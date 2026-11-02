"""SPECPILOT_D4P_GDNCAP: GDN prefill extends replayed as CUDA graphs inside the breakable prefill graph.

The breakable prefill graph runs every GDN extend eagerly (real token count, varlen metadata): ~10 kernel launches
plus Python glue per layer, which leaves the GPU idle once the prefill linears are fast. Here each (layer, token
bucket B) extend is captured once, at the breakable graph's capture time, over the whole padded bucket with the real
length L on device: the conv and chunked delta-rule kernels get the real cu_seqlens [0, L] from a device buffer
(their grids cover the bucket; they stop at L), so they compute exactly what the eager path computes; padded output
rows are zeroed (as the eager path does). Used only for batch-1 prefills without a cached prefix (the c1
regime), with contiguous state pools and no state checkpoints / tracking; anything else runs the eager path.

Knob (read once at import): SPECPILOT_D4P_GDNCAP=1; SPECPILOT_D4P_GDNCAP_MAX (default 448) = largest bucket captured;
SPECPILOT_D4P_GDNCAP_CHECK=N compares graph vs eager (outputs and states) on the first N eligible calls and logs it.
"""
from __future__ import annotations

import logging
import os
from typing import Dict, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

D4P_GDNCAP = int(os.environ.get("SPECPILOT_D4P_GDNCAP", "0").strip() or 0)
# 448: a server with bucket-512 graphs crashed (host segfault in a segment replay, probe p6) at P 502; GPU-bound above
_MAX_B = int(os.environ.get("SPECPILOT_D4P_GDNCAP_MAX", "448").strip() or 448)
_CHECK = int(os.environ.get("SPECPILOT_D4P_GDNCAP_CHECK", "0").strip() or 0)
# SPECPILOT_D4P_GDNCAP_DEVLEN (default 1): the kernels read the real cu_seqlens [0, L] from device, i.e. process the
# same L tokens as the eager path -> outputs, conv and ssm states bitwise equal (96/96 live checks, probe p9); 0 = the
# whole-bucket variant (beta 0 / unit decay on padding), whose fp32 state rounding differs (57/120 identical, d4-008)
_DEVLEN = int(os.environ.get("SPECPILOT_D4P_GDNCAP_DEVLEN", "1").strip() or 1)

# (layer_id, B, ptrs of mixed_qkv / a / b / output) -> CUDAGraph, or None when capture failed (eager forever)
_GRAPHS: Dict[Tuple, Optional[torch.cuda.CUDAGraph]] = {}
_STATIC: dict = {}
_STATS = {"captured": 0, "failed": 0, "replays": 0, "checked": 0}


def _static(dev):
    st = _STATIC.get(dev)
    if st is None:
        st = {
            "L": torch.zeros(1, dtype=torch.int64, device=dev),
            "idx": torch.zeros(1, dtype=torch.int32, device=dev),
            "noinit": torch.zeros(1, dtype=torch.bool, device=dev),
            "ar": torch.arange(max(_MAX_B, 8), dtype=torch.int64, device=dev),
            "qsl": {},
            "qdev": torch.zeros(2, dtype=torch.int32, device=dev),  # [0, L]: real cu_seqlens, set per forward
            "pool": torch.cuda.graph_pool_handle(),
        }
        _STATIC[dev] = st
    return st


def _qsl(st, B, dev):
    q = st["qsl"].get(B)
    if q is None:
        q = torch.tensor([0, B], dtype=torch.int32, device=dev)
        st["qsl"][B] = q
    return q


def _gdn_backend():
    from sglang.srt.model_executor.forward_context import get_attn_backend

    be = get_attn_backend()
    return getattr(be, "linear_attn_backend", be)


def _key(layer_id, mixed_qkv, a, b, output):
    return (layer_id, mixed_qkv.shape[0], mixed_qkv.data_ptr(), a.data_ptr(), b.data_ptr(), output.data_ptr())


def _body(gdn, layer, mixed_qkv, a, b, output, st):
    """Whole-bucket GDN extend for one sequence of real length st['L'] (device), cache slot st['idx']."""
    from sglang.kernels.ops.attention.fla.fused_gdn_gating import fused_gdn_gating
    from sglang.kernels.ops.attention.triton_gdn_fused_proj import fused_qkv_split_gdn_prefill
    from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_fn
    from sglang.srt.layers.attention.linear.gdn_backend import MAX_FUSED_QKV_SPLIT_DIM

    B = mixed_qkv.shape[0]
    dev = mixed_qkv.device
    qsl = st["qdev"] if _DEVLEN else _qsl(st, B, dev)
    cache = gdn.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
    conv_states, ssm_states = cache.conv[0], cache.temporal
    L, idx = st["L"], st["idx"]
    x = mixed_qkv.transpose(0, 1)
    y = causal_conv1d_fn(
        x, layer.conv_weights, layer.bias, activation=layer.activation, conv_states=conv_states,
        has_initial_state=st["noinit"], cache_indices=idx, query_start_loc=qsl, seq_lens_cpu=[B],
    ).transpose(0, 1)[:B]
    if _DEVLEN:  # kernels bounded by the device cu_seqlens: no conv-state rewrite, no padding mask
        w1 = 0
    else:
        w1 = conv_states.shape[2]
    if w1:
        pos = L - w1 + st["ar"][:w1]
        win = x.index_select(1, pos.clamp(min=0))
        win = torch.where((pos >= 0)[None, :], win, torch.zeros_like(win))
        conv_states.index_copy_(0, idx.to(torch.int64), win.unsqueeze(0).to(conv_states.dtype))
    if layer.q_dim + layer.k_dim + layer.v_dim <= MAX_FUSED_QKV_SPLIT_DIM:
        query, key, value = fused_qkv_split_gdn_prefill(
            y, layer.num_q_heads, layer.num_k_heads, layer.num_v_heads, layer.head_q_dim, layer.head_k_dim,
            layer.head_v_dim,
        )
    else:  # as the eager path: views of the conv output
        query, key, value = torch.split(y, [layer.q_dim, layer.k_dim, layer.v_dim], dim=-1)
        query = query.view(1, B, layer.num_q_heads, layer.head_q_dim)
        key = key.view(1, B, layer.num_k_heads, layer.head_k_dim)
        value = value.view(1, B, layer.num_v_heads, layer.head_v_dim)
    g, beta = fused_gdn_gating(layer.A_log, a, b, layer.dt_bias)
    valid = (st["ar"][:B] < L).view(1, B, 1)
    if not _DEVLEN:
        g = torch.where(valid, g, torch.zeros_like(g))
        beta = torch.where(valid, beta, torch.zeros_like(beta))
    gdn.kernel_dispatcher.extend(
        q=query, k=key, v=value, g=g, beta=beta, ssm_states=ssm_states, cache_indices=idx, query_start_loc=qsl,
        state_checkpoint_cu_starts=None, num_state_checkpoints=0, state_checkpoint_every_n_tokens=0, output=output,
    )
    output.masked_fill_(~valid.view(1, B, 1, 1), 0)


def _supported(gdn, layer, mixed_qkv, output) -> bool:
    if output is None or output.dim() != 4 or not output.is_contiguous() or gdn.mis_metadata is not None:
        return False
    cache = gdn.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
    conv_states, ssm_states = cache.conv[0], cache.temporal
    return (conv_states.is_contiguous() and ssm_states.is_contiguous()
            and conv_states.shape[2] == layer.conv_weights.shape[1] - 1 and mixed_qkv.shape[0] <= _MAX_B)


_PENDING: list = []


def precapture(mixed_qkv, a, b, output, layer_id, layer) -> None:
    """Called from the breakable graph's capture stub: record this (layer, bucket); the graphs are captured after the
    breakable capture, on the first eligible replay (the server's warm-up request), outside any other capture."""
    output.zero_()
    key = _key(layer_id, mixed_qkv, a, b, output)
    if key in _GRAPHS or mixed_qkv.shape[0] > _MAX_B:
        return
    _GRAPHS[key] = None
    _PENDING.append((key, mixed_qkv, a, b, output, layer))


def _capture_pending() -> None:
    gdn = _gdn_backend()
    logger.info("SPECPILOT_D4P_GDNCAP: capturing %d GDN prefill graphs (buckets <= %d)", len(_PENDING), _MAX_B)
    s = torch.cuda.Stream()
    for key, mixed_qkv, a, b, output, layer in _PENDING:
        if not _supported(gdn, layer, mixed_qkv, output):
            continue
        st = _static(mixed_qkv.device)
        st["L"].fill_(mixed_qkv.shape[0])
        st["qdev"][1:2].fill_(mixed_qkv.shape[0])
        st["idx"].zero_()  # the reserved dummy slot
        try:
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                _body(gdn, layer, mixed_qkv, a, b, output, st)  # warm-up (Triton / cute-DSL compile)
                g = torch.cuda.CUDAGraph()
                g.capture_begin(pool=st["pool"])
                try:
                    _body(gdn, layer, mixed_qkv, a, b, output, st)
                finally:
                    g.capture_end()
            torch.cuda.current_stream().wait_stream(s)
            _GRAPHS[key] = g
            _STATS["captured"] += 1
        except Exception as e:  # capture-unsafe op on this path: keep the eager extend for this key
            _STATS["failed"] += 1
            if _STATS["failed"] <= 3:
                logger.warning("SPECPILOT_D4P_GDNCAP capture failed (layer %d, B %d): %s", layer.layer_id,
                               mixed_qkv.shape[0], e)
    _PENDING.clear()
    torch.cuda.synchronize()
    logger.info("SPECPILOT_D4P_GDNCAP: %d GDN prefill graphs captured (%d failed)", _STATS["captured"], _STATS["failed"])


def try_run(mixed_qkv, a, b, output, layer, forward_batch) -> bool:
    """Replay the captured whole-bucket extend when this prefill is eligible; False -> caller runs the eager path."""
    if _PENDING:
        _capture_pending()
    g = _GRAPHS.get(_key(layer.layer_id, mixed_qkv, a, b, output))
    if g is None:
        return False
    fb = forward_batch
    if fb.batch_size != 1 or not fb.forward_mode.is_extend() or fb.forward_mode.is_target_verify():
        return False
    prefix = fb.extend_prefix_lens_cpu
    if prefix is None or len(prefix) != 1 or prefix[0] != 0:
        return False
    gdn = _gdn_backend()
    md = gdn.forward_metadata
    if gdn.mis_metadata is not None or md.has_mamba_track_mask or md.num_state_checkpoints:
        return False
    L = min(int(fb.global_num_token_non_padded_cpu), mixed_qkv.shape[0])
    if L <= 0:
        return False
    st = _static(mixed_qkv.device)
    if st.get("fwd") is not fb:  # once per forward: real length and cache slot
        st["L"].fill_(L)
        st["qdev"][1:2].fill_(L)
        st["idx"].copy_(md.mamba_cache_indices[:1])
        st["fwd"] = fb
    if _STATS["checked"] < _CHECK:
        _check(g, gdn, layer, mixed_qkv, a, b, output, fb, L)
        return True
    g.replay()
    _STATS["replays"] += 1
    return True


def _check(g, gdn, layer, mixed_qkv, a, b, output, fb, L) -> None:
    """Graph vs eager on the live call: same outputs (real rows) and states within bf16/fp32 rounding."""
    from sglang.srt.layers.radix_linear_attention import _linear_attention_with_output_impl

    cache = gdn.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
    conv_states, ssm_states = cache.conv[0], cache.temporal
    i = int(gdn.forward_metadata.mamba_cache_indices[0].item())
    c0, s0 = conv_states[i].clone(), ssm_states[i].clone()
    _linear_attention_with_output_impl(mixed_qkv, a, b, output, layer, fb)
    oe, ce, se = output.clone(), conv_states[i].clone(), ssm_states[i].clone()
    conv_states[i].copy_(c0)
    ssm_states[i].copy_(s0)
    g.replay()
    do = (output[:, :L].float() - oe[:, :L].float()).abs().max().item()
    dc = (conv_states[i].float() - ce.float()).abs().max().item()
    ds = (ssm_states[i].float() - se.float()).abs().max().item()
    pad_zero = bool((output[:, L:] == 0).all().item())
    _STATS["checked"] += 1
    logger.info("SPECPILOT_D4P_GDNCAP check layer %d L %d B %d: max|d out| %.3g (|out| %.3g) max|d conv| %.3g "
                "max|d ssm| %.3g (|ssm| %.3g) pad_zero %s", layer.layer_id, L, mixed_qkv.shape[0], do,
                oe[:, :L].float().abs().max().item(), dc, ds, se.float().abs().max().item(), pad_zero)
