"""SPECPILOT_D2 (stream W, sm_120): lib w12_sm120 (specpilot_d2_w12, mma.sync stream-K) on the Muse-Glimmer target.

At the end of ``load_weights`` (``after_load`` hook, before KV-pool sizing) every selected bf16 linear of the 52
decoder layers is re-encoded as a ``W12Weight`` (bit-exact roundtrip enforced per tensor, raises otherwise), stored as
``lin._d2_w12`` and its bf16 weight freed:

- ``self_attn.qkv_proj`` [4608, 6656], ``self_attn.output_gate_proj`` [4096, 6656], ``mlp.down_proj`` [6656, 19968]
  and, with SPECPILOT_D2_W12_OUT, ``self_attn.o_proj`` [6656, 4096]: served by the lib's generic
  ``UnquantizedLinearMethod.apply`` dispatch (unquant.py, ``layer._d2_w12``);
- ``mlp.gate_up_proj`` [39936, 6656] with the fused SiLU(gate) * up epilogue: served by the ``mlp_forward`` hook
  (``d2_gate_up`` then ``down_proj``; M > 128 decodes to the shared scratch and runs cuBLAS + ``act_fn``);
- with SPECPILOT_D2_W12_HEAD the ``lm_head``: only its first ``org_vocab_size`` rows (202048; the 64 zero padding rows
  of the 128-padded ParallelLMHead would all be escapes in one tile), so the W12 logits are [M, 202048] bf16, the
  width ``_copy_logits_to_buffer`` slices stock's padded logits to; logits_processor.py and dflash.py dispatch on
  ``lm_head._d2_w12`` (target logits, then stock's bf16 x0.196 and fp32 softcap unchanged; drafter candidates).

Every encoded linear keeps the nested W12 record format, so the two-level verify Q view (SPECPILOT_D4_QVIEW,
``q_pass()``) and the D3 Q copies (SPECPILOT_D3_QCOPY, attached here as in ``d2_encode_target``) apply unchanged.

Knobs (read once at import; with SPECPILOT_D2_W12 unset nothing here runs: ENABLED False, no hook bound):
  SPECPILOT_D2_W12=1         (lib knob) this stream; it also arms the unquant.py dispatch
  SPECPILOT_D2_W12_OUT=1     (lib knob) also o_proj
  SPECPILOT_D2_W12_HEAD=1    (lib knob) also the lm_head
  SPECPILOT_D2W_ENC          comma list of the per-layer linears to encode among qkv, gate, gate_up, down
                             (default: all four; o_proj follows SPECPILOT_D2_W12_OUT)
  SPECPILOT_D7W_W11=1        encode the same linears as W11o (specpilot_d7_w11: lossless 11-bit code, 7-binade
                             per-row window + ordered escape byte stream, ~11.24 b/w) instead of W12; same dispatch
  SPECPILOT_INT_W11_HEAD=1   (int-003) with D7W_W11 also the lm_head in W11o; default off: the head stays W12 so the
                             TLV Q pass keeps its nested-8 head view (W11o has no nested view and would run exact)
"""
from __future__ import annotations

import logging
import os
from typing import Dict, Optional

import torch

logger = logging.getLogger(__name__)


def _env_on(name: str) -> bool:
    return os.environ.get(name, "").strip() not in ("", "0")


ENABLED = _env_on("SPECPILOT_D2_W12")
ENC_OUT = _env_on("SPECPILOT_D2_W12_OUT")
ENC_HEAD = _env_on("SPECPILOT_D2_W12_HEAD")
D7_W11 = _env_on("SPECPILOT_D7W_W11")
INT_W11_HEAD = _env_on("SPECPILOT_INT_W11_HEAD")
_KNOWN = ("qkv", "gate", "gate_up", "down")
_ENC = frozenset(s for s in os.environ.get("SPECPILOT_D2W_ENC", ",".join(_KNOWN)).replace(" ", "").split(",") if s)
if ENABLED and not _ENC <= set(_KNOWN):
    raise ValueError(f"SPECPILOT_D2W_ENC: unknown linears {sorted(_ENC - set(_KNOWN))} (known {list(_KNOWN)})")


def _new_stats() -> Dict[str, Dict[str, int]]:
    return {}


def _account(stats, *, kind: str, enc, w: torch.Tensor) -> None:
    s = stats.setdefault(kind, {"n": 0, "weights": 0, "bf16": 0, "w12": 0, "esc": 0, "inexact": 0, "max_row_esc": 0})
    s["n"] += 1
    s["weights"] += w.numel()
    s["bf16"] += w.numel() * 2
    s["w12"] += enc.nbytes()
    s["esc"] += enc.n_esc
    s["inexact"] += enc.n_inexact
    s["max_row_esc"] = max(s["max_row_esc"], enc.max_row_esc)


def _encode(w: torch.Tensor, *, silu: bool, head: bool, what: str, kind: str, stats):
    from sglang.kernels.ops.gemm import specpilot_d2_w12 as w12

    if D7_W11 and (not head or INT_W11_HEAD):
        from sglang.kernels.ops.gemm import specpilot_d7_w11 as w11

        enc = w11.W11Weight(w, silu=silu, head=head)
    else:
        enc = w12.W12Weight(w, silu=silu, head=head)
    dec = enc.decode()
    if not torch.equal(dec, w):
        bad = int((dec != w).sum().item())
        raise RuntimeError(f"SPECPILOT_D2 W12/W11 not bit-exact at {what} {tuple(w.shape)}: {bad} mismatches")
    del dec
    _account(stats, kind=kind, enc=enc, w=w)
    return enc


def _plain(lin) -> bool:
    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

    w = lin.weight
    return (isinstance(lin.quant_method, UnquantizedLinearMethod) and lin.bias is None and w.dtype == torch.bfloat16
            and w.dim() == 2 and w.shape[0] > 0 and w.shape[1] % 64 == 0 and lin.tp_size == 1)


def _encode_linear(lin, *, silu: bool, what: str, kind: str, stats) -> None:
    from sglang.kernels.ops.gemm.specpilot_d2_w12 import _free

    if not _plain(lin) or (silu and lin.weight.shape[0] % 128):
        logger.info("SPECPILOT_D2 W12: %s left bf16 (quantized, biased, tp > 1 or odd shape)", what)
        return
    lin._d2_w12 = _encode(lin.weight.data, silu=silu, head=False, what=what, kind=kind, stats=stats)
    _free(lin.weight)


def _encode_layer(layer, *, prefix: str, stats) -> None:
    attn, mlp = layer.self_attn, layer.mlp
    picks = []
    if "qkv" in _ENC and not attn.unfused_qkv:
        picks.append((attn.qkv_proj, False, "self_attn.qkv_proj", "qkv"))
    if "gate" in _ENC and attn.use_output_gate:
        picks.append((attn.output_gate_proj, False, "self_attn.output_gate_proj", "gate"))
    if ENC_OUT:
        picks.append((attn.o_proj, False, "self_attn.o_proj", "o"))
    if "gate_up" in _ENC:
        picks.append((mlp.gate_up_proj, True, "mlp.gate_up_proj", "gate_up"))
    if "down" in _ENC:
        picks.append((mlp.down_proj, False, "mlp.down_proj", "down"))
    for lin, silu, name, kind in picks:
        _encode_linear(lin, silu=silu, what=f"{prefix}.{name}", kind=kind, stats=stats)


def _encode_head(model, *, stats) -> None:
    from sglang.kernels.ops.gemm import specpilot_d2_w12 as w12
    from sglang.srt.layers.quantization.unquant import UnquantizedEmbeddingMethod

    head = model.lm_head
    w = head.weight
    if not (isinstance(head.quant_method, UnquantizedEmbeddingMethod) and w.dtype == torch.bfloat16 and w.dim() == 2
            and w.shape[1] % 64 == 0 and head.tp_size == 1 and not w12._lm_head_fp32(model)
            and not w12._tied(model, head)):
        logger.info("SPECPILOT_D2 W12: lm_head not encoded (quantized, tied, tp > 1 or fp32 logits)")
        return
    org = int(head.org_vocab_size)
    enc = _encode(w.data[:org], silu=False, head=True, what="lm_head", kind="head", stats=stats)
    head._d2_w12 = enc
    if w12._drafter_derives_head():
        # SPECPILOT_INT: the drafter derives its candidate head from these bf16 rows, then frees them
        head._d2_free_pending = True
    else:
        w12._free(w)


def after_load(model) -> None:
    from sglang.kernels.ops.gemm import specpilot_d2_w12 as w12

    stats = _new_stats()
    for i, layer in enumerate(model.model.layers):
        _encode_layer(layer, prefix=f"model.layers.{i}", stats=stats)
        torch.cuda.empty_cache()
    if ENC_HEAD:
        _encode_head(model, stats=stats)
        torch.cuda.empty_cache()
    if w12.D3_QCOPY and w12.D4_QVIEW:  # SPECPILOT_D3_QCOPY (as d2_encode_target)
        w12.d3_attach_qcopies(model)
    if w12._D4_L2PF:  # SPECPILOT_D4_L2PF (d1-020, Muse branch): per-layer prefetch plans, after the Q copies
        w12._l2pf.build_plans(model)
    w12._warmup_kernels(model)
    torch.cuda.empty_cache()
    _log_stats(stats)


def _log_stats(stats) -> None:
    tot = {k: sum(s[k] for s in stats.values()) for k in ("n", "weights", "bf16", "w12", "esc", "inexact")}
    for kind, s in sorted(stats.items()):
        logger.info(
            "SPECPILOT_D2 W12 Muse %s: %d tensors, %.3f GB bf16 -> %.3f GB W12 (%.4f), escapes %d (rate %.3g), "
            "max escapes in a row %d, escape corrections below fp32 range %d", kind, s["n"], s["bf16"] / 1e9,
            s["w12"] / 1e9, s["w12"] / max(1, s["bf16"]), s["esc"], s["esc"] / max(1, s["weights"]), s["max_row_esc"],
            s["inexact"])
    logger.info(
        "SPECPILOT_D2 W12: encoded %d Muse tensors (%s, o %d, head %d), %.2f GB bf16 -> %.2f GB W12 (%.4f), escapes %d "
        "(rate %.3g), roundtrip bit-exact (enforced)", tot["n"], ",".join(sorted(_ENC)), ENC_OUT, ENC_HEAD,
        tot["bf16"] / 1e9, tot["w12"] / 1e9, tot["w12"] / max(1, tot["bf16"]), tot["esc"],
        tot["esc"] / max(1, tot["weights"]))


def mlp_forward(mlp, x: torch.Tensor) -> Optional[torch.Tensor]:
    enc = getattr(mlp.gate_up_proj, "_d2_w12", None)  # set only on encoded layers (after_load)
    if enc is None:
        return None
    from sglang.kernels.ops.gemm.specpilot_d2_w12 import d2_gate_up

    h, fused = d2_gate_up(enc, x)
    out, _ = mlp.down_proj(h if fused else mlp.act_fn(h))
    return out


# hook table for the SPECPILOT_D1 dispatcher (specpilot_muse.py); empty with the knobs unset
MUSE_HOOKS = {}
if ENABLED:
    MUSE_HOOKS = {"after_load": after_load}
    if "gate_up" in _ENC:
        MUSE_HOOKS["mlp_forward"] = mlp_forward
