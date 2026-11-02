"""SPECPILOT_D1R3_GLUE2=1: verify-row glue fixes at the packed verify rows (MIN_M <= M <= MAX_M), ported from the
library module glue2_load (items 1-3; its item 4, the gate folded into the suffix merge, only fires at identity row
layout and the FP8 seed verifies packed rows, so it is not ported).

1. (kernel kept for its tests, NOT hooked: microbench 713c50576d9f measured it slower than flashinfer's
   gemma_fused_add_rmsnorm on this stack, 2.89 -> 3.30 us/call at M 192) Residual-add GemmaRMSNorm: one Triton program per row, x = h + residual (fp32), residual = bf16(x),
   h = x * rsqrt(mean(x^2) + eps) * (1 + w) in place (the gemma_fused_add_rmsnorm contract).
2. Attention Q/K GemmaRMSNorm + partial NeoX RoPE (fused_qk_gemma_rmsnorm_rope_gate's math, same rounding points):
   HB heads per program instead of one, and no gate copy: the gate is a strided [T, heads, head_dim] view of the
   interleaved Q+gate GEMM output, read by fused_sigmoid_mul's strided-gate path.
3. FA3 tree-verify prefix scheduler metadata computed once per forward (first attention layer) instead of by
   prepare_varlen_num_blocks in all 16 attention layers.

Read once at import; strict no-op when unset.
"""
import os

import torch
import triton
import triton.language as tl

GLUE2 = os.environ.get("SPECPILOT_D1R3_GLUE2", "0").strip() not in ("", "0")
MIN_M, MAX_M = 64, 256
# item 3 runs on the dense tree rows (bs x 16, up to 384 at bs 24); the metadata is per forward, not per row count
FA_MAX_M = 512
QNR_HEADS_PER_PROGRAM = 4
QNR_WARPS = 4


@triton.jit
def _d1r3_g2_add_gemma_rmsnorm_kernel(X, stride_xm, R, stride_rm, W, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    xp = X + row * stride_xm + cols
    rp = R + row * stride_rm + cols
    x = tl.load(xp, mask=mask, other=0.0).to(tl.float32) + tl.load(rp, mask=mask, other=0.0).to(tl.float32)
    tl.store(rp, x.to(tl.bfloat16), mask=mask)
    rstd = tl.rsqrt(tl.sum(x * x, axis=0) / N + eps)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32) + 1.0
    tl.store(xp, (x * rstd * w).to(tl.bfloat16), mask=mask)


def norm_eligible(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor) -> bool:
    return (
        x.is_cuda
        and x.dim() == 2
        and x.dtype == torch.bfloat16
        and residual.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and MIN_M <= x.shape[0] <= MAX_M
        and residual.shape == x.shape
        and x.stride(1) == 1
        and residual.stride(1) == 1
        and weight.dim() == 1
        and weight.shape[0] == x.shape[1]
        and weight.stride(0) == 1
        and x.shape[1] <= 8192
        and not torch.compiler.is_compiling()
    )


def add_gemma_rmsnorm(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float) -> None:
    """In place: ``residual = bf16(x + residual)``; ``x = GemmaRMSNorm(x + residual)`` (fp32 sum)."""
    m, n = x.shape
    _d1r3_g2_add_gemma_rmsnorm_kernel[(m,)](
        x, x.stride(0), residual, residual.stride(0), weight, n, eps, BLOCK=triton.next_power_of_2(n), num_warps=8
    )


@triton.jit
def _d1r3_g2_qk_norm_rope_kernel(
    q_gate_ptr, k_ptr, q_out_ptr, k_out_ptr, q_weight_ptr, k_weight_ptr, cos_sin_cache_ptr, positions_ptr,
    mrope_axis_map_ptr, stride_qg_t, stride_k_t, stride_qo_t, stride_ko_t, stride_cos_t, stride_pos_axis,
    NUM_Q_HEADS: tl.constexpr, HEAD_DIM: tl.constexpr, ROTARY_DIM: tl.constexpr, HALF_ROTARY: tl.constexpr,
    HB: tl.constexpr, ROT_HALF_BLOCK: tl.constexpr, EPS: tl.constexpr, MROPE: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    h0 = tl.program_id(1) * HB
    hh = tl.arange(0, HB)
    if h0 >= NUM_Q_HEADS:
        lh = h0 - NUM_Q_HEADS + hh
        in_base = k_ptr + token * stride_k_t + lh * HEAD_DIM
        w_ptr = k_weight_ptr
        out_base = k_out_ptr + token * stride_ko_t + lh * HEAD_DIM
    else:
        lh = h0 + hh
        in_base = q_gate_ptr + token * stride_qg_t + lh * (2 * HEAD_DIM)
        w_ptr = q_weight_ptr
        out_base = q_out_ptr + token * stride_qo_t + lh * HEAD_DIM
    d = tl.arange(0, HEAD_DIM)
    x = tl.load(in_base[:, None] + d[None, :]).to(tl.float32)
    w = tl.load(w_ptr + d).to(tl.float32)
    inv_rms = tl.rsqrt(tl.sum(x * x, axis=1) / HEAD_DIM + EPS)
    x_norm = (x * inv_rms[:, None] * (w[None, :] + 1.0)).to(tl.bfloat16)
    tl.store(out_base[:, None] + d[None, :], x_norm, mask=(d >= ROTARY_DIM)[None, :])

    r = tl.arange(0, ROT_HALF_BLOCK)
    rm = r < HALF_ROTARY
    m2 = rm[None, :]
    xr1 = tl.load(in_base[:, None] + r[None, :], mask=m2, other=0.0).to(tl.float32)
    xr2 = tl.load(in_base[:, None] + HALF_ROTARY + r[None, :], mask=m2, other=0.0).to(tl.float32)
    wr1 = tl.load(w_ptr + r, mask=rm, other=0.0).to(tl.float32)
    wr2 = tl.load(w_ptr + HALF_ROTARY + r, mask=rm, other=0.0).to(tl.float32)
    xr1 = (xr1 * inv_rms[:, None] * (wr1[None, :] + 1.0)).to(tl.bfloat16).to(tl.float32)
    xr2 = (xr2 * inv_rms[:, None] * (wr2[None, :] + 1.0)).to(tl.bfloat16).to(tl.float32)
    if MROPE:
        axis = tl.load(mrope_axis_map_ptr + r, mask=rm, other=0)
        pos = tl.load(positions_ptr + axis * stride_pos_axis + token, mask=rm, other=0)
    else:
        pos = tl.load(positions_ptr + token)
    off = pos.to(tl.int64) * stride_cos_t
    cos = tl.load(cos_sin_cache_ptr + off + r, mask=rm, other=0.0).to(tl.float32)[None, :]
    sin = tl.load(cos_sin_cache_ptr + off + HALF_ROTARY + r, mask=rm, other=0.0).to(tl.float32)[None, :]
    tl.store(out_base[:, None] + r[None, :], (xr1 * cos - xr2 * sin).to(tl.bfloat16), mask=m2)
    tl.store(out_base[:, None] + HALF_ROTARY + r[None, :], (xr2 * cos + xr1 * sin).to(tl.bfloat16), mask=m2)


def qnr_eligible(q_gate, k, cos_sin_cache, positions, num_q_heads, num_kv_heads, head_dim, rotary_dim) -> bool:
    hb = QNR_HEADS_PER_PROGRAM
    t = q_gate.shape[0]
    return (
        q_gate.is_cuda
        and q_gate.dtype == torch.bfloat16
        and k.dtype == torch.bfloat16
        and MIN_M <= t <= MAX_M
        and q_gate.dim() == 2
        and k.dim() == 2
        and q_gate.shape[1] == 2 * num_q_heads * head_dim
        and k.shape == (t, num_kv_heads * head_dim)
        and q_gate.stride(1) == 1
        and k.stride(1) == 1
        and cos_sin_cache.stride(-1) == 1
        and head_dim == triton.next_power_of_2(head_dim)
        and 0 < rotary_dim <= head_dim
        and rotary_dim % 2 == 0
        and num_q_heads % hb == 0
        and num_kv_heads % hb == 0
        and (positions.dim() == 1 or (positions.dim() == 2 and positions.shape[0] == 3 and positions.stride(1) == 1))
        and positions.shape[-1] == t
        and not torch.compiler.is_compiling()
    )


def qk_norm_rope(q_gate, k, q_weight, k_weight, cos_sin_cache, positions, eps, num_q_heads, num_kv_heads, head_dim,
                 rotary_dim, mrope_axis_map=None):
    """fused_qk_gemma_rmsnorm_rope_gate(has_gate=True) outputs, except the gate is a strided [T, heads, head_dim]
    view of ``q_gate`` (no copy)."""
    t = q_gate.shape[0]
    mrope = positions.dim() == 2
    assert mrope == (mrope_axis_map is not None)
    q_out = torch.empty(t, num_q_heads * head_dim, dtype=q_gate.dtype, device=q_gate.device)
    k_out = torch.empty(t, num_kv_heads * head_dim, dtype=k.dtype, device=k.device)
    half = rotary_dim // 2
    hb = QNR_HEADS_PER_PROGRAM
    _d1r3_g2_qk_norm_rope_kernel[(t, (num_q_heads + num_kv_heads) // hb)](
        q_gate, k, q_out, k_out, q_weight, k_weight, cos_sin_cache, positions, mrope_axis_map,
        q_gate.stride(0), k.stride(0), q_out.stride(0), k_out.stride(0), cos_sin_cache.stride(0), positions.stride(0),
        NUM_Q_HEADS=num_q_heads, HEAD_DIM=head_dim, ROTARY_DIM=rotary_dim, HALF_ROTARY=half, HB=hb,
        ROT_HALF_BLOCK=triton.next_power_of_2(half), EPS=eps, MROPE=mrope, num_warps=QNR_WARPS,
    )
    gate = q_gate.view(t, num_q_heads, 2 * head_dim)[:, :, head_dim:]
    return q_out, k_out, gate


def prefix_scheduler_metadata(get_scheduler_metadata, metadata, layer_id, *, q, key_cache, page_table, cache_seqlens,
                              cu_seqlens_q, max_seqlen_q, window_size, softcap, num_splits):
    """3. FA3 scheduler metadata of the tree-verify prefix call (flash_attn_with_kvcache on the paged cache, non-causal,
    split-KV), computed once per forward instead of by prepare_varlen_num_blocks in every attention layer. The first
    full-attention layer (smallest layer_id) recomputes it on every forward, eager or captured; later layers reuse it.
    Arguments mirror what mha_fwd derives for that call: no k_new (so no cu_seqlens_k_new), max_seqlen_k = pages per
    row x page size."""
    first = getattr(metadata, "_d1r3_g2_sched_layer", None)
    if first is None or layer_id <= first:
        page_size = key_cache.shape[1]
        metadata._d1r3_g2_sched = get_scheduler_metadata(
            batch_size=cache_seqlens.shape[0],
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=page_table.shape[1] * page_size,
            num_heads=q.shape[-2],
            num_heads_k=key_cache.shape[-2],
            headdim=q.shape[-1],
            cache_seqlens=cache_seqlens,
            qkv_dtype=key_cache.dtype,
            cu_seqlens_q=cu_seqlens_q,
            page_size=page_size,
            causal=False,
            window_size=window_size,
            has_softcap=softcap > 0,
            num_splits=num_splits,
        )
        metadata._d1r3_g2_sched_layer = layer_id
    return metadata._d1r3_g2_sched
