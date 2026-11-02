"""SPECPILOT_D3_GNQ=1: GDN gated RMSNorm + FP8 1x128 group quant of out_proj's input in one kernel.

The GDN output path ran fla's gated RMSNorm (``layernorm_gated._layer_norm_fwd_1pass_kernel``: per 128-wide head row
y = x * rsqrt(mean(x^2) + eps) * w, y *= act(z)), stored y in bf16, then ``sglang_per_token_group_quant_fp8`` re-read
it and wrote (q, s) for out_proj's DeepGEMM. A quant group (128) is exactly one head row, so the norm kernel can
quantise its own bf16-rounded row: amax = max(|y|, 1e-10), s = amax * (1 / 448), q = satfinite(min(y * (448 / amax),
448)) with the reference's arithmetic (fast-math division, fp32 scale, round-to-nearest fp8). The row arithmetic of the
norm is fla's (same tile shape, warps and rows per block), so q and s are meant to be bitwise equal to the two-kernel
path. Knob read once at import.
"""
from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

GNQ = os.environ.get("SPECPILOT_D3_GNQ", "0") not in ("", "0")
GROUP = 128
_FP8_MAX = 448.0


@triton.jit
def _gated_norm_quant_kernel(
    X, W, Z, Q, S,
    stride_x_row, stride_z_row,
    M,  # rows = tokens * heads
    HEADS,  # heads per token (= quant groups per token)
    s_stride,  # scale buffer stride between groups (align4(tokens))
    eps,
    inv_max,  # fp32(1 / 448)
    N: tl.constexpr, BLOCK_N: tl.constexpr, ROWS_PER_BLOCK: tl.constexpr, ACTIVATION: tl.constexpr,
):
    row_start = tl.program_id(0) * ROWS_PER_BLOCK
    rows = row_start + tl.arange(0, ROWS_PER_BLOCK)
    cols = tl.arange(0, BLOCK_N)
    row_mask = rows[:, None] < M
    col_mask = cols[None, :] < N
    mask = row_mask & col_mask
    x = tl.load(X + rows[:, None] * stride_x_row + cols[None, :], mask=mask, other=0.0).to(tl.float32)
    xbar = tl.where(mask, x, 0.0)
    var = tl.sum(xbar * xbar, axis=1) / N
    rstd = tl.rsqrt(var + eps)
    w = tl.load(W + cols, mask=cols < N, other=0.0).to(tl.float32)
    y = x * rstd[:, None] * w[None, :]
    z = tl.load(Z + rows[:, None] * stride_z_row + cols[None, :], mask=mask, other=0.0).to(tl.float32)
    if ACTIVATION == "swish" or ACTIVATION == "silu":
        y *= z * tl.sigmoid(z)
    elif ACTIVATION == "sigmoid":
        y *= tl.sigmoid(z)
    yb = y.to(tl.bfloat16).to(tl.float32)  # the reference stores y in bf16 and quantises that
    amax = tl.maximum(tl.max(tl.where(mask, tl.abs(yb), 0.0), axis=1), 1e-10)
    scale = amax * inv_max
    qs = tl.fdiv(tl.full(amax.shape, 448.0, tl.float32), amax, ieee_rounding=False)
    q = tl.minimum(yb * qs[:, None], 448.0).to(tl.float8e4nv)
    tl.store(Q + rows[:, None] * N + cols[None, :], q, mask=mask)
    tok = rows // HEADS
    grp = rows % HEADS
    tl.store(S + grp.to(tl.int64) * s_stride + tok, scale, mask=rows < M)


def _rows_per_block(M: int, device) -> int:
    from sglang.kernels.ops.attention.fla.layernorm_gated import calc_rows_per_block

    return calc_rows_per_block(M, device)


def gated_norm_quant(x: torch.Tensor, z: torch.Tensor, weight: torch.Tensor, eps: float, activation: str,
                     tokens: int):
    """x, z [tokens * heads, 128] bf16 (row-major rows) -> (q [tokens, heads * 128] fp8 e4m3,
    s [tokens, heads] fp32 with stride (1, align4(tokens))), as sglang_per_token_group_quant_fp8(
    column_major_scales=True, scale_tma_aligned=True) of the gated-norm output."""
    M, N = x.shape
    heads = M // tokens
    assert N == GROUP and heads * tokens == M and x.stride(-1) == 1 and z.stride(-1) == 1
    m_al = (tokens + 3) // 4 * 4
    q = torch.empty((tokens, heads * N), dtype=torch.float8_e4m3fn, device=x.device)
    s_base = torch.empty((heads, m_al), dtype=torch.float32, device=x.device)
    rpb = _rows_per_block(M, x.device)
    inv_max = float(torch.tensor(1.0, dtype=torch.float32) / torch.tensor(_FP8_MAX, dtype=torch.float32))
    _gated_norm_quant_kernel[(triton.cdiv(M, rpb),)](
        x, weight, z, q, s_base, x.stride(0), z.stride(0), M, heads, m_al, eps, inv_max,
        N=N, BLOCK_N=triton.next_power_of_2(N), ROWS_PER_BLOCK=rpb, ACTIVATION=activation, num_warps=1,
    )
    return q, s_base.t()[:tokens]


def eligible(x: torch.Tensor, z: torch.Tensor, norm, out_proj) -> bool:
    """The DeepGEMM block-FP8 out_proj path that consumes (q, s) from sglang_per_token_group_quant_fp8, and the
    plain gated-norm configuration (per-head RMS norm, gate after the norm, no bias)."""
    from sglang.srt.layers.quantization import fp8_utils
    from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod

    qm = out_proj.quant_method
    return (
        isinstance(qm, Fp8LinearMethod)
        and qm.block_quant
        and not qm.use_marlin
        and qm.w8a8_block_fp8_linear is fp8_utils.deepgemm_w8a8_block_fp8_linear_with_fallback
        and not fp8_utils.deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0
        and x.dtype == torch.bfloat16
        and z.dtype == torch.bfloat16
        and x.dim() == 2
        and x.shape == z.shape
        and x.shape[-1] == GROUP
        and x.shape[0] > 0
        and x.is_contiguous()
        and z.stride(-1) == 1
        and norm.group_size is None
        and norm.norm_before_gate
        and norm.bias is None
        and norm.activation in ("swish", "silu", "sigmoid")
        and out_proj.weight.shape[0] % 64 == 0
    )
