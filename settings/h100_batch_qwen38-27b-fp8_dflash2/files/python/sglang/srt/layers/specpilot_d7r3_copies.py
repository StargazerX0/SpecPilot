"""SPECPILOT_D7R3 copy removals in the packed verify (SPECPILOT_DB_RV_MODE=pack); knobs read once at import.

SPECPILOT_D7R3_GATHER1=1  attention layers: the packed -> dense gather of q, k and v in one Triton launch (one program
                          per dense row) instead of three index_select kernels. Same bytes: dense row d = packed row
                          unpack[d] (0 for the unused rows, as index_select).
SPECPILOT_D7R3_ZVIEW=1    GDN layers: the DB_RV_FUSE split (d4 split_row) no longer copies z; z is the token-strided
                          view in_proj_qkvz_out[:, QKV:].view(T, HV, Dv). It equals the copied z on every verified
                          packed row (split_row wrote z[rows_out[d]] = qkvz[unpack[d], QKV:] and rows_out == unpack
                          there); the unused tail rows read their own in-projection row instead of uninitialised memory.
                          The gated norm + FP8 quant (d3 GNQ arithmetic, unchanged) reads z through token/head strides.
"""
from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

GATHER1 = os.environ.get("SPECPILOT_D7R3_GATHER1", "0") not in ("", "0")
ZVIEW = os.environ.get("SPECPILOT_D7R3_ZVIEW", "0") not in ("", "0")
_BLOCK = 2048


@triton.jit
def _copy_row(dst, src, W: tl.constexpr, BLOCK: tl.constexpr):
    for c0 in tl.static_range(0, W, BLOCK):
        o = c0 + tl.arange(0, BLOCK)
        m = o < W
        tl.store(dst + o, tl.load(src + o, mask=m), mask=m)


@triton.jit
def _gather3_kernel(q, k, v, qd, kd, vd, unpack_ptr, q_stride, k_stride, v_stride, QW: tl.constexpr,
                    KW: tl.constexpr, VW: tl.constexpr, BQ: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr):
    d = tl.program_id(0).to(tl.int64)
    src = tl.load(unpack_ptr + d)
    _copy_row(qd + d * QW, q + src * q_stride, QW, BQ)
    _copy_row(kd + d * KW, k + src * k_stride, KW, BK)
    _copy_row(vd + d * VW, v + src * v_stride, VW, BV)


def gather3_eligible(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> bool:
    return all(x.dim() == 2 and x.stride(-1) == 1 and x.dtype == q.dtype and x.shape[0] == q.shape[0]
               for x in (q, k, v)) and q.shape[0] > 0


def gather3_dense(p, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    """(q, k, v) packed rows [T, W] -> dense [bs * N, W] contiguous, = tuple(x.index_select(0, p.unpack))."""
    rows = p.unpack.shape[0]
    qd = q.new_empty((rows, q.shape[1]))
    kd = k.new_empty((rows, k.shape[1]))
    vd = v.new_empty((rows, v.shape[1]))
    if rows == 0:
        return qd, kd, vd
    _gather3_kernel[(rows,)](
        q, k, v, qd, kd, vd, p.unpack, q.stride(0), k.stride(0), v.stride(0),
        QW=q.shape[1], KW=k.shape[1], VW=v.shape[1],
        BQ=min(_BLOCK, triton.next_power_of_2(q.shape[1])), BK=min(_BLOCK, triton.next_power_of_2(k.shape[1])),
        BV=min(_BLOCK, triton.next_power_of_2(v.shape[1])), num_warps=4,
    )
    return qd, kd, vd


@triton.jit
def _split_row_noz_kernel(mixed_qkv, b, a, mixed_qkvz, mixed_ba, qkvz_row_stride, ba_row_stride, unpack_ptr,
                          QKV: tl.constexpr, HV: tl.constexpr, HVP: tl.constexpr, BLOCK: tl.constexpr):
    d = tl.program_id(0).to(tl.int64)
    src = tl.load(unpack_ptr + d)
    in_row = mixed_qkvz + src * qkvz_row_stride
    out_row = mixed_qkv + d * QKV
    for c0 in tl.static_range(0, QKV, BLOCK):
        o = c0 + tl.arange(0, BLOCK)
        m = o < QKV
        tl.store(out_row + o, tl.load(in_row + o, mask=m), mask=m)
    oh = tl.arange(0, HVP)
    mh = oh < HV
    ba_row = mixed_ba + src * ba_row_stride
    tl.store(b + d * HV + oh, tl.load(ba_row + oh, mask=mh), mask=mh)
    tl.store(a + d * HV + oh, tl.load(ba_row + HV + oh, mask=mh), mask=mh)


def split_qkvzba_dense_zview(p, mixed_qkvz, mixed_ba, num_heads_qk, num_heads_v, head_qk, head_v):
    """specpilot_d4_glue.split_qkvzba_dense without the z copy: (mixed_qkv dense, z view [T, HV, Dv], b, a)."""
    rows = p.bs * p.n_nodes
    T = mixed_qkvz.shape[0]
    qkv = num_heads_qk * head_qk * 2 + num_heads_v * head_v
    tv = num_heads_v * head_v
    assert mixed_qkvz.shape[1] == qkv + tv and mixed_qkvz.stride(-1) == 1 and mixed_ba.stride(-1) == 1
    mixed_qkv = mixed_qkvz.new_empty((rows, qkv))
    b = mixed_ba.new_empty((rows, num_heads_v))
    a = torch.empty_like(b)
    _split_row_noz_kernel[(rows,)](mixed_qkv, b, a, mixed_qkvz, mixed_ba, mixed_qkvz.stride(0), mixed_ba.stride(0),
                                   p.unpack, QKV=qkv, HV=num_heads_v, HVP=triton.next_power_of_2(num_heads_v),
                                   BLOCK=2048, num_warps=4)
    z = mixed_qkvz[:, qkv:].view(T, num_heads_v, head_v)
    return mixed_qkv, z, b, a


@triton.jit
def _gated_norm_quant_ztok_kernel(
    X, W, Z, Q, S,
    stride_x_row, stride_z_tok, stride_z_head,
    M,  # rows = tokens * heads
    HEADS,  # heads per token (= quant groups per token)
    s_stride,  # scale buffer stride between groups (align4(tokens))
    eps,
    inv_max,  # fp32(1 / 448)
    N: tl.constexpr, BLOCK_N: tl.constexpr, ROWS_PER_BLOCK: tl.constexpr, ACTIVATION: tl.constexpr,
):
    # specpilot_d3_gnq._gated_norm_quant_kernel with z addressed as [token, head, col] (token-strided view)
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
    tok = rows // HEADS
    grp = rows % HEADS
    z_off = tok.to(tl.int64) * stride_z_tok + grp.to(tl.int64) * stride_z_head
    z = tl.load(Z + z_off[:, None] + cols[None, :], mask=mask, other=0.0).to(tl.float32)
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
    tl.store(S + grp.to(tl.int64) * s_stride + tok, scale, mask=rows < M)


def zview_eligible(x: torch.Tensor, z3: torch.Tensor, norm, out_proj) -> bool:
    """d3 GNQ's eligibility (x as the shape proxy for z) and a [T, HV, 128] z with unit column stride."""
    from sglang.srt.layers import specpilot_d3_gnq as _gnq

    return (
        _gnq.GNQ
        and z3.dim() == 3
        and z3.dtype == torch.bfloat16
        and z3.stride(-1) == 1
        and z3.shape[0] * z3.shape[1] == x.shape[0]
        and z3.shape[2] == x.shape[-1]
        and _gnq.eligible(x, x, norm, out_proj)
    )


def gated_norm_quant_zview(x: torch.Tensor, z3: torch.Tensor, weight: torch.Tensor, eps: float, activation: str):
    """x [T * HV, 128] contiguous, z3 [T, HV, 128] token-strided -> (q, s) as specpilot_d3_gnq.gated_norm_quant."""
    from sglang.srt.layers import specpilot_d3_gnq as _gnq

    tokens, heads, _ = z3.shape
    M, N = x.shape
    assert N == _gnq.GROUP and heads * tokens == M and x.stride(-1) == 1 and z3.stride(-1) == 1
    m_al = (tokens + 3) // 4 * 4
    q = torch.empty((tokens, heads * N), dtype=torch.float8_e4m3fn, device=x.device)
    s_base = torch.empty((heads, m_al), dtype=torch.float32, device=x.device)
    rpb = _gnq._rows_per_block(M, x.device)
    inv_max = float(torch.tensor(1.0, dtype=torch.float32) / torch.tensor(_gnq._FP8_MAX, dtype=torch.float32))
    _gated_norm_quant_ztok_kernel[(triton.cdiv(M, rpb),)](
        x, weight, z3, q, s_base, x.stride(0), z3.stride(0), z3.stride(1), M, heads, m_al, eps, inv_max,
        N=N, BLOCK_N=triton.next_power_of_2(N), ROWS_PER_BLOCK=rpb, ACTIVATION=activation, num_warps=1,
    )
    return q, s_base.t()[:tokens]
