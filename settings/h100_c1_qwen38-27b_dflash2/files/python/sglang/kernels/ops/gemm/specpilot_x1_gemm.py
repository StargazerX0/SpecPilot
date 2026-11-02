"""SPECPILOT_X1: verify-width (M <= 64) BF16 weight-streaming GEMM for the target linears at batch 1.

out[M, N] = x[M, K] @ w[N, K]^T. Swap-AB: the weight rows are the MMA M dimension (BLOCK_N), the verify rows the
MMA N dimension (BM = 16 / 32 / 64, zero-padded); the weight is streamed through a host TMA descriptor with a
multi-stage pipeline, no split-K (one CTA per BLOCK_N rows, deterministic, no reduce kernel).
``gate_up_silu``: MergedColumn gate_up ([2I, K]) + SiLU(gate) * up epilogue in the same kernel; the gate and up
accumulators are rounded to bf16 before the activation, as the cuBLAS + act_and_mul path does.

Knob (read once at import): SPECPILOT_X1_GEMM=1 enables the dispatch; unset = cuBLAS (the parent's path).
Only (N, K) shapes in ``_CFG`` are taken; anything else (prefill, M > 64, other shapes) returns None.
"""
from __future__ import annotations

import os
from typing import Optional

import torch
import triton
import triton.language as tl


def read_x1_knobs() -> int:
    v = os.environ.get("SPECPILOT_X1_GEMM", "").strip()
    return int(v) if v else 0


X1_GEMM = read_x1_knobs()
_D2_PRE = os.environ.get("SPECPILOT_D2_PRE", "").strip() not in ("", "0")  # SPECPILOT_D2

# (N_out, K, silu) -> {BM: (BLOCK_N, BLOCK_K, num_stages, num_warps)}; tuned on H100 SXM (scratch x1/tma.py),
# dependent kernel before each GEMM, CUDA graph replay, cold weights.
_CFG = {
    (17408, 5120, True): {16: (64, 128, 6, 4), 32: (128, 64, 3, 4), 64: (64, 64, 3, 4)},  # gate_up + SiLU*mul
    (5120, 17408, False): {16: (64, 256, 4, 4), 32: (64, 256, 4, 4), 64: (64, 128, 6, 4)},  # down_proj
    (16384, 5120, False): {16: (64, 128, 3, 4), 32: (64, 128, 3, 4), 64: (64, 128, 3, 4)},  # in_proj_qkvz
    (16480, 5120, False): {16: (64, 128, 3, 4), 32: (64, 128, 3, 4), 64: (64, 128, 3, 4)},  # qkvz + ba folded
    (5120, 6144, False): {16: (64, 256, 4, 4), 32: (64, 128, 6, 4), 64: (64, 128, 6, 4)},  # out_proj / o_proj
    (14336, 5120, False): {16: (64, 128, 3, 4), 32: (64, 128, 3, 4), 64: (64, 128, 3, 4)},  # attention qkv(+gate)
}
_MAX_M = 64


@triton.jit
def _x1_tma_gemm_kernel(x_ptr, w_desc, out_ptr, M, NOUT, K, stride_xm, stride_om,
                        BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, BM: tl.constexpr, SILU: tl.constexpr):
    pid = tl.program_id(0)
    n0 = pid * BLOCK_N
    offs_m = tl.arange(0, BM)
    offs_k = tl.arange(0, BLOCK_K)
    mmask = offs_m[None, :] < M
    x_ptrs = x_ptr + offs_m[None, :] * stride_xm + offs_k[:, None]
    acc = tl.zeros((BLOCK_N, BM), dtype=tl.float32)
    acc_u = tl.zeros((BLOCK_N, BM), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        xt = tl.load(x_ptrs, mask=mmask, other=0.0)
        w = w_desc.load([n0, k])
        acc = tl.dot(w, xt, acc)
        if SILU:
            u = w_desc.load([n0 + NOUT, k])
            acc_u = tl.dot(u, xt, acc_u)
        x_ptrs += BLOCK_K
    offs_n = n0 + tl.arange(0, BLOCK_N)
    o_ptrs = out_ptr + offs_m[None, :] * stride_om + offs_n[:, None]
    if SILU:
        g = acc.to(tl.bfloat16).to(tl.float32)
        uu = acc_u.to(tl.bfloat16).to(tl.float32)
        res = g * tl.sigmoid(g) * uu
    else:
        res = acc
    tl.store(o_ptrs, res.to(tl.bfloat16), mask=mmask & (offs_n[:, None] < NOUT))


_DESC_CACHE: dict = {}


def _bm_for(m: int) -> int:
    return 16 if m <= 16 else (32 if m <= 32 else 64)


def _desc(w: torch.Tensor, bn: int, bk: int):
    from triton.tools.tensor_descriptor import TensorDescriptor

    key = (w.data_ptr(), tuple(w.shape), bn, bk)
    d = _DESC_CACHE.get(key)
    if d is None:
        d = TensorDescriptor.from_tensor(w, [bn, bk])
        _DESC_CACHE[key] = d
    return d


def _eligible(x: torch.Tensor, w: torch.Tensor) -> bool:
    return (
        x.is_cuda
        and x.dtype == torch.bfloat16
        and w.dtype == torch.bfloat16
        and w.dim() == 2
        and w.is_contiguous()
        and x.shape[-1] == w.shape[1]
        and x.stride(-1) == 1
        and not torch.compiler.is_compiling()
    )


def x1_gemm_out(x2: torch.Tensor, w: torch.Tensor, nout: int, silu: bool) -> torch.Tensor:
    if _D2_PRE:  # SPECPILOT_D2: PDL weight pre-staging GEMM for the same shapes
        from sglang.kernels.ops.gemm.specpilot_d2_gemm import d2_eligible, d2_gemm_out

        if d2_eligible(x2, w, nout, silu):
            return d2_gemm_out(x2, w, nout, silu)
    m, k = x2.shape
    bn, bk, ns, nw = _CFG[(nout, k, silu)][_bm_for(m)]
    out = torch.empty((m, nout), dtype=x2.dtype, device=x2.device)
    _x1_tma_gemm_kernel[(triton.cdiv(nout, bn),)](
        x2, _desc(w, bn, bk), out, m, nout, k, x2.stride(0), out.stride(0),
        BLOCK_N=bn, BLOCK_K=bk, BM=_bm_for(m), SILU=silu, num_warps=nw, num_stages=ns,
    )
    return out


def x1_try_linear(x: torch.Tensor, w: torch.Tensor) -> Optional[torch.Tensor]:
    """y = x @ w^T through the x1 kernel when the shape is tuned and M <= 64, else None (caller uses cuBLAS)."""
    if not _eligible(x, w):
        return None
    k = w.shape[1]
    m = x.numel() // k
    if m == 0 or m > _MAX_M or (w.shape[0], k, False) not in _CFG:
        return None
    x2 = x.reshape(m, k)
    if x2.stride(-1) != 1 or x2.stride(0) % 8:
        return None
    return x1_gemm_out(x2, w, w.shape[0], False).view(*x.shape[:-1], w.shape[0])


def x1_try_gate_up_silu(x: torch.Tensor, w: torch.Tensor) -> Optional[torch.Tensor]:
    """h = SiLU(x @ w[:I]^T) * (x @ w[I:]^T) for a MergedColumn gate_up weight [2I, K], else None."""
    if not _eligible(x, w) or w.shape[0] % 2:
        return None
    k = w.shape[1]
    inter = w.shape[0] // 2
    m = x.numel() // k
    if m == 0 or m > _MAX_M or (inter, k, True) not in _CFG:
        return None
    x2 = x.reshape(m, k)
    if x2.stride(-1) != 1 or x2.stride(0) % 8:
        return None
    return x1_gemm_out(x2, w, inter, True).view(*x.shape[:-1], inter)
