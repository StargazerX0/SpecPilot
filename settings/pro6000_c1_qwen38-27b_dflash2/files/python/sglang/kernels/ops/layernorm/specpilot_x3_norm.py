# SPECPILOT_X3: small-M Gemma fused add + RMSNorm (one program per row, PDL).
"""residual <- input + residual (fp32 sum stored in the residual dtype);
input <- (sum * rsqrt(mean(sum^2) + eps)) * (1 + weight), all on the fp32 sum,
the semantics of flashinfer's gemma_fused_add_rmsnorm. The weight load is issued
before the PDL wait so it overlaps the previous kernel's tail."""

import os

import torch
import triton
import triton.language as tl

from sglang.kernels.jit.utils import is_arch_support_pdl

# Rows above this keep the stock kernel; 0 (unset) disables the path.
X3_NORM_MAX_ROWS = int(os.environ.get("SPECPILOT_X3_NORM_ROWS", "0") or 0)
X3_NORM_WARPS = int(os.environ.get("SPECPILOT_X3_NORM_WARPS", "8"))
# SPECPILOT_D2: trigger the dependent (next GEMM, PDL weight pre-staging) at kernel start
# SPECPILOT_D2: SPECPILOT_D2_EARLY bitmask (1 add-norm, 2 gated norm, 4 WY, 8 attention)
D2_EARLY = bool(int(__import__("os").environ.get("SPECPILOT_D2_EARLY", "0").strip() or 0) & 1)


@triton.jit
def _x3_gemma_add_rmsnorm_kernel(
    x_ptr,
    r_ptr,
    w_ptr,
    stride_x,
    stride_r,
    eps,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
    USE_GDC: tl.constexpr = False,
    D2_EARLY: tl.constexpr = False,
):
    if D2_EARLY:  # SPECPILOT_D2
        tl.extra.cuda.gdc_launch_dependents()
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    m = cols < N
    w = tl.load(w_ptr + cols, m, 0.0).to(tl.float32)
    if USE_GDC:
        tl.extra.cuda.gdc_wait()
    x = tl.load(x_ptr + row * stride_x + cols, m, 0.0).to(tl.float32)
    r = tl.load(r_ptr + row * stride_r + cols, m, 0.0).to(tl.float32)
    s = x + r
    tl.store(r_ptr + row * stride_r + cols, s.to(r_ptr.dtype.element_ty), m)
    var = tl.sum(s * s, axis=0) / N
    rstd = tl.rsqrt(var + eps)
    y = s * rstd * (1.0 + w)
    if USE_GDC:
        tl.extra.cuda.gdc_launch_dependents()
    tl.store(x_ptr + row * stride_x + cols, y.to(x_ptr.dtype.element_ty), m)


def x3_gemma_add_rmsnorm_eligible(x: torch.Tensor, residual: torch.Tensor) -> bool:
    return (
        0 < x.shape[0] <= X3_NORM_MAX_ROWS
        and x.dim() == 2
        and x.is_cuda
        and x.stride(-1) == 1
        and residual.stride(-1) == 1
        and x.shape == residual.shape
    )


def x3_gemma_add_rmsnorm(
    *, x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> None:
    M, N = x.shape
    pdl_kwargs = {"USE_GDC": True, "launch_pdl": True} if is_arch_support_pdl() else {}
    if D2_EARLY and pdl_kwargs:  # SPECPILOT_D2
        pdl_kwargs["D2_EARLY"] = True
    _x3_gemma_add_rmsnorm_kernel[(M,)](
        x,
        residual,
        weight,
        x.stride(0),
        residual.stride(0),
        eps,
        N=N,
        BLOCK=triton.next_power_of_2(N),
        num_warps=X3_NORM_WARPS,
        **pdl_kwargs,
    )
