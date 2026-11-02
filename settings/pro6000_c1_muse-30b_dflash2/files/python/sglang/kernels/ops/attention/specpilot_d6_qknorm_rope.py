# SPECPILOT_D6: Muse qk RMSNorm (+ RoPE) in one kernel; q is written to a new contiguous tensor.
"""Bitwise equal to stock's fused_inplace_qknorm -> apply_rope_with_cos_sin_cache_inplace -> q.contiguous()
(see csrc/elementwise/specpilot_d6_qknorm_rope.cuh). k is normalized (and rotated) in place, as stock."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import cache_once, is_arch_support_pdl, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module

HEAD_DIM = 128


@cache_once
def _module(use_rope: bool, is_neox: bool) -> Module:
    args = make_cpp_args(use_rope, is_neox, is_arch_support_pdl())
    return load_jit(
        "specpilot_d6_qknorm_rope",
        *args,
        cuda_files=["elementwise/specpilot_d6_qknorm_rope.cuh"],
        cuda_wrappers=[("run", f"D6QKNormRopeKernel<{args}>::run")],
    )


_DUMMY_CACHE = {}


def _dummy_cache(device: torch.device) -> torch.Tensor:
    t = _DUMMY_CACHE.get(device)
    if t is None:
        t = _DUMMY_CACHE[device] = torch.zeros((1, HEAD_DIM), dtype=torch.float32, device=device)
    return t


def eligible(q: torch.Tensor, k: torch.Tensor, head_dim: int) -> bool:
    return (
        head_dim == HEAD_DIM
        and q.dim() == 2
        and k.dim() == 2
        and q.shape[0] == k.shape[0]
        and q.shape[0] > 0
        and q.dtype == torch.bfloat16
        and k.dtype == torch.bfloat16
        and q.stride(-1) == 1
        and k.stride(-1) == 1
        and q.shape[1] % HEAD_DIM == 0
        and k.shape[1] % HEAD_DIM == 0
    )


def qknorm_rope(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
    positions: torch.Tensor,
    cos_sin_cache: Optional[torch.Tensor] = None,
    is_neox: bool = False,
) -> torch.Tensor:
    """q [T, HQ*D] (row-strided view ok), k [T, HKV*D] (modified in place). Returns new contiguous q [T, HQ*D].
    RoPE (positions [T] int32/int64, fp32 cos_sin_cache [max_pos, D]) is applied when ``cos_sin_cache`` is given."""
    T = q.shape[0]
    q3 = q.view(T, -1, HEAD_DIM)
    k3 = k.view(T, -1, HEAD_DIM)
    q_out = torch.empty((T, q3.shape[1], HEAD_DIM), dtype=q.dtype, device=q.device)
    use_rope = cos_sin_cache is not None
    if not use_rope:
        cos_sin_cache = _dummy_cache(q.device)
    _module(use_rope, bool(is_neox)).run(q3, k3, q_out, q_weight, k_weight, cos_sin_cache, positions, eps)
    return q_out.view(T, -1)
