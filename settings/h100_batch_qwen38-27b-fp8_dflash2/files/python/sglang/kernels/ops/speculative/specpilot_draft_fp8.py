"""SPECPILOT_K_DRAFT_FP8: draft-only block-scaled FP8 copies of the DFlash drafter MLP weights and their GEMM.

Weights get 128x128 block scales (amax / 448, e4m3), activations 1x128 group scales; the GEMM is the target's
deep_gemm block-FP8 path (w8a8_block_fp8_matmul_deepgemm). The drafter's bf16 weights stay as they are (other paths,
e.g. the fused draft-KV append, keep reading them); the target verify never sees these copies.
"""
from __future__ import annotations

import os

import torch

_K_DRAFT_FP8 = os.environ.get("SPECPILOT_K_DRAFT_FP8", "0") not in ("", "0")
_BLOCK = 128
_FP8_MAX = 448.0


def draft_fp8_enabled() -> bool:
    return _K_DRAFT_FP8


def quantize_block_fp8(weight: torch.Tensor):
    """[N, K] bf16 -> ([N, K] e4m3, [N/128, K/128] fp32 block scales)."""
    n, k = weight.shape
    assert n % _BLOCK == 0 and k % _BLOCK == 0, (n, k)
    wb = weight.detach().float().view(n // _BLOCK, _BLOCK, k // _BLOCK, _BLOCK)
    scale = (wb.abs().amax(dim=(1, 3)) / _FP8_MAX).clamp(min=1e-12)
    wq = (wb / scale[:, None, :, None]).clamp(-_FP8_MAX, _FP8_MAX).view(n, k).to(torch.float8_e4m3fn)
    return wq.contiguous(), scale.contiguous()


def block_fp8_linear(x: torch.Tensor, wq: torch.Tensor, ws: torch.Tensor) -> torch.Tensor:
    """y = x @ W^T for bf16 x [..., K] with a block-FP8 weight copy; output bf16 [..., N]."""
    from sglang.kernels.ops.quantization.fp8_kernel import (
        sglang_per_token_group_quant_fp8,
        w8a8_block_fp8_matmul_deepgemm,
    )

    x2 = x.reshape(-1, x.shape[-1])
    if not x2.is_contiguous():
        x2 = x2.contiguous()
    xq, xs = sglang_per_token_group_quant_fp8(x2, _BLOCK, column_major_scales=True, scale_tma_aligned=True)
    y = w8a8_block_fp8_matmul_deepgemm(xq, wq, xs, ws, [_BLOCK, _BLOCK], torch.bfloat16)
    return y.view(*x.shape[:-1], wq.shape[0])


def prepare_mlp(mlp) -> None:
    """Attach FP8 copies of a DFlashMLP's gate_up / down weights (call after the weights are loaded)."""
    mlp._specpilot_fp8 = (
        quantize_block_fp8(mlp.gate_up_proj.weight),
        quantize_block_fp8(mlp.down_proj.weight),
    )


# SPECPILOT_X2: block-FP8 drafter attention projections and candidate head (read once; unset = lib draft_fp8 only).
_X2_ATTN = os.environ.get("SPECPILOT_X2_DFP8_ATTN", "0") not in ("", "0")
_X2_HEAD = os.environ.get("SPECPILOT_X2_DFP8_HEAD", "0") not in ("", "0")


def attn_fp8_enabled() -> bool:
    return _X2_ATTN


def head_fp8_enabled() -> bool:
    return _X2_HEAD


def prepare_attn(attn) -> None:
    """Attach FP8 copies of a DFlashAttention's fused qkv and o weights (bias-free, TP=1)."""
    if attn.qkv_proj.bias is not None or attn.o_proj.bias is not None:
        return
    attn._specpilot_x2_fp8 = (quantize_block_fp8(attn.qkv_proj.weight), quantize_block_fp8(attn.o_proj.weight))


def prepare_head(weight: torch.Tensor):
    """FP8 copy of the static candidate-vocabulary head rows [V_sub, H] (V_sub and H multiples of 128)."""
    return quantize_block_fp8(weight)
