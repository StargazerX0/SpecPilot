"""SPECPILOT_D2_MASKED: route listed (N, K, M) cells of the block-FP8 linear to DeepGEMM's 1-group masked grouped GEMM.

The plain fp8_gemm_nt heuristic picks 64-row tiles for in_proj_qkvz (16384 x 5120) at M > 160 (2x slower than the
weight read); the masked grouped kernel at a fixed expected_m takes a better config. Same inputs and promotion order:
outputs are bitwise identical to the plain kernel.

Knob (read once at import; unset = the parent's plain DeepGEMM path):
  SPECPILOT_D2_MASKED="NxK:mlo-mhi:expected_m[,...]"   e.g. 16384x5120:161-256:256
"""
from __future__ import annotations

import os
from typing import Optional

import torch


def _parse(spec: str) -> dict:
    cells = {}
    for item in filter(None, (s.strip() for s in spec.split(","))):
        nk, mr, em = item.split(":")
        n, k = (int(v) for v in nk.split("x"))
        lo, hi = (int(v) for v in mr.split("-"))
        cells.setdefault((n, k), []).append((lo, hi, int(em)))
    return cells


D2_MASKED = _parse(os.environ.get("SPECPILOT_D2_MASKED", ""))
_MAX_M = 1024
_masked_m: dict = {}


def prepare(device: torch.device) -> None:
    """Device arange holding every masked_m value, created outside CUDA-graph capture."""
    if D2_MASKED and device not in _masked_m:
        _masked_m[device] = torch.arange(_MAX_M + 1, dtype=torch.int32, device=device)


def expected_m_for(n: int, k: int, m: int) -> Optional[int]:
    for lo, hi, em in D2_MASKED.get((n, k), ()):
        if lo <= m <= hi and m <= min(em, _MAX_M):
            return em
    return None


def masked_gemm(q: torch.Tensor, s: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor,
                expected_m: int) -> torch.Tensor:
    """q [M, K] fp8, s [M, K/128] fp32 MN-major TMA-aligned (the engine's quant layout) -> [M, N] bf16."""
    import deep_gemm

    m, k = q.shape
    out = torch.empty((1, m, weight.shape[0]), dtype=torch.bfloat16, device=q.device)
    mm = _masked_m[q.device][m:m + 1]
    deep_gemm.fp8_m_grouped_gemm_nt_masked(
        (q.view(1, m, k), s.unsqueeze(0)), (weight.unsqueeze(0), weight_scale.unsqueeze(0)), out, mm, expected_m)
    return out[0]
