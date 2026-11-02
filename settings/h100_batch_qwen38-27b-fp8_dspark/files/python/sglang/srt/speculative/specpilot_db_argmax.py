"""SPECPILOT_DB_RA_ARGMAX: two-stage row argmax of the verify logits (budget family, db-3).

torch.argmax over [T, V] (V = 248320, bf16) runs at ~2.7x its byte floor at decode-verify sizes. Stage 1: one program
per (row, BLOCK-wide vocab chunk) -> chunk max and the first index reaching it; stage 2: one program per row -> the
first chunk index among the chunks reaching the global max. Comparisons are exact on the values widened to fp32, so
the result equals torch.argmax, including first-index tie-breaking (NaN-free logits).
"""
from __future__ import annotations

import functools
import os

import torch
import triton
import triton.language as tl

_BIG = 2147483647


@functools.lru_cache(maxsize=None)
def argmax_enabled() -> bool:
    return os.environ.get("SPECPILOT_DB_RA_ARGMAX", "").strip() not in ("", "0", "off")


@triton.jit
def _argmax_stage1(x_ptr, pm_ptr, pi_ptr, V, stride, NCH, BLOCK: tl.constexpr):
    r = tl.program_id(0).to(tl.int64)
    c = tl.program_id(1)
    offs = c * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + r * stride + offs, mask=offs < V, other=float("-inf")).to(tl.float32)
    m = tl.max(x, 0)
    i = tl.min(tl.where(x == m, offs, 2147483647), 0)
    tl.store(pm_ptr + r * NCH + c, m)
    tl.store(pi_ptr + r * NCH + c, i)


@triton.jit
def _argmax_stage2(pm_ptr, pi_ptr, out_ptr, NCH, NCHP: tl.constexpr):
    r = tl.program_id(0).to(tl.int64)
    cs = tl.arange(0, NCHP)
    m = tl.load(pm_ptr + r * NCH + cs, mask=cs < NCH, other=float("-inf"))
    i = tl.load(pi_ptr + r * NCH + cs, mask=cs < NCH, other=2147483647)
    mm = tl.max(m, 0)
    tl.store(out_ptr + r, tl.min(tl.where(m == mm, i, 2147483647), 0).to(tl.int64))


def row_argmax(x: torch.Tensor, block: int = 8192) -> torch.Tensor:
    """argmax over the last dim of a 2D [T, V] tensor (unit stride along V); int64 [T]."""
    assert x.dim() == 2 and x.stride(1) == 1
    T, V = x.shape
    out = torch.empty((T,), dtype=torch.int64, device=x.device)
    if T == 0:
        return out
    nch = triton.cdiv(V, block)
    pm = torch.empty((T * nch,), dtype=torch.float32, device=x.device)
    pi = torch.empty((T * nch,), dtype=torch.int32, device=x.device)
    _argmax_stage1[(T, nch)](x, pm, pi, V, x.stride(0), nch, BLOCK=block, num_warps=8)
    _argmax_stage2[(T,)](pm, pi, out, nch, NCHP=triton.next_power_of_2(nch), num_warps=4)
    return out
