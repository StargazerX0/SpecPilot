"""SPECPILOT_D3R3_WSMMA: efficiency variants of the seed's weight-stationary FP8 block-scaled GEMM (specpilot_d2_fp8)
for the packed verify at M 129..192 (3 math warpgroups x 64 tokens, data-parallel grid).

Kernel source ``specpilot_d3r3_wsmma.cu`` (same directory). Every variant computes the seed kernel's result bit for bit
(same per-k-block wgmma sums, same DeepGEMM-order fp32 promotion); variants differ in
  W, X   depths of the separate weight (HBM) and token (L2) smem rings (X0 = the seed's coupled [W | X] stage),
  P      0 = the seed's round-robin wgmma issue order across math warpgroups, 2 = no issue order,
         1 = split-N software pipeline per math warpgroup (promote one column half while the other half's MMA runs),
  D      diagnostics (timing only, wrong numerics), bit flags: 1 no wgmma, 2 no TMA loads, 4 no promotion.
A variant is named like ``W5X3P1`` (+ ``D1``/``D2``/``D4``).

Knob (read once at import; unset = the parent's d2 path). The engine hook (fp8_utils) routes a call here only where
the parent's SPECPILOT_D2_GEMM path serves it, so every routed output is bitwise the parent's:
  SPECPILOT_D3R3_WSMMA="NxK:lo-hi:VARIANT[,...]"  e.g. 34816x5120:129-192:W4X0P2,16384x5120:129-192:W5X0P2
  SPECPILOT_D3R3_WSMMA=1                          = ROUTED below (the seed's coupled stage, unordered wgmma issue;
                                                  microbench 7aa8e1f36203)
"""
from __future__ import annotations

import hashlib
import os
import re
from typing import Optional

import torch

_SRC_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "specpilot_d3r3_wsmma.cu")
_CPP = ("void d3r3w_gemm(torch::Tensor xq, torch::Tensor xs, torch::Tensor w, torch::Tensor ws, torch::Tensor out, "
        "int64_t code, int64_t G);")
_VAR = re.compile(r"^W(\d+)X(\d+)P([012])(?:D([1-7]))?$")
_mod = None


ROUTED = "34816x5120:129-192:W4X0P2,16384x5120:129-192:W5X0P2,14336x5120:129-192:W5X0P2"


def _parse(spec: str) -> dict:
    if spec.strip() == "1":
        spec = ROUTED
    cells = {}
    for item in filter(None, (s.strip() for s in spec.split(","))):
        nk, mr, var = item.split(":")
        n, k = (int(v) for v in nk.split("x"))
        lo, hi = (int(v) for v in mr.split("-"))
        assert 129 <= lo <= hi <= 192 and _VAR.match(var), f"SPECPILOT_D3R3_WSMMA: bad cell {item}"
        cells.setdefault((n, k), []).append((lo, hi, var))
    return cells


WSMMA = _parse(os.environ.get("SPECPILOT_D3R3_WSMMA", ""))
# d1r3 WSTAIL's tail-piece layout in the d2 kernel beats P2 on 136-row tiles (gate_up, microbench 191cf23ce534),
# so with it on those calls stay on d2; 128-row tiles (qkvz, qkv) keep the wsmma cell.
YIELD_TAIL = os.environ.get("SPECPILOT_D1R3_WSTAIL", "0") == "1"


def _ext():
    global _mod
    if _mod is None:
        import deep_gemm
        from torch.utils.cpp_extension import load_inline

        src = open(_SRC_PATH).read()
        cflags = ["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-std=c++20", "--expt-relaxed-constexpr", "-DNDEBUG"]
        key = hashlib.sha1("\0".join([_CPP, src, *cflags]).encode()).hexdigest()[:16]
        base = os.environ.get("SGLANG_CACHE_DIR") or os.path.join(os.path.expanduser("~"), ".cache", "sglang")
        bdir = os.path.join(base, "specpilot_d3r3_wsmma", key)
        os.makedirs(bdir, exist_ok=True)
        _mod = load_inline(
            name=f"specpilot_d3r3_wsmma_{key}", cpp_sources=_CPP, cuda_sources=src, functions=["d3r3w_gemm"],
            extra_include_paths=[os.path.join(os.path.dirname(deep_gemm.__file__), "include")],
            extra_cuda_cflags=cflags, build_directory=bdir, verbose=False)
    return _mod


def prepare(device: torch.device) -> None:
    """Build the extension outside any CUDA-graph capture."""
    if WSMMA:
        _ext()


def tile_rows(n: int, sms: int) -> int:
    """The seed's plan at M 129..192: 136-row tiles when N splits into 2 tiles per CTA (gate_up 34816), else 128."""
    return 136 if (n % 136 == 0 and n // 128 > sms) else 128


def code(variant: str, n: int, sms: int) -> tuple:
    """(variant code, grid) for weight rows n; data-parallel only (1 or 2 tiles per CTA, as the seed's plan)."""
    m = _VAR.match(variant)
    assert m, variant
    ws, xs, pipe, diag = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4) or 0)
    bn = tile_rows(n, sms)
    tiles = n // bn
    if tiles <= sms:
        grid, ded = tiles, False
    else:
        assert tiles % 2 == 0 and tiles // 2 <= sms, "data-parallel grid only"
        grid, ded = tiles // 2, True
    return diag + 10 * pipe + 100 * xs + 1000 * ws + 100000 * (bn == 136) + 1000000 * ded, grid


def eligible(m: int, weight: torch.Tensor, weight_scale: torch.Tensor) -> Optional[str]:
    """The variant serving this call, or None (the caller falls back to the parent's path)."""
    n, k = weight.shape
    cells = WSMMA.get((n, k), ())
    if cells and YIELD_TAIL and tile_rows(n, _device_sms(weight.device)) == 136:
        return None
    for lo, hi, var in cells:
        if lo <= m <= hi and weight.dtype == torch.float8_e4m3fn and weight.is_contiguous() \
                and weight_scale.shape == (n // 128, k // 128) and weight_scale.stride(1) == 1:
            return var
    return None


_sms: dict = {}


def _device_sms(device: torch.device) -> int:
    dev = device.index if device.index is not None else torch.cuda.current_device()
    sms = _sms.get(dev)
    if sms is None:
        sms = _sms[dev] = torch.cuda.get_device_properties(dev).multi_processor_count
    return sms


def wsmma_gemm(xq: torch.Tensor, xs: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor, variant: str,
               out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """out = (xq * xs) @ (weight * weight_scale)^T, bitwise the seed's d2_fp8_gemm at 129 <= M <= 192."""
    m, n = xq.shape[0], weight.shape[0]
    assert 129 <= m <= 192, "served token range is 129..192"
    if out is None:
        out = torch.empty((m, n), dtype=torch.bfloat16, device=xq.device)
    c, grid = code(variant, n, _device_sms(xq.device))
    _ext().d3r3w_gemm(xq, xs, weight, weight_scale, out, c, grid)
    return out
