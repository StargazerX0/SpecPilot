"""SPECPILOT_D2_GEMM: weight-stationary FP8 block-scaled GEMM for the packed verify (M <= 256 tokens).

out[M, N] = (xq * xs) @ (w * ws)^T with 1x128 activation scales (column-major, TMA-aligned, as
``sglang_per_token_group_quant_fp8(column_major_scales=True, scale_tma_aligned=True)`` writes them) and 128x128
weight block scales. One CTA owns BN weight rows for all tokens (tokens = wgmma M over 2-3 math warpgroups), so the
weight is read once (M <= 192: 2-3 warpgroups x 64 tokens over 128/136-row tiles; 193..256: 4 warpgroups over
64-row tiles). Kernel source ``specpilot_d2_fp8.cu`` (same directory; ``load_weight_tile`` is the weight
operand-load stage), JIT-built once per source hash. The grid is data-parallel when the tiles fill the SMs at most
twice (bitwise DeepGEMM's result), else stream-K with a fixed-order fp32 fix-up.

Knobs (read once at import; unset = the parent's DeepGEMM path):
  SPECPILOT_D2_GEMM="NxK:lo-hi[,...]"  weight shapes and token-count ranges served (hi <= 256),
                                       e.g. 16384x5120:129-192,14336x5120:129-192,34816x5120:129-192
  SPECPILOT_D1R3_WSTAIL=1              136-row tiles use the tail-piece layout (scale block t + 8 tail rows), so the
                                       promotion has a compile-time scale boundary (bitwise the same result)
"""
from __future__ import annotations

import hashlib
import os
from typing import Optional

import torch


def _parse(spec: str) -> dict:
    cells = {}
    for item in filter(None, (s.strip() for s in spec.split(","))):
        nk, mr = item.split(":")
        n, k = (int(v) for v in nk.split("x"))
        lo, hi = (int(v) for v in mr.split("-"))
        assert hi <= 256, "SPECPILOT_D2_GEMM: the kernel serves at most 256 tokens"
        cells.setdefault((n, k), []).append((lo, hi))
    return cells


D2_GEMM = _parse(os.environ.get("SPECPILOT_D2_GEMM", ""))
D1R3_WSTAIL = os.environ.get("SPECPILOT_D1R3_WSTAIL", "0") == "1"  # SPECPILOT_D1R3

_SRC_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "specpilot_d2_fp8.cu")
_CPP = ("void d2_gemm(torch::Tensor xq, torch::Tensor xs, torch::Tensor w, torch::Tensor ws, torch::Tensor out, "
        "torch::Tensor part, torch::Tensor cnt, int64_t bt, int64_t stages, int64_t G, int64_t cl);")
_mod = None
_ws: dict = {}


def _ext():
    global _mod
    if _mod is None:
        import deep_gemm
        from torch.utils.cpp_extension import load_inline

        src = open(_SRC_PATH).read()
        cflags = ["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-std=c++20", "--expt-relaxed-constexpr", "-DNDEBUG"]
        key = hashlib.sha1("\0".join([_CPP, src, *cflags]).encode()).hexdigest()[:16]
        base = os.environ.get("SGLANG_CACHE_DIR") or os.path.join(os.path.expanduser("~"), ".cache", "sglang")
        bdir = os.path.join(base, "specpilot_d2_fp8", key)
        os.makedirs(bdir, exist_ok=True)
        _mod = load_inline(
            name=f"specpilot_d2_fp8_{key}", cpp_sources=_CPP, cuda_sources=src, functions=["d2_gemm"],
            extra_include_paths=[os.path.join(os.path.dirname(deep_gemm.__file__), "include")],
            extra_cuda_cflags=cflags, build_directory=bdir, verbose=False)
    return _mod


def _workspace(device: torch.device):
    """(partial-tile slots, per-tile counters, SM count); allocated by prepare() before graph capture."""
    key = (device.type, device.index if device.index is not None else torch.cuda.current_device())
    w = _ws.get(key)
    if w is None:
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        w = (torch.empty(2 * sms * 136 * 256, dtype=torch.float32, device=device),
             torch.zeros(4096, dtype=torch.int32, device=device), sms)
        _ws[key] = w
    return w


def prepare(device: torch.device) -> None:
    """Build the extension and allocate the workspace outside any CUDA-graph capture."""
    if D2_GEMM:
        _ext()
        _workspace(device)


def eligible(m: int, weight: torch.Tensor, weight_scale: torch.Tensor) -> bool:
    n, k = weight.shape
    return (
        any(lo <= m <= hi for lo, hi in D2_GEMM.get((n, k), ()))
        and weight.dtype == torch.float8_e4m3fn
        and weight_scale.dtype == torch.float32
        and n % 128 == 0
        and k % 128 == 0
        and k // 128 <= 256
        and weight.is_contiguous()
        and weight_scale.shape == (n // 128, k // 128)
        and weight_scale.stride(1) == 1
    )


def _grid(tiles: int, sms: int) -> int:
    """Data-parallel grid with 1 or 2 tiles per CTA when possible, else stream-K over all SMs."""
    if tiles <= sms:
        return tiles
    if tiles % 2 == 0 and tiles // 2 <= sms:
        return tiles // 2
    return sms


def _plan(n: int, m: int, sms: int):
    """(token tile, variant code, grid). Variant code = stages + 100 (136-row tiles) / 200 (64-row tiles) + 10000
    (dedicated output staging, used when a CTA owns 2 tiles so the producer runs ahead into the second)."""
    if m <= 192:
        bt = 128 if m <= 128 else 192
        bn = 136 if (n % 136 == 0 and n // 128 > sms) else 128
        tiles = n // bn
        grid = _grid(tiles, sms)
        tail = 20000 if (D1R3_WSTAIL and bn == 136) else 0  # SPECPILOT_D1R3
        if bt == 192 and grid * 2 == tiles:
            return bt, 4 + (100 if bn == 136 else 0) + 10000 + tail, grid
        return bt, 5 + (100 if bn == 136 else 0) + tail, grid
    tiles = n // 64
    grid = _grid(tiles, sms)
    return 256, (10204 if grid * 2 == tiles else 205), grid


def d2_fp8_gemm(xq: torch.Tensor, xs: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor,
                out: Optional[torch.Tensor] = None) -> torch.Tensor:
    m = xq.shape[0]
    n = weight.shape[0]
    if out is None:
        out = torch.empty((m, n), dtype=torch.bfloat16, device=xq.device)
    part, cnt, sms = _workspace(xq.device)
    bt, variant, grid = _plan(n, m, sms)
    _ext().d2_gemm(xq, xs, weight, weight_scale, out, part, cnt, bt, variant, grid, 1)
    return out
