"""SPECPILOT_D7: drafter-only INT4 weights (W4A16) for the DFlash2 drafter at batch-1 verify widths.

The drafter only proposes tokens (the target verifies every one), so any approximation of the drafter is lossless
for the output and is judged by tokens/verify against the bytes it saves. At batch 1 the draft graph streams its
weights once per step (FP8 W8A16 in the parent: ~2.2 GB, ~0.9 ms); this module derives, at load, symmetric INT4
copies with one fp32 scale per group of GS consecutive input channels (round-to-nearest with a per-group
MSE-optimal clipping factor), and multiplies them with bf16 activations in one Triton kernel:

- packing: for every group (K tile) of GS channels, byte j (j < GS/2) holds q[k0 + j] in its low nibble and
  q[k0 + GS/2 + j] in its high nibble, so the kernel splits one byte tile into two operands without any interleave;
- dequant in registers by PTX (prmt + lop3: nibble | 0x4300 = bf16(128 + q), then one bf16x2 fma to q - 8, exact),
  swap-AB (weights are the MMA's M side, the 16 drafter rows its N side); per group the two dots chain into one
  fp32 tile that is scaled once: acc += s_g * dot(x_g, q_g - 8);
- narrow-N shapes split K across programs; the last program of an N tile (acq_rel ticket) sums the fp32 partials
  in a fixed order (deterministic, one launch, no separate reduce kernel);
- the tree-verify row cut of the parent (N verify rows per request, drafter rows < B used) is done in-kernel: only
  rows < B are computed and the other rows are stored as zeros (no zero-fill / copy kernels).

Knobs (read once at model init; all unset = parent behaviour):
  SPECPILOT_D7_W4      comma (or + / ;) list of drafter parts in INT4: mlp (gate_up + down), qkv, o, conv (kernel_projection),
                       head (the derived candidate head), fc (draft-KV append projection); "all" = every part
  SPECPILOT_D7_W8      parts (not in W4) on the FP8 split-K / row-cut twin kernel (parent's fp8 copies)
  SPECPILOT_D7_W4_G    group size in channels (256 default; 128)
  SPECPILOT_D7_W4_FMT  sym (default: q in [-8, 7], one scale per group = one K tile) | asym (scale + min per group,
                       two groups per K tile, fp32 min term m * sum(x) in the epilogue)
  SPECPILOT_D7_HEAD_FMT  candidate-head format when it differs from the body (sym -> group 256, asym -> 128)
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

MAX_M = 64
_PARTS = ("mlp", "qkv", "o", "conv", "head", "fc")


def _parse_parts(name: str) -> set:
    raw = os.environ.get(name, "").strip().lower()
    if not raw or raw == "0":
        return set()
    parts = set(_PARTS) if raw == "all" else {p.strip() for p in raw.replace(";", ",").replace("+", ",").split(",") if p.strip()}
    bad = parts - set(_PARTS)
    if bad:
        raise ValueError(f"{name}: unknown parts {sorted(bad)} (known: {_PARTS})")
    return parts


def read_d7_knobs() -> Optional[dict]:
    """None = parent. 'w4' parts take INT4; 'w8' parts (not in w4) take the FP8 split-K / row-cut kernel."""
    w4, w8 = _parse_parts("SPECPILOT_D7_W4"), _parse_parts("SPECPILOT_D7_W8")
    if not w4 and not w8:
        return None
    g = int(os.environ.get("SPECPILOT_D7_W4_G", "256") or 256)
    if g not in (128, 256):
        raise ValueError("SPECPILOT_D7_W4_G must be 128 or 256")
    fmt = (os.environ.get("SPECPILOT_D7_W4_FMT", "") or "sym").strip().lower()
    if fmt not in ("sym", "asym"):
        raise ValueError("SPECPILOT_D7_W4_FMT must be sym or asym")
    head_fmt = (os.environ.get("SPECPILOT_D7_HEAD_FMT", "") or fmt).strip().lower()
    if head_fmt not in ("sym", "asym"):
        raise ValueError("SPECPILOT_D7_HEAD_FMT must be sym or asym")
    out = {"parts": w4, "w8": w8 - w4, "group": g, "fmt": fmt}
    if head_fmt != fmt:
        out["head_fmt"] = head_fmt  # the head then uses group 256 (sym) / 128 (asym)
    return out


# ------------------------------------------------------------------------------------------------ quantization


def quantize_w4(weight: torch.Tensor, group: int = 256, chunk: int = 2048, grid=(1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7),
                row_ids: Optional[torch.Tensor] = None, asym: bool = False):
    """bf16 [N, K] (or the rows `row_ids` of it) -> (packed uint8 [N, K/2], scale fp32 [K/GS, N], minv fp32 [K/GS, N]
    or None). Packing tile T = GS (sym) or 2 GS (asym): byte j < T/2 of a tile = (q[k0 + j], q[k0 + T/2 + j]).
    sym: q = clamp(round(w / s), -8, 7) + 8, s = a * amax / 7; asym: q = clamp(round((w - m) / s), 0, 15) with the
    range [lo, hi] shrunk around its centre by a; a = the per-group MSE-best factor of `grid` (RTN, no data)."""
    k = weight.shape[1]
    n = weight.shape[0] if row_ids is None else int(row_ids.numel())
    tile = 2 * group if asym else group
    assert k % tile == 0, (k, tile)
    ng = k // group
    packed = torch.empty((n, k // 2), device=weight.device, dtype=torch.uint8)
    scale = torch.empty((ng, n), device=weight.device, dtype=torch.float32)
    minv = torch.empty((ng, n), device=weight.device, dtype=torch.float32) if asym else None
    for r0 in range(0, n, chunk):
        rows = weight[r0:r0 + chunk] if row_ids is None else weight.index_select(0, row_ids[r0:r0 + chunk])
        w = rows.float().view(-1, ng, group)  # [c, ng, GS]
        if asym:
            lo, hi = w.amin(-1, keepdim=True), w.amax(-1, keepdim=True)
            mid, half = (lo + hi) * 0.5, (hi - lo) * 0.5
        else:
            amax = w.abs().amax(-1, keepdim=True).clamp_min(1e-10)
        best_err = best_s = best_m = None
        for a in grid:
            if asym:
                m = mid - half * a
                sc = (2 * half * a / 15.0).clamp_min(1e-10)
                q = ((w - m) / sc).round_().clamp_(0, 15)
                err = ((q * sc + m - w) ** 2).sum(-1, keepdim=True)
            else:
                m = None
                sc = amax * (a / 7.0)
                q = (w / sc).round_().clamp_(-8, 7)
                err = ((q * sc - w) ** 2).sum(-1, keepdim=True)
            if best_err is None:
                best_err, best_s, best_m = err, sc, m
            else:
                take = err < best_err
                best_err = torch.where(take, err, best_err)
                best_s = torch.where(take, sc, best_s)
                if asym:
                    best_m = torch.where(take, m, best_m)
        if asym:
            q = ((w - best_m) / best_s).round_().clamp_(0, 15)
        else:
            q = (w / best_s).round_().clamp_(-8, 7) + 8
        q = q.to(torch.uint8).view(w.shape[0], k // tile, 2, tile // 2)
        packed[r0:r0 + chunk] = (q[:, :, 0] | (q[:, :, 1] << 4)).reshape(w.shape[0], k // 2)
        scale[:, r0:r0 + chunk] = best_s.view(-1, ng).t()
        if asym:
            minv[:, r0:r0 + chunk] = best_m.view(-1, ng).t()
    return packed, scale, minv


def dequantize_w4(packed: torch.Tensor, scale: torch.Tensor, group: int, minv: Optional[torch.Tensor] = None):
    """Reference: -> fp32 [N, K]."""
    n, kh = packed.shape
    k = kh * 2
    tile = group if minv is None else 2 * group
    p = packed.view(n, k // tile, tile // 2)
    q = torch.stack([(p & 0xF), (p >> 4)], dim=2).view(n, k // group, group).float()
    if minv is None:
        return ((q - 8.0) * scale.t().reshape(n, -1, 1)).view(n, k)
    return (q * scale.t().reshape(n, -1, 1) + minv.t().reshape(n, -1, 1)).view(n, k)


# ------------------------------------------------------------------------------------------------ kernel


# nibble -> bf16 (q - 8) for 4 packed bytes per asm call (pack=4): prmt spreads bytes (b0, b1) / (b2, b3) to the two
# halves of a 32-bit word, lop3 keeps one nibble per half and ORs the bf16 exponent of 128 (0x4300 | q = 128 + q),
# the shift selects the high nibbles, and one bf16x2 fma (x * 1 - 136) gives q - 8 exactly. Out 0 = low nibbles of
# the 4 bytes, out 1 = high nibbles (elementwise order kept).
_UNPACK_ASM = tl.constexpr("""
{
.reg .b32 t0, t1, s0, s1, a0, a1, a2, a3, one, n136;
mov.b32 one, 0x3F803F80;
mov.b32 n136, 0xC308C308;
prmt.b32 t0, $4, 0, 0x4140;
prmt.b32 t1, $4, 0, 0x4342;
lop3.b32 a0, t0, 0x000F000F, 0x43004300, 0xEA;
lop3.b32 a1, t1, 0x000F000F, 0x43004300, 0xEA;
shr.u32 s0, t0, 4;
shr.u32 s1, t1, 4;
lop3.b32 a2, s0, 0x000F000F, 0x43004300, 0xEA;
lop3.b32 a3, s1, 0x000F000F, 0x43004300, 0xEA;
fma.rn.bf16x2 $0, a0, one, n136;
fma.rn.bf16x2 $1, a1, one, n136;
fma.rn.bf16x2 $2, a2, one, n136;
fma.rn.bf16x2 $3, a3, one, n136;
}
""")


@triton.jit
def _unpack_nibbles(pk):
    return tl.inline_asm_elementwise(_UNPACK_ASM, "=r,=r,=r,=r,r", [pk], dtype=(tl.bfloat16, tl.bfloat16),
                                     is_pure=True, pack=4)


@triton.jit
def _d7_w4a16_kernel(x_ptr, w_ptr, s_ptr, m_ptr, out_ptr, part_ptr, cnt_ptr,
                     M, N, K, stride_xm, stride_om, NR, BR, P,
                     G: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, SPLIT: tl.constexpr,
                     TPS: tl.constexpr, ROWCUT: tl.constexpr, BLOCK_P: tl.constexpr, ASYM: tl.constexpr):
    """Swap-AB, G = bytes per row of a K tile (2G channels).
    sym:  acc[n, m] += s[t, n] * (dot(q_lo - 8, x_lo) + dot(q_hi - 8, x_hi))       (one group per tile)
    asym: acc[n, m] += s_lo * dot(q_lo - 8, x_lo) + m8_lo * sum(x_lo) + (hi likewise), m8 = min + 8 s"""
    pid_n = tl.program_id(0)
    ks = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_g = tl.arange(0, G)
    n_mask = offs_n < N
    m_mask = offs_m < M
    if ROWCUT:
        rows = (offs_m // BR) * NR + offs_m % BR
    else:
        rows = offs_m
    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    t0 = ks * TPS
    for t in range(t0, t0 + TPS):
        k0 = t * 2 * G
        pk = tl.load(w_ptr + offs_n[:, None] * (K // 2) + (t * G + offs_g)[None, :], mask=n_mask[:, None], other=0)
        ql, qh = _unpack_nibbles(pk)
        if ASYM:  # coalesced row-major x tiles: the group sums reduce along the contiguous axis
            xlr = tl.load(x_ptr + rows[:, None] * stride_xm + (k0 + offs_g)[None, :], mask=m_mask[:, None], other=0.0)
            xhr = tl.load(x_ptr + rows[:, None] * stride_xm + (k0 + G + offs_g)[None, :], mask=m_mask[:, None], other=0.0)
            xl = tl.trans(xlr)
            xh = tl.trans(xhr)
        else:
            xl = tl.load(x_ptr + rows[None, :] * stride_xm + (k0 + offs_g)[:, None], mask=m_mask[None, :], other=0.0)
            xh = tl.load(x_ptr + rows[None, :] * stride_xm + (k0 + G + offs_g)[:, None], mask=m_mask[None, :], other=0.0)
        if ASYM:
            sl = tl.load(s_ptr + (2 * t) * N + offs_n, mask=n_mask, other=0.0)
            sh = tl.load(s_ptr + (2 * t + 1) * N + offs_n, mask=n_mask, other=0.0)
            ml = tl.load(m_ptr + (2 * t) * N + offs_n, mask=n_mask, other=0.0)
            mh = tl.load(m_ptr + (2 * t + 1) * N + offs_n, mask=n_mask, other=0.0)
            sxl = tl.sum(xlr.to(tl.float32), axis=1)
            sxh = tl.sum(xhr.to(tl.float32), axis=1)
            acc += (tl.dot(ql, xl) * sl[:, None] + tl.dot(qh, xh) * sh[:, None]
                    + ml[:, None] * sxl[None, :] + mh[:, None] * sxh[None, :])
        else:
            tmp = tl.dot(ql, xl)
            tmp = tl.dot(qh, xh, tmp)
            acc += tmp * tl.load(s_ptr + t * N + offs_n, mask=n_mask, other=0.0)[:, None]
    out_mask = n_mask[:, None] & m_mask[None, :]
    if ROWCUT:
        if ks == 0:
            offs_p = tl.arange(0, BLOCK_P)
            prow = (offs_p // (NR - BR)) * NR + BR + offs_p % (NR - BR)
            tl.store(out_ptr + prow[None, :] * stride_om + offs_n[:, None],
                     tl.zeros((BLOCK_N, BLOCK_P), dtype=out_ptr.dtype.element_ty),
                     mask=n_mask[:, None] & (offs_p < P)[None, :])
    if SPLIT == 1:
        tl.store(out_ptr + rows[None, :] * stride_om + offs_n[:, None], acc.to(out_ptr.dtype.element_ty), mask=out_mask)
    else:
        pbase = part_ptr + ks * BLOCK_M * N
        tl.store(pbase + offs_m[None, :] * N + offs_n[:, None], acc, mask=out_mask)
        tl.debug_barrier()
        ticket = tl.atomic_add(cnt_ptr + pid_n, 1, sem="acq_rel", scope="gpu")
        if ticket == SPLIT - 1:
            tot = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
            for s in tl.static_range(SPLIT):
                tot += tl.load(part_ptr + s * BLOCK_M * N + offs_m[None, :] * N + offs_n[:, None], mask=out_mask,
                               other=0.0, cache_modifier=".cg")
            tl.store(out_ptr + rows[None, :] * stride_om + offs_n[:, None], tot.to(out_ptr.dtype.element_ty),
                     mask=out_mask)
            tl.atomic_xchg(cnt_ptr + pid_n, 0, sem="relaxed", scope="gpu")


@triton.jit
def _d7_w8a16_kernel(x_ptr, w_ptr, s_ptr, out_ptr, part_ptr, cnt_ptr,
                     M, N, K, stride_xm, stride_om, NR, BR, P,
                     BLOCK_K: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, SPLIT: tl.constexpr,
                     TPS: tl.constexpr, ROWCUT: tl.constexpr, BLOCK_P: tl.constexpr):
    """FP8 (per-row e4m3, the parent's copy) twin of the W4 kernel: swap-AB, serial split-K, in-kernel row cut."""
    pid_n = tl.program_id(0)
    ks = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    n_mask = offs_n < N
    m_mask = offs_m < M
    if ROWCUT:
        rows = (offs_m // BR) * NR + offs_m % BR
    else:
        rows = offs_m
    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    t0 = ks * TPS
    for t in range(t0, t0 + TPS):
        k0 = t * BLOCK_K
        w = tl.load(w_ptr + offs_n[:, None] * K + (k0 + offs_k)[None, :], mask=n_mask[:, None], other=0.0)
        x = tl.load(x_ptr + rows[None, :] * stride_xm + (k0 + offs_k)[:, None], mask=m_mask[None, :], other=0.0)
        acc += tl.dot(w.to(tl.bfloat16), x)
    sc = tl.load(s_ptr + offs_n, mask=n_mask, other=0.0)
    acc = acc * sc[:, None]
    out_mask = n_mask[:, None] & m_mask[None, :]
    if ROWCUT:
        if ks == 0:
            offs_p = tl.arange(0, BLOCK_P)
            prow = (offs_p // (NR - BR)) * NR + BR + offs_p % (NR - BR)
            tl.store(out_ptr + prow[None, :] * stride_om + offs_n[:, None],
                     tl.zeros((BLOCK_N, BLOCK_P), dtype=out_ptr.dtype.element_ty),
                     mask=n_mask[:, None] & (offs_p < P)[None, :])
    if SPLIT == 1:
        tl.store(out_ptr + rows[None, :] * stride_om + offs_n[:, None], acc.to(out_ptr.dtype.element_ty), mask=out_mask)
    else:
        pbase = part_ptr + ks * BLOCK_M * N
        tl.store(pbase + offs_m[None, :] * N + offs_n[:, None], acc, mask=out_mask)
        tl.debug_barrier()
        ticket = tl.atomic_add(cnt_ptr + pid_n, 1, sem="acq_rel", scope="gpu")
        if ticket == SPLIT - 1:
            tot = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
            for s in tl.static_range(SPLIT):
                tot += tl.load(part_ptr + s * BLOCK_M * N + offs_m[None, :] * N + offs_n[:, None], mask=out_mask,
                               other=0.0, cache_modifier=".cg")
            tl.store(out_ptr + rows[None, :] * stride_om + offs_n[:, None], tot.to(out_ptr.dtype.element_ty),
                     mask=out_mask)
            tl.atomic_xchg(cnt_ptr + pid_n, 0, sem="relaxed", scope="gpu")


# (N, K) -> split for the DFlash2 drafter of Qwen3.8-27B at group 256 (dev-loop grid scratch/d7/tune_w4.py, H100,
# 16 computed rows): gate_up 48.6 us, down 28.4, qkv 10.8, o 9.6, kernel_projection 5.8, head 118.3, fc 46.8.
_TUNED_SPLIT = {(34816, 5120): 2, (5120, 17408): 4, (6144, 5120): 4, (5120, 4096): 4, (1280, 5120): 10,
                (98304, 5120): 1, (5120, 25600): 5}


def pick_config(n: int, k: int, tile: int) -> dict:
    """(block_n, split, warps, stages) for M <= 64 on H100 (132 SMs); `tile` = K channels per loop step."""
    tiles = k // tile
    if tile == 256 and (n, k) in _TUNED_SPLIT:
        return {"block_n": 64, "split": _TUNED_SPLIT[(n, k)], "num_warps": 4, "num_stages": 3}
    block_n = 64
    ctas = triton.cdiv(n, block_n)
    split = 1
    for s in (2, 3, 4, 5, 6, 8):
        if ctas * split >= 528:
            break
        if tiles % s == 0 and tiles // s >= 2:
            split = s
    return {"block_n": block_n, "split": split, "num_warps": 4, "num_stages": 3}


class W4Linear:
    """INT4 copy of a bf16 [N, K] weight with the small-M kernel; bf16 fallback above MAX_M rows."""

    def __init__(self, weight: torch.Tensor, group: int = 256, cfg: Optional[dict] = None,
                 row_ids: Optional[torch.Tensor] = None, asym: bool = False):
        # bf16 kept for the large-M fallback (None for a row subset: the caller keeps its own fallback)
        self.weight = weight if row_ids is None else None
        self.n = weight.shape[0] if row_ids is None else int(row_ids.numel())
        self.k = weight.shape[1]
        self.group = group
        self.asym = bool(asym)
        self.tile = 2 * group if self.asym else group
        self.packed, self.scale, self.minv = quantize_w4(weight.data, group, row_ids=row_ids, asym=self.asym)
        self.minv8 = self.minv + 8.0 * self.scale if self.asym else None  # kernel offset for the (q - 8) operand
        self.cfg = dict(cfg or pick_config(self.n, self.k, self.tile))
        n_tiles = triton.cdiv(self.n, self.cfg["block_n"])
        self.counters = torch.zeros((n_tiles,), device=weight.device, dtype=torch.int32)
        # fp32 split-K partials, allocated once here (never inside a graph capture)
        self._part = (torch.empty((self.cfg["split"] * MAX_M * self.n,), device=weight.device, dtype=torch.float32)
                      if self.cfg["split"] > 1 else None)

    def nbytes(self) -> int:
        return self.packed.numel() + self.scale.numel() * 4 * (3 if self.asym else 1) + (self._part.numel() * 4 if self._part is not None else 0)

    def __call__(self, x2: torch.Tensor, row_cut: Optional[tuple] = None) -> torch.Tensor:
        """x2 [R, K] -> [R, N] bf16. row_cut=(NR, BR): R = bs*NR rows, compute rows i % NR < BR, zero the others."""
        rows_total = x2.shape[0]
        if row_cut is not None:
            nr, br = row_cut
            bs = rows_total // nr
            m = bs * br
            p = rows_total - m
        else:
            nr, br, m, p = 1, 1, rows_total, 0
        if m > MAX_M:
            return torch.nn.functional.linear(x2, self.weight)
        if x2.dtype != torch.bfloat16:
            x2 = x2.to(torch.bfloat16)
        if x2.stride(-1) != 1:
            x2 = x2.contiguous()
        out = torch.empty((rows_total, self.n), device=x2.device, dtype=torch.bfloat16)
        block_m = max(16, triton.next_power_of_2(max(m, 1)))
        split = self.cfg["split"]
        tiles = self.k // self.tile
        part = self._part if split > 1 else out
        grid = (triton.cdiv(self.n, self.cfg["block_n"]), split)
        _d7_w4a16_kernel[grid](
            x2, self.packed, self.scale, self.minv8 if self.asym else self.scale, out, part, self.counters,
            m, self.n, self.k, x2.stride(0), out.stride(0), nr, br, p,
            G=self.tile // 2, ASYM=self.asym, BLOCK_M=block_m, BLOCK_N=self.cfg["block_n"], SPLIT=split, TPS=tiles // split,
            ROWCUT=row_cut is not None, BLOCK_P=max(16, triton.next_power_of_2(max(p, 1))),
            num_warps=self.cfg["num_warps"], num_stages=self.cfg["num_stages"])
        return out


class W8Linear:
    """FP8 per-row (e4m3) small-M linear with serial split-K and in-kernel row cut (reuses an fp8 copy if given)."""

    def __init__(self, weight: torch.Tensor, qs=None, cfg: Optional[dict] = None, row_ids=None):
        from sglang.kernels.ops.speculative.dflash_de_head import quantize_rows_fp8

        self.weight = weight if row_ids is None else None
        if qs is None:
            rows = weight.data if row_ids is None else weight.data.index_select(0, row_ids)
            qs = quantize_rows_fp8(rows, chunk=2048)
            del rows
        self.q, self.s = qs
        self.n, self.k = self.q.shape
        self.cfg = dict(cfg or pick_config_w8(self.n, self.k))
        n_tiles = triton.cdiv(self.n, self.cfg["block_n"])
        self.counters = torch.zeros((n_tiles,), device=self.q.device, dtype=torch.int32)
        self._part = (torch.empty((self.cfg["split"] * MAX_M * self.n,), device=self.q.device, dtype=torch.float32)
                      if self.cfg["split"] > 1 else None)

    def __call__(self, x2: torch.Tensor, row_cut: Optional[tuple] = None) -> torch.Tensor:
        rows_total = x2.shape[0]
        if row_cut is not None:
            nr, br = row_cut
            m = rows_total // nr * br
            p = rows_total - m
        else:
            nr, br, m, p = 1, 1, rows_total, 0
        if m > MAX_M and self.weight is not None:
            return torch.nn.functional.linear(x2, self.weight)
        if x2.dtype != torch.bfloat16:
            x2 = x2.to(torch.bfloat16)
        if x2.stride(-1) != 1:
            x2 = x2.contiguous()
        out = torch.empty((rows_total, self.n), device=x2.device, dtype=torch.bfloat16)
        split, bk = self.cfg["split"], self.cfg["block_k"]
        _d7_w8a16_kernel[(triton.cdiv(self.n, self.cfg["block_n"]), split)](
            x2, self.q, self.s, out, self._part if split > 1 else out, self.counters,
            m, self.n, self.k, x2.stride(0), out.stride(0), nr, br, p,
            BLOCK_K=bk, BLOCK_M=max(16, triton.next_power_of_2(max(m, 1))), BLOCK_N=self.cfg["block_n"], SPLIT=split,
            TPS=self.k // bk // split, ROWCUT=row_cut is not None,
            BLOCK_P=max(16, triton.next_power_of_2(max(p, 1))),
            num_warps=self.cfg["num_warps"], num_stages=self.cfg["num_stages"])
        return out


def pick_config_w8(n: int, k: int) -> dict:
    bk = 256 if k % 256 == 0 else 128
    tiles = k // bk
    block_n = 64
    ctas = triton.cdiv(n, block_n)
    split = 1
    for s in (2, 3, 4, 5, 6, 8):
        if ctas * split >= 528:
            break
        if tiles % s == 0 and tiles // s >= 2:
            split = s
    return {"block_n": block_n, "block_k": bk, "split": split, "num_warps": 4, "num_stages": 4}


# ------------------------------------------------------------------------------------------------ engine wiring


def _row_cut(rows_total: int):
    """The parent's tree-verify row layout (dflash_de_head._X4_BLOCK_ROWS = (N, B)), read at call time (set by the
    worker before capture): (N, B) when the rows are whole requests and the computed rows fit the small-M kernel."""
    from sglang.kernels.ops.speculative import dflash_de_head

    nb = dflash_de_head._X4_BLOCK_ROWS
    if nb is None or rows_total % nb[0]:
        return None
    if rows_total // nb[0] * nb[1] > MAX_M:
        return None
    return nb


def _patch_sglang_linear(mod, lin: W4Linear, cut: bool) -> None:
    def forward(x, _lin=lin, _cut=cut):
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        y = _lin(x2, _row_cut(x2.shape[0]) if _cut else None)
        return y.reshape(*shape[:-1], y.shape[-1]), None

    object.__setattr__(mod, "forward", forward)
    object.__setattr__(mod, "_d7_w4", lin)
    if hasattr(mod, "_de_fp8"):  # drop the parent's fp8 copy (its closure goes with the old forward)
        object.__delattr__(mod, "_de_fp8")


def _patch_torch_linear(mod, lin: W4Linear, cut: bool) -> None:
    def forward(x, _lin=lin, _cut=cut):
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        y = _lin(x2, _row_cut(x2.shape[0]) if _cut else None)
        return y.reshape(*shape[:-1], y.shape[-1])

    object.__setattr__(mod, "forward", forward)
    object.__setattr__(mod, "_d7_w4", lin)


def _ok_weight(mod, group: int) -> bool:
    w = getattr(mod, "weight", None)
    return (w is not None and w.dim() == 2 and w.dtype == torch.bfloat16 and getattr(mod, "bias", None) is None
            and getattr(mod, "tp_size", 1) == 1 and w.shape[1] % group == 0)


class _W4Head:
    """Replaces DEDraftHead.project with the INT4 copy of its kept rows (fp8 copy freed)."""

    def __init__(self, de_head, lm_head, group: int, asym: bool = False):
        weight = lm_head.weight
        ids = de_head.sub_to_full
        if ids is None:
            ids = torch.arange(int(lm_head.org_vocab_size), device=weight.device)
        self.lin = W4Linear(weight, group, row_ids=ids, asym=asym)
        self.de_head = de_head

    def project(self, hidden: torch.Tensor) -> torch.Tensor:
        h2 = hidden.reshape(-1, hidden.shape[-1])
        if h2.shape[0] > MAX_M:
            return torch.matmul(h2.to(torch.bfloat16), self.de_head_rows_t())
        return self.lin(h2)

    def de_head_rows_t(self):  # large-M fallback only (eager, outside the decode graphs): dequantized rows
        return dequantize_w4(self.lin.packed, self.lin.scale, self.lin.group, self.lin.minv).to(torch.bfloat16).t()


# SPECPILOT_D4_W4S_PARTS: parts the stream-K twin takes ("all" = every part). Unset: every part on sm_120, where W4S
# also beats the Triton kernel on the narrow Muse drafter shapes (qkv 21 vs 35 us, o 17 vs 26, conv 12 vs 27 at 16
# rows; microbench d0a2e84e1fd8); the wide parts mlp, fc, head elsewhere (the H100 measurement).
_W4S_PARTS_ENV = _parse_parts("SPECPILOT_D4_W4S_PARTS")


def _w4s_parts() -> set:
    if _W4S_PARTS_ENV:
        return _W4S_PARTS_ENV
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 12:
        return set(_PARTS)
    return {"mlp", "fc", "head"}


def _w4s(lin, part):
    """SPECPILOT_D5_W4S: the stream-K CUDA twin for the parts in _w4s_parts(); d7's object stays the fallback."""
    from sglang.kernels.ops.speculative import specpilot_d5_w4s as d5w4s

    if d5w4s.D5_W4S and part in _w4s_parts():
        return d5w4s.W4SLinear(lin)
    return lin


def _make(mod, parts_w4, parts_w8, part, g, asym=False):
    """W4Linear / W8Linear (reusing the parent's fp8 copy when present) / None for one module."""
    if part in parts_w4:
        return _w4s(W4Linear(mod.weight, g, asym=asym), part)
    if part in parts_w8:
        return W8Linear(mod.weight, qs=getattr(mod, "_de_fp8", None))
    return None


def apply_d7_w4_(model, lm_head, knobs: dict) -> list:
    """Attach INT4 / FP8-split-K forwards to the drafter parts named in the knobs (see module doc)."""
    w4p, w8p, g = knobs["parts"], knobs.get("w8", set()), knobs["group"]
    asym = knobs.get("fmt", "sym") == "asym"
    tile = 2 * g if asym else g
    allp = w4p | w8p
    patched = []
    with torch.no_grad():
        for i, layer in enumerate(model.layers):
            attn, mlp = getattr(layer, "self_attn", None), getattr(layer, "mlp", None)
            todo = [("gate_up_proj", getattr(mlp, "gate_up_proj", None), "mlp"),
                    ("down_proj", getattr(mlp, "down_proj", None), "mlp"),
                    ("qkv_proj", getattr(attn, "qkv_proj", None), "qkv"),
                    ("o_proj", getattr(attn, "o_proj", None), "o")]
            for name, mod, part in todo:
                if part in allp and mod is not None and _ok_weight(mod, tile):
                    lin = _make(mod, w4p, w8p, part, g, asym)
                    _patch_sglang_linear(mod, lin, cut=True)
                    patched.append(f"layers.{i}.{name}:{type(lin).__name__}")
            if "conv" in allp:
                for cname in ("attention_conv", "mlp_conv"):
                    kp = getattr(getattr(layer, cname, None), "kernel_projection", None)
                    if isinstance(kp, torch.nn.Linear) and _ok_weight(kp, tile):
                        lin = _make(kp, w4p, w8p, "conv", g, asym)
                        _patch_torch_linear(kp, lin, cut=True)
                        patched.append(f"layers.{i}.{cname}.kernel_projection:{type(lin).__name__}")
        fc = getattr(model, "fc", None)
        if "fc" in allp and isinstance(fc, torch.nn.Linear) and _ok_weight(fc, tile):
            lin = _make(fc, w4p, w8p, "fc", g, asym)
            _patch_torch_linear(fc, lin, cut=False)
            patched.append(f"fc:{type(lin).__name__}")
        de_head = getattr(getattr(model, "candidate_selector", None), "de_head", None)
        if "head" in allp and de_head is not None and lm_head is not None and lm_head.weight.shape[1] % tile == 0:
            if "head" in w4p:
                hf = knobs.get("head_fmt")
                if hf is None:
                    w4h = _W4Head(de_head, lm_head, g, asym)
                else:
                    w4h = _W4Head(de_head, lm_head, 256 if hf == "sym" else 128, hf == "asym")
                w4h.lin = _w4s(w4h.lin, "head")
                object.__setattr__(de_head, "project", w4h.project)
                object.__setattr__(de_head, "_d7_w4", w4h)
                de_head.weight, de_head.scale = None, None  # free the fp8 copy
                patched.append(f"head[{w4h.lin.n}]:W4")
            elif de_head.scale is not None:  # fp8 head of the parent -> split-free swap-AB twin
                lin8 = W8Linear(lm_head.weight, qs=(de_head.weight, de_head.scale), row_ids=de_head.sub_to_full)
                orig = de_head.project

                def proj(hidden, _l=lin8, _o=orig):
                    h2 = hidden.reshape(-1, hidden.shape[-1])
                    return _o(hidden) if h2.shape[0] > MAX_M else _l(h2)

                object.__setattr__(de_head, "project", proj)
                patched.append(f"head[{lin8.n}]:W8")
    torch.cuda.empty_cache()
    logger.info("SPECPILOT_D7 drafter linears (g=%d %s): %d patched (%s)", g, "asym" if asym else "sym", len(patched),
                ", ".join(patched[:8]))
    return patched
