"""SPECPILOT_D2: W12 lossless 12-bit weight streaming for the target linears on sm_120 (RTX PRO 6000 Blackwell).

Record format = lib w12_nest (bit-identical encoder): per (128-row tile, 64-k chunk) a 12 KB record = Q region
[8 warps][1 KB] (byte [c0 m6 m5 m4 | s c3 c2 c1], A-fragment lane order) + B region [8 warps][512 B] ([m3..m0]);
w = s * 2^(c-15) * 1.m * 2^(E_n-127) with a per-row 16-binade window top E_n; out-of-window weights (zeros,
denormals, outliers) decode as 2^-15 and are corrected exactly from a CSR list (x*w~_true - x*2^-15).
GEMM (M <= 64): persistent stream-K kernel, 1 producer warp (cp.async.bulk 12 KB records + 2-D TMA SW128 x stages,
2-3 stage ring), 8 consumer warps decoding in registers straight into mma.sync m16n8k16 A fragments (swap-AB: weight
rows as the M=16 side, tokens as N=8 tiles via ldmatrix.x4), 2 escape warps (double-buffered smem corrections), fp32
accumulation, deterministic split-K reduction (the owner of a tile's k=0 segment sums the partials in segment
order), fused SiLU(gate)*up for gate_up, PDL (weight prologue before griddepcontrol.wait; dependents triggered at
start, every PDL dependent is a W12 kernel that waits before reading). M > 64: 64-row chunks up to 128, else the
weight is decoded into a shared bf16 scratch and cuBLAS runs.

Knob (read once at import): SPECPILOT_D2_W12=1 encodes the target's dense-MLP gate_up/down, the GDN in_proj
(qkvz + ba, folded) and the attention qkv_proj at load (bit-exact roundtrip enforced; the bf16 weights are freed);
unset = stock, untouched. SPECPILOT_D2_PDL (default 1) = programmatic dependent launch of the W12 kernels.
Sub-knobs (d2-003 / d2-004; all unset = d2-002 behaviour):
  SPECPILOT_D2_W12_OUT=1      also the GDN out_proj and the attention o_proj (5120 x 6144)
  SPECPILOT_D2_W12_HEAD=1     also the target lm_head (vocab x hidden): target logits (bf16, as stock) and the DFlash2
                              drafter's candidate projection; never decoded whole (M > 64 without PREFILL: 64-row chunks)
  SPECPILOT_D2_W12_PREFILL=1  M > PREFILL_M: L2-resident sliced path (decode ~SLICE_MB of 128-row tiles into a small
                              reused scratch, exact escape patch, cuBLAS on the slice, SiluAndMul per slice for
                              gate_up) instead of 64-row W12 chunks (M <= 128) / whole-weight decode + cuBLAS
  SPECPILOT_D2_W12_SLICE_MB   slice size in MB of bf16 (default 48); SPECPILOT_D2_W12_PREFILL_M (default 64)
  SPECPILOT_D4_QVIEW=1        two-level verify: q_pass() selects the 8-bit nested Q-view kernels (M <= 48)
  SPECPILOT_D4_QHEAD=1        with QVIEW: inside q_pass() the lm_head = exact W12 logits over the drafter's vocabulary
                              subset (SPECPILOT_DE_VOCAB_SIZE rows, own W12 copy), -inf elsewhere
  (microbench 7702a7d4b985: the sliced path loses to d2-002's prefill path at M <= 128 and is within +-10% above;
  per-slice GEMMs are too narrow for M >= 256 -> PREFILL stays off in d2-004)
"""
from __future__ import annotations

import hashlib
import logging
import os
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def read_d2_knobs() -> int:
    v = os.environ.get("SPECPILOT_D2_W12", "").strip()
    return int(v) if v else 0


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name, "").strip()
    return int(v) if v else default


D2_W12 = read_d2_knobs()
D2_PDL = _env_int("SPECPILOT_D2_PDL", 1)
D2_OUT = _env_int("SPECPILOT_D2_W12_OUT", 0)
D2_HEAD = _env_int("SPECPILOT_D2_W12_HEAD", 0)
D2_PREFILL = _env_int("SPECPILOT_D2_W12_PREFILL", 0)
D2_SLICE_MB = _env_int("SPECPILOT_D2_W12_SLICE_MB", 48)
D2_PREFILL_M = _env_int("SPECPILOT_D2_W12_PREFILL_M", 64)
# SPECPILOT_D4_QVIEW (two-level verify, d4): inside q_pass() the W12 GEMMs at M <= 48 stream only the 8-bit nested Q
# view of each record (sign, 4-bit exponent code, top 3 mantissa bits; low mantissa = midpoint; escapes exact) -- an
# approximate copy of the same weights at zero extra memory. Read at call time, so a CUDA graph captured inside
# q_pass() replays the Q kernels. Unset = parent behaviour (q_pass() is a no-op).
D4_QVIEW = _env_int("SPECPILOT_D4_QVIEW", 0)
D4_QHEAD = _env_int("SPECPILOT_D4_QHEAD", 0)
# SPECPILOT_D3_QCOPY (d3): with QVIEW, q_pass() reads a separately stored scaled INT6/INT8 g128 copy of each covered
# linear (built at load from the bf16 weights, derived data) instead of the nested 8-bit view, at M <= 32.
# SPECPILOT_D3_QBITS_MLP / _IN / _REST: bits of the MLP (gate_up, down) / GDN in_proj / other (qkv, out, o) copies
# (6 or 8; 0 = nested view for that group).
D3_QCOPY = _env_int("SPECPILOT_D3_QCOPY", 0)
D3_QBITS_MLP = _env_int("SPECPILOT_D3_QBITS_MLP", 6)
D3_QBITS_IN = _env_int("SPECPILOT_D3_QBITS_IN", 6)
D3_QBITS_REST = _env_int("SPECPILOT_D3_QBITS_REST", 8)
_D3_MAX_M = 32
# SPECPILOT_D4_L2PF (d4): L2 prefetch of the next layer records during the GDN / attention section
from sglang.kernels.ops.gemm import specpilot_d4_l2pf as _l2pf  # noqa: E402

_D4_L2PF = _l2pf.D4_L2PF
_QPASS = [False]
_Q_MAX_M = 48


class q_pass:
    """Context manager: `with specpilot_d2_w12.q_pass(): ...` runs the W12 GEMMs in Q mode (SPECPILOT_D4_QVIEW only)."""

    def __init__(self, enabled: bool = True):
        self.enabled = bool(enabled) and bool(D4_QVIEW)

    def __enter__(self):
        self._prev, _QPASS[0] = _QPASS[0], self.enabled
        return self

    def __exit__(self, *exc):
        _QPASS[0] = self._prev
        return False
_RB = 128
_MAX_M = 64
_CHUNK_M = 128
ROT = 37
# (NT) -> (ring depth, corr buffers); measured (d2 microbench rounds 1-2 on sm_120): 2-3 stages saturate GDDR7,
# double-buffered escape corrections hide the whole-tile correction latency of gate_up
CONFIGS = {8: (3, 2), 16: (3, 2), 24: (3, 2), 32: (2, 2), 40: (2, 2), 48: (2, 2), 64: (2, 1)}
# d4 Q view: ring depths compiled per NT (stage = 8 KB Q region + x); QCONFIGS = the default pick
# NT=48 fits only 2 stages in ~99 KB (3 x (8 KB + 6 KB) + corr 48 KB + SiLU 12 KB > optin): ring depth 2 there
QCONFIGS_ALL = {8: (3, 4), 16: (3, 4, 5), 24: (3, 4, 5), 32: (2, 3, 4), 48: (2,)}
QCONFIGS = {8: 4, 16: 4, 24: 4, 32: 3, 48: 2}


def _pick_nt(m: int) -> int:
    for nt in sorted(CONFIGS):
        if m <= nt:
            return nt
    raise ValueError(f"W12: M={m} > {_MAX_M}")


# ------------------------------------------------------------------------------------------------------------ encode

def _frag(p, R, K):
    # row = rg*16 + h*8 + g, col = kc*64 + j*16 + c8*8 + t*2 + b  ->  [rg, kc, lane=(g,t), w=(j, c8, h, b)]
    return p.view(R // 16, 2, 8, K // 64, 4, 2, 4, 2).permute(0, 3, 2, 6, 4, 5, 1, 7).reshape(R // 16, K // 64, 32, 32)


def _encode_rows(wc: torch.Tensor):
    R, K = wc.shape
    u = wc.view(torch.int16).to(torch.int32) & 0xFFFF
    e = (u >> 7) & 0xFF
    hist = torch.zeros(R, 257, dtype=torch.int32, device=wc.device)
    hist.scatter_add_(1, e.to(torch.int64), torch.ones_like(e))
    hist[:, 0] = 0
    cum = torch.cumsum(hist, dim=1)
    tops = torch.arange(15, 256, device=wc.device)
    cover = cum[:, tops] - cum[:, tops - 16].masked_fill((tops - 16 < 0)[None, :], 0)
    best = cover.max(dim=1, keepdim=True).values
    E = tops[(cover == best).int().cumsum(1).argmax(dim=1)].to(torch.int32)
    c = e - (E[:, None] - 15)
    esc = (c < 0) | (c > 15) | (e == 0)
    c = torch.where(esc, torch.zeros_like(c), c)
    s = torch.where(esc, torch.zeros_like(u), u >> 15)
    m = torch.where(esc, torch.zeros_like(u), u & 0x7F)
    low = (((c & 1) << 7) | m).to(torch.uint8)
    nb = ((s << 3) | (c >> 1)).to(torch.uint8)
    del u, e, c, s, m
    qf = _frag((low & 0xF0) | nb, R, K)
    qb = qf.view(R // 16, K // 64, 32, 2, 16).transpose(2, 3).reshape(R // 16, K // 64, 1024)
    bf = _frag(low & 0x0F, R, K).view(R // 16, K // 64, 32, 4, 2, 4)
    bb = (bf[..., 0, :] | (bf[..., 1, :] << 4)).reshape(R // 16, K // 64, 512)
    T_ = R // _RB
    qb = qb.view(T_, _RB // 16, K // 64, 1024).transpose(1, 2).reshape(T_, K // 64, 8 * 1024)
    bb = bb.view(T_, _RB // 16, K // 64, 512).transpose(1, 2).reshape(T_, K // 64, 8 * 512)
    blocks = torch.cat([qb, bb], dim=2).view(T_, K // 64, 8, 1536).contiguous()
    rows, cols = esc.nonzero(as_tuple=True)
    return blocks, E, rows, cols


def _maxseg(T, S, P):
    U = T * S
    own = lambda u: ((u + 1) * P - 1) // U  # noqa: E731
    return max(own(t * S + S - 1) - own(t * S) + 1 for t in range(T))


class W12Weight:
    """Bit-exact W12 (nested) encoding of a [N, K] bf16 weight; silu=True interleaves gate_up as [64 gate | 64 up]."""

    def __init__(self, w: torch.Tensor, silu: bool = False, chunk_rows: int = 2048, nsm: int | None = None,
                 head: bool = False):
        assert w.dtype == torch.bfloat16 and w.dim() == 2
        N, K = w.shape
        assert K % 64 == 0
        self.N, self.K, self.silu = N, K, int(silu)
        self.d4_sub = None  # SPECPILOT_D4_QHEAD: (subset row ids, W12Weight of those rows) on the lm_head only
        self.d3q = None  # SPECPILOT_D3_QCOPY: scaled low-bit Q copy (D3QLinear) used inside q_pass()
        self.d4_pf = None  # SPECPILOT_D4_L2PF: {pass mode: L2 prefetch plan} of the layer's next records (first GEMM only)
        if silu:
            I = N // 2
            wi = w.view(2, I // 64, 64, K).transpose(0, 1).reshape(N, K)
            self.nout = I
        else:
            wi = w
            self.nout = N
        T = (N + _RB - 1) // _RB
        self.T, self.S = T, K // 64
        dev = w.device
        self.planes = torch.empty((T, self.S * 12288), dtype=torch.uint8, device=dev)
        E_all = torch.empty(T * _RB, dtype=torch.int32, device=dev)
        rows_l, cols_l = [], []
        for r0 in range(0, T * _RB, chunk_rows):
            r1 = min(T * _RB, r0 + chunk_rows)
            wc = wi[r0:min(r1, N)]
            if wc.shape[0] < r1 - r0:
                wc = torch.cat([wc, torch.ones(r1 - r0 - wc.shape[0], K, dtype=w.dtype, device=dev)])
            blocks, E, rows, cols = _encode_rows(wc.contiguous())
            self.planes[r0 // _RB: r1 // _RB] = blocks.view(blocks.shape[0], -1)
            E_all[r0:r1] = E
            rows_l.append(rows + r0)
            cols_l.append(cols)
        rows = torch.cat(rows_l)
        cols = torch.cat(cols_l)
        self.E = E_all
        self.scale = torch.pow(2.0, (E_all - 127).double()).float()
        wt = wi[rows, cols].double() * torch.pow(2.0, (127 - E_all[rows]).double())
        d = wt.float()
        self.n_inexact = int((d.double() != wt).sum().item()) if d.numel() else 0
        counts = torch.bincount(rows, minlength=T * _RB)
        self.rp = torch.zeros(T * _RB + 1, dtype=torch.int32, device=dev)
        self.rp[1:] = torch.cumsum(counts, 0).to(torch.int32)
        order = torch.argsort(rows * K + cols)
        self.col = cols[order].to(torch.int32).contiguous()
        self.val = d[order].contiguous()
        rs = rows[order]
        orig = rs
        if silu:
            t64, wr = rs // 128, rs % 128
            orig = torch.where(wr < 64, t64 * 64 + wr, N // 2 + t64 * 64 + wr - 64)
        self.esc_flat = (orig * K + cols[order]).contiguous()  # decode path: true bf16 values scattered back
        self.esc_bf16 = w.reshape(-1)[self.esc_flat].contiguous()
        self._esc_rows_il, self._esc_cols = rs, cols[order]  # interleaved-row escapes (sliced decode)
        self.slices = None
        self.n_esc = int(rows.numel())
        self.max_row_esc = int(counts.max().item()) if self.n_esc else 0
        self.esc_seg = 0 if silu else 1
        self.head = bool(head)  # lm_head: never decoded whole (no N x K scratch)
        if nsm is None:
            nsm = torch.cuda.get_device_properties(dev).multi_processor_count
        self.nsm = nsm
        self.set_P(None)

    def set_P(self, P, whole_decode: Optional[bool] = None):
        T, S = self.T, self.S
        if P is None:
            P = T * max(1, self.nsm // T) if T <= self.nsm else self.nsm
        self.P = min(P, T * S)
        self.maxseg = _maxseg(T, S, self.P)
        if whole_decode is None:
            whole_decode = not D2_PREFILL and not self.head
        _ensure_shared(self.planes.device, T * self.maxseg * _RB * _MAX_M, T, self.N * self.K if whole_decode else 0)

    def build_slices(self, slice_mb: Optional[int] = None) -> None:
        """Host-side plan of the sliced prefill path (no syncs at run time): consecutive 128-row tiles of ~slice_mb
        MB of bf16 each, with each slice's escapes as (flat index in the slice's decoded layout, true bf16 value).
        A gate_up slice of tiles [t0, t1) decodes as [gate rows t0*64.. | up rows t0*64..] (2 x (t1-t0)*64 rows)."""
        mb = slice_mb or D2_SLICE_MB
        tps = max(1, int(mb * 2 ** 20 // (_RB * self.K * 2)))
        rows_il, cols = self._esc_rows_il, self._esc_cols
        tile_of = rows_il // _RB
        sl = []
        for t0 in range(0, self.T, tps):
            t1 = min(self.T, t0 + tps)
            nts = t1 - t0
            if self.silu:
                nrows = 2 * 64 * nts
            else:
                nrows = min(self.N, t1 * _RB) - t0 * _RB
            sel = ((tile_of >= t0) & (tile_of < t1)).nonzero(as_tuple=True)[0]
            r = rows_il[sel] - t0 * _RB
            if self.silu:
                tl, wr = r // _RB, r % _RB
                r = torch.where(wr < 64, tl * 64 + wr, nts * 64 + tl * 64 + wr - 64)
            idx = (r * self.K + cols[sel]).contiguous()
            sl.append((t0, t1, nrows, idx, self.esc_bf16[sel].contiguous()))
        self.slices = sl
        sh = _SHARED.setdefault(self.planes.device, {})
        need = max(s[2] for s in sl) * self.K
        if sh.get("slice") is None or sh["slice"].numel() < need:
            sh["slice"] = torch.empty(need, dtype=torch.bfloat16, device=self.planes.device)

    def decode_slice(self, i: int) -> torch.Tensor:
        """Decoded bf16 rows of slice i (escapes patched) in the shared slice scratch: [nrows, K]."""
        t0, t1, nrows, idx, val = self.slices[i]
        out = _SHARED[self.planes.device]["slice"][: nrows * self.K].view(nrows, self.K)
        _ext().w12m_decode(self.planes[t0:t1], self.scale[t0 * _RB:], out, nrows, self.K, self.silu)
        if idx.numel():
            out.view(-1)[idx] = val
        return out

    def sliced_linear(self, x2: torch.Tensor, rows: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """Prefill path: y = x2 @ w[rows]^T (gate_up: SiLU(gate) * up), one slice of tiles at a time (the decoded
        slice stays in L2 for its cuBLAS GEMM)."""
        if self.slices is None:
            self.build_slices()
        m = x2.shape[0]
        r0, r1 = rows if rows is not None else (0, self.nout)
        out = torch.empty((m, r1 - r0), dtype=x2.dtype, device=x2.device)
        for i, (t0, t1, nrows, _, _) in enumerate(self.slices):
            if self.silu:
                a, b = t0 * 64, t0 * 64 + nrows // 2
            else:
                a, b = t0 * _RB, t0 * _RB + nrows
            lo, hi = max(a, r0), min(b, r1)
            if lo >= hi:
                continue
            w = self.decode_slice(i)
            if self.silu:
                gu = F.linear(x2, w if x2.dtype == w.dtype else w.to(x2.dtype))
                h = _silu_and_mul(gu)
                out[:, lo - r0: hi - r0] = h[:, lo - a: hi - a]
            else:
                ws_ = w[lo - a: hi - a]
                out[:, lo - r0: hi - r0] = F.linear(x2, ws_ if x2.dtype == ws_.dtype else ws_.to(x2.dtype))
        return out

    def nbytes(self):
        return sum(t.numel() * t.element_size()
                   for t in (self.planes, self.scale, self.rp, self.col, self.val, self.esc_flat, self.esc_bf16))

    def decode(self) -> torch.Tensor:
        """bf16 [N, K] (original row order) in the shared scratch (a fresh tensor when the scratch is smaller: the
        load-time roundtrip check of weights that never use the whole-weight decode path)."""
        sc = _SHARED[self.planes.device].get("scratch")
        if sc is not None and sc.numel() >= self.N * self.K:
            out = sc[: self.N * self.K].view(self.N, self.K)
        else:
            out = torch.empty((self.N, self.K), dtype=torch.bfloat16, device=self.planes.device)
        _ext().w12m_decode(self.planes, self.scale, out, self.N, self.K, self.silu)
        if self.n_esc:
            out.view(-1)[self.esc_flat] = self.esc_bf16
        return out

    def gemm(self, x2: torch.Tensor, out: Optional[torch.Tensor] = None, qv: int = 0, ns: int = 0) -> torch.Tensor:
        """x2 [M <= 64, K] bf16 (row-major, 16-B aligned rows) -> [M, nout] (gate_up: SiLU(gate) * up).
        qv=1: the 8-bit nested Q view (midpoint low mantissa, exact escapes), M <= 48."""
        m = x2.shape[0]
        if x2.stride(-1) != 1 or x2.stride(0) % 8 or x2.data_ptr() % 16:
            x2 = x2.contiguous()
        if qv:
            nt = next(n for n in sorted(QCONFIGS) if m <= n)
            ns, ncb = (ns or QCONFIGS[nt]), 2
        else:
            nt = _pick_nt(m)
            ns, ncb = CONFIGS[nt]
        sh = _SHARED[x2.device]
        if out is None:
            out = torch.empty((m, self.nout), dtype=torch.bfloat16, device=x2.device)
        _ext().w12m_gemm(x2, self.planes, self.scale, self.rp, self.col, self.val, out, sh["ws"], sh["cnt"], self.P,
                         self.maxseg, self.N, self.silu, nt, ns, self.esc_seg, D2_PDL, D2_PDL, ROT, 0, ncb, qv)
        return out



# -------------------------------------------------------------------------------------------------------------- CUDA

_SRC = r"""
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

#define NCW 8
#define RB 128
#define WREC 12288

__device__ __forceinline__ void mbar_init(uint32_t a, int cnt) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(a), "r"(cnt));
}
__device__ __forceinline__ void mbar_expect_tx(uint32_t a, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(a), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint32_t a) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(a) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint32_t a, uint32_t parity) {
  asm volatile("{\n .reg .pred p;\n W12W_%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra W12W_%=;\n}\n"
               ::"r"(a), "r"(parity) : "memory");
}
__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t mbar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;\n"
               ::"r"(dst), "l"(src), "r"(bytes), "r"(mbar), "l"(0x12F0000000000000ull) : "memory");
}
__device__ __forceinline__ void tma2d(uint32_t dst, const CUtensorMap* map, int c0, int c1, uint32_t mbar) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n"
               ::"r"(dst), "l"(map), "r"(c0), "r"(c1), "r"(mbar) : "memory");
}
__device__ __forceinline__ float ldg_early(const float* p) {
  float v;
  asm volatile("ld.global.nc.f32 %0, [%1];\n" : "=f"(v) : "l"(p));
  return v;
}
__device__ __forceinline__ void cons_bar(int n) { asm volatile("bar.sync 1, %0;\n" ::"r"(n) : "memory"); }
__device__ __forceinline__ void ldsm_x4(uint32_t (&b)[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(b[0]), "=r"(b[1]), "=r"(b[2]), "=r"(b[3]) : "r"(addr));
}
__device__ __forceinline__ void ldsm_x2(uint32_t (&b)[2], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(b[0]), "=r"(b[1]) : "r"(addr));
}
__device__ __forceinline__ void mma16816(float* d, const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// = lib w12_nest decode_n<0>: Q words (2 uint4: one byte per weight [c0 m6 m5 m4 | s c3 c2 c1]) + B words (1 uint4:
// [m3..m0] nibbles; word j low nibbles -> Q word 2j, high -> 2j+1) -> A fragments of 4 k16 steps
__device__ __forceinline__ void decode_n(const uint4 cq0, const uint4 cq1, const uint4 cb, uint32_t (&A)[4][4]) {
  const uint32_t Q[8] = {cq0.x, cq0.y, cq0.z, cq0.w, cq1.x, cq1.y, cq1.z, cq1.w};
  const uint32_t B[4] = {cb.x, cb.y, cb.z, cb.w};
#pragma unroll
  for (int j = 0; j < 4; ++j) {
#pragma unroll
    for (int c8 = 0; c8 < 2; ++c8) {
      const uint32_t q = Q[2 * j + c8];
      const uint32_t b = c8 ? (B[j] >> 4) & 0x0F0F0F0Fu : B[j] & 0x0F0F0F0Fu;
      const uint32_t lo = (q & 0xF0F0F0F0u) | b;
      const uint32_t hi = ((((q << 4) & 0x80808080u) | (q & 0x07070707u))) | 0x38383838u;
      A[j][2 * c8] = __byte_perm(lo, hi, 0x5140);
      A[j][2 * c8 + 1] = __byte_perm(lo, hi, 0x7362);
    }
  }
}

// Q view (d4): the 8-bit nested prefix only, low mantissa nibble = midpoint 1000
__device__ __forceinline__ void decode_q(const uint4 cq0, const uint4 cq1, uint32_t (&A)[4][4]) {
  const uint32_t Q[8] = {cq0.x, cq0.y, cq0.z, cq0.w, cq1.x, cq1.y, cq1.z, cq1.w};
#pragma unroll
  for (int j = 0; j < 4; ++j) {
#pragma unroll
    for (int c8 = 0; c8 < 2; ++c8) {
      const uint32_t q = Q[2 * j + c8];
      const uint32_t lo = (q & 0xF0F0F0F0u) | 0x08080808u;
      const uint32_t hi = ((((q << 4) & 0x80808080u) | (q & 0x07070707u))) | 0x38383838u;
      A[j][2 * c8] = __byte_perm(lo, hi, 0x5140);
      A[j][2 * c8 + 1] = __byte_perm(lo, hi, 0x7362);
    }
  }
}

// ABL ablation bits (benchmarks only): 1 no decode (raw Q words as A), 2 no mma/ldmatrix, 4 no x (no TMA, B = const),
// 8 no escapes, 16 consumers skip the corr wait/add (escape warps still compute), 32 escape warps signal without
// computing. ABL = 0 is the production kernel. NCB: 1 or 2 corr buffers (2 = escape warps run one segment ahead).
template <int NT, int NS, int ABL, int NCB, int QV>
__global__ void __launch_bounds__(NCW * 32 + 96, 1) w12m_kernel(
    const __grid_constant__ CUtensorMap xmap, const __nv_bfloat16* __restrict__ x, int64_t ldx, int M,
    const unsigned char* __restrict__ planes, int64_t tstride, int S, int U,
    const float* __restrict__ scale, const int* __restrict__ esc_rp, const int* __restrict__ esc_col,
    const float* __restrict__ esc_val, __nv_bfloat16* __restrict__ out, int64_t ldo, int N,
    float* __restrict__ ws, int maxseg, int* __restrict__ counters, int silu, int nout, int esc_seg, int trig, int rot) {
  constexpr int NQ = NT / 8;
  constexpr int XB = NT * 128;
  constexpr int RBY = QV ? 8192 : WREC;  // Q pass streams only the Q region (first 8 KB) of each 12 KB record
  constexpr int SB = RBY + XB;  // multiple of 1024: x stages stay 1024-B aligned (128B swizzle atom)
  constexpr int CB = RB * NT;
  constexpr uint32_t TXB = (ABL & 4) ? RBY : SB;
  extern __shared__ __align__(1024) unsigned char smem[];
  __shared__ __align__(8) uint64_t bars[2 * NS + 2 * NCB];
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int P = gridDim.x, p = blockIdx.x;
  const int u0 = (int)(((int64_t)p * U) / P), u1 = (int)(((int64_t)(p + 1) * U) / P);
  const uint32_t sraw = (uint32_t)__cvta_generic_to_shared(smem);
  const uint32_t sbase = (sraw + 1023u) & ~1023u;
  unsigned char* const smem_al = smem + (sbase - sraw);
  float* corrbuf = reinterpret_cast<float*>(smem_al + NS * SB);
  float* xchg = corrbuf + NCB * CB;
  const uint32_t full0 = (uint32_t)__cvta_generic_to_shared(bars), empty0 = full0 + 8 * NS;
  const uint32_t cfull0 = full0 + 16 * NS, cempty0 = cfull0 + 8 * NCB;
  if (threadIdx.x == 0) {
    for (int s = 0; s < NS; ++s) { mbar_init(full0 + 8 * s, 1); mbar_init(empty0 + 8 * s, NCW); }
    for (int b = 0; b < NCB; ++b) { mbar_init(cfull0 + 8 * b, 64); mbar_init(cempty0 + 8 * b, NCW * 32); }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();
  if (trig) asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
  if (u0 >= u1) return;
  if (warp == NCW) {  // producer
    if (lane == 0) {
      int tile = 0, sbeg = 0, slen = 1, srot = 0;
      auto walk = [&](int u) -> int {
        if (u == u0 || u - tile * S == S) {
          tile = u / S;
          sbeg = u - tile * S;
          slen = min(u1, (tile + 1) * S) - u;
          srot = rot ? (int)(((int64_t)tile * rot + p) % slen) : 0;
        }
        int s = (u - tile * S) - sbeg + srot;
        if (s >= slen) s -= slen;
        return s + sbeg;
      };
      const int npre = min(NS, u1 - u0);
      int spre[NS];
#pragma unroll
      for (int i = 0; i < NS; ++i) {
        if (i < npre) {
          const int s = walk(u0 + i);
          spre[i] = s;
          const uint32_t fb = full0 + 8 * i;
          mbar_expect_tx(fb, TXB);
          bulk_g2s(sbase + i * SB, planes + (int64_t)tile * tstride + (int64_t)s * WREC, RBY, fb);
        }
      }
      asm volatile("griddepcontrol.wait;\n" ::: "memory");
      if (!(ABL & 4)) {
#pragma unroll
        for (int i = 0; i < NS; ++i)
          if (i < npre) tma2d(sbase + i * SB + RBY, &xmap, spre[i] * 64, 0, full0 + 8 * i);
      }
      for (int u = u0 + npre; u < u1; ++u) {
        const int i = u - u0, slot = i % NS;
        mbar_wait(empty0 + 8 * slot, ((i / NS) - 1) & 1);
        const int s = walk(u);
        const uint32_t fb = full0 + 8 * slot, dst = sbase + slot * SB;
        mbar_expect_tx(fb, TXB);
        bulk_g2s(dst, planes + (int64_t)tile * tstride + (int64_t)s * WREC, RBY, fb);
        if (!(ABL & 4)) tma2d(dst + RBY, &xmap, s * 64, 0, fb);
      }
    }
    return;
  }
  if (warp > NCW) {  // 2 escape warps: exact correction of this segment's k-range into the (single) corr buffer
    if (ABL & 8) return;
    const int ew = warp - NCW - 1;
    asm volatile("griddepcontrol.wait;\n" ::: "memory");
    int j = 0;
    for (int u = u0; u < u1;) {
      const int tile = u / S, s0 = u - tile * S;
      const int uend = min(u1, (tile + 1) * S), s1 = uend - tile * S;
      if (!esc_seg && s0 != 0) { u = uend; continue; }
      const int k0 = esc_seg ? s0 * 64 : 0, k1 = esc_seg ? s1 * 64 : S * 64;
      const int b = j % NCB;
      if (!(ABL & 16) && j >= NCB) mbar_wait(cempty0 + 8 * b, ((j / NCB) - 1) & 1);
      float* cb = corrbuf + b * CB;
      for (int r = lane + 32 * ew; r < RB && !(ABL & 32); r += 64) {
        float c[NT];
#pragma unroll
        for (int m = 0; m < NT; ++m) c[m] = 0.f;
        const int n = tile * RB + r;
        if (n < N) {
          const int e0 = esc_rp[n], e1 = esc_rp[n + 1];
          for (int e = e0; e < e1; ++e) {
            const int col = esc_col[e];
            if (col < k0 || col >= k1) continue;
            const float ev = esc_val[e];
            const __nv_bfloat16* xc = x + col;
#pragma unroll
            for (int m = 0; m < NT; ++m)
              if (m < M) {
                const float xv = __bfloat162float(xc[(int64_t)m * ldx]);
                c[m] = fmaf(ev, xv, fmaf(QV ? -3.2424926757812e-05f : -3.0517578125e-05f, xv, c[m]));
              }
          }
        }
#pragma unroll
        for (int m = 0; m < NT; ++m) cb[r * NT + ((m + r) % NT)] = c[m];
      }
      mbar_arrive(cfull0 + 8 * b);
      ++j;
      u = uend;
    }
    return;
  }
  // consumers: warp w owns weight rows [w*16, w*16+16) of the tile; tokens as NQ n-tiles of 8
  const int g = lane / 4, t = lane % 4;
  float acc[NQ * 4];
  // ldmatrix.x4 lane address: matrix mi = lane/8 (bit0: k half, bit1: n-tile of the pair), row r = lane%8
  const int lm_r = lane & 7, lm_kh = (lane >> 3) & 1, lm_nt = lane >> 4;
  auto stage = [&](int i) {
    const int slot = i % NS;
    mbar_wait(full0 + 8 * slot, (i / NS) & 1);
    const uint4* sw = reinterpret_cast<const uint4*>(smem_al + slot * SB);
    uint32_t A[4][4];
    const uint4 q0 = sw[warp * 64 + lane], q1 = sw[warp * 64 + 32 + lane];
    const uint4 bb = QV ? make_uint4(0u, 0u, 0u, 0u) : sw[512 + warp * 32 + lane];
    if (ABL & 1) {
      const uint32_t Q[12] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w, bb.x, bb.y, bb.z, bb.w};
#pragma unroll
      for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int r = 0; r < 4; ++r) A[j][r] = Q[(j * 4 + r) % 12];
    } else if (QV) {
      decode_q(q0, q1, A);
    } else {
      decode_n(q0, q1, bb, A);
    }
    const uint32_t xs = sbase + slot * SB + RBY;
    if (ABL & 2) {
#pragma unroll
      for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int r = 0; r < 4; ++r) acc[(j * 4 + r) % (NQ * 4)] += __uint_as_float(A[j][r] & 0x3F003F00u);
    } else {
#pragma unroll
      for (int j = 0; j < 4; ++j) {
#pragma unroll
        for (int qp = 0; qp < NQ / 2; ++qp) {
          uint32_t b[4];
          if (ABL & 4) {
            b[0] = b[1] = b[2] = b[3] = 0x3F803F80u;
          } else {
            const int n = (2 * qp + lm_nt) * 8 + lm_r, ch = 2 * j + lm_kh;
            ldsm_x4(b, xs + n * 128 + ((ch ^ lm_r) << 4));
          }
          mma16816(acc + (2 * qp) * 4, A[j], b[0], b[1]);
          mma16816(acc + (2 * qp + 1) * 4, A[j], b[2], b[3]);
        }
        if (NQ & 1) {
          uint32_t b[2];
          if (ABL & 4) {
            b[0] = b[1] = 0x3F803F80u;
          } else {
            const int n = (NQ - 1) * 8 + lm_r, ch = 2 * j + lm_kh;
            ldsm_x2(b, xs + n * 128 + ((ch ^ lm_r) << 4));
          }
          mma16816(acc + (NQ - 1) * 4, A[j], b[0], b[1]);
        }
      }
    }
    __syncwarp();
    // the slot was read through the generic proxy (LDS, ldmatrix); the producer refills it with TMA (async proxy):
    // without this fence a warp's late reads raced the refill (~2% of back-to-back calls at NT=24 differed)
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    if (lane == 0) mbar_arrive(empty0 + 8 * slot);
  };
  int jc = 0;
  for (int u = u0; u < u1;) {
    const int tile = u / S, s0 = u - tile * S;
    const int uend = min(u1, (tile + 1) * S);
    const int n0 = tile * RB + warp * 16 + g;
    const float sc0 = n0 < N ? ldg_early(scale + n0) : 1.f, sc1 = n0 + 8 < N ? ldg_early(scale + n0 + 8) : 1.f;
#pragma unroll
    for (int q = 0; q < NQ * 4; ++q) acc[q] = 0.f;
    for (int uu = u; uu < uend; ++uu) stage(uu - u0);
    if (!(ABL & 24) && (esc_seg || s0 == 0)) {
      const int b = jc % NCB;
      const float* cb = corrbuf + b * CB;
      mbar_wait(cfull0 + 8 * b, (jc / NCB) & 1);
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int r = warp * 16 + g + 8 * h;
#pragma unroll
        for (int q = 0; q < NQ; ++q)
#pragma unroll
          for (int v = 0; v < 2; ++v) {
            const int m = q * 8 + 2 * t + v;
            acc[q * 4 + 2 * h + v] += cb[r * NT + ((m + r) % NT)];
          }
      }
      mbar_arrive(cempty0 + 8 * b);
      ++jc;
    }
    const bool whole = (s0 == 0) && (uend == (tile + 1) * S);
    bool emit = whole;
    if (!whole) {
      const int ufirst = tile * S, ulast = ufirst + S - 1;
      const int pfirst = (int)((((int64_t)ufirst + 1) * P - 1) / U), plast = (int)((((int64_t)ulast + 1) * P - 1) / U);
      const int seg = p - pfirst, nseg = plast - pfirst + 1;
      float* wt = ws + ((int64_t)tile * maxseg) * RB * NT;
      const int rl = warp * 16 + g;
      if (seg > 0) {
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
          for (int q = 0; q < NQ; ++q)
            __stcg(reinterpret_cast<float2*>(wt + ((int64_t)seg * RB + rl + 8 * h) * NT + q * 8 + 2 * t),
                   make_float2(acc[q * 4 + 2 * h], acc[q * 4 + 2 * h + 1]));
        cons_bar(NCW * 32);
        if (threadIdx.x == 0) asm volatile("red.release.gpu.global.add.s32 [%0], 1;\n" ::"l"(counters + tile) : "memory");
      } else {
        if (threadIdx.x == 0) {
          int c;
          do {
            asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\n" : "=r"(c) : "l"(counters + tile) : "memory");
          } while (c < nseg - 1);
          counters[tile] = 0;
        }
        cons_bar(NCW * 32);
        constexpr int PB = NT <= 32 ? 4 : (NT <= 48 ? 2 : 1);
        for (int s2 = 1; s2 < nseg; s2 += PB) {
          float2 v[PB][2][NQ];
#pragma unroll
          for (int jj = 0; jj < PB; ++jj)
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
              for (int q = 0; q < NQ; ++q)
                v[jj][h][q] = s2 + jj < nseg ? __ldcg(reinterpret_cast<const float2*>(
                                                   wt + ((int64_t)(s2 + jj) * RB + rl + 8 * h) * NT + q * 8 + 2 * t))
                                             : make_float2(0.f, 0.f);
#pragma unroll
          for (int jj = 0; jj < PB; ++jj)
            if (s2 + jj < nseg) {
#pragma unroll
              for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int q = 0; q < NQ; ++q) {
                  acc[q * 4 + 2 * h] += v[jj][h][q].x;
                  acc[q * 4 + 2 * h + 1] += v[jj][h][q].y;
                }
            }
        }
        emit = true;
      }
    }
    if (emit) {
      if (!silu) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int n = n0 + 8 * h;
          const float sc = h ? sc1 : sc0;
#pragma unroll
          for (int q = 0; q < NQ; ++q)
#pragma unroll
            for (int v = 0; v < 2; ++v) {
              const int m = q * 8 + 2 * t + v;
              if (m < M && n < N) out[(int64_t)m * ldo + n] = __float2bfloat16(acc[q * 4 + 2 * h + v] * sc);
            }
        }
      } else {
        const int wr = (warp & 3) * 16 + g;
        if (warp >= 4) {
#pragma unroll
          for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int q = 0; q < NQ; ++q)
#pragma unroll
              for (int v = 0; v < 2; ++v)
                xchg[(wr + 8 * h) * NT + q * 8 + 2 * t + v] =
                    __bfloat162float(__float2bfloat16(acc[q * 4 + 2 * h + v] * (h ? sc1 : sc0)));
        }
        cons_bar(NCW * 32);
        if (warp < 4) {
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const int n = tile * 64 + wr + 8 * h;
#pragma unroll
            for (int q = 0; q < NQ; ++q)
#pragma unroll
              for (int v = 0; v < 2; ++v) {
                const int m = q * 8 + 2 * t + v;
                const float gg = __bfloat162float(__float2bfloat16(acc[q * 4 + 2 * h + v] * (h ? sc1 : sc0)));
                const float uu = xchg[(wr + 8 * h) * NT + q * 8 + 2 * t + v];
                if (m < M && n < nout) out[(int64_t)m * ldo + n] = __float2bfloat16(gg / (1.f + __expf(-gg)) * uu);
              }
          }
        }
        cons_bar(NCW * 32);
      }
    }
    u = uend;
  }
}

typedef CUresult (*EncodeTiled)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*,
                                const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                                CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
static EncodeTiled get_encode() {
  static EncodeTiled fn = nullptr;
  if (!fn) {
    cudaDriverEntryPointQueryResult q;
    cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", (void**)&fn, cudaEnableDefault, &q);
    TORCH_CHECK(fn != nullptr, "no cuTensorMapEncodeTiled");
  }
  return fn;
}

template <int NT, int NS, int ABL, int NCB, int QV>
void launch(torch::Tensor x, torch::Tensor planes, torch::Tensor scale, torch::Tensor rp, torch::Tensor col,
            torch::Tensor val, torch::Tensor out, torch::Tensor ws, torch::Tensor cnt, int64_t P, int64_t maxseg,
            int64_t N, int64_t silu, int64_t esc_seg, int64_t pdl, int64_t trig, int64_t rot) {
  const int M = x.size(0), K = x.size(1), nout = out.size(1);
  TORCH_CHECK(M <= NT && M > 0, "W12m: bad M");
  const int T = planes.size(0), S = K / 64, U = T * S;
  CUtensorMap map;
  cuuint64_t gdim[2] = {(cuuint64_t)K, (cuuint64_t)M};
  cuuint64_t gstr[1] = {(cuuint64_t)x.stride(0) * 2};
  cuuint32_t box[2] = {64, NT};
  cuuint32_t es[2] = {1, 1};
  CUresult r = get_encode()(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, x.data_ptr(), gdim, gstr, box, es,
                            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "W12m tensor map encode failed ", (int)r);
  constexpr int SB = (QV ? 8192 : WREC) + NT * 128;
  const int smem = NS * SB + NCB * RB * NT * 4 + (silu ? 64 * NT * 4 : 0) + 1024;
  auto kern = w12m_kernel<NT, NS, ABL, NCB, QV>;
  static bool attr = false;
  if (!attr) {
    cudaError_t ea = cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, NS * SB + NCB * RB * NT * 4 + 64 * NT * 4 + 1024);
    TORCH_CHECK(ea == cudaSuccess, "W12m smem attr failed: ", cudaGetErrorString(ea));
    attr = true;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((unsigned)P);
  cfg.blockDim = dim3(NCW * 32 + 96);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = c10::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
  cfg.attrs = at;
  cfg.numAttrs = 1;
  cudaError_t e = cudaLaunchKernelEx(&cfg, kern,
      map, reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), (int64_t)x.stride(0), M,
      reinterpret_cast<const unsigned char*>(planes.data_ptr()), (int64_t)planes.stride(0), S, U,
      (const float*)scale.data_ptr<float>(), (const int*)rp.data_ptr<int>(), (const int*)col.data_ptr<int>(),
      (const float*)val.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), (int64_t)out.stride(0),
      (int)N, ws.data_ptr<float>(), (int)maxseg, cnt.data_ptr<int>(), (int)silu, nout, (int)esc_seg, (int)trig, (int)rot);
  TORCH_CHECK(e == cudaSuccess, "W12m launch failed: ", cudaGetErrorString(e));
}

@@DISPATCH@@


// planes -> bf16 weight in the original row order (escapes are scattered back by the caller); one block = one
// (tile, 64-k chunk): decode the 128 x 64 tile into smem, then coalesced 16-B row stores
__global__ void __launch_bounds__(256) w12m_decode_kernel(const uint4* __restrict__ planes, int64_t tstride16, int nkc,
                                                          const float* __restrict__ scale, __nv_bfloat16* __restrict__ out,
                                                          int N, int K, int silu) {
  __shared__ uint32_t tl[128][33];
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  const int kc = blockIdx.x % nkc, tile = blockIdx.x / nkc;
  const uint4* blk = planes + tile * tstride16 + (int64_t)kc * 768;
  uint32_t A[4][4];
  decode_n(blk[w * 64 + lane], blk[w * 64 + 32 + lane], blk[512 + w * 32 + lane], A);
  const int g = lane / 4, t = lane % 4;
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const int rl = w * 16 + g + 8 * h;
    const uint32_t E = (__float_as_uint(scale[tile * 128 + rl]) >> 23) & 0xFF;
    const uint32_t add = ((E - 127u) << 7) & 0xFFFFu;
    const uint32_t add2 = add | (add << 16);
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
      for (int c8 = 0; c8 < 2; ++c8) tl[rl][j * 8 + c8 * 4 + t] = __vadd2(A[j][2 * c8 + h], add2);
  }
  __syncthreads();
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    const int idx = threadIdx.x + 256 * q, rl = idx >> 3, c16 = idx & 7;
    const int r = tile * 128 + rl;
    int orig = r;
    if (silu) orig = rl < 64 ? tile * 64 + rl : N / 2 + tile * 64 + rl - 64;
    if (orig >= N) continue;
    const uint4 v = make_uint4(tl[rl][c16 * 4], tl[rl][c16 * 4 + 1], tl[rl][c16 * 4 + 2], tl[rl][c16 * 4 + 3]);
    *reinterpret_cast<uint4*>(out + (int64_t)orig * K + kc * 64 + c16 * 8) = v;
  }
}

void w12m_decode(torch::Tensor planes, torch::Tensor scale, torch::Tensor out, int64_t N, int64_t K, int64_t silu) {
  const int T = planes.size(0), nkc = K / 64;
  w12m_decode_kernel<<<(unsigned)(T * nkc), 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const uint4*>(planes.data_ptr()), planes.stride(0) / 16, nkc, scale.data_ptr<float>(),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), (int)N, (int)K, (int)silu);
}
"""


def _dispatch_src():
    lines = ["void w12m_gemm(torch::Tensor x, torch::Tensor planes, torch::Tensor scale, torch::Tensor rp, torch::Tensor col,"
             " torch::Tensor val, torch::Tensor out, torch::Tensor ws, torch::Tensor cnt, int64_t P, int64_t maxseg, int64_t N,"
             " int64_t silu, int64_t nt, int64_t ns, int64_t esc_seg, int64_t pdl, int64_t trig, int64_t rot, int64_t abl,"
             " int64_t ncb, int64_t qv) {"]
    args = "x, planes, scale, rp, col, val, out, ws, cnt, P, maxseg, N, silu, esc_seg, pdl, trig, rot"
    for nt, (ns, ncb) in CONFIGS.items():
        lines.append(f"  if (nt == {nt} && ns == {ns} && ncb == {ncb} && abl == 0 && qv == 0) "
                     f"{{ launch<{nt}, {ns}, 0, {ncb}, 0>({args}); return; }}")
    for nt, nss in QCONFIGS_ALL.items():
        for ns in nss:
            lines.append(f"  if (nt == {nt} && ns == {ns} && ncb == 2 && abl == 0 && qv == 1) "
                         f"{{ launch<{nt}, {ns}, 0, 2, 1>({args}); return; }}")
    lines.append('  TORCH_CHECK(false, "W12m: no instantiation for nt ", nt, " ns ", ns, " ncb ", ncb, " abl ", abl);\n}')
    return "\n".join(lines)


_CPP = (
    "void w12m_gemm(torch::Tensor x, torch::Tensor planes, torch::Tensor scale, torch::Tensor rp, torch::Tensor col, "
    "torch::Tensor val, torch::Tensor out, torch::Tensor ws, torch::Tensor cnt, int64_t P, int64_t maxseg, int64_t N, "
    "int64_t silu, int64_t nt, int64_t ns, int64_t esc_seg, int64_t pdl, int64_t trig, int64_t rot, int64_t abl, int64_t ncb, int64_t qv);\n"
    "void w12m_decode(torch::Tensor planes, torch::Tensor scale, torch::Tensor out, int64_t N, int64_t K, int64_t silu);\n"
)

_mod = None


def _ext():
    global _mod
    if _mod is None:
        from torch.utils.cpp_extension import load_inline

        os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0a"
        src = _SRC.replace("@@DISPATCH@@", _dispatch_src())
        cflags = ["-O3", "-gencode=arch=compute_120a,code=sm_120a", "--use_fast_math"]
        # one build directory per source version: versions sharing a directory rebuild a .so other processes mapped
        key = hashlib.sha1("\0".join([_CPP, src, *cflags]).encode()).hexdigest()[:16]
        base = os.environ.get("SGLANG_CACHE_DIR") or os.path.join(os.path.expanduser("~"), ".cache", "sglang")
        bdir = os.path.join(base, "specpilot_d2_w12", key)
        os.makedirs(bdir, exist_ok=True)
        _mod = load_inline(name=f"specpilot_d2_w12_{key}", cpp_sources=_CPP, cuda_sources=src,
                           functions=["w12m_gemm", "w12m_decode"], extra_cuda_cflags=cflags, build_directory=bdir,
                           verbose=False)
    return _mod


# ----------------------------------------------------------------------------------------------------------- runtime

_SHARED: dict = {}


def _ensure_shared(dev, ws_floats, ntiles, scratch_elems):
    """One fp32 split-K workspace, one tile-counter array (reset by the kernel's reducers) and one bf16 decode scratch
    per device, shared by every W12 layer (kernels run in stream order). Allocated at load, before the KV pool."""
    sh = _SHARED.setdefault(dev, {})
    if sh.get("ws") is None or sh["ws"].numel() < ws_floats:
        sh["ws"] = torch.empty(ws_floats, dtype=torch.float32, device=dev)
    if sh.get("cnt") is None or sh["cnt"].numel() < ntiles:
        sh["cnt"] = torch.zeros(ntiles, dtype=torch.int32, device=dev)
    if sh.get("scratch") is None or sh["scratch"].numel() < scratch_elems:
        sh["scratch"] = torch.empty(scratch_elems, dtype=torch.bfloat16, device=dev)


def _silu_and_mul(gu: torch.Tensor) -> torch.Tensor:
    """Stock's MLP activation (SiluAndMul over [gate | up])."""
    from sglang.srt.layers.activation import SiluAndMul

    global _ACT
    if _ACT is None:
        _ACT = SiluAndMul()
    return _ACT(gu)


_ACT = None


def _gemm_rows(enc: W12Weight, x2: torch.Tensor) -> torch.Tensor:
    m = x2.shape[0]
    if _QPASS[0] and m <= _Q_MAX_M:  # SPECPILOT_D4_QVIEW
        if enc.d3q is not None and m <= _D3_MAX_M:  # SPECPILOT_D3_QCOPY
            return enc.d3q(x2)
        return enc.gemm(x2, qv=1)
    if m <= _MAX_M:
        return enc.gemm(x2)
    return torch.cat([enc.gemm(x2[i:i + _MAX_M]) for i in range(0, m, _MAX_M)])


def d2_linear(enc: W12Weight, x: torch.Tensor, rows: Optional[Tuple[int, int]] = None) -> torch.Tensor:
    """y = x @ w[rows]^T for an encoded (non-SiLU) weight."""
    k = enc.K
    m = x.numel() // k
    x2 = x.reshape(m, k)
    r0, r1 = rows if rows is not None else (0, enc.N)
    if _D4_L2PF and (enc.head or enc.d4_pf is not None):  # SPECPILOT_D4_L2PF: join the previous layer's prefetch
        _l2pf.join()
    if D2_PREFILL and (m > D2_PREFILL_M or x2.dtype != torch.bfloat16 or torch.compiler.is_compiling()):
        y = enc.sliced_linear(x2, rows)
    elif (enc.head and _QPASS[0] and m <= _Q_MAX_M and enc.d4_sub is not None
          and (r0, r1) == (0, enc.N)):
        # SPECPILOT_D4_QHEAD: Q pass head = exact logits over the drafter's subset rows (tree tokens are subset tokens)
        sub, enc_sub = enc.d4_sub
        ys = enc_sub.gemm(x2 if x2.dtype == torch.bfloat16 else x2.to(torch.bfloat16))
        y = torch.full((m, enc.N), float("-inf"), dtype=ys.dtype, device=ys.device)
        y.index_copy_(1, sub, ys)
    elif enc.head or (0 < m <= _CHUNK_M and x2.dtype == torch.bfloat16 and not torch.compiler.is_compiling()):
        # the head is never decoded whole (2.5 GB scratch): 64-row W12 chunks for any M
        y = _gemm_rows(enc, x2 if x2.dtype == torch.bfloat16 else x2.to(torch.bfloat16))
        if (r0, r1) != (0, enc.N):
            y = y[:, r0:r1]
    else:
        y = F.linear(x2, enc.decode()[r0:r1])
    if (_D4_L2PF and enc.d4_pf is not None and 0 < m <= _MAX_M and x2.dtype == torch.bfloat16
            and not torch.compiler.is_compiling()):  # SPECPILOT_D4_L2PF: stream the next records into L2
        _l2pf.fork(enc.d4_pf, _d4_pf_mode(m))
    return y.view(*x.shape[:-1], r1 - r0)


def _d4_pf_mode(m: int) -> str:
    """SPECPILOT_D4_L2PF: which stream the layer's next GEMMs read (mirrors _gemm_rows' routing)."""
    if not (_QPASS[0] and m <= _Q_MAX_M):
        return "exact"
    return "qd3" if m <= _D3_MAX_M else "qnest"


def d2_gate_up(enc: W12Weight, x: torch.Tensor):
    """(h, fused): fused=True -> h = SiLU(gate) * up [.., I]; fused=False -> h = gate_up [.., 2I] (caller applies act)."""
    k = enc.K
    m = x.numel() // k
    x2 = x.reshape(m, k)
    if D2_PREFILL and (m > D2_PREFILL_M or x2.dtype != torch.bfloat16 or torch.compiler.is_compiling()):
        return enc.sliced_linear(x2).view(*x.shape[:-1], enc.nout), True
    if 0 < m <= _CHUNK_M and x2.dtype == torch.bfloat16 and not torch.compiler.is_compiling():
        return _gemm_rows(enc, x2).view(*x.shape[:-1], enc.nout), True
    return F.linear(x2, enc.decode()).view(*x.shape[:-1], enc.N), False


# ---------------------------------------------------------------------------------------------------- model encoding


def _free(param):
    param.data = torch.empty((0, param.shape[1]), dtype=param.dtype, device=param.device)


def _drafter_derives_head() -> bool:
    """SPECPILOT_INT: True when the drafter builds a derived candidate head from the target's bf16 lm_head rows
    (SPECPILOT_DE head knobs set); the W12-encoded head's bf16 copy is then freed after that build."""
    try:
        from sglang.kernels.ops.speculative.dflash_de_head import read_de_head_knobs
    except ImportError:
        return False
    return read_de_head_knobs() is not None


def d2_free_pending_head(head) -> None:
    """SPECPILOT_INT: free a W12-encoded lm_head's bf16 weight kept for the drafter's derived head (no-op otherwise)."""
    if getattr(head, "_d2_free_pending", False) and getattr(head, "_d2_w12", None) is not None:
        _free(head.weight)
        head._d2_free_pending = False
        torch.cuda.empty_cache()


def _warmup_kernels(model: torch.nn.Module) -> None:
    """Launch every GEMM instantiation once (M = each NT, fused and plain) before the KV pool and graph capture:
    a first launch inside capture with the memory already handed to the KV pool failed (lazy module load / local
    memory sizing: "W12m launch failed: out of memory")."""
    encs = {}
    for mod in model.modules():
        enc = getattr(mod, "_d2_w12", None)
        if enc is not None:
            encs.setdefault(enc.silu, enc)
    for enc in encs.values():
        for nt in CONFIGS:
            enc.gemm(torch.zeros(nt, enc.K, dtype=torch.bfloat16, device=enc.planes.device))
        if D4_QVIEW:  # SPECPILOT_D4_QVIEW: the Q instantiations too
            for nt in QCONFIGS:
                enc.gemm(torch.zeros(nt, enc.K, dtype=torch.bfloat16, device=enc.planes.device), qv=1)
    for mod in model.modules():  # SPECPILOT_D3_QCOPY: every Q-copy instantiation (nt 16 / 32) once before capture
        enc = getattr(mod, "_d2_w12", None)
        if enc is not None and enc.d3q is not None:
            for m in (16, 32):
                enc.d3q(torch.zeros(m, enc.K, dtype=torch.bfloat16, device=enc.planes.device))
    torch.cuda.synchronize()


def d3_attach_qcopies(model: torch.nn.Module) -> None:
    """SPECPILOT_D3_QCOPY: attach a scaled INT6/INT8 g128 Q copy (from the exact decoded bf16 weight) to every encoded
    target linear except the lm_head; bits per group: MLP D3_QBITS_MLP, GDN in_proj D3_QBITS_IN, qkv/out/o D3_QBITS_REST."""
    from sglang.kernels.ops.gemm.specpilot_d3_qcopy import D3QLinear

    seen, nbytes, n = set(), 0, 0
    for name, mod in model.named_modules():
        enc = getattr(mod, "_d2_w12", None)
        if enc is None or enc.head or id(enc) in seen:
            continue
        seen.add(id(enc))
        if name.endswith("gate_up_proj") or name.endswith("down_proj"):
            bits = D3_QBITS_MLP
        elif name.endswith("in_proj_qkvz") or name.endswith("in_proj_ba"):
            bits = D3_QBITS_IN
        else:
            bits = D3_QBITS_REST
        if bits == 0:
            continue
        w = enc.decode()
        if enc.silu:  # decode() returns [gate | up] in the original row order
            enc.d3q = D3QLinear(w, bits, silu=True)
        else:
            enc.d3q = D3QLinear(w, bits)
        nbytes += enc.d3q.nbytes()
        n += 1
        del w
        torch.cuda.empty_cache()
    logger.info("SPECPILOT_D3_QCOPY: %d Q copies (MLP %d bit, in_proj %d bit, rest %d bit), %.2f GB", n, D3_QBITS_MLP,
                D3_QBITS_IN, D3_QBITS_REST, nbytes / 1e9)


def d4_attach_qhead(enc: W12Weight, w: torch.Tensor, nsub: int) -> None:
    """SPECPILOT_D4_QHEAD: attach a W12 copy of the drafter's vocabulary-subset rows (build_vocab_subset order, the
    rows the drafter can propose) to the encoded lm_head `enc` of bf16 weight `w`; q_pass() then serves the head from it."""
    from sglang.kernels.ops.speculative.dflash_de_head import build_vocab_subset

    sub = build_vocab_subset(nsub, w.shape[0]).to(device=w.device, dtype=torch.int64)
    enc.d4_sub = (sub, W12Weight(w.index_select(0, sub).contiguous(), head=True))


def _tied(model: torch.nn.Module, head) -> bool:
    emb = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
    ew = getattr(emb, "weight", None)
    return emb is head or (ew is not None and ew.data_ptr() == head.weight.data_ptr())


def _lm_head_fp32(model: torch.nn.Module) -> bool:
    lp = getattr(model, "logits_processor", None)
    return bool(getattr(lp, "use_fp32_lm_head", True))


def d2_encode_target(model: torch.nn.Module) -> None:
    """Encode the target's dense-MLP gate_up/down, GDN in_proj (qkvz + ba folded) and attention qkv_proj (bf16,
    unquantized, no bias, tp 1); with SPECPILOT_D2_W12_OUT also the GDN out_proj and attention o_proj, with
    SPECPILOT_D2_W12_HEAD the lm_head (bf16-logits configurations only); free their bf16 weights. Raises if any
    decoded weight is not bit-exact."""
    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
    from sglang.srt.models.qwen2_moe import Qwen2MoeMLP

    def plain(lin):
        return (isinstance(lin.quant_method, UnquantizedLinearMethod) and lin.bias is None
                and lin.weight.dtype == torch.bfloat16 and lin.weight.dim() == 2 and lin.weight.shape[0] > 0
                and lin.weight.shape[1] % 64 == 0 and lin.tp_size == 1)

    st = {"layers": 0, "bf16": 0, "w12": 0, "esc": 0, "weights": 0, "inexact": 0, "max_row_esc": 0}

    def account(enc, w, what):
        if not torch.equal(enc.decode(), w):
            bad = int((enc.decode() != w).sum().item())
            raise RuntimeError(f"SPECPILOT_D2 W12 not bit-exact at {what} {tuple(w.shape)}: {bad} mismatches")
        st["layers"] += 1
        st["bf16"] += w.numel() * 2
        st["w12"] += enc.nbytes()
        st["esc"] += enc.n_esc
        st["weights"] += w.numel()
        st["inexact"] += enc.n_inexact
        st["max_row_esc"] = max(st["max_row_esc"], enc.max_row_esc)

    def encode_linear(lin, what):
        enc = W12Weight(lin.weight.data)
        account(enc, lin.weight.data, what)
        lin._d2_w12 = enc
        _free(lin.weight)

    for name, mod in model.named_modules():
        if isinstance(mod, Qwen2MoeMLP) and plain(mod.gate_up_proj) and plain(mod.down_proj):
            gu, dn = mod.gate_up_proj, mod.down_proj
            if gu.weight.shape[0] % 128 == 0:
                enc = W12Weight(gu.weight.data, silu=True)
                account(enc, gu.weight.data, name + ".gate_up_proj")
                gu._d2_w12 = enc
                _free(gu.weight)
            encode_linear(dn, name + ".down_proj")
        elif type(mod).__name__ == "Qwen3_5GatedDeltaNet" and plain(mod.in_proj_qkvz) and plain(mod.in_proj_ba):
            width = mod.in_proj_qkvz.weight.shape[0]
            # SPECPILOT_INT: with SPECPILOT_X3_FUSE_BA the weights already live in the folded _fused_in_proj_weight
            # (qkvz/ba are views of it): encode that tensor and drop it, else the bf16 copy would stay resident.
            fused = getattr(mod, "_fused_in_proj_weight", None)
            if fused is None:
                fused = torch.cat([mod.in_proj_qkvz.weight.data, mod.in_proj_ba.weight.data], dim=0).contiguous()
            enc = W12Weight(fused)
            account(enc, fused, name + ".in_proj(qkvz+ba)")
            mod._d2_in_proj, mod._d2_in_proj_width = enc, width
            mod.in_proj_qkvz._d2_w12, mod.in_proj_qkvz._d2_rows = enc, (0, width)
            mod.in_proj_ba._d2_w12, mod.in_proj_ba._d2_rows = enc, (width, enc.N)
            mod._fused_in_proj_weight = None
            _free(mod.in_proj_qkvz.weight)
            _free(mod.in_proj_ba.weight)
            del fused
            if D2_OUT and plain(mod.out_proj):
                encode_linear(mod.out_proj, name + ".out_proj")
        elif hasattr(mod, "qkv_proj") and hasattr(mod, "o_proj") and plain(mod.qkv_proj):
            encode_linear(mod.qkv_proj, name + ".qkv_proj")
            if D2_OUT and plain(mod.o_proj):
                encode_linear(mod.o_proj, name + ".o_proj")
        torch.cuda.empty_cache()
    head = getattr(model, "lm_head", None)
    if D2_HEAD and head is not None:
        from sglang.srt.layers.quantization.unquant import UnquantizedEmbeddingMethod

        w = getattr(head, "weight", None)
        if (isinstance(getattr(head, "quant_method", None), UnquantizedEmbeddingMethod) and w is not None
                and w.dtype == torch.bfloat16 and w.dim() == 2 and w.shape[1] % 64 == 0
                and getattr(head, "tp_size", 1) == 1 and getattr(head, "bias", None) is None
                and not _lm_head_fp32(model) and not _tied(model, head)):
            enc = W12Weight(w.data, head=True)
            account(enc, w.data, "lm_head")
            if D4_QVIEW and D4_QHEAD:  # SPECPILOT_D4_QHEAD: W12 copy of the drafter's subset rows for the Q pass
                d4_attach_qhead(enc, w.data, _env_int("SPECPILOT_DE_VOCAB_SIZE", 98304))
                logger.info("SPECPILOT_D4_QHEAD: Q-pass head over %d of %d rows", enc.d4_sub[0].numel(), w.shape[0])
            head._d2_w12 = enc
            if _drafter_derives_head():
                # SPECPILOT_INT: the DFlash2 drafter derives its own candidate head (SPECPILOT_DE fp8 subset / D7 INT4)
                # from the bf16 rows when it attaches the target head; it frees them after (d2_free_pending_head).
                head._d2_free_pending = True
            else:
                _free(head.weight)
        else:
            logger.info("SPECPILOT_D2 W12: lm_head not encoded (quantized, tied, biased, tp > 1 or fp32 logits)")
        torch.cuda.empty_cache()
    if D2_PREFILL:
        for mod in model.modules():
            enc = getattr(mod, "_d2_w12", None)
            if enc is not None and enc.slices is None:
                enc.build_slices()
    if D3_QCOPY and D4_QVIEW:  # SPECPILOT_D3_QCOPY
        d3_attach_qcopies(model)
    if _D4_L2PF:  # SPECPILOT_D4_L2PF: per-layer prefetch plans (after the Q copies: their records are the Q stream)
        _l2pf.build_plans(model)
    _warmup_kernels(model)
    torch.cuda.empty_cache()
    logger.info(
        "SPECPILOT_D2 W12: encoded %d target linears (out/o_proj %d, lm_head %d, sliced prefill %d), %.2f GB bf16 -> "
        "%.2f GB W12 (%.4f), escapes %d (rate %.3g), max escapes in a row %d, roundtrip bit-exact (enforced), escape "
        "corrections below fp32 range %d", st["layers"], D2_OUT, D2_HEAD, D2_PREFILL, st["bf16"] / 1e9,
        st["w12"] / 1e9, st["w12"] / max(1, st["bf16"]), st["esc"], st["esc"] / max(1, st["weights"]),
        st["max_row_esc"], st["inexact"])
