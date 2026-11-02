"""SPECPILOT_D7W_W11: W11o, a lossless 11-bit weight code with an ordered (position-free) escape side stream, sm_120.

Record per (128-row tile, 64-k chunk), variable size (16-B multiple), offsets in ``roff`` (16-B units):
  SM plane   8192 B  [warp 8][half 2][lane 32][16 B]: byte [s m6..m0] per weight, A-fragment lane order (as W12's Q region)
  code plane 3072 B  [warp 8][word 3][lane 32][4 B]: 3-bit exponent code c per weight; byte b of words W0..W2 packs the
                     codes of the lane's 8 groups i (weights 4i+b): W0 = c0 | c1<<3 | c6[1:0]<<6,
                     W1 = c2 | c3<<3 | c6[2]<<6 | c7[0]<<7, W2 = c4 | c5<<3 | c7[2:1]<<6
  header     16 B    u16 start index of each warp's escapes in the stream
  stream     n B     one byte X = 128 ^ f_rel per escape (c == 7), in scan order: lane slot q = warp*32+lane,
                     then group a (i 0,1), b (i 2,3), c (i 4,5), d (i 6), e (i 7), then bit k = 8b + 3*(i&1) ascending
w = s * 2^(f_rel - 127) * 1.m * 2^(E_n - 127): per-row 7-binade window top E_n (best placement), in-window
f_rel = 121 + c; escapes (c = 7: outside the window, zeros, denormals) take f_rel = (121 + 7) ^ X, exact (zeros:
f_rel = 0). Kernel: W12's stream-K mma.sync kernel (1 producer, 8 decode consumers, no escape warps); each consumer
lane finds its escapes from the code masks (popc + warp shfl scan + the header's warp start), reads their stream bytes
and decodes F = (c + 121) ^ X in registers. Decoded weights are bit-identical; outputs equal W12's wherever W12 had no escape.

``W11Weight`` duck-types ``specpilot_d2_w12.W12Weight`` for the d2 entry points (d2_linear, d2_gate_up,
_warmup_kernels): gemm(x2) for M <= 64, decode() into a shared bf16 scratch for the M > 128 path. It has no nested Q
view (q_pass GEMMs run the exact kernel unless a D3Q copy is attached), no prefill slices and no L2-prefetch plan.
"""
from __future__ import annotations

import hashlib
import os

import torch

_RB = 128
_FIX = 11264
_ESCCAP = 1024
_MAX_M = 64
ROT = 37
CONFIGS = {16: 3, 32: 3, 48: 2, 64: 2}


def _frag(p, R, K):
    # row = rg*16 + h*8 + g, col = kc*64 + j*16 + c8*8 + t*2 + b  ->  [rg, kc, lane=(g,t), w=(j, c8, h, b)]
    return p.view(R // 16, 2, 8, K // 64, 4, 2, 4, 2).permute(0, 3, 2, 6, 4, 5, 1, 7).reshape(R // 16, K // 64, 32, 32)


def _stream_perm():
    keys = []
    for w in range(32):
        i, b = w // 4, w % 4
        g = i // 2 if i < 6 else (3 if i == 6 else 4)
        k = 8 * b + (3 * (i & 1) if i < 6 else 0)
        keys.append(g * 32 + k)
    return torch.tensor(sorted(range(32), key=lambda w: keys[w]))


_PERM = _stream_perm()


def _maxseg(T, S, P):
    U = T * S
    own = lambda u: ((u + 1) * P - 1) // U  # noqa: E731
    return max(own(t * S + S - 1) - own(t * S) + 1 for t in range(T))


def _encode_chunk(wc: torch.Tensor):
    R, K = wc.shape
    S = K // 64
    dev = wc.device
    u = wc.view(torch.int16).to(torch.int32) & 0xFFFF
    e = (u >> 7) & 0xFF
    m = u & 0x7F
    hist = torch.zeros(R, 257, dtype=torch.int32, device=dev)
    hist.scatter_add_(1, e.to(torch.int64), torch.ones_like(e))
    hist[:, 0] = 0
    cum = torch.cumsum(hist, dim=1)
    tops = torch.arange(6, 256, device=dev)
    lo = tops - 7
    cover = cum[:, tops] - torch.where(lo[None, :] >= 0, cum[:, lo.clamp(min=0)], torch.zeros_like(cum[:, :1]))
    best = cover.max(dim=1, keepdim=True).values
    E = tops[(cover == best).int().cumsum(1).argmax(dim=1)].to(torch.int32)
    c = e - (E[:, None] - 6)
    esc = (c < 0) | (c > 6) | (e == 0)
    c = torch.where(esc, torch.full_like(c, 7), c)
    f = e - E[:, None] + 127
    inexact = esc & (((e == 0) & (m != 0)) | ((e != 0) & (f <= 0)) | (f > 255))
    f = torch.where(e == 0, torch.zeros_like(f), f.clamp(0, 255))
    X = torch.where(esc, 128 ^ f, torch.zeros_like(f))
    sm = ((u >> 8) & 0x80) | m
    T = R // _RB
    smf = _frag(sm.to(torch.uint8), R, K).view(R // 16, S, 32, 2, 16).transpose(2, 3).reshape(R // 16, S, 1024)
    smrec = smf.view(T, 8, S, 1024).transpose(1, 2).reshape(T, S, 8192)
    cf = _frag(c.to(torch.uint8), R, K).view(R // 16, S, 32, 8, 4)
    ci = [cf[:, :, :, i, :] for i in range(8)]
    W0 = ci[0] | (ci[1] << 3) | ((ci[6] & 3) << 6)
    W1 = ci[2] | (ci[3] << 3) | (((ci[6] >> 2) & 1) << 6) | ((ci[7] & 1) << 7)
    W2 = ci[4] | (ci[5] << 3) | (((ci[7] >> 1) & 3) << 6)
    cw = torch.stack([W0, W1, W2], dim=2).reshape(R // 16, S, 384)  # [rg, s, word, lane, byte]
    crec = cw.view(T, 8, S, 384).transpose(1, 2).reshape(T, S, 3072)
    fixed = torch.cat([smrec, crec], dim=2).reshape(T * S, _FIX)
    perm = _PERM.to(dev)
    escf = _frag(esc.to(torch.uint8), R, K)[..., perm].view(T, 8, S, 32, 32).transpose(1, 2).reshape(T * S, 8192)
    Xf = _frag(X.to(torch.uint8), R, K)[..., perm].view(T, 8, S, 32, 32).transpose(1, 2).reshape(T * S, 8192)
    cnt = escf.sum(1, dtype=torch.int64)
    if int(cnt.max()) > _ESCCAP - 16:
        raise ValueError(f"W11o: {int(cnt.max())} escapes in one record > {_ESCCAP - 16}")
    wc = escf.view(-1, 8, 1024).sum(2, dtype=torch.int64)
    hdr = (torch.cumsum(wc, 1) - wc).to(torch.int32)
    hdrb = torch.stack([hdr & 0xFF, hdr >> 8], dim=2).reshape(-1, 16).to(torch.uint8)
    fixed = torch.cat([fixed, hdrb], dim=1)
    size = _FIX + 16 + (cnt + 15) // 16 * 16
    off = torch.cumsum(size, 0) - size
    total = int(size.sum())
    data = torch.zeros(total, dtype=torch.uint8, device=dev)
    nrec = T * S
    idx = off[:, None] + torch.arange(_FIX + 16, device=dev)[None, :]
    data[idx.view(-1)] = fixed.view(-1)
    escb = escf.bool()
    rank = torch.cumsum(escf.to(torch.int64), 1) - 1
    pos = off[:, None] + _FIX + 16 + rank
    data[pos[escb]] = Xf[escb]
    del idx
    return data, size, E, int(esc.sum()), int(inexact.sum()), cnt, nrec


class W11Weight:
    """Bit-exact W11o encoding of a [N, K] bf16 weight; silu=True interleaves gate_up as [64 gate | 64 up]."""

    def __init__(self, w: torch.Tensor, silu: bool = False, chunk_rows: int = 2048, nsm: int | None = None,
                 head: bool = False):
        assert w.dtype == torch.bfloat16 and w.dim() == 2
        N, K = w.shape
        assert K % 64 == 0
        self.N, self.K, self.silu = N, K, int(silu)
        self.head = bool(head)  # lm_head: never decoded whole (no N x K scratch)
        self.d3q = None  # W12Weight interface: D3Q copy, Q-head subset, L2 prefetch plan, prefill slices
        self.d4_sub = None
        self.d4_pf = None
        self.slices = None
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
        datas, sizes = [], []
        E_all = torch.empty(T * _RB, dtype=torch.int32, device=dev)
        self.n_esc = self.n_inexact = 0
        cnts = []
        for r0 in range(0, T * _RB, chunk_rows):
            r1 = min(T * _RB, r0 + chunk_rows)
            wc = wi[r0:min(r1, N)]
            if wc.shape[0] < r1 - r0:
                wc = torch.cat([wc, torch.ones(r1 - r0 - wc.shape[0], K, dtype=w.dtype, device=dev)])
            data, size, E, ne, ni, cnt, _ = _encode_chunk(wc.contiguous())
            datas.append(data)
            sizes.append(size)
            cnts.append(cnt)
            E_all[r0:r1] = E
            self.n_esc += ne
            self.n_inexact += ni
        self.data = torch.cat(datas)
        size = torch.cat(sizes)
        self.rec_esc = torch.cat(cnts)
        self.max_row_esc = int(self.rec_esc.max()) if self.rec_esc.numel() else 0  # W12 stats field: max per record here
        roff = torch.zeros(size.numel() + 1, dtype=torch.int64, device=dev)
        roff[1:] = torch.cumsum(size, 0)
        assert int(roff[-1]) // 16 < 2 ** 31
        self.roff = (roff // 16).to(torch.int32)
        self.E = E_all
        self.scale = torch.pow(2.0, (E_all - 127).double()).float()
        if nsm is None:
            nsm = torch.cuda.get_device_properties(dev).multi_processor_count if dev.type == "cuda" else 188
        self.nsm = nsm
        self.set_P(None)

    def set_P(self, P):
        T, S = self.T, self.S
        if P is None:
            P = T * max(1, self.nsm // T) if T <= self.nsm else self.nsm
        self.P = min(P, T * S)
        self.maxseg = _maxseg(T, S, self.P)
        self.maxrec = -(-T * S // self.P) + 1
        if self.data.device.type == "cuda":
            _ensure_shared(self.data.device, T * self.maxseg * _RB * _MAX_M, T, 0 if self.head else self.N * self.K)

    @property
    def planes(self) -> torch.Tensor:
        return self.data

    def nbytes(self):
        return self.data.numel() + self.roff.numel() * 4 + self.scale.numel() * 4

    def decode(self) -> torch.Tensor:
        """bf16 [N, K] (original row order) via the GPU decode kernel, in the shared scratch when it is large enough."""
        sc = _SHARED[self.data.device].get("scratch")
        if sc is not None and sc.numel() >= self.N * self.K:
            out = sc[: self.N * self.K].view(self.N, self.K)
        else:
            out = torch.empty((self.N, self.K), dtype=torch.bfloat16, device=self.data.device)
        _ext().w11_decode(self.data, self.roff, self.scale, out, self.N, self.K, self.silu)
        return out

    def decode_reference(self, tiles_per_batch: int = 16) -> torch.Tensor:
        """Reference decoder (torch, mask-based scan order, independent of the CUDA code): bf16 [N, K]."""
        K = self.K
        rows = torch.cat([self._decode_tiles(t0, min(self.T, t0 + tiles_per_batch))
                          for t0 in range(0, self.T, tiles_per_batch)])
        if self.silu:
            I = self.N // 2
            rows = rows.view(I // 64, 2, 64, K).transpose(0, 1).reshape(self.N, K)
        return rows[: self.N]

    def _decode_tiles(self, t0: int, t1: int) -> torch.Tensor:
        T, S, K = t1 - t0, self.S, self.K
        dev = self.data.device
        nrec = T * S
        off = self.roff[t0 * S: t1 * S].to(torch.int64) * 16
        fixed = self.data[(off[:, None] + torch.arange(_FIX, device=dev)[None, :]).view(-1)].view(nrec, _FIX)
        smp = fixed[:, :8192].view(T, S, 8, 2, 32, 16).transpose(3, 4).reshape(T, S, 8, 32, 32)  # [.., lane, w]
        cw = fixed[:, 8192:].view(T, S, 8, 3, 32, 4).to(torch.int32)  # [.., word, lane, byte]
        W0, W1, W2 = cw[:, :, :, 0], cw[:, :, :, 1], cw[:, :, :, 2]
        C = [W0 & 7, (W0 >> 3) & 7, W1 & 7, (W1 >> 3) & 7, W2 & 7, (W2 >> 3) & 7,
             ((W0 >> 6) & 3) | (((W1 >> 6) & 1) << 2), ((W1 >> 7) & 1) | (((W2 >> 6) & 3) << 1)]
        c = torch.stack(C, dim=4).reshape(T, S, 8, 32, 32)  # w = 4i + b
        # helper order: masks per group, bit k = 8b + {0, 3}
        A = [(Wk & (Wk >> 1) & (Wk >> 2)) for Wk in (W0, W1, W2)]
        flags = torch.zeros(T, S, 8, 32, 5, 32, dtype=torch.bool, device=dev)
        bidx = torch.arange(4, device=dev)
        for gi, Ak in enumerate(A):
            flags[..., gi, 8 * bidx] = (Ak & 1).bool()
            flags[..., gi, 8 * bidx + 3] = ((Ak >> 3) & 1).bool()
        flags[..., 3, 8 * bidx] = ((W0 >> 6) & (W0 >> 7) & (W1 >> 6) & 1).bool()
        flags[..., 4, 8 * bidx] = ((W1 >> 7) & (W2 >> 6) & (W2 >> 7) & 1).bool()
        fl = flags.view(nrec, -1)
        rank = torch.cumsum(fl.to(torch.int64), 1) - 1
        pos = off[:, None] + _FIX + 16 + rank
        Xs = torch.zeros_like(fl, dtype=torch.int32)
        Xs[fl] = self.data[pos[fl]].to(torch.int32)
        Xs = Xs.view(T, S, 8, 32, 5, 32)
        # (group, k) -> w
        X = torch.zeros(T, S, 8, 32, 32, dtype=torch.int32, device=dev)
        Fw = torch.zeros(T, S, 8, 32, 32, dtype=torch.bool, device=dev)
        fl6 = flags.view(T, S, 8, 32, 5, 32)
        for w in range(32):
            i, b = w // 4, w % 4
            g = i // 2 if i < 6 else (3 if i == 6 else 4)
            k = 8 * b + (3 * (i & 1) if i < 6 else 0)
            X[..., w] = Xs[..., g, k]
            Fw[..., w] = fl6[..., g, k]
        assert torch.equal(c == 7, Fw), "escape flags disagree with codes"
        F = (c + 121) ^ X
        sm = smp.to(torch.int32)
        # absolute bf16 bits: e = F + E - 127 (F == 0 -> 0)
        Er = self.E[t0 * _RB: t1 * _RB].view(T, 8, 2, 8)  # rows: tile, warp, h, g
        # w index -> h = (w >> 1) & 1 ; lane -> g = lane // 4
        hh = (torch.arange(32, device=dev) >> 1) & 1
        gg = torch.arange(32, device=dev) // 4
        Erow = Er[:, :, hh[None, :], gg[:, None]]  # [T, 8, lane, w]
        Erow = Erow[:, None]  # [T, 1, 8, lane, w]
        eabs = torch.where(F == 0, torch.zeros_like(F), F + Erow - 127)
        bits = ((sm & 0x80) << 8) | (eabs << 7) | (sm & 0x7F)
        # back to [rows, K]: [T, S, warp, lane, w] -> frag inverse
        b5 = bits.permute(0, 2, 1, 3, 4).reshape(T * 8, S, 32, 32)
        # frag: [rg, kc, lane=(g,t), w=(j,c8,h,b)] -> rows (rg, h, g), cols (kc, j, c8, t, b)
        r = b5.view(T * 8, S, 8, 4, 4, 2, 2, 2).permute(0, 6, 2, 1, 4, 5, 3, 7).reshape(T * _RB, K)
        out16 = (r & 0xFFFF).to(torch.int32)
        out16 = torch.where(out16 >= 32768, out16 - 65536, out16).to(torch.int16)
        return out16.view(torch.bfloat16)

    def gemm(self, x2: torch.Tensor, out=None, qv: int = 0, ns: int = 0, abl: int = 0) -> torch.Tensor:
        """x2 [M <= 64, K] bf16 -> [M, nout]; qv is accepted for the W12 interface and runs the exact kernel."""
        m = x2.shape[0]
        if x2.stride(-1) != 1 or x2.stride(0) % 8 or x2.data_ptr() % 16:
            x2 = x2.contiguous()
        nt = next(n for n in sorted(CONFIGS) if m <= n)
        ns = ns or CONFIGS[nt]
        sh = _SHARED[x2.device]
        if out is None:
            out = torch.empty((m, self.nout), dtype=torch.bfloat16, device=x2.device)
        _ext().w11_gemm(x2, self.data, self.roff, self.scale, out, sh["ws"], sh["cnt"], self.P, self.maxseg,
                        self.maxrec, self.N, self.T, self.silu, nt, ns, 1, 1, ROT, abl)
        return out


_SHARED: dict = {}


def _ensure_shared(dev, ws_floats, ntiles, scratch_elems=0):
    """Split-K workspace, tile counters and bf16 decode scratch shared by every W11 weight on the device (load time)."""
    sh = _SHARED.setdefault(dev, {})
    if sh.get("scratch") is None or sh["scratch"].numel() < scratch_elems:
        sh["scratch"] = torch.empty(scratch_elems, dtype=torch.bfloat16, device=dev)
    if sh.get("ws") is None or sh["ws"].numel() < ws_floats:
        sh["ws"] = torch.empty(ws_floats, dtype=torch.float32, device=dev)
    if sh.get("cnt") is None or sh["cnt"].numel() < ntiles:
        sh["cnt"] = torch.zeros(ntiles, dtype=torch.int32, device=dev)


_SRC = r"""
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

#define NCW 8
#define RB 128
#define FIX 11264
#define RCAP 12288

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
  asm volatile("{\n .reg .pred p;\n W11W_%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra W11W_%=;\n}\n"
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

// SM bytes (2 uint4) + code words (3) + escape X bytes (8 words, 0 off escapes) -> A fragments of 4 k16 steps
__device__ __forceinline__ void decode11(const uint4 s0, const uint4 s1, uint32_t W0, uint32_t W1, uint32_t W2,
                                         const uint32_t (&X)[8], uint32_t (&A)[4][4]) {
  const uint32_t SM[8] = {s0.x, s0.y, s0.z, s0.w, s1.x, s1.y, s1.z, s1.w};
  uint32_t C[8];
  C[0] = W0 & 0x07070707u;
  C[1] = (W0 >> 3) & 0x07070707u;
  C[2] = W1 & 0x07070707u;
  C[3] = (W1 >> 3) & 0x07070707u;
  C[4] = W2 & 0x07070707u;
  C[5] = (W2 >> 3) & 0x07070707u;
  C[6] = ((W0 >> 6) & 0x03030303u) | ((W1 >> 4) & 0x04040404u);
  C[7] = ((W1 >> 7) & 0x01010101u) | ((W2 >> 5) & 0x06060606u);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const uint32_t F = (C[i] + 0x79797979u) ^ X[i];
    const uint32_t lo = (SM[i] & 0x7F7F7F7Fu) | ((F << 7) & 0x80808080u);
    const uint32_t hi = (SM[i] & 0x80808080u) | ((F >> 1) & 0x7F7F7F7Fu);
    A[i >> 1][2 * (i & 1)] = __byte_perm(lo, hi, 0x5140);
    A[i >> 1][2 * (i & 1) + 1] = __byte_perm(lo, hi, 0x7362);
  }
}

// one lane's 32 weights of a record (smem or global) -> A fragments; esc = false skips the escape patch (bench)
__device__ __forceinline__ void w11_lane_A(const unsigned char* rec, int warp, int lane, bool esc, uint32_t (&A)[4][4]) {
  const uint4* sw = reinterpret_cast<const uint4*>(rec);
  const uint32_t* cw = reinterpret_cast<const uint32_t*>(rec + 8192 + warp * 384);
  const uint4 s0 = sw[warp * 64 + lane], s1 = sw[warp * 64 + 32 + lane];
  const uint32_t W0 = cw[lane], W1 = cw[32 + lane], W2 = cw[64 + lane];
  uint32_t X[8] = {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u};
  if (esc) {
    const uint32_t mg[5] = {W0 & (W0 >> 1) & (W0 >> 2) & 0x09090909u, W1 & (W1 >> 1) & (W1 >> 2) & 0x09090909u,
                            W2 & (W2 >> 1) & (W2 >> 2) & 0x09090909u, (W0 >> 6) & (W0 >> 7) & (W1 >> 6) & 0x01010101u,
                            (W1 >> 7) & (W2 >> 6) & (W2 >> 7) & 0x01010101u};
    const int cnt = __popc(mg[0]) + __popc(mg[1]) + __popc(mg[2]) + __popc(mg[3]) + __popc(mg[4]);
    int incl = cnt;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
      const int v = __shfl_up_sync(0xFFFFFFFFu, incl, d);
      if (lane >= d) incl += v;
    }
    int pos = (int)reinterpret_cast<const uint16_t*>(rec + FIX)[warp] + incl - cnt;
    const unsigned char* st = rec + FIX + 16;
#pragma unroll
    for (int gq = 0; gq < 3; ++gq) {
      uint32_t mm = mg[gq];
      while (mm) {
        const int k = __ffs(mm) - 1;
        mm &= mm - 1;
        const uint32_t v = (uint32_t)st[pos++] << (k & 0x18);
        if (k & 7) X[2 * gq + 1] |= v; else X[2 * gq] |= v;
      }
    }
#pragma unroll
    for (int gq = 3; gq < 5; ++gq) {
      uint32_t mm = mg[gq];
      while (mm) {
        const int k = __ffs(mm) - 1;
        mm &= mm - 1;
        X[gq + 3] |= (uint32_t)st[pos++] << k;
      }
    }
  }
  decode11(s0, s1, W0, W1, W2, X, A);
}

// ABL bits (benchmarks only): 1 = no escape processing (X = 0: no-escape bound)
template <int NT, int NS, int ABL>
__global__ void __launch_bounds__(NCW * 32 + 32, 1) w11_kernel(
    const __grid_constant__ CUtensorMap xmap, int M, const unsigned char* __restrict__ data,
    const int* __restrict__ roff, int S, int U, const float* __restrict__ scale,
    __nv_bfloat16* __restrict__ out, int64_t ldo, int N, float* __restrict__ ws, int maxseg, int maxrec,
    int* __restrict__ counters, int silu, int nout, int trig, int rot) {
  constexpr int NQ = NT / 8;
  constexpr int XB = NT * 128;
  constexpr int SB = RCAP + XB;  // multiple of 1024
  extern __shared__ __align__(1024) unsigned char smem[];
  __shared__ __align__(8) uint64_t bars[2 * NS];
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int P = gridDim.x, p = blockIdx.x;
  const int u0 = (int)(((int64_t)p * U) / P), u1 = (int)(((int64_t)(p + 1) * U) / P);
  const uint32_t sraw = (uint32_t)__cvta_generic_to_shared(smem);
  const uint32_t sbase = (sraw + 1023u) & ~1023u;
  unsigned char* const smem_al = smem + (sbase - sraw);
  float* xchg = reinterpret_cast<float*>(smem_al + NS * SB);
  int* soff = reinterpret_cast<int*>(smem_al + NS * SB + (silu ? 64 * NT * 4 : 0));
  const uint32_t full0 = (uint32_t)__cvta_generic_to_shared(bars), empty0 = full0 + 8 * NS;
  if (threadIdx.x == 0) {
    for (int s = 0; s < NS; ++s) { mbar_init(full0 + 8 * s, 1); mbar_init(empty0 + 8 * s, NCW); }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  // record offsets of this CTA's range [u0, u1] (16-B units)
  for (int r = threadIdx.x; r <= u1 - u0; r += blockDim.x) soff[r] = roff[u0 + r];
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
          const int r = tile * S + s - u0;
          const uint32_t bytes = (uint32_t)(soff[r + 1] - soff[r]) * 16u;
          const uint32_t fb = full0 + 8 * i;
          mbar_expect_tx(fb, bytes + XB);
          bulk_g2s(sbase + i * SB, data + (int64_t)soff[r] * 16, bytes, fb);
        }
      }
      asm volatile("griddepcontrol.wait;\n" ::: "memory");
#pragma unroll
      for (int i = 0; i < NS; ++i)
        if (i < npre) tma2d(sbase + i * SB + RCAP, &xmap, spre[i] * 64, 0, full0 + 8 * i);
      for (int u = u0 + npre; u < u1; ++u) {
        const int i = u - u0, slot = i % NS;
        mbar_wait(empty0 + 8 * slot, ((i / NS) - 1) & 1);
        const int s = walk(u);
        const int r = tile * S + s - u0;
        const uint32_t bytes = (uint32_t)(soff[r + 1] - soff[r]) * 16u;
        const uint32_t fb = full0 + 8 * slot, dst = sbase + slot * SB;
        mbar_expect_tx(fb, bytes + XB);
        bulk_g2s(dst, data + (int64_t)soff[r] * 16, bytes, fb);
        tma2d(dst + RCAP, &xmap, s * 64, 0, fb);
      }
    }
    return;
  }
  // consumers: warp w owns weight rows [w*16, w*16+16) of the tile; tokens as NQ n-tiles of 8
  const int g = lane / 4, t = lane % 4;
  float acc[NQ * 4];
  const int lm_r = lane & 7, lm_kh = (lane >> 3) & 1, lm_nt = lane >> 4;
  auto stage = [&](int i) {
    const int slot = i % NS;
    mbar_wait(full0 + 8 * slot, (i / NS) & 1);
    const unsigned char* rec = smem_al + slot * SB;
    uint32_t A[4][4];
    w11_lane_A(rec, warp, lane, !(ABL & 1), A);
    const uint32_t xs = sbase + slot * SB + RCAP;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
#pragma unroll
      for (int qp = 0; qp < NQ / 2; ++qp) {
        uint32_t b[4];
        const int n = (2 * qp + lm_nt) * 8 + lm_r, ch = 2 * j + lm_kh;
        ldsm_x4(b, xs + n * 128 + ((ch ^ lm_r) << 4));
        mma16816(acc + (2 * qp) * 4, A[j], b[0], b[1]);
        mma16816(acc + (2 * qp + 1) * 4, A[j], b[2], b[3]);
      }
      if (NQ & 1) {
        uint32_t b[2];
        const int n = (NQ - 1) * 8 + lm_r, ch = 2 * j + lm_kh;
        ldsm_x2(b, xs + n * 128 + ((ch ^ lm_r) << 4));
        mma16816(acc + (NQ - 1) * 4, A[j], b[0], b[1]);
      }
    }
    __syncwarp();
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    if (lane == 0) mbar_arrive(empty0 + 8 * slot);
  };
  for (int u = u0; u < u1;) {
    const int tile = u / S, s0 = u - tile * S;
    const int uend = min(u1, (tile + 1) * S);
    const int n0 = tile * RB + warp * 16 + g;
    const float sc0 = n0 < N ? ldg_early(scale + n0) : 1.f, sc1 = n0 + 8 < N ? ldg_early(scale + n0 + 8) : 1.f;
#pragma unroll
    for (int q = 0; q < NQ * 4; ++q) acc[q] = 0.f;
    for (int uu = u; uu < uend; ++uu) stage(uu - u0);
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

template <int NT, int NS, int ABL>
void launch(torch::Tensor x, torch::Tensor data, torch::Tensor roff, torch::Tensor scale, torch::Tensor out,
            torch::Tensor ws, torch::Tensor cnt, int64_t P, int64_t maxseg, int64_t maxrec, int64_t N, int64_t T,
            int64_t silu, int64_t pdl, int64_t trig, int64_t rot) {
  const int M = x.size(0), K = x.size(1), nout = out.size(1);
  TORCH_CHECK(M <= NT && M > 0, "W11: bad M");
  const int S = K / 64, U = (int)T * S;
  CUtensorMap map;
  cuuint64_t gdim[2] = {(cuuint64_t)K, (cuuint64_t)M};
  cuuint64_t gstr[1] = {(cuuint64_t)x.stride(0) * 2};
  cuuint32_t box[2] = {64, NT};
  cuuint32_t es[2] = {1, 1};
  CUresult r = get_encode()(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, x.data_ptr(), gdim, gstr, box, es,
                            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "W11 tensor map encode failed ", (int)r);
  constexpr int SB = RCAP + NT * 128;
  const int smem = NS * SB + (silu ? 64 * NT * 4 : 0) + (int)(maxrec + 2) * 4 + 1024;
  auto kern = w11_kernel<NT, NS, ABL>;
  static int attr = 0;
  if (smem > attr) {
    cudaError_t ea = cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    TORCH_CHECK(ea == cudaSuccess, "W11 smem attr failed: ", cudaGetErrorString(ea), " smem ", smem);
    attr = smem;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((unsigned)P);
  cfg.blockDim = dim3(NCW * 32 + 32);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = c10::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
  cfg.attrs = at;
  cfg.numAttrs = 1;
  cudaError_t e = cudaLaunchKernelEx(&cfg, kern, map, M, reinterpret_cast<const unsigned char*>(data.data_ptr()),
      (const int*)roff.data_ptr<int>(), S, U, (const float*)scale.data_ptr<float>(),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), (int64_t)out.stride(0), (int)N, ws.data_ptr<float>(),
      (int)maxseg, (int)maxrec, cnt.data_ptr<int>(), (int)silu, nout, (int)trig, (int)rot);
  TORCH_CHECK(e == cudaSuccess, "W11 launch failed: ", cudaGetErrorString(e));
}

@@DISPATCH@@

// records -> bf16 weight in the original row order; one block = one (tile, 64-k chunk) record
__global__ void __launch_bounds__(256) w11_decode_kernel(const unsigned char* __restrict__ data, const int* __restrict__ roff,
                                                         int S, const float* __restrict__ scale,
                                                         __nv_bfloat16* __restrict__ out, int N, int K, int silu) {
  __shared__ uint32_t tl[128][33];
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  const int kc = blockIdx.x % S, tile = blockIdx.x / S;
  uint32_t A[4][4];
  w11_lane_A(data + (int64_t)roff[blockIdx.x] * 16, w, lane, true, A);
  const int g = lane / 4, t = lane % 4;
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const int rl = w * 16 + g + 8 * h;
    const uint32_t E = (__float_as_uint(scale[tile * 128 + rl]) >> 23) & 0xFF;
    const uint32_t add = ((E - 127u) << 7) & 0xFFFFu;
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
      for (int c8 = 0; c8 < 2; ++c8) {
        const uint32_t v = A[j][2 * c8 + h];
        uint32_t lo = v & 0xFFFFu, hi = v >> 16;
        if (lo & 0x7F80u) lo = (lo + add) & 0xFFFFu;  // exponent field 0 (zeros, denormals): absolute already
        if (hi & 0x7F80u) hi = (hi + add) & 0xFFFFu;
        tl[rl][j * 8 + c8 * 4 + t] = lo | (hi << 16);
      }
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

void w11_decode(torch::Tensor data, torch::Tensor roff, torch::Tensor scale, torch::Tensor out, int64_t N, int64_t K,
                int64_t silu) {
  const int S = K / 64, R = (int)roff.size(0) - 1;
  w11_decode_kernel<<<(unsigned)R, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const unsigned char*>(data.data_ptr()), roff.data_ptr<int>(), S, scale.data_ptr<float>(),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), (int)N, (int)K, (int)silu);
}
"""

_NS_ALL = {16: (3, 4), 32: (3, 4), 48: (2, 3), 64: (2, 3)}


def _dispatch_src():
    lines = ["void w11_gemm(torch::Tensor x, torch::Tensor data, torch::Tensor roff, torch::Tensor scale, torch::Tensor out,"
             " torch::Tensor ws, torch::Tensor cnt, int64_t P, int64_t maxseg, int64_t maxrec, int64_t N, int64_t T,"
             " int64_t silu, int64_t nt, int64_t ns, int64_t pdl, int64_t trig, int64_t rot, int64_t abl) {"]
    args = "x, data, roff, scale, out, ws, cnt, P, maxseg, maxrec, N, T, silu, pdl, trig, rot"
    for nt, nss in _NS_ALL.items():
        for ns in nss:
            for abl in (0, 1):
                lines.append(f"  if (nt == {nt} && ns == {ns} && abl == {abl}) {{ launch<{nt}, {ns}, {abl}>({args}); return; }}")
    lines.append('  TORCH_CHECK(false, "W11: no instantiation for nt ", nt, " ns ", ns, " abl ", abl);\n}')
    return "\n".join(lines)


_CPP = ("void w11_gemm(torch::Tensor x, torch::Tensor data, torch::Tensor roff, torch::Tensor scale, torch::Tensor out, "
        "torch::Tensor ws, torch::Tensor cnt, int64_t P, int64_t maxseg, int64_t maxrec, int64_t N, int64_t T, "
        "int64_t silu, int64_t nt, int64_t ns, int64_t pdl, int64_t trig, int64_t rot, int64_t abl);\n"
        "void w11_decode(torch::Tensor data, torch::Tensor roff, torch::Tensor scale, torch::Tensor out, int64_t N, "
        "int64_t K, int64_t silu);\n")

_mod = None


def _ext():
    global _mod
    if _mod is None:
        from torch.utils.cpp_extension import load_inline

        os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0a"
        src = _SRC.replace("@@DISPATCH@@", _dispatch_src())
        cflags = ["-O3", "-gencode=arch=compute_120a,code=sm_120a", "--use_fast_math"]
        key = hashlib.sha1("\0".join([_CPP, src, *cflags]).encode()).hexdigest()[:16]
        base = os.environ.get("SPECPILOT_D7W_BUILD") or os.environ.get("SGLANG_CACHE_DIR") or os.path.join(os.path.expanduser("~"), ".cache", "sglang")
        bdir = os.path.join(base, "specpilot_d7_w11", key)
        os.makedirs(bdir, exist_ok=True)
        _mod = load_inline(name=f"specpilot_d7_w11_{key}", cpp_sources=_CPP, cuda_sources=src,
                           functions=["w11_gemm", "w11_decode"], extra_cuda_cflags=cflags, build_directory=bdir, verbose=False)
    return _mod
