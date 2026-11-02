"""SPECPILOT_D5: W12 lossless 12-bit split-plane weight streaming for the target's verify-width linears.

Coding (bit-exact) of a bf16 weight w = s * 2^(e-127) * 1.m of row n with window top E_n (the 16-binade window
[E_n-15, E_n] covering the most weights of the row, ties -> highest; >= 15):
c = e - (E_n - 15) in [0, 15]; the decoded w~ = s * 2^(c-15) * 1.m has bf16 bits [s 0111 c3 c2 c1 | c0 m6..m0], so a
low-byte plane stores [c0 m] and a 4-bit plane stores [s c3 c2 c1] (1.5 B/weight). w = w~ * 2^(E_n-127) exactly: the
kernel scales the fp32 accumulator by scale[n] = 2^(E_n-127). Out-of-window weights (e outside the window, zeros, denormals) are
stored as w~ = 2^-15 and corrected exactly: acc += x * w~_true - x * 2^-15 (CSR per row; w~_true is exact in
fp32, checked at encode; both products are exact).
NESTED PLANES: per (128-row tile, 64-k chunk) a 12 KB record = Q region [8 warps][1 KB] (one byte per weight,
[c0 m6 m5 m4 | s c3 c2 c1]) + B region [8 warps][512 B] ([m3..m0] nibbles), both in wgmma-A-fragment lane order. The
exact GEMM streams the whole record (bitwise-equal to the earlier 2-plane layout); the Q pass (`q_pass()`, two-level
verify) streams only the Q regions (1 B/weight, 128-k stages: sign, exponent code and 3 mantissa bits + the midpoint
of the dropped ones; only the escapes above the row window (|w~| >= 2) are corrected, with the Q placeholder
1.0625 * 2^-15; the ones below it keep the placeholder) -- an approximate copy
at zero extra memory.

GEMM (M <= 64): persistent stream-K Hopper kernel (one CTA per SM, P CTAs over the flattened tile x k-stage range,
stage order rotated per segment). Roles: 2 consumer warpgroups (lds planes -> 2 byte_perm + 2 LOP3 per weight pair ->
wgmma m64nNTk16 with A from registers, B = x from smem), a producer warp (weights by 1-D bulk TMA, x by a 3-D tensor
map into wgmma no-swizzle K-major core matrices, 5-stage full/empty mbarrier ring), an escape warp (exact sparse
correction tile into double-buffered smem). Split tiles: contributors publish fp32 partials (red.release), the owner
of the tile's k = 0 segment reduces them in segment order (deterministic for a fixed P; all partial loads in flight
before the ordered sum). gate_up and down kernels trigger their dependents (the down GEMM; the next x3 add+RMSNorm) at start. gate_up: tiles of [64 gate |
64 up] rows with a fused SiLU(gate) * up epilogue on bf16-rounded operands (as the act_and_mul path).
M > 64 (prefill): the layer weight is decoded into a shared bf16 scratch and cuBLAS runs (F.linear).

Knob (read once at import): SPECPILOT_D5_W12=1 encodes the target's dense-MLP gate_up/down, the folded GDN in_proj
(qkvz + ba) and the attention qkv_proj at load (the bf16 weights are freed); unset = the parent's path, untouched.
"""
from __future__ import annotations

import logging
import os
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def read_d5_knobs() -> int:
    v = os.environ.get("SPECPILOT_D5_W12", "").strip()
    return int(v) if v else 0


D5_W12 = read_d5_knobs()
# SPECPILOT_D5_PDL=1 (default 1 when W12 is on): launch W12 GEMMs with programmatic dependent launch; the first ring
# stages of weight planes are issued before griddepcontrol.wait (they overlap the predecessor's tail when it triggers
# early, as the parent's d2 early-trigger norms do); 0 = plain stream order.
D5_PDL = int(os.environ.get("SPECPILOT_D5_PDL", "1").strip() or 1)
# SPECPILOT_D4P_W12Q=1: 64 < M <= 128 rows in one W12 launch (w12q_kernel, bitwise equal to the two 64-row launches)
D4P_W12Q = int(os.environ.get("SPECPILOT_D4P_W12Q", "0").strip() or 0)
_RB = 128
_QCPS = 2  # = D5_QCPS of the kernel source
_MAX_M = 64
# 64 < M <= _CHUNK_M (short prefills): the W12 GEMM in 64-row chunks streams 1.5 B/weight per chunk and beats
# decode-to-scratch (2 + 1.5 B/weight) + cuBLAS up to 2 chunks (scratch/d5/chunk_t.py: M=128 293 vs 353 us)
_CHUNK_M = 128


def _wgmma_fn(nt):
    nreg = nt // 2
    outs = ",".join(f"%{i}" for i in range(nreg))
    a = ",".join(f"%{nreg + i}" for i in range(4))
    cons_d = ", ".join(f'"+f"(d[{i}])' for i in range(nreg))
    return f"""
__device__ __forceinline__ void wgmma_n{nt}(float* d, const uint32_t* a, uint64_t desc) {{
  asm volatile("{{\\n.reg .pred p;\\nsetp.ne.b32 p, %{nreg + 5}, 0;\\n"
               "wgmma.mma_async.sync.aligned.m64n{nt}k16.f32.bf16.bf16 {{{outs}}}, {{{a}}}, %{nreg + 4}, p, 1, 1, 0;\\n}}\\n"
               : {cons_d}
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(1));
}}
"""



_KERN = r"""
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

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
__device__ __forceinline__ void tma3d(uint32_t dst, const CUtensorMap* map, int c0, int c1, int c2, uint32_t mbar) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];\n"
               ::"r"(dst), "l"(map), "r"(c0), "r"(c1), "r"(c2), "r"(mbar) : "memory");
}
__device__ __forceinline__ void tma2d(uint32_t dst, const CUtensorMap* map, int c0, int c1, uint32_t mbar) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n"
               ::"r"(dst), "l"(map), "r"(c0), "r"(c1), "r"(mbar) : "memory");
}
// wgmma descriptor of a K-major 128B-swizzled bf16 tile (rows of 64 k = 128 B, 8-row groups 1024 B apart)
__device__ __forceinline__ uint64_t mkdesc_sw128(uint32_t saddr) {
  return (uint64_t)((saddr & 0x3FFFF) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)(1024 >> 4) << 32) | ((uint64_t)1 << 62);
}
__device__ __forceinline__ uint64_t mkdesc(uint32_t saddr, uint32_t lbo, uint32_t sbo) {
  return (uint64_t)((saddr & 0x3FFFF) >> 4) | ((uint64_t)((lbo >> 4) & 0x3FFF) << 16) |
         ((uint64_t)((sbo >> 4) & 0x3FFF) << 32);
}
__device__ __forceinline__ float ldg_early(const float* p) {
  float v;
  asm volatile("ld.global.nc.f32 %0, [%1];\n" : "=f"(v) : "l"(p));
  return v;
}
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int N> __device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory"); }
__device__ __forceinline__ void cons_bar(int n) { asm volatile("bar.sync 1, %0;\n" ::"r"(n) : "memory"); }

@@WGMMA@@

template <int NT> __device__ __forceinline__ void wgmma_nt(float* d, const uint32_t* a, uint64_t desc);
template <> __device__ __forceinline__ void wgmma_nt<16>(float* d, const uint32_t* a, uint64_t desc) { wgmma_n16(d, a, desc); }
template <> __device__ __forceinline__ void wgmma_nt<32>(float* d, const uint32_t* a, uint64_t desc) { wgmma_n32(d, a, desc); }
template <> __device__ __forceinline__ void wgmma_nt<48>(float* d, const uint32_t* a, uint64_t desc) { wgmma_n48(d, a, desc); }
template <> __device__ __forceinline__ void wgmma_nt<64>(float* d, const uint32_t* a, uint64_t desc) { wgmma_n64(d, a, desc); }
template <> __device__ __forceinline__ void wgmma_nt<128>(float* d, const uint32_t* a, uint64_t desc) { wgmma_n128(d, a, desc); }

// one warp-chunk (16 rows x 64 k) from its 3 uint4 -> A fragments of 4 k16 steps (bf16 bits [s 0111 c3c2c1 | c0 m]).
// Per nibble word (8 weights): clean nibbles n (mask), high bytes h = ((n * 17) & 0x87) | 0x38 (n * 17 = n | n << 4
// for n < 16: s -> bit 7, c3c2c1 stay in bits 2..0), two PRMTs pair the high bytes, one PRMT per A register merges
// them with the low-plane bytes: ~3.3 ops per register (was ~7, with selectors re-materialised per PRMT)
__device__ __forceinline__ void decode(const uint4 cL0, const uint4 cL1, const uint4 cN, uint32_t (&A)[4][4]) {
  const uint32_t S[8] = {cL0.x, cL0.y, cL0.z, cL0.w, cL1.x, cL1.y, cL1.z, cL1.w};
  const uint32_t NWd[4] = {cN.x, cN.y, cN.z, cN.w};
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const uint32_t nr = NWd[j];
    const uint32_t xl = (((nr & 0x0F0F0F0Fu) * 17u) & 0x87878787u) | 0x38383838u;          // slot-0 high bytes
    const uint32_t xh = ((((nr >> 4) & 0x0F0F0F0Fu) * 17u) & 0x87878787u) | 0x38383838u;   // slot-1 high bytes
    const uint32_t y01 = __byte_perm(xl, xh, 0x5140), y23 = __byte_perm(xl, xh, 0x7362);
    const uint32_t s0 = S[2 * j], s1 = S[2 * j + 1];
    A[j][0] = __byte_perm(s0, y01, 0x5140);
    A[j][1] = __byte_perm(s0, y01, 0x7362);
    A[j][2] = __byte_perm(s1, y23, 0x5140);
    A[j][3] = __byte_perm(s1, y23, 0x7362);
  }
}

// the same decode, mask/shift form (kept for NT >= 48: lower register pressure next to the larger accumulators)
__device__ __forceinline__ void decode_v1(const uint4 cL0, const uint4 cL1, const uint4 cN, uint32_t (&A)[4][4]) {
  const uint32_t S[8] = {cL0.x, cL0.y, cL0.z, cL0.w, cL1.x, cL1.y, cL1.z, cL1.w};
  const uint32_t NWd[4] = {cN.x, cN.y, cN.z, cN.w};
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const uint32_t nr = NWd[j], ns = nr >> 4;
#pragma unroll
    for (int p = 0; p < 4; ++p) {
      const uint32_t sw = S[2 * j + (p >> 1)];
      const uint32_t lo = __byte_perm(sw, 0x38383838u, (p & 1) ? 0x4342 : 0x4140);
      const uint32_t pp = __byte_perm(nr, ns, ((4 + p) << 12) | (p << 8) | (p << 4) | p);
      A[j][p] = (pp & 0x07000700u) | ((pp << 4) & 0x80008000u) | lo;
    }
  }
}

// NESTED PLANES (per warp-chunk of 16 rows x 64 k, per lane): Q words (2 uint4) = one byte per weight
// [c0 m6 m5 m4 | s c3 c2 c1] in the low-byte order of the A fragments, B words (1 uint4) = [m3..m0] nibbles
// (word j: low nibbles -> Q word 2j, high nibbles -> Q word 2j+1). Exact bf16: lo = (q & 0xF0) | b,
// hi = s << 7 | 0x38 | c3c2c1; Q-only (b = 0): the mantissa truncated to 3 bits (an e4m3-like value in the row window).
template <int QM>
__device__ __forceinline__ void decode_n(const uint4 cq0, const uint4 cq1, const uint4 cb, uint32_t (&A)[4][4]) {
  const uint32_t Q[8] = {cq0.x, cq0.y, cq0.z, cq0.w, cq1.x, cq1.y, cq1.z, cq1.w};
  const uint32_t B[4] = {cb.x, cb.y, cb.z, cb.w};
#pragma unroll
  for (int j = 0; j < 4; ++j) {
#pragma unroll
    for (int c8 = 0; c8 < 2; ++c8) {
      const uint32_t q = Q[2 * j + c8];
      uint32_t lo;
      if (QM) {
        lo = (q & 0xF0F0F0F0u) | 0x08080808u;  // midpoint of the dropped mantissa bits: [m6 m5 m4 1 0 0 0]
      } else {
        const uint32_t b = c8 ? (B[j] >> 4) & 0x0F0F0F0Fu : B[j] & 0x0F0F0F0Fu;
        lo = (q & 0xF0F0F0F0u) | b;
      }
      const uint32_t hi = ((((q << 4) & 0x80808080u) | (q & 0x07070707u))) | 0x38383838u;
      A[j][2 * c8] = __byte_perm(lo, hi, 0x5140);
      A[j][2 * c8 + 1] = __byte_perm(lo, hi, 0x7362);
    }
  }
}

template <int NWG, int NT, int CPS, int NS, int MINB, int QM>
__global__ void __launch_bounds__(NWG * 128 + 96, MINB) w12s_kernel(
    const __grid_constant__ CUtensorMap xmap, const __nv_bfloat16* __restrict__ x, int64_t ldx, int M,
    const unsigned char* __restrict__ planes, int64_t tstride, int S, int U,
    const float* __restrict__ scale, const int* __restrict__ esc_rp, const int* __restrict__ esc_col,
    const float* __restrict__ esc_val, __nv_bfloat16* __restrict__ out, int64_t ldo, int N,
    float* __restrict__ ws, int maxseg, int* __restrict__ counters, int silu, int nout, int esc_seg, int trig) {
  constexpr int NCW = NWG * 4, RB = NWG * 64;
  constexpr int WREC1 = NCW * 1536;                   // record bytes per 64-k chunk in global (Q 8 KB + B 4 KB)
  constexpr int WB1 = QM ? NCW * 1024 : WREC1;        // bytes streamed per chunk (Q-only: its first 8 KB)
  constexpr int WB = CPS * WB1;                       // bytes streamed per stage (CPS chunks)
  constexpr int XB = CPS * 8 * NT * 16;
  constexpr int SB = WB + XB;
  constexpr int NREG = NT / 2;
  constexpr int CB = RB * NT;  // floats per corr buffer
  extern __shared__ __align__(1024) unsigned char smem[];
  __shared__ __align__(8) uint64_t bars[2 * NS + 4];
  __shared__ int last_flag;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int P = gridDim.x, p = blockIdx.x;
  const int u0 = (int)(((int64_t)p * U) / P), u1 = (int)(((int64_t)(p + 1) * U) / P);
  // x stages are 128B-swizzled (TMA + wgmma): the ring starts at a 1024-B aligned offset (1 KB of slack allocated)
  const uint32_t sraw = (uint32_t)__cvta_generic_to_shared(smem);
  const uint32_t sbase = (sraw + 1023u) & ~1023u;
  unsigned char* const smem_al = smem + (sbase - sraw);
  float* corrbuf = reinterpret_cast<float*>(smem_al + NS * SB);
  float* xchg = corrbuf + 2 * CB;  // [64][NT] up values for the fused SiLU epilogue  // [64][NT] up values for the fused SiLU epilogue
  const uint32_t full0 = (uint32_t)__cvta_generic_to_shared(bars), empty0 = full0 + 8 * NS;
  const uint32_t cfull0 = full0 + 16 * NS, cempty0 = cfull0 + 16;
  if (threadIdx.x == 0) {
    for (int s = 0; s < NS; ++s) { mbar_init(full0 + 8 * s, 1); mbar_init(empty0 + 8 * s, NCW); }
    for (int b = 0; b < 2; ++b) { mbar_init(cfull0 + 8 * b, 64); mbar_init(cempty0 + 8 * b, NCW * 32); }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();
  // trig (gate_up and down kernels): let the dependent launch now. PDL dependents in the engine: gate_up -> the down
  // W12 GEMM (or the next 64-row chunk), down -> the next layer's x3 add+RMSNorm; both read this kernel's output,
  // the workspace and the counters only after griddepcontrol.wait / gdc_wait
  if (trig) asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
  if (u0 >= u1) return;
  if (warp == NCW) {  // producer
    if (lane == 0) {
      // sequential walk over (tile, k-stage) with the per-segment rotation of stage_of (no divisions per unit)
      int tile = 0, sbeg = 0, slen = 1, srot = 0;
      auto walk = [&](int u) -> int {
        if (u == u0 || u - tile * S == S) {
          tile = u / S;
          sbeg = u - tile * S;
          slen = min(u1, (tile + 1) * S) - u;
          srot = D5_W12_ROT ? (int)(((int64_t)tile * D5_W12_ROT + p) % slen) : 0;
        }
        int s = (u - tile * S) - sbeg + srot;
        if (s >= slen) s -= slen;
        return s + sbeg;
      };
      // PDL: the first NS weight stages do not depend on the predecessor kernel -> issued before griddepcontrol.wait
      const int npre = min(NS, u1 - u0);
      int spre[NS];
#pragma unroll
      for (int i = 0; i < NS; ++i) {
        if (i < npre) {
          const int s = walk(u0 + i);
          spre[i] = s;
          const uint32_t fb = full0 + 8 * i, dst = sbase + i * SB;
          mbar_expect_tx(fb, SB);
          if (QM) {
#pragma unroll
            for (int cc = 0; cc < CPS; ++cc)
              bulk_g2s(dst + cc * WB1, planes + (int64_t)tile * tstride + (int64_t)(s * CPS + cc) * WREC1, WB1, fb);
          } else {
            bulk_g2s(dst, planes + (int64_t)tile * tstride + (int64_t)s * CPS * WREC1, WB, fb);
          }
        }
      }
      asm volatile("griddepcontrol.wait;\n" ::: "memory");
#pragma unroll
      for (int i = 0; i < NS; ++i)
        if (i < npre)
#pragma unroll
          for (int cc = 0; cc < CPS; ++cc)
            tma2d(sbase + i * SB + WB + cc * NT * 128, &xmap, (spre[i] * CPS + cc) * 64, 0, full0 + 8 * i);
      for (int u = u0 + npre; u < u1; ++u) {
        const int i = u - u0, slot = i % NS;
        mbar_wait(empty0 + 8 * slot, ((i / NS) - 1) & 1);
        const int s = walk(u);
        const uint32_t fb = full0 + 8 * slot, dst = sbase + slot * SB;
        mbar_expect_tx(fb, SB);
        if (QM) {
#pragma unroll
          for (int cc = 0; cc < CPS; ++cc)
            bulk_g2s(dst + cc * WB1, planes + (int64_t)tile * tstride + (int64_t)(s * CPS + cc) * WREC1, WB1, fb);
        } else {
          bulk_g2s(dst, planes + (int64_t)tile * tstride + (int64_t)s * CPS * WREC1, WB, fb);
        }
#pragma unroll
        for (int cc = 0; cc < CPS; ++cc) tma2d(dst + WB + cc * NT * 128, &xmap, (s * CPS + cc) * 64, 0, fb);
      }
    }
    return;
  }
  if (warp > NCW) {  // 2 escape warps: exact correction of this segment's k-range (rows split between the warps)
    const int ew = warp - NCW - 1;
    asm volatile("griddepcontrol.wait;\n" ::: "memory");  // x is the predecessor's output
    int j = 0;
    for (int u = u0; u < u1;) {
      const int tile = u / S, s0 = u - tile * S;
      const int uend = min(u1, (tile + 1) * S), s1 = uend - tile * S;
      if (!esc_seg && s0 != 0) { u = uend; continue; }  // whole-tile mode: the k = 0 segment corrects all of k
      const int k0 = esc_seg ? s0 * 64 * CPS : 0, k1 = esc_seg ? s1 * 64 * CPS : S * 64 * CPS;
      const int b = j & 1;
      if (j >= 2) mbar_wait(cempty0 + 8 * b, ((j >> 1) - 1) & 1);
      float* cb = corrbuf + b * CB;
      for (int r = lane + 32 * ew; r < RB; r += 64) {
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
              if (m < M) {  // x * w~_true - x * 2^-15: two exact products
                const float xv = __bfloat162float(xc[(int64_t)m * ldx]);
                // Q pass: the placeholder decodes as 1.0625 * 2^-15 (midpoint mantissa); both products stay exact
                c[m] = fmaf(ev, xv, fmaf(QM ? -3.24249267578125e-05f : -3.0517578125e-05f, xv, c[m]));
              }
          }
        }
#pragma unroll
        for (int m = 0; m < NT; ++m) cb[r * NT + ((m + r) % NT)] = c[m];  // rotated: conflict-free
      }
      mbar_arrive(cfull0 + 8 * b);
      ++j;
      u = uend;
    }
    return;
  }
  // consumers
  const int g = lane / 4, t = lane % 4;
  float acc[NREG];
  uint32_t A0[CPS][4][4], A1[CPS][4][4];
  int pending = -1;
  auto stage = [&](int i, uint32_t (&A)[CPS][4][4]) {
    const int slot = i % NS;
    mbar_wait(full0 + 8 * slot, (i / NS) & 1);
    const uint4* sw = reinterpret_cast<const uint4*>(smem_al + slot * SB);
#pragma unroll
    for (int cc = 0; cc < CPS; ++cc) {
      const uint4* bq = sw + cc * (WB1 / 16) + warp * 64;
      const uint4* bb = sw + cc * (WB1 / 16) + NCW * 64 + warp * 32;
      decode_n<QM>(bq[lane], bq[32 + lane], QM ? bq[lane] : bb[lane], A[cc]);
    }
#pragma unroll
    for (int q = 0; q < NREG; ++q) asm volatile("" : "+f"(acc[q])::"memory");
    wg_fence();
    const uint32_t xs = sbase + slot * SB + WB;
#pragma unroll
    for (int cc = 0; cc < CPS; ++cc)
#pragma unroll
      for (int j = 0; j < 4; ++j)
        wgmma_nt<NT>(acc, A[cc][j], mkdesc_sw128(xs + cc * NT * 128 + j * 32));  // k16 step j: +32 B in the swizzled rows
    wg_commit();
#pragma unroll
    for (int q = 0; q < NREG; ++q) asm volatile("" : "+f"(acc[q])::"memory");
    wg_wait<1>();
    if (pending >= 0 && lane == 0) mbar_arrive(empty0 + 8 * pending);
    pending = slot;
  };
  int jc = 0;
  for (int u = u0; u < u1;) {
    const int tile = u / S, s0 = u - tile * S;
    const int uend = min(u1, (tile + 1) * S);
    const int n0 = tile * RB + warp * 16 + g;
    const float sc0 = n0 < N ? ldg_early(scale + n0) : 1.f, sc1 = n0 + 8 < N ? ldg_early(scale + n0 + 8) : 1.f;
#pragma unroll
    for (int q = 0; q < NREG; ++q) acc[q] = 0.f;
    for (int uu = u; uu < uend; ++uu) {
      const int i = uu - u0;
      if (i & 1) stage(i, A1); else stage(i, A0);
    }
    wg_wait<0>();
#pragma unroll
    for (int q = 0; q < NREG; ++q) asm volatile("" : "+f"(acc[q])::"memory");
    if (lane == 0) mbar_arrive(empty0 + 8 * pending);
    pending = -1;
    if (esc_seg || s0 == 0) {  // add this segment's (or, whole-tile mode, the tile's) exact escape correction
      const int b = jc & 1;
      mbar_wait(cfull0 + 8 * b, (jc >> 1) & 1);
      const float* cb = corrbuf + b * CB;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int r = warp * 16 + g + 8 * h;
#pragma unroll
        for (int q = 0; q < NT / 8; ++q)
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
      if (seg > 0) {  // contributor: publish the partial (release), keep streaming
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
          for (int q = 0; q < NT / 8; ++q)
            __stcg(reinterpret_cast<float2*>(wt + ((int64_t)seg * RB + rl + 8 * h) * NT + q * 8 + 2 * t),
                   make_float2(acc[q * 4 + 2 * h], acc[q * 4 + 2 * h + 1]));
        cons_bar(NCW * 32);
        if (threadIdx.x == 0) asm volatile("red.release.gpu.global.add.s32 [%0], 1;\n" ::"l"(counters + tile) : "memory");
      } else {  // reducer = owner of the tile's k = 0 segment (finishes last): seg 0 + seg 1..nseg-1 in order
        if (threadIdx.x == 0) {
          int c;
          do {
            asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\n" : "=r"(c) : "l"(counters + tile) : "memory");
          } while (c < nseg - 1);
          counters[tile] = 0;
        }
        cons_bar(NCW * 32);
        // all partial loads of up to 4 segments in flight at once, then summed in segment order (same fp result as
        // the serial loop; the serial form exposed one L2/DRAM round trip per (h, q, seg) at the CTA's tail)
        constexpr int PB = NT <= 32 ? 4 : (NT <= 48 ? 2 : 1);  // partials in flight (register budget)
        for (int s2 = 1; s2 < nseg; s2 += PB) {
          float2 v[PB][2][NT / 8];
#pragma unroll
          for (int j = 0; j < PB; ++j)
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
              for (int q = 0; q < NT / 8; ++q)
                v[j][h][q] = s2 + j < nseg ? __ldcg(reinterpret_cast<const float2*>(
                                                 wt + ((int64_t)(s2 + j) * RB + rl + 8 * h) * NT + q * 8 + 2 * t))
                                           : make_float2(0.f, 0.f);
#pragma unroll
          for (int j = 0; j < PB; ++j)
            if (s2 + j < nseg) {
#pragma unroll
              for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int q = 0; q < NT / 8; ++q) {
                  acc[q * 4 + 2 * h] += v[j][h][q].x;
                  acc[q * 4 + 2 * h + 1] += v[j][h][q].y;
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
          for (int q = 0; q < NT / 8; ++q)
#pragma unroll
            for (int v = 0; v < 2; ++v) {
              const int m = q * 8 + 2 * t + v;
              if (m < M && n < N) out[(int64_t)m * ldo + n] = __float2bfloat16(acc[q * 4 + 2 * h + v] * sc);
            }
        }
      } else {  // fused SiLU(gate) * up: tile = [64 gate rows | 64 up rows]; bf16-rounded operands as act_and_mul
        const int wr = (warp & 3) * 16 + g;  // row within the warpgroup
        if (warp >= 4) {
#pragma unroll
          for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int q = 0; q < NT / 8; ++q)
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
            for (int q = 0; q < NT / 8; ++q)
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

// SPECPILOT_D4P_W12Q: w12s_kernel for 64 < M <= 128 prefill rows in one launch (NT = 128): each decoded weight stage
// feeds one 128-wide wgmma instead of two 64-row launches streaming the weights twice. Every output element sees the
// same operation sequence as in the incumbent's two launches (same P, segments, rotation, per-element escape
// correction and reduction order, epilogue), so the outputs are bitwise equal; one correction buffer (smem budget).
template <int NWG, int NT, int CPS, int NS, int MINB>
__global__ void __launch_bounds__(NWG * 128 + 96, MINB) w12q_kernel(
    const __grid_constant__ CUtensorMap xmap, const __nv_bfloat16* __restrict__ x, int64_t ldx, int M,
    const unsigned char* __restrict__ planes, int64_t tstride, int S, int U,
    const float* __restrict__ scale, const int* __restrict__ esc_rp, const int* __restrict__ esc_col,
    const float* __restrict__ esc_val, __nv_bfloat16* __restrict__ out, int64_t ldo, int N,
    float* __restrict__ ws, int maxseg, int* __restrict__ counters, int silu, int nout, int esc_seg, int trig) {
  constexpr int NCW = NWG * 4, RB = NWG * 64;
  constexpr int WB = CPS * NCW * 1536;
  constexpr int XB = CPS * 8 * NT * 16;
  constexpr int SB = WB + XB;
  constexpr int NREG = NT / 2;
  constexpr int CB = RB * NT;  // floats per corr buffer
  extern __shared__ __align__(1024) unsigned char smem[];
  __shared__ __align__(8) uint64_t bars[2 * NS + 4];
  __shared__ int last_flag;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int P = gridDim.x, p = blockIdx.x;
  const int u0 = (int)(((int64_t)p * U) / P), u1 = (int)(((int64_t)(p + 1) * U) / P);
  // x stages are 128B-swizzled (TMA + wgmma): the ring starts at a 1024-B aligned offset (1 KB of slack allocated)
  const uint32_t sraw = (uint32_t)__cvta_generic_to_shared(smem);
  const uint32_t sbase = (sraw + 1023u) & ~1023u;
  unsigned char* const smem_al = smem + (sbase - sraw);
  float* corrbuf = reinterpret_cast<float*>(smem_al + NS * SB);
  float* xchg = corrbuf + CB;  // single correction buffer (smem budget at NT = 128)
  const uint32_t full0 = (uint32_t)__cvta_generic_to_shared(bars), empty0 = full0 + 8 * NS;
  const uint32_t cfull0 = full0 + 16 * NS, cempty0 = cfull0 + 16;
  if (threadIdx.x == 0) {
    for (int s = 0; s < NS; ++s) { mbar_init(full0 + 8 * s, 1); mbar_init(empty0 + 8 * s, NCW); }
    for (int b = 0; b < 2; ++b) { mbar_init(cfull0 + 8 * b, 64); mbar_init(cempty0 + 8 * b, NCW * 32); }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();
  // trig (gate_up and down kernels): let the dependent launch now. PDL dependents in the engine: gate_up -> the down
  // W12 GEMM (or the next 64-row chunk), down -> the next layer's x3 add+RMSNorm; both read this kernel's output,
  // the workspace and the counters only after griddepcontrol.wait / gdc_wait
  if (trig) asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
  if (u0 >= u1) return;
  if (warp == NCW) {  // producer
    if (lane == 0) {
      // sequential walk over (tile, k-stage) with the per-segment rotation of stage_of (no divisions per unit)
      int tile = 0, sbeg = 0, slen = 1, srot = 0;
      auto walk = [&](int u) -> int {
        if (u == u0 || u - tile * S == S) {
          tile = u / S;
          sbeg = u - tile * S;
          slen = min(u1, (tile + 1) * S) - u;
          srot = D5_W12_ROT ? (int)(((int64_t)tile * D5_W12_ROT + p) % slen) : 0;
        }
        int s = (u - tile * S) - sbeg + srot;
        if (s >= slen) s -= slen;
        return s + sbeg;
      };
      // PDL: the first NS weight stages do not depend on the predecessor kernel -> issued before griddepcontrol.wait
      const int npre = min(NS, u1 - u0);
      int spre[NS];
#pragma unroll
      for (int i = 0; i < NS; ++i) {
        if (i < npre) {
          const int s = walk(u0 + i);
          spre[i] = s;
          const uint32_t fb = full0 + 8 * i, dst = sbase + i * SB;
          mbar_expect_tx(fb, SB);
          bulk_g2s(dst, planes + (int64_t)tile * tstride + (int64_t)s * WB, WB, fb);
        }
      }
      asm volatile("griddepcontrol.wait;\n" ::: "memory");
#pragma unroll
      for (int i = 0; i < NS; ++i)
        if (i < npre) tma2d(sbase + i * SB + WB, &xmap, spre[i] * 64, 0, full0 + 8 * i);
      for (int u = u0 + npre; u < u1; ++u) {
        const int i = u - u0, slot = i % NS;
        mbar_wait(empty0 + 8 * slot, ((i / NS) - 1) & 1);
        const int s = walk(u);
        const uint32_t fb = full0 + 8 * slot, dst = sbase + slot * SB;
        mbar_expect_tx(fb, SB);
        bulk_g2s(dst, planes + (int64_t)tile * tstride + (int64_t)s * WB, WB, fb);
        tma2d(dst + WB, &xmap, s * 64, 0, fb);
      }
    }
    return;
  }
  if (warp > NCW) {  // 2 escape warps: exact correction of this segment's k-range (rows split between the warps)
    const int ew = warp - NCW - 1;
    asm volatile("griddepcontrol.wait;\n" ::: "memory");  // x is the predecessor's output
    int j = 0;
    for (int u = u0; u < u1;) {
      const int tile = u / S, s0 = u - tile * S;
      const int uend = min(u1, (tile + 1) * S), s1 = uend - tile * S;
      if (!esc_seg && s0 != 0) { u = uend; continue; }  // whole-tile mode: the k = 0 segment corrects all of k
      const int k0 = esc_seg ? s0 * 64 * CPS : 0, k1 = esc_seg ? s1 * 64 * CPS : S * 64 * CPS;
      const int b = 0;
      if (j >= 1) mbar_wait(cempty0, (j - 1) & 1);
      float* cb = corrbuf + b * CB;
      for (int r = lane + 32 * ew; r < RB; r += 64) {
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
              if (m < M) {  // x * w~_true - x * 2^-15: two exact products
                const float xv = __bfloat162float(xc[(int64_t)m * ldx]);
                c[m] = fmaf(ev, xv, fmaf(-3.0517578125e-05f, xv, c[m]));
              }
          }
        }
#pragma unroll
        for (int m = 0; m < NT; ++m) cb[r * NT + ((m + r) % NT)] = c[m];  // rotated: conflict-free
      }
      mbar_arrive(cfull0 + 8 * b);
      ++j;
      u = uend;
    }
    return;
  }
  // consumers
  const int g = lane / 4, t = lane % 4;
  float acc[NREG];
  uint32_t A0[CPS][4][4], A1[CPS][4][4];
  int pending = -1;
  auto stage = [&](int i, uint32_t (&A)[CPS][4][4]) {
    const int slot = i % NS;
    mbar_wait(full0 + 8 * slot, (i / NS) & 1);
    const uint4* sw = reinterpret_cast<const uint4*>(smem_al + slot * SB);
#pragma unroll
    for (int cc = 0; cc < CPS; ++cc) {
      // nested-plane record (d5-019): Q region [8 warps][1 KB] | B region [8 warps][512 B]; exact decode
      const uint4* bq = sw + cc * (WB / CPS / 16) + warp * 64;
      const uint4* bb = sw + cc * (WB / CPS / 16) + NCW * 64 + warp * 32;
      decode_n<0>(bq[lane], bq[32 + lane], bb[lane], A[cc]);
    }
#pragma unroll
    for (int q = 0; q < NREG; ++q) asm volatile("" : "+f"(acc[q])::"memory");
    wg_fence();
    const uint32_t xs = sbase + slot * SB + WB;
#pragma unroll
    for (int cc = 0; cc < CPS; ++cc)
#pragma unroll
      for (int j = 0; j < 4; ++j)
        wgmma_nt<NT>(acc, A[cc][j], mkdesc_sw128(xs + j * 32));  // k16 step j: +32 B inside the swizzled rows
    wg_commit();
#pragma unroll
    for (int q = 0; q < NREG; ++q) asm volatile("" : "+f"(acc[q])::"memory");
    wg_wait<1>();
    if (pending >= 0 && lane == 0) mbar_arrive(empty0 + 8 * pending);
    pending = slot;
  };
  int jc = 0;
  for (int u = u0; u < u1;) {
    const int tile = u / S, s0 = u - tile * S;
    const int uend = min(u1, (tile + 1) * S);
    const int n0 = tile * RB + warp * 16 + g;
    const float sc0 = n0 < N ? ldg_early(scale + n0) : 1.f, sc1 = n0 + 8 < N ? ldg_early(scale + n0 + 8) : 1.f;
#pragma unroll
    for (int q = 0; q < NREG; ++q) acc[q] = 0.f;
    for (int uu = u; uu < uend; ++uu) {
      const int i = uu - u0;
      if (i & 1) stage(i, A1); else stage(i, A0);
    }
    wg_wait<0>();
#pragma unroll
    for (int q = 0; q < NREG; ++q) asm volatile("" : "+f"(acc[q])::"memory");
    if (lane == 0) mbar_arrive(empty0 + 8 * pending);
    pending = -1;
    if (esc_seg || s0 == 0) {  // add this segment's (or, whole-tile mode, the tile's) exact escape correction
      const int b = 0;
      mbar_wait(cfull0, jc & 1);
      const float* cb = corrbuf + b * CB;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int r = warp * 16 + g + 8 * h;
#pragma unroll
        for (int q = 0; q < NT / 8; ++q)
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
      if (seg > 0) {  // contributor: publish the partial (release), keep streaming
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
          for (int q = 0; q < NT / 8; ++q)
            __stcg(reinterpret_cast<float2*>(wt + ((int64_t)seg * RB + rl + 8 * h) * NT + q * 8 + 2 * t),
                   make_float2(acc[q * 4 + 2 * h], acc[q * 4 + 2 * h + 1]));
        cons_bar(NCW * 32);
        if (threadIdx.x == 0) asm volatile("red.release.gpu.global.add.s32 [%0], 1;\n" ::"l"(counters + tile) : "memory");
      } else {  // reducer = owner of the tile's k = 0 segment (finishes last): seg 0 + seg 1..nseg-1 in order
        if (threadIdx.x == 0) {
          int c;
          do {
            asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\n" : "=r"(c) : "l"(counters + tile) : "memory");
          } while (c < nseg - 1);
          counters[tile] = 0;
        }
        cons_bar(NCW * 32);
        // all partial loads of up to 4 segments in flight at once, then summed in segment order (same fp result as
        // the serial loop; the serial form exposed one L2/DRAM round trip per (h, q, seg) at the CTA's tail)
        constexpr int PB = NT <= 32 ? 4 : (NT <= 48 ? 2 : 1);  // partials in flight (register budget)
        for (int s2 = 1; s2 < nseg; s2 += PB) {
          float2 v[PB][2][NT / 8];
#pragma unroll
          for (int j = 0; j < PB; ++j)
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
              for (int q = 0; q < NT / 8; ++q)
                v[j][h][q] = s2 + j < nseg ? __ldcg(reinterpret_cast<const float2*>(
                                                 wt + ((int64_t)(s2 + j) * RB + rl + 8 * h) * NT + q * 8 + 2 * t))
                                           : make_float2(0.f, 0.f);
#pragma unroll
          for (int j = 0; j < PB; ++j)
            if (s2 + j < nseg) {
#pragma unroll
              for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int q = 0; q < NT / 8; ++q) {
                  acc[q * 4 + 2 * h] += v[j][h][q].x;
                  acc[q * 4 + 2 * h + 1] += v[j][h][q].y;
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
          for (int q = 0; q < NT / 8; ++q)
#pragma unroll
            for (int v = 0; v < 2; ++v) {
              const int m = q * 8 + 2 * t + v;
              if (m < M && n < N) out[(int64_t)m * ldo + n] = __float2bfloat16(acc[q * 4 + 2 * h + v] * sc);
            }
        }
      } else {  // fused SiLU(gate) * up: tile = [64 gate rows | 64 up rows]; bf16-rounded operands as act_and_mul
        const int wr = (warp & 3) * 16 + g;  // row within the warpgroup
        if (warp >= 4) {
#pragma unroll
          for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int q = 0; q < NT / 8; ++q)
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
            for (int q = 0; q < NT / 8; ++q)
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

"""

_HOST = r"""
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

template <int NT, int QM>
void launch(torch::Tensor x, torch::Tensor planes, torch::Tensor scale, torch::Tensor rp, torch::Tensor col,
            torch::Tensor val, torch::Tensor out, torch::Tensor ws, torch::Tensor cnt, int64_t P, int64_t maxseg,
            int64_t N, int64_t silu, int64_t esc_seg, int64_t pdl, int64_t trig) {
  constexpr int NWG = 2, NS = (QM && NT == 64) ? 4 : 5;  // ring depth 5: measured best (Q at NT=64: 4, smem budget)
  constexpr int CPS = QM ? D5_QCPS : 1;  // Q pass: CPS 64-k chunks per stage (1 B/w: keep the stage at >= 8 KB x CPS)
  const int M = x.size(0), K = x.size(1), nout = out.size(1);
  TORCH_CHECK(M <= NT && M > 0, "W12: bad M");
  const int T = planes.size(0), S = K / 64 / CPS, U = T * S;
  CUtensorMap map;
  cuuint64_t gdim[2] = {(cuuint64_t)K, (cuuint64_t)M};
  cuuint64_t gstr[1] = {(cuuint64_t)x.stride(0) * 2};
  cuuint32_t box[2] = {64, NT};  // one x stage: NT rows x 64 k (128-B rows, OOB rows zero-filled)
  cuuint32_t es[2] = {1, 1};
  CUresult r = get_encode()(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, x.data_ptr(), gdim, gstr, box, es,
                            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "W12 tensor map encode failed ", (int)r);
  constexpr int RB = NWG * 64;
  constexpr int SB = CPS * NWG * 4 * (QM ? 1024 : 1536) + CPS * 8 * NT * 16;
  const int smem = NS * SB + 2 * RB * NT * 4 + 64 * NT * 4 + 1024;
  auto kern = w12s_kernel<NWG, NT, CPS, NS, 1, QM>;
  TORCH_CHECK(maxseg > 0 && K % (64 * CPS) == 0, "W12: bad K for this pass");
  static bool attr = false;
  if (!attr) { cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); attr = true; }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((unsigned)P);
  cfg.blockDim = dim3(NWG * 128 + 96);
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
      (int)N, ws.data_ptr<float>(), (int)maxseg, cnt.data_ptr<int>(), (int)silu, nout, (int)esc_seg, (int)trig);
  TORCH_CHECK(e == cudaSuccess, "W12 launch failed: ", cudaGetErrorString(e));
}

template <int NT>
void launch_q(torch::Tensor x, torch::Tensor planes, torch::Tensor scale, torch::Tensor rp, torch::Tensor col,
            torch::Tensor val, torch::Tensor out, torch::Tensor ws, torch::Tensor cnt, int64_t P, int64_t maxseg,
            int64_t N, int64_t silu, int64_t esc_seg, int64_t pdl, int64_t trig) {
  constexpr int NWG = 2, CPS = 1, NS = 4;  // fits 227 KB with one 64 KB correction buffer at NT = 128
  const int M = x.size(0), K = x.size(1), nout = out.size(1);
  TORCH_CHECK(M <= NT && M > 0, "W12: bad M");
  const int T = planes.size(0), S = K / 64 / CPS, U = T * S;
  CUtensorMap map;
  static_assert(CPS == 1, "128B-swizzled x stages hold one 64-k chunk");
  cuuint64_t gdim[2] = {(cuuint64_t)K, (cuuint64_t)M};
  cuuint64_t gstr[1] = {(cuuint64_t)x.stride(0) * 2};
  cuuint32_t box[2] = {64, NT};  // one x stage: NT rows x 64 k (128-B rows, OOB rows zero-filled)
  cuuint32_t es[2] = {1, 1};
  CUresult r = get_encode()(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, x.data_ptr(), gdim, gstr, box, es,
                            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "W12 tensor map encode failed ", (int)r);
  constexpr int RB = NWG * 64;
  constexpr int SB = CPS * NWG * 4 * 1536 + CPS * 8 * NT * 16;
  const int smem = NS * SB + RB * NT * 4 + 64 * NT * 4 + 1024;
  auto kern = w12q_kernel<NWG, NT, CPS, NS, 1>;
  static bool attr = false;
  if (!attr) { cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); attr = true; }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((unsigned)P);
  cfg.blockDim = dim3(NWG * 128 + 96);
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
      (int)N, ws.data_ptr<float>(), (int)maxseg, cnt.data_ptr<int>(), (int)silu, nout, (int)esc_seg, (int)trig);
  TORCH_CHECK(e == cudaSuccess, "W12Q launch failed: ", cudaGetErrorString(e));
}

void w12_gemm_q(torch::Tensor x, torch::Tensor planes, torch::Tensor scale, torch::Tensor rp, torch::Tensor col,
                torch::Tensor val, torch::Tensor out, torch::Tensor ws, torch::Tensor cnt, int64_t P, int64_t maxseg,
                int64_t N, int64_t silu, int64_t esc_seg, int64_t pdl, int64_t trig) {
  launch_q<128>(x, planes, scale, rp, col, val, out, ws, cnt, P, maxseg, N, silu, esc_seg, pdl, trig);
}

void w12_gemm(torch::Tensor x, torch::Tensor planes, torch::Tensor scale, torch::Tensor rp, torch::Tensor col,
              torch::Tensor val, torch::Tensor out, torch::Tensor ws, torch::Tensor cnt, int64_t P, int64_t maxseg,
              int64_t N, int64_t silu, int64_t nt, int64_t esc_seg, int64_t pdl, int64_t trig, int64_t qm) {
  switch (nt) {
    case 16: if (qm) launch<16, 1>(x, planes, scale, rp, col, val, out, ws, cnt, P, maxseg, N, silu, esc_seg, pdl, trig);
             else launch<16, 0>(x, planes, scale, rp, col, val, out, ws, cnt, P, maxseg, N, silu, esc_seg, pdl, trig); break;
    case 32: if (qm) launch<32, 1>(x, planes, scale, rp, col, val, out, ws, cnt, P, maxseg, N, silu, esc_seg, pdl, trig);
             else launch<32, 0>(x, planes, scale, rp, col, val, out, ws, cnt, P, maxseg, N, silu, esc_seg, pdl, trig); break;
    case 48: if (qm) launch<48, 1>(x, planes, scale, rp, col, val, out, ws, cnt, P, maxseg, N, silu, esc_seg, pdl, trig);
             else launch<48, 0>(x, planes, scale, rp, col, val, out, ws, cnt, P, maxseg, N, silu, esc_seg, pdl, trig); break;
    case 64: if (qm) launch<64, 1>(x, planes, scale, rp, col, val, out, ws, cnt, P, maxseg, N, silu, esc_seg, pdl, trig);
             else launch<64, 0>(x, planes, scale, rp, col, val, out, ws, cnt, P, maxseg, N, silu, esc_seg, pdl, trig); break;
    default: TORCH_CHECK(false, "W12: bad nt");
  }
}

// planes -> bf16 weight in the original row order (out-of-window entries are fixed up by the caller's scatter)
__global__ void __launch_bounds__(256) w12_decode_kernel(const uint4* __restrict__ planes, int64_t tstride16, int nkc,
                                  const float* __restrict__ scale, __nv_bfloat16* __restrict__ out, int N, int K, int silu,
                                  int64_t total) {
  // one block = one (tile, 64-k chunk): decode the 128 x 64 tile into smem, then coalesced 16-B row stores
  __shared__ uint32_t tl[128][33];
  const int64_t id = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  const int64_t rest = id >> 8;
  const int kc = rest % nkc, tile = rest / nkc;
  const uint4* blk = planes + tile * tstride16 + (int64_t)kc * 768;  // stage record: Q 512 uint4 | B 256 uint4
  uint32_t A[4][4];
  decode_n<0>(blk[w * 64 + lane], blk[w * 64 + 32 + lane], blk[512 + w * 32 + lane], A);
  const int g = lane / 4, t = lane % 4;
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const int rl = w * 16 + g + 8 * h;
    const uint32_t E = (__float_as_uint(scale[tile * 128 + rl]) >> 23) & 0xFF;
    const uint32_t add = ((E - 127u) << 7) & 0xFFFFu;  // exponent shift mod 2^16, added per halfword
    const uint32_t add2 = add | (add << 16);
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
      for (int c8 = 0; c8 < 2; ++c8)  // per-halfword: exponent c + E - 15 stays in [0, 255]
        tl[rl][j * 8 + c8 * 4 + t] = __vadd2(A[j][2 * c8 + h], add2);
  }
  __syncthreads();
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    const int idx = threadIdx.x + 256 * q, rl = idx >> 3, c16 = idx & 7;
    const int r = tile * 128 + rl;
    int orig = r;
    if (silu) {
      const int I = N / 2;
      orig = rl < 64 ? tile * 64 + rl : I + tile * 64 + rl - 64;
    }
    if (orig >= N) continue;
    const uint4 v = make_uint4(tl[rl][c16 * 4], tl[rl][c16 * 4 + 1], tl[rl][c16 * 4 + 2], tl[rl][c16 * 4 + 3]);
    *reinterpret_cast<uint4*>(out + (int64_t)orig * K + kc * 64 + c16 * 8) = v;
  }
  (void)total;
}

void w12_decode(torch::Tensor planes, torch::Tensor scale, torch::Tensor out, int64_t N, int64_t K, int64_t silu) {
  const int T = planes.size(0), nkc = K / 64;
  const int64_t total = (int64_t)T * nkc * 256;
  w12_decode_kernel<<<(unsigned)((total + 255) / 256), 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const uint4*>(planes.data_ptr()), planes.stride(0) / 16, nkc, scale.data_ptr<float>(),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), (int)N, (int)K, (int)silu, total);
}
"""


_SRC = r"""
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#define D5_W12_ROT 37
#ifndef D5_QCPS
#define D5_QCPS 2
#endif
""" + _KERN + _HOST

_CPP = (
    "void w12_gemm(torch::Tensor x, torch::Tensor planes, torch::Tensor scale, torch::Tensor rp, torch::Tensor col, "
    "torch::Tensor val, torch::Tensor out, torch::Tensor ws, torch::Tensor cnt, int64_t P, int64_t maxseg, int64_t N, "
    "int64_t silu, int64_t nt, int64_t esc_seg, int64_t pdl, int64_t trig, int64_t qm);\n"
    "void w12_decode(torch::Tensor planes, torch::Tensor scale, torch::Tensor out, int64_t N, int64_t K, int64_t silu);\n"
    "void w12_gemm_q(torch::Tensor x, torch::Tensor planes, torch::Tensor scale, torch::Tensor rp, torch::Tensor col, "
    "torch::Tensor val, torch::Tensor out, torch::Tensor ws, torch::Tensor cnt, int64_t P, int64_t maxseg, int64_t N, "
    "int64_t silu, int64_t esc_seg, int64_t pdl, int64_t trig);"
)

_mod = None


def _ext():
    global _mod
    if _mod is None:
        from torch.utils.cpp_extension import load_inline

        import hashlib

        base = os.environ.get("SGLANG_CACHE_DIR") or os.path.join(os.path.expanduser("~"), ".cache", "sglang")
        fns = "".join(_wgmma_fn(nt) for nt in (16, 32, 48, 64, 128))
        src = _SRC.replace("@@WGMMA@@", fns)
        cflags = ["-O3", "-gencode=arch=compute_90a,code=sm_90a", "--use_fast_math"]
        # one build directory per source: W12 versions sharing a directory overwrite each other's cuda.cu /
        # build.ninja and rebuild a .so other processes have mapped (lazy-loaded kernel -> invalid resource handle)
        key = hashlib.sha1("\0".join([_CPP, src, *cflags]).encode()).hexdigest()[:16]
        bdir = os.path.join(base, "specpilot_int012_w12_src", key)  # private, source-keyed
        os.makedirs(bdir, exist_ok=True)
        _mod = load_inline(
            name="specpilot_int012_w12_v27q", cpp_sources=_CPP, cuda_sources=src,
            functions=["w12_gemm", "w12_decode", "w12_gemm_q"],
            extra_cuda_cflags=cflags, build_directory=bdir, verbose=False)
    return _mod


# ----------------------------------------------------------------------------------------------------------------- encode


def _encode_rows(wc: torch.Tensor):
    """wc: [R, K] bf16, R % 128 == 0 -> (blocks [R/128, K/64, 8, 1536] uint8, E [R] int32, esc (rows, cols) local)."""
    R, K = wc.shape
    u = wc.view(torch.int16).to(torch.int32) & 0xFFFF
    e = (u >> 7) & 0xFF
    # per-row 16-binade window [E-15, E] that covers the most weights (ties -> the highest window): outlier rows
    # (a few huge weights) keep the bulk in the window and send the outliers to the exact sparse correction
    hist = torch.zeros(R, 257, dtype=torch.int32, device=wc.device)
    hist.scatter_add_(1, e.to(torch.int64), torch.ones_like(e))
    hist[:, 0] = 0  # zeros / denormals always escape
    cum = torch.cumsum(hist, dim=1)  # cum[:, t] = #(1 <= e <= t)
    tops = torch.arange(15, 256, device=wc.device)
    cover = cum[:, tops] - cum[:, tops - 16].masked_fill((tops - 16 < 0)[None, :], 0)
    best = cover.max(dim=1, keepdim=True).values
    E = tops[(cover == best).int().cumsum(1).argmax(dim=1)].to(torch.int32)  # last (highest) maximiser
    c = e - (E[:, None] - 15)
    esc = (c < 0) | (c > 15) | (e == 0)
    c = torch.where(esc, torch.zeros_like(c), c)
    s = torch.where(esc, torch.zeros_like(u), u >> 15)
    m = torch.where(esc, torch.zeros_like(u), u & 0x7F)
    low = (((c & 1) << 7) | m).to(torch.uint8)
    nb = ((s << 3) | (c >> 1)).to(torch.uint8)
    del u, e, c, s, m
    # row = rg*16 + h*8 + g, col = kc*64 + j*16 + c8*8 + t*2 + b  ->  [rg, kc, lane=(g,t), w=(j, c8, h, b)]
    def frag(p):
        return (p.view(R // 16, 2, 8, K // 64, 4, 2, 4, 2).permute(0, 3, 2, 6, 4, 5, 1, 7).reshape(R // 16, K // 64, 32, 32))
    # nested planes: Q byte = [c0 m6 m5 m4 | s c3 c2 c1] (one byte per weight, read alone by the Q pass), B = [m3..m0]
    qf = frag((low & 0xF0) | nb)
    qb = qf.view(R // 16, K // 64, 32, 2, 16).transpose(2, 3).reshape(R // 16, K // 64, 1024)
    bf = frag(low & 0x0F).view(R // 16, K // 64, 32, 4, 2, 4)  # [.., lane, j, c8, (h, b)]
    bb = (bf[..., 0, :] | (bf[..., 1, :] << 4)).reshape(R // 16, K // 64, 512)
    T_ = R // _RB
    qb = qb.view(T_, _RB // 16, K // 64, 1024).transpose(1, 2).reshape(T_, K // 64, 8 * 1024)
    bb = bb.view(T_, _RB // 16, K // 64, 512).transpose(1, 2).reshape(T_, K // 64, 8 * 512)
    blocks = torch.cat([qb, bb], dim=2).view(T_, K // 64, 8, 1536).contiguous()  # per stage: [Q 8 x 1 KB | B 8 x 512 B]
    rows, cols = esc.nonzero(as_tuple=True)
    return blocks, E, rows, cols


class W12Weight:
    """Bit-exact W12 encoding of a [N, K] bf16 weight (optionally gate_up interleaved for the fused SiLU)."""

    def __init__(self, w: torch.Tensor, silu: bool = False, chunk_rows: int = 2048):
        assert w.dtype == torch.bfloat16 and w.dim() == 2 and w.is_cuda
        N, K = w.shape
        assert K % 64 == 0, "W12 needs K % 64 == 0"
        self.N, self.K, self.silu = N, K, int(silu)
        if silu:
            assert N % 128 == 0
            I = N // 2
            wi = w.view(2, I // 64, 64, K).transpose(0, 1).reshape(N, K)
            self.nout = I
        else:
            wi = w
            self.nout = N
        T = (N + _RB - 1) // _RB
        self.T = T
        dev = w.device
        self.planes = torch.empty((T, (K // 64) * (_RB // 16) * 1536), dtype=torch.uint8, device=dev)
        E_all = torch.empty(T * _RB, dtype=torch.int32, device=dev)
        rows_l, cols_l = [], []
        for r0 in range(0, T * _RB, chunk_rows):
            r1 = min(T * _RB, r0 + chunk_rows)
            wc = wi[r0:min(r1, N)]
            if wc.shape[0] < r1 - r0:  # pad rows (never stored): ones, no escapes
                wc = torch.cat([wc, torch.ones(r1 - r0 - wc.shape[0], K, dtype=w.dtype, device=dev)])
            blocks, E, rows, cols = _encode_rows(wc.contiguous())
            self.planes[r0 // _RB: r1 // _RB] = blocks.view(blocks.shape[0], -1)
            E_all[r0:r1] = E
            rows_l.append(rows + r0)
            cols_l.append(cols)
            del blocks, wc
        rows = torch.cat(rows_l)
        cols = torch.cat(cols_l)
        self.scale = torch.pow(2.0, (E_all - 127).double()).float()
        wt = wi[rows, cols].double() * torch.pow(2.0, (127 - E_all[rows]).double())
        d = wt.float()  # the kernel adds x * d - x * 2^-15 (decoded escapes read as 2^-15)
        # exact unless w~_true underflows fp32 (|w| < 2^(E_n-276): far below fp32 accumulation resolution)
        self.n_inexact = int((d.double() != wt).sum().item()) if d.numel() else 0
        self.exact_corr = self.n_inexact == 0
        counts = torch.bincount(rows, minlength=T * _RB)
        self.rp = torch.zeros(T * _RB + 1, dtype=torch.int32, device=dev)
        self.rp[1:] = torch.cumsum(counts, 0).to(torch.int32)
        order = torch.argsort(rows * K + cols)
        self.col = cols[order].to(torch.int32).contiguous()
        self.val = d[order].contiguous()
        rs = rows[order]
        # Q pass: only the escapes ABOVE the row window (|w~| >= 2: the large outliers) are corrected; the ones below it
        # (zeros, tiny values) keep the Q placeholder 1.0625 * 2^-15 (negligible for the Q argmax)
        big = self.val.abs() >= 2.0
        self.col_q = self.col[big].contiguous()
        self.val_q = self.val[big].contiguous()
        self.rp_q = torch.zeros(T * _RB + 1, dtype=torch.int32, device=dev)
        self.rp_q[1:] = torch.cumsum(torch.bincount(rs[big], minlength=T * _RB), 0).to(torch.int32)
        self.n_esc_q = int(big.sum().item())
        orig = rs
        if silu:
            t64, wr = rs // 128, rs % 128
            orig = torch.where(wr < 64, t64 * 64 + wr, N // 2 + t64 * 64 + wr - 64)
        self.esc_flat = (orig * K + cols[order]).contiguous()
        self.esc_bf16 = w.view(-1)[self.esc_flat].contiguous()
        self.n_esc = int(rows.numel())
        self.max_row_esc = int(counts.max().item()) if self.n_esc else 0
        nsm = torch.cuda.get_device_properties(dev).multi_processor_count
        S = K // 64
        self.P = T * max(1, nsm // T) if T <= nsm else nsm
        self.P = min(self.P, T * S)
        self.maxseg = _maxseg(T, S, self.P)
        self.maxseg_q = _maxseg(T, S // _QCPS, self.P) if S % _QCPS == 0 else 0  # Q pass: 64*_QCPS-channel stages
        # escape correction per segment k-range (split tiles share the work) except for gate_up (few escapes per
        # tile, short tail segments): there the k = 0 segment corrects the whole tile (measured scratch/d5 cmp_ws)
        self.esc_seg = 0 if silu else 1
        # early PDL trigger at kernel start (only where every PDL dependent waits before reading: gate_up; the
        # dense-MLP down projection is enabled by d5_encode_target)
        self.trigger = int(silu)
        _ensure_shared(dev, T * self.maxseg * _RB * (2 * _MAX_M if D4P_W12Q else _MAX_M), T, N * K)

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.planes, self.scale, self.rp, self.col, self.val,
                                                           self.rp_q, self.col_q, self.val_q,
                                                           self.esc_flat, self.esc_bf16))

    def decode(self) -> torch.Tensor:
        """bf16 [N, K] (original row order) in the shared scratch."""
        out = _SHARED[self.planes.device]["scratch"][: self.N * self.K].view(self.N, self.K)
        _ext().w12_decode(self.planes, self.scale, out, self.N, self.K, self.silu)
        if self.n_esc:
            out.view(-1)[self.esc_flat] = self.esc_bf16
        return out

    def gemm(self, x2: torch.Tensor, qmode: bool = False) -> torch.Tensor:
        """x2 [M <= 64, K] bf16 -> [M, nout] (gate_up: SiLU(gate) * up); qmode: the Q-plane pass (approximate)."""
        qmode = bool(qmode) and self.maxseg_q > 0
        m = x2.shape[0]
        nt = 16 if m <= 16 else (32 if m <= 32 else (48 if m <= 48 else 64))
        if x2.stride(-1) != 1 or x2.stride(0) % 8 or x2.data_ptr() % 16:
            x2 = x2.contiguous()
        sh = _SHARED[x2.device]
        out = torch.empty((m, self.nout), dtype=torch.bfloat16, device=x2.device)
        rp, col, val = (self.rp_q, self.col_q, self.val_q) if qmode else (self.rp, self.col, self.val)
        _ext().w12_gemm(x2, self.planes, self.scale, rp, col, val, out, sh["ws"], sh["cnt"], self.P,
                        self.maxseg_q if qmode else self.maxseg, self.N, self.silu, nt, self.esc_seg, D5_PDL, self.trigger,
                        int(qmode))
        return out


def _gemm_q(enc: "W12Weight", x2: torch.Tensor) -> torch.Tensor:
    """x2 [64 < M <= 128, K] bf16 -> [M, nout] in one launch (same results as two 64-row enc.gemm calls)."""
    m = x2.shape[0]
    if x2.stride(-1) != 1 or x2.stride(0) % 8 or x2.data_ptr() % 16:
        x2 = x2.contiguous()
    sh = _SHARED[x2.device]
    assert sh["ws"].numel() >= enc.T * enc.maxseg * _RB * 2 * _MAX_M, "W12Q: split-K workspace sized without the knob"
    out = torch.empty((m, enc.nout), dtype=torch.bfloat16, device=x2.device)
    _ext().w12_gemm_q(x2, enc.planes, enc.scale, enc.rp, enc.col, enc.val, out, sh["ws"], sh["cnt"], enc.P,
                      enc.maxseg, enc.N, enc.silu, enc.esc_seg, D5_PDL, enc.trigger)
    return out


def _chunked(enc: "W12Weight", x2: torch.Tensor) -> torch.Tensor:
    m = x2.shape[0]
    if m <= _MAX_M:
        return enc.gemm(x2)
    if D4P_W12Q:
        return _gemm_q(enc, x2)
    return torch.cat([enc.gemm(x2[i:i + _MAX_M]) for i in range(0, m, _MAX_M)])


def _maxseg(T, S, P):
    U = T * S
    own = lambda u: ((u + 1) * P - 1) // U
    return max(own(t * S + S - 1) - own(t * S) + 1 for t in range(T))


_SHARED: dict = {}


def _ensure_shared(dev, ws_floats, ntiles, scratch_elems):
    """One fp32 split-K workspace, one tile-counter array (reset by the kernel) and one bf16 prefill scratch per
    device, shared by every W12 layer (kernels run in stream order). Allocated at load, before the KV pool."""
    sh = _SHARED.setdefault(dev, {})
    if sh.get("ws") is None or sh["ws"].numel() < ws_floats:
        sh["ws"] = torch.empty(ws_floats, dtype=torch.float32, device=dev)
    if sh.get("cnt") is None or sh["cnt"].numel() < ntiles:
        sh["cnt"] = torch.zeros(ntiles, dtype=torch.int32, device=dev)
    if sh.get("scratch") is None or sh["scratch"].numel() < scratch_elems:
        sh["scratch"] = torch.empty(scratch_elems, dtype=torch.bfloat16, device=dev)


# --------------------------------------------------------------------------------------------------------------- dispatch


# Q pass (two-level verify, d8): while enabled, the small-M W12 GEMMs read only the Q plane (1 B/weight: sign, the
# 4-bit exponent code, the top 3 mantissa bits + the midpoint of the dropped ones; escapes corrected exactly) -- an
# approximate copy of the same weights at zero extra memory. Read at call time, so a CUDA graph captured inside `q_pass()` replays the Q kernels.
_QPASS = False


class q_pass:
    """Context manager: `with specpilot_d5_w12.q_pass(): ...` runs the W12 linears in Q mode (M <= 64 only)."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def __enter__(self):
        global _QPASS
        self._prev, _QPASS = _QPASS, bool(self.enabled)
        return self

    def __exit__(self, *exc):
        global _QPASS
        _QPASS = self._prev
        return False


def d5_linear(enc: W12Weight, x: torch.Tensor, rows: Optional[Tuple[int, int]] = None) -> torch.Tensor:
    """y = x @ w[rows]^T for an encoded (non-SiLU) weight."""
    k = enc.K
    m = x.numel() // k
    x2 = x.reshape(m, k)
    r0, r1 = rows if rows is not None else (0, enc.N)
    if 0 < m <= _CHUNK_M and x2.dtype == torch.bfloat16 and not torch.compiler.is_compiling():
        qm = _QPASS and m <= _MAX_M
        y = enc.gemm(x2, qm) if m <= _MAX_M else _chunked(enc, x2)
        if rows is not None and (r0, r1) != (0, enc.N):
            y = y[:, r0:r1]
    else:
        y = F.linear(x2, enc.decode()[r0:r1])
    return y.view(*x.shape[:-1], r1 - r0)


def d5_gate_up(enc: W12Weight, x: torch.Tensor):
    """(h, fused): fused=True -> h = SiLU(gate) * up [.., I]; fused=False -> h = gate_up [.., 2I] (caller applies act)."""
    k = enc.K
    m = x.numel() // k
    x2 = x.reshape(m, k)
    if 0 < m <= _CHUNK_M and x2.dtype == torch.bfloat16 and not torch.compiler.is_compiling():
        qm = _QPASS and m <= _MAX_M
        y = enc.gemm(x2, qm) if m <= _MAX_M else _chunked(enc, x2)
        return y.view(*x.shape[:-1], enc.nout), True
    return F.linear(x2, enc.decode()).view(*x.shape[:-1], enc.N), False


# ------------------------------------------------------------------------------------------------------- model encoding


def _check_roundtrip(enc: W12Weight, w: torch.Tensor) -> bool:
    return bool(torch.equal(enc.decode(), w))


def _free(param):
    param.data = torch.empty((0, param.shape[1]), dtype=param.dtype, device=param.device)


def d5_encode_target(model: torch.nn.Module) -> None:
    """Encode the target's dense MLP gate_up/down, folded GDN in_proj and attention qkv_proj (bf16, unquantized,
    tp 1); frees the bf16 weights. Raises if any decoded weight is not bit-exact."""
    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
    from sglang.srt.models.qwen2_moe import Qwen2MoeMLP

    def plain(lin):
        return (isinstance(getattr(lin, "quant_method", None), UnquantizedLinearMethod) and getattr(lin, "bias", None) is None
                and lin.weight.dtype == torch.bfloat16 and lin.weight.dim() == 2 and lin.weight.shape[0] > 0
                and lin.weight.shape[1] % 64 == 0 and getattr(lin, "tp_size", 1) == 1)

    stats = {"layers": 0, "bf16_bytes": 0, "w12_bytes": 0, "esc": 0, "esc_q": 0, "weights": 0, "exact": True, "inexact": 0,
             "max_row_esc": 0}

    def account(enc, w, what):
        rt = _check_roundtrip(enc, w)
        ok = rt
        stats["inexact"] += enc.n_inexact
        stats["exact"] &= ok
        stats["layers"] += 1
        stats["bf16_bytes"] += w.numel() * 2
        stats["w12_bytes"] += enc.nbytes()
        stats["esc"] += enc.n_esc
        stats["esc_q"] += enc.n_esc_q
        stats["max_row_esc"] = max(stats["max_row_esc"], enc.max_row_esc)
        stats["weights"] += w.numel()
        if not ok:
            bad = int((enc.decode() != w).sum().item())
            raise RuntimeError(f"SPECPILOT_D5 W12 not bit-exact at {what} {tuple(w.shape)}: roundtrip={rt} "
                               f"({bad} mismatches), exact_corr={enc.exact_corr}, escapes={enc.n_esc}")

    for name, mod in model.named_modules():
        if isinstance(mod, Qwen2MoeMLP) and plain(mod.gate_up_proj) and plain(mod.down_proj):
            gu, dn = mod.gate_up_proj, mod.down_proj
            if gu.weight.shape[0] % 128 == 0:
                enc = W12Weight(gu.weight.data, silu=True)
                account(enc, gu.weight.data, name + '.gate_up_proj')
                gu._d5_w12 = enc
                _free(gu.weight)
            enc = W12Weight(dn.weight.data)
            account(enc, dn.weight.data, name + '.down_proj')
            enc.trigger = D5_PDL  # its PDL dependent is the next layer's x3 add+RMSNorm (gdc_wait before reading)
            dn._d5_w12 = enc
            _free(dn.weight)
        elif type(mod).__name__ == "Qwen3_5GatedDeltaNet" and getattr(mod, "_fused_in_proj_weight", None) is not None:
            fused = mod._fused_in_proj_weight
            enc = W12Weight(fused)
            account(enc, fused, name + '.in_proj(fused)')
            width = mod._fused_in_proj_qkvz_width
            mod._d5_in_proj = enc
            mod.in_proj_qkvz._d5_w12, mod.in_proj_qkvz._d5_rows = enc, (0, width)
            mod.in_proj_ba._d5_w12, mod.in_proj_ba._d5_rows = enc, (width, enc.N)
            mod._fused_in_proj_weight = None
            _free(mod.in_proj_qkvz.weight)
            _free(mod.in_proj_ba.weight)
            del fused
        elif hasattr(mod, "qkv_proj") and hasattr(mod, "o_proj") and plain(mod.qkv_proj):
            q = mod.qkv_proj
            enc = W12Weight(q.weight.data)
            account(enc, q.weight.data, name + '.qkv_proj')
            q._d5_w12 = enc
            _free(q.weight)
        torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    logger.info(
        "SPECPILOT_D5 W12: encoded %d target linears, %.2f GB bf16 -> %.2f GB W12 (%.4f), escapes %d (rate %.3g), "
        "roundtrip bit-exact=%s, escape corrections below fp32 range %d, max escapes in a row %d, above-window (Q-pass) "
        "escapes %d", stats["layers"], stats["bf16_bytes"] / 1e9, stats["w12_bytes"] / 1e9,
        stats["w12_bytes"] / max(1, stats["bf16_bytes"]), stats["esc"], stats["esc"] / max(1, stats["weights"]),
        stats["exact"], stats["inexact"], stats["max_row_esc"], stats["esc_q"])
