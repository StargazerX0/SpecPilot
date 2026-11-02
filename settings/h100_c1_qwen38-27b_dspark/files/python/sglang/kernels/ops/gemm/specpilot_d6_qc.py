"""SPECPILOT_D6_QC: separately stored INT6 / INT8 Q copy of a target linear for TLV's Q pass (H100, sm_90a).

Format: symmetric INT6 or INT8, one bf16 scale per (row, 128 input channels), RTN at the per-group MSE-best clip of
_GRID (no data); quantizer from the Pro 6000 campaign's d3-001-qcopy (specpilot_d3_qcopy.py, owner-allowed read-only
reference). Records per (128-row tile, 128-channel stage), byte-aligned for a PRMT decode (build_records2):
INT6 = per lane and 64-channel half 6 words (16 weights per 3 words: 12 direct bytes, 4 weights in bits 6..7),
INT8 = per lane 8 words of u = q + 128; then bf16 scales [128]. Kernel: the W12 nested-Q stream-K structure (lib
w12 / w12_nest): persistent CTAs over (tile, stage) units, one producer warp (record by 1-D bulk TMA, x by two
128B-swizzled 2-D TMA boxes, NS-slot mbarrier ring, records issued before griddepcontrol.wait), 2 consumer
warpgroups decoding into bf16 wgmma A fragments (bf16(0x43 | byte) by PRMT; INT6 (f - 160) * s, INT8
min(s (f - 256), s (f / 2 - 128))) -> wgmma m64nNTk16 RS with B = x from smem, deterministic split-tile reduction
by the k = 0 owner, optional fused SiLU(gate) * up epilogue on W12's [64 gate | 64 up] tile interleave. M <= 64.
"""
from __future__ import annotations

import hashlib
import os

import torch

_RB = 128
_STK = 128
_GRID = (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7)
# ring slots per CTA: microbench scratch/d6/mb6.log (4 best of 4 / 6 / 8 at M=32)
_NSMAX = int(os.environ.get("SPECPILOT_D6_QC_NS", "4") or 4)


def _wgmma_fn(nt):
    nreg = nt // 2
    outs = ",".join(f"%{i}" for i in range(nreg))
    a = ",".join(f"%{nreg + i}" for i in range(4))
    cons_d = ", ".join(f'"+f"(d[{i}])' for i in range(nreg))
    return f"""
__device__ __forceinline__ void wgmma_n{nt}(float* d, const uint32_t* a, uint64_t desc, int sd) {{
  asm volatile("{{\\n.reg .pred p;\\nsetp.ne.b32 p, %{nreg + 5}, 0;\\n"
               "wgmma.mma_async.sync.aligned.m64n{nt}k16.f32.bf16.bf16 {{{outs}}}, {{{a}}}, %{nreg + 4}, p, 1, 1, 0;\\n}}\\n"
               : {cons_d}
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(sd));
}}
"""


_KERN = r"""
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

namespace {  // internal linkage: kernel variants built as separate modules must not share template statics
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
  asm volatile("{\n .reg .pred p;\n QCW_%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra QCW_%=;\n}\n"
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
__device__ __forceinline__ uint64_t mkdesc_sw128(uint32_t saddr) {
  return (uint64_t)((saddr & 0x3FFFF) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)(1024 >> 4) << 32) | ((uint64_t)1 << 62);
}
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int N> __device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory"); }
__device__ __forceinline__ void cons_bar(int n) { asm volatile("bar.sync 1, %0;\n" ::"r"(n) : "memory"); }

@@WGMMA@@

template <int NT> __device__ __forceinline__ void wgmma_nt(float* d, const uint32_t* a, uint64_t desc, int sd = 1);
template <> __device__ __forceinline__ void wgmma_nt<16>(float* d, const uint32_t* a, uint64_t desc, int sd) { wgmma_n16(d, a, desc, sd); }
template <> __device__ __forceinline__ void wgmma_nt<32>(float* d, const uint32_t* a, uint64_t desc, int sd) { wgmma_n32(d, a, desc, sd); }
template <> __device__ __forceinline__ void wgmma_nt<48>(float* d, const uint32_t* a, uint64_t desc, int sd) { wgmma_n48(d, a, desc, sd); }
template <> __device__ __forceinline__ void wgmma_nt<64>(float* d, const uint32_t* a, uint64_t desc, int sd) { wgmma_n64(d, a, desc, sd); }

__device__ __forceinline__ uint32_t lop3_or_and(uint32_t a, uint32_t b, uint32_t c) {  // (a & b) | c
  uint32_t d;
  asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  return d;
}
__device__ __forceinline__ uint32_t bf2_fma(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t d;
  asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  return d;
}
__device__ __forceinline__ uint32_t bf2_mul(uint32_t a, uint32_t b) {
  uint32_t d;
  asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
  return d;
}

// FORMAT 2 (d6): byte-aligned planes, decode mostly by PRMT into bf16(128 + byte) + bf16x2 FP ops.
// INT8: per lane 8 words, word 2j + rh: bytes (rr 0 e 0, rr 0 e 1, rr 1 e 0, rr 1 e 1) of reg r = rr + 2 rh at k16 step
// j, u = q + 128; f = bf16(0x43 | u) = 128 + u (u < 128) or 2u (u >= 128): q * s = min(s (f - 256), s (f / 2 - 128))
// (both exact before the one rounding; the min picks the right branch for s > 0).
__device__ __forceinline__ uint32_t bf2_min(uint32_t a, uint32_t b) {
  uint32_t d;
  asm("min.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
  return d;
}
struct QC8S { uint32_t s, hs, n256, n128; };
__device__ __forceinline__ QC8S qc8s(float s) {
  const __nv_bfloat162 a = __floats2bfloat162_rn(s, s), b = __floats2bfloat162_rn(0.5f * s, 0.5f * s);
  const __nv_bfloat162 c = __floats2bfloat162_rn(-256.f * s, -256.f * s), d = __floats2bfloat162_rn(-128.f * s, -128.f * s);
  QC8S r;
  r.s = *reinterpret_cast<const uint32_t*>(&a); r.hs = *reinterpret_cast<const uint32_t*>(&b);
  r.n256 = *reinterpret_cast<const uint32_t*>(&c); r.n128 = *reinterpret_cast<const uint32_t*>(&d);
  return r;
}
__device__ __forceinline__ uint32_t qc8_one(uint32_t f, const QC8S& c) {
  return bf2_min(bf2_fma(f, c.s, c.n256), bf2_fma(f, c.hs, c.n128));
}
__device__ __forceinline__ void qc_decode8v2(const uint4 ca, const uint4 cb, const QC8S& c0, const QC8S& c1,
                                             uint32_t (&A)[4][4]) {
  const uint32_t W[8] = {ca.x, ca.y, ca.z, ca.w, cb.x, cb.y, cb.z, cb.w};
  const uint32_t MAGW = 0x43434343u;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    A[j][0] = qc8_one(__byte_perm(W[2 * j], MAGW, 0x4140), c0);
    A[j][1] = qc8_one(__byte_perm(W[2 * j], MAGW, 0x4342), c1);
    A[j][2] = qc8_one(__byte_perm(W[2 * j + 1], MAGW, 0x4140), c0);
    A[j][3] = qc8_one(__byte_perm(W[2 * j + 1], MAGW, 0x4342), c1);
  }
}
// INT6: per lane 6 words = 2 groups (k16 steps 2jh, 2jh + 1) of 3 words: bytes 0..11 hold u of regs (j0 r0, j0 r1),
// (j0 r2, j0 r3), (j1 r0, j1 r1) in bits 0..5 (pairs e 0, e 1); u of (j1 r2, j1 r3) is spread over bits 6..7 of the
// three words (byte k of word w = bits 2w, 2w + 1 of packed weight k). q * s = (bf16(128 + u) - 160) * s.
__device__ __forceinline__ uint32_t qc6_one(uint32_t f, uint32_t s) {
  return bf2_mul(bf2_fma(f, 0x3F803F80u, 0xC320C320u), s);
}
__device__ __forceinline__ void qc_decode6v2(const uint32_t (&w)[6], const uint32_t s0, const uint32_t s1,
                                             uint32_t (&A)[4][4]) {
  const uint32_t MAGW = 0x43434343u, M6 = 0x3F3F3F3Fu;
#pragma unroll
  for (int jh = 0; jh < 2; ++jh) {
    const uint32_t W0 = w[3 * jh], W1 = w[3 * jh + 1], W2 = w[3 * jh + 2];
    const uint32_t d0 = W0 & M6, d1 = W1 & M6, d2 = W2 & M6;
    const uint32_t z = lop3_or_and(W2 >> 2, 0x30303030u, lop3_or_and(W1 >> 4, 0x0C0C0C0Cu, (W0 >> 6) & 0x03030303u));
    A[2 * jh][0] = qc6_one(__byte_perm(d0, MAGW, 0x4140), s0);
    A[2 * jh][1] = qc6_one(__byte_perm(d0, MAGW, 0x4342), s1);
    A[2 * jh][2] = qc6_one(__byte_perm(d1, MAGW, 0x4140), s0);
    A[2 * jh][3] = qc6_one(__byte_perm(d1, MAGW, 0x4342), s1);
    A[2 * jh + 1][0] = qc6_one(__byte_perm(d2, MAGW, 0x4140), s0);
    A[2 * jh + 1][1] = qc6_one(__byte_perm(d2, MAGW, 0x4342), s1);
    A[2 * jh + 1][2] = qc6_one(__byte_perm(z, MAGW, 0x4140), s0);
    A[2 * jh + 1][3] = qc6_one(__byte_perm(z, MAGW, 0x4342), s1);
  }
}

template <int NT, int BITS, int NS>
__global__ void __launch_bounds__(288, 1) qc_kernel(
    const __grid_constant__ CUtensorMap xmap, int M, const unsigned char* __restrict__ recs, int64_t tstride, int S,
    int U, __nv_bfloat16* __restrict__ out, int64_t ldo, int N, float* __restrict__ ws, int maxseg,
    int* __restrict__ counters, int silu, int nout, int trig) {
  constexpr int NCW = 8, RB = 128;
  constexpr int RECB = BITS == 6 ? 12544 : 16640;   // planes 12 / 16 KB + bf16 scales [128] (256 B)
  constexpr int XOFF = BITS == 6 ? 13312 : 17408;   // x stage offset in a slot (1 KB aligned for the 128B swizzle)
  constexpr int XB = 2 * NT * 128;
  constexpr int SB = XOFF + XB;
  constexpr int NREG = NT / 2;
  extern __shared__ __align__(1024) unsigned char smem[];
  __shared__ __align__(8) uint64_t bars[2 * NS];
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int P = gridDim.x, p = blockIdx.x;
  const int u0 = (int)(((int64_t)p * U) / P), u1 = (int)(((int64_t)(p + 1) * U) / P);
  const uint32_t sraw = (uint32_t)__cvta_generic_to_shared(smem);
  const uint32_t sbase = (sraw + 1023u) & ~1023u;
  unsigned char* const smem_al = smem + (sbase - sraw);
  float* xchg = reinterpret_cast<float*>(smem_al + NS * SB);  // [64][NT] up values for the fused SiLU epilogue
  const uint32_t full0 = (uint32_t)__cvta_generic_to_shared(bars), empty0 = full0 + 8 * NS;
  if (threadIdx.x == 0) {
    for (int s = 0; s < NS; ++s) { mbar_init(full0 + 8 * s, 1); mbar_init(empty0 + 8 * s, NCW); }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();
  if (trig) asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
  if (u0 >= u1) return;
  if (warp == NCW) {  // producer
    if (lane == 0) {
      const int npre = min(NS, u1 - u0);
      for (int i = 0; i < npre; ++i) {  // PDL: weight records do not depend on the predecessor
        const int u = u0 + i, tile = u / S, s = u - tile * S;
        const uint32_t fb = full0 + 8 * i;
        mbar_expect_tx(fb, RECB + XB);
        bulk_g2s(sbase + i * SB, recs + (int64_t)tile * tstride + (int64_t)s * RECB, RECB, fb);
      }
      asm volatile("griddepcontrol.wait;\n" ::: "memory");
      for (int i = 0; i < npre; ++i) {
        const int u = u0 + i, tile = u / S, s = u - tile * S;
#pragma unroll
        for (int hh = 0; hh < 2; ++hh)
          tma2d(sbase + i * SB + XOFF + hh * NT * 128, &xmap, s * 128 + hh * 64, 0, full0 + 8 * i);
      }
      for (int u = u0 + npre; u < u1; ++u) {
        const int i = u - u0, slot = i % NS, tile = u / S, s = u - tile * S;
        mbar_wait(empty0 + 8 * slot, ((i / NS) - 1) & 1);
        const uint32_t fb = full0 + 8 * slot, dst = sbase + slot * SB;
        mbar_expect_tx(fb, RECB + XB);
        bulk_g2s(dst, recs + (int64_t)tile * tstride + (int64_t)s * RECB, RECB, fb);
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) tma2d(dst + XOFF + hh * NT * 128, &xmap, s * 128 + hh * 64, 0, fb);
      }
    }
    return;
  }
  // consumers: warp w = rows w*16 .. +15 of the tile (warpgroup = 64 rows = one wgmma m64)
  const int rw = warp, g = lane / 4, t = lane % 4;
  float acc[NREG];
  uint32_t A0[2][4][4], A1[2][4][4];
  int pending = -1;
  auto stage = [&](int i, uint32_t (&A)[2][4][4]) {
    const int slot = i % NS;
    mbar_wait(full0 + 8 * slot, (i / NS) & 1);
    const unsigned char* sl = smem_al + slot * SB;
    const __nv_bfloat16* scs = reinterpret_cast<const __nv_bfloat16*>(sl + RECB - 256);
    if (BITS == 6) {
      const uint32_t sa = (uint32_t)__bfloat16_as_ushort(scs[rw * 16 + g]);
      const uint32_t sb = (uint32_t)__bfloat16_as_ushort(scs[rw * 16 + g + 8]);
      const uint32_t s0 = sa | (sa << 16), s1 = sb | (sb << 16);
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const uint2* pl = reinterpret_cast<const uint2*>(sl) + (hh * 8 + rw) * 3 * 32 + lane;
        const uint2 c0 = pl[0], c1 = pl[32], c2 = pl[64];
        const uint32_t w6[6] = {c0.x, c0.y, c1.x, c1.y, c2.x, c2.y};
        qc_decode6v2(w6, s0, s1, A[hh]);
      }
    } else {
      const QC8S q8a = qc8s(__bfloat162float(scs[rw * 16 + g])), q8b = qc8s(__bfloat162float(scs[rw * 16 + g + 8]));
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const uint4* pl = reinterpret_cast<const uint4*>(sl) + (hh * 8 + rw) * 2 * 32 + lane;
        qc_decode8v2(pl[0], pl[32], q8a, q8b, A[hh]);
      }
    }
#pragma unroll
    for (int q = 0; q < NREG; ++q) asm volatile("" : "+f"(acc[q])::"memory");
    wg_fence();
    const uint32_t xs = sbase + slot * SB + XOFF;
#pragma unroll
    for (int hh = 0; hh < 2; ++hh)
#pragma unroll
      for (int j = 0; j < 4; ++j) wgmma_nt<NT>(acc, A[hh][j], mkdesc_sw128(xs + hh * NT * 128 + j * 32));
    wg_commit();
#pragma unroll
    for (int q = 0; q < NREG; ++q) asm volatile("" : "+f"(acc[q])::"memory");
    wg_wait<1>();
    if (pending >= 0 && lane == 0) mbar_arrive(empty0 + 8 * pending);
    pending = slot;
  };
  for (int u = u0; u < u1;) {
    const int tile = u / S, s0 = u - tile * S;
    const int uend = min(u1, (tile + 1) * S);
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
    const bool whole = (s0 == 0) && (uend == (tile + 1) * S);
    bool emit = whole;
    if (!whole) {
      const int ufirst = tile * S, ulast = ufirst + S - 1;
      const int pfirst = (int)((((int64_t)ufirst + 1) * P - 1) / U), plast = (int)((((int64_t)ulast + 1) * P - 1) / U);
      const int seg = p - pfirst, nseg = plast - pfirst + 1;
      float* wt = ws + ((int64_t)tile * maxseg) * RB * NT;
      const int rl = rw * 16 + g;
      if (seg > 0) {
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
          for (int q = 0; q < NT / 8; ++q)
            __stcg(reinterpret_cast<float2*>(wt + ((int64_t)seg * RB + rl + 8 * h) * NT + q * 8 + 2 * t),
                   make_float2(acc[q * 4 + 2 * h], acc[q * 4 + 2 * h + 1]));
        cons_bar(256);
        if (threadIdx.x == 0) asm volatile("red.release.gpu.global.add.s32 [%0], 1;\n" ::"l"(counters + tile) : "memory");
      } else {
        if (threadIdx.x == 0) {
          int c;
          do {
            asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\n" : "=r"(c) : "l"(counters + tile) : "memory");
          } while (c < nseg - 1);
          counters[tile] = 0;
        }
        cons_bar(256);
        constexpr int PB = NT <= 32 ? 4 : (NT <= 48 ? 2 : 1);
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
          const int n = tile * RB + warp * 16 + g + 8 * h;
#pragma unroll
          for (int q = 0; q < NT / 8; ++q)
#pragma unroll
            for (int v = 0; v < 2; ++v) {
              const int m = q * 8 + 2 * t + v;
              if (m < M && n < N) out[(int64_t)m * ldo + n] = __float2bfloat16(acc[q * 4 + 2 * h + v]);
            }
        }
      } else {  // tile = [64 gate rows | 64 up rows]; bf16-rounded operands as act_and_mul
        const int wr = (warp & 3) * 16 + g;
        if (warp >= 4) {
#pragma unroll
          for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int q = 0; q < NT / 8; ++q)
#pragma unroll
              for (int v = 0; v < 2; ++v)
                xchg[(wr + 8 * h) * NT + q * 8 + 2 * t + v] = __bfloat162float(__float2bfloat16(acc[q * 4 + 2 * h + v]));
        }
        cons_bar(256);
        if (warp < 4) {
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const int n = tile * 64 + wr + 8 * h;
#pragma unroll
            for (int q = 0; q < NT / 8; ++q)
#pragma unroll
              for (int v = 0; v < 2; ++v) {
                const int m = q * 8 + 2 * t + v;
                const float gg = __bfloat162float(__float2bfloat16(acc[q * 4 + 2 * h + v]));
                const float uu = xchg[(wr + 8 * h) * NT + q * 8 + 2 * t + v];
                if (m < M && n < nout) out[(int64_t)m * ldo + n] = __float2bfloat16(gg / (1.f + __expf(-gg)) * uu);
              }
          }
        }
        cons_bar(256);
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

template <int NT, int BITS>
void launch(torch::Tensor x, int64_t M, torch::Tensor recs, int64_t S, torch::Tensor out, torch::Tensor ws,
            torch::Tensor cnt, int64_t P, int64_t maxseg, int64_t N, int64_t silu, int64_t pdl, int64_t trig) {
  constexpr int XOFF = BITS == 6 ? 13312 : 17408;
  constexpr int SB = XOFF + 2 * NT * 128;
  constexpr int EXTRA = 64 * NT * 4;
  constexpr int NS0 = (210 * 1024 - EXTRA) / SB;
  constexpr int NS = NS0 > QC_NSMAX ? QC_NSMAX : NS0;
  const int K = x.size(1);
  TORCH_CHECK(M <= NT && M > 0 && K % 128 == 0, "QC: bad M / K");
  const int T = recs.size(0), U = T * (int)S;
  CUtensorMap map;
  cuuint64_t gdim[2] = {(cuuint64_t)K, (cuuint64_t)M};
  cuuint64_t gstr[1] = {(cuuint64_t)x.stride(0) * 2};
  cuuint32_t box[2] = {64, NT};
  cuuint32_t es[2] = {1, 1};
  CUresult r = get_encode()(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, x.data_ptr(), gdim, gstr, box, es,
                            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "QC tensor map encode failed ", (int)r);
  const int smem = NS * SB + EXTRA + 1024;
  auto kern = qc_kernel<NT, BITS, NS>;
  static bool attr = false;
  if (!attr) { cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); attr = true; }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((unsigned)P);
  cfg.blockDim = dim3(288);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = c10::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
  cfg.attrs = at;
  cfg.numAttrs = 1;
  cudaError_t e = cudaLaunchKernelEx(&cfg, kern, map, (int)M,
      reinterpret_cast<const unsigned char*>(recs.data_ptr()), (int64_t)recs.stride(0), (int)S, U,
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), (int64_t)out.stride(0), (int)N,
      ws.data_ptr<float>(), (int)maxseg, cnt.data_ptr<int>(), (int)silu, (int)out.size(1), (int)trig);
  TORCH_CHECK(e == cudaSuccess, "QC launch failed: ", cudaGetErrorString(e));
}

}  // namespace

void qc_gemm(torch::Tensor x, int64_t M, torch::Tensor recs, int64_t S, torch::Tensor out, torch::Tensor ws,
             torch::Tensor cnt, int64_t P, int64_t maxseg, int64_t N, int64_t bits, int64_t nt, int64_t silu,
             int64_t pdl, int64_t trig) {
#define QC_L(NT_, B_) if (nt == NT_ && bits == B_) return launch<NT_, B_>(x, M, recs, S, out, ws, cnt, P, maxseg, N, silu, pdl, trig);
  QC_L(16, 6) QC_L(32, 6) QC_L(48, 6) QC_L(64, 6) QC_L(16, 8) QC_L(32, 8) QC_L(48, 8) QC_L(64, 8)
#undef QC_L
  TORCH_CHECK(false, "QC: bad nt / bits");
}
"""

_CPP = ("void qc_gemm(torch::Tensor x, int64_t M, torch::Tensor recs, int64_t S, torch::Tensor out, torch::Tensor ws, "
        "torch::Tensor cnt, int64_t P, int64_t maxseg, int64_t N, int64_t bits, int64_t nt, int64_t silu, int64_t pdl, "
        "int64_t trig);")
_CFLAGS = ["-O3", "-gencode=arch=compute_90a,code=sm_90a", "--use_fast_math"]
_mod = None


def _ext():
    global _mod
    if _mod is None:
        from torch.utils.cpp_extension import load_inline

        fns = "".join(_wgmma_fn(nt) for nt in (16, 32, 48, 64))
        src = "#define QC_NSMAX %d\n" % _NSMAX + _KERN.replace("@@WGMMA@@", fns) + _HOST
        base = os.environ.get("SGLANG_CACHE_DIR") or os.path.join(os.path.expanduser("~"), ".cache", "sglang")
        key = hashlib.sha1("\0".join([_CPP, src, *_CFLAGS]).encode()).hexdigest()[:16]
        bdir = os.path.join(base, "specpilot_d6_qc_src", key)
        os.makedirs(bdir, exist_ok=True)
        _mod = load_inline(name="specpilot_d6_qc_" + key, cpp_sources=_CPP, cuda_sources=src, functions=["qc_gemm"],
                           extra_cuda_cflags=_CFLAGS, build_directory=bdir, verbose=False)
    return _mod


# ------------------------------------------------------------------------------------------------------- quantize


def quantize_q(w: torch.Tensor, bits: int, chunk: int = 2048):
    """bf16 [N, K] -> (u uint8 [N, K] in 0..2^bits-1, scale bf16 [N, K/128]); q = u - 2^(bits-1), w ~ q * s,
    s = a * amax / (2^(bits-1) - 1) with a the per-group MSE-best factor of _GRID (RTN, no data)."""
    n, k = w.shape
    assert k % _STK == 0
    qmax = 2 ** (bits - 1) - 1
    u = torch.empty((n, k), dtype=torch.uint8, device=w.device)
    sc = torch.empty((n, k // _STK), dtype=torch.bfloat16, device=w.device)
    for r0 in range(0, n, chunk):
        x = w[r0:r0 + chunk].float().view(-1, k // _STK, _STK)
        amax = x.abs().amax(-1, keepdim=True)
        best_err = best_s = None
        for a in _GRID:
            s = (amax * (a / qmax)).to(torch.bfloat16).float().clamp_min(1e-12)
            q = torch.clamp(torch.round(x / s), -qmax - 1, qmax)
            err = (q * s - x).pow(2).sum(-1, keepdim=True)
            if best_err is None:
                best_err, best_s = err, s
            else:
                take = err < best_err
                best_err = torch.where(take, err, best_err)
                best_s = torch.where(take, s, best_s)
        q = torch.clamp(torch.round(x / best_s), -qmax - 1, qmax)
        u[r0:r0 + chunk] = (q + qmax + 1).to(torch.uint8).view(-1, k)
        sc[r0:r0 + chunk] = best_s.view(-1, k // _STK).to(torch.bfloat16)
    return u, sc


def dequantize_q(u: torch.Tensor, sc: torch.Tensor, bits: int) -> torch.Tensor:
    """fp32 q * s (the reference; the kernel uses bf16(q * s))."""
    n, k = u.shape
    q = u.float() - 2 ** (bits - 1)
    return (q.view(n, k // _STK, _STK) * sc.float().unsqueeze(-1)).view(n, k)


def build_records2(u: torch.Tensor, sc: torch.Tensor, bits: int, chunk_tiles: int = 16):
    """FORMAT 2 records (byte-aligned planes, see qc_decode6v2 / qc_decode8v2) -> (recs, rec_bytes, xoff).
    Per (tile, stage): plane [hh 2][rw 8][lane 32] x (INT6: [wd 3][..][8 B] / INT8: [half 2][..][16 B]) + scales."""
    n, k = u.shape
    S = k // _STK
    T = (n + _RB - 1) // _RB
    pb = 8192 if bits == 8 else 6144
    rec_bytes = 2 * pb + 256
    xoff = (rec_bytes + 1023) // 1024 * 1024
    recs = torch.empty((T, S * rec_bytes), dtype=torch.uint8, device=u.device)
    mid = 2 ** (bits - 1)
    for t0 in range(0, T, chunk_tiles):
        t1 = min(T, t0 + chunk_tiles)
        nt = t1 - t0
        r0, r1 = t0 * _RB, min(n, t1 * _RB)
        uu = torch.full((nt * _RB, k), mid, dtype=torch.uint8, device=u.device)
        uu[: r1 - r0] = u[r0:r1]
        # row = rw*16 + rr*8 + g, k = s*128 + hh*64 + j*16 + rh*8 + t*2 + e -> [T, S, hh, rw, lane(g, t), j, rh, rr, e]
        L = uu.view(nt, 8, 2, 8, S, 2, 4, 2, 4, 2).permute(0, 4, 5, 1, 3, 8, 6, 7, 2, 9).reshape(nt, S, 2, 8, 32, 32)
        if bits == 8:
            pl = L.view(nt, S, 2, 8, 32, 2, 16).permute(0, 1, 2, 3, 5, 4, 6).reshape(nt, S, 2 * pb)
        else:
            v = L.view(nt, S, 2, 8, 32, 2, 16).to(torch.int32)  # [.., group jh, 16 weights]
            pk = v[..., 12:16]  # packed weights (j1 r2 e0, e1, j1 r3 e0, e1)
            byt = torch.stack([v[..., 4 * w:4 * w + 4] | (((pk >> (2 * w)) & 3) << 6) for w in range(3)], -2)
            byt = byt.reshape(nt, S, 2, 8, 32, 24).to(torch.uint8)  # lane's 6 words = [jh][w][byte k]
            pl = byt.view(nt, S, 2, 8, 32, 3, 8).permute(0, 1, 2, 3, 5, 4, 6).reshape(nt, S, 2 * pb)
        scs = torch.zeros((nt * _RB, S), dtype=torch.bfloat16, device=u.device)
        scs[: r1 - r0] = sc[r0:r1]
        sb = scs.view(nt, _RB, S).permute(0, 2, 1).contiguous().view(torch.uint8).view(nt, S, 256)
        recs[t0:t1] = torch.cat([pl, sb], dim=2).view(nt, S * rec_bytes)
    return recs, rec_bytes, xoff


def _maxseg(T, S, P):
    U = T * S
    own = lambda u: ((u + 1) * P - 1) // U  # noqa: E731
    return max(own(t * S + S - 1) - own(t * S) + 1 for t in range(T))


_SHARED: dict = {}


def _ensure_shared(dev, ws_floats, ntiles):
    sh = _SHARED.setdefault(dev, {})
    if sh.get("ws") is None or sh["ws"].numel() < ws_floats:
        sh["ws"] = torch.empty(ws_floats, dtype=torch.float32, device=dev)
    if sh.get("cnt") is None or sh["cnt"].numel() < ntiles:
        sh["cnt"] = torch.zeros(ntiles, dtype=torch.int32, device=dev)
    return sh


class QCLinear:
    """Scaled INT6 / INT8 g128 Q copy of a bf16 [N, K] weight; __call__(x2 [M <= 64, K] bf16) -> [M, N] bf16
    (silu: [M, N/2] = SiLU(gate) * up of the [gate | up] weight, fused in the epilogue)."""

    MAX_M = 64

    def __init__(self, w: torch.Tensor, bits: int, silu: bool = False, pdl: int = 1, trigger: int = 0):
        assert bits in (6, 8) and w.dim() == 2 and w.shape[1] % _STK == 0
        self.bits, self.n, self.k, self.silu = bits, w.shape[0], w.shape[1], int(silu)
        if silu:  # [gate | up] -> per 128-row tile [64 gate | 64 up] (as W12); the kernel emits SiLU(gate) * up
            assert self.n % 128 == 0
            w = w.view(2, self.n // 128, 64, self.k).transpose(0, 1).reshape(self.n, self.k)
        self.nout = self.n // 2 if silu else self.n
        u, sc = quantize_q(w, bits)
        self.recs, self.rec_bytes, self.xoff = build_records2(u, sc, bits)
        del u, sc
        self.T, self.S = self.recs.shape[0], self.k // _STK
        nsm = torch.cuda.get_device_properties(w.device).multi_processor_count
        self.P = min(nsm, self.T * self.S)
        self.maxseg = _maxseg(self.T, self.S, self.P)
        self.pdl = pdl
        # PDL trigger at kernel start: only where every dependent waits (griddepcontrol) before reading the output
        self.trigger = trigger
        _ensure_shared(w.device, self.T * self.maxseg * _RB * self.MAX_M, self.T)

    def nbytes(self) -> int:
        return self.recs.numel()

    def __call__(self, x2: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        m = x2.shape[0]
        assert 0 < m <= self.MAX_M and x2.dtype == torch.bfloat16
        if x2.stride(-1) != 1 or x2.stride(0) % 8 or x2.data_ptr() % 16:
            x2 = x2.contiguous()
        nt = 16 if m <= 16 else (32 if m <= 32 else (48 if m <= 48 else 64))
        if out is None:
            out = torch.empty((m, self.nout), device=x2.device, dtype=torch.bfloat16)
        sh = _SHARED[x2.device]
        _ext().qc_gemm(x2, m, self.recs, self.S, out, sh["ws"], sh["cnt"], self.P, self.maxseg, self.n, self.bits, nt,
                       self.silu, self.pdl, self.trigger)
        return out
