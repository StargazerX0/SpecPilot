"""SPECPILOT_D5_W4S: d7's INT4 drafter linears (asym g128 body, sym g256 head) on a stream-K Hopper kernel.

The drafter only proposes tokens, so its numerics need not match anything bitwise; this module keeps d7's exact
numbers (same q / s / m8, per 128-channel group acc += s * dot(x, q - 8) (+ m8 * sum(x)), fp32 accumulation, bf16
out; sym g256 = two 128-channel halves with the same scale) and only changes how they are streamed.

Records (built at load from a d7 W4Linear): per (128-row tile, 256-channel stage) one contiguous record = nibbles
[half 2][warp 8][hh 2][lane 32][16 B] in mma-A-fragment order (one 32-bit word = the 4 A registers of one k16 step:
bytes (lo, hi) = (reg0.e, reg2.e) / (reg1.e, reg3.e)) + scales [half 2][s | m8][128] fp32 (sym: [s][128]).
Kernel: persistent stream-K over (tile, stage) units (one CTA per SM, PDL: the first ring records are issued before
griddepcontrol.wait), a producer warp (record by 1-D bulk TMA, x by four 128B-swizzled 2-D TMA boxes per stage),
8 consumer warps (16 weight rows each: nibble -> bf16(q - 8) by prmt/lop3/fma, mma.sync m16n8k16 with B by
ldmatrix from the swizzled x stage, per-group fold), an aux warp (asym group sums of x by a ones-A mma.sync), split
tiles reduced by the k = 0 owner in segment order (deterministic for a fixed P), in-kernel tree-verify row cut (rows
>= M of the R output rows are zeros). Used for the computed rows M <= 32 of one request; anything else falls back to
the d7 object. Knob SPECPILOT_D5_W4S=1 (read once at import; applied by dflash_d7_w4.apply_d7_w4_ to the MLP,
fc and head parts, the narrow qkv / o / conv shapes stay on d7's kernel).
"""
from __future__ import annotations

import hashlib
import os

import torch

D5_W4S = os.environ.get("SPECPILOT_D5_W4S", "").strip() not in ("", "0")
_RB = 128          # weight rows per tile
_STK = 256         # channels per stage
_WREC = 16384      # nibble bytes per (tile, stage)
_XOFF = 18432      # smem offset of the x stage inside a ring slot (1 KB aligned)


_KERN = r"""
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <type_traits>

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
  asm volatile("{\n .reg .pred p;\n W4W_%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra W4W_%=;\n}\n"
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
__device__ __forceinline__ void cons_bar(int n) { asm volatile("bar.sync 1, %0;\n" ::"r"(n) : "memory"); }

__device__ __forceinline__ void ldsm_x4(uint32_t addr, uint32_t (&b)[4]) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(b[0]), "=r"(b[1]), "=r"(b[2]), "=r"(b[3]) : "r"(addr));
}
__device__ __forceinline__ void mma16816(float* d, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// 4 packed nibble words (one 64-channel quarter: 4 k16 steps) -> A[4][4] bf16x2 (q - 8, exact): per word bytes
// (b0, b1, b2, b3), regs = (lo b0, lo b1), (lo b2, lo b3), (hi b0, hi b1), (hi b2, hi b3) as d7's unpack
__device__ __forceinline__ void w4_decode(const uint4 c, uint32_t (&A)[4][4]) {
  const uint32_t W[4] = {c.x, c.y, c.z, c.w};
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const uint32_t t0 = __byte_perm(W[j], 0, 0x4140), t1 = __byte_perm(W[j], 0, 0x4342);
    uint32_t a0, a1, a2, a3;
    asm("lop3.b32 %0, %1, 0x000F000F, 0x43004300, 0xEA;" : "=r"(a0) : "r"(t0));
    asm("lop3.b32 %0, %1, 0x000F000F, 0x43004300, 0xEA;" : "=r"(a1) : "r"(t1));
    asm("lop3.b32 %0, %1, 0x000F000F, 0x43004300, 0xEA;" : "=r"(a2) : "r"(t0 >> 4));
    asm("lop3.b32 %0, %1, 0x000F000F, 0x43004300, 0xEA;" : "=r"(a3) : "r"(t1 >> 4));
    // bf16x2 (128 + q) * 1 - 136 = q - 8 (exact)
    asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(A[j][0]) : "r"(a0), "r"(0x3F803F80u), "r"(0xC308C308u));
    asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(A[j][1]) : "r"(a1), "r"(0x3F803F80u), "r"(0xC308C308u));
    asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(A[j][2]) : "r"(a2), "r"(0x3F803F80u), "r"(0xC308C308u));
    asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(A[j][3]) : "r"(a3), "r"(0x3F803F80u), "r"(0xC308C308u));
  }
}

template <int NT, int ASYM, int NS>
__global__ void __launch_bounds__(320, 1) w4s_kernel(
    const __grid_constant__ CUtensorMap xmap, int M, int R,
    const unsigned char* __restrict__ recs, int64_t tstride, int S, int U, int rec_bytes,
    __nv_bfloat16* __restrict__ out, int64_t ldo, int N,
    float* __restrict__ ws, int maxseg, int* __restrict__ counters) {
  constexpr int NCW = 8, RB = 128;
  constexpr int XB = NT * 512;               // x stage: 4 boxes of NT rows x 64 channels (128-B rows)
  constexpr int SB = 18432 + XB;             // ring slot: record (<= 18 KB) + x stage
  constexpr int NREG = NT / 2;
  extern __shared__ __align__(1024) unsigned char smem[];
  __shared__ __align__(8) uint64_t bars[3 * NS];
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int P = gridDim.x, p = blockIdx.x;
  const int u0 = (int)(((int64_t)p * U) / P), u1 = (int)(((int64_t)(p + 1) * U) / P);
  const uint32_t sraw = (uint32_t)__cvta_generic_to_shared(smem);
  const uint32_t sbase = (sraw + 1023u) & ~1023u;
  unsigned char* const smem_al = smem + (sbase - sraw);
  const uint32_t full0 = (uint32_t)__cvta_generic_to_shared(bars), empty0 = full0 + 8 * NS, sxf0 = empty0 + 8 * NS;
  float* sxs = reinterpret_cast<float*>(smem_al + NS * SB);  // [slot][half][NT] fp32 group sums of x (asym)
  if (threadIdx.x == 0) {
    for (int s = 0; s < NS; ++s) { mbar_init(full0 + 8 * s, 1); mbar_init(empty0 + 8 * s, NCW + ASYM); mbar_init(sxf0 + 8 * s, 1); }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();
  if (u0 >= u1) return;
  if (warp == NCW) {  // producer
    if (lane == 0) {
      const int npre = min(NS, u1 - u0);
      // PDL: the first NS records do not depend on the predecessor -> issued before griddepcontrol.wait
      for (int i = 0; i < npre; ++i) {
        const int u = u0 + i, tile = u / S, s = u - tile * S;
        const uint32_t fb = full0 + 8 * i;
        mbar_expect_tx(fb, rec_bytes + XB);
        bulk_g2s(sbase + i * SB, recs + (int64_t)tile * tstride + (int64_t)s * rec_bytes, rec_bytes, fb);
      }
      asm volatile("griddepcontrol.wait;\n" ::: "memory");
      for (int i = 0; i < npre; ++i) {
        const int u = u0 + i, tile = u / S, s = u - tile * S;
#pragma unroll
        for (int q = 0; q < 4; ++q) tma2d(sbase + i * SB + 18432 + q * NT * 128, &xmap, s * 256 + q * 64, 0, full0 + 8 * i);
      }
      for (int u = u0 + npre; u < u1; ++u) {
        const int i = u - u0, slot = i % NS, tile = u / S, s = u - tile * S;
        mbar_wait(empty0 + 8 * slot, ((i / NS) - 1) & 1);
        const uint32_t fb = full0 + 8 * slot, dst = sbase + slot * SB;
        mbar_expect_tx(fb, rec_bytes + XB);
        bulk_g2s(dst, recs + (int64_t)tile * tstride + (int64_t)s * rec_bytes, rec_bytes, fb);
#pragma unroll
        for (int q = 0; q < 4; ++q) tma2d(dst + 18432 + q * NT * 128, &xmap, s * 256 + q * 64, 0, fb);
      }
    }
    return;
  }
  if (warp == NCW + 1) {  // aux warp (asym): fp32 sums of x per (row, 128-channel group) by a ones-A mma.sync
    if (!ASYM) return;
    const uint32_t ONES[4] = {0x3F803F80u, 0x3F803F80u, 0x3F803F80u, 0x3F803F80u};
    const int mi = lane >> 3, r8 = lane & 7;
    for (int i = 0; u0 + i < u1; ++i) {
      const int slot = i % NS;
      mbar_wait(full0 + 8 * slot, (i / NS) & 1);
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        float d[NT / 8][4];
#pragma unroll
        for (int nb = 0; nb < NT / 8; ++nb) d[nb][0] = d[nb][1] = d[nb][2] = d[nb][3] = 0.f;
#pragma unroll
        for (int qb = 0; qb < 2; ++qb) {
          const uint32_t xb = sbase + slot * SB + 18432 + (half * 2 + qb) * NT * 128;
#pragma unroll
          for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int np = 0; np < NT / 16; ++np) {
              const int n = np * 16 + (mi >> 1) * 8 + r8, c = 2 * j + (mi & 1);
              uint32_t bb[4];
              ldsm_x4(xb + n * 128 + ((c ^ (n & 7)) << 4), bb);
              mma16816(d[np * 2], ONES, bb[0], bb[1]);
              mma16816(d[np * 2 + 1], ONES, bb[2], bb[3]);
            }
        }
        if (lane < 4) {  // row 0 of D: column sums (cols 2t, 2t+1 of each 8-wide n block)
#pragma unroll
          for (int nb = 0; nb < NT / 8; ++nb) {
            sxs[(slot * 2 + half) * NT + nb * 8 + 2 * lane] = d[nb][0];
            sxs[(slot * 2 + half) * NT + nb * 8 + 2 * lane + 1] = d[nb][1];
          }
        }
      }
      __syncwarp();
      if (lane == 0) { mbar_arrive(sxf0 + 8 * slot); mbar_arrive(empty0 + 8 * slot); }
    }
    return;
  }
  // consumers: 8 warps = 2 warpgroups x 64 rows; quarter = 64 channels (4 k16 steps), group = 128 channels
  const int g = lane / 4, t = lane % 4;
  const int wr = warp * 16 + g;  // tile row of acc[.. 2h ..] h = 0 (h = 1: + 8)
  float acc[NREG];
  const int mi = lane >> 3, r8 = lane & 7;
  int ibase = 0;
  for (int u = u0; u < u1;) {
    const int tile = u / S, s0 = u - tile * S;
    const int uend = min(u1, (tile + 1) * S);
#pragma unroll
    for (int q = 0; q < NREG; ++q) acc[q] = 0.f;
    for (int uu = u; uu < uend; ++uu) {
      const int i = ibase + (uu - u), slot = i % NS;
      mbar_wait(full0 + 8 * slot, (i / NS) & 1);
      const uint4* rw = reinterpret_cast<const uint4*>(smem_al + slot * SB);
      const float* sc = reinterpret_cast<const float*>(smem_al + slot * SB + _WREC_);
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        float aq[NT / 8][4];  // [n block][c0..c3]: two independent chains per n block (even / odd k16)
        float aq2[NT / 8][4];
#pragma unroll
        for (int nb = 0; nb < NT / 8; ++nb)
#pragma unroll
          for (int c = 0; c < 4; ++c) aq[nb][c] = aq2[nb][c] = 0.f;
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
          uint32_t A[4][4];
          w4_decode(rw[((half * 8 + warp) * 2 + hh) * 32 + lane], A);
          const uint32_t xb = sbase + slot * SB + 18432 + (half * 2 + hh) * NT * 128;
#pragma unroll
          for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int np = 0; np < NT / 16; ++np) {
              const int n = np * 16 + (mi >> 1) * 8 + r8, c = 2 * j + (mi & 1);
              uint32_t bb[4];
              ldsm_x4(xb + n * 128 + ((c ^ (n & 7)) << 4), bb);
              float (*dst)[4] = (j & 1) ? aq2 : aq;
              mma16816(dst[np * 2], A[j], bb[0], bb[1]);
              mma16816(dst[np * 2 + 1], A[j], bb[2], bb[3]);
            }
        }
        // fold: acc += s * dot (+ m8 * sum)
        float s0v, s1v, m0v = 0.f, m1v = 0.f;
        if (ASYM) {
          s0v = sc[(half * 2) * RB + wr]; s1v = sc[(half * 2) * RB + wr + 8];
          m0v = sc[(half * 2 + 1) * RB + wr]; m1v = sc[(half * 2 + 1) * RB + wr + 8];
          if (half == 0) mbar_wait(sxf0 + 8 * slot, (i / NS) & 1);
        } else {
          s0v = sc[wr]; s1v = sc[wr + 8];
        }
        const float* sx = sxs + (slot * 2 + half) * NT;
#pragma unroll
        for (int nb = 0; nb < NT / 8; ++nb)
#pragma unroll
          for (int v = 0; v < 2; ++v) {
            acc[nb * 4 + v] = fmaf(s0v, aq[nb][v] + aq2[nb][v], acc[nb * 4 + v]);
            acc[nb * 4 + 2 + v] = fmaf(s1v, aq[nb][2 + v] + aq2[nb][2 + v], acc[nb * 4 + 2 + v]);
            if (ASYM) {
              const float xs = sx[nb * 8 + 2 * t + v];
              acc[nb * 4 + v] = fmaf(m0v, xs, acc[nb * 4 + v]);
              acc[nb * 4 + 2 + v] = fmaf(m1v, xs, acc[nb * 4 + 2 + v]);
            }
          }
      }
      __syncwarp();
      if (lane == 0) mbar_arrive(empty0 + 8 * slot);
    }
    ibase += uend - u;
    const bool whole = (s0 == 0) && (uend == (tile + 1) * S);
    bool emit = whole;
    if (!whole) {
      const int ufirst = tile * S, ulast = ufirst + S - 1;
      const int pfirst = (int)((((int64_t)ufirst + 1) * P - 1) / U), plast = (int)((((int64_t)ulast + 1) * P - 1) / U);
      const int seg = p - pfirst, nseg = plast - pfirst + 1;
      float* wt = ws + ((int64_t)tile * maxseg) * RB * NT;
      if (seg > 0) {
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
          for (int q = 0; q < NT / 8; ++q)
            __stcg(reinterpret_cast<float2*>(wt + ((int64_t)seg * RB + wr + 8 * h) * NT + q * 8 + 2 * t),
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
        for (int s2 = 1; s2 < nseg; s2 += 4) {
          float2 v[4][2][NT / 8];
#pragma unroll
          for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
              for (int q = 0; q < NT / 8; ++q)
                v[j][h][q] = s2 + j < nseg ? __ldcg(reinterpret_cast<const float2*>(
                                                 wt + ((int64_t)(s2 + j) * RB + wr + 8 * h) * NT + q * 8 + 2 * t))
                                           : make_float2(0.f, 0.f);
#pragma unroll
          for (int j = 0; j < 4; ++j)
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
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int n = tile * RB + wr + 8 * h;
#pragma unroll
        for (int q = 0; q < NT / 8; ++q)
#pragma unroll
          for (int v = 0; v < 2; ++v) {
            const int m = q * 8 + 2 * t + v;
            if (m < M && n < N) out[(int64_t)m * ldo + n] = __float2bfloat16(acc[q * 4 + 2 * h + v]);
          }
      }
      // tree-verify row cut: rows M..R-1 of this tile's columns are zeros
      for (int idx = threadIdx.x; idx < (R - M) * RB; idx += NCW * 32) {
        const int m = M + idx / RB, n = tile * RB + idx % RB;
        if (n < N) out[(int64_t)m * ldo + n] = __float2bfloat16(0.f);
      }
    }
    u = uend;
  }
}
""".replace("_WREC_", str(_WREC))

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

template <int NT, int ASYM>
void launch(torch::Tensor x, int64_t M, torch::Tensor recs, int64_t S, int64_t rec_bytes, torch::Tensor out,
            torch::Tensor ws, torch::Tensor cnt, int64_t P, int64_t maxseg, int64_t N, int64_t pdl) {
  constexpr int NS = 5;
  const int K = x.size(1), R = out.size(0);
  const int T = recs.size(0), U = T * (int)S;
  CUtensorMap map;
  cuuint64_t gdim[2] = {(cuuint64_t)K, (cuuint64_t)x.size(0)};
  cuuint64_t gstr[1] = {(cuuint64_t)x.stride(0) * 2};
  cuuint32_t box[2] = {64, NT};
  cuuint32_t es[2] = {1, 1};
  CUresult r = get_encode()(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, x.data_ptr(), gdim, gstr, box, es,
                            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "W4S tensor map encode failed ", (int)r);
  constexpr int SB = 18432 + NT * 512;
  const int smem = NS * SB + NS * 2 * NT * 4 + 1024;
  auto kern = w4s_kernel<NT, ASYM, NS>;
  static bool attr = false;
  if (!attr) { cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); attr = true; }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((unsigned)P);
  cfg.blockDim = dim3(320);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = c10::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
  cfg.attrs = at;
  cfg.numAttrs = 1;
  cudaError_t e = cudaLaunchKernelEx(&cfg, kern, map, (int)M, R,
      reinterpret_cast<const unsigned char*>(recs.data_ptr()), (int64_t)recs.stride(0), (int)S, U, (int)rec_bytes,
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), (int64_t)out.stride(0), (int)N,
      ws.data_ptr<float>(), (int)maxseg, cnt.data_ptr<int>());
  TORCH_CHECK(e == cudaSuccess, "W4S launch failed: ", cudaGetErrorString(e));
}

void w4s_gemm(torch::Tensor x, int64_t M, torch::Tensor recs, int64_t S, int64_t rec_bytes, torch::Tensor out,
              torch::Tensor ws, torch::Tensor cnt, int64_t P, int64_t maxseg, int64_t N, int64_t asym, int64_t nt,
              int64_t pdl) {
  if (nt == 16 && asym) launch<16, 1>(x, M, recs, S, rec_bytes, out, ws, cnt, P, maxseg, N, pdl);
  else if (nt == 16) launch<16, 0>(x, M, recs, S, rec_bytes, out, ws, cnt, P, maxseg, N, pdl);
  else if (nt == 32 && asym) launch<32, 1>(x, M, recs, S, rec_bytes, out, ws, cnt, P, maxseg, N, pdl);
  else if (nt == 32) launch<32, 0>(x, M, recs, S, rec_bytes, out, ws, cnt, P, maxseg, N, pdl);
  else TORCH_CHECK(false, "W4S: bad nt");
}
"""

_SRC = _KERN + _HOST
_CPP = ("void w4s_gemm(torch::Tensor x, int64_t M, torch::Tensor recs, int64_t S, int64_t rec_bytes, torch::Tensor out, "
        "torch::Tensor ws, torch::Tensor cnt, int64_t P, int64_t maxseg, int64_t N, int64_t asym, int64_t nt, int64_t pdl);")
_CFLAGS = ["-O3", "-gencode=arch=compute_90a,code=sm_90a", "--use_fast_math"]
_mod = None


def _ext():
    global _mod
    if _mod is None:
        from torch.utils.cpp_extension import load_inline

        src = _SRC
        base = os.environ.get("SGLANG_CACHE_DIR") or os.path.join(os.path.expanduser("~"), ".cache", "sglang")
        key = hashlib.sha1("\0".join([_CPP, src, *_CFLAGS]).encode()).hexdigest()[:16]
        bdir = os.path.join(base, "specpilot_d5_w4s_src", key)
        os.makedirs(bdir, exist_ok=True)
        _mod = load_inline(name="specpilot_d5_w4s_v1", cpp_sources=_CPP, cuda_sources=src, functions=["w4s_gemm"],
                           extra_cuda_cflags=_CFLAGS, build_directory=bdir, verbose=False)
    return _mod


# ------------------------------------------------------------------------------------------------------- records


def _q_full(packed: torch.Tensor) -> torch.Tensor:
    """d7 packing (256-channel tiles, byte j = (q[k0 + j], q[k0 + 128 + j])) -> q uint8 [N, K] in 0..15."""
    n, kh = packed.shape
    p = packed.view(n, kh // 128, 128)
    return torch.stack([p & 0xF, p >> 4], dim=2).reshape(n, kh * 2)


def build_records(packed, scale, minv8, asym: bool, chunk_tiles: int = 16):
    """-> recs uint8 [T, S * rec_bytes] (rows padded to 128 with q = 8, s = m8 = 0), rec_bytes."""
    n, kh = packed.shape
    k = kh * 2
    assert k % _STK == 0
    S = k // _STK
    T = (n + _RB - 1) // _RB
    rec_bytes = _WREC + (2048 if asym else 512)
    dev = packed.device
    recs = torch.empty((T, S * rec_bytes), dtype=torch.uint8, device=dev)
    # per lane (g, t), k16 step j, reg r, element e: row = w*16 + g + 8 (r & 1), k = j*16 + 2t + 8 (r >> 1) + e
    for t0 in range(0, T, chunk_tiles):
        t1 = min(T, t0 + chunk_tiles)
        r0, r1 = t0 * _RB, min(n, t1 * _RB)
        q = torch.full(((t1 - t0) * _RB, k), 8, dtype=torch.uint8, device=dev)
        q[: r1 - r0] = _q_full(packed[r0:r1])
        # q[tile, w, gg, rr(r&1), stage, half, j(8), r>>1, tt, e] with k = stage*256 + half*128 + j*16 + (r>>1)*8 + 2tt + e
        qv = q.view(t1 - t0, 8, 2, 8, S, 2, 8, 2, 4, 2)
        #            tile  w  rr  gg  st  hf  j  rh  tt  e
        # word j of lane (gg, tt): bytes b(e) = (lo r0.e, hi r2.e) = (rr=0,rh=0 | rr=0,rh=1), b(2+e) = (rr=1,rh=0 | rr=1,rh=1)
        lo = qv[:, :, :, :, :, :, :, 0]  # rh = 0 -> regs r0 (rr=0), r1 (rr=1)
        hi = qv[:, :, :, :, :, :, :, 1]  # rh = 1 -> regs r2, r3
        byte = lo | (hi << 4)            # [tile, w, rr, gg, st, hf, j, tt, e]
        # byte index within word = rr*2 + e; word = j; lane = gg*4 + tt
        byte = byte.permute(0, 4, 5, 1, 3, 7, 6, 2, 8)  # [tile, st, hf, w, gg, tt, j, rr, e]
        byte = byte.reshape(t1 - t0, S, 2, 8, 32, 8, 4)  # lane = gg*4+tt, j, byte
        # [half][warp][hh][lane][16 B]: hh = j // 4, 16 B = words j%4 (4 B each)
        byte = byte.view(t1 - t0, S, 2, 8, 32, 2, 16).permute(0, 1, 2, 3, 5, 4, 6).contiguous()
        w = byte.view(t1 - t0, S, _WREC)
        # scales [half][s | m8][128] (asym) or [s][128] (sym, the 256-channel group's scale)
        if asym:
            sc = torch.zeros((t1 - t0) * _RB, 2 * S, 2, dtype=torch.float32, device=dev)  # [row, group, s|m8]
            sc[: r1 - r0, :, 0] = scale[:, r0:r1].t()
            sc[: r1 - r0, :, 1] = minv8[:, r0:r1].t()
            sc = sc.view(t1 - t0, _RB, S, 2, 2).permute(0, 2, 3, 4, 1).contiguous()  # [tile, st, half, s|m8, row]
            sb = sc.view(torch.uint8).view(t1 - t0, S, 2048)
        else:
            sc = torch.zeros((t1 - t0) * _RB, S, dtype=torch.float32, device=dev)
            sc[: r1 - r0] = scale[:, r0:r1].t()
            sc = sc.view(t1 - t0, _RB, S).permute(0, 2, 1).contiguous()  # [tile, st, row]
            sb = sc.view(torch.uint8).view(t1 - t0, S, 512)
        recs[t0:t1] = torch.cat([w, sb], dim=2).view(t1 - t0, S * rec_bytes)
        del q, qv, byte, w, sc, sb
    return recs, rec_bytes


def _maxseg(T, S, P):
    U = T * S
    own = lambda u: ((u + 1) * P - 1) // U
    return max(own(t * S + S - 1) - own(t * S) + 1 for t in range(T))


_SHARED: dict = {}


def _ensure_shared(dev, ws_floats, ntiles):
    sh = _SHARED.setdefault(dev, {})
    if sh.get("ws") is None or sh["ws"].numel() < ws_floats:
        sh["ws"] = torch.empty(ws_floats, dtype=torch.float32, device=dev)
    if sh.get("cnt") is None or sh["cnt"].numel() < ntiles:
        sh["cnt"] = torch.zeros(ntiles, dtype=torch.int32, device=dev)
    return sh


class W4SLinear:
    """Stream-K twin of a d7 W4Linear (same q / s / m8). Small-M path: bs = 1 row cut or plain rows <= 32; every
    other call (and attribute) goes to the d7 object."""

    def __init__(self, lin, pdl: int = 1):
        self.lin = lin
        self.n, self.k = lin.n, lin.k
        self.asym = bool(lin.asym)
        if self.asym:
            assert lin.group == 128
        else:
            assert lin.group == 256
        self.recs, self.rec_bytes = build_records(lin.packed, lin.scale, lin.minv8 if self.asym else None, self.asym)
        self.T = self.recs.shape[0]
        self.S = self.k // _STK
        nsm = torch.cuda.get_device_properties(self.recs.device).multi_processor_count
        U = self.T * self.S
        self.P = min(nsm, U)
        self.maxseg = _maxseg(self.T, self.S, self.P)
        self.pdl = pdl
        _ensure_shared(self.recs.device, self.T * self.maxseg * _RB * 32, self.T)

    def __getattr__(self, name):  # packed / scale / group / minv / weight ... of the d7 object
        return getattr(self.__dict__["lin"], name)

    def __call__(self, x2: torch.Tensor, row_cut=None) -> torch.Tensor:
        rows_total = x2.shape[0]
        if row_cut is not None:
            nr, br = row_cut
            if rows_total != nr:  # bs > 1: rows not contiguous
                return self.lin(x2, row_cut)
            m = br
        else:
            m = rows_total
        if m > 32 or x2.dtype != torch.bfloat16 or x2.stride(-1) != 1 or x2.stride(0) % 8 or x2.data_ptr() % 16:
            return self.lin(x2, row_cut)
        nt = 16 if m <= 16 else 32
        out = torch.empty((rows_total, self.n), device=x2.device, dtype=torch.bfloat16)
        sh = _SHARED[x2.device]
        _ext().w4s_gemm(x2, m, self.recs, self.S, self.rec_bytes, out, sh["ws"], sh["cnt"], self.P, self.maxseg,
                        self.n, int(self.asym), nt, self.pdl)
        return out
