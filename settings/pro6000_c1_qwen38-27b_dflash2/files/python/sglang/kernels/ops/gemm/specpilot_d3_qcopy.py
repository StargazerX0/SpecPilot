"""SPECPILOT_D3: separately stored, scaled low-bit Q copy of a target linear for TLV's Q pass (sm_120).

Format: symmetric INT6 or INT8, one bf16 scale per (row, 128 input channels), round-to-nearest at the per-group
MSE-best clip of _GRID (no data). u = q + 2^(b-1) is split into a nibble plane (low 4 bits) and a high plane (2 bits
for INT6, a second nibble plane for INT8). Records per (128-row tile, 128-channel stage): nibble plane
[warp 8][hh 2][lane 32][16 B] in mma-A-fragment order (the d5 W4S layout), high plane [warp][hh][lane][8 B | 16 B],
scales bf16 [128]. Kernel: persistent stream-K over (tile, stage) units, 1 producer warp (1-D bulk copy of the record +
two 128B-swizzled 2-D TMA boxes of x per stage), 8 consumer warps decoding in registers into bf16 A fragments
(INT6: bf16(128 + u) by lop3 then one bf16x2 fma; INT8: two nibble decodes + 3 fma, exact), mma.sync m16n8k16
swap-AB, per-group fold acc += s * dot, deterministic split-tile reduction by the k = 0 owner. M <= 32 rows.
"""
from __future__ import annotations

import hashlib
import os

import torch

_RB = 128
_STK = 128
_GRID = (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7)
_NS = {6: 4, 8: 3}
# SPECPILOT_K_D3Q_TRIG: 1 = D3Q triggers its PDL dependents at start (after the mbarrier init), like W12; read once
_K_TRIG = int(os.environ.get("SPECPILOT_K_D3Q_TRIG", "0") or 0)

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
  asm volatile("{\n .reg .pred p;\n D3W_%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra D3W_%=;\n}\n"
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
__device__ __forceinline__ uint32_t nib(uint32_t t) {  // bf16x2 (128 + (t & 15)) per half
  uint32_t a;
  asm("lop3.b32 %0, %1, 0x000F000F, 0x43004300, 0xEA;" : "=r"(a) : "r"(t));
  return a;
}
__device__ __forceinline__ uint32_t orhi(uint32_t a, uint32_t h) {  // a | (h & 0x00300030)
  uint32_t r;
  asm("lop3.b32 %0, %1, %2, 0x00300030, 0xF8;" : "=r"(r) : "r"(a), "r"(h));
  return r;
}
__device__ __forceinline__ uint32_t bfma(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;
  asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}

// one lane's quarter (64 channels = 4 k16 steps): nibble words c, high words h -> A[4][4] bf16x2 q (exact)
template <int HB>
__device__ __forceinline__ void q_decode(const uint4 c, const uint4 h, uint32_t (&A)[4][4]) {
  const uint32_t W[4] = {c.x, c.y, c.z, c.w};
  const uint32_t H[4] = {h.x, h.y, h.z, h.w};
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const uint32_t t0 = __byte_perm(W[j], 0, 0x4140), t1 = __byte_perm(W[j], 0, 0x4342);
    uint32_t a0 = nib(t0), a1 = nib(t1), a2 = nib(t0 >> 4), a3 = nib(t1 >> 4);
    if (HB == 2) {
      const uint32_t hw = H[j >> 1] >> (2 * (j & 1));
      const uint32_t h0 = __byte_perm(hw, 0, 0x4140), h1 = __byte_perm(hw, 0, 0x4342);
      a0 = orhi(a0, h0 << 4); a1 = orhi(a1, h1 << 4); a2 = orhi(a2, h0); a3 = orhi(a3, h1);
      // bf16x2 (128 + u) - 160 = u - 32 = q
      A[j][0] = bfma(a0, 0x3F803F80u, 0xC320C320u); A[j][1] = bfma(a1, 0x3F803F80u, 0xC320C320u);
      A[j][2] = bfma(a2, 0x3F803F80u, 0xC320C320u); A[j][3] = bfma(a3, 0x3F803F80u, 0xC320C320u);
    } else {
      const uint32_t s0 = __byte_perm(H[j], 0, 0x4140), s1 = __byte_perm(H[j], 0, 0x4342);
      const uint32_t b0 = nib(s0), b1 = nib(s1), b2 = nib(s0 >> 4), b3 = nib(s1 >> 4);
      // q = 16 (hi - 8) + lo: hi - 8 = (128 + hi) - 136, lo = (128 + lo) - 128, all exact in bf16
      A[j][0] = bfma(bfma(b0, 0x3F803F80u, 0xC308C308u), 0x41804180u, bfma(a0, 0x3F803F80u, 0xC300C300u));
      A[j][1] = bfma(bfma(b1, 0x3F803F80u, 0xC308C308u), 0x41804180u, bfma(a1, 0x3F803F80u, 0xC300C300u));
      A[j][2] = bfma(bfma(b2, 0x3F803F80u, 0xC308C308u), 0x41804180u, bfma(a2, 0x3F803F80u, 0xC300C300u));
      A[j][3] = bfma(bfma(b3, 0x3F803F80u, 0xC308C308u), 0x41804180u, bfma(a3, 0x3F803F80u, 0xC300C300u));
    }
  }
}

template <int NT, int HB, int NS, int SILU>
__global__ void __launch_bounds__(288, 1) d3q_kernel(
    const __grid_constant__ CUtensorMap xmap, int M, int R,
    const unsigned char* __restrict__ recs, int64_t tstride, int S, int U, int rec_bytes, int xoff,
    __nv_bfloat16* __restrict__ out, int64_t ldo, int N,
    float* __restrict__ ws, int maxseg, int* __restrict__ counters, int trig) {
  constexpr int NCW = 8, RB = 128;
  constexpr int XB = NT * 256;               // x stage: 2 boxes of NT rows x 64 channels (128-B rows)
  constexpr int NREG = NT / 2;
  constexpr int HOFF = 8192;                 // high plane offset in the record
  constexpr int HQ = HB == 2 ? 8 : 16;       // high-plane bytes per lane and quarter
  constexpr int SOFF = 8192 + HQ * 512;      // bf16 scales [128]
  extern __shared__ __align__(1024) unsigned char smem[];
  __shared__ __align__(8) uint64_t bars[2 * NS];
  const int SB = xoff + XB;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int P = gridDim.x, p = blockIdx.x;
  const int u0 = (int)(((int64_t)p * U) / P), u1 = (int)(((int64_t)(p + 1) * U) / P);
  const uint32_t sraw = (uint32_t)__cvta_generic_to_shared(smem);
  const uint32_t sbase = (sraw + 1023u) & ~1023u;
  unsigned char* const smem_al = smem + (sbase - sraw);
  const uint32_t full0 = (uint32_t)__cvta_generic_to_shared(bars), empty0 = full0 + 8 * NS;
  if (threadIdx.x == 0) {
    for (int s = 0; s < NS; ++s) { mbar_init(full0 + 8 * s, 1); mbar_init(empty0 + 8 * s, NCW); }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();
  // SPECPILOT_K_D3Q_TRIG: let the dependents (x3 add+RMSNorm, the next D3Q, conv / qk-norm) launch now, as W12 does;
  // each of them waits on griddepcontrol.wait before reading this kernel's output.
  if (trig) asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
  if (u0 >= u1) return;
  if (warp == NCW) {  // producer
    if (lane == 0) {
      const int npre = min(NS, u1 - u0);
      for (int i = 0; i < npre; ++i) {  // weight records do not depend on the predecessor (PDL)
        const int u = u0 + i, tile = u / S, s = u - tile * S;
        const uint32_t fb = full0 + 8 * i;
        mbar_expect_tx(fb, rec_bytes + XB);
        bulk_g2s(sbase + i * SB, recs + (int64_t)tile * tstride + (int64_t)s * rec_bytes, rec_bytes, fb);
      }
      asm volatile("griddepcontrol.wait;\n" ::: "memory");
      for (int i = 0; i < npre; ++i) {
        const int u = u0 + i, tile = u / S, s = u - tile * S;
#pragma unroll
        for (int q = 0; q < 2; ++q) tma2d(sbase + i * SB + xoff + q * NT * 128, &xmap, s * 128 + q * 64, 0, full0 + 8 * i);
      }
      for (int u = u0 + npre; u < u1; ++u) {
        const int i = u - u0, slot = i % NS, tile = u / S, s = u - tile * S;
        mbar_wait(empty0 + 8 * slot, ((i / NS) - 1) & 1);
        const uint32_t fb = full0 + 8 * slot, dst = sbase + slot * SB;
        mbar_expect_tx(fb, rec_bytes + XB);
        bulk_g2s(dst, recs + (int64_t)tile * tstride + (int64_t)s * rec_bytes, rec_bytes, fb);
#pragma unroll
        for (int q = 0; q < 2; ++q) tma2d(dst + xoff + q * NT * 128, &xmap, s * 128 + q * 64, 0, fb);
      }
    }
    return;
  }
  const int g = lane / 4, t = lane % 4;
  const int wr = warp * 16 + g;
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
      const unsigned char* rb = smem_al + slot * SB;
      const uint4* rw = reinterpret_cast<const uint4*>(rb);
      const __nv_bfloat16* sc = reinterpret_cast<const __nv_bfloat16*>(rb + SOFF);
      float aq[NT / 8][4], aq2[NT / 8][4];
#pragma unroll
      for (int nb = 0; nb < NT / 8; ++nb)
#pragma unroll
        for (int c = 0; c < 4; ++c) aq[nb][c] = aq2[nb][c] = 0.f;
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        uint32_t A[4][4];
        const int li = (warp * 2 + hh) * 32 + lane;
        uint4 hv;
        if (HB == 2) {
          const uint2 h2 = reinterpret_cast<const uint2*>(rb + HOFF)[li];
          hv = make_uint4(h2.x, h2.y, 0u, 0u);
        } else {
          hv = reinterpret_cast<const uint4*>(rb + HOFF)[li];
        }
        q_decode<HB>(rw[li], hv, A);
        const uint32_t xb = sbase + slot * SB + xoff + hh * NT * 128;
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
      const float s0v = __bfloat162float(sc[wr]), s1v = __bfloat162float(sc[wr + 8]);
#pragma unroll
      for (int nb = 0; nb < NT / 8; ++nb)
#pragma unroll
        for (int v = 0; v < 2; ++v) {
          acc[nb * 4 + v] = fmaf(s0v, aq[nb][v] + aq2[nb][v], acc[nb * 4 + v]);
          acc[nb * 4 + 2 + v] = fmaf(s1v, aq[nb][2 + v] + aq2[nb][2 + v], acc[nb * 4 + 2 + v]);
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
        for (int s2 = 1; s2 < nseg; ++s2) {
#pragma unroll
          for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int q = 0; q < NT / 8; ++q) {
              const float2 v = __ldcg(reinterpret_cast<const float2*>(wt + ((int64_t)s2 * RB + wr + 8 * h) * NT + q * 8 + 2 * t));
              acc[q * 4 + 2 * h] += v.x;
              acc[q * 4 + 2 * h + 1] += v.y;
            }
        }
        emit = true;
      }
    }
    if (emit && SILU) {
      // tile rows [64 gate | 64 up] (W12's interleave): up warps (4..7) hand their rows to the gate warps (0..3)
      float* xu = reinterpret_cast<float*>(smem_al + NS * SB);  // [64 rows][NT]
      if (warp >= 4) {
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
          for (int q = 0; q < NT / 8; ++q)
            *reinterpret_cast<float2*>(xu + (wr - 64 + 8 * h) * NT + q * 8 + 2 * t) =
                make_float2(acc[q * 4 + 2 * h], acc[q * 4 + 2 * h + 1]);
      }
      cons_bar(NCW * 32);
      if (warp < 4) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int n = tile * 64 + wr + 8 * h;
#pragma unroll
          for (int q = 0; q < NT / 8; ++q) {
            const float2 up = *reinterpret_cast<const float2*>(xu + (wr + 8 * h) * NT + q * 8 + 2 * t);
#pragma unroll
            for (int v = 0; v < 2; ++v) {
              const int m = q * 8 + 2 * t + v;
              const float gv = acc[q * 4 + 2 * h + v];
              const float y = gv / (1.f + __expf(-gv)) * (v ? up.y : up.x);
              if (m < M && n < N) out[(int64_t)m * ldo + n] = __float2bfloat16(y);
            }
          }
        }
      }
      for (int idx = threadIdx.x; idx < (R - M) * 64; idx += NCW * 32) {
        const int m = M + idx / 64, n = tile * 64 + idx % 64;
        if (n < N) out[(int64_t)m * ldo + n] = __float2bfloat16(0.f);
      }
      cons_bar(NCW * 32);
    } else if (emit) {
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
      for (int idx = threadIdx.x; idx < (R - M) * RB; idx += NCW * 32) {
        const int m = M + idx / RB, n = tile * RB + idx % RB;
        if (n < N) out[(int64_t)m * ldo + n] = __float2bfloat16(0.f);
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

template <int NT, int HB, int NS, int SILU>
void launch(torch::Tensor x, int64_t M, torch::Tensor recs, int64_t S, int64_t rec_bytes, int64_t xoff, torch::Tensor out,
            torch::Tensor ws, torch::Tensor cnt, int64_t P, int64_t maxseg, int64_t N, int64_t pdl) {
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
  TORCH_CHECK(r == CUDA_SUCCESS, "D3Q tensor map encode failed ", (int)r);
  const int SB = (int)xoff + NT * 256;
  const int smem = NS * SB + 1024 + (SILU ? 64 * NT * 4 : 0);
  auto kern = d3q_kernel<NT, HB, NS, SILU>;
  static int attr = 0;
  if (attr < smem) {
    cudaError_t ea = cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    TORCH_CHECK(ea == cudaSuccess, "D3Q smem attr failed: ", cudaGetErrorString(ea), " smem=", smem);
    attr = smem;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((unsigned)P);
  cfg.blockDim = dim3(288);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = c10::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = (pdl & 1) ? 1 : 0;  // SPECPILOT_K_D3Q_TRIG: bit 1 = trigger
  cfg.attrs = at;
  cfg.numAttrs = 1;
  cudaError_t e = cudaLaunchKernelEx(&cfg, kern, map, (int)M, R,
      reinterpret_cast<const unsigned char*>(recs.data_ptr()), (int64_t)recs.stride(0), (int)S, U, (int)rec_bytes, (int)xoff,
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), (int64_t)out.stride(0), (int)N,
      ws.data_ptr<float>(), (int)maxseg, cnt.data_ptr<int>(), (int)((pdl >> 1) & 1));
  TORCH_CHECK(e == cudaSuccess, "D3Q launch failed: ", cudaGetErrorString(e));
}

void d3q_gemm(torch::Tensor x, int64_t M, torch::Tensor recs, int64_t S, int64_t rec_bytes, int64_t xoff, torch::Tensor out,
              torch::Tensor ws, torch::Tensor cnt, int64_t P, int64_t maxseg, int64_t N, int64_t bits, int64_t nt,
              int64_t pdl, int64_t silu) {
#define D3Q_L(NT_, HB_, NS_, SI_) launch<NT_, HB_, NS_, SI_>(x, M, recs, S, rec_bytes, xoff, out, ws, cnt, P, maxseg, N, pdl)
  const int key = (int)(nt == 32) * 4 + (int)(bits == 8) * 2 + (int)(silu != 0);
  switch (key) {
    case 0: D3Q_L(16, 2, NS6, 0); break;
    case 1: D3Q_L(16, 2, NS6, 1); break;
    case 2: D3Q_L(16, 4, NS8, 0); break;
    case 3: D3Q_L(16, 4, NS8, 1); break;
    case 4: D3Q_L(32, 2, NS6, 0); break;
    case 5: D3Q_L(32, 2, NS6, 1); break;
    case 6: D3Q_L(32, 4, NS8, 0); break;
    case 7: D3Q_L(32, 4, NS8, 1); break;
  }
#undef D3Q_L
}
"""

_CPP = ("void d3q_gemm(torch::Tensor x, int64_t M, torch::Tensor recs, int64_t S, int64_t rec_bytes, int64_t xoff, "
        "torch::Tensor out, torch::Tensor ws, torch::Tensor cnt, int64_t P, int64_t maxseg, int64_t N, int64_t bits, "
        "int64_t nt, int64_t pdl, int64_t silu);")
_CFLAGS = ["-O3", "-gencode=arch=compute_120a,code=sm_120a", "--use_fast_math"]
_mod = None


def _ext():
    global _mod
    if _mod is None:
        from torch.utils.cpp_extension import load_inline

        src = _KERN + _HOST.replace("NS6", str(_NS[6])).replace("NS8", str(_NS[8]))
        base = os.environ.get("SGLANG_CACHE_DIR") or os.path.join(os.path.expanduser("~"), ".cache", "sglang")
        key = hashlib.sha1("\0".join([_CPP, src, *_CFLAGS]).encode()).hexdigest()[:16]
        bdir = os.path.join(base, "specpilot_d3_qcopy_src", key)
        os.makedirs(bdir, exist_ok=True)
        _mod = load_inline(name="specpilot_d3_qcopy_" + key, cpp_sources=_CPP, cuda_sources=src, functions=["d3q_gemm"],
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
    n, k = u.shape
    q = u.float() - 2 ** (bits - 1)
    return (q.view(n, k // _STK, _STK) * sc.float().unsqueeze(-1)).view(n, k)


def _lane_order(v: torch.Tensor, T: int, S: int) -> torch.Tensor:
    """v [T*128, S*128] (one value per weight) -> [T, S, warp 8, hh 2, lane 32, j 4, byte 4, nib 2] with
    row = w*16 + g + 8 rr, k = s*128 + hh*64 + j*16 + 2t + 8 rh + e; byte = rr*2 + e, nib = rh (lo/hi)."""
    x = v.view(T, 8, 2, 8, S, 2, 4, 2, 4, 2)
    #          T  w rr  g  s hh  j rh  t  e
    x = x.permute(0, 4, 1, 5, 3, 8, 6, 2, 9, 7)  # T s w hh g t j rr e rh
    return x.reshape(T, S, 8, 2, 32, 4, 4, 2)


def build_records(u: torch.Tensor, sc: torch.Tensor, bits: int, chunk_tiles: int = 16):
    """-> (recs uint8 [T, S * rec_bytes], rec_bytes, xoff). Padding rows: u = 2^(bits-1) (q = 0), s = 0."""
    n, k = u.shape
    S = k // _STK
    T = (n + _RB - 1) // _RB
    hq = 8 if bits == 6 else 16
    rec_bytes = 8192 + hq * 512 + 256
    xoff = (rec_bytes + 1023) // 1024 * 1024
    recs = torch.empty((T, S * rec_bytes), dtype=torch.uint8, device=u.device)
    mid = 2 ** (bits - 1)
    for t0 in range(0, T, chunk_tiles):
        t1 = min(T, t0 + chunk_tiles)
        r0, r1 = t0 * _RB, min(n, t1 * _RB)
        uu = torch.full(((t1 - t0) * _RB, k), mid, dtype=torch.uint8, device=u.device)
        uu[: r1 - r0] = u[r0:r1]
        L = _lane_order(uu, t1 - t0, S)  # [.., lane, j, byte, nib]
        lo = L & 0xF
        nibw = (lo[..., 0] | (lo[..., 1] << 4))  # [T, S, w, hh, lane, j, byte]
        hi = L >> 4
        if bits == 6:
            # high word jh = j >> 1, shift 2 (j & 1): byte bits [0..1] = lo-nib value, [4..5] = hi-nib value
            h = hi.view(t1 - t0, S, 8, 2, 32, 2, 2, 4, 2)  # j -> (jh, jl)
            hb = (h[..., 0, :, 0] | (h[..., 0, :, 1] << 4) | (h[..., 1, :, 0] << 2) | (h[..., 1, :, 1] << 6))
            hplane = hb.reshape(t1 - t0, S, 8 * 2 * 32 * 8)
        else:
            hplane = (hi[..., 0] | (hi[..., 1] << 4)).reshape(t1 - t0, S, 8 * 2 * 32 * 16)
        scs = torch.zeros(((t1 - t0) * _RB, S), dtype=torch.bfloat16, device=u.device)
        scs[: r1 - r0] = sc[r0:r1]
        sb = scs.view(t1 - t0, _RB, S).permute(0, 2, 1).contiguous().view(torch.uint8).view(t1 - t0, S, 256)
        rec = torch.cat([nibw.reshape(t1 - t0, S, 8192).to(torch.uint8), hplane.to(torch.uint8), sb], dim=2)
        recs[t0:t1] = rec.view(t1 - t0, S * rec_bytes)
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


class D3QLinear:
    """Scaled INT6 / INT8 Q copy of a bf16 [N, K] weight; __call__(x2 [M <= 32, K] bf16) -> [M, N] bf16
    (silu: [M, N/2] = SiLU(gate) * up of the [gate | up] weight, fused in the epilogue)."""

    def __init__(self, w: torch.Tensor, bits: int, silu: bool = False, pdl: int = 1, nsm: int | None = None,
                 trig: int | None = None):
        assert bits in (6, 8) and w.dim() == 2 and w.shape[1] % _STK == 0
        self.bits, self.n, self.k, self.silu = bits, w.shape[0], w.shape[1], int(silu)
        if silu:  # [gate | up] -> per 128-row tile [64 gate | 64 up] (as W12); the kernel emits SiLU(gate) * up
            assert self.n % 128 == 0
            w = w.view(2, self.n // 128, 64, self.k).transpose(0, 1).reshape(self.n, self.k)
        self.nout = self.n // 2 if silu else self.n
        u, sc = quantize_q(w, bits)
        self.recs, self.rec_bytes, self.xoff = build_records(u, sc, bits)
        del u, sc
        self.T, self.S = self.recs.shape[0], self.k // _STK
        nsm = nsm or torch.cuda.get_device_properties(w.device).multi_processor_count
        self.P = min(nsm, self.T * self.S)
        self.maxseg = _maxseg(self.T, self.S, self.P)
        trig = _K_TRIG if trig is None else trig  # SPECPILOT_K_D3Q_TRIG (only with PDL on)
        self.pdl = (1 if pdl else 0) | (2 if pdl and trig else 0)
        _ensure_shared(w.device, self.T * self.maxseg * _RB * 32, self.T)

    def nbytes(self) -> int:
        return self.recs.numel()

    def __call__(self, x2: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        m = x2.shape[0]
        assert 0 < m <= 32 and x2.dtype == torch.bfloat16 and x2.stride(-1) == 1 and x2.stride(0) % 8 == 0
        nt = 16 if m <= 16 else 32
        if out is None:
            out = torch.empty((m, self.nout), device=x2.device, dtype=torch.bfloat16)
        sh = _SHARED[x2.device]
        _ext().d3q_gemm(x2, m, self.recs, self.S, self.rec_bytes, self.xoff, out, sh["ws"], sh["cnt"], self.P,
                        self.maxseg, self.nout, self.bits, nt, self.pdl, self.silu)
        return out
