"""SPECPILOT_D3W_GDNF (N9b kernel, N9d consumer pipeline): persistent warp-specialized CUDA lazy GDN tree verify for trees of <= 16 nodes (K = V = 128).

Per layer, apply the previous step's accepted path to the fp32 state in closed form (seed SPECPILOT_K_WY_LAZY math,
h' = e^{gam_l} h + sum_{s on path} e^{gam_l - gam_s} v'_s kn_s^T), write h' back once, verify the new tree from h'.

Persistent CTAs (grid <= #SMs, 1 per SM), work item = (request, value head) = one 128 x 128 fp32 state tile, 2-stage
smem ring with mbarriers:
  producer warp:  slot records of all its items prefetched once; per item TMA bulk copies (cp.async.bulk +
                  complete_tx) of the state tile (4 x 16 KB), the 16 q / k rows and the pending commit's on-path
                  kn / v' rows; e^{gam_l - gam_s} weights computed by its lanes.
  2 prep warps:   (prep warp p serves stage p) g / beta / gam / norms / Gram (mma) / X, P /
                  M' = (I-X)(I+X^2)(I+X^4)[(I+X^8) only for trees deeper than 7] diag(beta) / W = P M' (fp32) into the
                  stage's prep slot; kn / gam commit inputs for the next step -> global.
  8 consumer warps (16 state rows each): smem -> registers (mma B-fragment layout, k permuted inside each 16-k step),
                  pending commit on CUDA cores (fp32), h' -> stage tile -> one TMA bulk store, Kn h' / Qn h' with
                  mma.m16n8k16 on a 3-term bf16 split of h' (seed precision), release the stage, then the fp32
                  epilogue u = v - e^gam rk A, v' = M' u, o = e^gam rq scale QH + W u.
N9d: the consumers run item i - 1's epilogue after item i's commit + mma (software pipeline across items), so the
per-item v loads and the prep chain leave the critical path; the prep waits for its slot (pfree) only before writing
st.pre. Bitwise equal to the N9b order.
N9e: the state tile loads / h' bulk stores, the pending commit's kn / v' row copies and the v' buffer stores carry an
L2 evict_first cache hint (the 3 GB/step state stream no longer evicts the L2-resident q / k / v / o rows).
"""
from __future__ import annotations

import hashlib
import os

import torch

_CPP = (
    "void d3w_gdnf2(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor a, torch::Tensor b, "
    "torch::Tensor A_log, torch::Tensor dt_bias, torch::Tensor anc, torch::Tensor state, torch::Tensor cidx, "
    "torch::Tensor leaf, torch::Tensor mask, torch::Tensor parity, torch::Tensor kn_buf, torch::Tensor vp_buf, "
    "torch::Tensor gam_buf, torch::Tensor o, torch::Tensor out_rows, int64_t packed, int64_t N, int64_t grid, "
    "int64_t abl, double scale, double sp_beta, double sp_thresh);"
)

_CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace d3w2 {

constexpr int KD = 128;
constexpr int VD = 128;
constexpr int CW = 8;      // consumer warps (16 state rows each)
constexpr int NT = (CW + 3) * 32;  // + producer + 2 prep warps (prep warp p serves stage p)
constexpr int HLD = 128;   // fp32 row stride of the staged state tile (contiguous: whole-tile bulk copies)
constexpr int QLD = 144;   // bf16 row stride of the q / k tiles (288 B)
constexpr int PLD = 20;    // fp32 row stride of the 16 x 16 prep matrices

struct Params {
  const __nv_bfloat16* q; const __nv_bfloat16* k; const __nv_bfloat16* v;
  long long sq, sk, sv;
  const __nv_bfloat16* a; const __nv_bfloat16* b; long long sa, sb;
  const void* A_log; const void* dt_bias; int alog_bf16, dtb_bf16;
  const int* anc;
  float* state; long long s_slot;
  const int* cidx;
  const int* leaf; const int* mask; const int* parity; int n_slots;
  float* kn_buf; float* vp_buf; float* gam_buf;
  __nv_bfloat16* o; const long long* out_rows; int packed;
  float scale, sp_beta, sp_thresh;
  int N, H, HV, n_items;
  int abl;
};

struct __align__(16) Meta {
  long long wrow, rrow;
  int idx, leaf, np, onb;
  float gl;
  float gsc[16];
};

struct __align__(16) Prep {
  float M[16 * PLD], W[16 * PLD];
  float cA[16], cQ[16];
};

struct __align__(128) Stage {
  float h[128 * HLD];
  __nv_bfloat16 qs[16 * QLD];
  __nv_bfloat16 ks[16 * QLD];
  float knc[16 * KD];
  float vpc[16 * VD];
  Prep pre;
  Meta meta;
};

struct PrepScratch {
  float X[16 * PLD], X2[16 * PLD], X4[16 * PLD], X8[16 * PLD], R[16 * PLD], T[16 * PLD], P[16 * PLD];
  float gam[16], beta[16], rk[16], rqs[16];
  int bits[16];
};

struct Smem {
  Stage st[2];
  PrepScratch ps[2];
  float uS[CW][16 * 16];
  unsigned long long full[2], empty[2], prepd[2], pfree[2];
};

__device__ __forceinline__ float ldf(const void* p, int i, int is_bf16) {
  return is_bf16 ? __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(p)[i]) : reinterpret_cast<const float*>(p)[i];
}
__device__ __forceinline__ uint32_t sptr(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void mb_init(unsigned long long* b, int cnt) {
  asm volatile("mbarrier.init.shared.b64 [%0], %1;" ::"r"(sptr(b)), "r"(cnt) : "memory");
}
__device__ __forceinline__ void mb_arrive(unsigned long long* b) {
  asm volatile("mbarrier.arrive.shared.b64 _, [%0];" ::"r"(sptr(b)) : "memory");
}
__device__ __forceinline__ void mb_arrive_tx(unsigned long long* b, uint32_t tx) {
  asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;" ::"r"(sptr(b)), "r"(tx) : "memory");
}
__device__ __forceinline__ void mb_wait(unsigned long long* b, uint32_t parity) {
  asm volatile(
      "{\n.reg .pred p;\nWAIT_%=:\n"
      "mbarrier.try_wait.parity.shared.b64 p, [%0], %1;\n"
      "@!p bra WAIT_%=;\n}\n" ::"r"(sptr(b)), "r"(parity) : "memory");
}
__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, uint32_t bytes, unsigned long long* b) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               ::"r"(sptr(dst)), "l"(src), "r"(bytes), "r"(sptr(b)) : "memory");
}
__device__ __forceinline__ uint64_t pol_evict_first() {
  uint64_t pol;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
  return pol;
}
__device__ __forceinline__ void bulk_g2s_ef(void* dst, const void* src, uint32_t bytes, unsigned long long* b) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"
               ::"r"(sptr(dst)), "l"(src), "r"(bytes), "r"(sptr(b)), "l"(pol_evict_first()) : "memory");
}
__device__ __forceinline__ void bulk_s2g_ef(void* dst, const void* src, uint32_t bytes) {
  asm volatile("cp.async.bulk.global.shared::cta.bulk_group.L2::cache_hint [%0], [%1], %2, %3;"
               ::"l"(dst), "r"(sptr(src)), "r"(bytes), "l"(pol_evict_first()) : "memory");
  asm volatile("cp.async.bulk.commit_group;" ::: "memory");
}
__device__ __forceinline__ void st_f2_ef(float* dst, float x, float y, uint64_t pol) {
  asm volatile("st.global.L2::cache_hint.v2.f32 [%0], {%1, %2}, %3;" ::"l"(dst), "f"(x), "f"(y), "l"(pol) : "memory");
}
__device__ __forceinline__ void bulk_s2g(void* dst, const void* src, uint32_t bytes) {
  asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;" ::"l"(dst), "r"(sptr(src)), "r"(bytes) : "memory");
  asm volatile("cp.async.bulk.commit_group;" ::: "memory");
}
__device__ __forceinline__ void bulk_wait_read() { asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory"); }
__device__ __forceinline__ void bulk_wait_all() { asm volatile("cp.async.bulk.wait_group 0;" ::: "memory"); }
__device__ __forceinline__ void fence_async_smem() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
__device__ __forceinline__ void bar_sync(int id, int cnt) { asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(cnt) : "memory"); }
__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
  __nv_bfloat162 t = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&t);
}
__device__ __forceinline__ void mma16816(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
// C = A @ B (+ Cin), 16 x 16 fp32 in smem, one warp; lane: row t = lane & 15, cols (lane >> 4) * 8 .. +8.
__device__ __forceinline__ void mm16(const float* A, const float* B, const float* Cin, float sgn, float* C, int lane) {
  const int t = lane & 15, j0 = (lane >> 4) * 8;
  float acc[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) acc[j] = 0.f;
#pragma unroll
  for (int s = 0; s < 16; ++s) {
    const float av = A[t * PLD + s];
    const float4 b0 = *reinterpret_cast<const float4*>(B + s * PLD + j0);
    const float4 b1 = *reinterpret_cast<const float4*>(B + s * PLD + j0 + 4);
    acc[0] = fmaf(av, b0.x, acc[0]); acc[1] = fmaf(av, b0.y, acc[1]);
    acc[2] = fmaf(av, b0.z, acc[2]); acc[3] = fmaf(av, b0.w, acc[3]);
    acc[4] = fmaf(av, b1.x, acc[4]); acc[5] = fmaf(av, b1.y, acc[5]);
    acc[6] = fmaf(av, b1.z, acc[6]); acc[7] = fmaf(av, b1.w, acc[7]);
  }
  if (Cin) {
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[j] = Cin[t * PLD + j0 + j] + sgn * acc[j];
  }
  __syncwarp();
#pragma unroll
  for (int j = 0; j < 8; ++j) C[t * PLD + j0 + j] = acc[j];
  __syncwarp();
}
__device__ __forceinline__ int nth_bit(int bits, int j) {
  int cnt = 0, r = 0;
#pragma unroll
  for (int s = 0; s < 16; ++s) {
    if ((bits >> s) & 1) { if (cnt == j) r = s; ++cnt; }
  }
  return r;
}

__global__ void __launch_bounds__(NT, 1) gdnf2_kernel(const Params p) {
  extern __shared__ __align__(128) unsigned char smem_raw[];
  Smem& sm = *reinterpret_cast<Smem*>(smem_raw);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int N = p.N, HV = p.HV, H = p.H, S = p.n_slots;
  if (threadIdx.x == 0) {
    for (int s = 0; s < 2; ++s) {
      mb_init(&sm.full[s], 2); mb_init(&sm.empty[s], CW + 1); mb_init(&sm.prepd[s], 1); mb_init(&sm.pfree[s], CW);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  const int n_my = p.n_items > (int)blockIdx.x ? (p.n_items - 1 - (int)blockIdx.x) / (int)gridDim.x + 1 : 0;

  if (warp == CW) {
    // ---------------------------------------------------------------- producer warp
    // prologue: lane j holds the slot record of this CTA's item j (all loads in flight together)
    int r_idx = -1, r_leaf = -1, r_pp = 0, r_onb = 0;
    if (lane < n_my) {
      const int it = blockIdx.x + lane * gridDim.x;
      r_idx = p.cidx[it / HV];
      if (r_idx >= 0) { r_leaf = p.leaf[r_idx]; r_pp = p.parity[r_idx]; r_onb = p.mask[r_idx] & ((1 << N) - 1); }
      if (p.abl & 2) r_leaf = -1;
      if (r_leaf < 0) r_onb = 0;
    }
    for (int i = 0; i < n_my; ++i) {
      const int it = blockIdx.x + i * gridDim.x, s = i & 1;
      const int n = it / HV, hv = it % HV, ih = hv / (HV / H);
      const long long tok0 = (long long)n * N;
      const int idx = __shfl_sync(0xffffffffu, r_idx, i & 31), leaf = __shfl_sync(0xffffffffu, r_leaf, i & 31);
      const int pp = __shfl_sync(0xffffffffu, r_pp, i & 31), onb = __shfl_sync(0xffffffffu, r_onb, i & 31);
      const long long rrow = (long long)pp * S + (idx >= 0 ? idx : 0);
      const int np_ = __popc(onb);
      // the stage's slot-dependent smem payload (gsc) is written after the copies are issued: second arrival
      float gl = 0.f, gsv = 0.f;
      if (leaf >= 0) {
        gl = p.gam_buf[(rrow * N + leaf) * HV + hv];
        if (lane < np_) gsv = p.gam_buf[(rrow * N + nth_bit(onb, lane)) * HV + hv];
      }
      if (i >= 2) mb_wait(&sm.empty[s], ((i >> 1) - 1) & 1);
      Stage& st = sm.st[s];
      const uint32_t tx = (idx >= 0 ? 128u * KD * 4u : 0u) + 2u * N * KD * 2u + (uint32_t)np_ * (KD + VD) * 4u;
      if (lane == 0) mb_arrive_tx(&sm.full[s], tx);
      __syncwarp();
      if (idx >= 0) {
        const float* hsrc = p.state + (long long)idx * p.s_slot + (long long)hv * VD * KD;
        if (lane < 4) bulk_g2s_ef(st.h + lane * 32 * KD, hsrc + (long long)lane * 32 * KD, 32 * KD * 4, &sm.full[s]);
      }
      if (lane < N) {
        bulk_g2s(st.qs + lane * QLD, p.q + (tok0 + lane) * p.sq + (long long)ih * KD, KD * 2, &sm.full[s]);
        bulk_g2s(st.ks + lane * QLD, p.k + (tok0 + lane) * p.sk + (long long)ih * KD, KD * 2, &sm.full[s]);
      }
      if (lane < np_) {
        const int sn = nth_bit(onb, lane);
        bulk_g2s_ef(st.knc + lane * KD, p.kn_buf + ((rrow * N + sn) * H + ih) * KD, KD * 4, &sm.full[s]);
        bulk_g2s_ef(st.vpc + lane * VD, p.vp_buf + ((rrow * N + sn) * HV + hv) * VD, VD * 4, &sm.full[s]);
      }
      for (int i2 = lane; i2 < (16 - N) * (QLD / 8); i2 += 32) {  // rows >= N of q / k: zero (no copy)
        const int t = N + i2 / (QLD / 8), ch = i2 % (QLD / 8);
        *reinterpret_cast<uint4*>(st.qs + t * QLD + ch * 8) = make_uint4(0, 0, 0, 0);
        *reinterpret_cast<uint4*>(st.ks + t * QLD + ch * 8) = make_uint4(0, 0, 0, 0);
      }
      if (lane < np_) st.meta.gsc[lane] = expf(gl - gsv);
      if (lane == 0) {
        st.meta.idx = idx; st.meta.leaf = leaf; st.meta.np = np_; st.meta.onb = onb; st.meta.gl = gl;
        st.meta.rrow = rrow;
        st.meta.wrow = idx >= 0 ? (long long)(1 - pp) * S + idx : 2LL * S;
      }
      __syncwarp();
      if (lane == 0) mb_arrive(&sm.full[s]);
    }
    return;
  }

  if (warp >= CW + 1) {
    // ---------------------------------------------------------------- prep warps (prep warp pw: items i = pw mod 2)
    const int g = lane >> 2, c = lane & 3;
    const int pw = warp - CW - 1;
    PrepScratch& ps = sm.ps[pw];
    for (int i = pw; i < n_my; i += 2) {
      const int it = blockIdx.x + i * gridDim.x, s = i & 1;
      const int n = it / HV, hv = it % HV, ih = hv / (HV / H);
      const long long tok0 = (long long)n * N;
      Stage& st = sm.st[s];
      // gates (global loads, before the stage wait)
      float g_t = 0.f, beta_t = 0.f;
      int bits_t = 0;
      if (lane < N) {
        const float A_l = ldf(p.A_log, hv, p.alog_bf16), dtb = ldf(p.dt_bias, hv, p.dtb_bf16);
        const float av = __bfloat162float(p.a[(tok0 + lane) * p.sa + hv]);
        const float bv = __bfloat162float(p.b[(tok0 + lane) * p.sb + hv]);
        const float x = av + dtb;
        const float bx = p.sp_beta * x;
        const float spx = bx <= p.sp_thresh ? (1.f / p.sp_beta) * logf(1.f + expf(bx)) : x;
        g_t = -expf(A_l) * spx;
        beta_t = 1.f / (1.f + expf(-bv));
        bits_t = p.anc[tok0 + lane];
      }
      float gam_t = 0.f;
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        const float gs = __shfl_sync(0xffffffffu, g_t, j);
        if ((bits_t >> j) & 1) gam_t += gs;
      }
      mb_wait(&sm.full[s], (i >> 1) & 1);
      const long long wrow = st.meta.wrow;
      float sq = 0.f, sk2 = 0.f;
      {
        const int t = lane >> 1, h0 = (lane & 1) * 64;
#pragma unroll 8
        for (int j = 0; j < 64; j += 2) {
          const float2 qf = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(st.qs + t * QLD + h0 + j));
          const float2 kf = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(st.ks + t * QLD + h0 + j));
          sq = fmaf(qf.x, qf.x, fmaf(qf.y, qf.y, sq));
          sk2 = fmaf(kf.x, kf.x, fmaf(kf.y, kf.y, sk2));
        }
        sq += __shfl_xor_sync(0xffffffffu, sq, 1);
        sk2 += __shfl_xor_sync(0xffffffffu, sk2, 1);
      }
      const float rq_r = 1.f / sqrtf(sq + 1e-6f), rk_r = 1.f / sqrtf(sk2 + 1e-6f);
      const float rq_l = __shfl_sync(0xffffffffu, rq_r, (2 * lane) & 31);
      const float rk_l = __shfl_sync(0xffffffffu, rk_r, (2 * lane) & 31);
      if (lane < 16) {
        ps.gam[lane] = gam_t; ps.beta[lane] = beta_t; ps.bits[lane] = bits_t;
        ps.rk[lane] = rk_l; ps.rqs[lane] = rq_l * p.scale;
      }
      const float cA_l = expf(gam_t) * rk_l, cQ_l = expf(gam_t) * rq_l * p.scale;
      __syncwarp();
      float kk[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}}, qk[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
#pragma unroll
      for (int ks = 0; ks < 8; ++ks) {
        const uint2 k0 = *reinterpret_cast<const uint2*>(st.ks + g * QLD + ks * 16 + 4 * c);
        const uint2 k1 = *reinterpret_cast<const uint2*>(st.ks + (g + 8) * QLD + ks * 16 + 4 * c);
        const uint2 q0 = *reinterpret_cast<const uint2*>(st.qs + g * QLD + ks * 16 + 4 * c);
        const uint2 q1 = *reinterpret_cast<const uint2*>(st.qs + (g + 8) * QLD + ks * 16 + 4 * c);
        mma16816(kk[0], k0.x, k1.x, k0.y, k1.y, k0.x, k0.y);
        mma16816(kk[1], k0.x, k1.x, k0.y, k1.y, k1.x, k1.y);
        mma16816(qk[0], q0.x, q1.x, q0.y, q1.y, k0.x, k0.y);
        mma16816(qk[1], q0.x, q1.x, q0.y, q1.y, k1.x, k1.y);
      }
      // commit inputs for the next step (kn of the key head from its first value head; gam), from the staged k rows
      if (hv % (HV / H) == 0) {
        for (int i2 = lane; i2 < N * (KD / 2); i2 += 32) {
          const int t = i2 / (KD / 2), j = (i2 % (KD / 2)) * 2;
          const float2 kf = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(st.ks + t * QLD + j));
          const float r = ps.rk[t];
          *reinterpret_cast<float2*>(p.kn_buf + ((wrow * N + t) * H + ih) * KD + j) = make_float2(kf.x * r, kf.y * r);
        }
      }
      if (lane < N) p.gam_buf[(wrow * N + lane) * HV + hv] = gam_t;
      __syncwarp();
      if (lane == 0) mb_arrive(&sm.empty[s]);  // done with the stage's q / k rows
#pragma unroll
      for (int nt = 0; nt < 2; ++nt) {
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int t = g + (e >> 1) * 8, sn = nt * 8 + 2 * c + (e & 1);
          const int ae = (ps.bits[t] >> sn) & 1;
          const float dg = ae ? ps.gam[t] - ps.gam[sn] : 0.f;
          const float ex = expf(dg);
          const float kkv = kk[nt][e] * (ps.rk[t] * ps.rk[sn]);
          const float qkv = qk[nt][e] * (ps.rqs[t] * ps.rk[sn]);
          ps.X[t * PLD + sn] = (ae && sn != t) ? ps.beta[t] * ex * kkv : 0.f;
          ps.P[t * PLD + sn] = ae ? ex * qkv : 0.f;
          ps.R[t * PLD + sn] = (sn == t) ? ps.beta[sn] : 0.f;
        }
      }
      __syncwarp();
      // X^8 == 0 exactly when every node has depth <= 7 (<= 8 ancestor-or-self bits): r + X^8 r == r bitwise
      const bool deep = __any_sync(0xffffffffu, __popc(bits_t) > 8);
      mm16(ps.X, ps.X, nullptr, 1.f, ps.X2, lane);
      mm16(ps.X2, ps.X2, nullptr, 1.f, ps.X4, lane);
      float* rr = ps.R;
      if (deep) {
        mm16(ps.X4, ps.X4, nullptr, 1.f, ps.X8, lane);
        mm16(ps.X8, ps.R, ps.R, 1.f, ps.T, lane);     // r = r + X8 r
        mm16(ps.X4, ps.T, ps.T, 1.f, ps.R, lane);     // r = r + X4 r
      } else {
        mm16(ps.X4, ps.R, ps.R, 1.f, ps.T, lane);     // r = r + X4 r
        rr = ps.T;
      }
      float* rt = rr == ps.R ? ps.T : ps.R;
      mm16(ps.X2, rr, rr, 1.f, rt, lane);             // r = r + X2 r
      if (i >= 2) mb_wait(&sm.pfree[s], ((i >> 1) - 1) & 1);  // the consumers' epilogue of item i - 2 is done
      if (lane < 16) { st.pre.cA[lane] = cA_l; st.pre.cQ[lane] = cQ_l; }
      mm16(ps.X, rt, rt, -1.f, st.pre.M, lane);       // M' = r - X r
      mm16(ps.P, st.pre.M, nullptr, 1.f, st.pre.W, lane);  // W = P M'
      if (lane == 0) mb_arrive(&sm.prepd[s]);
    }
    return;
  }

  // ------------------------------------------------------------------ consumer warps
  const int g = lane >> 2, c = lane & 3;
  float* uS = sm.uS[warp];
  const int vcol0 = warp * 16;
  // item i - 1's epilogue inputs, carried across one iteration
  float vinP[2][2][2], accAP[2][4], accQP[2][4];
  long long wrowP = 0;
  auto epilogue = [&](const int ip) {
    const int it = blockIdx.x + ip * gridDim.x, s = ip & 1;
    const int n = it / HV, hv = it % HV;
    const long long tok0 = (long long)n * N;
    const Stage& st = sm.st[s];
    mb_wait(&sm.prepd[s], (ip >> 1) & 1);
    const Prep& pr = st.pre;
#pragma unroll
    for (int nt = 0; nt < 2; ++nt)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int tr = e >> 1, t = g + 8 * tr, vl = nt * 8 + 2 * c + (e & 1);
        uS[t * 16 + vl] = vinP[tr][nt][e & 1] - pr.cA[t] * accAP[nt][e];
      }
    __syncwarp();
    float vp[2][2][2], oo[2][2][2];
#pragma unroll
    for (int tr = 0; tr < 2; ++tr) {
      const int t = g + 8 * tr;
#pragma unroll
      for (int nt = 0; nt < 2; ++nt) {
        vp[tr][nt][0] = vp[tr][nt][1] = 0.f;
        oo[tr][nt][0] = pr.cQ[t] * accQP[nt][2 * tr];
        oo[tr][nt][1] = pr.cQ[t] * accQP[nt][2 * tr + 1];
      }
    }
#pragma unroll
    for (int sn = 0; sn < 16; ++sn) {
      const float2 u0 = *reinterpret_cast<const float2*>(uS + sn * 16 + 2 * c);
      const float2 u1 = *reinterpret_cast<const float2*>(uS + sn * 16 + 8 + 2 * c);
#pragma unroll
      for (int tr = 0; tr < 2; ++tr) {
        const int t = g + 8 * tr;
        const float m = pr.M[t * PLD + sn], w = pr.W[t * PLD + sn];
        vp[tr][0][0] = fmaf(m, u0.x, vp[tr][0][0]); vp[tr][0][1] = fmaf(m, u0.y, vp[tr][0][1]);
        vp[tr][1][0] = fmaf(m, u1.x, vp[tr][1][0]); vp[tr][1][1] = fmaf(m, u1.y, vp[tr][1][1]);
        oo[tr][0][0] = fmaf(w, u0.x, oo[tr][0][0]); oo[tr][0][1] = fmaf(w, u0.y, oo[tr][0][1]);
        oo[tr][1][0] = fmaf(w, u1.x, oo[tr][1][0]); oo[tr][1][1] = fmaf(w, u1.y, oo[tr][1][1]);
      }
    }
    __syncwarp();
    if (lane == 0) mb_arrive(&sm.pfree[s]);  // prep slot of this stage consumed
#pragma unroll
    for (int tr = 0; tr < 2; ++tr) {
      const int t = g + 8 * tr;
      if (t >= N) continue;
      long long orow = tok0 + t;
      if (p.packed) orow = p.out_rows[tok0 + t];
#pragma unroll
      for (int nt = 0; nt < 2; ++nt) {
        const int vg = vcol0 + nt * 8 + 2 * c;
        st_f2_ef(p.vp_buf + ((wrowP * N + t) * HV + hv) * VD + vg, vp[tr][nt][0], vp[tr][nt][1], pol_evict_first());
        if (orow >= 0)
          *reinterpret_cast<uint32_t*>(p.o + (orow * HV + hv) * VD + vg) = pack_bf16(oo[tr][nt][0], oo[tr][nt][1]);
      }
    }
  };
  for (int i = 0; i < n_my; ++i) {
    const int it = blockIdx.x + i * gridDim.x, s = i & 1;
    const int n = it / HV, hv = it % HV;
    const long long tok0 = (long long)n * N;
    Stage& st = sm.st[s];
    float vin[2][2][2];
#pragma unroll
    for (int tr = 0; tr < 2; ++tr)
#pragma unroll
      for (int nt = 0; nt < 2; ++nt) {
        const int t = g + 8 * tr;
        float2 f = make_float2(0.f, 0.f);
        if (t < N)
          f = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(
              p.v + (tok0 + t) * p.sv + (long long)hv * VD + vcol0 + nt * 8 + 2 * c));
        vin[tr][nt][0] = f.x; vin[tr][nt][1] = f.y;
      }
    mb_wait(&sm.full[s], (i >> 1) & 1);
    const int idx = st.meta.idx, leaf = st.meta.leaf, np_ = st.meta.np;
    const long long wrow = st.meta.wrow;
    const int r0 = warp * 16 + g;
    float h[8][2][4];
    if (idx >= 0) {
#pragma unroll
      for (int ks = 0; ks < 8; ++ks)
#pragma unroll
        for (int r = 0; r < 2; ++r) {
          const float4 t4 = *reinterpret_cast<const float4*>(st.h + (r0 + 8 * r) * HLD + ks * 16 + 4 * c);
          h[ks][r][0] = t4.x; h[ks][r][1] = t4.y; h[ks][r][2] = t4.z; h[ks][r][3] = t4.w;
        }
    } else {
#pragma unroll
      for (int ks = 0; ks < 8; ++ks)
#pragma unroll
        for (int r = 0; r < 2; ++r)
#pragma unroll
          for (int e = 0; e < 4; ++e) h[ks][r][e] = 0.f;
    }
    if (leaf >= 0) {
      const float eg = expf(st.meta.gl);
#pragma unroll
      for (int ks = 0; ks < 8; ++ks)
#pragma unroll
        for (int r = 0; r < 2; ++r)
#pragma unroll
          for (int e = 0; e < 4; ++e) h[ks][r][e] *= eg;
#pragma unroll 2
      for (int j = 0; j < np_; ++j) {
        const float wj = st.meta.gsc[j];
        const float a0 = wj * st.vpc[j * VD + r0], a1 = wj * st.vpc[j * VD + r0 + 8];
#pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
          const float4 kv = *reinterpret_cast<const float4*>(st.knc + j * KD + ks * 16 + 4 * c);
          h[ks][0][0] = fmaf(a0, kv.x, h[ks][0][0]); h[ks][0][1] = fmaf(a0, kv.y, h[ks][0][1]);
          h[ks][0][2] = fmaf(a0, kv.z, h[ks][0][2]); h[ks][0][3] = fmaf(a0, kv.w, h[ks][0][3]);
          h[ks][1][0] = fmaf(a1, kv.x, h[ks][1][0]); h[ks][1][1] = fmaf(a1, kv.y, h[ks][1][1]);
          h[ks][1][2] = fmaf(a1, kv.z, h[ks][1][2]); h[ks][1][3] = fmaf(a1, kv.w, h[ks][1][3]);
        }
      }
#pragma unroll
      for (int ks = 0; ks < 8; ++ks)
#pragma unroll
        for (int r = 0; r < 2; ++r)
          *reinterpret_cast<float4*>(st.h + (r0 + 8 * r) * HLD + ks * 16 + 4 * c) =
              make_float4(h[ks][r][0], h[ks][r][1], h[ks][r][2], h[ks][r][3]);
      fence_async_smem();
      bar_sync(1, CW * 32);
      if (warp == 0 && lane == 0)
        bulk_s2g_ef(p.state + (long long)idx * p.s_slot + (long long)hv * VD * KD, st.h, 128 * KD * 4);
    }
    float accA[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}}, accQ[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
#pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
      const uint2 k0 = *reinterpret_cast<const uint2*>(st.ks + g * QLD + ks * 16 + 4 * c);
      const uint2 k1 = *reinterpret_cast<const uint2*>(st.ks + (g + 8) * QLD + ks * 16 + 4 * c);
      const uint2 q0 = *reinterpret_cast<const uint2*>(st.qs + g * QLD + ks * 16 + 4 * c);
      const uint2 q1 = *reinterpret_cast<const uint2*>(st.qs + (g + 8) * QLD + ks * 16 + 4 * c);
#pragma unroll
      for (int nt = 0; nt < 2; ++nt) {
        const float x0 = h[ks][nt][0], x1 = h[ks][nt][1], x2 = h[ks][nt][2], x3 = h[ks][nt][3];
        const __nv_bfloat162 H0 = __floats2bfloat162_rn(x0, x1), H1 = __floats2bfloat162_rn(x2, x3);
        const float2 h0f = __bfloat1622float2(H0), h1f = __bfloat1622float2(H1);
        const float r0 = x0 - h0f.x, r1 = x1 - h0f.y, r2 = x2 - h1f.x, r3 = x3 - h1f.y;
        const __nv_bfloat162 M0 = __floats2bfloat162_rn(r0, r1), M1 = __floats2bfloat162_rn(r2, r3);
        const float2 m0f = __bfloat1622float2(M0), m1f = __bfloat1622float2(M1);
        const __nv_bfloat162 L0 = __floats2bfloat162_rn(r0 - m0f.x, r1 - m0f.y), L1 = __floats2bfloat162_rn(r2 - m1f.x, r3 - m1f.y);
        const uint32_t bh0 = *reinterpret_cast<const uint32_t*>(&H0), bh1 = *reinterpret_cast<const uint32_t*>(&H1);
        const uint32_t bm0 = *reinterpret_cast<const uint32_t*>(&M0), bm1 = *reinterpret_cast<const uint32_t*>(&M1);
        const uint32_t bl0 = *reinterpret_cast<const uint32_t*>(&L0), bl1 = *reinterpret_cast<const uint32_t*>(&L1);
        mma16816(accA[nt], k0.x, k1.x, k0.y, k1.y, bh0, bh1);
        mma16816(accA[nt], k0.x, k1.x, k0.y, k1.y, bm0, bm1);
        mma16816(accA[nt], k0.x, k1.x, k0.y, k1.y, bl0, bl1);
        mma16816(accQ[nt], q0.x, q1.x, q0.y, q1.y, bh0, bh1);
        mma16816(accQ[nt], q0.x, q1.x, q0.y, q1.y, bm0, bm1);
        mma16816(accQ[nt], q0.x, q1.x, q0.y, q1.y, bl0, bl1);
      }
    }
    __syncwarp();
    if (warp == 0 && lane == 0 && leaf >= 0) bulk_wait_read();  // the bulk store has read the stage tile
    if (lane == 0) mb_arrive(&sm.empty[s]);  // stage rows consumed (state, q / k, commit inputs)
    if (i > 0) epilogue(i - 1);
#pragma unroll
    for (int nt = 0; nt < 2; ++nt)
#pragma unroll
      for (int e = 0; e < 4; ++e) { accAP[nt][e] = accA[nt][e]; accQP[nt][e] = accQ[nt][e]; }
#pragma unroll
    for (int tr = 0; tr < 2; ++tr)
#pragma unroll
      for (int nt = 0; nt < 2; ++nt) { vinP[tr][nt][0] = vin[tr][nt][0]; vinP[tr][nt][1] = vin[tr][nt][1]; }
    wrowP = wrow;
  }
  if (n_my > 0) epilogue(n_my - 1);
  if (warp == 0 && lane == 0) bulk_wait_all();
}

}  // namespace d3w2

void d3w_gdnf2(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor a, torch::Tensor b,
               torch::Tensor A_log, torch::Tensor dt_bias, torch::Tensor anc, torch::Tensor state, torch::Tensor cidx,
               torch::Tensor leaf, torch::Tensor mask, torch::Tensor parity, torch::Tensor kn_buf, torch::Tensor vp_buf,
               torch::Tensor gam_buf, torch::Tensor o, torch::Tensor out_rows, int64_t packed, int64_t N, int64_t grid,
               int64_t abl, double scale, double sp_beta, double sp_thresh) {
  d3w2::Params p;
  p.q = reinterpret_cast<const __nv_bfloat16*>(q.data_ptr());
  p.k = reinterpret_cast<const __nv_bfloat16*>(k.data_ptr());
  p.v = reinterpret_cast<const __nv_bfloat16*>(v.data_ptr());
  p.sq = q.stride(0); p.sk = k.stride(0); p.sv = v.stride(0);
  p.a = reinterpret_cast<const __nv_bfloat16*>(a.data_ptr());
  p.b = reinterpret_cast<const __nv_bfloat16*>(b.data_ptr());
  p.sa = a.stride(0); p.sb = b.stride(0);
  p.A_log = A_log.data_ptr(); p.dt_bias = dt_bias.data_ptr();
  p.alog_bf16 = A_log.scalar_type() == at::kBFloat16; p.dtb_bf16 = dt_bias.scalar_type() == at::kBFloat16;
  p.anc = anc.data_ptr<int>();
  p.state = state.data_ptr<float>(); p.s_slot = state.stride(0);
  p.cidx = cidx.data_ptr<int>();
  p.leaf = leaf.data_ptr<int>(); p.mask = mask.data_ptr<int>(); p.parity = parity.data_ptr<int>();
  p.n_slots = (int)state.size(0);
  p.kn_buf = kn_buf.data_ptr<float>(); p.vp_buf = vp_buf.data_ptr<float>(); p.gam_buf = gam_buf.data_ptr<float>();
  p.o = reinterpret_cast<__nv_bfloat16*>(o.data_ptr());
  p.out_rows = reinterpret_cast<const long long*>(out_rows.data_ptr()); p.packed = (int)packed;
  p.scale = (float)scale; p.sp_beta = (float)sp_beta; p.sp_thresh = (float)sp_thresh;
  p.abl = (int)abl;
  p.N = (int)N; p.H = (int)k.size(1); p.HV = (int)v.size(1);
  const int bs = (int)cidx.size(0);
  p.n_items = bs * p.HV;
  if (bs == 0) return;
  const size_t smem = sizeof(d3w2::Smem);
  static bool attr = false;
  if (!attr) {
    cudaFuncSetAttribute(d3w2::gdnf2_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    attr = true;
  }
  const int g = (int)std::min<int64_t>(std::max<int64_t>(grid, (p.n_items + 31) / 32), p.n_items);  // <= 32 items per CTA
  d3w2::gdnf2_kernel<<<g, d3w2::NT, smem, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""

CFLAGS = ["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-std=c++17"]
_mod = None
_NSM = {}


def _ext():
    global _mod
    if _mod is None:
        from torch.utils.cpp_extension import load_inline

        base = os.environ.get("SGLANG_CACHE_DIR") or os.path.join(os.path.expanduser("~"), ".cache", "sglang")
        key = hashlib.sha1("\0".join([_CPP, _CUDA, *CFLAGS]).encode()).hexdigest()[:16]
        bdir = os.path.join(base, "specpilot_d3w_gdnf3", key)
        os.makedirs(bdir, exist_ok=True)
        _mod = load_inline(name=f"specpilot_d3w_gdnf3_{key}", cpp_sources=_CPP, cuda_sources=_CUDA,
                           functions=["d3w_gdnf2"], extra_cuda_cflags=CFLAGS, build_directory=bdir,
                           verbose=bool(os.environ.get("D3W_GDNF_VERBOSE")))
    return _mod


def ensure_built():
    """Build / load the extension (SGLANG_CACHE_DIR cache) before any CUDA graph capture."""
    _ext()


def _dummy64(dev):
    t = _NSM.get(("d64", dev))
    if t is None:
        t = torch.zeros(1, dtype=torch.int64, device=dev)
        _NSM[("d64", dev)] = t
    return t


def _num_sms(dev):
    n = _NSM.get(dev)
    if n is None:
        n = torch.cuda.get_device_properties(dev).multi_processor_count
        _NSM[dev] = n
    return n


def gdnf_cuda_verify(*, q, k, v, a, b, A_log, dt_bias, anc, ssm_states, cache_indices, leaf, mask, parity, kn_buf,
                     vp_buf, gam_buf, o, out_rows=None, N=16, grid=0, abl=0, scale=None, softplus_beta=1.0,
                     softplus_threshold=20.0, mw=None):
    """q, k [T, H, K] / v [T, HV, V] bf16 (token-strided views, last dim contiguous); a, b [T, HV] bf16;
    anc [bs, N] int32 ancestor-or-self bits; ssm_states [S, HV, V, K] fp32; cache_indices [bs] int32 (-1 padding);
    leaf / mask / parity [>= S] int32 (wy_lazy state); bufs of this layer [2S+1, N, ...] fp32 (kn, v', gam); o [rows,
    HV, V] bf16 (packed rows from out_rows [T] int64 when given, else token rows); grid = persistent CTAs (0 = #SMs)."""
    T, H, K = q.shape
    HV, V = v.shape[1], v.shape[2]
    S = ssm_states.shape[0]
    bs = cache_indices.shape[0]
    assert K == 128 and V == 128 and 1 <= N <= 16 and T == bs * N and HV % H == 0, (q.shape, v.shape, N, bs)
    assert q.dtype == k.dtype == v.dtype == a.dtype == b.dtype == o.dtype == torch.bfloat16
    assert q.stride(2) == 1 and k.stride(2) == 1 and v.stride(2) == 1 and q.stride(1) == K and k.stride(1) == K
    assert v.stride(1) == V and a.stride(1) == 1 and b.stride(1) == 1
    assert q.stride(0) % 8 == 0 and k.stride(0) % 8 == 0 and v.stride(0) % 2 == 0
    assert q.data_ptr() % 16 == 0 and k.data_ptr() % 16 == 0 and v.data_ptr() % 4 == 0
    assert ssm_states.dtype == torch.float32 and ssm_states.stride(3) == 1 and ssm_states.stride(2) == K
    assert ssm_states.stride(1) == K * V and ssm_states.stride(0) % 4 == 0 and ssm_states.data_ptr() % 16 == 0
    assert cache_indices.dtype == torch.int32 and cache_indices.is_contiguous()
    assert anc.dtype == torch.int32 and anc.is_contiguous() and anc.numel() >= bs * N
    for t_ in (leaf, mask, parity):
        assert t_.dtype == torch.int32 and t_.shape[0] >= S and t_.is_contiguous()
    for t_, tail in ((kn_buf, (N, H, K)), (vp_buf, (N, HV, V)), (gam_buf, (N, HV))):
        assert t_.dtype == torch.float32 and t_.shape[0] >= 2 * S + 1 and tuple(t_.shape[1:]) == tail, (t_.shape, tail)
        assert t_.is_contiguous() and t_.data_ptr() % 16 == 0
    assert o.stride(2) == 1 and o.stride(1) == V and o.stride(0) == HV * V
    packed = out_rows is not None
    if packed:
        assert out_rows.dtype == torch.int64 and out_rows.numel() >= T
    else:
        assert o.shape[0] >= T
        out_rows = _dummy64(q.device)
    if scale is None:
        scale = K ** -0.5
    _ext().d3w_gdnf2(q, k, v, a, b, A_log, dt_bias, anc, ssm_states, cache_indices, leaf, mask, parity, kn_buf,
                     vp_buf, gam_buf, o, out_rows, int(packed), int(N), int(grid or _num_sms(q.device)), int(abl),
                     float(scale), float(softplus_beta), float(softplus_threshold))
    return o
