// SPECPILOT_D7R3_NQ: residual add + GemmaRMSNorm + FP8 e4m3 1x128 group quant of the normed row, one kernel.
// Reference = flashinfer's CuTe FusedAddRMSNormKernel (weight_bias 1.0; H = 5120 bf16 -> 64 threads per row, 8-wide
// vectors, thread t owns columns 8t + v + 512j, fp32 h = x + r, residual = bf16(h), per-thread fma sum of h*h, butterfly
// over the warp, then the row's two warp sums; rstd = rsqrt.approx(div.rn(sum, H) + eps); y = bf16(h * rstd * (w + 1)))
// followed by sglang per_token_group_quant (fp32 column-major TMA-aligned scales, --use_fast_math: q = satfinite(
// min(y * (448 / amax), 448)), s = amax * (1 / 448), amax = max(|y|, 1e-10)). The thread->column map and the
// reduction order are the reference's, so residual, y, q and s are meant to be bitwise equal to the two-kernel path.
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace d7r3 {

constexpr int kVec = 8;
constexpr int kTpr = 64;            // threads per row (flashinfer: 4096 < H <= 6144)
constexpr int kGroup = 128;
constexpr int kLanesPerGroup = kGroup / kVec;  // 16
constexpr float kFp8Max = 448.0f;

template <int VB, int RPB, bool WRITE_Y>
__global__ void __launch_bounds__(kTpr * RPB) add_norm_quant_kernel(
    __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ res, const __nv_bfloat16* __restrict__ w,
    __nv_fp8_e4m3* __restrict__ q, float* __restrict__ s, int M, int s_ld, float eps) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
    constexpr int H = VB * kVec * kTpr;
    __shared__ float red[RPB][2];
    const int t = threadIdx.x % kTpr;
    const int rl = threadIdx.x / kTpr;
    const int row = blockIdx.x * RPB + rl;
    const bool live = row < M;
    const long long base = (long long)row * H + t * kVec;

    float h[VB][kVec];
    uint4 xv[VB], rv[VB], wv[VB];
#pragma unroll
    for (int j = 0; j < VB; ++j) wv[j] = __ldg(reinterpret_cast<const uint4*>(w + t * kVec + j * kVec * kTpr));
    if (live) {
#pragma unroll
        for (int j = 0; j < VB; ++j) {
            xv[j] = *reinterpret_cast<const uint4*>(x + base + j * kVec * kTpr);
            rv[j] = *reinterpret_cast<const uint4*>(res + base + j * kVec * kTpr);
        }
    } else {
#pragma unroll
        for (int j = 0; j < VB; ++j) xv[j] = rv[j] = make_uint4(0, 0, 0, 0);
    }
#pragma unroll
    for (int j = 0; j < VB; ++j) {
        const __nv_bfloat16* xb = reinterpret_cast<const __nv_bfloat16*>(&xv[j]);
        const __nv_bfloat16* rb = reinterpret_cast<const __nv_bfloat16*>(&rv[j]);
        __align__(16) __nv_bfloat16 ob[kVec];
#pragma unroll
        for (int v = 0; v < kVec; ++v) {
            h[j][v] = __fadd_rn(__bfloat162float(xb[v]), __bfloat162float(rb[v]));
            ob[v] = __float2bfloat16_rn(h[j][v]);
        }
        if (live) *reinterpret_cast<uint4*>(res + base + j * kVec * kTpr) = *reinterpret_cast<const uint4*>(ob);
    }

    // per-thread sum of squares in the reference's order: fma over (j, v), fragment order
    float acc = 0.f;
#pragma unroll
    for (int j = 0; j < VB; ++j)
#pragma unroll
        for (int v = 0; v < kVec; ++v) acc = __fmaf_rn(h[j][v], h[j][v], acc);
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) acc = __fadd_rn(acc, __shfl_xor_sync(0xffffffffu, acc, d));
    if ((threadIdx.x & 31) == 0) red[rl][t / 32] = acc;
    __syncthreads();
    const float sum = __fadd_rn(red[rl][0], red[rl][1]);
    const float mean = __fdiv_rn(sum, (float)H);
    const float rstd = rsqrtf(__fadd_rn(mean, eps));

    // y = bf16(h * rstd * (w + 1)) kept as fp32 in h; per-vector |y| max; the 16-lane group max of all VB groups
    // is reduced with interleaved shuffles
    float amax[VB];
#pragma unroll
    for (int j = 0; j < VB; ++j) {
        const __nv_bfloat16* wb = reinterpret_cast<const __nv_bfloat16*>(&wv[j]);
        __align__(16) __nv_bfloat16 yb[kVec];
        amax[j] = 0.f;
#pragma unroll
        for (int v = 0; v < kVec; ++v) {
            yb[v] = __float2bfloat16_rn(__fmul_rn(__fmul_rn(h[j][v], rstd), __fadd_rn(__bfloat162float(wb[v]), 1.0f)));
            h[j][v] = __bfloat162float(yb[v]);
            amax[j] = fmaxf(amax[j], fabsf(h[j][v]));
        }
        if constexpr (WRITE_Y) {
            if (live) *reinterpret_cast<uint4*>(x + base + j * kVec * kTpr) = *reinterpret_cast<const uint4*>(yb);
        }
    }
#pragma unroll
    for (int d = kLanesPerGroup / 2; d >= 1; d >>= 1)
#pragma unroll
        for (int j = 0; j < VB; ++j) amax[j] = fmaxf(amax[j], __shfl_xor_sync(0xffffffffu, amax[j], d));
#pragma unroll
    for (int j = 0; j < VB; ++j) {
        const float am = fmaxf(amax[j], 1e-10f);
        const float qs = kFp8Max / am;
        __nv_fp8x2_storage_t p[kVec / 2];
#pragma unroll
        for (int i = 0; i < kVec / 2; ++i) {
            const float2 v2 = make_float2(fminf(h[j][2 * i] * qs, kFp8Max), fminf(h[j][2 * i + 1] * qs, kFp8Max));
            p[i] = __nv_cvt_float2_to_fp8x2(v2, __NV_SATFINITE, __NV_E4M3);
        }
        if (live) {
            *reinterpret_cast<uint2*>(q + base + j * kVec * kTpr) = *reinterpret_cast<const uint2*>(p);
            if ((t % kLanesPerGroup) == 0)
                s[(long long)(j * (kTpr / kLanesPerGroup) + t / kLanesPerGroup) * s_ld + row] = am * (1.f / kFp8Max);
        }
    }
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}

// Wide variant: one row per block, NT = 640 / VPT threads; thread i owns 8-wide vectors i, i + NT, ... (vector c covers
// columns [8c, 8c + 8)). Loads, the residual store, y and the quant run on all NT threads; only the sum of squares runs
// on the reference's 64 owner threads (owner t sums vectors t + 64j, j = 0..9, from fp32 h staged in smem), so the
// result is bitwise the 64-thread kernel's.
template <int VPT, bool WRITE_Y>
__global__ void __launch_bounds__(640 / VPT) add_norm_quant_wide_kernel(
    __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ res, const __nv_bfloat16* __restrict__ w,
    __nv_fp8_e4m3* __restrict__ q, float* __restrict__ s, int M, int s_ld, float eps) {
    constexpr int NV = 640;  // vectors per row (H = 5120)
    constexpr int NT = NV / VPT;
    constexpr int H = NV * kVec;
    __shared__ __align__(16) float hs[H];
    __shared__ float red[2];
    const int i = threadIdx.x;
    const int row = blockIdx.x;
    const long long rbase = (long long)row * H;

    float h[VPT][kVec];
    uint4 xv[VPT], rv[VPT], wv[VPT];
#pragma unroll
    for (int k = 0; k < VPT; ++k) wv[k] = __ldg(reinterpret_cast<const uint4*>(w + (i + k * NT) * kVec));
    // PDL: the weight is a constant; everything else waits for the producer of x / residual
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
#pragma unroll
    for (int k = 0; k < VPT; ++k) {
        const int c = i + k * NT;
        xv[k] = *reinterpret_cast<const uint4*>(x + rbase + c * kVec);
        rv[k] = *reinterpret_cast<const uint4*>(res + rbase + c * kVec);
    }
#pragma unroll
    for (int k = 0; k < VPT; ++k) {
        const int c = i + k * NT;
        const __nv_bfloat16* xb = reinterpret_cast<const __nv_bfloat16*>(&xv[k]);
        const __nv_bfloat16* rb = reinterpret_cast<const __nv_bfloat16*>(&rv[k]);
        __align__(16) __nv_bfloat16 ob[kVec];
#pragma unroll
        for (int v = 0; v < kVec; ++v) {
            h[k][v] = __fadd_rn(__bfloat162float(xb[v]), __bfloat162float(rb[v]));
            ob[v] = __float2bfloat16_rn(h[k][v]);
        }
        *reinterpret_cast<uint4*>(res + rbase + c * kVec) = *reinterpret_cast<const uint4*>(ob);
        *reinterpret_cast<float4*>(hs + c * kVec) = make_float4(h[k][0], h[k][1], h[k][2], h[k][3]);
        *reinterpret_cast<float4*>(hs + c * kVec + 4) = make_float4(h[k][4], h[k][5], h[k][6], h[k][7]);
    }
    __syncthreads();
    if (i < kTpr) {
        float acc = 0.f;
#pragma unroll
        for (int j = 0; j < NV / kTpr; ++j) {
            const float4 a = *reinterpret_cast<const float4*>(hs + (j * kTpr + i) * kVec);
            const float4 b = *reinterpret_cast<const float4*>(hs + (j * kTpr + i) * kVec + 4);
            acc = __fmaf_rn(a.x, a.x, acc); acc = __fmaf_rn(a.y, a.y, acc);
            acc = __fmaf_rn(a.z, a.z, acc); acc = __fmaf_rn(a.w, a.w, acc);
            acc = __fmaf_rn(b.x, b.x, acc); acc = __fmaf_rn(b.y, b.y, acc);
            acc = __fmaf_rn(b.z, b.z, acc); acc = __fmaf_rn(b.w, b.w, acc);
        }
#pragma unroll
        for (int d = 1; d < 32; d <<= 1) acc = __fadd_rn(acc, __shfl_xor_sync(0xffffffffu, acc, d));
        if ((i & 31) == 0) red[i / 32] = acc;
    }
    __syncthreads();
    const float sum = __fadd_rn(red[0], red[1]);
    const float mean = __fdiv_rn(sum, (float)H);
    const float rstd = rsqrtf(__fadd_rn(mean, eps));

    float amax[VPT];
#pragma unroll
    for (int k = 0; k < VPT; ++k) {
        const int c = i + k * NT;
        const __nv_bfloat16* wb = reinterpret_cast<const __nv_bfloat16*>(&wv[k]);
        __align__(16) __nv_bfloat16 yb[kVec];
        amax[k] = 0.f;
#pragma unroll
        for (int v = 0; v < kVec; ++v) {
            yb[v] = __float2bfloat16_rn(__fmul_rn(__fmul_rn(h[k][v], rstd), __fadd_rn(__bfloat162float(wb[v]), 1.0f)));
            h[k][v] = __bfloat162float(yb[v]);
            amax[k] = fmaxf(amax[k], fabsf(h[k][v]));
        }
        if constexpr (WRITE_Y) *reinterpret_cast<uint4*>(x + rbase + c * kVec) = *reinterpret_cast<const uint4*>(yb);
    }
#pragma unroll
    for (int d = kLanesPerGroup / 2; d >= 1; d >>= 1)
#pragma unroll
        for (int k = 0; k < VPT; ++k) amax[k] = fmaxf(amax[k], __shfl_xor_sync(0xffffffffu, amax[k], d));
#pragma unroll
    for (int k = 0; k < VPT; ++k) {
        const int c = i + k * NT;
        const float am = fmaxf(amax[k], 1e-10f);
        const float qs = kFp8Max / am;
        __nv_fp8x2_storage_t p[kVec / 2];
#pragma unroll
        for (int e = 0; e < kVec / 2; ++e) {
            const float2 v2 = make_float2(fminf(h[k][2 * e] * qs, kFp8Max), fminf(h[k][2 * e + 1] * qs, kFp8Max));
            p[e] = __nv_cvt_float2_to_fp8x2(v2, __NV_SATFINITE, __NV_E4M3);
        }
        *reinterpret_cast<uint2*>(q + rbase + c * kVec) = *reinterpret_cast<const uint2*>(p);
        if ((c % kLanesPerGroup) == 0) s[(long long)(c / kLanesPerGroup) * s_ld + row] = am * (1.f / kFp8Max);
    }
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}

}  // namespace d7r3

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <tuple>

template <int RPB, bool WY>
static void launch(torch::Tensor& x, torch::Tensor& r, torch::Tensor& w, torch::Tensor& q, torch::Tensor& s, int M,
                   double eps) {
    const int blocks = (M + RPB - 1) / RPB;
    d7r3::add_norm_quant_kernel<10, RPB, WY><<<blocks, d7r3::kTpr * RPB, 0, c10::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<__nv_bfloat16*>(x.data_ptr()), reinterpret_cast<__nv_bfloat16*>(r.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(w.data_ptr()), reinterpret_cast<__nv_fp8_e4m3*>(q.data_ptr()),
        s.data_ptr<float>(), M, (int)s.size(1), (float)eps);
}

// rpb = rows per 64-thread-per-row block (1, 2 or 4)
void d7r3_add_norm_quant(torch::Tensor x, torch::Tensor r, torch::Tensor w, torch::Tensor q, torch::Tensor s_base,
                         double eps, bool write_y, int64_t rpb, bool pdl) {
    TORCH_CHECK(x.size(1) == 5120 && x.is_contiguous() && r.is_contiguous() && q.is_contiguous() && w.is_contiguous());
    const int M = (int)x.size(0);
    if (M == 0) return;
    if (rpb == 1) return write_y ? launch<1, true>(x, r, w, q, s_base, M, eps) : launch<1, false>(x, r, w, q, s_base, M, eps);
    if (rpb == 2) return write_y ? launch<2, true>(x, r, w, q, s_base, M, eps) : launch<2, false>(x, r, w, q, s_base, M, eps);
    if (rpb == 10 || rpb == 20) {
        auto args = std::make_tuple(reinterpret_cast<__nv_bfloat16*>(x.data_ptr()), reinterpret_cast<__nv_bfloat16*>(r.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(w.data_ptr()), reinterpret_cast<__nv_fp8_e4m3*>(q.data_ptr()),
            s_base.data_ptr<float>(), M, (int)s_base.size(1), (float)eps);
        auto st = c10::cuda::getCurrentCUDAStream();
        auto go = [&](auto kern, int nt) {
            cudaLaunchConfig_t cfg = {};
            cfg.gridDim = dim3(M); cfg.blockDim = dim3(nt); cfg.dynamicSmemBytes = 0; cfg.stream = st;
            cudaLaunchAttribute at[1];
            at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
            at[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
            cfg.attrs = at; cfg.numAttrs = 1;
            std::apply([&](auto... a) {
                cudaError_t e = cudaLaunchKernelEx(&cfg, kern, a...);
                TORCH_CHECK(e == cudaSuccess, "d7r3 launch failed: ", cudaGetErrorString(e));
            }, args);
        };
        if (rpb == 10) { if (write_y) go(d7r3::add_norm_quant_wide_kernel<1, true>, 640); else go(d7r3::add_norm_quant_wide_kernel<1, false>, 640); }
        else { if (write_y) go(d7r3::add_norm_quant_wide_kernel<2, true>, 320); else go(d7r3::add_norm_quant_wide_kernel<2, false>, 320); }
        return;
    }
    if (rpb == 4) return write_y ? launch<4, true>(x, r, w, q, s_base, M, eps) : launch<4, false>(x, r, w, q, s_base, M, eps);
    TORCH_CHECK(false, "d7r3_add_norm_quant: rpb must be 1, 2 or 4");
}
