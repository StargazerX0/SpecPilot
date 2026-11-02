// SPECPILOT_D4_GLUE: SiLU(gate) * up -> bf16 -> per-1x128-group FP8 quant, one kernel. Bitwise equal to sglang's
// act_and_mul_kernel (activation.cuh, kRoundActivation = false) followed by per_token_group_quant (fp32 column-major
// TMA-aligned scales); both references are built with --use_fast_math on sm_90, and so is this file.
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace d4 {

constexpr int kGroup = 128;
constexpr int kVec = 8;                    // bf16 per 16-byte load
constexpr int kLanesPerGroup = kGroup / kVec;  // 16
constexpr float kFp8Max = 448.0f;

__device__ __forceinline__ float silu_f32(float x) { return x / (1.0f + expf(-x)); }

// grid: ceil(M * I / 8 / 256) blocks of 256 threads; thread = 8 consecutive outputs of one row
__global__ void __launch_bounds__(256) act_mul_quant_kernel(const __nv_bfloat16* __restrict__ in,
                                                            __nv_fp8_e4m3* __restrict__ q, float* __restrict__ s,
                                                            int M, int I, int s_ld) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
    const int vecs_per_row = I / kVec;
    const long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const int row = (int)(tid / vecs_per_row);
    const int vi = (int)(tid % vecs_per_row);
    const bool live = row < M;
    float out[kVec];
    float amax = 0.f;
    if (live) {
        const uint4 g4 = *reinterpret_cast<const uint4*>(in + (long long)row * 2 * I + vi * kVec);
        const uint4 u4 = *reinterpret_cast<const uint4*>(in + (long long)row * 2 * I + I + vi * kVec);
        const __nv_bfloat16* g = reinterpret_cast<const __nv_bfloat16*>(&g4);
        const __nv_bfloat16* u = reinterpret_cast<const __nv_bfloat16*>(&u4);
#pragma unroll
        for (int i = 0; i < kVec; ++i) {
            const __nv_bfloat16 o = __float2bfloat16(silu_f32(__bfloat162float(g[i])) * __bfloat162float(u[i]));
            out[i] = __bfloat162float(o);
            amax = fmaxf(amax, fabsf(out[i]));
        }
    }
#pragma unroll
    for (int d = kLanesPerGroup / 2; d >= 1; d >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, d));
    if (live) {
        amax = fmaxf(amax, 1e-10f);
        const float qs = kFp8Max / amax;
        __nv_fp8x2_storage_t p[kVec / 2];
#pragma unroll
        for (int i = 0; i < kVec / 2; ++i) {
            const float2 v = make_float2(fminf(out[2 * i] * qs, kFp8Max), fminf(out[2 * i + 1] * qs, kFp8Max));
            p[i] = __nv_cvt_float2_to_fp8x2(v, __NV_SATFINITE, __NV_E4M3);
        }
        *reinterpret_cast<uint2*>(q + (long long)row * I + vi * kVec) = *reinterpret_cast<const uint2*>(p);
        if ((vi % kLanesPerGroup) == 0) s[(long long)(vi / kLanesPerGroup) * s_ld + row] = amax * (1.f / kFp8Max);
    }
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}

}  // namespace d4

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

void d4_act_mul_quant(torch::Tensor in, torch::Tensor q, torch::Tensor s_base, int64_t M, int64_t I) {
    TORCH_CHECK(I % d4::kGroup == 0 && in.is_contiguous() && q.is_contiguous());
    const long long threads = M * (I / d4::kVec);
    if (threads == 0) return;
    const int blocks = (int)((threads + 255) / 256);
    d4::act_mul_quant_kernel<<<blocks, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(in.data_ptr()), reinterpret_cast<__nv_fp8_e4m3*>(q.data_ptr()),
        s_base.data_ptr<float>(), (int)M, (int)I, (int)s_base.size(1));
}
