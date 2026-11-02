// SPECPILOT_D6: Muse per-head qk RMSNorm (+ RoPE on sliding layers) in one kernel, q written contiguous.
//
// Bitwise equal to stock's fused_qknorm_warp (qknorm.cuh, in place) -> fused_rope_kernel (rope.cuh, in place)
// -> q.contiguous() (attention backend): the norm is norm::apply_norm_warp on the same per-lane storage, the
// rotation is rope.cuh's rope_rotate_head with stock's vector size and 16 work lanes. One warp per (token, head):
// the warp normalizes the head, stores it (q: to q_out, k: in place), __syncwarp (orders the warp's global
// stores), then lanes 0..15 rotate it in place.
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/tile.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <sgl_kernel/impl/norm.cuh>

#include "rope.cuh"

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

struct D6QKNormRopeParams {
  const void* __restrict__ q;  // [T, HQ, D] row stride q_stride (elements)
  void* __restrict__ k;        // [T, HKV, D] row stride k_stride (elements), in place
  void* __restrict__ q_out;    // [T, HQ, D] contiguous
  const void* __restrict__ q_weight;
  const void* __restrict__ k_weight;
  const void* __restrict__ cos_sin_cache;  // [max_pos, D] fp32 (cos half | sin half)
  const void* __restrict__ positions;
  int64_t q_stride;
  int64_t k_stride;
  uint32_t num_qo_heads;
  uint32_t num_kv_heads;
  uint32_t num_tokens;
  float eps;
};

constexpr uint32_t kD6WarpsPerBlock = 4;
constexpr uint32_t kD6Threads = kD6WarpsPerBlock * device::kWarpThreads;
constexpr int64_t kD6HeadDim = 128;
constexpr uint32_t kD6RopeLanes = 16;  // rope.cuh FusedRopeKernel<.., 128, .., bf16>::kWorkThreads

template <bool kUseRope, bool kIsNeox, bool kUsePDL, typename IdType>
__global__ void d6_qknorm_rope_kernel(const D6QKNormRopeParams __grid_constant__ params) {
  using namespace device;
  using DType = bf16_t;
  using Storage = norm::StorageType<DType, kD6HeadDim>;
  constexpr int64_t kVecSize = next_pow2(kD6HeadDim, (2 * kD6RopeLanes * (1 + kIsNeox)));
  constexpr int64_t kCosSinStrideBytes = kD6HeadDim * sizeof(float);
  static_assert(kD6HeadDim == kVecSize * 2 * (1 + kIsNeox) * kD6RopeLanes);

  const auto& p = params;
  const auto num_heads = p.num_qo_heads + p.num_kv_heads;
  const auto num_works = num_heads * p.num_tokens;
  const auto num_workers = gridDim.x * kD6WarpsPerBlock;
  const auto lane_id = threadIdx.x % kWarpThreads;
  const auto gmem = tile::Memory<Storage>::warp();

  const auto q_weight_vec = gmem.load(p.q_weight);
  const auto k_weight_vec = gmem.load(p.k_weight);

  PDLWaitPrimary<kUsePDL>();

  for (auto idx = blockIdx.x * kD6WarpsPerBlock + threadIdx.x / kWarpThreads; idx < num_works; idx += num_workers) {
    const int64_t token_id = idx / num_heads;
    const int64_t head_id = idx % num_heads;
    const bool is_q = head_id < p.num_qo_heads;
    const void* input;
    void* output;
    if (is_q) {
      input = pointer::offset(p.q, 2 * (token_id * p.q_stride + head_id * kD6HeadDim));
      output = pointer::offset(p.q_out, 2 * ((token_id * p.num_qo_heads + head_id) * kD6HeadDim));
    } else {
      const int64_t kv_head = head_id - p.num_qo_heads;
      output = pointer::offset(p.k, 2 * (token_id * p.k_stride + kv_head * kD6HeadDim));
      input = output;
    }
    const auto input_vec = gmem.load(input);
    const auto output_vec = norm::apply_norm_warp<kD6HeadDim>(input_vec, is_q ? q_weight_vec : k_weight_vec, p.eps);
    gmem.store(output, output_vec);
    if constexpr (kUseRope) {
      __syncwarp();
      const auto pos = static_cast<const IdType*>(p.positions)[token_id];
      const auto cos_ptr = pointer::offset(p.cos_sin_cache, pos * kCosSinStrideBytes);
      const auto sin_ptr = pointer::offset(p.cos_sin_cache, pos * kCosSinStrideBytes + kCosSinStrideBytes / 2);
      if (lane_id < kD6RopeLanes) {
        rope_rotate_head<DType, kIsNeox, kD6HeadDim, kVecSize>(output, cos_ptr, sin_ptr, lane_id);
      }
      __syncwarp();
    }
  }

  PDLTriggerSecondary<kUsePDL>();
}

template <bool kUseRope, bool kIsNeox, bool kUsePDL>
struct D6QKNormRopeKernel {
  template <typename IdType>
  static constexpr auto _kernel = d6_qknorm_rope_kernel<kUseRope, kIsNeox, kUsePDL, IdType>;

  static void
  run(const tvm::ffi::TensorView q,
      const tvm::ffi::TensorView k,
      const tvm::ffi::TensorView q_out,
      const tvm::ffi::TensorView q_weight,
      const tvm::ffi::TensorView k_weight,
      const tvm::ffi::TensorView cos_sin_cache,
      const tvm::ffi::TensorView positions,
      float eps) {
    using namespace host;
    auto N = SymbolicSize{"num_tokens"};
    auto Q = SymbolicSize{"num_qo_heads"};
    auto K = SymbolicSize{"num_kv_heads"};
    auto D = SymbolicSize{"head_dim"};
    auto Sq = SymbolicSize{"q_stride"};
    auto Sk = SymbolicSize{"k_stride"};
    auto device = SymbolicDevice{};
    auto id_type = SymbolicDType{};
    D.set_value(kD6HeadDim);
    device.set_options<kDLCUDA>();

    TensorMatcher({N, Q, D}).with_strides({Sq, D, 1}).with_dtype<bf16_t>().with_device(device).verify(q);
    TensorMatcher({N, K, D}).with_strides({Sk, D, 1}).with_dtype<bf16_t>().with_device(device).verify(k);
    TensorMatcher({N, Q, D}).with_dtype<bf16_t>().with_device(device).verify(q_out);
    TensorMatcher({D}).with_dtype<bf16_t>().with_device(device).verify(q_weight).verify(k_weight);
    TensorMatcher({-1, D}).with_dtype<float>().with_device(device).verify(cos_sin_cache);
    TensorMatcher({N}).with_dtype<int32_t, int64_t>(id_type).with_device(device).verify(positions);

    const auto num_tokens = static_cast<uint32_t>(N.unwrap());
    const auto num_qo_heads = static_cast<uint32_t>(Q.unwrap());
    const auto num_kv_heads = static_cast<uint32_t>(K.unwrap());
    const auto params = D6QKNormRopeParams{
        .q = q.data_ptr(),
        .k = k.data_ptr(),
        .q_out = q_out.data_ptr(),
        .q_weight = q_weight.data_ptr(),
        .k_weight = k_weight.data_ptr(),
        .cos_sin_cache = cos_sin_cache.data_ptr(),
        .positions = positions.data_ptr(),
        .q_stride = static_cast<int64_t>(Sq.unwrap()),
        .k_stride = static_cast<int64_t>(Sk.unwrap()),
        .num_qo_heads = num_qo_heads,
        .num_kv_heads = num_kv_heads,
        .num_tokens = num_tokens,
        .eps = eps,
    };
    const auto is_int32 = id_type.is_type<int32_t>();
    const auto kernel = is_int32 ? _kernel<int32_t> : _kernel<int64_t>;
    static const uint32_t kNumSM = runtime::get_sm_count(device.unwrap().device_id);
    static const uint32_t kOcc = runtime::get_blocks_per_sm(_kernel<int64_t>, kD6Threads);
    const auto num_works = (num_qo_heads + num_kv_heads) * num_tokens;
    const auto num_blocks = std::min(kNumSM * kOcc, div_ceil(num_works, kD6WarpsPerBlock));
    LaunchKernel(num_blocks, kD6Threads, device.unwrap()).enable_pdl(kUsePDL)(kernel, params);
  }
};

}  // namespace sglang
