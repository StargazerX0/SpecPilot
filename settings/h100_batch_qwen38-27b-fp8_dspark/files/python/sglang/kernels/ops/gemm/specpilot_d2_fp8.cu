// d2: weight-stationary FP8 block-scaled GEMM for Hopper (sm_90a).
// out[M, N] (bf16) = sum_kb (xq[:, kb] . w[:, kb]^T) * xs[:, kb] * ws[n / 128, kb]
// One CTA owns BN weight rows (the wgmma N dimension; 128, or 136 with two 128-row scale blocks per tile) for all
// packed tokens (the wgmma M dimension, NWG math warpgroups x 64 tokens), so each weight byte is read from HBM once
// per call. A (weight + token) ring stage is released as soon as its wgmmas retire; token scales stream through their
// own ring; the epilogue stages the output tile in the drained ring. Round-robin ordered wgmma issue across the math
// warpgroups overlaps one's promotion with the others' MMAs. Per k-block promotion fin += (sx[token] * ws) * acc in
// DeepGEMM's order, so a data-parallel grid is bitwise DeepGEMM's result; stream-K (tiles > 2x SMs) uses a fixed-order
// fp32 fix-up by the last arriving CTA. DED: a dedicated output-staging buffer lets the producer stream the next
// tile while the epilogue runs. TAIL (SPECPILOT_D1R3_WSTAIL, BN 136 only): tile t = 128-row scale block t + the 8 rows
// 128T+8t.. of the tail region [128T, 136T) (two TMA boxes), so every tile's scale boundary sits at the compile-time
// column 128 and the promotion needs no per-element scale select; same per-element arithmetic (bitwise). SPLIT > 1 (k-split clusters with a DSMEM reduction) is kept for experiments.
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/mma/sm90.cuh>
#include <deep_gemm/ptx/wgmma.cuh>
#include <cooperative_groups.h>

namespace d2 {

using Barrier = cutlass::arch::ClusterTransactionBarrier;

constexpr uint64_t kEvictFirst = 0x12F0000000000000ull;
constexpr uint64_t kEvictLast = 0x14F0000000000000ull;
constexpr int kDStride = 136;  // bf16 elements per staged output row (128 + 8 pad: staggers smem banks)
constexpr int kWBytes = 128 * 128;
constexpr int kSxStages = 8;

struct Params {
    const float* ws;   // [N/128, KB] fp32
    const float* xs;   // column-major [KB][xs_ld] fp32
    __nv_bfloat16* out;
    float* part;       // [2 * G][128 * BT] fp32 partial-tile slots
    int* cnt;          // [num_tiles] arrival counters (left at 0 after every call)
    int M, N, KB, ws_ld, xs_ld, out_ld, num_tiles, G;
    long long U;       // num_tiles * KB
};

// Operand-load stage of the weight stream: one (tile rows from row0) x 128-k FP8 weight tile into smem (128B swizzle) via TMA,
// completing its bytes on `bar`. A decode-in-loop variant (re-coded weights) replaces this function.
__device__ __forceinline__ void load_weight_tile(const CUtensorMap* tm_w, uint64_t* bar, uint8_t* dst, int row0, int kb) {
    cute::SM90_TMA_LOAD_2D::copy(tm_w, bar, kEvictFirst, dst, kb * 128, row0);
}

__device__ __forceinline__ long long unit_begin(long long U, int G, int g) { return (U * g) / G; }

__device__ __forceinline__ int group_of_unit(long long U, int G, long long u) {
    int g = (int)((u * G) / U);
    while (g > 0 && unit_begin(U, G, g) > u) --g;
    while (g + 1 < G && unit_begin(U, G, g + 1) <= u) ++g;
    return g;
}

template <int NWG, int STAGES, int BN, int SPLIT, bool DED, bool TAIL = false>
__global__ void __launch_bounds__(128 * (NWG + 1), 1)
d2_ws_kernel(const __grid_constant__ CUtensorMap tm_w, const __grid_constant__ CUtensorMap tm_x, const Params p,
             const __grid_constant__ CUtensorMap tm_w8) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    // tokens are the wgmma M dimension (NWG math warpgroups x 64 tokens), the tile's 128 weight rows the N dimension
    constexpr int BT = 64 * NWG;
    constexpr int SXS = DED ? 4 : kSxStages;  // token-scale ring depth (smaller when the output stage is dedicated)
    constexpr int kMath = 128 * NWG;
    // BN weight rows per tile (128, or 136 so that 34816-row gate_up splits into 256 tiles = 2 per CTA)
    using WGMMA = typename deep_gemm::mma::sm90::FP8MMASelector<BN>::type;
    constexpr int W_BYTES = BN * 128;
    constexpr int X_BYTES = BT * 128;
    constexpr int STAGE_BYTES = W_BYTES + X_BYTES;
    constexpr int kAcc = WGMMA::kNumAccum;  // 64 tokens x BN rows / 128 threads
    constexpr int kDS = BN + 8;             // staged output row stride (bf16)
    static_assert(W_BYTES % 1024 == 0 && X_BYTES % 1024 == 0, "tiles must keep 1024-byte swizzle alignment");
    static_assert(DED || BT * kDS * 2 <= STAGES * STAGE_BYTES, "output staging must fit in the ring");
    constexpr int D_BYTES = DED ? (BT * kDS * 2 + 1023) / 1024 * 1024 : 0;
    static_assert(!TAIL || BN == 136, "TAIL: 128-row scale block + 8 tail rows per tile");

    extern __shared__ __align__(1024) uint8_t smem[];
    uint8_t* smem_stage = smem;                                                     // [STAGES][W | X]
    // DED: dedicated output staging after the ring (the producer runs ahead into the next tile); else the drained ring
    __nv_bfloat16* smem_d = reinterpret_cast<__nv_bfloat16*>(DED ? smem + STAGES * STAGE_BYTES : smem);
    float* smem_sx = reinterpret_cast<float*>(smem + STAGES * STAGE_BYTES + D_BYTES);  // [SXS][BT]
    float* smem_sw = smem_sx + SXS * BT;                                      // [2][256]: scale blocks a, b
    Barrier* full = reinterpret_cast<Barrier*>(smem_sw + 512);
    Barrier* empty = full + STAGES;
    Barrier* sfull = empty + STAGES;
    Barrier* sempty = sfull + SXS;
    Barrier* ring_free = sempty + SXS;
    __shared__ int s_last;

    const int tid = threadIdx.x;
    const int warp = tid / 32, lane = tid % 32;
    const int g = blockIdx.x;
    // SPLIT > 1: a cluster of SPLIT CTAs splits one tile's k-blocks; partial sums are reduced through DSMEM
    const int split_rank = SPLIT > 1 ? (int)cute::block_rank_in_cluster() : 0;
    long long u_begin, u_end;
    if constexpr (SPLIT > 1) {
        const long long t = blockIdx.x / SPLIT;
        u_begin = t * p.KB + (p.KB * split_rank) / SPLIT;
        u_end = t * p.KB + (p.KB * (split_rank + 1)) / SPLIT;
    } else {
        u_begin = unit_begin(p.U, p.G, g);
        u_end = unit_begin(p.U, p.G, g + 1);
    }

    if (tid == kMath) {
        cute::prefetch_tma_descriptor(&tm_w);
        cute::prefetch_tma_descriptor(&tm_x);
        if constexpr (TAIL) cute::prefetch_tma_descriptor(&tm_w8);  // SPECPILOT_D1R3
    }
    if (tid == kMath + 32) {
        for (int i = 0; i < STAGES; ++i) {
            full[i].init(1);
            empty[i].init(4 * NWG);  // one arrival per math warp
        }
        for (int i = 0; i < SXS; ++i) {
            sfull[i].init(1);
            sempty[i].init(4 * NWG);
        }
        ring_free->init(4 * NWG);
        cutlass::arch::fence_barrier_init();
    }
    // token-scale tails beyond M are never written by the bulk copies: zero them once
    for (int i = tid; i < SXS * BT; i += kMath + 128) smem_sx[i] = 0.f;
    __syncthreads();

    const int xs_bytes = ((p.M + 3) / 4) * 4 * 4;

    if (warp >= 4 * NWG) {
        cutlass::arch::warpgroup_reg_dealloc<NWG >= 3 ? 24 : 40>();
        if (tid == kMath) {
            int stage = 0, phase = 0, sst = 0, sph = 0, ntile = 0;
            long long u = u_begin;
            while (u < u_end) {
                const int tile = (int)(u / p.KB);
                const int kb0 = (int)(u % p.KB);
                const int kb1 = (int)min((long long)p.KB, kb0 + (u_end - u));
                // the previous tile's epilogue staged its output in the ring
                if (!DED && ntile > 0) ring_free->wait((ntile - 1) & 1);
                for (int kb = kb0; kb < kb1; ++kb) {
                    sempty[sst].wait(sph ^ 1);
                    sfull[sst].arrive_and_expect_tx(xs_bytes);
                    const uint32_t dst = static_cast<uint32_t>(__cvta_generic_to_shared(smem_sx + sst * BT));
                    const uint32_t mb = static_cast<uint32_t>(__cvta_generic_to_shared(&sfull[sst]));
                    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
                                 :: "r"(dst), "l"(p.xs + (long long)kb * p.xs_ld), "r"(xs_bytes), "r"(mb) : "memory");
                    sst = sst == SXS - 1 ? 0 : sst + 1;
                    sph ^= sst == 0;

                    empty[stage].wait(phase ^ 1);
                    uint8_t* st = smem_stage + stage * STAGE_BYTES;
                    uint64_t* bar = reinterpret_cast<uint64_t*>(&full[stage]);
                    full[stage].arrive_and_expect_tx(W_BYTES + X_BYTES);
                    if constexpr (TAIL) {  // SPECPILOT_D1R3: scale block `tile` + its 8 tail rows
                        cute::SM90_TMA_LOAD_2D::copy(&tm_w, bar, kEvictFirst, st, kb * 128, tile * 128);
                        cute::SM90_TMA_LOAD_2D::copy(&tm_w8, bar, kEvictFirst, st + 128 * 128, kb * 128, p.num_tiles * 128 + tile * 8);
                    } else {
                        load_weight_tile(&tm_w, bar, st, tile * BN, kb);
                    }
                    cute::SM90_TMA_LOAD_2D::copy(&tm_x, bar, kEvictLast, st + W_BYTES, kb * 128, 0);
                    stage = stage == STAGES - 1 ? 0 : stage + 1;
                    phase ^= stage == 0;
                }
                u += kb1 - kb0;
                ++ntile;
            }
        }
        if constexpr (SPLIT > 1) {
            cute::cluster_arrive();
            cute::cluster_wait();
            cute::cluster_arrive();
            cute::cluster_wait();
        }
    } else {
        // register pool per CTA = threads x launch-time count (96 at 640 threads): 4 x 128 x 112 + 128 x 24 fits
        cutlass::arch::warpgroup_reg_alloc<NWG == 4 ? 112 : (NWG == 3 ? 160 : 232)>();
        const int wg = tid / 128;                         // tokens [wg*64, wg*64+64)
        const int t0 = wg * 64 + (warp % 4) * 16 + lane / 4;  // this thread's token rows t0, t0 + 8
        const int cq = (lane % 4) * 2;                    // weight-row pair offset within an 8-row group
        int stage = 0, phase = 0, sst = 0, sph = 0;
        long long u = u_begin;
        float accum[kAcc], fin[kAcc];
        bool issued = false;

#define D2_STORE_TILE(TILE)                                                                              \
        do {                                                                                             \
            cutlass::arch::NamedBarrier::sync(kMath, 2);                                                 \
            _Pragma("unroll") for (int j = 0; j < kAcc / 4; ++j) {                                       \
                const int n = j * 8 + cq;                                                                \
                *reinterpret_cast<__nv_bfloat162*>(smem_d + t0 * kDS + n) =                              \
                    __float22bfloat162_rn(make_float2(fin[j * 4 + 0], fin[j * 4 + 1]));                  \
                *reinterpret_cast<__nv_bfloat162*>(smem_d + (t0 + 8) * kDS + n) =                        \
                    __float22bfloat162_rn(make_float2(fin[j * 4 + 2], fin[j * 4 + 3]));                  \
            }                                                                                            \
            cutlass::arch::NamedBarrier::sync(kMath, 2);                                                 \
            __nv_bfloat16* dst_ = p.out + (long long)(TILE) * BN;                                        \
            constexpr int kCh = BN / 8;                                                                  \
            for (int i = tid; i < p.M * kCh; i += kMath) {                                               \
                const int t = i / kCh, ch = i % kCh;                                                     \
                __nv_bfloat16* dc_ = !TAIL ? dst_ + ch * 8                                               \
                    : (ch < 16 ? p.out + (long long)(TILE) * 128 + ch * 8                                \
                               : p.out + (long long)p.num_tiles * 128 + (TILE) * 8);                     \
                *reinterpret_cast<int4*>(dc_ + (long long)t * p.out_ld) =                                \
                    *reinterpret_cast<const int4*>(smem_d + t * kDS + ch * 8);                           \
            }                                                                                            \
        } while (0)

        while (u < u_end) {
            const int tile = (int)(u / p.KB);
            const int kb0 = (int)(u % p.KB);
            const int kb1 = (int)min((long long)p.KB, kb0 + (u_end - u));
            // weight block scales of this tile (its 128 rows share one 128x128 scale block per k-block)
            cutlass::arch::NamedBarrier::sync(kMath, 1);
            const int n0 = tile * BN;
            const int blk_a = TAIL ? tile : n0 / 128;
            const int blk_b = TAIL ? (p.num_tiles * 128 + tile * 8) / 128 : (n0 + BN - 1) / 128;
            const int split_col = TAIL ? 128 : (blk_a + 1) * 128 - n0;  // tile columns >= split_col use scale block b
            for (int i = tid; i < kb1 - kb0; i += kMath) {
                smem_sw[i] = p.ws[(long long)blk_a * p.ws_ld + kb0 + i];
                smem_sw[256 + i] = p.ws[(long long)blk_b * p.ws_ld + kb0 + i];
            }
            cutlass::arch::NamedBarrier::sync(kMath, 1);
#pragma unroll
            for (int i = 0; i < kAcc; ++i) fin[i] = 0.f;
            for (int kb = kb0; kb < kb1; ++kb) {
                const float sw = smem_sw[kb - kb0], swb = smem_sw[256 + kb - kb0];
                full[stage].wait(phase);
                uint8_t* st = smem_stage + stage * STAGE_BYTES;
                auto a_desc = deep_gemm::mma::sm90::make_smem_desc(st + W_BYTES + wg * 64 * 128, 1);
                auto b_desc = deep_gemm::mma::sm90::make_smem_desc(st, 1);
                const uint32_t a_lo = a_desc.reg32_[0], b_lo = b_desc.reg32_[0];
                // round-robin ordered issue: warpgroup i issues k-block kb after warpgroup i-1
                if (NWG > 1) {
                    if (wg == 0) {
                        if (issued) asm volatile("bar.sync %0, 256;\n" :: "r"(12) : "memory");
                    } else {
                        asm volatile("bar.sync %0, 256;\n" :: "r"(12 + wg) : "memory");
                    }
                }
#pragma unroll
                for (int i = 0; i < kAcc; ++i) deep_gemm::ptx::warpgroup_fence_operand(accum[i]);
                deep_gemm::ptx::warpgroup_arrive();
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    a_desc.reg32_[0] = a_lo + k * 32 / 16;
                    b_desc.reg32_[0] = b_lo + k * 32 / 16;
                    WGMMA::wgmma(a_desc, b_desc, accum, k);
                }
                deep_gemm::ptx::warpgroup_commit_batch();
                if (NWG > 1) asm volatile("bar.arrive %0, 256;\n" :: "r"(12 + (wg + 1) % NWG) : "memory");
                issued = true;
#pragma unroll
                for (int i = 0; i < kAcc; ++i) deep_gemm::ptx::warpgroup_fence_operand(accum[i]);
                deep_gemm::ptx::warpgroup_wait<0>();
                if (lane == 0) empty[stage].arrive();
                sfull[sst].wait(sph);
                const float x0 = smem_sx[sst * BT + t0], x1 = smem_sx[sst * BT + t0 + 8];
                __syncwarp();
                if (lane == 0) sempty[sst].arrive();
                const float s0 = sw * x0, s1 = sw * x1;
                const float s0b = BN <= 128 ? s0 : swb * x0, s1b = BN <= 128 ? s1 : swb * x1;
#pragma unroll
                for (int j = 0; j < kAcc / 4; ++j) {
                    // an 8-column group never straddles split_col (a multiple of 8)
                    const bool lo = BN <= 128 || (TAIL ? j < 16 : j * 8 < split_col);
                    fin[j * 4 + 0] += (lo ? s0 : s0b) * accum[j * 4 + 0];
                    fin[j * 4 + 1] += (lo ? s0 : s0b) * accum[j * 4 + 1];
                    fin[j * 4 + 2] += (lo ? s1 : s1b) * accum[j * 4 + 2];
                    fin[j * 4 + 3] += (lo ? s1 : s1b) * accum[j * 4 + 3];
                }
                stage = stage == STAGES - 1 ? 0 : stage + 1;
                phase ^= stage == 0;
                sst = sst == SXS - 1 ? 0 : sst + 1;
                sph ^= sst == 0;
            }
            if constexpr (SPLIT > 1) {
                // ranks > 0 park their fp32 partials in their drained ring; rank 0 adds them in rank order
                float* red = reinterpret_cast<float*>(smem);
                cutlass::arch::NamedBarrier::sync(kMath, 2);
                if (split_rank != 0) {
#pragma unroll
                    for (int i = 0; i < kAcc; ++i) red[i * kMath + tid] = fin[i];
                }
                cute::cluster_arrive();
                cute::cluster_wait();
                if (split_rank == 0) {
                    auto cluster = cooperative_groups::this_cluster();
#pragma unroll 1
                    for (int r = 1; r < SPLIT; ++r) {
                        const float* peer = cluster.map_shared_rank(red, r);
#pragma unroll
                        for (int i = 0; i < kAcc; ++i) fin[i] += peer[i * kMath + tid];
                    }
                }
                cute::cluster_arrive();
                cute::cluster_wait();
                if (split_rank == 0) D2_STORE_TILE(tile);
            } else if (kb0 == 0 && kb1 == p.KB) {
                D2_STORE_TILE(tile);
            } else {
                // partial tile: slot 2g (first range segment) or 2g+1 (last range segment),
                // layout [kAcc/4][kMath threads][4] floats so each warp store covers 512 contiguous bytes
                const int my_slot = 2 * g + (u == u_begin ? 0 : 1);
                float* slot = p.part + (long long)my_slot * (BN * BT) + tid * 4;
#pragma unroll
                for (int i = 0; i < kAcc; i += 4)
                    __stcg(reinterpret_cast<float4*>(slot + i * kMath), make_float4(fin[i], fin[i + 1], fin[i + 2], fin[i + 3]));
                __threadfence();
                cutlass::arch::NamedBarrier::sync(kMath, 1);
                const long long tu0 = (long long)tile * p.KB, tu1 = tu0 + p.KB;
                const int g_lo = group_of_unit(p.U, p.G, tu0), g_hi = group_of_unit(p.U, p.G, tu1 - 1);
                if (tid == 0) {
                    const int old = atomicAdd(p.cnt + tile, 1);
                    s_last = (old == g_hi - g_lo) ? 1 : 0;
                }
                cutlass::arch::NamedBarrier::sync(kMath, 1);
                if (s_last) {
                    __threadfence();
#pragma unroll
                    for (int i = 0; i < kAcc; ++i) fin[i] = 0.f;
                    for (int gg = g_lo; gg <= g_hi; ++gg) {
                        // group gg covers this tile with its first range segment iff its range starts inside it
                        const int sl = 2 * gg + (unit_begin(p.U, p.G, gg) >= tu0 ? 0 : 1);
                        const float* src = p.part + (long long)sl * (BN * BT) + tid * 4;
#pragma unroll
                        for (int i = 0; i < kAcc; i += 4) {
                            const float4 v = __ldcg(reinterpret_cast<const float4*>(src + i * kMath));
                            fin[i] += v.x; fin[i + 1] += v.y; fin[i + 2] += v.z; fin[i + 3] += v.w;
                        }
                    }
                    D2_STORE_TILE(tile);
                    if (tid == 0) p.cnt[tile] = 0;
                }
            }
            // every math warp is done with the staged output: the producer may refill the ring
            cutlass::arch::NamedBarrier::sync(kMath, 2);
            if (lane == 0) ring_free->arrive();
            u += kb1 - kb0;
        }
        if (NWG > 1 && wg == 0 && issued) asm volatile("bar.sync %0, 256;\n" :: "r"(12) : "memory");
#undef D2_STORE_TILE
    }
#endif
}

}  // namespace d2

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cudaTypedefs.h>

static PFN_cuTensorMapEncodeTiled_v12000 get_encode() {
    static PFN_cuTensorMapEncodeTiled_v12000 fn = nullptr;
    if (!fn) {
        cudaDriverEntryPointQueryResult q;
        void* ptr = nullptr;
        cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &ptr, cudaEnableDefault, &q);
        TORCH_CHECK(ptr != nullptr, "cuTensorMapEncodeTiled not found");
        fn = reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(ptr);
    }
    return fn;
}

static CUtensorMap make_map_fp8(const torch::Tensor& t, int box_rows) {
    CUtensorMap m;
    const uint64_t rows = t.size(0), cols = t.size(1);
    cuuint64_t dims[2] = {cols, rows};
    cuuint64_t strides[1] = {(cuuint64_t)t.stride(0)};
    cuuint32_t box[2] = {128, (cuuint32_t)box_rows};
    cuuint32_t estr[2] = {1, 1};
    CUresult r = get_encode()(&m, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, t.data_ptr(), dims, strides, box, estr,
                              CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                              CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed ", (int)r);
    return m;
}

template <int NWG, int STAGES, int BN, int SPLIT, bool DED, bool TAIL = false>
static void launch(const torch::Tensor& xq, const torch::Tensor& xs, const torch::Tensor& w, const torch::Tensor& ws,
                   torch::Tensor& out, torch::Tensor& part, torch::Tensor& cnt, int G) {
    auto tm_w = make_map_fp8(w, TAIL ? 128 : BN);
    auto tm_w8 = TAIL ? make_map_fp8(w, 8) : tm_w;  // SPECPILOT_D1R3
    constexpr int BT = 64 * NWG;
    auto tm_x = make_map_fp8(xq, BT);
    d2::Params p;
    p.ws = ws.data_ptr<float>();
    p.xs = xs.data_ptr<float>();
    p.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
    p.part = part.data_ptr<float>();
    p.cnt = cnt.data_ptr<int>();
    p.M = xq.size(0); p.N = w.size(0); p.KB = w.size(1) / 128;
    p.ws_ld = ws.stride(0); p.xs_ld = xs.stride(1); p.out_ld = out.stride(0);
    TORCH_CHECK(p.N % BN == 0, "N must be a multiple of the tile rows");
    p.num_tiles = p.N / BN; p.G = G; p.U = (long long)p.num_tiles * p.KB;
    TORCH_CHECK(p.M <= BT && xs.size(0) >= p.M, "token count exceeds the tile");
    constexpr int dbytes = DED ? ((64 * NWG) * (BN + 8) * 2 + 1023) / 1024 * 1024 : 0;
    constexpr int smem = STAGES * (BN * 128 + 64 * NWG * 128) + dbytes + (DED ? 4 : d2::kSxStages) * 64 * NWG * 4 + 512 * 4 +
                         (2 * STAGES + 2 * (DED ? 4 : d2::kSxStages) + 1) * 8;
    static_assert(smem <= 232448 - 1024, "smem (dynamic + static)");
    auto kern = d2::d2_ws_kernel<NWG, STAGES, BN, SPLIT, DED, TAIL>;
    static bool attr = false;
    if (!attr) { cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); attr = true; }
    if (SPLIT > 1) G = p.num_tiles * SPLIT;
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(G); cfg.blockDim = dim3(128 * (NWG + 1)); cfg.dynamicSmemBytes = smem;
    cfg.stream = c10::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute attrs[2];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = SPLIT; attrs[0].val.clusterDim.y = 1; attrs[0].val.clusterDim.z = 1;
    cfg.attrs = attrs; cfg.numAttrs = 1;
    cudaError_t e = cudaLaunchKernelEx(&cfg, kern, tm_w, tm_x, p, tm_w8);
    TORCH_CHECK(e == cudaSuccess, "d2 launch failed: ", cudaGetErrorString(e));
}

// variant code: stages * 10 + nsub
void d2_gemm(torch::Tensor xq, torch::Tensor xs, torch::Tensor w, torch::Tensor ws, torch::Tensor out,
             torch::Tensor part, torch::Tensor cnt, int64_t bt, int64_t stages, int64_t G, int64_t cl) {
    TORCH_CHECK(cl == 1, "cluster size must be 1");
// stages argument: S + 100 for 136-row / 200 for 64-row weight tiles + 1000 * SPLIT (k-split cluster)
// + 10000 for a dedicated output staging buffer
#define D2_CASE(B, S, BN, SP, DED) if (bt == B && stages == S + (BN == 136 ? 100 : (BN == 64 ? 200 : 0)) + (SP > 1 ? 1000 * SP : 0) + (DED ? 10000 : 0)) return launch<B / 64, S, BN, SP, DED>(xq, xs, w, ws, out, part, cnt, (int)G);
    D2_CASE(128, 5, 128, 1, false) D2_CASE(192, 5, 128, 1, false) D2_CASE(128, 5, 136, 1, false) D2_CASE(192, 5, 136, 1, false)
    D2_CASE(256, 5, 64, 1, false) D2_CASE(256, 4, 64, 1, false) D2_CASE(192, 4, 136, 1, true) D2_CASE(192, 4, 128, 1, true)
    D2_CASE(256, 4, 64, 1, true)
    // SPECPILOT_D1R3: + 20000 = TAIL tile layout (136-row tiles)
    if (bt == 192 && stages == 4 + 100 + 10000 + 20000) return launch<3, 4, 136, 1, true, true>(xq, xs, w, ws, out, part, cnt, (int)G);
    if (bt == 192 && stages == 5 + 100 + 20000) return launch<3, 5, 136, 1, false, true>(xq, xs, w, ws, out, part, cnt, (int)G);
    if (bt == 128 && stages == 5 + 100 + 20000) return launch<2, 5, 136, 1, false, true>(xq, xs, w, ws, out, part, cnt, (int)G);
    TORCH_CHECK(false, "unsupported bt/stages ", bt, " ", stages);
}
