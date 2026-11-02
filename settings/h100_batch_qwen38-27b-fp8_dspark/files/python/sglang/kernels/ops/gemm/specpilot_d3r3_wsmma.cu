// d3r3 wsmma: compute/latency-efficiency variants of the seed's weight-stationary FP8 block-scaled GEMM
// (specpilot_d2_fp8.cu, data-parallel path) for the packed verify at M 129..192 (3 math warpgroups x 64 tokens).
// out[M, N] (bf16) = sum_kb (xq[:, kb] . w[:, kb]^T) * xs[:, kb] * ws[n / 128, kb], promotion fin += (sw * sx) * acc per
// k-block in DeepGEMM's order, so every variant is bitwise the seed kernel's (and DeepGEMM's) result.
// Changes vs the seed:
//  - XS = 0: the seed's coupled [W | X] stage (one barrier; P/DIAG variants of the seed kernel itself);
//  - split operand rings: the HBM weight tile (BN x 128) and the L2-resident token tile (BT x 128) have separate smem
//    rings of depths WS / XS, each fed by its own producer thread (the seed couples them in one stage);
//  - PIPE=1: split-N software pipeline inside each math warpgroup: the tile's columns are issued as two wgmma halves
//    (n64 + n72 at BN 136, n56 + n72 at BN 128); a half's promotion runs while the other half's MMA is in flight (wait_group 1); same registers;
//  - PIPE=2: the seed's mainloop without the round-robin wgmma issue order across math warpgroups;
//  - DIAG (compile-time, numerically wrong, timing only): 1 = no wgmma, 2 = no TMA loads, 4 = no promotion FFMAs.
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

namespace d3r3w {

using Barrier = cutlass::arch::ClusterTransactionBarrier;

constexpr uint64_t kEvictFirst = 0x12F0000000000000ull;
constexpr uint64_t kEvictLast = 0x14F0000000000000ull;

struct Params {
    const float* ws;   // [N/128, KB] fp32
    const float* xs;   // column-major [KB][xs_ld] fp32
    __nv_bfloat16* out;
    int M, N, KB, ws_ld, xs_ld, out_ld, num_tiles, tiles_per_cta;
};

template <int NWG, int WS, int XS, int BN, bool DED>
struct Layout {
    static constexpr int BT = 64 * NWG;
    static constexpr int SXS = DED ? 4 : 8;
    static constexpr int W_BYTES = BN * 128;
    static constexpr int X_BYTES = BT * 128;
    static constexpr int kDS = BN + 8;
    static constexpr int D_BYTES = DED ? (BT * kDS * 2 + 1023) / 1024 * 1024 : 0;
    static constexpr bool COUPLED = XS == 0;  // XS = 0: the seed's coupled stage [W | X] with one barrier
    static constexpr int W_STRIDE = COUPLED ? W_BYTES + X_BYTES : W_BYTES;
    static constexpr int X_OFF = WS * W_STRIDE;
    static constexpr int D_OFF = X_OFF + XS * X_BYTES;
    static constexpr int SX_OFF = D_OFF + D_BYTES;
    static constexpr int SW_OFF = SX_OFF + SXS * BT * 4;
    static constexpr int BAR_OFF = SW_OFF + 512 * 4;
    static constexpr int SMEM = BAR_OFF + (2 * WS + 2 * XS + 2 * SXS + 1) * 8;
    static_assert(W_BYTES % 1024 == 0 && X_BYTES % 1024 == 0, "tiles must keep 1024-byte swizzle alignment");
    static_assert(DED || BT * kDS * 2 <= D_OFF, "output staging must fit in the drained rings");
    static_assert(SMEM <= 232448 - 1024, "smem");
};

template <int NWG, int WS, int XS, int BN, bool DED, int PIPE, int DIAG>
__global__ void __launch_bounds__(128 * (NWG + 1), 1)
wsmma_kernel(const __grid_constant__ CUtensorMap tm_w, const __grid_constant__ CUtensorMap tm_x, const Params p) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    using L = Layout<NWG, WS, XS, BN, DED>;
    constexpr int BT = L::BT, SXS = L::SXS, W_BYTES = L::W_BYTES, X_BYTES = L::X_BYTES, kDS = L::kDS;
    constexpr int kMath = 128 * NWG;
    constexpr bool kNoMMA = DIAG & 1, kNoLoad = DIAG & 2, kNoPromo = DIAG & 4;
    constexpr bool kCoupled = L::COUPLED;
    constexpr int W_STRIDE = L::W_STRIDE;

    extern __shared__ __align__(1024) uint8_t smem[];
    uint8_t* smem_w = smem;
    uint8_t* smem_x = smem + L::X_OFF;
    __nv_bfloat16* smem_d = reinterpret_cast<__nv_bfloat16*>(DED ? smem + L::D_OFF : smem);
    float* smem_sx = reinterpret_cast<float*>(smem + L::SX_OFF);
    float* smem_sw = reinterpret_cast<float*>(smem + L::SW_OFF);
    Barrier* wfull = reinterpret_cast<Barrier*>(smem + L::BAR_OFF);
    Barrier* wempty = wfull + WS;
    Barrier* xfull = wempty + WS;
    Barrier* xempty = xfull + XS;
    Barrier* sfull = xempty + XS;
    Barrier* sempty = sfull + SXS;
    Barrier* ring_free = sempty + SXS;

    const int tid = threadIdx.x;
    const int warp = tid / 32, lane = tid % 32;
    const int tile_begin = blockIdx.x * p.tiles_per_cta, tile_end = tile_begin + p.tiles_per_cta;

    if (tid == kMath) {
        cute::prefetch_tma_descriptor(&tm_w);
        cute::prefetch_tma_descriptor(&tm_x);
    }
    if (tid == kMath + 32) {
        for (int i = 0; i < WS; ++i) { wfull[i].init(1); wempty[i].init(4 * NWG); }
        for (int i = 0; i < XS; ++i) { xfull[i].init(1); xempty[i].init(4 * NWG); }
        for (int i = 0; i < SXS; ++i) { sfull[i].init(1); sempty[i].init(4 * NWG); }
        ring_free->init(4 * NWG);
        cutlass::arch::fence_barrier_init();
    }
    for (int i = tid; i < SXS * BT; i += kMath + 128) smem_sx[i] = 0.f;
    __syncthreads();

    const int xs_bytes = ((p.M + 3) / 4) * 4 * 4;

    if (warp >= 4 * NWG) {
        cutlass::arch::warpgroup_reg_dealloc<NWG >= 3 ? 24 : 40>();
        if (tid == kMath) {
            // weight producer (HBM stream)
            int st = 0, ph = 0;
            for (int tile = tile_begin; tile < tile_end; ++tile) {
                if (!DED && tile > tile_begin) ring_free->wait((tile - tile_begin - 1) & 1);
                for (int kb = 0; kb < p.KB; ++kb) {
                    wempty[st].wait(ph ^ 1);
                    if constexpr (kNoLoad) {
                        wfull[st].arrive();
                    } else {
                        wfull[st].arrive_and_expect_tx(kCoupled ? W_BYTES + X_BYTES : W_BYTES);
                        cute::SM90_TMA_LOAD_2D::copy(&tm_w, reinterpret_cast<uint64_t*>(&wfull[st]), kEvictFirst,
                                                     smem_w + st * W_STRIDE, kb * 128, tile * BN);
                        if constexpr (kCoupled)
                            cute::SM90_TMA_LOAD_2D::copy(&tm_x, reinterpret_cast<uint64_t*>(&wfull[st]), kEvictLast,
                                                         smem_w + st * W_STRIDE + W_BYTES, kb * 128, 0);
                    }
                    st = st == WS - 1 ? 0 : st + 1;
                    ph ^= st == 0;
                }
            }
        } else if (tid == kMath + 32) {
            // token producer (L2-resident activations + their scales)
            int st = 0, ph = 0, sst = 0, sph = 0;
            for (int tile = tile_begin; tile < tile_end; ++tile) {
                if (!DED && tile > tile_begin) ring_free->wait((tile - tile_begin - 1) & 1);
                for (int kb = 0; kb < p.KB; ++kb) {
                    sempty[sst].wait(sph ^ 1);
                    if constexpr (kNoLoad) {
                        sfull[sst].arrive();
                    } else {
                        sfull[sst].arrive_and_expect_tx(xs_bytes);
                        const uint32_t dst = static_cast<uint32_t>(__cvta_generic_to_shared(smem_sx + sst * BT));
                        const uint32_t mb = static_cast<uint32_t>(__cvta_generic_to_shared(&sfull[sst]));
                        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
                                     :: "r"(dst), "l"(p.xs + (long long)kb * p.xs_ld), "r"(xs_bytes), "r"(mb) : "memory");
                    }
                    sst = sst == SXS - 1 ? 0 : sst + 1;
                    sph ^= sst == 0;

                    if constexpr (kCoupled) continue;
                    xempty[st].wait(ph ^ 1);
                    if constexpr (kNoLoad) {
                        xfull[st].arrive();
                    } else {
                        xfull[st].arrive_and_expect_tx(X_BYTES);
                        cute::SM90_TMA_LOAD_2D::copy(&tm_x, reinterpret_cast<uint64_t*>(&xfull[st]), kEvictLast,
                                                     smem_x + st * X_BYTES, kb * 128, 0);
                    }
                    st = st == XS - 1 ? 0 : st + 1;
                    ph ^= st == 0;
                }
            }
        }
    } else {
        cutlass::arch::warpgroup_reg_alloc<NWG == 4 ? 112 : (NWG == 3 ? 160 : 232)>();
        const int wg = tid / 128;
        const int t0 = wg * 64 + (warp % 4) * 16 + lane / 4;
        const int cq = (lane % 4) * 2;
        int wst = 0, wph = 0, xst = 0, xph = 0, sst = 0, sph = 0;
        constexpr int kAcc = BN / 2;  // 64 tokens x BN columns / 128 threads
        float fin[kAcc];

#define D3W_ADV(S, P, N) do { S = S == (N) - 1 ? 0 : S + 1; P ^= S == 0; } while (0)
#define D3W_STORE_TILE(TILE)                                                                             \
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
                *reinterpret_cast<int4*>(dst_ + (long long)t * p.out_ld + ch * 8) =                      \
                    *reinterpret_cast<const int4*>(smem_d + t * kDS + ch * 8);                           \
            }                                                                                            \
        } while (0)
// promotion of accumulator ACC (columns [J0*8, J1*8) of the tile) for k-block scales (SW, SWB) x (X0, X1)
#define D3W_PROMOTE(ACC, J0, J1, SW, SWB, X0, X1)                                                        \
        do {                                                                                             \
            _Pragma("unroll") for (int i = 0; i < (J1 - J0) * 4; ++i) deep_gemm::ptx::warpgroup_fence_operand(ACC[i]); \
            const float s0 = (SW) * (X0), s1 = (SW) * (X1);                                              \
            const float s0b = BN <= 128 ? s0 : (SWB) * (X0), s1b = BN <= 128 ? s1 : (SWB) * (X1);        \
            _Pragma("unroll") for (int j = J0; j < J1; ++j) {                                            \
                const bool lo = BN <= 128 || j * 8 < split_col;                                          \
                const int a = (j - J0) * 4;                                                              \
                if constexpr (kNoPromo) {                                                                \
                    fin[j * 4 + 0] = ACC[a + 0]; fin[j * 4 + 1] = ACC[a + 1];                            \
                    fin[j * 4 + 2] = ACC[a + 2]; fin[j * 4 + 3] = ACC[a + 3];                            \
                } else {                                                                                 \
                    fin[j * 4 + 0] += (lo ? s0 : s0b) * ACC[a + 0];                                      \
                    fin[j * 4 + 1] += (lo ? s0 : s0b) * ACC[a + 1];                                      \
                    fin[j * 4 + 2] += (lo ? s1 : s1b) * ACC[a + 2];                                      \
                    fin[j * 4 + 3] += (lo ? s1 : s1b) * ACC[a + 3];                                      \
                }                                                                                        \
            }                                                                                            \
            _Pragma("unroll") for (int i = 0; i < (J1 - J0) * 4; ++i) deep_gemm::ptx::warpgroup_fence_operand(ACC[i]); \
        } while (0)
// issue one k-block (4 x k32) of MMA into ACC from token tile XPTR and weight rows at WPTR
#define D3W_ISSUE(MMA, ACC, NACC, XPTR, WPTR)                                                            \
        do {                                                                                             \
            auto a_desc = deep_gemm::mma::sm90::make_smem_desc((XPTR) + wg * 64 * 128, 1);               \
            auto b_desc = deep_gemm::mma::sm90::make_smem_desc((WPTR), 1);                               \
            const uint32_t a_lo = a_desc.reg32_[0], b_lo = b_desc.reg32_[0];                             \
            _Pragma("unroll") for (int i = 0; i < NACC; ++i) deep_gemm::ptx::warpgroup_fence_operand(ACC[i]); \
            deep_gemm::ptx::warpgroup_arrive();                                                          \
            _Pragma("unroll") for (int k = 0; k < 4; ++k) {                                              \
                a_desc.reg32_[0] = a_lo + k * 32 / 16;                                                   \
                b_desc.reg32_[0] = b_lo + k * 32 / 16;                                                   \
                if constexpr (!kNoMMA) MMA::wgmma(a_desc, b_desc, ACC, k);                               \
            }                                                                                            \
            deep_gemm::ptx::warpgroup_commit_batch();                                                    \
            _Pragma("unroll") for (int i = 0; i < NACC; ++i) deep_gemm::ptx::warpgroup_fence_operand(ACC[i]); \
        } while (0)

        bool issued = false;
        for (int tile = tile_begin; tile < tile_end; ++tile) {
            cutlass::arch::NamedBarrier::sync(kMath, 1);
            const int n0 = tile * BN, blk_a = n0 / 128, blk_b = (n0 + BN - 1) / 128;
            const int split_col = (blk_a + 1) * 128 - n0;
            for (int i = tid; i < p.KB; i += kMath) {
                smem_sw[i] = p.ws[(long long)blk_a * p.ws_ld + i];
                smem_sw[256 + i] = p.ws[(long long)blk_b * p.ws_ld + i];
            }
            cutlass::arch::NamedBarrier::sync(kMath, 1);
#pragma unroll
            for (int i = 0; i < kAcc; ++i) fin[i] = 0.f;

            if constexpr (PIPE != 1) {
                using WGMMA = typename deep_gemm::mma::sm90::FP8MMASelector<BN>::type;
                float accum[kAcc];
                if constexpr (kNoMMA) {
#pragma unroll
                    for (int i = 0; i < kAcc; ++i) accum[i] = 0.f;
                }
                for (int kb = 0; kb < p.KB; ++kb) {
                    const float sw = smem_sw[kb], swb = smem_sw[256 + kb];
                    wfull[wst].wait(wph);
                    if constexpr (!kCoupled) xfull[xst].wait(xph);
                    if (NWG > 1 && PIPE == 0) {
                        if (wg == 0) {
                            if (issued) asm volatile("bar.sync %0, 256;\n" :: "r"(12) : "memory");
                        } else {
                            asm volatile("bar.sync %0, 256;\n" :: "r"(12 + wg) : "memory");
                        }
                    }
                    D3W_ISSUE(WGMMA, accum, kAcc, (kCoupled ? smem_w + wst * W_STRIDE + W_BYTES : smem_x + xst * X_BYTES), smem_w + wst * W_STRIDE);
                    if (NWG > 1 && PIPE == 0) asm volatile("bar.arrive %0, 256;\n" :: "r"(12 + (wg + 1) % NWG) : "memory");
                    issued = true;
                    deep_gemm::ptx::warpgroup_wait<0>();
                    if (lane == 0) { wempty[wst].arrive(); if constexpr (!kCoupled) xempty[xst].arrive(); }
                    sfull[sst].wait(sph);
                    const float x0 = smem_sx[sst * BT + t0], x1 = smem_sx[sst * BT + t0 + 8];
                    __syncwarp();
                    if (lane == 0) sempty[sst].arrive();
                    D3W_PROMOTE(accum, 0, kAcc / 4, sw, swb, x0, x1);
                    D3W_ADV(wst, wph, WS);
                    if constexpr (!kCoupled) D3W_ADV(xst, xph, XS);
                    D3W_ADV(sst, sph, SXS);
                }
            } else {
                // split-N pipeline: L = columns [0, NL), H = columns [NL, BN) (NL = 64, or 56 at BN 128: two equal
                // halves of one wgmma type make ptxas serialize the pipeline)
                constexpr int NL = BN == 128 ? 56 : 64, NH = BN - NL;
                using MMA_L = typename deep_gemm::mma::sm90::FP8MMASelector<NL>::type;
                using MMA_H = typename deep_gemm::mma::sm90::FP8MMASelector<NH>::type;
                constexpr int JL = NL / 8, JH = BN / 8;
                float accL[NL / 2], accH[NH / 2];
                if constexpr (kNoMMA) {
#pragma unroll
                    for (int i = 0; i < NL / 2; ++i) accL[i] = 0.f;
#pragma unroll
                    for (int i = 0; i < NH / 2; ++i) accH[i] = 0.f;
                }
                // half-k-block stream L0 H0 L1 H1 ...: issue the next half, wait_group 1 (the previous half retired),
                // promote it; straight-line steady state with one pending group at the back-edge (always L(kb+1))
                wfull[wst].wait(wph);
                if constexpr (!kCoupled) xfull[xst].wait(xph);
                D3W_ISSUE(MMA_L, accL, NL / 2, (kCoupled ? smem_w + wst * W_STRIDE + W_BYTES : smem_x + xst * X_BYTES), smem_w + wst * W_STRIDE);
                float sw = 0.f, swb = 0.f, x0 = 0.f, x1 = 0.f;
#define D3W_HALF_L(KB)                                                                                   \
                do {                                                                                     \
                    D3W_ISSUE(MMA_H, accH, NH / 2, (kCoupled ? smem_w + wst * W_STRIDE + W_BYTES : smem_x + xst * X_BYTES), smem_w + wst * W_STRIDE + NL * 128); \
                    deep_gemm::ptx::warpgroup_wait<1>();  /* L(kb) retired */                            \
                    sw = smem_sw[KB];                                                                    \
                    swb = smem_sw[256 + (KB)];                                                           \
                    sfull[sst].wait(sph);                                                                \
                    x0 = smem_sx[sst * BT + t0];                                                         \
                    x1 = smem_sx[sst * BT + t0 + 8];                                                     \
                    __syncwarp();                                                                        \
                    if (lane == 0) sempty[sst].arrive();                                                 \
                    D3W_ADV(sst, sph, SXS);                                                              \
                    D3W_PROMOTE(accL, 0, JL, sw, swb, x0, x1);                                           \
                } while (0)
                int kb = 0;
                for (; kb + 1 < p.KB; ++kb) {
                    D3W_HALF_L(kb);
                    const int pw = wst, px = xst;
                    D3W_ADV(wst, wph, WS);
                    if constexpr (!kCoupled) D3W_ADV(xst, xph, XS);
                    wfull[wst].wait(wph);
                    if constexpr (!kCoupled) xfull[xst].wait(xph);
                    D3W_ISSUE(MMA_L, accL, NL / 2, (kCoupled ? smem_w + wst * W_STRIDE + W_BYTES : smem_x + xst * X_BYTES), smem_w + wst * W_STRIDE);
                    deep_gemm::ptx::warpgroup_wait<1>();  // H(kb) retired
                    if (lane == 0) { wempty[pw].arrive(); if constexpr (!kCoupled) xempty[px].arrive(); }
                    D3W_PROMOTE(accH, JL, JH, sw, swb, x0, x1);
                }
                // tail; the else branch never runs (KB >= 1) but keeps ptxas's wgmma pipeline analysis exact: with an
                // unconditional tail ptxas (cu13 and 12.8) serializes every wgmma of the kernel (C7514)
                if (kb + 1 == p.KB) {
                    D3W_HALF_L(kb);
                    deep_gemm::ptx::warpgroup_wait<0>();
                    if (lane == 0) { wempty[wst].arrive(); if constexpr (!kCoupled) xempty[xst].arrive(); }
                    D3W_PROMOTE(accH, JL, JH, sw, swb, x0, x1);
                } else {
                    deep_gemm::ptx::warpgroup_wait<0>();
                    D3W_PROMOTE(accL, 0, JL, sw, swb, x0, x1);
                }
                D3W_ADV(wst, wph, WS);
                if constexpr (!kCoupled) D3W_ADV(xst, xph, XS);
#undef D3W_HALF_L
            }
            D3W_STORE_TILE(tile);
            cutlass::arch::NamedBarrier::sync(kMath, 2);
            if (lane == 0) ring_free->arrive();
        }
        if (PIPE == 0 && NWG > 1 && wg == 0 && issued) asm volatile("bar.sync %0, 256;\n" :: "r"(12) : "memory");
#undef D3W_STORE_TILE
#undef D3W_PROMOTE
#undef D3W_ISSUE
#undef D3W_ADV
    }
#endif
}

}  // namespace d3r3w

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cudaTypedefs.h>

static PFN_cuTensorMapEncodeTiled_v12000 d3r3w_get_encode() {
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

static CUtensorMap d3r3w_map_fp8(const torch::Tensor& t, int box_rows) {
    CUtensorMap m;
    const uint64_t rows = t.size(0), cols = t.size(1);
    cuuint64_t dims[2] = {cols, rows};
    cuuint64_t strides[1] = {(cuuint64_t)t.stride(0)};
    cuuint32_t box[2] = {128, (cuuint32_t)box_rows};
    cuuint32_t estr[2] = {1, 1};
    CUresult r = d3r3w_get_encode()(&m, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, t.data_ptr(), dims, strides, box, estr,
                                    CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                                    CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed ", (int)r);
    return m;
}

template <int NWG, int WS, int XS, int BN, bool DED, int PIPE, int DIAG>
static void d3r3w_launch(const torch::Tensor& xq, const torch::Tensor& xs, const torch::Tensor& w,
                         const torch::Tensor& ws, torch::Tensor& out, int G) {
    using L = d3r3w::Layout<NWG, WS, XS, BN, DED>;
    auto tm_w = d3r3w_map_fp8(w, BN);
    auto tm_x = d3r3w_map_fp8(xq, L::BT);
    d3r3w::Params p;
    p.ws = ws.data_ptr<float>();
    p.xs = xs.data_ptr<float>();
    p.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
    p.M = xq.size(0); p.N = w.size(0); p.KB = w.size(1) / 128;
    p.ws_ld = ws.stride(0); p.xs_ld = xs.stride(1); p.out_ld = out.stride(0);
    TORCH_CHECK(p.N % BN == 0, "N must be a multiple of the tile rows");
    TORCH_CHECK(p.KB <= 256, "K too large");
    p.num_tiles = p.N / BN;
    TORCH_CHECK(p.num_tiles % G == 0, "data-parallel grid only");
    p.tiles_per_cta = p.num_tiles / G;
    TORCH_CHECK(p.M <= L::BT && xs.size(0) >= p.M, "token count exceeds the tile");
    auto kern = d3r3w::wsmma_kernel<NWG, WS, XS, BN, DED, PIPE, DIAG>;
    static bool attr = false;
    if (!attr) { cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, L::SMEM); attr = true; }
    // launched exactly as the seed's d2 kernel (cudaLaunchKernelEx with a cluster-dimension attribute of 1): a plain
    // <<<>>> launch measured +0.4 ms per 128-GEMM chain at M192 (3083c01c40ce) from worse kernel-boundary overlap
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(G); cfg.blockDim = dim3(128 * (NWG + 1)); cfg.dynamicSmemBytes = L::SMEM;
    cfg.stream = c10::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 1; attrs[0].val.clusterDim.y = 1; attrs[0].val.clusterDim.z = 1;
    cfg.attrs = attrs; cfg.numAttrs = 1;
    cudaError_t e = cudaLaunchKernelEx(&cfg, kern, tm_w, tm_x, p);
    TORCH_CHECK(e == cudaSuccess, "d3r3w launch failed: ", cudaGetErrorString(e));
}

// code = DIAG + 10 * PIPE + 100 * XS + 1000 * WS + 100000 * (BN == 136) + 1000000 * DED
void d3r3w_gemm(torch::Tensor xq, torch::Tensor xs, torch::Tensor w, torch::Tensor ws, torch::Tensor out,
                int64_t code, int64_t G) {
#define D3W_CASE(WS, XS, BN, DED, PIPE, DIAG)                                                                   \
    if (code == DIAG + 10 * PIPE + 100 * XS + 1000 * WS + 100000 * (BN == 136) + 1000000 * DED)                \
        return d3r3w_launch<3, WS, XS, BN, DED, PIPE, DIAG>(xq, xs, w, ws, out, (int)G);
    // gate_up (BN 136, 2 tiles per CTA, dedicated output staging)
    D3W_CASE(4, 0, 136, true, 0, 0) D3W_CASE(4, 0, 136, true, 2, 0) D3W_CASE(4, 0, 136, true, 1, 0)
    D3W_CASE(4, 0, 136, true, 0, 2) D3W_CASE(4, 0, 136, true, 0, 6) D3W_CASE(4, 0, 136, true, 2, 2) D3W_CASE(4, 0, 136, true, 2, 6)
    D3W_CASE(4, 0, 136, true, 1, 2) D3W_CASE(4, 0, 136, true, 1, 6)
    D3W_CASE(4, 4, 136, true, 0, 0) D3W_CASE(5, 3, 136, true, 0, 0) D3W_CASE(6, 2, 136, true, 0, 0)
    D3W_CASE(5, 3, 136, true, 2, 0) D3W_CASE(4, 4, 136, true, 1, 0) D3W_CASE(5, 3, 136, true, 1, 0) D3W_CASE(6, 2, 136, true, 1, 0)
    D3W_CASE(4, 4, 136, true, 0, 1) D3W_CASE(4, 4, 136, true, 0, 2) D3W_CASE(4, 4, 136, true, 0, 4)
    D3W_CASE(5, 3, 136, true, 0, 1) D3W_CASE(6, 2, 136, true, 0, 1)
    // qkvz / qkv (BN 128, 1 tile per CTA, staging in the drained rings)
    D3W_CASE(5, 0, 128, false, 0, 0) D3W_CASE(5, 0, 128, false, 2, 0) D3W_CASE(5, 0, 128, false, 1, 0)
    D3W_CASE(5, 0, 128, false, 0, 6) D3W_CASE(5, 0, 128, false, 2, 6)
    D3W_CASE(5, 5, 128, false, 0, 0) D3W_CASE(8, 3, 128, false, 0, 0) D3W_CASE(10, 2, 128, false, 0, 0)
    D3W_CASE(8, 3, 128, false, 2, 0) D3W_CASE(5, 5, 128, false, 1, 0) D3W_CASE(8, 3, 128, false, 1, 0) D3W_CASE(10, 2, 128, false, 1, 0)
    D3W_CASE(5, 5, 128, false, 0, 1) D3W_CASE(5, 5, 128, false, 0, 2) D3W_CASE(5, 5, 128, false, 0, 4)
    D3W_CASE(8, 3, 128, false, 0, 1) D3W_CASE(10, 2, 128, false, 0, 1)
#undef D3W_CASE
    TORCH_CHECK(false, "d3r3w: unsupported variant code ", code);
}
