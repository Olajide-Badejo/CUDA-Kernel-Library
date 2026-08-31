// Top tensor core GEMM: 128 by 128 block tile, K step 32, ldmatrix fragment
// loads, a three stage cp.async pipeline, mma.sync.m16n8k16, FP16 storage with
// FP32 accumulate. This is the kernel the ladder drives toward the compute bound
// gate.
//
// Round 6 showed that ldmatrix on a 64 by 64 tile still ran the L1 / TEX pipe at
// 95 percent, because a small tile loads too many bytes per fused multiply add:
// staging 2048 halves to do 64 by 64 by 16 MACs. A 128 by 128 tile with K step
// 32 stages 8192 halves for 128 by 128 by 32 MACs, roughly a fourfold better
// load to compute ratio, so the fragment traffic stops dominating and the tensor
// pipe can lead. cp.async overlaps the next stages' global loads with the
// current stage's math.
//
// Eight warps (256 threads) in a 2 by 4 layout; each warp owns a 64 by 32
// region, a 4 by 4 grid of 16 by 8 mma tiles, over two K substeps of 16 per
// stage. Aligned fast path (m by 128, n by 128, k by 32); other shapes reuse the
// WMMA scalar fallback.
//
// Three stages at 16 KB each is 49152 bytes, above the 48 KB static __shared__
// ceiling on plain sm_120, so the tile lives in dynamic shared memory and the
// launcher raises cudaFuncAttributeMaxDynamicSharedMemorySize once per process.
// The per block opt-in ceiling on this part is 99 KB, so the budget fits with
// room to spare; the launcher checks it and refuses rather than falling back.

#include <cstddef>
#include <cstdint>
#include <string>

#include <cuda_fp16.h>
#include <cuda_pipeline.h>

#include "ckl/context.hpp"
#include "ckl/cuda_check.hpp"
#include "ckl/gemm.hpp"

#include "detail/gemm_swizzle.hpp"

namespace ckl {

namespace {

constexpr int kBM = detail::kSwzTileM;  // 128
constexpr int kBN = detail::kSwzTileN;  // 128
constexpr int kBK = detail::kSwzTileK;  // 32
constexpr int kWarpsM = 2;
constexpr int kWarpsN = 4;
constexpr int kThreads = kWarpsM * kWarpsN * 32;  // 256
constexpr int kWarpM = kBM / kWarpsM;             // 64
constexpr int kWarpN = kBN / kWarpsN;             // 32
// Used only inside the guarded device body, so below the architecture floor
// they have no reader and nvcc's unreferenced variable diagnostic, which is an
// error under CKL_WERROR, would fire.
[[maybe_unused]] constexpr int kMTiles = kWarpM / 16;  // 4
[[maybe_unused]] constexpr int kNTiles = kWarpN / 8;   // 4
// One ldmatrix.x4.trans covers two adjacent 16 by 8 output tiles.
[[maybe_unused]] constexpr int kNPairs = kNTiles / 2;  // 2
[[maybe_unused]] constexpr int kKSub = kBK / 16;       // 2

// Pipeline depth. Two stages need a barrier before the issue and another after
// the math, because the buffer about to be overwritten is the one just read;
// with three or more the buffer written at iteration s was last read at
// iteration s-1, which the iteration-s barrier already fences, so one barrier
// per K stage is enough.
constexpr int kStages = 3;
constexpr int kATileHalves = kBM * kBK;  // 4096
constexpr int kBTileHalves = kBK * kBN;  // 4096
constexpr std::size_t kSmemBytes =
    static_cast<std::size_t>(kStages) * (kATileHalves + kBTileHalves) * sizeof(__half);

// cp.async, ldmatrix and mma.sync.m16n8k16 together: sm_80 is the floor.
constexpr int kMinArch = 80;

// The PTX and the cp.async intrinsics below only assemble on sm_80 and newer,
// and an unused device function still reaches ptxas, so the helpers sit inside
// the same guard as the kernel body that calls them.
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800

__device__ inline uint32_t smem_u32(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ inline void ldmatrix_x4(uint32_t (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(smem_u32(p)));
}

// Four transposed 8 by 8 tiles in one instruction: lanes 0 to 7 address the
// first, 8 to 15 the second, and so on, so a single call fills the B operands of
// two neighbouring 16 by 8 mma tiles and halves the B wavefront count against
// the x2 form this kernel used before.
__device__ inline void ldmatrix_x4_trans(uint32_t (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(smem_u32(p)));
}

__device__ inline void mma_m16n8k16(float (&d)[4], const uint32_t (&a)[4], uint32_t b0,
                                    uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

#endif  // __CUDA_ARCH__ >= 800

// FUSE_BIAS folds a per column bias add and a ReLU into the epilogue, while the
// accumulator is still in registers, so no extra pass over C is needed. The
// default (false) path is the plain kernel used everywhere else.
template <bool FUSE_BIAS>
__global__ __launch_bounds__(kThreads) void gemm_mma_opt_kernel(const __half* __restrict__ a,
                                                                const __half* __restrict__ b,
                                                                float* __restrict__ c, int m, int n,
                                                                int k, float alpha, float beta,
                                                                const float* __restrict__ bias) {
#if __CUDA_ARCH__ >= 800
    // kStages A tiles then kStages B tiles. The runtime aligns the dynamic block
    // to 16 bytes and every offset below is a multiple of eight halves, so both
    // the cp.async stores and the ldmatrix loads stay 16 byte aligned.
    extern __shared__ __align__(16) char ckl_mma_opt_smem[];
    __half* const as = reinterpret_cast<__half*>(ckl_mma_opt_smem);
    __half* const bs = as + kStages * kATileHalves;

    const int block_row = blockIdx.y * kBM;
    const int block_col = blockIdx.x * kBN;
    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int lane = tid % 32;
    const int warp_m = warp / kWarpsN;
    const int warp_n = warp % kWarpsN;

    float acc[kMTiles][kNTiles][4] = {};

    const int num_stages = k / kBK;

    // Each thread stages two float4 (sixteen halves) of each of A and B per
    // stage: A is 128 by 32 (512 float4), B is 32 by 128 (512 float4). Past the
    // last K stage the call still commits, empty, so the number of outstanding
    // batches the wait below counts against stays constant to the end of the
    // loop.
    auto stage = [&](int buf, int stage_idx) {
        if (stage_idx < num_stages) {
            const int kk = stage_idx * kBK;
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                const int f = tid + i * kThreads;  // 0..511
                const int a_row = f / (kBK / 8);   // kBK/8 = 4 float4 per row
                const int a_col = (f % (kBK / 8)) * 8;
                __pipeline_memcpy_async(
                    &as[buf * kATileHalves + detail::swizzle_a(a_row, a_col)],
                    &a[static_cast<long long>(block_row + a_row) * k + kk + a_col], sizeof(float4));
                const int b_row = f / (kBN / 8);  // kBN/8 = 16 float4 per row
                const int b_col = (f % (kBN / 8)) * 8;
                __pipeline_memcpy_async(
                    &bs[buf * kBTileHalves + detail::swizzle_b(b_row, b_col)],
                    &b[static_cast<long long>(kk + b_row) * n + block_col + b_col], sizeof(float4));
            }
        }
        __pipeline_commit();
    };

#pragma unroll
    for (int s = 0; s < kStages - 1; ++s) {
        stage(s, s);
    }

    int read_buf = 0;
    int write_buf = kStages - 1;
    for (int s = 0; s < num_stages; ++s) {
        __pipeline_wait_prior(kStages - 2);
        __syncthreads();  // the only barrier in the K loop
        // Issued after the barrier: the buffer being filled here was last read
        // at iteration s-1, which that barrier has already fenced.
        stage(write_buf, s + kStages - 1);

        const __half* const a_tile = as + read_buf * kATileHalves;
        const __half* const b_tile = bs + read_buf * kBTileHalves;

        // B fragments are double buffered in registers across the two K
        // substeps, so the loads for substep ks+1 are in flight while the mma of
        // substep ks runs.
        uint32_t b_frag[2][kNPairs][4];
        // The destination comes in by reference rather than by slot index so the
        // fragment array is never addressed dynamically and stays in registers.
        auto load_b = [&](uint32_t (&dst)[kNPairs][4], int k_off) {
#pragma unroll
            for (int nj = 0; nj < kNPairs; ++nj) {
                const int col_base = warp_n * kWarpN + nj * 16;
                ldmatrix_x4_trans(
                    dst[nj],
                    &b_tile[detail::swizzle_b(k_off + (lane % 16), col_base + (lane / 16) * 8)]);
            }
        };
        load_b(b_frag[0], 0);

#pragma unroll
        for (int ks = 0; ks < kKSub; ++ks) {
            const int k_off = ks * 16;
            uint32_t a_frag[kMTiles][4];
#pragma unroll
            for (int mi = 0; mi < kMTiles; ++mi) {
                const int row_base = warp_m * kWarpM + mi * 16;
                ldmatrix_x4(
                    a_frag[mi],
                    &a_tile[detail::swizzle_a(row_base + (lane % 16), k_off + (lane / 16) * 8)]);
            }
            if (ks + 1 < kKSub) {
                load_b(b_frag[(ks + 1) & 1], (ks + 1) * 16);
            }
#pragma unroll
            for (int nj = 0; nj < kNPairs; ++nj) {
#pragma unroll
                for (int mi = 0; mi < kMTiles; ++mi) {
                    mma_m16n8k16(acc[mi][2 * nj], a_frag[mi], b_frag[ks & 1][nj][0],
                                 b_frag[ks & 1][nj][1]);
                    mma_m16n8k16(acc[mi][2 * nj + 1], a_frag[mi], b_frag[ks & 1][nj][2],
                                 b_frag[ks & 1][nj][3]);
                }
            }
        }

        read_buf = (read_buf + 1 == kStages) ? 0 : read_buf + 1;
        write_buf = (write_buf + 1 == kStages) ? 0 : write_buf + 1;
    }

    // Epilogue. The m16n8k16 accumulator gives each lane two adjacent columns of
    // two rows per tile, so a float2 is the widest store the fragment layout
    // permits; a 128 bit store would need the accumulators exchanged through
    // shared memory first. n is a multiple of 128 and the column index is even
    // on this path, and the launcher checks that C itself is eight byte aligned,
    // so the pair store is always legal here.
    const int group = lane / 4;
    const int tpair = lane % 4;
#pragma unroll
    for (int mi = 0; mi < kMTiles; ++mi) {
        const int row_base = block_row + warp_m * kWarpM + mi * 16;
#pragma unroll
        for (int ni = 0; ni < kNTiles; ++ni) {
            const int col_base = block_col + warp_n * kWarpN + ni * 8;
            const int c0 = col_base + 2 * tpair;
            float2* const p0 =
                reinterpret_cast<float2*>(c + static_cast<long long>(row_base + group) * n + c0);
            float2* const p1 = reinterpret_cast<float2*>(
                c + static_cast<long long>(row_base + group + 8) * n + c0);
            if constexpr (FUSE_BIAS) {
                // C = relu(alpha * A*B + bias[col]); no read of C, no extra pass.
                const auto relu = [](float v) { return v > 0.0f ? v : 0.0f; };
                const float b0 = bias[c0];
                const float b1 = bias[c0 + 1];
                *p0 = make_float2(relu(alpha * acc[mi][ni][0] + b0),
                                  relu(alpha * acc[mi][ni][1] + b1));
                *p1 = make_float2(relu(alpha * acc[mi][ni][2] + b0),
                                  relu(alpha * acc[mi][ni][3] + b1));
            } else if (beta == 0.0f) {
                // C is not read when beta is zero, per the BLAS contract, so an
                // uninitialized or NaN C is legal input.
                *p0 = make_float2(alpha * acc[mi][ni][0], alpha * acc[mi][ni][1]);
                *p1 = make_float2(alpha * acc[mi][ni][2], alpha * acc[mi][ni][3]);
            } else {
                const float2 v0 = *p0;
                const float2 v1 = *p1;
                *p0 = make_float2(alpha * acc[mi][ni][0] + beta * v0.x,
                                  alpha * acc[mi][ni][1] + beta * v0.y);
                *p1 = make_float2(alpha * acc[mi][ni][2] + beta * v1.x,
                                  alpha * acc[mi][ni][3] + beta * v1.y);
            }
        }
    }
#else
    // cp.async, ldmatrix and mma.sync all need sm_80. The launcher checks the
    // running device before it reaches the aligned path, so this body is never
    // launched.
    (void)a;
    (void)b;
    (void)c;
    (void)m;
    (void)n;
    (void)k;
    (void)alpha;
    (void)beta;
    (void)bias;
#endif
}

// Standalone bias plus ReLU epilogue, the unfused path: reads C, adds the column
// bias, applies ReLU, writes C. A separate memory bound pass over C.
__global__ void bias_relu_kernel(float* __restrict__ c, const float* __restrict__ bias, int m,
                                 int n) {
    const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < static_cast<long long>(m) * n) {
        const float v = c[idx] + bias[idx % n];
        c[idx] = v > 0.0f ? v : 0.0f;
    }
}

bool aligned(int m, int n, int k) {
    return m % kBM == 0 && n % kBN == 0 && k % kBK == 0 && k > 0;
}

// Opt-in shared memory ceiling of the current device, read once. Plain sm_120
// reports 101376 bytes; the 48 KB figure that limits static __shared__ is a
// different limit and does not apply once the attribute below is set.
int shared_mem_per_block_optin() {
    static const int bytes = [] {
        int device = 0;
        CKL_CUDA_CHECK(cudaGetDevice(&device));
        int value = 0;
        CKL_CUDA_CHECK(
            cudaDeviceGetAttribute(&value, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
        return value;
    }();
    return bytes;
}

// Raises the dynamic shared memory ceiling for one instantiation, once per
// process, the way the compute capability is cached. A device that cannot give
// the block 49152 bytes gets kNotSupported: there is no slower shape of this
// kernel to fall back to, and returning a wrong answer or a silently different
// path is worse than a status the caller can act on.
template <bool FUSE_BIAS>
void enable_dynamic_smem(const char* what) {
    static const bool ready = [what] {
        const int optin = shared_mem_per_block_optin();
        if (optin < static_cast<int>(kSmemBytes)) {
            throw Error(Status::kNotSupported,
                        std::string(what) + " needs " + std::to_string(kSmemBytes) +
                            " bytes of dynamic shared memory per block; this device allows " +
                            std::to_string(optin));
        }
        CKL_CUDA_CHECK(cudaFuncSetAttribute(gemm_mma_opt_kernel<FUSE_BIAS>,
                                            cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            static_cast<int>(kSmemBytes)));
        return true;
    }();
    (void)ready;
}

// The epilogue stores C two floats at a time, so the base pointer has to carry
// the alignment of a float2. Every device allocator returns at least 256 bytes
// of alignment; a hand offset pointer might not, and that is a caller error
// worth naming rather than a misaligned address fault deep in the kernel.
void require_pair_aligned(const float* c, const char* what) {
    if ((reinterpret_cast<std::uintptr_t>(c) & 0x7U) != 0U) {
        throw Error(Status::kInvalidValue,
                    std::string(what) +
                        " needs C aligned to eight bytes; its epilogue issues "
                        "64 bit stores");
    }
}

}  // namespace

void gemm_mma_opt(const __half* a, const __half* b, float* c, int m, int n, int k, float alpha,
                  float beta, cudaStream_t stream) {
    if (m <= 0 || n <= 0) {
        return;
    }
    if (!aligned(m, n, k)) {
        gemm_wmma_fp16(a, b, c, m, n, k, alpha, beta, stream);
        return;
    }
    detail::require_arch(kMinArch, "gemm_mma_opt");
    require_pair_aligned(c, "gemm_mma_opt");
    enable_dynamic_smem<false>("gemm_mma_opt");
    const dim3 block(kThreads);
    const dim3 grid(n / kBN, m / kBM);
    gemm_mma_opt_kernel<false>
        <<<grid, block, kSmemBytes, stream>>>(a, b, c, m, n, k, alpha, beta, nullptr);
    CKL_CUDA_LAST_ERROR(false);
}

void gemm_mma_opt_bias(const __half* a, const __half* b, float* c, const float* bias, int m, int n,
                       int k, float alpha, cudaStream_t stream) {
    if (m <= 0 || n <= 0) {
        return;  // no output elements, so nothing to write
    }
    if (!aligned(m, n, k)) {
        // This used to return without writing C, which reads to the caller as a
        // successful fused GEMM that produced garbage. The fused epilogue has no
        // unaligned form; the caller runs an unfused GEMM plus gemm_bias_relu
        // instead, and the dispatcher routes it that way.
        throw Error(Status::kNotSupported,
                    "gemm_mma_opt_bias needs m and n divisible by 128 and k divisible by 32; run "
                    "an unfused GEMM followed by gemm_bias_relu for other shapes");
    }
    detail::require_arch(kMinArch, "gemm_mma_opt_bias");
    require_pair_aligned(c, "gemm_mma_opt_bias");
    enable_dynamic_smem<true>("gemm_mma_opt_bias");
    const dim3 block(kThreads);
    const dim3 grid(n / kBN, m / kBM);
    gemm_mma_opt_kernel<true>
        <<<grid, block, kSmemBytes, stream>>>(a, b, c, m, n, k, alpha, 0.0f, bias);
    CKL_CUDA_LAST_ERROR(false);
}

void gemm_bias_relu(float* c, const float* bias, int m, int n, cudaStream_t stream) {
    if (m <= 0 || n <= 0) {
        return;
    }
    constexpr int kBlock = 256;
    const long long total = static_cast<long long>(m) * n;
    const int grid = static_cast<int>((total + kBlock - 1) / kBlock);
    bias_relu_kernel<<<grid, kBlock, 0, stream>>>(c, bias, m, n);
    CKL_CUDA_LAST_ERROR(false);
}

}  // namespace ckl
