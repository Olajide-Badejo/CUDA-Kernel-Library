// Top tensor core GEMM: 128 by 128 block tile, K step 32, ldmatrix fragment
// loads, cp.async double buffering, mma.sync.m16n8k16, FP16 storage with FP32
// accumulate. This is the kernel the ladder drives toward the compute bound
// gate.
//
// Round 6 showed that ldmatrix on a 64 by 64 tile still ran the L1 / TEX pipe at
// 95 percent, because a small tile loads too many bytes per fused multiply add:
// staging 2048 halves to do 64 by 64 by 16 MACs. A 128 by 128 tile with K step
// 32 stages 8192 halves for 128 by 128 by 32 MACs, roughly a fourfold better
// load to compute ratio, so the fragment traffic stops dominating and the tensor
// pipe can lead. cp.async overlaps the next stage's global load with the current
// stage's math.
//
// Eight warps (256 threads) in a 2 by 4 layout; each warp owns a 64 by 32
// region, a 4 by 4 grid of 16 by 8 mma tiles, over two K substeps of 16 per
// stage. Aligned fast path (m by 128, n by 128, k by 32); other shapes reuse the
// WMMA scalar fallback.

#include <cstdint>

#include <cuda_fp16.h>
#include <cuda_pipeline.h>

#include "ckl/gemm.hpp"

namespace ckl {

namespace {

constexpr int kBM = 128;
constexpr int kBN = 128;
constexpr int kBK = 32;
constexpr int kWarpsM = 2;
constexpr int kWarpsN = 4;
constexpr int kThreads = kWarpsM * kWarpsN * 32;  // 256
constexpr int kWarpM = kBM / kWarpsM;             // 64
constexpr int kWarpN = kBN / kWarpsN;             // 32
constexpr int kMTiles = kWarpM / 16;              // 4
constexpr int kNTiles = kWarpN / 8;               // 4
constexpr int kKSub = kBK / 16;                   // 2

__device__ inline uint32_t smem_u32(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// Shared memory swizzle. Round 7 left the kernel co limited by the shared read
// pipe, and the bank conflict counter confirmed the cause: the plain row major
// tile makes ldmatrix collide, about 0.8 conflicts per shared load wavefront.
// Permuting the eight half (sixteen byte) chunk position within each row by the
// row index spreads consecutive rows across banks. The same permutation is used
// on the cp.async store and the ldmatrix load, so it is a per row bijection and
// the data read back is always correct; only the bank mapping changes. The mask
// is NCHUNK-1 for A (four chunks per row) and 7 for B (a 128 byte bank window is
// eight chunks wide).
__device__ inline int swz(int row, int col, int stride_halves, int mask) {
    const int chunk = col >> 3;
    const int swz_chunk = chunk ^ (row & mask);
    return row * stride_halves + (swz_chunk << 3) + (col & 7);
}

constexpr int kSwzA = (kBK / 8) - 1;  // 3
constexpr int kSwzB = 7;

__device__ inline void ldmatrix_x4(uint32_t (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(smem_u32(p)));
}

__device__ inline void ldmatrix_x2_trans(uint32_t (&r)[2], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r[0]), "=r"(r[1])
                 : "r"(smem_u32(p)));
}

__device__ inline void mma_m16n8k16(float (&d)[4], const uint32_t (&a)[4],
                                    const uint32_t (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// FUSE_BIAS folds a per column bias add and a ReLU into the epilogue, while the
// accumulator is still in registers, so no extra pass over C is needed. The
// default (false) path is the plain kernel used everywhere else.
template <bool FUSE_BIAS>
__global__ __launch_bounds__(kThreads) void gemm_mma_opt_kernel(
    const __half* __restrict__ a, const __half* __restrict__ b, float* __restrict__ c,
    int m, int n, int k, float alpha, float beta, const float* __restrict__ bias) {
    __shared__ __align__(16) __half as[2][kBM * kBK];  // [buf][row][kk]
    __shared__ __align__(16) __half bs[2][kBK * kBN];  // [buf][kk][col]

    const int block_row = blockIdx.y * kBM;
    const int block_col = blockIdx.x * kBN;
    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int lane = tid % 32;
    const int warp_m = warp / kWarpsN;
    const int warp_n = warp % kWarpsN;

    float acc[kMTiles][kNTiles][4] = {};

    // Each thread stages two float4 (sixteen halves) of each of A and B per
    // stage: A is 128 by 32 (512 float4), B is 32 by 128 (512 float4).
    auto stage = [&](int buf, int kk) {
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int fa = tid + i * kThreads;      // 0..511
            const int a_row = fa / (kBK / 8);       // kBK/8 = 4 float4 per row
            const int a_col = (fa % (kBK / 8)) * 8;
            __pipeline_memcpy_async(
                &as[buf][swz(a_row, a_col, kBK, kSwzA)],
                &a[static_cast<long long>(block_row + a_row) * k + kk + a_col],
                sizeof(float4));
            const int fb = tid + i * kThreads;
            const int b_row = fb / (kBN / 8);       // kBN/8 = 16 float4 per row
            const int b_col = (fb % (kBN / 8)) * 8;
            __pipeline_memcpy_async(
                &bs[buf][swz(b_row, b_col, kBN, kSwzB)],
                &b[static_cast<long long>(kk + b_row) * n + block_col + b_col],
                sizeof(float4));
        }
        __pipeline_commit();
    };

    const int num_stages = k / kBK;
    stage(0, 0);

    for (int s = 0; s < num_stages; ++s) {
        const int cur = s & 1;
        const bool has_next = (s + 1) < num_stages;
        if (has_next) {
            stage((s + 1) & 1, (s + 1) * kBK);
        }
        __pipeline_wait_prior(has_next ? 1 : 0);
        __syncthreads();

#pragma unroll
        for (int ks = 0; ks < kKSub; ++ks) {
            const int k_off = ks * 16;
            uint32_t a_frag[kMTiles][4];
#pragma unroll
            for (int mi = 0; mi < kMTiles; ++mi) {
                const int row_base = warp_m * kWarpM + mi * 16;
                ldmatrix_x4(a_frag[mi],
                            &as[cur][swz(row_base + (lane % 16), k_off + (lane / 16) * 8, kBK, kSwzA)]);
            }
#pragma unroll
            for (int ni = 0; ni < kNTiles; ++ni) {
                const int col_base = warp_n * kWarpN + ni * 8;
                uint32_t b_frag[2];
                ldmatrix_x2_trans(b_frag, &bs[cur][swz(k_off + (lane % 16), col_base, kBN, kSwzB)]);
#pragma unroll
                for (int mi = 0; mi < kMTiles; ++mi) {
                    mma_m16n8k16(acc[mi][ni], a_frag[mi], b_frag);
                }
            }
        }
        __syncthreads();
    }

    const int group = lane / 4;
    const int tpair = lane % 4;
#pragma unroll
    for (int mi = 0; mi < kMTiles; ++mi) {
        const int row_base = block_row + warp_m * kWarpM + mi * 16;
#pragma unroll
        for (int ni = 0; ni < kNTiles; ++ni) {
            const int col_base = block_col + warp_n * kWarpN + ni * 8;
            const int r0 = row_base + group;
            const int r1 = row_base + group + 8;
            const int c0 = col_base + 2 * tpair;
            const int c1 = col_base + 2 * tpair + 1;
            const long long i00 = static_cast<long long>(r0) * n + c0;
            const long long i01 = static_cast<long long>(r0) * n + c1;
            const long long i10 = static_cast<long long>(r1) * n + c0;
            const long long i11 = static_cast<long long>(r1) * n + c1;
            if constexpr (FUSE_BIAS) {
                // C = relu(alpha * A*B + bias[col]); no read of C, no extra pass.
                const auto relu = [](float v) { return v > 0.0f ? v : 0.0f; };
                c[i00] = relu(alpha * acc[mi][ni][0] + bias[c0]);
                c[i01] = relu(alpha * acc[mi][ni][1] + bias[c1]);
                c[i10] = relu(alpha * acc[mi][ni][2] + bias[c0]);
                c[i11] = relu(alpha * acc[mi][ni][3] + bias[c1]);
            } else {
                c[i00] = alpha * acc[mi][ni][0] + beta * c[i00];
                c[i01] = alpha * acc[mi][ni][1] + beta * c[i01];
                c[i10] = alpha * acc[mi][ni][2] + beta * c[i10];
                c[i11] = alpha * acc[mi][ni][3] + beta * c[i11];
            }
        }
    }
}

// Standalone bias plus ReLU epilogue, the unfused path: reads C, adds the column
// bias, applies ReLU, writes C. A separate memory bound pass over C.
__global__ void bias_relu_kernel(float* __restrict__ c, const float* __restrict__ bias,
                                 int m, int n) {
    const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < static_cast<long long>(m) * n) {
        const float v = c[idx] + bias[idx % n];
        c[idx] = v > 0.0f ? v : 0.0f;
    }
}

bool aligned(int m, int n, int k) {
    return m % kBM == 0 && n % kBN == 0 && k % kBK == 0 && k > 0;
}

}  // namespace

void gemm_mma_opt(const __half* a, const __half* b, float* c,
                  int m, int n, int k, float alpha, float beta,
                  cudaStream_t stream) {
    if (m <= 0 || n <= 0) {
        return;
    }
    if (!aligned(m, n, k)) {
        gemm_wmma_fp16(a, b, c, m, n, k, alpha, beta, stream);
        return;
    }
    const dim3 block(kThreads);
    const dim3 grid(n / kBN, m / kBM);
    gemm_mma_opt_kernel<false><<<grid, block, 0, stream>>>(a, b, c, m, n, k, alpha, beta, nullptr);
}

void gemm_mma_opt_bias(const __half* a, const __half* b, float* c, const float* bias,
                       int m, int n, int k, float alpha, cudaStream_t stream) {
    if (m <= 0 || n <= 0 || !aligned(m, n, k)) {
        return;  // fusion study drives aligned shapes only
    }
    const dim3 block(kThreads);
    const dim3 grid(n / kBN, m / kBM);
    gemm_mma_opt_kernel<true><<<grid, block, 0, stream>>>(a, b, c, m, n, k, alpha, 0.0f, bias);
}

void gemm_bias_relu(float* c, const float* bias, int m, int n, cudaStream_t stream) {
    if (m <= 0 || n <= 0) {
        return;
    }
    constexpr int kBlock = 256;
    const long long total = static_cast<long long>(m) * n;
    const int grid = static_cast<int>((total + kBlock - 1) / kBlock);
    bias_relu_kernel<<<grid, kBlock, 0, stream>>>(c, bias, m, n);
}

}  // namespace ckl
