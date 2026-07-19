// Register blocked, vectorized FP32 GEMM. This is the decisive rung: Round 1
// showed the tiled kernel losing to naive because it dropped to 66 percent
// occupancy and bottlenecked on the shared memory (MIO) pipe while doing one
// output per thread. Here each thread computes an 8 by 8 register tile, so a
// 128 by 128 output block needs only 256 threads, and every value staged into
// shared memory is reused eight times from registers before the next shared
// read. That raises the work done per shared memory instruction and per byte of
// global traffic, which is the Volkov and Demmel move (SC 2008).
//
// Loads are float4 (128 bit) to cut the number of memory instructions. The A
// tile is transposed on the way into shared memory so the inner product reads
// contiguous rows. The vectorized path requires the block factors to divide the
// problem; the dispatcher below falls back to the tiled kernel for shapes that
// do not align, which keeps correctness on odd and degenerate sizes.

#include "ckl/gemm.hpp"

namespace ckl {

namespace {

constexpr int kBM = 128;                             // output rows per block
constexpr int kBN = 128;                             // output cols per block
constexpr int kBK = 8;                               // contraction depth per shared stage
constexpr int kTM = 8;                               // output rows per thread
constexpr int kTN = 8;                               // output cols per thread
constexpr int kThreads = (kBM / kTM) * (kBN / kTN);  // 16 * 16 = 256

// Requires m % kBM == 0, n % kBN == 0, k % kBK == 0. The dispatcher guarantees
// this; the loads and stores below assume it so they can stay branchless.
__global__ __launch_bounds__(kThreads) void gemm_register_kernel(const float* __restrict__ a,
                                                                 const float* __restrict__ b,
                                                                 float* __restrict__ c, int m,
                                                                 int n, int k, float alpha,
                                                                 float beta) {
    __shared__ float as[kBK][kBM];  // transposed: as[e][row]
    __shared__ float bs[kBK][kBN];  // bs[e][col]

    const int block_row = blockIdx.y * kBM;
    const int block_col = blockIdx.x * kBN;
    const int tid = threadIdx.x;

    // Thread's 8 by 8 output tile origin within the block.
    const int thread_row = (tid / (kBN / kTN)) * kTM;  // 0..120 step 8
    const int thread_col = (tid % (kBN / kTN)) * kTN;

    // Indices for the vectorized staging loads. A tile is 128 rows by 8 cols;
    // two float4 per row, 256 loads for 256 threads. B tile is 8 rows by 128
    // cols; 32 float4 per row, 256 loads.
    const int a_load_row = tid / 2;
    const int a_load_col = (tid % 2) * 4;
    const int b_load_row = tid / 32;
    const int b_load_col = (tid % 32) * 4;

    float acc[kTM][kTN] = {};
    float reg_a[kTM];
    float reg_b[kTN];

    for (int kk = 0; kk < k; kk += kBK) {
        // Stage A transposed into shared memory.
        const float4 av = *reinterpret_cast<const float4*>(
            &a[static_cast<long long>(block_row + a_load_row) * k + kk + a_load_col]);
        as[a_load_col + 0][a_load_row] = av.x;
        as[a_load_col + 1][a_load_row] = av.y;
        as[a_load_col + 2][a_load_row] = av.z;
        as[a_load_col + 3][a_load_row] = av.w;

        // Stage B directly.
        const float4 bv = *reinterpret_cast<const float4*>(
            &b[static_cast<long long>(kk + b_load_row) * n + block_col + b_load_col]);
        *reinterpret_cast<float4*>(&bs[b_load_row][b_load_col]) = bv;

        __syncthreads();

#pragma unroll
        for (int e = 0; e < kBK; ++e) {
#pragma unroll
            for (int i = 0; i < kTM; ++i) {
                reg_a[i] = as[e][thread_row + i];
            }
#pragma unroll
            for (int j = 0; j < kTN; ++j) {
                reg_b[j] = bs[e][thread_col + j];
            }
#pragma unroll
            for (int i = 0; i < kTM; ++i) {
#pragma unroll
                for (int j = 0; j < kTN; ++j) {
                    acc[i][j] += reg_a[i] * reg_b[j];
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < kTM; ++i) {
        const int row = block_row + thread_row + i;
#pragma unroll
        for (int j = 0; j < kTN; ++j) {
            const int col = block_col + thread_col + j;
            const long long idx = static_cast<long long>(row) * n + col;
            c[idx] = alpha * acc[i][j] + beta * c[idx];
        }
    }
}

bool aligned(int m, int n, int k) {
    return m % kBM == 0 && n % kBN == 0 && k % kBK == 0 && m > 0 && n > 0 && k > 0;
}

}  // namespace

void gemm_register(const float* a, const float* b, float* c, int m, int n, int k, float alpha,
                   float beta, cudaStream_t stream) {
    if (m <= 0 || n <= 0) {
        return;
    }
    if (!aligned(m, n, k)) {
        // Odd, sub tile, or zero k shapes: the vectorized kernel's branchless
        // loads assume alignment, so use the boundary safe tiled kernel.
        gemm_tiled(a, b, c, m, n, k, alpha, beta, stream);
        return;
    }
    const dim3 block(kThreads);
    const dim3 grid(n / kBN, m / kBM);
    gemm_register_kernel<<<grid, block, 0, stream>>>(a, b, c, m, n, k, alpha, beta);
}

}  // namespace ckl
