// Double buffered FP32 GEMM using cp.async (asynchronous global to shared copy,
// Ampere and later; CUDA C++ Programming Guide). Round 2 left the register
// kernel bound by the shared read pipe with global loads not fully hidden. Here
// two shared buffers are kept: while the SM computes on the current tile, the
// next tile streams in through cp.async, so the global load latency overlaps the
// math instead of stalling in front of it.
//
// Compared with the register kernel this stores the A tile in natural
// (non transposed) order. That matters for cp.async, which copies a contiguous
// run of bytes and cannot transpose on the way in, and it costs nothing on the
// read side: every thread of a warp shares the same A row index, so the A shared
// reads broadcast and do not add bank conflicts. B keeps its natural layout too.
//
// Same alignment contract as the register kernel (m by 128, n by 128, k by 8);
// the dispatcher falls back to the tiled kernel otherwise.
//
// cp.async needs compute capability 8.0, so the device body is compiled only for
// sm_80 and newer and the launcher refuses on anything older. An empty kernel
// that launches and writes nothing would be the worst kind of silent fallback,
// which is why the check is on the host side and returns rather than launching.

#include <cuda_pipeline.h>

#include "ckl/context.hpp"
#include "ckl/cuda_check.hpp"
#include "ckl/gemm.hpp"

namespace ckl {

namespace {

constexpr int kBM = 128;
constexpr int kBN = 128;
constexpr int kBK = 8;
constexpr int kTM = 8;
constexpr int kTN = 8;
constexpr int kThreads = (kBM / kTM) * (kBN / kTN);  // 256
constexpr int kMinArch = 80;                         // cp.async

__global__ __launch_bounds__(kThreads) void gemm_cp_async_kernel(const float* __restrict__ a,
                                                                 const float* __restrict__ b,
                                                                 float* __restrict__ c, int m,
                                                                 int n, int k, float alpha,
                                                                 float beta) {
#if __CUDA_ARCH__ >= 800
    // Double buffered. A stored [row][e] naturally, B stored [e][col].
    __shared__ __align__(16) float as[2][kBM * kBK];
    __shared__ __align__(16) float bs[2][kBK * kBN];

    const int block_row = blockIdx.y * kBM;
    const int block_col = blockIdx.x * kBN;
    const int tid = threadIdx.x;

    const int thread_row = (tid / (kBN / kTN)) * kTM;
    const int thread_col = (tid % (kBN / kTN)) * kTN;

    // One float4 per thread stages each of A and B (256 threads, 256 float4).
    const int a_row = tid / 2;
    const int a_col = (tid % 2) * 4;
    const int b_row = tid / 32;
    const int b_col = (tid % 32) * 4;

    auto stage = [&](int buf, int kk) {
        __pipeline_memcpy_async(&as[buf][a_row * kBK + a_col],
                                &a[static_cast<long long>(block_row + a_row) * k + kk + a_col],
                                sizeof(float4));
        __pipeline_memcpy_async(&bs[buf][b_row * kBN + b_col],
                                &b[static_cast<long long>(kk + b_row) * n + block_col + b_col],
                                sizeof(float4));
        __pipeline_commit();
    };

    float acc[kTM][kTN] = {};
    float reg_a[kTM];
    float reg_b[kTN];

    const int num_tiles = k / kBK;
    stage(0, 0);

    for (int t = 0; t < num_tiles; ++t) {
        const int cur = t & 1;
        const bool has_next = (t + 1) < num_tiles;
        if (has_next) {
            stage((t + 1) & 1, (t + 1) * kBK);
        }
        // Leave the next tile's group in flight (1) while waiting for the
        // current tile; on the last tile there is no next group, so wait for all.
        __pipeline_wait_prior(has_next ? 1 : 0);
        __syncthreads();

#pragma unroll
        for (int e = 0; e < kBK; ++e) {
#pragma unroll
            for (int i = 0; i < kTM; ++i) {
                reg_a[i] = as[cur][(thread_row + i) * kBK + e];
            }
#pragma unroll
            for (int j = 0; j < kTN; ++j) {
                reg_b[j] = bs[cur][e * kBN + thread_col + j];
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

    // C is not read when beta is zero: BLAS allows an uninitialized or NaN C in
    // that case, and one uniform branch beats 64 multiplications by zero.
    if (beta == 0.0f) {
#pragma unroll
        for (int i = 0; i < kTM; ++i) {
            const int row = block_row + thread_row + i;
#pragma unroll
            for (int j = 0; j < kTN; ++j) {
                const int col = block_col + thread_col + j;
                c[static_cast<long long>(row) * n + col] = alpha * acc[i][j];
            }
        }
        return;
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
#else
    // Below sm_80 there is no cp.async, so there is no kernel. The launcher
    // never gets here: it checks the running device first.
    (void)a;
    (void)b;
    (void)c;
    (void)m;
    (void)n;
    (void)k;
    (void)alpha;
    (void)beta;
#endif
}

bool aligned(int m, int n, int k) {
    return m % kBM == 0 && n % kBN == 0 && k % kBK == 0 && m > 0 && n > 0 && k > 0;
}

}  // namespace

void gemm_cp_async(const float* a, const float* b, float* c, int m, int n, int k, float alpha,
                   float beta, cudaStream_t stream) {
    if (m <= 0 || n <= 0) {
        return;
    }
    if (!aligned(m, n, k)) {
        gemm_tiled(a, b, c, m, n, k, alpha, beta, stream);
        return;
    }
    detail::require_arch(kMinArch, "gemm_cp_async");
    const dim3 block(kThreads);
    const dim3 grid(n / kBN, m / kBM);
    gemm_cp_async_kernel<<<grid, block, 0, stream>>>(a, b, c, m, n, k, alpha, beta);
    CKL_CUDA_LAST_ERROR(false);
}

}  // namespace ckl
