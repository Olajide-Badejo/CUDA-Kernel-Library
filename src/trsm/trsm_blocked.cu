// Blocked TRSM. The matrix is swept in diagonal blocks of width BS. For each
// block: solve the small BS by BS triangular system for every right hand side
// (limited work, done by substitution on the block), then subtract that block's
// contribution from all trailing rows. That subtraction, B_below -= L_below,k *
// X_k, is a dense matrix multiply, and it is where almost all the flops live. So
// the blocked solve is a thin substitution wrapper around a GEMM, which is how
// BLAS libraries make TRSM run at GEMM speed.
//
// alpha is applied once by scaling B up front, so the block solves and the update
// run with alpha folded in.

#include "ckl/trsm.hpp"

namespace ckl {

namespace {

constexpr int kBS = 32;

__global__ void scale_kernel(float* __restrict__ b, long long count, float alpha) {
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) {
        b[i] *= alpha;
    }
}

// Solve one diagonal block L[k:k+kb, k:k+kb] X = B[k:k+kb, :] in place. One thread
// per right hand side column, substitution within the block.
__global__ void block_solve_kernel(const float* __restrict__ a, float* __restrict__ b, int m, int n,
                                   int k, int kb) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= n) {
        return;
    }
    for (int ii = 0; ii < kb; ++ii) {
        const int i = k + ii;
        float s = b[static_cast<long long>(i) * n + col];
        for (int jj = 0; jj < ii; ++jj) {
            const int j = k + jj;
            s -= a[static_cast<long long>(i) * m + j] * b[static_cast<long long>(j) * n + col];
        }
        b[static_cast<long long>(i) * n + col] = s / a[static_cast<long long>(i) * m + i];
    }
}

// Trailing update: B[i, c] -= sum_{j in block} L[i, j] * X[j, c] for i below the
// block. This is the GEMM step; one thread per trailing (row, col) element,
// reading the just solved block of X from B.
__global__ void trailing_update_kernel(const float* __restrict__ a, float* __restrict__ b, int m,
                                       int n, int k, int kb) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = (k + kb) + blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= n || i >= m) {
        return;
    }
    float s = 0.0f;
    for (int jj = 0; jj < kb; ++jj) {
        const int j = k + jj;
        s += a[static_cast<long long>(i) * m + j] * b[static_cast<long long>(j) * n + col];
    }
    b[static_cast<long long>(i) * n + col] -= s;
}

}  // namespace

void trsm_blocked(const float* a, float* b, int m, int n, float alpha, cudaStream_t stream) {
    if (m <= 0 || n <= 0) {
        return;
    }
    const long long count = static_cast<long long>(m) * n;
    if (alpha != 1.0f) {
        constexpr int kThreads = 256;
        const int grid = static_cast<int>((count + kThreads - 1) / kThreads);
        scale_kernel<<<grid, kThreads, 0, stream>>>(b, count, alpha);
    }

    for (int k = 0; k < m; k += kBS) {
        const int kb = (m - k < kBS) ? (m - k) : kBS;
        {
            constexpr int kBlock = 128;
            const int grid = (n + kBlock - 1) / kBlock;
            block_solve_kernel<<<grid, kBlock, 0, stream>>>(a, b, m, n, k, kb);
        }
        const int rows_below = m - (k + kb);
        if (rows_below > 0) {
            const dim3 block(32, 8);
            const dim3 grid((n + block.x - 1) / block.x, (rows_below + block.y - 1) / block.y);
            trailing_update_kernel<<<grid, block, 0, stream>>>(a, b, m, n, k, kb);
        }
    }
}

}  // namespace ckl
