// Naive TRSM by forward substitution: one thread per right hand side column.
// Each thread walks the rows top to bottom, and by the time it reaches row i the
// solution for rows 0 to i-1 is already written back into B, so it can subtract
// their contribution. Parallelism is limited to the n columns, which is the
// weakness the blocked variant addresses by turning most of the work into a GEMM.

#include "ckl/trsm.hpp"

namespace ckl {

namespace {

__global__ void trsm_naive_kernel(const float* __restrict__ a, float* __restrict__ b,
                                  int m, int n, float alpha) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= n) {
        return;
    }
    for (int i = 0; i < m; ++i) {
        float s = alpha * b[static_cast<long long>(i) * n + col];
        for (int j = 0; j < i; ++j) {
            s -= a[static_cast<long long>(i) * m + j] * b[static_cast<long long>(j) * n + col];
        }
        b[static_cast<long long>(i) * n + col] = s / a[static_cast<long long>(i) * m + i];
    }
}

}  // namespace

void trsm_naive(const float* a, float* b, int m, int n, float alpha, cudaStream_t stream) {
    if (m <= 0 || n <= 0) {
        return;
    }
    constexpr int kBlock = 128;
    const int grid = (n + kBlock - 1) / kBlock;
    trsm_naive_kernel<<<grid, kBlock, 0, stream>>>(a, b, m, n, alpha);
}

}  // namespace ckl
