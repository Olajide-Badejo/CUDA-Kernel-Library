// Naive CSR SpMV: one thread per row. Each thread walks its own row's nonzeros.
// Correct and simple, but when the row degree distribution is skewed the threads
// assigned the long rows run far longer than their warp neighbors, so the warp
// retires at the speed of its slowest row. The warp per row variant fixes that.

#include "ckl/sparse.hpp"

namespace ckl {

namespace {

__global__ void spmv_csr_naive_kernel(const int* __restrict__ row_ptr,
                                      const int* __restrict__ col_idx,
                                      const float* __restrict__ values, const float* __restrict__ x,
                                      float* __restrict__ y, int m, float alpha, float beta) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m) {
        return;
    }
    const int start = row_ptr[row];
    const int end = row_ptr[row + 1];
    float sum = 0.0f;
    for (int k = start; k < end; ++k) {
        sum += values[k] * x[col_idx[k]];
    }
    y[row] = alpha * sum + beta * y[row];
}

}  // namespace

void spmv_csr_naive(const int* row_ptr, const int* col_idx, const float* values, const float* x,
                    float* y, int m, int n, int nnz, float alpha, float beta, cudaStream_t stream) {
    (void)n;
    (void)nnz;
    if (m <= 0) {
        return;
    }
    constexpr int kBlock = 128;
    const int grid = (m + kBlock - 1) / kBlock;
    spmv_csr_naive_kernel<<<grid, kBlock, 0, stream>>>(row_ptr, col_idx, values, x, y, m, alpha,
                                                       beta);
}

}  // namespace ckl
