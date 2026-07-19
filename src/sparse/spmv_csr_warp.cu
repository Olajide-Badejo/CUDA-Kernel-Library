// Warp per row CSR SpMV. Each warp owns one row and its 32 lanes stride across
// that row's nonzeros, then a shfl_down reduction sums the lane partials. A long
// row is now shared across 32 lanes instead of stalling one thread, so a skewed
// degree distribution no longer serializes the kernel. The lane strided read of
// col_idx and values also coalesces within a row.

#include "ckl/sparse.hpp"

namespace ckl {

namespace {

__device__ inline float warp_reduce_sum(float v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    return v;
}

__global__ void spmv_csr_warp_kernel(const int* __restrict__ row_ptr,
                                     const int* __restrict__ col_idx,
                                     const float* __restrict__ values,
                                     const float* __restrict__ x, float* __restrict__ y,
                                     int m, float alpha, float beta) {
    const int row = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane = threadIdx.x % 32;
    if (row >= m) {
        return;
    }
    const int start = row_ptr[row];
    const int end = row_ptr[row + 1];
    float sum = 0.0f;
    for (int k = start + lane; k < end; k += 32) {
        sum += values[k] * x[col_idx[k]];
    }
    sum = warp_reduce_sum(sum);
    if (lane == 0) {
        y[row] = alpha * sum + beta * y[row];
    }
}

}  // namespace

void spmv_csr_warp(const int* row_ptr, const int* col_idx, const float* values,
                   const float* x, float* y, int m, int n, int nnz,
                   float alpha, float beta, cudaStream_t stream) {
    (void)n;
    (void)nnz;
    if (m <= 0) {
        return;
    }
    constexpr int kWarpsPerBlock = 4;
    constexpr int kBlock = kWarpsPerBlock * 32;
    const int grid = (m + kWarpsPerBlock - 1) / kWarpsPerBlock;
    spmv_csr_warp_kernel<<<grid, kBlock, 0, stream>>>(row_ptr, col_idx, values, x, y,
                                                      m, alpha, beta);
}

}  // namespace ckl
