// Vector CSR SpMV: a lane group of 2, 4, 8, 16 or 32 threads works one row.
//
// The warp per row kernel spends a full 32 lane warp on every row. On the
// regular classes of the suite that is mostly waste: parabolic_fem averages 7
// nonzeros per row and mc2depi has exactly 4, so 25 to 28 lanes hold nothing and
// the kernel still runs all five shfl steps to combine them. Narrowing the group
// to the width the row degree actually needs puts those lanes on other rows
// instead, and drops the reduction to log2(width) steps.
//
// The width is a plan decision, not a kernel decision: SpmvPlan builds a
// histogram of nonzeros per row once and hands the answer down. The kernel is
// templated on it so the reduction unrolls to a fixed depth and the compiler
// knows the group is a power of two subset of a warp, which is what makes the
// shfl mask a constant.

#include "ckl/cuda_check.hpp"
#include "ckl/sparse.hpp"

namespace ckl {

namespace {

// Rows worked by one block. 128 threads is four warps, which is enough to hide
// the dependent col_idx then x load without asking for more registers than the
// tail of a long row needs.
constexpr int kBlock = 128;

// Sum across a lane group of Width consecutive lanes. __shfl_down_sync with a
// full warp mask is correct here because every lane of the warp participates:
// the groups are aligned power of two slices of the warp, so a shuffle of
// offset below Width never crosses a group boundary.
template <int Width>
__device__ inline float group_reduce_sum(float v) {
#pragma unroll
    for (int offset = Width / 2; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, offset, Width);
    }
    return v;
}

template <int Width>
__global__ void spmv_csr_vector_kernel(const int* __restrict__ row_ptr,
                                       const int* __restrict__ col_idx,
                                       const float* __restrict__ values,
                                       const float* __restrict__ x, float* __restrict__ y, int m,
                                       float alpha, float beta) {
    const int thread = static_cast<int>(blockIdx.x) * kBlock + static_cast<int>(threadIdx.x);
    const int row = thread / Width;
    const int lane = thread % Width;
    if (row >= m) {
        return;
    }
    const int start = row_ptr[row];
    const int end = row_ptr[row + 1];
    float sum = 0.0f;
    for (int k = start + lane; k < end; k += Width) {
        sum += values[k] * x[col_idx[k]];
    }
    sum = group_reduce_sum<Width>(sum);
    if (lane == 0) {
        // y is not read when beta is zero, per the BLAS contract.
        y[row] = (beta == 0.0f) ? alpha * sum : alpha * sum + beta * y[row];
    }
}

template <int Width>
void launch(const int* row_ptr, const int* col_idx, const float* values, const float* x, float* y,
            int m, float alpha, float beta, cudaStream_t stream) {
    const long long threads = static_cast<long long>(m) * Width;
    const int grid = static_cast<int>((threads + kBlock - 1) / kBlock);
    spmv_csr_vector_kernel<Width>
        <<<grid, kBlock, 0, stream>>>(row_ptr, col_idx, values, x, y, m, alpha, beta);
    CKL_CUDA_LAST_ERROR(false);
}

}  // namespace

void spmv_csr_vector(const int* row_ptr, const int* col_idx, const float* values, const float* x,
                     float* y, int m, int n, int nnz, float alpha, float beta, int width,
                     cudaStream_t stream) {
    (void)n;
    (void)nnz;
    if (m <= 0) {
        return;
    }
    // Rounding up rather than refusing: every width in the set is correct on
    // every matrix, so an out of range request costs performance and never an
    // answer. The plan never sends anything but a member of the set.
    if (width <= 2) {
        launch<2>(row_ptr, col_idx, values, x, y, m, alpha, beta, stream);
    } else if (width <= 4) {
        launch<4>(row_ptr, col_idx, values, x, y, m, alpha, beta, stream);
    } else if (width <= 8) {
        launch<8>(row_ptr, col_idx, values, x, y, m, alpha, beta, stream);
    } else if (width <= 16) {
        launch<16>(row_ptr, col_idx, values, x, y, m, alpha, beta, stream);
    } else {
        launch<32>(row_ptr, col_idx, values, x, y, m, alpha, beta, stream);
    }
}

}  // namespace ckl
