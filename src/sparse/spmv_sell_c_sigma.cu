// SELL-C-sigma SpMV: sliced ELLPACK with C = 32 and a sort window of sigma rows.
//
// Both CSR kernels walk values and col_idx in row order, so the moment a lane
// group crosses a row boundary its reads stop being contiguous, and both of them
// reload row_ptr in the inner loop to know where that boundary is. SELL removes
// the indirection: rows are sorted by length inside a window of sigma rows and
// packed into slices of C rows padded to the slice maximum, stored so that the
// C values a slice needs at step j sit next to each other. One warp per slice,
// one lane per row, and every step is a single 128 byte column read with no
// offset lookup at all.
//
// What it costs is the padding. A slice is as long as its longest row, so the
// arrays hold nnz_padded entries rather than nnz, and a larger sigma cuts that
// padding by sorting over a wider window while scattering the y write further
// and adding a permutation. That trade is exactly why nnz_padded over nnz rides
// on every benchmark row this variant produces: a variant that wins ten percent
// at a padding ratio of two is not a win.
//
// The conversion belongs to SpmvPlan, which does it once per matrix on the host.
// This file is only the kernel.

#include "ckl/cuda_check.hpp"
#include "ckl/sparse.hpp"

namespace ckl {

namespace {

constexpr int kSliceHeight = 32;
constexpr int kWarpsPerBlock = 4;
constexpr int kBlock = kWarpsPerBlock * kSliceHeight;

__global__ void spmv_sell_kernel(const int* __restrict__ slice_ptr, const int* __restrict__ col_idx,
                                 const float* __restrict__ values, const int* __restrict__ perm,
                                 const int* __restrict__ row_length, const float* __restrict__ x,
                                 float* __restrict__ y, int m, int slices, float alpha,
                                 float beta) {
    const int thread = static_cast<int>(blockIdx.x) * kBlock + static_cast<int>(threadIdx.x);
    const int slice = thread / kSliceHeight;
    const int lane = thread % kSliceHeight;
    if (slice >= slices) {
        return;
    }
    const int slot = slice * kSliceHeight + lane;
    const int row = perm[slot];
    // The last slice is padded up to C rows when m is not a multiple of 32, and
    // those slots carry no row at all.
    if (row < 0 || row >= m) {
        return;
    }
    // The true length rather than the slice length: the padded tail holds zeros
    // and would contribute nothing, but reading it would still fetch x at a
    // column index that means nothing, and a NaN parked there would propagate.
    const int len = row_length[slot];
    const int base = slice_ptr[slice] + lane;
    float sum = 0.0f;
    for (int j = 0; j < len; ++j) {
        const int at = base + j * kSliceHeight;
        sum += values[at] * x[col_idx[at]];
    }
    // y is not read when beta is zero, per the BLAS contract.
    y[row] = (beta == 0.0f) ? alpha * sum : alpha * sum + beta * y[row];
}

}  // namespace

void spmv_sell_c_sigma(const int* slice_ptr, const int* col_idx, const float* values,
                       const int* perm, const int* row_length, const float* x, float* y, int m,
                       int slices, float alpha, float beta, cudaStream_t stream) {
    if (m <= 0 || slices <= 0) {
        return;
    }
    const long long threads = static_cast<long long>(slices) * kSliceHeight;
    const int grid = static_cast<int>((threads + kBlock - 1) / kBlock);
    spmv_sell_kernel<<<grid, kBlock, 0, stream>>>(slice_ptr, col_idx, values, perm, row_length, x,
                                                  y, m, slices, alpha, beta);
    CKL_CUDA_LAST_ERROR(false);
}

}  // namespace ckl
