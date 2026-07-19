#pragma once

// CSR sparse matrix vector product, y = alpha * (A * x) + beta * y, A stored in
// compressed sparse row form (row_ptr length m+1, col_idx and values length nnz).
// Two hand written variants plus cuSPARSE as oracle and baseline.
//
// The interesting case is a skewed row degree distribution: with uniform rows the
// thread per row and warp per row kernels perform about the same, but with a few
// very long rows the thread per row kernel serializes on those rows while the warp
// per row kernel spreads each row across 32 lanes. The tests use a skewed matrix
// so that difference is visible.

#include <cuda_runtime.h>

namespace ckl {

// One thread per row: simple, but a long row stalls its whole warp.
void spmv_csr_naive(const int* row_ptr, const int* col_idx, const float* values, const float* x,
                    float* y, int m, int n, int nnz, float alpha, float beta,
                    cudaStream_t stream = nullptr);

// One warp per row with a shfl reduction: the row's nonzeros are shared across 32
// lanes, so a few long rows do not serialize.
void spmv_csr_warp(const int* row_ptr, const int* col_idx, const float* values, const float* x,
                   float* y, int m, int n, int nnz, float alpha, float beta,
                   cudaStream_t stream = nullptr);

// cuSPARSE cusparseSpMV oracle and baseline (CSR, FP32), same result.
void spmv_cusparse(const int* row_ptr, const int* col_idx, const float* values, const float* x,
                   float* y, int m, int n, int nnz, float alpha, float beta,
                   cudaStream_t stream = nullptr);

}  // namespace ckl
