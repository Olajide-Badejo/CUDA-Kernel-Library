#pragma once

/**
 * @file sparse.hpp
 * @brief CSR sparse matrix vector product, y = alpha * (A * x) + beta * y.
 *
 * A is stored in compressed sparse row form: row_ptr has length m+1, col_idx and
 * values have length nnz. Two hand written variants plus cuSPARSE as oracle and
 * baseline.
 *
 * The interesting case is a skewed row degree distribution: with uniform rows the
 * thread per row and warp per row kernels perform about the same, but with a few
 * very long rows the thread per row kernel serializes on those rows while the warp
 * per row kernel spreads each row across 32 lanes. The tests use a skewed matrix
 * so that difference is visible.
 */

#include <cuda_runtime.h>

#include "ckl/ckl_export.h"

namespace ckl {

/**
 * @brief One thread per row.
 * @param row_ptr Device pointer to the CSR row offsets, length m+1.
 * @param col_idx Device pointer to the CSR column indices, length nnz.
 * @param values Device pointer to the CSR values, length nnz.
 * @param x Device pointer to x, length n.
 * @param y Device pointer to y, length m, updated in place.
 * @param m Rows of A and length of y.
 * @param n Columns of A and length of x.
 * @param nnz Number of stored nonzeros.
 * @param alpha Scale on the product.
 * @param beta Scale on the incoming y; when zero, y is not read.
 * @param stream Stream to enqueue on; nullptr means the default stream.
 * @note Simple, but a long row stalls its whole warp.
 * @note Asynchronous. Synchronize on stream before reading y.
 */
CKL_EXPORT void spmv_csr_naive(const int* row_ptr, const int* col_idx, const float* values, const float* x,
                    float* y, int m, int n, int nnz, float alpha, float beta,
                    cudaStream_t stream = nullptr);

/**
 * @brief One warp per row with a shfl reduction.
 * @param row_ptr Device pointer to the CSR row offsets, length m+1.
 * @param col_idx Device pointer to the CSR column indices, length nnz.
 * @param values Device pointer to the CSR values, length nnz.
 * @param x Device pointer to x, length n.
 * @param y Device pointer to y, length m, updated in place.
 * @param m Rows of A and length of y.
 * @param n Columns of A and length of x.
 * @param nnz Number of stored nonzeros.
 * @param alpha Scale on the product.
 * @param beta Scale on the incoming y; when zero, y is not read.
 * @param stream Stream to enqueue on; nullptr means the default stream.
 * @note The row's nonzeros are shared across 32 lanes, so a few long rows do not
 *       serialize.
 */
CKL_EXPORT void spmv_csr_warp(const int* row_ptr, const int* col_idx, const float* values, const float* x,
                   float* y, int m, int n, int nnz, float alpha, float beta,
                   cudaStream_t stream = nullptr);

/**
 * @brief cuSPARSE cusparseSpMV oracle and baseline, CSR and FP32, same result.
 * @param row_ptr Device pointer to the CSR row offsets, length m+1.
 * @param col_idx Device pointer to the CSR column indices, length nnz.
 * @param values Device pointer to the CSR values, length nnz.
 * @param x Device pointer to x, length n.
 * @param y Device pointer to y, length m, updated in place.
 * @param m Rows of A and length of y.
 * @param n Columns of A and length of x.
 * @param nnz Number of stored nonzeros.
 * @param alpha Scale on the product.
 * @param beta Scale on the incoming y.
 * @param stream Stream to enqueue on; nullptr means the default stream.
 * @note Runs on the process wide default Context and takes its lock. cuSPARSE
 *       allocates its own scratch inside this call.
 */
CKL_EXPORT void spmv_cusparse(const int* row_ptr, const int* col_idx, const float* values, const float* x,
                   float* y, int m, int n, int nnz, float alpha, float beta,
                   cudaStream_t stream = nullptr);

}  // namespace ckl
