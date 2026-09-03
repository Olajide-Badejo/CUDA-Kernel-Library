#pragma once

/**
 * @file gemv.hpp
 * @brief GEMV interface. Row major, single precision: y = alpha * (A * x) + beta * y.
 *
 * A is m by n, x is length n, y is length m.
 *
 * GEMV is a memory bound kernel: every matrix element is read once and used in a
 * single multiply add, so the arithmetic intensity is fixed near 2 FLOPs per 4
 * bytes no matter how the work is tiled. The three variants exist to show that
 * the win comes from reading A efficiently (coalescing, vectorization), not from
 * arithmetic, and the docs back that with roofline evidence. cuBLAS SGEMV is the
 * baseline.
 *
 * Every entry point here is asynchronous on the stream it is given.
 */

#include <cuda_runtime.h>

#include "ckl/ckl_export.h"

namespace ckl {

/**
 * @brief One thread per output row, scalar loads across the row.
 * @param a Device pointer to A, m by n, row major, leading dimension n.
 * @param x Device pointer to x, length n.
 * @param y Device pointer to y, length m, updated in place.
 * @param m Rows of A and length of y.
 * @param n Columns of A and length of x.
 * @param alpha Scale on the product.
 * @param beta Scale on the incoming y; when zero, y is not read.
 * @param stream Stream to enqueue on; nullptr means the default stream.
 * @note Simple and coalescing poor: adjacent threads read down a column of A,
 *       one row apart in memory. It is the honest baseline for this family.
 */
CKL_EXPORT void gemv_naive(const float* a, const float* x, float* y, int m, int n, float alpha,
                           float beta, cudaStream_t stream = nullptr);

/**
 * @brief One warp per output row with a shfl reduction.
 * @param a Device pointer to A, m by n, row major.
 * @param x Device pointer to x, length n.
 * @param y Device pointer to y, length m, updated in place.
 * @param m Rows of A and length of y.
 * @param n Columns of A and length of x.
 * @param alpha Scale on the product.
 * @param beta Scale on the incoming y; when zero, y is not read.
 * @param stream Stream to enqueue on; nullptr means the default stream.
 * @note Each lane strides across the row and the partial sums are combined with
 *       a shfl reduction, so adjacent lanes read adjacent A elements and the row
 *       read coalesces.
 */
CKL_EXPORT void gemv_warp(const float* a, const float* x, float* y, int m, int n, float alpha,
                          float beta, cudaStream_t stream = nullptr);

/**
 * @brief One warp per row with float4 loads of A and x.
 * @param a Device pointer to A, m by n, row major.
 * @param x Device pointer to x, length n.
 * @param y Device pointer to y, length m, updated in place.
 * @param m Rows of A and length of y.
 * @param n Columns of A and length of x; a multiple of 4 takes the fast path.
 * @param alpha Scale on the product.
 * @param beta Scale on the incoming y; when zero, y is not read.
 * @param stream Stream to enqueue on; nullptr means the default stream.
 * @note Quarters the number of memory instructions. Other n fall back to
 *       gemv_warp. The float4 loads need A and x 16 byte aligned.
 */
CKL_EXPORT void gemv_vectorized(const float* a, const float* x, float* y, int m, int n, float alpha,
                                float beta, cudaStream_t stream = nullptr);

/**
 * @brief cuBLAS SGEMV oracle and baseline, producing the same row major result.
 * @param a Device pointer to A, m by n, row major.
 * @param x Device pointer to x, length n.
 * @param y Device pointer to y, length m, updated in place.
 * @param m Rows of A and length of y.
 * @param n Columns of A and length of x.
 * @param alpha Scale on the product.
 * @param beta Scale on the incoming y.
 * @param stream Stream to enqueue on; nullptr means the default stream.
 * @note Runs on the process wide default Context and takes its lock.
 */
CKL_EXPORT void gemv_cublas(const float* a, const float* x, float* y, int m, int n, float alpha,
                            float beta, cudaStream_t stream = nullptr);

}  // namespace ckl
