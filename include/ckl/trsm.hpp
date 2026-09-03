#pragma once

/**
 * @file trsm.hpp
 * @brief Triangular solve with multiple right hand sides: L X = alpha B.
 *
 * L is an m by m lower triangular matrix with non unit diagonal, and B (and the
 * solution X, written in place over B) is m by n. Row major, single precision.
 * Left side, no transpose, lower triangular: one concrete case is enough to show
 * the naive versus blocked contrast and to check against cuBLAS.
 *
 * The blocked variant is the point: it solves small diagonal blocks directly and
 * pushes the bulk of the work into a trailing matrix update, which is a GEMM. That
 * is how BLAS turns TRSM into mostly GEMM.
 */

#include <cuda_runtime.h>

#include "ckl/ckl_export.h"

namespace ckl {

/**
 * @brief Naive forward substitution: one thread per right hand side column.
 * @param a Device pointer to L, m by m, row major, lower triangular, non unit diagonal.
 * @param b Device pointer to B, m by n, row major; overwritten by X.
 * @param m Order of L and rows of B.
 * @param n Right hand sides, the columns of B.
 * @param alpha Scale applied to B before the solve.
 * @param stream Stream to enqueue on; nullptr means the default stream.
 * @note Correct and simple, with limited parallelism: n columns, worked
 *       sequentially down the rows. Kept as the honest baseline.
 * @note Asynchronous. Synchronize on stream before reading B.
 */
CKL_EXPORT void trsm_naive(const float* a, float* b, int m, int n, float alpha,
                           cudaStream_t stream = nullptr);

/**
 * @brief Blocked triangular solve: diagonal block solves plus GEMM style trailing updates.
 * @param a Device pointer to L, m by m, row major, lower triangular, non unit diagonal.
 * @param b Device pointer to B, m by n, row major; overwritten by X.
 * @param m Order of L and rows of B.
 * @param n Right hand sides, the columns of B.
 * @param alpha Scale applied to B before the solve.
 * @param stream Stream to enqueue on; nullptr means the default stream.
 * @note Scales by alpha, then for each diagonal block solves the block system and
 *       subtracts its contribution from the trailing rows.
 */
CKL_EXPORT void trsm_blocked(const float* a, float* b, int m, int n, float alpha,
                             cudaStream_t stream = nullptr);

/**
 * @brief cuBLAS STRSM oracle and baseline, same row major result.
 * @param a Device pointer to L, m by m, row major, lower triangular, non unit diagonal.
 * @param b Device pointer to B, m by n, row major; overwritten by X.
 * @param m Order of L and rows of B.
 * @param n Right hand sides, the columns of B.
 * @param alpha Scale applied to B before the solve.
 * @param stream Stream to enqueue on; nullptr means the default stream.
 * @note Runs on the process wide default Context and takes its lock.
 */
CKL_EXPORT void trsm_cublas(const float* a, float* b, int m, int n, float alpha,
                            cudaStream_t stream = nullptr);

}  // namespace ckl
