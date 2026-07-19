#pragma once

// Triangular solve with multiple right hand sides. Solves L X = alpha B for X,
// where L is an m by m lower triangular matrix with non unit diagonal, and B (and
// the solution X, written in place over B) is m by n. Row major, single
// precision. Left side, no transpose, lower triangular: one concrete case is
// enough to show the naive versus blocked contrast and to check against cuBLAS.
//
// The blocked variant is the point: it solves small diagonal blocks directly and
// pushes the bulk of the work into a trailing matrix update, which is a GEMM. That
// is how BLAS turns TRSM into mostly GEMM.

#include <cuda_runtime.h>

namespace ckl {

// Naive forward substitution: one thread per right hand side column, sequential
// down the rows. Correct and simple, limited parallelism (n columns).
void trsm_naive(const float* a, float* b, int m, int n, float alpha,
                cudaStream_t stream = nullptr);

// Blocked: scale by alpha, then for each diagonal block solve the block system
// and subtract its contribution from the trailing rows with a GEMM style update.
void trsm_blocked(const float* a, float* b, int m, int n, float alpha,
                  cudaStream_t stream = nullptr);

// cuBLAS STRSM oracle and baseline, same row major result.
void trsm_cublas(const float* a, float* b, int m, int n, float alpha,
                 cudaStream_t stream = nullptr);

}  // namespace ckl
