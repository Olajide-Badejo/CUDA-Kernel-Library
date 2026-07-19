#pragma once

// GEMV interface. Row major, single precision: A is m by n, x is length n, y is
// length m, and y = alpha * (A * x) + beta * y.
//
// GEMV is a memory bound kernel: every matrix element is read once and used in a
// single multiply add, so the arithmetic intensity is fixed near 2 FLOPs per 4
// bytes no matter how the work is tiled. The three variants exist to show that
// the win comes from reading A efficiently (coalescing, vectorization), not from
// arithmetic, and the docs back that with roofline evidence. cuBLAS SGEMV is the
// baseline.

#include <cuda_runtime.h>

namespace ckl {

// One thread per output row, scalar loads across the row. Simple and coalescing
// poor: adjacent threads read down a column of A, one row apart in memory.
void gemv_naive(const float* a, const float* x, float* y,
                int m, int n, float alpha, float beta,
                cudaStream_t stream = nullptr);

// One warp per output row, each lane strides across the row and the partial sums
// are combined with a shfl reduction. Adjacent lanes read adjacent A elements, so
// the row read coalesces.
void gemv_warp(const float* a, const float* x, float* y,
               int m, int n, float alpha, float beta,
               cudaStream_t stream = nullptr);

// One warp per row with float4 loads of A and x, quartering the number of memory
// instructions. Requires n a multiple of 4; other n fall back to the warp kernel.
void gemv_vectorized(const float* a, const float* x, float* y,
                     int m, int n, float alpha, float beta,
                     cudaStream_t stream = nullptr);

// cuBLAS SGEMV oracle and baseline, producing the same row major result.
void gemv_cublas(const float* a, const float* x, float* y,
                 int m, int n, float alpha, float beta,
                 cudaStream_t stream = nullptr);

}  // namespace ckl
