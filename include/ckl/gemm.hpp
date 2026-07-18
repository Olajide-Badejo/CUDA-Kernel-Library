#pragma once

// Public GEMM interface. All kernels in this library use the same convention so
// the benchmark harness and the correctness tests can drive any variant through
// one signature.
//
// Convention: row major, single precision unless a variant says otherwise.
//   A is m by k, leading dimension k
//   B is k by n, leading dimension n
//   C is m by n, leading dimension n
//   C = alpha * (A * B) + beta * C
//
// Row major is the natural indexing for hand written kernels. The cuBLAS oracle
// (column major) is wrapped so it produces the identical row major C, letting
// every performance claim be a same shape, same process comparison.

#include <cuda_runtime.h>

namespace ckl {

// Ladder rungs plus the two library baselines. Kept in optimization order.
enum class GemmVariant {
    kNaive,
    kTiled,
    kRegister,
    kCpAsync,
    kWmmaFp16,
    kWmmaBf16,
    kMmaPtx,
    kCutlass,
    kCublas,
};

const char* gemm_variant_name(GemmVariant v);

// Naive FP32 GEMM: one thread per output element, natural indexing. This is the
// honest baseline, not a strawman.
void gemm_naive(const float* a, const float* b, float* c,
                int m, int n, int k, float alpha, float beta,
                cudaStream_t stream = nullptr);

// cuBLAS SGEMM oracle producing the same row major C. Manages a cached handle
// internally. Used as both the correctness oracle and the performance baseline.
void gemm_cublas(const float* a, const float* b, float* c,
                 int m, int n, int k, float alpha, float beta,
                 cudaStream_t stream = nullptr);

}  // namespace ckl
