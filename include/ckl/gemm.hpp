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

// Shared memory tiled FP32 GEMM: reuse each staged tile TILE times, cutting
// global traffic by the tile factor over the naive kernel.
void gemm_tiled(const float* a, const float* b, float* c,
                int m, int n, int k, float alpha, float beta,
                cudaStream_t stream = nullptr);

// Register blocked, float4 vectorized FP32 GEMM: 128x128 block, 8x8 register
// tile per thread. Falls back to the tiled kernel for shapes that do not divide
// the block factors (m by 128, n by 128, k by 8).
void gemm_register(const float* a, const float* b, float* c,
                   int m, int n, int k, float alpha, float beta,
                   cudaStream_t stream = nullptr);

// cp.async double buffered FP32 GEMM: overlaps the next tile's global to shared
// copy with the current tile's math. Same alignment contract and fallback as the
// register kernel.
void gemm_cp_async(const float* a, const float* b, float* c,
                   int m, int n, int k, float alpha, float beta,
                   cudaStream_t stream = nullptr);

// cuBLAS SGEMM oracle producing the same row major C. Manages a cached handle
// internally. Used as both the correctness oracle and the performance baseline.
void gemm_cublas(const float* a, const float* b, float* c,
                 int m, int n, int k, float alpha, float beta,
                 cudaStream_t stream = nullptr);

}  // namespace ckl
