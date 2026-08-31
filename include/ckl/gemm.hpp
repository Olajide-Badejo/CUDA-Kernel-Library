#pragma once

// Public GEMM interface.
//
// Two layers live here. The descriptor driven ckl::gemm is the primary entry
// point: it takes a Context, validates the shape, checks the device against
// what the selected path needs, dispatches, and reports the path it took
// through the chosen out-param. Below it sit the v1 free functions, one per
// ladder rung, which every benchmark and correctness test drives directly.
//
// Free function convention: row major, single precision unless a variant says
// otherwise.
//   A is m by k, leading dimension k
//   B is k by n, leading dimension n
//   C is m by n, leading dimension n
//   C = alpha * (A * B) + beta * C
//
// Row major is the natural indexing for hand written kernels. The cuBLAS oracle
// (column major) is wrapped so it produces the identical row major C, letting
// every performance claim be a same shape, same process comparison.
//
// BLAS contract on beta: when beta is zero, C is not read. An uninitialized or
// NaN C is legal input in that case, and every kernel here honors it.

#include <cstddef>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "ckl/ckl_export.h"
#include "ckl/context.hpp"
#include "ckl/status.hpp"
#include "ckl/types.hpp"

namespace ckl {

// ---------------------------------------------------------------------------
// Descriptor driven entry points
// ---------------------------------------------------------------------------

// C = alpha * op(A) * op(B) + beta * C as described by desc. alpha and beta are
// host pointers to float. When chosen is non-null it is always written, on
// success and on failure alike, with the path the call took or would have
// taken; kAuto never resolves silently.
//
// Returns kInvalidValue for a malformed descriptor (bad dimensions, leading
// dimensions too small, null buffers), kArchMismatch when the running device
// cannot execute the selected path, and kNotSupported when an explicitly named
// algorithm cannot take the shape. An explicit algorithm is never rerouted.
CKL_EXPORT Status gemm(Context& ctx, const GemmDesc& desc, const void* alpha, const void* a,
                       const void* b, const void* beta, void* c, Algo* chosen = nullptr);

// What kAuto would pick for this descriptor. Launches nothing. Returns kCublas
// for any shape the hand written kernels cannot take, which is also what the
// dispatcher would do.
CKL_EXPORT Algo gemm_query(const Context& ctx, const GemmDesc& desc);

// Bytes of scratch the descriptor needs. Every path shipped in 1.1.0 answers 0.
CKL_EXPORT std::size_t gemm_workspace_size(const Context& ctx, const GemmDesc& desc);

// ---------------------------------------------------------------------------
// Ladder rungs
// ---------------------------------------------------------------------------

// Naive FP32 GEMM: one thread per output element, natural indexing. This is the
// honest baseline, not a strawman.
CKL_EXPORT void gemm_naive(const float* a, const float* b, float* c, int m, int n, int k,
                           float alpha, float beta, cudaStream_t stream = nullptr);

// Shared memory tiled FP32 GEMM: reuse each staged tile TILE times, cutting
// global traffic by the tile factor over the naive kernel.
CKL_EXPORT void gemm_tiled(const float* a, const float* b, float* c, int m, int n, int k,
                           float alpha, float beta, cudaStream_t stream = nullptr);

// Register blocked, float4 vectorized FP32 GEMM: 128x128 block, 8x8 register
// tile per thread. Falls back to the tiled kernel for shapes that do not divide
// the block factors (m by 128, n by 128, k by 8).
CKL_EXPORT void gemm_register(const float* a, const float* b, float* c, int m, int n, int k,
                              float alpha, float beta, cudaStream_t stream = nullptr);

// cp.async double buffered FP32 GEMM: overlaps the next tile's global to shared
// copy with the current tile's math. Same alignment contract and fallback as the
// register kernel. Needs compute capability 8.0 on the aligned path.
CKL_EXPORT void gemm_cp_async(const float* a, const float* b, float* c, int m, int n, int k,
                              float alpha, float beta, cudaStream_t stream = nullptr);

// cuBLAS SGEMM oracle producing the same row major C. Runs on the process wide
// default Context. Used as both the correctness oracle and the performance
// baseline.
CKL_EXPORT void gemm_cublas(const float* a, const float* b, float* c, int m, int n, int k,
                            float alpha, float beta, cudaStream_t stream = nullptr);

// Tensor core rungs. Inputs are FP16 or BF16 storage, accumulate is FP32, and
// the output C is FP32. WMMA 16x16x16 fragments; aligned fast path (m by 64,
// n by 64, k by 16), which needs compute capability 8.0, with a scalar half
// input fallback that runs anywhere for other shapes.
CKL_EXPORT void gemm_wmma_fp16(const __half* a, const __half* b, float* c, int m, int n, int k,
                               float alpha, float beta, cudaStream_t stream = nullptr);
CKL_EXPORT void gemm_wmma_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b, float* c, int m,
                               int n, int k, float alpha, float beta,
                               cudaStream_t stream = nullptr);

// mma.sync PTX tensor core variant (FP16 storage, FP32 accumulate) with scalar
// shared fragment reads. Same alignment contract as the WMMA fast path.
CKL_EXPORT void gemm_mma_ptx(const __half* a, const __half* b, float* c, int m, int n, int k,
                             float alpha, float beta, cudaStream_t stream = nullptr);

// mma.sync with ldmatrix fragment loads: the optimized tensor variant that
// replaces scalar shared loads with one swizzled ldmatrix per fragment.
CKL_EXPORT void gemm_mma_ldm(const __half* a, const __half* b, float* c, int m, int n, int k,
                             float alpha, float beta, cudaStream_t stream = nullptr);

// Top tensor kernel: 128x128 tile, ldmatrix, cp.async double buffering. The FP16
// kernel driven toward the compute bound gate.
CKL_EXPORT void gemm_mma_opt(const __half* a, const __half* b, float* c, int m, int n, int k,
                             float alpha, float beta, cudaStream_t stream = nullptr);

// Fusion study: the top kernel with a per column bias add and ReLU folded into
// the epilogue (computes C = relu(alpha * A*B + bias)), versus the unfused path of
// a plain GEMM followed by gemm_bias_relu (a separate memory bound pass over C).
// The fused kernel takes aligned shapes only and throws ckl::Error with
// kNotSupported otherwise, because a return that writes no C is a silent wrong
// answer.
CKL_EXPORT void gemm_mma_opt_bias(const __half* a, const __half* b, float* c, const float* bias,
                                  int m, int n, int k, float alpha,
                                  cudaStream_t stream = nullptr);
CKL_EXPORT void gemm_bias_relu(float* c, const float* bias, int m, int n,
                               cudaStream_t stream = nullptr);

// cuBLAS tensor core oracles (cublasGemmEx, FP16 or BF16 in, FP32 accumulate),
// producing the same row major C. Baseline for the tensor path per precision.
CKL_EXPORT void gemm_cublas_fp16(const __half* a, const __half* b, float* c, int m, int n, int k,
                                 float alpha, float beta, cudaStream_t stream = nullptr);
CKL_EXPORT void gemm_cublas_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b, float* c, int m,
                                 int n, int k, float alpha, float beta,
                                 cudaStream_t stream = nullptr);

}  // namespace ckl
