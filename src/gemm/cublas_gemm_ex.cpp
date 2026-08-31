// cuBLAS tensor core oracles via cublasGemmEx: FP16 or BF16 inputs, FP32
// accumulate, FP32 output, producing the same row major C as the hand written
// tensor kernels through the same transpose identity used for SGEMM. These are
// the per precision baselines the WMMA and mma.sync variants are measured
// against.
//
// The handle comes from the process wide default Context rather than from a
// function local static. The old static was never destroyed and cublasSetStream
// on it raced across threads; the lock makes the set-stream plus call pair
// atomic, and a caller that wants no lock creates its own Context.

#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "ckl/context.hpp"
#include "ckl/gemm.hpp"
#include "ckl/status.hpp"

#include "detail/cublas_scal.hpp"

namespace ckl {

namespace {

void check(cublasStatus_t s, const char* expr) {
    if (s != CUBLAS_STATUS_SUCCESS) {
        throw Error(s == CUBLAS_STATUS_ARCH_MISMATCH ? Status::kArchMismatch
                                                     : Status::kExecutionFailed,
                    std::string("cuBLAS error ") + cublasGetStatusName(s) + ": " + expr);
    }
}

// Shared body: compute C_transpose(n by m) = B_transpose * A_transpose so the
// column major result lands as our row major C. a_type is the cuda data type of
// the FP16 or BF16 inputs.
void gemm_ex(const void* a, const void* b, float* c, int m, int n, int k, float alpha, float beta,
             cudaStream_t stream, cudaDataType_t in_type) {
    if (m <= 0 || n <= 0) {
        return;
    }
    auto* h = static_cast<cublasHandle_t>(detail::default_context().cublas());
    std::lock_guard<std::mutex> lock(detail::default_context_mutex());
    check(cublasSetStream(h, stream), "cublasSetStream");
    if (k <= 0) {
        // Empty contraction: C = beta * C. Scale the FP32 output directly, with
        // the element count in 64 bits.
        detail::scal_all(h, static_cast<std::int64_t>(m) * n, beta, c);
        return;
    }
    check(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, b, in_type, n, a, in_type, k,
                       &beta, c, CUDA_R_32F, n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
          "cublasGemmEx");
}

}  // namespace

void gemm_cublas_fp16(const __half* a, const __half* b, float* c, int m, int n, int k, float alpha,
                      float beta, cudaStream_t stream) {
    gemm_ex(a, b, c, m, n, k, alpha, beta, stream, CUDA_R_16F);
}

void gemm_cublas_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b, float* c, int m, int n, int k,
                      float alpha, float beta, cudaStream_t stream) {
    gemm_ex(a, b, c, m, n, k, alpha, beta, stream, CUDA_R_16BF);
}

}  // namespace ckl
