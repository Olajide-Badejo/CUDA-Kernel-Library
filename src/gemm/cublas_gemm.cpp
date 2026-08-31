// cuBLAS SGEMM wrapped to produce the same row major C as the hand written
// kernels, so a comparison is same shape and same process. cuBLAS is column
// major, so we use the identity: a row major C = A * B occupies the same bytes
// as a column major C transpose = B transpose * A transpose. Feeding cuBLAS our
// row major B and A unchanged, it reads them as those transposes, and the
// column major result it writes is exactly our row major C.
//
// The handle comes from the process wide default Context rather than from a
// function local static. The old static was never destroyed and cublasSetStream
// on it raced across threads; the lock below makes the set-stream plus call pair
// atomic, and a caller that wants no lock at all creates its own Context and
// goes through ckl::gemm.

#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "ckl/context.hpp"
#include "ckl/gemm.hpp"
#include "ckl/status.hpp"

#include "detail/cublas_scal.hpp"

namespace ckl {

namespace {

const char* cublas_status_string(cublasStatus_t s) {
    switch (s) {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED:
            return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR:
            return "CUBLAS_STATUS_LICENSE_ERROR";
        default:
            return "CUBLAS_STATUS_UNKNOWN";
    }
}

void check_cublas(cublasStatus_t s, const char* expr) {
    if (s != CUBLAS_STATUS_SUCCESS) {
        Status mapped = Status::kExecutionFailed;
        if (s == CUBLAS_STATUS_ARCH_MISMATCH) {
            mapped = Status::kArchMismatch;
        } else if (s == CUBLAS_STATUS_NOT_INITIALIZED) {
            mapped = Status::kNotInitialized;
        } else if (s == CUBLAS_STATUS_ALLOC_FAILED) {
            mapped = Status::kAllocFailed;
        } else if (s == CUBLAS_STATUS_INVALID_VALUE) {
            mapped = Status::kInvalidValue;
        } else if (s == CUBLAS_STATUS_NOT_SUPPORTED) {
            mapped = Status::kNotSupported;
        }
        throw Error(mapped, std::string("cuBLAS error ") + cublas_status_string(s) + ": " + expr);
    }
}

}  // namespace

void gemm_cublas(const float* a, const float* b, float* c, int m, int n, int k, float alpha,
                 float beta, cudaStream_t stream) {
    if (m <= 0 || n <= 0) {
        return;
    }
    auto* h = static_cast<cublasHandle_t>(detail::default_context().cublas());
    std::lock_guard<std::mutex> lock(detail::default_context_mutex());
    check_cublas(cublasSetStream(h, stream), "cublasSetStream");
    if (k <= 0) {
        // Empty contraction: C = beta * C. cuBLAS rejects a zero leading
        // dimension, so scale directly rather than calling SGEMM with lda == 0.
        detail::scal_all(h, static_cast<std::int64_t>(m) * n, beta, c);
        return;
    }
    // See the file header for the transpose identity. Compute
    // C_transpose(n by m) = B_transpose(n by k) * A_transpose(k by m).
    check_cublas(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, b, n, a, k, &beta, c, n),
                 "cublasSgemm");
}

}  // namespace ckl
