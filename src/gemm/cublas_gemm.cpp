// cuBLAS SGEMM wrapped to produce the same row major C as the hand written
// kernels, so a comparison is same shape and same process. cuBLAS is column
// major, so we use the identity: a row major C = A * B occupies the same bytes
// as a column major C transpose = B transpose * A transpose. Feeding cuBLAS our
// row major B and A unchanged, it reads them as those transposes, and the
// column major result it writes is exactly our row major C.

#include <stdexcept>
#include <string>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "ckl/gemm.hpp"

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
        throw std::runtime_error(std::string("cuBLAS error ") + cublas_status_string(s) + ": " +
                                 expr);
    }
}

// One cached handle for the process. cuBLAS handles are not cheap to create and
// the sweep calls this thousands of times.
cublasHandle_t handle() {
    static cublasHandle_t h = [] {
        cublasHandle_t created = nullptr;
        check_cublas(cublasCreate(&created), "cublasCreate");
        return created;
    }();
    return h;
}

}  // namespace

void gemm_cublas(const float* a, const float* b, float* c, int m, int n, int k, float alpha,
                 float beta, cudaStream_t stream) {
    if (m <= 0 || n <= 0) {
        return;
    }
    cublasHandle_t h = handle();
    check_cublas(cublasSetStream(h, stream), "cublasSetStream");
    if (k <= 0) {
        // Empty contraction: C = beta * C. cuBLAS rejects a zero leading
        // dimension, so scale directly rather than calling SGEMM with lda == 0.
        check_cublas(cublasSscal(h, m * n, &beta, c, 1), "cublasSscal");
        return;
    }
    // See the file header for the transpose identity. Compute
    // C_transpose(n by m) = B_transpose(n by k) * A_transpose(k by m).
    check_cublas(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, b, n, a, k, &beta, c, n),
                 "cublasSgemm");
}

}  // namespace ckl
