// cuBLAS SGEMV oracle producing the same row major result. cuBLAS is column
// major, so our row major A (m by n) is a column major (n by m) matrix; asking
// cuBLAS for op(A) = transpose with dimensions (n, m) computes A_rowmajor times x.
//
// The handle comes from the process wide default Context rather than from a
// function local static. The old static was never destroyed and cublasSetStream
// on it raced across threads; the lock makes the set-stream plus call pair
// atomic, and a caller that wants no lock creates its own Context.

#include <mutex>
#include <stdexcept>
#include <string>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "ckl/context.hpp"
#include "ckl/gemv.hpp"
#include "ckl/status.hpp"

namespace ckl {

namespace {

void check(cublasStatus_t s, const char* expr) {
    if (s != CUBLAS_STATUS_SUCCESS) {
        throw Error(s == CUBLAS_STATUS_ARCH_MISMATCH ? Status::kArchMismatch
                                                     : Status::kExecutionFailed,
                    std::string("cuBLAS error ") + cublasGetStatusName(s) + ": " + expr);
    }
}

}  // namespace

void gemv_cublas(const float* a, const float* x, float* y, int m, int n, float alpha, float beta,
                 cudaStream_t stream) {
    if (m <= 0) {
        return;
    }
    auto* h = static_cast<cublasHandle_t>(detail::default_context().cublas());
    std::lock_guard<std::mutex> lock(detail::default_context_mutex());
    check(cublasSetStream(h, stream), "cublasSetStream");
    if (n <= 0) {
        check(cublasSscal(h, m, &beta, y, 1), "cublasSscal");
        return;
    }
    // Column major (n by m) matrix, transposed, times x gives row major A times x.
    check(cublasSgemv(h, CUBLAS_OP_T, n, m, &alpha, a, n, x, 1, &beta, y, 1), "cublasSgemv");
}

}  // namespace ckl
