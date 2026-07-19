// cuBLAS SGEMV oracle producing the same row major result. cuBLAS is column
// major, so our row major A (m by n) is a column major (n by m) matrix; asking
// cuBLAS for op(A) = transpose with dimensions (n, m) computes A_rowmajor times x.

#include <stdexcept>
#include <string>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "ckl/gemv.hpp"

namespace ckl {

namespace {

void check(cublasStatus_t s, const char* expr) {
    if (s != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("cuBLAS error ") + std::to_string(s) + ": " + expr);
    }
}

cublasHandle_t handle() {
    static cublasHandle_t h = [] {
        cublasHandle_t created = nullptr;
        check(cublasCreate(&created), "cublasCreate");
        return created;
    }();
    return h;
}

}  // namespace

void gemv_cublas(const float* a, const float* x, float* y,
                 int m, int n, float alpha, float beta, cudaStream_t stream) {
    if (m <= 0) {
        return;
    }
    cublasHandle_t h = handle();
    check(cublasSetStream(h, stream), "cublasSetStream");
    if (n <= 0) {
        check(cublasSscal(h, m, &beta, y, 1), "cublasSscal");
        return;
    }
    // Column major (n by m) matrix, transposed, times x gives row major A times x.
    check(cublasSgemv(h, CUBLAS_OP_T, n, m, &alpha, a, n, x, 1, &beta, y, 1), "cublasSgemv");
}

}  // namespace ckl
