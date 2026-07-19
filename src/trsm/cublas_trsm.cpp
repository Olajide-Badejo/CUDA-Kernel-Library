// cuBLAS STRSM oracle producing the same row major result. We solve, in row
// major, L X = alpha B with L lower triangular. Transposing gives
// X_transpose * L_transpose = alpha B_transpose, a right side solve, and our row
// major L buffer is exactly L_transpose (upper triangular) in cuBLAS's column
// major view, our row major B buffer is B_transpose. So the call is a right side,
// upper, no transpose STRSM with swapped dimensions.

#include <stdexcept>
#include <string>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "ckl/trsm.hpp"

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

void trsm_cublas(const float* a, float* b, int m, int n, float alpha, cudaStream_t stream) {
    if (m <= 0 || n <= 0) {
        return;
    }
    cublasHandle_t h = handle();
    check(cublasSetStream(h, stream), "cublasSetStream");
    check(cublasStrsm(h, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER,
                      CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                      n, m, &alpha, a, m, b, n),
          "cublasStrsm");
}

}  // namespace ckl
