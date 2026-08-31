// cuBLAS STRSM oracle producing the same row major result. We solve, in row
// major, L X = alpha B with L lower triangular. Transposing gives
// X_transpose * L_transpose = alpha B_transpose, a right side solve, and our row
// major L buffer is exactly L_transpose (upper triangular) in cuBLAS's column
// major view, our row major B buffer is B_transpose. So the call is a right side,
// upper, no transpose STRSM with swapped dimensions.
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
#include "ckl/status.hpp"
#include "ckl/trsm.hpp"

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

void trsm_cublas(const float* a, float* b, int m, int n, float alpha, cudaStream_t stream) {
    if (m <= 0 || n <= 0) {
        return;
    }
    auto* h = static_cast<cublasHandle_t>(detail::default_context().cublas());
    std::lock_guard<std::mutex> lock(detail::default_context_mutex());
    check(cublasSetStream(h, stream), "cublasSetStream");
    check(cublasStrsm(h, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
                      CUBLAS_DIAG_NON_UNIT, n, m, &alpha, a, m, b, n),
          "cublasStrsm");
}

}  // namespace ckl
