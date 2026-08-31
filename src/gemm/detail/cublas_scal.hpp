#pragma once

// C = beta * C for an empty contraction, counted in 64 bits.
//
// The element count of an m by n FP32 matrix overflows a signed int at
// m = n = 46341, and the v1 wrappers computed m * n in int before handing it to
// cublasSscal. Count in int64_t and chunk the call, so the only limit left is
// the allocation itself. Internal to the GEMM library; not installed.

#include <cstdint>
#include <string>

#include <cublas_v2.h>

#include "ckl/status.hpp"

namespace ckl {
namespace detail {

inline void scal_all(cublasHandle_t h, std::int64_t count, float beta, float* x) {
    constexpr std::int64_t kChunk = 1073741824;  // 2^30 elements per call
    std::int64_t done = 0;
    while (done < count) {
        const std::int64_t n = (count - done) < kChunk ? (count - done) : kChunk;
        const cublasStatus_t s = cublasSscal(h, static_cast<int>(n), &beta, x + done, 1);
        if (s != CUBLAS_STATUS_SUCCESS) {
            throw Error(Status::kExecutionFailed,
                        "cuBLAS error " + std::to_string(static_cast<int>(s)) + ": cublasSscal");
        }
        done += n;
    }
}

}  // namespace detail
}  // namespace ckl
