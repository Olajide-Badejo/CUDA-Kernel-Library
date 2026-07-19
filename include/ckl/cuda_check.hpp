#pragma once

// Thin error-checking wrappers for the CUDA runtime and the math libraries.
// Every CUDA call in this project goes through one of these macros so a
// failure surfaces at the call site with file and line rather than as a
// silent wrong answer three kernels later.

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

namespace ckl {

[[noreturn]] inline void fail(const char* what, const char* expr, const char* file, int line) {
    std::string msg =
        std::string(what) + " failed: " + expr + " at " + file + ":" + std::to_string(line);
    throw std::runtime_error(msg);
}

inline void check_cuda(cudaError_t status, const char* expr, const char* file, int line) {
    if (status != cudaSuccess) {
        std::string msg = std::string("CUDA error ") + cudaGetErrorName(status) + " (" +
                          cudaGetErrorString(status) + "): " + expr + " at " + file + ":" +
                          std::to_string(line);
        throw std::runtime_error(msg);
    }
}

}  // namespace ckl

// Wrap any cudaError_t returning call.
#define CKL_CUDA_CHECK(expr) ::ckl::check_cuda((expr), #expr, __FILE__, __LINE__)

// Check for asynchronous kernel launch errors. Pass true to also synchronize,
// which is what tests and single-shot timing want; the sweep leaves it false
// on the hot path and synchronizes explicitly through events.
#define CKL_CUDA_LAST_ERROR(sync)                                                         \
    do {                                                                                  \
        ::ckl::check_cuda(cudaGetLastError(), "kernel launch", __FILE__, __LINE__);       \
        if (sync) {                                                                       \
            ::ckl::check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize", __FILE__, \
                              __LINE__);                                                  \
        }                                                                                 \
    } while (0)
