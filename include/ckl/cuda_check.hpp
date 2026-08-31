#pragma once

// Thin error-checking wrappers for the CUDA runtime and the math libraries.
// Every CUDA call in this project goes through one of these macros so a
// failure surfaces at the call site with file and line rather than as a
// silent wrong answer three kernels later.
//
// The exception carries a ckl::Status, so the C ABI shim can map a failure to a
// code without parsing the message. The mapping that matters most is
// cudaErrorNoKernelImageForDevice, which is what a fatbin with no SASS for the
// running device produces: that is an architecture mismatch, not an internal
// error, and the caller needs to be told so.

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

#include "ckl/status.hpp"

namespace ckl {

// Header only, so no translation unit picks up a link dependency just for
// checking a CUDA call.
inline Status status_from_cuda(cudaError_t e) {
    switch (e) {
        case cudaSuccess:
            return Status::kSuccess;
        case cudaErrorNoKernelImageForDevice:
        case cudaErrorInvalidDeviceFunction:
        case cudaErrorUnsupportedPtxVersion:
        case cudaErrorJitCompilationDisabled:
            return Status::kArchMismatch;
        case cudaErrorMemoryAllocation:
            return Status::kAllocFailed;
        case cudaErrorNoDevice:
        case cudaErrorInsufficientDriver:
        case cudaErrorInitializationError:
        case cudaErrorDevicesUnavailable:
            return Status::kNotInitialized;
        case cudaErrorNotSupported:
            return Status::kNotSupported;
        case cudaErrorInvalidValue:
        case cudaErrorInvalidDevice:
        case cudaErrorInvalidPitchValue:
        case cudaErrorInvalidMemcpyDirection:
            return Status::kInvalidValue;
        case cudaErrorLaunchFailure:
        case cudaErrorLaunchTimeout:
        case cudaErrorLaunchOutOfResources:
        case cudaErrorIllegalAddress:
        case cudaErrorIllegalInstruction:
        case cudaErrorMisalignedAddress:
            return Status::kExecutionFailed;
        default:
            return Status::kInternal;
    }
}

[[noreturn]] inline void fail(const char* what, const char* expr, const char* file, int line) {
    std::string msg =
        std::string(what) + " failed: " + expr + " at " + file + ":" + std::to_string(line);
    throw Error(Status::kInternal, msg);
}

inline void check_cuda(cudaError_t status, const char* expr, const char* file, int line) {
    if (status != cudaSuccess) {
        std::string msg = std::string("CUDA error ") + cudaGetErrorName(status) + " (" +
                          cudaGetErrorString(status) + "): " + expr + " at " + file + ":" +
                          std::to_string(line);
        throw Error(status_from_cuda(status), msg);
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
