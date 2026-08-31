#pragma once

/**
 * @file cuda_check.hpp
 * @brief Thin error checking wrappers for the CUDA runtime and the math libraries.
 *
 * Every CUDA call in this project goes through one of these macros so a failure
 * surfaces at the call site with file and line rather than as a silent wrong
 * answer three kernels later.
 *
 * The exception carries a ckl::Status, so the C ABI shim can map a failure to a
 * code without parsing the message. The mapping that matters most is
 * cudaErrorNoKernelImageForDevice, which is what a fatbin with no SASS for the
 * running device produces: that is an architecture mismatch, not an internal
 * error, and the caller needs to be told so.
 */

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

#include "ckl/status.hpp"

namespace ckl {

/**
 * @brief Maps a CUDA runtime error onto the library's status enum.
 * @param e Error the runtime returned.
 * @return The status a caller can act on; Status::kInternal for anything unmapped.
 * @note Header only, so no translation unit picks up a link dependency just for
 *       checking a CUDA call.
 */
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

/**
 * @brief Throws for a failed call that carries no cudaError_t of its own.
 * @param what Name of the API that failed.
 * @param expr Source text of the failing expression.
 * @param file Source file, normally __FILE__.
 * @param line Source line, normally __LINE__.
 * @throws ckl::Error with Status::kInternal, always.
 */
[[noreturn]] inline void fail(const char* what, const char* expr, const char* file, int line) {
    std::string msg =
        std::string(what) + " failed: " + expr + " at " + file + ":" + std::to_string(line);
    throw Error(Status::kInternal, msg);
}

/**
 * @brief Throws a ckl::Error when a CUDA runtime call did not succeed.
 * @param status Value the runtime returned.
 * @param expr Source text of the call, for the message.
 * @param file Source file, normally __FILE__.
 * @param line Source line, normally __LINE__.
 * @throws ckl::Error carrying status_from_cuda(status) and a message naming the
 *         error, the expression, and the source location.
 */
inline void check_cuda(cudaError_t status, const char* expr, const char* file, int line) {
    if (status != cudaSuccess) {
        std::string msg = std::string("CUDA error ") + cudaGetErrorName(status) + " (" +
                          cudaGetErrorString(status) + "): " + expr + " at " + file + ":" +
                          std::to_string(line);
        throw Error(status_from_cuda(status), msg);
    }
}

}  // namespace ckl

/**
 * @brief Wraps any cudaError_t returning call and throws on failure.
 * @param expr The CUDA runtime call to check.
 */
#define CKL_CUDA_CHECK(expr) ::ckl::check_cuda((expr), #expr, __FILE__, __LINE__)

/**
 * @brief Checks for asynchronous kernel launch errors after a launch.
 * @param sync Pass true to also synchronize the device, which is what tests and
 *        single shot timing want; the sweep leaves it false on the hot path and
 *        synchronizes explicitly through events.
 */
#define CKL_CUDA_LAST_ERROR(sync)                                                         \
    do {                                                                                  \
        ::ckl::check_cuda(cudaGetLastError(), "kernel launch", __FILE__, __LINE__);       \
        if (sync) {                                                                       \
            ::ckl::check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize", __FILE__, \
                              __LINE__);                                                  \
        }                                                                                 \
    } while (0)
