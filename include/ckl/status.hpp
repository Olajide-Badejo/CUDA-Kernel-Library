#pragma once

/**
 * @file status.hpp
 * @brief Status codes for the C++ API and the exception type the library throws.
 *
 * Every path that can fail either returns a Status or throws ckl::Error carrying
 * one, so the C ABI shim can map a failure to a code without parsing a message.
 */

#include <stdexcept>
#include <string>

#include "ckl/ckl_export.h"

/// Everything the CUDA Kernel Library exposes.
namespace ckl {

/**
 * @brief Outcome of a CKL call.
 *
 * The values match ckl_status_t in ckl.h one for one and in the same order.
 */
enum class Status {
    kSuccess = 0,      ///< The call did what it was asked to do.
    kNotInitialized,   ///< No usable device, or a vendor handle would not initialize.
    kInvalidValue,     ///< Malformed argument: bad dimension, small leading dimension, null buffer.
    kArchMismatch,     ///< The running device cannot execute the selected code path.
    kNotSupported,     ///< The shape or type combination is outside what this entry point takes.
    kAllocFailed,      ///< A device or host allocation failed.
    kExecutionFailed,  ///< A launch or a vendor call failed at run time.
    kInternal,         ///< A defect in the library; the message says what happened.
};

/**
 * @brief Short symbolic name of a status, for example "kInvalidValue".
 * @param s Status to name.
 * @return A static string, never null.
 */
CKL_EXPORT const char* status_string(Status s);

/**
 * @brief The exception the library throws internally, carrying a Status.
 *
 * Header only on purpose. An exception thrown inside libckl_api and caught in a
 * consumer needs its type information visible on both sides of the library
 * boundary; a class level visibility attribute gives that without making every
 * translation unit that touches CKL_CUDA_CHECK link against a specific library.
 *
 * @note Nothing derived from this type crosses the C ABI. Every C entry point
 *       catches it and returns the mapped ckl_status_t instead.
 */
class CKL_EXPORT Error : public std::runtime_error {
public:
    /**
     * @brief Builds an error from a status and a human readable message.
     * @param status Status the C ABI shim maps this failure to.
     * @param message Detail, reachable afterwards through what().
     */
    Error(Status status, const std::string& message)
        : std::runtime_error(message), status_(status) {}

    /**
     * @brief The status this failure maps to.
     * @return The status handed to the constructor.
     */
    Status status() const noexcept { return status_; }

private:
    Status status_;
};

}  // namespace ckl
