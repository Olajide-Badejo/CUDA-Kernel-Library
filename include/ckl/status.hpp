#pragma once

// Status codes for the C++ API and the exception type the library throws
// internally. Every path that can fail either returns a Status or throws
// ckl::Error carrying one, so the C ABI shim can map a failure to a code
// without parsing a message.

#include <stdexcept>
#include <string>

#include "ckl/ckl_export.h"

namespace ckl {

enum class Status {
    kSuccess = 0,
    kNotInitialized,
    kInvalidValue,
    kArchMismatch,
    kNotSupported,
    kAllocFailed,
    kExecutionFailed,
    kInternal,
};

// Short symbolic name, never null.
CKL_EXPORT const char* status_string(Status s);

// Header only on purpose. An exception thrown inside libckl_api and caught in a
// consumer needs its type information visible on both sides of the library
// boundary; a class level visibility attribute gives that without making every
// translation unit that touches CKL_CUDA_CHECK link against a specific library.
class CKL_EXPORT Error : public std::runtime_error {
public:
    Error(Status status, const std::string& message)
        : std::runtime_error(message), status_(status) {}

    Status status() const noexcept { return status_; }

private:
    Status status_;
};

}  // namespace ckl
