#pragma once

// The thread local detail of the last failure. The dispatcher writes it, the C
// ABI reads it back through ckl_last_error, and both live in ckl_api so the
// buffer is one object. Not installed: a C++ caller gets the same information
// from the ckl::Error it catches.

#include <string>

namespace ckl {
namespace detail {

void set_last_error(const std::string& message);
void clear_last_error();

// Never null; empty when nothing has failed on this thread.
const char* last_error_message();

}  // namespace detail
}  // namespace ckl
