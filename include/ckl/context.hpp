#pragma once

// ckl::Context is the primary handle of the library. It owns one cuBLAS, one
// cuSPARSE and one cuSOLVER handle per instance, created on first use and
// destroyed newest first, plus the current stream, an optional user workspace,
// and the device properties, which are queried once at construction and read
// from cache afterwards so no dispatch decision costs a driver round trip.
//
// The v1 free functions (gemm_cublas, gemv_cublas, spmv_cusparse, trsm_cublas)
// stay source compatible by routing through detail::default_context(). They are
// conveniences; the Context API is the primary one, and anything that runs on
// more than one stream or more than one thread should create its own Context.

#include <cstddef>
#include <memory>
#include <mutex>

#include <cuda_runtime.h>

#include "ckl/ckl_export.h"
#include "ckl/status.hpp"

namespace ckl {

class CKL_EXPORT Context {
public:
    // Queries the current device and caches its properties. Throws ckl::Error
    // with kNotInitialized when there is no usable device.
    Context();
    ~Context();

    Context(Context&&) noexcept;
    Context& operator=(Context&&) noexcept;
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;

    void set_stream(cudaStream_t stream);
    cudaStream_t stream() const;

    // A caller supplied scratch buffer. Nothing shipped in 1.1.0 needs one; the
    // plumbing exists so a split-K or stream-K rung can take it later without an
    // ABI break.
    void set_workspace(void* ws, std::size_t bytes);
    void* workspace() const;
    std::size_t workspace_size() const;

    int device() const;

    // 10 * major + minor, so an RTX 5070 reads 120. Cached at construction.
    int compute_capability() const;

    const cudaDeviceProp& device_properties() const;

    // Vendor handles, created on first use. They come back as void* so this
    // header stays free of cublas_v2.h, cusparse.h and cusolverDn.h; each of
    // those handle types is itself a pointer, so the cast back is exact.
    void* cublas() const;
    void* cusparse() const;
    void* cusolver() const;

private:
    struct Impl;

    // Every accessor goes through this, so a moved from Context reports itself
    // rather than dereferencing null.
    Impl& live() const;

    std::unique_ptr<Impl> impl_;
};

namespace detail {

// The one process wide Context the v1 free functions route through. It is
// deliberately never destroyed: releasing vendor handles during static
// destruction, after the CUDA runtime has begun tearing the primary context
// down, is worse than not releasing them.
CKL_EXPORT Context& default_context();

// Serializes the set-stream plus call pair the free functions perform on the
// shared handle. A Context of your own avoids this lock entirely.
CKL_EXPORT std::mutex& default_context_mutex();

// Compute capability of the current device as 10 * major + minor, queried once
// per process. The free functions have no Context to read it from.
CKL_EXPORT int device_compute_capability();

// Throws ckl::Error(kArchMismatch) when the running device is below what the
// named code path needs. Called by the launchers of every kernel whose device
// body is compiled out below its floor, so a no-op kernel can never launch.
CKL_EXPORT void require_arch(int minimum_cc, const char* what);

}  // namespace detail

}  // namespace ckl
