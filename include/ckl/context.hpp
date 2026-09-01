#pragma once

/**
 * @file context.hpp
 * @brief ckl::Context, the primary handle of the library.
 *
 * The v1 free functions (gemm_cublas, gemv_cublas, spmv_cusparse, trsm_cublas)
 * stay source compatible by routing through ckl::detail::default_context(). They
 * are conveniences; the Context API is the primary one, and anything that runs
 * on more than one stream or more than one thread should create its own Context.
 */

#include <cstddef>
#include <memory>
#include <mutex>

#include <cuda_runtime.h>

#include "ckl/ckl_export.h"
#include "ckl/status.hpp"

namespace ckl {

/**
 * @brief One cuBLAS, cuSPARSE and cuSOLVER handle, one stream, one workspace.
 *
 * A Context owns one handle of each vendor library, created on first use and
 * destroyed newest first, plus the current stream, an optional user workspace,
 * and the device properties, which are queried once at construction and read
 * from cache afterwards so no dispatch decision costs a driver round trip.
 *
 * @note A Context is externally synchronized. Concurrent calls on one Context
 *       from two threads are a data race; give each thread its own Context, or
 *       hold a lock of your own across every call that touches the shared one.
 */
class CKL_EXPORT Context {
public:
    /**
     * @brief Selects the current device and caches its properties.
     * @throws ckl::Error with Status::kNotInitialized when there is no usable device.
     */
    Context();

    /// @brief Destroys the vendor handles newest first.
    ~Context();

    /**
     * @brief Move constructor.
     * @param other Context to move from; it holds nothing afterwards.
     */
    Context(Context&& other) noexcept;

    /**
     * @brief Move assignment.
     * @param other Context to move from; it holds nothing afterwards.
     * @return This context.
     */
    Context& operator=(Context&& other) noexcept;

    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;

    /**
     * @brief Sets the stream every subsequent call on this Context enqueues on.
     * @param stream Stream to use; nullptr means the default stream.
     * @note This does not synchronize. Work already enqueued on the previous
     *       stream stays there.
     */
    void set_stream(cudaStream_t stream);

    /**
     * @brief The stream this Context enqueues on.
     * @return The stream last handed to set_stream, or nullptr.
     */
    cudaStream_t stream() const;

    /**
     * @brief Hands the Context a caller supplied scratch buffer.
     * @param ws Device pointer to the scratch, or nullptr for none.
     * @param bytes Size of the buffer in bytes.
     * @note The split-K and stream-K GEMM paths use it; every other path answers
     *       zero from ckl::gemm_workspace_size and ignores whatever is here. A
     *       path that needs scratch and finds none allocates and frees its own
     *       around the launch, which is correct but puts an allocation inside
     *       the call. The Context does not own the buffer and never frees it.
     */
    void set_workspace(void* ws, std::size_t bytes);

    /**
     * @brief The workspace pointer this Context holds.
     * @return The pointer last handed to set_workspace, or nullptr.
     */
    void* workspace() const;

    /**
     * @brief The workspace size this Context holds.
     * @return Bytes last handed to set_workspace, or zero.
     */
    std::size_t workspace_size() const;

    /**
     * @brief The CUDA device ordinal this Context was built on.
     * @return The device index as the runtime reported it at construction.
     */
    int device() const;

    /**
     * @brief Compute capability as 10 * major + minor, so an RTX 5070 reads 120.
     * @return The cached capability; no driver call is made.
     */
    int compute_capability() const;

    /**
     * @brief Full device properties, queried once at construction.
     * @return A reference to the cached cudaDeviceProp.
     */
    const cudaDeviceProp& device_properties() const;

    /**
     * @brief The cuBLAS handle, created on first use.
     * @return A cublasHandle_t as void*, so this header stays free of cublas_v2.h.
     * @note The handle type is itself a pointer, so the cast back is exact.
     */
    void* cublas() const;

    /**
     * @brief The cuSPARSE handle, created on first use.
     * @return A cusparseHandle_t as void*.
     */
    void* cusparse() const;

    /**
     * @brief The cuSOLVER dense handle, created on first use.
     * @return A cusolverDnHandle_t as void*.
     */
    void* cusolver() const;

private:
    struct Impl;

    // Every accessor goes through this, so a moved from Context reports itself
    // rather than dereferencing null.
    Impl& live() const;

    std::unique_ptr<Impl> impl_;
};

/// Internals that the free function layer needs and a consumer normally does not.
namespace detail {

/**
 * @brief The one process wide Context the v1 free functions route through.
 * @return A reference to the shared Context.
 * @note It is deliberately never destroyed: releasing vendor handles during
 *       static destruction, after the CUDA runtime has begun tearing the primary
 *       context down, is worse than not releasing them.
 */
CKL_EXPORT Context& default_context();

/**
 * @brief The lock serializing the set-stream plus call pair on the shared handle.
 * @return A reference to the process wide mutex.
 * @note A Context of your own avoids this lock entirely.
 */
CKL_EXPORT std::mutex& default_context_mutex();

/**
 * @brief Compute capability of the current device, queried once per process.
 * @return 10 * major + minor.
 * @note The free functions have no Context to read it from, which is why this
 *       exists alongside Context::compute_capability.
 */
CKL_EXPORT int device_compute_capability();

/**
 * @brief Refuses to launch a kernel the running device cannot execute.
 * @param minimum_cc Lowest compute capability the code path needs, as 10 * major + minor.
 * @param what Name of the path, used in the message.
 * @throws ckl::Error with Status::kArchMismatch when the device is below the floor.
 * @note Called by the launchers of every kernel whose device body is compiled
 *       out below its floor, so a no-op kernel can never launch.
 */
CKL_EXPORT void require_arch(int minimum_cc, const char* what);

}  // namespace detail

}  // namespace ckl
