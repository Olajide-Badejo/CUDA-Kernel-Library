#pragma once

/**
 * @file device_buffer.hpp
 * @brief Owning device allocation with move semantics and no copies.
 *
 * Wrapping raw cudaMalloc in RAII means every test and benchmark path is leak
 * free even when an exception unwinds through it, and the ownership is explicit
 * in the type rather than living in a comment next to a bare pointer.
 */

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "ckl/cuda_check.hpp"

namespace ckl {

/**
 * @brief An owning cudaMalloc allocation of count elements of T.
 *
 * @tparam T Element type; it is allocated, never constructed, so it has to be
 *         trivially copyable to be meaningful on the device.
 *
 * @note A class template, so it carries no CKL_EXPORT: it is instantiated in the
 *       consumer's own translation unit and there is nothing in the library to
 *       import.
 * @note Every transfer here uses the default stream and blocks the host. This is
 *       a setup and teardown helper, not something to put inside a timed region.
 */
template <typename T>
class DeviceBuffer {
public:
    /// @brief Creates an empty buffer that owns nothing.
    DeviceBuffer() = default;

    /**
     * @brief Allocates count elements on the current device.
     * @param count Number of elements; zero allocates nothing.
     * @throws ckl::Error with Status::kAllocFailed when cudaMalloc fails.
     */
    explicit DeviceBuffer(std::size_t count) : count_(count) {
        if (count_ > 0) {
            CKL_CUDA_CHECK(cudaMalloc(&ptr_, count_ * sizeof(T)));
        }
    }

    /// @brief Frees the allocation.
    ~DeviceBuffer() { reset(); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    /**
     * @brief Move constructor.
     * @param other Buffer to take ownership from; it becomes empty.
     */
    DeviceBuffer(DeviceBuffer&& other) noexcept : ptr_(other.ptr_), count_(other.count_) {
        other.ptr_ = nullptr;
        other.count_ = 0;
    }

    /**
     * @brief Move assignment; frees whatever this buffer held first.
     * @param other Buffer to take ownership from; it becomes empty.
     * @return This buffer.
     */
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            reset();
            ptr_ = other.ptr_;
            count_ = other.count_;
            other.ptr_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }

    /**
     * @brief The device pointer.
     * @return The allocation, or nullptr for an empty buffer.
     */
    T* data() { return ptr_; }

    /**
     * @brief The device pointer, const overload.
     * @return The allocation, or nullptr for an empty buffer.
     */
    const T* data() const { return ptr_; }

    /**
     * @brief Element count.
     * @return How many elements were allocated.
     */
    std::size_t size() const { return count_; }

    /**
     * @brief Allocation size.
     * @return size() times sizeof(T).
     */
    std::size_t bytes() const { return count_ * sizeof(T); }

    /**
     * @brief Copies count elements from host memory into this buffer.
     * @param host Source pointer in host memory.
     * @param count Elements to copy; must not exceed size().
     * @throws std::invalid_argument when count is past the end of the allocation.
     * @throws std::runtime_error when a pointer is null and count is non zero.
     * @note Validates before touching the driver. A count past the end used to be
     *       a silent out of bounds copy, and a copy on a default constructed
     *       buffer used to pass a null device pointer to cudaMemcpy and get back
     *       an error that named the wrong thing.
     */
    void copy_from_host(const T* host, std::size_t count) {
        check_transfer(host, count, "copy_from_host");
        if (count == 0) {
            return;
        }
        CKL_CUDA_CHECK(cudaMemcpy(ptr_, host, count * sizeof(T), cudaMemcpyHostToDevice));
    }

    /**
     * @brief Copies a whole vector into this buffer.
     * @param host Source vector; its size must not exceed size().
     */
    void copy_from_host(const std::vector<T>& host) { copy_from_host(host.data(), host.size()); }

    /**
     * @brief Copies count elements out of this buffer into host memory.
     * @param host Destination pointer in host memory.
     * @param count Elements to copy; must not exceed size().
     * @throws std::invalid_argument when count is past the end of the allocation.
     * @throws std::runtime_error when a pointer is null and count is non zero.
     */
    void copy_to_host(T* host, std::size_t count) const {
        check_transfer(host, count, "copy_to_host");
        if (count == 0) {
            return;
        }
        CKL_CUDA_CHECK(cudaMemcpy(host, ptr_, count * sizeof(T), cudaMemcpyDeviceToHost));
    }

    /**
     * @brief Copies the whole buffer into a fresh vector.
     * @return A vector of size() elements holding the buffer's contents.
     */
    std::vector<T> to_host() const {
        std::vector<T> host(count_);
        copy_to_host(host.data(), count_);
        return host;
    }

    /// @brief Fills the whole allocation with zero bytes.
    void zero() {
        if (count_ > 0) {
            CKL_CUDA_CHECK(cudaMemset(ptr_, 0, count_ * sizeof(T)));
        }
    }

private:
    void check_transfer(const void* host, std::size_t count, const char* what) const {
        if (count > count_) {
            throw std::invalid_argument(std::string("DeviceBuffer::") + what + ": count " +
                                        std::to_string(count) + " exceeds the buffer's " +
                                        std::to_string(count_) + " elements");
        }
        if (count > 0 && (ptr_ == nullptr || host == nullptr)) {
            throw std::runtime_error(std::string("DeviceBuffer::") + what +
                                     ": null pointer with a non zero count");
        }
    }

    void reset() {
        if (ptr_ != nullptr) {
            cudaFree(ptr_);
            ptr_ = nullptr;
        }
        count_ = 0;
    }

    T* ptr_ = nullptr;
    std::size_t count_ = 0;
};

}  // namespace ckl
