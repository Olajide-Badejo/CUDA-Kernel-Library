#pragma once

// Owning device allocation with move semantics and no copies. Wrapping raw
// cudaMalloc in RAII means every test and benchmark path is leak free even
// when an exception unwinds through it, and the ownership is explicit in the
// type rather than living in a comment next to a bare pointer.

#include <cstddef>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "ckl/cuda_check.hpp"

namespace ckl {

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;

    explicit DeviceBuffer(std::size_t count) : count_(count) {
        if (count_ > 0) {
            CKL_CUDA_CHECK(cudaMalloc(&ptr_, count_ * sizeof(T)));
        }
    }

    ~DeviceBuffer() { reset(); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept : ptr_(other.ptr_), count_(other.count_) {
        other.ptr_ = nullptr;
        other.count_ = 0;
    }

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

    T* data() { return ptr_; }
    const T* data() const { return ptr_; }
    std::size_t size() const { return count_; }
    std::size_t bytes() const { return count_ * sizeof(T); }

    void copy_from_host(const T* host, std::size_t count) {
        CKL_CUDA_CHECK(cudaMemcpy(ptr_, host, count * sizeof(T), cudaMemcpyHostToDevice));
    }

    void copy_from_host(const std::vector<T>& host) { copy_from_host(host.data(), host.size()); }

    void copy_to_host(T* host, std::size_t count) const {
        CKL_CUDA_CHECK(cudaMemcpy(host, ptr_, count * sizeof(T), cudaMemcpyDeviceToHost));
    }

    std::vector<T> to_host() const {
        std::vector<T> host(count_);
        copy_to_host(host.data(), count_);
        return host;
    }

    void zero() {
        if (count_ > 0) {
            CKL_CUDA_CHECK(cudaMemset(ptr_, 0, count_ * sizeof(T)));
        }
    }

private:
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
