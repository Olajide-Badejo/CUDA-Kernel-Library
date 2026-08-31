// Context implementation. The device properties are read once here and never
// again; every later question about the device (compute capability, shared
// memory per block, SM count) is answered from this cache, so no dispatch
// decision costs a driver round trip.
//
// The three vendor handles are created on first use, because a caller that only
// runs hand written kernels should not pay for a cuBLAS context, and destroyed
// newest first in the destructor.

#include "ckl/context.hpp"

#include <mutex>
#include <string>

#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cusparse.h>

#include "ckl/cuda_check.hpp"

namespace ckl {

namespace {

std::string cuda_message(const char* what, cudaError_t e) {
    return std::string(what) + ": " + cudaGetErrorName(e) + " (" + cudaGetErrorString(e) + ")";
}

}  // namespace

struct Context::Impl {
    int device = 0;
    cudaDeviceProp props{};
    int compute_capability = 0;
    cudaStream_t stream = nullptr;
    void* workspace = nullptr;
    std::size_t workspace_bytes = 0;

    // Lazy handles. The mutex covers creation only; a Context is not meant to be
    // driven from two threads at once, but two threads racing to create the same
    // handle would leak one of them.
    std::mutex handle_mutex;
    cublasHandle_t cublas = nullptr;
    cusparseHandle_t cusparse = nullptr;
    cusolverDnHandle_t cusolver = nullptr;

    Impl() {
        cudaError_t e = cudaGetDevice(&device);
        if (e != cudaSuccess) {
            throw Error(status_from_cuda(e), cuda_message("cudaGetDevice", e));
        }
        e = cudaGetDeviceProperties(&props, device);
        if (e != cudaSuccess) {
            throw Error(status_from_cuda(e), cuda_message("cudaGetDeviceProperties", e));
        }
        compute_capability = props.major * 10 + props.minor;
    }

    ~Impl() {
        // Newest first, so nothing outlives a handle it was created against.
        if (cusolver != nullptr) {
            cusolverDnDestroy(cusolver);
        }
        if (cusparse != nullptr) {
            cusparseDestroy(cusparse);
        }
        if (cublas != nullptr) {
            cublasDestroy(cublas);
        }
    }

    Impl(const Impl&) = delete;
    Impl& operator=(const Impl&) = delete;
};

Context::Context() : impl_(std::make_unique<Impl>()) {}

Context::~Context() = default;

Context::Context(Context&&) noexcept = default;

Context& Context::operator=(Context&&) noexcept = default;

Context::Impl& Context::live() const {
    if (!impl_) {
        throw Error(Status::kNotInitialized, "Context used after it was moved from");
    }
    return *impl_;
}

void Context::set_stream(cudaStream_t stream) {
    live().stream = stream;
}

cudaStream_t Context::stream() const {
    return live().stream;
}

void Context::set_workspace(void* ws, std::size_t bytes) {
    Impl& impl = live();
    if (ws == nullptr && bytes != 0) {
        throw Error(Status::kInvalidValue, "Context::set_workspace: null pointer with a size");
    }
    impl.workspace = ws;
    impl.workspace_bytes = bytes;
}

void* Context::workspace() const {
    return live().workspace;
}

std::size_t Context::workspace_size() const {
    return live().workspace_bytes;
}

int Context::device() const {
    return live().device;
}

int Context::compute_capability() const {
    return live().compute_capability;
}

const cudaDeviceProp& Context::device_properties() const {
    return live().props;
}

void* Context::cublas() const {
    Impl& impl = live();
    std::lock_guard<std::mutex> lock(impl.handle_mutex);
    if (impl.cublas == nullptr) {
        cublasStatus_t s = cublasCreate(&impl.cublas);
        if (s != CUBLAS_STATUS_SUCCESS) {
            throw Error(Status::kNotInitialized,
                        "cublasCreate failed with status " + std::to_string(static_cast<int>(s)));
        }
    }
    return impl.cublas;
}

void* Context::cusparse() const {
    Impl& impl = live();
    std::lock_guard<std::mutex> lock(impl.handle_mutex);
    if (impl.cusparse == nullptr) {
        cusparseStatus_t s = cusparseCreate(&impl.cusparse);
        if (s != CUSPARSE_STATUS_SUCCESS) {
            throw Error(Status::kNotInitialized,
                        std::string("cusparseCreate failed: ") + cusparseGetErrorString(s));
        }
    }
    return impl.cusparse;
}

void* Context::cusolver() const {
    Impl& impl = live();
    std::lock_guard<std::mutex> lock(impl.handle_mutex);
    if (impl.cusolver == nullptr) {
        cusolverStatus_t s = cusolverDnCreate(&impl.cusolver);
        if (s != CUSOLVER_STATUS_SUCCESS) {
            throw Error(Status::kNotInitialized, "cusolverDnCreate failed with status " +
                                                     std::to_string(static_cast<int>(s)));
        }
    }
    return impl.cusolver;
}

namespace detail {

Context& default_context() {
    // Leaked on purpose. The v1 free functions can be called from anywhere,
    // including from another static object's destructor, and releasing vendor
    // handles after the CUDA runtime has begun tearing the primary context down
    // is worse than not releasing them. A Context you create yourself is
    // destroyed properly.
    static Context* ctx = new Context();
    return *ctx;
}

std::mutex& default_context_mutex() {
    static std::mutex m;
    return m;
}

int device_compute_capability() {
    // Queried once per process, not per launch. The free functions have no
    // Context, and cudaGetDeviceProperties on the hot path would cost more than
    // some of the kernels it guards.
    static const int cc = [] {
        int device = 0;
        CKL_CUDA_CHECK(cudaGetDevice(&device));
        cudaDeviceProp props{};
        CKL_CUDA_CHECK(cudaGetDeviceProperties(&props, device));
        return props.major * 10 + props.minor;
    }();
    return cc;
}

void require_arch(int minimum_cc, const char* what) {
    const int cc = device_compute_capability();
    if (cc < minimum_cc) {
        throw Error(Status::kArchMismatch,
                    std::string(what) + " needs compute capability " +
                        std::to_string(minimum_cc / 10) + "." + std::to_string(minimum_cc % 10) +
                        " or newer; this device reports " + std::to_string(cc / 10) + "." +
                        std::to_string(cc % 10));
    }
}

}  // namespace detail

}  // namespace ckl
