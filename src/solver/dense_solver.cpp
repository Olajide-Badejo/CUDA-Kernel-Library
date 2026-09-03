// cuSOLVER dense solver implementation. The Impl holds the handle and reusable
// scratch; each solve queries the workspace size, grows the buffers if needed,
// runs the factorization and the triangular solves, and checks device info.

#include "ckl/solver.hpp"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cusolverDn.h>

#include "ckl/cuda_check.hpp"
#include "ckl/device_buffer.hpp"
#include "ckl/status.hpp"

namespace ckl {

namespace {

void check(cusolverStatus_t s, const char* expr) {
    if (s != CUSOLVER_STATUS_SUCCESS) {
        throw Error(
            Status::kExecutionFailed,
            std::string("cuSOLVER error ") + std::to_string(static_cast<int>(s)) + ": " + expr);
    }
}

int read_info(const DeviceBuffer<int>& info) {
    int host = 0;
    info.copy_to_host(&host, 1);
    return host;
}

}  // namespace

// The handle is created and destroyed by Impl itself. The v1 code created it in
// the DenseSolver constructor after new Impl(), so a throwing cusolverDnCreate
// leaked the Impl; here the allocation is owned by a unique_ptr before anything
// can throw, and the members already built unwind normally.
struct DenseSolver::Impl {
    cusolverDnHandle_t handle = nullptr;
    cudaStream_t stream = nullptr;
    DeviceBuffer<float> workspace;
    DeviceBuffer<int> pivots;
    DeviceBuffer<int> info{1};

    Impl() { check(cusolverDnCreate(&handle), "cusolverDnCreate"); }

    ~Impl() {
        if (handle != nullptr) {
            cusolverDnDestroy(handle);
        }
    }

    Impl(const Impl&) = delete;
    Impl& operator=(const Impl&) = delete;

    void ensure_workspace(int lwork) {
        if (static_cast<int>(workspace.size()) < lwork) {
            workspace = DeviceBuffer<float>(static_cast<std::size_t>(lwork));
        }
    }
};

DenseSolver::DenseSolver() : impl_(std::make_unique<Impl>()) {}

DenseSolver::~DenseSolver() = default;

DenseSolver::DenseSolver(DenseSolver&&) noexcept = default;

DenseSolver& DenseSolver::operator=(DenseSolver&&) noexcept = default;

void DenseSolver::set_stream(cudaStream_t stream) {
    if (!impl_) {
        throw Error(Status::kNotInitialized, "DenseSolver used after it was moved from");
    }
    impl_->stream = stream;
    check(cusolverDnSetStream(impl_->handle, stream), "cusolverDnSetStream");
}

void DenseSolver::solve_lu(float* a, float* b, int n, int nrhs) {
    if (!impl_) {
        throw Error(Status::kNotInitialized, "DenseSolver used after it was moved from");
    }
    if (n <= 0 || nrhs <= 0) {
        return;
    }
    int lwork = 0;
    check(cusolverDnSgetrf_bufferSize(impl_->handle, n, n, a, n, &lwork),
          "cusolverDnSgetrf_bufferSize");
    impl_->ensure_workspace(lwork);
    if (impl_->pivots.size() < static_cast<std::size_t>(n)) {
        impl_->pivots = DeviceBuffer<int>(static_cast<std::size_t>(n));
    }

    check(cusolverDnSgetrf(impl_->handle, n, n, a, n, impl_->workspace.data(), impl_->pivots.data(),
                           impl_->info.data()),
          "cusolverDnSgetrf");
    if (int info = read_info(impl_->info); info != 0) {
        throw Error(Status::kExecutionFailed,
                    "LU factorization failed, U is singular at pivot " + std::to_string(info));
    }
    check(cusolverDnSgetrs(impl_->handle, CUBLAS_OP_N, n, nrhs, a, n, impl_->pivots.data(), b, n,
                           impl_->info.data()),
          "cusolverDnSgetrs");
    if (int info = read_info(impl_->info); info != 0) {
        throw Error(Status::kInvalidValue,
                    "LU solve reported invalid argument " + std::to_string(info));
    }
}

void DenseSolver::solve_cholesky(float* a, float* b, int n, int nrhs, Fill fill) {
    if (!impl_) {
        throw Error(Status::kNotInitialized, "DenseSolver used after it was moved from");
    }
    if (n <= 0 || nrhs <= 0) {
        return;
    }
    const cublasFillMode_t uplo =
        fill == Fill::kLower ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;

    int lwork = 0;
    check(cusolverDnSpotrf_bufferSize(impl_->handle, uplo, n, a, n, &lwork),
          "cusolverDnSpotrf_bufferSize");
    impl_->ensure_workspace(lwork);

    check(cusolverDnSpotrf(impl_->handle, uplo, n, a, n, impl_->workspace.data(), lwork,
                           impl_->info.data()),
          "cusolverDnSpotrf");
    if (int info = read_info(impl_->info); info != 0) {
        throw Error(Status::kExecutionFailed, "Cholesky factorization failed, leading minor " +
                                                  std::to_string(info) +
                                                  " is not positive definite");
    }
    check(cusolverDnSpotrs(impl_->handle, uplo, n, nrhs, a, n, b, n, impl_->info.data()),
          "cusolverDnSpotrs");
    if (int info = read_info(impl_->info); info != 0) {
        throw Error(Status::kInvalidValue,
                    "Cholesky solve reported invalid argument " + std::to_string(info));
    }
}

}  // namespace ckl
