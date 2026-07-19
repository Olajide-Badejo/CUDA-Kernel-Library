// cuSOLVER dense solver implementation. The Impl holds the handle and reusable
// scratch; each solve queries the workspace size, grows the buffers if needed,
// runs the factorization and the triangular solves, and checks device info.

#include "ckl/solver.hpp"

#include <stdexcept>
#include <string>
#include <vector>

#include <cusolverDn.h>

#include "ckl/cuda_check.hpp"
#include "ckl/device_buffer.hpp"

namespace ckl {

namespace {

void check(cusolverStatus_t s, const char* expr) {
    if (s != CUSOLVER_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("cuSOLVER error ") + std::to_string(s) + ": " + expr);
    }
}

int read_info(const DeviceBuffer<int>& info) {
    int host = 0;
    info.copy_to_host(&host, 1);
    return host;
}

}  // namespace

struct DenseSolver::Impl {
    cusolverDnHandle_t handle = nullptr;
    cudaStream_t stream = nullptr;
    DeviceBuffer<float> workspace;
    DeviceBuffer<int> pivots;
    DeviceBuffer<int> info{1};

    void ensure_workspace(int lwork) {
        if (static_cast<int>(workspace.size()) < lwork) {
            workspace = DeviceBuffer<float>(static_cast<std::size_t>(lwork));
        }
    }
};

DenseSolver::DenseSolver() : impl_(new Impl()) {
    check(cusolverDnCreate(&impl_->handle), "cusolverDnCreate");
}

DenseSolver::~DenseSolver() {
    if (impl_ != nullptr) {
        if (impl_->handle != nullptr) {
            cusolverDnDestroy(impl_->handle);
        }
        delete impl_;
    }
}

void DenseSolver::set_stream(cudaStream_t stream) {
    impl_->stream = stream;
    check(cusolverDnSetStream(impl_->handle, stream), "cusolverDnSetStream");
}

void DenseSolver::solve_lu(float* a, float* b, int n, int nrhs) {
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

    check(cusolverDnSgetrf(impl_->handle, n, n, a, n, impl_->workspace.data(),
                           impl_->pivots.data(), impl_->info.data()),
          "cusolverDnSgetrf");
    if (int info = read_info(impl_->info); info != 0) {
        throw std::runtime_error("LU factorization failed, U is singular at pivot " +
                                 std::to_string(info));
    }
    check(cusolverDnSgetrs(impl_->handle, CUBLAS_OP_N, n, nrhs, a, n,
                           impl_->pivots.data(), b, n, impl_->info.data()),
          "cusolverDnSgetrs");
    if (int info = read_info(impl_->info); info != 0) {
        throw std::runtime_error("LU solve reported invalid argument " + std::to_string(info));
    }
}

void DenseSolver::solve_cholesky(float* a, float* b, int n, int nrhs, Fill fill) {
    if (n <= 0 || nrhs <= 0) {
        return;
    }
    const cublasFillMode_t uplo =
        fill == Fill::kLower ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;

    int lwork = 0;
    check(cusolverDnSpotrf_bufferSize(impl_->handle, uplo, n, a, n, &lwork),
          "cusolverDnSpotrf_bufferSize");
    impl_->ensure_workspace(lwork);

    check(cusolverDnSpotrf(impl_->handle, uplo, n, a, n, impl_->workspace.data(),
                           lwork, impl_->info.data()),
          "cusolverDnSpotrf");
    if (int info = read_info(impl_->info); info != 0) {
        throw std::runtime_error("Cholesky factorization failed, leading minor " +
                                 std::to_string(info) + " is not positive definite");
    }
    check(cusolverDnSpotrs(impl_->handle, uplo, n, nrhs, a, n, b, n, impl_->info.data()),
          "cusolverDnSpotrs");
    if (int info = read_info(impl_->info); info != 0) {
        throw std::runtime_error("Cholesky solve reported invalid argument " +
                                 std::to_string(info));
    }
}

}  // namespace ckl
