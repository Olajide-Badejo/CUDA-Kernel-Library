#pragma once

// RAII wrapper over cuSOLVER dense factorizations. It owns the cuSOLVER handle
// and the scratch it needs (workspace sized by the _bufferSize query, pivot
// array, device info), so a solve is one call and nothing leaks when an exception
// unwinds. Column major (LAPACK convention), single precision.
//
// Two solvers: LU (getrf then getrs) for general systems, and Cholesky (potrf
// then potrs) for symmetric positive definite systems. Both solve A X = B in
// place over B and leave the factorization in A. "Done" for these means the
// residual norm check in the tests passes, not merely that no CUDA error fired.

#include <cstdint>
#include <memory>

#include <cuda_runtime.h>

#include "ckl/ckl_export.h"

namespace ckl {

enum class Fill { kLower, kUpper };

class CKL_EXPORT DenseSolver {
public:
    DenseSolver();
    ~DenseSolver();

    DenseSolver(const DenseSolver&) = delete;
    DenseSolver& operator=(const DenseSolver&) = delete;

    // Movable, so a solver can be returned or stored in a container. A moved
    // from solver holds no handle; calling a solve on it throws.
    DenseSolver(DenseSolver&&) noexcept;
    DenseSolver& operator=(DenseSolver&&) noexcept;

    void set_stream(cudaStream_t stream);

    // Solve A X = B for a general A (n by n) by LU with partial pivoting. A is
    // overwritten by its LU factors; B (n by nrhs) is overwritten by X. Throws on
    // a singular factor (device info nonzero).
    void solve_lu(float* a, float* b, int n, int nrhs);

    // Solve A X = B for a symmetric positive definite A by Cholesky. Only the
    // named triangle of A is read; A is overwritten by its Cholesky factor.
    void solve_cholesky(float* a, float* b, int n, int nrhs, Fill fill = Fill::kLower);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace ckl
