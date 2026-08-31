#pragma once

/**
 * @file solver.hpp
 * @brief RAII wrapper over cuSOLVER dense factorizations.
 *
 * Column major (LAPACK convention), single precision. Two solvers: LU (getrf
 * then getrs) for general systems, and Cholesky (potrf then potrs) for symmetric
 * positive definite systems. Both solve A X = B in place over B and leave the
 * factorization in A.
 *
 * "Done" for these means the residual norm check in the tests passes, not merely
 * that no CUDA error fired.
 */

#include <cstdint>
#include <memory>

#include <cuda_runtime.h>

#include "ckl/ckl_export.h"

namespace ckl {

/// @brief Which triangle of a symmetric matrix carries the data.
enum class Fill {
    kLower,  ///< The lower triangle is read; the upper is ignored.
    kUpper,  ///< The upper triangle is read; the lower is ignored.
};

/**
 * @brief Owns a cuSOLVER handle and the scratch its factorizations need.
 *
 * The workspace is sized by the matching _bufferSize query and held alongside the
 * pivot array and the device info word, so a solve is one call and nothing leaks
 * when an exception unwinds.
 *
 * @note Externally synchronized, like Context: one solver per thread, or a lock
 *       of your own around it.
 */
class CKL_EXPORT DenseSolver {
public:
    /**
     * @brief Creates the cuSOLVER handle.
     * @throws ckl::Error when cuSOLVER will not initialize.
     */
    DenseSolver();

    /// @brief Releases the handle and every buffer it allocated.
    ~DenseSolver();

    DenseSolver(const DenseSolver&) = delete;
    DenseSolver& operator=(const DenseSolver&) = delete;

    /**
     * @brief Move constructor, so a solver can be returned or stored in a container.
     * @param other Solver to move from; it holds no handle afterwards and calling
     *        a solve on it throws.
     */
    DenseSolver(DenseSolver&& other) noexcept;

    /**
     * @brief Move assignment.
     * @param other Solver to move from; it holds no handle afterwards.
     * @return This solver.
     */
    DenseSolver& operator=(DenseSolver&& other) noexcept;

    /**
     * @brief Sets the stream the factorizations enqueue on.
     * @param stream Stream to use; nullptr means the default stream.
     */
    void set_stream(cudaStream_t stream);

    /**
     * @brief Solves A X = B for a general A by LU with partial pivoting.
     * @param a Device pointer to A, n by n, column major; overwritten by its LU factors.
     * @param b Device pointer to B, n by nrhs, column major; overwritten by X.
     * @param n Order of A.
     * @param nrhs Right hand sides, the columns of B.
     * @throws ckl::Error on a singular factor, which is a nonzero device info word.
     * @note Unlike the GEMM and GEMV entry points this one is synchronous: it
     *       reads the device info word back with a blocking copy after each step,
     *       so it cannot return before the factorization has finished.
     */
    void solve_lu(float* a, float* b, int n, int nrhs);

    /**
     * @brief Solves A X = B for a symmetric positive definite A by Cholesky.
     * @param a Device pointer to A, n by n, column major; overwritten by its Cholesky factor.
     * @param b Device pointer to B, n by nrhs, column major; overwritten by X.
     * @param n Order of A.
     * @param nrhs Right hand sides, the columns of B.
     * @param fill Which triangle of A holds the data; only that one is read.
     * @throws ckl::Error when the factorization finds A not positive definite.
     * @note Synchronous for the same reason as solve_lu.
     */
    void solve_cholesky(float* a, float* b, int n, int nrhs, Fill fill = Fill::kLower);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace ckl
