// cuSOLVER DenseSolver correctness by residual norm. Column major throughout, as
// cuSOLVER expects. LU is checked on a general well conditioned matrix, Cholesky
// on a symmetric positive definite matrix built as B_transpose B + n I. The
// factorization overwrites A, so the residual is formed against a saved copy of
// the original A. "Done" is a small residual, not just the absence of a CUDA
// error.

#include <cmath>
#include <cstdio>
#include <vector>

#include "ckl/cuda_check.hpp"
#include "ckl/device_buffer.hpp"
#include "ckl/reference.hpp"
#include "ckl/solver.hpp"

namespace {

// Column major element (i, j) of an n by n matrix.
inline float& at(std::vector<float>& a, int n, int i, int j) {
    return a[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * n];
}

// Residual ||A X - B|| / ||B|| in double precision, column major, using the
// original A (n by n) and the right hand side B (n by nrhs).
double residual(const std::vector<float>& a, const std::vector<float>& x,
                const std::vector<float>& b, int n, int nrhs) {
    double num = 0.0;
    double den = 0.0;
    for (int c = 0; c < nrhs; ++c) {
        for (int i = 0; i < n; ++i) {
            double ax = 0.0;
            for (int j = 0; j < n; ++j) {
                ax += static_cast<double>(
                          a[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * n]) *
                      static_cast<double>(
                          x[static_cast<std::size_t>(j) + static_cast<std::size_t>(c) * n]);
            }
            const double bij = b[static_cast<std::size_t>(i) + static_cast<std::size_t>(c) * n];
            num += (ax - bij) * (ax - bij);
            den += bij * bij;
        }
    }
    return std::sqrt(num) / std::sqrt(den);
}

std::vector<float> run_solver(bool cholesky, const std::vector<float>& a0,
                              const std::vector<float>& b0, int n, int nrhs) {
    ckl::DenseSolver solver;
    ckl::DeviceBuffer<float> da(a0.size());
    ckl::DeviceBuffer<float> db(b0.size());
    da.copy_from_host(a0);
    db.copy_from_host(b0);
    if (cholesky) {
        solver.solve_cholesky(da.data(), db.data(), n, nrhs, ckl::Fill::kLower);
    } else {
        solver.solve_lu(da.data(), db.data(), n, nrhs);
    }
    CKL_CUDA_CHECK(cudaDeviceSynchronize());
    return db.to_host();
}

}  // namespace

int main() {
    const int n = 512;
    const int nrhs = 8;
    constexpr double kTol = 1e-3;
    bool ok = true;
    std::printf("cuSOLVER DenseSolver residual checks (n=%d, nrhs=%d, tol=%.0e)\n", n, nrhs, kTol);

    // LU on a general, diagonally dominant matrix.
    {
        auto a = ckl::random_matrix(n, n, 0xA1);
        for (int i = 0; i < n; ++i) {
            at(a, n, i, i) += static_cast<float>(n);
        }
        const auto b = ckl::random_matrix(n, nrhs, 0xB2);
        const auto x = run_solver(false, a, b, n, nrhs);
        const double res = residual(a, x, b, n, nrhs);
        const bool pass = res < kTol;
        ok = ok && pass;
        std::printf("  LU        residual = %.3e  %s\n", res, pass ? "PASS" : "FAIL");
    }

    // Cholesky on B_transpose B + n I (symmetric positive definite).
    {
        const auto bm = ckl::random_matrix(n, n, 0xC3);
        std::vector<float> a(static_cast<std::size_t>(n) * n, 0.0f);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                double s = 0.0;
                for (int p = 0; p < n; ++p) {
                    s += static_cast<double>(
                             bm[static_cast<std::size_t>(p) + static_cast<std::size_t>(i) * n]) *
                         static_cast<double>(
                             bm[static_cast<std::size_t>(p) + static_cast<std::size_t>(j) * n]);
                }
                at(a, n, i, j) = static_cast<float>(s);
            }
            at(a, n, i, i) += static_cast<float>(n);
        }
        const auto b = ckl::random_matrix(n, nrhs, 0xD4);
        const auto x = run_solver(true, a, b, n, nrhs);
        const double res = residual(a, x, b, n, nrhs);
        const bool pass = res < kTol;
        ok = ok && pass;
        std::printf("  Cholesky  residual = %.3e  %s\n", res, pass ? "PASS" : "FAIL");
    }

    std::printf("%s\n", ok ? "all solver cases passed" : "solver cases FAILED");
    return ok ? 0 : 1;
}
