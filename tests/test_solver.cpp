// cuSOLVER DenseSolver: LU with partial pivoting and Cholesky, both solving
// A X = B in place over B. Column major throughout, as cuSOLVER expects.
//
// Done here means the residual bound holds, not that no CUDA error fired. Two
// bounds are used, for the same reason as in the TRSM suite: the scaled residual
// is backward stable and holds at tol(n) whatever the conditioning, while the
// forward error against the double solve carries the measured condition number.
// v1 solved only diagonally dominant systems, where those two bounds are
// indistinguishable and neither is being tested.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include <gtest/gtest.h>

#include "ckl/cuda_check.hpp"
#include "ckl/device_buffer.hpp"
#include "ckl/solver.hpp"
#include "ckl/status.hpp"
#include "gpu_environment.hpp"
#include "reference.hpp"

namespace {

using ckl::test::GpuTest;
using ckl::test::GpuTestWithParam;

std::size_t elems(int rows, int cols) {
    return static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
}

// Column major element (i, j) of an n by rows matrix.
float& at(std::vector<float>& a, int n, int i, int j) {
    return a[static_cast<std::size_t>(i) +
             static_cast<std::size_t>(j) * static_cast<std::size_t>(n)];
}

float at(const std::vector<float>& a, int n, int i, int j) {
    return a[static_cast<std::size_t>(i) +
             static_cast<std::size_t>(j) * static_cast<std::size_t>(n)];
}

// Row major double copy, which is what the condition number helper wants.
std::vector<double> row_major_double(const std::vector<float>& a, int n) {
    std::vector<double> out(elems(n, n));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            out[static_cast<std::size_t>(i) * static_cast<std::size_t>(n) +
                static_cast<std::size_t>(j)] = static_cast<double>(at(a, n, i, j));
        }
    }
    return out;
}

// max_i |(A X)_i - B_i| divided by sum_j |A_ij| |X_j|.
double scaled_residual(const std::vector<float>& a, const std::vector<float>& x,
                       const std::vector<float>& b, int n, int nrhs) {
    double worst = 0.0;
    for (int c = 0; c < nrhs; ++c) {
        for (int i = 0; i < n; ++i) {
            double ax = 0.0;
            double mag = 0.0;
            for (int j = 0; j < n; ++j) {
                const double av = static_cast<double>(at(a, n, i, j));
                const double xv = static_cast<double>(at(x, n, j, c));
                ax += av * xv;
                mag += std::fabs(av) * std::fabs(xv);
            }
            const double bij = static_cast<double>(at(b, n, i, c));
            const double denom = mag > 1e-30 ? mag : 1.0;
            const double r = std::fabs(ax - bij) / denom;
            if (r > worst) {
                worst = r;
            }
        }
    }
    return worst;
}

// Exact solution in double by Gaussian elimination with partial pivoting.
std::vector<double> solve_reference(const std::vector<float>& a, const std::vector<float>& b, int n,
                                    int nrhs) {
    const std::size_t sn = static_cast<std::size_t>(n);
    std::vector<double> m(sn * sn);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            m[static_cast<std::size_t>(i) * sn + static_cast<std::size_t>(j)] =
                static_cast<double>(at(a, n, i, j));
        }
    }
    std::vector<double> rhs(sn * static_cast<std::size_t>(nrhs));
    for (int c = 0; c < nrhs; ++c) {
        for (int i = 0; i < n; ++i) {
            rhs[static_cast<std::size_t>(i) + static_cast<std::size_t>(c) * sn] =
                static_cast<double>(at(b, n, i, c));
        }
    }
    for (int col = 0; col < n; ++col) {
        int pivot = col;
        double best =
            std::fabs(m[static_cast<std::size_t>(col) * sn + static_cast<std::size_t>(col)]);
        for (int r = col + 1; r < n; ++r) {
            const double v =
                std::fabs(m[static_cast<std::size_t>(r) * sn + static_cast<std::size_t>(col)]);
            if (v > best) {
                best = v;
                pivot = r;
            }
        }
        if (pivot != col) {
            for (int c = 0; c < n; ++c) {
                std::swap(m[static_cast<std::size_t>(col) * sn + static_cast<std::size_t>(c)],
                          m[static_cast<std::size_t>(pivot) * sn + static_cast<std::size_t>(c)]);
            }
            for (int c = 0; c < nrhs; ++c) {
                std::swap(rhs[static_cast<std::size_t>(col) + static_cast<std::size_t>(c) * sn],
                          rhs[static_cast<std::size_t>(pivot) + static_cast<std::size_t>(c) * sn]);
            }
        }
        const double d = m[static_cast<std::size_t>(col) * sn + static_cast<std::size_t>(col)];
        for (int r = col + 1; r < n; ++r) {
            const double f =
                m[static_cast<std::size_t>(r) * sn + static_cast<std::size_t>(col)] / d;
            if (f == 0.0) {
                continue;
            }
            for (int c = col; c < n; ++c) {
                m[static_cast<std::size_t>(r) * sn + static_cast<std::size_t>(c)] -=
                    f * m[static_cast<std::size_t>(col) * sn + static_cast<std::size_t>(c)];
            }
            for (int c = 0; c < nrhs; ++c) {
                rhs[static_cast<std::size_t>(r) + static_cast<std::size_t>(c) * sn] -=
                    f * rhs[static_cast<std::size_t>(col) + static_cast<std::size_t>(c) * sn];
            }
        }
    }
    std::vector<double> x(sn * static_cast<std::size_t>(nrhs), 0.0);
    for (int c = 0; c < nrhs; ++c) {
        for (int i = n - 1; i >= 0; --i) {
            double s = rhs[static_cast<std::size_t>(i) + static_cast<std::size_t>(c) * sn];
            for (int j = i + 1; j < n; ++j) {
                s -= m[static_cast<std::size_t>(i) * sn + static_cast<std::size_t>(j)] *
                     x[static_cast<std::size_t>(j) + static_cast<std::size_t>(c) * sn];
            }
            x[static_cast<std::size_t>(i) + static_cast<std::size_t>(c) * sn] =
                s / m[static_cast<std::size_t>(i) * sn + static_cast<std::size_t>(i)];
        }
    }
    return x;
}

std::vector<float> make_dominant(int n, std::uint64_t seed) {
    auto a = ckl::random_matrix(n, n, seed);
    for (int i = 0; i < n; ++i) {
        at(a, n, i, i) += static_cast<float>(n);
    }
    return a;
}

// Two nearly parallel columns: column 3 is column 2 plus delta times a unit
// perturbation, so A sits delta away from rank deficient and its condition
// number lands near 1 / delta. Row scaling would not do: solving D A x = D b is
// the same computation as solving A x = b, so a row graded matrix has a large
// condition number and no extra forward error, which would make the bound below
// vacuous. A near degenerate column is the real thing.
std::vector<float> make_near_singular(int n, std::uint64_t seed, float delta) {
    auto a = make_dominant(n, seed);
    const auto pert = ckl::random_matrix(n, 1, seed ^ 0x5A5A5A5AULL);
    for (int i = 0; i < n; ++i) {
        at(a, n, i, 3) = at(a, n, i, 2) + delta * pert[static_cast<std::size_t>(i)];
    }
    return a;
}

// B_transpose B + n I: symmetric positive definite by construction.
std::vector<float> make_spd(int n, std::uint64_t seed) {
    const auto bm = ckl::random_matrix(n, n, seed);
    std::vector<float> a(elems(n, n), 0.0f);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double s = 0.0;
            for (int p = 0; p < n; ++p) {
                s += static_cast<double>(at(bm, n, p, i)) * static_cast<double>(at(bm, n, p, j));
            }
            at(a, n, i, j) = static_cast<float>(s);
        }
        at(a, n, i, i) += static_cast<float>(n);
    }
    return a;
}

enum class Method { kLu, kCholesky };

const char* method_name(Method m) {
    return m == Method::kLu ? "lu" : "cholesky";
}

std::vector<float> run_solver(Method method, const std::vector<float>& a0,
                              const std::vector<float>& b0, int n, int nrhs,
                              cudaStream_t stream = nullptr) {
    ckl::DenseSolver solver;
    if (stream != nullptr) {
        solver.set_stream(stream);
    }
    ckl::DeviceBuffer<float> da(a0.size());
    ckl::DeviceBuffer<float> db(b0.size());
    da.copy_from_host(a0);
    db.copy_from_host(b0);
    if (method == Method::kCholesky) {
        solver.solve_cholesky(da.data(), db.data(), n, nrhs, ckl::Fill::kLower);
    } else {
        solver.solve_lu(da.data(), db.data(), n, nrhs);
    }
    CKL_CUDA_CHECK(cudaDeviceSynchronize());
    return db.to_host();
}

// ---------------------------------------------------------------------------

struct SolveCase {
    const char* label;
    Method method;
    int n;
    int nrhs;
};

class SolverCase : public GpuTestWithParam<SolveCase> {};

std::string solve_case_name(const ::testing::TestParamInfo<SolveCase>& info) {
    return info.param.label;
}

TEST_P(SolverCase, ResidualAndForwardError) {
    const SolveCase c = GetParam();
    const auto a = (c.method == Method::kCholesky)
                       ? make_spd(c.n, ckl::test::seed_stream(6101))
                       : make_dominant(c.n, ckl::test::seed_stream(6102));
    const auto b = ckl::random_matrix(c.n, c.nrhs, ckl::test::seed_stream(6103));

    const auto x = run_solver(c.method, a, b, c.n, c.nrhs);
    ASSERT_TRUE(ckl::all_finite(x)) << method_name(c.method) << " produced a non finite solution";
    EXPECT_LT(scaled_residual(a, x, b, c.n, c.nrhs), ckl::tol(c.n))
        << method_name(c.method) << " " << c.label << " scaled residual";

    const auto exact = solve_reference(a, b, c.n, c.nrhs);
    EXPECT_LT(ckl::relative_frobenius_error(x, exact), ckl::tol(c.n))
        << method_name(c.method) << " " << c.label << " forward error";
}

INSTANTIATE_TEST_SUITE_P(Dense, SolverCase,
                         ::testing::Values(SolveCase{"lu_256_8", Method::kLu, 256, 8},
                                           SolveCase{"lu_128_1", Method::kLu, 128, 1},
                                           SolveCase{"lu_64_16", Method::kLu, 64, 16},
                                           SolveCase{"lu_1_1", Method::kLu, 1, 1},
                                           SolveCase{"cholesky_256_8", Method::kCholesky, 256, 8},
                                           SolveCase{"cholesky_128_1", Method::kCholesky, 128, 1},
                                           SolveCase{"cholesky_64_16", Method::kCholesky, 64, 16}),
                         solve_case_name);

class Solver : public GpuTest {};

// The conditioning case: a matrix one part in 1e-3 away from rank deficient.
// Measured condition number at the default seed is about 7e5, inside the 1e4 to
// 1e6 band Section 14 asks for.
// The test measures how ill conditioned it actually is and puts that number in
// the forward error bound rather than assuming one.
TEST_F(Solver, IllConditionedLu) {
    const int n = 128;
    const int nrhs = 4;
    const auto a = make_near_singular(n, ckl::test::seed_stream(6201), 1.0e-3f);
    const auto b = ckl::random_matrix(n, nrhs, ckl::test::seed_stream(6202));

    const double cond = ckl::condition_number_inf(row_major_double(a, n), n);
    ::testing::Test::RecordProperty("condition_number", ckl::test::sci(cond));
    ASSERT_GT(cond, 1.0e4);
    ASSERT_LT(cond, 1.0e7);

    const auto x = run_solver(Method::kLu, a, b, n, nrhs);
    ASSERT_TRUE(ckl::all_finite(x));

    // Backward stability is not excused by conditioning.
    EXPECT_LT(scaled_residual(a, x, b, n, nrhs), ckl::tol(n));

    const auto exact = solve_reference(a, b, n, nrhs);
    const double forward_bound = ckl::tol(n) * cond;
    const double forward = ckl::relative_frobenius_error(x, exact);
    ::testing::Test::RecordProperty("forward_error", ckl::test::sci(forward));
    EXPECT_LT(forward, forward_bound)
        << "forward error bound tol(n) * cond = " << forward_bound << " with cond " << cond;
}

TEST_F(Solver, SetStreamRunsTheSolveOnThatStream) {
    const int n = 128;
    const int nrhs = 4;
    const auto a = make_dominant(n, ckl::test::seed_stream(6301));
    const auto b = ckl::random_matrix(n, nrhs, ckl::test::seed_stream(6302));

    cudaStream_t s = nullptr;
    CKL_CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    const auto x = run_solver(Method::kLu, a, b, n, nrhs, s);
    CKL_CUDA_CHECK(cudaStreamSynchronize(s));
    CKL_CUDA_CHECK(cudaStreamDestroy(s));

    EXPECT_LT(scaled_residual(a, x, b, n, nrhs), ckl::tol(n));
    const auto exact = solve_reference(a, b, n, nrhs);
    EXPECT_LT(ckl::relative_frobenius_error(x, exact), ckl::tol(n));
}

TEST_F(Solver, SingularMatrixIsReportedNotIgnored) {
    const int n = 64;
    auto a = make_dominant(n, ckl::test::seed_stream(6401));
    // Column 3 is all zeros, so A is exactly rank deficient and the pivot search
    // for that column finds nothing: getrf has to report a zero pivot rather
    // than factor on and hand back a solution made of infinities.
    for (int i = 0; i < n; ++i) {
        at(a, n, i, 3) = 0.0f;
    }
    const auto b = ckl::random_matrix(n, 1, ckl::test::seed_stream(6402));

    ckl::DenseSolver solver;
    ckl::DeviceBuffer<float> da(a.size());
    ckl::DeviceBuffer<float> db(b.size());
    da.copy_from_host(a);
    db.copy_from_host(b);
    try {
        solver.solve_lu(da.data(), db.data(), n, 1);
        FAIL() << "a singular factor has to be reported";
    } catch (const ckl::Error& e) {
        EXPECT_EQ(e.status(), ckl::Status::kExecutionFailed) << e.what();
    }
    CKL_CUDA_CHECK(cudaDeviceSynchronize());
}

TEST_F(Solver, CholeskyRefusesAnIndefiniteMatrix) {
    const int n = 32;
    std::vector<float> a(elems(n, n), 0.0f);
    for (int i = 0; i < n; ++i) {
        at(a, n, i, i) = (i == 0) ? -1.0f : 1.0f;  // not positive definite
    }
    const auto b = ckl::random_matrix(n, 1, ckl::test::seed_stream(6501));

    ckl::DenseSolver solver;
    ckl::DeviceBuffer<float> da(a.size());
    ckl::DeviceBuffer<float> db(b.size());
    da.copy_from_host(a);
    db.copy_from_host(b);
    try {
        solver.solve_cholesky(da.data(), db.data(), n, 1, ckl::Fill::kLower);
        FAIL() << "an indefinite matrix has to be reported, not factored anyway";
    } catch (const ckl::Error& e) {
        EXPECT_EQ(e.status(), ckl::Status::kExecutionFailed) << e.what();
    }
    CKL_CUDA_CHECK(cudaDeviceSynchronize());
}

TEST_F(Solver, MovedFromSolverThrowsRatherThanCrashing) {
    ckl::DenseSolver a;
    ckl::DenseSolver b(std::move(a));

    ckl::DeviceBuffer<float> da(4);
    ckl::DeviceBuffer<float> db(2);
    da.zero();
    db.zero();
    // NOLINTNEXTLINE(bugprone-use-after-move)
    EXPECT_THROW(a.solve_lu(da.data(), db.data(), 2, 1), ckl::Error);
    // NOLINTNEXTLINE(bugprone-use-after-move)
    EXPECT_THROW(a.set_stream(nullptr), ckl::Error);
}

TEST_F(Solver, MoveAssignmentKeepsTheTargetUsable) {
    const int n = 64;
    const int nrhs = 2;
    const auto a = make_dominant(n, ckl::test::seed_stream(6601));
    const auto b = ckl::random_matrix(n, nrhs, ckl::test::seed_stream(6602));

    ckl::DenseSolver first;
    ckl::DenseSolver second;
    second = std::move(first);

    ckl::DeviceBuffer<float> da(a.size());
    ckl::DeviceBuffer<float> db(b.size());
    da.copy_from_host(a);
    db.copy_from_host(b);
    second.solve_lu(da.data(), db.data(), n, nrhs);
    CKL_CUDA_CHECK(cudaDeviceSynchronize());
    const auto x = db.to_host();
    EXPECT_LT(scaled_residual(a, x, b, n, nrhs), ckl::tol(n));
}

// One solver reused for several systems, which is the whole reason it owns its
// workspace: the second solve must not depend on the first having happened.
TEST_F(Solver, OneSolverHandlesSeveralSystems) {
    ckl::DenseSolver solver;
    for (int n : {32, 96, 64}) {
        SCOPED_TRACE(std::string("n = ") + std::to_string(n));
        const int nrhs = 3;
        const auto a =
            make_dominant(n, ckl::test::seed_stream(6701 + static_cast<std::uint64_t>(n)));
        const auto b = ckl::random_matrix(n, nrhs, ckl::test::seed_stream(6801));
        ckl::DeviceBuffer<float> da(a.size());
        ckl::DeviceBuffer<float> db(b.size());
        da.copy_from_host(a);
        db.copy_from_host(b);
        solver.solve_lu(da.data(), db.data(), n, nrhs);
        CKL_CUDA_CHECK(cudaDeviceSynchronize());
        const auto x = db.to_host();
        EXPECT_LT(scaled_residual(a, x, b, n, nrhs), ckl::tol(n));
    }
}

}  // namespace
