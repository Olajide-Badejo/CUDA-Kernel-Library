// Triangular solve: L X = alpha B, L lower triangular non unit diagonal, row
// major, X written in place over B.
//
// v1 only ever solved diagonally dominant systems, where the condition number is
// about one and any implementation looks accurate. Two things are checked here
// instead of one. The scaled residual max_i |(L X)_i - alpha B_i| divided by
// sum_j |L_ij| |X_j| is backward stable, so it holds at tol(m) whatever the
// conditioning is. The forward error against the double solution is not, so its
// bound carries the condition number explicitly, and the ill conditioned case
// below is the one that tells those two apart.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <tuple>
#include <vector>

#include <cuda_runtime.h>

#include <gtest/gtest.h>

#include "ckl/cuda_check.hpp"
#include "ckl/device_buffer.hpp"
#include "ckl/trsm.hpp"
#include "gpu_environment.hpp"
#include "reference.hpp"

namespace {

using ckl::test::GpuTest;
using ckl::test::GpuTestWithParam;

using LaunchFn = std::function<void(const float*, float*, int, int, float, cudaStream_t)>;

enum class Variant { kNaive, kBlocked, kCublas };

const char* variant_name(Variant v) {
    switch (v) {
        case Variant::kNaive:
            return "naive";
        case Variant::kBlocked:
            return "blocked";
        case Variant::kCublas:
            return "cublas";
    }
    return "unknown";
}

LaunchFn variant_launch(Variant v) {
    switch (v) {
        case Variant::kNaive:
            return ckl::trsm_naive;
        case Variant::kBlocked:
            return ckl::trsm_blocked;
        case Variant::kCublas:
            return ckl::trsm_cublas;
    }
    return ckl::trsm_cublas;
}

const std::vector<Variant>& all_variants() {
    static const std::vector<Variant> v = {Variant::kNaive, Variant::kBlocked, Variant::kCublas};
    return v;
}

struct Shape {
    int m;
    int n;
    const char* label;
};

std::size_t elems(int rows, int cols) {
    return static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
}

// Diagonally dominant lower triangular: the well conditioned case.
std::vector<float> make_dominant(int m, std::uint64_t seed) {
    auto full = ckl::random_matrix(m, m, seed);
    for (int i = 0; i < m; ++i) {
        for (int j = i + 1; j < m; ++j) {
            full[static_cast<std::size_t>(i) * static_cast<std::size_t>(m) +
                 static_cast<std::size_t>(j)] = 0.0f;
        }
        full[static_cast<std::size_t>(i) * static_cast<std::size_t>(m) +
             static_cast<std::size_t>(i)] = static_cast<float>(m + 1);
    }
    return full;
}

// One nearly singular row. Every diagonal entry is 1 except row m/2, which
// carries small_diag, and the off diagonal entries are scaled by 1/m so that
// L inverse stays bounded except through that one tiny pivot. The condition
// number then lands near 1 / small_diag.
//
// Grading the whole diagonal geometrically, which is the obvious first attempt,
// does not work: the inverse of a triangular matrix with order one off diagonals
// and shrinking pivots grows like a product down the column, and a 128 row
// matrix reaches 1e22 rather than the 1e4 to 1e6 the spec asks for. One bad
// pivot gives a condition number you can aim at.
std::vector<float> make_ill_conditioned(int m, std::uint64_t seed, float small_diag) {
    auto full = ckl::random_matrix(m, m, seed);
    const float off_scale = 1.0f / static_cast<float>(m);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j) {
            const std::size_t idx = static_cast<std::size_t>(i) * static_cast<std::size_t>(m) +
                                    static_cast<std::size_t>(j);
            if (j > i) {
                full[idx] = 0.0f;
            } else if (j < i) {
                full[idx] *= off_scale;
            }
        }
        full[static_cast<std::size_t>(i) * static_cast<std::size_t>(m) +
             static_cast<std::size_t>(i)] = (i == m / 2) ? small_diag : 1.0f;
    }
    return full;
}

std::vector<double> to_double(const std::vector<float>& v) {
    return std::vector<double>(v.begin(), v.end());
}

// Exact solution by forward substitution in double.
std::vector<double> trsm_reference(const std::vector<float>& a, const std::vector<float>& b, int m,
                                   int n, float alpha) {
    std::vector<double> x(elems(m, n), 0.0);
    for (int c = 0; c < n; ++c) {
        for (int i = 0; i < m; ++i) {
            double s =
                static_cast<double>(alpha) *
                static_cast<double>(b[static_cast<std::size_t>(i) * static_cast<std::size_t>(n) +
                                      static_cast<std::size_t>(c)]);
            for (int j = 0; j < i; ++j) {
                s -= static_cast<double>(
                         a[static_cast<std::size_t>(i) * static_cast<std::size_t>(m) +
                           static_cast<std::size_t>(j)]) *
                     x[static_cast<std::size_t>(j) * static_cast<std::size_t>(n) +
                       static_cast<std::size_t>(c)];
            }
            x[static_cast<std::size_t>(i) * static_cast<std::size_t>(n) +
              static_cast<std::size_t>(c)] =
                s /
                static_cast<double>(a[static_cast<std::size_t>(i) * static_cast<std::size_t>(m) +
                                      static_cast<std::size_t>(i)]);
        }
    }
    return x;
}

// max_i |(L X)_i - alpha B_i| divided by sum_j |L_ij| |X_j|. Backward stable, so
// it holds at tol(m) no matter how ill conditioned L is.
double scaled_residual(const std::vector<float>& a, const std::vector<float>& x,
                       const std::vector<float>& b, int m, int n, float alpha) {
    double worst = 0.0;
    for (int i = 0; i < m; ++i) {
        for (int c = 0; c < n; ++c) {
            double lx = 0.0;
            double mag = 0.0;
            for (int j = 0; j <= i; ++j) {
                const double av = static_cast<double>(
                    a[static_cast<std::size_t>(i) * static_cast<std::size_t>(m) +
                      static_cast<std::size_t>(j)]);
                const double xv = static_cast<double>(
                    x[static_cast<std::size_t>(j) * static_cast<std::size_t>(n) +
                      static_cast<std::size_t>(c)]);
                lx += av * xv;
                mag += std::fabs(av) * std::fabs(xv);
            }
            const double target =
                static_cast<double>(alpha) *
                static_cast<double>(b[static_cast<std::size_t>(i) * static_cast<std::size_t>(n) +
                                      static_cast<std::size_t>(c)]);
            const double denom = mag > 1e-30 ? mag : 1.0;
            const double r = std::fabs(lx - target) / denom;
            if (r > worst) {
                worst = r;
            }
        }
    }
    return worst;
}

std::vector<float> run_trsm(Variant v, const std::vector<float>& a, const std::vector<float>& b0,
                            int m, int n, float alpha) {
    ckl::DeviceBuffer<float> da(a.size());
    ckl::DeviceBuffer<float> db(b0.size());
    da.copy_from_host(a);
    db.copy_from_host(b0);
    variant_launch(v)(da.data(), db.data(), m, n, alpha, nullptr);
    CKL_CUDA_CHECK(cudaDeviceSynchronize());
    return db.to_host();
}

// ---------------------------------------------------------------------------

using ShapeParam = std::tuple<Variant, Shape>;

class TrsmShape : public GpuTestWithParam<ShapeParam> {};

std::string shape_param_name(const ::testing::TestParamInfo<ShapeParam>& info) {
    return std::string(variant_name(std::get<0>(info.param))) + "_" + std::get<1>(info.param).label;
}

TEST_P(TrsmShape, ResidualAndForwardError) {
    const auto [v, s] = GetParam();
    const float alpha = 1.5f;
    const auto a = make_dominant(s.m, ckl::test::seed_stream(5101));
    const auto b0 = ckl::random_matrix(s.m, s.n, ckl::test::seed_stream(5102));

    const auto x = run_trsm(v, a, b0, s.m, s.n, alpha);
    ASSERT_TRUE(ckl::all_finite(x)) << variant_name(v) << " produced a non finite solution";
    EXPECT_LT(scaled_residual(a, x, b0, s.m, s.n, alpha), ckl::tol(s.m))
        << variant_name(v) << " " << s.label << " scaled residual";

    const auto exact = trsm_reference(a, b0, s.m, s.n, alpha);
    EXPECT_LT(ckl::relative_frobenius_error(x, exact), ckl::tol(s.m))
        << variant_name(v) << " " << s.label << " forward error against the double solve";
}

INSTANTIATE_TEST_SUITE_P(Lower, TrsmShape,
                         ::testing::Combine(::testing::ValuesIn(all_variants()),
                                            ::testing::Values(Shape{256, 128, "square_rhs"},
                                                              Shape{512, 64, "tall_few_rhs"},
                                                              Shape{129, 200, "odd_block"},
                                                              Shape{64, 1, "single_rhs"})),
                         shape_param_name);

class TrsmVariant : public GpuTestWithParam<Variant> {};

std::string variant_param_name(const ::testing::TestParamInfo<Variant>& info) {
    return variant_name(info.param);
}

TEST_P(TrsmVariant, AlphaSweep) {
    const Variant v = GetParam();
    const Shape s{128, 64, "alpha"};
    const auto a = make_dominant(s.m, ckl::test::seed_stream(5201));
    const auto b0 = ckl::random_matrix(s.m, s.n, ckl::test::seed_stream(5202));
    for (float alpha : {1.0f, 1.5f, -0.75f, 0.0f}) {
        SCOPED_TRACE(std::string("alpha ") + std::to_string(alpha));
        const auto x = run_trsm(v, a, b0, s.m, s.n, alpha);
        ASSERT_TRUE(ckl::all_finite(x));
        EXPECT_LT(scaled_residual(a, x, b0, s.m, s.n, alpha), ckl::tol(s.m));
        const auto exact = trsm_reference(a, b0, s.m, s.n, alpha);
        EXPECT_LT(ckl::relative_frobenius_error(x, exact), ckl::tol(s.m));
    }
}

// The conditioning case. The residual bound is unchanged, because a triangular
// solve is backward stable; the forward error bound is the one that has to grow,
// and it grows with the measured condition number rather than with a constant
// somebody guessed.
TEST_P(TrsmVariant, IllConditionedSystem) {
    const Variant v = GetParam();
    const int m = 128;
    const int n = 32;
    const float alpha = 1.0f;
    // A single pivot of 1e-5 puts the condition number in the 1e4 to 1e6 band
    // the spec asks for. Measured at the default seed it is about 1.9e5, and the
    // assertions below state what it actually measured on the run.
    const auto a = make_ill_conditioned(m, ckl::test::seed_stream(5301), 1.0e-5f);
    const auto b0 = ckl::random_matrix(m, n, ckl::test::seed_stream(5302));

    const double cond = ckl::condition_number_inf(to_double(a), m);
    ::testing::Test::RecordProperty("condition_number", ckl::test::sci(cond));
    ASSERT_GT(cond, 1.0e4) << "the graded matrix is meant to be ill conditioned";
    ASSERT_LT(cond, 1.0e7) << "and not so ill conditioned that float cannot solve it at all";

    const auto x = run_trsm(v, a, b0, m, n, alpha);
    ASSERT_TRUE(ckl::all_finite(x)) << variant_name(v) << " produced a non finite solution";

    EXPECT_LT(scaled_residual(a, x, b0, m, n, alpha), ckl::tol(m))
        << variant_name(v) << " residual, which conditioning does not excuse";

    const auto exact = trsm_reference(a, b0, m, n, alpha);
    const double forward_bound = ckl::tol(m) * cond;
    const double forward = ckl::relative_frobenius_error(x, exact);
    ::testing::Test::RecordProperty("forward_error", ckl::test::sci(forward));
    EXPECT_LT(forward, forward_bound)
        << variant_name(v) << " forward error, bound tol(m) * cond = " << forward_bound
        << " with cond " << cond;
}

TEST_P(TrsmVariant, DegenerateShapes) {
    const Variant v = GetParam();
    const std::vector<Shape> shapes = {
        {1, 1, "1x1"}, {1, 8, "1x8"}, {8, 1, "8x1"}, {2, 3, "2x3"}, {33, 5, "33x5"},
    };
    for (const auto& s : shapes) {
        SCOPED_TRACE(s.label);
        const float alpha = 1.5f;
        const auto a = make_dominant(s.m, ckl::test::seed_stream(5401));
        const auto b0 = ckl::random_matrix(s.m, s.n, ckl::test::seed_stream(5402));
        const auto x = run_trsm(v, a, b0, s.m, s.n, alpha);
        ASSERT_TRUE(ckl::all_finite(x));
        EXPECT_LT(scaled_residual(a, x, b0, s.m, s.n, alpha), ckl::tol(s.m));
        const auto exact = trsm_reference(a, b0, s.m, s.n, alpha);
        EXPECT_LT(ckl::relative_frobenius_error(x, exact), ckl::tol(s.m));
    }
}

TEST_P(TrsmVariant, ZeroDimensionsLeaveTheRightHandSideAlone) {
    const Variant v = GetParam();
    const float canary = -555.25f;
    const auto a = make_dominant(32, ckl::test::seed_stream(5501));
    const std::vector<float> b0(elems(32, 32), canary);
    for (int which = 0; which < 2; ++which) {
        SCOPED_TRACE(which == 0 ? "m = 0" : "n = 0");
        const int m = which == 0 ? 0 : 32;
        const int n = which == 0 ? 32 : 0;
        const auto x = run_trsm(v, a, b0, m, n, 1.5f);
        ASSERT_EQ(x.size(), b0.size());
        for (std::size_t i = 0; i < x.size(); ++i) {
            ASSERT_EQ(x[i], canary) << "element " << i << " was written for an empty solve";
        }
    }
}

INSTANTIATE_TEST_SUITE_P(Lower, TrsmVariant, ::testing::ValuesIn(all_variants()),
                         variant_param_name);

class TrsmStreams : public GpuTest {};

TEST_F(TrsmStreams, TwoConcurrentStreams) {
    const int m = 256;
    const int n = 64;
    const float alpha = 1.5f;
    const auto a = make_dominant(m, ckl::test::seed_stream(5601));
    const auto b0 = ckl::random_matrix(m, n, ckl::test::seed_stream(5602));

    cudaStream_t s1 = nullptr;
    cudaStream_t s2 = nullptr;
    CKL_CUDA_CHECK(cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking));
    CKL_CUDA_CHECK(cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking));

    ckl::DeviceBuffer<float> da(a.size());
    ckl::DeviceBuffer<float> db1(b0.size());
    ckl::DeviceBuffer<float> db2(b0.size());
    da.copy_from_host(a);
    db1.copy_from_host(b0);
    db2.copy_from_host(b0);

    ckl::trsm_naive(da.data(), db1.data(), m, n, alpha, s1);
    ckl::trsm_blocked(da.data(), db2.data(), m, n, alpha, s2);
    CKL_CUDA_CHECK(cudaStreamSynchronize(s1));
    CKL_CUDA_CHECK(cudaStreamSynchronize(s2));
    const auto x1 = db1.to_host();
    const auto x2 = db2.to_host();
    CKL_CUDA_CHECK(cudaStreamDestroy(s1));
    CKL_CUDA_CHECK(cudaStreamDestroy(s2));

    EXPECT_LT(scaled_residual(a, x1, b0, m, n, alpha), ckl::tol(m)) << "naive on stream 1";
    EXPECT_LT(scaled_residual(a, x2, b0, m, n, alpha), ckl::tol(m)) << "blocked on stream 2";
}

}  // namespace
