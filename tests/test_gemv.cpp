// GEMV correctness: y = alpha * (A * x) + beta * y, row major, A is m by n.
//
// Same structure as the FP32 GEMM suite and the same gate. The contraction
// length here is n, so the tolerance is tol(n), and the elementwise residual is
// scaled by sum_j |a_ij| |x_j| rather than by |y_i|, which keeps one bound valid
// whether the row sums cleanly or cancels.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <string>
#include <tuple>
#include <vector>

#include <cuda_runtime.h>

#include <gtest/gtest.h>

#include "ckl/cuda_check.hpp"
#include "ckl/device_buffer.hpp"
#include "ckl/gemv.hpp"
#include "gpu_environment.hpp"
#include "reference.hpp"

namespace {

using ckl::test::GpuTest;
using ckl::test::GpuTestWithParam;

using LaunchFn =
    std::function<void(const float*, const float*, float*, int, int, float, float, cudaStream_t)>;

enum class Variant { kNaive, kWarp, kVectorized, kCublas };

const char* variant_name(Variant v) {
    switch (v) {
        case Variant::kNaive:
            return "naive";
        case Variant::kWarp:
            return "warp";
        case Variant::kVectorized:
            return "vec";
        case Variant::kCublas:
            return "cublas";
    }
    return "unknown";
}

LaunchFn variant_launch(Variant v) {
    switch (v) {
        case Variant::kNaive:
            return ckl::gemv_naive;
        case Variant::kWarp:
            return ckl::gemv_warp;
        case Variant::kVectorized:
            return ckl::gemv_vectorized;
        case Variant::kCublas:
            return ckl::gemv_cublas;
    }
    return ckl::gemv_cublas;
}

const std::vector<Variant>& all_variants() {
    static const std::vector<Variant> v = {Variant::kNaive, Variant::kWarp, Variant::kVectorized,
                                           Variant::kCublas};
    return v;
}

struct Shape {
    int m;
    int n;
    const char* label;
};

struct AlphaBeta {
    float alpha;
    float beta;
    const char* label;
};

const std::vector<AlphaBeta>& alpha_beta_cases() {
    static const std::vector<AlphaBeta> cases = {
        {1.0f, 0.0f, "a1_b0"},         {0.0f, 1.0f, "a0_b1"}, {1.0f, 1.0f, "a1_b1"},
        {2.5f, -0.75f, "a2p5_bm0p75"}, {0.0f, 0.0f, "a0_b0"},
    };
    return cases;
}

struct GemvRef {
    std::vector<double> y;
    std::vector<double> scale;
};

GemvRef gemv_reference(const std::vector<float>& a, const std::vector<float>& x,
                       const std::vector<float>& y0, int m, int n, float alpha, float beta) {
    GemvRef ref;
    ref.y.assign(static_cast<std::size_t>(m), 0.0);
    ref.scale.assign(static_cast<std::size_t>(m), 0.0);
    for (int i = 0; i < m; ++i) {
        double acc = 0.0;
        double mag = 0.0;
        for (int j = 0; j < n; ++j) {
            const double av =
                static_cast<double>(a[static_cast<std::size_t>(i) * static_cast<std::size_t>(n) +
                                      static_cast<std::size_t>(j)]);
            const double xv = static_cast<double>(x[static_cast<std::size_t>(j)]);
            acc += av * xv;
            mag += std::fabs(av) * std::fabs(xv);
        }
        const double y0i = static_cast<double>(y0[static_cast<std::size_t>(i)]);
        ref.y[static_cast<std::size_t>(i)] =
            static_cast<double>(alpha) * acc + static_cast<double>(beta) * y0i;
        ref.scale[static_cast<std::size_t>(i)] =
            std::fabs(static_cast<double>(alpha)) * mag +
            std::fabs(static_cast<double>(beta)) * std::fabs(y0i);
    }
    return ref;
}

double worst_scaled(const std::vector<float>& actual, const GemvRef& ref) {
    double worst = 0.0;
    for (std::size_t i = 0; i < ref.y.size(); ++i) {
        const double denom = ref.scale[i] > 1e-30 ? ref.scale[i] : 1.0;
        const double r = std::fabs(static_cast<double>(actual[i]) - ref.y[i]) / denom;
        if (r > worst) {
            worst = r;
        }
    }
    return worst;
}

std::vector<float> run_gemv(const LaunchFn& launch, const std::vector<float>& a,
                            const std::vector<float>& x, const std::vector<float>& y0, int m, int n,
                            float alpha, float beta) {
    ckl::DeviceBuffer<float> da(a.size());
    ckl::DeviceBuffer<float> dx(x.size());
    ckl::DeviceBuffer<float> dy(y0.size());
    da.copy_from_host(a);
    dx.copy_from_host(x);
    dy.copy_from_host(y0);
    launch(da.data(), dx.data(), dy.data(), m, n, alpha, beta, nullptr);
    CKL_CUDA_CHECK(cudaDeviceSynchronize());
    return dy.size() == 0 ? std::vector<float>{} : dy.to_host();
}

void expect_gemv_matches(Variant v, const Shape& s, float alpha, float beta,
                         std::uint64_t stream_id) {
    const auto a = ckl::random_matrix(s.m, s.n, ckl::test::seed_stream(stream_id));
    const auto x = ckl::random_matrix(s.n, 1, ckl::test::seed_stream(stream_id + 1));
    const auto y0 = ckl::random_matrix(s.m, 1, ckl::test::seed_stream(stream_id + 2));

    const auto out = run_gemv(variant_launch(v), a, x, y0, s.m, s.n, alpha, beta);
    const GemvRef ref = gemv_reference(a, x, y0, s.m, s.n, alpha, beta);

    ASSERT_TRUE(ckl::all_finite(out)) << variant_name(v) << " produced a non finite result";
    const double gate = ckl::tol(s.n);
    EXPECT_LT(ckl::relative_frobenius_error(out, ref.y), gate)
        << variant_name(v) << " " << s.label << " frobenius, gate " << gate;
    EXPECT_LT(worst_scaled(out, ref), gate)
        << variant_name(v) << " " << s.label << " elementwise scaled residual";
}

// ---------------------------------------------------------------------------

using ShapeParam = std::tuple<Variant, Shape>;

class GemvShape : public GpuTestWithParam<ShapeParam> {};

TEST_P(GemvShape, MatchesDoubleReference) {
    const auto [v, s] = GetParam();
    expect_gemv_matches(v, s, 1.3f, 0.4f, 3101);
}

std::string shape_param_name(const ::testing::TestParamInfo<ShapeParam>& info) {
    return std::string(variant_name(std::get<0>(info.param))) + "_" + std::get<1>(info.param).label;
}

const std::vector<Shape>& sweep_shapes() {
    static const std::vector<Shape> shapes = {
        {1024, 1024, "square"},          {2048, 512, "tall"},  {512, 2048, "wide"},
        {777, 333, "non_multiple_of_4"}, {1, 2048, "one_row"}, {64, 63, "odd_row_length"},
    };
    return shapes;
}

INSTANTIATE_TEST_SUITE_P(Fp32, GemvShape,
                         ::testing::Combine(::testing::ValuesIn(all_variants()),
                                            ::testing::ValuesIn(sweep_shapes())),
                         shape_param_name);

using AbParam = std::tuple<Variant, AlphaBeta>;

class GemvAlphaBeta : public GpuTestWithParam<AbParam> {};

// n divisible by four takes the float4 path, n not divisible falls back to the
// scalar warp kernel; the epilogue is written once per kernel, so both need the
// alpha and beta sweep.
TEST_P(GemvAlphaBeta, VectorizedAndScalarRowLengths) {
    const auto [v, ab] = GetParam();
    {
        SCOPED_TRACE("n divisible by four");
        expect_gemv_matches(v, Shape{256, 256, "vec"}, ab.alpha, ab.beta, 3201);
    }
    {
        SCOPED_TRACE("n not divisible by four");
        expect_gemv_matches(v, Shape{129, 131, "scalar"}, ab.alpha, ab.beta, 3207);
    }
}

std::string ab_param_name(const ::testing::TestParamInfo<AbParam>& info) {
    return std::string(variant_name(std::get<0>(info.param))) + "_" + std::get<1>(info.param).label;
}

INSTANTIATE_TEST_SUITE_P(Fp32, GemvAlphaBeta,
                         ::testing::Combine(::testing::ValuesIn(all_variants()),
                                            ::testing::ValuesIn(alpha_beta_cases())),
                         ab_param_name);

struct FillKind {
    float value;
    const char* label;
};

using NonFiniteParam = std::tuple<Variant, FillKind>;

class GemvBetaZeroNonFinite : public GpuTestWithParam<NonFiniteParam> {};

TEST_P(GemvBetaZeroNonFinite, OutputIsFiniteAndCorrect) {
    const auto [v, fill] = GetParam();
    const Shape s{512, 512, "beta_zero"};
    const auto a = ckl::random_matrix(s.m, s.n, ckl::test::seed_stream(3301));
    const auto x = ckl::random_matrix(s.n, 1, ckl::test::seed_stream(3302));
    const std::vector<float> poisoned(static_cast<std::size_t>(s.m), fill.value);
    const std::vector<float> zeros(static_cast<std::size_t>(s.m), 0.0f);

    const auto out = run_gemv(variant_launch(v), a, x, poisoned, s.m, s.n, 1.0f, 0.0f);
    ASSERT_TRUE(ckl::all_finite(out))
        << variant_name(v) << " read y while beta was zero (y was " << fill.label << ")";
    const GemvRef ref = gemv_reference(a, x, zeros, s.m, s.n, 1.0f, 0.0f);
    EXPECT_LT(worst_scaled(out, ref), ckl::tol(s.n));
}

std::string non_finite_param_name(const ::testing::TestParamInfo<NonFiniteParam>& info) {
    return std::string(variant_name(std::get<0>(info.param))) + "_" + std::get<1>(info.param).label;
}

const std::vector<FillKind>& non_finite_fills() {
    static const std::vector<FillKind> fills = {
        {std::numeric_limits<float>::quiet_NaN(), "nan"},
        {std::numeric_limits<float>::infinity(), "inf"},
    };
    return fills;
}

INSTANTIATE_TEST_SUITE_P(Fp32, GemvBetaZeroNonFinite,
                         ::testing::Combine(::testing::ValuesIn(all_variants()),
                                            ::testing::ValuesIn(non_finite_fills())),
                         non_finite_param_name);

class GemvVariant : public GpuTestWithParam<Variant> {};

std::string variant_param_name(const ::testing::TestParamInfo<Variant>& info) {
    return variant_name(info.param);
}

TEST_P(GemvVariant, DegenerateShapes) {
    const std::vector<Shape> shapes = {
        {1, 1, "1x1"}, {1, 4, "1x4"}, {4, 1, "4x1"}, {1, 3, "1x3"}, {3, 1, "3x1"}, {2, 5, "2x5"},
    };
    for (const auto& s : shapes) {
        SCOPED_TRACE(s.label);
        expect_gemv_matches(GetParam(), s, 1.3f, 0.4f, 3401);
    }
}

// One warp handles a row, so 32 and its multiples are where a lane predicate
// goes wrong, and the float4 path changes behaviour at multiples of four.
TEST_P(GemvVariant, RowLengthCrossings) {
    const std::vector<Shape> shapes = {
        {64, 63, "63"},   {64, 64, "64"},  {64, 65, "65"},  {64, 127, "127"}, {64, 128, "128"},
        {64, 129, "129"}, {33, 32, "m33"}, {32, 32, "m32"}, {31, 32, "m31"},
    };
    for (const auto& s : shapes) {
        SCOPED_TRACE(s.label);
        expect_gemv_matches(GetParam(), s, 1.3f, 0.4f, 3501);
    }
}

TEST_P(GemvVariant, LongRows) {
    expect_gemv_matches(GetParam(), Shape{64, 8192, "64x8192"}, 1.3f, 0.4f, 3601);
}

TEST_P(GemvVariant, MixedMagnitudeNearCancellation) {
    const Shape s{64, 512, "mixed"};
    const auto a = ckl::mixed_magnitude_matrix(s.m, s.n, ckl::test::seed_stream(3701), 1.0e12f);
    const auto x = ckl::mixed_magnitude_matrix(s.n, 1, ckl::test::seed_stream(3702), 1.0e12f);
    const std::vector<float> y0(static_cast<std::size_t>(s.m), 0.0f);

    const auto out = run_gemv(variant_launch(GetParam()), a, x, y0, s.m, s.n, 1.0f, 0.0f);
    ASSERT_TRUE(ckl::all_finite(out));
    const GemvRef ref = gemv_reference(a, x, y0, s.m, s.n, 1.0f, 0.0f);
    EXPECT_LT(worst_scaled(out, ref), ckl::tol(s.n));
}

TEST_P(GemvVariant, ZeroRowsLeaveTheOutputAlone) {
    const float canary = -98765.5f;
    const int n = 64;
    const auto a = ckl::random_matrix(64, n, ckl::test::seed_stream(3801));
    const auto x = ckl::random_matrix(n, 1, ckl::test::seed_stream(3802));
    const std::vector<float> y0(64, canary);
    const auto out = run_gemv(variant_launch(GetParam()), a, x, y0, 0, n, 1.3f, 0.4f);
    ASSERT_EQ(out.size(), y0.size());
    for (std::size_t i = 0; i < out.size(); ++i) {
        ASSERT_EQ(out[i], canary) << "element " << i << " was written for a zero row output";
    }
}

// An empty row is not an empty output: y = beta * y, and beta zero clears rather
// than scales, because y is not read when beta is zero.
TEST_P(GemvVariant, ZeroColumnsApplyBetaOnly) {
    const int m = 64;
    const auto a = ckl::random_matrix(m, 1, ckl::test::seed_stream(3811));
    const auto x = ckl::random_matrix(1, 1, ckl::test::seed_stream(3812));
    const auto y0 = ckl::random_matrix(m, 1, ckl::test::seed_stream(3813));
    {
        SCOPED_TRACE("beta 0.4 scales y");
        const auto out = run_gemv(variant_launch(GetParam()), a, x, y0, m, 0, 1.3f, 0.4f);
        for (std::size_t i = 0; i < out.size(); ++i) {
            ASSERT_FLOAT_EQ(out[i], 0.4f * y0[i]) << "element " << i;
        }
    }
    {
        SCOPED_TRACE("beta 0 clears y without reading it");
        const std::vector<float> nan_y(static_cast<std::size_t>(m),
                                       std::numeric_limits<float>::quiet_NaN());
        const auto out = run_gemv(variant_launch(GetParam()), a, x, nan_y, m, 0, 1.3f, 0.0f);
        for (std::size_t i = 0; i < out.size(); ++i) {
            ASSERT_EQ(out[i], 0.0f) << "element " << i << " came back " << out[i];
        }
    }
}

INSTANTIATE_TEST_SUITE_P(Fp32, GemvVariant, ::testing::ValuesIn(all_variants()),
                         variant_param_name);

class GemvStreams : public GpuTest {};

TEST_F(GemvStreams, TwoConcurrentStreams) {
    const int m = 512;
    const int n = 512;
    const float alpha = 1.3f;
    const float beta = 0.4f;
    const auto a = ckl::random_matrix(m, n, ckl::test::seed_stream(3901));
    const auto x = ckl::random_matrix(n, 1, ckl::test::seed_stream(3902));
    const auto y0 = ckl::random_matrix(m, 1, ckl::test::seed_stream(3903));

    cudaStream_t s1 = nullptr;
    cudaStream_t s2 = nullptr;
    CKL_CUDA_CHECK(cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking));
    CKL_CUDA_CHECK(cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking));

    ckl::DeviceBuffer<float> da(a.size());
    ckl::DeviceBuffer<float> dx(x.size());
    ckl::DeviceBuffer<float> dy1(y0.size());
    ckl::DeviceBuffer<float> dy2(y0.size());
    da.copy_from_host(a);
    dx.copy_from_host(x);
    dy1.copy_from_host(y0);
    dy2.copy_from_host(y0);

    ckl::gemv_warp(da.data(), dx.data(), dy1.data(), m, n, alpha, beta, s1);
    ckl::gemv_vectorized(da.data(), dx.data(), dy2.data(), m, n, alpha, beta, s2);
    CKL_CUDA_CHECK(cudaStreamSynchronize(s1));
    CKL_CUDA_CHECK(cudaStreamSynchronize(s2));
    const auto out1 = dy1.to_host();
    const auto out2 = dy2.to_host();
    CKL_CUDA_CHECK(cudaStreamDestroy(s1));
    CKL_CUDA_CHECK(cudaStreamDestroy(s2));

    const GemvRef ref = gemv_reference(a, x, y0, m, n, alpha, beta);
    EXPECT_LT(worst_scaled(out1, ref), ckl::tol(n)) << "warp on stream 1";
    EXPECT_LT(worst_scaled(out2, ref), ckl::tol(n)) << "vectorized on stream 2";
}

}  // namespace
