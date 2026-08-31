// FP32 GEMM ladder correctness.
//
// Every rung, including the cuBLAS oracle itself, is checked against a double
// precision CPU reference computed for the same data. Checking cuBLAS too is
// what catches a wrong transpose or leading dimension in the oracle, which a
// kernel versus oracle comparison alone would happily agree with.
//
// The gate is shape derived: tol(k) = 8 * sqrt(k) * FLT_EPSILON, applied twice.
// Once as a relative Frobenius error, which is tight enough that a kernel
// dropping its final K stage fails it. Once elementwise, as a residual scaled by
// sum_p |a_ip| |b_pj|, which is the bound that still means something when the
// data is built to cancel. The old flat 1e-4 was three orders of magnitude
// looser than the measured error at every shape here.

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
#include "ckl/gemm.hpp"
#include "gpu_environment.hpp"
#include "reference.hpp"

namespace {

using ckl::test::GpuTest;
using ckl::test::GpuTestWithParam;

using LaunchFn = std::function<void(const float*, const float*, float*, int, int, int, float, float,
                                    cudaStream_t)>;

enum class Variant { kNaive, kTiled, kRegister, kCpAsync, kCublas };

const char* variant_name(Variant v) {
    switch (v) {
        case Variant::kNaive:
            return "naive";
        case Variant::kTiled:
            return "tiled";
        case Variant::kRegister:
            return "reg";
        case Variant::kCpAsync:
            return "cpasync";
        case Variant::kCublas:
            return "cublas";
    }
    return "unknown";
}

LaunchFn variant_launch(Variant v) {
    switch (v) {
        case Variant::kNaive:
            return ckl::gemm_naive;
        case Variant::kTiled:
            return ckl::gemm_tiled;
        case Variant::kRegister:
            return ckl::gemm_register;
        case Variant::kCpAsync:
            return ckl::gemm_cp_async;
        case Variant::kCublas:
            return ckl::gemm_cublas;
    }
    return ckl::gemm_cublas;
}

const std::vector<Variant>& all_variants() {
    static const std::vector<Variant> v = {Variant::kNaive, Variant::kTiled, Variant::kRegister,
                                           Variant::kCpAsync, Variant::kCublas};
    return v;
}

struct Shape {
    int m;
    int n;
    int k;
    const char* label;
};

struct AlphaBeta {
    float alpha;
    float beta;
    const char* label;
};

// The five pairs Section 14 names. (1, 0) is the benchmarked configuration, so
// it has to be one of the correctness tested ones; v1 tested it nowhere.
const std::vector<AlphaBeta>& alpha_beta_cases() {
    static const std::vector<AlphaBeta> cases = {
        {1.0f, 0.0f, "a1_b0"},         {0.0f, 1.0f, "a0_b1"}, {1.0f, 1.0f, "a1_b1"},
        {2.5f, -0.75f, "a2p5_bm0p75"}, {0.0f, 0.0f, "a0_b0"},
    };
    return cases;
}

std::size_t elems(int rows, int cols) {
    return static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
}

// Runs one launcher and returns the host C. c_in is copied in first, so a
// caller can seed it with NaN, a canary, or real data.
std::vector<float> run_gemm(const LaunchFn& launch, const std::vector<float>& a,
                            const std::vector<float>& b, const std::vector<float>& c_in, int m,
                            int n, int k, float alpha, float beta) {
    ckl::DeviceBuffer<float> da(a.size());
    ckl::DeviceBuffer<float> db(b.size());
    ckl::DeviceBuffer<float> dc(c_in.size());
    da.copy_from_host(a);
    db.copy_from_host(b);
    dc.copy_from_host(c_in);
    launch(da.data(), db.data(), dc.data(), m, n, k, alpha, beta, nullptr);
    CKL_CUDA_CHECK(cudaDeviceSynchronize());
    return dc.size() == 0 ? std::vector<float>{} : dc.to_host();
}

// The whole gate for one benign case, in one place: finite output, tight
// Frobenius, and a tight elementwise residual.
void expect_gemm_matches(Variant v, const Shape& s, float alpha, float beta,
                         std::uint64_t stream_id) {
    const auto a = ckl::random_matrix(s.m, s.k, ckl::test::seed_stream(stream_id));
    const auto b = ckl::random_matrix(s.k, s.n, ckl::test::seed_stream(stream_id + 1));
    const auto c0 = ckl::random_matrix(s.m, s.n, ckl::test::seed_stream(stream_id + 2));

    const auto out = run_gemm(variant_launch(v), a, b, c0, s.m, s.n, s.k, alpha, beta);
    const ckl::GemmRef ref = ckl::gemm_reference_scaled(a, b, c0, s.m, s.n, s.k, alpha, beta);

    const double gate = ckl::tol(s.k);
    ASSERT_TRUE(ckl::all_finite(out)) << "non finite output from " << variant_name(v);
    EXPECT_LT(ckl::relative_frobenius_error(out, ref.c), gate)
        << variant_name(v) << " " << s.label << " frobenius, gate " << gate;
    EXPECT_LT(ckl::max_scaled_residual(out, ref), gate)
        << variant_name(v) << " " << s.label << " elementwise scaled residual, gate " << gate;
}

// ---------------------------------------------------------------------------
// Shape sweep
// ---------------------------------------------------------------------------

using ShapeParam = std::tuple<Variant, Shape>;

class GemmShape : public GpuTestWithParam<ShapeParam> {};

TEST_P(GemmShape, MatchesDoubleReference) {
    const auto [v, s] = GetParam();
    expect_gemm_matches(v, s, 1.25f, 0.5f, 101);
}

std::string shape_param_name(const ::testing::TestParamInfo<ShapeParam>& info) {
    return std::string(variant_name(std::get<0>(info.param))) + "_" + std::get<1>(info.param).label;
}

const std::vector<Shape>& sweep_shapes() {
    static const std::vector<Shape> shapes = {
        {128, 128, 128, "square_aligned"},   {192, 256, 64, "non_square_aligned"},
        {129, 257, 193, "non_tile_aligned"}, {7, 5, 11, "smaller_than_one_tile"},
        {256, 128, 320, "non_square_large"},
    };
    return shapes;
}

INSTANTIATE_TEST_SUITE_P(Fp32, GemmShape,
                         ::testing::Combine(::testing::ValuesIn(all_variants()),
                                            ::testing::ValuesIn(sweep_shapes())),
                         shape_param_name);

// ---------------------------------------------------------------------------
// Alpha and beta
// ---------------------------------------------------------------------------

using AbParam = std::tuple<Variant, AlphaBeta>;

class GemmAlphaBeta : public GpuTestWithParam<AbParam> {};

TEST_P(GemmAlphaBeta, AlignedAndUnaligned) {
    const auto [v, ab] = GetParam();
    // Both a shape the vectorized rungs take on their fast path and one that
    // sends them down the fallback, so a beta ignoring epilogue is caught on
    // either path.
    const Shape aligned{128, 128, 128, "aligned"};
    const Shape unaligned{65, 33, 17, "unaligned"};
    {
        SCOPED_TRACE("aligned 128x128x128");
        expect_gemm_matches(v, aligned, ab.alpha, ab.beta, 211);
    }
    {
        SCOPED_TRACE("unaligned 65x33x17");
        expect_gemm_matches(v, unaligned, ab.alpha, ab.beta, 217);
    }
}

std::string ab_param_name(const ::testing::TestParamInfo<AbParam>& info) {
    return std::string(variant_name(std::get<0>(info.param))) + "_" + std::get<1>(info.param).label;
}

INSTANTIATE_TEST_SUITE_P(Fp32, GemmAlphaBeta,
                         ::testing::Combine(::testing::ValuesIn(all_variants()),
                                            ::testing::ValuesIn(alpha_beta_cases())),
                         ab_param_name);

// ---------------------------------------------------------------------------
// beta = 0 must not read C
// ---------------------------------------------------------------------------

struct FillKind {
    float value;
    const char* label;
};

using NonFiniteParam = std::tuple<Variant, FillKind>;

class GemmBetaZeroNonFinite : public GpuTestWithParam<NonFiniteParam> {};

// BLAS says C is not read when beta is zero, so a C full of NaN or Inf is legal
// input. Every v1 kernel computed alpha * acc + 0.0f * c, which turns the whole
// output non finite. This test is the guard on the Part 03 fix.
TEST_P(GemmBetaZeroNonFinite, OutputIsFiniteAndCorrect) {
    const auto [v, fill] = GetParam();
    const Shape s{128, 128, 192, "beta_zero"};
    const auto a = ckl::random_matrix(s.m, s.k, ckl::test::seed_stream(301));
    const auto b = ckl::random_matrix(s.k, s.n, ckl::test::seed_stream(302));
    const std::vector<float> poisoned(elems(s.m, s.n), fill.value);
    const std::vector<float> zeros(elems(s.m, s.n), 0.0f);

    const auto out = run_gemm(variant_launch(v), a, b, poisoned, s.m, s.n, s.k, 1.0f, 0.0f);
    ASSERT_TRUE(ckl::all_finite(out))
        << variant_name(v) << " read C while beta was zero (C was " << fill.label << ")";

    const ckl::GemmRef ref = ckl::gemm_reference_scaled(a, b, zeros, s.m, s.n, s.k, 1.0f, 0.0f);
    const double gate = ckl::tol(s.k);
    EXPECT_LT(ckl::relative_frobenius_error(out, ref.c), gate);
    EXPECT_LT(ckl::max_scaled_residual(out, ref), gate);
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

INSTANTIATE_TEST_SUITE_P(Fp32, GemmBetaZeroNonFinite,
                         ::testing::Combine(::testing::ValuesIn(all_variants()),
                                            ::testing::ValuesIn(non_finite_fills())),
                         non_finite_param_name);

// ---------------------------------------------------------------------------
// Edge shapes
// ---------------------------------------------------------------------------

class GemmVariant : public GpuTestWithParam<Variant> {};

std::string variant_param_name(const ::testing::TestParamInfo<Variant>& info) {
    return variant_name(info.param);
}

TEST_P(GemmVariant, DegenerateShapes) {
    const std::vector<Shape> shapes = {
        {1, 1, 1, "1x1x1"}, {1, 1, 128, "1x1x128"}, {1, 8, 16, "1x8x16"},   {8, 1, 16, "8x1x16"},
        {2, 3, 5, "2x3x5"}, {1, 128, 1, "1x128x1"}, {128, 1, 1, "128x1x1"}, {16, 16, 1, "16x16x1"},
    };
    for (const auto& s : shapes) {
        SCOPED_TRACE(s.label);
        expect_gemm_matches(GetParam(), s, 1.25f, 0.5f, 401);
    }
}

// 128 is every block factor in the tree, 64 is the WMMA tile, and a kernel that
// gets its boundary predicate off by one fails on exactly one side of each.
TEST_P(GemmVariant, TileBoundaryCrossings) {
    const std::vector<Shape> shapes = {
        {127, 127, 127, "127"}, {128, 128, 128, "128"}, {129, 129, 129, "129"},
        {63, 63, 63, "63"},     {64, 64, 64, "64"},     {65, 65, 65, "65"},
    };
    for (const auto& s : shapes) {
        SCOPED_TRACE(s.label);
        expect_gemm_matches(GetParam(), s, 1.25f, 0.5f, 501);
    }
}

// The sweep runs k up to 8192 while v1's largest tested k was 640, so the error
// growth over a long contraction was never gated. tol(8192) is 8.6e-5, two
// orders tighter than the flat 1e-4 it replaces.
TEST_P(GemmVariant, LargeK) {
    const Shape small{64, 64, 8192, "64x64x8192"};
    {
        SCOPED_TRACE("small m and n");
        expect_gemm_matches(GetParam(), small, 1.25f, 0.5f, 601);
    }
    const Shape aligned{128, 128, 8192, "128x128x8192"};
    {
        SCOPED_TRACE("block aligned");
        expect_gemm_matches(GetParam(), aligned, 1.0f, 0.0f, 611);
    }
}

// Near cancellation. Elements sit at 1e12 and 1e-12, so a row of A meets a
// column of B in partial products spanning 48 orders of magnitude and the exact
// sum is far smaller than its largest term. A plain relative error against the
// double result is meaningless here; the scaled residual is the bound that
// holds, and it is the only one asserted.
TEST_P(GemmVariant, MixedMagnitudeNearCancellation) {
    const Shape s{96, 96, 256, "mixed"};
    const auto a = ckl::mixed_magnitude_matrix(s.m, s.k, ckl::test::seed_stream(701), 1.0e12f);
    const auto b = ckl::mixed_magnitude_matrix(s.k, s.n, ckl::test::seed_stream(702), 1.0e12f);
    const std::vector<float> c0(elems(s.m, s.n), 0.0f);

    const auto out = run_gemm(variant_launch(GetParam()), a, b, c0, s.m, s.n, s.k, 1.0f, 0.0f);
    const ckl::GemmRef ref = ckl::gemm_reference_scaled(a, b, c0, s.m, s.n, s.k, 1.0f, 0.0f);

    ASSERT_TRUE(ckl::all_finite(out)) << "the dataset stays inside float range; a non finite "
                                         "result means the kernel overflowed on its own";
    EXPECT_LT(ckl::max_scaled_residual(out, ref), ckl::tol(s.k));
}

// A zero dimension is where a short circuit hides. The old suite compared two
// empty vectors and passed unconditionally. Here C is filled with a canary and
// the assertion is that the canary survives byte for byte, so a kernel that
// wrote anything at all into a zero sized output is caught.
TEST_P(GemmVariant, ZeroDimensionsLeaveTheOutputAlone) {
    const Variant v = GetParam();
    const float canary = -12345.75f;
    const std::vector<Shape> empty_shapes = {{0, 64, 32, "m0"}, {64, 0, 32, "n0"}};
    for (const auto& s : empty_shapes) {
        SCOPED_TRACE(s.label);
        // The buffer is a full 64 by 64 regardless of the shape, so anything the
        // kernel writes lands somewhere the check looks at.
        const std::size_t guard = elems(64, 64);
        const auto a = ckl::random_matrix(64, s.k, ckl::test::seed_stream(801));
        const auto b = ckl::random_matrix(s.k, 64, ckl::test::seed_stream(802));
        const std::vector<float> c_in(guard, canary);
        const auto out = run_gemm(variant_launch(v), a, b, c_in, s.m, s.n, s.k, 1.25f, 0.5f);
        ASSERT_EQ(out.size(), guard);
        for (std::size_t i = 0; i < guard; ++i) {
            ASSERT_EQ(out[i], canary) << "element " << i << " was written for an empty output";
        }
    }
}

// An empty contraction is not an empty output: BLAS says C = beta * C, and with
// beta zero that is a cleared C, not a scaled one, because a NaN C is legal
// input when beta is zero.
TEST_P(GemmVariant, ZeroKAppliesBetaOnly) {
    const Variant v = GetParam();
    const int m = 64;
    const int n = 64;
    const auto a = ckl::random_matrix(m, 1, ckl::test::seed_stream(811));
    const auto b = ckl::random_matrix(1, n, ckl::test::seed_stream(812));
    const auto c0 = ckl::random_matrix(m, n, ckl::test::seed_stream(813));

    {
        SCOPED_TRACE("beta 0.5 scales C");
        const auto out = run_gemm(variant_launch(v), a, b, c0, m, n, 0, 1.25f, 0.5f);
        ASSERT_EQ(out.size(), elems(m, n));
        for (std::size_t i = 0; i < out.size(); ++i) {
            ASSERT_FLOAT_EQ(out[i], 0.5f * c0[i]) << "element " << i;
        }
    }
    {
        SCOPED_TRACE("beta 0 clears C without reading it");
        const std::vector<float> nan_c(elems(m, n), std::numeric_limits<float>::quiet_NaN());
        const auto out = run_gemm(variant_launch(v), a, b, nan_c, m, n, 0, 1.25f, 0.0f);
        ASSERT_EQ(out.size(), elems(m, n));
        for (std::size_t i = 0; i < out.size(); ++i) {
            ASSERT_EQ(out[i], 0.0f) << "element " << i << " came back " << out[i];
        }
    }
}

INSTANTIATE_TEST_SUITE_P(Fp32, GemmVariant, ::testing::ValuesIn(all_variants()),
                         variant_param_name);

// ---------------------------------------------------------------------------
// Streams, aliasing
// ---------------------------------------------------------------------------

class GemmStreams : public GpuTest {};

// Two non-blocking streams carrying different kernels at once. The point is not
// speed: it is that neither launcher touches the default stream or global state
// behind the caller's back, which a kernel that ignored its stream argument
// would show up as a wrong result here.
TEST_F(GemmStreams, TwoConcurrentStreams) {
    const int m = 128;
    const int n = 128;
    const int k = 256;
    const float alpha = 1.25f;
    const float beta = 0.5f;
    const auto a = ckl::random_matrix(m, k, ckl::test::seed_stream(901));
    const auto b = ckl::random_matrix(k, n, ckl::test::seed_stream(902));
    const auto c0 = ckl::random_matrix(m, n, ckl::test::seed_stream(903));

    cudaStream_t s1 = nullptr;
    cudaStream_t s2 = nullptr;
    CKL_CUDA_CHECK(cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking));
    CKL_CUDA_CHECK(cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking));

    ckl::DeviceBuffer<float> da(a.size());
    ckl::DeviceBuffer<float> db(b.size());
    ckl::DeviceBuffer<float> dc1(c0.size());
    ckl::DeviceBuffer<float> dc2(c0.size());
    da.copy_from_host(a);
    db.copy_from_host(b);
    dc1.copy_from_host(c0);
    dc2.copy_from_host(c0);

    ckl::gemm_naive(da.data(), db.data(), dc1.data(), m, n, k, alpha, beta, s1);
    ckl::gemm_tiled(da.data(), db.data(), dc2.data(), m, n, k, alpha, beta, s2);
    CKL_CUDA_CHECK(cudaStreamSynchronize(s1));
    CKL_CUDA_CHECK(cudaStreamSynchronize(s2));

    const auto out1 = dc1.to_host();
    const auto out2 = dc2.to_host();
    CKL_CUDA_CHECK(cudaStreamDestroy(s1));
    CKL_CUDA_CHECK(cudaStreamDestroy(s2));

    const ckl::GemmRef ref = ckl::gemm_reference_scaled(a, b, c0, m, n, k, alpha, beta);
    const double gate = ckl::tol(k);
    EXPECT_LT(ckl::max_scaled_residual(out1, ref), gate) << "naive on stream 1";
    EXPECT_LT(ckl::max_scaled_residual(out2, ref), gate) << "tiled on stream 2";
}

// The aliasing contract, written down as a test because it is not written down
// anywhere else. C must not alias A or B: every kernel here reads A and B while
// it writes C, with no ordering between blocks, so an aliased call reads a mix
// of original and overwritten values. The library does not detect it and does
// not pretend to; what this locks in is that it does not crash or corrupt memory
// outside the buffer, so compute-sanitizer stays clean and the failure mode is a
// wrong number rather than an illegal address.
class GemmAliasing : public GpuTest {};

TEST_F(GemmAliasing, CAliasingAIsAcceptedAndStaysInBounds) {
    const int n = 128;  // square, so A, B and C are the same size
    const auto a = ckl::random_matrix(n, n, ckl::test::seed_stream(921));
    const auto b = ckl::random_matrix(n, n, ckl::test::seed_stream(922));

    ckl::DeviceBuffer<float> da(a.size());
    ckl::DeviceBuffer<float> db(b.size());
    da.copy_from_host(a);
    db.copy_from_host(b);

    // C is A. Undefined by the contract; the assertion is bounded memory and a
    // clean CUDA state, not a particular answer.
    ckl::gemm_naive(da.data(), db.data(), da.data(), n, n, n, 1.0f, 0.0f, nullptr);
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    const auto out = da.to_host();
    EXPECT_EQ(out.size(), elems(n, n));
}

TEST_F(GemmAliasing, CAliasingBIsAcceptedAndStaysInBounds) {
    const int n = 128;
    const auto a = ckl::random_matrix(n, n, ckl::test::seed_stream(931));
    const auto b = ckl::random_matrix(n, n, ckl::test::seed_stream(932));

    ckl::DeviceBuffer<float> da(a.size());
    ckl::DeviceBuffer<float> db(b.size());
    da.copy_from_host(a);
    db.copy_from_host(b);

    ckl::gemm_tiled(da.data(), db.data(), db.data(), n, n, n, 1.0f, 0.0f, nullptr);
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    const auto out = db.to_host();
    EXPECT_EQ(out.size(), elems(n, n));
}

}  // namespace
