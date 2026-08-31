// Tensor core GEMM correctness: WMMA, mma.sync with scalar fragment reads,
// mma.sync with ldmatrix, and the optimized 128x128 kernel, in FP16 and BF16.
//
// Three errors are measured for every case, because they answer different
// questions and v1 conflated them.
//
//   vs_oracle   kernel against cuBLAS at the same input precision. Both consume
//               the same rounded inputs and accumulate in FP32, so the only
//               difference is summation order. Measured at the shapes below it
//               runs between 6e-8 and 4e-7, so the gate is 1e-5 relative
//               Frobenius; v1's was 5e-2, which a kernel that dropped its final
//               K stage would have sailed through. Every test records the number
//               it measured as a vs_oracle property in the XML report.
//   vs_rounded  kernel against a double precision reference over the same
//               rounded inputs. This isolates the accumulation error with no
//               input rounding mixed in, so tol(k) applies unchanged.
//   vs_fp32     kernel against a double precision reference over the original
//               FP32 data. This is the honest accuracy of the precision, and it
//               is bounded by the unit roundoff of the storage format rather
//               than by anything the kernel does.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <tuple>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <gtest/gtest.h>

#include "ckl/cuda_check.hpp"
#include "ckl/device_buffer.hpp"
#include "ckl/gemm.hpp"
#include "ckl/status.hpp"
#include "gpu_environment.hpp"
#include "reference.hpp"

namespace {

using ckl::test::GpuTest;
using ckl::test::GpuTestWithParam;

// Storage is carried as raw 16 bit words so one code path serves both formats;
// the launchers get the pointer reinterpreted at the call.
using Bits = std::vector<std::uint16_t>;

enum class TVariant { kWmmaFp16, kMmaPtx, kMmaLdm, kMmaOpt, kWmmaBf16 };

bool is_bf16(TVariant v) {
    return v == TVariant::kWmmaBf16;
}

const char* tvariant_name(TVariant v) {
    switch (v) {
        case TVariant::kWmmaFp16:
            return "wmma_fp16";
        case TVariant::kMmaPtx:
            return "mma_ptx";
        case TVariant::kMmaLdm:
            return "mma_ldm";
        case TVariant::kMmaOpt:
            return "mma_opt";
        case TVariant::kWmmaBf16:
            return "wmma_bf16";
    }
    return "unknown";
}

// Unit roundoff of the storage format. FP16 carries 11 significand bits, so
// u = 2^-11; BF16 carries 8, so u = 2^-8. Every vs_fp32 bound below is a
// multiple of this and of nothing else.
double storage_roundoff(TVariant v) {
    return is_bf16(v) ? 0x1p-8 : 0x1p-11;
}

const std::vector<TVariant>& all_tvariants() {
    static const std::vector<TVariant> v = {TVariant::kWmmaFp16, TVariant::kMmaPtx,
                                            TVariant::kMmaLdm, TVariant::kMmaOpt,
                                            TVariant::kWmmaBf16};
    return v;
}

void launch_kernel(TVariant v, const std::uint16_t* a, const std::uint16_t* b, float* c, int m,
                   int n, int k, float alpha, float beta, cudaStream_t stream) {
    const auto* ha = reinterpret_cast<const __half*>(a);
    const auto* hb = reinterpret_cast<const __half*>(b);
    const auto* ba = reinterpret_cast<const __nv_bfloat16*>(a);
    const auto* bb = reinterpret_cast<const __nv_bfloat16*>(b);
    switch (v) {
        case TVariant::kWmmaFp16:
            ckl::gemm_wmma_fp16(ha, hb, c, m, n, k, alpha, beta, stream);
            return;
        case TVariant::kMmaPtx:
            ckl::gemm_mma_ptx(ha, hb, c, m, n, k, alpha, beta, stream);
            return;
        case TVariant::kMmaLdm:
            ckl::gemm_mma_ldm(ha, hb, c, m, n, k, alpha, beta, stream);
            return;
        case TVariant::kMmaOpt:
            ckl::gemm_mma_opt(ha, hb, c, m, n, k, alpha, beta, stream);
            return;
        case TVariant::kWmmaBf16:
            ckl::gemm_wmma_bf16(ba, bb, c, m, n, k, alpha, beta, stream);
            return;
    }
}

void launch_oracle(TVariant v, const std::uint16_t* a, const std::uint16_t* b, float* c, int m,
                   int n, int k, float alpha, float beta, cudaStream_t stream) {
    if (is_bf16(v)) {
        ckl::gemm_cublas_bf16(reinterpret_cast<const __nv_bfloat16*>(a),
                              reinterpret_cast<const __nv_bfloat16*>(b), c, m, n, k, alpha, beta,
                              stream);
    } else {
        ckl::gemm_cublas_fp16(reinterpret_cast<const __half*>(a),
                              reinterpret_cast<const __half*>(b), c, m, n, k, alpha, beta, stream);
    }
}

// The raw structs are the supported way to move the bit pattern in and out of
// __half and __nv_bfloat16; memcpy over them trips -Wclass-memaccess because
// their storage member is protected.
Bits to_storage(TVariant v, const std::vector<float>& src) {
    Bits out(src.size());
    for (std::size_t i = 0; i < src.size(); ++i) {
        if (is_bf16(v)) {
            const auto raw = static_cast<__nv_bfloat16_raw>(__float2bfloat16(src[i]));
            out[i] = raw.x;
        } else {
            const auto raw = static_cast<__half_raw>(__float2half(src[i]));
            out[i] = raw.x;
        }
    }
    return out;
}

// The values the kernel actually contracts, back in float. The reference has to
// use these, or the accumulation error being measured would be swamped by the
// input rounding.
std::vector<float> from_storage(TVariant v, const Bits& src) {
    std::vector<float> out(src.size());
    for (std::size_t i = 0; i < src.size(); ++i) {
        if (is_bf16(v)) {
            __nv_bfloat16_raw raw{};
            raw.x = src[i];
            out[i] = __bfloat162float(__nv_bfloat16(raw));
        } else {
            __half_raw raw{};
            raw.x = src[i];
            out[i] = __half2float(__half(raw));
        }
    }
    return out;
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

// Everything one case needs: the two device inputs already in storage format,
// their rounded float images, and the C seed.
struct Case {
    Bits a_bits;
    Bits b_bits;
    std::vector<float> a_round;
    std::vector<float> b_round;
    std::vector<float> a_f32;
    std::vector<float> b_f32;
    std::vector<float> c0;
};

Case make_case(TVariant v, const Shape& s, std::uint64_t stream_id) {
    Case c;
    c.a_f32 = ckl::random_matrix(s.m, s.k, ckl::test::seed_stream(stream_id));
    c.b_f32 = ckl::random_matrix(s.k, s.n, ckl::test::seed_stream(stream_id + 1));
    c.c0 = ckl::random_matrix(s.m, s.n, ckl::test::seed_stream(stream_id + 2));
    c.a_bits = to_storage(v, c.a_f32);
    c.b_bits = to_storage(v, c.b_f32);
    c.a_round = from_storage(v, c.a_bits);
    c.b_round = from_storage(v, c.b_bits);
    return c;
}

// Runs kernel and oracle into separate buffers from the same inputs.
struct Outputs {
    std::vector<float> kernel;
    std::vector<float> oracle;
};

Outputs run_both(TVariant v, const Case& c, const Shape& s, float alpha, float beta) {
    ckl::DeviceBuffer<std::uint16_t> da(c.a_bits.size());
    ckl::DeviceBuffer<std::uint16_t> db(c.b_bits.size());
    ckl::DeviceBuffer<float> dk(c.c0.size());
    ckl::DeviceBuffer<float> doracle(c.c0.size());
    da.copy_from_host(c.a_bits);
    db.copy_from_host(c.b_bits);
    dk.copy_from_host(c.c0);
    doracle.copy_from_host(c.c0);

    launch_kernel(v, da.data(), db.data(), dk.data(), s.m, s.n, s.k, alpha, beta, nullptr);
    launch_oracle(v, da.data(), db.data(), doracle.data(), s.m, s.n, s.k, alpha, beta, nullptr);
    CKL_CUDA_CHECK(cudaDeviceSynchronize());

    Outputs out;
    out.kernel = dk.size() == 0 ? std::vector<float>{} : dk.to_host();
    out.oracle = doracle.size() == 0 ? std::vector<float>{} : doracle.to_host();
    return out;
}

// The gate Section 14 names for the tensor path. Both the kernel and the oracle
// consume identical rounded inputs and accumulate in FP32, so anything above
// this is a difference in what was computed, not in how it was rounded.
constexpr double kOracleGate = 1e-5;

void expect_tensor_matches(TVariant v, const Shape& s, float alpha, float beta,
                           std::uint64_t stream_id) {
    const Case c = make_case(v, s, stream_id);
    const Outputs out = run_both(v, c, s, alpha, beta);

    ASSERT_TRUE(ckl::all_finite(out.kernel)) << tvariant_name(v) << " produced a non finite result";

    const std::vector<double> oracle_ref(out.oracle.begin(), out.oracle.end());
    const double vs_oracle = ckl::relative_frobenius_error(out.kernel, oracle_ref);
    EXPECT_LT(vs_oracle, kOracleGate)
        << tvariant_name(v) << " " << s.label << " against the same precision cuBLAS oracle";

    const ckl::GemmRef rounded =
        ckl::gemm_reference_scaled(c.a_round, c.b_round, c.c0, s.m, s.n, s.k, alpha, beta);
    const double gate = ckl::tol(s.k);
    EXPECT_LT(ckl::relative_frobenius_error(out.kernel, rounded.c), gate)
        << tvariant_name(v) << " " << s.label << " accumulation error, gate " << gate;
    EXPECT_LT(ckl::max_scaled_residual(out.kernel, rounded), gate)
        << tvariant_name(v) << " " << s.label << " elementwise accumulation residual";

    // The honest accuracy of the precision. Input rounding dominates, so the
    // bound is a multiple of the storage roundoff and says nothing about the
    // kernel; it is here so a regression that silently changed the storage
    // format would be visible.
    const std::vector<double> exact =
        ckl::gemm_reference(c.a_f32, c.b_f32, c.c0, s.m, s.n, s.k, alpha, beta);
    const double vs_fp32 = ckl::relative_frobenius_error(out.kernel, exact);
    EXPECT_LT(vs_fp32, 8.0 * storage_roundoff(v))
        << tvariant_name(v) << " " << s.label << " against the FP32 data it was rounded from";
    ::testing::Test::RecordProperty("vs_oracle", ckl::test::sci(vs_oracle));
    ::testing::Test::RecordProperty("vs_fp32", ckl::test::sci(vs_fp32));
}

// ---------------------------------------------------------------------------
// Shape sweep
// ---------------------------------------------------------------------------

const std::vector<Shape>& sweep_shapes() {
    static const std::vector<Shape> shapes = {
        {128, 128, 128, "square_aligned"},   {256, 128, 64, "non_square_aligned"},
        {129, 257, 193, "non_tile_aligned"}, {7, 5, 11, "smaller_than_one_tile"},
        {256, 128, 320, "non_square_large"},
    };
    return shapes;
}

using ShapeParam = std::tuple<TVariant, Shape>;

class TensorShape : public GpuTestWithParam<ShapeParam> {};

TEST_P(TensorShape, MatchesOracleAndReference) {
    const auto [v, s] = GetParam();
    expect_tensor_matches(v, s, 1.0f, 0.0f, 1101);
}

std::string shape_param_name(const ::testing::TestParamInfo<ShapeParam>& info) {
    return std::string(tvariant_name(std::get<0>(info.param))) + "_" +
           std::get<1>(info.param).label;
}

INSTANTIATE_TEST_SUITE_P(Tensor, TensorShape,
                         ::testing::Combine(::testing::ValuesIn(all_tvariants()),
                                            ::testing::ValuesIn(sweep_shapes())),
                         shape_param_name);

// ---------------------------------------------------------------------------
// Alpha and beta
// ---------------------------------------------------------------------------

using AbParam = std::tuple<TVariant, AlphaBeta>;

class TensorAlphaBeta : public GpuTestWithParam<AbParam> {};

// v1 fixed beta at zero for every tensor variant, so a kernel that ignored beta
// passed the whole suite. Both an aligned shape and one that falls back to the
// scalar path are covered, because the two epilogues are separate code.
TEST_P(TensorAlphaBeta, AlignedAndUnaligned) {
    const auto [v, ab] = GetParam();
    {
        SCOPED_TRACE("aligned 128x128x128");
        expect_tensor_matches(v, Shape{128, 128, 128, "aligned"}, ab.alpha, ab.beta, 1201);
    }
    {
        SCOPED_TRACE("unaligned 65x33x17");
        expect_tensor_matches(v, Shape{65, 33, 17, "unaligned"}, ab.alpha, ab.beta, 1207);
    }
}

std::string ab_param_name(const ::testing::TestParamInfo<AbParam>& info) {
    return std::string(tvariant_name(std::get<0>(info.param))) + "_" +
           std::get<1>(info.param).label;
}

INSTANTIATE_TEST_SUITE_P(Tensor, TensorAlphaBeta,
                         ::testing::Combine(::testing::ValuesIn(all_tvariants()),
                                            ::testing::ValuesIn(alpha_beta_cases())),
                         ab_param_name);

// ---------------------------------------------------------------------------
// beta = 0 must not read C
// ---------------------------------------------------------------------------

struct FillKind {
    float value;
    const char* label;
};

using NonFiniteParam = std::tuple<TVariant, FillKind>;

class TensorBetaZeroNonFinite : public GpuTestWithParam<NonFiniteParam> {};

TEST_P(TensorBetaZeroNonFinite, OutputIsFiniteAndCorrect) {
    const auto [v, fill] = GetParam();
    const Shape s{128, 128, 128, "beta_zero"};
    Case c = make_case(v, s, 1301);
    c.c0.assign(elems(s.m, s.n), fill.value);

    const Outputs out = run_both(v, c, s, 1.0f, 0.0f);
    ASSERT_TRUE(ckl::all_finite(out.kernel))
        << tvariant_name(v) << " read C while beta was zero (C was " << fill.label << ")";

    const std::vector<float> zeros(elems(s.m, s.n), 0.0f);
    const ckl::GemmRef ref =
        ckl::gemm_reference_scaled(c.a_round, c.b_round, zeros, s.m, s.n, s.k, 1.0f, 0.0f);
    EXPECT_LT(ckl::relative_frobenius_error(out.kernel, ref.c), ckl::tol(s.k));
    EXPECT_LT(ckl::max_scaled_residual(out.kernel, ref), ckl::tol(s.k));
}

std::string non_finite_param_name(const ::testing::TestParamInfo<NonFiniteParam>& info) {
    return std::string(tvariant_name(std::get<0>(info.param))) + "_" +
           std::get<1>(info.param).label;
}

const std::vector<FillKind>& non_finite_fills() {
    static const std::vector<FillKind> fills = {
        {std::numeric_limits<float>::quiet_NaN(), "nan"},
        {std::numeric_limits<float>::infinity(), "inf"},
    };
    return fills;
}

INSTANTIATE_TEST_SUITE_P(Tensor, TensorBetaZeroNonFinite,
                         ::testing::Combine(::testing::ValuesIn(all_tvariants()),
                                            ::testing::ValuesIn(non_finite_fills())),
                         non_finite_param_name);

// ---------------------------------------------------------------------------
// Edge shapes, large K, canaries
// ---------------------------------------------------------------------------

class TensorVariant : public GpuTestWithParam<TVariant> {};

std::string tvariant_param_name(const ::testing::TestParamInfo<TVariant>& info) {
    return tvariant_name(info.param);
}

TEST_P(TensorVariant, DegenerateShapes) {
    const std::vector<Shape> shapes = {
        {1, 1, 1, "1x1x1"},   {1, 1, 128, "1x1x128"}, {1, 8, 16, "1x8x16"},
        {8, 1, 16, "8x1x16"}, {2, 3, 5, "2x3x5"},     {16, 16, 1, "16x16x1"},
    };
    for (const auto& s : shapes) {
        SCOPED_TRACE(s.label);
        expect_tensor_matches(GetParam(), s, 1.25f, 0.5f, 1401);
    }
}

// 64 is the WMMA tile and 128 is the mma_opt tile; a boundary predicate that is
// off by one fails on exactly one side of each.
TEST_P(TensorVariant, TileBoundaryCrossings) {
    const std::vector<Shape> shapes = {
        {127, 127, 127, "127"}, {128, 128, 128, "128"}, {129, 129, 129, "129"},
        {63, 63, 63, "63"},     {64, 64, 64, "64"},     {65, 65, 65, "65"},
    };
    for (const auto& s : shapes) {
        SCOPED_TRACE(s.label);
        expect_tensor_matches(GetParam(), s, 1.25f, 0.5f, 1501);
    }
}

// The sweep runs k to 8192; the largest k v1 ever tested was 640.
TEST_P(TensorVariant, LargeK) {
    {
        SCOPED_TRACE("small m and n");
        expect_tensor_matches(GetParam(), Shape{64, 64, 8192, "64x64x8192"}, 1.25f, 0.5f, 1601);
    }
    {
        SCOPED_TRACE("block aligned");
        expect_tensor_matches(GetParam(), Shape{128, 128, 8192, "128x128x8192"}, 1.0f, 0.0f, 1611);
    }
}

TEST_P(TensorVariant, ZeroDimensionsLeaveTheOutputAlone) {
    const TVariant v = GetParam();
    const float canary = -12345.75f;
    const std::vector<Shape> empty_shapes = {{0, 64, 32, "m0"}, {64, 0, 32, "n0"}};
    for (const auto& s : empty_shapes) {
        SCOPED_TRACE(s.label);
        const std::size_t guard = elems(64, 64);
        const auto fa = ckl::random_matrix(64, s.k, ckl::test::seed_stream(1701));
        const auto fb = ckl::random_matrix(s.k, 64, ckl::test::seed_stream(1702));
        const Bits a_bits = to_storage(v, fa);
        const Bits b_bits = to_storage(v, fb);

        ckl::DeviceBuffer<std::uint16_t> da(a_bits.size());
        ckl::DeviceBuffer<std::uint16_t> db(b_bits.size());
        ckl::DeviceBuffer<float> dc(guard);
        da.copy_from_host(a_bits);
        db.copy_from_host(b_bits);
        const std::vector<float> seed(guard, canary);
        dc.copy_from_host(seed);

        launch_kernel(v, da.data(), db.data(), dc.data(), s.m, s.n, s.k, 1.25f, 0.5f, nullptr);
        CKL_CUDA_CHECK(cudaDeviceSynchronize());
        const auto out = dc.to_host();
        for (std::size_t i = 0; i < guard; ++i) {
            ASSERT_EQ(out[i], canary) << "element " << i << " was written for an empty output";
        }
    }
}

TEST_P(TensorVariant, ZeroKAppliesBetaOnly) {
    const TVariant v = GetParam();
    const int m = 64;
    const int n = 64;
    const auto fa = ckl::random_matrix(m, 1, ckl::test::seed_stream(1711));
    const auto fb = ckl::random_matrix(1, n, ckl::test::seed_stream(1712));
    const auto c0 = ckl::random_matrix(m, n, ckl::test::seed_stream(1713));
    const Bits a_bits = to_storage(v, fa);
    const Bits b_bits = to_storage(v, fb);

    ckl::DeviceBuffer<std::uint16_t> da(a_bits.size());
    ckl::DeviceBuffer<std::uint16_t> db(b_bits.size());
    da.copy_from_host(a_bits);
    db.copy_from_host(b_bits);

    {
        SCOPED_TRACE("beta 0.5 scales C");
        ckl::DeviceBuffer<float> dc(c0.size());
        dc.copy_from_host(c0);
        launch_kernel(v, da.data(), db.data(), dc.data(), m, n, 0, 1.25f, 0.5f, nullptr);
        CKL_CUDA_CHECK(cudaDeviceSynchronize());
        const auto out = dc.to_host();
        for (std::size_t i = 0; i < out.size(); ++i) {
            ASSERT_FLOAT_EQ(out[i], 0.5f * c0[i]) << "element " << i;
        }
    }
    {
        SCOPED_TRACE("beta 0 clears C without reading it");
        const std::vector<float> nan_c(elems(m, n), std::numeric_limits<float>::quiet_NaN());
        ckl::DeviceBuffer<float> dc(nan_c.size());
        dc.copy_from_host(nan_c);
        launch_kernel(v, da.data(), db.data(), dc.data(), m, n, 0, 1.25f, 0.0f, nullptr);
        CKL_CUDA_CHECK(cudaDeviceSynchronize());
        const auto out = dc.to_host();
        for (std::size_t i = 0; i < out.size(); ++i) {
            ASSERT_EQ(out[i], 0.0f) << "element " << i << " came back " << out[i];
        }
    }
}

INSTANTIATE_TEST_SUITE_P(Tensor, TensorVariant, ::testing::ValuesIn(all_tvariants()),
                         tvariant_param_name);

// ---------------------------------------------------------------------------
// Fusion
// ---------------------------------------------------------------------------

// The 8.4 percent saving from folding the bias and ReLU into the mma_opt
// epilogue is a headline number and had no test at all. The fused kernel
// computes relu(alpha * A*B + bias) in registers; the unfused path stores
// alpha * A*B, then reads it back and applies the same two operations.
//
// The two are not bit identical, and that is not a bug: the fused expression is
// a single multiply add, which nvcc contracts into an FMA that keeps the full
// product before adding the bias, while the unfused path rounds alpha * A*B on
// its way to memory. The measured gap is under one ULP, so the gate is a few
// units of FLT_EPSILON relative to the magnitude, which is still orders of
// magnitude tighter than any real epilogue bug would produce.
class Fusion : public GpuTestWithParam<Shape> {};

std::string fusion_param_name(const ::testing::TestParamInfo<Shape>& info) {
    return info.param.label;
}

TEST_P(Fusion, FusedEqualsUnfused) {
    const Shape s = GetParam();
    const Case c = make_case(TVariant::kMmaOpt, s, 1801);
    const auto bias = ckl::random_matrix(1, s.n, ckl::test::seed_stream(1804));
    const float alpha = 1.25f;

    ckl::DeviceBuffer<std::uint16_t> da(c.a_bits.size());
    ckl::DeviceBuffer<std::uint16_t> db(c.b_bits.size());
    ckl::DeviceBuffer<float> dbias(bias.size());
    ckl::DeviceBuffer<float> dfused(elems(s.m, s.n));
    ckl::DeviceBuffer<float> dplain(elems(s.m, s.n));
    da.copy_from_host(c.a_bits);
    db.copy_from_host(c.b_bits);
    dbias.copy_from_host(bias);
    dfused.zero();
    dplain.zero();

    const auto* ha = reinterpret_cast<const __half*>(da.data());
    const auto* hb = reinterpret_cast<const __half*>(db.data());

    ckl::gemm_mma_opt_bias(ha, hb, dfused.data(), dbias.data(), s.m, s.n, s.k, alpha, nullptr);
    ckl::gemm_mma_opt(ha, hb, dplain.data(), s.m, s.n, s.k, alpha, 0.0f, nullptr);
    ckl::gemm_bias_relu(dplain.data(), dbias.data(), s.m, s.n, nullptr);
    CKL_CUDA_CHECK(cudaDeviceSynchronize());

    const auto fused = dfused.to_host();
    const auto unfused = dplain.to_host();
    ASSERT_EQ(fused.size(), unfused.size());
    ASSERT_TRUE(ckl::all_finite(fused));
    ASSERT_TRUE(ckl::all_finite(unfused));
    double worst = 0.0;
    std::size_t worst_at = 0;
    for (int i = 0; i < s.m; ++i) {
        for (int j = 0; j < s.n; ++j) {
            const std::size_t idx = static_cast<std::size_t>(i) * static_cast<std::size_t>(s.n) +
                                    static_cast<std::size_t>(j);
            const double denom = std::fabs(static_cast<double>(unfused[idx])) +
                                 std::fabs(static_cast<double>(bias[static_cast<std::size_t>(j)]));
            const double diff =
                std::fabs(static_cast<double>(fused[idx]) - static_cast<double>(unfused[idx]));
            const double r = diff / (denom > 1e-30 ? denom : 1.0);
            if (r > worst) {
                worst = r;
                worst_at = idx;
            }
        }
    }
    EXPECT_LT(worst, 4.0 * static_cast<double>(FLT_EPSILON))
        << "worst at element " << worst_at << ": fused " << fused[worst_at] << " unfused "
        << unfused[worst_at];

    // Both are also checked against the CPU, so a shared bug in the epilogue
    // cannot make them agree on a wrong answer.
    const std::vector<float> zeros(elems(s.m, s.n), 0.0f);
    const ckl::GemmRef ref =
        ckl::gemm_reference_scaled(c.a_round, c.b_round, zeros, s.m, s.n, s.k, alpha, 0.0f);
    ckl::GemmRef biased = ref;
    for (int i = 0; i < s.m; ++i) {
        for (int j = 0; j < s.n; ++j) {
            const std::size_t idx = static_cast<std::size_t>(i) * static_cast<std::size_t>(s.n) +
                                    static_cast<std::size_t>(j);
            const double v = biased.c[idx] + static_cast<double>(bias[static_cast<std::size_t>(j)]);
            biased.c[idx] = v > 0.0 ? v : 0.0;
            biased.scale[idx] += std::fabs(static_cast<double>(bias[static_cast<std::size_t>(j)]));
        }
    }
    EXPECT_LT(ckl::max_scaled_residual(fused, biased), ckl::tol(s.k));
}

INSTANTIATE_TEST_SUITE_P(Tensor, Fusion,
                         ::testing::Values(Shape{128, 128, 128, "128x128x128"},
                                           Shape{256, 256, 256, "256x256x256"},
                                           Shape{128, 256, 512, "128x256x512"}),
                         fusion_param_name);

class FusionContract : public GpuTest {};

// The fused kernel has no unaligned form. Returning without writing C would read
// to the caller as a successful GEMM that produced garbage, so it throws, and
// this is the test that says so.
TEST_F(FusionContract, UnalignedShapeIsRefusedNotSilentlySkipped) {
    const Shape s{100, 100, 100, "unaligned"};
    const Case c = make_case(TVariant::kMmaOpt, s, 1901);
    const auto bias = ckl::random_matrix(1, s.n, ckl::test::seed_stream(1904));

    ckl::DeviceBuffer<std::uint16_t> da(c.a_bits.size());
    ckl::DeviceBuffer<std::uint16_t> db(c.b_bits.size());
    ckl::DeviceBuffer<float> dbias(bias.size());
    ckl::DeviceBuffer<float> dc(elems(s.m, s.n));
    da.copy_from_host(c.a_bits);
    db.copy_from_host(c.b_bits);
    dbias.copy_from_host(bias);
    const float canary = -4242.5f;
    const std::vector<float> seed(elems(s.m, s.n), canary);
    dc.copy_from_host(seed);

    try {
        ckl::gemm_mma_opt_bias(reinterpret_cast<const __half*>(da.data()),
                               reinterpret_cast<const __half*>(db.data()), dc.data(), dbias.data(),
                               s.m, s.n, s.k, 1.0f, nullptr);
        FAIL() << "an unaligned fused GEMM has to be refused, not silently skipped";
    } catch (const ckl::Error& e) {
        EXPECT_EQ(e.status(), ckl::Status::kNotSupported) << e.what();
    }

    CKL_CUDA_CHECK(cudaDeviceSynchronize());
    const auto out = dc.to_host();
    for (std::size_t i = 0; i < out.size(); ++i) {
        ASSERT_EQ(out[i], canary) << "a refused call still wrote element " << i;
    }
}

// The standalone epilogue, on its own, against the CPU.
TEST_F(FusionContract, BiasReluMatchesHost) {
    const int m = 96;
    const int n = 80;
    const auto c0 = ckl::random_matrix(m, n, ckl::test::seed_stream(1911));
    const auto bias = ckl::random_matrix(1, n, ckl::test::seed_stream(1912));

    ckl::DeviceBuffer<float> dc(c0.size());
    ckl::DeviceBuffer<float> dbias(bias.size());
    dc.copy_from_host(c0);
    dbias.copy_from_host(bias);
    ckl::gemm_bias_relu(dc.data(), dbias.data(), m, n, nullptr);
    CKL_CUDA_CHECK(cudaDeviceSynchronize());
    const auto out = dc.to_host();

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            const std::size_t idx = static_cast<std::size_t>(i) * static_cast<std::size_t>(n) +
                                    static_cast<std::size_t>(j);
            const float v = c0[idx] + bias[static_cast<std::size_t>(j)];
            ASSERT_EQ(out[idx], v > 0.0f ? v : 0.0f) << "element " << idx;
        }
    }
}

TEST_F(FusionContract, ZeroDimensionBiasReluWritesNothing) {
    const float canary = 7.5f;
    ckl::DeviceBuffer<float> dc(64);
    ckl::DeviceBuffer<float> dbias(64);
    const std::vector<float> seed(64, canary);
    dc.copy_from_host(seed);
    dbias.zero();
    ckl::gemm_bias_relu(dc.data(), dbias.data(), 0, 64, nullptr);
    ckl::gemm_bias_relu(dc.data(), dbias.data(), 64, 0, nullptr);
    CKL_CUDA_CHECK(cudaDeviceSynchronize());
    for (float x : dc.to_host()) {
        ASSERT_EQ(x, canary);
    }
}

// ---------------------------------------------------------------------------
// Streams
// ---------------------------------------------------------------------------

class TensorStreams : public GpuTest {};

TEST_F(TensorStreams, TwoConcurrentStreams) {
    const Shape s{128, 128, 256, "streams"};
    const Case c = make_case(TVariant::kMmaOpt, s, 2001);

    cudaStream_t s1 = nullptr;
    cudaStream_t s2 = nullptr;
    CKL_CUDA_CHECK(cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking));
    CKL_CUDA_CHECK(cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking));

    ckl::DeviceBuffer<std::uint16_t> da(c.a_bits.size());
    ckl::DeviceBuffer<std::uint16_t> db(c.b_bits.size());
    ckl::DeviceBuffer<float> d1(c.c0.size());
    ckl::DeviceBuffer<float> d2(c.c0.size());
    da.copy_from_host(c.a_bits);
    db.copy_from_host(c.b_bits);
    d1.copy_from_host(c.c0);
    d2.copy_from_host(c.c0);

    launch_kernel(TVariant::kMmaOpt, da.data(), db.data(), d1.data(), s.m, s.n, s.k, 1.25f, 0.5f,
                  s1);
    launch_kernel(TVariant::kMmaLdm, da.data(), db.data(), d2.data(), s.m, s.n, s.k, 1.25f, 0.5f,
                  s2);
    CKL_CUDA_CHECK(cudaStreamSynchronize(s1));
    CKL_CUDA_CHECK(cudaStreamSynchronize(s2));
    const auto out1 = d1.to_host();
    const auto out2 = d2.to_host();
    CKL_CUDA_CHECK(cudaStreamDestroy(s1));
    CKL_CUDA_CHECK(cudaStreamDestroy(s2));

    const ckl::GemmRef ref =
        ckl::gemm_reference_scaled(c.a_round, c.b_round, c.c0, s.m, s.n, s.k, 1.25f, 0.5f);
    EXPECT_LT(ckl::max_scaled_residual(out1, ref), ckl::tol(s.k)) << "mma_opt on stream 1";
    EXPECT_LT(ckl::max_scaled_residual(out2, ref), ckl::tol(s.k)) << "mma_ldm on stream 2";
}

}  // namespace
