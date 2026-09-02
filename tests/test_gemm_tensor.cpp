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
#include "gemm/detail/gemm_swizzle.hpp"
#include "gemm/detail/gemm_tile_registry.hpp"
#include "gemm/detail/gemm_tile_swizzle.hpp"
#include "gpu_environment.hpp"
#include "reference.hpp"

namespace {

using ckl::test::GpuTest;
using ckl::test::GpuTestWithParam;

// Storage is carried as raw 16 bit words so one code path serves both formats;
// the launchers get the pointer reinterpreted at the call.
using Bits = std::vector<std::uint16_t>;

// The tile family and its two escalations join the ladder rungs here rather
// than in a suite of their own, so every property the older kernels are held to
// (shape sweep, alpha and beta, a non finite C under beta zero, degenerate
// shapes, tile boundary crossings, large K, zero dimensions) applies to all six
// instantiations without a second copy of the harness.
enum class TVariant {
    kWmmaFp16,
    kMmaPtx,
    kMmaLdm,
    kMmaOpt,
    kWmmaBf16,
    kTile0,
    kTile1,
    kTile2,
    kTile3,
    kTile4,
    kTile5,
    kSplitK,
    kStreamK,
    // The reference line. It is not in all_tvariants because it refuses shapes
    // whose n or k is not a multiple of 8, and the suites below run 1x1x1 and
    // 65x33x17 at every variant they are given. It gets the same three error
    // measurements over the shapes it does take, in a suite of its own.
    kCutlass,
};

// Family index of a tile variant, or -1 for the rungs that are not one.
int tile_index_of(TVariant v) {
    switch (v) {
        case TVariant::kTile0:
            return 0;
        case TVariant::kTile1:
            return 1;
        case TVariant::kTile2:
            return 2;
        case TVariant::kTile3:
            return 3;
        case TVariant::kTile4:
            return 4;
        case TVariant::kTile5:
            return 5;
        default:
            return -1;
    }
}

// Split count the split-K variant asks for. Four is enough to exercise the
// second pass and the alpha and beta placement without making the fixup the
// whole cost of the test.
constexpr int kTestSplits = 4;

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
        case TVariant::kTile0:
            return "tile0";
        case TVariant::kTile1:
            return "tile1";
        case TVariant::kTile2:
            return "tile2";
        case TVariant::kTile3:
            return "tile3";
        case TVariant::kTile4:
            return "tile4";
        case TVariant::kTile5:
            return "tile5";
        case TVariant::kSplitK:
            return "split_k";
        case TVariant::kStreamK:
            return "stream_k";
        case TVariant::kCutlass:
            return "cutlass";
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
    static const std::vector<TVariant> v = {
        TVariant::kWmmaFp16, TVariant::kMmaPtx, TVariant::kMmaLdm, TVariant::kMmaOpt,
        TVariant::kWmmaBf16, TVariant::kTile0,  TVariant::kTile1,  TVariant::kTile2,
        TVariant::kTile3,    TVariant::kTile4,  TVariant::kTile5,  TVariant::kSplitK,
        TVariant::kStreamK};
    return v;
}

// The family, split-K and stream-K only: the suites that are about the tile
// family rather than about the ladder.
const std::vector<TVariant>& family_tvariants() {
    static const std::vector<TVariant> v = {TVariant::kTile0,  TVariant::kTile1,  TVariant::kTile2,
                                            TVariant::kTile3,  TVariant::kTile4,  TVariant::kTile5,
                                            TVariant::kSplitK, TVariant::kStreamK};
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
        case TVariant::kSplitK:
            // No caller workspace: the driver takes one for the duration, which
            // is the path a caller that never asked gemm_workspace_size gets.
            ckl::gemm_split_k(ha, hb, c, m, n, k, alpha, beta, kTestSplits, 0, nullptr, 0, stream);
            return;
        case TVariant::kStreamK:
            ckl::gemm_stream_k(ha, hb, c, m, n, k, alpha, beta, 0, nullptr, 0, stream);
            return;
        case TVariant::kCutlass:
            ckl::gemm_cutlass(ha, hb, c, m, n, k, alpha, beta, stream);
            return;
        default:
            ckl::gemm_tile_family(ha, hb, c, m, n, k, alpha, beta, tile_index_of(v), stream);
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
// The CUTLASS reference line
// ---------------------------------------------------------------------------
//
// Same gates as the hand written rungs (1e-5 against the same precision cuBLAS
// oracle, tol(k) against a double precision reference over the rounded inputs,
// a storage roundoff bound against the original FP32 data), over the shapes the
// rung accepts. It refuses everything else instead of rerouting, and the last
// test here is what holds it to that.

const std::vector<Shape>& cutlass_shapes() {
    static const std::vector<Shape> shapes = {
        {128, 128, 128, "square_aligned"},   {256, 128, 64, "non_square_aligned"},
        {129, 256, 192, "m_predicated"},     {127, 128, 128, "m_below_one_tile"},
        {7, 8, 8, "smaller_than_one_tile"},  {1, 8, 16, "single_row"},
        {256, 128, 320, "non_square_large"},
    };
    return shapes;
}

class CutlassShape : public GpuTestWithParam<Shape> {};

TEST_P(CutlassShape, MatchesOracleAndReference) {
    expect_tensor_matches(TVariant::kCutlass, GetParam(), 1.0f, 0.0f, 2101);
}

std::string cutlass_shape_name(const ::testing::TestParamInfo<Shape>& info) {
    return info.param.label;
}

INSTANTIATE_TEST_SUITE_P(Tensor, CutlassShape, ::testing::ValuesIn(cutlass_shapes()),
                         cutlass_shape_name);

class CutlassAlphaBeta : public GpuTestWithParam<AlphaBeta> {};

TEST_P(CutlassAlphaBeta, TileAlignedAndPredicated) {
    const AlphaBeta ab = GetParam();
    {
        SCOPED_TRACE("tile aligned 128x128x128");
        expect_tensor_matches(TVariant::kCutlass, Shape{128, 128, 128, "aligned"}, ab.alpha,
                              ab.beta, 2201);
    }
    {
        SCOPED_TRACE("predicated 65x32x16");
        expect_tensor_matches(TVariant::kCutlass, Shape{65, 32, 16, "predicated"}, ab.alpha,
                              ab.beta, 2207);
    }
}

std::string cutlass_ab_name(const ::testing::TestParamInfo<AlphaBeta>& info) {
    return info.param.label;
}

INSTANTIATE_TEST_SUITE_P(Tensor, CutlassAlphaBeta, ::testing::ValuesIn(alpha_beta_cases()),
                         cutlass_ab_name);

class Cutlass : public GpuTest {};

TEST_F(Cutlass, BetaZeroDoesNotReadC) {
    const Shape s{128, 128, 128, "beta_zero"};
    for (const FillKind& fill : non_finite_fills()) {
        SCOPED_TRACE(fill.label);
        Case c = make_case(TVariant::kCutlass, s, 2301);
        c.c0.assign(elems(s.m, s.n), fill.value);
        const Outputs out = run_both(TVariant::kCutlass, c, s, 1.0f, 0.0f);
        ASSERT_TRUE(ckl::all_finite(out.kernel))
            << "cutlass read C while beta was zero (C was " << fill.label << ")";
        const std::vector<float> zeros(elems(s.m, s.n), 0.0f);
        const ckl::GemmRef ref =
            ckl::gemm_reference_scaled(c.a_round, c.b_round, zeros, s.m, s.n, s.k, 1.0f, 0.0f);
        EXPECT_LT(ckl::max_scaled_residual(out.kernel, ref), ckl::tol(s.k));
    }
}

TEST_F(Cutlass, LargeK) {
    {
        SCOPED_TRACE("small m and n");
        expect_tensor_matches(TVariant::kCutlass, Shape{64, 64, 8192, "64x64x8192"}, 1.25f, 0.5f,
                              2401);
    }
    {
        SCOPED_TRACE("tile aligned");
        expect_tensor_matches(TVariant::kCutlass, Shape{128, 128, 8192, "128x128x8192"}, 1.0f, 0.0f,
                              2411);
    }
}

TEST_F(Cutlass, ZeroDimensionsLeaveTheOutputAlone) {
    const float canary = -12345.75f;
    const std::vector<Shape> empty_shapes = {{0, 64, 32, "m0"}, {64, 0, 32, "n0"}};
    for (const auto& s : empty_shapes) {
        SCOPED_TRACE(s.label);
        const std::size_t guard = elems(64, 64);
        const auto fa = ckl::random_matrix(64, s.k, ckl::test::seed_stream(2501));
        const auto fb = ckl::random_matrix(s.k, 64, ckl::test::seed_stream(2502));
        const Bits a_bits = to_storage(TVariant::kCutlass, fa);
        const Bits b_bits = to_storage(TVariant::kCutlass, fb);

        ckl::DeviceBuffer<std::uint16_t> da(a_bits.size());
        ckl::DeviceBuffer<std::uint16_t> db(b_bits.size());
        ckl::DeviceBuffer<float> dc(guard);
        da.copy_from_host(a_bits);
        db.copy_from_host(b_bits);
        dc.copy_from_host(std::vector<float>(guard, canary));

        launch_kernel(TVariant::kCutlass, da.data(), db.data(), dc.data(), s.m, s.n, s.k, 1.25f,
                      0.5f, nullptr);
        CKL_CUDA_CHECK(cudaDeviceSynchronize());
        const auto out = dc.to_host();
        for (std::size_t i = 0; i < guard; ++i) {
            ASSERT_EQ(out[i], canary) << "element " << i << " was written for an empty output";
        }
    }
}

// An empty contraction is C = beta * C, and beta zero clears rather than scales,
// because BLAS says C is not read when beta is zero and 0 * NaN is still NaN.
TEST_F(Cutlass, ZeroKAppliesBetaOnly) {
    const int m = 64;
    const int n = 64;
    const auto c0 = ckl::random_matrix(m, n, ckl::test::seed_stream(2601));
    ckl::DeviceBuffer<std::uint16_t> da(1);
    ckl::DeviceBuffer<std::uint16_t> db(1);
    da.zero();
    db.zero();
    {
        SCOPED_TRACE("beta 0.5 scales C");
        ckl::DeviceBuffer<float> dc(c0.size());
        dc.copy_from_host(c0);
        launch_kernel(TVariant::kCutlass, da.data(), db.data(), dc.data(), m, n, 0, 1.25f, 0.5f,
                      nullptr);
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
        launch_kernel(TVariant::kCutlass, da.data(), db.data(), dc.data(), m, n, 0, 1.25f, 0.0f,
                      nullptr);
        CKL_CUDA_CHECK(cudaDeviceSynchronize());
        const auto out = dc.to_host();
        for (std::size_t i = 0; i < out.size(); ++i) {
            ASSERT_EQ(out[i], 0.0f) << "element " << i << " came back " << out[i];
        }
    }
}

// The shape rule, and the promise that a refusal is a throw rather than a
// quietly different kernel. A CUTLASS row that measured a hand kernel would be
// worse than no row at all.
TEST_F(Cutlass, RefusesTheShapesItCannotRunRatherThanRerouting) {
    EXPECT_TRUE(ckl::gemm_cutlass_supports(129, 256, 192));
    EXPECT_TRUE(ckl::gemm_cutlass_supports(1, 8, 8));
    EXPECT_FALSE(ckl::gemm_cutlass_supports(128, 129, 128)) << "n is not a multiple of 8";
    EXPECT_FALSE(ckl::gemm_cutlass_supports(128, 128, 129)) << "k is not a multiple of 8";

    ckl::DeviceBuffer<std::uint16_t> da(elems(128, 132));
    ckl::DeviceBuffer<std::uint16_t> db(elems(132, 128));
    ckl::DeviceBuffer<float> dc(elems(128, 128));
    da.zero();
    db.zero();
    dc.zero();
    EXPECT_THROW(ckl::gemm_cutlass(reinterpret_cast<const __half*>(da.data()),
                                   reinterpret_cast<const __half*>(db.data()), dc.data(), 128, 128,
                                   132, 1.0f, 0.0f, nullptr),
                 ckl::Error);
}

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

// ---------------------------------------------------------------------------
// Shared memory swizzle, checked on the host
// ---------------------------------------------------------------------------
//
// The mma_opt mainloop is correct for any map applied identically at the
// cp.async store and the ldmatrix load, so the GEMM results above say nothing
// about whether the map does its job. What it has to be is a bijection (nothing
// overwritten, nothing unread), 16 byte chunk preserving (cp.async and ldmatrix
// both move eight halves at a time), and conflict free across the eight lanes of
// an ldmatrix wavefront. All three are integer arithmetic, so they are checked
// here rather than inferred from a profiler counter.
//
// The old A map XOR-ed the four chunk index of a 64 byte row with the row index,
// which cannot separate eight lanes; its lane bases came out 0, 20, 8, 28, 0,
// 20, 8, 28, a two way conflict on every A wavefront. The current map packs two
// rows into one 128 byte line first.

constexpr int kSwzM = ckl::detail::kSwzTileM;
constexpr int kSwzN = ckl::detail::kSwzTileN;
constexpr int kSwzK = ckl::detail::kSwzTileK;

// 32 banks of 4 bytes; the offsets the maps return count halves.
int bank_of(int half_offset) {
    return (half_offset / 2) % 32;
}

TEST(GemmOptSwizzle, ATileMapIsABijection) {
    std::vector<int> seen(static_cast<std::size_t>(kSwzM) * kSwzK, -1);
    for (int r = 0; r < kSwzM; ++r) {
        for (int c = 0; c < kSwzK; ++c) {
            const int o = ckl::detail::swizzle_a(r, c);
            ASSERT_GE(o, 0) << "row " << r << " col " << c;
            ASSERT_LT(o, kSwzM * kSwzK) << "row " << r << " col " << c;
            ASSERT_EQ(seen[static_cast<std::size_t>(o)], -1)
                << "offset " << o << " is written by two different (row, col) pairs";
            seen[static_cast<std::size_t>(o)] = r * kSwzK + c;
        }
    }
}

TEST(GemmOptSwizzle, BTileMapIsABijection) {
    std::vector<int> seen(static_cast<std::size_t>(kSwzK) * kSwzN, -1);
    for (int r = 0; r < kSwzK; ++r) {
        for (int c = 0; c < kSwzN; ++c) {
            const int o = ckl::detail::swizzle_b(r, c);
            ASSERT_GE(o, 0) << "row " << r << " col " << c;
            ASSERT_LT(o, kSwzK * kSwzN) << "row " << r << " col " << c;
            ASSERT_EQ(seen[static_cast<std::size_t>(o)], -1)
                << "offset " << o << " is written by two different (row, col) pairs";
            seen[static_cast<std::size_t>(o)] = r * kSwzN + c;
        }
    }
}

// A cp.async of a float4 and an ldmatrix row both move eight contiguous halves
// from one 16 byte aligned address, so the map has to keep those eight together.
TEST(GemmOptSwizzle, MapsKeepSixteenByteChunksTogether) {
    for (int r = 0; r < kSwzM; ++r) {
        for (int c = 0; c < kSwzK; c += 8) {
            const int base = ckl::detail::swizzle_a(r, c);
            ASSERT_EQ(base % 8, 0) << "A chunk at row " << r << " col " << c << " is not aligned";
            for (int j = 0; j < 8; ++j) {
                ASSERT_EQ(ckl::detail::swizzle_a(r, c + j), base + j);
            }
        }
    }
    for (int r = 0; r < kSwzK; ++r) {
        for (int c = 0; c < kSwzN; c += 8) {
            const int base = ckl::detail::swizzle_b(r, c);
            ASSERT_EQ(base % 8, 0) << "B chunk at row " << r << " col " << c << " is not aligned";
            for (int j = 0; j < 8; ++j) {
                ASSERT_EQ(ckl::detail::swizzle_b(r, c + j), base + j);
            }
        }
    }
}

// The eight lane bases of the first wavefront of an A ldmatrix.x4 at the tile
// origin. The spec fixes this sequence; anything else means the map changed.
TEST(GemmOptSwizzle, ALdmatrixLaneBasesMatchTheDesignedBanks) {
    const std::vector<int> expected = {0, 16, 4, 20, 8, 24, 12, 28};
    for (int lane = 0; lane < 8; ++lane) {
        EXPECT_EQ(bank_of(ckl::detail::swizzle_a(lane, 0)),
                  expected[static_cast<std::size_t>(lane)])
            << "lane " << lane;
    }
}

// Every A wavefront the mainloop actually issues: row_base runs over the four
// 16 row mma tiles of both warp rows, k_off over the two K substeps, and the x4
// form splits its 32 lanes into four groups of eight, each group one wavefront.
TEST(GemmOptSwizzle, EveryALdmatrixWavefrontIsConflictFree) {
    for (int row_base = 0; row_base < kSwzM; row_base += 16) {
        for (int k_off = 0; k_off < kSwzK; k_off += 16) {
            for (int group = 0; group < 4; ++group) {
                std::vector<int> banks;
                for (int i = 0; i < 8; ++i) {
                    const int lane = group * 8 + i;
                    const int r = row_base + (lane % 16);
                    const int c = k_off + (lane / 16) * 8;
                    banks.push_back(bank_of(ckl::detail::swizzle_a(r, c)));
                }
                for (std::size_t x = 0; x < banks.size(); ++x) {
                    for (std::size_t y = x + 1; y < banks.size(); ++y) {
                        ASSERT_NE(banks[x], banks[y])
                            << "A row_base " << row_base << " k_off " << k_off << " group " << group
                            << ": lanes " << x << " and " << y << " share bank " << banks[x];
                    }
                }
            }
        }
    }
}

// The same for B, which is now read with ldmatrix.x4.trans: lanes 16 to 31
// address the next eight columns, so the wavefronts changed and the swizzle has
// to still separate them.
TEST(GemmOptSwizzle, EveryBLdmatrixWavefrontIsConflictFree) {
    for (int col_base = 0; col_base < kSwzN; col_base += 16) {
        for (int k_off = 0; k_off < kSwzK; k_off += 16) {
            for (int group = 0; group < 4; ++group) {
                std::vector<int> banks;
                for (int i = 0; i < 8; ++i) {
                    const int lane = group * 8 + i;
                    const int r = k_off + (lane % 16);
                    const int c = col_base + (lane / 16) * 8;
                    banks.push_back(bank_of(ckl::detail::swizzle_b(r, c)));
                }
                for (std::size_t x = 0; x < banks.size(); ++x) {
                    for (std::size_t y = x + 1; y < banks.size(); ++y) {
                        ASSERT_NE(banks[x], banks[y])
                            << "B col_base " << col_base << " k_off " << k_off << " group " << group
                            << ": lanes " << x << " and " << y << " share bank " << banks[x];
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// The tile family: the shapes that used to fall off the ladder
// ---------------------------------------------------------------------------
//
// Section 9.5 of the build spec: any shape not divisible by 128, 128, 32 used to
// route to gemm_wmma_fp16, and anything not divisible by 64, 64, 16 routed on to
// a scalar kernel one thread per output element. Four of the seven tensor test
// shapes missed the fast path, so those rows were re-testing the fallback under
// different names. The family predicates its edges instead, and this is the
// matrix the spec names for it.
//
// These shapes are too large for a double precision host reference (8192 by 64
// by 4096 alone is 4.3 billion multiply adds, and 1000 cubed is a billion), so
// the check here is against the cuBLAS tensor oracle only, at the same 1e-5 gate
// the rest of the suite uses. The accumulation error against a host reference is
// measured at the smaller shapes above, where it is affordable.

struct BigShape {
    int m;
    int n;
    int k;
    const char* label;
};

// Half precision inputs without the float mirrors make_case keeps. At 8192 by
// 4096 those mirrors would be half a gigabyte of host memory for a reference
// nothing computes.
Bits random_half_bits(int rows, int cols, std::uint64_t stream_id) {
    return to_storage(TVariant::kMmaOpt,
                      ckl::random_matrix(rows, cols, ckl::test::seed_stream(stream_id)));
}

class TileFamilyUnaligned : public GpuTestWithParam<BigShape> {};

std::string big_shape_name(const ::testing::TestParamInfo<BigShape>& info) {
    return info.param.label;
}

TEST_P(TileFamilyUnaligned, EveryFamilyMemberMatchesTheOracle) {
    const BigShape s = GetParam();
    const float alpha = 1.25f;
    const float beta = 0.5f;

    const Bits a_bits = random_half_bits(s.m, s.k, 2801);
    const Bits b_bits = random_half_bits(s.k, s.n, 2802);
    const std::vector<float> c0 = ckl::random_matrix(s.m, s.n, ckl::test::seed_stream(2803));

    ckl::DeviceBuffer<std::uint16_t> da(a_bits.size());
    ckl::DeviceBuffer<std::uint16_t> db(b_bits.size());
    ckl::DeviceBuffer<float> dk(c0.size());
    ckl::DeviceBuffer<float> doracle(c0.size());
    da.copy_from_host(a_bits);
    db.copy_from_host(b_bits);
    doracle.copy_from_host(c0);
    launch_oracle(TVariant::kMmaOpt, da.data(), db.data(), doracle.data(), s.m, s.n, s.k, alpha,
                  beta, nullptr);
    CKL_CUDA_CHECK(cudaDeviceSynchronize());
    const auto oracle = doracle.to_host();
    const std::vector<double> ref(oracle.begin(), oracle.end());

    for (TVariant v : family_tvariants()) {
        SCOPED_TRACE(tvariant_name(v));
        dk.copy_from_host(c0);
        launch_kernel(v, da.data(), db.data(), dk.data(), s.m, s.n, s.k, alpha, beta, nullptr);
        CKL_CUDA_CHECK(cudaDeviceSynchronize());
        const auto got = dk.to_host();
        ASSERT_TRUE(ckl::all_finite(got)) << tvariant_name(v) << " produced a non finite result";
        const double err = ckl::relative_frobenius_error(got, ref);
        EXPECT_LT(err, kOracleGate) << tvariant_name(v) << " at " << s.label;
        ::testing::Test::RecordProperty(std::string("vs_oracle_") + tvariant_name(v),
                                        ckl::test::sci(err));
    }
}

INSTANTIATE_TEST_SUITE_P(Tensor, TileFamilyUnaligned,
                         ::testing::Values(BigShape{127, 127, 127, "127x127x127"},
                                           BigShape{1000, 1000, 1000, "1000x1000x1000"},
                                           BigShape{4096, 4090, 4096, "4096x4090x4096"},
                                           BigShape{129, 257, 193, "129x257x193"},
                                           BigShape{8192, 64, 4096, "8192x64x4096"}),
                         big_shape_name);

// ---------------------------------------------------------------------------
// The workspace contract
// ---------------------------------------------------------------------------

class FamilyWorkspace : public GpuTest {};

// A caller that asks for the size and hands one in has to get the same answer as
// a caller that lets the driver allocate. If the two disagreed, the benchmark
// protocol (ask for the size, allocate outside the timed region) would be
// measuring a different computation from the one the tests check.
TEST_F(FamilyWorkspace, SuppliedWorkspaceMatchesTheDriverAllocatedOne) {
    const Shape s{256, 256, 1024, "workspace"};
    const Bits a_bits = random_half_bits(s.m, s.k, 2901);
    const Bits b_bits = random_half_bits(s.k, s.n, 2902);
    const std::vector<float> c0 = ckl::random_matrix(s.m, s.n, ckl::test::seed_stream(2903));
    const float alpha = 2.5f;
    const float beta = -0.75f;

    ckl::DeviceBuffer<std::uint16_t> da(a_bits.size());
    ckl::DeviceBuffer<std::uint16_t> db(b_bits.size());
    ckl::DeviceBuffer<float> owned(c0.size());
    ckl::DeviceBuffer<float> given(c0.size());
    da.copy_from_host(a_bits);
    db.copy_from_host(b_bits);
    const auto* ha = reinterpret_cast<const __half*>(da.data());
    const auto* hb = reinterpret_cast<const __half*>(db.data());

    for (int tile = 0; tile < ckl::gemm_tile_family_count(); ++tile) {
        SCOPED_TRACE("tile " + std::to_string(tile));
        {
            const std::size_t need =
                ckl::gemm_split_k_workspace_size(s.m, s.n, s.k, kTestSplits, tile);
            ASSERT_GT(need, 0u) << "this shape does split, so it needs partial planes";
            ckl::DeviceBuffer<std::uint8_t> scratch(need);
            owned.copy_from_host(c0);
            given.copy_from_host(c0);
            ckl::gemm_split_k(ha, hb, owned.data(), s.m, s.n, s.k, alpha, beta, kTestSplits, tile,
                              nullptr, 0, nullptr);
            ckl::gemm_split_k(ha, hb, given.data(), s.m, s.n, s.k, alpha, beta, kTestSplits, tile,
                              scratch.data(), scratch.bytes(), nullptr);
            CKL_CUDA_CHECK(cudaDeviceSynchronize());
            EXPECT_EQ(owned.to_host(), given.to_host()) << "split-K disagreed with itself";
        }
        {
            const std::size_t need = ckl::gemm_stream_k_workspace_size(s.m, s.n, s.k, tile);
            ckl::DeviceBuffer<std::uint8_t> scratch(need > 0 ? need : 1);
            owned.copy_from_host(c0);
            given.copy_from_host(c0);
            ckl::gemm_stream_k(ha, hb, owned.data(), s.m, s.n, s.k, alpha, beta, tile, nullptr, 0,
                               nullptr);
            ckl::gemm_stream_k(ha, hb, given.data(), s.m, s.n, s.k, alpha, beta, tile,
                               scratch.data(), scratch.bytes(), nullptr);
            CKL_CUDA_CHECK(cudaDeviceSynchronize());
            EXPECT_EQ(owned.to_host(), given.to_host()) << "stream-K disagreed with itself";
        }
    }
}

// A workspace too small for the shape is a caller error, and a caller error that
// ran anyway would write outside the buffer it was given.
TEST_F(FamilyWorkspace, TooSmallAWorkspaceIsRefused) {
    const Shape s{256, 256, 1024, "small_workspace"};
    ckl::DeviceBuffer<std::uint16_t> da(elems(s.m, s.k));
    ckl::DeviceBuffer<std::uint16_t> db(elems(s.k, s.n));
    ckl::DeviceBuffer<float> dc(elems(s.m, s.n));
    ckl::DeviceBuffer<std::uint8_t> tiny(64);
    da.zero();
    db.zero();
    dc.zero();
    tiny.zero();
    const auto* ha = reinterpret_cast<const __half*>(da.data());
    const auto* hb = reinterpret_cast<const __half*>(db.data());

    try {
        ckl::gemm_split_k(ha, hb, dc.data(), s.m, s.n, s.k, 1.0f, 0.0f, kTestSplits, 0, tiny.data(),
                          tiny.bytes(), nullptr);
        FAIL() << "split-K accepted a workspace far too small for the shape";
    } catch (const ckl::Error& e) {
        EXPECT_EQ(e.status(), ckl::Status::kInvalidValue) << e.what();
    }
    try {
        ckl::gemm_stream_k(ha, hb, dc.data(), s.m, s.n, s.k, 1.0f, 0.0f, 0, tiny.data(),
                           tiny.bytes(), nullptr);
        FAIL() << "stream-K accepted a workspace far too small for the shape";
    } catch (const ckl::Error& e) {
        EXPECT_EQ(e.status(), ckl::Status::kInvalidValue) << e.what();
    }
    CKL_CUDA_CHECK(cudaDeviceSynchronize());
}

TEST_F(FamilyWorkspace, TileIndexOutsideTheFamilyIsRefused) {
    ckl::DeviceBuffer<std::uint16_t> da(elems(64, 64));
    ckl::DeviceBuffer<float> dc(elems(64, 64));
    da.zero();
    dc.zero();
    const auto* h = reinterpret_cast<const __half*>(da.data());
    for (int bad : {-1, ckl::gemm_tile_family_count()}) {
        EXPECT_THROW(ckl::gemm_tile_family(h, h, dc.data(), 64, 64, 64, 1.0f, 0.0f, bad, nullptr),
                     ckl::Error);
        EXPECT_THROW(
            ckl::gemm_split_k(h, h, dc.data(), 64, 64, 64, 1.0f, 0.0f, 2, bad, nullptr, 0, nullptr),
            ckl::Error);
        EXPECT_THROW(
            ckl::gemm_stream_k(h, h, dc.data(), 64, 64, 64, 1.0f, 0.0f, bad, nullptr, 0, nullptr),
            ckl::Error);
        EXPECT_EQ(ckl::gemm_tile_family_shape(bad).m, 0);
        EXPECT_EQ(ckl::gemm_tile_family_blocks_per_sm(bad), 0);
    }
    CKL_CUDA_CHECK(cudaDeviceSynchronize());
}

// The split count a caller asks for is a request. Reporting back what was
// actually used is what keeps ckl::GemmPlan::splits from being a guess.
TEST_F(FamilyWorkspace, SplitCountIsRoundedToWholeKSteps) {
    for (int tile = 0; tile < ckl::gemm_tile_family_count(); ++tile) {
        const ckl::GemmTile t = ckl::gemm_tile_family_shape(tile);
        SCOPED_TRACE("tile " + std::to_string(tile));
        // A K shorter than two steps cannot be cut in two.
        EXPECT_EQ(ckl::gemm_split_k_slices(t.k, 4, tile), 1);
        // Eight steps cut four ways is two steps each, which is exact.
        EXPECT_EQ(ckl::gemm_split_k_slices(8 * t.k, 4, tile), 4);
        // A K that is not a multiple of the step still covers the whole range.
        const int odd = 8 * t.k + 1;
        const int used = ckl::gemm_split_k_slices(odd, 4, tile);
        EXPECT_GE(used, 1);
        EXPECT_LE(used, 4);
        EXPECT_EQ(
            ckl::gemm_split_k_workspace_size(16, 16, odd, 4, tile),
            used > 1 ? static_cast<std::size_t>(used) * 16 * 16 * sizeof(float) : std::size_t{0});
    }
}

TEST_F(FamilyWorkspace, EveryShapeInTheFamilyFitsTheHardware) {
    ASSERT_EQ(ckl::gemm_tile_family_count(), CKL_TILE_FAMILY_COUNT);
    for (int i = 0; i < ckl::gemm_tile_family_count(); ++i) {
        const ckl::GemmTile t = ckl::gemm_tile_family_shape(i);
        SCOPED_TRACE("tile " + std::to_string(i));
        EXPECT_GT(t.m, 0);
        EXPECT_GT(t.n, 0);
        EXPECT_GT(t.k, 0);
        // Three stages of A and B, inside the per block opt-in this part allows.
        const std::size_t smem = 3u * static_cast<std::size_t>(t.m * t.k + t.k * t.n) * 2u;
        EXPECT_LE(smem, 101376u) << "tile " << i << " asks for " << smem << " bytes";
        EXPECT_LE(t.warps_m * t.warps_n, 48) << "a block cannot hold more warps than an SM";
        // The occupancy answer is what the dispatch heuristic divides by, so a
        // zero here would make every wave count infinite.
        EXPECT_GT(ckl::gemm_tile_family_blocks_per_sm(i), 0);
    }
}

// ---------------------------------------------------------------------------
// The templated shared memory maps
// ---------------------------------------------------------------------------
//
// gemm_tile_swizzle.hpp generalizes the two maps gemm_swizzle.hpp fixes at
// 128 by 128 by 32. The properties are the ones the fixed maps are held to, and
// they are checked here for every shape in the family, on the host, because they
// are integer arithmetic and not something a profiler counter would answer.

int tile_bank_of(int half_offset) {
    return (half_offset / 2) % 32;
}

template <int BM, int BK>
void check_a_map() {
    std::vector<int> seen(static_cast<std::size_t>(BM) * BK, -1);
    for (int r = 0; r < BM; ++r) {
        for (int c = 0; c < BK; ++c) {
            const int o = ckl::detail::swizzle_a_tile<BK>(r, c);
            ASSERT_GE(o, 0);
            ASSERT_LT(o, BM * BK);
            ASSERT_EQ(seen[static_cast<std::size_t>(o)], -1)
                << "A offset " << o << " is written twice at BM " << BM << " BK " << BK;
            seen[static_cast<std::size_t>(o)] = r * BK + c;
        }
    }
    for (int r = 0; r < BM; ++r) {
        for (int c = 0; c < BK; c += 8) {
            const int base = ckl::detail::swizzle_a_tile<BK>(r, c);
            ASSERT_EQ(base % 8, 0);
            for (int j = 0; j < 8; ++j) {
                ASSERT_EQ(ckl::detail::swizzle_a_tile<BK>(r, c + j), base + j);
            }
        }
    }
    // Every A wavefront the mainloop issues: row_base over the 16 row mma tiles,
    // k_off over the K substeps, and the x4 form's four groups of eight lanes.
    for (int row_base = 0; row_base < BM; row_base += 16) {
        for (int k_off = 0; k_off < BK; k_off += 16) {
            for (int group = 0; group < 4; ++group) {
                std::vector<int> banks;
                for (int i = 0; i < 8; ++i) {
                    const int lane = group * 8 + i;
                    banks.push_back(tile_bank_of(ckl::detail::swizzle_a_tile<BK>(
                        row_base + (lane % 16), k_off + (lane / 16) * 8)));
                }
                for (std::size_t x = 0; x < banks.size(); ++x) {
                    for (std::size_t y = x + 1; y < banks.size(); ++y) {
                        ASSERT_NE(banks[x], banks[y])
                            << "A conflict at BM " << BM << " BK " << BK << " row_base " << row_base
                            << " k_off " << k_off << " group " << group;
                    }
                }
            }
        }
    }
}

template <int BN, int BK>
void check_b_map() {
    std::vector<int> seen(static_cast<std::size_t>(BK) * BN, -1);
    for (int r = 0; r < BK; ++r) {
        for (int c = 0; c < BN; ++c) {
            const int o = ckl::detail::swizzle_b_tile<BN>(r, c);
            ASSERT_GE(o, 0);
            ASSERT_LT(o, BK * BN);
            ASSERT_EQ(seen[static_cast<std::size_t>(o)], -1)
                << "B offset " << o << " is written twice at BN " << BN << " BK " << BK;
            seen[static_cast<std::size_t>(o)] = r * BN + c;
        }
    }
    for (int r = 0; r < BK; ++r) {
        for (int c = 0; c < BN; c += 8) {
            const int base = ckl::detail::swizzle_b_tile<BN>(r, c);
            ASSERT_EQ(base % 8, 0);
            for (int j = 0; j < 8; ++j) {
                ASSERT_EQ(ckl::detail::swizzle_b_tile<BN>(r, c + j), base + j);
            }
        }
    }
    for (int col_base = 0; col_base < BN; col_base += 16) {
        for (int k_off = 0; k_off < BK; k_off += 16) {
            for (int group = 0; group < 4; ++group) {
                std::vector<int> banks;
                for (int i = 0; i < 8; ++i) {
                    const int lane = group * 8 + i;
                    banks.push_back(tile_bank_of(ckl::detail::swizzle_b_tile<BN>(
                        k_off + (lane % 16), col_base + (lane / 16) * 8)));
                }
                for (std::size_t x = 0; x < banks.size(); ++x) {
                    for (std::size_t y = x + 1; y < banks.size(); ++y) {
                        ASSERT_NE(banks[x], banks[y])
                            << "B conflict at BN " << BN << " BK " << BK << " col_base " << col_base
                            << " k_off " << k_off << " group " << group;
                    }
                }
            }
        }
    }
}

TEST(TileSwizzle, EveryFamilyShapeIsABijectionAndConflictFree){
#define CKL_CHECK_TILE_MAPS(IDX, BM, BN, BK, WM, WN) \
    {                                                \
        SCOPED_TRACE("tile " #IDX);                  \
        check_a_map<BM, BK>();                       \
        check_b_map<BN, BK>();                       \
    }
    CKL_TILE_FAMILY_FOR_EACH(CKL_CHECK_TILE_MAPS)
#undef CKL_CHECK_TILE_MAPS
}

// The templated maps and the fixed ones in gemm_swizzle.hpp describe the same
// tile at 128 by 128 by 32. Two copies that drifted apart would put the cp.async
// store of one kernel and the ldmatrix load of another at different addresses,
// which is the kind of bug a shared header exists to prevent.
TEST(TileSwizzle, MatchesTheFixedShapeMapsAt128x128x32) {
    for (int r = 0; r < ckl::detail::kSwzTileM; ++r) {
        for (int c = 0; c < ckl::detail::kSwzTileK; ++c) {
            ASSERT_EQ(ckl::detail::swizzle_a_tile<32>(r, c), ckl::detail::swizzle_a(r, c))
                << "A map differs at row " << r << " col " << c;
        }
    }
    for (int r = 0; r < ckl::detail::kSwzTileK; ++r) {
        for (int c = 0; c < ckl::detail::kSwzTileN; ++c) {
            ASSERT_EQ(ckl::detail::swizzle_b_tile<128>(r, c), ckl::detail::swizzle_b(r, c))
                << "B map differs at row " << r << " col " << c;
        }
    }
}

}  // namespace
