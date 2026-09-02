// The descriptor driven API: ckl::Context, ckl::gemm, ckl::gemm_query, and the
// promise that dispatch never lies about what it did.
//
// Ground rule 6 says a silent fallback is a bug: either the fast path runs, or a
// status comes back, or chosen reports the path actually taken. That rule is
// only worth having if something checks it, which is what most of this file is.
// The shapes below are chosen so each one has exactly one right answer, and a
// regression that made everything fall back to cuBLAS would fail more than half
// of them.

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <gtest/gtest.h>

#include "ckl/context.hpp"
#include "ckl/cuda_check.hpp"
#include "ckl/device_buffer.hpp"
#include "ckl/gemm.hpp"
#include "ckl/status.hpp"
#include "ckl/types.hpp"
#include "gpu_environment.hpp"
#include "reference.hpp"

namespace {

using ckl::Algo;
using ckl::DType;
using ckl::GemmDesc;
using ckl::Layout;
using ckl::Op;
using ckl::Status;
using ckl::test::GpuTest;
using ckl::test::GpuTestWithParam;

std::size_t elems(std::int64_t rows, std::int64_t cols) {
    return static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
}

// Square, packed, row major, one batch. The dtype decides which rung kAuto can
// reach, so it is the only thing that varies.
GemmDesc square_desc(DType dt, std::int64_t n, Algo algo) {
    GemmDesc d;
    d.layout = Layout::kRowMajor;
    d.m = n;
    d.n = n;
    d.k = n;
    d.dt_a = dt;
    d.dt_b = dt;
    d.dt_c = DType::kR32F;
    d.lda = n;
    d.ldb = n;
    d.ldc = n;
    d.algo = algo;
    return d;
}

// Host storage for the inputs in whatever precision the descriptor names, plus
// the float images of those stored values so a host reference can use exactly
// what the device saw.
struct Operand {
    std::vector<std::uint8_t> bytes;
    std::vector<float> values;
};

Operand make_operand(DType dt, std::size_t count, std::uint64_t stream_id) {
    const auto f32 =
        ckl::random_matrix(static_cast<int>(count), 1, ckl::test::seed_stream(stream_id));
    Operand op;
    op.values.resize(count);
    op.bytes.resize(count * ckl::dtype_size(dt));
    // The 16 bit formats go through their raw structs rather than memcpy: the
    // storage member of __half and __nv_bfloat16 is protected, and copying over
    // it is a -Wclass-memaccess warning that this tree treats as a real one.
    auto store16 = [&op](std::size_t i, unsigned short bits) {
        op.bytes[i * 2] = static_cast<std::uint8_t>(bits & 0xFFu);
        op.bytes[i * 2 + 1] = static_cast<std::uint8_t>((bits >> 8) & 0xFFu);
    };
    for (std::size_t i = 0; i < count; ++i) {
        switch (dt) {
            case DType::kR16F: {
                const __half h = __float2half(f32[i]);
                store16(i, static_cast<__half_raw>(h).x);
                op.values[i] = __half2float(h);
                break;
            }
            case DType::kR16BF: {
                const __nv_bfloat16 h = __float2bfloat16(f32[i]);
                store16(i, static_cast<__nv_bfloat16_raw>(h).x);
                op.values[i] = __bfloat162float(h);
                break;
            }
            case DType::kR32F:
            default: {
                std::memcpy(op.bytes.data() + i * 4, &f32[i], 4);
                op.values[i] = f32[i];
                break;
            }
        }
    }
    return op;
}

// One descriptor driven call, with real device memory behind every pointer.
struct DescRun {
    Status status = Status::kInternal;
    Algo chosen = Algo::kAuto;
    std::vector<float> c;
};

DescRun run_desc(ckl::Context& ctx, const GemmDesc& d, const Operand& a, const Operand& b,
                 const std::vector<float>& c0, float alpha, float beta) {
    ckl::DeviceBuffer<std::uint8_t> da(a.bytes.size());
    ckl::DeviceBuffer<std::uint8_t> db(b.bytes.size());
    ckl::DeviceBuffer<float> dc(c0.size());
    da.copy_from_host(a.bytes);
    db.copy_from_host(b.bytes);
    dc.copy_from_host(c0);

    DescRun r;
    r.chosen = Algo::kAuto;
    r.status = ckl::gemm(ctx, d, &alpha, da.data(), db.data(), &beta, dc.data(), &r.chosen);
    if (r.status == Status::kSuccess) {
        CKL_CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
        r.c = dc.to_host();
    }
    return r;
}

// ---------------------------------------------------------------------------
// What kAuto picks
// ---------------------------------------------------------------------------

struct AutoCase {
    const char* label;
    DType dt;
    std::int64_t n;
    Algo expect;
};

class DispatchAuto : public GpuTestWithParam<AutoCase> {};

std::string auto_case_name(const ::testing::TestParamInfo<AutoCase>& info) {
    return info.param.label;
}

TEST_P(DispatchAuto, PicksTheDocumentedPathAndSaysSo) {
    const AutoCase c = GetParam();
    ckl::Context ctx;
    const GemmDesc d = square_desc(c.dt, c.n, Algo::kAuto);

    // The query answers without launching anything, and it has to agree with
    // what the call reports; two answers that drift apart would make the
    // benchmark rows label the wrong kernel.
    EXPECT_EQ(ckl::gemm_query(ctx, d), c.expect) << "gemm_query disagrees";

    const Operand a = make_operand(c.dt, elems(c.n, c.n), 2101);
    const Operand b = make_operand(c.dt, elems(c.n, c.n), 2102);
    const std::vector<float> c0(elems(c.n, c.n), 0.0f);

    const DescRun got = run_desc(ctx, d, a, b, c0, 1.0f, 0.0f);
    ASSERT_EQ(got.status, Status::kSuccess);
    EXPECT_EQ(got.chosen, c.expect) << "chosen said " << ckl::algo_name(got.chosen) << ", expected "
                                    << ckl::algo_name(c.expect);

    // Cross check the numbers against the vendor path for the same descriptor,
    // so "it picked the right rung" cannot pass while the rung computes garbage.
    GemmDesc vendor = d;
    vendor.algo = Algo::kCublas;
    const DescRun ref = run_desc(ctx, vendor, a, b, c0, 1.0f, 0.0f);
    ASSERT_EQ(ref.status, Status::kSuccess);
    const std::vector<double> ref_d(ref.c.begin(), ref.c.end());
    EXPECT_LT(ckl::relative_frobenius_error(got.c, ref_d), ckl::tol(static_cast<int>(c.n)));
}

INSTANTIATE_TEST_SUITE_P(Gemm, DispatchAuto,
                         ::testing::Values(
                             // FP16 now goes to the tile family whatever the shape, so its rows
                             // live in DispatchFamily below, which checks the Section 9.6 rule
                             // instead of a remembered algorithm name.
                             AutoCase{"bf16_128_wmma", DType::kR16BF, 128, Algo::kWmmaBf16},
                             AutoCase{"bf16_100_cublas", DType::kR16BF, 100, Algo::kCublas},
                             // FP32: large and block aligned goes to the double buffered kernel,
                             // small stays on one launch, everything between goes to cuBLAS.
                             AutoCase{"fp32_512_cp_async", DType::kR32F, 512, Algo::kCpAsync},
                             AutoCase{"fp32_128_naive", DType::kR32F, 128, Algo::kNaive},
                             AutoCase{"fp32_100_naive", DType::kR32F, 100, Algo::kNaive},
                             AutoCase{"fp32_640_cp_async", DType::kR32F, 640, Algo::kCpAsync},
                             AutoCase{"fp32_256_cublas", DType::kR32F, 256, Algo::kCublas},
                             // Large enough for the double buffered kernel but not divisible by its
                             // 128 by 128 block, so it lands on the vendor path and says so.
                             AutoCase{"fp32_520_cublas", DType::kR32F, 520, Algo::kCublas}),
                         auto_case_name);

// ---------------------------------------------------------------------------
// An explicitly named algorithm is never rerouted
// ---------------------------------------------------------------------------

struct RefusalCase {
    const char* label;
    Algo algo;
    DType dt;
    std::int64_t n;
    Status expect;
};

class DispatchRefusal : public GpuTestWithParam<RefusalCase> {};

std::string refusal_case_name(const ::testing::TestParamInfo<RefusalCase>& info) {
    return info.param.label;
}

// This is the no silent fallback gate. Every case names an algorithm the shape
// or the precision rules out; the call has to come back with a status and with
// chosen still naming what the caller asked for, not what the dispatcher would
// rather have run.
TEST_P(DispatchRefusal, NamedAlgorithmIsRefusedNotRerouted) {
    const RefusalCase c = GetParam();
    ckl::Context ctx;
    const GemmDesc d = square_desc(c.dt, c.n, c.algo);
    const Operand a = make_operand(c.dt, elems(c.n, c.n), 2201);
    const Operand b = make_operand(c.dt, elems(c.n, c.n), 2202);
    const std::vector<float> c0(elems(c.n, c.n), 3.5f);

    const DescRun got = run_desc(ctx, d, a, b, c0, 1.0f, 0.0f);
    EXPECT_EQ(got.status, c.expect);
    EXPECT_EQ(got.chosen, c.algo) << "chosen was rewritten to " << ckl::algo_name(got.chosen);
}

INSTANTIATE_TEST_SUITE_P(
    Gemm, DispatchRefusal,
    ::testing::Values(
        // FP32 only rungs handed FP16 input.
        RefusalCase{"naive_on_fp16", Algo::kNaive, DType::kR16F, 128, Status::kNotSupported},
        RefusalCase{"tiled_on_fp16", Algo::kTiled, DType::kR16F, 128, Status::kNotSupported},
        // Block factors the shape does not divide.
        RefusalCase{"register_unaligned", Algo::kRegister, DType::kR32F, 100,
                    Status::kNotSupported},
        RefusalCase{"cp_async_unaligned", Algo::kCpAsync, DType::kR32F, 100, Status::kNotSupported},
        RefusalCase{"wmma_fp16_unaligned", Algo::kWmmaFp16, DType::kR16F, 100,
                    Status::kNotSupported},
        RefusalCase{"mma_ptx_unaligned", Algo::kMmaPtx, DType::kR16F, 100, Status::kNotSupported},
        RefusalCase{"mma_ldm_unaligned", Algo::kMmaLdm, DType::kR16F, 100, Status::kNotSupported},
        RefusalCase{"mma_opt_unaligned", Algo::kMmaOpt, DType::kR16F, 100, Status::kNotSupported},
        // 64 divides the WMMA block but not the mma_opt one, so naming mma_opt
        // at 64 is a refusal even though kAuto would have found a tensor rung.
        RefusalCase{"mma_opt_64", Algo::kMmaOpt, DType::kR16F, 64, Status::kNotSupported},
        // Wrong precision for the rung.
        RefusalCase{"wmma_bf16_on_fp16", Algo::kWmmaBf16, DType::kR16F, 128, Status::kNotSupported},
        RefusalCase{"wmma_fp16_on_bf16", Algo::kWmmaFp16, DType::kR16BF, 128,
                    Status::kNotSupported},
        RefusalCase{"mma_opt_on_fp32", Algo::kMmaOpt, DType::kR32F, 128, Status::kNotSupported},
        // The family is FP16 in and FP32 out. Handed FP32 it refuses, the same
        // way every other precision specific rung does.
        RefusalCase{"tile_family_on_fp32", Algo::kTileFamily, DType::kR32F, 128,
                    Status::kNotSupported},
        RefusalCase{"split_k_on_fp32", Algo::kSplitK, DType::kR32F, 128, Status::kNotSupported},
        RefusalCase{"stream_k_on_fp32", Algo::kStreamK, DType::kR32F, 128, Status::kNotSupported},
        // The CUTLASS reference line is FP16 in and FP32 out like the rest of
        // the tensor rungs, and its mainloop moves 16 bytes along the contiguous
        // axis, so a shape whose n or k is not a multiple of 8 is refused rather
        // than quietly handed to another kernel.
        RefusalCase{"cutlass_on_fp32", Algo::kCutlass, DType::kR32F, 128, Status::kNotSupported},
        RefusalCase{"cutlass_unaligned", Algo::kCutlass, DType::kR16F, 100, Status::kNotSupported}),
    refusal_case_name);

// ---------------------------------------------------------------------------
// Malformed descriptors
// ---------------------------------------------------------------------------

enum class Break {
    kNegativeM,
    kNegativeN,
    kNegativeK,
    kZeroBatch,
    kNullAlpha,
    kNullBeta,
    kSmallLda,
    kSmallLdb,
    kSmallLdc,
    kNullC,
    kNullA,
    kNullB,
    kHalfOutput,
    kMixedInputs,
};

struct BreakCase {
    const char* label;
    Break how;
    Status expect;
};

class DispatchValidation : public GpuTestWithParam<BreakCase> {};

std::string break_case_name(const ::testing::TestParamInfo<BreakCase>& info) {
    return info.param.label;
}

TEST_P(DispatchValidation, MalformedDescriptorIsRejected) {
    const BreakCase c = GetParam();
    ckl::Context ctx;
    const std::int64_t n = 128;
    GemmDesc d = square_desc(DType::kR32F, n, Algo::kAuto);

    ckl::DeviceBuffer<float> da(elems(n, n));
    ckl::DeviceBuffer<float> db(elems(n, n));
    ckl::DeviceBuffer<float> dc(elems(n, n));
    da.zero();
    db.zero();
    dc.zero();

    const float alpha = 1.0f;
    const float beta = 0.0f;
    const float* palpha = &alpha;
    const float* pbeta = &beta;
    const void* pa = da.data();
    const void* pb = db.data();
    void* pc = dc.data();

    switch (c.how) {
        case Break::kNegativeM:
            d.m = -1;
            break;
        case Break::kNegativeN:
            d.n = -1;
            break;
        case Break::kNegativeK:
            d.k = -1;
            break;
        case Break::kZeroBatch:
            d.batch_count = 0;
            break;
        case Break::kNullAlpha:
            palpha = nullptr;
            break;
        case Break::kNullBeta:
            pbeta = nullptr;
            break;
        case Break::kSmallLda:
            d.lda = 1;
            break;
        case Break::kSmallLdb:
            d.ldb = 1;
            break;
        case Break::kSmallLdc:
            d.ldc = 1;
            break;
        case Break::kNullC:
            pc = nullptr;
            break;
        case Break::kNullA:
            pa = nullptr;
            break;
        case Break::kNullB:
            pb = nullptr;
            break;
        case Break::kHalfOutput:
            d.dt_c = DType::kR16F;
            break;
        case Break::kMixedInputs:
            d.dt_a = DType::kR16F;
            break;
    }

    Algo chosen = Algo::kAuto;
    const Status s = ckl::gemm(ctx, d, palpha, pa, pb, pbeta, pc, &chosen);
    EXPECT_EQ(s, c.expect);
}

INSTANTIATE_TEST_SUITE_P(
    Gemm, DispatchValidation,
    ::testing::Values(BreakCase{"negative_m", Break::kNegativeM, Status::kInvalidValue},
                      BreakCase{"negative_n", Break::kNegativeN, Status::kInvalidValue},
                      BreakCase{"negative_k", Break::kNegativeK, Status::kInvalidValue},
                      BreakCase{"zero_batch", Break::kZeroBatch, Status::kInvalidValue},
                      BreakCase{"null_alpha", Break::kNullAlpha, Status::kInvalidValue},
                      BreakCase{"null_beta", Break::kNullBeta, Status::kInvalidValue},
                      BreakCase{"lda_too_small", Break::kSmallLda, Status::kInvalidValue},
                      BreakCase{"ldb_too_small", Break::kSmallLdb, Status::kInvalidValue},
                      BreakCase{"ldc_too_small", Break::kSmallLdc, Status::kInvalidValue},
                      BreakCase{"null_c", Break::kNullC, Status::kInvalidValue},
                      BreakCase{"null_a", Break::kNullA, Status::kInvalidValue},
                      BreakCase{"null_b", Break::kNullB, Status::kInvalidValue},
                      BreakCase{"half_output", Break::kHalfOutput, Status::kNotSupported},
                      BreakCase{"mixed_inputs", Break::kMixedInputs, Status::kNotSupported}),
    break_case_name);

// ---------------------------------------------------------------------------
// Layout and transpose
// ---------------------------------------------------------------------------

// Reads the logical (i, j) element of an operand, which is where the row major
// to column major identity in dispatch.cpp either holds or does not.
double logical(const std::vector<float>& data, Layout layout, Op op, std::int64_t ld,
               std::int64_t i, std::int64_t j) {
    const std::int64_t r = (op == Op::kN) ? i : j;
    const std::int64_t c = (op == Op::kN) ? j : i;
    const std::size_t idx = layout == Layout::kRowMajor ? static_cast<std::size_t>(r * ld + c)
                                                        : static_cast<std::size_t>(c * ld + r);
    return static_cast<double>(data[idx]);
}

struct LayoutCase {
    const char* label;
    Layout layout;
    Op op_a;
    Op op_b;
};

class DispatchLayout : public GpuTestWithParam<LayoutCase> {};

std::string layout_case_name(const ::testing::TestParamInfo<LayoutCase>& info) {
    return info.param.label;
}

// Everything except row major with both operands untransposed goes to cuBLAS,
// and the answer still has to be the right one in the caller's own layout. This
// is the test for the transpose identity the dispatcher documents.
TEST_P(DispatchLayout, MatchesTheHostReferenceInTheCallersLayout) {
    const LayoutCase lc = GetParam();
    ckl::Context ctx;

    const std::int64_t m = 96;
    const std::int64_t n = 64;
    const std::int64_t k = 80;
    GemmDesc d;
    d.layout = lc.layout;
    d.op_a = lc.op_a;
    d.op_b = lc.op_b;
    d.m = m;
    d.n = n;
    d.k = k;
    d.dt_a = DType::kR32F;
    d.dt_b = DType::kR32F;
    d.dt_c = DType::kR32F;

    const std::int64_t a_rows = (lc.op_a == Op::kN) ? m : k;
    const std::int64_t a_cols = (lc.op_a == Op::kN) ? k : m;
    const std::int64_t b_rows = (lc.op_b == Op::kN) ? k : n;
    const std::int64_t b_cols = (lc.op_b == Op::kN) ? n : k;
    d.lda = (lc.layout == Layout::kRowMajor) ? a_cols : a_rows;
    d.ldb = (lc.layout == Layout::kRowMajor) ? b_cols : b_rows;
    d.ldc = (lc.layout == Layout::kRowMajor) ? n : m;

    const auto a = ckl::random_matrix(static_cast<int>(a_rows), static_cast<int>(a_cols),
                                      ckl::test::seed_stream(2301));
    const auto b = ckl::random_matrix(static_cast<int>(b_rows), static_cast<int>(b_cols),
                                      ckl::test::seed_stream(2302));
    const auto c0 =
        ckl::random_matrix(static_cast<int>(m), static_cast<int>(n), ckl::test::seed_stream(2303));
    const float alpha = 1.25f;
    const float beta = 0.5f;

    ckl::DeviceBuffer<float> da(a.size());
    ckl::DeviceBuffer<float> db(b.size());
    ckl::DeviceBuffer<float> dc(c0.size());
    da.copy_from_host(a);
    db.copy_from_host(b);
    dc.copy_from_host(c0);

    Algo chosen = Algo::kAuto;
    ASSERT_EQ(ckl::gemm(ctx, d, &alpha, da.data(), db.data(), &beta, dc.data(), &chosen),
              Status::kSuccess);
    CKL_CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
    const auto out = dc.to_host();

    const bool hand_shape =
        lc.layout == Layout::kRowMajor && lc.op_a == Op::kN && lc.op_b == Op::kN;
    if (!hand_shape) {
        EXPECT_EQ(chosen, Algo::kCublas)
            << "a layout no hand kernel can take has to report the vendor path";
    }

    double worst = 0.0;
    for (std::int64_t i = 0; i < m; ++i) {
        for (std::int64_t j = 0; j < n; ++j) {
            double acc = 0.0;
            double mag = 0.0;
            for (std::int64_t p = 0; p < k; ++p) {
                const double av = logical(a, lc.layout, lc.op_a, d.lda, i, p);
                const double bv = logical(b, lc.layout, lc.op_b, d.ldb, p, j);
                acc += av * bv;
                mag += std::fabs(av) * std::fabs(bv);
            }
            const std::size_t idx = lc.layout == Layout::kRowMajor
                                        ? static_cast<std::size_t>(i * d.ldc + j)
                                        : static_cast<std::size_t>(j * d.ldc + i);
            const double want =
                static_cast<double>(alpha) * acc + static_cast<double>(beta) * c0[idx];
            const double scale = std::fabs(static_cast<double>(alpha)) * mag +
                                 std::fabs(static_cast<double>(beta) * c0[idx]);
            const double denom = scale > 1e-30 ? scale : 1.0;
            const double r = std::fabs(static_cast<double>(out[idx]) - want) / denom;
            if (r > worst) {
                worst = r;
            }
        }
    }
    EXPECT_LT(worst, ckl::tol(static_cast<int>(k)));
}

INSTANTIATE_TEST_SUITE_P(Gemm, DispatchLayout,
                         ::testing::Values(LayoutCase{"row_nn", Layout::kRowMajor, Op::kN, Op::kN},
                                           LayoutCase{"row_tn", Layout::kRowMajor, Op::kT, Op::kN},
                                           LayoutCase{"row_nt", Layout::kRowMajor, Op::kN, Op::kT},
                                           LayoutCase{"row_tt", Layout::kRowMajor, Op::kT, Op::kT},
                                           LayoutCase{"col_nn", Layout::kColMajor, Op::kN, Op::kN},
                                           LayoutCase{"col_tn", Layout::kColMajor, Op::kT, Op::kN},
                                           LayoutCase{"col_nt", Layout::kColMajor, Op::kN, Op::kT},
                                           LayoutCase{"col_tt", Layout::kColMajor, Op::kT, Op::kT},
                                           // kC is accepted and treated as a plain transpose, since
                                           // every type the library ships is real.
                                           LayoutCase{"row_cn", Layout::kRowMajor, Op::kC, Op::kN}),
                         layout_case_name);

// ---------------------------------------------------------------------------
// Batching, workspace, and the odds and ends
// ---------------------------------------------------------------------------

class Dispatch : public GpuTest {};

TEST_F(Dispatch, StridedBatchedGoesToCublasAndSaysSo) {
    ckl::Context ctx;
    const std::int64_t n = 128;
    const std::int32_t batches = 3;
    GemmDesc d = square_desc(DType::kR32F, n, Algo::kAuto);
    d.batch_count = batches;
    d.stride_a = n * n;
    d.stride_b = n * n;
    d.stride_c = n * n;

    const std::size_t per = elems(n, n);
    const auto a =
        ckl::random_matrix(static_cast<int>(per * batches), 1, ckl::test::seed_stream(2401));
    const auto b =
        ckl::random_matrix(static_cast<int>(per * batches), 1, ckl::test::seed_stream(2402));
    const std::vector<float> c0(per * batches, 0.0f);

    ckl::DeviceBuffer<float> da(a.size());
    ckl::DeviceBuffer<float> db(b.size());
    ckl::DeviceBuffer<float> dc(c0.size());
    da.copy_from_host(a);
    db.copy_from_host(b);
    dc.copy_from_host(c0);

    const float alpha = 1.0f;
    const float beta = 0.0f;
    Algo chosen = Algo::kAuto;
    ASSERT_EQ(ckl::gemm(ctx, d, &alpha, da.data(), db.data(), &beta, dc.data(), &chosen),
              Status::kSuccess);
    EXPECT_EQ(chosen, Algo::kCublas) << "batching has no hand written form; chosen must say so";
    CKL_CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
    const auto out = dc.to_host();

    // Each batch has to equal the same single GEMM run on its own slice.
    for (std::int32_t bi = 0; bi < batches; ++bi) {
        const std::size_t off = static_cast<std::size_t>(bi) * per;
        const std::vector<float> a_slice(a.begin() + static_cast<std::ptrdiff_t>(off),
                                         a.begin() + static_cast<std::ptrdiff_t>(off + per));
        const std::vector<float> b_slice(b.begin() + static_cast<std::ptrdiff_t>(off),
                                         b.begin() + static_cast<std::ptrdiff_t>(off + per));
        const std::vector<float> zero(per, 0.0f);
        const ckl::GemmRef ref =
            ckl::gemm_reference_scaled(a_slice, b_slice, zero, static_cast<int>(n),
                                       static_cast<int>(n), static_cast<int>(n), alpha, beta);
        const std::vector<float> got(out.begin() + static_cast<std::ptrdiff_t>(off),
                                     out.begin() + static_cast<std::ptrdiff_t>(off + per));
        EXPECT_LT(ckl::max_scaled_residual(got, ref), ckl::tol(static_cast<int>(n)))
            << "batch " << bi;
    }
}

TEST_F(Dispatch, ExplicitHandKernelRefusesToBatch) {
    ckl::Context ctx;
    const std::int64_t n = 128;
    GemmDesc d = square_desc(DType::kR16F, n, Algo::kMmaOpt);
    d.batch_count = 2;
    d.stride_a = n * n;
    d.stride_b = n * n;
    d.stride_c = n * n;

    ckl::DeviceBuffer<std::uint8_t> da(elems(n, n) * 2 * 2);
    ckl::DeviceBuffer<std::uint8_t> db(elems(n, n) * 2 * 2);
    ckl::DeviceBuffer<float> dc(elems(n, n) * 2);
    da.zero();
    db.zero();
    dc.zero();

    const float alpha = 1.0f;
    const float beta = 0.0f;
    Algo chosen = Algo::kAuto;
    EXPECT_EQ(ckl::gemm(ctx, d, &alpha, da.data(), db.data(), &beta, dc.data(), &chosen),
              Status::kNotSupported);
    EXPECT_EQ(chosen, Algo::kMmaOpt);
}

TEST_F(Dispatch, ChosenIsWrittenEvenWhenTheCallFails) {
    ckl::Context ctx;
    GemmDesc d = square_desc(DType::kR32F, 128, Algo::kRegister);
    d.ldc = 1;  // malformed, so the call fails before anything runs
    ckl::DeviceBuffer<float> da(elems(128, 128));
    ckl::DeviceBuffer<float> db(elems(128, 128));
    ckl::DeviceBuffer<float> dc(elems(128, 128));
    da.zero();
    db.zero();
    dc.zero();

    const float alpha = 1.0f;
    const float beta = 0.0f;
    Algo chosen = Algo::kCublas;  // deliberately not the answer
    EXPECT_EQ(ckl::gemm(ctx, d, &alpha, da.data(), db.data(), &beta, dc.data(), &chosen),
              Status::kInvalidValue);
    EXPECT_EQ(chosen, Algo::kRegister) << "a caller reading chosen after a failure has to see what "
                                          "the dispatcher was aiming at";
}

// The reference line runs when a caller names it, and kAuto never lands on it.
// A dispatcher that started choosing CUTLASS for the library's own kAuto path
// would make every ladder comparison a comparison of CUTLASS with itself.
TEST_F(Dispatch, CutlassRungRunsWhenNamedAndIsNeverAutoChosen) {
    ckl::Context ctx;
    const std::int64_t n = 256;
    const GemmDesc named = square_desc(DType::kR16F, n, Algo::kCutlass);
    const Operand a = make_operand(DType::kR16F, elems(n, n), 2401);
    const Operand b = make_operand(DType::kR16F, elems(n, n), 2402);
    const std::vector<float> c0(elems(n, n), 0.0f);

    const DescRun got = run_desc(ctx, named, a, b, c0, 1.0f, 0.0f);
    ASSERT_EQ(got.status, Status::kSuccess);
    EXPECT_EQ(got.chosen, Algo::kCutlass);

    const std::vector<double> ref =
        ckl::gemm_reference(a.values, b.values, c0, static_cast<int>(n), static_cast<int>(n),
                            static_cast<int>(n), 1.0f, 0.0f);
    EXPECT_LT(ckl::relative_frobenius_error(got.c, ref), ckl::tol(static_cast<int>(n)));

    for (std::int64_t side : {static_cast<std::int64_t>(256), static_cast<std::int64_t>(1024)}) {
        const GemmDesc probe = square_desc(DType::kR16F, side, Algo::kAuto);
        EXPECT_NE(ckl::gemm_query(ctx, probe), Algo::kCutlass)
            << "kAuto reached for the reference line at side " << side;
    }

    // The instantiation the docs describe, read off the template rather than
    // off a comment.
    const ckl::GemmTile tile = ckl::gemm_cutlass_tile();
    EXPECT_EQ(tile.m, 128);
    EXPECT_EQ(tile.n, 128);
    EXPECT_EQ(tile.k, 32);
    EXPECT_EQ(tile.warps_m, 2);
    EXPECT_EQ(tile.warps_n, 2);
}

TEST_F(Dispatch, NullChosenIsAllowed) {
    ckl::Context ctx;
    const GemmDesc d = square_desc(DType::kR32F, 64, Algo::kAuto);
    ckl::DeviceBuffer<float> da(elems(64, 64));
    ckl::DeviceBuffer<float> db(elems(64, 64));
    ckl::DeviceBuffer<float> dc(elems(64, 64));
    da.zero();
    db.zero();
    dc.zero();
    const float alpha = 1.0f;
    const float beta = 0.0f;
    EXPECT_EQ(ckl::gemm(ctx, d, &alpha, da.data(), db.data(), &beta, dc.data(), nullptr),
              Status::kSuccess);
    CKL_CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
}

TEST_F(Dispatch, QueryLaunchesNothing) {
    ckl::Context ctx;
    // No device memory is allocated at all here: a query that touched the
    // buffers would fault, and one that launched would leave a pending error.
    const GemmDesc d = square_desc(DType::kR16F, 1024, Algo::kAuto);
    const Algo picked = ckl::gemm_query(ctx, d);
    EXPECT_TRUE(picked == Algo::kTileFamily || picked == Algo::kSplitK || picked == Algo::kStreamK)
        << "an FP16 square goes to the tile family; the query said " << ckl::algo_name(picked);
    // The occupancy query the heuristic runs is not a launch, and it must not
    // leave a pending error behind either.
    EXPECT_EQ(cudaGetLastError(), cudaSuccess);
}

// The data parallel paths run out of registers and shared memory only. Split-K
// and stream-K are the two that ask for scratch, and they are covered in
// DispatchFamily below, which checks the answer against each driver's own query.
TEST_F(Dispatch, WorkspaceSizeIsZeroForThePathsThatNeedNoScratch) {
    ckl::Context ctx;
    for (std::int64_t n : {64, 128, 1024}) {
        EXPECT_EQ(ckl::gemm_workspace_size(ctx, square_desc(DType::kR32F, n, Algo::kAuto)), 0u);
        EXPECT_EQ(ckl::gemm_workspace_size(ctx, square_desc(DType::kR16F, n, Algo::kMmaOpt)), 0u);
        EXPECT_EQ(ckl::gemm_workspace_size(ctx, square_desc(DType::kR16F, n, Algo::kTileFamily)),
                  0u);
    }
}

TEST_F(Dispatch, EmptyOutputSucceedsAndWritesNothing) {
    ckl::Context ctx;
    const float canary = -777.25f;
    const std::vector<float> seed(elems(64, 64), canary);
    ckl::DeviceBuffer<float> da(elems(64, 64));
    ckl::DeviceBuffer<float> db(elems(64, 64));
    ckl::DeviceBuffer<float> dc(seed.size());
    da.zero();
    db.zero();
    dc.copy_from_host(seed);

    const float alpha = 1.25f;
    const float beta = 0.5f;
    for (int which = 0; which < 2; ++which) {
        GemmDesc d = square_desc(DType::kR32F, 64, Algo::kAuto);
        if (which == 0) {
            d.m = 0;
            d.ldc = 64;
        } else {
            d.n = 0;
            d.ldb = 0;
            d.ldc = 0;
        }
        Algo chosen = Algo::kAuto;
        ASSERT_EQ(ckl::gemm(ctx, d, &alpha, da.data(), db.data(), &beta, dc.data(), &chosen),
                  Status::kSuccess)
            << "case " << which;
    }
    CKL_CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
    for (float x : dc.to_host()) {
        ASSERT_EQ(x, canary);
    }
}

TEST_F(Dispatch, EmptyContractionAppliesBetaAndClearsOnZero) {
    ckl::Context ctx;
    const std::int64_t n = 64;
    const auto c0 =
        ckl::random_matrix(static_cast<int>(n), static_cast<int>(n), ckl::test::seed_stream(2501));
    ckl::DeviceBuffer<float> da(elems(n, n));
    ckl::DeviceBuffer<float> db(elems(n, n));
    ckl::DeviceBuffer<float> dc(c0.size());
    da.zero();
    db.zero();

    GemmDesc d = square_desc(DType::kR32F, n, Algo::kAuto);
    d.k = 0;
    d.lda = 0;
    d.ldb = n;

    {
        dc.copy_from_host(c0);
        const float alpha = 1.0f;
        const float beta = 0.5f;
        ASSERT_EQ(ckl::gemm(ctx, d, &alpha, da.data(), db.data(), &beta, dc.data(), nullptr),
                  Status::kSuccess);
        CKL_CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
        const auto out = dc.to_host();
        for (std::size_t i = 0; i < out.size(); ++i) {
            ASSERT_FLOAT_EQ(out[i], 0.5f * c0[i]) << "element " << i;
        }
    }
    {
        // beta zero clears C rather than scaling it, because a NaN C is legal
        // input when beta is zero and 0 * NaN is still NaN.
        const std::vector<float> nan_c(c0.size(), std::nan(""));
        dc.copy_from_host(nan_c);
        const float alpha = 1.0f;
        const float beta = 0.0f;
        ASSERT_EQ(ckl::gemm(ctx, d, &alpha, da.data(), db.data(), &beta, dc.data(), nullptr),
                  Status::kSuccess);
        CKL_CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
        for (float x : dc.to_host()) {
            ASSERT_EQ(x, 0.0f);
        }
    }
}

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------

class ContextTest : public GpuTest {};

TEST_F(ContextTest, ReportsItsDeviceAndCapability) {
    ckl::Context ctx;
    EXPECT_GE(ctx.device(), 0);
    EXPECT_GE(ctx.compute_capability(), 10);
    const cudaDeviceProp& prop = ctx.device_properties();
    EXPECT_EQ(ctx.compute_capability(), prop.major * 10 + prop.minor);
    EXPECT_GT(prop.multiProcessorCount, 0);
}

TEST_F(ContextTest, StreamDefaultsToTheDefaultStreamAndCanBeSet) {
    ckl::Context ctx;
    EXPECT_EQ(ctx.stream(), cudaStream_t{nullptr});
    cudaStream_t s = nullptr;
    CKL_CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    ctx.set_stream(s);
    EXPECT_EQ(ctx.stream(), s);
    ctx.set_stream(nullptr);
    CKL_CUDA_CHECK(cudaStreamDestroy(s));
}

TEST_F(ContextTest, WorkspaceRoundTrips) {
    ckl::Context ctx;
    EXPECT_EQ(ctx.workspace(), nullptr);
    EXPECT_EQ(ctx.workspace_size(), 0u);
    ckl::DeviceBuffer<float> scratch(1024);
    ctx.set_workspace(scratch.data(), scratch.bytes());
    EXPECT_EQ(ctx.workspace(), scratch.data());
    EXPECT_EQ(ctx.workspace_size(), scratch.bytes());
}

TEST_F(ContextTest, VendorHandlesAreCreatedOnFirstUseAndCached) {
    ckl::Context ctx;
    void* blas = ctx.cublas();
    ASSERT_NE(blas, nullptr);
    EXPECT_EQ(ctx.cublas(), blas) << "the handle has to be cached, not recreated per call";
    EXPECT_NE(ctx.cusparse(), nullptr);
    EXPECT_NE(ctx.cusolver(), nullptr);
}

TEST_F(ContextTest, MovedFromContextReportsItselfRatherThanCrashing) {
    ckl::Context a;
    const int device = a.device();
    ckl::Context b(std::move(a));
    EXPECT_EQ(b.device(), device);
    // Reading through the moved from handle has to throw, not dereference null.
    EXPECT_THROW((void)a.device(), ckl::Error);  // NOLINT(bugprone-use-after-move)
}

TEST_F(ContextTest, MoveAssignmentKeepsTheTargetUsable) {
    ckl::Context a;
    ckl::Context b;
    const int device = a.device();
    b = std::move(a);
    EXPECT_EQ(b.device(), device);
    EXPECT_NE(b.cublas(), nullptr);
}

TEST_F(ContextTest, DefaultContextIsShared) {
    ckl::Context& one = ckl::detail::default_context();
    ckl::Context& two = ckl::detail::default_context();
    EXPECT_EQ(&one, &two);
    EXPECT_EQ(ckl::detail::device_compute_capability(), one.compute_capability());
}

TEST_F(ContextTest, RequireArchThrowsAboveTheRunningDevice) {
    // 999 is above anything that exists, so the refusal is unconditional and the
    // test does not depend on the machine it runs on.
    EXPECT_THROW(ckl::detail::require_arch(999, "a fictional path"), ckl::Error);
    EXPECT_NO_THROW(ckl::detail::require_arch(1, "anything at all"));
}

// Two contexts, two streams, one device. Each carries its own stream, so the
// two calls overlap and neither reaches into the other's state.
TEST_F(ContextTest, TwoContextsOnTwoStreams) {
    const std::int64_t n = 256;
    const auto a =
        ckl::random_matrix(static_cast<int>(n), static_cast<int>(n), ckl::test::seed_stream(2601));
    const auto b =
        ckl::random_matrix(static_cast<int>(n), static_cast<int>(n), ckl::test::seed_stream(2602));
    const std::vector<float> c0(elems(n, n), 0.0f);

    cudaStream_t s1 = nullptr;
    cudaStream_t s2 = nullptr;
    CKL_CUDA_CHECK(cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking));
    CKL_CUDA_CHECK(cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking));

    ckl::Context ctx1;
    ckl::Context ctx2;
    ctx1.set_stream(s1);
    ctx2.set_stream(s2);

    ckl::DeviceBuffer<float> da(a.size());
    ckl::DeviceBuffer<float> db(b.size());
    ckl::DeviceBuffer<float> d1(c0.size());
    ckl::DeviceBuffer<float> d2(c0.size());
    da.copy_from_host(a);
    db.copy_from_host(b);
    d1.copy_from_host(c0);
    d2.copy_from_host(c0);

    const float alpha = 1.0f;
    const float beta = 0.0f;
    GemmDesc naive = square_desc(DType::kR32F, n, Algo::kNaive);
    GemmDesc vendor = square_desc(DType::kR32F, n, Algo::kCublas);
    ASSERT_EQ(ckl::gemm(ctx1, naive, &alpha, da.data(), db.data(), &beta, d1.data(), nullptr),
              Status::kSuccess);
    ASSERT_EQ(ckl::gemm(ctx2, vendor, &alpha, da.data(), db.data(), &beta, d2.data(), nullptr),
              Status::kSuccess);
    CKL_CUDA_CHECK(cudaStreamSynchronize(s1));
    CKL_CUDA_CHECK(cudaStreamSynchronize(s2));
    const auto out1 = d1.to_host();
    const auto out2 = d2.to_host();
    CKL_CUDA_CHECK(cudaStreamDestroy(s1));
    CKL_CUDA_CHECK(cudaStreamDestroy(s2));

    const ckl::GemmRef ref = ckl::gemm_reference_scaled(
        a, b, c0, static_cast<int>(n), static_cast<int>(n), static_cast<int>(n), alpha, beta);
    EXPECT_LT(ckl::max_scaled_residual(out1, ref), ckl::tol(static_cast<int>(n)));
    EXPECT_LT(ckl::max_scaled_residual(out2, ref), ckl::tol(static_cast<int>(n)));
}

// ---------------------------------------------------------------------------
// Allocation failure
// ---------------------------------------------------------------------------

class Allocation : public GpuTest {};

// An allocation that cannot succeed has to come back as a status, not as a
// crash and not as a null pointer the next kernel walks off the end of.
TEST_F(Allocation, AbsurdRequestFailsCleanly) {
    // 2^44 floats is 64 TiB: larger than any device and small enough that the
    // byte count does not overflow size_t on the way in.
    constexpr std::size_t kAbsurd = std::size_t{1} << 44;
    try {
        ckl::DeviceBuffer<float> impossible(kAbsurd);
        FAIL() << "a 64 TiB allocation must not report success";
    } catch (const ckl::Error& e) {
        EXPECT_EQ(e.status(), ckl::Status::kAllocFailed) << e.what();
    }
    // The failed allocation leaves the runtime's last error set. Clearing it is
    // part of the contract being tested: the failure travelled out through the
    // status and the exception, not as a poison pill for the next call.
    EXPECT_NE(cudaGetLastError(), cudaSuccess);
    EXPECT_EQ(cudaGetLastError(), cudaSuccess);
}

TEST_F(Allocation, ContextStillWorksAfterAFailedAllocation) {
    constexpr std::size_t kAbsurd = std::size_t{1} << 44;
    EXPECT_THROW(ckl::DeviceBuffer<float> impossible(kAbsurd), ckl::Error);
    cudaGetLastError();

    ckl::Context ctx;
    const GemmDesc d = square_desc(DType::kR32F, 64, Algo::kNaive);
    ckl::DeviceBuffer<float> da(elems(64, 64));
    ckl::DeviceBuffer<float> db(elems(64, 64));
    ckl::DeviceBuffer<float> dc(elems(64, 64));
    da.zero();
    db.zero();
    dc.zero();
    const float alpha = 1.0f;
    const float beta = 0.0f;
    EXPECT_EQ(ckl::gemm(ctx, d, &alpha, da.data(), db.data(), &beta, dc.data(), nullptr),
              Status::kSuccess);
    CKL_CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
}

TEST_F(Allocation, DeviceBufferRejectsAnOversizedTransfer) {
    ckl::DeviceBuffer<float> buf(16);
    const std::vector<float> host(32, 1.0f);
    EXPECT_THROW(buf.copy_from_host(host), std::invalid_argument);
    std::vector<float> out(32);
    EXPECT_THROW(buf.copy_to_host(out.data(), 32), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Names
// ---------------------------------------------------------------------------

// These strings end up in benchmark rows and in the chosen report, so they are
// part of the interface: never null, never empty, never two rungs sharing one.
TEST(Names, EveryEnumeratorHasAUniqueName) {
    const std::vector<Algo> algos = {
        Algo::kAuto,       Algo::kNaive,    Algo::kTiled,   Algo::kRegister, Algo::kCpAsync,
        Algo::kWmmaFp16,   Algo::kWmmaBf16, Algo::kMmaPtx,  Algo::kMmaLdm,   Algo::kMmaOpt,
        Algo::kTileFamily, Algo::kSplitK,   Algo::kStreamK, Algo::kCutlass,  Algo::kCublas};
    std::vector<std::string> seen;
    for (Algo a : algos) {
        const char* name = ckl::algo_name(a);
        ASSERT_NE(name, nullptr);
        EXPECT_NE(std::string(name), "");
        for (const auto& other : seen) {
            EXPECT_NE(other, std::string(name)) << "two algorithms share the name " << name;
        }
        seen.emplace_back(name);
    }

    for (DType t : {DType::kR32F, DType::kR16F, DType::kR16BF}) {
        ASSERT_NE(ckl::dtype_name(t), nullptr);
        EXPECT_GT(ckl::dtype_size(t), 0u);
    }
    for (Op o : {Op::kN, Op::kT, Op::kC}) {
        ASSERT_NE(ckl::op_name(o), nullptr);
    }
    for (Layout l : {Layout::kRowMajor, Layout::kColMajor}) {
        ASSERT_NE(ckl::layout_name(l), nullptr);
    }
    const std::vector<Status> statuses = {Status::kSuccess,         Status::kNotInitialized,
                                          Status::kInvalidValue,    Status::kArchMismatch,
                                          Status::kNotSupported,    Status::kAllocFailed,
                                          Status::kExecutionFailed, Status::kInternal};
    for (Status s : statuses) {
        ASSERT_NE(ckl::status_string(s), nullptr);
        EXPECT_NE(std::string(ckl::status_string(s)), "");
    }
}

TEST(Names, ElementSizesMatchTheFormats) {
    EXPECT_EQ(ckl::dtype_size(DType::kR32F), 4u);
    EXPECT_EQ(ckl::dtype_size(DType::kR16F), 2u);
    EXPECT_EQ(ckl::dtype_size(DType::kR16BF), 2u);
}

// ---------------------------------------------------------------------------
// The Section 9.6 tile family heuristic
// ---------------------------------------------------------------------------
//
// Two things are checked here, and they are different in kind. The first is the
// rule itself: for every FP16 shape the dispatcher has to land in the family,
// name a tile, and escalate exactly when the grid cannot fill one wave. That is
// arithmetic on the public accessors, so the test recomputes it rather than
// asserting a golden algorithm name that would only be right on this GPU.
//
// The second is that a committed sweep actually overrides the quantization only
// fallback. The tree ships no sweep, because a sweep is owner work at locked
// clocks and an invented one would be a measurement claim; the fixture CSV the
// build hands over through CKL_TILE_SWEEP_FIXTURE carries obviously synthetic
// throughput numbers and exists only to prove the override happens.

// Blocks of one output tile the grid needs, divided by the blocks the machine
// can hold. This is the waves term of the heuristic, recomputed from the public
// accessors so the test does not repeat the dispatcher's arithmetic by copying
// it.
double family_waves(const ckl::Context& ctx, const GemmDesc& d, int tile_index) {
    const ckl::GemmTile t = ckl::gemm_tile_family_shape(tile_index);
    const int bps = ckl::gemm_tile_family_blocks_per_sm(tile_index);
    EXPECT_GT(bps, 0);
    const double tiles =
        std::ceil(static_cast<double>(d.m) / t.m) * std::ceil(static_cast<double>(d.n) / t.n);
    return tiles / (static_cast<double>(ctx.device_properties().multiProcessorCount) *
                    static_cast<double>(bps));
}

double quantization(double waves) {
    return waves > 0.0 ? waves / std::ceil(waves) : 0.0;
}

// The 128 by 128 by 32 shape, found by its extents so a reordered roster cannot
// make this test agree with the dispatcher for the wrong reason.
int reference_tile_index() {
    for (int i = 0; i < ckl::gemm_tile_family_count(); ++i) {
        const ckl::GemmTile t = ckl::gemm_tile_family_shape(i);
        if (t.m == 128 && t.n == 128 && t.k == 32) {
            return i;
        }
    }
    return -1;
}

// The band boundary of the untuned guard: at or above this many waves of the
// reference tile, the untuned path keeps that tile instead of asking which shape
// quantizes best.
constexpr double kUntunedGuardWaves = 2.0;

// Asserts that a tile is the one the quantization only rule would have picked:
// nothing scores higher, and nothing of equal score is larger.
void expect_quantization_rule_choice(const ckl::Context& ctx, const GemmDesc& d, int chosen_index) {
    const ckl::GemmTile chosen_tile = ckl::gemm_tile_family_shape(chosen_index);
    const double chosen = quantization(family_waves(ctx, d, chosen_index));
    const long long chosen_area =
        static_cast<long long>(chosen_tile.m) * static_cast<long long>(chosen_tile.n);
    for (int i = 0; i < ckl::gemm_tile_family_count(); ++i) {
        const double score = quantization(family_waves(ctx, d, i));
        const ckl::GemmTile t = ckl::gemm_tile_family_shape(i);
        const long long area = static_cast<long long>(t.m) * static_cast<long long>(t.n);
        EXPECT_LE(score, chosen + 1e-12) << "tile " << i << " quantizes better than the choice";
        if (std::fabs(score - chosen) < 1e-12) {
            EXPECT_LE(area, chosen_area) << "tile " << i << " ties on quantization and is larger";
        }
    }
}

void point_at_sweep(const char* path) {
#if defined(_WIN32)
    _putenv_s("CKL_TILE_SWEEP_CSV", path);
#else
    setenv("CKL_TILE_SWEEP_CSV", path, 1);
#endif
}

struct FamilyCase {
    const char* label;
    std::int64_t m;
    std::int64_t n;
    std::int64_t k;
};

class DispatchFamily : public GpuTestWithParam<FamilyCase> {};

std::string family_case_name(const ::testing::TestParamInfo<FamilyCase>& info) {
    return info.param.label;
}

TEST_P(DispatchFamily, LandsInTheFamilyAndEscalatesOnTheWaveRule) {
    // No sweep: this is the quantization only path, which is what the tree
    // ships until the owner commits one.
    point_at_sweep("this/path/does/not/exist.csv");
    const FamilyCase fc = GetParam();
    ckl::Context ctx;
    GemmDesc d;
    d.m = fc.m;
    d.n = fc.n;
    d.k = fc.k;
    d.dt_a = DType::kR16F;
    d.dt_b = DType::kR16F;
    d.dt_c = DType::kR32F;
    d.lda = fc.k;
    d.ldb = fc.n;
    d.ldc = fc.n;
    d.algo = Algo::kAuto;

    const ckl::GemmPlan plan = ckl::gemm_plan(ctx, d);
    EXPECT_EQ(ckl::gemm_query(ctx, d), plan.algo) << "the short answer and the plan disagree";
    EXPECT_FALSE(plan.tuned) << "there is no committed sweep, so nothing can be tuned";
    ASSERT_GE(plan.tile_index, 0) << "an FP16 shape has to name a tile of the family";
    ASSERT_LT(plan.tile_index, ckl::gemm_tile_family_count());
    EXPECT_EQ(plan.tile.m, ckl::gemm_tile_family_shape(plan.tile_index).m);

    // The untuned path has two bands, and which one applies is decided by the
    // reference tile's wave count, so the test asks the same question the
    // dispatcher does rather than remembering an answer per shape.
    const int ref_tile = reference_tile_index();
    ASSERT_GE(ref_tile, 0) << "the family has to contain the 128 by 128 by 32 reference tile";
    if (family_waves(ctx, d, ref_tile) >= kUntunedGuardWaves) {
        EXPECT_EQ(plan.tile_index, ref_tile)
            << "above two waves the untuned guard keeps the reference tile, but the plan chose "
            << plan.tile.m << "x" << plan.tile.n << "x" << plan.tile.k;
    } else {
        expect_quantization_rule_choice(ctx, d, plan.tile_index);
    }

    // The escalation rule, exactly as Section 9.6 states it.
    const double waves = family_waves(ctx, d, plan.tile_index);
    if (waves >= 1.0) {
        EXPECT_EQ(plan.algo, Algo::kTileFamily)
            << "a grid that fills a wave has no reason to split K";
        EXPECT_EQ(plan.splits, 1);
        EXPECT_EQ(ckl::gemm_workspace_size(ctx, d), 0u);
    } else {
        EXPECT_TRUE(plan.algo == Algo::kSplitK || plan.algo == Algo::kStreamK)
            << "waves is " << waves << " but the plan says " << ckl::algo_name(plan.algo);
    }
    if (plan.algo == Algo::kSplitK) {
        EXPECT_GE(plan.splits, 2) << "split-K with one slice is not split-K";
        EXPECT_EQ(
            ckl::gemm_workspace_size(ctx, d),
            ckl::gemm_split_k_workspace_size(static_cast<int>(d.m), static_cast<int>(d.n),
                                             static_cast<int>(d.k), plan.splits, plan.tile_index));
    }
    if (plan.algo == Algo::kStreamK) {
        EXPECT_EQ(ckl::gemm_workspace_size(ctx, d),
                  ckl::gemm_stream_k_workspace_size(static_cast<int>(d.m), static_cast<int>(d.n),
                                                    static_cast<int>(d.k), plan.tile_index));
    }

    // And it has to compute the right answer, or picking the right rung means
    // nothing.
    const Operand a = make_operand(DType::kR16F, elems(fc.m, fc.k), 2701);
    const Operand b = make_operand(DType::kR16F, elems(fc.k, fc.n), 2702);
    const std::vector<float> c0(elems(fc.m, fc.n), 0.25f);
    const DescRun got = run_desc(ctx, d, a, b, c0, 1.25f, 0.5f);
    ASSERT_EQ(got.status, Status::kSuccess);
    EXPECT_EQ(got.chosen, plan.algo) << "chosen has to agree with the plan";

    GemmDesc vendor = d;
    vendor.algo = Algo::kCublas;
    const DescRun ref = run_desc(ctx, vendor, a, b, c0, 1.25f, 0.5f);
    ASSERT_EQ(ref.status, Status::kSuccess);
    const std::vector<double> ref_d(ref.c.begin(), ref.c.end());
    EXPECT_LT(ckl::relative_frobenius_error(got.c, ref_d), 1e-5);
}

INSTANTIATE_TEST_SUITE_P(Gemm, DispatchFamily,
                         ::testing::Values(FamilyCase{"fp16_64", 64, 64, 64},
                                           FamilyCase{"fp16_100", 100, 100, 100},
                                           FamilyCase{"fp16_128", 128, 128, 128},
                                           FamilyCase{"fp16_192", 192, 192, 192},
                                           FamilyCase{"fp16_1024", 1024, 1024, 1024},
                                           FamilyCase{"fp16_2048", 2048, 2048, 2048},
                                           // Deep K on a small output is the case
                                           // split-K exists for.
                                           FamilyCase{"fp16_deep_k", 128, 128, 8192},
                                           // Skinny, and unaligned in every
                                           // dimension: the two shapes that used
                                           // to fall through to the scalar path.
                                           FamilyCase{"fp16_skinny", 4096, 64, 2048},
                                           FamilyCase{"fp16_unaligned", 1000, 1000, 1000}),
                         family_case_name);

class DispatchHeuristic : public GpuTest {};

GemmDesc fp16_square(std::int64_t n) {
    GemmDesc d;
    d.m = n;
    d.n = n;
    d.k = n;
    d.dt_a = DType::kR16F;
    d.dt_b = DType::kR16F;
    d.dt_c = DType::kR32F;
    d.lda = n;
    d.ldb = n;
    d.ldc = n;
    return d;
}

// The untuned guard, at the shape it exists for. Wave quantization on its own
// prefers the smallest tile in the family at 4096 cubed, which stages far more
// bytes per multiply add than the reference shape for a fraction of a percent of
// quantization. Without a sweep to settle that, the untuned path keeps the tile
// the ladder has always run rather than betting the flagship shape on an
// unmeasured hypothesis.
TEST_F(DispatchHeuristic, UntunedGuardKeepsTheReferenceTileWhenTheGridIsDeep) {
    point_at_sweep("this/path/does/not/exist.csv");
    ckl::Context ctx;
    const GemmDesc d = fp16_square(4096);
    const int ref = reference_tile_index();
    ASSERT_GE(ref, 0);

    // The precondition the guard keys on. If a future device made 4096 cubed
    // shallower than two waves this test would be checking the wrong band, so it
    // says so rather than passing quietly.
    ASSERT_GE(family_waves(ctx, d, ref), kUntunedGuardWaves)
        << "4096 cubed is no longer two waves deep on this device";

    const ckl::GemmPlan plan = ckl::gemm_plan(ctx, d);
    EXPECT_EQ(plan.tile.m, 128);
    EXPECT_EQ(plan.tile.n, 128);
    EXPECT_EQ(plan.tile.k, 32);
    EXPECT_EQ(plan.tile_index, ref);
    EXPECT_EQ(plan.algo, Algo::kTileFamily) << "a grid this deep fills plenty of waves";
    EXPECT_EQ(plan.splits, 1);
    EXPECT_FALSE(plan.tuned) << "a guard is not a measurement";

    // The guard is a band, not a blanket: the smallest tile really does quantize
    // better here, which is exactly why the guard has to overrule it.
    bool a_smaller_tile_quantizes_better = false;
    const double ref_score = quantization(family_waves(ctx, d, ref));
    for (int i = 0; i < ckl::gemm_tile_family_count(); ++i) {
        const ckl::GemmTile t = ckl::gemm_tile_family_shape(i);
        if (static_cast<long long>(t.m) * t.n < 128LL * 128 &&
            quantization(family_waves(ctx, d, i)) > ref_score + 1e-12) {
            a_smaller_tile_quantizes_better = true;
        }
    }
    EXPECT_TRUE(a_smaller_tile_quantizes_better)
        << "the guard is overruling nothing here, so it is not being tested";
}

// Below the band the rule is unchanged, and the escalation below one wave is
// unchanged with it.
TEST_F(DispatchHeuristic, UntunedShallowGridsStillFollowTheQuantizationRule) {
    point_at_sweep("this/path/does/not/exist.csv");
    ckl::Context ctx;
    const GemmDesc d = fp16_square(512);
    const int ref = reference_tile_index();
    ASSERT_GE(ref, 0);
    ASSERT_LT(family_waves(ctx, d, ref), kUntunedGuardWaves)
        << "512 cubed has to sit below the guard band for this test to mean anything";

    const ckl::GemmPlan plan = ckl::gemm_plan(ctx, d);
    ASSERT_GE(plan.tile_index, 0);
    EXPECT_FALSE(plan.tuned);
    expect_quantization_rule_choice(ctx, d, plan.tile_index);

    // And the escalation rule still reads off the chosen tile, not the reference
    // one: 512 cubed does not fill a wave for any shape in the family.
    const double waves = family_waves(ctx, d, plan.tile_index);
    if (waves < 1.0) {
        EXPECT_TRUE(plan.algo == Algo::kSplitK || plan.algo == Algo::kStreamK)
            << "waves is " << waves << " but the plan says " << ckl::algo_name(plan.algo);
    } else {
        EXPECT_EQ(plan.algo, Algo::kTileFamily);
    }
}

// waves below one is the whole reason split-K and stream-K exist, so the suite
// has to contain at least one shape that triggers each. If a future change made
// every shape fill a wave, this fails rather than quietly leaving both paths
// unreached.
TEST_F(DispatchHeuristic, BothEscalationsAreReachable) {
    point_at_sweep("this/path/does/not/exist.csv");
    ckl::Context ctx;
    bool saw_split = false;
    bool saw_stream = false;
    bool saw_plain = false;
    const std::vector<std::array<std::int64_t, 3>> shapes = {
        {64, 64, 64},      {128, 128, 128},    {128, 128, 8192},
        {256, 256, 16384}, {2048, 2048, 2048}, {4096, 4096, 4096}};
    for (const auto& s : shapes) {
        GemmDesc d;
        d.m = s[0];
        d.n = s[1];
        d.k = s[2];
        d.dt_a = DType::kR16F;
        d.dt_b = DType::kR16F;
        d.dt_c = DType::kR32F;
        d.lda = s[2];
        d.ldb = s[1];
        d.ldc = s[1];
        const ckl::GemmPlan p = ckl::gemm_plan(ctx, d);
        saw_split = saw_split || p.algo == Algo::kSplitK;
        saw_stream = saw_stream || p.algo == Algo::kStreamK;
        saw_plain = saw_plain || p.algo == Algo::kTileFamily;
    }
    EXPECT_TRUE(saw_plain) << "no shape in the list filled a wave";
    EXPECT_TRUE(saw_split) << "no shape in the list escalated to split-K";
    EXPECT_TRUE(saw_stream) << "no shape in the list escalated to stream-K";
}

// The five percent throughput filter. The fixture leaves exactly one tile above
// the bar at each swept shape, so the answer is fixed by the data and not by the
// occupancy of the machine the test runs on.
TEST_F(DispatchHeuristic, ACommittedSweepOverridesQuantization) {
    ckl::Context ctx;
    auto plan_at = [&ctx](std::int64_t n) {
        GemmDesc d;
        d.m = n;
        d.n = n;
        d.k = n;
        d.dt_a = DType::kR16F;
        d.dt_b = DType::kR16F;
        d.dt_c = DType::kR32F;
        d.lda = n;
        d.ldb = n;
        d.ldc = n;
        return ckl::gemm_plan(ctx, d);
    };

    point_at_sweep(CKL_TILE_SWEEP_FIXTURE);
    {
        // Nearest swept shape is 4096 cubed, where the fixture puts only
        // tile_128x128x32 within five percent of the best.
        const ckl::GemmPlan p = plan_at(4000);
        EXPECT_TRUE(p.tuned) << "a readable sweep has to be reported as one";
        EXPECT_EQ(p.tile.m, 128);
        EXPECT_EQ(p.tile.n, 128);
        EXPECT_EQ(p.tile.k, 32);
    }
    {
        // Nearest swept shape is 1024 cubed, where the winner is a tile neither
        // the quantization rule nor the untuned guard would ever reach, so this
        // is the case that proves the sweep is what decided. It also checks that
        // the nearest shape lookup is doing something, since the answer differs
        // from the 4096 row above.
        const ckl::GemmPlan p = plan_at(1100);
        EXPECT_TRUE(p.tuned);
        EXPECT_EQ(p.tile.m, 256);
        EXPECT_EQ(p.tile.n, 128);
        EXPECT_EQ(p.tile.k, 32);
    }

    // Point it back at nothing and the answer has to go untuned again, which is
    // also the check that the cache follows the path rather than latching the
    // first file it ever saw.
    point_at_sweep("this/path/does/not/exist.csv");
    EXPECT_FALSE(plan_at(4000).tuned);
}

// An explicitly named family algorithm is still never rerouted, and the plan
// fills in the tile the caller did not name.
TEST_F(DispatchHeuristic, NamedFamilyAlgorithmsRunOnAnyFp16Shape) {
    point_at_sweep("this/path/does/not/exist.csv");
    ckl::Context ctx;
    const std::int64_t m = 129;
    const std::int64_t n = 257;
    const std::int64_t k = 193;
    const Operand a = make_operand(DType::kR16F, elems(m, k), 2711);
    const Operand b = make_operand(DType::kR16F, elems(k, n), 2712);
    const std::vector<float> c0(elems(m, n), -1.5f);

    GemmDesc base;
    base.m = m;
    base.n = n;
    base.k = k;
    base.dt_a = DType::kR16F;
    base.dt_b = DType::kR16F;
    base.dt_c = DType::kR32F;
    base.lda = k;
    base.ldb = n;
    base.ldc = n;

    GemmDesc vendor = base;
    vendor.algo = Algo::kCublas;
    const DescRun ref = run_desc(ctx, vendor, a, b, c0, 2.5f, -0.75f);
    ASSERT_EQ(ref.status, Status::kSuccess);
    const std::vector<double> ref_d(ref.c.begin(), ref.c.end());

    for (Algo named : {Algo::kTileFamily, Algo::kSplitK, Algo::kStreamK}) {
        SCOPED_TRACE(ckl::algo_name(named));
        GemmDesc d = base;
        d.algo = named;
        const DescRun got = run_desc(ctx, d, a, b, c0, 2.5f, -0.75f);
        ASSERT_EQ(got.status, Status::kSuccess);
        EXPECT_EQ(got.chosen, named) << "a named algorithm is never rerouted";
        EXPECT_LT(ckl::relative_frobenius_error(got.c, ref_d), 1e-5);
    }
}

// A workspace the Context already owns keeps the allocation out of the call, and
// the answer has to be the same either way.
TEST_F(DispatchHeuristic, AContextWorkspaceIsUsedWhenItIsBigEnough) {
    point_at_sweep("this/path/does/not/exist.csv");
    ckl::Context ctx;
    const std::int64_t n = 256;
    const std::int64_t k = 8192;
    GemmDesc d;
    d.m = n;
    d.n = n;
    d.k = k;
    d.dt_a = DType::kR16F;
    d.dt_b = DType::kR16F;
    d.dt_c = DType::kR32F;
    d.lda = k;
    d.ldb = n;
    d.ldc = n;

    const ckl::GemmPlan plan = ckl::gemm_plan(ctx, d);
    const std::size_t need = ckl::gemm_workspace_size(ctx, d);
    ASSERT_GT(need, 0u) << "this shape was chosen because it escalates and needs scratch, but the "
                        << "plan says " << ckl::algo_name(plan.algo);

    const Operand a = make_operand(DType::kR16F, elems(n, k), 2721);
    const Operand b = make_operand(DType::kR16F, elems(k, n), 2722);
    const std::vector<float> c0(elems(n, n), 0.5f);

    const DescRun without = run_desc(ctx, d, a, b, c0, 1.25f, 0.5f);
    ASSERT_EQ(without.status, Status::kSuccess);

    ckl::DeviceBuffer<std::uint8_t> scratch(need);
    ctx.set_workspace(scratch.data(), scratch.bytes());
    const DescRun with = run_desc(ctx, d, a, b, c0, 1.25f, 0.5f);
    ctx.set_workspace(nullptr, 0);
    ASSERT_EQ(with.status, Status::kSuccess);
    EXPECT_EQ(without.c, with.c) << "the workspace changed the answer, which it must not";
}

}  // namespace
