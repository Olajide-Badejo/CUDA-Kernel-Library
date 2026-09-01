// FFT correctness: the round trip at every size in scope, the forward transform
// against a long double CPU DFT where an O(N^2) reference is affordable, the
// transpose study, the 2D driver and the real transforms.
//
// The tolerance is derived rather than picked. Error grows with the number of
// stages, so the bound is rms_rel(N) <= 8 log2(N) FLT_EPSILON, which is 2.3e-5 at
// 2^24 and 9.5e-6 at 2^10. A flat tolerance would either pass a broken 2^24 or
// fail a correct one, which is the same argument ckl::tol(k) makes for GEMM.
// ckl::fft_tolerance is the one implementation of it and every assertion here
// goes through it.
//
// The round trip test prints its table. The numbers in it are a property of the
// arithmetic and not of the clock, so unlike a benchmark they belong in the test
// output, and the report cites them.
//
// Three things are asserted that a "does it compute an FFT" suite would not.
// Every rung is checked at every size it claims to support, so a rung that
// quietly agreed with cuFFT by falling back to cuFFT would fail on the chosen
// out-param rather than pass on a right answer from the wrong kernel. Every
// refusal is checked to carry a reason, so a size the shared resident kernel
// cannot hold reports the shared memory arithmetic instead of a bare status. And
// the fast twiddle rung is measured beside the default one, because a rung whose
// whole cost is accuracy has to have that cost on the record.

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include <gtest/gtest.h>

#include "ckl/ckl.h"
#include "ckl/context.hpp"
#include "ckl/cuda_check.hpp"
#include "ckl/device_buffer.hpp"
#include "ckl/fft.hpp"
#include "gpu_environment.hpp"
#include "reference.hpp"

namespace {

using ckl::test::GpuTest;
using ckl::test::GpuTestWithParam;

constexpr int kMinLog2 = 10;
constexpr int kMaxLog2 = 24;

std::vector<float2> random_signal(int count, std::uint64_t seed) {
    const std::vector<float> flat = ckl::random_matrix(2 * count, 1, seed);
    std::vector<float2> out(static_cast<std::size_t>(count));
    for (int i = 0; i < count; ++i) {
        out[static_cast<std::size_t>(i)] = make_float2(flat[static_cast<std::size_t>(2 * i)],
                                                       flat[static_cast<std::size_t>(2 * i) + 1]);
    }
    return out;
}

// sqrt(sum |got - want|^2 / sum |want|^2), the measure the tolerance model is
// stated in.
double relative_rms(const std::vector<float2>& got, const std::vector<float2>& want) {
    double num = 0.0;
    double den = 0.0;
    for (std::size_t i = 0; i < want.size(); ++i) {
        const double dx = static_cast<double>(got[i].x) - static_cast<double>(want[i].x);
        const double dy = static_cast<double>(got[i].y) - static_cast<double>(want[i].y);
        num += dx * dx + dy * dy;
        den += static_cast<double>(want[i].x) * static_cast<double>(want[i].x) +
               static_cast<double>(want[i].y) * static_cast<double>(want[i].y);
    }
    return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
}

double relative_rms_real(const std::vector<float>& got, const std::vector<float>& want) {
    double num = 0.0;
    double den = 0.0;
    for (std::size_t i = 0; i < want.size(); ++i) {
        const double d = static_cast<double>(got[i]) - static_cast<double>(want[i]);
        num += d * d;
        den += static_cast<double>(want[i]) * static_cast<double>(want[i]);
    }
    return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
}

// The O(N^2) reference, in long double. Affordable to 2^12 and no further, which
// is exactly the range the spec asks it to cover.
std::vector<float2> cpu_dft(const std::vector<float2>& in, bool forward) {
    const std::size_t n = in.size();
    const long double sign = forward ? -1.0L : 1.0L;
    const long double two_pi = 6.283185307179586476925286766559L;
    std::vector<float2> out(n);
    for (std::size_t k = 0; k < n; ++k) {
        long double re = 0.0L;
        long double im = 0.0L;
        for (std::size_t j = 0; j < n; ++j) {
            const long double angle = sign * two_pi * static_cast<long double>(k) *
                                      static_cast<long double>(j) / static_cast<long double>(n);
            const long double c = std::cos(angle);
            const long double s = std::sin(angle);
            const long double xr = static_cast<long double>(in[j].x);
            const long double xi = static_cast<long double>(in[j].y);
            re += xr * c - xi * s;
            im += xr * s + xi * c;
        }
        out[k] = make_float2(static_cast<float>(re), static_cast<float>(im));
    }
    return out;
}

const char* name_of(ckl::FftAlgo a) {
    return ckl::fft_algo_name(a);
}

// Named rather than written as a lambda at the instantiation site: GoogleTest's
// generated code has a parameter called info, and a lambda parameter of the same
// name shadows it.
std::string size_param_name(const ::testing::TestParamInfo<int>& info) {
    return "n" + std::to_string(info.param);
}

std::string packed_param_name(const ::testing::TestParamInfo<int>& info) {
    return "packed" + std::to_string(info.param);
}

std::string transpose_param_name(const ::testing::TestParamInfo<ckl::FftTranspose>& info) {
    return std::string(ckl::fft_transpose_name(info.param)).substr(1);
}

std::vector<ckl::FftAlgo> all_algos() {
    return {ckl::FftAlgo::kRadix2Global, ckl::FftAlgo::kSharedResident, ckl::FftAlgo::kRadix4Global,
            ckl::FftAlgo::kRadix8Global, ckl::FftAlgo::kFourStep,       ckl::FftAlgo::kCufft};
}

// One forward then one inverse then the 1/n the inverse deliberately does not
// apply, compared with the input.
double round_trip(ckl::FftPlan& plan, ckl::FftAlgo algo, const std::vector<float2>& host_in,
                  ckl::DeviceBuffer<float2>& da, ckl::DeviceBuffer<float2>& db,
                  ckl::FftAlgo* chosen_out) {
    const int n = plan.n();
    const int batch = plan.batch();
    da.copy_from_host(host_in);
    ckl::FftAlgo chosen = ckl::FftAlgo::kAuto;
    ckl::Status st =
        ckl::fft(plan, algo, ckl::FftDirection::kForward, da.data(), db.data(), &chosen);
    EXPECT_EQ(st, ckl::Status::kSuccess) << name_of(algo);
    EXPECT_EQ(chosen, algo) << "a variant that runs a path its name does not claim is a defect";
    st = ckl::fft(plan, algo, ckl::FftDirection::kInverse, db.data(), da.data(), &chosen);
    EXPECT_EQ(st, ckl::Status::kSuccess) << name_of(algo);
    ckl::fft_scale(da.data(), static_cast<long long>(n) * batch, 1.0f / static_cast<float>(n));
    CKL_CUDA_CHECK(cudaDeviceSynchronize());
    if (chosen_out != nullptr) {
        *chosen_out = chosen;
    }
    return relative_rms(da.to_host(), host_in);
}

// ---------------------------------------------------------------------------
// The round trip, at every size from 2^10 to 2^24
// ---------------------------------------------------------------------------

class FftRoundTrip : public GpuTest {};

TEST_F(FftRoundTrip, EverySizeAndEveryRungStayInsideTheDerivedBound) {
    std::printf("\n%-8s %-10s %-18s %12s %12s %8s\n", "log2(n)", "n", "variant", "rms_rel", "bound",
                "margin");
    for (int bits = kMinLog2; bits <= kMaxLog2; ++bits) {
        const int n = 1 << bits;
        // The batch fills the machine at the small sizes without making the large
        // ones enormous: 48 SMs want at least 48 blocks, and one block runs one
        // shared resident transform.
        const int batch = n <= 4096 ? 64 : 1;
        ckl::FftPlanOptions opt;
        opt.batch = batch;
        ckl::FftPlan plan(n, opt);
        const double bound = ckl::fft_tolerance(n);
        const std::vector<float2> host_in =
            random_signal(n * batch, ckl::test::seed_stream(static_cast<std::uint64_t>(bits)));
        ckl::DeviceBuffer<float2> da(host_in.size());
        ckl::DeviceBuffer<float2> db(host_in.size());

        for (ckl::FftAlgo algo : all_algos()) {
            if (!plan.supports(algo)) {
                EXPECT_NE(std::string(plan.refusal(algo)), std::string())
                    << "a refusal has to say why";
                continue;
            }
            const double err = round_trip(plan, algo, host_in, da, db, nullptr);
            std::printf("%-8d %-10d %-18s %12.4e %12.4e %8.3f\n", bits, n, name_of(algo), err,
                        bound, bound > 0.0 ? err / bound : 0.0);
            EXPECT_LE(err, bound) << "n = " << n << ", variant " << name_of(algo);
        }
    }
    std::printf("\n");
}

// ---------------------------------------------------------------------------
// The forward transform against a long double CPU DFT
// ---------------------------------------------------------------------------

class FftAgainstCpu : public GpuTestWithParam<int> {};

TEST_P(FftAgainstCpu, ForwardMatchesALongDoubleDft) {
    const int n = GetParam();
    ckl::FftPlan plan(n);
    const std::vector<float2> host_in =
        random_signal(n, ckl::test::seed_stream(static_cast<std::uint64_t>(n) + 7000));
    const std::vector<float2> want = cpu_dft(host_in, true);
    const double bound = ckl::fft_tolerance(n);

    ckl::DeviceBuffer<float2> da(host_in.size());
    ckl::DeviceBuffer<float2> db(host_in.size());
    da.copy_from_host(host_in);
    for (ckl::FftAlgo algo : all_algos()) {
        if (!plan.supports(algo)) {
            continue;
        }
        ckl::FftAlgo chosen = ckl::FftAlgo::kAuto;
        ASSERT_EQ(ckl::fft(plan, algo, ckl::FftDirection::kForward, da.data(), db.data(), &chosen),
                  ckl::Status::kSuccess)
            << name_of(algo);
        ASSERT_EQ(chosen, algo);
        CKL_CUDA_CHECK(cudaDeviceSynchronize());
        const double err = relative_rms(db.to_host(), want);
        RecordProperty(std::string("rms_") + name_of(algo), ckl::test::sci(err));
        EXPECT_LE(err, bound) << "n = " << n << ", variant " << name_of(algo);
    }
}

TEST_P(FftAgainstCpu, InverseMatchesALongDoubleDft) {
    const int n = GetParam();
    ckl::FftPlan plan(n);
    const std::vector<float2> host_in =
        random_signal(n, ckl::test::seed_stream(static_cast<std::uint64_t>(n) + 9100));
    const std::vector<float2> want = cpu_dft(host_in, false);
    const double bound = ckl::fft_tolerance(n);

    ckl::DeviceBuffer<float2> da(host_in.size());
    ckl::DeviceBuffer<float2> db(host_in.size());
    da.copy_from_host(host_in);
    for (ckl::FftAlgo algo : all_algos()) {
        if (!plan.supports(algo)) {
            continue;
        }
        ASSERT_EQ(ckl::fft(plan, algo, ckl::FftDirection::kInverse, da.data(), db.data(), nullptr),
                  ckl::Status::kSuccess)
            << name_of(algo);
        CKL_CUDA_CHECK(cudaDeviceSynchronize());
        EXPECT_LE(relative_rms(db.to_host(), want), bound) << "n = " << n << ", " << name_of(algo);
    }
}

INSTANTIATE_TEST_SUITE_P(SmallEnoughForOnSquared, FftAgainstCpu,
                         ::testing::Values(1024, 2048, 4096), size_param_name);

// ---------------------------------------------------------------------------
// Batching, which is what makes a small transform measurable at all
// ---------------------------------------------------------------------------

class FftBatch : public GpuTest {};

TEST_F(FftBatch, EveryTransformInABatchIsIndependent) {
    constexpr int kN = 1024;
    constexpr int kBatch = 96;  // twice the 48 SMs, so every SM has work
    ckl::FftPlanOptions opt;
    opt.batch = kBatch;
    ckl::FftPlan plan(kN, opt);
    ASSERT_EQ(plan.batch(), kBatch);

    const std::vector<float2> host_in = random_signal(kN * kBatch, ckl::test::seed_stream(31));
    ckl::DeviceBuffer<float2> da(host_in.size());
    ckl::DeviceBuffer<float2> db(host_in.size());
    da.copy_from_host(host_in);
    ASSERT_EQ(ckl::fft(plan, ckl::FftAlgo::kSharedResident, ckl::FftDirection::kForward, da.data(),
                       db.data(), nullptr),
              ckl::Status::kSuccess);
    CKL_CUDA_CHECK(cudaDeviceSynchronize());
    const std::vector<float2> got = db.to_host();

    // Two batch entries transformed on their own have to match their slice of the
    // batched answer, or the batch index arithmetic is wrong in a way a single
    // transform would never show.
    ckl::FftPlan single(kN);
    for (int b : {0, kBatch - 1, kBatch / 3}) {
        const std::vector<float2> slice(host_in.begin() + static_cast<std::ptrdiff_t>(b) * kN,
                                        host_in.begin() + static_cast<std::ptrdiff_t>(b + 1) * kN);
        ckl::DeviceBuffer<float2> sa(slice.size());
        ckl::DeviceBuffer<float2> sb(slice.size());
        sa.copy_from_host(slice);
        ASSERT_EQ(ckl::fft(single, ckl::FftAlgo::kSharedResident, ckl::FftDirection::kForward,
                           sa.data(), sb.data(), nullptr),
                  ckl::Status::kSuccess);
        CKL_CUDA_CHECK(cudaDeviceSynchronize());
        const std::vector<float2> want = sb.to_host();
        const std::vector<float2> mine(got.begin() + static_cast<std::ptrdiff_t>(b) * kN,
                                       got.begin() + static_cast<std::ptrdiff_t>(b + 1) * kN);
        EXPECT_LE(relative_rms(mine, want), ckl::fft_tolerance(kN)) << "batch entry " << b;
    }
}

// ---------------------------------------------------------------------------
// The fast twiddle rung, measured beside the default
// ---------------------------------------------------------------------------

class FftTwiddleRung : public GpuTest {};

TEST_F(FftTwiddleRung, SincosfCostsAccuracyAndTheTestSaysHowMuch) {
    constexpr int kN = 1 << 20;
    const std::vector<float2> host_in = random_signal(kN, ckl::test::seed_stream(77));
    ckl::DeviceBuffer<float2> da(host_in.size());
    ckl::DeviceBuffer<float2> db(host_in.size());

    ckl::FftPlan exact(kN);
    const double exact_err = round_trip(exact, ckl::FftAlgo::kFourStep, host_in, da, db, nullptr);

    ckl::FftPlanOptions fast_opt;
    fast_opt.fast_twiddles = true;
    ckl::FftPlan fast(kN, fast_opt);
    ASSERT_TRUE(fast.fast_twiddles());
    const double fast_err = round_trip(fast, ckl::FftAlgo::kFourStep, host_in, da, db, nullptr);

    std::printf("fast twiddles at 2^20: double table %.4e, __sincosf table %.4e, bound %.4e\n",
                exact_err, fast_err, ckl::fft_tolerance(kN));
    RecordProperty("rms_double_table", ckl::test::sci(exact_err));
    RecordProperty("rms_sincosf_table", ckl::test::sci(fast_err));

    // Both stay inside the bound at this size; the rung is a speed hypothesis and
    // an accuracy cost, and this is the cost on the record.
    EXPECT_LE(exact_err, ckl::fft_tolerance(kN));
    EXPECT_LE(fast_err, ckl::fft_tolerance(kN));
    EXPECT_GT(fast_err, exact_err) << "if __sincosf were as accurate as a rounded double table "
                                      "there would be no reason to keep two paths";
}

// ---------------------------------------------------------------------------
// The transpose study
// ---------------------------------------------------------------------------

class FftTransposeVariant : public GpuTestWithParam<ckl::FftTranspose> {};

TEST_P(FftTransposeVariant, MatchesAHostTranspose) {
    const ckl::FftTranspose variant = GetParam();
    for (const auto& shape : {std::pair<int, int>{64, 64}, std::pair<int, int>{128, 512},
                              std::pair<int, int>{1024, 256}, std::pair<int, int>{33, 97}}) {
        const int rows = shape.first;
        const int cols = shape.second;
        const std::vector<float2> host_in =
            random_signal(rows * cols, ckl::test::seed_stream(static_cast<std::uint64_t>(rows)));
        std::vector<float2> want(host_in.size());
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                want[static_cast<std::size_t>(c) * rows + r] =
                    host_in[static_cast<std::size_t>(r) * cols + c];
            }
        }
        ckl::DeviceBuffer<float2> da(host_in.size());
        ckl::DeviceBuffer<float2> db(host_in.size());
        da.copy_from_host(host_in);
        ckl::fft_transpose(da.data(), db.data(), rows, cols, 1, variant, nullptr, 1.0f);
        CKL_CUDA_CHECK(cudaDeviceSynchronize());
        const std::vector<float2> got = db.to_host();
        for (std::size_t i = 0; i < want.size(); ++i) {
            ASSERT_FLOAT_EQ(got[i].x, want[i].x) << ckl::fft_transpose_name(variant) << " at "
                                                 << rows << " by " << cols << ", i " << i;
            ASSERT_FLOAT_EQ(got[i].y, want[i].y) << ckl::fft_transpose_name(variant);
        }
    }
}

TEST_P(FftTransposeVariant, TheEpilogueMultipliesAtTheOutputIndex) {
    const ckl::FftTranspose variant = GetParam();
    constexpr int kRows = 64;
    constexpr int kCols = 32;
    const std::vector<float2> host_in = random_signal(kRows * kCols, ckl::test::seed_stream(404));
    const std::vector<float2> host_mul = random_signal(kRows * kCols, ckl::test::seed_stream(405));
    ckl::DeviceBuffer<float2> da(host_in.size());
    ckl::DeviceBuffer<float2> db(host_in.size());
    ckl::DeviceBuffer<float2> dm(host_mul.size());
    da.copy_from_host(host_in);
    dm.copy_from_host(host_mul);
    ckl::fft_transpose(da.data(), db.data(), kRows, kCols, 1, variant, dm.data(), 0.5f);
    CKL_CUDA_CHECK(cudaDeviceSynchronize());
    const std::vector<float2> got = db.to_host();
    for (int r = 0; r < kRows; ++r) {
        for (int c = 0; c < kCols; ++c) {
            const std::size_t out_i = static_cast<std::size_t>(c) * kRows + r;
            const float2 v = host_in[static_cast<std::size_t>(r) * kCols + c];
            const float2 m = host_mul[out_i];
            const float want_x = 0.5f * (v.x * m.x - v.y * m.y);
            const float want_y = 0.5f * (v.x * m.y + v.y * m.x);
            ASSERT_NEAR(got[out_i].x, want_x, 1e-5f) << ckl::fft_transpose_name(variant);
            ASSERT_NEAR(got[out_i].y, want_y, 1e-5f) << ckl::fft_transpose_name(variant);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(Study, FftTransposeVariant,
                         ::testing::Values(ckl::FftTranspose::kNaive,
                                           ckl::FftTranspose::kTiledPadded,
                                           ckl::FftTranspose::kStridedShared),
                         transpose_param_name);

// ---------------------------------------------------------------------------
// The 2D driver
// ---------------------------------------------------------------------------

class Fft2d : public GpuTestWithParam<ckl::FftTranspose> {};

TEST_P(Fft2d, MatchesASeparableCpuDft) {
    const ckl::FftTranspose variant = GetParam();
    constexpr int kRows = 16;
    constexpr int kCols = 32;
    ckl::Fft2dPlanOptions opt;
    opt.transpose = variant;
    ckl::Fft2dPlan plan(kRows, kCols, opt);

    const std::vector<float2> host_in = random_signal(kRows * kCols, ckl::test::seed_stream(51));
    // Row transforms then column transforms, both in long double on the host.
    std::vector<float2> mid(host_in.size());
    for (int r = 0; r < kRows; ++r) {
        const std::vector<float2> row(host_in.begin() + static_cast<std::ptrdiff_t>(r) * kCols,
                                      host_in.begin() + static_cast<std::ptrdiff_t>(r + 1) * kCols);
        const std::vector<float2> t = cpu_dft(row, true);
        std::copy(t.begin(), t.end(), mid.begin() + static_cast<std::ptrdiff_t>(r) * kCols);
    }
    std::vector<float2> want(host_in.size());
    for (int c = 0; c < kCols; ++c) {
        std::vector<float2> col(static_cast<std::size_t>(kRows));
        for (int r = 0; r < kRows; ++r) {
            col[static_cast<std::size_t>(r)] = mid[static_cast<std::size_t>(r) * kCols + c];
        }
        const std::vector<float2> t = cpu_dft(col, true);
        for (int r = 0; r < kRows; ++r) {
            want[static_cast<std::size_t>(r) * kCols + c] = t[static_cast<std::size_t>(r)];
        }
    }

    ckl::DeviceBuffer<float2> da(host_in.size());
    ckl::DeviceBuffer<float2> db(host_in.size());
    da.copy_from_host(host_in);
    ASSERT_EQ(ckl::fft2d(plan, ckl::FftDirection::kForward, da.data(), db.data()),
              ckl::Status::kSuccess);
    CKL_CUDA_CHECK(cudaDeviceSynchronize());
    EXPECT_LE(relative_rms(db.to_host(), want), ckl::fft_tolerance(kRows * kCols))
        << ckl::fft_transpose_name(variant);
}

TEST_P(Fft2d, RoundTripsAtASizeWorthMeasuring) {
    const ckl::FftTranspose variant = GetParam();
    constexpr int kRows = 512;
    constexpr int kCols = 512;
    ckl::Fft2dPlanOptions opt;
    opt.transpose = variant;
    ckl::Fft2dPlan plan(kRows, kCols, opt);
    EXPECT_EQ(plan.passes(), variant == ckl::FftTranspose::kStridedShared ? 2 : 4);
    EXPECT_EQ(plan.transpose_model_bytes() == 0, variant == ckl::FftTranspose::kStridedShared);

    const std::vector<float2> host_in = random_signal(kRows * kCols, ckl::test::seed_stream(52));
    ckl::DeviceBuffer<float2> da(host_in.size());
    ckl::DeviceBuffer<float2> db(host_in.size());
    da.copy_from_host(host_in);
    ASSERT_EQ(ckl::fft2d(plan, ckl::FftDirection::kForward, da.data(), db.data()),
              ckl::Status::kSuccess);
    ASSERT_EQ(ckl::fft2d(plan, ckl::FftDirection::kInverse, db.data(), da.data()),
              ckl::Status::kSuccess);
    ckl::fft_scale(da.data(), static_cast<long long>(kRows) * kCols,
                   1.0f / static_cast<float>(kRows * kCols));
    CKL_CUDA_CHECK(cudaDeviceSynchronize());
    EXPECT_LE(relative_rms(da.to_host(), host_in), ckl::fft_tolerance(kRows * kCols))
        << ckl::fft_transpose_name(variant);
}

INSTANTIATE_TEST_SUITE_P(RowColumn, Fft2d,
                         ::testing::Values(ckl::FftTranspose::kNaive,
                                           ckl::FftTranspose::kTiledPadded,
                                           ckl::FftTranspose::kStridedShared),
                         transpose_param_name);

// ---------------------------------------------------------------------------
// Real to complex, rung 5
// ---------------------------------------------------------------------------

class FftReal : public GpuTestWithParam<int> {};

TEST_P(FftReal, ForwardMatchesTheComplexTransformOfTheSameSignal) {
    const int packed = GetParam();
    const int real_len = 2 * packed;
    if (real_len > 4096) {
        // The reference is O(N^2) in long double. Past 2^12 it is the test that
        // would be the bottleneck, which is the same bound the spec puts on the
        // CPU DFT check, and the round trip below covers the larger sizes.
        GTEST_SKIP() << "the long double reference is affordable only to 2^12";
    }
    ckl::FftPlanOptions opt;
    opt.build_real = true;
    ckl::FftPlan plan(packed, opt);

    const std::vector<float> host_real =
        ckl::random_matrix(real_len, 1, ckl::test::seed_stream(static_cast<std::uint64_t>(packed)));
    // The reference is the full length complex transform of the same reals.
    std::vector<float2> as_complex(static_cast<std::size_t>(real_len));
    for (int i = 0; i < real_len; ++i) {
        as_complex[static_cast<std::size_t>(i)] =
            make_float2(host_real[static_cast<std::size_t>(i)], 0.0f);
    }
    const std::vector<float2> want = cpu_dft(as_complex, true);

    ckl::DeviceBuffer<float> dr(host_real.size());
    ckl::DeviceBuffer<float2> dspec(static_cast<std::size_t>(packed + 1));
    dr.copy_from_host(host_real);
    ckl::FftAlgo chosen = ckl::FftAlgo::kAuto;
    ASSERT_EQ(ckl::fft_r2c(plan, ckl::FftAlgo::kAuto, dr.data(), dspec.data(), &chosen),
              ckl::Status::kSuccess);
    EXPECT_NE(chosen, ckl::FftAlgo::kAuto) << "chosen has to name a real path";
    CKL_CUDA_CHECK(cudaDeviceSynchronize());
    const std::vector<float2> got = dspec.to_host();
    const std::vector<float2> want_half(want.begin(),
                                        want.begin() + static_cast<std::ptrdiff_t>(packed) + 1);
    EXPECT_LE(relative_rms(got, want_half), ckl::fft_tolerance(real_len));
}

TEST_P(FftReal, RoundTripsThroughTheHermitianSpectrum) {
    const int packed = GetParam();
    const int real_len = 2 * packed;
    ckl::FftPlanOptions opt;
    opt.build_real = true;
    ckl::FftPlan plan(packed, opt);

    const std::vector<float> host_real = ckl::random_matrix(
        real_len, 1, ckl::test::seed_stream(static_cast<std::uint64_t>(packed) + 500));
    ckl::DeviceBuffer<float> dr(host_real.size());
    ckl::DeviceBuffer<float> dback(host_real.size());
    ckl::DeviceBuffer<float2> dspec(static_cast<std::size_t>(packed + 1));
    dr.copy_from_host(host_real);
    ASSERT_EQ(ckl::fft_r2c(plan, ckl::FftAlgo::kAuto, dr.data(), dspec.data(), nullptr),
              ckl::Status::kSuccess);
    ASSERT_EQ(ckl::fft_c2r(plan, ckl::FftAlgo::kAuto, dspec.data(), dback.data(), nullptr),
              ckl::Status::kSuccess);
    CKL_CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> got = dback.to_host();
    // fft_c2r leaves the packed transform's own factor of n on the answer, which
    // the header states and this undoes.
    for (float& v : got) {
        v /= static_cast<float>(packed);
    }
    EXPECT_LE(relative_rms_real(got, host_real), ckl::fft_tolerance(real_len));
}

INSTANTIATE_TEST_SUITE_P(PackedHalfLength, FftReal, ::testing::Values(512, 1024, 4096, 65536),
                         packed_param_name);

TEST_F(FftRoundTrip, ARealTransformOnAPlanWithoutTheRealTablesIsRefused) {
    ckl::FftPlan plan(1024);
    ckl::DeviceBuffer<float> dr(2048);
    ckl::DeviceBuffer<float2> dspec(1025);
    EXPECT_EQ(ckl::fft_r2c(plan, ckl::FftAlgo::kAuto, dr.data(), dspec.data(), nullptr),
              ckl::Status::kNotSupported);
    char message[512] = {0};
    ckl_last_error(message, sizeof(message));
    EXPECT_NE(std::string(message).find("build_real"), std::string::npos) << message;
}

// ---------------------------------------------------------------------------
// The plan's own state and the dispatch contract
// ---------------------------------------------------------------------------

class FftPlanState : public GpuTest {};

TEST_F(FftPlanState, TheSharedResidentBoundIsTheSharedMemoryArithmetic) {
    EXPECT_EQ(ckl::FftPlan::shared_resident_max(), 4096);
    // 16n bytes for two ping pong buffers: 2^12 costs 65536 and fits under the
    // 101376 byte per block limit, 2^13 costs 131072 and does not.
    EXPECT_LE(16 * ckl::FftPlan::shared_resident_max(), 101376);
    EXPECT_GT(16 * 2 * ckl::FftPlan::shared_resident_max(), 101376);

    ckl::FftPlan small(4096);
    EXPECT_TRUE(small.supports(ckl::FftAlgo::kSharedResident));
    EXPECT_EQ(std::string(small.refusal(ckl::FftAlgo::kSharedResident)), std::string());

    ckl::FftPlan big(8192);
    EXPECT_FALSE(big.supports(ckl::FftAlgo::kSharedResident));
    EXPECT_NE(std::string(big.refusal(ckl::FftAlgo::kSharedResident)).find("101376"),
              std::string::npos)
        << big.refusal(ckl::FftAlgo::kSharedResident);
}

TEST_F(FftPlanState, ThePassCountsAreTheOnesTheTrafficModelDeclares) {
    ckl::FftPlan plan(1 << 20);
    EXPECT_EQ(plan.passes(ckl::FftAlgo::kRadix2Global), 20);
    EXPECT_EQ(plan.passes(ckl::FftAlgo::kRadix4Global), 10);
    // 20 is not divisible by 3, so the radix 8 ladder opens with a radix 4 stage
    // and then runs six radix 8 stages.
    EXPECT_EQ(plan.passes(ckl::FftAlgo::kRadix8Global), 7);
    EXPECT_EQ(plan.passes(ckl::FftAlgo::kFourStep), 5);
    EXPECT_EQ(plan.passes(ckl::FftAlgo::kCufft), 0);

    const long long n = 1 << 20;
    EXPECT_EQ(plan.model_bytes(ckl::FftAlgo::kFourStep), 5 * 16 * n);
    EXPECT_EQ(plan.model_bytes(ckl::FftAlgo::kRadix2Global), 20 * 16 * n);
    EXPECT_EQ(plan.model_bytes(ckl::FftAlgo::kCufft), 0);

    // The twiddle band brackets the honest answer: the low end is the whole
    // factored table, which is what a perfectly cached one would cost.
    EXPECT_GT(plan.twiddle_bytes_low(ckl::FftAlgo::kRadix2Global), 0);
    EXPECT_LT(plan.twiddle_bytes_low(ckl::FftAlgo::kRadix2Global), 65536);
    EXPECT_GT(plan.twiddle_bytes_high(ckl::FftAlgo::kRadix2Global),
              plan.twiddle_bytes_low(ckl::FftAlgo::kRadix2Global));
}

TEST_F(FftPlanState, TheFourStepFactorsBothFitTheSharedBound) {
    for (int bits = 13; bits <= 24; ++bits) {
        ckl::FftPlan plan(1 << bits);
        ASSERT_TRUE(plan.supports(ckl::FftAlgo::kFourStep)) << "2^" << bits;
        EXPECT_LE(plan.four_step_n1(), ckl::FftPlan::shared_resident_max()) << "2^" << bits;
        EXPECT_LE(plan.four_step_n2(), ckl::FftPlan::shared_resident_max()) << "2^" << bits;
        EXPECT_EQ(plan.four_step_n1() * plan.four_step_n2(), 1 << bits);
    }
}

TEST_F(FftPlanState, ANamedFactorizationOverridesTheDefaultSplit) {
    ckl::FftPlanOptions opt;
    opt.four_step_n1 = 4096;
    ckl::FftPlan plan(1 << 20, opt);
    EXPECT_EQ(plan.four_step_n1(), 4096);
    EXPECT_EQ(plan.four_step_n2(), 256);

    const std::vector<float2> host_in = random_signal(1 << 20, ckl::test::seed_stream(88));
    ckl::DeviceBuffer<float2> da(host_in.size());
    ckl::DeviceBuffer<float2> db(host_in.size());
    EXPECT_LE(round_trip(plan, ckl::FftAlgo::kFourStep, host_in, da, db, nullptr),
              ckl::fft_tolerance(1 << 20));
}

TEST_F(FftPlanState, AutoPicksSharedBelowTheBoundAndFourStepAbove) {
    EXPECT_EQ(ckl::FftPlan(1024).query(), ckl::FftAlgo::kSharedResident);
    EXPECT_EQ(ckl::FftPlan(4096).query(), ckl::FftAlgo::kSharedResident);
    EXPECT_EQ(ckl::FftPlan(8192).query(), ckl::FftAlgo::kFourStep);
    EXPECT_EQ(ckl::FftPlan(1 << 24).query(), ckl::FftAlgo::kFourStep);
}

TEST_F(FftPlanState, EveryAlgorithmAndVariantHasAName) {
    for (ckl::FftAlgo a :
         {ckl::FftAlgo::kAuto, ckl::FftAlgo::kRadix2Global, ckl::FftAlgo::kSharedResident,
          ckl::FftAlgo::kRadix4Global, ckl::FftAlgo::kRadix8Global, ckl::FftAlgo::kFourStep,
          ckl::FftAlgo::kCufft}) {
        EXPECT_NE(std::string(ckl::fft_algo_name(a)), std::string("unknown"));
    }
    for (ckl::FftTranspose t : {ckl::FftTranspose::kNaive, ckl::FftTranspose::kTiledPadded,
                                ckl::FftTranspose::kStridedShared}) {
        EXPECT_NE(std::string(ckl::fft_transpose_name(t)), std::string("unknown"));
    }
    EXPECT_EQ(std::string(ckl::fft_direction_name(ckl::FftDirection::kForward)), "forward");
    EXPECT_EQ(std::string(ckl::fft_direction_name(ckl::FftDirection::kInverse)), "inverse");
}

TEST_F(FftPlanState, TheToleranceGrowsWithTheStageCount) {
    EXPECT_LT(ckl::fft_tolerance(1024), ckl::fft_tolerance(1 << 24));
    // 8 log2(N) FLT_EPSILON, evaluated. FLT_EPSILON is 1.1920929e-7, so 2^24 gives
    // 8 * 24 * 1.1920929e-7 = 2.29e-5 and 2^10 gives 9.54e-6. The build spec quotes
    // 1.1e-5 at 2^24 beside the same formula, which is the formula evaluated at the
    // unit roundoff FLT_EPSILON/2 rather than at FLT_EPSILON; docs/fft.md records
    // the discrepancy and this library implements the formula as written.
    EXPECT_NEAR(ckl::fft_tolerance(1 << 24), 8.0 * 24.0 * static_cast<double>(FLT_EPSILON), 1e-12);
    EXPECT_NEAR(ckl::fft_tolerance(1 << 10), 8.0 * 10.0 * static_cast<double>(FLT_EPSILON), 1e-12);
}

class FftDispatch : public GpuTest {};

TEST_F(FftDispatch, ARefusedVariantSaysSoRatherThanFallingBack) {
    ckl::FftPlan plan(1 << 16);
    const std::vector<float2> host_in = random_signal(1 << 16, ckl::test::seed_stream(9));
    ckl::DeviceBuffer<float2> da(host_in.size());
    ckl::DeviceBuffer<float2> db(host_in.size());
    da.copy_from_host(host_in);

    ckl::FftAlgo chosen = ckl::FftAlgo::kAuto;
    const ckl::Status st = ckl::fft(plan, ckl::FftAlgo::kSharedResident,
                                    ckl::FftDirection::kForward, da.data(), db.data(), &chosen);
    EXPECT_EQ(st, ckl::Status::kNotSupported);
    // chosen is written on failure as well as on success, so a test can see which
    // path was asked for.
    EXPECT_EQ(chosen, ckl::FftAlgo::kSharedResident);
    char message[512] = {0};
    ckl_last_error(message, sizeof(message));
    EXPECT_NE(std::string(message).find("shared memory"), std::string::npos) << message;
}

TEST_F(FftDispatch, AnInPlaceCallIsRefusedRatherThanSilentlyWrong) {
    ckl::FftPlan plan(1024);
    ckl::DeviceBuffer<float2> da(1024);
    EXPECT_EQ(ckl::fft(plan, ckl::FftAlgo::kAuto, ckl::FftDirection::kForward, da.data(), da.data(),
                       nullptr),
              ckl::Status::kInvalidValue);
}

TEST_F(FftDispatch, AutoReportsTheAlgorithmItRan) {
    ckl::FftPlan plan(2048);
    const std::vector<float2> host_in = random_signal(2048, ckl::test::seed_stream(12));
    ckl::DeviceBuffer<float2> da(host_in.size());
    ckl::DeviceBuffer<float2> db(host_in.size());
    da.copy_from_host(host_in);
    ckl::FftAlgo chosen = ckl::FftAlgo::kAuto;
    ASSERT_EQ(ckl::fft(plan, ckl::FftAlgo::kAuto, ckl::FftDirection::kForward, da.data(), db.data(),
                       &chosen),
              ckl::Status::kSuccess);
    EXPECT_EQ(chosen, ckl::FftAlgo::kSharedResident);
    EXPECT_NE(chosen, ckl::FftAlgo::kAuto);
}

TEST_F(FftDispatch, ABadLengthIsRejectedByTheConstructor) {
    EXPECT_THROW(ckl::FftPlan(1000), ckl::Error);
    EXPECT_THROW(ckl::FftPlan(1 << 25), ckl::Error);
    ckl::FftPlanOptions opt;
    opt.batch = 0;
    EXPECT_THROW(ckl::FftPlan(1024, opt), ckl::Error);
}

TEST_F(FftDispatch, TheCEntryPointDispatchesAndReportsWhatItRan) {
    ckl_handle_t h = nullptr;
    ASSERT_EQ(ckl_create(&h), CKL_STATUS_SUCCESS);
    constexpr int kN = 4096;
    const std::vector<float2> host_in = random_signal(kN, ckl::test::seed_stream(64));
    ckl::DeviceBuffer<float2> da(host_in.size());
    ckl::DeviceBuffer<float2> db(host_in.size());
    da.copy_from_host(host_in);

    ckl_fft_algo_t chosen = CKL_FFT_AUTO;
    EXPECT_EQ(ckl_fft_c2c(h, kN, 1, da.data(), db.data(), CKL_FFT_FORWARD, CKL_FFT_AUTO, &chosen),
              CKL_STATUS_SUCCESS);
    EXPECT_EQ(chosen, CKL_FFT_SHARED_RESIDENT);
    EXPECT_EQ(ckl_fft_c2c(h, kN, 1, db.data(), da.data(), CKL_FFT_INVERSE, CKL_FFT_AUTO, nullptr),
              CKL_STATUS_SUCCESS);
    ckl::fft_scale(da.data(), kN, 1.0f / static_cast<float>(kN));
    EXPECT_EQ(ckl_device_synchronize(), CKL_STATUS_SUCCESS);
    EXPECT_LE(relative_rms(da.to_host(), host_in), ckl::fft_tolerance(kN));

    // Out of range arguments are rejected rather than cast into something.
    EXPECT_EQ(ckl_fft_c2c(h, 1000, 1, da.data(), db.data(), CKL_FFT_FORWARD, CKL_FFT_AUTO, nullptr),
              CKL_STATUS_INVALID_VALUE);
    EXPECT_EQ(ckl_fft_c2c(h, kN, 1, da.data(), db.data(), CKL_FFT_FORWARD,
                          static_cast<ckl_fft_algo_t>(CKL_FFT_CUFFT + 1), nullptr),
              CKL_STATUS_INVALID_VALUE);
    EXPECT_EQ(ckl_destroy(h), CKL_STATUS_SUCCESS);
}

TEST_F(FftDispatch, TheFreeFunctionBaselineAgreesWithThePlan) {
    constexpr int kN = 8192;
    const std::vector<float2> host_in = random_signal(kN, ckl::test::seed_stream(71));
    ckl::DeviceBuffer<float2> da(host_in.size());
    ckl::DeviceBuffer<float2> db(host_in.size());
    ckl::DeviceBuffer<float2> dc(host_in.size());
    da.copy_from_host(host_in);

    ckl::FftPlan plan(kN);
    ASSERT_EQ(ckl::fft(plan, ckl::FftAlgo::kCufft, ckl::FftDirection::kForward, da.data(),
                       db.data(), nullptr),
              ckl::Status::kSuccess);
    ckl::fft_cufft(da.data(), dc.data(), kN, 1, ckl::FftDirection::kForward);
    CKL_CUDA_CHECK(cudaDeviceSynchronize());
    const std::vector<float2> through_plan = db.to_host();
    const std::vector<float2> through_free = dc.to_host();
    for (std::size_t i = 0; i < through_plan.size(); ++i) {
        ASSERT_FLOAT_EQ(through_free[i].x, through_plan[i].x) << "i " << i;
        ASSERT_FLOAT_EQ(through_free[i].y, through_plan[i].y) << "i " << i;
    }
}

TEST_F(FftDispatch, EveryRungAgreesWithCufftOnTheSameInput) {
    for (int bits : {10, 13, 16, 20}) {
        const int n = 1 << bits;
        ckl::FftPlan plan(n);
        const std::vector<float2> host_in =
            random_signal(n, ckl::test::seed_stream(static_cast<std::uint64_t>(bits) + 3000));
        ckl::DeviceBuffer<float2> da(host_in.size());
        ckl::DeviceBuffer<float2> db(host_in.size());
        ckl::DeviceBuffer<float2> dref(host_in.size());
        da.copy_from_host(host_in);
        ASSERT_EQ(ckl::fft(plan, ckl::FftAlgo::kCufft, ckl::FftDirection::kForward, da.data(),
                           dref.data(), nullptr),
                  ckl::Status::kSuccess);
        CKL_CUDA_CHECK(cudaDeviceSynchronize());
        const std::vector<float2> want = dref.to_host();
        for (ckl::FftAlgo algo : all_algos()) {
            if (algo == ckl::FftAlgo::kCufft || !plan.supports(algo)) {
                continue;
            }
            ASSERT_EQ(
                ckl::fft(plan, algo, ckl::FftDirection::kForward, da.data(), db.data(), nullptr),
                ckl::Status::kSuccess)
                << name_of(algo);
            CKL_CUDA_CHECK(cudaDeviceSynchronize());
            EXPECT_LE(relative_rms(db.to_host(), want), ckl::fft_tolerance(n))
                << "2^" << bits << " " << name_of(algo);
        }
    }
}

class FftStreams : public GpuTest {};

TEST_F(FftStreams, TwoConcurrentStreamsDoNotInterfere) {
    constexpr int kN = 16384;
    cudaStream_t s1 = nullptr;
    cudaStream_t s2 = nullptr;
    CKL_CUDA_CHECK(cudaStreamCreateWithFlags(&s1, cudaStreamDefault));
    CKL_CUDA_CHECK(cudaStreamCreateWithFlags(&s2, cudaStreamDefault));

    ckl::FftPlan p1(kN);
    ckl::FftPlan p2(kN);
    const std::vector<float2> in1 = random_signal(kN, ckl::test::seed_stream(101));
    const std::vector<float2> in2 = random_signal(kN, ckl::test::seed_stream(102));
    ckl::DeviceBuffer<float2> a1(in1.size());
    ckl::DeviceBuffer<float2> b1(in1.size());
    ckl::DeviceBuffer<float2> a2(in2.size());
    ckl::DeviceBuffer<float2> b2(in2.size());
    a1.copy_from_host(in1);
    a2.copy_from_host(in2);

    ASSERT_EQ(ckl::fft(p1, ckl::FftAlgo::kFourStep, ckl::FftDirection::kForward, a1.data(),
                       b1.data(), nullptr, s1),
              ckl::Status::kSuccess);
    ASSERT_EQ(ckl::fft(p2, ckl::FftAlgo::kFourStep, ckl::FftDirection::kForward, a2.data(),
                       b2.data(), nullptr, s2),
              ckl::Status::kSuccess);
    ASSERT_EQ(ckl::fft(p1, ckl::FftAlgo::kFourStep, ckl::FftDirection::kInverse, b1.data(),
                       a1.data(), nullptr, s1),
              ckl::Status::kSuccess);
    ASSERT_EQ(ckl::fft(p2, ckl::FftAlgo::kFourStep, ckl::FftDirection::kInverse, b2.data(),
                       a2.data(), nullptr, s2),
              ckl::Status::kSuccess);
    ckl::fft_scale(a1.data(), kN, 1.0f / static_cast<float>(kN), s1);
    ckl::fft_scale(a2.data(), kN, 1.0f / static_cast<float>(kN), s2);
    CKL_CUDA_CHECK(cudaStreamSynchronize(s1));
    CKL_CUDA_CHECK(cudaStreamSynchronize(s2));

    EXPECT_LE(relative_rms(a1.to_host(), in1), ckl::fft_tolerance(kN));
    EXPECT_LE(relative_rms(a2.to_host(), in2), ckl::fft_tolerance(kN));
    CKL_CUDA_CHECK(cudaStreamDestroy(s1));
    CKL_CUDA_CHECK(cudaStreamDestroy(s2));
}

}  // namespace
