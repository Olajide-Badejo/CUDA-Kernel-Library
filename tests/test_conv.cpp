// Convolution correctness: every variant against the same double precision time
// domain reference, at several signal lengths and several filter lengths.
//
// The reference is the definition, computed in double on the host: out[i] is the
// sum over j of signal[j] * filter[i - j], output length N + M - 1. Both FFT
// variants and both direct variants have to agree with it, which is the only
// check that catches the failure this family is prone to: an off by one in the
// zero padding, or a filter spectrum that was cached from the wrong array, gives
// an answer that is smooth, plausible and shifted.
//
// The tolerance is the transform's own, at the padded length L, because that is
// where the error comes from. A direct variant is held to the same bound even
// though it does no transform at all, which is generous to it and still tight
// enough to fail a wrong answer: its error is a length M dot product in FP32.
//
// The three things asserted here that a "does it convolve" suite would not: every
// variant is checked through the plan with chosen asserted, so a variant that
// fell back to another one fails rather than passing on the right answer from the
// wrong kernel; the filter spectrum is checked to be cached, by running the same
// plan twice and asserting the second call does not need the filter buffer to
// still be intact; and a filter too long for the constant bank is checked to be
// refused with a reason rather than truncated.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include <cuda_runtime.h>

#include <gtest/gtest.h>

#include "ckl/ckl.h"
#include "ckl/cuda_check.hpp"
#include "ckl/device_buffer.hpp"
#include "ckl/fft.hpp"
#include "gpu_environment.hpp"
#include "reference.hpp"

namespace {

using ckl::test::GpuTest;
using ckl::test::GpuTestWithParam;

// The definition, in double. O(N M), so the suite keeps N at or below 2^16.
std::vector<double> cpu_convolution(const std::vector<float>& signal,
                                    const std::vector<float>& filter) {
    const std::size_t n = signal.size();
    const std::size_t m = filter.size();
    std::vector<double> out(n + m - 1, 0.0);
    for (std::size_t j = 0; j < n; ++j) {
        const double s = static_cast<double>(signal[j]);
        for (std::size_t t = 0; t < m; ++t) {
            out[j + t] += s * static_cast<double>(filter[t]);
        }
    }
    return out;
}

double relative_rms(const std::vector<float>& got, const std::vector<double>& want) {
    double num = 0.0;
    double den = 0.0;
    for (std::size_t i = 0; i < want.size(); ++i) {
        const double d = static_cast<double>(got[i]) - want[i];
        num += d * d;
        den += want[i] * want[i];
    }
    return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
}

// Named rather than written as a lambda at the instantiation site: GoogleTest's
// generated code has a parameter called info, and a lambda parameter of the same
// name shadows it.
std::string shape_param_name(const ::testing::TestParamInfo<std::tuple<int, int>>& info) {
    return "n" + std::to_string(std::get<0>(info.param)) + "m" +
           std::to_string(std::get<1>(info.param));
}

std::vector<ckl::ConvAlgo> all_algos() {
    return {ckl::ConvAlgo::kFftSeparate, ckl::ConvAlgo::kFftFused, ckl::ConvAlgo::kDirectShared,
            ckl::ConvAlgo::kDirectConstant, ckl::ConvAlgo::kCufft};
}

const char* name_of(ckl::ConvAlgo a) {
    return ckl::conv_algo_name(a);
}

// ---------------------------------------------------------------------------
// Every variant against the double reference
// ---------------------------------------------------------------------------

class ConvAgainstReference : public GpuTestWithParam<std::tuple<int, int>> {};

TEST_P(ConvAgainstReference, EveryVariantMatchesTheDoubleTimeDomainAnswer) {
    const int n = std::get<0>(GetParam());
    const int m = std::get<1>(GetParam());
    const std::vector<float> signal =
        ckl::random_matrix(n, 1, ckl::test::seed_stream(static_cast<std::uint64_t>(n) * 31 + 1));
    const std::vector<float> filter =
        ckl::random_matrix(m, 1, ckl::test::seed_stream(static_cast<std::uint64_t>(m) * 17 + 2));
    const std::vector<double> want = cpu_convolution(signal, filter);

    ckl::DeviceBuffer<float> ds(signal.size());
    ckl::DeviceBuffer<float> df(filter.size());
    ckl::DeviceBuffer<float> dout(want.size());
    ds.copy_from_host(signal);
    df.copy_from_host(filter);

    ckl::ConvPlan plan(n, df.data(), m);
    ASSERT_EQ(plan.output_length(), static_cast<int>(want.size()));
    const double bound = ckl::fft_tolerance(plan.transform_length());

    for (ckl::ConvAlgo algo : all_algos()) {
        if (!plan.supports(algo)) {
            EXPECT_NE(std::string(plan.refusal(algo)), std::string()) << "a refusal has to say why";
            continue;
        }
        dout.zero();
        ckl::ConvAlgo chosen = ckl::ConvAlgo::kAuto;
        ASSERT_EQ(ckl::conv(plan, algo, ds.data(), dout.data(), &chosen), ckl::Status::kSuccess)
            << name_of(algo);
        EXPECT_EQ(chosen, algo) << "a variant that runs a path its name does not claim is a defect";
        CKL_CUDA_CHECK(cudaDeviceSynchronize());
        const double err = relative_rms(dout.to_host(), want);
        RecordProperty(std::string("rms_") + name_of(algo), ckl::test::sci(err));
        EXPECT_LE(err, bound) << "N = " << n << ", M = " << m << ", variant " << name_of(algo);
    }
}

INSTANTIATE_TEST_SUITE_P(SignalAndFilter, ConvAgainstReference,
                         ::testing::Values(std::make_tuple(1024, 8), std::make_tuple(1024, 64),
                                           std::make_tuple(4096, 129), std::make_tuple(16384, 1024),
                                           std::make_tuple(65536, 8), std::make_tuple(65536, 4096),
                                           std::make_tuple(65536, 16384)),
                         shape_param_name);

// ---------------------------------------------------------------------------
// The plan's own state
// ---------------------------------------------------------------------------

class ConvPlanState : public GpuTest {};

TEST_F(ConvPlanState, ThePaddedLengthIsTheNextPowerOfTwoAtOrAboveTheOutput) {
    ckl::DeviceBuffer<float> df(64);
    df.zero();
    ckl::ConvPlan plan(1000, df.data(), 64);
    EXPECT_EQ(plan.signal_length(), 1000);
    EXPECT_EQ(plan.filter_length(), 64);
    EXPECT_EQ(plan.output_length(), 1063);
    EXPECT_EQ(plan.transform_length(), 2048);
}

TEST_F(ConvPlanState, AFilterTooLongForTheConstantBankIsRefusedWithAReason) {
    const int m = ckl::conv_direct_constant_max_taps() + 1;
    ckl::DeviceBuffer<float> df(static_cast<std::size_t>(m));
    df.zero();
    ckl::ConvPlan plan(1 << 15, df.data(), m);
    EXPECT_FALSE(plan.supports(ckl::ConvAlgo::kDirectConstant));
    EXPECT_NE(std::string(plan.refusal(ckl::ConvAlgo::kDirectConstant))
                  .find(std::to_string(ckl::conv_direct_constant_max_taps())),
              std::string::npos)
        << plan.refusal(ckl::ConvAlgo::kDirectConstant);
    EXPECT_TRUE(plan.supports(ckl::ConvAlgo::kDirectShared));
}

TEST_F(ConvPlanState, ARefusedVariantIsRefusedAtTheCallToo) {
    const int m = ckl::conv_direct_constant_max_taps() + 1;
    const std::vector<float> signal = ckl::random_matrix(4096, 1, ckl::test::seed_stream(1));
    const std::vector<float> filter = ckl::random_matrix(m, 1, ckl::test::seed_stream(2));
    ckl::DeviceBuffer<float> ds(signal.size());
    ckl::DeviceBuffer<float> df(filter.size());
    ckl::DeviceBuffer<float> dout(static_cast<std::size_t>(4096 + m - 1));
    ds.copy_from_host(signal);
    df.copy_from_host(filter);
    ckl::ConvPlan plan(4096, df.data(), m);

    ckl::ConvAlgo chosen = ckl::ConvAlgo::kAuto;
    EXPECT_EQ(ckl::conv(plan, ckl::ConvAlgo::kDirectConstant, ds.data(), dout.data(), &chosen),
              ckl::Status::kNotSupported);
    // chosen is written on failure as well as on success.
    EXPECT_EQ(chosen, ckl::ConvAlgo::kDirectConstant);
    char message[512] = {0};
    ckl_last_error(message, sizeof(message));
    EXPECT_NE(std::string(message).find("constant memory rung"), std::string::npos) << message;
}

TEST_F(ConvPlanState, TheFusedVariantIsRefusedOnARungWithNoStoreEpilogue) {
    ckl::DeviceBuffer<float> df(32);
    df.zero();
    ckl::ConvPlanOptions opt;
    opt.fft_algo = ckl::FftAlgo::kRadix2Global;
    ckl::ConvPlan plan(4096, df.data(), 32, opt);
    EXPECT_EQ(plan.fft_algo(), ckl::FftAlgo::kRadix2Global);
    EXPECT_FALSE(plan.supports(ckl::ConvAlgo::kFftFused));
    EXPECT_NE(std::string(plan.refusal(ckl::ConvAlgo::kFftFused)).find("store epilogue"),
              std::string::npos)
        << plan.refusal(ckl::ConvAlgo::kFftFused);
    EXPECT_TRUE(plan.supports(ckl::ConvAlgo::kFftSeparate));
}

TEST_F(ConvPlanState, TheFusedModelMovesLessThanTheSeparateOne) {
    ckl::DeviceBuffer<float> df(256);
    df.zero();
    ckl::ConvPlan plan(1 << 20, df.data(), 256);
    const long long separate = plan.model_bytes(ckl::ConvAlgo::kFftSeparate);
    const long long fused = plan.model_bytes(ckl::ConvAlgo::kFftFused);
    EXPECT_LT(fused, separate);
    // The saving is exactly the pointwise pass the fused epilogue removes: it read
    // two spectra and wrote one, and the epilogue reads one and writes none extra.
    EXPECT_EQ(separate - fused, 16LL * plan.transform_length());
    // Both FFT variants carry the same number of transform passes.
    EXPECT_EQ(plan.passes(ckl::ConvAlgo::kFftSeparate), plan.passes(ckl::ConvAlgo::kFftFused));
    EXPECT_EQ(plan.passes(ckl::ConvAlgo::kDirectShared), 0);
}

TEST_F(ConvPlanState, TheFlopModelsAreTheOnesTheReportQuotes) {
    ckl::DeviceBuffer<float> df(64);
    df.zero();
    const int n = 1 << 20;
    ckl::ConvPlan plan(n, df.data(), 64);
    EXPECT_DOUBLE_EQ(plan.flop_model(ckl::ConvAlgo::kDirectShared),
                     2.0 * static_cast<double>(n) * 64.0);
    const double l = static_cast<double>(plan.transform_length());
    EXPECT_DOUBLE_EQ(plan.flop_model(ckl::ConvAlgo::kFftFused),
                     2.0 * 5.0 * l * std::log2(l) + 6.0 * l);
}

TEST_F(ConvPlanState, AutoPrefersDirectAtAShortFilterAndTheTransformAtALongOne) {
    ckl::DeviceBuffer<float> tiny(4);
    tiny.zero();
    ckl::ConvPlan shortest(1 << 20, tiny.data(), 4);
    EXPECT_EQ(shortest.query(), ckl::ConvAlgo::kDirectConstant);

    ckl::DeviceBuffer<float> huge(16384);
    huge.zero();
    ckl::ConvPlan longest(1 << 20, huge.data(), 16384);
    EXPECT_EQ(longest.query(), ckl::ConvAlgo::kFftFused);
}

TEST_F(ConvPlanState, EveryAlgorithmHasAName) {
    for (ckl::ConvAlgo a :
         {ckl::ConvAlgo::kAuto, ckl::ConvAlgo::kFftSeparate, ckl::ConvAlgo::kFftFused,
          ckl::ConvAlgo::kDirectShared, ckl::ConvAlgo::kDirectConstant, ckl::ConvAlgo::kCufft}) {
        EXPECT_NE(std::string(ckl::conv_algo_name(a)), std::string("unknown"));
    }
}

TEST_F(ConvPlanState, ABadLengthIsRejectedByTheConstructor) {
    ckl::DeviceBuffer<float> df(8);
    df.zero();
    EXPECT_THROW(ckl::ConvPlan(0, df.data(), 8), ckl::Error);
    EXPECT_THROW(ckl::ConvPlan(64, df.data(), 0), ckl::Error);
    EXPECT_THROW(ckl::ConvPlan(64, nullptr, 8), ckl::Error);
    // N + M - 1 past 2^24 needs a padded length past 2^24, which is out of scope.
    EXPECT_THROW(ckl::ConvPlan(1 << 24, df.data(), 8), ckl::Error);
}

// ---------------------------------------------------------------------------
// The cached filter spectrum
// ---------------------------------------------------------------------------

class ConvCaching : public GpuTest {};

TEST_F(ConvCaching, TheFilterSpectrumSurvivesTheFilterBufferGoingAway) {
    constexpr int kN = 4096;
    constexpr int kM = 63;
    const std::vector<float> signal = ckl::random_matrix(kN, 1, ckl::test::seed_stream(21));
    const std::vector<float> filter = ckl::random_matrix(kM, 1, ckl::test::seed_stream(22));
    const std::vector<double> want = cpu_convolution(signal, filter);

    ckl::DeviceBuffer<float> ds(signal.size());
    ckl::DeviceBuffer<float> dout(want.size());
    ds.copy_from_host(signal);

    ckl::DeviceBuffer<float> df(filter.size());
    df.copy_from_host(filter);
    ckl::ConvPlan plan(kN, df.data(), kM);
    // The plan took its own copy of the taps and cached the spectrum, so trashing
    // the caller's filter must not change the answer. A plan that held the
    // caller's pointer and transformed it per call would fail here.
    df.zero();
    CKL_CUDA_CHECK(cudaDeviceSynchronize());

    for (ckl::ConvAlgo algo : all_algos()) {
        if (!plan.supports(algo)) {
            continue;
        }
        dout.zero();
        ASSERT_EQ(ckl::conv(plan, algo, ds.data(), dout.data(), nullptr), ckl::Status::kSuccess)
            << name_of(algo);
        CKL_CUDA_CHECK(cudaDeviceSynchronize());
        EXPECT_LE(relative_rms(dout.to_host(), want), ckl::fft_tolerance(plan.transform_length()))
            << name_of(algo);
    }
}

TEST_F(ConvCaching, TwoPlansCanShareTheConstantBankByRebinding) {
    constexpr int kN = 2048;
    const std::vector<float> signal = ckl::random_matrix(kN, 1, ckl::test::seed_stream(41));
    const std::vector<float> f1 = ckl::random_matrix(32, 1, ckl::test::seed_stream(42));
    const std::vector<float> f2 = ckl::random_matrix(48, 1, ckl::test::seed_stream(43));
    const std::vector<double> w1 = cpu_convolution(signal, f1);
    const std::vector<double> w2 = cpu_convolution(signal, f2);

    ckl::DeviceBuffer<float> ds(signal.size());
    ckl::DeviceBuffer<float> d1(f1.size());
    ckl::DeviceBuffer<float> d2(f2.size());
    ckl::DeviceBuffer<float> o1(w1.size());
    ckl::DeviceBuffer<float> o2(w2.size());
    ds.copy_from_host(signal);
    d1.copy_from_host(f1);
    d2.copy_from_host(f2);

    ckl::ConvPlan p1(kN, d1.data(), 32);
    ckl::ConvPlan p2(kN, d2.data(), 48);
    // p2 bound last, so p1 has to rebind before it can run. The alternative would
    // be convolving with p2's filter and reporting success.
    for (int repeat = 0; repeat < 2; ++repeat) {
        ASSERT_EQ(ckl::conv(p1, ckl::ConvAlgo::kDirectConstant, ds.data(), o1.data(), nullptr),
                  ckl::Status::kSuccess);
        ASSERT_EQ(ckl::conv(p2, ckl::ConvAlgo::kDirectConstant, ds.data(), o2.data(), nullptr),
                  ckl::Status::kSuccess);
        CKL_CUDA_CHECK(cudaDeviceSynchronize());
        EXPECT_LE(relative_rms(o1.to_host(), w1), ckl::fft_tolerance(p1.transform_length()));
        EXPECT_LE(relative_rms(o2.to_host(), w2), ckl::fft_tolerance(p2.transform_length()));
    }
}

// ---------------------------------------------------------------------------
// The C entry point
// ---------------------------------------------------------------------------

class ConvCApi : public GpuTest {};

TEST_F(ConvCApi, DispatchesAndReportsWhatItRan) {
    ckl_handle_t h = nullptr;
    ASSERT_EQ(ckl_create(&h), CKL_STATUS_SUCCESS);
    constexpr int kN = 8192;
    constexpr int kM = 96;
    const std::vector<float> signal = ckl::random_matrix(kN, 1, ckl::test::seed_stream(61));
    const std::vector<float> filter = ckl::random_matrix(kM, 1, ckl::test::seed_stream(62));
    const std::vector<double> want = cpu_convolution(signal, filter);

    ckl::DeviceBuffer<float> ds(signal.size());
    ckl::DeviceBuffer<float> df(filter.size());
    ckl::DeviceBuffer<float> dout(want.size());
    ds.copy_from_host(signal);
    df.copy_from_host(filter);

    ckl_conv_algo_t chosen = CKL_CONV_AUTO;
    EXPECT_EQ(
        ckl_conv_r2r(h, kN, ds.data(), kM, df.data(), dout.data(), CKL_CONV_FFT_FUSED, &chosen),
        CKL_STATUS_SUCCESS);
    EXPECT_EQ(chosen, CKL_CONV_FFT_FUSED);
    EXPECT_EQ(ckl_device_synchronize(), CKL_STATUS_SUCCESS);
    EXPECT_LE(relative_rms(dout.to_host(), want), ckl::fft_tolerance(16384));

    EXPECT_EQ(ckl_conv_r2r(h, kN, ds.data(), kM, df.data(), dout.data(),
                           static_cast<ckl_conv_algo_t>(CKL_CONV_CUFFT + 1), &chosen),
              CKL_STATUS_INVALID_VALUE);
    EXPECT_EQ(ckl_conv_r2r(h, 0, ds.data(), kM, df.data(), dout.data(), CKL_CONV_AUTO, &chosen),
              CKL_STATUS_INVALID_VALUE);
    EXPECT_EQ(ckl_destroy(h), CKL_STATUS_SUCCESS);
}

// ---------------------------------------------------------------------------
// The direct kernels on their own
// ---------------------------------------------------------------------------

class ConvDirectKernels : public GpuTest {};

TEST_F(ConvDirectKernels, TheSharedFormHandlesFiltersLongerThanOneTapChunk) {
    // 256 taps is one chunk, so 257 and 1000 exercise the chunk loop and its tail.
    for (int m : {1, 2, 255, 256, 257, 1000}) {
        constexpr int kN = 3000;
        const std::vector<float> signal =
            ckl::random_matrix(kN, 1, ckl::test::seed_stream(static_cast<std::uint64_t>(m) + 90));
        const std::vector<float> filter =
            ckl::random_matrix(m, 1, ckl::test::seed_stream(static_cast<std::uint64_t>(m) + 91));
        const std::vector<double> want = cpu_convolution(signal, filter);

        ckl::DeviceBuffer<float> ds(signal.size());
        ckl::DeviceBuffer<float> df(filter.size());
        ckl::DeviceBuffer<float> dout(want.size());
        ds.copy_from_host(signal);
        df.copy_from_host(filter);
        dout.zero();
        ckl::conv_direct_shared(ds.data(), df.data(), dout.data(), kN, m);
        CKL_CUDA_CHECK(cudaDeviceSynchronize());
        EXPECT_LE(relative_rms(dout.to_host(), want), ckl::fft_tolerance(4096)) << "m = " << m;
    }
}

TEST_F(ConvDirectKernels, TheConstantFormRefusesAnUnboundOwner) {
    ckl::DeviceBuffer<float> ds(256);
    ckl::DeviceBuffer<float> dout(256 + 8 - 1);
    ds.zero();
    int not_an_owner = 0;
    EXPECT_THROW(ckl::conv_direct_constant(ds.data(), dout.data(), 256, 8, &not_an_owner),
                 ckl::Error);
    EXPECT_FALSE(ckl::conv_direct_constant_bound(&not_an_owner));
    EXPECT_FALSE(ckl::conv_direct_constant_bound(nullptr));
}

TEST_F(ConvDirectKernels, TheConstantBankHoldsSixteenKilobytesOfTaps) {
    EXPECT_EQ(ckl::conv_direct_constant_max_taps(), 4096);
    EXPECT_EQ(ckl::conv_direct_constant_max_taps() * static_cast<int>(sizeof(float)), 16384);
}

}  // namespace
