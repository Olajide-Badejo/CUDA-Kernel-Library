// Reduction and scan correctness: every rung of both ladders, every operator the
// plan admits, every element type, at the lengths that break things.
//
// The length list is the one that matters rather than a set of round numbers.
// Zero, one and two catch the launcher that computes a grid of zero blocks or
// reads one element past the end. Thirty one, thirty two and thirty three catch
// a warp primitive that assumed a full warp. A thousand and twenty three, four
// and five catch a block scan that assumed a full block. The prime 1048573
// catches a tile decomposition that assumed the length divided by anything. And
// 2^28 catches the level recursion running out of levels, which nothing smaller
// does.
//
// The dataset list is the same idea. A single nonzero at the LAST index is the
// one that catches a dropped final tile, and it is the reason it is in the list:
// every other dataset here would still look right with the last tile missing.
// The mixed magnitude dataset of 1e30 and 1e-30 exists because the tolerance
// model below is a model for random data and says nothing about cancellation, so
// those runs are checked against a double precision reference scaled by the sum
// of magnitudes instead.
//
// The tolerance model is tol(N) = c * sqrt(N) * FLT_EPSILON on the residual
//
//     max_i |got_i - ref_i| / max_j |x_j|
//
// which is the error measured in units of the input scale. For random signed
// data the partial sums do a random walk of size sqrt(N) times that scale and
// the rounding error is machine epsilon times that walk, times a slowly growing
// factor for the depth of the reduction tree, so the ratio of the residual to
// sqrt(N) * FLT_EPSILON is O(1) and c absorbs the depth term. c was calibrated
// by ScanCalibration.DISABLED_ToleranceConstant below, which sweeps seeds,
// lengths and rungs and reports the worst ratio it saw; the committed value in
// ckl::scan_tolerance_c is that maximum with slack over it and no more, so the
// tolerance is still a gate that can fail. docs/scan.md records the run.
//
// Integer types and the exact operators (max, min, last nonzero) are compared
// exactly. There is no rounding in any of them, so a tolerance would only hide a
// defect.
//
// Two rungs do not fit that model and are not made to. ReduceAlgo::kAtomic and
// ReduceAlgo::kShuffle both finish through a single global atomic, so their last
// combine is a serial accumulation of however many terms reached that address:
// all N elements for r0, one partial per 256 element block for r2. The textbook
// bound for a recursive sum grows with the length of that chain and not with its
// square root, so both get tol_serial(L) = c_serial * L * FLT_EPSILON on the
// chain length L their rung produces, calibrated from the same sweep and against
// the same measure. That the two atomic rungs are less accurate as well as
// slower is a result, not an exemption: one accumulator is one long dependent
// chain, and a long dependent chain is where floating point error comes from.

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include <gtest/gtest.h>

#include "ckl/ckl.h"
#include "ckl/context.hpp"
#include "ckl/cuda_check.hpp"
#include "ckl/device_buffer.hpp"
#include "ckl/scan.hpp"
#include "gpu_environment.hpp"
#include "reference.hpp"

namespace {

using ckl::test::GpuTest;
using ckl::test::GpuTestWithParam;

// ---------------------------------------------------------------------------
// Lengths and datasets
// ---------------------------------------------------------------------------

// 1000003 is the non power of two mid size; 1048573 is the prime the spec names.
const std::vector<long long> kLengths = {0,    1,    2,    31,     32,      33,     1023,
                                         1024, 1025, 4095, 100000, 1000003, 1048573};

enum class Dataset {
    kRandom,
    kAllEqual,
    kAllZero,
    kLastOnly,
    kAlternating,
    kAdversarial,
};

const char* dataset_name(Dataset d) {
    switch (d) {
        case Dataset::kRandom:
            return "random";
        case Dataset::kAllEqual:
            return "all_equal";
        case Dataset::kAllZero:
            return "all_zero";
        case Dataset::kLastOnly:
            return "single_nonzero_at_last_index";
        case Dataset::kAlternating:
            return "alternating_signs";
        case Dataset::kAdversarial:
            return "adversarial_magnitudes";
    }
    return "unknown";
}

std::vector<float> make_dataset(Dataset kind, long long n, std::uint64_t seed) {
    const std::size_t count = static_cast<std::size_t>(n);
    switch (kind) {
        case Dataset::kRandom:
            return ckl::random_matrix(static_cast<int>(n), 1, seed);
        case Dataset::kAllEqual:
            return std::vector<float>(count, 0.75f);
        case Dataset::kAllZero:
            return std::vector<float>(count, 0.0f);
        case Dataset::kLastOnly: {
            std::vector<float> v(count, 0.0f);
            if (count > 0) {
                v[count - 1] = 3.5f;
            }
            return v;
        }
        case Dataset::kAlternating: {
            std::vector<float> v(count);
            for (std::size_t i = 0; i < count; ++i) {
                v[i] = (i % 2 == 0) ? 1.0f : -1.0f;
            }
            return v;
        }
        case Dataset::kAdversarial: {
            // Every element is huge or tiny, so the partial sums cancel down to
            // something far smaller than the terms they were built from.
            return ckl::mixed_magnitude_matrix(static_cast<int>(n), 1, seed, 1e30f);
        }
    }
    return std::vector<float>(count, 0.0f);
}

// ---------------------------------------------------------------------------
// Host references, in double
// ---------------------------------------------------------------------------

double apply_op(ckl::ScanOp op, double a, double b) {
    switch (op) {
        case ckl::ScanOp::kSum:
            return a + b;
        case ckl::ScanOp::kMax:
            return a > b ? a : b;
        case ckl::ScanOp::kMin:
            return a < b ? a : b;
        case ckl::ScanOp::kLastNonZero:
            return b != 0.0 ? b : a;
    }
    return a + b;
}

double op_identity(ckl::ScanOp op) {
    switch (op) {
        case ckl::ScanOp::kSum:
            return 0.0;
        case ckl::ScanOp::kMax:
            return -HUGE_VAL;
        case ckl::ScanOp::kMin:
            return HUGE_VAL;
        case ckl::ScanOp::kLastNonZero:
            return 0.0;
    }
    return 0.0;
}

std::vector<double> reference_scan(const std::vector<float>& in, ckl::ScanOp op, bool exclusive) {
    std::vector<double> out(in.size());
    double running = op_identity(op);
    for (std::size_t i = 0; i < in.size(); ++i) {
        const double value = static_cast<double>(in[i]);
        if (exclusive) {
            out[i] = running;
            running = apply_op(op, running, value);
        } else {
            running = apply_op(op, running, value);
            out[i] = running;
        }
    }
    return out;
}

double reference_reduce(const std::vector<float>& in, ckl::ScanOp op) {
    double running = op_identity(op);
    for (float x : in) {
        running = apply_op(op, running, static_cast<double>(x));
    }
    return running;
}

// The running sum of magnitudes, which is what bounds the rounding error of a
// prefix sum whatever the data does. It is the denominator the adversarial
// datasets are checked against, because their answers cancel to near zero and a
// relative error against the answer would be meaningless.
std::vector<double> magnitude_prefix(const std::vector<float>& in) {
    std::vector<double> out(in.size());
    double running = 0.0;
    for (std::size_t i = 0; i < in.size(); ++i) {
        running += std::fabs(static_cast<double>(in[i]));
        out[i] = running;
    }
    return out;
}

double largest_magnitude(const std::vector<float>& in) {
    double worst = 0.0;
    for (float x : in) {
        worst = std::max(worst, std::fabs(static_cast<double>(x)));
    }
    return worst > 0.0 ? worst : 1.0;
}

// The residual the tolerance model is stated in.
double scaled_residual(const std::vector<float>& got, const std::vector<double>& ref,
                       double scale) {
    double worst = 0.0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        worst = std::max(worst, std::fabs(static_cast<double>(got[i]) - ref[i]) / scale);
    }
    return worst;
}

// The adversarial residual: divided by the sum of magnitudes up to that index.
double magnitude_residual(const std::vector<float>& got, const std::vector<double>& ref,
                          const std::vector<double>& magnitudes) {
    double worst = 0.0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        const double denom = magnitudes[i] > 1e-300 ? magnitudes[i] : 1.0;
        worst = std::max(worst, std::fabs(static_cast<double>(got[i]) - ref[i]) / denom);
    }
    return worst;
}

// FNV-1a over the raw bytes of a result. The determinism artifact is this
// number: two runs that agree on it agree bit for bit.
std::uint64_t hash_bytes(const void* data, std::size_t bytes) {
    const auto* p = static_cast<const unsigned char*>(data);
    std::uint64_t h = 1469598103934665603ULL;
    for (std::size_t i = 0; i < bytes; ++i) {
        h ^= p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

std::string hex(std::uint64_t v) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(v));
    return std::string(buf);
}

// ---------------------------------------------------------------------------
// Device side scaffolding
// ---------------------------------------------------------------------------

// Runs one reduction and brings the answer back.
float run_reduce(ckl::ScanPlan& plan, ckl::ReduceAlgo algo, ckl::ScanOp op,
                 const ckl::DeviceBuffer<float>& in, ckl::ReduceAlgo* chosen) {
    ckl::DeviceBuffer<float> out(1);
    out.zero();
    const ckl::Status st = ckl::reduce(plan, algo, op, in.data(), out.data(), chosen, nullptr);
    EXPECT_EQ(st, ckl::Status::kSuccess) << ckl::reduce_algo_name(algo) << " "
                                         << ckl::scan_op_name(op) << ": " << ckl::status_string(st);
    CKL_CUDA_CHECK(cudaDeviceSynchronize());
    return out.to_host()[0];
}

std::vector<float> run_scan(ckl::ScanPlan& plan, ckl::ScanAlgo algo, ckl::ScanOp op, bool exclusive,
                            const ckl::DeviceBuffer<float>& in, long long n,
                            ckl::ScanAlgo* chosen) {
    ckl::DeviceBuffer<float> out(static_cast<std::size_t>(n));
    if (n > 0) {
        out.zero();
    }
    const ckl::Status st =
        ckl::scan(plan, algo, op, exclusive, in.data(), out.data(), chosen, nullptr);
    EXPECT_EQ(st, ckl::Status::kSuccess) << ckl::scan_algo_name(algo) << " "
                                         << ckl::scan_op_name(op) << ": " << ckl::status_string(st);
    CKL_CUDA_CHECK(cudaDeviceSynchronize());
    return out.to_host();
}

ckl::DeviceBuffer<float> upload(const std::vector<float>& host) {
    ckl::DeviceBuffer<float> d(host.size());
    if (!host.empty()) {
        d.copy_from_host(host);
    }
    return d;
}

// The serial model's constant, calibrated by the same disabled case that
// calibrated ckl::scan_tolerance_c, over the same sweep and against the same
// measure, with slack over the worst observed ratio and no more. docs/scan.md
// records the run.
constexpr double kSerialToleranceC = 16.0;

// How long the dependent chain of floating point additions is on this rung. Two
// rungs finish through a single global atomic, so their final combine is a
// serial accumulation whatever happened before it: r0 accumulates all N elements
// that way and r2 accumulates one partial per 256 element block. Everything else
// finishes through a tree.
long long serial_chain(ckl::ReduceAlgo algo, long long n) {
    switch (algo) {
        case ckl::ReduceAlgo::kAtomic:
            return n;
        case ckl::ReduceAlgo::kShuffle:
            return (n + 255) / 256;
        default:
            return 0;
    }
}

double tolerance_for(ckl::ReduceAlgo algo, long long n) {
    const long long chain = serial_chain(algo, n);
    if (chain > 0) {
        return kSerialToleranceC * static_cast<double>(chain) * static_cast<double>(FLT_EPSILON);
    }
    return ckl::scan_tolerance(n);
}

const ckl::ReduceAlgo kReduceRungs[] = {
    ckl::ReduceAlgo::kAtomic,        ckl::ReduceAlgo::kSharedTree, ckl::ReduceAlgo::kShuffle,
    ckl::ReduceAlgo::kVec4,          ckl::ReduceAlgo::kSinglePass, ckl::ReduceAlgo::kTwoPass,
    ckl::ReduceAlgo::kDeterministic, ckl::ReduceAlgo::kKahan,      ckl::ReduceAlgo::kCub,
};

const ckl::ScanAlgo kScanRungs[] = {
    ckl::ScanAlgo::kHillisSteele, ckl::ScanAlgo::kBlelloch,
    ckl::ScanAlgo::kThreeKernel,  ckl::ScanAlgo::kReduceThenScan,
    ckl::ScanAlgo::kLookback,     ckl::ScanAlgo::kDeterministic,
    ckl::ScanAlgo::kCub,
};

}  // namespace

// ---------------------------------------------------------------------------
// The reduction ladder
// ---------------------------------------------------------------------------

class ReduceLadder : public GpuTestWithParam<ckl::ReduceAlgo> {};

TEST_P(ReduceLadder, MatchesTheDoubleReferenceAtEveryLength) {
    const ckl::ReduceAlgo algo = GetParam();
    for (long long n : kLengths) {
        ckl::ScanPlan plan(n);
        ASSERT_TRUE(plan.supports_reduce(algo, ckl::ScanOp::kSum))
            << plan.reduce_refusal(algo, ckl::ScanOp::kSum);
        for (Dataset kind : {Dataset::kRandom, Dataset::kAllEqual, Dataset::kAllZero,
                             Dataset::kLastOnly, Dataset::kAlternating}) {
            const std::vector<float> host = make_dataset(kind, n, ckl::test::seed_stream(1));
            const ckl::DeviceBuffer<float> in = upload(host);
            ckl::ReduceAlgo chosen = ckl::ReduceAlgo::kAuto;
            const float got = run_reduce(plan, algo, ckl::ScanOp::kSum, in, &chosen);
            EXPECT_EQ(chosen, algo) << "a named rung must never be rerouted";

            const double want = reference_reduce(host, ckl::ScanOp::kSum);
            const double residual =
                std::fabs(static_cast<double>(got) - want) / largest_magnitude(host);
            EXPECT_LE(residual, tolerance_for(algo, n))
                << ckl::reduce_algo_name(algo) << " n=" << n << " " << dataset_name(kind)
                << " residual " << ckl::test::sci(residual) << " against "
                << ckl::test::sci(tolerance_for(algo, n));
        }
    }
}

TEST_P(ReduceLadder, AdversarialMagnitudesAgainstTheDoubleReference) {
    const ckl::ReduceAlgo algo = GetParam();
    for (long long n : {1023LL, 1024LL, 1048573LL}) {
        ckl::ScanPlan plan(n);
        const std::vector<float> host =
            make_dataset(Dataset::kAdversarial, n, ckl::test::seed_stream(2));
        const ckl::DeviceBuffer<float> in = upload(host);
        const float got = run_reduce(plan, algo, ckl::ScanOp::kSum, in, nullptr);
        const double want = reference_reduce(host, ckl::ScanOp::kSum);
        double magnitude = 0.0;
        for (float x : host) {
            magnitude += std::fabs(static_cast<double>(x));
        }
        const double residual = std::fabs(static_cast<double>(got) - want) / magnitude;
        EXPECT_LE(residual, tolerance_for(algo, n))
            << ckl::reduce_algo_name(algo) << " n=" << n << " adversarial residual "
            << ckl::test::sci(residual);
    }
}

TEST_P(ReduceLadder, EmptyInputWritesTheIdentityAndLaunchesNothingEmpty) {
    const ckl::ReduceAlgo algo = GetParam();
    ckl::ScanPlan plan(0);
    ckl::DeviceBuffer<float> out(1);
    // Poisoned, so a rung that returned success without writing fails here.
    const std::vector<float> poison(1, -12345.0f);
    out.copy_from_host(poison);
    ckl::ReduceAlgo chosen = ckl::ReduceAlgo::kAuto;
    const ckl::Status st =
        ckl::reduce(plan, algo, ckl::ScanOp::kSum, nullptr, out.data(), &chosen, nullptr);
    EXPECT_EQ(st, ckl::Status::kSuccess);
    CKL_CUDA_CHECK(cudaDeviceSynchronize());
    EXPECT_EQ(out.to_host()[0], 0.0f);
    EXPECT_EQ(chosen, algo);
    // A zero sized grid is cudaErrorInvalidConfiguration, which the fixture's
    // TearDown would report; getting here with no pending error is the check.
    EXPECT_EQ(cudaPeekAtLastError(), cudaSuccess);
}

INSTANTIATE_TEST_SUITE_P(Rungs, ReduceLadder, ::testing::ValuesIn(kReduceRungs),
                         [](const ::testing::TestParamInfo<ckl::ReduceAlgo>& entry) {
                             return std::string(ckl::reduce_algo_name(entry.param)).substr(1);
                         });

// ---------------------------------------------------------------------------
// The scan ladder
// ---------------------------------------------------------------------------

class ScanLadder : public GpuTestWithParam<ckl::ScanAlgo> {};

TEST_P(ScanLadder, MatchesTheDoubleReferenceAtEveryLength) {
    const ckl::ScanAlgo algo = GetParam();
    for (long long n : kLengths) {
        ckl::ScanPlan plan(n);
        for (bool exclusive : {false, true}) {
            for (Dataset kind : {Dataset::kRandom, Dataset::kAllEqual, Dataset::kAllZero,
                                 Dataset::kLastOnly, Dataset::kAlternating}) {
                const std::vector<float> host = make_dataset(kind, n, ckl::test::seed_stream(3));
                const ckl::DeviceBuffer<float> in = upload(host);
                ckl::ScanAlgo chosen = ckl::ScanAlgo::kAuto;
                const std::vector<float> got =
                    run_scan(plan, algo, ckl::ScanOp::kSum, exclusive, in, n, &chosen);
                EXPECT_EQ(chosen, algo) << "a named rung must never be rerouted";
                if (n == 0) {
                    continue;
                }
                const std::vector<double> want = reference_scan(host, ckl::ScanOp::kSum, exclusive);
                const double residual = scaled_residual(got, want, largest_magnitude(host));
                EXPECT_LE(residual, ckl::scan_tolerance(n))
                    << ckl::scan_algo_name(algo) << " n=" << n
                    << (exclusive ? " exclusive " : " inclusive ") << dataset_name(kind)
                    << " residual " << ckl::test::sci(residual) << " against "
                    << ckl::test::sci(ckl::scan_tolerance(n));
            }
        }
    }
}

TEST_P(ScanLadder, AdversarialMagnitudesAgainstTheDoubleReference) {
    const ckl::ScanAlgo algo = GetParam();
    for (long long n : {1025LL, 100000LL, 1048573LL}) {
        ckl::ScanPlan plan(n);
        const std::vector<float> host =
            make_dataset(Dataset::kAdversarial, n, ckl::test::seed_stream(4));
        const ckl::DeviceBuffer<float> in = upload(host);
        const std::vector<double> magnitudes = magnitude_prefix(host);
        for (bool exclusive : {false, true}) {
            const std::vector<float> got =
                run_scan(plan, algo, ckl::ScanOp::kSum, exclusive, in, n, nullptr);
            const std::vector<double> want = reference_scan(host, ckl::ScanOp::kSum, exclusive);
            const double residual = magnitude_residual(got, want, magnitudes);
            EXPECT_LE(residual, ckl::scan_tolerance(n))
                << ckl::scan_algo_name(algo) << " n=" << n
                << (exclusive ? " exclusive" : " inclusive") << " adversarial residual "
                << ckl::test::sci(residual);
        }
    }
}

// max, min and the non commutative operator, compared exactly: none of the three
// rounds anything, so a tolerance would only hide a defect. kLastNonZero is the
// one that matters here. It is associative, so every rung is entitled to run it,
// and it is not commutative, so a block scan, a level recursion or a look-back
// window that combined its operands in the wrong direction gets a different
// answer and fails, while every sum test it could have run would still pass.
TEST_P(ScanLadder, FunctorOperatorsExactly) {
    const ckl::ScanAlgo algo = GetParam();
    for (long long n : {33LL, 1025LL, 4095LL, 1048573LL}) {
        ckl::ScanPlan plan(n);
        for (ckl::ScanOp op : {ckl::ScanOp::kMax, ckl::ScanOp::kMin, ckl::ScanOp::kLastNonZero}) {
            ASSERT_TRUE(plan.supports_scan(algo, op)) << plan.scan_refusal(algo, op);
            for (bool exclusive : {false, true}) {
                // A dataset with real zeros in it, so kLastNonZero has something
                // to carry forward past.
                std::vector<float> host =
                    ckl::random_matrix(static_cast<int>(n), 1, ckl::test::seed_stream(5));
                for (std::size_t i = 0; i < host.size(); i += 3) {
                    host[i] = 0.0f;
                }
                const ckl::DeviceBuffer<float> in = upload(host);
                const std::vector<float> got = run_scan(plan, algo, op, exclusive, in, n, nullptr);
                const std::vector<double> want = reference_scan(host, op, exclusive);
                for (std::size_t i = 0; i < want.size(); ++i) {
                    // The identity of kMax and kMin is an infinity, and an
                    // exclusive scan writes it at index zero on purpose.
                    ASSERT_EQ(static_cast<double>(got[i]), want[i])
                        << ckl::scan_algo_name(algo) << " " << ckl::scan_op_name(op) << " n=" << n
                        << (exclusive ? " exclusive" : " inclusive") << " at index " << i;
                }
            }
        }
    }
}

TEST_P(ScanLadder, RunsInPlace) {
    const ckl::ScanAlgo algo = GetParam();
    for (long long n : {1025LL, 100000LL}) {
        ckl::ScanPlan plan(n);
        const std::vector<float> host =
            make_dataset(Dataset::kRandom, n, ckl::test::seed_stream(6));
        ckl::DeviceBuffer<float> buffer = upload(host);
        const ckl::Status st = ckl::scan(plan, algo, ckl::ScanOp::kSum, false, buffer.data(),
                                         buffer.data(), nullptr, nullptr);
        ASSERT_EQ(st, ckl::Status::kSuccess);
        CKL_CUDA_CHECK(cudaDeviceSynchronize());
        const std::vector<double> want = reference_scan(host, ckl::ScanOp::kSum, false);
        const double residual = scaled_residual(buffer.to_host(), want, largest_magnitude(host));
        EXPECT_LE(residual, ckl::scan_tolerance(n)) << ckl::scan_algo_name(algo) << " in place";
    }
}

TEST_P(ScanLadder, EmptyInputSucceedsAndLaunchesNothingEmpty) {
    const ckl::ScanAlgo algo = GetParam();
    ckl::ScanPlan plan(0);
    ckl::ScanAlgo chosen = ckl::ScanAlgo::kAuto;
    const ckl::Status st =
        ckl::scan(plan, algo, ckl::ScanOp::kSum, false, nullptr, nullptr, &chosen, nullptr);
    EXPECT_EQ(st, ckl::Status::kSuccess);
    EXPECT_EQ(chosen, algo);
    CKL_CUDA_CHECK(cudaDeviceSynchronize());
    EXPECT_EQ(cudaPeekAtLastError(), cudaSuccess);
}

INSTANTIATE_TEST_SUITE_P(Rungs, ScanLadder, ::testing::ValuesIn(kScanRungs),
                         [](const ::testing::TestParamInfo<ckl::ScanAlgo>& entry) {
                             return std::string(ckl::scan_algo_name(entry.param)).substr(1);
                         });

// ---------------------------------------------------------------------------
// Element types
// ---------------------------------------------------------------------------

class ScanDTypes : public GpuTest {};

TEST_F(ScanDTypes, DoubleSumOnEveryRung) {
    for (long long n : {1025LL, 1048573LL}) {
        ckl::ScanPlan plan(n, ckl::ScanDType::kF64);
        std::vector<double> host(static_cast<std::size_t>(n));
        const std::vector<float> seedv =
            ckl::random_matrix(static_cast<int>(n), 1, ckl::test::seed_stream(7));
        for (std::size_t i = 0; i < host.size(); ++i) {
            host[i] = static_cast<double>(seedv[i]);
        }
        ckl::DeviceBuffer<double> in(host.size());
        in.copy_from_host(host);
        long double exact = 0.0L;
        for (double x : host) {
            exact += static_cast<long double>(x);
        }

        for (ckl::ReduceAlgo algo : kReduceRungs) {
            ckl::DeviceBuffer<double> out(1);
            out.zero();
            ASSERT_EQ(
                ckl::reduce(plan, algo, ckl::ScanOp::kSum, in.data(), out.data(), nullptr, nullptr),
                ckl::Status::kSuccess);
            CKL_CUDA_CHECK(cudaDeviceSynchronize());
            const long double got = static_cast<long double>(out.to_host()[0]);
            // Double precision, so the tolerance model's FLT_EPSILON is replaced
            // by DBL_EPSILON at the same constant and the same sqrt(N) shape.
            const long double bound = static_cast<long double>(ckl::scan_tolerance_c()) *
                                      std::sqrt(static_cast<long double>(n)) *
                                      static_cast<long double>(DBL_EPSILON);
            const long double scale = 1.0L;
            EXPECT_LE(std::fabs(got - exact) / scale, bound)
                << ckl::reduce_algo_name(algo) << " fp64 n=" << n;
        }

        for (ckl::ScanAlgo algo : kScanRungs) {
            ckl::DeviceBuffer<double> out(host.size());
            out.zero();
            ASSERT_EQ(ckl::scan(plan, algo, ckl::ScanOp::kSum, false, in.data(), out.data(),
                                nullptr, nullptr),
                      ckl::Status::kSuccess);
            CKL_CUDA_CHECK(cudaDeviceSynchronize());
            const std::vector<double> got = out.to_host();
            long double running = 0.0L;
            long double worst = 0.0L;
            for (std::size_t i = 0; i < host.size(); ++i) {
                running += static_cast<long double>(host[i]);
                worst = std::max(worst, std::fabs(static_cast<long double>(got[i]) - running));
            }
            const long double bound = static_cast<long double>(ckl::scan_tolerance_c()) *
                                      std::sqrt(static_cast<long double>(n)) *
                                      static_cast<long double>(DBL_EPSILON);
            EXPECT_LE(worst, bound) << ckl::scan_algo_name(algo) << " fp64 n=" << n;
        }
    }
}

TEST_F(ScanDTypes, IntegerSumsAreExact) {
    for (long long n : {1025LL, 100000LL}) {
        // int32 and int64 both, and both compared exactly: an integer sum that
        // is one off is a defect, not a rounding difference.
        {
            ckl::ScanPlan plan(n, ckl::ScanDType::kI32);
            std::vector<int> host(static_cast<std::size_t>(n));
            for (std::size_t i = 0; i < host.size(); ++i) {
                host[i] = static_cast<int>(i % 7) - 3;
            }
            ckl::DeviceBuffer<int> in(host.size());
            in.copy_from_host(host);
            long long exact = 0;
            for (int x : host) {
                exact += x;
            }
            for (ckl::ReduceAlgo algo : kReduceRungs) {
                ckl::DeviceBuffer<int> out(1);
                out.zero();
                ASSERT_EQ(ckl::reduce(plan, algo, ckl::ScanOp::kSum, in.data(), out.data(), nullptr,
                                      nullptr),
                          ckl::Status::kSuccess);
                CKL_CUDA_CHECK(cudaDeviceSynchronize());
                EXPECT_EQ(static_cast<long long>(out.to_host()[0]), exact)
                    << ckl::reduce_algo_name(algo) << " int32 n=" << n;
            }
            for (ckl::ScanAlgo algo : kScanRungs) {
                ckl::DeviceBuffer<int> out(host.size());
                out.zero();
                ASSERT_EQ(ckl::scan(plan, algo, ckl::ScanOp::kSum, true, in.data(), out.data(),
                                    nullptr, nullptr),
                          ckl::Status::kSuccess);
                CKL_CUDA_CHECK(cudaDeviceSynchronize());
                const std::vector<int> got = out.to_host();
                int running = 0;
                for (std::size_t i = 0; i < host.size(); ++i) {
                    ASSERT_EQ(got[i], running)
                        << ckl::scan_algo_name(algo) << " int32 exclusive at " << i;
                    running += host[i];
                }
            }
        }
        {
            ckl::ScanPlan plan(n, ckl::ScanDType::kI64);
            std::vector<long long> host(static_cast<std::size_t>(n));
            for (std::size_t i = 0; i < host.size(); ++i) {
                host[i] = static_cast<long long>(i % 11) * 1000000007LL - 5LL;
            }
            ckl::DeviceBuffer<long long> in(host.size());
            in.copy_from_host(host);
            long long exact = 0;
            for (long long x : host) {
                exact += x;
            }
            for (ckl::ReduceAlgo algo : kReduceRungs) {
                ckl::DeviceBuffer<long long> out(1);
                out.zero();
                ASSERT_EQ(ckl::reduce(plan, algo, ckl::ScanOp::kSum, in.data(), out.data(), nullptr,
                                      nullptr),
                          ckl::Status::kSuccess);
                CKL_CUDA_CHECK(cudaDeviceSynchronize());
                EXPECT_EQ(out.to_host()[0], exact)
                    << ckl::reduce_algo_name(algo) << " int64 n=" << n;
            }
            for (ckl::ScanAlgo algo : kScanRungs) {
                ckl::DeviceBuffer<long long> out(host.size());
                out.zero();
                ASSERT_EQ(ckl::scan(plan, algo, ckl::ScanOp::kSum, false, in.data(), out.data(),
                                    nullptr, nullptr),
                          ckl::Status::kSuccess);
                CKL_CUDA_CHECK(cudaDeviceSynchronize());
                const std::vector<long long> got = out.to_host();
                long long running = 0;
                for (std::size_t i = 0; i < host.size(); ++i) {
                    running += host[i];
                    ASSERT_EQ(got[i], running)
                        << ckl::scan_algo_name(algo) << " int64 inclusive at " << i;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Refusals, dispatch and the plan's own answers
// ---------------------------------------------------------------------------

class ScanDispatch : public GpuTest {};

TEST_F(ScanDispatch, TheReductionLadderRefusesTheNonCommutativeOperatorWithAReason) {
    ckl::ScanPlan plan(1024);
    const std::vector<float> host = make_dataset(Dataset::kRandom, 1024, 1);
    const ckl::DeviceBuffer<float> in = upload(host);
    ckl::DeviceBuffer<float> out(1);
    for (ckl::ReduceAlgo algo : kReduceRungs) {
        EXPECT_FALSE(plan.supports_reduce(algo, ckl::ScanOp::kLastNonZero));
        const std::string why = plan.reduce_refusal(algo, ckl::ScanOp::kLastNonZero);
        EXPECT_NE(why.find("commutative"), std::string::npos)
            << "a refusal has to name its reason, not just refuse: " << why;
        ckl::ReduceAlgo chosen = ckl::ReduceAlgo::kAuto;
        const ckl::Status st = ckl::reduce(plan, algo, ckl::ScanOp::kLastNonZero, in.data(),
                                           out.data(), &chosen, nullptr);
        EXPECT_EQ(st, ckl::Status::kNotSupported);
        // chosen is written on failure too, which is the whole point of it.
        EXPECT_EQ(chosen, algo);
        char detail[512] = {0};
        ckl_last_error(detail, sizeof(detail));
        EXPECT_NE(std::string(detail).find("commutative"), std::string::npos);
    }
    // And the scan ladder takes it on every rung.
    for (ckl::ScanAlgo algo : kScanRungs) {
        EXPECT_TRUE(plan.supports_scan(algo, ckl::ScanOp::kLastNonZero))
            << plan.scan_refusal(algo, ckl::ScanOp::kLastNonZero);
    }
}

TEST_F(ScanDispatch, NonFloatTypesCarrySumOnly) {
    for (ckl::ScanDType dtype :
         {ckl::ScanDType::kF64, ckl::ScanDType::kI32, ckl::ScanDType::kI64}) {
        ckl::ScanPlan plan(1024, dtype);
        for (ckl::ScanOp op : {ckl::ScanOp::kMax, ckl::ScanOp::kMin}) {
            EXPECT_FALSE(plan.supports_scan(ckl::ScanAlgo::kLookback, op));
            EXPECT_NE(plan.scan_refusal(ckl::ScanAlgo::kLookback, op).find("32 bit float"),
                      std::string::npos);
        }
        EXPECT_TRUE(plan.supports_scan(ckl::ScanAlgo::kLookback, ckl::ScanOp::kSum));
    }
}

TEST_F(ScanDispatch, CompensationRefusesTheOperatorsThatHaveNothingToCompensate) {
    ckl::ScanPlan plan(1024);
    for (ckl::ScanOp op : {ckl::ScanOp::kMax, ckl::ScanOp::kMin}) {
        EXPECT_FALSE(plan.supports_reduce(ckl::ReduceAlgo::kKahan, op));
        EXPECT_NE(plan.reduce_refusal(ckl::ReduceAlgo::kKahan, op).find("exact"),
                  std::string::npos);
    }
    EXPECT_TRUE(plan.supports_reduce(ckl::ReduceAlgo::kKahan, ckl::ScanOp::kSum));
}

TEST_F(ScanDispatch, APlanWithoutCubStorageRefusesTheBaseline) {
    ckl::ScanPlanOptions opt;
    opt.build_cub = false;
    ckl::ScanPlan plan(1024, ckl::ScanDType::kF32, opt);
    EXPECT_FALSE(plan.supports_reduce(ckl::ReduceAlgo::kCub, ckl::ScanOp::kSum));
    EXPECT_FALSE(plan.supports_scan(ckl::ScanAlgo::kCub, ckl::ScanOp::kSum));
    EXPECT_NE(plan.reduce_refusal(ckl::ReduceAlgo::kCub, ckl::ScanOp::kSum).find("build_cub"),
              std::string::npos);
}

TEST_F(ScanDispatch, AutoChoosesAndSaysWhatItChose) {
    ckl::ScanPlan small(1024);
    ckl::ScanPlan large(1 << 20);
    EXPECT_EQ(small.query_scan(), ckl::ScanAlgo::kThreeKernel);
    EXPECT_EQ(large.query_scan(), ckl::ScanAlgo::kLookback);
    EXPECT_EQ(large.query_reduce(), ckl::ReduceAlgo::kSinglePass);

    const std::vector<float> host = make_dataset(Dataset::kRandom, 1 << 20, 1);
    const ckl::DeviceBuffer<float> in = upload(host);
    ckl::DeviceBuffer<float> out(host.size());
    ckl::ScanAlgo chosen = ckl::ScanAlgo::kAuto;
    ASSERT_EQ(ckl::scan(large, ckl::ScanAlgo::kAuto, ckl::ScanOp::kSum, false, in.data(),
                        out.data(), &chosen, nullptr),
              ckl::Status::kSuccess);
    CKL_CUDA_CHECK(cudaDeviceSynchronize());
    EXPECT_EQ(chosen, ckl::ScanAlgo::kLookback);
}

TEST_F(ScanDispatch, DeterministicModeIsAnOptionAndIsReported) {
    ckl::ScanPlanOptions opt;
    opt.deterministic = true;
    ckl::ScanPlan plan(1 << 20, ckl::ScanDType::kF32, opt);
    EXPECT_TRUE(plan.deterministic());
    EXPECT_EQ(plan.query_reduce(), ckl::ReduceAlgo::kDeterministic);
    EXPECT_EQ(plan.query_scan(), ckl::ScanAlgo::kDeterministic);

    const std::vector<float> host = make_dataset(Dataset::kRandom, 1 << 20, 1);
    const ckl::DeviceBuffer<float> in = upload(host);
    ckl::DeviceBuffer<float> out(1);
    ckl::ReduceAlgo chosen = ckl::ReduceAlgo::kAuto;
    ASSERT_EQ(ckl::reduce(plan, ckl::ReduceAlgo::kAuto, ckl::ScanOp::kSum, in.data(), out.data(),
                          &chosen, nullptr),
              ckl::Status::kSuccess);
    CKL_CUDA_CHECK(cudaDeviceSynchronize());
    EXPECT_EQ(chosen, ckl::ReduceAlgo::kDeterministic);
}

TEST_F(ScanDispatch, TheDeterministicGeometryIsAFunctionOfTheLengthAlone) {
    // The whole promise rests on this: the block count never comes from the
    // occupancy API, so it cannot change with what else is resident.
    for (long long n : {1LL << 16, 1LL << 20, 1LL << 24}) {
        ckl::ScanPlan a(n);
        ckl::ScanPlan b(n);
        EXPECT_EQ(a.deterministic_blocks(), b.deterministic_blocks());
        EXPECT_LE(a.deterministic_blocks(), 1024);
    }
}

TEST_F(ScanDispatch, TheTrafficModelsSayWhatEachRungMoves) {
    const long long n = 1 << 22;
    ckl::ScanPlan plan(n);
    const long long body = n * 4;
    EXPECT_EQ(plan.reduce_model_bytes(ckl::ReduceAlgo::kAtomic), body);
    EXPECT_EQ(plan.scan_model_bytes(ckl::ScanAlgo::kThreeKernel), 4 * body);
    EXPECT_EQ(plan.scan_model_bytes(ckl::ScanAlgo::kReduceThenScan), 3 * body);
    EXPECT_GT(plan.scan_model_bytes(ckl::ScanAlgo::kLookback), 2 * body);
    EXPECT_LT(plan.scan_model_bytes(ckl::ScanAlgo::kLookback), 3 * body);
    EXPECT_GT(plan.lookback_status_bytes(), 0);
}

// The C entry points, through the handle and the one entry plan cache behind
// them. Nothing but C crosses that boundary, so the enums are checked for range
// on the way in and chosen comes back as the C enum.
TEST_F(ScanDispatch, TheCEntryPointsAgreeWithTheCppOnes) {
    ckl_handle_t handle = nullptr;
    ASSERT_EQ(ckl_create(&handle), CKL_STATUS_SUCCESS);

    const long long n = 1 << 20;
    const std::vector<float> host = make_dataset(Dataset::kRandom, n, ckl::test::seed_stream(10));
    const ckl::DeviceBuffer<float> in = upload(host);
    ckl::DeviceBuffer<float> single(1);
    ckl::DeviceBuffer<float> out(host.size());
    single.zero();
    out.zero();

    ckl_reduce_algo_t taken = CKL_REDUCE_AUTO;
    ASSERT_EQ(ckl_reduce(handle, n, CKL_SCAN_F32, CKL_SCAN_OP_SUM, in.data(), single.data(),
                         CKL_REDUCE_VEC4, &taken),
              CKL_STATUS_SUCCESS);
    EXPECT_EQ(taken, CKL_REDUCE_VEC4);

    ckl_scan_algo_t scan_taken = CKL_SCAN_AUTO;
    ASSERT_EQ(ckl_scan(handle, n, CKL_SCAN_F32, CKL_SCAN_OP_SUM, 0, in.data(), out.data(),
                       CKL_SCAN_AUTO, &scan_taken),
              CKL_STATUS_SUCCESS);
    EXPECT_EQ(scan_taken, CKL_SCAN_LOOKBACK);
    ASSERT_EQ(ckl_device_synchronize(), CKL_STATUS_SUCCESS);

    const double reference = reference_reduce(host, ckl::ScanOp::kSum);
    const double scale = largest_magnitude(host);
    EXPECT_LE(std::fabs(static_cast<double>(single.to_host()[0]) - reference) / scale,
              ckl::scan_tolerance(n));
    const std::vector<double> want = reference_scan(host, ckl::ScanOp::kSum, false);
    EXPECT_LE(scaled_residual(out.to_host(), want, scale), ckl::scan_tolerance(n));

    // The refusal crosses as a status, and the detail is readable afterwards.
    const ckl_status_t refused = ckl_reduce(handle, n, CKL_SCAN_F32, CKL_SCAN_OP_LAST_NONZERO,
                                            in.data(), single.data(), CKL_REDUCE_VEC4, &taken);
    EXPECT_EQ(refused, CKL_STATUS_NOT_SUPPORTED);
    char detail[512] = {0};
    ckl_last_error(detail, sizeof(detail));
    EXPECT_NE(std::string(detail).find("commutative"), std::string::npos);

    // A malformed argument is caught before anything is dispatched. The enum
    // range checks are the same shape and are exercised from C, where an enum is
    // an int and a caller really can pass anything; forming an out of range
    // enumerator here would be unspecified in C++ and the compiler says so.
    EXPECT_EQ(ckl_scan(handle, -1, CKL_SCAN_F32, CKL_SCAN_OP_SUM, 0, in.data(), out.data(),
                       CKL_SCAN_AUTO, nullptr),
              CKL_STATUS_INVALID_VALUE);
    EXPECT_EQ(ckl_reduce(nullptr, n, CKL_SCAN_F32, CKL_SCAN_OP_SUM, in.data(), single.data(),
                         CKL_REDUCE_AUTO, nullptr),
              CKL_STATUS_INVALID_VALUE);

    EXPECT_EQ(ckl_destroy(handle), CKL_STATUS_SUCCESS);
}

TEST_F(ScanDispatch, TheCubVersionIsOnTheRecord) {
    const int version = ckl::ScanPlan::cub_version();
    EXPECT_GT(version, 0);
    std::printf("[ ckl      ] CUB_VERSION %d (CUB %d.%d.%d)\n", version, version / 100000,
                version / 100 % 1000, version % 100);
}

// ---------------------------------------------------------------------------
// Determinism
// ---------------------------------------------------------------------------

class ScanDeterminism : public GpuTest {};

TEST_F(ScanDeterminism, TenConsecutiveRunsAreBitIdenticalAt2To24) {
    const long long n = 1 << 24;
    ckl::ScanPlan plan(n);
    const std::vector<float> host = make_dataset(Dataset::kRandom, n, ckl::test::seed_stream(8));
    const ckl::DeviceBuffer<float> in = upload(host);

    std::uint64_t reduce_hash = 0;
    std::uint64_t scan_hash = 0;
    for (int run = 0; run < 10; ++run) {
        ckl::DeviceBuffer<float> out(1);
        out.zero();
        ASSERT_EQ(ckl::reduce(plan, ckl::ReduceAlgo::kDeterministic, ckl::ScanOp::kSum, in.data(),
                              out.data(), nullptr, nullptr),
                  ckl::Status::kSuccess);
        CKL_CUDA_CHECK(cudaDeviceSynchronize());
        const std::vector<float> value = out.to_host();
        const std::uint64_t h = hash_bytes(value.data(), value.size() * sizeof(float));
        if (run == 0) {
            reduce_hash = h;
        }
        EXPECT_EQ(h, reduce_hash) << "deterministic reduction changed on run " << run;

        ckl::DeviceBuffer<float> scanned(host.size());
        scanned.zero();
        ASSERT_EQ(ckl::scan(plan, ckl::ScanAlgo::kDeterministic, ckl::ScanOp::kSum, false,
                            in.data(), scanned.data(), nullptr, nullptr),
                  ckl::Status::kSuccess);
        CKL_CUDA_CHECK(cudaDeviceSynchronize());
        const std::vector<float> scanned_host = scanned.to_host();
        const std::uint64_t sh =
            hash_bytes(scanned_host.data(), scanned_host.size() * sizeof(float));
        if (run == 0) {
            scan_hash = sh;
        }
        EXPECT_EQ(sh, scan_hash) << "deterministic scan changed on run " << run;
    }

    // The artifact. It is a property of the arithmetic and not of the clock, so
    // it belongs in the test output and under experiments/results, and it is not
    // a performance number.
    std::printf("[ ckl      ] determinism artifact, seed %llu, n=%lld, fp32, kSum\n",
                static_cast<unsigned long long>(ckl::test::seed()), n);
    std::printf("[ ckl      ]   reduce kDeterministic fnv1a64 %s\n", hex(reduce_hash).c_str());
    std::printf("[ ckl      ]   scan   kDeterministic fnv1a64 %s\n", hex(scan_hash).c_str());
}

TEST_F(ScanDeterminism, TheNonDeterministicRungsAreStillCorrect) {
    // Not a determinism claim: the point is that r3 and r4 may legitimately give
    // a different answer from the deterministic rung, and both are inside the
    // tolerance. A test that demanded they agree bitwise would be asserting
    // something the library does not promise.
    const long long n = 1 << 22;
    ckl::ScanPlan plan(n);
    const std::vector<float> host = make_dataset(Dataset::kRandom, n, ckl::test::seed_stream(9));
    const ckl::DeviceBuffer<float> in = upload(host);
    const double reference = reference_reduce(host, ckl::ScanOp::kSum);
    const double scale = largest_magnitude(host);
    for (ckl::ReduceAlgo algo :
         {ckl::ReduceAlgo::kVec4, ckl::ReduceAlgo::kSinglePass, ckl::ReduceAlgo::kDeterministic}) {
        const float got = run_reduce(plan, algo, ckl::ScanOp::kSum, in, nullptr);
        EXPECT_LE(std::fabs(static_cast<double>(got) - reference) / scale, tolerance_for(algo, n))
            << ckl::reduce_algo_name(algo);
    }
}

// ---------------------------------------------------------------------------
// The compensated rung
// ---------------------------------------------------------------------------

TEST_F(ScanDeterminism, CompensationBeatsThePlainSumOnDataThatNeedsIt) {
    // A long run of small terms added to one large one: the classic case where
    // an uncompensated float sum loses every small term. The claim being checked
    // is that the compensated rung is closer to the exact answer, not that the
    // plain one is wrong.
    const long long n = 1 << 20;
    ckl::ScanPlan plan(n);
    std::vector<float> host(static_cast<std::size_t>(n), 1.0f / 3.0f);
    host[0] = 1.0e7f;
    const ckl::DeviceBuffer<float> in = upload(host);

    long double exact = 0.0L;
    for (float x : host) {
        exact += static_cast<long double>(x);
    }
    const long double plain = static_cast<long double>(
        run_reduce(plan, ckl::ReduceAlgo::kVec4, ckl::ScanOp::kSum, in, nullptr));
    const long double compensated = static_cast<long double>(
        run_reduce(plan, ckl::ReduceAlgo::kKahan, ckl::ScanOp::kSum, in, nullptr));
    const long double plain_error = std::fabs(plain - exact);
    const long double kahan_error = std::fabs(compensated - exact);
    std::printf("[ ckl      ] n=%lld plain error %.3Le, compensated error %.3Le\n", n, plain_error,
                kahan_error);
    EXPECT_LE(kahan_error, plain_error);
}

// ---------------------------------------------------------------------------
// The large case
// ---------------------------------------------------------------------------
//
// Named with Slow in it so ctest can select or skip the subset; tests/CMakeLists
// registers it as its own ctest entry with the slow label. The datasets here
// repeat a small pattern, so the reference is a closed form rather than a two
// gigabyte host array.

class ScanSlow : public GpuTest {};

namespace {

template <typename T>
void fill_repeating(ckl::DeviceBuffer<T>& device, const std::vector<T>& pattern, long long n) {
    const long long period = static_cast<long long>(pattern.size());
    std::vector<T> chunk;
    const long long chunk_elements = 1LL << 22;
    chunk.reserve(static_cast<std::size_t>(chunk_elements));
    for (long long i = 0; i < chunk_elements; ++i) {
        chunk.push_back(pattern[static_cast<std::size_t>(i % period)]);
    }
    long long done = 0;
    while (done < n) {
        const long long take = std::min<long long>(chunk_elements, n - done);
        CKL_CUDA_CHECK(cudaMemcpy(device.data() + done, chunk.data(),
                                  static_cast<std::size_t>(take) * sizeof(T),
                                  cudaMemcpyHostToDevice));
        done += take;
    }
}

}  // namespace

TEST_F(ScanSlow, ReductionAt2To28) {
    const long long n = 1LL << 28;
    // A repeating pattern whose period divides n, so the exact sum is a product
    // in double and no host array of a gigabyte is needed to know the answer.
    std::vector<float> pattern(1024);
    for (std::size_t i = 0; i < pattern.size(); ++i) {
        pattern[i] = static_cast<float>(static_cast<double>(i % 17) * 0.125 - 1.0);
    }
    double period_sum = 0.0;
    double period_magnitude = 0.0;
    for (float x : pattern) {
        period_sum += static_cast<double>(x);
        period_magnitude = std::max(period_magnitude, std::fabs(static_cast<double>(x)));
    }
    const double exact = period_sum * static_cast<double>(n / 1024);

    ckl::DeviceBuffer<float> in(static_cast<std::size_t>(n));
    fill_repeating(in, pattern, n);
    ckl::ScanPlan plan(n);
    for (ckl::ReduceAlgo algo : kReduceRungs) {
        const float got = run_reduce(plan, algo, ckl::ScanOp::kSum, in, nullptr);
        const double residual = std::fabs(static_cast<double>(got) - exact) / period_magnitude;
        EXPECT_LE(residual, tolerance_for(algo, n))
            << ckl::reduce_algo_name(algo) << " at 2^28 residual " << ckl::test::sci(residual);
    }
}

TEST_F(ScanSlow, ScanAt2To28) {
    // int32 so the answer is exact and checkable in closed form at every index,
    // which a float prefix sum of 2^28 terms is not.
    const long long n = 1LL << 28;
    const std::vector<int> pattern = {1};
    ckl::DeviceBuffer<int> in(static_cast<std::size_t>(n));
    fill_repeating(in, pattern, n);
    ckl::DeviceBuffer<int> out(static_cast<std::size_t>(n));
    ckl::ScanPlan plan(n, ckl::ScanDType::kI32);

    for (ckl::ScanAlgo algo : kScanRungs) {
        out.zero();
        ASSERT_EQ(ckl::scan(plan, algo, ckl::ScanOp::kSum, false, in.data(), out.data(), nullptr,
                            nullptr),
                  ckl::Status::kSuccess)
            << ckl::scan_algo_name(algo);
        CKL_CUDA_CHECK(cudaDeviceSynchronize());
        // Pulled back in windows rather than all at once, and every element of
        // every window is checked: sampling would let a wrong tile through.
        const long long window = 1LL << 22;
        std::vector<int> host(static_cast<std::size_t>(window));
        for (long long base = 0; base < n; base += window) {
            CKL_CUDA_CHECK(cudaMemcpy(host.data(), out.data() + base,
                                      static_cast<std::size_t>(window) * sizeof(int),
                                      cudaMemcpyDeviceToHost));
            for (long long i = 0; i < window; ++i) {
                ASSERT_EQ(static_cast<long long>(host[static_cast<std::size_t>(i)]), base + i + 1)
                    << ckl::scan_algo_name(algo) << " at index " << (base + i);
            }
        }
    }
}

TEST_F(ScanSlow, DeterminismAt2To28) {
    const long long n = 1LL << 28;
    std::vector<float> pattern(1024);
    for (std::size_t i = 0; i < pattern.size(); ++i) {
        pattern[i] = static_cast<float>(static_cast<double>(i % 13) * 0.0625 - 0.5);
    }
    ckl::DeviceBuffer<float> in(static_cast<std::size_t>(n));
    fill_repeating(in, pattern, n);
    ckl::ScanPlan plan(n);

    std::uint64_t first = 0;
    for (int run = 0; run < 10; ++run) {
        ckl::DeviceBuffer<float> out(1);
        out.zero();
        ASSERT_EQ(ckl::reduce(plan, ckl::ReduceAlgo::kDeterministic, ckl::ScanOp::kSum, in.data(),
                              out.data(), nullptr, nullptr),
                  ckl::Status::kSuccess);
        CKL_CUDA_CHECK(cudaDeviceSynchronize());
        const std::vector<float> value = out.to_host();
        const std::uint64_t h = hash_bytes(value.data(), value.size() * sizeof(float));
        if (run == 0) {
            first = h;
        }
        EXPECT_EQ(h, first) << "deterministic reduction changed on run " << run;
    }
    std::printf("[ ckl      ] determinism artifact, n=2^28, fp32, kSum, repeating pattern\n");
    std::printf("[ ckl      ]   reduce kDeterministic fnv1a64 %s\n", hex(first).c_str());
}

// ---------------------------------------------------------------------------
// The tolerance calibration
// ---------------------------------------------------------------------------
//
// Disabled by default because it is a calibration, not a check: it sweeps seeds,
// lengths and rungs, computes the residual the tolerance model is stated in, and
// reports the worst ratio of that residual to sqrt(N) * FLT_EPSILON. The
// committed ckl::scan_tolerance_c is that maximum with slack over it. Run it
// with
//
//   ./ckl_test_scan --gtest_also_run_disabled_tests
//       --gtest_filter=ScanCalibration.DISABLED_ToleranceConstant
//
// and record the output in docs/scan.md next to the value it produced.

class ScanCalibration : public GpuTest {};

TEST_F(ScanCalibration, DISABLED_ToleranceConstant) {
    const std::vector<long long> lengths = {1023,    1024,    4095,    100000,
                                            1000003, 1048573, 1 << 22, 1 << 24};
    double worst_parallel = 0.0;
    std::string worst_parallel_where;
    double worst_serial = 0.0;
    std::string worst_serial_where;
    std::printf("%-18s %10s %6s %14s %14s %8s\n", "rung", "n", "seed", "residual", "ratio",
                "model");
    for (long long n : lengths) {
        ckl::ScanPlan plan(n);
        for (std::uint64_t seed = 1; seed <= 8; ++seed) {
            const std::vector<float> host = make_dataset(Dataset::kRandom, n, seed);
            const ckl::DeviceBuffer<float> in = upload(host);
            const double scale = largest_magnitude(host);
            const double count = static_cast<double>(n);
            const double sqrt_unit = std::sqrt(count) * static_cast<double>(FLT_EPSILON);

            const double want_reduce = reference_reduce(host, ckl::ScanOp::kSum);
            for (ckl::ReduceAlgo algo : kReduceRungs) {
                const float got = run_reduce(plan, algo, ckl::ScanOp::kSum, in, nullptr);
                const double residual = std::fabs(static_cast<double>(got) - want_reduce) / scale;
                const long long chain = serial_chain(algo, n);
                const bool serial = chain > 0;
                const double ratio = residual / (serial ? static_cast<double>(chain) *
                                                              static_cast<double>(FLT_EPSILON)
                                                        : sqrt_unit);
                if (serial) {
                    if (ratio > worst_serial) {
                        worst_serial = ratio;
                        worst_serial_where = std::string("reduce ") + ckl::reduce_algo_name(algo);
                    }
                } else if (ratio > worst_parallel) {
                    worst_parallel = ratio;
                    worst_parallel_where = std::string("reduce ") + ckl::reduce_algo_name(algo);
                }
                std::printf("%-18s %10lld %6llu %14.6e %14.6e %8s\n", ckl::reduce_algo_name(algo),
                            n, static_cast<unsigned long long>(seed), residual, ratio,
                            serial ? "chain" : "sqrt(n)");
            }

            const std::vector<double> want_scan = reference_scan(host, ckl::ScanOp::kSum, false);
            for (ckl::ScanAlgo algo : kScanRungs) {
                const std::vector<float> got =
                    run_scan(plan, algo, ckl::ScanOp::kSum, false, in, n, nullptr);
                const double residual = scaled_residual(got, want_scan, scale);
                const double ratio = residual / sqrt_unit;
                if (ratio > worst_parallel) {
                    worst_parallel = ratio;
                    worst_parallel_where = std::string("scan ") + ckl::scan_algo_name(algo);
                }
                std::printf("%-18s %10lld %6llu %14.6e %14.6e %8s\n", ckl::scan_algo_name(algo), n,
                            static_cast<unsigned long long>(seed), residual, ratio, "sqrt(n)");
            }
        }
    }
    std::printf("\nworst sqrt(n) ratio %.6e at %s; committed c is %.3f\n", worst_parallel,
                worst_parallel_where.c_str(), ckl::scan_tolerance_c());
    std::printf("worst n ratio      %.6e at %s; committed c_serial is %.3f\n", worst_serial,
                worst_serial_where.c_str(), kSerialToleranceC);
    EXPECT_GT(ckl::scan_tolerance_c(), worst_parallel)
        << "the committed constant has to be above the worst observed ratio, and it should not "
           "be far above it: a tolerance nothing can fail is not a gate";
    EXPECT_GT(kSerialToleranceC, worst_serial);
}
