// CSR SpMV correctness: y = alpha * (A * x) + beta * y.
//
// v1 tested one skewed matrix and checked it with a single Frobenius number over
// the whole vector, which lets one wrong long row hide inside four thousand
// right short ones. Here the check is per row, against a double CPU reference,
// with the tolerance derived from that row's own nonzero count: a row with three
// nonzeros gets a much tighter bound than one with five hundred, which is the
// point.
//
// The matrix zoo is the second half. Empty rows, a matrix with no nonzeros at
// all, a single row, column indices out of order, and duplicated column indices
// that have to sum rather than overwrite: every one of those is a shape a real
// caller produces and none of them was covered.
//
// The third half, which arrived with the tuned family, is that every one of
// those runs against every variant through ckl::SpmvPlan, and that the plan
// reports which path it took. Six hand written kernels and two cuSPARSE
// algorithms all have to agree with the same double reference on the same zoo,
// and a variant that silently fell back to another one fails on the chosen
// assertion rather than passing on a right answer from the wrong kernel.
//
// The symmetric positive definite case is the one that cannot be faked. A is
// B transpose times B plus n times the identity at 1024 by 1024, held as CSR and
// as dense at the same time, and the sparse answer is checked against ckl::gemv
// on the dense copy as well as against the CPU reference, with x transpose A x
// asserted positive so a sign error inside a reduction cannot pass.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include <cuda_runtime.h>

#include <gtest/gtest.h>

#include "ckl/ckl.h"
#include "ckl/context.hpp"
#include "ckl/cuda_check.hpp"
#include "ckl/device_buffer.hpp"
#include "ckl/gemm.hpp"
#include "ckl/gemv.hpp"
#include "ckl/sparse.hpp"
#include "gpu_environment.hpp"
#include "reference.hpp"

namespace {

using ckl::test::GpuTest;
using ckl::test::GpuTestWithParam;

enum class Variant {
    kNaive,
    kWarp,
    kVector,
    kMerge,
    kSell,
    kBsr,
    kCusparseDefault,
    kCusparseAlg2,
};

const char* variant_name(Variant v) {
    switch (v) {
        case Variant::kNaive:
            return "naive";
        case Variant::kWarp:
            return "warp";
        case Variant::kVector:
            return "vector";
        case Variant::kMerge:
            return "merge";
        case Variant::kSell:
            return "sell";
        case Variant::kBsr:
            return "bsr";
        case Variant::kCusparseDefault:
            return "cusparse_default";
        case Variant::kCusparseAlg2:
            return "cusparse_alg2";
    }
    return "unknown";
}

ckl::SpmvAlgo variant_algo(Variant v) {
    switch (v) {
        case Variant::kNaive:
            return ckl::SpmvAlgo::kCsrNaive;
        case Variant::kWarp:
            return ckl::SpmvAlgo::kCsrWarp;
        case Variant::kVector:
            return ckl::SpmvAlgo::kCsrVector;
        case Variant::kMerge:
            return ckl::SpmvAlgo::kMerge;
        case Variant::kSell:
            return ckl::SpmvAlgo::kSellCSigma;
        case Variant::kBsr:
            return ckl::SpmvAlgo::kBsr;
        case Variant::kCusparseDefault:
            return ckl::SpmvAlgo::kCusparseDefault;
        case Variant::kCusparseAlg2:
            return ckl::SpmvAlgo::kCusparseAlg2;
    }
    return ckl::SpmvAlgo::kCusparseDefault;
}

const std::vector<Variant>& all_variants() {
    static const std::vector<Variant> v = {
        Variant::kNaive, Variant::kWarp, Variant::kVector,          Variant::kMerge,
        Variant::kSell,  Variant::kBsr,  Variant::kCusparseDefault, Variant::kCusparseAlg2};
    return v;
}

// The hand written variants only; cuSPARSE documents CSR with sorted, unique
// column indices, so the cases that deliberately break that contract are run
// against the six kernels this project owns.
const std::vector<Variant>& hand_variants() {
    static const std::vector<Variant> v = {Variant::kNaive, Variant::kWarp, Variant::kVector,
                                           Variant::kMerge, Variant::kSell, Variant::kBsr};
    return v;
}

struct Csr {
    std::vector<int> row_ptr;
    std::vector<int> col_idx;
    std::vector<float> values;
    int m = 0;
    int n = 0;
    int nnz = 0;
};

// Row degree small for most rows and large for a few. This is the distribution
// where the warp per row kernel earns its keep, and correctness has to hold for
// every variant regardless.
Csr make_skewed(int m, int n, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    Csr a;
    a.m = m;
    a.n = n;
    a.row_ptr.assign(static_cast<std::size_t>(m) + 1, 0);
    for (int i = 0; i < m; ++i) {
        int degree = 4 + static_cast<int>(rng() % 9);
        if (rng() % 50 == 0) {
            degree = std::min(n, 300 + static_cast<int>(rng() % 400));
        }
        degree = std::min(degree, n);
        std::vector<int> cols;
        while (static_cast<int>(cols.size()) < degree) {
            const int c = static_cast<int>(rng() % static_cast<std::uint64_t>(n));
            if (std::find(cols.begin(), cols.end(), c) == cols.end()) {
                cols.push_back(c);
            }
        }
        std::sort(cols.begin(), cols.end());
        for (int c : cols) {
            a.col_idx.push_back(c);
            a.values.push_back(ckl::bits_to_signed_unit(rng()));
        }
        a.row_ptr[static_cast<std::size_t>(i) + 1] =
            a.row_ptr[static_cast<std::size_t>(i)] + degree;
    }
    a.nnz = a.row_ptr.back();
    return a;
}

// Every third row is empty. An empty row still has to produce beta * y_i.
Csr make_empty_rows(int m, int n, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    Csr a;
    a.m = m;
    a.n = n;
    a.row_ptr.assign(static_cast<std::size_t>(m) + 1, 0);
    for (int i = 0; i < m; ++i) {
        const int degree = (i % 3 == 0) ? 0 : 1 + static_cast<int>(rng() % 6);
        std::vector<int> cols;
        while (static_cast<int>(cols.size()) < degree) {
            const int c = static_cast<int>(rng() % static_cast<std::uint64_t>(n));
            if (std::find(cols.begin(), cols.end(), c) == cols.end()) {
                cols.push_back(c);
            }
        }
        std::sort(cols.begin(), cols.end());
        for (int c : cols) {
            a.col_idx.push_back(c);
            a.values.push_back(ckl::bits_to_signed_unit(rng()));
        }
        a.row_ptr[static_cast<std::size_t>(i) + 1] =
            a.row_ptr[static_cast<std::size_t>(i)] + degree;
    }
    a.nnz = a.row_ptr.back();
    return a;
}

Csr make_zero_nnz(int m, int n) {
    Csr a;
    a.m = m;
    a.n = n;
    a.row_ptr.assign(static_cast<std::size_t>(m) + 1, 0);
    a.nnz = 0;
    return a;
}

Csr make_single_row(int n, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    Csr a;
    a.m = 1;
    a.n = n;
    a.row_ptr = {0, n};
    for (int j = 0; j < n; ++j) {
        a.col_idx.push_back(j);
        a.values.push_back(ckl::bits_to_signed_unit(rng()));
    }
    a.nnz = n;
    return a;
}

// Same matrix as make_skewed, with each row's column indices shuffled. CSR does
// not require sorted columns, and a kernel that assumed sorted order would be
// wrong here and right everywhere else.
Csr make_unsorted(int m, int n, std::uint64_t seed) {
    Csr a = make_skewed(m, n, seed);
    std::mt19937_64 rng(seed ^ 0xABCDEF01ULL);
    for (int i = 0; i < m; ++i) {
        const int start = a.row_ptr[static_cast<std::size_t>(i)];
        const int end = a.row_ptr[static_cast<std::size_t>(i) + 1];
        for (int k = end - 1; k > start; --k) {
            const int j =
                start + static_cast<int>(rng() % static_cast<std::uint64_t>(k - start + 1));
            std::swap(a.col_idx[static_cast<std::size_t>(k)],
                      a.col_idx[static_cast<std::size_t>(j)]);
            std::swap(a.values[static_cast<std::size_t>(k)], a.values[static_cast<std::size_t>(j)]);
        }
    }
    return a;
}

// Every row lists the same column twice with different values. The CSR contract
// is that duplicates sum, and a kernel that indexed by column instead of
// accumulating would drop one of them. The BSR conversion has to sum them too,
// which is the case that catches a block writer that assigns instead of adds.
Csr make_duplicates(int m, int n, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    Csr a;
    a.m = m;
    a.n = n;
    a.row_ptr.assign(static_cast<std::size_t>(m) + 1, 0);
    for (int i = 0; i < m; ++i) {
        const int base = static_cast<int>(rng() % static_cast<std::uint64_t>(n));
        const int other = static_cast<int>(rng() % static_cast<std::uint64_t>(n));
        const int cols[4] = {base, other, base, base};
        for (int c : cols) {
            a.col_idx.push_back(c);
            a.values.push_back(ckl::bits_to_signed_unit(rng()));
        }
        a.row_ptr[static_cast<std::size_t>(i) + 1] = a.row_ptr[static_cast<std::size_t>(i)] + 4;
    }
    a.nnz = a.row_ptr.back();
    return a;
}

// Uniform degree. The width heuristic reads this off the histogram, and the two
// widths below sit on opposite sides of the point where a full warp per row
// stops wasting lanes.
Csr make_uniform(int m, int n, int degree, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    Csr a;
    a.m = m;
    a.n = n;
    a.row_ptr.assign(static_cast<std::size_t>(m) + 1, 0);
    const int deg = std::min(degree, n);
    for (int i = 0; i < m; ++i) {
        std::vector<int> cols;
        while (static_cast<int>(cols.size()) < deg) {
            const int c = static_cast<int>(rng() % static_cast<std::uint64_t>(n));
            if (std::find(cols.begin(), cols.end(), c) == cols.end()) {
                cols.push_back(c);
            }
        }
        std::sort(cols.begin(), cols.end());
        for (int c : cols) {
            a.col_idx.push_back(c);
            a.values.push_back(ckl::bits_to_signed_unit(rng()));
        }
        a.row_ptr[static_cast<std::size_t>(i) + 1] = a.row_ptr[static_cast<std::size_t>(i)] + deg;
    }
    a.nnz = a.row_ptr.back();
    return a;
}

// Uniform except for one row long enough to trip the power law rule.
Csr make_power_law(int m, int n, std::uint64_t seed) {
    Csr a = make_uniform(m, n, 4, seed);
    Csr out;
    out.m = m;
    out.n = n;
    out.row_ptr.assign(static_cast<std::size_t>(m) + 1, 0);
    std::mt19937_64 rng(seed ^ 0x5151ULL);
    for (int i = 0; i < m; ++i) {
        const int start = a.row_ptr[static_cast<std::size_t>(i)];
        const int end = a.row_ptr[static_cast<std::size_t>(i) + 1];
        for (int k = start; k < end; ++k) {
            out.col_idx.push_back(a.col_idx[static_cast<std::size_t>(k)]);
            out.values.push_back(a.values[static_cast<std::size_t>(k)]);
        }
        if (i == m / 2) {
            for (int c = 0; c < n; ++c) {
                out.col_idx.push_back(c);
                out.values.push_back(ckl::bits_to_signed_unit(rng()));
            }
        }
        out.row_ptr[static_cast<std::size_t>(i) + 1] = static_cast<int>(out.col_idx.size());
    }
    // The long row's columns were appended after the short row's, so this one
    // row is unsorted. Sorting it keeps the matrix legal for cuSPARSE too.
    for (int i = 0; i < m; ++i) {
        const int start = out.row_ptr[static_cast<std::size_t>(i)];
        const int end = out.row_ptr[static_cast<std::size_t>(i) + 1];
        std::vector<std::pair<int, float>> pairs;
        for (int k = start; k < end; ++k) {
            pairs.emplace_back(out.col_idx[static_cast<std::size_t>(k)],
                               out.values[static_cast<std::size_t>(k)]);
        }
        std::sort(pairs.begin(), pairs.end(),
                  [](const auto& l, const auto& r) { return l.first < r.first; });
        for (int k = start; k < end; ++k) {
            out.col_idx[static_cast<std::size_t>(k)] =
                pairs[static_cast<std::size_t>(k - start)].first;
            out.values[static_cast<std::size_t>(k)] =
                pairs[static_cast<std::size_t>(k - start)].second;
        }
    }
    out.nnz = out.row_ptr.back();
    return out;
}

enum class MatrixKind {
    kSkewed,
    kEmptyRows,
    kZeroNnz,
    kSingleRow,
    kUnsorted,
    kDuplicates,
};

const char* kind_name(MatrixKind k) {
    switch (k) {
        case MatrixKind::kSkewed:
            return "skewed";
        case MatrixKind::kEmptyRows:
            return "empty_rows";
        case MatrixKind::kZeroNnz:
            return "zero_nnz";
        case MatrixKind::kSingleRow:
            return "single_row";
        case MatrixKind::kUnsorted:
            return "unsorted_cols";
        case MatrixKind::kDuplicates:
            return "duplicate_cols";
    }
    return "unknown";
}

Csr make_matrix(MatrixKind k, std::uint64_t seed) {
    switch (k) {
        case MatrixKind::kSkewed:
            return make_skewed(1024, 1024, seed);
        case MatrixKind::kEmptyRows:
            return make_empty_rows(512, 256, seed);
        case MatrixKind::kZeroNnz:
            return make_zero_nnz(256, 256);
        case MatrixKind::kSingleRow:
            return make_single_row(2048, seed);
        case MatrixKind::kUnsorted:
            return make_unsorted(512, 512, seed);
        case MatrixKind::kDuplicates:
            return make_duplicates(256, 128, seed);
    }
    return make_zero_nnz(1, 1);
}

struct SpmvRef {
    std::vector<double> y;
    std::vector<double> scale;
};

SpmvRef spmv_reference(const Csr& a, const std::vector<float>& x, const std::vector<float>& y0,
                       float alpha, float beta) {
    SpmvRef ref;
    ref.y.assign(static_cast<std::size_t>(a.m), 0.0);
    ref.scale.assign(static_cast<std::size_t>(a.m), 0.0);
    for (int i = 0; i < a.m; ++i) {
        double acc = 0.0;
        double mag = 0.0;
        for (int k = a.row_ptr[static_cast<std::size_t>(i)];
             k < a.row_ptr[static_cast<std::size_t>(i) + 1]; ++k) {
            const double v = static_cast<double>(a.values[static_cast<std::size_t>(k)]);
            const double xv = static_cast<double>(
                x[static_cast<std::size_t>(a.col_idx[static_cast<std::size_t>(k)])]);
            acc += v * xv;
            mag += std::fabs(v) * std::fabs(xv);
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

// Every device side buffer one matrix needs, uploaded once. The plan holds
// pointers into these and never copies them, so this has to outlive the plan.
struct DeviceCsr {
    ckl::DeviceBuffer<int> row_ptr;
    ckl::DeviceBuffer<int> col_idx;
    ckl::DeviceBuffer<float> values;

    explicit DeviceCsr(const Csr& a)
        : row_ptr(a.row_ptr.size()), col_idx(a.col_idx.size()), values(a.values.size()) {
        row_ptr.copy_from_host(a.row_ptr);
        if (!a.col_idx.empty()) {
            col_idx.copy_from_host(a.col_idx);
            values.copy_from_host(a.values);
        }
    }

    ckl::SpmvCsr view(const Csr& a) const {
        ckl::SpmvCsr out;
        out.row_ptr = row_ptr.data();
        out.col_idx = col_idx.data();
        out.values = values.data();
        out.m = a.m;
        out.n = a.n;
        out.nnz = a.nnz;
        return out;
    }
};

// One variant, through the plan, with the chosen out-param asserted. A variant
// that quietly ran another kernel fails here rather than passing on the right
// answer from the wrong path.
std::vector<float> run_spmv(Variant v, const Csr& a, const std::vector<float>& x,
                            const std::vector<float>& y0, float alpha, float beta) {
    const DeviceCsr device(a);
    ckl::DeviceBuffer<float> dx(x.size());
    ckl::DeviceBuffer<float> dy(y0.size());
    dx.copy_from_host(x);
    dy.copy_from_host(y0);

    ckl::SpmvPlan plan(device.view(a));
    const ckl::SpmvAlgo want = variant_algo(v);
    ckl::SpmvAlgo chosen = ckl::SpmvAlgo::kAuto;
    const ckl::Status st = ckl::spmv(plan, want, alpha, dx.data(), beta, dy.data(), &chosen);
    EXPECT_EQ(st, ckl::Status::kSuccess)
        << variant_name(v) << " returned " << ckl::status_string(st);
    EXPECT_EQ(chosen, want) << variant_name(v) << " asked for " << ckl::spmv_algo_name(want)
                            << " and the plan reports " << ckl::spmv_algo_name(chosen);
    CKL_CUDA_CHECK(cudaDeviceSynchronize());
    return dy.to_host();
}

// Per row, with that row's own nonzero count setting the bound. The flat matrix
// wide number this replaces could not tell a wrong five hundred element row from
// rounding noise in the other thousand.
void expect_rows_match(Variant v, const Csr& a, const std::vector<float>& out, const SpmvRef& ref) {
    for (int i = 0; i < a.m; ++i) {
        const int nnz_row =
            a.row_ptr[static_cast<std::size_t>(i) + 1] - a.row_ptr[static_cast<std::size_t>(i)];
        const double denom = ref.scale[static_cast<std::size_t>(i)] > 1e-30
                                 ? ref.scale[static_cast<std::size_t>(i)]
                                 : 1.0;
        const double diff = std::fabs(static_cast<double>(out[static_cast<std::size_t>(i)]) -
                                      ref.y[static_cast<std::size_t>(i)]);
        ASSERT_LT(diff / denom, ckl::tol(nnz_row))
            << variant_name(v) << " row " << i << " with " << nnz_row << " nonzeros";
    }
}

void expect_spmv_matches(Variant v, const Csr& a, float alpha, float beta,
                         std::uint64_t stream_id) {
    const auto x = ckl::random_matrix(a.n, 1, ckl::test::seed_stream(stream_id));
    const auto y0 = ckl::random_matrix(a.m, 1, ckl::test::seed_stream(stream_id + 1));
    const auto out = run_spmv(v, a, x, y0, alpha, beta);
    ASSERT_TRUE(ckl::all_finite(out)) << variant_name(v) << " produced a non finite result";
    const SpmvRef ref = spmv_reference(a, x, y0, alpha, beta);
    expect_rows_match(v, a, out, ref);
}

// ---------------------------------------------------------------------------

using KindParam = std::tuple<Variant, MatrixKind>;

class SpmvMatrix : public GpuTestWithParam<KindParam> {};

std::string kind_param_name(const ::testing::TestParamInfo<KindParam>& info) {
    return std::string(variant_name(std::get<0>(info.param))) + "_" +
           kind_name(std::get<1>(info.param));
}

TEST_P(SpmvMatrix, EveryRowMatchesTheDoubleReference) {
    const auto [v, kind] = GetParam();
    const Csr a = make_matrix(kind, ckl::test::seed_stream(4101));
    expect_spmv_matches(v, a, 1.1f, 0.3f, 4102);
}

// The shapes every implementation agrees on.
INSTANTIATE_TEST_SUITE_P(
    Csr, SpmvMatrix,
    ::testing::Combine(::testing::ValuesIn(all_variants()),
                       ::testing::Values(MatrixKind::kSkewed, MatrixKind::kEmptyRows,
                                         MatrixKind::kZeroNnz, MatrixKind::kSingleRow)),
    kind_param_name);

// Unsorted and duplicated column indices are legal CSR but outside what cuSPARSE
// promises, so only the kernels this project owns are held to them.
INSTANTIATE_TEST_SUITE_P(CsrHandWritten, SpmvMatrix,
                         ::testing::Combine(::testing::ValuesIn(hand_variants()),
                                            ::testing::Values(MatrixKind::kUnsorted,
                                                              MatrixKind::kDuplicates)),
                         kind_param_name);

struct AlphaBeta {
    float alpha;
    float beta;
    const char* label;
};

using AbParam = std::tuple<Variant, AlphaBeta>;

class SpmvAlphaBeta : public GpuTestWithParam<AbParam> {};

std::string ab_param_name(const ::testing::TestParamInfo<AbParam>& info) {
    return std::string(variant_name(std::get<0>(info.param))) + "_" + std::get<1>(info.param).label;
}

TEST_P(SpmvAlphaBeta, SkewedAndEmptyRows) {
    const auto [v, ab] = GetParam();
    {
        SCOPED_TRACE("skewed");
        expect_spmv_matches(v, make_matrix(MatrixKind::kSkewed, ckl::test::seed_stream(4201)),
                            ab.alpha, ab.beta, 4202);
    }
    {
        SCOPED_TRACE("empty rows");
        expect_spmv_matches(v, make_matrix(MatrixKind::kEmptyRows, ckl::test::seed_stream(4211)),
                            ab.alpha, ab.beta, 4212);
    }
}

INSTANTIATE_TEST_SUITE_P(Csr, SpmvAlphaBeta,
                         ::testing::Combine(::testing::ValuesIn(all_variants()),
                                            ::testing::Values(AlphaBeta{1.0f, 0.0f, "a1_b0"},
                                                              AlphaBeta{0.0f, 1.0f, "a0_b1"},
                                                              AlphaBeta{1.0f, 1.0f, "a1_b1"},
                                                              AlphaBeta{2.5f, -0.75f,
                                                                        "a2p5_bm0p75"},
                                                              AlphaBeta{0.0f, 0.0f, "a0_b0"})),
                         ab_param_name);

class SpmvVariant : public GpuTestWithParam<Variant> {};

std::string variant_param_name(const ::testing::TestParamInfo<Variant>& info) {
    return variant_name(info.param);
}

TEST_P(SpmvVariant, BetaZeroDoesNotReadY) {
    const Variant v = GetParam();
    const Csr a = make_matrix(MatrixKind::kSkewed, ckl::test::seed_stream(4301));
    const auto x = ckl::random_matrix(a.n, 1, ckl::test::seed_stream(4302));
    for (float fill :
         {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity()}) {
        const std::vector<float> poisoned(static_cast<std::size_t>(a.m), fill);
        const auto out = run_spmv(v, a, x, poisoned, 1.0f, 0.0f);
        ASSERT_TRUE(ckl::all_finite(out)) << variant_name(v) << " read y while beta was zero";
        const std::vector<float> zeros(static_cast<std::size_t>(a.m), 0.0f);
        const SpmvRef ref = spmv_reference(a, x, zeros, 1.0f, 0.0f);
        expect_rows_match(v, a, out, ref);
    }
}

TEST_P(SpmvVariant, ZeroRowsLeaveTheOutputAlone) {
    const float canary = -31337.5f;
    Csr a = make_matrix(MatrixKind::kSkewed, ckl::test::seed_stream(4401));
    const auto x = ckl::random_matrix(a.n, 1, ckl::test::seed_stream(4402));
    const std::vector<float> y0(static_cast<std::size_t>(a.m), canary);
    const int real_m = a.m;
    a.m = 0;  // the buffers stay full size, so anything written is visible
    const auto out = run_spmv(GetParam(), a, x, y0, 1.1f, 0.3f);
    a.m = real_m;
    for (std::size_t i = 0; i < out.size(); ++i) {
        ASSERT_EQ(out[i], canary) << "element " << i << " was written for a zero row matrix";
    }
}

// The generator seeds. One --seed moves all five, so a CI job that runs the
// suite under five seeds covers twenty five distinct matrices per variant.
TEST_P(SpmvVariant, FiveGeneratorSeeds) {
    const Variant v = GetParam();
    for (int s = 1; s <= 5; ++s) {
        SCOPED_TRACE("generator seed " + std::to_string(s));
        const Csr a = make_skewed(512, 512, ckl::test::seed_stream(4500 + s));
        expect_spmv_matches(v, a, 1.25f, -0.5f, 4600 + s);
    }
}

INSTANTIATE_TEST_SUITE_P(Csr, SpmvVariant, ::testing::ValuesIn(all_variants()), variant_param_name);

// ---------------------------------------------------------------------------
// The symmetric positive definite case
// ---------------------------------------------------------------------------

// A = B^T B + n I at 1024 by 1024, built on the device because a host side
// 1024 cubed loop is a second of CI time per run for no extra confidence.
std::vector<float> make_spd_dense(int n, std::uint64_t seed) {
    const auto b = ckl::random_matrix(n, n, seed);
    ckl::Context ctx;
    ckl::DeviceBuffer<float> db(b.size());
    ckl::DeviceBuffer<float> da(b.size());
    db.copy_from_host(b);

    ckl::GemmDesc d;
    d.layout = ckl::Layout::kRowMajor;
    d.op_a = ckl::Op::kT;
    d.m = n;
    d.n = n;
    d.k = n;
    d.lda = n;
    d.ldb = n;
    d.ldc = n;
    d.algo = ckl::Algo::kCublas;
    const float one = 1.0f;
    const float zero = 0.0f;
    const ckl::Status st = ckl::gemm(ctx, d, &one, db.data(), db.data(), &zero, da.data(), nullptr);
    EXPECT_EQ(st, ckl::Status::kSuccess) << "building B^T B failed: " << ckl::status_string(st);
    CKL_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> dense = da.to_host();
    for (int i = 0; i < n; ++i) {
        dense[static_cast<std::size_t>(i) * n + i] += static_cast<float>(n);
    }
    return dense;
}

Csr dense_to_csr(const std::vector<float>& dense, int m, int n) {
    Csr a;
    a.m = m;
    a.n = n;
    a.row_ptr.assign(static_cast<std::size_t>(m) + 1, 0);
    a.col_idx.reserve(dense.size());
    a.values.reserve(dense.size());
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            a.col_idx.push_back(j);
            a.values.push_back(dense[static_cast<std::size_t>(i) * n + j]);
        }
        a.row_ptr[static_cast<std::size_t>(i) + 1] = (i + 1) * n;
    }
    a.nnz = a.row_ptr.back();
    return a;
}

class SpmvSpd : public GpuTestWithParam<Variant> {};

TEST_P(SpmvSpd, MatchesTheDenseGemvAndStaysPositiveDefinite) {
    constexpr int kN = 1024;
    const Variant v = GetParam();
    const std::vector<float> dense = make_spd_dense(kN, ckl::test::seed_stream(4701));
    const Csr a = dense_to_csr(dense, kN, kN);
    const auto x = ckl::random_matrix(kN, 1, ckl::test::seed_stream(4702));
    const auto y0 = ckl::random_matrix(kN, 1, ckl::test::seed_stream(4703));
    const float alpha = 1.0f;
    const float beta = 0.0f;

    const auto sparse = run_spmv(v, a, x, y0, alpha, beta);
    ASSERT_TRUE(ckl::all_finite(sparse)) << variant_name(v) << " produced a non finite result";

    // The double reference, per row, at the full contraction length.
    const SpmvRef ref = spmv_reference(a, x, y0, alpha, beta);
    expect_rows_match(v, a, sparse, ref);

    // The same product through the dense path. Two independent implementations
    // of A times x agreeing is what makes this case worth its runtime.
    ckl::DeviceBuffer<float> dd(dense.size());
    ckl::DeviceBuffer<float> dx(x.size());
    ckl::DeviceBuffer<float> dy(y0.size());
    dd.copy_from_host(dense);
    dx.copy_from_host(x);
    dy.copy_from_host(y0);
    ckl::gemv_cublas(dd.data(), dx.data(), dy.data(), kN, kN, alpha, beta, nullptr);
    CKL_CUDA_CHECK(cudaDeviceSynchronize());
    const auto gemv = dy.to_host();
    for (int i = 0; i < kN; ++i) {
        const double denom = ref.scale[static_cast<std::size_t>(i)] > 1e-30
                                 ? ref.scale[static_cast<std::size_t>(i)]
                                 : 1.0;
        const double diff = std::fabs(static_cast<double>(sparse[static_cast<std::size_t>(i)]) -
                                      static_cast<double>(gemv[static_cast<std::size_t>(i)]));
        ASSERT_LT(diff / denom, ckl::tol(kN))
            << variant_name(v) << " and ckl::gemv disagree at row " << i;
    }

    // x^T A x for a positive definite A is positive. A reduction that lost a
    // sign somewhere can still land inside the elementwise tolerance on a
    // cancelling dataset; it cannot pass this.
    double quadratic = 0.0;
    for (int i = 0; i < kN; ++i) {
        quadratic += static_cast<double>(x[static_cast<std::size_t>(i)]) *
                     static_cast<double>(sparse[static_cast<std::size_t>(i)]);
    }
    ASSERT_GT(quadratic, 0.0) << variant_name(v) << " gives a non positive x^T A x of "
                              << ckl::test::sci(quadratic);
}

INSTANTIATE_TEST_SUITE_P(Spd, SpmvSpd, ::testing::ValuesIn(all_variants()), variant_param_name);

// ---------------------------------------------------------------------------
// The plan itself
// ---------------------------------------------------------------------------

class SpmvPlanState : public GpuTest {};

TEST_F(SpmvPlanState, HistogramCountsEveryRow) {
    const Csr a = make_matrix(MatrixKind::kSkewed, ckl::test::seed_stream(4801));
    const DeviceCsr device(a);
    const ckl::SpmvPlan plan(device.view(a));
    long long counted = 0;
    for (int b = 0; b < ckl::SpmvPlan::histogram_buckets(); ++b) {
        counted += plan.histogram(b);
    }
    EXPECT_EQ(counted, a.m);
    EXPECT_NEAR(plan.mean_nnz_per_row(), static_cast<double>(a.nnz) / a.m, 1e-9);

    int longest = 0;
    for (int i = 0; i < a.m; ++i) {
        longest = std::max(longest, a.row_ptr[static_cast<std::size_t>(i) + 1] -
                                        a.row_ptr[static_cast<std::size_t>(i)]);
    }
    EXPECT_EQ(plan.max_nnz_per_row(), longest);
}

TEST_F(SpmvPlanState, SellPaddingIsAtLeastOneAndTheSliceHeightIsThirtyTwo) {
    EXPECT_EQ(ckl::SpmvPlan::sell_slice_height(), 32);
    const Csr a = make_matrix(MatrixKind::kSkewed, ckl::test::seed_stream(4811));
    const DeviceCsr device(a);
    const ckl::SpmvPlan plan(device.view(a));
    EXPECT_GE(plan.sell_nnz_padded(), a.nnz);
    EXPECT_GE(plan.sell_padding_ratio(), 1.0);
    // v1's generator puts two percent of its rows at 300 to 700 nonzeros against
    // a mean near 8, so the plan calls it power law and gives it the wide sort
    // window. That is worth stating in a test rather than in a comment: the
    // matrix the v1 numbers came from is a power law matrix, not a regular one.
    EXPECT_TRUE(plan.power_law());
    EXPECT_EQ(plan.sigma(), 8192);

    // A uniform degree matrix takes the other branch.
    const Csr regular = make_uniform(2048, 2048, 7, ckl::test::seed_stream(4812));
    const DeviceCsr regular_device(regular);
    const ckl::SpmvPlan regular_plan(regular_device.view(regular));
    EXPECT_FALSE(regular_plan.power_law());
    EXPECT_EQ(regular_plan.sigma(), 256);
    EXPECT_EQ(regular_plan.vector_width(), 8);
}

TEST_F(SpmvPlanState, AWiderSigmaPadsNoMoreThanANarrowOne) {
    const Csr a = make_matrix(MatrixKind::kSkewed, ckl::test::seed_stream(4821));
    const DeviceCsr device(a);
    ckl::SpmvPlanOptions narrow;
    narrow.sigma = 1;
    narrow.build_bsr = false;
    ckl::SpmvPlanOptions wide;
    wide.sigma = 8192;
    wide.build_bsr = false;
    const ckl::SpmvPlan tight(device.view(a), narrow);
    const ckl::SpmvPlan loose(device.view(a), wide);
    EXPECT_LE(loose.sell_nnz_padded(), tight.sell_nnz_padded());
    EXPECT_EQ(tight.sigma(), 1);
    EXPECT_EQ(loose.sigma(), 8192);
}

TEST_F(SpmvPlanState, EverySigmaGivesTheSameAnswer) {
    const Csr a = make_matrix(MatrixKind::kSkewed, ckl::test::seed_stream(4831));
    const auto x = ckl::random_matrix(a.n, 1, ckl::test::seed_stream(4832));
    const auto y0 = ckl::random_matrix(a.m, 1, ckl::test::seed_stream(4833));
    const SpmvRef ref = spmv_reference(a, x, y0, 1.1f, 0.3f);
    const DeviceCsr device(a);
    for (int sigma : {1, 256, 8192}) {
        SCOPED_TRACE("sigma " + std::to_string(sigma));
        ckl::SpmvPlanOptions opt;
        opt.sigma = sigma;
        opt.build_bsr = false;
        ckl::SpmvPlan plan(device.view(a), opt);
        ckl::DeviceBuffer<float> dx(x.size());
        ckl::DeviceBuffer<float> dy(y0.size());
        dx.copy_from_host(x);
        dy.copy_from_host(y0);
        ckl::SpmvAlgo chosen = ckl::SpmvAlgo::kAuto;
        ASSERT_EQ(
            ckl::spmv(plan, ckl::SpmvAlgo::kSellCSigma, 1.1f, dx.data(), 0.3f, dy.data(), &chosen),
            ckl::Status::kSuccess);
        ASSERT_EQ(chosen, ckl::SpmvAlgo::kSellCSigma);
        CKL_CUDA_CHECK(cudaDeviceSynchronize());
        expect_rows_match(Variant::kSell, a, dy.to_host(), ref);
    }
}

TEST_F(SpmvPlanState, ABlockedMatrixPicksAWideBlockAndADiagonalOnePicksTheNarrowest) {
    // Four by four dense blocks on the diagonal: exactly the structure BSR
    // exists for, and the detection pass has to find it.
    constexpr int kBlocks = 64;
    constexpr int kDim = 4;
    Csr blocked;
    blocked.m = kBlocks * kDim;
    blocked.n = kBlocks * kDim;
    blocked.row_ptr.assign(static_cast<std::size_t>(blocked.m) + 1, 0);
    std::mt19937_64 rng(ckl::test::seed_stream(4841));
    for (int i = 0; i < blocked.m; ++i) {
        const int base = (i / kDim) * kDim;
        for (int c = 0; c < kDim; ++c) {
            blocked.col_idx.push_back(base + c);
            blocked.values.push_back(ckl::bits_to_signed_unit(rng()));
        }
        blocked.row_ptr[static_cast<std::size_t>(i) + 1] =
            blocked.row_ptr[static_cast<std::size_t>(i)] + kDim;
    }
    blocked.nnz = blocked.row_ptr.back();
    const DeviceCsr blocked_device(blocked);
    const ckl::SpmvPlan blocked_plan(blocked_device.view(blocked));
    EXPECT_GE(blocked_plan.bsr_block_dim(), 4);
    EXPECT_LE(blocked_plan.bsr_nnz_padded(), 2 * blocked.nnz);
    EXPECT_GE(blocked_plan.bsr_detect_ms(), 0.0);

    // One nonzero per row on the diagonal: no block structure at all, so the
    // widest dimension that reaches half fill is the narrowest one offered.
    Csr diagonal;
    diagonal.m = 512;
    diagonal.n = 512;
    diagonal.row_ptr.assign(static_cast<std::size_t>(diagonal.m) + 1, 0);
    for (int i = 0; i < diagonal.m; ++i) {
        diagonal.col_idx.push_back(i);
        diagonal.values.push_back(1.0f + static_cast<float>(i % 7));
        diagonal.row_ptr[static_cast<std::size_t>(i) + 1] = i + 1;
    }
    diagonal.nnz = diagonal.m;
    const DeviceCsr diagonal_device(diagonal);
    const ckl::SpmvPlan diagonal_plan(diagonal_device.view(diagonal));
    EXPECT_EQ(diagonal_plan.bsr_block_dim(), 2);
}

TEST_F(SpmvPlanState, ANamedBlockDimensionOverridesTheDetectionPass) {
    const Csr a = make_matrix(MatrixKind::kSkewed, ckl::test::seed_stream(4861));
    const DeviceCsr device(a);
    for (int bd : {2, 4, 8}) {
        SCOPED_TRACE("block_dim " + std::to_string(bd));
        ckl::SpmvPlanOptions opt;
        opt.block_dim = bd;
        opt.build_sell = false;
        const ckl::SpmvPlan plan(device.view(a), opt);
        EXPECT_EQ(plan.bsr_block_dim(), bd);
    }
    // A dimension the kernel is not instantiated for is refused with a reason
    // rather than converted into something the kernel would then decline.
    ckl::SpmvPlanOptions odd;
    odd.block_dim = 3;
    odd.build_sell = false;
    const ckl::SpmvPlan refused(device.view(a), odd);
    EXPECT_EQ(refused.bsr_block_dim(), 0);
}

TEST_F(SpmvPlanState, TheTrafficModelBandBracketsTheHonestAnswer) {
    const Csr a = make_matrix(MatrixKind::kSkewed, ckl::test::seed_stream(4851));
    const DeviceCsr device(a);
    const ckl::SpmvPlan plan(device.view(a));
    // With no reuse the x term is 4 nnz; with perfect residency it is 4 n. The
    // suite's matrices all have nnz above n, so low is below high.
    EXPECT_LT(plan.model_bytes_low(false), plan.model_bytes_high(false));
    EXPECT_EQ(plan.model_bytes_low(true) - plan.model_bytes_low(false), 4LL * a.m);
    const long long expect_low = 8LL * a.nnz + 4LL * (a.m + 1) + 4LL * a.n + 4LL * a.m;
    EXPECT_EQ(plan.model_bytes_low(false), expect_low);
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

class SpmvDispatch : public GpuTest {};

TEST_F(SpmvDispatch, AutoPicksTheVectorKernelOnAShortRegularMatrix) {
    const Csr a = make_uniform(4096, 4096, 4, ckl::test::seed_stream(4901));
    const DeviceCsr device(a);
    const ckl::SpmvPlan plan(device.view(a));
    EXPECT_FALSE(plan.power_law());
    EXPECT_EQ(plan.query(), ckl::SpmvAlgo::kCsrVector);
    EXPECT_EQ(plan.vector_width(), 4);
}

TEST_F(SpmvDispatch, AutoPicksTheWarpKernelOnALongRegularMatrix) {
    const Csr a = make_uniform(2048, 4096, 64, ckl::test::seed_stream(4911));
    const DeviceCsr device(a);
    const ckl::SpmvPlan plan(device.view(a));
    EXPECT_FALSE(plan.power_law());
    EXPECT_EQ(plan.query(), ckl::SpmvAlgo::kCsrWarp);
    EXPECT_EQ(plan.vector_width(), 32);
}

TEST_F(SpmvDispatch, AutoPicksTheMergePathOnAPowerLawMatrix) {
    const Csr a = make_power_law(1024, 4096, ckl::test::seed_stream(4921));
    const DeviceCsr device(a);
    const ckl::SpmvPlan plan(device.view(a));
    EXPECT_TRUE(plan.power_law());
    EXPECT_EQ(plan.query(), ckl::SpmvAlgo::kMerge);
    EXPECT_EQ(plan.sigma(), 8192);
}

TEST_F(SpmvDispatch, AutoReportsTheAlgorithmItRan) {
    const Csr a = make_power_law(1024, 4096, ckl::test::seed_stream(4931));
    const auto x = ckl::random_matrix(a.n, 1, ckl::test::seed_stream(4932));
    const auto y0 = ckl::random_matrix(a.m, 1, ckl::test::seed_stream(4933));
    const DeviceCsr device(a);
    ckl::SpmvPlan plan(device.view(a));
    ckl::DeviceBuffer<float> dx(x.size());
    ckl::DeviceBuffer<float> dy(y0.size());
    dx.copy_from_host(x);
    dy.copy_from_host(y0);
    ckl::SpmvAlgo chosen = ckl::SpmvAlgo::kAuto;
    ASSERT_EQ(ckl::spmv(plan, ckl::SpmvAlgo::kAuto, 1.0f, dx.data(), 0.0f, dy.data(), &chosen),
              ckl::Status::kSuccess);
    // kAuto is the only mode allowed to choose, and it has to say what it chose.
    ASSERT_NE(chosen, ckl::SpmvAlgo::kAuto);
    EXPECT_EQ(chosen, plan.query());
    CKL_CUDA_CHECK(cudaDeviceSynchronize());
    const SpmvRef ref = spmv_reference(a, x, y0, 1.0f, 0.0f);
    expect_rows_match(Variant::kMerge, a, dy.to_host(), ref);
}

TEST_F(SpmvDispatch, ARefusedVariantSaysSoRatherThanFallingBack) {
    const Csr a = make_matrix(MatrixKind::kSkewed, ckl::test::seed_stream(4941));
    const auto x = ckl::random_matrix(a.n, 1, ckl::test::seed_stream(4942));
    const DeviceCsr device(a);
    ckl::SpmvPlanOptions opt;
    opt.build_sell = false;
    opt.build_bsr = false;
    ckl::SpmvPlan plan(device.view(a), opt);
    ckl::DeviceBuffer<float> dx(x.size());
    ckl::DeviceBuffer<float> dy(static_cast<std::size_t>(a.m));
    dx.copy_from_host(x);
    dy.zero();
    for (ckl::SpmvAlgo algo : {ckl::SpmvAlgo::kSellCSigma, ckl::SpmvAlgo::kBsr}) {
        SCOPED_TRACE(ckl::spmv_algo_name(algo));
        ckl::SpmvAlgo chosen = ckl::SpmvAlgo::kAuto;
        // A silent fallback is a bug: the plan has no state for this variant, so
        // the call fails and chosen still names the path that was asked for.
        EXPECT_EQ(ckl::spmv(plan, algo, 1.0f, dx.data(), 0.0f, dy.data(), &chosen),
                  ckl::Status::kNotSupported);
        EXPECT_EQ(chosen, algo);
    }
}

TEST_F(SpmvDispatch, EveryAlgorithmHasAName) {
    for (ckl::SpmvAlgo algo :
         {ckl::SpmvAlgo::kAuto, ckl::SpmvAlgo::kCsrNaive, ckl::SpmvAlgo::kCsrWarp,
          ckl::SpmvAlgo::kCsrVector, ckl::SpmvAlgo::kMerge, ckl::SpmvAlgo::kSellCSigma,
          ckl::SpmvAlgo::kBsr, ckl::SpmvAlgo::kCusparseDefault, ckl::SpmvAlgo::kCusparseAlg2}) {
        EXPECT_STRNE(ckl::spmv_algo_name(algo), "unknown");
    }
}

// ---------------------------------------------------------------------------
// The v1 free functions, still source compatible
// ---------------------------------------------------------------------------

class SpmvStreams : public GpuTest {};

TEST_F(SpmvStreams, TwoConcurrentStreams) {
    const Csr a = make_matrix(MatrixKind::kSkewed, ckl::test::seed_stream(4501));
    const auto x = ckl::random_matrix(a.n, 1, ckl::test::seed_stream(4502));
    const auto y0 = ckl::random_matrix(a.m, 1, ckl::test::seed_stream(4503));
    const float alpha = 1.1f;
    const float beta = 0.3f;

    cudaStream_t s1 = nullptr;
    cudaStream_t s2 = nullptr;
    CKL_CUDA_CHECK(cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking));
    CKL_CUDA_CHECK(cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking));

    const DeviceCsr device(a);
    ckl::DeviceBuffer<float> dx(x.size());
    ckl::DeviceBuffer<float> dy1(y0.size());
    ckl::DeviceBuffer<float> dy2(y0.size());
    dx.copy_from_host(x);
    dy1.copy_from_host(y0);
    dy2.copy_from_host(y0);

    ckl::spmv_csr_naive(device.row_ptr.data(), device.col_idx.data(), device.values.data(),
                        dx.data(), dy1.data(), a.m, a.n, a.nnz, alpha, beta, s1);
    ckl::spmv_csr_warp(device.row_ptr.data(), device.col_idx.data(), device.values.data(),
                       dx.data(), dy2.data(), a.m, a.n, a.nnz, alpha, beta, s2);
    CKL_CUDA_CHECK(cudaStreamSynchronize(s1));
    CKL_CUDA_CHECK(cudaStreamSynchronize(s2));
    const auto out1 = dy1.to_host();
    const auto out2 = dy2.to_host();
    CKL_CUDA_CHECK(cudaStreamDestroy(s1));
    CKL_CUDA_CHECK(cudaStreamDestroy(s2));

    const SpmvRef ref = spmv_reference(a, x, y0, alpha, beta);
    expect_rows_match(Variant::kNaive, a, out1, ref);
    expect_rows_match(Variant::kWarp, a, out2, ref);
}

// The C ABI entry point. It holds a plan cache of its own, so the second call on
// the same matrix has to give the same answer as the first, and chosen has to
// come back through the C enum.
TEST_F(SpmvStreams, TheCEntryPointDispatchesAndReportsWhatItRan) {
    const Csr a = make_matrix(MatrixKind::kSkewed, ckl::test::seed_stream(4521));
    const auto x = ckl::random_matrix(a.n, 1, ckl::test::seed_stream(4522));
    const auto y0 = ckl::random_matrix(a.m, 1, ckl::test::seed_stream(4523));
    const SpmvRef ref = spmv_reference(a, x, y0, 1.1f, 0.3f);
    const DeviceCsr device(a);
    ckl::DeviceBuffer<float> dx(x.size());
    ckl::DeviceBuffer<float> dy(y0.size());
    dx.copy_from_host(x);

    ckl_handle_t h = nullptr;
    ASSERT_EQ(ckl_create(&h), CKL_STATUS_SUCCESS);
    for (int call = 0; call < 2; ++call) {
        SCOPED_TRACE("call " + std::to_string(call));
        dy.copy_from_host(y0);
        ckl_spmv_algo_t chosen = CKL_SPMV_AUTO;
        ASSERT_EQ(
            ckl_spmv_csr(h, a.m, a.n, a.nnz, 1.1f, device.row_ptr.data(), device.col_idx.data(),
                         device.values.data(), dx.data(), 0.3f, dy.data(), CKL_SPMV_MERGE, &chosen),
            CKL_STATUS_SUCCESS);
        EXPECT_EQ(chosen, CKL_SPMV_MERGE);
        CKL_CUDA_CHECK(cudaDeviceSynchronize());
        expect_rows_match(Variant::kMerge, a, dy.to_host(), ref);
    }
    ckl_spmv_algo_t chosen = CKL_SPMV_AUTO;
    EXPECT_EQ(ckl_spmv_csr(h, a.m, a.n, a.nnz, 1.0f, device.row_ptr.data(), device.col_idx.data(),
                           device.values.data(), dx.data(), 0.0f, dy.data(),
                           // 15 is inside the enum's representable range and is
                           // not one of its nine enumerators, which is exactly
                           // the value a C caller can produce by accident.
                           static_cast<ckl_spmv_algo_t>(15), &chosen),
              CKL_STATUS_INVALID_VALUE);
    EXPECT_EQ(ckl_destroy(h), CKL_STATUS_SUCCESS);
}

TEST_F(SpmvStreams, TheV1CusparseFreeFunctionStillAgreesWithThePlan) {
    const Csr a = make_matrix(MatrixKind::kSkewed, ckl::test::seed_stream(4511));
    const auto x = ckl::random_matrix(a.n, 1, ckl::test::seed_stream(4512));
    const auto y0 = ckl::random_matrix(a.m, 1, ckl::test::seed_stream(4513));
    const DeviceCsr device(a);
    ckl::DeviceBuffer<float> dx(x.size());
    ckl::DeviceBuffer<float> dy(y0.size());
    dx.copy_from_host(x);
    dy.copy_from_host(y0);
    ckl::spmv_cusparse(device.row_ptr.data(), device.col_idx.data(), device.values.data(),
                       dx.data(), dy.data(), a.m, a.n, a.nnz, 1.1f, 0.3f, nullptr);
    CKL_CUDA_CHECK(cudaDeviceSynchronize());
    const SpmvRef ref = spmv_reference(a, x, y0, 1.1f, 0.3f);
    expect_rows_match(Variant::kCusparseDefault, a, dy.to_host(), ref);
}

}  // namespace
