// CSR SpMV correctness: y = alpha * (A * x) + beta * y.
//
// v1 tested one skewed matrix and checked it with a single Frobenius number over
// the whole vector, which lets one wrong long row hide inside four thousand
// right short ones. Here the check is per row, against a double CPU reference,
// with the tolerance derived from that row's own nonzero count: a row with three
// nonzeros gets a much tighter bound than one with five hundred, which is the
// point.
//
// The matrix zoo is the other half. Empty rows, a matrix with no nonzeros at
// all, a single row, column indices out of order, and duplicated column indices
// that have to sum rather than overwrite: every one of those is a shape a real
// caller produces and none of them was covered.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include <cuda_runtime.h>

#include <gtest/gtest.h>

#include "ckl/cuda_check.hpp"
#include "ckl/device_buffer.hpp"
#include "ckl/sparse.hpp"
#include "gpu_environment.hpp"
#include "reference.hpp"

namespace {

using ckl::test::GpuTest;
using ckl::test::GpuTestWithParam;

using LaunchFn = std::function<void(const int*, const int*, const float*, const float*, float*, int,
                                    int, int, float, float, cudaStream_t)>;

enum class Variant { kNaive, kWarp, kCusparse };

const char* variant_name(Variant v) {
    switch (v) {
        case Variant::kNaive:
            return "naive";
        case Variant::kWarp:
            return "warp";
        case Variant::kCusparse:
            return "cusparse";
    }
    return "unknown";
}

LaunchFn variant_launch(Variant v) {
    switch (v) {
        case Variant::kNaive:
            return ckl::spmv_csr_naive;
        case Variant::kWarp:
            return ckl::spmv_csr_warp;
        case Variant::kCusparse:
            return ckl::spmv_cusparse;
    }
    return ckl::spmv_cusparse;
}

const std::vector<Variant>& all_variants() {
    static const std::vector<Variant> v = {Variant::kNaive, Variant::kWarp, Variant::kCusparse};
    return v;
}

// The hand written variants only; cuSPARSE documents CSR with sorted, unique
// column indices, so the cases that deliberately break that contract are run
// against the two kernels this project owns.
const std::vector<Variant>& hand_variants() {
    static const std::vector<Variant> v = {Variant::kNaive, Variant::kWarp};
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
// both kernels regardless.
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
// accumulating would drop one of them.
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

std::vector<float> run_spmv(Variant v, const Csr& a, const std::vector<float>& x,
                            const std::vector<float>& y0, float alpha, float beta) {
    ckl::DeviceBuffer<int> drp(a.row_ptr.size());
    ckl::DeviceBuffer<int> dci(a.col_idx.size());
    ckl::DeviceBuffer<float> dv(a.values.size());
    ckl::DeviceBuffer<float> dx(x.size());
    ckl::DeviceBuffer<float> dy(y0.size());
    drp.copy_from_host(a.row_ptr);
    dci.copy_from_host(a.col_idx);
    dv.copy_from_host(a.values);
    dx.copy_from_host(x);
    dy.copy_from_host(y0);
    variant_launch(v)(drp.data(), dci.data(), dv.data(), dx.data(), dy.data(), a.m, a.n, a.nnz,
                      alpha, beta, nullptr);
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

INSTANTIATE_TEST_SUITE_P(Csr, SpmvVariant, ::testing::ValuesIn(all_variants()), variant_param_name);

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

    ckl::DeviceBuffer<int> drp(a.row_ptr.size());
    ckl::DeviceBuffer<int> dci(a.col_idx.size());
    ckl::DeviceBuffer<float> dv(a.values.size());
    ckl::DeviceBuffer<float> dx(x.size());
    ckl::DeviceBuffer<float> dy1(y0.size());
    ckl::DeviceBuffer<float> dy2(y0.size());
    drp.copy_from_host(a.row_ptr);
    dci.copy_from_host(a.col_idx);
    dv.copy_from_host(a.values);
    dx.copy_from_host(x);
    dy1.copy_from_host(y0);
    dy2.copy_from_host(y0);

    ckl::spmv_csr_naive(drp.data(), dci.data(), dv.data(), dx.data(), dy1.data(), a.m, a.n, a.nnz,
                        alpha, beta, s1);
    ckl::spmv_csr_warp(drp.data(), dci.data(), dv.data(), dx.data(), dy2.data(), a.m, a.n, a.nnz,
                       alpha, beta, s2);
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

}  // namespace
