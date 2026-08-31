#pragma once

// Host side helpers for correctness tests and benchmarks: deterministic random
// fill, double precision CPU references, and the error measures the gates are
// written against.
//
// Two things changed in 1.1.0. The fill no longer uses
// std::uniform_real_distribution, whose output is implementation defined, so the
// same seed gave different data under libstdc++ and libc++ and a failing shape
// could not be reproduced off the machine that found it. The bytes now come from
// mt19937_64, which the standard pins exactly, mapped to a float by explicit bit
// arithmetic. And seeds are passed in rather than derived from the shape, so one
// --seed on the command line moves every dataset in a suite at once.

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

namespace ckl {

// One engine output to a float in [-1, 1]. The top 24 bits are taken because a
// float carries 24 bits of significand, so every representable step is reachable
// and none is produced twice. mt19937_64 is specified by the standard down to
// the constants, and this arithmetic is exact in double, so the result is the
// same on every conforming standard library.
inline float bits_to_signed_unit(std::uint64_t bits) {
    constexpr std::uint64_t kMask = (1ULL << 24) - 1ULL;
    const std::uint64_t m = (bits >> 40) & kMask;
    constexpr double kScale = 2.0 / 16777215.0;  // 2 / (2^24 - 1)
    return static_cast<float>(static_cast<double>(m) * kScale - 1.0);
}

// Fills with values in [-1, 1] from an explicit seed.
inline std::vector<float> random_matrix(int rows, int cols, std::uint64_t seed) {
    std::vector<float> data(static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols));
    std::mt19937_64 rng(seed);
    for (auto& x : data) {
        x = bits_to_signed_unit(rng());
    }
    return data;
}

// The near cancellation dataset. Every element is a unit value scaled either up
// or down by the same factor, chosen from one engine bit, so a row of A meets a
// column of B in a mix of huge and tiny partial products that largely cancel.
// The exact sum is tiny next to the largest term it is built from, which is what
// makes a plain relative error meaningless here and a scaled residual the only
// bound worth asserting.
inline std::vector<float> mixed_magnitude_matrix(int rows, int cols, std::uint64_t seed,
                                                 float factor) {
    std::vector<float> data(static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols));
    std::mt19937_64 rng(seed);
    for (auto& x : data) {
        const std::uint64_t bits = rng();
        const float unit = bits_to_signed_unit(bits);
        x = ((bits & 1ULL) != 0ULL) ? unit * factor : unit / factor;
    }
    return data;
}

// Shape derived tolerance. The error of a length k dot product accumulated in
// float grows like sqrt(k) for random signs, so a flat 1e-4 is far too loose at
// k = 64 and would not catch a kernel that dropped its last K stage. The factor
// of 8 is the slack over the statistical bound; nothing in the suite comes close
// to it, which is the point of recording the measured numbers next to it.
inline double tol(int k) {
    const int kk = k > 1 ? k : 1;
    return 8.0 * std::sqrt(static_cast<double>(kk)) * static_cast<double>(FLT_EPSILON);
}

// A double reference plus, per output element, the magnitude the rounding error
// is actually bounded by: |alpha| * sum_p |a_ip| |b_pj| + |beta| |c_ij|. Dividing
// by that instead of by |c_ij| is what makes one bound work for both a benign
// dataset and a cancelling one.
struct GemmRef {
    std::vector<double> c;
    std::vector<double> scale;
};

inline GemmRef gemm_reference_scaled(const std::vector<float>& a, const std::vector<float>& b,
                                     const std::vector<float>& c_in, int m, int n, int k,
                                     float alpha, float beta) {
    GemmRef ref;
    const std::size_t elems = static_cast<std::size_t>(m) * static_cast<std::size_t>(n);
    ref.c.assign(elems, 0.0);
    ref.scale.assign(elems, 0.0);
    const double da = static_cast<double>(alpha);
    const double db = static_cast<double>(beta);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            double acc = 0.0;
            double mag = 0.0;
            for (int p = 0; p < k; ++p) {
                const double av = static_cast<double>(
                    a[static_cast<std::size_t>(i) * static_cast<std::size_t>(k) +
                      static_cast<std::size_t>(p)]);
                const double bv = static_cast<double>(
                    b[static_cast<std::size_t>(p) * static_cast<std::size_t>(n) +
                      static_cast<std::size_t>(j)]);
                acc += av * bv;
                mag += std::fabs(av) * std::fabs(bv);
            }
            const std::size_t idx = static_cast<std::size_t>(i) * static_cast<std::size_t>(n) +
                                    static_cast<std::size_t>(j);
            const double c0 = static_cast<double>(c_in[idx]);
            ref.c[idx] = da * acc + db * c0;
            ref.scale[idx] = std::fabs(da) * mag + std::fabs(db) * std::fabs(c0);
        }
    }
    return ref;
}

// Row major double precision reference: C = alpha * (A * B) + beta * C.
inline std::vector<double> gemm_reference(const std::vector<float>& a, const std::vector<float>& b,
                                          const std::vector<float>& c_in, int m, int n, int k,
                                          float alpha, float beta) {
    return gemm_reference_scaled(a, b, c_in, m, n, k, alpha, beta).c;
}

// Relative Frobenius error ||actual - ref||_F / ||ref||_F. One number for a
// whole matrix, so it hides a single wrong row inside a large right answer; the
// suites use it only where the spec names it and prefer the elementwise measures
// below everywhere else.
inline double relative_frobenius_error(const std::vector<float>& actual,
                                       const std::vector<double>& ref) {
    double num = 0.0;
    double den = 0.0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        const double diff = static_cast<double>(actual[i]) - ref[i];
        num += diff * diff;
        den += ref[i] * ref[i];
    }
    const double rn = std::sqrt(num);
    const double rd = std::sqrt(den);
    return rd > 1e-30 ? rn / rd : rn;
}

// Worst elementwise |actual - ref| / scale. One bad element fails it, whatever
// the rest of the matrix does.
inline double max_scaled_residual(const std::vector<float>& actual, const GemmRef& ref) {
    double worst = 0.0;
    for (std::size_t i = 0; i < ref.c.size(); ++i) {
        const double diff = std::fabs(static_cast<double>(actual[i]) - ref.c[i]);
        const double denom = ref.scale[i] > 1e-30 ? ref.scale[i] : 1.0;
        const double r = diff / denom;
        if (r > worst) {
            worst = r;
        }
    }
    return worst;
}

// Worst elementwise relative error against a double reference, with the
// reference magnitude floored so a near zero entry does not divide the whole
// check into meaninglessness.
inline double max_relative_error(const std::vector<float>& actual, const std::vector<double>& ref,
                                 double floor_magnitude) {
    double worst = 0.0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        const double mag = std::fabs(ref[i]);
        const double denom = mag > floor_magnitude ? mag : floor_magnitude;
        const double r = std::fabs(static_cast<double>(actual[i]) - ref[i]) / denom;
        if (r > worst) {
            worst = r;
        }
    }
    return worst;
}

// Infinity norm condition number of an n by n row major double matrix, computed
// by explicit Gauss-Jordan inversion with partial pivoting. O(n^3), so it is for
// the small systems the conditioning tests build, and it exists so a test that
// claims to be ill conditioned can say by how much instead of asserting it.
// Returns infinity for a numerically singular matrix.
inline double condition_number_inf(const std::vector<double>& a, int n) {
    const std::size_t sn = static_cast<std::size_t>(n);
    std::vector<double> work(a);
    std::vector<double> inv(sn * sn, 0.0);
    for (int i = 0; i < n; ++i) {
        inv[static_cast<std::size_t>(i) * sn + static_cast<std::size_t>(i)] = 1.0;
    }
    for (int col = 0; col < n; ++col) {
        int pivot = col;
        double best =
            std::fabs(work[static_cast<std::size_t>(col) * sn + static_cast<std::size_t>(col)]);
        for (int r = col + 1; r < n; ++r) {
            const double v =
                std::fabs(work[static_cast<std::size_t>(r) * sn + static_cast<std::size_t>(col)]);
            if (v > best) {
                best = v;
                pivot = r;
            }
        }
        if (best == 0.0) {
            return std::numeric_limits<double>::infinity();
        }
        if (pivot != col) {
            for (int c = 0; c < n; ++c) {
                std::swap(work[static_cast<std::size_t>(col) * sn + static_cast<std::size_t>(c)],
                          work[static_cast<std::size_t>(pivot) * sn + static_cast<std::size_t>(c)]);
                std::swap(inv[static_cast<std::size_t>(col) * sn + static_cast<std::size_t>(c)],
                          inv[static_cast<std::size_t>(pivot) * sn + static_cast<std::size_t>(c)]);
            }
        }
        const double d = work[static_cast<std::size_t>(col) * sn + static_cast<std::size_t>(col)];
        for (int c = 0; c < n; ++c) {
            work[static_cast<std::size_t>(col) * sn + static_cast<std::size_t>(c)] /= d;
            inv[static_cast<std::size_t>(col) * sn + static_cast<std::size_t>(c)] /= d;
        }
        for (int r = 0; r < n; ++r) {
            if (r == col) {
                continue;
            }
            const double f = work[static_cast<std::size_t>(r) * sn + static_cast<std::size_t>(col)];
            if (f == 0.0) {
                continue;
            }
            for (int c = 0; c < n; ++c) {
                work[static_cast<std::size_t>(r) * sn + static_cast<std::size_t>(c)] -=
                    f * work[static_cast<std::size_t>(col) * sn + static_cast<std::size_t>(c)];
                inv[static_cast<std::size_t>(r) * sn + static_cast<std::size_t>(c)] -=
                    f * inv[static_cast<std::size_t>(col) * sn + static_cast<std::size_t>(c)];
            }
        }
    }
    double norm_a = 0.0;
    double norm_inv = 0.0;
    for (int r = 0; r < n; ++r) {
        double ra = 0.0;
        double ri = 0.0;
        for (int c = 0; c < n; ++c) {
            ra += std::fabs(a[static_cast<std::size_t>(r) * sn + static_cast<std::size_t>(c)]);
            ri += std::fabs(inv[static_cast<std::size_t>(r) * sn + static_cast<std::size_t>(c)]);
        }
        norm_a = ra > norm_a ? ra : norm_a;
        norm_inv = ri > norm_inv ? ri : norm_inv;
    }
    return norm_a * norm_inv;
}

inline bool all_finite(const std::vector<float>& v) {
    for (float x : v) {
        if (!std::isfinite(x)) {
            return false;
        }
    }
    return true;
}

}  // namespace ckl
