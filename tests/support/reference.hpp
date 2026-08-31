#pragma once

// Host side helpers for correctness tests: deterministic random fill, a double
// precision CPU GEMM reference for small shapes, and a relative Frobenius error.
// Seeds are explicit so a failing shape reproduces exactly.

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace ckl {

// Fills with uniform values in [-1, 1] from a fixed seed. Same seed and size
// give the same data on every run, which the sweep records per row.
inline std::vector<float> random_matrix(int rows, int cols, std::uint64_t seed) {
    std::vector<float> data(static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols));
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : data) {
        x = dist(rng);
    }
    return data;
}

// Row major double precision reference: C = alpha * (A * B) + beta * C.
// A is m by k, B is k by n, C is m by n. Only for shapes small enough to afford
// the O(m n k) host cost; the tests pick those.
inline std::vector<double> gemm_reference(const std::vector<float>& a, const std::vector<float>& b,
                                          const std::vector<float>& c_in, int m, int n, int k,
                                          float alpha, float beta) {
    std::vector<double> c(static_cast<std::size_t>(m) * static_cast<std::size_t>(n));
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            double acc = 0.0;
            for (int p = 0; p < k; ++p) {
                acc += static_cast<double>(a[static_cast<std::size_t>(i) * k + p]) *
                       static_cast<double>(b[static_cast<std::size_t>(p) * n + j]);
            }
            const std::size_t idx = static_cast<std::size_t>(i) * n + j;
            c[idx] = static_cast<double>(alpha) * acc +
                     static_cast<double>(beta) * static_cast<double>(c_in[idx]);
        }
    }
    return c;
}

// Relative Frobenius error ||actual - ref||_F / ||ref||_F. The ref is double;
// the actual is the float result promoted. A near zero ref norm falls back to
// the absolute error so a zero matrix does not divide by zero.
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

}  // namespace ckl
