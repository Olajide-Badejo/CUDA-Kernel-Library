// TRSM correctness against cuBLAS and a double precision CPU reference, plus a
// residual check ||L X - alpha B|| / ||alpha B||. L is built diagonally dominant
// so the system is well conditioned and a float solve stays close to the double
// reference. Shapes cover square and rectangular right hand sides.

#include <cmath>
#include <cstdio>
#include <functional>
#include <vector>

#include "ckl/cuda_check.hpp"
#include "ckl/device_buffer.hpp"
#include "ckl/reference.hpp"
#include "ckl/trsm.hpp"

namespace {

// Lower triangular, diagonally dominant: off diagonal in [-1, 1], diagonal set to
// m + 1 so the solve is well conditioned. Upper part is zero.
std::vector<float> make_lower(int m, std::uint64_t seed) {
    auto full = ckl::random_matrix(m, m, seed);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j) {
            if (j > i) {
                full[static_cast<std::size_t>(i) * m + j] = 0.0f;
            }
        }
        full[static_cast<std::size_t>(i) * m + i] = static_cast<float>(m + 1);
    }
    return full;
}

std::vector<double> trsm_reference(const std::vector<float>& a, const std::vector<float>& b, int m,
                                   int n, float alpha) {
    std::vector<double> x(static_cast<std::size_t>(m) * n);
    for (int c = 0; c < n; ++c) {
        for (int i = 0; i < m; ++i) {
            double s = static_cast<double>(alpha) * b[static_cast<std::size_t>(i) * n + c];
            for (int j = 0; j < i; ++j) {
                s -= static_cast<double>(a[static_cast<std::size_t>(i) * m + j]) *
                     x[static_cast<std::size_t>(j) * n + c];
            }
            x[static_cast<std::size_t>(i) * n + c] = s / a[static_cast<std::size_t>(i) * m + i];
        }
    }
    return x;
}

// Residual ||L X - alpha B|| / ||alpha B|| in double precision.
double residual(const std::vector<float>& a, const std::vector<float>& x,
                const std::vector<float>& b, int m, int n, float alpha) {
    double num = 0.0;
    double den = 0.0;
    for (int i = 0; i < m; ++i) {
        for (int c = 0; c < n; ++c) {
            double lx = 0.0;
            for (int j = 0; j <= i; ++j) {
                lx += static_cast<double>(a[static_cast<std::size_t>(i) * m + j]) *
                      static_cast<double>(x[static_cast<std::size_t>(j) * n + c]);
            }
            const double target = static_cast<double>(alpha) *
                                  static_cast<double>(b[static_cast<std::size_t>(i) * n + c]);
            num += (lx - target) * (lx - target);
            den += target * target;
        }
    }
    return std::sqrt(num) / std::sqrt(den);
}

using LaunchFn = std::function<void(const float*, float*, int, int, float, cudaStream_t)>;

std::vector<float> run(const LaunchFn& launch, const std::vector<float>& a,
                       const std::vector<float>& b0, int m, int n, float alpha) {
    ckl::DeviceBuffer<float> da(a.size());
    ckl::DeviceBuffer<float> db(b0.size());
    da.copy_from_host(a);
    db.copy_from_host(b0);
    launch(da.data(), db.data(), m, n, alpha, nullptr);
    CKL_CUDA_CHECK(cudaDeviceSynchronize());
    return db.to_host();
}

struct Shape {
    int m;
    int n;
    const char* label;
};

bool run_case(const Shape& s) {
    const float alpha = 1.5f;
    const auto a = make_lower(s.m, 0x71 + static_cast<std::uint64_t>(s.m));
    const auto b0 = ckl::random_matrix(s.m, s.n, 0x92 + static_cast<std::uint64_t>(s.n));

    const auto x_cublas = run(ckl::trsm_cublas, a, b0, s.m, s.n, alpha);
    const std::vector<double> cublas_ref(x_cublas.begin(), x_cublas.end());
    const auto cpu_ref = trsm_reference(a, b0, s.m, s.n, alpha);
    const double err_oracle = ckl::relative_frobenius_error(x_cublas, cpu_ref);
    const double res_cublas = residual(a, x_cublas, b0, s.m, s.n, alpha);
    std::printf("  %-10s m=%-5d n=%-5d cublas_vs_cpu=%.3e residual=%.3e\n", s.label, s.m, s.n,
                err_oracle, res_cublas);

    bool ok = err_oracle < 1e-3 && res_cublas < 1e-3;
    const std::vector<std::pair<const char*, LaunchFn>> variants = {
        {"naive", ckl::trsm_naive},
        {"blocked", ckl::trsm_blocked},
    };
    for (const auto& v : variants) {
        const auto x = run(v.second, a, b0, s.m, s.n, alpha);
        const double err = ckl::relative_frobenius_error(x, cublas_ref);
        const double res = residual(a, x, b0, s.m, s.n, alpha);
        const bool pass = err < 1e-3 && res < 1e-3;
        ok = ok && pass;
        std::printf("      %-8s vs cuBLAS = %.3e  residual = %.3e  %s\n", v.first, err, res,
                    pass ? "PASS" : "FAIL");
    }
    return ok;
}

}  // namespace

int main() {
    const std::vector<Shape> shapes = {
        {256, 128, "square rhs"},
        {512, 64, "tall few rhs"},
        {129, 200, "odd block"},
    };
    std::printf("TRSM correctness (lower, diagonally dominant; tolerance 1e-3)\n");
    bool all_ok = true;
    for (const auto& s : shapes) {
        all_ok = run_case(s) && all_ok;
    }
    std::printf("%s\n", all_ok ? "all TRSM cases passed" : "TRSM cases FAILED");
    return all_ok ? 0 : 1;
}
