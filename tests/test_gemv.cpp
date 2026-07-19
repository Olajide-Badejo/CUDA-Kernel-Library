// GEMV correctness against cuBLAS and a double precision CPU reference. Shapes
// cover square, tall, wide, non multiple of four (to exercise the vectorized
// fallback), a single row, and a zero dimension.

#include <cstdio>
#include <functional>
#include <vector>

#include "ckl/cuda_check.hpp"
#include "ckl/device_buffer.hpp"
#include "ckl/gemv.hpp"
#include "ckl/reference.hpp"

namespace {

using LaunchFn =
    std::function<void(const float*, const float*, float*, int, int, float, float, cudaStream_t)>;

struct NamedVariant {
    const char* name;
    LaunchFn launch;
};

struct Shape {
    int m;
    int n;
    const char* label;
};

constexpr double kTolerance = 1e-4;

std::vector<double> gemv_reference(const std::vector<float>& a, const std::vector<float>& x,
                                   const std::vector<float>& y0, int m, int n, float alpha,
                                   float beta) {
    std::vector<double> y(static_cast<std::size_t>(m));
    for (int i = 0; i < m; ++i) {
        double acc = 0.0;
        for (int j = 0; j < n; ++j) {
            acc += static_cast<double>(a[static_cast<std::size_t>(i) * n + j]) *
                   static_cast<double>(x[j]);
        }
        y[i] = static_cast<double>(alpha) * acc + static_cast<double>(beta) * y0[i];
    }
    return y;
}

std::vector<float> run(const LaunchFn& launch, const std::vector<float>& a,
                       const std::vector<float>& x, const std::vector<float>& y0, int m, int n,
                       float alpha, float beta) {
    ckl::DeviceBuffer<float> da(a.size());
    ckl::DeviceBuffer<float> dx(x.size());
    ckl::DeviceBuffer<float> dy(y0.size());
    if (!a.empty())
        da.copy_from_host(a);
    if (!x.empty())
        dx.copy_from_host(x);
    if (!y0.empty())
        dy.copy_from_host(y0);
    launch(da.data(), dx.data(), dy.data(), m, n, alpha, beta, nullptr);
    CKL_CUDA_CHECK(cudaDeviceSynchronize());
    return y0.empty() ? std::vector<float>{} : dy.to_host();
}

bool run_case(const Shape& s) {
    const float alpha = 1.3f;
    const float beta = 0.4f;
    const auto a = ckl::random_matrix(s.m, s.n, 0x21 + static_cast<std::uint64_t>(s.m));
    const auto x = ckl::random_matrix(s.n, 1, 0x43 + static_cast<std::uint64_t>(s.n));
    const auto y0 = ckl::random_matrix(s.m, 1, 0x65);

    const auto y_cublas = run(ckl::gemv_cublas, a, x, y0, s.m, s.n, alpha, beta);
    const std::vector<double> cublas_ref(y_cublas.begin(), y_cublas.end());
    const auto cpu_ref = gemv_reference(a, x, y0, s.m, s.n, alpha, beta);
    const double err_oracle =
        y_cublas.empty() ? 0.0 : ckl::relative_frobenius_error(y_cublas, cpu_ref);

    std::printf("  %-16s m=%-6d n=%-6d cublas_vs_cpu=%.3e\n", s.label, s.m, s.n, err_oracle);
    bool ok = err_oracle < kTolerance;

    const std::vector<NamedVariant> variants = {
        {"naive", ckl::gemv_naive},
        {"warp", ckl::gemv_warp},
        {"vectorized", ckl::gemv_vectorized},
    };
    for (const auto& v : variants) {
        const auto y = run(v.launch, a, x, y0, s.m, s.n, alpha, beta);
        const double err = y.empty() ? 0.0 : ckl::relative_frobenius_error(y, cublas_ref);
        const bool pass = err < kTolerance;
        ok = ok && pass;
        std::printf("      %-12s vs cuBLAS = %.3e  %s\n", v.name, err, pass ? "PASS" : "FAIL");
    }
    return ok;
}

}  // namespace

int main() {
    const std::vector<Shape> shapes = {
        {1024, 1024, "square"},          {4096, 512, "tall"},     {512, 4096, "wide"},
        {777, 333, "non multiple of 4"}, {1, 2048, "single row"}, {2048, 0, "zero n"},
    };
    std::printf("GEMV correctness (tolerance %.0e)\n", kTolerance);
    bool all_ok = true;
    for (const auto& s : shapes) {
        all_ok = run_case(s) && all_ok;
    }
    std::printf("%s\n", all_ok ? "all GEMV cases passed" : "GEMV cases FAILED");
    return all_ok ? 0 : 1;
}
