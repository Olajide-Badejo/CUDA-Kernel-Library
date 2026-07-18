// GEMM correctness against cuBLAS and, on small shapes, against a double
// precision CPU reference that also validates the cuBLAS row major orientation.
// Shapes cover square, non square, non tile aligned, smaller than one tile, and
// a zero dimension, per the Phase 1 gate in Section 10.
//
// Exit code 0 means every case passed its tolerance; non zero means at least
// one failed. No test framework on purpose: this keeps the dependency surface
// small and the failure output is a plain table.

#include <cstdio>
#include <functional>
#include <string>
#include <vector>

#include "ckl/cuda_check.hpp"
#include "ckl/device_buffer.hpp"
#include "ckl/gemm.hpp"
#include "ckl/reference.hpp"

namespace {

using LaunchFn = std::function<void(const float*, const float*, float*, int, int, int,
                                    float, float, cudaStream_t)>;

// Runs a GEMM launcher on device and returns the host C result.
std::vector<float> run_device_gemm(const LaunchFn& launch,
                                   const std::vector<float>& a,
                                   const std::vector<float>& b,
                                   const std::vector<float>& c_in,
                                   int m, int n, int k, float alpha, float beta) {
    ckl::DeviceBuffer<float> da(a.size());
    ckl::DeviceBuffer<float> db(b.size());
    ckl::DeviceBuffer<float> dc(c_in.size());
    if (!a.empty()) da.copy_from_host(a);
    if (!b.empty()) db.copy_from_host(b);
    if (!c_in.empty()) dc.copy_from_host(c_in);

    launch(da.data(), db.data(), dc.data(), m, n, k, alpha, beta, nullptr);
    CKL_CUDA_CHECK(cudaDeviceSynchronize());

    return c_in.empty() ? std::vector<float>{} : dc.to_host();
}

struct Shape {
    int m;
    int n;
    int k;
    const char* label;
};

constexpr double kTolerance = 1e-4;

struct NamedVariant {
    const char* name;
    LaunchFn launch;
};

// Every hand written FP32 variant is checked against cuBLAS through this list;
// a new ladder rung is added with one line.
const std::vector<NamedVariant>& hand_written_variants() {
    static const std::vector<NamedVariant> variants = {
        {"naive", ckl::gemm_naive},
        {"tiled", ckl::gemm_tiled},
    };
    return variants;
}

bool run_case(const Shape& s) {
    const float alpha = 1.25f;
    const float beta = 0.5f;  // exercises the C read-modify-write path
    const auto a = ckl::random_matrix(s.m, s.k, 0x1234 + static_cast<std::uint64_t>(s.m));
    const auto b = ckl::random_matrix(s.k, s.n, 0x5678 + static_cast<std::uint64_t>(s.n));
    const auto c0 = ckl::random_matrix(s.m, s.n, 0x9abc + static_cast<std::uint64_t>(s.k));

    const auto c_cublas = run_device_gemm(ckl::gemm_cublas, a, b, c0, s.m, s.n, s.k, alpha, beta);
    const std::vector<double> cublas_as_ref(c_cublas.begin(), c_cublas.end());

    // Cross check on small shapes: cuBLAS versus a CPU double reference. This is
    // what catches a wrong transpose or leading dimension in the oracle.
    double err_cublas_vs_cpu = 0.0;
    if (static_cast<long long>(s.m) * s.n * s.k <= (512LL * 512 * 512) && !c_cublas.empty()) {
        const auto ref = ckl::gemm_reference(a, b, c0, s.m, s.n, s.k, alpha, beta);
        err_cublas_vs_cpu = ckl::relative_frobenius_error(c_cublas, ref);
    }

    bool ok = err_cublas_vs_cpu < kTolerance;
    std::printf("  %-22s m=%-5d n=%-5d k=%-5d  cublas_vs_cpu=%.3e\n",
                s.label, s.m, s.n, s.k, err_cublas_vs_cpu);
    for (const auto& v : hand_written_variants()) {
        const auto out = run_device_gemm(v.launch, a, b, c0, s.m, s.n, s.k, alpha, beta);
        const double err = out.empty() ? 0.0 : ckl::relative_frobenius_error(out, cublas_as_ref);
        const bool pass = err < kTolerance;
        ok = ok && pass;
        std::printf("      %-10s vs cuBLAS = %.3e  %s\n", v.name, err, pass ? "PASS" : "FAIL");
    }
    return ok;
}

}  // namespace

int main() {
    const std::vector<Shape> shapes = {
        {256, 256, 256, "square aligned"},
        {384, 512, 128, "non square aligned"},
        {129, 257, 193, "non tile aligned"},
        {7, 5, 11, "smaller than one tile"},
        {512, 384, 640, "non square large"},
        {0, 128, 128, "zero m dimension"},
        {128, 0, 128, "zero n dimension"},
        {64, 64, 0, "zero k dimension"},
    };

    std::printf("GEMM correctness (tolerance relative Frobenius < %.0e)\n", kTolerance);
    bool all_ok = true;
    for (const auto& s : shapes) {
        all_ok = run_case(s) && all_ok;
    }
    std::printf("%s\n", all_ok ? "all GEMM cases passed" : "GEMM cases FAILED");
    return all_ok ? 0 : 1;
}
