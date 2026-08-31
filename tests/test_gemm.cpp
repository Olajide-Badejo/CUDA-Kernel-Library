// GEMM correctness against cuBLAS and, on small shapes, against a double
// precision CPU reference that also validates the cuBLAS row major orientation.
// Shapes cover square, non square, non tile aligned, smaller than one tile, and
// a zero dimension, per the Phase 1 gate in Section 10.
//
// Two more sections follow the shape table. One asserts that beta zero never
// reads C, by handing every rung a C full of NaN and requiring a finite result:
// that is the BLAS contract, and it used to fail here. The other asserts that
// dispatch is honest, by checking what kAuto picks and that an explicitly named
// algorithm the shape cannot take is refused instead of quietly rerouted.
//
// Exit code 0 means every case passed its tolerance; non zero means at least
// one failed. No test framework on purpose: this keeps the dependency surface
// small and the failure output is a plain table.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <limits>
#include <string>
#include <vector>

#include <cuda_fp16.h>

#include "ckl/context.hpp"
#include "ckl/cuda_check.hpp"
#include "ckl/device_buffer.hpp"
#include "ckl/gemm.hpp"
#include "ckl/status.hpp"
#include "ckl/types.hpp"
#include "reference.hpp"

namespace {

using LaunchFn = std::function<void(const float*, const float*, float*, int, int, int, float, float,
                                    cudaStream_t)>;

// Runs a GEMM launcher on device and returns the host C result.
std::vector<float> run_device_gemm(const LaunchFn& launch, const std::vector<float>& a,
                                   const std::vector<float>& b, const std::vector<float>& c_in,
                                   int m, int n, int k, float alpha, float beta) {
    ckl::DeviceBuffer<float> da(a.size());
    ckl::DeviceBuffer<float> db(b.size());
    ckl::DeviceBuffer<float> dc(c_in.size());
    if (!a.empty())
        da.copy_from_host(a);
    if (!b.empty())
        db.copy_from_host(b);
    if (!c_in.empty())
        dc.copy_from_host(c_in);

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
        {"register", ckl::gemm_register},
        {"cp_async", ckl::gemm_cp_async},
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
    std::printf("  %-22s m=%-5d n=%-5d k=%-5d  cublas_vs_cpu=%.3e\n", s.label, s.m, s.n, s.k,
                err_cublas_vs_cpu);
    for (const auto& v : hand_written_variants()) {
        const auto out = run_device_gemm(v.launch, a, b, c0, s.m, s.n, s.k, alpha, beta);
        const double err = out.empty() ? 0.0 : ckl::relative_frobenius_error(out, cublas_as_ref);
        const bool pass = err < kTolerance;
        ok = ok && pass;
        std::printf("      %-10s vs cuBLAS = %.3e  %s\n", v.name, err, pass ? "PASS" : "FAIL");
    }
    return ok;
}

// beta zero must not read C. BLAS says an uninitialized or NaN C is legal input
// in that case, and the v1 kernels all computed alpha * acc + 0.0f * c[idx],
// which propagates the NaN. Every rung gets a C of NaN and has to come back
// finite and equal to alpha * (A * B).
bool run_beta_zero_case() {
    std::printf("beta zero does not read C (C seeded with NaN)\n");
    const int m = 256;
    const int n = 256;
    const int k = 256;
    const float alpha = 1.0f;
    const auto a = ckl::random_matrix(m, k, 0xbeef);
    const auto b = ckl::random_matrix(k, n, 0xcafe);
    const std::vector<float> nan_c(static_cast<std::size_t>(m) * n,
                                   std::numeric_limits<float>::quiet_NaN());
    const std::vector<float> zero_c(static_cast<std::size_t>(m) * n, 0.0f);

    // The reference is the same call with a zero C, which every kernel handles
    // whether or not it honors the contract.
    const auto ref = run_device_gemm(ckl::gemm_cublas, a, b, zero_c, m, n, k, alpha, 0.0f);
    const std::vector<double> ref_d(ref.begin(), ref.end());

    bool ok = true;
    for (const auto& v : hand_written_variants()) {
        const auto out = run_device_gemm(v.launch, a, b, nan_c, m, n, k, alpha, 0.0f);
        bool finite = true;
        for (float value : out) {
            if (!std::isfinite(value)) {
                finite = false;
                break;
            }
        }
        const double err = finite ? ckl::relative_frobenius_error(out, ref_d) : 1.0;
        const bool pass = finite && err < kTolerance;
        ok = ok && pass;
        std::printf("      %-10s finite=%d  err=%.3e  %s\n", v.name, finite ? 1 : 0, err,
                    pass ? "PASS" : "FAIL");
    }
    return ok;
}

// A square FP16 descriptor with packed leading dimensions.
ckl::GemmDesc fp16_desc(int n, ckl::Algo algo) {
    ckl::GemmDesc d;
    d.layout = ckl::Layout::kRowMajor;
    d.m = n;
    d.n = n;
    d.k = n;
    d.dt_a = ckl::DType::kR16F;
    d.dt_b = ckl::DType::kR16F;
    d.dt_c = ckl::DType::kR32F;
    d.lda = n;
    d.ldb = n;
    d.ldc = n;
    d.algo = algo;
    return d;
}

// Runs one descriptor over matrices of ones, so every output element should be
// exactly k. Returns the status and writes the algorithm the dispatcher took.
ckl::Status run_desc(ckl::Context& ctx, const ckl::GemmDesc& d, ckl::Algo* chosen,
                     double* max_abs_error) {
    const std::size_t elems = static_cast<std::size_t>(d.m) * static_cast<std::size_t>(d.k);
    ckl::DeviceBuffer<__half> da(elems);
    ckl::DeviceBuffer<__half> db(static_cast<std::size_t>(d.k) * static_cast<std::size_t>(d.n));
    ckl::DeviceBuffer<float> dc(static_cast<std::size_t>(d.m) * static_cast<std::size_t>(d.n));
    const std::vector<__half> ones(elems, __float2half(1.0f));
    da.copy_from_host(ones);
    db.copy_from_host(ones.data(), db.size());
    dc.zero();

    const float alpha = 1.0f;
    const float beta = 0.0f;
    const ckl::Status s = ckl::gemm(ctx, d, &alpha, da.data(), db.data(), &beta, dc.data(), chosen);
    if (s != ckl::Status::kSuccess) {
        return s;
    }
    CKL_CUDA_CHECK(cudaDeviceSynchronize());
    double worst = 0.0;
    for (float value : dc.to_host()) {
        worst = std::max(worst, std::fabs(static_cast<double>(value) - static_cast<double>(d.k)));
    }
    *max_abs_error = worst;
    return s;
}

// Dispatch has to say what it did. kAuto on a big aligned FP16 shape should land
// on the top tensor kernel; the same call on a shape no hand kernel can tile
// should report the vendor path it actually took; and naming a kernel the shape
// cannot take is refused rather than rerouted.
bool run_dispatch_smoke() {
    std::printf("dispatch honesty\n");
    ckl::Context ctx;
    bool ok = true;

    {
        ckl::Algo chosen = ckl::Algo::kAuto;
        double err = 0.0;
        const ckl::Status s = run_desc(ctx, fp16_desc(4096, ckl::Algo::kAuto), &chosen, &err);
        const bool pass = s == ckl::Status::kSuccess && chosen == ckl::Algo::kMmaOpt && err <= 0.0;
        ok = ok && pass;
        std::printf("      kAuto   4096^3 fp16 -> status=%-14s chosen=%-9s err=%.1f  %s\n",
                    ckl::status_string(s), ckl::algo_name(chosen), err, pass ? "PASS" : "FAIL");
    }
    {
        ckl::Algo chosen = ckl::Algo::kAuto;
        double err = 0.0;
        const ckl::Status s = run_desc(ctx, fp16_desc(100, ckl::Algo::kAuto), &chosen, &err);
        // 100 divides none of the block factors, so no hand kernel can take it.
        // What matters is that chosen names the path that actually ran.
        const bool pass = s == ckl::Status::kSuccess && chosen == ckl::Algo::kCublas && err <= 0.0;
        ok = ok && pass;
        std::printf("      kAuto   100^3  fp16 -> status=%-14s chosen=%-9s err=%.1f  %s\n",
                    ckl::status_string(s), ckl::algo_name(chosen), err, pass ? "PASS" : "FAIL");
    }
    {
        ckl::Algo chosen = ckl::Algo::kAuto;
        double err = 0.0;
        const ckl::Status s = run_desc(ctx, fp16_desc(100, ckl::Algo::kMmaOpt), &chosen, &err);
        const bool pass = s == ckl::Status::kNotSupported && chosen == ckl::Algo::kMmaOpt;
        ok = ok && pass;
        std::printf("      kMmaOpt 100^3  fp16 -> status=%-14s chosen=%-9s  %s\n",
                    ckl::status_string(s), ckl::algo_name(chosen), pass ? "PASS" : "FAIL");
    }
    {
        // A leading dimension smaller than the row length is a malformed
        // descriptor, not an unsupported one.
        ckl::GemmDesc d = fp16_desc(128, ckl::Algo::kAuto);
        d.ldc = 1;
        ckl::Algo chosen = ckl::Algo::kAuto;
        double err = 0.0;
        const ckl::Status s = run_desc(ctx, d, &chosen, &err);
        const bool pass = s == ckl::Status::kInvalidValue;
        ok = ok && pass;
        std::printf("      bad ldc             -> status=%-14s  %s\n", ckl::status_string(s),
                    pass ? "PASS" : "FAIL");
    }
    {
        // gemm_query answers without launching anything.
        const ckl::Algo q = ckl::gemm_query(ctx, fp16_desc(4096, ckl::Algo::kAuto));
        const bool pass = q == ckl::Algo::kMmaOpt;
        ok = ok && pass;
        std::printf("      gemm_query 4096^3   -> %-9s  %s\n", ckl::algo_name(q),
                    pass ? "PASS" : "FAIL");
    }
    return ok;
}

}  // namespace

int main() {
    const std::vector<Shape> shapes = {
        {256, 256, 256, "square aligned"},   {384, 512, 128, "non square aligned"},
        {129, 257, 193, "non tile aligned"}, {7, 5, 11, "smaller than one tile"},
        {512, 384, 640, "non square large"}, {0, 128, 128, "zero m dimension"},
        {128, 0, 128, "zero n dimension"},   {64, 64, 0, "zero k dimension"},
    };

    std::printf("GEMM correctness (tolerance relative Frobenius < %.0e)\n", kTolerance);
    bool all_ok = true;
    for (const auto& s : shapes) {
        all_ok = run_case(s) && all_ok;
    }
    all_ok = run_beta_zero_case() && all_ok;
    all_ok = run_dispatch_smoke() && all_ok;
    std::printf("%s\n", all_ok ? "all GEMM cases passed" : "GEMM cases FAILED");
    return all_ok ? 0 : 1;
}
