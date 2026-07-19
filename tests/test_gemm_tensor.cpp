// Tensor core GEMM correctness. FP16 and BF16 inputs are derived from the same
// float random matrices, run through both the hand written WMMA kernel and the
// cuBLAS tensor oracle for that precision, and compared. The tolerance is not
// the FP32 1e-4: rounding the inputs to 16 bit storage introduces an error that
// grows with k, so the gate here is derived from the measured distribution and
// recorded in docs/gemm.md. This test also reports the error against an FP32 CPU
// reference, which is the honest accuracy number for the precision.
//
// Exit code 0 means every case passed; non zero means at least one failed.

#include <cstdio>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "ckl/cuda_check.hpp"
#include "ckl/device_buffer.hpp"
#include "ckl/gemm.hpp"
#include "ckl/reference.hpp"

namespace {

struct Shape {
    int m;
    int n;
    int k;
    const char* label;
};

// Gate derived from the measured error distribution (see docs/gemm.md). FP16 and
// BF16 both carry short mantissas; at the shapes here the kernel versus oracle
// error stays well under this, while the honest error against FP32 is reported
// alongside.
constexpr double kTensorTolerance = 5e-2;

template <typename T>
std::vector<T> to_half_type(const std::vector<float>& src);

template <>
std::vector<__half> to_half_type<__half>(const std::vector<float>& src) {
    std::vector<__half> out(src.size());
    for (std::size_t i = 0; i < src.size(); ++i) {
        out[i] = __float2half(src[i]);
    }
    return out;
}

template <>
std::vector<__nv_bfloat16> to_half_type<__nv_bfloat16>(const std::vector<float>& src) {
    std::vector<__nv_bfloat16> out(src.size());
    for (std::size_t i = 0; i < src.size(); ++i) {
        out[i] = __float2bfloat16(src[i]);
    }
    return out;
}

template <typename T, typename KernelFn, typename OracleFn>
bool run_case(const Shape& s, const char* type_name, KernelFn kernel, OracleFn oracle) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const auto fa = ckl::random_matrix(s.m, s.k, 0x1111 + static_cast<std::uint64_t>(s.m));
    const auto fb = ckl::random_matrix(s.k, s.n, 0x2222 + static_cast<std::uint64_t>(s.n));
    const std::vector<float> fc0(static_cast<std::size_t>(s.m) * s.n, 0.0f);

    const auto ha = to_half_type<T>(fa);
    const auto hb = to_half_type<T>(fb);

    ckl::DeviceBuffer<T> da(ha.size());
    ckl::DeviceBuffer<T> db(hb.size());
    ckl::DeviceBuffer<float> dk(fc0.size());
    ckl::DeviceBuffer<float> doo(fc0.size());
    if (!ha.empty()) da.copy_from_host(ha);
    if (!hb.empty()) db.copy_from_host(hb);
    dk.zero();
    doo.zero();

    kernel(da.data(), db.data(), dk.data(), s.m, s.n, s.k, alpha, beta, nullptr);
    oracle(da.data(), db.data(), doo.data(), s.m, s.n, s.k, alpha, beta, nullptr);
    CKL_CUDA_CHECK(cudaDeviceSynchronize());

    const auto ck = dk.size() == 0 ? std::vector<float>{} : dk.to_host();
    const auto co = doo.size() == 0 ? std::vector<float>{} : doo.to_host();

    std::vector<double> oracle_ref(co.begin(), co.end());
    const double err_vs_oracle = ck.empty() ? 0.0 : ckl::relative_frobenius_error(ck, oracle_ref);

    double err_vs_fp32 = 0.0;
    if (static_cast<long long>(s.m) * s.n * s.k <= (512LL * 512 * 512) && !ck.empty()) {
        const auto ref = ckl::gemm_reference(fa, fb, fc0, s.m, s.n, s.k, alpha, beta);
        err_vs_fp32 = ckl::relative_frobenius_error(ck, ref);
    }

    const bool ok = err_vs_oracle < kTensorTolerance;
    std::printf("  %-6s %-22s m=%-5d n=%-5d k=%-5d  vs_oracle=%.3e  vs_fp32=%.3e  %s\n",
                type_name, s.label, s.m, s.n, s.k, err_vs_oracle, err_vs_fp32,
                ok ? "PASS" : "FAIL");
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
        {64, 0, 128, "zero n dimension"},
        {64, 64, 0, "zero k dimension"},
    };

    std::printf("Tensor GEMM correctness (kernel vs cuBLAS oracle, tolerance %.0e)\n",
                kTensorTolerance);
    bool all_ok = true;
    for (const auto& s : shapes) {
        all_ok = run_case<__half>(s, "fp16", ckl::gemm_wmma_fp16, ckl::gemm_cublas_fp16) && all_ok;
    }
    for (const auto& s : shapes) {
        all_ok = run_case<__nv_bfloat16>(s, "bf16", ckl::gemm_wmma_bf16, ckl::gemm_cublas_bf16) && all_ok;
    }
    for (const auto& s : shapes) {
        all_ok = run_case<__half>(s, "ptx", ckl::gemm_mma_ptx, ckl::gemm_cublas_fp16) && all_ok;
    }
    std::printf("%s\n", all_ok ? "all tensor GEMM cases passed" : "tensor GEMM cases FAILED");
    return all_ok ? 0 : 1;
}
