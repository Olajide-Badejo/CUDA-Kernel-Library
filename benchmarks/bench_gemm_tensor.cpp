// Tensor core GEMM benchmark: times the WMMA kernels against the cuBLAS tensor
// oracle for the same precision, reporting GFLOP/s and percent of cuBLAS. The
// comparison is per precision, as the compute bound gate and the 90 percent
// target require. Same timing protocol as the FP32 harness.

#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "ckl/cuda_check.hpp"
#include "ckl/device_buffer.hpp"
#include "ckl/event_timer.hpp"
#include "ckl/gemm.hpp"
#include "ckl/reference.hpp"

namespace {

template <typename T>
T from_float(float f);
template <>
__half from_float<__half>(float f) { return __float2half(f); }
template <>
__nv_bfloat16 from_float<__nv_bfloat16>(float f) { return __float2bfloat16(f); }

template <typename T>
std::vector<T> convert(const std::vector<float>& src) {
    std::vector<T> out(src.size());
    for (std::size_t i = 0; i < src.size(); ++i) {
        out[i] = from_float<T>(src[i]);
    }
    return out;
}

template <typename T, typename KernelFn, typename OracleFn>
void bench_precision(const char* type_name, int sz, KernelFn kernel, OracleFn oracle) {
    const int m = sz;
    const int n = sz;
    const int k = sz;
    const auto fa = ckl::random_matrix(m, k, 7);
    const auto fb = ckl::random_matrix(k, n, 9);
    const auto ha = convert<T>(fa);
    const auto hb = convert<T>(fb);

    ckl::DeviceBuffer<T> da(ha.size());
    ckl::DeviceBuffer<T> db(hb.size());
    ckl::DeviceBuffer<float> dc(static_cast<std::size_t>(m) * n);
    da.copy_from_host(ha);
    db.copy_from_host(hb);
    dc.zero();

    ckl::TimingStats ks = ckl::time_stream([&](cudaStream_t s) {
        kernel(da.data(), db.data(), dc.data(), m, n, k, 1.0f, 0.0f, s);
    });
    ckl::TimingStats os = ckl::time_stream([&](cudaStream_t s) {
        oracle(da.data(), db.data(), dc.data(), m, n, k, 1.0f, 0.0f, s);
    });

    const double kg = ckl::gemm_gflops(m, n, k, ks.median_ms);
    const double og = ckl::gemm_gflops(m, n, k, os.median_ms);
    const double pct = og > 0.0 ? 100.0 * kg / og : 0.0;
    std::printf("%-6s %-8d wmma %10.1f gflops   cublas %10.1f gflops   %6.1f%% of cublas\n",
                type_name, sz, kg, og, pct);
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<int> sizes = {512, 1024, 2048, 4096, 8192};
    if (argc > 1) {
        sizes.clear();
        for (int i = 1; i < argc; ++i) {
            sizes.push_back(std::atoi(argv[i]));
        }
    }
    for (int sz : sizes) {
        bench_precision<__half>("fp16", sz, ckl::gemm_wmma_fp16, ckl::gemm_cublas_fp16);
    }
    for (int sz : sizes) {
        bench_precision<__nv_bfloat16>("bf16", sz, ckl::gemm_wmma_bf16, ckl::gemm_cublas_bf16);
    }
    for (int sz : sizes) {
        bench_precision<__half>("ptx", sz, ckl::gemm_mma_ptx, ckl::gemm_cublas_fp16);
    }
    for (int sz : sizes) {
        bench_precision<__half>("ldm", sz, ckl::gemm_mma_ldm, ckl::gemm_cublas_fp16);
    }
    for (int sz : sizes) {
        bench_precision<__half>("opt", sz, ckl::gemm_mma_opt, ckl::gemm_cublas_fp16);
    }
    return 0;
}
