// Roofline profiler. Empirical mode measures the machine ceilings on this GPU
// (streaming bandwidth, FP32 compute from cuBLAS SGEMM, tensor compute from cuBLAS
// FP16 GEMM) and the achieved throughput of the ladder; analytical mode supplies
// the FLOP and byte counts per operation. It writes experiments/results/
// roofline.csv (label, intensity, achieved GFLOP/s, roof GFLOP/s, ceiling type,
// bound) and prints the ridge points. The report figure is generated from this
// CSV in the report phase.

#include <cstdio>
#include <string>
#include <vector>

#include <cuda_fp16.h>

#include "ckl/cuda_check.hpp"
#include "ckl/device_buffer.hpp"
#include "ckl/event_timer.hpp"
#include "ckl/gemm.hpp"
#include "ckl/gemv.hpp"
#include "ckl/reference.hpp"
#include "ckl/roofline.hpp"

namespace {

double measure_bandwidth() {
    const std::size_t bytes = std::size_t{256} << 20;  // 256 MiB
    ckl::DeviceBuffer<char> src(bytes);
    ckl::DeviceBuffer<char> dst(bytes);
    src.zero();
    dst.zero();
    for (int i = 0; i < 3; ++i) {
        CKL_CUDA_CHECK(cudaMemcpy(dst.data(), src.data(), bytes, cudaMemcpyDeviceToDevice));
    }
    ckl::TimingStats st = ckl::time_stream([&](cudaStream_t s) {
        CKL_CUDA_CHECK(cudaMemcpyAsync(dst.data(), src.data(), bytes, cudaMemcpyDeviceToDevice, s));
    });
    return 2.0 * static_cast<double>(bytes) / (st.median_ms / 1000.0);  // read + write
}

double measure_gemm_fp32(int n) {
    const auto a = ckl::random_matrix(n, n, 1);
    const auto b = ckl::random_matrix(n, n, 2);
    ckl::DeviceBuffer<float> da(a.size());
    ckl::DeviceBuffer<float> db(b.size());
    ckl::DeviceBuffer<float> dc(static_cast<std::size_t>(n) * n);
    da.copy_from_host(a);
    db.copy_from_host(b);
    dc.zero();
    ckl::TimingStats st = ckl::time_stream([&](cudaStream_t s) {
        ckl::gemm_cublas(da.data(), db.data(), dc.data(), n, n, n, 1.0f, 0.0f, s);
    });
    return ckl::gemm_flops(n, n, n) / (st.median_ms / 1000.0);
}

double measure_gemm_fp16(int n) {
    const auto fa = ckl::random_matrix(n, n, 1);
    const auto fb = ckl::random_matrix(n, n, 2);
    std::vector<__half> a(fa.size());
    std::vector<__half> b(fb.size());
    for (std::size_t i = 0; i < a.size(); ++i) { a[i] = __float2half(fa[i]); b[i] = __float2half(fb[i]); }
    ckl::DeviceBuffer<__half> da(a.size());
    ckl::DeviceBuffer<__half> db(b.size());
    ckl::DeviceBuffer<float> dc(static_cast<std::size_t>(n) * n);
    da.copy_from_host(a);
    db.copy_from_host(b);
    dc.zero();
    ckl::TimingStats st = ckl::time_stream([&](cudaStream_t s) {
        ckl::gemm_cublas_fp16(da.data(), db.data(), dc.data(), n, n, n, 1.0f, 0.0f, s);
    });
    return ckl::gemm_flops(n, n, n) / (st.median_ms / 1000.0);
}

// Measure a hand written FP32 GEMM variant at n cubed.
template <typename Fn>
double measure_variant_fp32(Fn fn, int n) {
    const auto a = ckl::random_matrix(n, n, 3);
    const auto b = ckl::random_matrix(n, n, 4);
    ckl::DeviceBuffer<float> da(a.size());
    ckl::DeviceBuffer<float> db(b.size());
    ckl::DeviceBuffer<float> dc(static_cast<std::size_t>(n) * n);
    da.copy_from_host(a);
    db.copy_from_host(b);
    dc.zero();
    ckl::TimingStats st = ckl::time_stream([&](cudaStream_t s) {
        fn(da.data(), db.data(), dc.data(), n, n, n, 1.0f, 0.0f, s);
    });
    return ckl::gemm_flops(n, n, n) / (st.median_ms / 1000.0);
}

template <typename Fn>
double measure_variant_fp16(Fn fn, int n) {
    const auto fa = ckl::random_matrix(n, n, 3);
    const auto fb = ckl::random_matrix(n, n, 4);
    std::vector<__half> a(fa.size());
    std::vector<__half> b(fb.size());
    for (std::size_t i = 0; i < a.size(); ++i) { a[i] = __float2half(fa[i]); b[i] = __float2half(fb[i]); }
    ckl::DeviceBuffer<__half> da(a.size());
    ckl::DeviceBuffer<__half> db(b.size());
    ckl::DeviceBuffer<float> dc(static_cast<std::size_t>(n) * n);
    da.copy_from_host(a);
    db.copy_from_host(b);
    dc.zero();
    ckl::TimingStats st = ckl::time_stream([&](cudaStream_t s) {
        fn(da.data(), db.data(), dc.data(), n, n, n, 1.0f, 0.0f, s);
    });
    return ckl::gemm_flops(n, n, n) / (st.median_ms / 1000.0);
}

}  // namespace

int main() {
    const double bw = measure_bandwidth();
    const double fp32_peak = measure_gemm_fp32(8192);
    const double tensor_peak = measure_gemm_fp16(8192);

    ckl::MachineCeilings ceil{bw, fp32_peak, tensor_peak};

    const int n = 4096;
    std::vector<ckl::RooflinePoint> points;
    points.push_back({"gemm_naive", ckl::gemm_flops(n, n, n), ckl::gemm_bytes(n, n, n, 4),
                      measure_variant_fp32(ckl::gemm_naive, n), false});
    points.push_back({"gemm_cp_async", ckl::gemm_flops(n, n, n), ckl::gemm_bytes(n, n, n, 4),
                      measure_variant_fp32(ckl::gemm_cp_async, n), false});
    points.push_back({"gemm_wmma_fp16", ckl::gemm_flops(n, n, n), ckl::gemm_bytes(n, n, n, 2),
                      measure_variant_fp16(ckl::gemm_wmma_fp16, n), true});
    points.push_back({"gemm_mma_opt", ckl::gemm_flops(n, n, n), ckl::gemm_bytes(n, n, n, 2),
                      measure_variant_fp16(ckl::gemm_mma_opt, n), true});

    // GEMV warp at 8192 (memory bound point).
    {
        const int gn = 8192;
        const auto a = ckl::random_matrix(gn, gn, 5);
        const auto x = ckl::random_matrix(gn, 1, 6);
        ckl::DeviceBuffer<float> da(a.size());
        ckl::DeviceBuffer<float> dx(x.size());
        ckl::DeviceBuffer<float> dy(static_cast<std::size_t>(gn));
        da.copy_from_host(a);
        dx.copy_from_host(x);
        dy.zero();
        ckl::TimingStats st = ckl::time_stream([&](cudaStream_t s) {
            ckl::gemv_warp(da.data(), dx.data(), dy.data(), gn, gn, 1.0f, 0.0f, s);
        });
        const double f = ckl::gemv_flops(gn, gn);
        points.push_back({"gemv_warp", f, ckl::gemv_bytes(gn, gn, 4),
                          f / (st.median_ms / 1000.0), false});
    }

    const double ridge_fp32 = ckl::ridge_intensity(fp32_peak, bw);
    const double ridge_tensor = ckl::ridge_intensity(tensor_peak, bw);

    std::printf("Roofline (measured ceilings on this GPU)\n");
    std::printf("  streaming bandwidth : %.1f GB/s\n", bw / 1.0e9);
    std::printf("  FP32 peak (cuBLAS)  : %.1f TFLOP/s\n", fp32_peak / 1.0e12);
    std::printf("  tensor peak (cuBLAS): %.1f TFLOP/s\n", tensor_peak / 1.0e12);
    std::printf("  ridge FP32          : %.1f FLOP/byte\n", ridge_fp32);
    std::printf("  ridge tensor        : %.1f FLOP/byte\n", ridge_tensor);
    std::printf("%-16s %12s %12s %12s %10s %s\n",
                "variant", "intensity", "gflops", "roof_gflops", "pct_roof", "bound");

    // Companion file with the measured ceilings so the plot can draw the roofs.
    if (std::FILE* cf = std::fopen("experiments/results/roofline_ceilings.csv", "w"); cf != nullptr) {
        std::fprintf(cf, "quantity,value\n");
        std::fprintf(cf, "bandwidth_gbps,%.3f\n", bw / 1.0e9);
        std::fprintf(cf, "fp32_gflops,%.3f\n", fp32_peak / 1.0e9);
        std::fprintf(cf, "tensor_gflops,%.3f\n", tensor_peak / 1.0e9);
        std::fprintf(cf, "ridge_fp32,%.3f\n", ridge_fp32);
        std::fprintf(cf, "ridge_tensor,%.3f\n", ridge_tensor);
        std::fclose(cf);
    }

    std::FILE* csv = std::fopen("experiments/results/roofline.csv", "w");
    if (csv != nullptr) {
        std::fprintf(csv, "label,intensity_flop_per_byte,achieved_gflops,roof_gflops,ceiling,bound\n");
    }
    for (const auto& p : points) {
        const double peak = p.tensor ? tensor_peak : fp32_peak;
        const double roof = ckl::roofline_flops(p.intensity(), peak, bw) / 1.0e9;
        const double pct = roof > 0.0 ? 100.0 * p.achieved_gflops() / roof : 0.0;
        const double ridge = p.tensor ? ridge_tensor : ridge_fp32;
        const char* bound = p.intensity() < ridge ? "memory" : "compute";
        std::printf("%-16s %12.2f %12.1f %12.1f %9.1f%% %s\n",
                    p.label.c_str(), p.intensity(), p.achieved_gflops(), roof, pct, bound);
        if (csv != nullptr) {
            std::fprintf(csv, "%s,%.4f,%.1f,%.1f,%s,%s\n", p.label.c_str(), p.intensity(),
                         p.achieved_gflops(), roof, p.tensor ? "tensor" : "fp32", bound);
        }
    }
    if (csv != nullptr) {
        std::fclose(csv);
        std::printf("wrote experiments/results/roofline.csv\n");
    }
    return 0;
}
