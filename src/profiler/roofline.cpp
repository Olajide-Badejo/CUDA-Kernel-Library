// Roofline profiler. It measures the streaming bandwidth ceiling and the achieved
// throughput of the ladder, computes the compute ceilings from the hardware, and
// writes experiments/results/roofline.csv plus roofline_ceilings.csv. The report
// figure is generated from those CSVs in the report phase.
//
// The compute roofs are hardware, not cuBLAS. V1 set the tensor roof to
// measure_gemm_fp16(8192), which is a cuBLAS call, and then reported "percent of
// roof" as if it meant something independent; it was "percent of cuBLAS" drawn on
// log axes. That is defect A3 in docs/CORRECTIONS.md. Here:
//
//   tensor roof = SMs * SM clock * 512 FLOP per cycle
//   FP32 roof   = SMs * SM clock * 256 FLOP per cycle  (128 lanes, 2 FLOP per FMA)
//
// 512 FLOP per cycle per SM is the FP16 input, FP32 accumulate rate on this
// consumer part, which is half the FP16 accumulate rate. Everything this project
// runs uses CUBLAS_COMPUTE_32F, so that is the rate that applies. The marketing
// TOPS figure is not a denominator.
//
// The clock is NVML's graphics clock sampled during the measurement window, not
// the boost clock from the device properties, because the part does not sit at
// boost. If NVML is unavailable the profiler says so in the CSV and on stdout
// rather than quietly substituting the boost clock.
//
// The cuBLAS measurements are kept, as attainable performance rather than as a
// ceiling: a cublas_attainable row in roofline.csv and cublas_* entries in the
// ceilings file.

#include <cstdio>
#include <string>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "ckl/cuda_check.hpp"
#include "ckl/device_buffer.hpp"
#include "ckl/event_timer.hpp"
#include "ckl/gemm.hpp"
#include "ckl/gemv.hpp"
#include "ckl/nvml_monitor.hpp"
#include "ckl/reference.hpp"
#include "ckl/roofline.hpp"

namespace {

// FP16 in, FP32 accumulate. Half the FP16 accumulate rate on a consumer part.
constexpr double kTensorFlopPerCyclePerSm = 512.0;
// 128 FP32 lanes per SM, two FLOP per fused multiply add.
constexpr double kFp32FlopPerCyclePerSm = 256.0;

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
    for (std::size_t i = 0; i < a.size(); ++i) {
        a[i] = __float2half(fa[i]);
        b[i] = __float2half(fb[i]);
    }
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
    ckl::TimingStats st = ckl::time_stream(
        [&](cudaStream_t s) { fn(da.data(), db.data(), dc.data(), n, n, n, 1.0f, 0.0f, s); });
    return ckl::gemm_flops(n, n, n) / (st.median_ms / 1000.0);
}

template <typename Fn>
double measure_variant_fp16(Fn fn, int n) {
    const auto fa = ckl::random_matrix(n, n, 3);
    const auto fb = ckl::random_matrix(n, n, 4);
    std::vector<__half> a(fa.size());
    std::vector<__half> b(fb.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        a[i] = __float2half(fa[i]);
        b[i] = __float2half(fb[i]);
    }
    ckl::DeviceBuffer<__half> da(a.size());
    ckl::DeviceBuffer<__half> db(b.size());
    ckl::DeviceBuffer<float> dc(static_cast<std::size_t>(n) * n);
    da.copy_from_host(a);
    db.copy_from_host(b);
    dc.zero();
    ckl::TimingStats st = ckl::time_stream(
        [&](cudaStream_t s) { fn(da.data(), db.data(), dc.data(), n, n, n, 1.0f, 0.0f, s); });
    return ckl::gemm_flops(n, n, n) / (st.median_ms / 1000.0);
}

}  // namespace

int main() {
    int device = 0;
    CKL_CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop{};
    CKL_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    // Sample the clock across the whole measurement window, so the roof is the
    // roof of the machine as it actually ran.
    ckl::NvmlMonitor monitor(25);
    monitor.start();

    const double bw = measure_bandwidth();
    const double cublas_fp32 = measure_gemm_fp32(8192);
    const double cublas_tensor = measure_gemm_fp16(8192);

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
        points.push_back(
            {"gemv_warp", f, ckl::gemv_bytes(gn, gn, 4), f / (st.median_ms / 1000.0), false});
    }

    const ckl::NvmlSummary telemetry = monitor.stop();

    // The clock the roofs are built on, and where it came from. When NVML is not
    // available the profiler falls back to the boost clock from the driver, says
    // so in clock_source, and prints a warning; a roof built on boost is optimistic
    // because the part does not hold boost under a sustained GEMM.
    const char* clock_source = "nvml_median_graphics_clock";
    double clock_mhz = telemetry.median_sm_clock_mhz;
    if (!telemetry.available || clock_mhz <= 0.0) {
        int core_clock_khz = 0;
        CKL_CUDA_CHECK(cudaDeviceGetAttribute(&core_clock_khz, cudaDevAttrClockRate, device));
        clock_mhz = static_cast<double>(core_clock_khz) / 1000.0;
        clock_source = "cuda_boost_clock_attribute";
        std::printf("WARNING: NVML unavailable (%s); the compute roofs below use the boost "
                    "clock, which the GPU does not hold under load.\n",
                    telemetry.note.empty() ? "no detail" : telemetry.note.c_str());
    }

    const double sms = static_cast<double>(prop.multiProcessorCount);
    const double clock_hz = clock_mhz * 1.0e6;
    const double fp32_roof = sms * clock_hz * kFp32FlopPerCyclePerSm;
    const double tensor_roof = sms * clock_hz * kTensorFlopPerCyclePerSm;

    const double ridge_fp32 = ckl::ridge_intensity(fp32_roof, bw);
    const double ridge_tensor = ckl::ridge_intensity(tensor_roof, bw);

    std::printf("Roofline on %s\n", prop.name);
    std::printf("  SM count               : %d\n", prop.multiProcessorCount);
    std::printf("  SM clock (%s) : %.0f MHz\n", clock_source, clock_mhz);
    std::printf("  streaming bandwidth    : %.1f GB/s (measured)\n", bw / 1.0e9);
    std::printf("  FP32 roof (hardware)   : %.1f TFLOP/s = %d SM x %.0f MHz x %.0f FLOP/cycle\n",
                fp32_roof / 1.0e12, prop.multiProcessorCount, clock_mhz, kFp32FlopPerCyclePerSm);
    std::printf("  tensor roof (hardware) : %.1f TFLOP/s = %d SM x %.0f MHz x %.0f FLOP/cycle\n",
                tensor_roof / 1.0e12, prop.multiProcessorCount, clock_mhz,
                kTensorFlopPerCyclePerSm);
    std::printf("  cuBLAS SGEMM attained  : %.1f TFLOP/s (%.1f%% of the FP32 roof)\n",
                cublas_fp32 / 1.0e12, 100.0 * cublas_fp32 / fp32_roof);
    std::printf("  cuBLAS FP16 attained   : %.1f TFLOP/s (%.1f%% of the tensor roof)\n",
                cublas_tensor / 1.0e12, 100.0 * cublas_tensor / tensor_roof);
    std::printf("  ridge FP32             : %.1f FLOP/byte\n", ridge_fp32);
    std::printf("  ridge tensor           : %.1f FLOP/byte\n", ridge_tensor);
    std::printf("%-20s %12s %12s %12s %10s %s\n", "variant", "intensity", "gflops", "roof_gflops",
                "pct_roof", "bound");

    // Companion file with the ceilings so the plot can draw the roofs and title
    // itself from the device name rather than a hard coded model.
    if (std::FILE* cf = std::fopen("experiments/results/roofline_ceilings.csv", "w");
        cf != nullptr) {
        std::fprintf(cf, "quantity,value\n");
        std::fprintf(cf, "device_name,%s\n", prop.name);
        std::fprintf(cf, "sm_count,%d\n", prop.multiProcessorCount);
        std::fprintf(cf, "clock_mhz,%.0f\n", clock_mhz);
        std::fprintf(cf, "clock_source,%s\n", clock_source);
        std::fprintf(cf, "bandwidth_gbps,%.3f\n", bw / 1.0e9);
        std::fprintf(cf, "fp32_gflops,%.3f\n", fp32_roof / 1.0e9);
        std::fprintf(cf, "tensor_gflops,%.3f\n", tensor_roof / 1.0e9);
        std::fprintf(cf, "cublas_fp32_gflops,%.3f\n", cublas_fp32 / 1.0e9);
        std::fprintf(cf, "cublas_tensor_gflops,%.3f\n", cublas_tensor / 1.0e9);
        std::fprintf(cf, "ridge_fp32,%.3f\n", ridge_fp32);
        std::fprintf(cf, "ridge_tensor,%.3f\n", ridge_tensor);
        std::fclose(cf);
    }

    // cuBLAS is an operating point like any other kernel, so it goes in the same
    // table. Its intensity is the FP16 GEMM at 8192 cubed, two bytes per element.
    const double cublas_flops = ckl::gemm_flops(8192, 8192, 8192);
    points.push_back({"cublas_attainable", cublas_flops, ckl::gemm_bytes(8192, 8192, 8192, 2),
                      cublas_tensor, true});

    std::FILE* csv = std::fopen("experiments/results/roofline.csv", "w");
    if (csv != nullptr) {
        std::fprintf(csv, "label,intensity_flop_per_byte,achieved_gflops,roof_gflops,ceiling,"
                          "bound,clock_mhz\n");
    }
    for (const auto& p : points) {
        const double peak = p.tensor ? tensor_roof : fp32_roof;
        const double roof = ckl::roofline_flops(p.intensity(), peak, bw) / 1.0e9;
        const double pct = roof > 0.0 ? 100.0 * p.achieved_gflops() / roof : 0.0;
        const double ridge = p.tensor ? ridge_tensor : ridge_fp32;
        const char* bound = p.intensity() < ridge ? "memory" : "compute";
        std::printf("%-20s %12.2f %12.1f %12.1f %9.1f%% %s\n", p.label.c_str(), p.intensity(),
                    p.achieved_gflops(), roof, pct, bound);
        if (csv != nullptr) {
            std::fprintf(csv, "%s,%.4f,%.1f,%.1f,%s,%s,%.0f\n", p.label.c_str(), p.intensity(),
                         p.achieved_gflops(), roof, p.tensor ? "tensor" : "fp32", bound, clock_mhz);
        }
    }
    if (csv != nullptr) {
        std::fclose(csv);
        std::printf("wrote experiments/results/roofline.csv\n");
    }
    return 0;
}
