// Exercises the NVML monitor by sampling while a GEMM runs for a short window,
// then prints the summary. Passing means NVML initialized and collected samples;
// the throttle flag is reported, not asserted, because whether a short burst
// throttles depends on the machine's thermal state. On a host without NVML the
// monitor reports unavailable and the test skips cleanly.

#include <chrono>
#include <cstdio>
#include <vector>

#include "ckl/cuda_check.hpp"
#include "ckl/device_buffer.hpp"
#include "ckl/gemm.hpp"
#include "ckl/nvml_monitor.hpp"
#include "ckl/reference.hpp"

int main() {
    const int n = 2048;
    const auto ha = ckl::random_matrix(n, n, 1);
    const auto hb = ckl::random_matrix(n, n, 2);
    ckl::DeviceBuffer<float> da(ha.size());
    ckl::DeviceBuffer<float> db(hb.size());
    ckl::DeviceBuffer<float> dc(static_cast<std::size_t>(n) * n);
    da.copy_from_host(ha);
    db.copy_from_host(hb);
    dc.zero();

    ckl::NvmlMonitor monitor(20);
    monitor.start();

    const auto t0 = std::chrono::steady_clock::now();
    while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < 0.5) {
        for (int i = 0; i < 20; ++i) {
            ckl::gemm_cublas(da.data(), db.data(), dc.data(), n, n, n, 1.0f, 0.0f, nullptr);
        }
        CKL_CUDA_CHECK(cudaDeviceSynchronize());
    }

    const ckl::NvmlSummary s = monitor.stop();
    std::printf("NVML monitor summary\n");
    std::printf("  available        : %s\n", s.available ? "yes" : "no");
    if (!s.available) {
        std::printf("  note             : %s\n", s.note.c_str());
        std::printf("NVML unavailable, skipping (not a failure)\n");
        return 0;
    }
    std::printf("  samples          : %d\n", s.samples);
    std::printf("  median SM clock  : %.0f MHz\n", s.median_sm_clock_mhz);
    std::printf("  max temperature  : %u C\n", s.max_temperature_c);
    std::printf("  max power        : %.1f W\n", s.max_power_w);
    std::printf("  mean GPU util    : %.1f percent\n", s.mean_gpu_util_pct);
    std::printf("  throttled        : %s (%s)\n", s.throttled ? "yes" : "no", s.note.c_str());

    const bool ok = s.samples > 0;
    std::printf("%s\n", ok ? "NVML monitor collected samples" : "NVML monitor FAILED to sample");
    return ok ? 0 : 1;
}
