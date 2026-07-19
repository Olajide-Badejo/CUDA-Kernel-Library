// GEMV benchmark: times each variant against cuBLAS SGEMV and reports the
// achieved effective bandwidth. GEMV moves about m*n*4 bytes to read A (which
// dominates x and y), so bandwidth is the honest yardstick here, not GFLOP/s.
// The number to watch is how close each variant gets to the measured streaming
// bandwidth from the device probe (about 540 GB/s).

#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

#include "ckl/cuda_check.hpp"
#include "ckl/device_buffer.hpp"
#include "ckl/event_timer.hpp"
#include "ckl/gemv.hpp"
#include "ckl/reference.hpp"

namespace {

using LaunchFn = std::function<void(const float*, const float*, float*, int, int,
                                    float, float, cudaStream_t)>;

double gemv_gbps(int m, int n, double ms) {
    // A dominates the traffic: m*n floats read once, plus x and y.
    const double bytes = (static_cast<double>(m) * n + n + 2.0 * m) * sizeof(float);
    return bytes / (ms / 1000.0) / 1.0e9;
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<int> sizes = {1024, 2048, 4096, 8192};
    if (argc > 1) {
        sizes.clear();
        for (int i = 1; i < argc; ++i) {
            sizes.push_back(std::atoi(argv[i]));
        }
    }
    const std::vector<std::pair<std::string, LaunchFn>> variants = {
        {"naive", ckl::gemv_naive},
        {"warp", ckl::gemv_warp},
        {"vectorized", ckl::gemv_vectorized},
        {"cublas", ckl::gemv_cublas},
    };

    std::printf("%-6s %-12s %12s %12s %10s\n", "size", "variant", "median_ms", "gbps", "pct_cublas");
    for (int sz : sizes) {
        const int m = sz;
        const int n = sz;
        const auto ha = ckl::random_matrix(m, n, 3);
        const auto hx = ckl::random_matrix(n, 1, 5);
        ckl::DeviceBuffer<float> da(ha.size());
        ckl::DeviceBuffer<float> dx(hx.size());
        ckl::DeviceBuffer<float> dy(static_cast<std::size_t>(m));
        da.copy_from_host(ha);
        dx.copy_from_host(hx);
        dy.zero();

        double cublas_gbps = 0.0;
        std::vector<std::pair<std::string, ckl::TimingStats>> rows;
        for (const auto& v : variants) {
            ckl::TimingStats st = ckl::time_stream([&](cudaStream_t s) {
                v.second(da.data(), dx.data(), dy.data(), m, n, 1.0f, 0.0f, s);
            });
            if (v.first == "cublas") {
                cublas_gbps = gemv_gbps(m, n, st.median_ms);
            }
            rows.emplace_back(v.first, st);
        }
        for (const auto& r : rows) {
            const double gbps = gemv_gbps(m, n, r.second.median_ms);
            const double pct = cublas_gbps > 0.0 ? 100.0 * gbps / cublas_gbps : 0.0;
            std::printf("%-6d %-12s %12.4f %12.1f %9.1f%%\n",
                        sz, r.first.c_str(), r.second.median_ms, gbps, pct);
        }
    }
    return 0;
}
