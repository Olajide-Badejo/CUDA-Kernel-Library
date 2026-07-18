// Minimal GEMM benchmark for Phase 1: times each available variant against
// cuBLAS on a few square shapes and prints achieved GFLOP/s and percent of the
// cuBLAS baseline measured in the same process. The full sweep with JSONL
// output, NVML sampling, and the resumable matrix is Phase 9; this is the
// harness the early ladder rungs are timed with.

#include <cstdio>
#include <functional>
#include <string>
#include <vector>

#include "ckl/cuda_check.hpp"
#include "ckl/device_buffer.hpp"
#include "ckl/event_timer.hpp"
#include "ckl/gemm.hpp"
#include "ckl/reference.hpp"

namespace {

using LaunchFn = std::function<void(const float*, const float*, float*, int, int, int,
                                    float, float, cudaStream_t)>;

struct Variant {
    const char* name;
    LaunchFn launch;
};

double bench_one(const Variant& v, const ckl::DeviceBuffer<float>& da,
                 const ckl::DeviceBuffer<float>& db, ckl::DeviceBuffer<float>& dc,
                 int m, int n, int k, ckl::TimingStats& stats_out) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    stats_out = ckl::time_stream([&](cudaStream_t s) {
        v.launch(da.data(), db.data(), dc.data(), m, n, k, alpha, beta, s);
    });
    return ckl::gemm_gflops(m, n, k, stats_out.median_ms);
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<int> sizes = {256, 512, 1024, 2048, 4096};
    if (argc > 1) {
        sizes.clear();
        for (int i = 1; i < argc; ++i) {
            sizes.push_back(std::atoi(argv[i]));
        }
    }

    const std::vector<Variant> variants = {
        {"naive", ckl::gemm_naive},
        {"cublas", ckl::gemm_cublas},
    };

    std::printf("%-8s %-8s %14s %14s %10s %12s\n",
                "size", "variant", "median_ms", "gflops", "pct_cublas", "iqr_ms");

    for (int sz : sizes) {
        const int m = sz;
        const int n = sz;
        const int k = sz;
        const auto ha = ckl::random_matrix(m, k, 11);
        const auto hb = ckl::random_matrix(k, n, 22);
        ckl::DeviceBuffer<float> da(ha.size());
        ckl::DeviceBuffer<float> db(hb.size());
        ckl::DeviceBuffer<float> dc(static_cast<std::size_t>(m) * n);
        da.copy_from_host(ha);
        db.copy_from_host(hb);
        dc.zero();

        // Measure every variant first, then report percent of cuBLAS so the
        // baseline is known regardless of print order.
        struct Row {
            std::string name;
            ckl::TimingStats stats;
            double gflops;
        };
        std::vector<Row> rows;
        double cublas_gflops = 0.0;
        for (const auto& v : variants) {
            ckl::TimingStats stats;
            const double gflops = bench_one(v, da, db, dc, m, n, k, stats);
            if (std::string(v.name) == "cublas") {
                cublas_gflops = gflops;
            }
            rows.push_back({v.name, stats, gflops});
        }
        for (const auto& r : rows) {
            const double pct = cublas_gflops > 0.0 ? 100.0 * r.gflops / cublas_gflops : 0.0;
            std::printf("%-8d %-8s %14.3f %14.1f %9.1f%% %12.4f\n",
                        sz, r.name.c_str(), r.stats.median_ms, r.gflops, pct, r.stats.iqr_ms);
        }
    }
    return 0;
}
