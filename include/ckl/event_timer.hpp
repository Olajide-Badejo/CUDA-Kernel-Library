#pragma once

// CUDA event based timing with the protocol the whole project uses: some warm up
// launches, then a fixed number of timed reps, reported as the median and the
// interquartile range so a single slow rep (a context switch, an NVML sample)
// does not move the headline. Times are milliseconds.

#include <algorithm>
#include <functional>
#include <vector>

#include <cuda_runtime.h>

#include "ckl/cuda_check.hpp"

namespace ckl {

struct TimingStats {
    double median_ms = 0.0;
    double iqr_ms = 0.0;
    double min_ms = 0.0;
    int reps = 0;
};

// Times a callable that launches work on the given stream. The callable takes
// the stream and must only enqueue asynchronous work; synchronization is handled
// here through events.
inline TimingStats time_stream(const std::function<void(cudaStream_t)>& launch,
                               cudaStream_t stream = nullptr,
                               int warmups = 5, int reps = 20) {
    cudaEvent_t start;
    cudaEvent_t stop;
    CKL_CUDA_CHECK(cudaEventCreate(&start));
    CKL_CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < warmups; ++i) {
        launch(stream);
    }
    CKL_CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(reps));
    for (int i = 0; i < reps; ++i) {
        CKL_CUDA_CHECK(cudaEventRecord(start, stream));
        launch(stream);
        CKL_CUDA_CHECK(cudaEventRecord(stop, stream));
        CKL_CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CKL_CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        samples.push_back(static_cast<double>(ms));
    }

    CKL_CUDA_CHECK(cudaEventDestroy(start));
    CKL_CUDA_CHECK(cudaEventDestroy(stop));

    std::sort(samples.begin(), samples.end());
    TimingStats stats;
    stats.reps = reps;
    stats.min_ms = samples.front();
    const auto q = [&samples](double frac) {
        double pos = frac * static_cast<double>(samples.size() - 1);
        auto lo = static_cast<std::size_t>(pos);
        double rem = pos - static_cast<double>(lo);
        if (lo + 1 < samples.size()) {
            return samples[lo] * (1.0 - rem) + samples[lo + 1] * rem;
        }
        return samples[lo];
    };
    stats.median_ms = q(0.5);
    stats.iqr_ms = q(0.75) - q(0.25);
    return stats;
}

// GEMM flop count: two flops per multiply add, m*n*k of them.
inline double gemm_gflops(int m, int n, int k, double ms) {
    const double flops = 2.0 * static_cast<double>(m) * static_cast<double>(n) * static_cast<double>(k);
    return flops / (ms / 1000.0) / 1.0e9;
}

}  // namespace ckl
