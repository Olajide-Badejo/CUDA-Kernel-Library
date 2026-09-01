#pragma once

// CUDA event based timing, and the measurement protocol of Section 13 built on
// top of it. Times are milliseconds throughout.
//
// The old protocol was five warmups and twenty fixed reps, reported as a median
// and an interquartile range. That still exists, because the per family drivers
// and the roofline tool are written against it, but three things were missing
// and every one of them changed a number rather than only its error bar.
//
//   Samples are kept. TimingStats now carries every rep, so a row in the
//   canonical JSONL can be re-reduced by whoever reads it instead of being
//   trusted as a median someone else computed.
//
//   Reps are earned rather than assumed. Adaptive mode runs to whichever of
//   1000 reps or two seconds comes first, so a 40 microsecond kernel is not
//   summarized from twenty samples while a 10 millisecond one wastes a minute.
//
//   The cache state and the launch path are inputs, not accidents. An L2 flush
//   can be enqueued between reps (a scratch write on the same stream, ordered
//   before the start event, so it is not inside the timed region), and the
//   inner launches can be captured into a CUDA graph so launch overhead can be
//   measured by difference instead of being conflated with kernel time.
//
// Everything here enqueues on the stream it is given and synchronizes only
// through events. Graph capture needs a stream that is not the legacy default
// one, so a caller that wants launch_mode graph has to create a stream.

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <functional>
#include <vector>

#include <cuda_runtime.h>

#include "ckl/cuda_check.hpp"

namespace ckl {

/// One timed window, reduced, with every sample it was reduced from.
struct TimingStats {
    double median_ms = 0.0;
    double iqr_ms = 0.0;
    double min_ms = 0.0;
    int reps = 0;
    /// One entry per timed rep, in the order measured. Sorted copies are made
    /// locally; this stays in launch order so a warmup trend stays visible.
    std::vector<double> samples;
};

/// How to run one timed window.
struct TimingOptions {
    /// Stream every launch is enqueued on. Graph capture refuses the legacy
    /// default stream, so leave this null only for stream mode.
    cudaStream_t stream = nullptr;
    int warmups = 5;
    /// Rep count in fixed mode, and the floor in adaptive mode.
    int reps = 20;
    /// Run to min(max_reps, budget_s) instead of exactly reps.
    bool adaptive = false;
    int max_reps = 1000;
    double budget_s = 2.0;
    /// Launches inside one event pair. The reported sample is the elapsed time
    /// divided by this, so a per launch number stays comparable across modes.
    int inner = 1;
    /// Capture the inner launches into a CUDA graph and launch the graph once
    /// per rep. Graphs are a measurement tool here, not a library feature.
    bool graph = false;
    /// Scratch buffer written before each event pair to evict L2. Null disables
    /// the flush. The write is stream ordered ahead of the start event, so it
    /// is outside the timed region.
    void* flush_buffer = nullptr;
    std::size_t flush_bytes = 0;
    /// Enqueued before each event pair, for kernels that consume their input
    /// (TRSM overwrites its right hand side). Also outside the timed region.
    std::function<void(cudaStream_t)> prologue;
};

namespace detail {

inline double quantile_of(const std::vector<double>& sorted, double frac) {
    const double pos = frac * static_cast<double>(sorted.size() - 1);
    const auto lo = static_cast<std::size_t>(pos);
    const double rem = pos - static_cast<double>(lo);
    if (lo + 1 < sorted.size()) {
        return sorted[lo] * (1.0 - rem) + sorted[lo + 1] * rem;
    }
    return sorted[lo];
}

inline TimingStats reduce_samples(std::vector<double> samples) {
    TimingStats stats;
    stats.reps = static_cast<int>(samples.size());
    stats.samples = samples;
    std::sort(samples.begin(), samples.end());
    stats.min_ms = samples.front();
    stats.median_ms = quantile_of(samples, 0.5);
    stats.iqr_ms = quantile_of(samples, 0.75) - quantile_of(samples, 0.25);
    return stats;
}

// The flush is a memset of an over-L2-size buffer. The byte value changes every
// call so nothing downstream can treat two flushes as the same write.
inline void enqueue_flush(const TimingOptions& opt, cudaStream_t stream, int rep) {
    if (opt.flush_buffer == nullptr || opt.flush_bytes == 0) {
        return;
    }
    CKL_CUDA_CHECK(cudaMemsetAsync(opt.flush_buffer, 1 + (rep & 0x7e), opt.flush_bytes, stream));
}

}  // namespace detail

/**
 * Times a callable that enqueues work on a stream, under the full protocol.
 *
 * The callable must only enqueue; every synchronization happens here. In graph
 * mode it is called once during capture and never again, so it must not depend
 * on host state that changes between reps.
 */
inline TimingStats time_stream_ex(const std::function<void(cudaStream_t)>& launch,
                                  const TimingOptions& opt) {
    cudaStream_t stream = opt.stream;
    const int inner = opt.inner > 0 ? opt.inner : 1;

    cudaEvent_t start;
    cudaEvent_t stop;
    CKL_CUDA_CHECK(cudaEventCreate(&start));
    CKL_CUDA_CHECK(cudaEventCreate(&stop));

    // Warm up on the plain path in both modes: the first launch of a kernel
    // pays for module load and, for cuBLAS, for its own workspace, and neither
    // belongs in a captured graph or in a sample.
    for (int i = 0; i < opt.warmups; ++i) {
        if (opt.prologue) {
            opt.prologue(stream);
        }
        launch(stream);
    }
    CKL_CUDA_CHECK(cudaStreamSynchronize(stream));

    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    if (opt.graph) {
        CKL_CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        for (int i = 0; i < inner; ++i) {
            launch(stream);
        }
        CKL_CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
        CKL_CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, 0));
        // One untimed graph launch so instantiation and the first upload are
        // not charged to rep zero.
        CKL_CUDA_CHECK(cudaGraphLaunch(graph_exec, stream));
        CKL_CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    const int floor_reps = opt.adaptive ? std::min(opt.reps, opt.max_reps) : opt.reps;
    const int ceiling_reps = opt.adaptive ? opt.max_reps : opt.reps;
    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(floor_reps));

    const auto wall_start = std::chrono::steady_clock::now();
    for (int i = 0; i < ceiling_reps; ++i) {
        if (opt.prologue) {
            opt.prologue(stream);
        }
        detail::enqueue_flush(opt, stream, i);
        CKL_CUDA_CHECK(cudaEventRecord(start, stream));
        if (opt.graph) {
            CKL_CUDA_CHECK(cudaGraphLaunch(graph_exec, stream));
        } else {
            for (int j = 0; j < inner; ++j) {
                launch(stream);
            }
        }
        CKL_CUDA_CHECK(cudaEventRecord(stop, stream));
        CKL_CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CKL_CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        samples.push_back(static_cast<double>(ms) / static_cast<double>(inner));
        if (opt.adaptive && i + 1 >= floor_reps) {
            const std::chrono::duration<double> spent =
                std::chrono::steady_clock::now() - wall_start;
            if (spent.count() >= opt.budget_s) {
                break;
            }
        }
    }

    if (graph_exec != nullptr) {
        CKL_CUDA_CHECK(cudaGraphExecDestroy(graph_exec));
    }
    if (graph != nullptr) {
        CKL_CUDA_CHECK(cudaGraphDestroy(graph));
    }
    CKL_CUDA_CHECK(cudaEventDestroy(start));
    CKL_CUDA_CHECK(cudaEventDestroy(stop));

    return detail::reduce_samples(std::move(samples));
}

/**
 * The fixed rep form the per family drivers and the roofline tool are written
 * against: warmups, then exactly reps timed launches on one stream.
 */
inline TimingStats time_stream(const std::function<void(cudaStream_t)>& launch,
                               cudaStream_t stream = nullptr, int warmups = 5, int reps = 20) {
    TimingOptions opt;
    opt.stream = stream;
    opt.warmups = warmups;
    opt.reps = reps;
    return time_stream_ex(launch, opt);
}

// GEMM flop count: two flops per multiply add, m*n*k of them.
inline double gemm_gflops(int m, int n, int k, double ms) {
    const double flops =
        2.0 * static_cast<double>(m) * static_cast<double>(n) * static_cast<double>(k);
    return flops / (ms / 1000.0) / 1.0e9;
}

}  // namespace ckl
