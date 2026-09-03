// The FFT family driver: one transform length, every rung, the cuFFT baseline
// measured once, the transpose study on its own, and the 2D driver with the
// transpose share of its time.
//
// Everything here follows the Section 13 measurement protocol that
// docs/benchmarking.md sets out.
//
//   The plan is built before anything is timed. It holds the twiddle tables, the
//   ping pong workspace and the cuFFT handle, so a timed call enqueues launches
//   and nothing else. A CUPTI counter is armed across every timed lambda and the
//   run exits non-zero if it moves, which is what turns that claim into a gate
//   rather than a comment. --gate-red puts a malloc back inside the timed region
//   so the gate can be seen failing.
//
//   Nothing is timed before it is verified. Every variant runs one forward and
//   one inverse into separate buffers and the round trip is compared against
//   ckl::fft_tolerance before its timing loop opens. --verify-perturb scales the
//   answer so that gate can be seen failing too.
//
//   The baseline is measured once. cuFFT is a variant row like any other and
//   every percent in the table is against that one measurement, never against a
//   fresh one taken inside each row.
//
//   Every sample is kept and the row carries the median, the interquartile range
//   and the rep count. The L2 flush is on by default for any working set that
//   fits in the 48 MB of L2, because an unflushed small transform measures the
//   cache and not the kernel.
//
//   Every row carries its declared model bytes and its batch count. A single
//   unbatched 2^12 transform measures launch overhead rather than the kernel, so
//   the default batch fills two blocks per SM at the sizes where one transform is
//   one block.
//
// Every number this prints is at whatever clocks the machine happened to be at.
// Nothing here belongs in a document; the row of record comes from bench_all
// under sweep.py with the clock locked.
//
// Usage: bench_fft [log2n] [flags]

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "allocation_gate.hpp"
#include "ckl/cuda_check.hpp"
#include "ckl/device_buffer.hpp"
#include "ckl/fft.hpp"
#include "event_timer.hpp"
#include "reference.hpp"

namespace {

struct Options {
    int log2n = 20;
    int batch = 0;  // zero asks for the fill-the-machine default
    int warmups = 5;
    int reps = 50;
    bool flush = true;
    bool fast_twiddles = false;
    ckl::FftTranspose transpose = ckl::FftTranspose::kTiledPadded;
    // Test only. Scales the verification answer so the gate can be shown red.
    double verify_perturb = 1.0;
    // Test only. Puts a cudaMalloc and a cudaFree inside the timed lambda.
    bool gate_red = false;
};

struct Row {
    std::string variant;
    std::string chosen;
    int passes = 0;
    long long model_bytes = 0;
    long long twiddle_low = 0;
    long long twiddle_high = 0;
    double median_ms = 0.0;
    double iqr_ms = 0.0;
    int reps = 0;
    double effective_gbs = 0.0;
    double gflops = 0.0;
    double residual = 0.0;
    long long allocations = 0;
    bool ok = false;
    std::string note;
};

const ckl::FftAlgo kVariants[] = {
    ckl::FftAlgo::kRadix2Global, ckl::FftAlgo::kSharedResident, ckl::FftAlgo::kRadix4Global,
    ckl::FftAlgo::kRadix8Global, ckl::FftAlgo::kFourStep,       ckl::FftAlgo::kCufft,
};

std::vector<float2> random_signal(long long count, std::uint64_t seed) {
    const std::vector<float> flat = ckl::random_matrix(static_cast<int>(2 * count), 1, seed);
    std::vector<float2> out(static_cast<std::size_t>(count));
    for (long long i = 0; i < count; ++i) {
        out[static_cast<std::size_t>(i)] = make_float2(flat[static_cast<std::size_t>(2 * i)],
                                                       flat[static_cast<std::size_t>(2 * i) + 1]);
    }
    return out;
}

double relative_rms(const std::vector<float2>& got, const std::vector<float2>& want,
                    double perturb) {
    double num = 0.0;
    double den = 0.0;
    for (std::size_t i = 0; i < want.size(); ++i) {
        const double dx = static_cast<double>(got[i].x) * perturb - static_cast<double>(want[i].x);
        const double dy = static_cast<double>(got[i].y) * perturb - static_cast<double>(want[i].y);
        num += dx * dx + dy * dy;
        den += static_cast<double>(want[i].x) * static_cast<double>(want[i].x) +
               static_cast<double>(want[i].y) * static_cast<double>(want[i].y);
    }
    return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
}

double gbs_of(long long bytes, double ms) {
    return ms > 0.0 ? static_cast<double>(bytes) / (ms / 1000.0) / 1.0e9 : 0.0;
}

bool parse_transpose(const std::string& s, ckl::FftTranspose* out) {
    if (s == "naive") {
        *out = ckl::FftTranspose::kNaive;
    } else if (s == "tiled") {
        *out = ckl::FftTranspose::kTiledPadded;
    } else if (s == "strided") {
        *out = ckl::FftTranspose::kStridedShared;
    } else {
        return false;
    }
    return true;
}

Options parse(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            std::printf(
                "usage: %s [log2n] [flags]\n"
                "  log2n: log2 of the transform length, 10 to 24 (default 20)\n"
                "flags:\n"
                "  --batch N        transforms per launch; 0 fills two blocks per SM (default 0)\n"
                "  --warmups N      untimed launches before the loop (default 5)\n"
                "  --reps N         timed reps (default 50)\n"
                "  --no-flush       leave L2 warm between reps\n"
                "  --fast-twiddles  build the twiddle tables with __sincosf; a labelled rung\n"
                "  --transpose T    naive, tiled or strided, for the four step and 2D rows\n"
                "  --verify-perturb X  TEST ONLY: scale the answer so verification fails\n"
                "  --gate-red       TEST ONLY: allocate inside the timed region\n",
                argv[0]);
            std::exit(0);
        } else if (arg == "--batch" && i + 1 < argc) {
            opt.batch = std::atoi(argv[++i]);
        } else if (arg == "--warmups" && i + 1 < argc) {
            opt.warmups = std::atoi(argv[++i]);
        } else if (arg == "--reps" && i + 1 < argc) {
            opt.reps = std::atoi(argv[++i]);
        } else if (arg == "--no-flush") {
            opt.flush = false;
        } else if (arg == "--fast-twiddles") {
            opt.fast_twiddles = true;
        } else if (arg == "--transpose" && i + 1 < argc) {
            if (!parse_transpose(argv[++i], &opt.transpose)) {
                std::fprintf(stderr, "--transpose takes naive, tiled or strided\n");
                std::exit(2);
            }
        } else if (arg == "--verify-perturb" && i + 1 < argc) {
            opt.verify_perturb = std::atof(argv[++i]);
        } else if (arg == "--gate-red") {
            opt.gate_red = true;
        } else if (arg.rfind("--", 0) == 0) {
            std::fprintf(stderr, "unknown flag %s\n", arg.c_str());
            std::exit(2);
        } else {
            opt.log2n = std::atoi(arg.c_str());
        }
    }
    if (opt.log2n < 10 || opt.log2n > 24) {
        std::fprintf(stderr, "log2n must be between 10 and 24, and is %d\n", opt.log2n);
        std::exit(2);
    }
    return opt;
}

}  // namespace

// NOLINTNEXTLINE(bugprone-exception-escape): a ckl::Error here is fatal by design
int main(int argc, char** argv) {
    const Options opt = parse(argc, argv);
    const int n = 1 << opt.log2n;

    int device = 0;
    CKL_CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop{};
    CKL_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    // A shared resident transform is one block, and a 64 KB block is one block per
    // SM, so a small size needs a batch before it can fill the machine at all. Two
    // blocks per SM is the smallest batch that leaves the tail wave covered.
    int batch = opt.batch;
    if (batch <= 0) {
        batch = n <= ckl::FftPlan::shared_resident_max() ? 2 * prop.multiProcessorCount : 1;
    }

    ckl::bench::AllocationGate gate;
    const std::string gate_error = gate.start();
    if (!gate_error.empty()) {
        std::fprintf(stderr,
                     "the allocation gate could not be armed: %s\n"
                     "The claim that a timed call only enqueues launches needs it, so this is a "
                     "failure rather than a warning.\n",
                     gate_error.c_str());
        return 3;
    }

    ckl::FftPlanOptions plan_opt;
    plan_opt.batch = batch;
    plan_opt.fast_twiddles = opt.fast_twiddles;
    plan_opt.transpose = opt.transpose;
    ckl::FftPlan plan(n, plan_opt);

    const long long elems = static_cast<long long>(n) * batch;
    const std::vector<float2> host_in = random_signal(elems, 7);
    ckl::DeviceBuffer<float2> da(static_cast<std::size_t>(elems));
    ckl::DeviceBuffer<float2> db(static_cast<std::size_t>(elems));

    cudaStream_t stream = nullptr;
    CKL_CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamDefault));

    const std::size_t l2 =
        prop.l2CacheSize > 0 ? static_cast<std::size_t>(prop.l2CacheSize) : (std::size_t{48} << 20);
    const std::size_t working_set = 2 * static_cast<std::size_t>(elems) * sizeof(float2);
    const bool flush = opt.flush;
    ckl::DeviceBuffer<unsigned char> scratch;
    if (flush) {
        scratch = ckl::DeviceBuffer<unsigned char>(2 * l2);
    }

    const double bound = ckl::fft_tolerance(n);
    std::printf("transform n = %d (2^%d), batch %d, %lld complex points per call\n", n, opt.log2n,
                batch, elems);
    std::printf(
        "plan: kAuto would run %s; four step factors %d by %d; twiddles from %s\n",
        ckl::fft_algo_name(plan.query()), plan.four_step_n1(), plan.four_step_n2(),
        plan.fast_twiddles() ? "__sincosf (a labelled rung)" : "double precision, rounded once");
    std::printf("working set %.1f MB against %.1f MB of L2; L2 flush between reps: %s\n",
                static_cast<double>(working_set) / 1048576.0, static_cast<double>(l2) / 1048576.0,
                flush ? "on" : "off");
    std::printf("round trip tolerance 8 log2(n) FLT_EPSILON = %.4e\n", bound);
    std::printf("dram__bytes.sum against the declared model: pending an ncu round\n\n");

    std::vector<Row> rows;
    long long total_allocations = 0;
    int failures = 0;

    for (ckl::FftAlgo algo : kVariants) {
        Row row;
        row.variant = ckl::fft_algo_name(algo);
        if (!plan.supports(algo)) {
            row.note = std::string("refused: ") + plan.refusal(algo);
            row.chosen = row.variant;
            rows.push_back(row);
            continue;
        }
        row.passes = plan.passes(algo);
        row.model_bytes = plan.model_bytes(algo);
        row.twiddle_low = plan.twiddle_bytes_low(algo);
        row.twiddle_high = plan.twiddle_bytes_high(algo);

        // Verification first, and outside the gate: a first call is allowed to
        // touch whatever it needs to.
        da.copy_from_host(host_in);
        ckl::FftAlgo chosen = ckl::FftAlgo::kAuto;
        ckl::Status st = ckl::fft(plan, algo, ckl::FftDirection::kForward, da.data(), db.data(),
                                  &chosen, stream);
        row.chosen = ckl::fft_algo_name(chosen);
        if (st == ckl::Status::kSuccess) {
            st = ckl::fft(plan, algo, ckl::FftDirection::kInverse, db.data(), da.data(), nullptr,
                          stream);
        }
        if (st != ckl::Status::kSuccess) {
            row.note = std::string("failed: ") + ckl::status_string(st);
            ++failures;
            rows.push_back(row);
            continue;
        }
        ckl::fft_scale(da.data(), elems, 1.0f / static_cast<float>(n), stream);
        CKL_CUDA_CHECK(cudaStreamSynchronize(stream));
        if (chosen != algo) {
            std::fprintf(stderr,
                         "%s ran %s instead; a benchmark that measures a path its row does not "
                         "name is a defect\n",
                         ckl::fft_algo_name(algo), row.chosen.c_str());
            ++failures;
            rows.push_back(row);
            continue;
        }
        row.residual = relative_rms(da.to_host(), host_in, opt.verify_perturb);
        if (!(row.residual <= bound)) {
            row.note = "verification failed";
            std::fprintf(stderr, "%s: round trip %.3e against a bound of %.3e; nothing timed\n",
                         ckl::fft_algo_name(algo), row.residual, bound);
            ++failures;
            rows.push_back(row);
            continue;
        }

        auto launch = [&plan, &da, &db, algo, &opt](cudaStream_t s) {
            void* leak = nullptr;
            if (opt.gate_red) {
                CKL_CUDA_CHECK(cudaMalloc(&leak, 1024));
            }
            const ckl::Status inner =
                ckl::fft(plan, algo, ckl::FftDirection::kForward, da.data(), db.data(), nullptr, s);
            if (inner != ckl::Status::kSuccess) {
                std::fprintf(stderr, "the timed call failed: %s\n", ckl::status_string(inner));
                std::exit(4);
            }
            if (opt.gate_red) {
                CKL_CUDA_CHECK(cudaFree(leak));
            }
        };

        ckl::TimingOptions timing;
        timing.stream = stream;
        timing.warmups = opt.warmups;
        timing.reps = opt.reps;
        timing.flush_buffer = flush ? scratch.data() : nullptr;
        timing.flush_bytes = flush ? 2 * l2 : 0;

        gate.arm();
        const ckl::TimingStats stats = ckl::time_stream_ex(launch, timing);
        row.allocations = gate.disarm();
        total_allocations += row.allocations;

        row.median_ms = stats.median_ms;
        row.iqr_ms = stats.iqr_ms;
        row.reps = stats.reps;
        // The honest metric: declared model bytes over measured time, against the
        // measured DRAM roof. The FLOP count is in the table for continuity with
        // the literature and is not what this family is judged on.
        row.effective_gbs = gbs_of(row.model_bytes, stats.median_ms);
        row.gflops =
            stats.median_ms > 0.0 ? plan.flop_model() / (stats.median_ms / 1000.0) / 1.0e9 : 0.0;
        row.ok = true;
        rows.push_back(row);
    }

    double baseline_ms = 0.0;
    for (const Row& r : rows) {
        if (r.variant == std::string(ckl::fft_algo_name(ckl::FftAlgo::kCufft)) && r.ok) {
            baseline_ms = r.median_ms;
        }
    }

    std::printf("%-18s %-18s %6s %14s %10s %9s %6s %11s %10s %8s %10s\n", "variant", "chosen",
                "passes", "model_bytes", "median_ms", "iqr_ms", "reps", "eff_GB/s", "GFLOP/s",
                "pct_cufft", "allocs");
    for (const Row& r : rows) {
        if (!r.ok) {
            std::printf("%-18s %-18s %6s %14s %10s %9s %6s %11s %10s %8s %10s  %s\n",
                        r.variant.c_str(), r.chosen.c_str(), "-", "-", "-", "-", "-", "-", "-", "-",
                        "-", r.note.c_str());
            continue;
        }
        char pct[16] = "-";
        if (baseline_ms > 0.0) {
            std::snprintf(pct, sizeof(pct), "%.1f%%", 100.0 * baseline_ms / r.median_ms);
        }
        std::printf("%-18s %-18s %6d %14lld %10.4f %9.4f %6d %11.1f %10.1f %8s %10lld\n",
                    r.variant.c_str(), r.chosen.c_str(), r.passes, r.model_bytes, r.median_ms,
                    r.iqr_ms, r.reps, r.effective_gbs, r.gflops, pct, r.allocations);
    }
    std::printf(
        "\nmodel_bytes is passes times 16 n batch, the declared traffic Gate X checks a measured\n"
        "dram__bytes.sum against. The twiddle tables add between %lld and %lld bytes on top;\n"
        "they are factored, so the low end is the whole table read once.\n",
        rows.empty() ? 0 : rows.front().twiddle_low, rows.empty() ? 0 : rows.front().twiddle_high);
    std::printf("pct_cufft is against %s measured once in this process, not per row.\n",
                ckl::fft_algo_name(ckl::FftAlgo::kCufft));
    std::printf("round trip residuals:");
    for (const Row& r : rows) {
        if (r.ok) {
            std::printf(" %s=%.2e", r.variant.c_str(), r.residual);
        }
    }
    std::printf("\n");

    // --- the transpose study, on its own ---
    {
        const int side = 1 << (opt.log2n / 2);
        const long long tbytes = 16LL * side * side;
        const std::vector<float2> tin = random_signal(static_cast<long long>(side) * side, 11);
        ckl::DeviceBuffer<float2> ta(tin.size());
        ckl::DeviceBuffer<float2> tb(tin.size());
        ta.copy_from_host(tin);
        std::printf("\ntranspose study at %d by %d, model %lld bytes per call\n", side, side,
                    tbytes);
        std::printf("%-18s %10s %9s %11s\n", "variant", "median_ms", "iqr_ms", "eff_GB/s");
        for (ckl::FftTranspose v : {ckl::FftTranspose::kNaive, ckl::FftTranspose::kTiledPadded,
                                    ckl::FftTranspose::kStridedShared}) {
            auto launch = [&ta, &tb, side, v](cudaStream_t s) {
                ckl::fft_transpose(ta.data(), tb.data(), side, side, 1, v, nullptr, 1.0f, s);
            };
            ckl::TimingOptions timing;
            timing.stream = stream;
            timing.warmups = opt.warmups;
            timing.reps = opt.reps;
            timing.flush_buffer = flush ? scratch.data() : nullptr;
            timing.flush_bytes = flush ? 2 * l2 : 0;
            gate.arm();
            const ckl::TimingStats stats = ckl::time_stream_ex(launch, timing);
            total_allocations += gate.disarm();
            std::printf("%-18s %10.4f %9.4f %11.1f\n", ckl::fft_transpose_name(v), stats.median_ms,
                        stats.iqr_ms, gbs_of(tbytes, stats.median_ms));
        }
    }

    // --- the 2D driver, and the share of its time the transposes take ---
    {
        const int side = std::min(1024, 1 << (opt.log2n / 2));
        const std::vector<float2> din = random_signal(static_cast<long long>(side) * side, 13);
        ckl::DeviceBuffer<float2> ta(din.size());
        ckl::DeviceBuffer<float2> tb(din.size());
        ta.copy_from_host(din);
        std::printf("\n2D row column at %d by %d\n", side, side);
        std::printf("%-18s %6s %14s %10s %11s %16s\n", "transpose", "passes", "model_bytes",
                    "median_ms", "eff_GB/s", "transpose_share");
        for (ckl::FftTranspose v : {ckl::FftTranspose::kNaive, ckl::FftTranspose::kTiledPadded,
                                    ckl::FftTranspose::kStridedShared}) {
            ckl::Fft2dPlanOptions opt2d;
            opt2d.transpose = v;
            opt2d.fast_twiddles = opt.fast_twiddles;
            ckl::Fft2dPlan plan2d(side, side, opt2d);
            auto launch = [&plan2d, &ta, &tb](cudaStream_t s) {
                const ckl::Status inner =
                    ckl::fft2d(plan2d, ckl::FftDirection::kForward, ta.data(), tb.data(), s);
                if (inner != ckl::Status::kSuccess) {
                    std::fprintf(stderr, "the timed 2D call failed: %s\n",
                                 ckl::status_string(inner));
                    std::exit(4);
                }
            };
            ckl::TimingOptions timing;
            timing.stream = stream;
            timing.warmups = opt.warmups;
            timing.reps = opt.reps;
            timing.flush_buffer = flush ? scratch.data() : nullptr;
            timing.flush_bytes = flush ? 2 * l2 : 0;
            gate.arm();
            const ckl::TimingStats total = ckl::time_stream_ex(launch, timing);
            total_allocations += gate.disarm();

            // The share is measured, not assumed: the same transpose is timed on
            // its own at the same shape and doubled, because the driver runs two.
            double share = 0.0;
            if (v != ckl::FftTranspose::kStridedShared) {
                auto tlaunch = [&ta, &tb, side, v](cudaStream_t s) {
                    ckl::fft_transpose(ta.data(), tb.data(), side, side, 1, v, nullptr, 1.0f, s);
                };
                const ckl::TimingStats one = ckl::time_stream_ex(tlaunch, timing);
                share = total.median_ms > 0.0 ? 2.0 * one.median_ms / total.median_ms : 0.0;
            }
            char share_text[24] = "0.0% (no transpose)";
            if (v != ckl::FftTranspose::kStridedShared) {
                std::snprintf(share_text, sizeof(share_text), "%.1f%%", 100.0 * share);
            }
            std::printf("%-18s %6d %14lld %10.4f %11.1f %16s\n", ckl::fft_transpose_name(v),
                        plan2d.passes(), plan2d.model_bytes(), total.median_ms,
                        gbs_of(plan2d.model_bytes(), total.median_ms), share_text);
        }
    }

    CKL_CUDA_CHECK(cudaStreamSynchronize(stream));
    CKL_CUDA_CHECK(cudaStreamDestroy(stream));

    if (total_allocations != 0) {
        std::fprintf(stderr,
                     "\nallocation gate FAILED: %lld cudaMalloc, cudaMallocAsync, cudaFree or "
                     "cudaFreeAsync call(s) happened inside a timed region. A measurement that "
                     "contains an allocation is measuring the allocator.\n",
                     total_allocations);
        return 5;
    }
    if (failures != 0) {
        std::fprintf(stderr, "\n%d variant(s) failed verification or dispatch\n", failures);
        return 6;
    }
    std::printf("\nallocation gate clean: zero allocations inside every timed region.\n");
    return 0;
}
