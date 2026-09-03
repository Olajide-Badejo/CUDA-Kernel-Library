// The convolution family driver, and the crossover sweep that is this family's
// headline figure.
//
// Two paths compute the same thing. The FFT path pads both operands to L, the
// next power of two at or above N + M - 1, transforms, multiplies pointwise and
// transforms back; its cost barely moves with M. The direct path computes the
// definition in the time domain; its cost is 2 N M and grows without bound. There
// is an M where they cross, and where it is on this part is a measurement, not an
// estimate. --sweep runs M in powers of two from 8 to 16384 at a fixed N and
// prints one row per M, which is what gen_report_assets.py turns into the chart
// with the crossing read out of the data.
//
// The protocol is the one bench_fft runs under and docs/benchmarking.md sets out:
// the plan and the cached filter spectrum are built before anything is timed, a
// CUPTI counter is armed across every timed lambda, every variant is verified
// against a double precision time domain reference before its loop opens, the
// cuFFT baseline is measured once and every percent is against that measurement,
// and every row carries its declared model bytes.
//
// Every number this prints is at whatever clocks the machine happened to be at.
// The rows of record come from bench_all under sweep.py with the clock locked.
//
// Usage: bench_conv [log2n] [flags]

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
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
    int filter = 256;
    int warmups = 5;
    int reps = 50;
    bool flush = true;
    bool sweep = false;
    double verify_perturb = 1.0;
    bool gate_red = false;
};

struct Row {
    std::string variant;
    std::string chosen;
    int passes = 0;
    long long model_bytes = 0;
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

const ckl::ConvAlgo kVariants[] = {
    ckl::ConvAlgo::kFftSeparate,    ckl::ConvAlgo::kFftFused, ckl::ConvAlgo::kDirectShared,
    ckl::ConvAlgo::kDirectConstant, ckl::ConvAlgo::kCufft,
};

// The definition, in double, on the host. O(N M), so the verification signal is
// short even when the timed one is not.
std::vector<double> cpu_convolution(const std::vector<float>& signal,
                                    const std::vector<float>& filter) {
    std::vector<double> out(signal.size() + filter.size() - 1, 0.0);
    for (std::size_t j = 0; j < signal.size(); ++j) {
        const double s = static_cast<double>(signal[j]);
        for (std::size_t t = 0; t < filter.size(); ++t) {
            out[j + t] += s * static_cast<double>(filter[t]);
        }
    }
    return out;
}

double relative_rms(const std::vector<float>& got, const std::vector<double>& want,
                    double perturb) {
    double num = 0.0;
    double den = 0.0;
    for (std::size_t i = 0; i < want.size(); ++i) {
        const double d = static_cast<double>(got[i]) * perturb - want[i];
        num += d * d;
        den += want[i] * want[i];
    }
    return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
}

double gbs_of(long long bytes, double ms) {
    return ms > 0.0 ? static_cast<double>(bytes) / (ms / 1000.0) / 1.0e9 : 0.0;
}

Options parse(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            std::printf(
                "usage: %s [log2n] [flags]\n"
                "  log2n: log2 of the signal length, 10 to 23 (default 20)\n"
                "flags:\n"
                "  --filter M       filter length for the single shot table (default 256)\n"
                "  --sweep          run M in powers of two from 8 to 16384 and print the\n"
                "                   crossover rows instead of the single shot table\n"
                "  --warmups N      untimed launches before the loop (default 5)\n"
                "  --reps N         timed reps (default 50)\n"
                "  --no-flush       leave L2 warm between reps\n"
                "  --verify-perturb X  TEST ONLY: scale the answer so verification fails\n"
                "  --gate-red       TEST ONLY: allocate inside the timed region\n",
                argv[0]);
            std::exit(0);
        } else if (arg == "--filter" && i + 1 < argc) {
            opt.filter = std::atoi(argv[++i]);
        } else if (arg == "--sweep") {
            opt.sweep = true;
        } else if (arg == "--warmups" && i + 1 < argc) {
            opt.warmups = std::atoi(argv[++i]);
        } else if (arg == "--reps" && i + 1 < argc) {
            opt.reps = std::atoi(argv[++i]);
        } else if (arg == "--no-flush") {
            opt.flush = false;
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
    if (opt.log2n < 10 || opt.log2n > 23) {
        std::fprintf(stderr, "log2n must be between 10 and 23, and is %d\n", opt.log2n);
        std::exit(2);
    }
    return opt;
}

// Everything one (N, M) pair needs. Built once and reused across every variant,
// which is the whole reason a plan exists.
struct Bench {
    int n = 0;
    int m = 0;
    ckl::DeviceBuffer<float> ds;
    ckl::DeviceBuffer<float> df;
    ckl::DeviceBuffer<float> dout;
    std::vector<double> want;
    std::vector<float> host_signal;
};

Bench build(int n, int m, int verify_len) {
    Bench b;
    b.n = n;
    b.m = m;
    b.host_signal = ckl::random_matrix(n, 1, 5);
    const std::vector<float> filter = ckl::random_matrix(m, 1, 6);
    b.ds = ckl::DeviceBuffer<float>(b.host_signal.size());
    b.df = ckl::DeviceBuffer<float>(filter.size());
    b.dout = ckl::DeviceBuffer<float>(static_cast<std::size_t>(n + m - 1));
    b.ds.copy_from_host(b.host_signal);
    b.df.copy_from_host(filter);
    // The reference is O(N M) on the host, so it is taken over a short prefix of
    // the signal rather than the whole thing. A wrong kernel is wrong on the
    // prefix too: the output there depends on the first verify_len + M - 1
    // samples and on every tap.
    const int short_n = std::min(n, verify_len);
    const std::vector<float> prefix(b.host_signal.begin(),
                                    b.host_signal.begin() + static_cast<std::ptrdiff_t>(short_n));
    b.want = cpu_convolution(prefix, filter);
    return b;
}

// Compares only the outputs the short reference actually determines: output i
// depends on signal[i - M + 1 .. i], so the first short_n - M + 1 entries are
// exact even though the device convolved the whole signal.
double verify(const Bench& b, int verify_len, double perturb) {
    const int short_n = std::min(b.n, verify_len);
    const int usable = short_n - b.m + 1;
    if (usable <= 0) {
        return 0.0;
    }
    const std::vector<float> got = b.dout.to_host();
    const std::vector<float> head(got.begin(), got.begin() + usable);
    const std::vector<double> ref(b.want.begin(), b.want.begin() + usable);
    return relative_rms(head, ref, perturb);
}

}  // namespace

// NOLINTNEXTLINE(bugprone-exception-escape): a ckl::Error here is fatal by design
int main(int argc, char** argv) {
    const Options opt = parse(argc, argv);
    const int n = 1 << opt.log2n;
    constexpr int kVerifyPrefix = 4096;

    ckl::bench::AllocationGate gate;
    const std::string gate_error = gate.start();
    if (!gate_error.empty()) {
        std::fprintf(stderr, "the allocation gate could not be armed: %s\n", gate_error.c_str());
        return 3;
    }

    int device = 0;
    CKL_CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop{};
    CKL_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    const std::size_t l2 =
        prop.l2CacheSize > 0 ? static_cast<std::size_t>(prop.l2CacheSize) : (std::size_t{48} << 20);

    cudaStream_t stream = nullptr;
    CKL_CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamDefault));
    ckl::DeviceBuffer<unsigned char> scratch;
    if (opt.flush) {
        scratch = ckl::DeviceBuffer<unsigned char>(2 * l2);
    }

    long long total_allocations = 0;
    int failures = 0;

    auto run_one = [&](ckl::ConvPlan& plan, ckl::ConvAlgo algo, Bench& b) {
        Row row;
        row.variant = ckl::conv_algo_name(algo);
        if (!plan.supports(algo)) {
            row.chosen = row.variant;
            row.note = std::string("refused: ") + plan.refusal(algo);
            return row;
        }
        row.passes = plan.passes(algo);
        row.model_bytes = plan.model_bytes(algo);

        b.dout.zero();
        ckl::ConvAlgo chosen = ckl::ConvAlgo::kAuto;
        const ckl::Status st = ckl::conv(plan, algo, b.ds.data(), b.dout.data(), &chosen, stream);
        row.chosen = ckl::conv_algo_name(chosen);
        CKL_CUDA_CHECK(cudaStreamSynchronize(stream));
        if (st != ckl::Status::kSuccess) {
            row.note = std::string("failed: ") + ckl::status_string(st);
            ++failures;
            return row;
        }
        if (chosen != algo) {
            std::fprintf(stderr, "%s ran %s instead; that is a defect\n", ckl::conv_algo_name(algo),
                         row.chosen.c_str());
            ++failures;
            return row;
        }
        row.residual = verify(b, kVerifyPrefix, opt.verify_perturb);
        const double bound = ckl::fft_tolerance(plan.transform_length());
        if (!(row.residual <= bound)) {
            row.note = "verification failed";
            std::fprintf(stderr, "%s: residual %.3e against a bound of %.3e; nothing timed\n",
                         ckl::conv_algo_name(algo), row.residual, bound);
            ++failures;
            return row;
        }

        auto launch = [&plan, &b, algo, &opt](cudaStream_t s) {
            void* leak = nullptr;
            if (opt.gate_red) {
                CKL_CUDA_CHECK(cudaMalloc(&leak, 1024));
            }
            const ckl::Status inner = ckl::conv(plan, algo, b.ds.data(), b.dout.data(), nullptr, s);
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
        timing.flush_buffer = opt.flush ? scratch.data() : nullptr;
        timing.flush_bytes = opt.flush ? 2 * l2 : 0;

        gate.arm();
        const ckl::TimingStats stats = ckl::time_stream_ex(launch, timing);
        row.allocations = gate.disarm();
        total_allocations += row.allocations;

        row.median_ms = stats.median_ms;
        row.iqr_ms = stats.iqr_ms;
        row.reps = stats.reps;
        row.effective_gbs = gbs_of(row.model_bytes, stats.median_ms);
        row.gflops = stats.median_ms > 0.0
                         ? plan.flop_model(algo) / (stats.median_ms / 1000.0) / 1.0e9
                         : 0.0;
        row.ok = true;
        return row;
    };

    if (opt.sweep) {
        // The crossover. One row per (M, variant), which is what the report's chart
        // is generated from; the crossing is read out of the data by the script and
        // never placed by hand.
        std::printf("crossover sweep at N = %d (2^%d)\n", n, opt.log2n);
        std::printf("%8s %8s %-18s %-18s %14s %10s %9s %11s %10s\n", "M", "L", "variant", "chosen",
                    "model_bytes", "median_ms", "iqr_ms", "eff_GB/s", "GFLOP/s");
        for (int m = 8; m <= 16384; m <<= 1) {
            Bench b = build(n, m, kVerifyPrefix);
            ckl::ConvPlan plan(n, b.df.data(), m);
            for (ckl::ConvAlgo algo : kVariants) {
                const Row row = run_one(plan, algo, b);
                if (!row.ok) {
                    std::printf("%8d %8d %-18s %-18s %14s %10s %9s %11s %10s  %s\n", m,
                                plan.transform_length(), row.variant.c_str(), row.chosen.c_str(),
                                "-", "-", "-", "-", "-", row.note.c_str());
                    continue;
                }
                std::printf("%8d %8d %-18s %-18s %14lld %10.4f %9.4f %11.1f %10.1f\n", m,
                            plan.transform_length(), row.variant.c_str(), row.chosen.c_str(),
                            row.model_bytes, row.median_ms, row.iqr_ms, row.effective_gbs,
                            row.gflops);
            }
        }
    } else {
        Bench b = build(n, opt.filter, kVerifyPrefix);
        ckl::ConvPlan plan(n, b.df.data(), opt.filter);
        std::printf("convolution N = %d (2^%d), M = %d, output %d, padded L = %d\n", n, opt.log2n,
                    opt.filter, plan.output_length(), plan.transform_length());
        std::printf("plan: transforms run on %s; kAuto would run %s\n",
                    ckl::fft_algo_name(plan.fft_algo()), ckl::conv_algo_name(plan.query()));
        std::printf("verification prefix %d samples against a double time domain reference\n",
                    kVerifyPrefix);
        std::printf("L2 flush between reps: %s\n", opt.flush ? "on" : "off");
        std::printf("dram__bytes.sum against the declared model: pending an ncu round\n\n");

        std::vector<Row> rows;
        for (ckl::ConvAlgo algo : kVariants) {
            rows.push_back(run_one(plan, algo, b));
        }
        double baseline_ms = 0.0;
        for (const Row& r : rows) {
            if (r.variant == std::string(ckl::conv_algo_name(ckl::ConvAlgo::kCufft)) && r.ok) {
                baseline_ms = r.median_ms;
            }
        }
        std::printf("%-18s %-18s %6s %14s %10s %9s %6s %11s %10s %10s %10s\n", "variant", "chosen",
                    "passes", "model_bytes", "median_ms", "iqr_ms", "reps", "eff_GB/s", "GFLOP/s",
                    "pct_cufft", "allocs");
        for (const Row& r : rows) {
            if (!r.ok) {
                std::printf("%-18s %-18s %6s %14s %10s %9s %6s %11s %10s %10s %10s  %s\n",
                            r.variant.c_str(), r.chosen.c_str(), "-", "-", "-", "-", "-", "-", "-",
                            "-", "-", r.note.c_str());
                continue;
            }
            char pct[16] = "-";
            if (baseline_ms > 0.0) {
                std::snprintf(pct, sizeof(pct), "%.1f%%", 100.0 * baseline_ms / r.median_ms);
            }
            std::printf("%-18s %-18s %6d %14lld %10.4f %9.4f %6d %11.1f %10.1f %10s %10lld\n",
                        r.variant.c_str(), r.chosen.c_str(), r.passes, r.model_bytes, r.median_ms,
                        r.iqr_ms, r.reps, r.effective_gbs, r.gflops, pct, r.allocations);
        }
        std::printf(
            "\npct_cufft is against %s, which is cuFFT plus a separate pointwise kernel. "
            "docs/fft.md\nrecords whether the cufftXtSetCallback probe linked and ran on this "
            "toolkit; where it did,\nthe vendor fused row is bench_conv_callback, built outside "
            "this binary against libcufft_static.\n",
            ckl::conv_algo_name(ckl::ConvAlgo::kCufft));
        std::printf("residuals:");
        for (const Row& r : rows) {
            if (r.ok) {
                std::printf(" %s=%.2e", r.variant.c_str(), r.residual);
            }
        }
        std::printf("\n");
    }

    CKL_CUDA_CHECK(cudaStreamSynchronize(stream));
    CKL_CUDA_CHECK(cudaStreamDestroy(stream));

    if (total_allocations != 0) {
        std::fprintf(stderr,
                     "\nallocation gate FAILED: %lld allocation call(s) happened inside a timed "
                     "region.\n",
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
