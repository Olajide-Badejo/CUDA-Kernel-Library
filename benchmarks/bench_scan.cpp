// The reduction and scan family driver: one length, every rung of both ladders,
// the CUB baseline, and the allocation counter around the timed region.
//
// It is the SpMV driver's shape because the claim is the same one. A
// ckl::ScanPlan is built before anything is timed, every rung is verified
// against a double precision CPU reference before it is timed, and a CUPTI
// counter is armed over the timed lambda so an allocation inside a measurement
// fails the run instead of inflating it. --gate-red puts a cudaMalloc and a
// cudaFree back inside the timed region so the counter can be seen failing,
// because a gate that has never failed is not known to be a gate.
//
// Three things this family needs that the others do not.
//
// The flush pair. Input plus output below about 2^22 elements is L2 resident on
// this 48 MB part, so every mid size row is measured twice, flushed and
// unflushed, and both are printed. Only the flushed number can carry a bandwidth
// claim; the unflushed one says how much of the difference was the cache.
//
// The graph pair. Below about 2^16 elements the kernel is shorter than the
// launch, so those lengths are measured stream launched and graph launched with
// the same number of inner launches, and the difference is launch overhead with
// nothing else in it. No bandwidth claim is made from either.
//
// CUB_VERSION on every row. CUB implements decoupled look-back itself, so parity
// is the goal and a percentage against CUB is a percentage against one CUB.
//
// Every number this prints is at whatever clocks the machine happened to be at.
// Nothing here belongs in a document; the row of record comes from bench_all
// under sweep.py with the clock locked.
//
// Usage: bench_scan [n] [flags]

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
#include "ckl/scan.hpp"
#include "event_timer.hpp"
#include "reference.hpp"

namespace {

struct Options {
    long long n = 1 << 24;
    std::uint64_t seed = 1;
    int warmups = 5;
    int reps = 50;
    bool exclusive = false;
    // Test only. Puts a cudaMalloc and a cudaFree back inside the timed lambda
    // so the allocation gate can be shown failing.
    bool gate_red = false;
};

struct Row {
    std::string family;
    std::string rung;
    std::string chosen;
    double median_ms = 0.0;
    double iqr_ms = 0.0;
    int reps = 0;
    long long model_bytes = 0;
    double effective_gbs = 0.0;
    double residual = 0.0;
    double tolerance = 0.0;
    long long allocations = 0;
    bool flushed = false;
    std::string launch_mode = "stream";
    bool ok = false;
    std::string note;
};

const ckl::ReduceAlgo kReduceRungs[] = {
    ckl::ReduceAlgo::kAtomic,        ckl::ReduceAlgo::kSharedTree, ckl::ReduceAlgo::kShuffle,
    ckl::ReduceAlgo::kVec4,          ckl::ReduceAlgo::kSinglePass, ckl::ReduceAlgo::kTwoPass,
    ckl::ReduceAlgo::kDeterministic, ckl::ReduceAlgo::kKahan,      ckl::ReduceAlgo::kCub,
};

const ckl::ScanAlgo kScanRungs[] = {
    ckl::ScanAlgo::kHillisSteele, ckl::ScanAlgo::kBlelloch,
    ckl::ScanAlgo::kThreeKernel,  ckl::ScanAlgo::kReduceThenScan,
    ckl::ScanAlgo::kLookback,     ckl::ScanAlgo::kDeterministic,
    ckl::ScanAlgo::kCub,
};

// The same two error models the correctness suite is written against, and for
// the same reason: two rungs finish through one global atomic, so their last
// combine is a serial accumulation and its error grows with the chain length
// rather than with its square root.
long long serial_chain(ckl::ReduceAlgo algo, long long n) {
    switch (algo) {
        case ckl::ReduceAlgo::kAtomic:
            return n;
        case ckl::ReduceAlgo::kShuffle:
            return (n + 255) / 256;
        default:
            return 0;
    }
}

constexpr double kSerialToleranceC = 16.0;

double reduce_tolerance(ckl::ReduceAlgo algo, long long n) {
    const long long chain = serial_chain(algo, n);
    if (chain > 0) {
        return kSerialToleranceC * static_cast<double>(chain) * 1.1920929e-07;
    }
    return ckl::scan_tolerance(n);
}

Options parse(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            std::printf(
                "usage: %s [n] [flags]\n"
                "  n: elements to reduce and scan (default 16777216)\n"
                "flags:\n"
                "  --seed N     seed for the input (default 1)\n"
                "  --warmups N  untimed launches before the loop (default 5)\n"
                "  --reps N     timed reps (default 50)\n"
                "  --exclusive  measure the exclusive scan instead of the inclusive one\n"
                "  --gate-red   TEST ONLY: allocate inside the timed region so the\n"
                "               allocation gate fails\n",
                argv[0]);
            std::exit(0);
        } else if (arg == "--seed" && i + 1 < argc) {
            opt.seed = std::strtoull(argv[++i], nullptr, 10);
        } else if (arg == "--warmups" && i + 1 < argc) {
            opt.warmups = std::atoi(argv[++i]);
        } else if (arg == "--reps" && i + 1 < argc) {
            opt.reps = std::atoi(argv[++i]);
        } else if (arg == "--exclusive") {
            opt.exclusive = true;
        } else if (arg == "--gate-red") {
            opt.gate_red = true;
        } else if (arg.rfind("--", 0) == 0) {
            std::fprintf(stderr, "unknown flag %s\n", arg.c_str());
            std::exit(2);
        } else {
            opt.n = std::strtoll(arg.c_str(), nullptr, 10);
        }
    }
    return opt;
}

std::vector<double> reference_scan(const std::vector<float>& in, bool exclusive) {
    std::vector<double> out(in.size());
    double running = 0.0;
    for (std::size_t i = 0; i < in.size(); ++i) {
        if (exclusive) {
            out[i] = running;
            running += static_cast<double>(in[i]);
        } else {
            running += static_cast<double>(in[i]);
            out[i] = running;
        }
    }
    return out;
}

double largest_magnitude(const std::vector<float>& in) {
    double worst = 0.0;
    for (float x : in) {
        worst = std::max(worst, std::fabs(static_cast<double>(x)));
    }
    return worst > 0.0 ? worst : 1.0;
}

}  // namespace

int main(int argc, char** argv) {
    const Options opt = parse(argc, argv);
    if (opt.n <= 0) {
        std::fprintf(stderr, "n must be positive\n");
        return 2;
    }

    ckl::bench::AllocationGate gate;
    const std::string gate_error = gate.start();
    if (!gate_error.empty()) {
        std::fprintf(stderr,
                     "the allocation gate could not be armed: %s\n"
                     "This driver needs it, so this is a failure rather than a warning.\n",
                     gate_error.c_str());
        return 3;
    }

    // Everything below is setup, and none of it is timed.
    const std::vector<float> host = ckl::random_matrix(static_cast<int>(opt.n), 1, opt.seed);
    ckl::DeviceBuffer<float> in(host.size());
    in.copy_from_host(host);
    ckl::DeviceBuffer<float> out(host.size());
    ckl::DeviceBuffer<float> single(1);
    out.zero();
    single.zero();

    ckl::ScanPlan plan(opt.n);

    const double scale = largest_magnitude(host);
    double reference_total = 0.0;
    for (float x : host) {
        reference_total += static_cast<double>(x);
    }
    const std::vector<double> reference_prefix = reference_scan(host, opt.exclusive);

    cudaStream_t stream = nullptr;
    CKL_CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamDefault));

    int device = 0;
    CKL_CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop{};
    CKL_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    const std::size_t l2 =
        prop.l2CacheSize > 0 ? static_cast<std::size_t>(prop.l2CacheSize) : (std::size_t{48} << 20);
    const std::size_t working_set = 2 * host.size() * sizeof(float);
    const bool l2_resident = working_set <= l2;
    // Twice L2, so the flush write cannot leave any of the previous working set
    // resident whatever the replacement policy does. Allocated once; allocating
    // it per rep is exactly what the gate is watching for.
    ckl::DeviceBuffer<unsigned char> scratch(2 * l2);

    // Below this the kernel is shorter than the launch, so those lengths get a
    // graph launched row beside the stream launched one and no bandwidth claim.
    constexpr long long kGraphMaxN = 1 << 16;
    constexpr int kGraphInner = 20;
    const bool graph_pair = opt.n <= kGraphMaxN;

    std::printf("n=%lld fp32 kSum %s, working set %.2f MiB, L2 %.0f MiB, %s\n", opt.n,
                opt.exclusive ? "exclusive" : "inclusive",
                static_cast<double>(working_set) / (1024.0 * 1024.0),
                static_cast<double>(l2) / (1024.0 * 1024.0),
                l2_resident ? "L2 resident: flushed and unflushed rows below"
                            : "larger than L2: flushed rows only");
    std::printf("CUB_VERSION %d (CUB %d.%d.%d), tolerance c %.3f, serial c %.3f\n",
                ckl::ScanPlan::cub_version(), ckl::ScanPlan::cub_version() / 100000,
                ckl::ScanPlan::cub_version() / 100 % 1000, ckl::ScanPlan::cub_version() % 100,
                ckl::scan_tolerance_c(), kSerialToleranceC);
    std::printf("plan: persistent blocks %d, deterministic blocks %d, look-back tiles %lld\n",
                plan.persistent_blocks(), plan.deterministic_blocks(),
                plan.tiles(ckl::ScanAlgo::kLookback));
    if (graph_pair) {
        std::printf("launch isolation: %d inner launches per event pair, stream and graph\n",
                    kGraphInner);
    }

    std::vector<Row> rows;
    long long total_allocations = 0;
    int failures = 0;

    // One timed window, under the protocol.
    auto measure = [&](Row& row, const std::function<void(cudaStream_t)>& launch, bool flushed,
                       bool graph, int inner) {
        ckl::TimingOptions timing;
        timing.stream = stream;
        timing.warmups = opt.warmups;
        timing.reps = opt.reps;
        timing.graph = graph;
        timing.inner = inner;
        timing.flush_buffer = flushed ? scratch.data() : nullptr;
        timing.flush_bytes = flushed ? 2 * l2 : 0;
        gate.arm();
        const ckl::TimingStats stats = ckl::time_stream_ex(launch, timing);
        row.allocations = gate.disarm();
        total_allocations += row.allocations;
        row.median_ms = stats.median_ms;
        row.iqr_ms = stats.iqr_ms;
        row.reps = stats.reps;
        row.flushed = flushed;
        row.launch_mode = graph ? "graph" : "stream";
        row.effective_gbs = stats.median_ms > 0.0 ? static_cast<double>(row.model_bytes) /
                                                        (stats.median_ms / 1000.0) / 1.0e9
                                                  : 0.0;
        row.ok = true;
    };

    for (ckl::ReduceAlgo algo : kReduceRungs) {
        Row row;
        row.family = "reduce";
        row.rung = ckl::reduce_algo_name(algo);
        row.model_bytes = plan.reduce_model_bytes(algo);
        row.tolerance = reduce_tolerance(algo, opt.n);

        // Verification first, and outside the gate: a first call is allowed to
        // touch whatever it needs to.
        ckl::ReduceAlgo chosen = ckl::ReduceAlgo::kAuto;
        single.zero();
        const ckl::Status st =
            ckl::reduce(plan, algo, ckl::ScanOp::kSum, in.data(), single.data(), &chosen, stream);
        CKL_CUDA_CHECK(cudaStreamSynchronize(stream));
        row.chosen = ckl::reduce_algo_name(chosen);
        if (st != ckl::Status::kSuccess) {
            row.note = std::string("refused: ") + ckl::status_string(st);
            rows.push_back(row);
            continue;
        }
        if (chosen != algo) {
            std::fprintf(stderr,
                         "%s ran %s instead; a benchmark that measures a path its row does not "
                         "name is a defect\n",
                         ckl::reduce_algo_name(algo), row.chosen.c_str());
            ++failures;
            rows.push_back(row);
            continue;
        }
        row.residual =
            std::fabs(static_cast<double>(single.to_host()[0]) - reference_total) / scale;
        if (!(row.residual <= row.tolerance)) {
            row.note = "verification failed";
            std::fprintf(stderr, "%s: residual %.3e against a tolerance of %.3e; nothing timed\n",
                         ckl::reduce_algo_name(algo), row.residual, row.tolerance);
            ++failures;
            rows.push_back(row);
            continue;
        }

        auto launch = [&](cudaStream_t s) {
            void* leak = nullptr;
            if (opt.gate_red) {
                CKL_CUDA_CHECK(cudaMalloc(&leak, 1024));
            }
            const ckl::Status inner =
                ckl::reduce(plan, algo, ckl::ScanOp::kSum, in.data(), single.data(), nullptr, s);
            if (inner != ckl::Status::kSuccess) {
                std::fprintf(stderr, "the timed call failed: %s\n", ckl::status_string(inner));
                std::exit(4);
            }
            if (opt.gate_red) {
                CKL_CUDA_CHECK(cudaFree(leak));
            }
        };

        Row flushed = row;
        measure(flushed, launch, true, false, 1);
        rows.push_back(flushed);
        if (l2_resident) {
            Row warm = row;
            measure(warm, launch, false, false, 1);
            rows.push_back(warm);
        }
        if (graph_pair) {
            Row stream_row = row;
            measure(stream_row, launch, true, false, kGraphInner);
            stream_row.note = "launch isolation pair";
            rows.push_back(stream_row);
            Row graph_row = row;
            measure(graph_row, launch, true, true, kGraphInner);
            graph_row.note = "launch isolation pair";
            rows.push_back(graph_row);
        }
    }

    for (ckl::ScanAlgo algo : kScanRungs) {
        Row row;
        row.family = "scan";
        row.rung = ckl::scan_algo_name(algo);
        row.model_bytes = plan.scan_model_bytes(algo);
        row.tolerance = ckl::scan_tolerance(opt.n);

        ckl::ScanAlgo chosen = ckl::ScanAlgo::kAuto;
        out.zero();
        const ckl::Status st = ckl::scan(plan, algo, ckl::ScanOp::kSum, opt.exclusive, in.data(),
                                         out.data(), &chosen, stream);
        CKL_CUDA_CHECK(cudaStreamSynchronize(stream));
        row.chosen = ckl::scan_algo_name(chosen);
        if (st != ckl::Status::kSuccess) {
            row.note = std::string("refused: ") + ckl::status_string(st);
            rows.push_back(row);
            continue;
        }
        if (chosen != algo) {
            std::fprintf(stderr,
                         "%s ran %s instead; a benchmark that measures a path its row does not "
                         "name is a defect\n",
                         ckl::scan_algo_name(algo), row.chosen.c_str());
            ++failures;
            rows.push_back(row);
            continue;
        }
        {
            const std::vector<float> got = out.to_host();
            double worst = 0.0;
            for (std::size_t i = 0; i < got.size(); ++i) {
                worst = std::max(
                    worst, std::fabs(static_cast<double>(got[i]) - reference_prefix[i]) / scale);
            }
            row.residual = worst;
        }
        if (!(row.residual <= row.tolerance)) {
            row.note = "verification failed";
            std::fprintf(stderr, "%s: residual %.3e against a tolerance of %.3e; nothing timed\n",
                         ckl::scan_algo_name(algo), row.residual, row.tolerance);
            ++failures;
            rows.push_back(row);
            continue;
        }

        auto launch = [&](cudaStream_t s) {
            void* leak = nullptr;
            if (opt.gate_red) {
                CKL_CUDA_CHECK(cudaMalloc(&leak, 1024));
            }
            const ckl::Status inner = ckl::scan(plan, algo, ckl::ScanOp::kSum, opt.exclusive,
                                                in.data(), out.data(), nullptr, s);
            if (inner != ckl::Status::kSuccess) {
                std::fprintf(stderr, "the timed call failed: %s\n", ckl::status_string(inner));
                std::exit(4);
            }
            if (opt.gate_red) {
                CKL_CUDA_CHECK(cudaFree(leak));
            }
        };

        Row flushed = row;
        measure(flushed, launch, true, false, 1);
        rows.push_back(flushed);
        if (l2_resident) {
            Row warm = row;
            measure(warm, launch, false, false, 1);
            rows.push_back(warm);
        }
        if (graph_pair) {
            Row stream_row = row;
            measure(stream_row, launch, true, false, kGraphInner);
            stream_row.note = "launch isolation pair";
            rows.push_back(stream_row);
            Row graph_row = row;
            measure(graph_row, launch, true, true, kGraphInner);
            graph_row.note = "launch isolation pair";
            rows.push_back(graph_row);
        }
    }

    CKL_CUDA_CHECK(cudaStreamSynchronize(stream));
    CKL_CUDA_CHECK(cudaStreamDestroy(stream));

    // The baseline of record for each ladder, measured once and quoted from
    // there. Every percentage below names which one it is against.
    double reduce_base = 0.0;
    double scan_base = 0.0;
    for (const Row& r : rows) {
        if (!r.ok || !r.flushed || r.launch_mode != "stream" || r.reps == 0) {
            continue;
        }
        if (r.family == "reduce" && r.rung == std::string("kCub") && reduce_base == 0.0) {
            reduce_base = r.median_ms;
        }
        if (r.family == "scan" && r.rung == std::string("kCub") && scan_base == 0.0) {
            scan_base = r.median_ms;
        }
    }

    std::printf("\n%-7s %-16s %-16s %7s %6s %11s %9s %9s %8s %10s %6s %s\n", "family", "rung",
                "chosen", "flush", "launch", "median_ms", "iqr_ms", "model_GB", "eff_GB/s",
                "pct_cub", "allocs", "residual");
    for (const Row& r : rows) {
        if (!r.ok) {
            std::printf("%-7s %-16s %-16s %7s %6s %11s %9s %9s %8s %10s %6s %s\n", r.family.c_str(),
                        r.rung.c_str(), r.chosen.c_str(), "-", "-", "-", "-", "-", "-", "-", "-",
                        r.note.c_str());
            continue;
        }
        const double base = r.family == "reduce" ? reduce_base : scan_base;
        char pct[16] = "-";
        if (base > 0.0 && r.median_ms > 0.0 && r.launch_mode == "stream" && r.flushed) {
            std::snprintf(pct, sizeof(pct), "%.1f%%", 100.0 * base / r.median_ms);
        }
        std::printf("%-7s %-16s %-16s %7s %6s %11.5f %9.5f %9.4f %8.1f %10s %6lld %.3e\n",
                    r.family.c_str(), r.rung.c_str(), r.chosen.c_str(), r.flushed ? "on" : "off",
                    r.launch_mode.c_str(), r.median_ms, r.iqr_ms,
                    static_cast<double>(r.model_bytes) / 1.0e9, r.effective_gbs, pct, r.allocations,
                    r.residual);
    }
    std::printf(
        "\npct_cub is the flushed stream row against kCub of the same ladder; every percent "
        "names its baseline.\n");
    std::printf(
        "eff_GB/s uses the rung's own declared model bytes, not a common denominator, so a rung "
        "that moves more data cannot look faster by moving it.\n");
    if (graph_pair) {
        std::printf(
            "the two launch isolation rows per rung differ only in the launch path, so their "
            "difference is launch overhead and nothing else.\n");
    }

    if (total_allocations != 0) {
        std::fprintf(stderr,
                     "\nallocation gate FAILED: %lld cudaMalloc, cudaMallocAsync, cudaFree or "
                     "cudaFreeAsync call(s) happened inside a timed region. A measurement that "
                     "contains an allocation is not a measurement of the kernel.\n",
                     total_allocations);
        return 5;
    }
    if (failures != 0) {
        std::fprintf(stderr, "\n%d rung(s) failed verification or dispatch\n", failures);
        return 6;
    }
    std::printf("\nallocation gate clean: zero allocations inside every timed region.\n");
    return 0;
}
