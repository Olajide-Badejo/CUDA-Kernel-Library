// The SpMV family driver: one matrix from the suite, every variant, both
// cuSPARSE algorithms, and Gate S's allocation counter around the timed region.
//
// v1 built one skewed matrix from a hard coded mt19937_64(2024), ran three
// variants and printed a percent of cuSPARSE whose denominator included three
// descriptor creations, a buffer sizing, a cudaMalloc and a cudaFree. This
// driver names a matrix from experiments/matrix_manifest.csv, builds one
// ckl::SpmvPlan for it before anything is timed, verifies every variant against
// a double precision CPU reference before timing it, and arms a CUPTI counter
// over the timed lambda so an allocation inside a measurement fails the run
// instead of inflating it.
//
// Every number this prints is at whatever clocks the machine happened to be at.
// Nothing here belongs in a document; the row of record comes from bench_all
// under sweep.py with the clock locked.
//
// Usage: bench_spmv [matrix] [flags]
//   matrix: a name from the manifest, or synthetic[:rows]

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
#include "ckl/sparse.hpp"
#include "event_timer.hpp"
#include "matrix_cache.hpp"
#include "reference.hpp"

namespace {

struct Options {
    std::string matrix = "synthetic:65536";
    std::uint64_t seed = 1;
    int warmups = 5;
    int reps = 50;
    bool flush = true;
    // Test only. Puts a cudaMalloc and a cudaFree back inside the timed lambda,
    // exactly where v1 had them, so the allocation gate can be shown failing.
    bool gate_red = false;
};

struct Row {
    ckl::SpmvAlgo algo = ckl::SpmvAlgo::kAuto;
    std::string chosen;
    double median_ms = 0.0;
    double gflops = 0.0;
    double residual = 0.0;
    long long allocations = 0;
    bool ok = false;
    std::string note;
};

// The double precision answer, and the magnitude the rounding error is bounded
// by, per row. Nothing is timed before it has matched this.
struct HostRef {
    std::vector<double> y;
    std::vector<double> scale;
};

HostRef reference_of(const ckl::bench::HostCsr& a, const std::vector<float>& x, float alpha) {
    HostRef ref;
    ref.y.assign(static_cast<std::size_t>(a.m), 0.0);
    ref.scale.assign(static_cast<std::size_t>(a.m), 0.0);
    for (int i = 0; i < a.m; ++i) {
        double acc = 0.0;
        double mag = 0.0;
        for (int k = a.row_ptr[static_cast<std::size_t>(i)];
             k < a.row_ptr[static_cast<std::size_t>(i) + 1]; ++k) {
            const double v = static_cast<double>(a.values[static_cast<std::size_t>(k)]);
            const double xv = static_cast<double>(
                x[static_cast<std::size_t>(a.col_idx[static_cast<std::size_t>(k)])]);
            acc += v * xv;
            mag += std::fabs(v) * std::fabs(xv);
        }
        ref.y[static_cast<std::size_t>(i)] = static_cast<double>(alpha) * acc;
        ref.scale[static_cast<std::size_t>(i)] = std::fabs(static_cast<double>(alpha)) * mag;
    }
    return ref;
}

double worst_residual(const ckl::bench::HostCsr& a, const std::vector<float>& got,
                      const HostRef& ref) {
    double worst = 0.0;
    for (int i = 0; i < a.m; ++i) {
        const double denom = ref.scale[static_cast<std::size_t>(i)] > 1e-30
                                 ? ref.scale[static_cast<std::size_t>(i)]
                                 : 1.0;
        worst = std::max(worst, std::fabs(static_cast<double>(got[static_cast<std::size_t>(i)]) -
                                          ref.y[static_cast<std::size_t>(i)]) /
                                    denom);
    }
    return worst;
}

const ckl::SpmvAlgo kVariants[] = {
    ckl::SpmvAlgo::kCsrNaive,        ckl::SpmvAlgo::kCsrWarp,      ckl::SpmvAlgo::kCsrVector,
    ckl::SpmvAlgo::kMerge,           ckl::SpmvAlgo::kSellCSigma,   ckl::SpmvAlgo::kBsr,
    ckl::SpmvAlgo::kCusparseDefault, ckl::SpmvAlgo::kCusparseAlg2,
};

Options parse(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            std::printf(
                "usage: %s [matrix] [flags]\n"
                "  matrix: a name from experiments/matrix_manifest.csv, or synthetic[:rows]\n"
                "flags:\n"
                "  --seed N     seed for the synthetic generator (default 1)\n"
                "  --warmups N  untimed launches before the loop (default 5)\n"
                "  --reps N     timed reps (default 50)\n"
                "  --no-flush   leave L2 warm between reps\n"
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
        } else if (arg == "--no-flush") {
            opt.flush = false;
        } else if (arg == "--gate-red") {
            opt.gate_red = true;
        } else if (arg.rfind("--", 0) == 0) {
            std::fprintf(stderr, "unknown flag %s\n", arg.c_str());
            std::exit(2);
        } else {
            opt.matrix = arg;
        }
    }
    return opt;
}

}  // namespace

int main(int argc, char** argv) {
    const Options opt = parse(argc, argv);

    ckl::bench::HostCsr a;
    std::string error;
    if (!ckl::bench::load_matrix(opt.matrix, opt.seed, &a, &error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 2;
    }

    ckl::bench::AllocationGate gate;
    const std::string gate_error = gate.start();
    if (!gate_error.empty()) {
        std::fprintf(stderr,
                     "the allocation gate could not be armed: %s\n"
                     "Gate S needs it, so this is a failure rather than a warning.\n",
                     gate_error.c_str());
        return 3;
    }

    // Everything below the plan is setup and none of it is timed.
    ckl::DeviceBuffer<int> drp(a.row_ptr.size());
    ckl::DeviceBuffer<int> dci(a.col_idx.size());
    ckl::DeviceBuffer<float> dv(a.values.size());
    drp.copy_from_host(a.row_ptr);
    if (a.nnz > 0) {
        dci.copy_from_host(a.col_idx);
        dv.copy_from_host(a.values);
    }

    ckl::SpmvCsr view;
    view.row_ptr = drp.data();
    view.col_idx = dci.data();
    view.values = dv.data();
    view.m = a.m;
    view.n = a.n;
    view.nnz = a.nnz;
    ckl::SpmvPlan plan(view);

    const auto x = ckl::random_matrix(a.n, 1, 7);
    ckl::DeviceBuffer<float> dx(x.size());
    ckl::DeviceBuffer<float> dy(static_cast<std::size_t>(a.m));
    dx.copy_from_host(x);
    dy.zero();

    const float alpha = 1.0f;
    const float beta = 0.0f;
    const HostRef ref = reference_of(a, x, alpha);
    int longest = 1;
    for (std::size_t i = 1; i < a.row_ptr.size(); ++i) {
        longest = std::max(longest, a.row_ptr[i] - a.row_ptr[i - 1]);
    }
    const double tolerance = ckl::tol(longest);

    cudaStream_t stream = nullptr;
    CKL_CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamDefault));

    // The L2 flush buffer, allocated once. Allocating it per rep would be the
    // very thing the gate is watching for.
    int device = 0;
    CKL_CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop{};
    CKL_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    const std::size_t l2 =
        prop.l2CacheSize > 0 ? static_cast<std::size_t>(prop.l2CacheSize) : (std::size_t{48} << 20);
    ckl::DeviceBuffer<unsigned char> scratch;
    if (opt.flush) {
        scratch = ckl::DeviceBuffer<unsigned char>(2 * l2);
    }

    std::printf("matrix %s: m=%d n=%d nnz=%d, mean %.2f nnz per row, longest %d, %s\n",
                a.name.c_str(), a.m, a.n, a.nnz, plan.mean_nnz_per_row(), plan.max_nnz_per_row(),
                plan.power_law() ? "power law" : "regular");
    std::printf(
        "plan: vector width %d, sigma %d, SELL padding %.4f, BSR block %d (detect %.2f ms)\n",
        plan.vector_width(), plan.sigma(), plan.sell_padding_ratio(), plan.bsr_block_dim(),
        plan.bsr_detect_ms());
    std::printf(
        "traffic model: %lld to %lld bytes per SpMV; measured dram__bytes.sum pending an "
        "ncu round\n",
        plan.model_bytes_low(beta != 0.0f), plan.model_bytes_high(beta != 0.0f));
    std::printf("L2 flush between reps: %s\n", opt.flush ? "on" : "off");

    std::vector<Row> rows;
    long long total_allocations = 0;
    int failures = 0;

    for (ckl::SpmvAlgo algo : kVariants) {
        Row row;
        row.algo = algo;

        // Verification first, and outside the gate: a first call is allowed to
        // touch whatever it needs to.
        ckl::SpmvAlgo chosen = ckl::SpmvAlgo::kAuto;
        dy.zero();
        ckl::Status st = ckl::spmv(plan, algo, alpha, dx.data(), beta, dy.data(), &chosen, stream);
        CKL_CUDA_CHECK(cudaStreamSynchronize(stream));
        row.chosen = ckl::spmv_algo_name(chosen);
        if (st != ckl::Status::kSuccess) {
            row.note = std::string("refused: ") + ckl::status_string(st);
            rows.push_back(row);
            continue;
        }
        if (chosen != algo) {
            std::fprintf(stderr,
                         "%s ran %s instead; a benchmark that measures a path its row does not "
                         "name is a defect\n",
                         ckl::spmv_algo_name(algo), row.chosen.c_str());
            ++failures;
            rows.push_back(row);
            continue;
        }
        row.residual = worst_residual(a, dy.to_host(), ref);
        if (!(row.residual <= tolerance)) {
            row.note = "verification failed";
            std::fprintf(stderr, "%s: residual %.3e against a tolerance of %.3e; nothing timed\n",
                         ckl::spmv_algo_name(algo), row.residual, tolerance);
            ++failures;
            rows.push_back(row);
            continue;
        }

        auto launch = [&](cudaStream_t s) {
            void* leak = nullptr;
            if (opt.gate_red) {
                // Exactly what the v1 cuSPARSE wrapper did on every call.
                CKL_CUDA_CHECK(cudaMalloc(&leak, 1024));
            }
            const ckl::Status inner =
                ckl::spmv(plan, algo, alpha, dx.data(), beta, dy.data(), nullptr, s);
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
        row.gflops = stats.median_ms > 0.0
                         ? 2.0 * static_cast<double>(a.nnz) / (stats.median_ms / 1000.0) / 1.0e9
                         : 0.0;
        row.ok = true;
        rows.push_back(row);
    }

    CKL_CUDA_CHECK(cudaStreamSynchronize(stream));
    CKL_CUDA_CHECK(cudaStreamDestroy(stream));

    double base_default = 0.0;
    double base_alg2 = 0.0;
    for (const Row& r : rows) {
        if (r.algo == ckl::SpmvAlgo::kCusparseDefault && r.ok) {
            base_default = r.gflops;
        }
        if (r.algo == ckl::SpmvAlgo::kCusparseAlg2 && r.ok) {
            base_alg2 = r.gflops;
        }
    }

    std::printf("\n%-18s %-18s %10s %10s %8s %8s %10s %s\n", "variant", "chosen", "median_ms",
                "gflops", "pct_def", "pct_alg2", "allocs", "residual");
    for (const Row& r : rows) {
        if (!r.ok) {
            std::printf("%-18s %-18s %10s %10s %8s %8s %10s %s\n", ckl::spmv_algo_name(r.algo),
                        r.chosen.c_str(), "-", "-", "-", "-", "-", r.note.c_str());
            continue;
        }
        char pct_def[16] = "-";
        char pct_alg2[16] = "-";
        if (base_default > 0.0) {
            std::snprintf(pct_def, sizeof(pct_def), "%.1f%%", 100.0 * r.gflops / base_default);
        }
        if (base_alg2 > 0.0) {
            std::snprintf(pct_alg2, sizeof(pct_alg2), "%.1f%%", 100.0 * r.gflops / base_alg2);
        }
        std::printf("%-18s %-18s %10.4f %10.1f %8s %8s %10lld %.3e\n", ckl::spmv_algo_name(r.algo),
                    r.chosen.c_str(), r.median_ms, r.gflops, pct_def, pct_alg2, r.allocations,
                    r.residual);
    }
    std::printf(
        "\npct_def is against %s and pct_alg2 against %s; every percent names its "
        "algorithm.\n",
        ckl::spmv_algo_name(ckl::SpmvAlgo::kCusparseDefault),
        ckl::spmv_algo_name(ckl::SpmvAlgo::kCusparseAlg2));
    std::printf("SELL rows carry a padding ratio of %.4f (nnz_padded %lld over nnz %d).\n",
                plan.sell_padding_ratio(), plan.sell_nnz_padded(), a.nnz);

    if (total_allocations != 0) {
        std::fprintf(stderr,
                     "\nallocation gate FAILED: %lld cudaMalloc, cudaMallocAsync, cudaFree or "
                     "cudaFreeAsync call(s) happened inside a timed region. A measurement that "
                     "contains an allocation is defect A2 all over again.\n",
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
