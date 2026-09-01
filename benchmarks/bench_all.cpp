// One configuration, one JSON row. bench_all runs a single (family, variant,
// dtype, shape) benchmark with NVML sampling on and prints one JSON object.
// sweep.py and tile_sweep.py drive it across their matrices and append the rows
// to the canonical results files. Keeping the measurement in a small binary that
// does exactly one config makes the sweep resumable, and makes an independent
// process repeat a matter of running the binary again.
//
// Four rules from the Section 13 measurement protocol shape this file.
//
//   The GEMM variants speak the descriptor API. A variant name maps to a
//   ckl::Algo through one table, the call goes through ckl::gemm on a Context
//   created once, and the chosen out-param is asserted equal to the algorithm
//   the row claims. A benchmark that silently measures a different path than its
//   row names is defect class A2, and an assertion is the only thing that stops
//   it happening again.
//
//   Nothing is timed before it is verified. One launch of the kernel and one of
//   the vendor oracle land in separate buffers and are compared against the
//   family tolerance before the timing loop opens. A mismatch prints a JSON
//   error object and exits non-zero, so a silently wrong or early returning
//   kernel cannot post a number.
//
//   The vendor baseline is a variant of its own (baseline_cublas_default,
//   baseline_cublas_autotune, baseline_cusparse), measured in its own process
//   once per shape and joined onto the variant rows by the sweep driver. v1
//   re-measured the baseline inside every row and saw a 1.9x spread on identical
//   calls, which is where the impossible percentages of defect A2 came from.
//
//   The cache state and the launch path are inputs, not accidents. --flush-l2
//   writes an over-L2-size scratch buffer between reps, outside the timed
//   region, and --launch-mode graph captures the rep loop so launch overhead can
//   be read off by difference. Both land in the row, so no two rows can be
//   compared without seeing which protocol produced them.
//
// Usage: bench_all <family> <variant> <dtype> <m> <n> <k> [commit] [flags]
//   family in {gemm, gemv, spmv, trsm}; dtype in {fp32, fp16, bf16}
// bench_all --help lists the flags.
//
// The spmv family takes a matrix name where the other three take m. One
// synthetic matrix cannot separate the SpMV variants, so a row names a matrix
// from experiments/matrix_manifest.csv and the shape it reports is the shape
// that matrix actually has. "synthetic" and "synthetic:<rows>" build v1's
// generator in process, so a sweep still runs on a machine whose matrix cache
// has not been populated.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "ckl/ckl.h"
#include "ckl/context.hpp"
#include "ckl/cuda_check.hpp"
#include "ckl/device_buffer.hpp"
#include "event_timer.hpp"
#include "ckl/gemm.hpp"
#include "ckl/gemv.hpp"
#include "matrix_cache.hpp"
#include "ckl/nvml_monitor.hpp"
#include "reference.hpp"
#include "ckl/sparse.hpp"
#include "ckl/trsm.hpp"
#include "ckl/types.hpp"

namespace {

// Schema version of the row this binary prints. The committed v1 sweep carries
// no such field at all, which is how a reader tells the two apart: a v1 row
// re-measured its own baseline inside the row and kept no samples.
constexpr int kSchemaVersion = 2;

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

struct Options {
    std::string family;
    std::string variant;
    std::string dtype;
    int m = 0;
    int n = 0;
    int k = 0;
    std::string commit = "unknown";

    // The spmv family names a matrix where the others give m. The name is what
    // a row carries, and it has to match a row of the manifest.
    std::string matrix;
    std::uint64_t seed = 1;

    int warmups = 5;
    int reps = 20;         // the exact count in fixed mode, the floor in adaptive
    bool adaptive = true;  // min(max_reps, budget_s), per Section 13
    int max_reps = 1000;
    double budget_s = 2.0;

    std::string flush = "auto";          // auto, on, off
    std::string launch_mode = "stream";  // stream, graph
    // Launches inside one event pair. One is the plain protocol. The graph
    // comparison sets the same value on both of its rows, so the difference
    // between them is launch overhead and nothing else.
    int inner = 1;

    bool disallow_reduced_precision = false;

    // Test only. Scales the oracle before the comparison so the verification
    // gate can be shown red on demand; ground rule 8 wants a gate that has been
    // seen failing. 1.0 is the only value any real measurement uses, and any
    // other value is stamped into the row.
    double verify_perturb = 1.0;

    bool probe_autotune = false;
};

[[noreturn]] void usage_and_exit(const char* argv0, int code) {
    std::fprintf(
        stderr,
        "usage: %s <family> <variant> <dtype> <m> <n> <k> [commit] [flags]\n"
        "  families: gemm gemv spmv trsm\n"
        "  dtypes:   fp32 fp16 bf16\n"
        "  gemm variants: naive tiled register cp_async wmma mma_ptx mma_ldm mma_opt auto\n"
        "                 tile_<BM>x<BN>x<BK> splitk streamk\n"
        "                 baseline_cublas_default baseline_cublas_autotune\n"
        "  gemv variants: naive warp vectorized baseline_cublas\n"
        "  spmv variants: naive warp vector merge sell bsr auto\n"
        "                 baseline_cusparse_default baseline_cusparse_alg2\n"
        "  spmv takes a matrix name in the m position: a name from\n"
        "  experiments/matrix_manifest.csv, or synthetic[:rows]. n and k are ignored.\n"
        "  trsm variants: naive blocked baseline_cublas\n"
        "flags:\n"
        "  --seed N               seed for the synthetic SpMV generator (default 1)\n"
        "  --warmups N            untimed launches before the loop (default 5)\n"
        "  --reps N               timed reps, and the floor in adaptive mode (default 20)\n"
        "  --fixed-reps           run exactly --reps instead of the adaptive budget\n"
        "  --max-reps N           adaptive ceiling (default 1000)\n"
        "  --budget-s S           adaptive wall clock budget in seconds (default 2.0)\n"
        "  --flush-l2 auto|on|off L2 flush between reps (default auto)\n"
        "  --launch-mode stream|graph\n"
        "  --inner N              launches inside one event pair (default 1)\n"
        "  --disallow-reduced-precision-reduction\n"
        "  --verify-perturb X     TEST ONLY: scale the oracle so verification fails\n"
        "  --probe-autotune       report whether CUBLAS_GEMM_AUTOTUNE runs, then exit\n",
        argv0);
    std::exit(code);
}

// ---------------------------------------------------------------------------
// JSON output
// ---------------------------------------------------------------------------

class JsonRow {
public:
    void str(const char* key, const std::string& value) {
        sep();
        body_ += '"';
        body_ += key;
        body_ += "\":\"";
        body_ += value;
        body_ += '"';
    }

    void num(const char* key, double value, int digits = 6) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%.*f", digits, value);
        raw(key, buf);
    }

    void sci(const char* key, double value) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%.6e", value);
        raw(key, buf);
    }

    void integer(const char* key, long long value) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%lld", value);
        raw(key, buf);
    }

    void boolean(const char* key, bool value) { raw(key, value ? "true" : "false"); }

    void array(const char* key, const std::vector<double>& values, int digits = 6) {
        std::string out = "[";
        char buf[64];
        for (std::size_t i = 0; i < values.size(); ++i) {
            if (i != 0) {
                out += ',';
            }
            std::snprintf(buf, sizeof(buf), "%.*f", digits, values[i]);
            out += buf;
        }
        out += ']';
        raw(key, out);
    }

    void raw(const char* key, const std::string& literal) {
        sep();
        body_ += '"';
        body_ += key;
        body_ += "\":";
        body_ += literal;
    }

    std::string text() const { return "{" + body_ + "}"; }

private:
    void sep() {
        if (!body_.empty()) {
            body_ += ',';
        }
    }
    std::string body_;
};

// Escapes what a diagnostic message can realistically contain. Every string here
// is built inside this binary, not read from input.
std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        if (c == '"' || c == '\\') {
            out += '\\';
            out += c;
        } else if (c == '\n') {
            out += "\\n";
        } else {
            out += c;
        }
    }
    return out;
}

const Options* g_opt = nullptr;

// A benchmark that cannot honestly produce a number says so on stdout, as an
// object with the same identifying fields a row carries, and exits non-zero.
// Nothing downstream has to guess what an empty line meant.
[[noreturn]] void fail_json(const char* stage, const std::string& message,
                            const std::function<void(JsonRow&)>& extra, int code) {
    JsonRow row;
    row.boolean("error", true);
    row.str("stage", stage);
    if (g_opt != nullptr) {
        row.str("family", g_opt->family);
        row.str("variant", g_opt->variant);
        row.str("dtype", g_opt->dtype);
        row.integer("m", g_opt->m);
        row.integer("n", g_opt->n);
        row.integer("k", g_opt->k);
        // For spmv the shape positions hold a matrix name, so the name is what
        // identifies the failed configuration.
        row.str("matrix", g_opt->family == "spmv" ? g_opt->matrix : std::string());
        row.str("commit", g_opt->commit);
    }
    if (extra) {
        extra(row);
    }
    row.str("message", json_escape(message));
    std::printf("%s\n", row.text().c_str());
    std::fflush(stdout);
    std::exit(code);
}

[[noreturn]] void fail_json(const char* stage, const std::string& message, int code) {
    fail_json(stage, message, nullptr, code);
}

// The detail the library left behind for the last failed call.
std::string library_detail() {
    char buf[512] = {0};
    ckl_last_error(buf, sizeof(buf));
    return std::string(buf);
}

// ---------------------------------------------------------------------------
// Host side helpers
// ---------------------------------------------------------------------------

template <typename T>
T from_float(float f);
template <>
float from_float<float>(float f) {
    return f;
}
template <>
__half from_float<__half>(float f) {
    return __float2half(f);
}
template <>
__nv_bfloat16 from_float<__nv_bfloat16>(float f) {
    return __float2bfloat16(f);
}

template <typename T>
std::vector<T> convert(const std::vector<float>& src) {
    std::vector<T> out(src.size());
    for (std::size_t i = 0; i < src.size(); ++i) {
        out[i] = from_float<T>(src[i]);
    }
    return out;
}

std::vector<float> absolute(const std::vector<float>& src) {
    std::vector<float> out(src.size());
    for (std::size_t i = 0; i < src.size(); ++i) {
        out[i] = std::fabs(src[i]);
    }
    return out;
}

double gflops_of(double flops, double ms) {
    return ms > 0.0 ? flops / (ms / 1000.0) / 1.0e9 : 0.0;
}

// The verification measure: worst elementwise |kernel - oracle| over the
// magnitude the rounding error is actually bounded by, which for a dot product
// is sum_p |a_ip| |b_pj|. That sum is itself a GEMM, so it is computed on the
// device from |A| and |B| rather than on the host, and the answer is directly
// comparable with ckl::tol(k), the same shape derived tolerance the correctness
// suite is written against. Dividing by |c| instead would make the bound depend
// on how much the dot product happened to cancel.
struct Residual {
    double worst = 0.0;
    long long index = -1;  ///< Where the worst element sits, so a red gate is debuggable.
    double got = 0.0;
    double oracle = 0.0;
    double scale = 0.0;
};

Residual scaled_residual(const std::vector<float>& got, const std::vector<float>& oracle,
                         const std::vector<float>& scale, double perturb) {
    Residual worst;
    for (std::size_t i = 0; i < got.size(); ++i) {
        const double ref = static_cast<double>(oracle[i]) * perturb;
        const double denom = scale[i] > 1e-30f ? static_cast<double>(scale[i]) : 1.0;
        const double r = std::fabs(static_cast<double>(got[i]) - ref) / denom;
        if (r > worst.worst) {
            worst.worst = r;
            worst.index = static_cast<long long>(i);
            worst.got = static_cast<double>(got[i]);
            worst.oracle = static_cast<double>(oracle[i]);
            worst.scale = static_cast<double>(scale[i]);
        }
    }
    return worst;
}

// ---------------------------------------------------------------------------
// What one run produces
// ---------------------------------------------------------------------------

struct Measured {
    ckl::TimingStats stats;
    double flops = 0.0;
    std::size_t working_set_bytes = 0;
    std::string chosen;
    std::string entry_point = "ckl_gemm";
    int tile_index = -1;
    std::string tile;
    int splits = 1;
    bool plan_tuned = false;
    std::string plan_algo;
    Residual verify;
    double verify_tol = 0.0;
    std::string cublas_gemm_algo;
    long long nnz = 0;
    bool l2_flushed = false;
    std::size_t flush_bytes = 0;

    // SpMV only. Every one of these rides on the row because the family's
    // claims cannot be read without them: a padding ratio decides whether a
    // SELL win is a win, and the model band is what a measured dram__bytes.sum
    // gets compared against.
    std::string matrix;
    int rows = 0;
    int cols = 0;
    double sell_padding_ratio = 0.0;
    long long sell_nnz_padded = 0;
    int vector_width = 0;
    int sigma = 0;
    int bsr_block_dim = 0;
    double bsr_detect_ms = 0.0;
    long long model_bytes_low = 0;
    long long model_bytes_high = 0;
    double mean_nnz_per_row = 0.0;
    int max_nnz_per_row = 0;
    bool power_law = false;
};

// ---------------------------------------------------------------------------
// The timed loop, shared by every family
// ---------------------------------------------------------------------------

std::size_t l2_bytes(const ckl::Context& ctx) {
    const int l2 = ctx.device_properties().l2CacheSize;
    return l2 > 0 ? static_cast<std::size_t>(l2) : (std::size_t{48} << 20);
}

// Whether this row flushes L2 between reps. The rule: on for the memory bound
// families, and on for any row whose whole working set fits in L2, because those
// are exactly the rows that would otherwise measure a cache no real caller
// arrives with.
bool decide_flush(const Options& opt, std::size_t working_set, std::size_t l2) {
    if (opt.flush == "on") {
        return true;
    }
    if (opt.flush == "off") {
        return false;
    }
    return opt.family == "gemv" || opt.family == "spmv" || working_set <= l2;
}

void run_timed(const Options& opt, const ckl::Context& ctx, cudaStream_t stream, Measured& out,
               const std::function<void(cudaStream_t)>& launch,
               const std::function<void(cudaStream_t)>& prologue) {
    const std::size_t l2 = l2_bytes(ctx);
    out.l2_flushed = decide_flush(opt, out.working_set_bytes, l2);

    // Twice L2 guarantees the flush write cannot leave any of the previous
    // working set resident, whatever the replacement policy does.
    ckl::DeviceBuffer<unsigned char> scratch;
    if (out.l2_flushed) {
        out.flush_bytes = 2 * l2;
        scratch = ckl::DeviceBuffer<unsigned char>(out.flush_bytes);
    }

    ckl::TimingOptions timing;
    timing.stream = stream;
    timing.warmups = opt.warmups;
    timing.reps = opt.reps;
    timing.adaptive = opt.adaptive;
    timing.max_reps = opt.max_reps;
    timing.budget_s = opt.budget_s;
    timing.graph = opt.launch_mode == "graph";
    // Both launch modes run the same number of launches inside one event pair,
    // so the whole difference between the paired rows is launch overhead and
    // nothing else. A stream row measured on its own uses one.
    timing.inner = opt.inner;
    timing.flush_buffer = out.l2_flushed ? scratch.data() : nullptr;
    timing.flush_bytes = out.l2_flushed ? out.flush_bytes : 0;
    timing.prologue = prologue;

    if (timing.graph && prologue) {
        // A prologue is host enqueued work between reps. It cannot live inside a
        // captured graph, and running it once per graph launch would restore a
        // buffer that the inner launches then consume inner times over.
        fail_json("usage",
                  "--launch-mode graph is not available for a family whose kernel consumes its "
                  "input and needs a restore between reps",
                  2);
    }
    out.stats = ckl::time_stream_ex(launch, timing);
}

// ---------------------------------------------------------------------------
// cuBLAS plumbing and the fairness protocol
// ---------------------------------------------------------------------------

void check_cublas(cublasStatus_t s, const char* what) {
    if (s != CUBLAS_STATUS_SUCCESS) {
        fail_json("cublas", std::string(what) + ": " + cublasGetStatusName(s), 5);
    }
}

// The row major identity cuBLAS is driven through everywhere in this project:
// C_t(n by m) = op(B)_t * op(A)_t, so a column major call writes exactly the row
// major C the hand written kernels produce. algo is the only thing that varies
// between the two named baseline variants; the protocol forbids sweeping it any
// further than default against autotune, because algo selection is a no-op on
// sm_80 and newer.
cublasStatus_t cublas_gemm_row_major(cublasHandle_t h, cudaDataType in_type, int m, int n, int k,
                                     float alpha, const void* a, const void* b, float beta,
                                     float* c, cublasGemmAlgo_t algo) {
    return cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, b, in_type, n, a, in_type, k,
                        &beta, c, CUDA_R_32F, n, CUBLAS_COMPUTE_32F, algo);
}

// cuBLAS still declares CUBLAS_GEMM_AUTOTUNE and still documents it as
// experimental, so nothing here assumes it runs: the value is probed against a
// real call through --probe-autotune and the answer is reported, not asserted.
#if defined(CUBLAS_VER_MAJOR) && CUBLAS_VER_MAJOR >= 12
constexpr bool kAutotuneDeclared = true;
constexpr cublasGemmAlgo_t kAutotuneAlgo = CUBLAS_GEMM_AUTOTUNE;
#else
constexpr bool kAutotuneDeclared = false;
constexpr cublasGemmAlgo_t kAutotuneAlgo = CUBLAS_GEMM_DEFAULT;
#endif

// The fairness protocol, in one place so it cannot drift between call sites.
//
//   cublasSetWorkspace is never called, at any size. One call forfeits the
//   default pool, which is 32 MiB on sm_120, and a baseline running without its
//   pool is not the baseline a user would get.
//   cublasSetStream comes first, because it resets the workspace to the default
//   pool; a workspace experiment that set the buffer first would lose it again.
//   The math mode is set explicitly and read back into the row, because with
//   FP16 in and CUBLAS_COMPUTE_32F cuBLAS may reduce split-K partials in reduced
//   precision, which moves both its speed and its accuracy as an oracle.
std::string configure_cublas(const Options& opt, cublasHandle_t h, cudaStream_t stream) {
    check_cublas(cublasSetStream(h, stream), "cublasSetStream");
    const cublasMath_t want =
        opt.disallow_reduced_precision
            ? static_cast<cublasMath_t>(CUBLAS_DEFAULT_MATH |
                                        CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION)
            : CUBLAS_DEFAULT_MATH;
    check_cublas(cublasSetMathMode(h, want), "cublasSetMathMode");
    cublasMath_t got = CUBLAS_DEFAULT_MATH;
    check_cublas(cublasGetMathMode(h, &got), "cublasGetMathMode");
    std::string name = "CUBLAS_DEFAULT_MATH";
    if ((static_cast<unsigned>(got) &
         static_cast<unsigned>(CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION)) != 0u) {
        name += "|CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION";
    }
    return name;
}

// ---------------------------------------------------------------------------
// The variant table
// ---------------------------------------------------------------------------

struct VariantEntry {
    const char* name;
    const char* dtype;  // empty means any dtype the family accepts
    ckl::Algo algo;
    bool baseline;
};

// Every GEMM variant name the sweep can ask for, and the ckl::Algo it must run.
// This replaces the v1 string if-chain over free function pointers: one table,
// one lookup, and the chosen out-param checked against the entry afterwards.
const VariantEntry kGemmVariants[] = {
    {"naive", "fp32", ckl::Algo::kNaive, false},
    {"tiled", "fp32", ckl::Algo::kTiled, false},
    {"register", "fp32", ckl::Algo::kRegister, false},
    {"cp_async", "fp32", ckl::Algo::kCpAsync, false},
    {"wmma", "fp16", ckl::Algo::kWmmaFp16, false},
    {"wmma", "bf16", ckl::Algo::kWmmaBf16, false},
    {"mma_ptx", "fp16", ckl::Algo::kMmaPtx, false},
    {"mma_ldm", "fp16", ckl::Algo::kMmaLdm, false},
    {"mma_opt", "fp16", ckl::Algo::kMmaOpt, false},
    {"splitk", "fp16", ckl::Algo::kSplitK, false},
    {"streamk", "fp16", ckl::Algo::kStreamK, false},
    {"auto", "", ckl::Algo::kAuto, false},
    {"baseline_cublas_default", "", ckl::Algo::kCublas, true},
    {"baseline_cublas_autotune", "", ckl::Algo::kCublas, true},
};

// "tile_128x128x32" back to a family index, or -1 when no shape matches. The
// name is built from the shape rather than from the index so a row in a
// committed sweep still means the same tile if the roster is ever reordered.
int tile_family_index(const std::string& variant) {
    for (int i = 0; i < ckl::gemm_tile_family_count(); ++i) {
        const ckl::GemmTile t = ckl::gemm_tile_family_shape(i);
        const std::string name =
            "tile_" + std::to_string(t.m) + "x" + std::to_string(t.n) + "x" + std::to_string(t.k);
        if (variant == name) {
            return i;
        }
    }
    return -1;
}

std::string tile_label(const ckl::GemmTile& t) {
    return std::to_string(t.m) + "x" + std::to_string(t.n) + "x" + std::to_string(t.k);
}

const VariantEntry* find_gemm_variant(const std::string& name, const std::string& dtype) {
    for (const VariantEntry& e : kGemmVariants) {
        if (name == e.name && (e.dtype[0] == '\0' || dtype == e.dtype)) {
            return &e;
        }
    }
    return nullptr;
}

ckl::DType dtype_of(const std::string& s) {
    if (s == "fp16") {
        return ckl::DType::kR16F;
    }
    if (s == "bf16") {
        return ckl::DType::kR16BF;
    }
    return ckl::DType::kR32F;
}

void require_verified(const Measured& out) {
    if (!(out.verify.worst <= out.verify_tol)) {
        const Residual r = out.verify;
        const double tolerance = out.verify_tol;
        fail_json(
            "verify",
            "the kernel and the reference disagree beyond the family tolerance; nothing was timed",
            [r, tolerance](JsonRow& row) {
                row.boolean("verify_ok", false);
                row.sci("verify_residual", r.worst);
                row.sci("verify_tol", tolerance);
                row.integer("verify_worst_index", r.index);
                row.sci("verify_worst_got", r.got);
                row.sci("verify_worst_reference", r.oracle);
                row.sci("verify_worst_scale", r.scale);
            },
            9);
    }
}

// ---------------------------------------------------------------------------
// GEMM
// ---------------------------------------------------------------------------

template <typename T>
Measured bench_gemm_typed(const Options& opt, ckl::Context& ctx, cudaStream_t stream,
                          const VariantEntry& entry, int pinned_tile, cudaDataType in_type) {
    const int m = opt.m;
    const int n = opt.n;
    const int k = opt.k;
    Measured out;
    out.flops = 2.0 * static_cast<double>(m) * n * k;
    out.working_set_bytes =
        (static_cast<std::size_t>(m) * k + static_cast<std::size_t>(k) * n) * sizeof(T) +
        static_cast<std::size_t>(m) * n * sizeof(float);

    const auto fa = ckl::random_matrix(m, k, 11);
    const auto fb = ckl::random_matrix(k, n, 22);
    ckl::DeviceBuffer<T> da(fa.size());
    ckl::DeviceBuffer<T> db(fb.size());
    ckl::DeviceBuffer<float> dc(static_cast<std::size_t>(m) * n);
    dc.zero();

    ckl::GemmDesc desc;
    desc.layout = ckl::Layout::kRowMajor;
    desc.m = m;
    desc.n = n;
    desc.k = k;
    desc.dt_a = dtype_of(opt.dtype);
    desc.dt_b = desc.dt_a;
    desc.dt_c = ckl::DType::kR32F;
    desc.lda = k;
    desc.ldb = n;
    desc.ldc = n;

    const bool pinned = pinned_tile >= 0;
    const bool autotune = std::string(entry.name) == "baseline_cublas_autotune";
    const ckl::Algo requested = pinned ? ckl::Algo::kTileFamily : entry.algo;

    // The plan, for the record: the whole dispatch decision, so a row can say
    // which rung and which tile kAuto would have run on this shape and whether a
    // committed tile sweep informed the answer.
    ckl::GemmDesc auto_desc = desc;
    auto_desc.algo = ckl::Algo::kAuto;
    const ckl::GemmPlan plan = ckl::gemm_plan(ctx, auto_desc);
    out.plan_tuned = plan.tuned;
    out.plan_algo = ckl::algo_name(plan.algo);

    ckl::GemmDesc named_desc = desc;
    named_desc.algo = requested;

    // Scratch for the split-K and stream-K drivers, sized from the API and given
    // to the Context, so no allocation happens inside a timed call.
    ckl::DeviceBuffer<unsigned char> workspace;
    const std::size_t ws = ckl::gemm_workspace_size(ctx, named_desc);
    if (ws > 0) {
        workspace = ckl::DeviceBuffer<unsigned char>(ws);
        ctx.set_workspace(workspace.data(), ws);
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
    float* raw_c = dc.data();
    auto* handle = static_cast<cublasHandle_t>(ctx.cublas());

    // The launch under test, exactly as the timed loop will run it. tile_* is
    // the one variant that cannot go through the descriptor API: GemmDesc names
    // a rung, not a family index, so a tile_* row calls the pinned entry point
    // and says so through entry_point. Everything else is ckl::gemm.
    std::function<void(cudaStream_t)> launch;
    if (pinned) {
        // The tile family is FP16 in only, which main() has already enforced;
        // the guard is what keeps the FP32 and BF16 instantiations of this
        // template from trying to name a __half entry point.
        if constexpr (std::is_same_v<T, __half>) {
            const int index = pinned_tile;
            launch = [&da, &db, raw_c, m, n, k, index](cudaStream_t s) {
                ckl::gemm_tile_family(da.data(), db.data(), raw_c, m, n, k, 1.0f, 0.0f, index, s);
            };
        }
        out.entry_point = "gemm_tile_family";
    } else if (autotune) {
        launch = [&da, &db, raw_c, handle, in_type, m, n, k](cudaStream_t s) {
            (void)s;  // the handle already carries the stream
            const cublasStatus_t st = cublas_gemm_row_major(
                handle, in_type, m, n, k, 1.0f, da.data(), db.data(), 0.0f, raw_c, kAutotuneAlgo);
            if (st != CUBLAS_STATUS_SUCCESS) {
                fail_json("launch",
                          std::string("cublasGemmEx with CUBLAS_GEMM_AUTOTUNE: ") +
                              cublasGetStatusName(st),
                          5);
            }
        };
        out.entry_point = "cublasGemmEx";
        out.cublas_gemm_algo = "CUBLAS_GEMM_AUTOTUNE";
    } else {
        launch = [&ctx, &named_desc, &da, &db, &alpha, &beta, raw_c](cudaStream_t s) {
            (void)s;  // the Context already carries the stream
            const ckl::Status st =
                ckl::gemm(ctx, named_desc, &alpha, da.data(), db.data(), &beta, raw_c, nullptr);
            if (st != ckl::Status::kSuccess) {
                fail_json("launch",
                          std::string("ckl::gemm returned ") + ckl::status_string(st) +
                              " inside the timing loop",
                          5);
            }
        };
        out.entry_point = "ckl_gemm";
        if (entry.algo == ckl::Algo::kCublas) {
            out.cublas_gemm_algo = "CUBLAS_GEMM_DEFAULT";
        }
    }

    // --- self verification, before a single timed launch ---
    {
        ckl::DeviceBuffer<float> d_scale(static_cast<std::size_t>(m) * n);
        ckl::DeviceBuffer<float> d_oracle(static_cast<std::size_t>(m) * n);

        ckl::GemmDesc vendor = desc;
        vendor.algo = ckl::Algo::kCublas;
        ckl::Algo vendor_chosen = ckl::Algo::kAuto;

        // sum_p |a_ip| |b_pj|, the magnitude the rounding error is bounded by,
        // computed on the device because on the host it is an m by n by k loop.
        da.copy_from_host(convert<T>(absolute(fa)));
        db.copy_from_host(convert<T>(absolute(fb)));
        ckl::Status st = ckl::gemm(ctx, vendor, &alpha, da.data(), db.data(), &beta, d_scale.data(),
                                   &vendor_chosen);
        if (st != ckl::Status::kSuccess) {
            fail_json("verify",
                      std::string("the magnitude reference call failed: ") + ckl::status_string(st),
                      6);
        }

        // The buffers about to be overwritten are still being read by the call
        // above. DeviceBuffer copies run on the default stream, and this stream
        // is non blocking, so nothing orders the two without this.
        CKL_CUDA_CHECK(cudaStreamSynchronize(stream));
        da.copy_from_host(convert<T>(fa));
        db.copy_from_host(convert<T>(fb));

        // One launch of the path under test, and the chosen assertion. An
        // explicitly named algorithm is never rerouted by the dispatcher, so a
        // mismatch here means the table and the library disagree about what the
        // name means, which is exactly the silent substitution A2 was.
        ckl::Algo chosen = ckl::Algo::kAuto;
        if (pinned) {
            // The dispatch probe goes into the oracle buffer, which the vendor
            // call below overwrites. C under test is written exactly once, by
            // the pinned tile, so nothing about the comparison depends on two
            // kernels having landed in the right order.
            ckl::GemmDesc family = desc;
            family.algo = ckl::Algo::kTileFamily;
            st = ckl::gemm(ctx, family, &alpha, da.data(), db.data(), &beta, d_oracle.data(),
                           &chosen);
            if (st != ckl::Status::kSuccess) {
                fail_json("dispatch",
                          std::string("the tile family refuses this shape: ") +
                              ckl::status_string(st) + "; " + library_detail(),
                          7);
            }
        }

        st = ckl::gemm(ctx, vendor, &alpha, da.data(), db.data(), &beta, d_oracle.data(),
                       &vendor_chosen);
        if (st != ckl::Status::kSuccess) {
            fail_json("verify",
                      std::string("the vendor oracle call failed: ") + ckl::status_string(st), 6);
        }

        if (pinned) {
            // The pinned tile, not the one the plan picked, is what the row is
            // about, so it is what gets timed and what gets verified.
            launch(stream);
        } else if (autotune) {
            chosen = ckl::Algo::kCublas;  // a direct cuBLAS call is the vendor path by construction
            // The first CUBLAS_GEMM_AUTOTUNE call for a shape is the tuning run
            // itself: cuBLAS benchmarks its candidates and caches the winner in
            // the handle. Whatever that leaves in C is not the answer, so the
            // call that gets verified is the one after it.
            launch(stream);
            CKL_CUDA_CHECK(cudaStreamSynchronize(stream));
            launch(stream);
        } else {
            st = ckl::gemm(ctx, named_desc, &alpha, da.data(), db.data(), &beta, raw_c, &chosen);
            if (st != ckl::Status::kSuccess) {
                fail_json("dispatch",
                          std::string("ckl::gemm returned ") + ckl::status_string(st) +
                              " for algo " + ckl::algo_name(requested) + "; " + library_detail(),
                          7);
            }
        }
        CKL_CUDA_CHECK(cudaStreamSynchronize(stream));

        const ckl::Algo expect = requested == ckl::Algo::kAuto ? chosen : requested;
        if (chosen != expect) {
            fail_json("dispatch",
                      std::string("variant ") + opt.variant + " asked for " +
                          ckl::algo_name(expect) + " and dispatch reports " +
                          ckl::algo_name(chosen) +
                          "; a benchmark that measures a path its row does not name is a defect",
                      8);
        }
        out.chosen = ckl::algo_name(chosen);

        const std::vector<float> got = dc.to_host();
        const std::vector<float> oracle = d_oracle.to_host();
        const std::vector<float> scale = d_scale.to_host();
        out.verify = scaled_residual(got, oracle, scale, opt.verify_perturb);
        out.verify_tol = ckl::tol(k);
        require_verified(out);
    }

    if (pinned) {
        out.tile_index = pinned_tile;
        out.tile = tile_label(ckl::gemm_tile_family_shape(pinned_tile));
    } else if (requested != ckl::Algo::kCublas && plan.tile_index >= 0) {
        out.tile_index = plan.tile_index;
        out.tile = tile_label(plan.tile);
        out.splits = plan.splits;
    }

    run_timed(opt, ctx, stream, out, launch, nullptr);
    return out;
}

// ---------------------------------------------------------------------------
// GEMV, SpMV and TRSM
// ---------------------------------------------------------------------------
//
// None of the three has a descriptor entry point in 1.1.0, so they keep their
// free function calls. What they do gain is the shared timing helper and the
// same verification against the vendor oracle, so the protocol is identical
// everywhere and only the call shape differs.

Measured bench_gemv(const Options& opt, const ckl::Context& ctx, cudaStream_t stream) {
    const int m = opt.m;
    const int n = opt.n;
    Measured out;
    out.flops = 2.0 * static_cast<double>(m) * n;
    out.working_set_bytes = (static_cast<std::size_t>(m) * n + static_cast<std::size_t>(n) +
                             static_cast<std::size_t>(m)) *
                            sizeof(float);
    out.entry_point = "free_function";

    std::function<void(const float*, const float*, float*, int, int, float, float, cudaStream_t)>
        kf;
    if (opt.variant == "naive") {
        kf = ckl::gemv_naive;
    } else if (opt.variant == "warp") {
        kf = ckl::gemv_warp;
    } else if (opt.variant == "vectorized") {
        kf = ckl::gemv_vectorized;
    } else if (opt.variant == "baseline_cublas") {
        kf = ckl::gemv_cublas;
        out.entry_point = "gemv_cublas";
        out.cublas_gemm_algo = "CUBLAS_GEMM_DEFAULT";
    } else {
        fail_json("variant", "unknown gemv variant " + opt.variant, 3);
    }
    out.chosen = opt.variant == "baseline_cublas" ? "cublas" : opt.variant;

    const auto a = ckl::random_matrix(m, n, 3);
    const auto x = ckl::random_matrix(n, 1, 5);
    ckl::DeviceBuffer<float> da(a.size());
    ckl::DeviceBuffer<float> dx(x.size());
    ckl::DeviceBuffer<float> dy(static_cast<std::size_t>(m));
    dy.zero();

    {
        ckl::DeviceBuffer<float> d_scale(static_cast<std::size_t>(m));
        ckl::DeviceBuffer<float> d_oracle(static_cast<std::size_t>(m));
        da.copy_from_host(absolute(a));
        dx.copy_from_host(absolute(x));
        ckl::gemv_cublas(da.data(), dx.data(), d_scale.data(), m, n, 1.0f, 0.0f, stream);
        CKL_CUDA_CHECK(cudaStreamSynchronize(stream));
        da.copy_from_host(a);
        dx.copy_from_host(x);
        ckl::gemv_cublas(da.data(), dx.data(), d_oracle.data(), m, n, 1.0f, 0.0f, stream);
        kf(da.data(), dx.data(), dy.data(), m, n, 1.0f, 0.0f, stream);
        CKL_CUDA_CHECK(cudaStreamSynchronize(stream));
        out.verify = scaled_residual(dy.to_host(), d_oracle.to_host(), d_scale.to_host(),
                                     opt.verify_perturb);
        out.verify_tol = ckl::tol(n);
        require_verified(out);
    }

    auto launch = [&da, &dx, &dy, kf, m, n](cudaStream_t s) {
        kf(da.data(), dx.data(), dy.data(), m, n, 1.0f, 0.0f, s);
    };
    run_timed(opt, ctx, stream, out, launch, nullptr);
    return out;
}

// Every SpMV variant name the sweep can ask for, and the ckl::SpmvAlgo it must
// run. Same table shape as the GEMM roster, and the chosen out-param is checked
// against the entry afterwards for the same reason.
struct SpmvVariantEntry {
    const char* name;
    ckl::SpmvAlgo algo;
};

const SpmvVariantEntry kSpmvVariants[] = {
    {"naive", ckl::SpmvAlgo::kCsrNaive},
    {"warp", ckl::SpmvAlgo::kCsrWarp},
    {"vector", ckl::SpmvAlgo::kCsrVector},
    {"merge", ckl::SpmvAlgo::kMerge},
    {"sell", ckl::SpmvAlgo::kSellCSigma},
    {"bsr", ckl::SpmvAlgo::kBsr},
    {"auto", ckl::SpmvAlgo::kAuto},
    {"baseline_cusparse_default", ckl::SpmvAlgo::kCusparseDefault},
    {"baseline_cusparse_alg2", ckl::SpmvAlgo::kCusparseAlg2},
    // The name the schema v2 rows already on file carry. Kept so an old command
    // line still runs; the sweep asks for the explicit name.
    {"baseline_cusparse", ckl::SpmvAlgo::kCusparseDefault},
};

const SpmvVariantEntry* find_spmv_variant(const std::string& name) {
    for (const SpmvVariantEntry& e : kSpmvVariants) {
        if (name == e.name) {
            return &e;
        }
    }
    return nullptr;
}

Measured bench_spmv(const Options& opt, const ckl::Context& ctx, cudaStream_t stream) {
    Measured out;
    out.entry_point = "ckl_spmv";

    const SpmvVariantEntry* entry = find_spmv_variant(opt.variant);
    if (entry == nullptr) {
        fail_json("variant", "unknown spmv variant " + opt.variant, 3);
    }

    ckl::bench::HostCsr a;
    std::string load_error;
    if (!ckl::bench::load_matrix(opt.matrix, opt.seed, &a, &load_error)) {
        fail_json("matrix", load_error, 10);
    }
    const int m = a.m;
    const int n = a.n;
    const int nnz = a.nnz;
    out.matrix = a.name;
    out.rows = m;
    out.cols = n;
    out.nnz = nnz;
    out.flops = 2.0 * static_cast<double>(nnz);
    out.working_set_bytes =
        static_cast<std::size_t>(nnz) * (sizeof(int) + sizeof(float)) +
        a.row_ptr.size() * sizeof(int) +
        (static_cast<std::size_t>(n) + static_cast<std::size_t>(m)) * sizeof(float);

    const auto x = ckl::random_matrix(n, 1, 7);
    ckl::DeviceBuffer<int> drp(a.row_ptr.size());
    ckl::DeviceBuffer<int> dci(a.col_idx.size());
    ckl::DeviceBuffer<float> dv(a.values.size());
    ckl::DeviceBuffer<float> dx(x.size());
    ckl::DeviceBuffer<float> dy(static_cast<std::size_t>(m));
    drp.copy_from_host(a.row_ptr);
    if (nnz > 0) {
        dci.copy_from_host(a.col_idx);
        dv.copy_from_host(a.values);
    }
    dy.zero();

    // The plan, built once, before anything is timed. It owns the descriptors,
    // both vendor workspaces, the histogram and the two converted layouts, and
    // that is the whole of the A2 fix: a timed call enqueues launches only.
    ckl::SpmvCsr view;
    view.row_ptr = drp.data();
    view.col_idx = dci.data();
    view.values = dv.data();
    view.m = m;
    view.n = n;
    view.nnz = nnz;
    ckl::SpmvPlan plan(view);
    out.sell_padding_ratio = plan.sell_padding_ratio();
    out.sell_nnz_padded = plan.sell_nnz_padded();
    out.vector_width = plan.vector_width();
    out.sigma = plan.sigma();
    out.bsr_block_dim = plan.bsr_block_dim();
    out.bsr_detect_ms = plan.bsr_detect_ms();
    out.model_bytes_low = plan.model_bytes_low(false);
    out.model_bytes_high = plan.model_bytes_high(false);
    out.mean_nnz_per_row = plan.mean_nnz_per_row();
    out.max_nnz_per_row = plan.max_nnz_per_row();
    out.power_law = plan.power_law();
    out.plan_algo = ckl::spmv_algo_name(plan.query());

    // --- self verification, before a single timed launch ---
    {
        ckl::DeviceBuffer<float> d_scale(static_cast<std::size_t>(m));
        ckl::DeviceBuffer<float> d_oracle(static_cast<std::size_t>(m));

        // sum_k |a_ik| |x_k|, the magnitude the rounding error is bounded by.
        // The plan holds pointers into dv and dx rather than copies of them, and
        // the sparsity pattern is untouched, so swapping the values in and out
        // is safe. The SELL and BSR copies were taken from the real values at
        // construction, which is why the oracle call below runs after the
        // restore rather than before it.
        if (nnz > 0) {
            dv.copy_from_host(absolute(a.values));
        }
        dx.copy_from_host(absolute(x));
        ckl::Status st = ckl::spmv(plan, ckl::SpmvAlgo::kCusparseDefault, 1.0f, dx.data(), 0.0f,
                                   d_scale.data(), nullptr, stream);
        if (st != ckl::Status::kSuccess) {
            fail_json("verify",
                      std::string("the magnitude reference call failed: ") + ckl::status_string(st),
                      6);
        }
        CKL_CUDA_CHECK(cudaStreamSynchronize(stream));
        if (nnz > 0) {
            dv.copy_from_host(a.values);
        }
        dx.copy_from_host(x);

        st = ckl::spmv(plan, ckl::SpmvAlgo::kCusparseDefault, 1.0f, dx.data(), 0.0f,
                       d_oracle.data(), nullptr, stream);
        if (st != ckl::Status::kSuccess) {
            fail_json("verify",
                      std::string("the vendor oracle call failed: ") + ckl::status_string(st), 6);
        }

        ckl::SpmvAlgo chosen = ckl::SpmvAlgo::kAuto;
        st = ckl::spmv(plan, entry->algo, 1.0f, dx.data(), 0.0f, dy.data(), &chosen, stream);
        if (st != ckl::Status::kSuccess) {
            fail_json("dispatch",
                      std::string("ckl::spmv returned ") + ckl::status_string(st) + " for algo " +
                          ckl::spmv_algo_name(entry->algo) + "; " + library_detail(),
                      7);
        }
        CKL_CUDA_CHECK(cudaStreamSynchronize(stream));

        const ckl::SpmvAlgo expect = entry->algo == ckl::SpmvAlgo::kAuto ? chosen : entry->algo;
        if (chosen != expect) {
            fail_json("dispatch",
                      std::string("variant ") + opt.variant + " asked for " +
                          ckl::spmv_algo_name(expect) + " and dispatch reports " +
                          ckl::spmv_algo_name(chosen) +
                          "; a benchmark that measures a path its row does not name is a defect",
                      8);
        }
        out.chosen = ckl::spmv_algo_name(chosen);

        out.verify = scaled_residual(dy.to_host(), d_oracle.to_host(), d_scale.to_host(),
                                     opt.verify_perturb);
        // The contraction length of an SpMV row is its nonzero count, so the
        // tolerance comes from the longest row rather than from n.
        int longest = 1;
        for (std::size_t i = 1; i < a.row_ptr.size(); ++i) {
            longest = std::max(longest, a.row_ptr[i] - a.row_ptr[i - 1]);
        }
        out.verify_tol = ckl::tol(longest);
        require_verified(out);
    }

    const ckl::SpmvAlgo run_algo = entry->algo;
    auto launch = [&plan, &dx, &dy, run_algo](cudaStream_t s) {
        const ckl::Status st =
            ckl::spmv(plan, run_algo, 1.0f, dx.data(), 0.0f, dy.data(), nullptr, s);
        if (st != ckl::Status::kSuccess) {
            fail_json("launch",
                      std::string("ckl::spmv returned ") + ckl::status_string(st) +
                          " inside the timing loop",
                      5);
        }
    };
    run_timed(opt, ctx, stream, out, launch, nullptr);
    return out;
}

Measured bench_trsm(const Options& opt, ckl::Context& ctx, cudaStream_t stream) {
    const int m = opt.m;
    const int n = opt.n;
    Measured out;
    // A triangular solve does about m*m*n flops.
    out.flops = static_cast<double>(m) * m * n;
    out.working_set_bytes =
        (static_cast<std::size_t>(m) * m + static_cast<std::size_t>(m) * n) * sizeof(float);
    out.entry_point = "free_function";

    std::function<void(const float*, float*, int, int, float, cudaStream_t)> kf;
    if (opt.variant == "naive") {
        kf = ckl::trsm_naive;
    } else if (opt.variant == "blocked") {
        kf = ckl::trsm_blocked;
    } else if (opt.variant == "baseline_cublas") {
        kf = ckl::trsm_cublas;
        out.entry_point = "trsm_cublas";
        out.cublas_gemm_algo = "CUBLAS_GEMM_DEFAULT";
    } else {
        fail_json("variant", "unknown trsm variant " + opt.variant, 3);
    }
    out.chosen = opt.variant == "baseline_cublas" ? "cublas" : opt.variant;

    auto a = ckl::random_matrix(m, m, 71);
    for (int i = 0; i < m; ++i) {
        for (int j = i + 1; j < m; ++j) {
            a[static_cast<std::size_t>(i) * m + j] = 0.0f;
        }
        a[static_cast<std::size_t>(i) * m + i] = static_cast<float>(m + 1);
    }
    const auto b0 = ckl::random_matrix(m, n, 92);
    ckl::DeviceBuffer<float> da(a.size());
    ckl::DeviceBuffer<float> db(b0.size());
    ckl::DeviceBuffer<float> db_pristine(b0.size());
    da.copy_from_host(a);
    // The pristine copy is uploaded once, outside every timed region.
    db_pristine.copy_from_host(b0);

    const std::size_t b_bytes = db.bytes();
    // TRSM overwrites its right hand side, so every rep needs a fresh B. v1 did
    // that with a blocking host to device copy inside the timed lambda, which
    // put a PCIe transfer between the two event records and inflated both the
    // kernel and the cuBLAS baseline (defect A2). This restore is a device to
    // device cudaMemcpyAsync on the same stream, enqueued before the start
    // event, so stream ordering guarantees it has finished by the time the timed
    // region opens. Same protocol for the kernel under test and for the baseline.
    auto restore = [&db, &db_pristine, b_bytes](cudaStream_t s) {
        CKL_CUDA_CHECK(
            cudaMemcpyAsync(db.data(), db_pristine.data(), b_bytes, cudaMemcpyDeviceToDevice, s));
    };

    {
        // A solve is verified by its backward residual, not against another
        // solve: multiply the answer back through L and see whether B comes out.
        // The magnitude bound is then the same |L| times |X| product the GEMM
        // check uses, which is what makes ckl::tol(m) the right yardstick here
        // too. Comparing two solves elementwise instead would need |L inverse|,
        // and (|L|) inverse is not it.
        restore(stream);
        kf(da.data(), db.data(), m, n, 1.0f, stream);
        CKL_CUDA_CHECK(cudaStreamSynchronize(stream));
        const std::vector<float> x = db.to_host();

        ckl::DeviceBuffer<float> d_abs_a(a.size());
        ckl::DeviceBuffer<float> d_abs_x(x.size());
        ckl::DeviceBuffer<float> d_prod(x.size());
        ckl::DeviceBuffer<float> d_scale(x.size());
        d_abs_a.copy_from_host(absolute(a));
        d_abs_x.copy_from_host(absolute(x));

        ckl::GemmDesc back;
        back.layout = ckl::Layout::kRowMajor;
        back.m = m;
        back.n = n;
        back.k = m;
        back.lda = m;
        back.ldb = n;
        back.ldc = n;
        back.algo = ckl::Algo::kCublas;
        const float one = 1.0f;
        const float zero = 0.0f;
        ckl::Status st =
            ckl::gemm(ctx, back, &one, da.data(), db.data(), &zero, d_prod.data(), nullptr);
        if (st == ckl::Status::kSuccess) {
            st = ckl::gemm(ctx, back, &one, d_abs_a.data(), d_abs_x.data(), &zero, d_scale.data(),
                           nullptr);
        }
        if (st != ckl::Status::kSuccess) {
            fail_json(
                "verify",
                std::string("the backward residual product failed: ") + ckl::status_string(st), 6);
        }
        CKL_CUDA_CHECK(cudaStreamSynchronize(stream));

        out.verify = scaled_residual(d_prod.to_host(), b0, d_scale.to_host(), opt.verify_perturb);
        out.verify_tol = ckl::tol(m);
        require_verified(out);
    }

    auto launch = [&da, &db, kf, m, n](cudaStream_t s) { kf(da.data(), db.data(), m, n, 1.0f, s); };
    run_timed(opt, ctx, stream, out, launch, restore);
    return out;
}

// ---------------------------------------------------------------------------
// Argument parsing
// ---------------------------------------------------------------------------

int need_int(const char* flag, int argc, char** argv, int& i) {
    if (i + 1 >= argc) {
        fail_json("usage", std::string(flag) + " needs a value", 2);
    }
    return std::atoi(argv[++i]);
}

double need_double(const char* flag, int argc, char** argv, int& i) {
    if (i + 1 >= argc) {
        fail_json("usage", std::string(flag) + " needs a value", 2);
    }
    return std::atof(argv[++i]);
}

std::string need_str(const char* flag, int argc, char** argv, int& i) {
    if (i + 1 >= argc) {
        fail_json("usage", std::string(flag) + " needs a value", 2);
    }
    return argv[++i];
}

Options parse(int argc, char** argv) {
    Options opt;
    std::vector<std::string> positional;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            usage_and_exit(argv[0], 0);
        } else if (arg == "--warmups") {
            opt.warmups = need_int("--warmups", argc, argv, i);
        } else if (arg == "--reps") {
            opt.reps = need_int("--reps", argc, argv, i);
        } else if (arg == "--fixed-reps") {
            opt.adaptive = false;
        } else if (arg == "--max-reps") {
            opt.max_reps = need_int("--max-reps", argc, argv, i);
        } else if (arg == "--budget-s") {
            opt.budget_s = need_double("--budget-s", argc, argv, i);
        } else if (arg == "--flush-l2") {
            opt.flush = need_str("--flush-l2", argc, argv, i);
        } else if (arg == "--launch-mode") {
            opt.launch_mode = need_str("--launch-mode", argc, argv, i);
        } else if (arg == "--inner") {
            opt.inner = need_int("--inner", argc, argv, i);
        } else if (arg == "--disallow-reduced-precision-reduction") {
            opt.disallow_reduced_precision = true;
        } else if (arg == "--seed") {
            opt.seed = static_cast<std::uint64_t>(need_int("--seed", argc, argv, i));
        } else if (arg == "--verify-perturb") {
            opt.verify_perturb = need_double("--verify-perturb", argc, argv, i);
        } else if (arg == "--probe-autotune") {
            opt.probe_autotune = true;
        } else if (arg.rfind("--", 0) == 0) {
            std::fprintf(stderr, "unknown flag %s\n", arg.c_str());
            usage_and_exit(argv[0], 2);
        } else {
            positional.push_back(arg);
        }
    }
    if (opt.probe_autotune) {
        return opt;
    }
    if (positional.size() < 6) {
        usage_and_exit(argv[0], 2);
    }
    opt.family = positional[0];
    opt.variant = positional[1];
    opt.dtype = positional[2];
    // The spmv family names a matrix here. m and n are filled in from the
    // matrix once it is loaded, so the row's shape is the shape the matrix has
    // and never a pair of numbers somebody typed next to a name.
    opt.matrix = positional[3];
    opt.m = std::atoi(positional[3].c_str());
    opt.n = std::atoi(positional[4].c_str());
    opt.k = std::atoi(positional[5].c_str());
    if (positional.size() > 6) {
        opt.commit = positional[6];
    }
    return opt;
}

int probe_autotune() {
    JsonRow row;
    row.boolean("cublas_autotune_declared", kAutotuneDeclared);
    if (!kAutotuneDeclared) {
        row.boolean("cublas_autotune_available", false);
        row.str("note", "this cuBLAS does not declare CUBLAS_GEMM_AUTOTUNE");
        std::printf("%s\n", row.text().c_str());
        return 0;
    }
    ckl::Context ctx;
    auto* h = static_cast<cublasHandle_t>(ctx.cublas());
    constexpr int kSide = 128;
    ckl::DeviceBuffer<__half> a(static_cast<std::size_t>(kSide) * kSide);
    ckl::DeviceBuffer<__half> b(static_cast<std::size_t>(kSide) * kSide);
    ckl::DeviceBuffer<float> c(static_cast<std::size_t>(kSide) * kSide);
    a.zero();
    b.zero();
    c.zero();
    const cublasStatus_t st =
        cublas_gemm_row_major(h, CUDA_R_16F, kSide, kSide, kSide, 1.0f, a.data(), b.data(), 0.0f,
                              c.data(), kAutotuneAlgo);
    const bool ok = st == CUBLAS_STATUS_SUCCESS && cudaDeviceSynchronize() == cudaSuccess;
    row.boolean("cublas_autotune_available", ok);
    row.str("status", cublasGetStatusName(st));
    std::printf("%s\n", row.text().c_str());
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    const Options opt = parse(argc, argv);
    g_opt = &opt;
    if (opt.probe_autotune) {
        return probe_autotune();
    }
    if (opt.launch_mode != "stream" && opt.launch_mode != "graph") {
        fail_json("usage", "--launch-mode takes stream or graph", 2);
    }
    if (opt.flush != "auto" && opt.flush != "on" && opt.flush != "off") {
        fail_json("usage", "--flush-l2 takes auto, on or off", 2);
    }

    ckl::NvmlMonitor monitor(25);
    monitor.start();

    // One Context for the whole process, and one stream that is not the legacy
    // default one, because graph capture refuses that one.
    //
    // The stream is deliberately created blocking rather than with
    // cudaStreamNonBlocking. DeviceBuffer uploads use plain cudaMemcpy, which
    // for pageable host memory returns as soon as the bytes reach the driver's
    // staging buffer and leaves the DMA to the device running on the legacy
    // default stream. A non blocking stream does not wait for that stream, so a
    // kernel enqueued right after an upload can read the previous contents of
    // the operand. That is not theoretical: it made one verification in about
    // twenty fail with a handful of elements computed from stale data, and it
    // took a standalone repro to pin on the harness rather than on the kernels.
    // A blocking stream carries the implicit ordering against the legacy stream
    // and costs nothing here, since nothing else runs concurrently.
    ckl::Context ctx;
    cudaStream_t stream = nullptr;
    CKL_CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamDefault));
    ctx.set_stream(stream);

    const std::string math_mode =
        configure_cublas(opt, static_cast<cublasHandle_t>(ctx.cublas()), stream);
    // The free function families route through the process wide Context, so its
    // handle gets the same treatment; otherwise the row's math mode field would
    // be a claim about a handle nothing used.
    configure_cublas(opt, static_cast<cublasHandle_t>(ckl::detail::default_context().cublas()),
                     stream);

    Measured measured;
    if (opt.family == "gemm") {
        const int pinned = tile_family_index(opt.variant);
        static const VariantEntry kPinnedEntry{"tile_family", "fp16", ckl::Algo::kTileFamily,
                                               false};
        const VariantEntry* entry =
            pinned >= 0 ? &kPinnedEntry : find_gemm_variant(opt.variant, opt.dtype);
        if (entry == nullptr) {
            fail_json("variant", "unknown gemm variant " + opt.variant + " for dtype " + opt.dtype,
                      3);
        }
        if (pinned >= 0 && opt.dtype != "fp16") {
            fail_json("variant", "the tile family is FP16 in and FP32 out only", 3);
        }
        if (opt.dtype == "fp32") {
            measured = bench_gemm_typed<float>(opt, ctx, stream, *entry, pinned, CUDA_R_32F);
        } else if (opt.dtype == "fp16") {
            measured = bench_gemm_typed<__half>(opt, ctx, stream, *entry, pinned, CUDA_R_16F);
        } else if (opt.dtype == "bf16") {
            measured =
                bench_gemm_typed<__nv_bfloat16>(opt, ctx, stream, *entry, pinned, CUDA_R_16BF);
        } else {
            fail_json("dtype", "unknown dtype " + opt.dtype, 3);
        }
    } else if (opt.family == "gemv") {
        measured = bench_gemv(opt, ctx, stream);
    } else if (opt.family == "spmv") {
        measured = bench_spmv(opt, ctx, stream);
    } else if (opt.family == "trsm") {
        measured = bench_trsm(opt, ctx, stream);
    } else {
        fail_json("family", "unknown family " + opt.family, 2);
    }

    const ckl::NvmlSummary nv = monitor.stop();
    int rt = 0;
    int drv = 0;
    cudaRuntimeGetVersion(&rt);
    cudaDriverGetVersion(&drv);

    JsonRow row;
    row.integer("schema_version", kSchemaVersion);
    row.str("family", opt.family);
    row.str("variant", opt.variant);
    row.str("dtype", opt.dtype);
    // For spmv the shape comes from the matrix that was loaded, not from the
    // command line, so a row cannot claim a shape its matrix does not have.
    const bool sparse = opt.family == "spmv";
    row.integer("m", sparse ? measured.rows : opt.m);
    row.integer("n", sparse ? measured.cols : opt.n);
    row.integer("k", opt.k);
    row.str("matrix", measured.matrix);
    row.num("median_ms", measured.stats.median_ms);
    row.num("iqr_ms", measured.stats.iqr_ms);
    row.num("min_ms", measured.stats.min_ms);
    row.integer("reps", measured.stats.reps);
    row.integer("warmups", opt.warmups);
    row.array("samples", measured.stats.samples);
    row.num("gflops", gflops_of(measured.flops, measured.stats.median_ms), 3);
    // A variant row carries no baseline of its own any more. The sweep driver
    // measures the vendor baseline once per shape as its own row and joins it
    // on, so every variant at a shape is quoted against the same measurement
    // instead of against a fresh one with a 1.9x spread.
    row.str("baseline_source", "joined_by_sweep");
    row.str("chosen", measured.chosen);
    row.str("entry_point", measured.entry_point);
    row.str("plan_algo", measured.plan_algo);
    row.boolean("plan_tuned", measured.plan_tuned);
    row.integer("tile_index", measured.tile_index);
    row.str("tile", measured.tile);
    row.integer("splits", measured.splits);
    row.integer("nnz", measured.nnz);
    if (sparse) {
        // The SpMV family's claims cannot be read without these. The padding
        // ratio decides whether a SELL win is a win, the model band is what a
        // measured dram__bytes.sum gets compared against, and the width, sigma
        // and block dimension are the plan decisions the row was measured under.
        row.num("sell_padding_ratio", measured.sell_padding_ratio, 4);
        row.integer("sell_nnz_padded", measured.sell_nnz_padded);
        row.integer("vector_width", measured.vector_width);
        row.integer("sigma", measured.sigma);
        row.integer("bsr_block_dim", measured.bsr_block_dim);
        row.num("bsr_detect_ms", measured.bsr_detect_ms, 4);
        row.integer("model_bytes_low", measured.model_bytes_low);
        row.integer("model_bytes_high", measured.model_bytes_high);
        row.num("mean_nnz_per_row", measured.mean_nnz_per_row, 4);
        row.integer("max_nnz_per_row", measured.max_nnz_per_row);
        row.boolean("power_law", measured.power_law);
        // The number of record for percent of roof. Nothing measures it here;
        // the ncu round does, and until it runs the field says so.
        row.str("dram_bytes_sum", "pending ncu round");
    }
    row.boolean("verify_ok", true);
    row.sci("verify_residual", measured.verify.worst);
    row.integer("verify_worst_index", measured.verify.index);
    row.sci("verify_tol", measured.verify_tol);
    row.num("verify_perturb", opt.verify_perturb, 6);
    row.boolean("l2_flushed", measured.l2_flushed);
    row.integer("flush_bytes", static_cast<long long>(measured.flush_bytes));
    row.integer("working_set_bytes", static_cast<long long>(measured.working_set_bytes));
    row.str("launch_mode", opt.launch_mode);
    row.integer("inner_launches", opt.inner);
    row.str("timing_mode", opt.adaptive ? "adaptive" : "fixed");
    row.num("budget_s", opt.budget_s, 3);
    row.integer("max_reps", opt.max_reps);
    row.str("cublas_math_mode", math_mode);
    row.boolean("cublas_reduced_precision_reduction_disallowed", opt.disallow_reduced_precision);
    row.str("cublas_gemm_algo", measured.cublas_gemm_algo);
    row.boolean("cublas_autotune_declared", kAutotuneDeclared);
    row.boolean("nvml_available", nv.available);
    row.num("median_sm_clock_mhz", nv.median_sm_clock_mhz, 0);
    row.integer("max_temp_c", nv.max_temperature_c);
    row.num("max_power_w", nv.max_power_w, 1);
    row.boolean("throttled", nv.throttled);
    row.integer("nvml_samples", nv.samples);
    row.integer("cuda_runtime", rt);
    row.integer("cuda_driver", drv);
    row.str("commit", opt.commit);
    std::printf("%s\n", row.text().c_str());

    CKL_CUDA_CHECK(cudaStreamSynchronize(stream));
    CKL_CUDA_CHECK(cudaStreamDestroy(stream));
    return 0;
}
