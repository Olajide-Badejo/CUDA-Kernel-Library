// GEMM dispatch: descriptor validation, the kAuto heuristic, the architecture
// gate, and the cuBLAS fallback path that covers every layout and transpose the
// hand written kernels do not.
//
// Two rules shape the whole file. First, *chosen is always written when it is
// non-null, on success and on failure alike, so a caller can see which rung ran
// and a regression that makes every shape fall back cannot pass a test.
// Second, an explicitly named algorithm is never rerouted: if the shape cannot
// take it the call returns kNotSupported and says why. kAuto is the only mode
// allowed to choose, and it says what it chose.
//
// The FP16 side of the kAuto heuristic is Section 9.6 of the build spec. For
// every block tile in the family it computes
//
//   waves = ceil(m/BM) * ceil(n/BN) / (SMs * blocksPerSM)
//
// with blocksPerSM from the occupancy API rather than from an assumption about
// registers, and picks the shape that maximizes waves / ceil(waves), the
// fraction of the last wave that is actually busy, among the shapes whose
// measured throughput at the nearest swept shape is within five percent of the
// best. When waves is below one the grid cannot fill the machine at all and the
// decision escalates to split-K or to stream-K.
//
// The measured part of that rule needs a committed tile sweep, and the sweep is
// owner work at locked clocks. Until one exists the throughput filter has
// nothing to filter with, so the untuned path decides in two bands. Above two
// waves of the 128 by 128 by 32 reference tile it keeps that tile: quantization
// alone would prefer the smallest shape in the family, which is the wrong trade
// where the grid is already deep, and the reference tile is the only one with
// measured history behind it. Below two waves it falls back to quantization
// alone, and below one wave it escalates. Neither band is a measurement, and
// every answer says so: ckl::GemmPlan::tuned is false for the whole untuned
// path.
//
// The FP32 and BF16 sides are unchanged and still provisional in the older
// sense: they divide the shape by the block factors each kernel needs and pick
// the largest tile that fits.

#include "ckl/gemm.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "ckl/cuda_check.hpp"
#include "ckl/status.hpp"
#include "ckl/types.hpp"

#include "detail/last_error.hpp"

namespace ckl {

namespace {

// ---------------------------------------------------------------------------
// Descriptor geometry
// ---------------------------------------------------------------------------

struct Stored {
    std::int64_t rows;
    std::int64_t cols;
};

Stored stored_a(const GemmDesc& d) {
    return d.op_a == Op::kN ? Stored{d.m, d.k} : Stored{d.k, d.m};
}

Stored stored_b(const GemmDesc& d) {
    return d.op_b == Op::kN ? Stored{d.k, d.n} : Stored{d.n, d.k};
}

// Smallest leading dimension the layout allows for a matrix of this shape.
std::int64_t min_ld(Layout layout, Stored s) {
    return layout == Layout::kRowMajor ? s.cols : s.rows;
}

bool packed(const GemmDesc& d) {
    return d.lda == min_ld(d.layout, stored_a(d)) && d.ldb == min_ld(d.layout, stored_b(d)) &&
           d.ldc == min_ld(d.layout, Stored{d.m, d.n});
}

// What the hand written kernels can take at all: row major, no transpose,
// packed, unbatched, and small enough that the int dimensions in their
// signatures are exact.
bool hand_kernel_shape(const GemmDesc& d) {
    constexpr std::int64_t kIntMax = 2147483647;
    return d.layout == Layout::kRowMajor && d.op_a == Op::kN && d.op_b == Op::kN && packed(d) &&
           d.batch_count == 1 && d.m <= kIntMax && d.n <= kIntMax && d.k <= kIntMax;
}

bool divides(const GemmDesc& d, std::int64_t bm, std::int64_t bn, std::int64_t bk) {
    return d.m % bm == 0 && d.n % bn == 0 && d.k % bk == 0;
}

bool is_fp16_in(const GemmDesc& d) {
    return d.dt_a == DType::kR16F && d.dt_b == DType::kR16F && d.dt_c == DType::kR32F;
}

bool is_bf16_in(const GemmDesc& d) {
    return d.dt_a == DType::kR16BF && d.dt_b == DType::kR16BF && d.dt_c == DType::kR32F;
}

bool is_fp32(const GemmDesc& d) {
    return d.dt_a == DType::kR32F && d.dt_b == DType::kR32F && d.dt_c == DType::kR32F;
}

// ---------------------------------------------------------------------------
// Per algorithm requirements
// ---------------------------------------------------------------------------

// Compute capability the algorithm's device body needs. Below it the body is
// compiled out, so the launcher must refuse rather than run an empty kernel.
int required_cc(Algo a) {
    switch (a) {
        case Algo::kCpAsync:
        case Algo::kWmmaFp16:
        case Algo::kWmmaBf16:
        case Algo::kMmaPtx:
        case Algo::kMmaLdm:
        case Algo::kMmaOpt:
        case Algo::kTileFamily:
        case Algo::kSplitK:
        case Algo::kStreamK:
            return 80;
        default:
            return 0;
    }
}

bool in_tile_family(Algo a) {
    return a == Algo::kTileFamily || a == Algo::kSplitK || a == Algo::kStreamK;
}

// Why this shape cannot take this algorithm, or an empty string when it can.
std::string why_not(Algo a, const GemmDesc& d) {
    if (!hand_kernel_shape(d)) {
        return "the hand written kernels take row major, no transpose, packed leading dimensions, "
               "one batch, and dimensions inside int range";
    }
    switch (a) {
        case Algo::kNaive:
        case Algo::kTiled:
            return is_fp32(d) ? "" : "this rung is FP32 in and FP32 out only";
        case Algo::kRegister:
        case Algo::kCpAsync:
            if (!is_fp32(d)) {
                return "this rung is FP32 in and FP32 out only";
            }
            return divides(d, 128, 128, 8)
                       ? ""
                       : "this rung needs m and n divisible by 128 and k divisible by 8";
        case Algo::kWmmaFp16:
            if (!is_fp16_in(d)) {
                return "this rung is FP16 in and FP32 out only";
            }
            return divides(d, 64, 64, 16)
                       ? ""
                       : "this rung needs m and n divisible by 64 and k divisible by 16";
        case Algo::kWmmaBf16:
            if (!is_bf16_in(d)) {
                return "this rung is BF16 in and FP32 out only";
            }
            return divides(d, 64, 64, 16)
                       ? ""
                       : "this rung needs m and n divisible by 64 and k divisible by 16";
        case Algo::kMmaPtx:
        case Algo::kMmaLdm:
            if (!is_fp16_in(d)) {
                return "this rung is FP16 in and FP32 out only";
            }
            return divides(d, 64, 64, 16)
                       ? ""
                       : "this rung needs m and n divisible by 64 and k divisible by 16";
        case Algo::kMmaOpt:
            if (!is_fp16_in(d)) {
                return "this rung is FP16 in and FP32 out only";
            }
            return divides(d, 128, 128, 32)
                       ? ""
                       : "this rung needs m and n divisible by 128 and k divisible by 32";
        case Algo::kTileFamily:
        case Algo::kSplitK:
        case Algo::kStreamK:
            // No divisibility clause: the family predicates its edges, so every
            // shape runs the same mainloop with masked tails.
            return is_fp16_in(d) ? "" : "this rung is FP16 in and FP32 out only";
        default:
            return "";
    }
}

// ---------------------------------------------------------------------------
// The tile sweep, when one is committed
// ---------------------------------------------------------------------------

// One swept shape and what each tile of the family measured on it. A tile with
// no row is absent rather than zero, so a partial sweep narrows the candidate
// set instead of silently ranking the missing shapes last.
struct SweptShape {
    double m = 0.0;
    double n = 0.0;
    double k = 0.0;
    std::vector<double> gflops;  // one per family index, negative when absent
};

// Column layout the sweep writes. Only these five are read here; the rest of the
// Section 13 fields are provenance and belong to the reader of the CSV, not to
// the dispatcher.
struct SweepColumns {
    int variant = -1;
    int m = -1;
    int n = -1;
    int k = -1;
    int gflops = -1;
};

std::vector<std::string> split_csv(const std::string& line) {
    std::vector<std::string> out;
    std::string field;
    std::istringstream in(line);
    while (std::getline(in, field, ',')) {
        out.push_back(field);
    }
    return out;
}

// "tile_128x128x32" for family index 0, which is what a sweep row's variant
// column carries.
std::string tile_variant_name(int index) {
    const GemmTile t = gemm_tile_family_shape(index);
    return "tile_" + std::to_string(t.m) + "x" + std::to_string(t.n) + "x" + std::to_string(t.k);
}

// Where the committed sweep lives. The environment variable exists so a test can
// point the dispatcher at a fixture, and so a consumer whose working directory
// is not the repository can still hand over a tuning file.
std::string sweep_path() {
    const char* env = std::getenv("CKL_TILE_SWEEP_CSV");
    if (env != nullptr && env[0] != '\0') {
        return std::string(env);
    }
    return "experiments/results/tile_sweep.csv";
}

std::vector<SweptShape> parse_sweep(const std::string& path) {
    std::vector<SweptShape> rows;
    std::ifstream in(path);
    if (!in) {
        return rows;
    }
    const int families = gemm_tile_family_count();
    std::vector<std::string> variants;
    variants.reserve(static_cast<std::size_t>(families));
    for (int i = 0; i < families; ++i) {
        variants.push_back(tile_variant_name(i));
    }

    std::string line;
    if (!std::getline(in, line)) {
        return rows;
    }
    SweepColumns col;
    const std::vector<std::string> header = split_csv(line);
    for (std::size_t i = 0; i < header.size(); ++i) {
        const int idx = static_cast<int>(i);
        if (header[i] == "variant") {
            col.variant = idx;
        } else if (header[i] == "m") {
            col.m = idx;
        } else if (header[i] == "n") {
            col.n = idx;
        } else if (header[i] == "k") {
            col.k = idx;
        } else if (header[i] == "gflops") {
            col.gflops = idx;
        }
    }
    if (col.variant < 0 || col.m < 0 || col.n < 0 || col.k < 0 || col.gflops < 0) {
        return rows;  // not a sweep this dispatcher understands
    }

    while (std::getline(in, line)) {
        const std::vector<std::string> f = split_csv(line);
        const std::size_t need = static_cast<std::size_t>(
            std::max(std::max(col.variant, col.m), std::max(std::max(col.n, col.k), col.gflops)));
        if (f.size() <= need) {
            continue;
        }
        int tile = -1;
        for (int i = 0; i < families; ++i) {
            if (f[static_cast<std::size_t>(col.variant)] == variants[static_cast<std::size_t>(i)]) {
                tile = i;
                break;
            }
        }
        if (tile < 0) {
            continue;  // some other family's row
        }
        double m = 0.0;
        double n = 0.0;
        double k = 0.0;
        double g = 0.0;
        try {
            m = std::stod(f[static_cast<std::size_t>(col.m)]);
            n = std::stod(f[static_cast<std::size_t>(col.n)]);
            k = std::stod(f[static_cast<std::size_t>(col.k)]);
            g = std::stod(f[static_cast<std::size_t>(col.gflops)]);
        } catch (const std::exception&) {
            continue;
        }
        if (!(m > 0.0) || !(n > 0.0) || !(k > 0.0) || !(g > 0.0)) {
            continue;
        }
        auto it = std::find_if(rows.begin(), rows.end(), [&](const SweptShape& s) {
            return s.m == m && s.n == n && s.k == k;
        });
        if (it == rows.end()) {
            SweptShape s;
            s.m = m;
            s.n = n;
            s.k = k;
            s.gflops.assign(static_cast<std::size_t>(families), -1.0);
            rows.push_back(s);
            it = rows.end() - 1;
        }
        it->gflops[static_cast<std::size_t>(tile)] = g;
    }
    return rows;
}

// Parsed once, then held. An absent or unreadable file is not an error: it means
// no sweep has been committed yet, and the caller is told through
// GemmPlan::tuned that the decision was made without one.
//
// The cache is keyed on the path rather than parsed unconditionally on first
// use, so pointing CKL_TILE_SWEEP_CSV somewhere else takes effect. That is a
// tuning and test facility; two threads pointing it at different files at the
// same time would invalidate the reference this hands out, which is the same
// external synchronization rule a Context carries.
const std::vector<SweptShape>& sweep_table() {
    static std::mutex guard;
    static std::string loaded_from;
    static std::vector<SweptShape> table;
    static bool loaded = false;
    const std::lock_guard<std::mutex> lock(guard);
    const std::string path = sweep_path();
    if (!loaded || path != loaded_from) {
        table = parse_sweep(path);
        loaded_from = path;
        loaded = true;
    }
    return table;
}

// Nearest swept shape in log space, so a factor of two in m counts the same as a
// factor of two in k and no dimension dominates by being large.
const SweptShape* nearest_swept(std::int64_t m, std::int64_t n, std::int64_t k) {
    const std::vector<SweptShape>& table = sweep_table();
    const SweptShape* best = nullptr;
    double best_d = 0.0;
    for (const SweptShape& s : table) {
        const double dm = std::log(static_cast<double>(m) / s.m);
        const double dn = std::log(static_cast<double>(n) / s.n);
        const double dk = std::log(static_cast<double>(k) / s.k);
        const double d = dm * dm + dn * dn + dk * dk;
        if (best == nullptr || d < best_d) {
            best = &s;
            best_d = d;
        }
    }
    return best;
}

// ---------------------------------------------------------------------------
// The 9.6 tile choice
// ---------------------------------------------------------------------------

// The wave quantization term: the fraction of the final wave that has work.
double wave_quantization(double waves) {
    if (!(waves > 0.0)) {
        return 0.0;
    }
    return waves / std::ceil(waves);
}

struct TileChoice {
    int index = -1;
    double waves = 0.0;
    int blocks_per_sm = 0;
    bool tuned = false;
};

// The 128 by 128 by 32 shape: the one the ladder has always run, and so the one
// tile in the family with measured history behind it.
constexpr int kReferenceTileM = 128;
constexpr int kReferenceTileN = 128;
constexpr int kReferenceTileK = 32;

// Found by shape rather than by index, so reordering the roster cannot silently
// point this at a different kernel.
int reference_tile_index() {
    for (int i = 0; i < gemm_tile_family_count(); ++i) {
        const GemmTile t = gemm_tile_family_shape(i);
        if (t.m == kReferenceTileM && t.n == kReferenceTileN && t.k == kReferenceTileK) {
            return i;
        }
    }
    return -1;
}

// Waves the grid needs for one tile, or a negative number when the device cannot
// host that shape at all.
double tile_waves(const GemmDesc& d, int index, int sms) {
    const int bps = gemm_tile_family_blocks_per_sm(index);
    if (bps <= 0) {
        return -1.0;
    }
    const GemmTile t = gemm_tile_family_shape(index);
    const double tiles =
        std::ceil(static_cast<double>(d.m) / t.m) * std::ceil(static_cast<double>(d.n) / t.n);
    return tiles / (static_cast<double>(sms) * bps);
}

// Above this many waves of the reference tile, the untuned path stops asking the
// quantization question and keeps the reference shape. See the guard below.
constexpr double kUntunedGuardWaves = 2.0;

TileChoice choose_tile(const Context& ctx, const GemmDesc& d) {
    TileChoice best;
    const int sms = ctx.device_properties().multiProcessorCount;
    if (sms <= 0) {
        return best;
    }
    const int families = gemm_tile_family_count();
    const SweptShape* swept = nearest_swept(d.m, d.n, d.k);

    // The throughput filter: within five percent of the best measured tile at
    // the nearest swept shape. With no sweep there is nothing to filter with.
    double bar = -1.0;
    if (swept != nullptr) {
        double top = -1.0;
        for (int i = 0; i < families; ++i) {
            top = std::max(top, swept->gflops[static_cast<std::size_t>(i)]);
        }
        if (top > 0.0) {
            bar = 0.95 * top;
        }
    }

    // The untuned guard. With no sweep the throughput term is missing entirely,
    // and wave quantization on its own prefers the smallest tile in the family,
    // because a small tile leaves the least of the final wave idle. That is a bad
    // trade wherever the grid is already many waves deep: quantization costs a
    // couple of percent there either way, while a smaller tile stages more bytes
    // per multiply add for the whole call. So above two waves of the reference
    // tile the untuned answer is the reference tile, the one shape with measured
    // history behind it. Below two waves the quantization rule decides, and below
    // one the caller escalates to split-K or stream-K. This guard exists only
    // because the throughput half of Section 9.6 has no data yet; a committed
    // sweep takes the branch below instead, and tuned stays false here either
    // way, because a guard is not a measurement.
    if (bar <= 0.0) {
        const int ref = reference_tile_index();
        const double ref_waves = ref >= 0 ? tile_waves(d, ref, sms) : -1.0;
        if (ref_waves >= kUntunedGuardWaves) {
            best.index = ref;
            best.waves = ref_waves;
            best.blocks_per_sm = gemm_tile_family_blocks_per_sm(ref);
            best.tuned = false;
            return best;
        }
    }

    double best_score = -1.0;
    long long best_area = -1;
    for (int i = 0; i < families; ++i) {
        const int bps = gemm_tile_family_blocks_per_sm(i);
        if (bps <= 0) {
            continue;  // this device cannot host the shape at all
        }
        if (bar > 0.0 && swept->gflops[static_cast<std::size_t>(i)] < bar) {
            continue;
        }
        const GemmTile t = gemm_tile_family_shape(i);
        const double waves = tile_waves(d, i, sms);
        const double score = wave_quantization(waves);
        const long long area = static_cast<long long>(t.m) * t.n;
        if (score > best_score || (score == best_score && area > best_area)) {
            best_score = score;
            best_area = area;
            best.index = i;
            best.waves = waves;
            best.blocks_per_sm = bps;
            best.tuned = bar > 0.0;
        }
    }
    return best;
}

// A K split is worth its second pass over C only when each slice still holds
// enough K steps for the three stage pipeline to fill. Below that, stream-K
// spreads the same work without the extra pass, which is why the escalation
// asks this question rather than always splitting.
constexpr int kMinStepsPerSplit = 4;

int split_count_for(const GemmDesc& d, const TileChoice& choice, int sms) {
    if (choice.index < 0 || d.k <= 0) {
        return 1;
    }
    const GemmTile t = gemm_tile_family_shape(choice.index);
    const double tiles =
        std::ceil(static_cast<double>(d.m) / t.m) * std::ceil(static_cast<double>(d.n) / t.n);
    if (!(tiles > 0.0)) {
        return 1;
    }
    const long long ctas = static_cast<long long>(sms) * choice.blocks_per_sm;
    const long long by_occupancy = static_cast<long long>(static_cast<double>(ctas) / tiles);
    const long long steps = (d.k + t.k - 1) / t.k;
    const long long by_depth = steps / kMinStepsPerSplit;
    const long long want = std::min(by_occupancy, by_depth);
    if (want < 2) {
        return 1;
    }
    // The rounding to whole K steps can still collapse the request to one slice,
    // and one slice is not split-K. Reporting the number the driver will really
    // use is what keeps GemmPlan::splits from being a guess.
    const int slices =
        gemm_split_k_slices(static_cast<int>(d.k), static_cast<int>(want), choice.index);
    return slices >= 2 ? slices : 1;
}

GemmPlan plan_tile_family(const Context& ctx, const GemmDesc& d) {
    GemmPlan plan;
    const TileChoice choice = choose_tile(ctx, d);
    if (choice.index < 0) {
        plan.algo = Algo::kCublas;
        return plan;
    }
    plan.tile_index = choice.index;
    plan.tile = gemm_tile_family_shape(choice.index);
    plan.tuned = choice.tuned;
    if (choice.waves >= 1.0) {
        plan.algo = Algo::kTileFamily;
        return plan;
    }
    // waves below one: the grid does not fill a single wave, so the K dimension
    // has to supply the missing parallelism.
    const int splits = split_count_for(d, choice, ctx.device_properties().multiProcessorCount);
    if (splits >= 2) {
        plan.algo = Algo::kSplitK;
        plan.splits = splits;
    } else {
        plan.algo = Algo::kStreamK;
    }
    return plan;
}

// ---------------------------------------------------------------------------
// The kAuto heuristic
// ---------------------------------------------------------------------------

GemmPlan pick_auto(const Context& ctx, const GemmDesc& d) {
    GemmPlan plan;
    plan.algo = Algo::kCublas;
    if (!hand_kernel_shape(d) || d.k <= 0) {
        return plan;
    }
    if (is_fp16_in(d)) {
        // Every FP16 shape the hand kernels can address goes to the family: the
        // predicated edges mean there is no shape it refuses, so there is no
        // fallback cliff to fall off.
        return plan_tile_family(ctx, d);
    }
    if (is_bf16_in(d)) {
        if (divides(d, 64, 64, 16)) {
            plan.algo = Algo::kWmmaBf16;
        }
        return plan;
    }
    if (is_fp32(d)) {
        // Large enough that the double buffered kernel's staging pays for
        // itself; below that its 128 by 128 blocks quantize badly on 48 SMs.
        const bool large = d.m >= 512 && d.n >= 512 && d.k >= 512;
        if (large && divides(d, 128, 128, 8)) {
            plan.algo = Algo::kCpAsync;
            return plan;
        }
        // One block per output tile beats a launch into cuBLAS's heuristics at
        // this size, and it keeps the whole call on one stream.
        const bool small = d.m <= 128 && d.n <= 128 && d.k <= 128;
        if (small) {
            plan.algo = Algo::kNaive;
        }
        return plan;
    }
    return plan;
}

// The plan for an explicitly named algorithm. A named rung is never rerouted, so
// this only fills in the detail the caller did not name: which tile of the
// family runs, and how many K slices.
GemmPlan plan_named(const Context& ctx, const GemmDesc& d, Algo named) {
    GemmPlan plan;
    plan.algo = named;
    if (!in_tile_family(named) || !hand_kernel_shape(d) || !is_fp16_in(d)) {
        return plan;
    }
    const TileChoice choice = choose_tile(ctx, d);
    if (choice.index < 0) {
        return plan;
    }
    plan.tile_index = choice.index;
    plan.tile = gemm_tile_family_shape(choice.index);
    plan.tuned = choice.tuned;
    if (named == Algo::kSplitK) {
        const int sms = ctx.device_properties().multiProcessorCount;
        const int wanted = split_count_for(d, choice, sms);
        // The caller asked for split-K on a shape the heuristic would not have
        // split. Two slices is the least that is still split-K, and the driver
        // reports back how many it could actually use.
        plan.splits =
            wanted >= 2 ? wanted : gemm_split_k_slices(static_cast<int>(d.k), 2, choice.index);
    }
    return plan;
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

Status validate(const GemmDesc& d, const void* alpha, const void* a, const void* b,
                const void* beta, const void* c) {
    if (d.m < 0 || d.n < 0 || d.k < 0) {
        detail::set_last_error("gemm: m, n and k must all be non negative");
        return Status::kInvalidValue;
    }
    if (d.batch_count < 1) {
        detail::set_last_error("gemm: batch_count must be at least 1");
        return Status::kInvalidValue;
    }
    if (alpha == nullptr || beta == nullptr) {
        detail::set_last_error("gemm: alpha and beta are host pointers and must not be null");
        return Status::kInvalidValue;
    }
    const std::int64_t need_a = min_ld(d.layout, stored_a(d));
    const std::int64_t need_b = min_ld(d.layout, stored_b(d));
    const std::int64_t need_c = min_ld(d.layout, Stored{d.m, d.n});
    if (d.lda < need_a || d.ldb < need_b || d.ldc < need_c) {
        detail::set_last_error(
            "gemm: leading dimensions too small for the described shape, need "
            "lda at least " +
            std::to_string(need_a) + ", ldb at least " + std::to_string(need_b) +
            ", ldc at least " + std::to_string(need_c));
        return Status::kInvalidValue;
    }
    if (d.m > 0 && d.n > 0) {
        if (c == nullptr) {
            detail::set_last_error("gemm: c is null for a non empty output");
            return Status::kInvalidValue;
        }
        if (d.k > 0 && (a == nullptr || b == nullptr)) {
            detail::set_last_error("gemm: a or b is null for a non empty contraction");
            return Status::kInvalidValue;
        }
    }
    if (d.dt_c != DType::kR32F) {
        detail::set_last_error("gemm: the only output type in 1.1.0 is 32 bit float");
        return Status::kNotSupported;
    }
    if (d.dt_a != d.dt_b) {
        detail::set_last_error("gemm: mixed input types are not supported");
        return Status::kNotSupported;
    }
    return Status::kSuccess;
}

// ---------------------------------------------------------------------------
// cuBLAS path
// ---------------------------------------------------------------------------

cublasOperation_t to_cublas(Op o) {
    // kC is a plain transpose here: every type the library ships is real.
    return o == Op::kN ? CUBLAS_OP_N : CUBLAS_OP_T;
}

cudaDataType to_cuda_dtype(DType t) {
    switch (t) {
        case DType::kR16F:
            return CUDA_R_16F;
        case DType::kR16BF:
            return CUDA_R_16BF;
        case DType::kR32F:
        default:
            return CUDA_R_32F;
    }
}

void check_cublas(cublasStatus_t s, const char* expr) {
    if (s != CUBLAS_STATUS_SUCCESS) {
        Status mapped = Status::kExecutionFailed;
        switch (s) {
            case CUBLAS_STATUS_NOT_INITIALIZED:
                mapped = Status::kNotInitialized;
                break;
            case CUBLAS_STATUS_ALLOC_FAILED:
                mapped = Status::kAllocFailed;
                break;
            case CUBLAS_STATUS_INVALID_VALUE:
                mapped = Status::kInvalidValue;
                break;
            case CUBLAS_STATUS_ARCH_MISMATCH:
                mapped = Status::kArchMismatch;
                break;
            case CUBLAS_STATUS_NOT_SUPPORTED:
                mapped = Status::kNotSupported;
                break;
            default:
                break;
        }
        throw Error(mapped, std::string("cuBLAS error ") + cublasGetStatusName(s) + ": " + expr);
    }
}

// C = beta * C for an empty contraction. beta zero has to clear C rather than
// scale it, because a NaN in C is legal input when beta is zero and 0 * NaN is
// still NaN.
void scale_c(cublasHandle_t h, const GemmDesc& d, float beta, void* c, cudaStream_t stream) {
    if (beta == 1.0f) {
        return;
    }
    const std::int64_t lead = d.layout == Layout::kRowMajor ? d.n : d.m;
    const std::int64_t other = d.layout == Layout::kRowMajor ? d.m : d.n;
    auto* base = static_cast<float*>(c);
    if (beta == 0.0f) {
        CKL_CUDA_CHECK(cudaMemset2DAsync(base, static_cast<std::size_t>(d.ldc) * sizeof(float), 0,
                                         static_cast<std::size_t>(lead) * sizeof(float),
                                         static_cast<std::size_t>(other), stream));
        return;
    }
    for (std::int64_t i = 0; i < other; ++i) {
        check_cublas(cublasSscal_64(h, lead, &beta, base + i * d.ldc, 1), "cublasSscal_64");
    }
}

void run_cublas(const Context& ctx, const GemmDesc& d, const void* alpha, const void* a,
                const void* b, const void* beta, void* c) {
    auto* h = static_cast<cublasHandle_t>(ctx.cublas());
    check_cublas(cublasSetStream(h, ctx.stream()), "cublasSetStream");

    if (d.k == 0) {
        scale_c(h, d, *static_cast<const float*>(beta), c, ctx.stream());
        return;
    }

    const cublasOperation_t opa = to_cublas(d.op_a);
    const cublasOperation_t opb = to_cublas(d.op_b);
    const cudaDataType ta = to_cuda_dtype(d.dt_a);
    const cudaDataType tb = to_cuda_dtype(d.dt_b);
    const cudaDataType tc = to_cuda_dtype(d.dt_c);

    if (d.batch_count > 1) {
        if (d.layout == Layout::kRowMajor) {
            check_cublas(cublasGemmStridedBatchedEx_64(
                             h, opb, opa, d.n, d.m, d.k, alpha, b, tb, d.ldb, d.stride_b, a, ta,
                             d.lda, d.stride_a, beta, c, tc, d.ldc, d.stride_c, d.batch_count,
                             CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
                         "cublasGemmStridedBatchedEx_64");
        } else {
            check_cublas(cublasGemmStridedBatchedEx_64(
                             h, opa, opb, d.m, d.n, d.k, alpha, a, ta, d.lda, d.stride_a, b, tb,
                             d.ldb, d.stride_b, beta, c, tc, d.ldc, d.stride_c, d.batch_count,
                             CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
                         "cublasGemmStridedBatchedEx_64");
        }
        return;
    }

    // Row major goes through the transpose identity: a row major buffer read
    // column major is its own transpose, so asking for C_t = op(B)_t * op(A)_t
    // with m and n swapped writes exactly the row major C we want, and each
    // operand keeps its own op flag.
    if (d.layout == Layout::kRowMajor) {
        check_cublas(cublasGemmEx_64(h, opb, opa, d.n, d.m, d.k, alpha, b, tb, d.ldb, a, ta, d.lda,
                                     beta, c, tc, d.ldc, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
                     "cublasGemmEx_64");
    } else {
        check_cublas(cublasGemmEx_64(h, opa, opb, d.m, d.n, d.k, alpha, a, ta, d.lda, b, tb, d.ldb,
                                     beta, c, tc, d.ldc, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
                     "cublasGemmEx_64");
    }
}

// ---------------------------------------------------------------------------
// Hand written kernel path
// ---------------------------------------------------------------------------

void run_hand_kernel(const Context& ctx, const GemmPlan& plan, const GemmDesc& d, float alpha,
                     const void* pa, const void* pb, float beta, void* pc, cudaStream_t stream) {
    const Algo a = plan.algo;
    const int m = static_cast<int>(d.m);
    const int n = static_cast<int>(d.n);
    const int k = static_cast<int>(d.k);
    const auto* f32a = static_cast<const float*>(pa);
    const auto* f32b = static_cast<const float*>(pb);
    const auto* f16a = static_cast<const __half*>(pa);
    const auto* f16b = static_cast<const __half*>(pb);
    const auto* bf16a = static_cast<const __nv_bfloat16*>(pa);
    const auto* bf16b = static_cast<const __nv_bfloat16*>(pb);
    auto* out = static_cast<float*>(pc);

    switch (a) {
        case Algo::kNaive:
            gemm_naive(f32a, f32b, out, m, n, k, alpha, beta, stream);
            return;
        case Algo::kTiled:
            gemm_tiled(f32a, f32b, out, m, n, k, alpha, beta, stream);
            return;
        case Algo::kRegister:
            gemm_register(f32a, f32b, out, m, n, k, alpha, beta, stream);
            return;
        case Algo::kCpAsync:
            gemm_cp_async(f32a, f32b, out, m, n, k, alpha, beta, stream);
            return;
        case Algo::kWmmaFp16:
            gemm_wmma_fp16(f16a, f16b, out, m, n, k, alpha, beta, stream);
            return;
        case Algo::kWmmaBf16:
            gemm_wmma_bf16(bf16a, bf16b, out, m, n, k, alpha, beta, stream);
            return;
        case Algo::kMmaPtx:
            gemm_mma_ptx(f16a, f16b, out, m, n, k, alpha, beta, stream);
            return;
        case Algo::kMmaLdm:
            gemm_mma_ldm(f16a, f16b, out, m, n, k, alpha, beta, stream);
            return;
        case Algo::kMmaOpt:
            gemm_mma_opt(f16a, f16b, out, m, n, k, alpha, beta, stream);
            return;
        case Algo::kTileFamily:
            gemm_tile_family(f16a, f16b, out, m, n, k, alpha, beta, plan.tile_index, stream);
            return;
        case Algo::kSplitK: {
            // A Context workspace big enough for the shape keeps the allocation
            // out of the call; anything else and the driver takes its own for
            // the duration and says so through gemm_workspace_size.
            const std::size_t need =
                gemm_split_k_workspace_size(m, n, k, plan.splits, plan.tile_index);
            const bool usable = ctx.workspace() != nullptr && ctx.workspace_size() >= need;
            gemm_split_k(f16a, f16b, out, m, n, k, alpha, beta, plan.splits, plan.tile_index,
                         usable ? ctx.workspace() : nullptr,
                         usable ? ctx.workspace_size() : std::size_t{0}, stream);
            return;
        }
        case Algo::kStreamK: {
            const std::size_t need = gemm_stream_k_workspace_size(m, n, k, plan.tile_index);
            const bool usable = ctx.workspace() != nullptr && ctx.workspace_size() >= need;
            gemm_stream_k(f16a, f16b, out, m, n, k, alpha, beta, plan.tile_index,
                          usable ? ctx.workspace() : nullptr,
                          usable ? ctx.workspace_size() : std::size_t{0}, stream);
            return;
        }
        default:
            throw Error(Status::kInternal, "dispatch reached a hand kernel case it does not own");
    }
}

}  // namespace

Algo gemm_query(const Context& ctx, const GemmDesc& desc) {
    // The short answer is the long one with the tile detail dropped. Two
    // separate heuristics would be two things to keep in step.
    return gemm_plan(ctx, desc).algo;
}

GemmPlan gemm_plan(const Context& ctx, const GemmDesc& desc) {
    // The heuristic is shape driven, but a device that cannot run the chosen
    // rung would make the answer a lie, so the architecture gate applies here
    // too and the answer degrades to the vendor path.
    GemmPlan plan = pick_auto(ctx, desc);
    if (ctx.compute_capability() < required_cc(plan.algo)) {
        GemmPlan vendor;
        vendor.algo = Algo::kCublas;  // kAuto is never an answer, only a request
        return vendor;
    }
    return plan;
}

std::size_t gemm_workspace_size(const Context& ctx, const GemmDesc& desc) {
    const GemmPlan plan =
        desc.algo == Algo::kAuto ? gemm_plan(ctx, desc) : plan_named(ctx, desc, desc.algo);
    if (desc.m <= 0 || desc.n <= 0 || desc.k <= 0 || plan.tile_index < 0) {
        return 0;
    }
    const int m = static_cast<int>(desc.m);
    const int n = static_cast<int>(desc.n);
    const int k = static_cast<int>(desc.k);
    switch (plan.algo) {
        case Algo::kSplitK:
            return gemm_split_k_workspace_size(m, n, k, plan.splits, plan.tile_index);
        case Algo::kStreamK:
            return gemm_stream_k_workspace_size(m, n, k, plan.tile_index);
        default:
            // The data parallel paths run out of registers and shared memory only.
            return 0;
    }
}

Status gemm(Context& ctx, const GemmDesc& desc, const void* alpha, const void* a, const void* b,
            const void* beta, void* c, Algo* chosen) {
    detail::clear_last_error();

    // Written before any early return, so a caller reading it after a failure
    // still sees what the dispatcher was aiming at.
    GemmPlan plan =
        desc.algo == Algo::kAuto ? pick_auto(ctx, desc) : plan_named(ctx, desc, desc.algo);
    Algo picked = plan.algo;
    if (chosen != nullptr) {
        *chosen = picked;
    }

    const Status valid = validate(desc, alpha, a, b, beta, c);
    if (valid != Status::kSuccess) {
        return valid;
    }

    if (picked == Algo::kCutlass) {
        detail::set_last_error(std::string("gemm: algorithm ") + algo_name(picked) +
                               " is declared for ABI stability but not implemented in 1.1.0");
        return Status::kNotSupported;
    }

    // Strided batched work goes to cuBLAS. Batching the hand kernels is future
    // work; looping them here would be a silent reinterpretation of the call.
    if (desc.batch_count > 1 && picked != Algo::kCublas) {
        if (desc.algo == Algo::kAuto) {
            picked = Algo::kCublas;
            plan = GemmPlan{};
            plan.algo = picked;
            if (chosen != nullptr) {
                *chosen = picked;
            }
        } else {
            detail::set_last_error(std::string("gemm: algorithm ") + algo_name(picked) +
                                   " has no strided batched form; use kAuto or kCublas");
            return Status::kNotSupported;
        }
    }

    if (picked != Algo::kCublas) {
        const std::string reason = why_not(picked, desc);
        if (!reason.empty()) {
            detail::set_last_error(std::string("gemm: algorithm ") + algo_name(picked) +
                                   " cannot take this shape: " + reason);
            return Status::kNotSupported;
        }
        const int need = required_cc(picked);
        if (ctx.compute_capability() < need) {
            // kAuto is allowed to choose again, and chosen reports where it
            // landed. An explicitly named algorithm is not rerouted: the caller
            // asked for that path and gets told the device cannot run it.
            if (desc.algo == Algo::kAuto) {
                picked = Algo::kCublas;
                plan = GemmPlan{};
                plan.algo = picked;
                if (chosen != nullptr) {
                    *chosen = picked;
                }
            } else {
                detail::set_last_error(std::string("gemm: algorithm ") + algo_name(picked) +
                                       " needs compute capability " + std::to_string(need / 10) +
                                       "." + std::to_string(need % 10) + "; this device reports " +
                                       std::to_string(ctx.compute_capability() / 10) + "." +
                                       std::to_string(ctx.compute_capability() % 10));
                return Status::kArchMismatch;
            }
        }
    }

    if (desc.m == 0 || desc.n == 0) {
        return Status::kSuccess;  // no output elements, nothing to write
    }

    try {
        if (picked == Algo::kCublas) {
            run_cublas(ctx, desc, alpha, a, b, beta, c);
        } else if (desc.k == 0) {
            // An empty contraction is C = beta * C. The hand kernels that carry
            // a beta only path handle it, but routing it through one place keeps
            // the zero beta clearing rule in a single implementation.
            auto* h = static_cast<cublasHandle_t>(ctx.cublas());
            check_cublas(cublasSetStream(h, ctx.stream()), "cublasSetStream");
            scale_c(h, desc, *static_cast<const float*>(beta), c, ctx.stream());
        } else {
            run_hand_kernel(ctx, plan, desc, *static_cast<const float*>(alpha), a, b,
                            *static_cast<const float*>(beta), c, ctx.stream());
            CKL_CUDA_LAST_ERROR(false);
        }
    } catch (const Error& e) {
        detail::set_last_error(e.what());
        return e.status();
    } catch (const std::exception& e) {
        detail::set_last_error(e.what());
        return Status::kInternal;
    }

    return Status::kSuccess;
}

}  // namespace ckl
