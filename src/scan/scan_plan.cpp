// The plan, the names, the tolerance model, and the two launchers the dispatcher
// calls.
//
// Host only. It owns every allocation the family makes and it makes them once:
// the per block partials, the level buffers the scan then propagate recursion
// carves its aggregate arrays out of, the tile status array and the two tile
// counters, and the CUB temporary storage. Running a rung through the plan
// enqueues launches and nothing else, which is the claim
// benchmarks/support/allocation_gate.hpp exists to check and the claim v1's
// SpMV wrapper got wrong.
//
// It is also where the two runtime enums turn into template arguments. The rungs
// are templates over the element type and the operator; the dispatch below is
// the one place that switch is written, and the two instantiation sets it
// switches over are the ones in scan_device.cuh, so a rung and its dispatch
// cannot drift apart without a link error.

#include "ckl/scan.hpp"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>

#include <cuda_runtime.h>

#include "ckl/cuda_check.hpp"
#include "ckl/device_buffer.hpp"
#include "ckl/status.hpp"

#include "scan_internal.hpp"
#include "scan_rungs.hpp"

namespace ckl {

namespace {

// The two calibrated constants of the tolerance model. See docs/scan.md for the
// calibration: tests/test_scan.cpp holds a disabled-by-default case that sweeps
// seeds, lengths, datasets and rungs and reports the worst observed ratio for
// each term, and these values are those maxima with deliberate slack over them
// and no more. A value large enough to pass anything would be a gate that cannot
// fail.
//
// kToleranceC scales the random walk term, which is what a tree combine of
// signed data produces: the partial sums reach about sqrt(N) times the input
// scale and the rounding walks with them.
//
// kMagnitudeC scales the accumulated magnitude term, which is what the two
// single pass rungs add on top: kLookback and kCub carry a tile to tile prefix
// chain whose length grows with N, and the rounding along that chain is bounded
// by the accumulated magnitude sum_j |x_j| rather than by sqrt(N). The first
// campaign found the missing term at 2^27 and 2^28, where a sqrt(N) only model
// is outgrown by a factor of two.
constexpr double kToleranceC = 12.0;
constexpr double kMagnitudeC = 0.01;

// The narrowest tile any rung uses. The level buffer has to be sized for it,
// because a narrower tile means more tiles at the first level and one more level
// underneath.
constexpr int kNarrowTile = 256;

// The deterministic rungs launch this many blocks, or fewer when the input
// cannot fill them. Both branches are functions of the length alone.
constexpr long long kDeterministicBlocks = 1024;
constexpr long long kDeterministicBlockSize = 256;

bool deterministic_from_environment() {
    const char* value = std::getenv("CKL_REDUCE_DETERMINISTIC");
    if (value == nullptr || value[0] == '\0') {
        return false;
    }
    return std::strcmp(value, "0") != 0;
}

}  // namespace

const char* reduce_algo_name(ReduceAlgo a) {
    switch (a) {
        case ReduceAlgo::kAuto:
            return "kAuto";
        case ReduceAlgo::kAtomic:
            return "kAtomic";
        case ReduceAlgo::kSharedTree:
            return "kSharedTree";
        case ReduceAlgo::kShuffle:
            return "kShuffle";
        case ReduceAlgo::kVec4:
            return "kVec4";
        case ReduceAlgo::kSinglePass:
            return "kSinglePass";
        case ReduceAlgo::kTwoPass:
            return "kTwoPass";
        case ReduceAlgo::kDeterministic:
            return "kDeterministic";
        case ReduceAlgo::kKahan:
            return "kKahan";
        case ReduceAlgo::kCub:
            return "kCub";
    }
    return "kUnknown";
}

const char* scan_algo_name(ScanAlgo a) {
    switch (a) {
        case ScanAlgo::kAuto:
            return "kAuto";
        case ScanAlgo::kHillisSteele:
            return "kHillisSteele";
        case ScanAlgo::kBlelloch:
            return "kBlelloch";
        case ScanAlgo::kThreeKernel:
            return "kThreeKernel";
        case ScanAlgo::kReduceThenScan:
            return "kReduceThenScan";
        case ScanAlgo::kLookback:
            return "kLookback";
        case ScanAlgo::kDeterministic:
            return "kDeterministic";
        case ScanAlgo::kCub:
            return "kCub";
    }
    return "kUnknown";
}

const char* scan_op_name(ScanOp op) {
    switch (op) {
        case ScanOp::kSum:
            return "kSum";
        case ScanOp::kMax:
            return "kMax";
        case ScanOp::kMin:
            return "kMin";
        case ScanOp::kLastNonZero:
            return "kLastNonZero";
    }
    return "kUnknown";
}

const char* scan_dtype_name(ScanDType t) {
    switch (t) {
        case ScanDType::kF32:
            return "kF32";
        case ScanDType::kF64:
            return "kF64";
        case ScanDType::kI32:
            return "kI32";
        case ScanDType::kI64:
            return "kI64";
    }
    return "kUnknown";
}

int scan_dtype_size(ScanDType t) {
    switch (t) {
        case ScanDType::kF64:
        case ScanDType::kI64:
            return 8;
        case ScanDType::kF32:
        case ScanDType::kI32:
        default:
            return 4;
    }
}

double scan_tolerance_c() {
    return kToleranceC;
}

double scan_magnitude_c() {
    return kMagnitudeC;
}

double scan_tolerance(long long n) {
    const double count = n > 1 ? static_cast<double>(n) : 1.0;
    return kToleranceC * std::sqrt(count) * static_cast<double>(FLT_EPSILON);
}

double scan_tolerance(long long n, double magnitude_ratio) {
    const double count = n > 1 ? static_cast<double>(n) : 1.0;
    const double ratio = magnitude_ratio > 0.0 ? magnitude_ratio : 0.0;
    return (kToleranceC * std::sqrt(count) + kMagnitudeC * ratio) *
           static_cast<double>(FLT_EPSILON);
}

// ---------------------------------------------------------------------------
// The plan
// ---------------------------------------------------------------------------

struct ScanPlan::Impl {
    long long n = 0;
    ScanDType dtype = ScanDType::kF32;
    int elem = 4;
    bool deterministic = false;
    bool cub_ready = false;

    int persistent_blocks = 1;
    int deterministic_blocks = 1;
    int wide_tile = 2048;
    long long level_capacity = 0;
    long long lookback_tiles = 0;
    std::size_t status_bytes_per_tile = 0;
    std::size_t cub_bytes = 0;

    DeviceBuffer<unsigned char> partials;
    DeviceBuffer<unsigned char> levels;
    DeviceBuffer<unsigned char> status;
    DeviceBuffer<unsigned int> counters;  // [0] single pass reduction, [1] look-back
    DeviceBuffer<unsigned char> cub_temp;
};

ScanPlan::ScanPlan(long long n, ScanDType dtype, const ScanPlanOptions& opt) : impl_(new Impl()) {
    if (n < 0) {
        throw Error(Status::kInvalidValue, "ScanPlan: the length must not be negative");
    }
    Impl& s = *impl_;
    s.n = n;
    s.dtype = dtype;
    s.elem = scan_dtype_size(dtype);
    s.deterministic = opt.deterministic || deterministic_from_environment();
    s.wide_tile = detail::scan_tile_elements(ScanAlgo::kThreeKernel, dtype);

    if (n == 0) {
        // An empty plan still answers for the baseline: CUB wants no temporary
        // storage for nothing, and refusing the rung here would make an empty
        // input the one length at which the baseline is missing.
        s.persistent_blocks = 1;
        s.deterministic_blocks = 1;
        s.cub_ready = opt.build_cub;
        return;
    }

    s.persistent_blocks = detail::reduce_persistent_blocks(dtype);
    const long long det_tiles = (n + kDeterministicBlockSize - 1) / kDeterministicBlockSize;
    s.deterministic_blocks = static_cast<int>(std::min<long long>(kDeterministicBlocks, det_tiles));

    // Partials. The shared memory tree rung writes one per 256 elements, which is
    // the widest demand; the compensated rung writes two per persistent block.
    const long long tree_partials = (n + 255) / 256;
    const long long need_partials =
        std::max<long long>({tree_partials, 2LL * s.persistent_blocks, s.deterministic_blocks});
    s.partials = DeviceBuffer<unsigned char>(static_cast<std::size_t>(need_partials) *
                                             static_cast<std::size_t>(s.elem));

    // Level buffers, sized for the narrowest tile because that is the rung with
    // the most levels and the largest first level.
    s.level_capacity = detail::scan_levels_capacity(n, kNarrowTile);
    if (s.level_capacity > 0) {
        s.levels = DeviceBuffer<unsigned char>(static_cast<std::size_t>(s.level_capacity) *
                                               static_cast<std::size_t>(s.elem));
    }

    // Look-back status, one record per tile of the wide decomposition.
    s.lookback_tiles = (n + s.wide_tile - 1) / s.wide_tile;
    s.status_bytes_per_tile = detail::lookback_status_bytes_per_tile(dtype);
    s.status = DeviceBuffer<unsigned char>(static_cast<std::size_t>(s.lookback_tiles) *
                                           s.status_bytes_per_tile);

    s.counters = DeviceBuffer<unsigned int>(2);
    s.counters.zero();

    if (opt.build_cub) {
        s.cub_bytes = detail::cub_temp_bytes(dtype, n);
        if (s.cub_bytes > 0) {
            s.cub_temp = DeviceBuffer<unsigned char>(s.cub_bytes);
        }
        s.cub_ready = true;
    }
}

ScanPlan::~ScanPlan() = default;
ScanPlan::ScanPlan(ScanPlan&& other) noexcept = default;
ScanPlan& ScanPlan::operator=(ScanPlan&& other) noexcept = default;

ScanPlan::Impl& ScanPlan::live() const {
    if (!impl_) {
        throw Error(Status::kInvalidValue, "ScanPlan: this plan has been moved from");
    }
    return *impl_;
}

long long ScanPlan::size() const {
    return live().n;
}

ScanDType ScanPlan::dtype() const {
    return live().dtype;
}

bool ScanPlan::deterministic() const {
    return live().deterministic;
}

int ScanPlan::tile_elements(ScanAlgo a) const {
    return detail::scan_tile_elements(a, live().dtype);
}

long long ScanPlan::tiles(ScanAlgo a) const {
    const Impl& s = live();
    if (s.n <= 0) {
        return 0;
    }
    const long long tile = tile_elements(a);
    return (s.n + tile - 1) / tile;
}

int ScanPlan::deterministic_blocks() const {
    return live().deterministic_blocks;
}

int ScanPlan::persistent_blocks() const {
    return live().persistent_blocks;
}

long long ScanPlan::lookback_status_bytes() const {
    const Impl& s = live();
    // Written once by the memset ahead of the launch, written again by each tile
    // as it publishes, and read by the tiles that look back at it. Two writes and
    // one read of every record is the model; the actual number of look-back reads
    // depends on how far each tile has to walk, which is what the rung's whole
    // behaviour is about and not something a static model can carry.
    return 3 * s.lookback_tiles * static_cast<long long>(s.status_bytes_per_tile);
}

long long ScanPlan::reduce_model_bytes(ReduceAlgo a) const {
    const Impl& s = live();
    if (s.n <= 0) {
        return 0;
    }
    const long long body = s.n * s.elem;
    switch (a) {
        case ReduceAlgo::kSharedTree: {
            // One partial per 256 elements, written and read back.
            const long long partials = 2 * ((s.n + 255) / 256) * s.elem;
            return body + partials;
        }
        case ReduceAlgo::kVec4:
        case ReduceAlgo::kTwoPass:
        case ReduceAlgo::kSinglePass:
        case ReduceAlgo::kKahan: {
            const long long partials = 2LL * s.persistent_blocks * s.elem;
            return body + partials;
        }
        case ReduceAlgo::kDeterministic: {
            const long long partials = 2LL * s.deterministic_blocks * s.elem;
            return body + partials;
        }
        default:
            // r0, r2 and CUB read the input once and write a single value.
            return body;
    }
}

long long ScanPlan::scan_model_bytes(ScanAlgo a) const {
    const Impl& s = live();
    if (s.n <= 0) {
        return 0;
    }
    const long long body = s.n * s.elem;
    switch (a) {
        case ScanAlgo::kHillisSteele:
        case ScanAlgo::kBlelloch:
        case ScanAlgo::kThreeKernel:
            // Read, write, read again, write again.
            return 4 * body;
        case ScanAlgo::kReduceThenScan:
        case ScanAlgo::kDeterministic:
            // Read, read again, write once.
            return 3 * body;
        case ScanAlgo::kLookback:
            return 2 * body + lookback_status_bytes();
        case ScanAlgo::kCub:
        default:
            // CUB implements decoupled look-back too, so the model is the same
            // read once, write once, plus its own tile state traffic, which is
            // not ours to model. The two body passes are what is declared.
            return 2 * body;
    }
}

ReduceAlgo ScanPlan::query_reduce() const {
    if (live().deterministic) {
        return ReduceAlgo::kDeterministic;
    }
    // r4 is r3's loop with one launch instead of two, so it is never the worse
    // of the pair and it is a whole launch cheaper below the size where the
    // kernel outruns the launch. This rule is provisional: there is no committed
    // sweep for this family yet and docs/scan.md says so.
    return ReduceAlgo::kSinglePass;
}

ScanAlgo ScanPlan::query_scan() const {
    const Impl& s = live();
    if (s.deterministic) {
        return ScanAlgo::kDeterministic;
    }
    if (s.n <= s.wide_tile) {
        // One tile covers the input, so there is nothing to look back at and the
        // status memset would be the only thing the rung added.
        return ScanAlgo::kThreeKernel;
    }
    return ScanAlgo::kLookback;
}

std::string ScanPlan::reduce_refusal(ReduceAlgo a, ScanOp op) const {
    const Impl& s = live();
    if (a == ReduceAlgo::kAuto) {
        return "reduce: kAuto is a request, not a rung; the plan resolves it before this point";
    }
    if (op == ScanOp::kLastNonZero) {
        return "the reduction ladder combines its block partials in an order the launch decides, "
               "so a non commutative operator has no honest rung here; ckl::scan carries "
               "kLastNonZero on every rung of the scan ladder";
    }
    if (s.dtype != ScanDType::kF32 && op != ScanOp::kSum) {
        return std::string(
                   "this release instantiates the operator ladder for 32 bit float only; ") +
               scan_dtype_name(s.dtype) +
               " carries kSum, which is what recomputing the roof "
               "per element type needs";
    }
    if (a == ReduceAlgo::kKahan && op != ScanOp::kSum) {
        return "compensated summation compensates a rounding error, and only addition has one; "
               "the other operators are exact";
    }
    if (a == ReduceAlgo::kCub && !s.cub_ready) {
        return "this plan was built with ScanPlanOptions::build_cub off, so it holds no CUB "
               "temporary storage and the baseline cannot run out of it";
    }
    return std::string();
}

std::string ScanPlan::scan_refusal(ScanAlgo a, ScanOp op) const {
    const Impl& s = live();
    if (a == ScanAlgo::kAuto) {
        return "scan: kAuto is a request, not a rung; the plan resolves it before this point";
    }
    if (s.dtype != ScanDType::kF32 && op != ScanOp::kSum) {
        return std::string(
                   "this release instantiates the operator ladder for 32 bit float only; ") +
               scan_dtype_name(s.dtype) +
               " carries kSum, which is what recomputing the roof "
               "per element type needs";
    }
    if (a == ScanAlgo::kCub && !s.cub_ready) {
        return "this plan was built with ScanPlanOptions::build_cub off, so it holds no CUB "
               "temporary storage and the baseline cannot run out of it";
    }
    return std::string();
}

bool ScanPlan::supports_reduce(ReduceAlgo a, ScanOp op) const {
    return reduce_refusal(a, op).empty();
}

bool ScanPlan::supports_scan(ScanAlgo a, ScanOp op) const {
    return scan_refusal(a, op).empty();
}

int ScanPlan::cub_version() {
    return detail::cub_version_macro();
}

// ---------------------------------------------------------------------------
// The launchers
// ---------------------------------------------------------------------------
//
// One switch over the element type and one over the operator, written once. The
// two instantiation sets differ, so the reduction switch and the scan switch are
// separate macros rather than one macro with a hole in it: a rung that is not
// instantiated for a pair must not be reachable from the dispatch at all, or the
// link would fail rather than the call being refused.

namespace {

#define CKL_REDUCE_OVER_OP(T, BODY)                                                   \
    switch (op) {                                                                     \
        case ScanOp::kSum:                                                            \
            BODY(T, ScanOp::kSum);                                                    \
            break;                                                                    \
        case ScanOp::kMax:                                                            \
            BODY(T, ScanOp::kMax);                                                    \
            break;                                                                    \
        case ScanOp::kMin:                                                            \
            BODY(T, ScanOp::kMin);                                                    \
            break;                                                                    \
        default:                                                                      \
            throw Error(Status::kNotSupported,                                        \
                        "reduce: this operator has no rung on the reduction ladder"); \
    }

#define CKL_SCAN_OVER_OP(T, BODY)          \
    switch (op) {                          \
        case ScanOp::kSum:                 \
            BODY(T, ScanOp::kSum);         \
            break;                         \
        case ScanOp::kMax:                 \
            BODY(T, ScanOp::kMax);         \
            break;                         \
        case ScanOp::kMin:                 \
            BODY(T, ScanOp::kMin);         \
            break;                         \
        case ScanOp::kLastNonZero:         \
            BODY(T, ScanOp::kLastNonZero); \
            break;                         \
    }

// The non float types carry kSum only, so their branch has no operator switch at
// all and cannot name an instantiation that does not exist.
#define CKL_OVER_DTYPE(OVER_OP, BODY)      \
    switch (s.dtype) {                     \
        case ScanDType::kF32:              \
            OVER_OP(float, BODY)           \
            break;                         \
        case ScanDType::kF64:              \
            BODY(double, ScanOp::kSum);    \
            break;                         \
        case ScanDType::kI32:              \
            BODY(int, ScanOp::kSum);       \
            break;                         \
        case ScanDType::kI64:              \
            BODY(long long, ScanOp::kSum); \
            break;                         \
    }

template <typename T>
T* typed(DeviceBuffer<unsigned char>& buffer) {
    return reinterpret_cast<T*>(buffer.data());
}

}  // namespace

namespace detail {

void reduce_launch(ScanPlan& plan, ReduceAlgo algo, ScanOp op, const void* in, void* out,
                   cudaStream_t stream) {
    ScanPlan::Impl& s = plan.live();
    const std::string refusal = plan.reduce_refusal(algo, op);
    if (!refusal.empty()) {
        throw Error(Status::kNotSupported, std::string("reduce: ") + refusal);
    }

#define CKL_RUN_REDUCE(T, OP)                                                                      \
    do {                                                                                           \
        const T* src = static_cast<const T*>(in);                                                  \
        T* dst = static_cast<T*>(out);                                                             \
        T* partials = typed<T>(s.partials);                                                        \
        switch (algo) {                                                                            \
            case ReduceAlgo::kAtomic:                                                              \
                reduce_atomic_launch<T, OP>(src, dst, s.n, stream);                                \
                break;                                                                             \
            case ReduceAlgo::kSharedTree:                                                          \
                reduce_shared_tree_launch<T, OP>(src, dst, s.n, partials, stream);                 \
                break;                                                                             \
            case ReduceAlgo::kShuffle:                                                             \
                reduce_shuffle_launch<T, OP>(src, dst, s.n, stream);                               \
                break;                                                                             \
            case ReduceAlgo::kVec4:                                                                \
                reduce_vec4_launch<T, OP>(src, dst, s.n, partials, s.persistent_blocks, stream);   \
                break;                                                                             \
            case ReduceAlgo::kSinglePass:                                                          \
                reduce_single_pass_launch<T, OP>(src, dst, s.n, partials, s.counters.data(),       \
                                                 s.persistent_blocks, stream);                     \
                break;                                                                             \
            case ReduceAlgo::kTwoPass:                                                             \
                reduce_two_pass_launch<T, OP>(src, dst, s.n, partials, s.persistent_blocks,        \
                                              stream);                                             \
                break;                                                                             \
            case ReduceAlgo::kDeterministic:                                                       \
                reduce_deterministic_launch<T, OP>(src, dst, s.n, partials,                        \
                                                   s.deterministic_blocks, stream);                \
                break;                                                                             \
            case ReduceAlgo::kKahan:                                                               \
                reduce_kahan_launch<T, OP>(src, dst, s.n, partials, s.persistent_blocks, stream);  \
                break;                                                                             \
            case ReduceAlgo::kCub:                                                                 \
                reduce_cub_launch<T, OP>(src, dst, s.n, s.cub_temp.data(), s.cub_bytes, stream);   \
                break;                                                                             \
            default:                                                                               \
                throw Error(Status::kInternal, "reduce: dispatch reached a rung it does not own"); \
        }                                                                                          \
    } while (false)

    CKL_OVER_DTYPE(CKL_REDUCE_OVER_OP, CKL_RUN_REDUCE)
#undef CKL_RUN_REDUCE
}

void scan_launch(ScanPlan& plan, ScanAlgo algo, ScanOp op, bool exclusive, const void* in,
                 void* out, cudaStream_t stream) {
    ScanPlan::Impl& s = plan.live();
    const std::string refusal = plan.scan_refusal(algo, op);
    if (!refusal.empty()) {
        throw Error(Status::kNotSupported, std::string("scan: ") + refusal);
    }

#define CKL_RUN_SCAN(T, OP)                                                                      \
    do {                                                                                         \
        const T* src = static_cast<const T*>(in);                                                \
        T* dst = static_cast<T*>(out);                                                           \
        T* levels = typed<T>(s.levels);                                                          \
        switch (algo) {                                                                          \
            case ScanAlgo::kHillisSteele:                                                        \
                scan_hillis_steele_launch<T, OP>(src, dst, s.n, exclusive, levels,               \
                                                 s.level_capacity, stream);                      \
                break;                                                                           \
            case ScanAlgo::kBlelloch:                                                            \
                scan_blelloch_launch<T, OP>(src, dst, s.n, exclusive, levels, s.level_capacity,  \
                                            stream);                                             \
                break;                                                                           \
            case ScanAlgo::kThreeKernel:                                                         \
                scan_three_kernel_launch<T, OP>(src, dst, s.n, exclusive, levels,                \
                                                s.level_capacity, stream);                       \
                break;                                                                           \
            case ScanAlgo::kReduceThenScan:                                                      \
            case ScanAlgo::kDeterministic:                                                       \
                scan_reduce_then_scan_launch<T, OP>(src, dst, s.n, exclusive, levels,            \
                                                    s.level_capacity, stream);                   \
                break;                                                                           \
            case ScanAlgo::kLookback:                                                            \
                scan_lookback_launch<T, OP>(src, dst, s.n, exclusive, s.status.data(),           \
                                            s.lookback_tiles, s.counters.data() + 1, stream);    \
                break;                                                                           \
            case ScanAlgo::kCub:                                                                 \
                scan_cub_launch<T, OP>(src, dst, s.n, exclusive, s.cub_temp.data(), s.cub_bytes, \
                                       stream);                                                  \
                break;                                                                           \
            default:                                                                             \
                throw Error(Status::kInternal, "scan: dispatch reached a rung it does not own"); \
        }                                                                                        \
    } while (false)

    CKL_OVER_DTYPE(CKL_SCAN_OVER_OP, CKL_RUN_SCAN)
#undef CKL_RUN_SCAN
}

}  // namespace detail

}  // namespace ckl
