#pragma once

/**
 * @file scan.hpp
 * @brief Device wide reduction and prefix scan, and the ladders behind them.
 *
 * Two ladders share one plan. The reduction ladder runs from a single global
 * atomic to a bandwidth bound persistent kernel; the scan ladder runs from a
 * Hillis-Steele block scan to single pass decoupled look-back. Both are reached
 * through ckl::ScanPlan, ckl::reduce and ckl::scan, and CUB is the baseline for
 * both. Thrust is not benchmarked separately: it is CUB behind a different call
 * signature, and docs/scan.md says so once.
 *
 * The plan is the entry point that matters. It is built once per length and
 * element type and owns everything a variant needs and nothing a variant should
 * allocate: the per block partials, the per level aggregate buffers the scan
 * then propagate rungs recurse through, the tile status array and the tile
 * counter the look-back rung claims tiles from, and the CUB temporary storage,
 * queried and allocated once. Running a variant through it enqueues launches and
 * nothing else, which is what benchmarks/support/allocation_gate.hpp checks.
 *
 * Inclusive and exclusive come out of one templated core rather than two copies,
 * and so do the four binary operators. ckl::ScanOp::kLastNonZero is on the list
 * because it is associative and not commutative, so a look-back that combined
 * its window in the wrong direction fails on it and passes on sum.
 *
 * Every entry point here is asynchronous on the stream it is given.
 */

#include <cstddef>
#include <memory>
#include <string>

#include <cuda_runtime.h>

#include "ckl/ckl_export.h"
#include "ckl/status.hpp"

namespace ckl {

/**
 * @brief Element type a plan reduces or scans.
 *
 * FP32 is the type the whole ladder is instantiated for. The other three carry
 * the sum operator only, which is what recomputing the roof per dtype needs;
 * docs/scan.md records why the operator by type cross product is not built.
 */
enum class ScanDType {
    kF32 = 0,  ///< 32 bit float.
    kF64,      ///< 64 bit float.
    kI32,      ///< 32 bit signed integer.
    kI64       ///< 64 bit signed integer.
};

/**
 * @brief The binary operator a reduction or a scan combines with.
 *
 * All four are associative, which is the only property either ladder depends
 * on. kLastNonZero is the one that is not commutative: `a op b` is `b` unless
 * `b` is zero, in which case it is `a`, so an inclusive scan of it carries the
 * most recent nonzero element forward and zero is its identity. It exists to
 * prove that the look-back path, the block scans and the level recursion all
 * combine their operands in the order the algorithm says and not in whichever
 * order happened to be convenient.
 */
enum class ScanOp {
    kSum = 0,     ///< Addition; identity zero.
    kMax,         ///< Maximum; identity the lowest finite value, or negative infinity.
    kMin,         ///< Minimum; identity the highest finite value, or positive infinity.
    kLastNonZero  ///< The right operand unless it is zero; associative, not commutative.
};

/**
 * @brief Every rung of the reduction ladder plus the vendor baseline.
 *
 * The values match ckl_reduce_algo_t in ckl.h one for one and in the same order.
 * r0 to r4 are the ladder of Section 12.3; kTwoPass is the comparison arm r4 is
 * measured against, and kDeterministic and kKahan are rungs of their own with
 * rows of their own.
 */
enum class ReduceAlgo {
    kAuto = 0,       ///< Let the plan choose and report the choice through chosen.
    kAtomic,         ///< r0: one atomic per element onto a single global accumulator.
    kSharedTree,     ///< r1: shared memory tree per block, one partial per block.
    kShuffle,        ///< r2: warp shuffle, one shared slot per warp, one atomic per block.
    kVec4,           ///< r3: 16 byte loads and a grid stride loop over persistent blocks.
    kSinglePass,     ///< r4: one launch, the last block combines the partials.
    kTwoPass,        ///< r4's comparison arm: a partials kernel and a combine kernel.
    kDeterministic,  ///< Fixed tree, fixed block count, no atomic accumulation.
    kKahan,          ///< Compensated summation; sum only.
    kCub             ///< cub::DeviceReduce, the baseline.
};

/**
 * @brief Every rung of the scan ladder plus the vendor baseline.
 *
 * The values match ckl_scan_algo_t in ckl.h one for one and in the same order.
 * kHillisSteele and kBlelloch are block scan primitives; they reach device wide
 * through the same scan then propagate recursion kThreeKernel uses, so the only
 * thing that changes between those three rungs is the block primitive and the
 * items each thread carries. docs/scan.md states that and why.
 */
enum class ScanAlgo {
    kAuto = 0,        ///< Let the plan choose and report the choice through chosen.
    kHillisSteele,    ///< s0: Hillis-Steele block scan, one item per thread.
    kBlelloch,        ///< s1: work efficient upsweep and downsweep, bank conflict padded.
    kThreeKernel,     ///< s2: scan then propagate, about four passes over the data.
    kReduceThenScan,  ///< s3: reduce then scan, about three passes.
    kLookback,        ///< s4: single pass decoupled look-back, about two passes.
    kDeterministic,   ///< Fixed tree, fixed block count; bit identical across runs.
    kCub              ///< cub::DeviceScan, the baseline.
};

/**
 * @brief Symbolic name of a reduction algorithm.
 * @param a Algorithm to name.
 * @return A static string such as "kVec4", never null.
 */
CKL_EXPORT const char* reduce_algo_name(ReduceAlgo a);

/**
 * @brief Symbolic name of a scan algorithm.
 * @param a Algorithm to name.
 * @return A static string such as "kLookback", never null.
 */
CKL_EXPORT const char* scan_algo_name(ScanAlgo a);

/**
 * @brief Symbolic name of a binary operator.
 * @param op Operator to name.
 * @return A static string such as "kLastNonZero", never null.
 */
CKL_EXPORT const char* scan_op_name(ScanOp op);

/**
 * @brief Symbolic name of an element type.
 * @param t Type to name.
 * @return A static string such as "kF32", never null.
 */
CKL_EXPORT const char* scan_dtype_name(ScanDType t);

/**
 * @brief Bytes one element of this type occupies.
 * @param t Type to size.
 * @return 4 or 8.
 */
CKL_EXPORT int scan_dtype_size(ScanDType t);

/**
 * @brief The calibrated constant in the random data tolerance model.
 * @return The committed value of c in tol(N) = c * sqrt(N) * FLT_EPSILON.
 * @note Calibrated once by the disabled-by-default calibration case in
 *       tests/test_scan.cpp, which sweeps seeds, lengths and rungs and reports
 *       the worst observed ratio. The method and the observed maximum are in
 *       docs/scan.md; the committed value carries deliberate slack over the
 *       maximum and no more, so the tolerance is still a gate that can fail.
 */
CKL_EXPORT double scan_tolerance_c();

/**
 * @brief Tolerance for a reduction or scan of random uniform data.
 * @param n Number of elements combined.
 * @return c * sqrt(n) * FLT_EPSILON.
 * @note The residual this bounds is max_i |got_i - ref_i| divided by the largest
 *       input magnitude, which is the error measured in units of the input
 *       scale. For random signed data the partial sums do a random walk of size
 *       sqrt(n) times that scale and the rounding error is machine epsilon times
 *       that walk, so the ratio is O(1) and c absorbs the depth of the reduction
 *       tree.
 * @note It is not a bound for adversarial input, so the mixed magnitude datasets
 *       are checked against a double precision reference divided by the running
 *       sum of magnitudes instead. It is also not the model for the two rungs
 *       that finish through a single global atomic, whose last combine is a
 *       serial accumulation; docs/scan.md carries that one and its calibration.
 */
CKL_EXPORT double scan_tolerance(long long n);

/**
 * @brief Tuning knobs a caller can set before the plan is built.
 *
 * Every field defaults to the rule the plan would apply on its own.
 */
struct CKL_EXPORT ScanPlanOptions {
    /// Query and allocate the CUB temporary storage. Off saves the allocation on
    /// a plan that will never run the baseline, and ReduceAlgo::kCub then reports
    /// kNotSupported with the reason.
    bool build_cub = true;
    /// Force kAuto onto the deterministic rungs. The environment variable
    /// CKL_REDUCE_DETERMINISTIC does the same thing for a caller that cannot
    /// reach this struct; either one is enough and the plan reports which
    /// through ScanPlan::deterministic.
    bool deterministic = false;
};

/**
 * @brief The per length state every reduction and scan variant runs out of.
 *
 * One plan per (length, element type), reused across every variant, every
 * operator and every repeat. Constructing it allocates the partials, the level
 * buffers, the tile status array, the tile counter and the CUB temporary
 * storage; calling ckl::reduce or ckl::scan on it does none of that.
 *
 * @note A plan is externally synchronized, like a Context: give each thread its
 *       own, or hold a lock across every call that touches a shared one.
 */
class CKL_EXPORT ScanPlan {
public:
    /**
     * @brief Builds the plan for one length and element type.
     * @param n Number of elements; zero is valid and allocates nothing.
     * @param dtype Element type the plan's buffers are sized for.
     * @param opt Tuning knobs; the default asks the plan to decide everything.
     * @throws ckl::Error when an allocation or a CUB storage query fails.
     */
    explicit ScanPlan(long long n, ScanDType dtype = ScanDType::kF32,
                      const ScanPlanOptions& opt = ScanPlanOptions());

    /// @brief Releases the partials, the level buffers and the CUB storage.
    ~ScanPlan();

    /**
     * @brief Move constructor.
     * @param other Plan to move from; it holds nothing afterwards.
     */
    ScanPlan(ScanPlan&& other) noexcept;

    /**
     * @brief Move assignment.
     * @param other Plan to move from; it holds nothing afterwards.
     * @return This plan.
     */
    ScanPlan& operator=(ScanPlan&& other) noexcept;

    ScanPlan(const ScanPlan&) = delete;
    ScanPlan& operator=(const ScanPlan&) = delete;

    /**
     * @brief Elements this plan was built for.
     * @return The length handed to the constructor.
     */
    long long size() const;

    /**
     * @brief Element type this plan was built for.
     * @return The type handed to the constructor.
     */
    ScanDType dtype() const;

    /**
     * @brief Whether kAuto resolves to the deterministic rungs.
     * @return True when the options asked for it or CKL_REDUCE_DETERMINISTIC is
     *         set in the environment to anything other than "0".
     */
    bool deterministic() const;

    /**
     * @brief Elements one tile of this rung covers.
     * @param a Scan rung to describe.
     * @return Threads per block times items per thread for that rung.
     */
    int tile_elements(ScanAlgo a) const;

    /**
     * @brief Tiles the first level of this rung splits the input into.
     * @param a Scan rung to describe.
     * @return ceil(size() / tile_elements(a)), or zero for an empty plan.
     */
    long long tiles(ScanAlgo a) const;

    /**
     * @brief Blocks the deterministic rungs launch.
     * @return A pure function of size() and the element type, never of the
     *         occupancy the device reports, which is what makes the answer bit
     *         identical from run to run and machine to machine.
     */
    int deterministic_blocks() const;

    /**
     * @brief Blocks the persistent rungs launch.
     * @return cudaOccupancyMaxActiveBlocksPerMultiprocessor times the SM count,
     *         capped at the tiles the input actually has.
     */
    int persistent_blocks() const;

    /**
     * @brief Declared DRAM traffic of one reduction, in bytes.
     * @param a Rung to model.
     * @return n * sizeof(T) for the single pass rungs, plus the partials traffic
     *         where a second pass reads them back.
     * @note This is the denominator of every effective GB/s this family quotes,
     *       and the number a measured dram__bytes.sum is checked against.
     */
    long long reduce_model_bytes(ReduceAlgo a) const;

    /**
     * @brief Declared DRAM traffic of one scan, in bytes.
     * @param a Rung to model.
     * @return About 4 * n * sizeof(T) for scan then propagate, 3 * n * sizeof(T)
     *         for reduce then scan, and 2 * n * sizeof(T) plus the tile status
     *         traffic for decoupled look-back.
     */
    long long scan_model_bytes(ScanAlgo a) const;

    /**
     * @brief Tile status traffic the look-back rung adds to its model.
     * @return Bytes of status words written and read once per tile, which is the
     *         term that makes the look-back model more than 2 * n * sizeof(T).
     */
    long long lookback_status_bytes() const;

    /**
     * @brief What kAuto would run for a reduction.
     * @return The rung the plan's rule picks; it launches nothing.
     */
    ReduceAlgo query_reduce() const;

    /**
     * @brief What kAuto would run for a scan.
     * @return The rung the plan's rule picks; it launches nothing.
     */
    ScanAlgo query_scan() const;

    /**
     * @brief Why this plan cannot run this reduction, or an empty string.
     * @param a Rung the caller wants.
     * @param op Operator the caller wants.
     * @return A sentence naming what is missing, or "" when the pair runs.
     */
    std::string reduce_refusal(ReduceAlgo a, ScanOp op) const;

    /**
     * @brief Why this plan cannot run this scan, or an empty string.
     * @param a Rung the caller wants.
     * @param op Operator the caller wants.
     * @return A sentence naming what is missing, or "" when the pair runs.
     */
    std::string scan_refusal(ScanAlgo a, ScanOp op) const;

    /**
     * @brief Whether this plan runs this reduction.
     * @param a Rung the caller wants.
     * @param op Operator the caller wants.
     * @return True when reduce_refusal is empty.
     */
    bool supports_reduce(ReduceAlgo a, ScanOp op) const;

    /**
     * @brief Whether this plan runs this scan.
     * @param a Rung the caller wants.
     * @param op Operator the caller wants.
     * @return True when scan_refusal is empty.
     */
    bool supports_scan(ScanAlgo a, ScanOp op) const;

    /**
     * @brief The CUB version this library was compiled against.
     * @return The CUB_VERSION macro, formatted MMMmmmpp, so 300304 is CUB 3.3.4.
     * @note It rides on every benchmark row and is cited in docs/scan.md, because
     *       a percentage against CUB is a percentage against one CUB.
     */
    static int cub_version();

    /// @cond INTERNAL
    // The plan's state, and the accessor the launchers reach it through. Public
    // because the dispatcher lives in another translation unit in another
    // library; Impl is only ever defined in src/scan.
    struct Impl;
    Impl& live() const;
    /// @endcond

private:
    std::unique_ptr<Impl> impl_;
};

/**
 * @brief Runs one reduction variant through a plan.
 * @param plan Plan built for the length and element type.
 * @param algo Rung to run, or ReduceAlgo::kAuto to let the plan choose.
 * @param op Binary operator to combine with.
 * @param in Device pointer to plan.size() elements of the plan's type.
 * @param out Device pointer to one element of the plan's type.
 * @param chosen Optional out-param receiving the rung taken; may be null. When
 *        non-null it is written on success and on failure alike.
 * @param stream Stream to enqueue on; nullptr means the default stream.
 * @return Status::kSuccess, or the failure status; ckl_last_error carries the detail.
 * @note An explicitly named rung is never rerouted. Only kAuto chooses, and
 *       chosen reports where it landed.
 * @note A length of zero writes the operator's identity to out, returns success
 *       and launches no zero sized grid.
 * @note Asynchronous. Synchronize on stream before reading out.
 */
CKL_EXPORT Status reduce(ScanPlan& plan, ReduceAlgo algo, ScanOp op, const void* in, void* out,
                         ReduceAlgo* chosen = nullptr, cudaStream_t stream = nullptr);

/**
 * @brief Runs one prefix scan variant through a plan.
 * @param plan Plan built for the length and element type.
 * @param algo Rung to run, or ScanAlgo::kAuto to let the plan choose.
 * @param op Binary operator to combine with.
 * @param exclusive True for an exclusive scan, false for an inclusive one.
 * @param in Device pointer to plan.size() elements of the plan's type.
 * @param out Device pointer to plan.size() elements; it may equal in, and every
 *        rung then runs in place, because every rung reads a tile whole before
 *        it writes any of it.
 * @param chosen Optional out-param receiving the rung taken; may be null. When
 *        non-null it is written on success and on failure alike.
 * @param stream Stream to enqueue on; nullptr means the default stream.
 * @return Status::kSuccess, or the failure status; ckl_last_error carries the detail.
 * @note An explicitly named rung is never rerouted. Only kAuto chooses, and
 *       chosen reports where it landed.
 * @note A length of zero returns success and launches no zero sized grid.
 * @note Asynchronous. Synchronize on stream before reading out.
 */
CKL_EXPORT Status scan(ScanPlan& plan, ScanAlgo algo, ScanOp op, bool exclusive, const void* in,
                       void* out, ScanAlgo* chosen = nullptr, cudaStream_t stream = nullptr);

}  // namespace ckl
