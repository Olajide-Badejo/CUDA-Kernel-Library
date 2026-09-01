#pragma once

/**
 * @file fft.hpp
 * @brief One dimensional complex FP32 FFT, the transpose study, and convolution.
 *
 * Powers of two from 2^10 to 2^24, forward and inverse, both first class. Every
 * hand written rung is Stockham autosort: each stage reads one buffer and writes
 * another with the digit permutation folded into the addressing, so no pass ever
 * scatters. Cooley-Tukey with a separate bit reversal pass is not implemented and
 * will not be; docs/fft.md carries the derivation of why.
 *
 * The plan is the entry point that matters, the same contract ckl::SpmvPlan
 * carries. It holds the twiddle tables, the ping pong workspace, the four step
 * factorization and the cuFFT handle. Building it computes the tables in double
 * and rounds them to FP32; running a variant through it enqueues launches and
 * nothing else. Recurrence generated twiddles are banned outright, because their
 * error grows with the stage index, and __sincosf is a labelled rung rather than
 * the default: FftPlanOptions::fast_twiddles turns it on and every accessor and
 * benchmark row says that it is on.
 *
 * Scaling: ckl::fft leaves the inverse transform unscaled, exactly as cuFFT does,
 * so the two are comparable without either side carrying a pass the other does
 * not. A caller that wants the round trip identity applies ckl::fft_scale, or
 * asks ckl::conv, which folds the 1/L into the epilogue of its last inverse pass.
 */

#include <cstddef>
#include <memory>

#include <cuda_runtime.h>

#include "ckl/ckl_export.h"
#include "ckl/status.hpp"

namespace ckl {

/// @brief Which way a transform runs.
enum class FftDirection {
    kForward = 0,  ///< exp(-2 pi i k n / N), unscaled.
    kInverse = 1   ///< exp(+2 pi i k n / N), unscaled, the same convention cuFFT uses.
};

/**
 * @brief Every 1D transform path the library can run, plus the vendor baseline.
 *
 * The rungs are the ladder of V2 spec Section 12.2 in order: one global memory
 * pass per radix 2 stage, then the whole transform resident in shared memory,
 * then wider butterflies, then the four step form for the sizes shared memory
 * cannot hold.
 */
enum class FftAlgo {
    kAuto = 0,        ///< Let the plan choose from the size and report the choice.
    kRadix2Global,    ///< r1: Stockham radix 2, one kernel launch per stage, global memory.
    kSharedResident,  ///< r2: one block per transform, both ping pong buffers in shared memory.
    kRadix4Global,    ///< r3: radix 4 butterflies, half the stages, more registers.
    kRadix8Global,    ///< r3: radix 8 butterflies mixed with one radix 2 or radix 4 stage.
    kFourStep,        ///< r4: N = N1 * N2, batched shared transforms and transposes between.
    kCufft            ///< cuFFT, planned once per size and reused, the baseline of record.
};

/**
 * @brief The three transposes measured against each other.
 *
 * A 2D FFT is row transforms, transpose, row transforms, transpose, and the
 * transposes are pure bandwidth, so they decide the time. These are measured on
 * their own before they are measured inside the 2D driver.
 */
enum class FftTranspose {
    kNaive = 0,     ///< One thread per element, coalesced on the write side only.
    kTiledPadded,   ///< 32 by 32 tile through shared memory, one element of padding.
    kStridedShared  ///< 32 by 32 tile loaded strided into unpadded shared, written coalesced.
};

/// @brief Every convolution path, plus the vendor baseline.
enum class ConvAlgo {
    kAuto = 0,        ///< Let the plan choose from the filter length and report the choice.
    kFftSeparate,     ///< Forward, a standalone pointwise multiply, inverse.
    kFftFused,        ///< The filter multiply and the 1/L folded into the transform epilogues.
    kDirectShared,    ///< Tiled time domain, signal tile plus halo and the taps in shared memory.
    kDirectConstant,  ///< The same tiling with the taps in constant memory.
    kCufft            ///< cuFFT forward, a standalone pointwise multiply, cuFFT inverse.
};

/**
 * @brief Symbolic name of a transform algorithm.
 * @param a Algorithm to name.
 * @return A static string such as "kFourStep", never null.
 */
CKL_EXPORT const char* fft_algo_name(FftAlgo a);

/**
 * @brief Symbolic name of a transpose variant.
 * @param t Variant to name.
 * @return A static string such as "kTiledPadded", never null.
 */
CKL_EXPORT const char* fft_transpose_name(FftTranspose t);

/**
 * @brief Symbolic name of a convolution algorithm.
 * @param a Algorithm to name.
 * @return A static string such as "kFftFused", never null.
 */
CKL_EXPORT const char* conv_algo_name(ConvAlgo a);

/**
 * @brief Direction of a transform as a printable name.
 * @param d Direction to name.
 * @return "forward" or "inverse", never null.
 */
CKL_EXPORT const char* fft_direction_name(FftDirection d);

/**
 * @brief The relative RMS error bound a correct transform of this size stays under.
 * @param n Transform length; values below 2 are treated as 2.
 * @return 8 * log2(n) * FLT_EPSILON, which is 2.3e-5 at 2^24 and 9.5e-6 at 2^10.
 * @note Error grows with the number of stages, so the bound has to as well. A flat
 *       tolerance would either pass a broken 2^24 or fail a correct one.
 */
CKL_EXPORT double fft_tolerance(int n);

/**
 * @brief Tuning knobs a caller can set before a transform plan is built.
 *
 * The defaults are what the plan would decide on its own, so a caller that wants
 * the library's answer passes the struct unchanged and reads back what it chose.
 */
struct CKL_EXPORT FftPlanOptions {
    /// Transforms of length n laid out back to back. The small size benchmarks
    /// batch until all 48 SMs have work; one unbatched 2^12 transform measures
    /// launch overhead rather than the kernel.
    int batch = 1;
    /// Build the twiddle tables with __sincosf instead of double precision sin
    /// and cos rounded to FP32. Its argument reduction gives about 2^-21 relative
    /// accuracy and shows up in the round trip number, so this is a rung that
    /// every row using it names, never the default.
    bool fast_twiddles = false;
    /// The first four step factor. Zero asks the plan to factor n itself into two
    /// powers of two that both sit at or below the shared resident bound.
    int four_step_n1 = 0;
    /// Which transpose the four step form runs. It is a plan option so the
    /// transpose study can be measured inside the rung that uses it and not only
    /// on its own.
    FftTranspose transpose = FftTranspose::kTiledPadded;
    /// Build the extra tables and buffers ckl::fft_r2c and ckl::fft_c2r need: the
    /// twiddles of the doubled length and two scratch arrays of n complex points.
    /// Off by default, because a plan that will only ever run complex transforms
    /// should not carry a third of a gigabyte of scratch it never touches.
    bool build_real = false;
};

/**
 * @brief The per size state every 1D transform variant runs out of.
 *
 * One plan per (length, batch), reused across every variant and every repeat. It
 * holds the twiddle tables, the ping pong workspace, the four step factorization
 * and its cross twiddle tables, the real to complex untangle table and the cuFFT
 * handle. Constructing it computes the tables in double precision on the host and
 * allocates the device buffers; calling ckl::fft on it does none of that.
 *
 * @note A plan is externally synchronized, like a Context: give each thread its
 *       own, or hold a lock across every call that touches a shared one.
 */
class CKL_EXPORT FftPlan {
public:
    /**
     * @brief Builds the plan for one transform length.
     * @param n Transform length; a power of two from 2^10 to 2^24.
     * @param opt Tuning knobs; the default asks the plan to decide everything.
     * @throws ckl::Error when n is out of range or not a power of two, when the
     *         batch is not positive, or when an allocation or a cuFFT plan fails.
     */
    explicit FftPlan(int n, const FftPlanOptions& opt = FftPlanOptions());

    /// @brief Releases the tables, the workspace and the cuFFT handle.
    ~FftPlan();

    /**
     * @brief Move constructor.
     * @param other Plan to move from; it holds nothing afterwards.
     */
    FftPlan(FftPlan&& other) noexcept;

    /**
     * @brief Move assignment.
     * @param other Plan to move from; it holds nothing afterwards.
     * @return This plan.
     */
    FftPlan& operator=(FftPlan&& other) noexcept;

    FftPlan(const FftPlan&) = delete;
    FftPlan& operator=(const FftPlan&) = delete;

    /**
     * @brief Transform length.
     * @return The n handed to the constructor.
     */
    int n() const;

    /**
     * @brief Transforms per call.
     * @return The batch count handed to the constructor.
     */
    int batch() const;

    /**
     * @brief Stage count of a radix 2 decomposition.
     * @return log2(n).
     */
    int log2n() const;

    /**
     * @brief Whether the twiddle tables came from __sincosf.
     * @return True when FftPlanOptions::fast_twiddles was set.
     */
    bool fast_twiddles() const;

    /**
     * @brief Largest length the single block shared resident kernel takes.
     * @return 4096.
     * @note A complex FP32 point is 8 bytes and the kernel holds two ping pong
     *       buffers, so it needs 16n bytes. 2^12 costs 65536 and fits inside the
     *       101376 byte opt in limit; 2^13 costs 131072 and does not. Holding one
     *       radix stage in registers would bring 2^13 back inside, and that trick
     *       is not implemented: above 2^12 the four step form takes over.
     */
    static int shared_resident_max();

    /**
     * @brief First four step factor.
     * @return N1 with n = N1 * N2, or zero when n is small enough not to need it.
     */
    int four_step_n1() const;

    /**
     * @brief Second four step factor.
     * @return N2 with n = N1 * N2, or zero when n is small enough not to need it.
     */
    int four_step_n2() const;

    /**
     * @brief Global memory passes one call of a variant makes over the data.
     * @param a Variant to count; kAuto is resolved through query() first.
     * @return The pass count, or zero for kCufft, whose internal pass count is
     *         not ours to declare.
     */
    int passes(FftAlgo a) const;

    /**
     * @brief Declared traffic model of one call, in bytes.
     * @param a Variant to model; kAuto is resolved through query() first.
     * @return passes(a) * 16 * n * batch, or zero for kCufft.
     * @note This is the number the Gate X 15 percent dram__bytes.sum check is
     *       against, and it is what stops effective GB/s being gamed by moving
     *       more data. The twiddle traffic is declared separately below because
     *       it is bounded rather than exact.
     */
    long long model_bytes(FftAlgo a) const;

    /**
     * @brief Lower end of the twiddle traffic, in bytes per call.
     * @param a Variant to model; kAuto is resolved through query() first.
     * @return 8 bytes per distinct twiddle the variant reads, summed over stages,
     *         which is what a perfectly cache resident table would cost.
     */
    long long twiddle_bytes_low(FftAlgo a) const;

    /**
     * @brief Upper end of the twiddle traffic, in bytes per call.
     * @param a Variant to model; kAuto is resolved through query() first.
     * @return 8 bytes per twiddle read by any thread, summed over stages, which
     *         is what no reuse at all would cost.
     * @note The tables are stored factored, as a pair whose sizes multiply to n,
     *       so the whole of both tables is under 64 KB even at 2^24 and the truth
     *       sits at the low end. The band is here so the ncu round can say so
     *       with a measurement instead of an assertion.
     */
    long long twiddle_bytes_high(FftAlgo a) const;

    /**
     * @brief The standard FLOP count of the transforms one call performs.
     * @return 5 * n * log2(n) * batch.
     * @note In the tables for continuity with the literature, and not the metric.
     *       Arithmetic intensity for one ideal pass is 5 n log2(n) / 16n, which is
     *       6.25 FLOP per byte at 2^20 against an FP32 ridge point of 53. Every
     *       size in scope sits far to the left of the ridge, so effective GB/s
     *       against the measured DRAM roof is the honest metric.
     */
    double flop_model() const;

    /**
     * @brief Whether this plan can run a variant at all.
     * @param a Variant to test.
     * @return True when a call would launch it.
     */
    bool supports(FftAlgo a) const;

    /**
     * @brief Why a variant is refused.
     * @param a Variant to test.
     * @return The reason, or an empty string when supports(a) is true. The pointer
     *         stays valid until the plan is destroyed.
     */
    const char* refusal(FftAlgo a) const;

    /**
     * @brief What kAuto would run at this size.
     * @return The variant the plan's rule picks; it launches nothing.
     * @note The rule is provisional and documented as such in docs/fft.md: shared
     *       resident at or below 2^12, four step above it. The radix 4 and radix 8
     *       rungs are never chosen automatically, because whether a wider butterfly
     *       wins depends on the occupancy its register count leaves, which only a
     *       measurement settles.
     */
    FftAlgo query() const;

    /// @cond INTERNAL
    // The plan's state, and the accessor ckl::detail::fft_launch reaches it
    // through. Public because the dispatcher lives in another translation unit in
    // another library; Impl is only ever defined in src/fft.
    struct Impl;
    Impl& live() const;
    /// @endcond

private:
    std::unique_ptr<Impl> impl_;
};

/**
 * @brief Runs one 1D transform variant through a plan.
 * @param plan Plan built for the length; it supplies the tables and the workspace.
 * @param algo Path to run, or FftAlgo::kAuto to let the plan choose.
 * @param dir Forward or inverse; neither is scaled.
 * @param in Device pointer to n * batch complex inputs.
 * @param out Device pointer to n * batch complex outputs; must not alias in.
 * @param chosen Optional out-param receiving the path taken; may be null. When
 *        non-null it is written on success and on failure alike.
 * @param stream Stream to enqueue on; nullptr means the default stream.
 * @return Status::kSuccess, or the failure status; ckl_last_error carries the detail.
 * @note An explicitly named algorithm is never rerouted. Only kAuto chooses, and
 *       chosen reports where it landed.
 * @note Asynchronous. Synchronize on stream before reading out.
 */
CKL_EXPORT Status fft(FftPlan& plan, FftAlgo algo, FftDirection dir, const float2* in, float2* out,
                      FftAlgo* chosen = nullptr, cudaStream_t stream = nullptr);

/**
 * @brief Real to complex transform: 2n reals in, n+1 complex bins out.
 * @param plan Plan built for the packed complex length n, so it transforms 2n reals.
 * @param algo Path the packed complex transform runs on, or FftAlgo::kAuto.
 * @param in Device pointer to 2 * n * batch reals.
 * @param out Device pointer to (n + 1) * batch complex bins.
 * @param chosen Optional out-param receiving the path taken; may be null.
 * @param stream Stream to enqueue on; nullptr means the default stream.
 * @return Status::kSuccess, or the failure status.
 * @note This is rung 5: the 2n reals are read as n complex points, transformed
 *       once at half the length, and untangled into the Hermitian spectrum by a
 *       single extra pass. A complex transform of the full length would move
 *       twice the bytes for the same answer.
 */
CKL_EXPORT Status fft_r2c(FftPlan& plan, FftAlgo algo, const float* in, float2* out,
                          FftAlgo* chosen = nullptr, cudaStream_t stream = nullptr);

/**
 * @brief Complex to real transform: n+1 complex bins in, 2n reals out, unscaled.
 * @param plan Plan built for the packed complex length n.
 * @param algo Path the packed complex transform runs on, or FftAlgo::kAuto.
 * @param in Device pointer to (n + 1) * batch complex bins.
 * @param out Device pointer to 2 * n * batch reals.
 * @param chosen Optional out-param receiving the path taken; may be null.
 * @param stream Stream to enqueue on; nullptr means the default stream.
 * @return Status::kSuccess, or the failure status.
 * @note The inverse of fft_r2c up to the factor n, which is plan.n() and not the
 *       real length 2n: the packed transform that does the work is n points long.
 *       The scaling is left to the caller for the same reason ckl::fft leaves the
 *       complex inverse unscaled.
 */
CKL_EXPORT Status fft_c2r(FftPlan& plan, FftAlgo algo, const float2* in, float* out,
                          FftAlgo* chosen = nullptr, cudaStream_t stream = nullptr);

/// @brief Knobs for a 2D plan.
struct CKL_EXPORT Fft2dPlanOptions {
    /// Which transpose the row column driver runs between its two row passes.
    FftTranspose transpose = FftTranspose::kTiledPadded;
    /// Build the twiddle tables with __sincosf; see FftPlanOptions::fast_twiddles.
    bool fast_twiddles = false;
};

/**
 * @brief Row column 2D transform state: two batched 1D plans and a workspace.
 *
 * Both extents are powers of two at or below the shared resident bound, so both
 * row passes are single block transforms and the only global traffic is the two
 * row passes and the two transposes. That is the point of the family: the
 * transposes are pure bandwidth and they dominate, which is what
 * FftTranspose exists to measure.
 */
class CKL_EXPORT Fft2dPlan {
public:
    /**
     * @brief Builds a plan for a rows by cols 2D transform.
     * @param rows Row count; a power of two from 2^4 to the shared resident bound.
     * @param cols Column count; the same range.
     * @param opt Which transpose to run and whether to use fast twiddles.
     * @throws ckl::Error when an extent is out of range or an allocation fails.
     */
    Fft2dPlan(int rows, int cols, const Fft2dPlanOptions& opt = Fft2dPlanOptions());

    /// @brief Releases the two 1D plans and the workspace.
    ~Fft2dPlan();

    /**
     * @brief Move constructor.
     * @param other Plan to move from; it holds nothing afterwards.
     */
    Fft2dPlan(Fft2dPlan&& other) noexcept;

    /**
     * @brief Move assignment.
     * @param other Plan to move from; it holds nothing afterwards.
     * @return This plan.
     */
    Fft2dPlan& operator=(Fft2dPlan&& other) noexcept;

    Fft2dPlan(const Fft2dPlan&) = delete;
    Fft2dPlan& operator=(const Fft2dPlan&) = delete;

    /**
     * @brief Row count.
     * @return The rows handed to the constructor.
     */
    int rows() const;

    /**
     * @brief Column count.
     * @return The cols handed to the constructor.
     */
    int cols() const;

    /**
     * @brief The transpose this plan runs.
     * @return The variant handed to the constructor.
     */
    FftTranspose transpose() const;

    /**
     * @brief Global memory passes one 2D call makes.
     * @return 4 for kNaive and kTiledPadded: a row pass, a transpose, a row pass
     *         and a transpose. 2 for kStridedShared, which runs the column pass
     *         out of strided loads and has no transpose at all.
     */
    int passes() const;

    /**
     * @brief Declared traffic model of one 2D call, in bytes.
     * @return passes() * 16 * rows * cols.
     */
    long long model_bytes() const;

    /**
     * @brief The share of model_bytes() the transposes account for.
     * @return 2 * 16 * rows * cols for the two transposing variants, and zero for
     *         kStridedShared.
     * @note Half the declared traffic of a transposing variant, which is why the
     *       transpose share of the measured time is the number this rung is
     *       judged on, and why removing the transposes entirely is a rung of its
     *       own rather than a tuning detail.
     */
    long long transpose_model_bytes() const;

    /// @cond INTERNAL
    struct Impl;
    Impl& live() const;
    /// @endcond

private:
    std::unique_ptr<Impl> impl_;
};

/**
 * @brief Runs a row column 2D transform through a plan.
 * @param plan Plan built for the extents.
 * @param dir Forward or inverse; neither is scaled.
 * @param in Device pointer to rows * cols complex inputs, row major.
 * @param out Device pointer to rows * cols complex outputs; must not alias in.
 * @param stream Stream to enqueue on; nullptr means the default stream.
 * @return Status::kSuccess, or the failure status.
 * @note Asynchronous. Synchronize on stream before reading out.
 */
CKL_EXPORT Status fft2d(Fft2dPlan& plan, FftDirection dir, const float2* in, float2* out,
                        cudaStream_t stream = nullptr);

/// @brief Knobs for a convolution plan.
struct CKL_EXPORT ConvPlanOptions {
    /// The transform variant the two FFT paths run on. kAuto asks the transform
    /// plan to decide from the padded length.
    FftAlgo fft_algo = FftAlgo::kAuto;
    /// Build the twiddle tables with __sincosf; see FftPlanOptions::fast_twiddles.
    bool fast_twiddles = false;
};

/**
 * @brief The per filter state every convolution variant runs out of.
 *
 * Linear convolution of a length N signal with a length M filter, output length
 * N + M - 1, computed either by zero padding both to L, the next power of two at
 * or above N + M - 1, or directly in the time domain. The filter spectrum is
 * computed once here and cached, so no timed region ever contains a transform of
 * the filter; the taps are also held on the device for the direct kernels and
 * bound into constant memory for the constant memory rung.
 *
 * @note Building the plan runs a transform of the filter and therefore launches
 *       kernels. It is setup, and it belongs outside every timed region.
 */
class CKL_EXPORT ConvPlan {
public:
    /**
     * @brief Builds the plan for one filter.
     * @param signal_length N, the signal length; at least 1.
     * @param filter Device pointer to the filter taps, length filter_length.
     * @param filter_length M, the filter length; at least 1.
     * @param opt Which transform variant to run and whether to use fast twiddles.
     * @throws ckl::Error when a length is out of range, when the padded length L
     *         would exceed 2^24, or when an allocation or a transform fails.
     */
    ConvPlan(int signal_length, const float* filter, int filter_length,
             const ConvPlanOptions& opt = ConvPlanOptions());

    /// @brief Releases the cached spectrum, the taps and the transform plan.
    ~ConvPlan();

    /**
     * @brief Move constructor.
     * @param other Plan to move from; it holds nothing afterwards.
     */
    ConvPlan(ConvPlan&& other) noexcept;

    /**
     * @brief Move assignment.
     * @param other Plan to move from; it holds nothing afterwards.
     * @return This plan.
     */
    ConvPlan& operator=(ConvPlan&& other) noexcept;

    ConvPlan(const ConvPlan&) = delete;
    ConvPlan& operator=(const ConvPlan&) = delete;

    /**
     * @brief Signal length.
     * @return N.
     */
    int signal_length() const;

    /**
     * @brief Filter length.
     * @return M.
     */
    int filter_length() const;

    /**
     * @brief Output length.
     * @return N + M - 1.
     */
    int output_length() const;

    /**
     * @brief Padded transform length.
     * @return L, the next power of two at or above N + M - 1.
     */
    int transform_length() const;

    /**
     * @brief The transform variant the FFT paths run.
     * @return The resolved variant, never FftAlgo::kAuto.
     */
    FftAlgo fft_algo() const;

    /**
     * @brief Global memory passes one call of a variant makes over the padded data.
     * @param a Variant to count; kAuto is resolved through query() first.
     * @return The pass count in units of a full L point complex pass, or zero for
     *         a direct variant, which makes one pass over the signal and none over
     *         a padded array.
     */
    int passes(ConvAlgo a) const;

    /**
     * @brief Declared traffic model of one call, in bytes.
     * @param a Variant to model; kAuto is resolved through query() first.
     * @return The byte count derived in docs/fft.md: the pack, the two transforms,
     *         the pointwise multiply where it is a separate pass, and the extract.
     *         For a direct variant it is the signal read, the tap reads and the
     *         output write.
     */
    long long model_bytes(ConvAlgo a) const;

    /**
     * @brief The FLOP count of one call.
     * @param a Variant to count; kAuto is resolved through query() first.
     * @return 2 * N * M for a direct variant, and 2 * 5 * L * log2(L) + 6 * L for
     *         an FFT variant: one forward transform, one inverse, and the complex
     *         multiply. The filter's own forward transform is not counted, because
     *         the plan paid for it once and a timed call does not run it.
     */
    double flop_model(ConvAlgo a) const;

    /**
     * @brief Whether this plan can run a variant at all.
     * @param a Variant to test.
     * @return True when a call would launch it.
     */
    bool supports(ConvAlgo a) const;

    /**
     * @brief Why a variant is refused.
     * @param a Variant to test.
     * @return The reason, or an empty string when supports(a) is true. The pointer
     *         stays valid until the plan is destroyed.
     */
    const char* refusal(ConvAlgo a) const;

    /**
     * @brief What kAuto would run for this signal and filter.
     * @return The variant the plan's rule picks; it launches nothing.
     * @note The rule is provisional and documented as such in docs/fft.md. There
     *       is no committed crossover sweep yet, and the crossover is exactly what
     *       the sweep exists to measure, so kAuto uses the arithmetic of the two
     *       models rather than a measurement, and says so.
     */
    ConvAlgo query() const;

    /// @cond INTERNAL
    struct Impl;
    Impl& live() const;
    /// @endcond

private:
    std::unique_ptr<Impl> impl_;
};

/**
 * @brief Runs one convolution variant through a plan.
 * @param plan Plan built for the filter; it supplies the cached spectrum.
 * @param algo Path to run, or ConvAlgo::kAuto to let the plan choose.
 * @param signal Device pointer to the signal, length plan.signal_length().
 * @param out Device pointer to the output, length plan.output_length().
 * @param chosen Optional out-param receiving the path taken; may be null. When
 *        non-null it is written on success and on failure alike.
 * @param stream Stream to enqueue on; nullptr means the default stream.
 * @return Status::kSuccess, or the failure status; ckl_last_error carries the detail.
 * @note An explicitly named algorithm is never rerouted. Only kAuto chooses, and
 *       chosen reports where it landed.
 * @note Asynchronous. Synchronize on stream before reading out.
 */
CKL_EXPORT Status conv(ConvPlan& plan, ConvAlgo algo, const float* signal, float* out,
                       ConvAlgo* chosen = nullptr, cudaStream_t stream = nullptr);

// ---------------------------------------------------------------------------
// The rungs as free functions. Each one launches exactly the kernels its name
// says and takes the tables and the scratch as arguments, because building them
// inside the call is what a plan exists to prevent.
// ---------------------------------------------------------------------------

/**
 * @brief A twiddle table, stored factored so it stays inside the caches.
 *
 * A direct table of exp(-2 pi i t / n) for every t costs 8n bytes, which is
 * 128 MB at 2^24 and comes from DRAM on the late stages. Writing
 * t = q * 2^lo_bits + r and holding one table of the coarse factors and one of
 * the fine factors costs 8 * (n / 2^lo_bits + 2^lo_bits) bytes instead, which is
 * under 64 KB at every size in scope, at the price of one complex multiply and
 * one extra rounding per twiddle. Both tables are computed in double precision
 * and rounded to FP32 once, when the plan is built.
 */
struct CKL_EXPORT FftTwiddles {
    /// exp(-2 pi i q * 2^lo_bits / n) for q in [0, n / 2^lo_bits).
    const float2* hi = nullptr;
    /// exp(-2 pi i r / n) for r in [0, 2^lo_bits).
    const float2* lo = nullptr;
    /// log2 of the fine table length.
    int lo_bits = 0;
    /// The length the tables are a decomposition of.
    int n = 0;
};

/**
 * @brief One Stockham radix 2 pass per stage, all of them in global memory.
 * @param in Device pointer to n * batch complex inputs.
 * @param out Device pointer to n * batch complex outputs.
 * @param work Device scratch of n * batch complex points for the ping pong.
 * @param tw Twiddle tables of length n.
 * @param n Transform length, a power of two.
 * @param batch Transforms laid out back to back.
 * @param dir Forward or inverse; neither is scaled.
 * @param stream Stream to enqueue on; nullptr means the default stream.
 * @note log2(n) launches, each of which reads n and writes n complex points, so
 *       the traffic is 16 n log2(n) bytes and this rung exists to be beaten.
 */
CKL_EXPORT void fft_radix2_global(const float2* in, float2* out, float2* work,
                                  const FftTwiddles& tw, int n, int batch, FftDirection dir,
                                  cudaStream_t stream = nullptr);

/**
 * @brief Radix 4 Stockham, with one radix 2 stage first when log2(n) is odd.
 * @param in Device pointer to n * batch complex inputs.
 * @param out Device pointer to n * batch complex outputs.
 * @param work Device scratch of n * batch complex points for the ping pong.
 * @param tw Twiddle tables of length n.
 * @param n Transform length, a power of two.
 * @param batch Transforms laid out back to back.
 * @param dir Forward or inverse; neither is scaled.
 * @param stream Stream to enqueue on; nullptr means the default stream.
 * @note Half the stages of radix 2 and therefore half the traffic, paid for in
 *       registers: four complex values and three twiddles stay live per thread.
 */
CKL_EXPORT void fft_radix4_global(const float2* in, float2* out, float2* work,
                                  const FftTwiddles& tw, int n, int batch, FftDirection dir,
                                  cudaStream_t stream = nullptr);

/**
 * @brief Radix 8 Stockham, mixed with one radix 2 or radix 4 stage as log2(n) needs.
 * @param in Device pointer to n * batch complex inputs.
 * @param out Device pointer to n * batch complex outputs.
 * @param work Device scratch of n * batch complex points for the ping pong.
 * @param tw Twiddle tables of length n.
 * @param n Transform length, a power of two.
 * @param batch Transforms laid out back to back.
 * @param dir Forward or inverse; neither is scaled.
 * @param stream Stream to enqueue on; nullptr means the default stream.
 * @note Eight complex values plus seven twiddles live per thread, so registers
 *       rather than shared memory are expected to limit occupancy here. That is
 *       why this rung and radix 4 are measured with registers per thread printed
 *       beside throughput.
 */
CKL_EXPORT void fft_radix8_global(const float2* in, float2* out, float2* work,
                                  const FftTwiddles& tw, int n, int batch, FftDirection dir,
                                  cudaStream_t stream = nullptr);

/**
 * @brief One block per transform, both ping pong buffers resident in shared memory.
 * @param in Device pointer to n * batch complex inputs.
 * @param out Device pointer to n * batch complex outputs.
 * @param tw Twiddle tables of length n.
 * @param n Transform length, a power of two at or below FftPlan::shared_resident_max().
 * @param batch Transforms laid out back to back, one block each.
 * @param dir Forward or inverse; neither is scaled.
 * @param epilogue_mul Device pointer to n complex values the result is multiplied
 *        by at its index inside the transform as it is stored, or null for none.
 *        This is where a convolution folds its filter spectrum.
 * @param epilogue_scale Scale applied to the result as it is stored; 1.0f for none.
 * @param stream Stream to enqueue on; nullptr means the default stream.
 * @throws ckl::Error when n is above the shared resident bound, or when the device
 *         refuses the dynamic shared memory opt in.
 * @note One read and one write of the whole array, whatever log2(n) is, which is
 *       the whole of what this rung buys. A 64 KB block is one block per SM against
 *       the 100 KB per SM limit, so a benchmark has to batch to fill 48 SMs.
 */
CKL_EXPORT void fft_shared_resident(const float2* in, float2* out, const FftTwiddles& tw, int n,
                                    int batch, FftDirection dir, const float2* epilogue_mul,
                                    float epilogue_scale, cudaStream_t stream = nullptr);

/**
 * @brief The four step form: three transposes around two batched shared transforms.
 * @param in Device pointer to n1 * n2 * batch complex inputs.
 * @param out Device pointer to n1 * n2 * batch complex outputs.
 * @param work Device scratch of n1 * n2 * batch complex points.
 * @param tw_n1 Twiddle tables of length n1, for the second sub transform.
 * @param tw_n2 Twiddle tables of length n2, for the first sub transform.
 * @param tw_full Twiddle tables of length n1 * n2, for the cross twiddle.
 * @param n1 First factor; a power of two at or below the shared resident bound.
 * @param n2 Second factor; the same bound, with n1 * n2 the transform length.
 * @param batch Transforms laid out back to back.
 * @param dir Forward or inverse; neither is scaled.
 * @param transpose Which variant the three transposes run.
 * @param epilogue_mul Device pointer to n1 * n2 complex values the result is
 *        multiplied by at the output index, or null for none.
 * @param epilogue_scale Scale applied to the result as it is stored; 1.0f for none.
 * @param stream Stream to enqueue on; nullptr means the default stream.
 * @throws ckl::Error when a factor is above the shared resident bound.
 * @note Five passes: transpose, batched size n2 transforms with the cross twiddle
 *       folded into their store epilogue, transpose, batched size n1 transforms,
 *       transpose. The epilogue arguments ride on that last transpose, so a
 *       convolution pays nothing for its pointwise multiply.
 */
CKL_EXPORT void fft_four_step(const float2* in, float2* out, float2* work, const FftTwiddles& tw_n1,
                              const FftTwiddles& tw_n2, const FftTwiddles& tw_full, int n1, int n2,
                              int batch, FftDirection dir, FftTranspose transpose,
                              const float2* epilogue_mul, float epilogue_scale,
                              cudaStream_t stream = nullptr);

/**
 * @brief Out of place complex transpose, in one of three variants.
 * @param in Device pointer to rows * cols * batch complex inputs, row major.
 * @param out Device pointer to cols * rows * batch complex outputs, row major.
 * @param rows Rows of the input.
 * @param cols Columns of the input.
 * @param batch Matrices laid out back to back.
 * @param variant Which of the three transposes to launch.
 * @param epilogue_mul Device pointer to cols * rows complex values the output is
 *        multiplied by at its own index, or null for none.
 * @param epilogue_scale Scale applied to the output as it is stored; 1.0f for none.
 * @param stream Stream to enqueue on; nullptr means the default stream.
 * @note The three variants differ only in where the strided access lands and
 *       whether the shared memory bank conflict is padded away, which is what
 *       makes them a study rather than three copies of one kernel.
 */
CKL_EXPORT void fft_transpose(const float2* in, float2* out, int rows, int cols, int batch,
                              FftTranspose variant, const float2* epilogue_mul,
                              float epilogue_scale, cudaStream_t stream = nullptr);

/**
 * @brief Untangles a packed half length spectrum into a Hermitian one.
 * @param z Device pointer to n * batch complex values, the transform of the packed signal.
 * @param out Device pointer to (n + 1) * batch complex bins.
 * @param tw_double Twiddle tables of length 2n.
 * @param n Packed complex length; the real signal holds 2n samples.
 * @param batch Transforms laid out back to back.
 * @param stream Stream to enqueue on; nullptr means the default stream.
 * @note The second half of rung 5. Bin k combines Z[k] with the conjugate of
 *       Z[n-k], which is why this is a pass of its own rather than an epilogue.
 */
CKL_EXPORT void fft_r2c_untangle(const float2* z, float2* out, const FftTwiddles& tw_double, int n,
                                 int batch, cudaStream_t stream = nullptr);

/**
 * @brief Packs a Hermitian spectrum back into a half length complex one.
 * @param spectrum Device pointer to (n + 1) * batch complex bins.
 * @param z Device pointer to n * batch complex values ready for the inverse transform.
 * @param tw_double Twiddle tables of length 2n.
 * @param n Packed complex length; the real signal holds 2n samples.
 * @param batch Transforms laid out back to back.
 * @param stream Stream to enqueue on; nullptr means the default stream.
 * @note The inverse of fft_r2c_untangle, and the first pass of fft_c2r.
 */
CKL_EXPORT void fft_c2r_pack(const float2* spectrum, float2* z, const FftTwiddles& tw_double, int n,
                             int batch, cudaStream_t stream = nullptr);

/**
 * @brief Reinterprets a packed complex array as interleaved reals.
 * @param z Device pointer to n * batch complex values.
 * @param out Device pointer to 2n * batch reals.
 * @param n Packed complex length.
 * @param batch Transforms laid out back to back.
 * @param scale Factor applied on the way out.
 * @param stream Stream to enqueue on; nullptr means the default stream.
 */
CKL_EXPORT void fft_unpack_real(const float2* z, float* out, int n, int batch, float scale,
                                cudaStream_t stream = nullptr);

/**
 * @brief Reads 2n interleaved reals as n complex points.
 * @param in Device pointer to 2n * batch reals.
 * @param z Device pointer to n * batch complex values.
 * @param n Packed complex length.
 * @param batch Transforms laid out back to back.
 * @param stream Stream to enqueue on; nullptr means the default stream.
 */
CKL_EXPORT void fft_pack_real(const float* in, float2* z, int n, int batch,
                              cudaStream_t stream = nullptr);

/**
 * @brief Scales a complex array in place.
 * @param data Device pointer to count complex values.
 * @param count Number of complex values.
 * @param scale Factor applied to both components.
 * @param stream Stream to enqueue on; nullptr means the default stream.
 * @note The 1/n a round trip needs. It is a separate pass on purpose: folding it
 *       into ckl::fft would make the inverse incomparable with cuFFT, which does
 *       not scale.
 */
CKL_EXPORT void fft_scale(float2* data, long long count, float scale,
                          cudaStream_t stream = nullptr);

/**
 * @brief Pointwise complex multiply of two spectra.
 * @param a Device pointer to count complex values.
 * @param b Device pointer to count complex values.
 * @param out Device pointer to count complex outputs; may alias a.
 * @param count Number of complex values.
 * @param scale Factor applied to the product.
 * @param stream Stream to enqueue on; nullptr means the default stream.
 * @note This is the pass the fused convolution variant removes. It reads two
 *       spectra and writes one, so it moves 24 bytes per point that the fused
 *       epilogue does not move at all.
 */
CKL_EXPORT void conv_pointwise(const float2* a, const float2* b, float2* out, long long count,
                               float scale, cudaStream_t stream = nullptr);

/**
 * @brief Zero pads a real signal into a complex array of the transform length.
 * @param signal Device pointer to n reals.
 * @param padded Device pointer to l complex values.
 * @param n Signal length.
 * @param l Padded transform length.
 * @param stream Stream to enqueue on; nullptr means the default stream.
 */
CKL_EXPORT void conv_pack(const float* signal, float2* padded, int n, int l,
                          cudaStream_t stream = nullptr);

/**
 * @brief Takes the real part of the first out_len points of a spectrum.
 * @param spectrum Device pointer to at least out_len complex values.
 * @param out Device pointer to out_len reals.
 * @param out_len Output length, which is n + m - 1 for a linear convolution.
 * @param scale Factor applied on the way out, 1/l when the inverse did not scale.
 * @param stream Stream to enqueue on; nullptr means the default stream.
 */
CKL_EXPORT void conv_extract(const float2* spectrum, float* out, int out_len, float scale,
                             cudaStream_t stream = nullptr);

/**
 * @brief Tiled time domain convolution, signal tile plus halo and taps in shared memory.
 * @param signal Device pointer to n reals.
 * @param filter Device pointer to m reals.
 * @param out Device pointer to n + m - 1 reals.
 * @param n Signal length.
 * @param m Filter length.
 * @param stream Stream to enqueue on; nullptr means the default stream.
 * @note One output per thread and 2 n m FLOP. The filter is walked in chunks so
 *       the shared footprint stays fixed whatever m is; a halo of m - 1 samples
 *       would not fit in shared memory at the top of the crossover sweep.
 */
CKL_EXPORT void conv_direct_shared(const float* signal, const float* filter, float* out, int n,
                                   int m, cudaStream_t stream = nullptr);

/**
 * @brief The same tiling with the taps read from constant memory.
 * @param signal Device pointer to n reals.
 * @param out Device pointer to n + m - 1 reals.
 * @param n Signal length.
 * @param m Filter length; at most conv_direct_constant_max_taps().
 * @param owner The object that bound the taps, checked against the binding.
 * @param stream Stream to enqueue on; nullptr means the default stream.
 * @throws ckl::Error when m is out of range, or when the taps currently in the
 *         bank were bound by another owner.
 * @note The constant bank is one per module, so a plan binds its taps once and
 *       this call refuses rather than silently convolving with another filter.
 */
CKL_EXPORT void conv_direct_constant(const float* signal, float* out, int n, int m,
                                     const void* owner, cudaStream_t stream = nullptr);

/**
 * @brief Copies taps into the constant bank and records who owns them.
 * @param filter Device pointer to m reals.
 * @param m Filter length; at most conv_direct_constant_max_taps().
 * @param owner The object claiming the bank, normally a ConvPlan::Impl.
 * @param stream Stream to enqueue on; nullptr means the default stream.
 * @throws ckl::Error when m is out of range.
 * @note Enqueued on the stream, so it is ordered with the launches around it. A
 *       plan binds at construction, which is outside every timed region.
 */
CKL_EXPORT void conv_direct_constant_bind(const float* filter, int m, const void* owner,
                                          cudaStream_t stream = nullptr);

/**
 * @brief Whether this owner still holds the constant bank.
 * @param owner The object to test.
 * @return True when the last successful bind was this owner's.
 */
CKL_EXPORT bool conv_direct_constant_bound(const void* owner);

/**
 * @brief Largest filter the constant memory rung takes.
 * @return 4096 taps, which is 16 KB, the budget the spec sets for this rung.
 */
CKL_EXPORT int conv_direct_constant_max_taps();

/**
 * @brief cuFFT baseline for one complex transform, planned once per size and reused.
 * @param in Device pointer to n * batch complex inputs.
 * @param out Device pointer to n * batch complex outputs.
 * @param n Transform length.
 * @param batch Transforms laid out back to back.
 * @param dir Forward or inverse; cuFFT does not scale either.
 * @param stream Stream to enqueue on; nullptr means the default stream.
 * @note Runs on the process wide default Context and takes its lock, through a one
 *       entry plan cache keyed on the length, the batch and the direction. A
 *       benchmark should build its own ckl::FftPlan and call ckl::fft with
 *       FftAlgo::kCufft instead: the cache holds one plan, so alternating between
 *       two sizes rebuilds on every call, and plan creation inside a timed region
 *       is exactly the defect a plan exists to remove.
 */
CKL_EXPORT void fft_cufft(const float2* in, float2* out, int n, int batch, FftDirection dir,
                          cudaStream_t stream = nullptr);

}  // namespace ckl
