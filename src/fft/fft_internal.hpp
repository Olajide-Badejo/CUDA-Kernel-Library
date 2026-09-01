#pragma once

// Pieces of the FFT family that cross a translation unit inside src/fft but are
// not part of the public API, plus the execution entry points shared between
// ckl_fft and ckl_api.
//
// ckl::fft, ckl::fft2d and ckl::conv live in src/api/dispatch.cpp because what
// they do is dispatch: validate, resolve kAuto, write chosen, and turn a thrown
// ckl::Error into a Status with a thread local message. What they dispatch to
// lives here, because running a variant needs the plan's private state: the
// twiddle tables, the ping pong workspace, the cached filter spectrum and the
// cuFFT handle.
//
// This header is not installed. It is an implementation detail of the split
// between the two libraries, not part of the public API. The split is what keeps
// ckl_api to ckl_fft acyclic, which a shared build requires.

#include <cuda_runtime.h>

#include "ckl/ckl_export.h"
#include "ckl/fft.hpp"

namespace ckl {
namespace detail {

/**
 * @brief Runs one already resolved 1D transform variant out of a plan.
 * @param plan Plan holding the tables, the workspace and the cuFFT handle.
 * @param algo The path to run; never FftAlgo::kAuto, which the caller resolves.
 * @param dir Forward or inverse.
 * @param in Device pointer to the input.
 * @param out Device pointer to the output.
 * @param stream Stream to enqueue on.
 * @throws ckl::Error when the plan refuses the variant or a launch fails.
 */
CKL_EXPORT void fft_launch(FftPlan& plan, FftAlgo algo, FftDirection dir, const float2* in,
                           float2* out, cudaStream_t stream);

/**
 * @brief Runs the real to complex transform out of a plan.
 * @param plan Plan holding the tables and the workspace.
 * @param algo The packed complex path to run; never FftAlgo::kAuto.
 * @param in Device pointer to 2 * n * batch reals.
 * @param out Device pointer to (n + 1) * batch complex bins.
 * @param stream Stream to enqueue on.
 * @throws ckl::Error when the plan refuses the variant or a launch fails.
 */
CKL_EXPORT void fft_r2c_launch(FftPlan& plan, FftAlgo algo, const float* in, float2* out,
                               cudaStream_t stream);

/**
 * @brief Runs the complex to real transform out of a plan.
 * @param plan Plan holding the tables and the workspace.
 * @param algo The packed complex path to run; never FftAlgo::kAuto.
 * @param in Device pointer to (n + 1) * batch complex bins.
 * @param out Device pointer to 2 * n * batch reals.
 * @param stream Stream to enqueue on.
 * @throws ckl::Error when the plan refuses the variant or a launch fails.
 */
CKL_EXPORT void fft_c2r_launch(FftPlan& plan, FftAlgo algo, const float2* in, float* out,
                               cudaStream_t stream);

/**
 * @brief Runs the row column 2D transform out of a plan.
 * @param plan Plan holding the two 1D plans and the workspace.
 * @param dir Forward or inverse.
 * @param in Device pointer to rows * cols complex inputs.
 * @param out Device pointer to rows * cols complex outputs.
 * @param stream Stream to enqueue on.
 * @throws ckl::Error when a launch fails.
 */
CKL_EXPORT void fft2d_launch(Fft2dPlan& plan, FftDirection dir, const float2* in, float2* out,
                             cudaStream_t stream);

/**
 * @brief Runs one already resolved convolution variant out of a plan.
 * @param plan Plan holding the cached filter spectrum and the transform plan.
 * @param algo The path to run; never ConvAlgo::kAuto.
 * @param signal Device pointer to the signal.
 * @param out Device pointer to the output.
 * @param stream Stream to enqueue on.
 * @throws ckl::Error when the plan refuses the variant or a launch fails.
 */
CKL_EXPORT void conv_launch(ConvPlan& plan, ConvAlgo algo, const float* signal, float* out,
                            cudaStream_t stream);

/**
 * @brief Grants the shared resident kernels the dynamic shared memory they need.
 * @param bytes Dynamic shared memory one block will ask for.
 * @throws ckl::Error when the device refuses the opt in.
 * @note Above 48 KB a block has to opt in through cudaFuncSetAttribute, and that
 *       is a driver call, so a plan does it once at construction rather than the
 *       launcher doing it on every call inside a timed region. Repeated calls
 *       with the same or a smaller budget do nothing.
 */
CKL_EXPORT void fft_shared_prepare(int bytes);

/**
 * @brief The shared resident transform with strides, so a column pass needs no transpose.
 * @param in Device pointer to the input.
 * @param out Device pointer to the output.
 * @param tw Twiddle tables of length n.
 * @param n Transform length, at or below the shared resident bound.
 * @param batch Transforms per launch, one block each.
 * @param dir Forward or inverse.
 * @param in_stride Elements between consecutive points of one input transform.
 * @param in_batch_stride Elements between the first points of consecutive transforms.
 * @param out_stride Elements between consecutive points of one output transform.
 * @param out_batch_stride Elements between the first points of consecutive outputs.
 * @param stream Stream to enqueue on.
 * @note This is what FftTranspose::kStridedShared means inside the 2D driver: the
 *       column pass reads its column straight into shared memory, so the 2D
 *       transform runs in two passes instead of a row pass, a transpose, a row
 *       pass and a transpose.
 */
CKL_EXPORT void fft_shared_strided(const float2* in, float2* out, const FftTwiddles& tw, int n,
                                   int batch, FftDirection dir, int in_stride,
                                   long long in_batch_stride, int out_stride,
                                   long long out_batch_stride, cudaStream_t stream);

/**
 * @brief Fills a factored twiddle table on the device with __sincosf.
 * @param hi Device pointer to hi_len complex values.
 * @param lo Device pointer to lo_len complex values.
 * @param hi_len Length of the coarse table.
 * @param lo_len Length of the fine table.
 * @param n The length the tables are a decomposition of.
 * @param stream Stream to enqueue on.
 * @note This is the fast_twiddles rung and nothing else uses it. __sincosf gives
 *       roughly 2^-21 relative accuracy, which is about sixteen times worse than
 *       an FP32 rounding of a double precision sine, and that shows up in the
 *       round trip number. The default path computes the table in double on the
 *       host and rounds it once.
 */
CKL_EXPORT void fft_fill_twiddles_fast(float2* hi, float2* lo, int hi_len, int lo_len, int n,
                                       cudaStream_t stream);

/**
 * @brief The stage plan of a mixed radix ladder.
 * @param log2n log2 of the transform length.
 * @param max_radix The widest butterfly the ladder may use: 2, 4 or 8.
 * @param radix_out Buffer of at least 32 ints receiving the radix of each stage.
 *        May be null when only the count is wanted.
 * @return The number of stages.
 * @note One implementation, used both by the launcher that runs the ladder and by
 *       the traffic model that counts its passes, so the two cannot disagree.
 */
CKL_EXPORT int fft_ladder(int log2n, int max_radix, int* radix_out);

}  // namespace detail
}  // namespace ckl
