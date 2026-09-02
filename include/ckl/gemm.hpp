#pragma once

/**
 * @file gemm.hpp
 * @brief Public GEMM interface: the descriptor driven entry points and the ladder rungs.
 *
 * Two layers live here. The descriptor driven ckl::gemm is the primary entry
 * point: it takes a Context, validates the shape, checks the device against
 * what the selected path needs, dispatches, and reports the path it took
 * through the chosen out-param. Below it sit the v1 free functions, one per
 * ladder rung, which every benchmark and correctness test drives directly.
 *
 * Free function convention: row major, single precision unless a variant says
 * otherwise.
 *   - A is m by k, leading dimension k
 *   - B is k by n, leading dimension n
 *   - C is m by n, leading dimension n
 *   - C = alpha * (A * B) + beta * C
 *
 * Row major is the natural indexing for hand written kernels. The cuBLAS oracle
 * (column major) is wrapped so it produces the identical row major C, letting
 * every performance claim be a same shape, same process comparison.
 *
 * BLAS contract on beta: when beta is zero, C is not read. An uninitialized or
 * NaN C is legal input in that case, and every kernel here honors it.
 *
 * Every entry point in this file is asynchronous with respect to the host: it
 * enqueues work on the stream it is given and returns. Synchronize on that
 * stream before reading C.
 */

#include <cstddef>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "ckl/ckl_export.h"
#include "ckl/context.hpp"
#include "ckl/status.hpp"
#include "ckl/types.hpp"

namespace ckl {

// ---------------------------------------------------------------------------
// Descriptor driven entry points
// ---------------------------------------------------------------------------

/**
 * @brief C = alpha * op(A) * op(B) + beta * C as described by desc.
 *
 * @param ctx Context supplying the stream, the vendor handles and the device capability.
 * @param desc The shape, the layout, the types and the requested algorithm.
 * @param alpha Host pointer to a float scale on the product; must not be null.
 * @param a Device pointer to A, or null only when the contraction is empty.
 * @param b Device pointer to B, or null only when the contraction is empty.
 * @param beta Host pointer to a float scale on the incoming C; must not be null.
 * @param c Device pointer to C, written in place.
 * @param chosen Optional out-param receiving the path taken. When non-null it is
 *        always written, on success and on failure alike, so Algo::kAuto never
 *        resolves silently.
 * @return Status::kSuccess, or Status::kInvalidValue for a malformed descriptor
 *         (bad dimensions, leading dimensions too small, null buffers),
 *         Status::kArchMismatch when the running device cannot execute the
 *         selected path, or Status::kNotSupported when an explicitly named
 *         algorithm cannot take the shape.
 * @note An explicit algorithm is never rerouted. Only Algo::kAuto is allowed to
 *       choose again, and chosen reports where it landed.
 * @note Asynchronous. The call returns once the work is enqueued on ctx.stream().
 * @note alpha and beta are host values. A device pointer mode is a documented
 *       non goal until CUDA graph capture support lands.
 */
CKL_EXPORT Status gemm(Context& ctx, const GemmDesc& desc, const void* alpha, const void* a,
                       const void* b, const void* beta, void* c, Algo* chosen = nullptr);

/**
 * @brief What Algo::kAuto would pick for this descriptor.
 * @param ctx Context whose device capability gates the answer.
 * @param desc The shape to plan for.
 * @return The algorithm the dispatcher would run, which is Algo::kCublas for any
 *         shape the hand written kernels cannot take.
 * @note Launches nothing and touches no operand.
 */
CKL_EXPORT Algo gemm_query(const Context& ctx, const GemmDesc& desc);

/**
 * @brief The whole dispatch decision, including which tile of the family would run.
 * @param ctx Context whose device capability and SM count feed the heuristic.
 * @param desc The shape to plan for.
 * @return The rung, the block tile, the K split count and whether a committed
 *         tile sweep informed the answer.
 * @note ckl::gemm_query returns the rung alone and is the stable short answer;
 *       this is the same decision with the tile family detail attached.
 * @note Launches nothing and touches no operand.
 */
CKL_EXPORT GemmPlan gemm_plan(const Context& ctx, const GemmDesc& desc);

/**
 * @brief Bytes of scratch the descriptor needs.
 * @param ctx Context the call would run on.
 * @param desc The shape to size for.
 * @return The scratch the planned path asks for: the split-K partial planes or
 *         the stream-K peer buffers when the plan lands on one of those, and
 *         zero for every other path.
 * @note Hand the answer to Context::set_workspace. A Context with no workspace,
 *       or one too small, makes the split-K and stream-K drivers allocate and
 *       free their own scratch around the launch, which costs an allocation the
 *       caller could have avoided.
 */
CKL_EXPORT std::size_t gemm_workspace_size(const Context& ctx, const GemmDesc& desc);

// ---------------------------------------------------------------------------
// Ladder rungs
// ---------------------------------------------------------------------------

/**
 * @brief Naive FP32 GEMM: one thread per output element, natural indexing.
 * @param a Device pointer to A, m by k, row major, leading dimension k.
 * @param b Device pointer to B, k by n, row major, leading dimension n.
 * @param c Device pointer to C, m by n, row major, leading dimension n.
 * @param m Rows of A and C.
 * @param n Columns of B and C.
 * @param k Contraction extent.
 * @param alpha Scale on the product.
 * @param beta Scale on the incoming C; when zero, C is not read.
 * @param stream Stream to enqueue on; nullptr means the default stream.
 * @note This is the honest baseline, not a strawman.
 * @note Asynchronous. Synchronize on stream before reading C.
 */
CKL_EXPORT void gemm_naive(const float* a, const float* b, float* c, int m, int n, int k,
                           float alpha, float beta, cudaStream_t stream = nullptr);

/**
 * @brief Shared memory tiled FP32 GEMM.
 * @param a Device pointer to A, m by k, row major.
 * @param b Device pointer to B, k by n, row major.
 * @param c Device pointer to C, m by n, row major.
 * @param m Rows of A and C.
 * @param n Columns of B and C.
 * @param k Contraction extent.
 * @param alpha Scale on the product.
 * @param beta Scale on the incoming C; when zero, C is not read.
 * @param stream Stream to enqueue on; nullptr means the default stream.
 * @note Reuses each staged tile TILE times, cutting global traffic by the tile
 *       factor over the naive kernel. It is boundary safe, so it is also the
 *       fallback the vectorized rungs use for shapes they cannot take.
 */
CKL_EXPORT void gemm_tiled(const float* a, const float* b, float* c, int m, int n, int k,
                           float alpha, float beta, cudaStream_t stream = nullptr);

/**
 * @brief Register blocked, float4 vectorized FP32 GEMM: 128x128 block, 8x8 register tile.
 * @param a Device pointer to A, m by k, row major.
 * @param b Device pointer to B, k by n, row major.
 * @param c Device pointer to C, m by n, row major.
 * @param m Rows of A and C.
 * @param n Columns of B and C.
 * @param k Contraction extent.
 * @param alpha Scale on the product.
 * @param beta Scale on the incoming C; when zero, C is not read.
 * @param stream Stream to enqueue on; nullptr means the default stream.
 * @note Falls back to gemm_tiled for shapes that do not divide the block factors
 *       (m by 128, n by 128, k by 8), which keeps correctness on odd sizes.
 * @note The float4 loads need every operand 16 byte aligned. cudaMalloc returns
 *       256 byte aligned memory, so a base pointer is fine; an offset into a
 *       buffer has to stay a multiple of four floats.
 */
CKL_EXPORT void gemm_register(const float* a, const float* b, float* c, int m, int n, int k,
                              float alpha, float beta, cudaStream_t stream = nullptr);

/**
 * @brief cp.async double buffered FP32 GEMM.
 * @param a Device pointer to A, m by k, row major.
 * @param b Device pointer to B, k by n, row major.
 * @param c Device pointer to C, m by n, row major.
 * @param m Rows of A and C.
 * @param n Columns of B and C.
 * @param k Contraction extent.
 * @param alpha Scale on the product.
 * @param beta Scale on the incoming C; when zero, C is not read.
 * @param stream Stream to enqueue on; nullptr means the default stream.
 * @note Overlaps the next tile's global to shared copy with the current tile's
 *       math. Same alignment contract and fallback as gemm_register.
 * @note The aligned path needs compute capability 8.0; below that the launcher
 *       throws ckl::Error with Status::kArchMismatch rather than running an
 *       empty kernel.
 */
CKL_EXPORT void gemm_cp_async(const float* a, const float* b, float* c, int m, int n, int k,
                              float alpha, float beta, cudaStream_t stream = nullptr);

/**
 * @brief cuBLAS SGEMM oracle producing the same row major C.
 * @param a Device pointer to A, m by k, row major.
 * @param b Device pointer to B, k by n, row major.
 * @param c Device pointer to C, m by n, row major.
 * @param m Rows of A and C.
 * @param n Columns of B and C.
 * @param k Contraction extent.
 * @param alpha Scale on the product.
 * @param beta Scale on the incoming C.
 * @param stream Stream to enqueue on; nullptr means the default stream.
 * @note Runs on the process wide default Context and takes its lock. Used as
 *       both the correctness oracle and the performance baseline.
 */
CKL_EXPORT void gemm_cublas(const float* a, const float* b, float* c, int m, int n, int k,
                            float alpha, float beta, cudaStream_t stream = nullptr);

/**
 * @brief WMMA tensor core GEMM, FP16 storage in, FP32 accumulate, FP32 out.
 * @param a Device pointer to A, m by k, row major, half precision.
 * @param b Device pointer to B, k by n, row major, half precision.
 * @param c Device pointer to C, m by n, row major, single precision.
 * @param m Rows of A and C.
 * @param n Columns of B and C.
 * @param k Contraction extent.
 * @param alpha Scale on the product.
 * @param beta Scale on the incoming C; when zero, C is not read.
 * @param stream Stream to enqueue on; nullptr means the default stream.
 * @note 16x16x16 fragments. The aligned fast path (m by 64, n by 64, k by 16)
 *       needs compute capability 8.0; other shapes take a scalar half input
 *       fallback that runs anywhere.
 */
CKL_EXPORT void gemm_wmma_fp16(const __half* a, const __half* b, float* c, int m, int n, int k,
                               float alpha, float beta, cudaStream_t stream = nullptr);

/**
 * @brief WMMA tensor core GEMM, BF16 storage in, FP32 accumulate, FP32 out.
 * @param a Device pointer to A, m by k, row major, bfloat16.
 * @param b Device pointer to B, k by n, row major, bfloat16.
 * @param c Device pointer to C, m by n, row major, single precision.
 * @param m Rows of A and C.
 * @param n Columns of B and C.
 * @param k Contraction extent.
 * @param alpha Scale on the product.
 * @param beta Scale on the incoming C; when zero, C is not read.
 * @param stream Stream to enqueue on; nullptr means the default stream.
 * @note Same fragment shape, alignment contract and fallback as gemm_wmma_fp16.
 */
CKL_EXPORT void gemm_wmma_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b, float* c, int m,
                               int n, int k, float alpha, float beta,
                               cudaStream_t stream = nullptr);

/**
 * @brief mma.sync PTX tensor core GEMM with scalar shared fragment reads.
 * @param a Device pointer to A, m by k, row major, half precision.
 * @param b Device pointer to B, k by n, row major, half precision.
 * @param c Device pointer to C, m by n, row major, single precision.
 * @param m Rows of A and C.
 * @param n Columns of B and C.
 * @param k Contraction extent.
 * @param alpha Scale on the product.
 * @param beta Scale on the incoming C; when zero, C is not read.
 * @param stream Stream to enqueue on; nullptr means the default stream.
 * @note FP16 storage, FP32 accumulate. Same alignment contract as the WMMA fast
 *       path (m by 64, n by 64, k by 16), and the same compute capability 8.0 floor.
 */
CKL_EXPORT void gemm_mma_ptx(const __half* a, const __half* b, float* c, int m, int n, int k,
                             float alpha, float beta, cudaStream_t stream = nullptr);

/**
 * @brief mma.sync with ldmatrix fragment loads.
 * @param a Device pointer to A, m by k, row major, half precision.
 * @param b Device pointer to B, k by n, row major, half precision.
 * @param c Device pointer to C, m by n, row major, single precision.
 * @param m Rows of A and C.
 * @param n Columns of B and C.
 * @param k Contraction extent.
 * @param alpha Scale on the product.
 * @param beta Scale on the incoming C; when zero, C is not read.
 * @param stream Stream to enqueue on; nullptr means the default stream.
 * @note The optimized tensor variant that replaces scalar shared loads with one
 *       swizzled ldmatrix per fragment.
 */
CKL_EXPORT void gemm_mma_ldm(const __half* a, const __half* b, float* c, int m, int n, int k,
                             float alpha, float beta, cudaStream_t stream = nullptr);

/**
 * @brief The top tensor kernel: 128x128 tile, ldmatrix, cp.async double buffering.
 * @param a Device pointer to A, m by k, row major, half precision.
 * @param b Device pointer to B, k by n, row major, half precision.
 * @param c Device pointer to C, m by n, row major, single precision.
 * @param m Rows of A and C.
 * @param n Columns of B and C.
 * @param k Contraction extent.
 * @param alpha Scale on the product.
 * @param beta Scale on the incoming C; when zero, C is not read.
 * @param stream Stream to enqueue on; nullptr means the default stream.
 * @note The FP16 kernel driven toward the compute bound gate. The fast path needs
 *       m and n divisible by 128 and k divisible by 32, and c 8-byte aligned
 *       because the epilogue issues 64-bit stores; a misaligned c throws
 *       Status::kInvalidValue.
 */
CKL_EXPORT void gemm_mma_opt(const __half* a, const __half* b, float* c, int m, int n, int k,
                             float alpha, float beta, cudaStream_t stream = nullptr);

/**
 * @brief Fusion study: the top kernel with a per column bias add and ReLU in the epilogue.
 * @param a Device pointer to A, m by k, row major, half precision.
 * @param b Device pointer to B, k by n, row major, half precision.
 * @param c Device pointer to C, m by n, row major, single precision, written not read.
 * @param bias Device pointer to a length n per column bias vector.
 * @param m Rows of A and C.
 * @param n Columns of B and C.
 * @param k Contraction extent.
 * @param alpha Scale on the product.
 * @param stream Stream to enqueue on; nullptr means the default stream.
 * @throws ckl::Error with Status::kNotSupported for an unaligned shape.
 * @note Computes C = relu(alpha * A*B + bias) in one pass, against the unfused
 *       path of a plain GEMM followed by gemm_bias_relu. It takes aligned shapes
 *       only and throws rather than returning, because a return that writes no C
 *       is a silent wrong answer. c must be 8-byte aligned for the 64-bit store
 *       epilogue; a misaligned c throws Status::kInvalidValue.
 */
CKL_EXPORT void gemm_mma_opt_bias(const __half* a, const __half* b, float* c, const float* bias,
                                  int m, int n, int k, float alpha, cudaStream_t stream = nullptr);

/**
 * @brief The unfused half of the fusion study: a memory bound pass adding bias and applying ReLU.
 * @param c Device pointer to C, m by n, row major, updated in place.
 * @param bias Device pointer to a length n per column bias vector.
 * @param m Rows of C.
 * @param n Columns of C.
 * @param stream Stream to enqueue on; nullptr means the default stream.
 */
CKL_EXPORT void gemm_bias_relu(float* c, const float* bias, int m, int n,
                               cudaStream_t stream = nullptr);

// ---------------------------------------------------------------------------
// The CUTLASS reference line
// ---------------------------------------------------------------------------

/**
 * @brief Whether the CUTLASS rung can run this shape.
 * @param m Rows of A and C.
 * @param n Columns of B and C.
 * @param k Contraction extent.
 * @return True when the shape meets the vector access rule below.
 * @note Row major A has leading dimension k and row major B and C have leading
 *       dimension n, and the mainloop moves 16 bytes per access, so n and k have
 *       to be divisible by 8. m is free: the M edge is predicated.
 */
CKL_EXPORT bool gemm_cutlass_supports(int m, int n, int k);

/**
 * @brief The block tile, warp split and K step the CUTLASS rung is instantiated at.
 * @return The threadblock shape and the warp counts along M and N.
 * @note Reported from the template parameters themselves rather than from a
 *       comment, so docs and benchmark rows cannot drift from what was compiled.
 */
CKL_EXPORT GemmTile gemm_cutlass_tile();

/**
 * @brief The CUTLASS reference line: an Ampere style multistage FP16 tensor GEMM.
 * @param a Device pointer to A, m by k, row major, half precision.
 * @param b Device pointer to B, k by n, row major, half precision.
 * @param c Device pointer to C, m by n, row major, single precision.
 * @param m Rows of A and C.
 * @param n Columns of B and C.
 * @param k Contraction extent.
 * @param alpha Scale on the product.
 * @param beta Scale on the incoming C; when zero, C is not read.
 * @param stream Stream to enqueue on; nullptr means the default stream.
 * @throws ckl::Error with Status::kNotSupported when gemm_cutlass_supports is
 *         false for the shape, or when the device is below compute capability
 *         8.0.
 * @note cutlass::gemm::device::GemmUniversal over arch::Sm80, OpClassTensorOp,
 *       a 128x128x32 threadblock tile, a 64x64x32 warp tile, instruction shape
 *       16x8x16 and a three stage cp.async mainloop. The shape it refuses it
 *       refuses; it is never rerouted to another rung, because a CUTLASS row
 *       that measured a hand kernel would be worse than no row at all.
 */
CKL_EXPORT void gemm_cutlass(const __half* a, const __half* b, float* c, int m, int n, int k,
                             float alpha, float beta, cudaStream_t stream = nullptr);

// ---------------------------------------------------------------------------
// The tile family, split-K and stream-K
// ---------------------------------------------------------------------------

/**
 * @brief How many block tile shapes the family carries.
 * @return The shape count; valid tile indices are 0 to the count minus one.
 */
CKL_EXPORT int gemm_tile_family_count();

/**
 * @brief The block tile at one index of the family.
 * @param index Family index, 0 to gemm_tile_family_count() minus one.
 * @return The tile extents and warp layout, or an all zero shape for an index
 *         outside the family.
 */
CKL_EXPORT GemmTile gemm_tile_family_shape(int index);

/**
 * @brief Blocks of this tile the occupancy API says fit on one SM.
 * @param index Family index.
 * @return The occupancy answer, queried once per shape per process and cached,
 *         or zero when the running device cannot host the shape at all.
 * @note This is the denominator of the wave quantization term the dispatch
 *       heuristic maximizes, which is why it comes from the API rather than from
 *       an assumption about registers.
 */
CKL_EXPORT int gemm_tile_family_blocks_per_sm(int index);

/**
 * @brief One tile shape of the family, run over the whole shape.
 * @param a Device pointer to A, m by k, row major, half precision.
 * @param b Device pointer to B, k by n, row major, half precision.
 * @param c Device pointer to C, m by n, row major, single precision.
 * @param m Rows of A and C.
 * @param n Columns of B and C.
 * @param k Contraction extent.
 * @param alpha Scale on the product.
 * @param beta Scale on the incoming C; when zero, C is not read.
 * @param tile_index Family index naming the block tile to run.
 * @param stream Stream to enqueue on; nullptr means the default stream.
 * @throws ckl::Error with Status::kInvalidValue for a tile index outside the
 *         family, or Status::kArchMismatch below compute capability 8.0.
 * @note Any shape runs: an m, n or k that does not divide the tile takes the
 *       predicated path, which masks the edges of the same mainloop rather than
 *       rerouting to a scalar kernel.
 * @note Alignment changes which instructions run, never whether the call works.
 *       The wide path wants both leading dimensions a multiple of 8 elements,
 *       a and b 16-byte aligned for its cp.async stages, and c 8-byte aligned
 *       for its 64-bit stores; anything else stages and stores narrower and
 *       still computes the same C.
 */
CKL_EXPORT void gemm_tile_family(const __half* a, const __half* b, float* c, int m, int n, int k,
                                 float alpha, float beta, int tile_index,
                                 cudaStream_t stream = nullptr);

/**
 * @brief Scratch bytes gemm_split_k needs for this shape.
 * @param m Rows of C.
 * @param n Columns of C.
 * @param k Contraction extent.
 * @param splits Requested K splits.
 * @param tile_index Family index naming the block tile.
 * @return Bytes of device scratch, or zero when the request degenerates to one
 *         split and no partials are needed.
 */
CKL_EXPORT std::size_t gemm_split_k_workspace_size(int m, int n, int k, int splits, int tile_index);

/**
 * @brief K slices gemm_split_k would actually use for this request.
 * @param k Contraction extent.
 * @param splits Requested K splits.
 * @param tile_index Family index naming the block tile.
 * @return The slice count after each slice is rounded up to a whole number of
 *         the tile's K steps, which is at most the request and at least one.
 * @note The request is a request, not a promise: a K that is short relative to
 *       the tile's K step cannot be cut as finely as asked. Reporting the answer
 *       is what keeps ckl::GemmPlan::splits honest.
 */
CKL_EXPORT int gemm_split_k_slices(int k, int splits, int tile_index);

/**
 * @brief The tile family split along K, with a fixup reduction applying alpha and beta once.
 * @param a Device pointer to A, m by k, row major, half precision.
 * @param b Device pointer to B, k by n, row major, half precision.
 * @param c Device pointer to C, m by n, row major, single precision.
 * @param m Rows of A and C.
 * @param n Columns of B and C.
 * @param k Contraction extent.
 * @param alpha Scale on the product.
 * @param beta Scale on the incoming C; when zero, C is not read.
 * @param splits Requested K splits; the driver rounds each split's K extent up
 *        to the tile's K step and may therefore use fewer.
 * @param tile_index Family index naming the block tile.
 * @param workspace Device scratch of at least gemm_split_k_workspace_size bytes,
 *        or nullptr to let the driver allocate and free its own.
 * @param workspace_bytes Size of workspace; ignored when workspace is null.
 * @param stream Stream to enqueue on; nullptr means the default stream.
 * @throws ckl::Error with Status::kInvalidValue for a bad tile index or a
 *         workspace too small for the shape.
 * @note The first pass writes raw partial sums, one m by n plane per split, and
 *       never touches C. The second pass sums the planes and applies alpha and
 *       beta once, so beta still does not read C when it is zero and every
 *       partial is scaled exactly once.
 */
CKL_EXPORT void gemm_split_k(const __half* a, const __half* b, float* c, int m, int n, int k,
                             float alpha, float beta, int splits, int tile_index, void* workspace,
                             std::size_t workspace_bytes, cudaStream_t stream = nullptr);

/**
 * @brief Scratch bytes gemm_stream_k needs for this shape.
 * @param m Rows of C.
 * @param n Columns of C.
 * @param k Contraction extent.
 * @param tile_index Family index naming the block tile.
 * @return Bytes of device scratch, or zero when the shape needs no stream-K
 *         CTAs and the whole grid is data parallel.
 */
CKL_EXPORT std::size_t gemm_stream_k_workspace_size(int m, int n, int k, int tile_index);

/**
 * @brief The tile family as a hybrid stream-K grid: a bounded number of persistent CTAs
 *        covering the K split remainder, the rest data parallel.
 * @param a Device pointer to A, m by k, row major, half precision.
 * @param b Device pointer to B, k by n, row major, half precision.
 * @param c Device pointer to C, m by n, row major, single precision.
 * @param m Rows of A and C.
 * @param n Columns of B and C.
 * @param k Contraction extent.
 * @param alpha Scale on the product.
 * @param beta Scale on the incoming C; when zero, C is not read.
 * @param tile_index Family index naming the block tile.
 * @param workspace Device scratch of at least gemm_stream_k_workspace_size
 *        bytes, or nullptr to let the driver allocate and free its own.
 * @param workspace_bytes Size of workspace; ignored when workspace is null.
 * @param stream Stream to enqueue on; nullptr means the default stream.
 * @throws ckl::Error with Status::kInvalidValue for a bad tile index or a
 *         workspace too small for the shape.
 * @note The stream-K CTAs cover at most two waves of output tiles between them,
 *       so the tail of a grid that does not fill the machine is spread over
 *       every SM instead of leaving most of them idle. A CTA whose share stops
 *       before the end of a tile publishes its partial through the workspace and
 *       raises a flag; the CTA that covers the tile's last K iteration waits on
 *       those peers, sums them in, and writes C once. Waits only ever run
 *       downward in CTA index, so no assumption about co-residency is needed.
 */
CKL_EXPORT void gemm_stream_k(const __half* a, const __half* b, float* c, int m, int n, int k,
                              float alpha, float beta, int tile_index, void* workspace,
                              std::size_t workspace_bytes, cudaStream_t stream = nullptr);

/**
 * @brief cuBLAS tensor core oracle, FP16 in and FP32 accumulate, producing the same row major C.
 * @param a Device pointer to A, m by k, row major, half precision.
 * @param b Device pointer to B, k by n, row major, half precision.
 * @param c Device pointer to C, m by n, row major, single precision.
 * @param m Rows of A and C.
 * @param n Columns of B and C.
 * @param k Contraction extent.
 * @param alpha Scale on the product.
 * @param beta Scale on the incoming C.
 * @param stream Stream to enqueue on; nullptr means the default stream.
 * @note cublasGemmEx. This is the baseline the FP16 rungs are quoted against.
 */
CKL_EXPORT void gemm_cublas_fp16(const __half* a, const __half* b, float* c, int m, int n, int k,
                                 float alpha, float beta, cudaStream_t stream = nullptr);

/**
 * @brief cuBLAS tensor core oracle, BF16 in and FP32 accumulate.
 * @param a Device pointer to A, m by k, row major, bfloat16.
 * @param b Device pointer to B, k by n, row major, bfloat16.
 * @param c Device pointer to C, m by n, row major, single precision.
 * @param m Rows of A and C.
 * @param n Columns of B and C.
 * @param k Contraction extent.
 * @param alpha Scale on the product.
 * @param beta Scale on the incoming C.
 * @param stream Stream to enqueue on; nullptr means the default stream.
 * @note The baseline the BF16 rung is quoted against.
 */
CKL_EXPORT void gemm_cublas_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b, float* c, int m,
                                 int n, int k, float alpha, float beta,
                                 cudaStream_t stream = nullptr);

}  // namespace ckl
