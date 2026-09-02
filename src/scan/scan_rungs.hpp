#pragma once

// The host side face of every rung, and the only thing scan_plan.cpp knows about
// them. Each entry point is a template over the element type and the operator,
// explicitly instantiated at the bottom of the translation unit that defines it,
// so the plan can dispatch on two runtime enums without being a CUDA
// translation unit itself.
//
// Every launcher takes the workspace it needs as a pointer rather than
// allocating one. That is the whole reason ckl::ScanPlan exists: the partials,
// the level buffers, the tile status array, the tile counters and the CUB
// temporary storage are allocated once when the plan is built, and a call
// enqueues launches and nothing else.
//
// This header is not installed and nothing outside src/scan includes it.

#include <cstddef>

#include <cuda_runtime.h>

#include "ckl/scan.hpp"

namespace ckl {
namespace detail {

// ---------------------------------------------------------------------------
// Shared helpers the plan asks for at construction time
// ---------------------------------------------------------------------------

/// Blocks the persistent reduction rungs launch on this device, from
/// cudaOccupancyMaxActiveBlocksPerMultiprocessor times the SM count. Asked once
/// when a plan is built, so no rung makes a driver call inside a timed region.
int reduce_persistent_blocks(ScanDType dtype);

/// Bytes one tile status record occupies, including the alignment padding.
std::size_t lookback_status_bytes_per_tile(ScanDType dtype);

/// Temporary storage CUB wants for the largest of the six calls this family
/// makes at this length and type. Queried once, allocated once, reused.
std::size_t cub_temp_bytes(ScanDType dtype, long long n);

/// CUB_VERSION as the toolkit's header defines it.
int cub_version_macro();

// ---------------------------------------------------------------------------
// Reduction ladder
// ---------------------------------------------------------------------------

/// Writes the operator's identity to out. Used for an empty input, and by the
/// atomic rung to initialise its accumulator before the atomics land on it.
template <typename T, ScanOp OP>
void reduce_fill_identity(T* out, cudaStream_t stream);

/// r0: one atomic per element onto a single global accumulator.
template <typename T, ScanOp OP>
void reduce_atomic_launch(const T* in, T* out, long long n, cudaStream_t stream);

/// r1: shared memory tree per block, one partial per block, then a combine.
template <typename T, ScanOp OP>
void reduce_shared_tree_launch(const T* in, T* out, long long n, T* partials, cudaStream_t stream);

/// r2: warp shuffle inside the warp, one shared slot per warp, one atomic per block.
template <typename T, ScanOp OP>
void reduce_shuffle_launch(const T* in, T* out, long long n, cudaStream_t stream);

/// r3: 16 byte loads and a grid stride loop over persistent blocks.
template <typename T, ScanOp OP>
void reduce_vec4_launch(const T* in, T* out, long long n, T* partials, int blocks,
                        cudaStream_t stream);

/// r4: one launch; the block that arrives last combines the partials.
template <typename T, ScanOp OP>
void reduce_single_pass_launch(const T* in, T* out, long long n, T* partials, unsigned int* counter,
                               int blocks, cudaStream_t stream);

/// r4's comparison arm: the same partials, combined by a second launch.
template <typename T, ScanOp OP>
void reduce_two_pass_launch(const T* in, T* out, long long n, T* partials, int blocks,
                            cudaStream_t stream);

/// Fixed tree, fixed block count, no atomic accumulation.
template <typename T, ScanOp OP>
void reduce_deterministic_launch(const T* in, T* out, long long n, T* partials, int blocks,
                                 cudaStream_t stream);

/// Compensated summation. Sum only; the plan refuses the other operators.
template <typename T, ScanOp OP>
void reduce_kahan_launch(const T* in, T* out, long long n, T* partials, int blocks,
                         cudaStream_t stream);

/// cub::DeviceReduce, the baseline.
template <typename T, ScanOp OP>
void reduce_cub_launch(const T* in, T* out, long long n, void* temp, std::size_t temp_bytes,
                       cudaStream_t stream);

// ---------------------------------------------------------------------------
// Scan ladder
// ---------------------------------------------------------------------------
//
// The four scan then propagate rungs share one signature. `levels` is the plan's
// level buffer, a single allocation the recursion carves its aggregate arrays
// out of, and `levels_capacity` is how many elements of it exist.

/// s0: Hillis-Steele block scan, one item per thread.
template <typename T, ScanOp OP>
void scan_hillis_steele_launch(const T* in, T* out, long long n, bool exclusive, T* levels,
                               long long levels_capacity, cudaStream_t stream);

/// s1: work efficient upsweep and downsweep with bank conflict padding.
template <typename T, ScanOp OP>
void scan_blelloch_launch(const T* in, T* out, long long n, bool exclusive, T* levels,
                          long long levels_capacity, cudaStream_t stream);

/// s2: scan then propagate over the production block scan and 16 byte loads.
template <typename T, ScanOp OP>
void scan_three_kernel_launch(const T* in, T* out, long long n, bool exclusive, T* levels,
                              long long levels_capacity, cudaStream_t stream);

/// s3: reduce then scan, two passes over the input.
template <typename T, ScanOp OP>
void scan_reduce_then_scan_launch(const T* in, T* out, long long n, bool exclusive, T* levels,
                                  long long levels_capacity, cudaStream_t stream);

/// s4: single pass decoupled look-back. The status array and the tile counter
/// are zeroed by a memset enqueued ahead of the kernel, never inside it.
template <typename T, ScanOp OP>
void scan_lookback_launch(const T* in, T* out, long long n, bool exclusive, void* status,
                          long long tiles, unsigned int* counter, cudaStream_t stream);

/// cub::DeviceScan, the baseline.
template <typename T, ScanOp OP>
void scan_cub_launch(const T* in, T* out, long long n, bool exclusive, void* temp,
                     std::size_t temp_bytes, cudaStream_t stream);

/// Elements one tile of a scan rung covers.
int scan_tile_elements(ScanAlgo algo, ScanDType dtype);

/// Elements the level buffer needs for the deepest recursion at this length.
long long scan_levels_capacity(long long n, int tile);

}  // namespace detail
}  // namespace ckl
