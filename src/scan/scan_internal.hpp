#pragma once

// The plan's execution entry points, shared between ckl_scan and ckl_api.
//
// ckl::reduce and ckl::scan live in src/api/dispatch.cpp because what they do is
// dispatch: validate, resolve kAuto, write chosen, and turn a thrown ckl::Error
// into a Status with a thread local message. What they dispatch to lives here,
// because running a variant needs the plan's private state: the partials, the
// level buffers, the tile status array, the tile counters and the CUB temporary
// storage.
//
// This header is not installed. It is an implementation detail of the split
// between the two libraries, not part of the public API. The split is what keeps
// ckl_api to ckl_scan acyclic, which a shared build requires.

#include <cuda_runtime.h>

#include "ckl/ckl_export.h"
#include "ckl/scan.hpp"

namespace ckl {
namespace detail {

/**
 * @brief Runs one already resolved reduction rung out of a plan.
 * @param plan Plan holding the workspace every rung runs out of.
 * @param algo The rung to run; never ReduceAlgo::kAuto, which the caller resolves.
 * @param op Binary operator to combine with.
 * @param in Device pointer to plan.size() elements.
 * @param out Device pointer to one element.
 * @param stream Stream to enqueue on.
 * @throws ckl::Error when the plan refuses the rung or a launch fails. It never
 *         returns a status; the caller maps the throw.
 */
CKL_EXPORT void reduce_launch(ScanPlan& plan, ReduceAlgo algo, ScanOp op, const void* in, void* out,
                              cudaStream_t stream);

/**
 * @brief Runs one already resolved scan rung out of a plan.
 * @param plan Plan holding the workspace every rung runs out of.
 * @param algo The rung to run; never ScanAlgo::kAuto, which the caller resolves.
 * @param op Binary operator to combine with.
 * @param exclusive True for an exclusive scan, false for an inclusive one.
 * @param in Device pointer to plan.size() elements.
 * @param out Device pointer to plan.size() elements.
 * @param stream Stream to enqueue on.
 * @throws ckl::Error when the plan refuses the rung or a launch fails.
 */
CKL_EXPORT void scan_launch(ScanPlan& plan, ScanAlgo algo, ScanOp op, bool exclusive,
                            const void* in, void* out, cudaStream_t stream);

}  // namespace detail
}  // namespace ckl
