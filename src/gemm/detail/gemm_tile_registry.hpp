#pragma once

// The tile family roster, and the one host entry point the split-K driver needs
// from the translation unit that owns the mainloop instantiations.
//
// The roster is a macro rather than a type list because two translation units
// have to instantiate the same six shapes and stay in lockstep: gemm_tile_family
// builds the data parallel and split-K kernels, gemm_streamk builds the
// persistent ones. Adding a shape in one place adds it in both.
//
// Columns: index, BM, BN, BK, warps in M, warps in N. Every warp tile is 64 by
// 32, which keeps the accumulator at 64 registers per thread across the family
// so the shapes differ in shared memory and quantization rather than in register
// pressure.

#include <cstddef>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define CKL_TILE_FAMILY_FOR_EACH(X) \
    X(0, 128, 128, 32, 2, 4)        \
    X(1, 128, 256, 32, 2, 8)        \
    X(2, 256, 128, 32, 4, 4)        \
    X(3, 128, 64, 64, 2, 2)         \
    X(4, 64, 128, 64, 1, 4)         \
    X(5, 128, 128, 64, 2, 4)

#define CKL_TILE_FAMILY_COUNT 6

namespace ckl {
namespace detail {

// Runs one tile shape over the whole grid.
//
// partials null: the kernel writes C = alpha * acc + beta * C.
// partials non null: it writes the raw accumulator into plane blockIdx.z of a
// splits by m by n scratch buffer, and the caller reduces.
//
// k_chunk is the K extent one split covers, rounded up to a multiple of the
// tile's K step by the caller. splits is the grid depth.
void tile_family_launch(int index, const __half* a, const __half* b, float* c, float* partials,
                        int m, int n, int k, int k_chunk, int splits, float alpha, float beta,
                        cudaStream_t stream);

}  // namespace detail
}  // namespace ckl
