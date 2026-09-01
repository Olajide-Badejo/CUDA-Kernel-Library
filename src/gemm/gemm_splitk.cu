// Split-K: the tile family run over slices of the contraction, then a fixup pass
// that sums the slices and applies alpha and beta once.
//
// The reason to want it is wave quantization. A 1024 by 1024 output at a 128 by
// 128 tile is 64 blocks on a 48 SM part: one full wave and a quarter, so a third
// of the machine idles for most of the call. Splitting K four ways turns the
// same work into 256 blocks, which fills the waves, at the cost of a second pass
// over C sized m by n by the split count.
//
// Two passes rather than atomics into C. Atomics would be one kernel and no
// scratch, but they would also make the result depend on the order the blocks
// happened to retire, which is not something a correctness test can pin down,
// and they would have to read C to apply beta from inside the mainloop. The
// first pass here writes raw partial sums into its own plane and never touches
// C; the second sums the planes, multiplies by alpha once, and adds beta times C
// once. beta zero still does not read C.
//
// The split count the caller asks for is a request, not a promise: each slice is
// rounded up to a whole number of the tile's K steps, so a shape whose K is
// small relative to the step ends up with fewer slices than asked for. The
// workspace query applies the same rounding, so the two never disagree.

#include <cstddef>
#include <string>

#include <cuda_fp16.h>

#include "ckl/context.hpp"
#include "ckl/cuda_check.hpp"
#include "ckl/gemm.hpp"

#include "detail/gemm_tile_registry.hpp"

namespace ckl {

namespace {

constexpr int kMinArch = 80;

// K extent of one slice: the even share, rounded up to a whole number of K
// steps so no slice boundary falls inside a mainloop stage.
int slice_extent(int k, int splits, int bk) {
    if (splits < 1) {
        splits = 1;
    }
    if (k <= 0 || bk <= 0) {
        return 1;
    }
    const long long share = (static_cast<long long>(k) + splits - 1) / splits;
    const long long rounded = ((share + bk - 1) / bk) * bk;
    return static_cast<int>(rounded);
}

int slices_used(int k, int extent) {
    if (k <= 0 || extent <= 0) {
        return 1;
    }
    return (k + extent - 1) / extent;
}

__global__ void splitk_fixup_kernel(float* __restrict__ c, const float* __restrict__ partials,
                                    long long count, int splits, float alpha, float beta) {
    const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= count) {
        return;
    }
    float sum = 0.0f;
    for (int s = 0; s < splits; ++s) {
        sum += partials[static_cast<long long>(s) * count + idx];
    }
    const float v = alpha * sum;
    // C is not read when beta is zero, per the BLAS contract.
    c[idx] = (beta == 0.0f) ? v : v + beta * c[idx];
}

}  // namespace

int gemm_split_k_slices(int k, int splits, int tile_index) {
    const GemmTile tile = gemm_tile_family_shape(tile_index);
    if (tile.k == 0) {
        return 1;
    }
    return slices_used(k, slice_extent(k, splits, tile.k));
}

std::size_t gemm_split_k_workspace_size(int m, int n, int k, int splits, int tile_index) {
    const GemmTile tile = gemm_tile_family_shape(tile_index);
    if (tile.k == 0 || m <= 0 || n <= 0 || k <= 0) {
        return 0;
    }
    const int used = slices_used(k, slice_extent(k, splits, tile.k));
    if (used <= 1) {
        return 0;  // one slice is the plain path, which writes C directly
    }
    return static_cast<std::size_t>(used) * static_cast<std::size_t>(m) *
           static_cast<std::size_t>(n) * sizeof(float);
}

void gemm_split_k(const __half* a, const __half* b, float* c, int m, int n, int k, float alpha,
                  float beta, int splits, int tile_index, void* workspace,
                  std::size_t workspace_bytes, cudaStream_t stream) {
    const GemmTile tile = gemm_tile_family_shape(tile_index);
    if (tile.k == 0) {
        throw Error(
            Status::kInvalidValue,
            "gemm_split_k: tile index " + std::to_string(tile_index) + " is outside the family");
    }
    if (m <= 0 || n <= 0) {
        return;  // no output elements, so nothing to write
    }
    detail::require_arch(kMinArch, "gemm_split_k");

    const int extent = slice_extent(k, splits, tile.k);
    const int used = slices_used(k, extent);
    if (used <= 1 || k <= 0) {
        // One slice is the plain grid; running the two pass machinery for it
        // would cost a whole extra pass over C for nothing.
        const int kk = k > 0 ? k : 0;
        detail::tile_family_launch(tile_index, a, b, c, nullptr, m, n, kk, kk > 0 ? kk : 1, 1,
                                   alpha, beta, stream);
        return;
    }

    const std::size_t need = gemm_split_k_workspace_size(m, n, k, splits, tile_index);
    void* scratch = workspace;
    bool owned = false;
    if (scratch == nullptr) {
        // No caller workspace, so the driver takes one for the duration. It is
        // stream ordered, which keeps it correct on any stream, and it is the
        // reason ckl::gemm_workspace_size exists: a benchmark that wants this
        // allocation out of its timed region asks for the size and hands one in.
        CKL_CUDA_CHECK(cudaMallocAsync(&scratch, need, stream));
        owned = true;
    } else if (workspace_bytes < need) {
        throw Error(Status::kInvalidValue,
                    "gemm_split_k: this shape needs " + std::to_string(need) +
                        " workspace bytes, the caller supplied " + std::to_string(workspace_bytes));
    }

    auto* partials = static_cast<float*>(scratch);
    try {
        detail::tile_family_launch(tile_index, a, b, c, partials, m, n, k, extent, used, 1.0f, 0.0f,
                                   stream);

        const long long count = static_cast<long long>(m) * n;
        constexpr int kBlock = 256;
        const long long blocks = (count + kBlock - 1) / kBlock;
        splitk_fixup_kernel<<<static_cast<unsigned>(blocks), kBlock, 0, stream>>>(
            c, partials, count, used, alpha, beta);
        CKL_CUDA_LAST_ERROR(false);
    } catch (...) {
        if (owned) {
            cudaFreeAsync(scratch, stream);
        }
        throw;
    }
    if (owned) {
        CKL_CUDA_CHECK(cudaFreeAsync(scratch, stream));
    }
}

}  // namespace ckl
