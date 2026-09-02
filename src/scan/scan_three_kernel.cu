// s2: device wide scan then propagate over the production block primitive.
//
// The same three step decomposition s0 and s1 reach device wide through, run on
// the block scan the rest of the family uses and on a tile wide enough to keep
// the memory system busy: 256 threads carrying 32 bytes each, loaded as two 16
// byte vectors, transposed through shared memory into the blocked arrangement a
// serial scan needs. The block primitive is a warp scan, a scan of the warp
// totals, and one combine, which is two shuffle rounds and one barrier against
// the eight or sixteen barriers the two rungs below pay.
//
// The traffic model is the honest cost of the decomposition: the tile pass reads
// the input and writes the output, and the propagate pass reads that output and
// writes it again, so about four times n times sizeof(T) crosses DRAM. That is
// twice what the top rung moves, and it is why the ladder does not stop here.
//
// This file also owns the two host helpers the plan sizes its level buffer from,
// because the level recursion is this rung's decomposition and the buffer has to
// be sized for the narrowest tile any rung uses.

#include "scan_device.cuh"
#include "scan_levels.cuh"
#include "scan_rungs.hpp"

namespace ckl {
namespace detail {

int scan_tile_elements(ScanAlgo algo, ScanDType dtype) {
    switch (algo) {
        case ScanAlgo::kHillisSteele:
        case ScanAlgo::kBlelloch:
            // One item per thread, so the tile is the block.
            return 256;
        default:
            break;
    }
    // 32 bytes per thread whatever the element size.
    return 256 * (32 / scan_dtype_size(dtype));
}

long long scan_levels_capacity(long long n, int tile) {
    return levels_capacity_for(n, tile);
}

template <typename T, ScanOp OP>
void scan_three_kernel_launch(const T* in, T* out, long long n, bool exclusive, T* levels,
                              long long levels_capacity, cudaStream_t stream) {
    using Op = typename OpOf<T, OP>::type;
    LevelDriver<T, Op, TileShape<T>::kBlock, TileShape<T>::kItems, WarpBlockScan>::run(
        in, out, n, exclusive, levels, levels_capacity, stream);
}

#define CKL_INSTANTIATE(T, OP)                                                                  \
    template void scan_three_kernel_launch<T, OP>(const T*, T*, long long, bool, T*, long long, \
                                                  cudaStream_t);
CKL_SCAN_SCAN_INSTANCES(CKL_INSTANTIATE)
#undef CKL_INSTANTIATE

}  // namespace detail
}  // namespace ckl
