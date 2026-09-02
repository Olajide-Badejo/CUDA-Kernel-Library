// s0: the Hillis-Steele block scan.
//
// The shortest correct block scan there is: log2(BLOCK) passes, every thread
// active in every pass, every pass through shared memory with a barrier on each
// side. That is O(BLOCK log BLOCK) combines to produce O(BLOCK) results, and
// naming the rung work inefficient is the point of having it. It stays on the
// ladder for two reasons. It fixes the barrier and shared memory floor the later
// rungs are measured against, so the s1 and s2 rows mean something. And it is the
// oracle for the block primitives: it is short enough to check by reading, so
// when a Blelloch downsweep or a warp scan disagrees with it the disagreement is
// in the clever one.
//
// It reaches device wide through the same scan then propagate recursion s2 uses,
// with one item per thread and no shared memory transpose, because one item per
// thread already loads coalesced. So the only thing that separates this row from
// the s1 and s2 rows is the block primitive and the tile width, which is what a
// ladder is for.

#include "scan_device.cuh"
#include "scan_levels.cuh"
#include "scan_rungs.hpp"

namespace ckl {
namespace detail {

template <typename T, ScanOp OP>
void scan_hillis_steele_launch(const T* in, T* out, long long n, bool exclusive, T* levels,
                               long long levels_capacity, cudaStream_t stream) {
    using Op = typename OpOf<T, OP>::type;
    LevelDriver<T, Op, 256, 1, HillisSteeleBlockScan>::run(in, out, n, exclusive, levels,
                                                           levels_capacity, stream);
}

#define CKL_INSTANTIATE(T, OP)                                                                   \
    template void scan_hillis_steele_launch<T, OP>(const T*, T*, long long, bool, T*, long long, \
                                                   cudaStream_t);
CKL_SCAN_SCAN_INSTANCES(CKL_INSTANTIATE)
#undef CKL_INSTANTIATE

}  // namespace detail
}  // namespace ckl
