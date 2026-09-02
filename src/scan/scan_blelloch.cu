// s1: the work efficient block scan, upsweep and downsweep, with bank conflict
// padding.
//
// Blelloch's scan does 2 * BLOCK combines where Hillis-Steele does BLOCK
// log2(BLOCK): an upsweep that builds a reduction tree in place, and a downsweep
// that turns that tree into an exclusive scan by swapping and combining at every
// node. The work is asymptotically better and the barrier count is worse, twice
// log2(BLOCK) instead of once, which is why the two rungs sit next to each other
// on the ladder rather than one replacing the other.
//
// Both sweeps address shared memory at a stride that doubles at every level, so
// without padding level d has every active thread hitting the same bank. The
// index map is i + i/32, which pushes each successive group of 32 elements one
// slot further along and breaks the alignment the conflict needs. docs/scan.md
// carries the ncu l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld page with
// the padding and without it.

#include "scan_device.cuh"
#include "scan_levels.cuh"
#include "scan_rungs.hpp"

namespace ckl {
namespace detail {

template <typename T, ScanOp OP>
void scan_blelloch_launch(const T* in, T* out, long long n, bool exclusive, T* levels,
                          long long levels_capacity, cudaStream_t stream) {
    using Op = typename OpOf<T, OP>::type;
    LevelDriver<T, Op, 256, 1, BlellochBlockScan>::run(in, out, n, exclusive, levels,
                                                       levels_capacity, stream);
}

#define CKL_INSTANTIATE(T, OP)                                                              \
    template void scan_blelloch_launch<T, OP>(const T*, T*, long long, bool, T*, long long, \
                                              cudaStream_t);
CKL_SCAN_SCAN_INSTANCES(CKL_INSTANTIATE)
#undef CKL_INSTANTIATE

}  // namespace detail
}  // namespace ckl
