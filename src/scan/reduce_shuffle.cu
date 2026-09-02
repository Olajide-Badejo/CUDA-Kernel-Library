// r2: a shuffle reduction inside the warp, one shared slot per warp, and one
// atomic per block.
//
// The difference from r1 is that the eight round trips through shared memory
// become five register to register shuffles with no barrier between them, and
// the only shared traffic left is one slot per warp. The contention that made r0
// hopeless drops by the block size: one atomic per 256 elements instead of one
// per element, which is few enough that the address stops being the limiter.
// The hypothesis is that this rung is limited by how fast it can issue loads,
// not by shared memory and not by contention.

#include "ckl/cuda_check.hpp"

#include "scan_device.cuh"
#include "scan_rungs.hpp"

namespace ckl {
namespace detail {

namespace {

constexpr int kBlock = 256;

template <typename T, class Op>
__global__ void shuffle_kernel(const T* in, T* out, long long n) {
    __shared__ typename BlockReduce<T, Op, kBlock>::TempStorage tmp;
    const long long i =
        static_cast<long long>(blockIdx.x) * kBlock + static_cast<long long>(threadIdx.x);
    const T value = i < n ? in[i] : Op::identity();
    const T total = BlockReduce<T, Op, kBlock>::run(tmp, value);
    if (threadIdx.x == 0) {
        AtomicCombine<T, Op>::run(out, total);
    }
}

}  // namespace

template <typename T, ScanOp OP>
void reduce_shuffle_launch(const T* in, T* out, long long n, cudaStream_t stream) {
    using Op = typename OpOf<T, OP>::type;
    reduce_fill_identity<T, OP>(out, stream);
    if (n <= 0) {
        return;
    }
    const long long blocks = (n + kBlock - 1) / kBlock;
    shuffle_kernel<T, Op><<<static_cast<unsigned int>(blocks), kBlock, 0, stream>>>(in, out, n);
    CKL_CUDA_LAST_ERROR(false);
}

#define CKL_INSTANTIATE(T, OP) \
    template void reduce_shuffle_launch<T, OP>(const T*, T*, long long, cudaStream_t);
CKL_SCAN_REDUCE_INSTANCES(CKL_INSTANTIATE)
#undef CKL_INSTANTIATE

}  // namespace detail
}  // namespace ckl
