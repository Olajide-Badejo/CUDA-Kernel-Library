// r0: one atomic per element onto a single global accumulator.
//
// The rung exists to be beaten. Every one of N elements issues an atomic to the
// same address, so the whole kernel serializes on one cache line whatever the
// memory system could have delivered, and the hypothesis the sweep tests is that
// it sits one to two orders below the roof. Nothing else on the ladder touches a
// single address from every thread.
//
// The accumulator has to start at the operator's identity, and that initial
// write is a launch of its own enqueued ahead of the atomics rather than a
// branch inside the kernel. The same one element kernel is what an empty input
// writes, which is why it lives here.

#include "ckl/cuda_check.hpp"

#include "scan_device.cuh"
#include "scan_rungs.hpp"

namespace ckl {
namespace detail {

namespace {

template <typename T, class Op>
__global__ void fill_identity_kernel(T* out) {
    *out = Op::identity();
}

template <typename T, class Op>
__global__ void reduce_atomic_kernel(const T* in, T* out, long long n) {
    const long long i =
        static_cast<long long>(blockIdx.x) * blockDim.x + static_cast<long long>(threadIdx.x);
    if (i < n) {
        AtomicCombine<T, Op>::run(out, in[i]);
    }
}

}  // namespace

template <typename T, ScanOp OP>
void reduce_fill_identity(T* out, cudaStream_t stream) {
    using Op = typename OpOf<T, OP>::type;
    fill_identity_kernel<T, Op><<<1, 1, 0, stream>>>(out);
    CKL_CUDA_LAST_ERROR(false);
}

template <typename T, ScanOp OP>
void reduce_atomic_launch(const T* in, T* out, long long n, cudaStream_t stream) {
    using Op = typename OpOf<T, OP>::type;
    reduce_fill_identity<T, OP>(out, stream);
    if (n <= 0) {
        return;
    }
    constexpr int kBlock = 256;
    const long long grid = (n + kBlock - 1) / kBlock;
    reduce_atomic_kernel<T, Op><<<static_cast<unsigned int>(grid), kBlock, 0, stream>>>(in, out, n);
    CKL_CUDA_LAST_ERROR(false);
}

#define CKL_INSTANTIATE(T, OP)                                   \
    template void reduce_fill_identity<T, OP>(T*, cudaStream_t); \
    template void reduce_atomic_launch<T, OP>(const T*, T*, long long, cudaStream_t);
CKL_SCAN_REDUCE_INSTANCES(CKL_INSTANTIATE)
#undef CKL_INSTANTIATE

}  // namespace detail
}  // namespace ckl
