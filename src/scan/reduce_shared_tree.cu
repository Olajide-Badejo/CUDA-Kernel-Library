// r1: a shared memory tree per block, one partial per block, then a combine.
//
// One element per thread, an explicit halving tree in shared memory, and a
// barrier between every level. That is log2(BLOCK) barriers and log2(BLOCK)
// round trips through shared memory to combine 256 values, where the next rung
// does the same work inside a warp with no barriers at all. The hypothesis is
// that this rung is barrier and shared bandwidth bound rather than DRAM bound,
// and the ncu page docs/scan.md asks for is the one that decides it.
//
// The second launch combines the per block partials in a single block, so no
// atomic is involved on this rung. That keeps the two halves of the cost
// separable: whatever this rung is limited by, it is not contention.

#include "ckl/cuda_check.hpp"

#include "scan_device.cuh"
#include "scan_rungs.hpp"

namespace ckl {
namespace detail {

namespace {

constexpr int kBlock = 256;

template <typename T, class Op>
__global__ void shared_tree_partials(const T* in, T* partials, long long n) {
    __shared__ T s[kBlock];
    const int tid = static_cast<int>(threadIdx.x);
    const long long i = static_cast<long long>(blockIdx.x) * kBlock + tid;
    s[tid] = i < n ? in[i] : Op::identity();
    __syncthreads();
    for (int half = kBlock / 2; half > 0; half >>= 1) {
        if (tid < half) {
            s[tid] = Op::apply(s[tid], s[tid + half]);
        }
        __syncthreads();
    }
    if (tid == 0) {
        partials[blockIdx.x] = s[0];
    }
}

// One block over however many partials there are. A grid stride keeps the
// launch geometry independent of the input length, and the same tree combines
// what each thread accumulated.
template <typename T, class Op>
__global__ void shared_tree_combine(const T* partials, T* out, long long count) {
    __shared__ T s[kBlock];
    const int tid = static_cast<int>(threadIdx.x);
    T acc = Op::identity();
    for (long long i = tid; i < count; i += kBlock) {
        acc = Op::apply(acc, partials[i]);
    }
    s[tid] = acc;
    __syncthreads();
    for (int half = kBlock / 2; half > 0; half >>= 1) {
        if (tid < half) {
            s[tid] = Op::apply(s[tid], s[tid + half]);
        }
        __syncthreads();
    }
    if (tid == 0) {
        *out = s[0];
    }
}

}  // namespace

template <typename T, ScanOp OP>
void reduce_shared_tree_launch(const T* in, T* out, long long n, T* partials, cudaStream_t stream) {
    using Op = typename OpOf<T, OP>::type;
    if (n <= 0) {
        reduce_fill_identity<T, OP>(out, stream);
        return;
    }
    const long long blocks = (n + kBlock - 1) / kBlock;
    shared_tree_partials<T, Op>
        <<<static_cast<unsigned int>(blocks), kBlock, 0, stream>>>(in, partials, n);
    CKL_CUDA_LAST_ERROR(false);
    shared_tree_combine<T, Op><<<1, kBlock, 0, stream>>>(partials, out, blocks);
    CKL_CUDA_LAST_ERROR(false);
}

#define CKL_INSTANTIATE(T, OP) \
    template void reduce_shared_tree_launch<T, OP>(const T*, T*, long long, T*, cudaStream_t);
CKL_SCAN_REDUCE_INSTANCES(CKL_INSTANTIATE)
#undef CKL_INSTANTIATE

}  // namespace detail
}  // namespace ckl
