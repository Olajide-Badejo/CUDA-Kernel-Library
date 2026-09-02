// The bitwise deterministic reduction: fixed tree shape, fixed block count, no
// atomic accumulation, two passes.
//
// Floating point addition is not associative, so a reduction gives a different
// answer when it combines its partials in a different order, and every other
// rung on this ladder lets the launch decide that order. r0 and r2 accumulate
// through an atomic, so the order is whatever the memory system chose that
// millisecond. r3 and r4 size their grid from occupancy, so the order changes
// with the device and with anything else resident on it. None of that is a
// defect; it is what a fast reduction costs. This rung is the one that pays the
// cost the other way round.
//
// Three things make it bit identical. The block count is a pure function of the
// length and nothing else: it never asks the occupancy API, so the same input
// gives the same grid on a busy device, on an idle one, and on a different card.
// The element to thread mapping is therefore fixed, and each thread combines its
// own elements in index order. And the two levels are combined by an explicit
// halving tree, so the shape of the tree is a property of the block size rather
// than of the schedule.
//
// The cost is a benchmark row of its own next to r3 and r4. It is not hidden and
// it is not defended; the sweep says what determinism costs at each length.

#include <algorithm>

#include "ckl/cuda_check.hpp"

#include "scan_device.cuh"
#include "scan_rungs.hpp"

namespace ckl {
namespace detail {

namespace {

constexpr int kBlock = 256;

// An ordered halving tree over one value per thread. Not BlockReduce, which
// combines through warp shuffles: the shuffle tree is just as deterministic, but
// writing the tree out is what makes "fixed tree shape" checkable by reading the
// rung rather than by trusting a shared primitive.
template <typename T, class Op>
__device__ inline T fixed_tree(T* s, T value) {
    const int tid = static_cast<int>(threadIdx.x);
    s[tid] = value;
    __syncthreads();
    for (int half = kBlock / 2; half > 0; half >>= 1) {
        if (tid < half) {
            s[tid] = Op::apply(s[tid], s[tid + half]);
        }
        __syncthreads();
    }
    const T total = s[0];
    __syncthreads();
    return total;
}

template <typename T, class Op>
__global__ void deterministic_partials(const T* in, T* partials, long long n) {
    __shared__ T s[kBlock];
    using Vec = typename Vec16<T>::type;
    constexpr int kLanes = Vec16<T>::kLanes;

    const long long vectors = n / kLanes;
    const long long stride = static_cast<long long>(gridDim.x) * kBlock;
    const long long start =
        static_cast<long long>(blockIdx.x) * kBlock + static_cast<long long>(threadIdx.x);

    T acc = Op::identity();
    const Vec* v = reinterpret_cast<const Vec*>(in);
    for (long long i = start; i < vectors; i += stride) {
        const Vec value = v[i];
#pragma unroll
        for (int j = 0; j < kLanes; ++j) {
            acc = Op::apply(acc, VecAccess<T>::get(value, j));
        }
    }
    for (long long i = vectors * kLanes + start; i < n; i += stride) {
        acc = Op::apply(acc, in[i]);
    }

    const T total = fixed_tree<T, Op>(s, acc);
    if (threadIdx.x == 0) {
        partials[blockIdx.x] = total;
    }
}

template <typename T, class Op>
__global__ void deterministic_combine(const T* partials, T* out, int count) {
    __shared__ T s[kBlock];
    const int tid = static_cast<int>(threadIdx.x);
    T acc = Op::identity();
    for (int i = tid; i < count; i += kBlock) {
        acc = Op::apply(acc, partials[i]);
    }
    const T total = fixed_tree<T, Op>(s, acc);
    if (tid == 0) {
        *out = total;
    }
}

}  // namespace

template <typename T, ScanOp OP>
void reduce_deterministic_launch(const T* in, T* out, long long n, T* partials, int blocks,
                                 cudaStream_t stream) {
    using Op = typename OpOf<T, OP>::type;
    if (n <= 0) {
        reduce_fill_identity<T, OP>(out, stream);
        return;
    }
    // blocks arrives from ScanPlan::deterministic_blocks, which is a pure
    // function of n. Clamping here would make it a function of something else,
    // so it is used as given.
    const int grid = blocks > 0 ? blocks : 1;
    deterministic_partials<T, Op>
        <<<static_cast<unsigned int>(grid), kBlock, 0, stream>>>(in, partials, n);
    CKL_CUDA_LAST_ERROR(false);
    deterministic_combine<T, Op><<<1, kBlock, 0, stream>>>(partials, out, grid);
    CKL_CUDA_LAST_ERROR(false);
}

#define CKL_INSTANTIATE(T, OP)                                                         \
    template void reduce_deterministic_launch<T, OP>(const T*, T*, long long, T*, int, \
                                                     cudaStream_t);
CKL_SCAN_REDUCE_INSTANCES(CKL_INSTANTIATE)
#undef CKL_INSTANTIATE

}  // namespace detail
}  // namespace ckl
