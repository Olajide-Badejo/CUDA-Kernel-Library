// r4: one launch instead of two.
//
// Both arms of this rung run the identical partials kernel. What differs is how
// the partials are combined. The single pass arm has every block publish its
// partial, fence, and take a ticket from a global counter; the block that draws
// the last ticket reduces the partials itself before it exits. The two pass arm
// ends the kernel there and combines in a second launch. Everything else is the
// same code, so the difference between the two rows is one kernel launch and
// nothing else, which is the only way the launch overhead question can be
// answered honestly.
//
// The hypothesis is that this changes nothing at large N, where one launch is
// noise against a millisecond of DRAM traffic, and buys the whole of the second
// launch back below about 2^16 elements, where the kernel is shorter than the
// launch. Both halves of that are on the record as rows rather than as a claim.
//
// The tile counter is zeroed by a memset enqueued on the same stream ahead of
// the kernel. Zeroing it inside the kernel would need a block to know it was
// first, which is the question the counter exists to answer.

#include <algorithm>

#include "ckl/cuda_check.hpp"

#include "scan_device.cuh"
#include "scan_rungs.hpp"

namespace ckl {
namespace detail {

namespace {

constexpr int kBlock = 256;

// The shared front half of both arms: a grid stride loop with 16 byte loads,
// reduced to one partial per block.
template <typename T, class Op>
__device__ inline T block_partial(const T* in, long long n,
                                  typename BlockReduce<T, Op, kBlock>::TempStorage& tmp) {
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
    return BlockReduce<T, Op, kBlock>::run(tmp, acc);
}

template <typename T, class Op>
__global__ void partials_only(const T* in, T* partials, long long n) {
    __shared__ typename BlockReduce<T, Op, kBlock>::TempStorage tmp;
    const T total = block_partial<T, Op>(in, n, tmp);
    if (threadIdx.x == 0) {
        partials[blockIdx.x] = total;
    }
}

template <typename T, class Op>
__global__ void combine_only(const T* partials, T* out, int count) {
    __shared__ typename BlockReduce<T, Op, kBlock>::TempStorage tmp;
    const int tid = static_cast<int>(threadIdx.x);
    T acc = Op::identity();
    for (int i = tid; i < count; i += kBlock) {
        acc = Op::apply(acc, partials[i]);
    }
    const T total = BlockReduce<T, Op, kBlock>::run(tmp, acc);
    if (tid == 0) {
        *out = total;
    }
}

template <typename T, class Op>
__global__ void single_pass(const T* in, T* out, long long n, T* partials, unsigned int* counter) {
    __shared__ typename BlockReduce<T, Op, kBlock>::TempStorage tmp;
    __shared__ bool s_last;

    const T total = block_partial<T, Op>(in, n, tmp);
    if (threadIdx.x == 0) {
        partials[blockIdx.x] = total;
    }
    // The partial has to be visible to whichever block draws the last ticket
    // before the ticket is drawn, which is what this fence buys and the only
    // thing it buys.
    __threadfence();
    if (threadIdx.x == 0) {
        const unsigned int ticket = atomicAdd(counter, 1u);
        s_last = ticket == gridDim.x - 1u;
    }
    __syncthreads();
    if (!s_last) {
        return;
    }
    // The matching acquire: everything published before the tickets this block
    // counted has to be visible to the reads below.
    __threadfence();

    const int tid = static_cast<int>(threadIdx.x);
    T acc = Op::identity();
    for (int i = tid; i < static_cast<int>(gridDim.x); i += kBlock) {
        acc = Op::apply(acc, partials[i]);
    }
    const T combined = BlockReduce<T, Op, kBlock>::run(tmp, acc);
    if (tid == 0) {
        *out = combined;
    }
}

int grid_for(long long n, int blocks) {
    const long long wanted = (n + kBlock - 1) / kBlock;
    const long long grid = std::min<long long>(blocks, wanted);
    return static_cast<int>(grid > 0 ? grid : 1);
}

}  // namespace

template <typename T, ScanOp OP>
void reduce_single_pass_launch(const T* in, T* out, long long n, T* partials, unsigned int* counter,
                               int blocks, cudaStream_t stream) {
    using Op = typename OpOf<T, OP>::type;
    if (n <= 0) {
        reduce_fill_identity<T, OP>(out, stream);
        return;
    }
    CKL_CUDA_CHECK(cudaMemsetAsync(counter, 0, sizeof(unsigned int), stream));
    const int grid = grid_for(n, blocks);
    single_pass<T, Op>
        <<<static_cast<unsigned int>(grid), kBlock, 0, stream>>>(in, out, n, partials, counter);
    CKL_CUDA_LAST_ERROR(false);
}

template <typename T, ScanOp OP>
void reduce_two_pass_launch(const T* in, T* out, long long n, T* partials, int blocks,
                            cudaStream_t stream) {
    using Op = typename OpOf<T, OP>::type;
    if (n <= 0) {
        reduce_fill_identity<T, OP>(out, stream);
        return;
    }
    const int grid = grid_for(n, blocks);
    partials_only<T, Op><<<static_cast<unsigned int>(grid), kBlock, 0, stream>>>(in, partials, n);
    CKL_CUDA_LAST_ERROR(false);
    combine_only<T, Op><<<1, kBlock, 0, stream>>>(partials, out, grid);
    CKL_CUDA_LAST_ERROR(false);
}

#define CKL_INSTANTIATE(T, OP)                                                                 \
    template void reduce_single_pass_launch<T, OP>(const T*, T*, long long, T*, unsigned int*, \
                                                   int, cudaStream_t);                         \
    template void reduce_two_pass_launch<T, OP>(const T*, T*, long long, T*, int, cudaStream_t);
CKL_SCAN_REDUCE_INSTANCES(CKL_INSTANTIATE)
#undef CKL_INSTANTIATE

}  // namespace detail
}  // namespace ckl
