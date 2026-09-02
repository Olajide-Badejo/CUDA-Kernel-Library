// r3: 16 byte loads and a grid stride loop over persistent blocks.
//
// Two changes from r2, and only two. Each thread reads 16 bytes at a time
// instead of 4, which quarters the number of load instructions a warp issues per
// byte delivered, and the grid is sized from occupancy rather than from the
// input, so the block count stops growing with N and the scheduler stops paying
// for a launch tail. This is the first rung whose limiter should be DRAM
// bandwidth rather than the machine's own issue rate, which is the hypothesis
// docs/scan.md records and the sweep tests.
//
// The persistent grid comes from cudaOccupancyMaxActiveBlocksPerMultiprocessor
// times the SM count, queried once when a plan is built and handed in, so no
// driver call happens inside a timed region. That is also why this file holds
// reduce_persistent_blocks: the occupancy of this kernel is the number the whole
// persistent half of the ladder is sized from.

#include <algorithm>

#include "ckl/cuda_check.hpp"

#include "scan_device.cuh"
#include "scan_rungs.hpp"

namespace ckl {
namespace detail {

namespace {

constexpr int kBlock = 256;

template <typename T, class Op>
__global__ void vec4_partials(const T* in, T* partials, long long n) {
    using Vec = typename Vec16<T>::type;
    constexpr int kLanes = Vec16<T>::kLanes;
    __shared__ typename BlockReduce<T, Op, kBlock>::TempStorage tmp;

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
    // The one to three elements a 16 byte vector cannot cover.
    for (long long i = vectors * kLanes + start; i < n; i += stride) {
        acc = Op::apply(acc, in[i]);
    }

    const T total = BlockReduce<T, Op, kBlock>::run(tmp, acc);
    if (threadIdx.x == 0) {
        partials[blockIdx.x] = total;
    }
}

template <typename T, class Op>
__global__ void combine_partials(const T* partials, T* out, int count) {
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

// The occupancy query, on the kernel the persistent rungs are shaped like. It
// takes a real kernel pointer, so it is one instantiation per element type.
template <typename T>
int persistent_blocks_for() {
    int device = 0;
    CKL_CUDA_CHECK(cudaGetDevice(&device));
    int sms = 0;
    CKL_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
    int per_sm = 0;
    CKL_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &per_sm, vec4_partials<T, SumOp<T>>, kBlock, 0));
    const int blocks = sms * (per_sm > 0 ? per_sm : 1);
    return blocks > 0 ? blocks : 1;
}

}  // namespace

int reduce_persistent_blocks(ScanDType dtype) {
    switch (dtype) {
        case ScanDType::kF64:
            return persistent_blocks_for<double>();
        case ScanDType::kI32:
            return persistent_blocks_for<int>();
        case ScanDType::kI64:
            return persistent_blocks_for<long long>();
        case ScanDType::kF32:
        default:
            return persistent_blocks_for<float>();
    }
}

template <typename T, ScanOp OP>
void reduce_vec4_launch(const T* in, T* out, long long n, T* partials, int blocks,
                        cudaStream_t stream) {
    using Op = typename OpOf<T, OP>::type;
    if (n <= 0) {
        reduce_fill_identity<T, OP>(out, stream);
        return;
    }
    // A grid wider than the work has blocks that reduce nothing; they would
    // still write an identity partial, but the launch is cheaper without them.
    const long long wanted = (n + kBlock - 1) / kBlock;
    const int grid = static_cast<int>(std::min<long long>(blocks, wanted));
    vec4_partials<T, Op><<<static_cast<unsigned int>(grid), kBlock, 0, stream>>>(in, partials, n);
    CKL_CUDA_LAST_ERROR(false);
    combine_partials<T, Op><<<1, kBlock, 0, stream>>>(partials, out, grid);
    CKL_CUDA_LAST_ERROR(false);
}

#define CKL_INSTANTIATE(T, OP) \
    template void reduce_vec4_launch<T, OP>(const T*, T*, long long, T*, int, cudaStream_t);
CKL_SCAN_REDUCE_INSTANCES(CKL_INSTANTIATE)
#undef CKL_INSTANTIATE

}  // namespace detail
}  // namespace ckl
