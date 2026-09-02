// s4: single pass decoupled look-back.
//
// Merrill and Garland, "Single-pass Parallel Prefix Scan with Decoupled
// Look-back", NVIDIA technical report NVR-2016-002, 2016. The idea is that a
// tile does not have to wait for its exclusive prefix to be computed for it: it
// can publish what it knows immediately, then walk backwards over its
// predecessors combining whatever each of them has already published until it
// meets one that knows its own inclusive prefix. The whole array is then read
// once and written once, which is half the traffic of scan then propagate and a
// third less than reduce then scan.
//
// Three implementation constraints are not optional, and all three are here.
//
// The tile index comes from an atomicAdd on a global counter, never from
// blockIdx.x. That is what makes the wait safe: a block that claimed index k did
// so after every block holding an index below k had already started, so those
// blocks are resident and can make progress. Indexing by blockIdx.x would let a
// resident block wait on one the scheduler has not launched yet, and that
// deadlocks.
//
// The status word is published with release semantics and read with acquire
// semantics, through cuda::atomic_ref at device scope. The payload beside it is
// a plain store, which is exactly what release and acquire exist to order: the
// release store on the flag publishes everything the same thread wrote before
// it, and the acquire load on the flag makes those writes visible to the reader
// before it touches them. A plain store on the flag would let a reader see the
// flag with the payload still in flight, and the racecheck run in docs/scan.md
// is the demonstration that this is a real ordering and not a decoration.
//
// The status array is zeroed by a memset enqueued on the same stream ahead of
// the kernel, never inside it. A block cannot zero its own slot, because a
// predecessor may already have read it, and it cannot zero anyone else's for the
// same reason.
//
// The look-back window is 32 tiles, one per lane of the first warp, so a whole
// window is inspected in one round of loads and one ballot. That is the width
// the algorithm was published with and the width a ballot can summarize in one
// instruction; a narrower window costs more rounds and a wider one costs a
// second ballot and a second reduction to combine them.

#include <cuda/atomic>

#include "ckl/cuda_check.hpp"

#include "scan_device.cuh"
#include "scan_rungs.hpp"

namespace ckl {
namespace detail {

namespace {

// The three states of a tile. Zero is the uninitialized state, so the memset
// ahead of the launch puts every tile into it.
constexpr unsigned int kFlagInvalid = 0;
constexpr unsigned int kFlagAggregate = 1;
constexpr unsigned int kFlagInclusive = 2;

// One aligned record per tile: what this tile alone sums to, what it and every
// tile before it sum to, and which of those two is valid.
template <typename T>
struct TileState {
    T aggregate;
    T inclusive;
    unsigned int flag;
    unsigned int padding;
};

using FlagRef = cuda::atomic_ref<unsigned int, cuda::thread_scope_device>;

// Lanes hold predecessors in decreasing tile order, so the ordered combine has
// to run from the highest lane down: lane 31 is the earliest tile and belongs on
// the left of every apply. Doubling offsets keep the operands in that order for
// lane zero, which is the only lane whose answer is read.
template <typename T, class Op>
__device__ inline T window_reduce(T v) {
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        const T other = __shfl_down_sync(kFullMask, v, offset);
        v = Op::apply(other, v);
    }
    return v;
}

template <typename T, class Op, int BLOCK, int ITEMS, bool EXCLUSIVE>
__global__ void lookback_kernel(const T* in, T* out, long long n, TileState<T>* status,
                                unsigned int* counter) {
    using Exchange = TileExchange<T, BLOCK, ITEMS>;
    using Scan = WarpBlockScan<T, Op, BLOCK>;
    __shared__ struct {
        typename Exchange::TempStorage exchange;
        typename Scan::TempStorage scan;
    } smem;
    __shared__ unsigned int s_tile;
    __shared__ T s_aggregate;
    __shared__ T s_prefix;

    if (threadIdx.x == 0) {
        s_tile = atomicAdd(counter, 1u);
    }
    __syncthreads();
    const long long tile = static_cast<long long>(s_tile);

    const long long base = tile * Exchange::kTile;
    const long long left = n - base;
    const int valid = left >= Exchange::kTile ? Exchange::kTile : static_cast<int>(left);

    T items[ITEMS];
    Exchange::load(smem.exchange, in, base, valid, Op::identity(), items);
    const T thread_aggregate = serial_reduce<T, Op, ITEMS>(items);
    T block_total = Op::identity();
    const T thread_prefix = Scan::exclusive(smem.scan, thread_aggregate, &block_total);
    if (threadIdx.x == 0) {
        s_aggregate = block_total;
    }
    __syncthreads();
    const T aggregate = s_aggregate;

    if (threadIdx.x < 32) {
        const int lane = static_cast<int>(threadIdx.x);

        if (lane == 0) {
            if (tile == 0) {
                // Tile zero's inclusive prefix is its own aggregate, and it is
                // known without looking at anybody.
                status[0].inclusive = aggregate;
                FlagRef(status[0].flag).store(kFlagInclusive, cuda::memory_order_release);
                s_prefix = Op::identity();
            } else {
                status[tile].aggregate = aggregate;
                FlagRef(status[tile].flag).store(kFlagAggregate, cuda::memory_order_release);
            }
        }

        if (tile != 0) {
            T running = Op::identity();
            long long look = tile - 1;
            while (true) {
                const long long idx = look - lane;
                unsigned int flag = kFlagInvalid;
                T value = Op::identity();
                if (idx >= 0) {
                    FlagRef ref(status[idx].flag);
                    // The spin. A predecessor always publishes something, so
                    // this ends; which of the two it published decides whether
                    // the walk stops here.
                    do {
                        flag = ref.load(cuda::memory_order_acquire);
                    } while (flag == kFlagInvalid);
                    value = flag == kFlagInclusive ? status[idx].inclusive : status[idx].aggregate;
                }

                const unsigned int inclusive_mask =
                    __ballot_sync(kFullMask, idx >= 0 && flag == kFlagInclusive);
                const unsigned int valid_mask = __ballot_sync(kFullMask, idx >= 0);
                // The nearest predecessor that knows its own inclusive prefix
                // ends the walk, and every lane past it is already covered by
                // that prefix.
                const int stop = inclusive_mask != 0u ? __ffs(static_cast<int>(inclusive_mask)) - 1
                                                      : 31 - __clz(static_cast<int>(valid_mask));
                const T contribution = (idx >= 0 && lane <= stop) ? value : Op::identity();
                const T window = window_reduce<T, Op>(contribution);
                if (lane == 0) {
                    running = Op::apply(window, running);
                }
                if (inclusive_mask != 0u) {
                    break;
                }
                look -= 32;
                if (look < 0) {
                    // Unreachable: tile zero publishes an inclusive prefix, so
                    // any window that reaches it ends the walk above.
                    break;
                }
            }

            if (lane == 0) {
                s_prefix = running;
                status[tile].inclusive = Op::apply(running, aggregate);
                FlagRef(status[tile].flag).store(kFlagInclusive, cuda::memory_order_release);
            }
        }
    }
    __syncthreads();

    const T prefix = s_prefix;
    serial_scan<T, Op, ITEMS, EXCLUSIVE>(items, Op::apply(prefix, thread_prefix));
    Exchange::store(smem.exchange, out, base, valid, items);
}

}  // namespace

std::size_t lookback_status_bytes_per_tile(ScanDType dtype) {
    switch (dtype) {
        case ScanDType::kF64:
            return sizeof(TileState<double>);
        case ScanDType::kI32:
            return sizeof(TileState<int>);
        case ScanDType::kI64:
            return sizeof(TileState<long long>);
        case ScanDType::kF32:
        default:
            return sizeof(TileState<float>);
    }
}

template <typename T, ScanOp OP>
void scan_lookback_launch(const T* in, T* out, long long n, bool exclusive, void* status,
                          long long tiles, unsigned int* counter, cudaStream_t stream) {
    using Op = typename OpOf<T, OP>::type;
    constexpr int kBlock = TileShape<T>::kBlock;
    constexpr int kItems = TileShape<T>::kItems;
    if (n <= 0) {
        return;
    }
    auto* state = static_cast<TileState<T>*>(status);
    // Both memsets are enqueued ahead of the kernel on the same stream. Neither
    // is inside it, and neither is inside the caller's timed region by accident:
    // they are part of what one call of this rung costs, and the traffic model
    // in ScanPlan::scan_model_bytes counts them.
    CKL_CUDA_CHECK(
        cudaMemsetAsync(state, 0, static_cast<std::size_t>(tiles) * sizeof(TileState<T>), stream));
    CKL_CUDA_CHECK(cudaMemsetAsync(counter, 0, sizeof(unsigned int), stream));

    const unsigned int grid = static_cast<unsigned int>(tiles);
    if (exclusive) {
        lookback_kernel<T, Op, kBlock, kItems, true>
            <<<grid, kBlock, 0, stream>>>(in, out, n, state, counter);
    } else {
        lookback_kernel<T, Op, kBlock, kItems, false>
            <<<grid, kBlock, 0, stream>>>(in, out, n, state, counter);
    }
    CKL_CUDA_LAST_ERROR(false);
}

#define CKL_INSTANTIATE(T, OP)                                                                 \
    template void scan_lookback_launch<T, OP>(const T*, T*, long long, bool, void*, long long, \
                                              unsigned int*, cudaStream_t);
CKL_SCAN_SCAN_INSTANCES(CKL_INSTANTIATE)
#undef CKL_INSTANTIATE

}  // namespace detail
}  // namespace ckl
