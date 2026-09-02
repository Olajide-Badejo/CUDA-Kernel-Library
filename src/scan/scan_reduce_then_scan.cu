// s3: reduce then scan.
//
// s2 writes its scanned output and then goes back over it to add each tile's
// prefix, which costs a read and a write of the whole array that produce nothing
// new. This rung spends a read instead: pass one reduces each tile to a single
// aggregate and writes only the aggregates, the aggregates are scanned, and pass
// two reads the input a second time and writes the finished output once. Three
// passes over the data instead of four.
//
// It is also the rung the deterministic scan mode runs on, and that is not a
// coincidence. Its tile decomposition is a pure function of the length: tile t
// always covers the same elements, always reduces them in the same order, and
// always receives the same prefix, whatever the device was doing at the time. A
// decoupled look-back cannot promise that, because how many predecessors a tile
// has to combine depends on which of them happened to finish first, and floating
// point addition is not associative.

#include "ckl/cuda_check.hpp"
#include "ckl/status.hpp"

#include "scan_device.cuh"
#include "scan_levels.cuh"
#include "scan_rungs.hpp"

namespace ckl {
namespace detail {

namespace {

// Pass one. Reads the input and writes one aggregate per tile, and nothing else.
template <typename T, class Op, int BLOCK, int ITEMS>
__global__ void tile_reduce_kernel(const T* in, T* aggregates, long long n) {
    using Exchange = TileExchange<T, BLOCK, ITEMS>;
    __shared__ struct {
        typename Exchange::TempStorage exchange;
        typename BlockReduce<T, Op, BLOCK>::TempStorage reduce;
    } smem;

    const long long base = static_cast<long long>(blockIdx.x) * Exchange::kTile;
    const long long left = n - base;
    const int valid = left >= Exchange::kTile ? Exchange::kTile : static_cast<int>(left);

    T items[ITEMS];
    Exchange::load(smem.exchange, in, base, valid, Op::identity(), items);
    const T thread_aggregate = serial_reduce<T, Op, ITEMS>(items);
    const T total = BlockReduce<T, Op, BLOCK>::run(smem.reduce, thread_aggregate);
    if (threadIdx.x == 0) {
        aggregates[blockIdx.x] = total;
    }
}

// Pass two. Reads the input again and writes the finished output once, starting
// each tile from the prefix the aggregate scan computed for it.
template <typename T, class Op, int BLOCK, int ITEMS, bool EXCLUSIVE>
__global__ void tile_apply_kernel(const T* in, T* out, const T* prefixes, long long n) {
    using Exchange = TileExchange<T, BLOCK, ITEMS>;
    using Scan = WarpBlockScan<T, Op, BLOCK>;
    __shared__ struct {
        typename Exchange::TempStorage exchange;
        typename Scan::TempStorage scan;
    } smem;

    const long long base = static_cast<long long>(blockIdx.x) * Exchange::kTile;
    const long long left = n - base;
    const int valid = left >= Exchange::kTile ? Exchange::kTile : static_cast<int>(left);
    const T prefix = prefixes[blockIdx.x];

    T items[ITEMS];
    Exchange::load(smem.exchange, in, base, valid, Op::identity(), items);
    const T thread_aggregate = serial_reduce<T, Op, ITEMS>(items);
    T block_total = Op::identity();
    const T thread_prefix = Scan::exclusive(smem.scan, thread_aggregate, &block_total);
    serial_scan<T, Op, ITEMS, EXCLUSIVE>(items, Op::apply(prefix, thread_prefix));
    Exchange::store(smem.exchange, out, base, valid, items);
}

}  // namespace

template <typename T, ScanOp OP>
void scan_reduce_then_scan_launch(const T* in, T* out, long long n, bool exclusive, T* levels,
                                  long long levels_capacity, cudaStream_t stream) {
    using Op = typename OpOf<T, OP>::type;
    constexpr int kBlock = TileShape<T>::kBlock;
    constexpr int kItems = TileShape<T>::kItems;
    constexpr int kTile = TileShape<T>::kTile;
    if (n <= 0) {
        return;
    }

    const long long tiles = (n + kTile - 1) / kTile;
    if (tiles == 1) {
        // One tile covers everything, so there is no aggregate to scan and the
        // second pass would be reading the input twice for nothing.
        LevelDriver<T, Op, kBlock, kItems, WarpBlockScan>::launch_tile(in, out, nullptr, n, 1,
                                                                       exclusive, stream);
        return;
    }
    const long long used = round_up_level(tiles);
    if (used > levels_capacity) {
        throw Error(Status::kInternal,
                    "scan: the plan's level buffer is too small for this length; it is sized from "
                    "ckl::detail::scan_levels_capacity and the two have drifted apart");
    }

    T* aggregates = levels;
    tile_reduce_kernel<T, Op, kBlock, kItems>
        <<<static_cast<unsigned int>(tiles), kBlock, 0, stream>>>(in, aggregates, n);
    CKL_CUDA_LAST_ERROR(false);

    LevelDriver<T, Op, kBlock, kItems, WarpBlockScan>::run(
        aggregates, aggregates, tiles, true, levels + used, levels_capacity - used, stream);

    if (exclusive) {
        tile_apply_kernel<T, Op, kBlock, kItems, true>
            <<<static_cast<unsigned int>(tiles), kBlock, 0, stream>>>(in, out, aggregates, n);
    } else {
        tile_apply_kernel<T, Op, kBlock, kItems, false>
            <<<static_cast<unsigned int>(tiles), kBlock, 0, stream>>>(in, out, aggregates, n);
    }
    CKL_CUDA_LAST_ERROR(false);
}

#define CKL_INSTANTIATE(T, OP)                                                           \
    template void scan_reduce_then_scan_launch<T, OP>(const T*, T*, long long, bool, T*, \
                                                      long long, cudaStream_t);
CKL_SCAN_SCAN_INSTANCES(CKL_INSTANTIATE)
#undef CKL_INSTANTIATE

}  // namespace detail
}  // namespace ckl
