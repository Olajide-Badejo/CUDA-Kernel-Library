#pragma once

// The scan then propagate driver, shared by s0, s1, s2 and by the aggregate half
// of s3.
//
// Three rungs of the ladder are the same device wide decomposition running on
// three different block primitives, and writing that decomposition three times
// would mean three places for it to be subtly different. So it is written once,
// here, as a template over the block scan flavour and the items each thread
// carries, and the rung files supply nothing but the policy and the explicit
// instantiations. What separates s0 from s1 from s2 is then exactly the block
// primitive and the tile width, which is what the ladder is supposed to be
// isolating.
//
// The recursion is on the aggregates. One block scans one tile and publishes its
// aggregate; the aggregates are themselves scanned, exclusively, by the same
// driver one level down; and a propagate pass folds each tile's exclusive prefix
// back into its outputs. At 2^28 elements and a 256 element tile that is four
// levels, and the deepest level is a single tile. The level buffers are carved
// out of one plan owned allocation, and each slice starts on a 32 element
// boundary so the 16 byte vector loads stay aligned.
//
// This header is an implementation detail of src/scan. It is not installed.

#include "ckl/cuda_check.hpp"
#include "ckl/status.hpp"

#include "scan_device.cuh"

namespace ckl {
namespace detail {

// Level slices start on this boundary, which is what keeps a 16 byte load off
// the end of the previous slice aligned.
constexpr long long kLevelAlign = 32;

inline long long round_up_level(long long v) {
    return ((v + kLevelAlign - 1) / kLevelAlign) * kLevelAlign;
}

// Elements the level buffer needs for the deepest recursion this tile width
// produces at this length.
inline long long levels_capacity_for(long long n, long long tile) {
    long long need = 0;
    long long count = n;
    while (count > tile) {
        count = (count + tile - 1) / tile;
        need += round_up_level(count);
    }
    return need;
}

// One block per tile. Loads the tile into the blocked arrangement, reduces each
// thread's items, block scans the thread aggregates exclusively, scans the items
// from that running value, and stores. The aggregate pointer is null at the
// level where one tile covers everything.
template <typename T, class Op, int BLOCK, int ITEMS, template <typename, class, int> class Flavor,
          bool EXCLUSIVE>
__global__ void tile_scan_kernel(const T* in, T* out, T* aggregates, long long n) {
    using Exchange = TileExchange<T, BLOCK, ITEMS>;
    using Scan = Flavor<T, Op, BLOCK>;
    __shared__ struct {
        typename Exchange::TempStorage exchange;
        typename Scan::TempStorage scan;
    } smem;

    const long long base = static_cast<long long>(blockIdx.x) * Exchange::kTile;
    const long long left = n - base;
    const int valid = left >= Exchange::kTile ? Exchange::kTile : static_cast<int>(left);

    T items[ITEMS];
    Exchange::load(smem.exchange, in, base, valid, Op::identity(), items);

    const T thread_aggregate = serial_reduce<T, Op, ITEMS>(items);
    T block_total = Op::identity();
    const T thread_prefix = Scan::exclusive(smem.scan, thread_aggregate, &block_total);
    serial_scan<T, Op, ITEMS, EXCLUSIVE>(items, thread_prefix);

    Exchange::store(smem.exchange, out, base, valid, items);
    if (aggregates != nullptr && threadIdx.x == 0) {
        aggregates[blockIdx.x] = block_total;
    }
}

// Folds tile t's exclusive prefix into tile t's outputs. Launched with one block
// per tile above the first, because tile zero's prefix is the identity and
// reading it back would be traffic that buys nothing.
template <typename T, class Op, int BLOCK, int ITEMS>
__global__ void propagate_kernel(T* out, const T* prefixes, long long n) {
    using Exchange = TileExchange<T, BLOCK, ITEMS>;
    using Vec = typename Vec16<T>::type;
    constexpr int kLanes = Vec16<T>::kLanes;

    const long long tile = static_cast<long long>(blockIdx.x) + 1;
    const T prefix = prefixes[tile];
    const long long base = tile * Exchange::kTile;
    const long long left = n - base;
    const int valid = left >= Exchange::kTile ? Exchange::kTile : static_cast<int>(left);
    const int tid = static_cast<int>(threadIdx.x);

    if (valid == Exchange::kTile) {
        Vec* v = reinterpret_cast<Vec*>(out + base);
        for (int i = tid; i < Exchange::kVecs; i += BLOCK) {
            Vec value = v[i];
#pragma unroll
            for (int j = 0; j < kLanes; ++j) {
                VecAccess<T>::set(value, j, Op::apply(prefix, VecAccess<T>::get(value, j)));
            }
            v[i] = value;
        }
    } else {
        for (int i = tid; i < valid; i += BLOCK) {
            out[base + i] = Op::apply(prefix, out[base + i]);
        }
    }
}

// The host side recursion. Every level is the same three steps; the only thing
// that changes going down is the length.
template <typename T, class Op, int BLOCK, int ITEMS, template <typename, class, int> class Flavor>
struct LevelDriver {
    static constexpr int kTile = BLOCK * ITEMS;

    static void launch_tile(const T* in, T* out, T* aggregates, long long n, long long tiles,
                            bool exclusive, cudaStream_t stream) {
        const unsigned int grid = static_cast<unsigned int>(tiles);
        if (exclusive) {
            tile_scan_kernel<T, Op, BLOCK, ITEMS, Flavor, true>
                <<<grid, BLOCK, 0, stream>>>(in, out, aggregates, n);
        } else {
            tile_scan_kernel<T, Op, BLOCK, ITEMS, Flavor, false>
                <<<grid, BLOCK, 0, stream>>>(in, out, aggregates, n);
        }
        CKL_CUDA_LAST_ERROR(false);
    }

    static void run(const T* in, T* out, long long n, bool exclusive, T* levels, long long capacity,
                    cudaStream_t stream) {
        if (n <= 0) {
            return;
        }
        const long long tiles = (n + kTile - 1) / kTile;
        if (tiles == 1) {
            launch_tile(in, out, nullptr, n, 1, exclusive, stream);
            return;
        }
        const long long used = round_up_level(tiles);
        if (used > capacity) {
            throw Error(Status::kInternal,
                        "scan: the plan's level buffer is too small for this length; it is sized "
                        "from ckl::detail::scan_levels_capacity and the two have drifted apart");
        }
        T* aggregates = levels;
        launch_tile(in, out, aggregates, n, tiles, exclusive, stream);
        // The aggregates always get an exclusive scan, whatever the top level
        // was asked for: what a tile needs from its predecessors is the prefix
        // that excludes itself.
        run(aggregates, aggregates, tiles, true, levels + used, capacity - used, stream);
        propagate_kernel<T, Op, BLOCK, ITEMS>
            <<<static_cast<unsigned int>(tiles - 1), BLOCK, 0, stream>>>(out, aggregates, n);
        CKL_CUDA_LAST_ERROR(false);
    }
};

}  // namespace detail
}  // namespace ckl
