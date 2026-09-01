// Stream-K: the CUTLASS style hybrid. A bounded number of persistent CTAs take
// an equal share of the K iterations belonging to the tiles that do not fill a
// whole wave, and every remaining tile runs data parallel, one CTA per tile.
//
// Split-K fixes wave quantization by cutting K, but it cuts every tile the same
// way and pays a full extra pass over C for it. Stream-K only touches the tiles
// that would have left SMs idle. The grid is laid out as
//
//   [ sk_blocks stream-K CTAs ][ dp_tiles data parallel CTAs ]
//
// with the data parallel CTAs taking tiles 0 to dp_tiles and the stream-K CTAs
// sharing out the K iterations of the tiles above that. The stream-K region is
// at most two waves of tiles: the remainder that does not fill a wave, plus one
// full wave to give the split something to absorb into.
//
// The handoff. A CTA whose share stops before the end of a tile cannot write
// that tile's C, because it holds only part of the sum. It publishes the partial
// into its own plane of the workspace, fences, and raises a flag. The CTA that
// covers the tile's last K iteration waits on the flag of every peer whose share
// ended inside that tile, adds the planes into its accumulator, and writes C
// once with alpha and beta.
//
// The direction of the wait is the whole safety argument, so it is deliberate:
// an owner only ever waits on CTAs with a lower index than its own. Blocks are
// dispatched in increasing index order, so a resident owner implies every peer
// it can wait on is already resident or already retired. Waiting upward instead
// would need every stream-K CTA co-resident, which is an assumption about
// occupancy that a profiler's instrumentation or a future register change could
// quietly break into a hang. The data parallel CTAs never wait at all.
//
// The two orderings that matter are both explicit. __threadfence between the
// partial stores and the flag makes the partial visible device wide before the
// flag that advertises it; __syncthreads between them makes that true for every
// thread of the block, not just the one that raises the flag. And the persistent
// loop carries a __syncthreads of its own, because the shared tile buffers are
// reused from one tile to the next and the next tile's cp.async stores would
// otherwise race the previous tile's ldmatrix reads. That last one is the
// barrier compute-sanitizer --tool racecheck reports when it is removed.

#include <cstddef>
#include <cstdint>
#include <string>

#include <cuda_fp16.h>

#include "ckl/context.hpp"
#include "ckl/cuda_check.hpp"
#include "ckl/gemm.hpp"

#include "detail/gemm_tile_mainloop.cuh"
#include "detail/gemm_tile_registry.hpp"

namespace ckl {

namespace {

using detail::TileShape;

constexpr int kMinArch = 80;

// How the grid is cut. Everything here is integer arithmetic on the shape and
// the occupancy, so the workspace query and the launch derive the same plan from
// the same inputs and cannot disagree.
struct StreamKPlan {
    int tiles = 0;
    int tiles_n = 0;
    int iters_per_tile = 1;
    int sk_tiles = 0;
    int sk_tile_begin = 0;
    int sk_blocks = 0;
    int sk_iters = 0;
    int dp_tiles = 0;
    int grid = 0;
};

StreamKPlan plan_stream_k(int m, int n, int k, const GemmTile& tile, int occupancy_ctas) {
    StreamKPlan p;
    const int tiles_m = (m + tile.m - 1) / tile.m;
    p.tiles_n = (n + tile.n - 1) / tile.n;
    p.tiles = tiles_m * p.tiles_n;
    p.iters_per_tile = k > 0 ? (k + tile.k - 1) / tile.k : 1;
    const int occ = occupancy_ctas > 0 ? occupancy_ctas : 1;

    const int remainder = p.tiles % occ;
    if (p.tiles > 0 && remainder != 0) {
        // The tiles that do not fill a wave, plus one full wave when there is
        // one, so the share each stream-K CTA gets is a fraction of a tile
        // rather than a whole one.
        p.sk_tiles = (p.tiles > occ) ? (remainder + occ) : p.tiles;
        p.sk_iters = p.sk_tiles * p.iters_per_tile;
        p.sk_blocks = occ < p.sk_iters ? occ : p.sk_iters;
    }
    p.sk_tile_begin = p.tiles - p.sk_tiles;
    p.dp_tiles = p.sk_tile_begin;
    p.grid = p.sk_blocks + p.dp_tiles;
    return p;
}

int multiprocessor_count() {
    static const int count = [] {
        int device = 0;
        CKL_CUDA_CHECK(cudaGetDevice(&device));
        int value = 0;
        CKL_CUDA_CHECK(cudaDeviceGetAttribute(&value, cudaDevAttrMultiProcessorCount, device));
        return value;
    }();
    return count;
}

template <typename S, bool TAIL>
__global__ __launch_bounds__(S::kThreads) void streamk_kernel(
    const __half* __restrict__ a, const __half* __restrict__ b, float* __restrict__ c,
    float* __restrict__ partials, int* __restrict__ flags, int m, int n, int k, int tiles_n,
    int iters_per_tile, int sk_blocks, int sk_tile_begin, int sk_iters, bool vec, float alpha,
    float beta) {
#if __CUDA_ARCH__ >= 800
    extern __shared__ __align__(16) char ckl_streamk_smem[];
    __half* const as = reinterpret_cast<__half*>(ckl_streamk_smem);
    __half* const bs = as + S::kStages * S::kATileHalves;

    const int cta = static_cast<int>(blockIdx.x);

    if (cta >= sk_blocks) {
        // Data parallel: one whole tile, whole K, straight into C.
        const int tile = cta - sk_blocks;
        const int block_row = (tile / tiles_n) * S::kBM;
        const int block_col = (tile % tiles_n) * S::kBN;
        float acc[S::kMTiles][S::kNTiles][4] = {};
        detail::tile_accumulate<S, TAIL>(a, b, m, n, k, 0, k, block_row, block_col, vec, acc, as,
                                         bs);
        detail::tile_write_c<S, TAIL>(c, m, n, block_row, block_col, acc, alpha, beta);
        return;
    }

    // An equal share of the stream-K iteration space, remainder spread over the
    // first few CTAs so no CTA is more than one iteration ahead of another.
    const int base = sk_iters / sk_blocks;
    const int rem = sk_iters % sk_blocks;
    const int my_begin = cta * base + (cta < rem ? cta : rem);
    const int my_end = my_begin + base + (cta < rem ? 1 : 0);

    int iter = my_begin;
    while (iter < my_end) {
        // The shared tile buffers are reused from one tile to the next, so the
        // stores of this tile's prologue have to wait for the loads of the last
        // tile's mainloop. Everything below is uniform across the block, so
        // every thread reaches this barrier the same number of times.
        __syncthreads();

        const int tile_local = iter / iters_per_tile;
        const int tile_begin = tile_local * iters_per_tile;
        const int local_begin = iter - tile_begin;
        int local_end = my_end - tile_begin;
        if (local_end > iters_per_tile) {
            local_end = iters_per_tile;
        }

        const int tile = sk_tile_begin + tile_local;
        const int block_row = (tile / tiles_n) * S::kBM;
        const int block_col = (tile % tiles_n) * S::kBN;

        int k_begin = local_begin * S::kBK;
        int k_end = local_end * S::kBK;
        if (k_begin > k) {
            k_begin = k;
        }
        if (k_end > k) {
            k_end = k;
        }

        float acc[S::kMTiles][S::kNTiles][4] = {};
        detail::tile_accumulate<S, TAIL>(a, b, m, n, k, k_begin, k_end, block_row, block_col, vec,
                                         acc, as, bs);

        if (local_end != iters_per_tile) {
            // A peer: its share stops before the end of the tile, so it holds a
            // slice of the sum and not the sum. A CTA can be in this position for
            // at most one tile, the last one its share reaches, which is why one
            // plane of the workspace per CTA is enough.
            detail::tile_store_partial<S>(partials + static_cast<long long>(cta) * S::kBM * S::kBN,
                                          acc);
            __threadfence();
            __syncthreads();
            if (threadIdx.x == 0) {
                atomicExch(&flags[cta], 1);
            }
        } else {
            // The owner: it covers the last K iteration of the tile, so it waits
            // for every peer whose share ends inside the tile and writes C.
            for (int peer = cta - 1; peer >= 0; --peer) {
                const int next = peer + 1;
                const int peer_end = next * base + (next < rem ? next : rem);
                if (peer_end <= tile_begin) {
                    break;
                }
                volatile int* const flag = flags + peer;
                while (*flag == 0) {
                    __nanosleep(64);
                }
                __threadfence();
                detail::tile_add_partial<S>(
                    partials + static_cast<long long>(peer) * S::kBM * S::kBN, acc);
            }
            detail::tile_write_c<S, TAIL>(c, m, n, block_row, block_col, acc, alpha, beta);
        }

        iter = tile_begin + local_end;
    }
#else
    (void)a;
    (void)b;
    (void)c;
    (void)partials;
    (void)flags;
    (void)m;
    (void)n;
    (void)k;
    (void)tiles_n;
    (void)iters_per_tile;
    (void)sk_blocks;
    (void)sk_tile_begin;
    (void)sk_iters;
    (void)vec;
    (void)alpha;
    (void)beta;
#endif
}

int shared_mem_per_block_optin() {
    static const int bytes = [] {
        int device = 0;
        CKL_CUDA_CHECK(cudaGetDevice(&device));
        int value = 0;
        CKL_CUDA_CHECK(
            cudaDeviceGetAttribute(&value, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
        return value;
    }();
    return bytes;
}

template <typename S, bool TAIL>
void prepare_once() {
    static const bool ready = [] {
        const int optin = shared_mem_per_block_optin();
        if (optin < static_cast<int>(S::kSmemBytes)) {
            throw Error(Status::kNotSupported,
                        "gemm_stream_k: this tile needs " + std::to_string(S::kSmemBytes) +
                            " bytes of dynamic shared memory per block; this device allows " +
                            std::to_string(optin));
        }
        CKL_CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<const void*>(streamk_kernel<S, TAIL>),
                                            cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            static_cast<int>(S::kSmemBytes)));
        return true;
    }();
    (void)ready;
}

template <typename S>
void launch_streamk(const __half* a, const __half* b, float* c, float* partials, int* flags,
                    const StreamKPlan& p, int m, int n, int k, float alpha, float beta,
                    cudaStream_t stream) {
    const dim3 block(S::kThreads);
    const dim3 grid(static_cast<unsigned>(p.grid));
    // Same two questions the data parallel launcher asks: can a sixteen byte
    // cp.async read these operands, and can a float2 store write this C.
    const bool vec = (k % 8 == 0) && (n % 8 == 0) &&
                     (reinterpret_cast<std::uintptr_t>(a) & 0xFU) == 0U &&
                     (reinterpret_cast<std::uintptr_t>(b) & 0xFU) == 0U;
    const bool wide_store = (reinterpret_cast<std::uintptr_t>(c) & 0x7U) == 0U;
    const bool aligned =
        vec && wide_store && (m % S::kBM == 0) && (n % S::kBN == 0) && (k % S::kBK == 0) && k > 0;
    if (aligned) {
        prepare_once<S, false>();
        streamk_kernel<S, false><<<grid, block, S::kSmemBytes, stream>>>(
            a, b, c, partials, flags, m, n, k, p.tiles_n, p.iters_per_tile, p.sk_blocks,
            p.sk_tile_begin, p.sk_iters, true, alpha, beta);
    } else {
        prepare_once<S, true>();
        streamk_kernel<S, true><<<grid, block, S::kSmemBytes, stream>>>(
            a, b, c, partials, flags, m, n, k, p.tiles_n, p.iters_per_tile, p.sk_blocks,
            p.sk_tile_begin, p.sk_iters, vec, alpha, beta);
    }
    CKL_CUDA_LAST_ERROR(false);
}

using StreamKFn = void (*)(const __half*, const __half*, float*, float*, int*, const StreamKPlan&,
                           int, int, int, float, float, cudaStream_t);

#define CKL_STREAMK_ENTRY(IDX, BM, BN, BK, WM, WN) &launch_streamk<TileShape<BM, BN, BK, WM, WN>>,
constexpr StreamKFn kStreamK[] = {CKL_TILE_FAMILY_FOR_EACH(CKL_STREAMK_ENTRY)};
#undef CKL_STREAMK_ENTRY

static_assert(sizeof(kStreamK) / sizeof(kStreamK[0]) == CKL_TILE_FAMILY_COUNT,
              "the stream-K table and the roster disagree");

// Partial planes first, then the flags, so the float2 stores into the planes
// keep the alignment the allocator handed out.
std::size_t partial_bytes(const StreamKPlan& p, const GemmTile& tile) {
    return static_cast<std::size_t>(p.sk_blocks) * static_cast<std::size_t>(tile.m) *
           static_cast<std::size_t>(tile.n) * sizeof(float);
}

std::size_t plan_bytes(const StreamKPlan& p, const GemmTile& tile) {
    if (p.sk_blocks <= 0) {
        return 0;
    }
    return partial_bytes(p, tile) + static_cast<std::size_t>(p.sk_blocks) * sizeof(int);
}

}  // namespace

std::size_t gemm_stream_k_workspace_size(int m, int n, int k, int tile_index) {
    const GemmTile tile = gemm_tile_family_shape(tile_index);
    if (tile.k == 0 || m <= 0 || n <= 0) {
        return 0;
    }
    const int occ = multiprocessor_count() * gemm_tile_family_blocks_per_sm(tile_index);
    const StreamKPlan p = plan_stream_k(m, n, k, tile, occ);
    return plan_bytes(p, tile);
}

void gemm_stream_k(const __half* a, const __half* b, float* c, int m, int n, int k, float alpha,
                   float beta, int tile_index, void* workspace, std::size_t workspace_bytes,
                   cudaStream_t stream) {
    const GemmTile tile = gemm_tile_family_shape(tile_index);
    if (tile.k == 0) {
        throw Error(
            Status::kInvalidValue,
            "gemm_stream_k: tile index " + std::to_string(tile_index) + " is outside the family");
    }
    if (m <= 0 || n <= 0) {
        return;  // no output elements, so nothing to write
    }
    detail::require_arch(kMinArch, "gemm_stream_k");

    const int kk = k > 0 ? k : 0;
    const int occ = multiprocessor_count() * gemm_tile_family_blocks_per_sm(tile_index);
    const StreamKPlan p = plan_stream_k(m, n, kk, tile, occ);

    if (p.sk_blocks <= 0) {
        // The grid already fills whole waves, so there is nothing for the
        // persistent CTAs to even out and the plain family grid is the answer.
        detail::tile_family_launch(tile_index, a, b, c, nullptr, m, n, kk, kk > 0 ? kk : 1, 1,
                                   alpha, beta, stream);
        return;
    }

    const std::size_t need = plan_bytes(p, tile);
    void* scratch = workspace;
    bool owned = false;
    if (scratch == nullptr) {
        CKL_CUDA_CHECK(cudaMallocAsync(&scratch, need, stream));
        owned = true;
    } else if (workspace_bytes < need) {
        throw Error(Status::kInvalidValue,
                    "gemm_stream_k: this shape needs " + std::to_string(need) +
                        " workspace bytes, the caller supplied " + std::to_string(workspace_bytes));
    }

    auto* partials = static_cast<float*>(scratch);
    auto* flags = reinterpret_cast<int*>(static_cast<char*>(scratch) + partial_bytes(p, tile));
    try {
        CKL_CUDA_CHECK(
            cudaMemsetAsync(flags, 0, static_cast<std::size_t>(p.sk_blocks) * sizeof(int), stream));
        kStreamK[tile_index](a, b, c, partials, flags, p, m, n, kk, alpha, beta, stream);
    } catch (...) {
        if (owned) {
            cudaFreeAsync(scratch, stream);
        }
        throw;
    }
    if (owned) {
        CKL_CUDA_CHECK(cudaFreeAsync(scratch, stream));
    }
}

}  // namespace ckl
