// The tile shape family: six instantiations of the round 10 mainloop, plus the
// predicated edge path that stops an unaligned shape falling off the ladder.
//
// One kernel template covers both the plain grid and the split-K first pass.
// They differ in two runtime values and nothing else: which slice of K a block
// covers, which comes from blockIdx.z, and whether the epilogue writes C with
// alpha and beta or a raw partial plane the fixup pass will reduce. Both are
// uniform across the block and are decided once in the epilogue, so folding them
// together costs nothing and halves the number of instantiations, which at six
// shapes times two edge forms is already twelve kernels.
//
// The edge form is a template parameter rather than a runtime flag because it
// changes the epilogue's store width: the aligned form keeps the float2 store
// the accumulator fragment allows, and the predicated form stores element by
// element under a bounds test. Making that a branch would cost the aligned path
// the store width on every shape.
//
// Every instantiation carries its constraints as static_asserts in TileShape:
// three stages inside the 99 KB per block opt-in, warps inside 48 per SM, the
// staging work dividing evenly across the block, and a warp tile the mma shape
// divides. A shape that violates one does not build.

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

// cp.async, ldmatrix and mma.sync together: sm_80 is the floor.
constexpr int kMinArch = 80;

template <typename S, bool TAIL>
__global__ __launch_bounds__(S::kThreads) void tile_kernel(const __half* __restrict__ a,
                                                           const __half* __restrict__ b,
                                                           float* __restrict__ c,
                                                           float* __restrict__ partials, int m,
                                                           int n, int k, int k_chunk, bool vec,
                                                           float alpha, float beta) {
#if __CUDA_ARCH__ >= 800
    // kStages A tiles then kStages B tiles. The runtime aligns the dynamic block
    // to 16 bytes and every offset is a multiple of eight halves, so both the
    // cp.async stores and the ldmatrix loads stay 16 byte aligned.
    extern __shared__ __align__(16) char ckl_tile_smem[];
    __half* const as = reinterpret_cast<__half*>(ckl_tile_smem);
    __half* const bs = as + S::kStages * S::kATileHalves;

    const int split = static_cast<int>(blockIdx.z);
    const int k_begin = split * k_chunk;
    if (k_begin >= k && k > 0) {
        return;  // a split the rounded chunk size left with nothing to do
    }
    int k_end = k_begin + k_chunk;
    if (k_end > k) {
        k_end = k;
    }

    const int block_row = static_cast<int>(blockIdx.y) * S::kBM;
    const int block_col = static_cast<int>(blockIdx.x) * S::kBN;

    float acc[S::kMTiles][S::kNTiles][4] = {};
    detail::tile_accumulate<S, TAIL>(a, b, m, n, k, k_begin, k_end, block_row, block_col, vec, acc,
                                     as, bs);

    if (partials != nullptr) {
        // Raw partial sums. alpha and beta are the fixup pass's business, so
        // that each is applied exactly once however many splits there are.
        float* const plane =
            partials + static_cast<long long>(split) * static_cast<long long>(m) * n;
        detail::tile_write_c<S, TAIL>(plane, m, n, block_row, block_col, acc, 1.0f, 0.0f);
    } else {
        detail::tile_write_c<S, TAIL>(c, m, n, block_row, block_col, acc, alpha, beta);
    }
#else
    // The mainloop needs sm_80. The launcher checks the running device, so this
    // body is never launched.
    (void)a;
    (void)b;
    (void)c;
    (void)partials;
    (void)m;
    (void)n;
    (void)k;
    (void)k_chunk;
    (void)vec;
    (void)alpha;
    (void)beta;
#endif
}

// Opt-in shared memory ceiling of the current device, read once. Plain sm_120
// reports 101376 bytes; the 48 KB figure that limits static __shared__ is a
// different limit and does not apply once the attribute below is set.
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

// Raises the dynamic shared memory ceiling for one instantiation, once per
// process. A device that cannot give the block its budget gets kNotSupported:
// there is no slower form of this shape to fall back to, and a silently
// different path is worse than a status the caller can act on.
template <typename S, bool TAIL>
void prepare_once() {
    static const bool ready = [] {
        const int optin = shared_mem_per_block_optin();
        if (optin < static_cast<int>(S::kSmemBytes)) {
            throw Error(Status::kNotSupported,
                        "gemm_tile_family: this tile needs " + std::to_string(S::kSmemBytes) +
                            " bytes of dynamic shared memory per block; this device allows " +
                            std::to_string(optin));
        }
        CKL_CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<const void*>(tile_kernel<S, TAIL>),
                                            cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            static_cast<int>(S::kSmemBytes)));
        return true;
    }();
    (void)ready;
}

template <typename S>
void launch_tile(const __half* a, const __half* b, float* c, float* partials, int m, int n, int k,
                 int k_chunk, int splits, float alpha, float beta, cudaStream_t stream) {
    const dim3 block(S::kThreads);
    const dim3 grid(static_cast<unsigned>((n + S::kBN - 1) / S::kBN),
                    static_cast<unsigned>((m + S::kBM - 1) / S::kBM),
                    static_cast<unsigned>(splits));

    // Both halves of the aligned form, kept apart because they fail for
    // different reasons. Staging asks whether a sixteen byte cp.async has a
    // legal source: an odd leading dimension puts every second row on an odd
    // address, and a hand offset base pointer can be misaligned on its own.
    // The epilogue asks whether the float2 store has a legal destination.
    float* const out = partials != nullptr ? partials : c;
    const bool vec = (k % 8 == 0) && (n % 8 == 0) &&
                     (reinterpret_cast<std::uintptr_t>(a) & 0xFU) == 0U &&
                     (reinterpret_cast<std::uintptr_t>(b) & 0xFU) == 0U;
    const bool wide_store = (reinterpret_cast<std::uintptr_t>(out) & 0x7U) == 0U;
    const bool aligned = vec && wide_store && (m % S::kBM == 0) && (n % S::kBN == 0) &&
                         (k % S::kBK == 0) && (k_chunk % S::kBK == 0) && k > 0;
    if (aligned) {
        prepare_once<S, false>();
        tile_kernel<S, false><<<grid, block, S::kSmemBytes, stream>>>(a, b, c, partials, m, n, k,
                                                                      k_chunk, true, alpha, beta);
    } else {
        prepare_once<S, true>();
        tile_kernel<S, true><<<grid, block, S::kSmemBytes, stream>>>(a, b, c, partials, m, n, k,
                                                                     k_chunk, vec, alpha, beta);
    }
    CKL_CUDA_LAST_ERROR(false);
}

using LaunchFn = void (*)(const __half*, const __half*, float*, float*, int, int, int, int, int,
                          float, float, cudaStream_t);

#define CKL_TILE_LAUNCH_ENTRY(IDX, BM, BN, BK, WM, WN) &launch_tile<TileShape<BM, BN, BK, WM, WN>>,
constexpr LaunchFn kLaunch[] = {CKL_TILE_FAMILY_FOR_EACH(CKL_TILE_LAUNCH_ENTRY)};
#undef CKL_TILE_LAUNCH_ENTRY

#define CKL_TILE_SHAPE_ENTRY(IDX, BM, BN, BK, WM, WN) GemmTile{BM, BN, BK, WM, WN},
constexpr GemmTile kShapes[] = {CKL_TILE_FAMILY_FOR_EACH(CKL_TILE_SHAPE_ENTRY)};
#undef CKL_TILE_SHAPE_ENTRY

static_assert(sizeof(kLaunch) / sizeof(kLaunch[0]) == CKL_TILE_FAMILY_COUNT,
              "the launch table and the roster disagree");
static_assert(sizeof(kShapes) / sizeof(kShapes[0]) == CKL_TILE_FAMILY_COUNT,
              "the shape table and the roster disagree");

// Occupancy of one instantiation, queried once and cached. Nothing here throws:
// the answer feeds a heuristic, and gemm_query promises to launch nothing and to
// leave no CUDA error behind, so a device that cannot host the shape reports
// zero blocks and the caller decides what to do about it.
template <typename S>
int blocks_per_sm_of() {
    static const int value = [] {
        const cudaError_t attr = cudaFuncSetAttribute(
            reinterpret_cast<const void*>(tile_kernel<S, false>),
            cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(S::kSmemBytes));
        if (attr != cudaSuccess) {
            cudaGetLastError();
            return 0;
        }
        int blocks = 0;
        const cudaError_t occ = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocks, reinterpret_cast<const void*>(tile_kernel<S, false>), S::kThreads,
            S::kSmemBytes);
        if (occ != cudaSuccess) {
            cudaGetLastError();
            return 0;
        }
        return blocks;
    }();
    return value;
}

using OccupancyFn = int (*)();

#define CKL_TILE_OCCUPANCY_ENTRY(IDX, BM, BN, BK, WM, WN) \
    &blocks_per_sm_of<TileShape<BM, BN, BK, WM, WN>>,
constexpr OccupancyFn kOccupancy[] = {CKL_TILE_FAMILY_FOR_EACH(CKL_TILE_OCCUPANCY_ENTRY)};
#undef CKL_TILE_OCCUPANCY_ENTRY

bool valid_index(int index) {
    return index >= 0 && index < CKL_TILE_FAMILY_COUNT;
}

}  // namespace

namespace detail {

void tile_family_launch(int index, const __half* a, const __half* b, float* c, float* partials,
                        int m, int n, int k, int k_chunk, int splits, float alpha, float beta,
                        cudaStream_t stream) {
    if (!valid_index(index)) {
        throw Error(Status::kInvalidValue, "gemm tile family: tile index " + std::to_string(index) +
                                               " is outside the family");
    }
    kLaunch[index](a, b, c, partials, m, n, k, k_chunk, splits, alpha, beta, stream);
}

}  // namespace detail

int gemm_tile_family_count() {
    return CKL_TILE_FAMILY_COUNT;
}

GemmTile gemm_tile_family_shape(int index) {
    return valid_index(index) ? kShapes[index] : GemmTile{};
}

int gemm_tile_family_blocks_per_sm(int index) {
    return valid_index(index) ? kOccupancy[index]() : 0;
}

void gemm_tile_family(const __half* a, const __half* b, float* c, int m, int n, int k, float alpha,
                      float beta, int tile_index, cudaStream_t stream) {
    if (!valid_index(tile_index)) {
        throw Error(Status::kInvalidValue, "gemm_tile_family: tile index " +
                                               std::to_string(tile_index) +
                                               " is outside the family");
    }
    if (m <= 0 || n <= 0) {
        return;  // no output elements, so nothing to write
    }
    detail::require_arch(kMinArch, "gemm_tile_family");
    const int kk = k > 0 ? k : 0;
    // One split covering the whole contraction. An empty contraction still runs,
    // because C = beta * C is the BLAS answer for it and the epilogue is where
    // that lives.
    detail::tile_family_launch(tile_index, a, b, c, nullptr, m, n, kk, kk > 0 ? kk : 1, 1, alpha,
                               beta, stream);
}

}  // namespace ckl
