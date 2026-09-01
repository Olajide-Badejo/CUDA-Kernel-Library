#pragma once

/**
 * @file types.hpp
 * @brief Descriptor types shared by the C++ GEMM entry points.
 *
 * These mirror the C ABI enums in ckl.h one for one, in the same order, so a
 * cast between them is a value preserving reinterpretation and the ABI stays
 * stable when a later part implements the rungs that currently report
 * Status::kNotSupported.
 */

#include <cstddef>
#include <cstdint>

#include "ckl/ckl_export.h"

namespace ckl {

/**
 * @brief Transpose flag applied to an operand.
 *
 * @note Op::kC is accepted and treated as Op::kT. The library ships real
 *       precisions only, so a conjugate transpose is a plain transpose; a
 *       complex type would have to revisit this.
 */
enum class Op {
    kN,  ///< No transpose.
    kT,  ///< Transpose.
    kC,  ///< Conjugate transpose, treated as kT.
};

/// @brief How a matrix is stored in memory.
enum class Layout {
    kRowMajor,  ///< Row elements adjacent; the leading dimension counts columns.
    kColMajor,  ///< Column elements adjacent; the leading dimension counts rows.
};

/// @brief Element type of a matrix operand.
enum class DType {
    kR32F,   ///< 32 bit float.
    kR16F,   ///< 16 bit IEEE half.
    kR16BF,  ///< 16 bit bfloat.
};

/**
 * @brief Every rung of the GEMM ladder plus the vendor baseline.
 *
 * @note kCutlass is declared now and implemented in a later release; dispatch
 *       returns Status::kNotSupported for it rather than quietly picking
 *       something else.
 */
enum class Algo {
    kAuto,      ///< Let the dispatcher choose, and report the choice through the chosen out-param.
    kNaive,     ///< One thread per output element, FP32.
    kTiled,     ///< Shared memory tiled, FP32.
    kRegister,  ///< Register blocked with float4 loads, FP32.
    kCpAsync,   ///< cp.async double buffered, FP32.
    kWmmaFp16,  ///< WMMA 16x16x16 fragments, FP16 in, FP32 accumulate.
    kWmmaBf16,  ///< WMMA 16x16x16 fragments, BF16 in, FP32 accumulate.
    kMmaPtx,    ///< Raw mma.sync PTX with scalar shared reads, FP16 in.
    kMmaLdm,    ///< mma.sync with ldmatrix fragment loads, FP16 in.
    kMmaOpt,    ///< The top tensor kernel: 128x128 tile, ldmatrix, cp.async double buffering.
    kTileFamily,  ///< The tile shape family with predicated edges, FP16 in; shape chosen by plan.
    kSplitK,      ///< The tile family split along K, with a fixup reduction pass.
    kStreamK,     ///< The tile family as persistent CTAs with a stream-K remainder.
    kCutlass,     ///< Declared for ABI stability; not implemented in 1.1.0.
    kCublas,      ///< The vendor path, which takes every shape the hand written kernels refuse.
};

/**
 * @brief One member of the tile shape family.
 *
 * @note A shape of all zeros means the path has no block tile, which is the
 *       answer for every rung outside the family and for the vendor path.
 */
struct GemmTile {
    int m = 0;        ///< Rows of the block tile (BM).
    int n = 0;        ///< Columns of the block tile (BN).
    int k = 0;        ///< K step of the mainloop (BK).
    int warps_m = 0;  ///< Warps along M inside the block.
    int warps_n = 0;  ///< Warps along N inside the block.
};

/**
 * @brief What Algo::kAuto decided, in full.
 *
 * ckl::gemm reports the rung through its chosen out-param, which is the answer
 * a benchmark row prints. The tile family needs one level more detail, because
 * "kTileFamily" alone does not say which of the six shapes ran, so the whole
 * decision is available through ckl::gemm_plan.
 */
struct GemmPlan {
    Algo algo = Algo::kAuto;  ///< The rung the dispatcher picked.
    GemmTile tile;            ///< The block tile, all zeros outside the family.
    int tile_index = -1;      ///< Index into the family, or -1 outside it.
    int splits = 1;           ///< K splits for Algo::kSplitK; 1 everywhere else.
    /// True when a committed tile sweep informed the choice, false when the
    /// decision was quantization only. A quantization only decision is a
    /// hypothesis about which shape wins, not a measurement.
    bool tuned = false;
};

/**
 * @brief One GEMM, described: C = alpha * op(A) * op(B) + beta * C.
 *
 * A is m by k, B is k by n and C is m by n after the ops are applied.
 *
 * Layout and transpose both reach cuBLAS through one identity. cuBLAS is column
 * major, and a row major matrix occupies the same bytes as its transpose read
 * column major. So a row major call becomes the column major call
 * C_t(n by m) = op(B)_t * op(A)_t: swap the operands, swap m and n, and keep
 * each operand's own op flag with it. A column major call passes straight
 * through.
 *
 * @note The hand written kernels are row major, no transpose, packed only;
 *       anything else dispatches to cuBLAS and says so through the chosen
 *       out-param of ckl::gemm.
 */
struct GemmDesc {
    Layout layout = Layout::kRowMajor;  ///< Storage order of all three matrices.
    Op op_a = Op::kN;                   ///< Transpose flag for A.
    Op op_b = Op::kN;                   ///< Transpose flag for B.
    std::int64_t m = 0;                 ///< Rows of op(A) and of C.
    std::int64_t n = 0;                 ///< Columns of op(B) and of C.
    std::int64_t k = 0;                 ///< Contraction extent.
    DType dt_a = DType::kR32F;          ///< Element type of A.
    DType dt_b = DType::kR32F;          ///< Element type of B; must equal dt_a in 1.1.0.
    DType dt_c = DType::kR32F;          ///< Element type of C; kR32F is the only one in 1.1.0.
    std::int64_t lda = 0;               ///< Leading dimension of A; the layout sets its minimum.
    std::int64_t ldb = 0;               ///< Leading dimension of B.
    std::int64_t ldc = 0;               ///< Leading dimension of C.
    std::int64_t stride_a = 0;          ///< Elements between batches of A; ignored when unbatched.
    std::int64_t stride_b = 0;          ///< Elements between batches of B.
    std::int64_t stride_c = 0;          ///< Elements between batches of C.
    std::int32_t batch_count = 1;       ///< Matrices in the batch; at least 1.
    Algo algo = Algo::kAuto;            ///< Requested path. A named algorithm is never rerouted.
};

/**
 * @brief Stable short name of an algorithm, for example "mma_opt".
 * @param a Algorithm to name.
 * @return A static string, never null. This is what a benchmark row prints.
 */
CKL_EXPORT const char* algo_name(Algo a);

/**
 * @brief Stable short name of an element type, for example "r16f".
 * @param t Type to name.
 * @return A static string, never null.
 */
CKL_EXPORT const char* dtype_name(DType t);

/**
 * @brief Stable short name of a transpose flag.
 * @param o Flag to name.
 * @return A static string, never null.
 */
CKL_EXPORT const char* op_name(Op o);

/**
 * @brief Stable short name of a layout.
 * @param l Layout to name.
 * @return A static string, never null.
 */
CKL_EXPORT const char* layout_name(Layout l);

/**
 * @brief Element size in bytes.
 * @param t Type to measure.
 * @return 4 for DType::kR32F, 2 for the two 16 bit types.
 */
CKL_EXPORT std::size_t dtype_size(DType t);

}  // namespace ckl
