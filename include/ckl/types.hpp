#pragma once

// Descriptor types shared by the C++ GEMM entry points. These mirror the C ABI
// enums in ckl.h one for one, in the same order, so a cast between them is a
// value preserving reinterpretation and the ABI stays stable when a later part
// implements the rungs that currently report kNotSupported.

#include <cstddef>
#include <cstdint>

#include "ckl/ckl_export.h"

namespace ckl {

// kC is accepted and treated as kT. The library ships real precisions only, so
// a conjugate transpose is a plain transpose; a complex type would have to
// revisit this.
enum class Op { kN, kT, kC };

enum class Layout { kRowMajor, kColMajor };

enum class DType { kR32F, kR16F, kR16BF };

// Every rung of the ladder plus the vendor baseline. The four entries after
// kMmaOpt are declared now and implemented in later parts; dispatch returns
// kNotSupported for them rather than quietly picking something else.
enum class Algo {
    kAuto,
    kNaive,
    kTiled,
    kRegister,
    kCpAsync,
    kWmmaFp16,
    kWmmaBf16,
    kMmaPtx,
    kMmaLdm,
    kMmaOpt,
    kTileFamily,
    kSplitK,
    kStreamK,
    kCutlass,
    kCublas,
};

// C = alpha * op(A) * op(B) + beta * C, with A m by k, B k by n, C m by n after
// the ops are applied.
//
// Layout and transpose both reach cuBLAS through one identity. cuBLAS is column
// major, and a row major matrix occupies the same bytes as its transpose read
// column major. So a row major call becomes the column major call
// C_t(n by m) = op(B)_t * op(A)_t: swap the operands, swap m and n, and keep
// each operand's own op flag with it. A column major call passes straight
// through. The hand written kernels are row major, no transpose, packed only;
// anything else dispatches to cuBLAS and says so through the chosen out-param.
struct GemmDesc {
    Layout layout = Layout::kRowMajor;
    Op op_a = Op::kN;
    Op op_b = Op::kN;
    std::int64_t m = 0;
    std::int64_t n = 0;
    std::int64_t k = 0;
    DType dt_a = DType::kR32F;
    DType dt_b = DType::kR32F;
    DType dt_c = DType::kR32F;
    std::int64_t lda = 0;
    std::int64_t ldb = 0;
    std::int64_t ldc = 0;
    std::int64_t stride_a = 0;  // strided batched; ignored when batch_count is 1
    std::int64_t stride_b = 0;
    std::int64_t stride_c = 0;
    std::int32_t batch_count = 1;
    Algo algo = Algo::kAuto;
};

// Stable short names, never null. These are what a benchmark row or a chosen
// report prints.
CKL_EXPORT const char* algo_name(Algo a);
CKL_EXPORT const char* dtype_name(DType t);
CKL_EXPORT const char* op_name(Op o);
CKL_EXPORT const char* layout_name(Layout l);

// Element size in bytes.
CKL_EXPORT std::size_t dtype_size(DType t);

}  // namespace ckl
