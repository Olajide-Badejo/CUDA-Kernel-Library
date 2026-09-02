// The CUTLASS reference line: the third line on the GEMM ladder.
//
// This rung exists so the hand written tensor kernels are measured against a
// tuned open source template library as well as against cuBLAS. cuBLAS is a
// closed binary, so a gap against it says only that a gap exists. CUTLASS is
// source, so a gap against it can be read technique by technique, which is what
// docs/cutlass.md is for.
//
// Why this instantiation, and not a CollectiveBuilder. As of the pinned release
// v4.7.1 there is no FP16 CollectiveBuilder path for SM120 at all: the SM120
// mainloop builder in include/cutlass/gemm/collective/builders/sm120_mma_builder.inl
// opens with
//
//   static_assert(detail::is_sm10x_f8f6f4_element<ElementA>() &&
//                 detail::is_sm10x_f8f6f4_element<ElementB>(), ...)
//
// at line 80, and closes with "Non-blockscaled collective builder only supports
// F8F6F4 MMA." at line 115. Half precision is not an f8f6f4 element, so a
// builder request for FP16 on SM120 fails to compile rather than producing a
// slow kernel. The blockscaled SM120 builders are narrow precision by
// construction. So the FP16 path here is the 2.x device API over an Ampere
// style multistage mainloop, which is the same shape of kernel cuBLAS
// effectively ships on this part: cp.async global to shared, ldmatrix into
// fragments, mma.sync.aligned.m16n8k16, FP32 accumulate. Every one of those
// instructions assembles on plain sm_120, so an arch::Sm80 kernel is a real
// kernel here and not an emulation.
//
// The instantiation, chosen to match the top hand kernel so the comparison is
// between implementations of one design rather than between two designs:
//
//   element A, B      cutlass::half_t          FP16 storage
//   element C, accum  float                    FP32 accumulate, FP32 output
//   layouts           row major A, B, C        the ladder convention
//   op class          OpClassTensorOp          tensor cores, not SIMT
//   arch              cutlass::arch::Sm80      multistage cp.async mainloop
//   threadblock tile  128 x 128 x 32           same as gemm_mma_opt
//   warp tile         64 x 64 x 32             four warps, 128 threads
//   instruction       16 x 8 x 16              the largest dense FP16 MMA here
//   stages            3                        same pipeline depth as mma_opt
//   alignment A, B    8 halves (16 bytes)      one cp.async.cg.128 per lane
//   epilogue          LinearCombination        alpha * acc + beta * C
//
// Three stages of 128 x 32 plus 32 x 128 halves is 49152 bytes of shared memory,
// over the 48 KB static ceiling, so CUTLASS raises the dynamic shared memory
// attribute once per device from GemmUniversalBase::init_device_props. The
// sm_120 opt-in ceiling is 101376 bytes, so it fits.
//
// Shape rule. Row major A has leading dimension k and row major B and C have
// leading dimension n, so the 16 byte access needs k and n divisible by 8. m is
// free: CUTLASS predicates the M edge. Anything else is refused rather than
// rerouted, in the dispatcher through why_not and here through a throw.
//
// CUTLASS stays private to this translation unit. Nothing above it in the tree
// sees a CUTLASS type, so the public headers and the C ABI are untouched by the
// dependency and a consumer who does not want it does not get it.

#include <string>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "ckl/context.hpp"
#include "ckl/cuda_check.hpp"
#include "ckl/gemm.hpp"
#include "ckl/status.hpp"

#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/device/gemm_universal.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/gemm/threadblock/threadblock_swizzle.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>

namespace ckl {

namespace {

// cp.async, ldmatrix and mma.sync.m16n8k16 together: sm_80 is the floor, the
// same floor every other tensor rung on the ladder carries.
constexpr int kMinArch = 80;

// Halves per vector access on A and B. 8 halves is 16 bytes, which is the widest
// cp.async this mainloop issues.
constexpr int kAlignIn = 8;

using ElementIn = cutlass::half_t;
using ElementOut = float;
using ElementAccum = float;
using LayoutRow = cutlass::layout::RowMajor;

// Floats per vector store in the epilogue: 128 bits of float is 4.
constexpr int kAlignOut = 128 / cutlass::sizeof_bits<ElementOut>::value;

using EpilogueOp =
    cutlass::epilogue::thread::LinearCombination<ElementOut, kAlignOut, ElementAccum, ElementAccum>;

using CutlassGemm = cutlass::gemm::device::GemmUniversal<
    ElementIn, LayoutRow, ElementIn, LayoutRow, ElementOut, LayoutRow, ElementAccum,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>, cutlass::gemm::GemmShape<16, 8, 16>, EpilogueOp,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3, kAlignIn, kAlignIn,
    cutlass::arch::OpMultiplyAdd>;

// C = beta * C for an empty contraction. CUTLASS is not asked to run a zero K
// problem; the answer is defined by BLAS and is one memory bound pass, so it is
// written here rather than pushed into a template that would have to be
// instantiated for a case with no math in it. beta zero clears rather than
// scales, because a NaN in C is legal input when beta is zero.
__global__ void cutlass_scale_c_kernel(float* c, int m, int n, float beta) {
    const long long total = static_cast<long long>(m) * n;
    const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    c[idx] = beta == 0.0f ? 0.0f : beta * c[idx];
}

void scale_c_only(float* c, int m, int n, float beta, cudaStream_t stream) {
    constexpr int kBlock = 256;
    const long long total = static_cast<long long>(m) * n;
    const int grid = static_cast<int>((total + kBlock - 1) / kBlock);
    cutlass_scale_c_kernel<<<grid, kBlock, 0, stream>>>(c, m, n, beta);
    CKL_CUDA_LAST_ERROR(false);
}

// One CUTLASS status turned into one ckl status, with the CUTLASS text kept so
// the message names what the template refused.
[[noreturn]] void raise(cutlass::Status s, const char* what) {
    const Status mapped = s == cutlass::Status::kErrorNotSupported ||
                                  s == cutlass::Status::kErrorInvalidProblem ||
                                  s == cutlass::Status::kErrorMisalignedOperand
                              ? Status::kNotSupported
                              : Status::kExecutionFailed;
    throw Error(mapped, std::string(what) + ": CUTLASS reports " + cutlassGetStatusString(s));
}

}  // namespace

bool gemm_cutlass_supports(int m, int n, int k) {
    return m >= 0 && n >= 0 && k >= 0 && n % kAlignIn == 0 && k % kAlignIn == 0;
}

GemmTile gemm_cutlass_tile() {
    GemmTile t;
    t.m = CutlassGemm::ThreadblockShape::kM;
    t.n = CutlassGemm::ThreadblockShape::kN;
    t.k = CutlassGemm::ThreadblockShape::kK;
    t.warps_m = CutlassGemm::ThreadblockShape::kM / CutlassGemm::WarpShape::kM;
    t.warps_n = CutlassGemm::ThreadblockShape::kN / CutlassGemm::WarpShape::kN;
    return t;
}

void gemm_cutlass(const __half* a, const __half* b, float* c, int m, int n, int k, float alpha,
                  float beta, cudaStream_t stream) {
    if (m <= 0 || n <= 0) {
        return;  // no output elements, so nothing to write
    }
    if (!gemm_cutlass_supports(m, n, k)) {
        // No reroute to another rung. A CUTLASS row that measured a hand kernel
        // would be worse than no row at all.
        throw Error(Status::kNotSupported,
                    "gemm_cutlass needs n and k divisible by 8, because row major A, B and C put "
                    "k, n and n on the contiguous axis and the mainloop moves 16 bytes per access");
    }
    detail::require_arch(kMinArch, "gemm_cutlass");

    if (k == 0) {
        scale_c_only(c, m, n, beta, stream);
        return;
    }

    const cutlass::gemm::GemmCoord problem(m, n, k);
    typename CutlassGemm::Arguments args(
        cutlass::gemm::GemmUniversalMode::kGemm, problem,
        1,  // one K slice: split-K is a rung of its own on this ladder
        typename EpilogueOp::Params(alpha, beta), reinterpret_cast<const void*>(a),
        reinterpret_cast<const void*>(b), reinterpret_cast<const void*>(c),
        reinterpret_cast<void*>(c), static_cast<std::int64_t>(0), static_cast<std::int64_t>(0),
        static_cast<std::int64_t>(0), static_cast<std::int64_t>(0),
        static_cast<std::int64_t>(k),   // lda: A is m by k, row major
        static_cast<std::int64_t>(n),   // ldb: B is k by n, row major
        static_cast<std::int64_t>(n),   // ldc
        static_cast<std::int64_t>(n));  // ldd

    const cutlass::Status ok = CutlassGemm::can_implement(args);
    if (ok != cutlass::Status::kSuccess) {
        raise(ok, "gemm_cutlass cannot run this problem");
    }

    // A single K slice in kGemm mode needs no semaphore and no reduction
    // scratch, so this is zero. It is checked rather than assumed: a workspace
    // allocated here would be an allocation inside a timed benchmark region.
    const std::size_t workspace = CutlassGemm::get_workspace_size(args);
    if (workspace != 0) {
        throw Error(Status::kInternal,
                    "gemm_cutlass expected a zero sized workspace for a single K slice and the "
                    "template asked for " +
                        std::to_string(workspace) + " bytes");
    }

    CutlassGemm op;
    const cutlass::Status started = op.initialize(args, nullptr, stream);
    if (started != cutlass::Status::kSuccess) {
        raise(started, "gemm_cutlass failed to initialize");
    }
    const cutlass::Status ran = op.run(stream);
    if (ran != cutlass::Status::kSuccess) {
        raise(ran, "gemm_cutlass failed to launch");
    }
    CKL_CUDA_LAST_ERROR(false);
}

}  // namespace ckl
