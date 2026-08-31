// GEMM dispatch: descriptor validation, the kAuto heuristic, the architecture
// gate, and the cuBLAS fallback path that covers every layout and transpose the
// hand written kernels do not.
//
// Two rules shape the whole file. First, *chosen is always written when it is
// non-null, on success and on failure alike, so a caller can see which rung ran
// and a regression that makes every shape fall back cannot pass a test.
// Second, an explicitly named algorithm is never rerouted: if the shape cannot
// take it the call returns kNotSupported and says why. kAuto is the only mode
// allowed to choose, and it says what it chose.
//
// The kAuto heuristic below is provisional. It divides the shape by the block
// factors each kernel needs and picks the largest tile that fits, which is the
// right shape of rule but not yet a tuned one. Part 05 replaces the constants
// with the tile sweep's measurements; until then this is a hypothesis, not a
// measured policy.

#include "ckl/gemm.hpp"

#include <cstdint>
#include <string>

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "ckl/cuda_check.hpp"
#include "ckl/status.hpp"
#include "ckl/types.hpp"

#include "detail/last_error.hpp"

namespace ckl {

namespace {

// ---------------------------------------------------------------------------
// Descriptor geometry
// ---------------------------------------------------------------------------

struct Stored {
    std::int64_t rows;
    std::int64_t cols;
};

Stored stored_a(const GemmDesc& d) {
    return d.op_a == Op::kN ? Stored{d.m, d.k} : Stored{d.k, d.m};
}

Stored stored_b(const GemmDesc& d) {
    return d.op_b == Op::kN ? Stored{d.k, d.n} : Stored{d.n, d.k};
}

// Smallest leading dimension the layout allows for a matrix of this shape.
std::int64_t min_ld(Layout layout, Stored s) {
    return layout == Layout::kRowMajor ? s.cols : s.rows;
}

bool packed(const GemmDesc& d) {
    return d.lda == min_ld(d.layout, stored_a(d)) && d.ldb == min_ld(d.layout, stored_b(d)) &&
           d.ldc == min_ld(d.layout, Stored{d.m, d.n});
}

// What the hand written kernels can take at all: row major, no transpose,
// packed, unbatched, and small enough that the int dimensions in their
// signatures are exact.
bool hand_kernel_shape(const GemmDesc& d) {
    constexpr std::int64_t kIntMax = 2147483647;
    return d.layout == Layout::kRowMajor && d.op_a == Op::kN && d.op_b == Op::kN && packed(d) &&
           d.batch_count == 1 && d.m <= kIntMax && d.n <= kIntMax && d.k <= kIntMax;
}

bool divides(const GemmDesc& d, std::int64_t bm, std::int64_t bn, std::int64_t bk) {
    return d.m % bm == 0 && d.n % bn == 0 && d.k % bk == 0;
}

bool is_fp16_in(const GemmDesc& d) {
    return d.dt_a == DType::kR16F && d.dt_b == DType::kR16F && d.dt_c == DType::kR32F;
}

bool is_bf16_in(const GemmDesc& d) {
    return d.dt_a == DType::kR16BF && d.dt_b == DType::kR16BF && d.dt_c == DType::kR32F;
}

bool is_fp32(const GemmDesc& d) {
    return d.dt_a == DType::kR32F && d.dt_b == DType::kR32F && d.dt_c == DType::kR32F;
}

// ---------------------------------------------------------------------------
// Per algorithm requirements
// ---------------------------------------------------------------------------

// Compute capability the algorithm's device body needs. Below it the body is
// compiled out, so the launcher must refuse rather than run an empty kernel.
int required_cc(Algo a) {
    switch (a) {
        case Algo::kCpAsync:
        case Algo::kWmmaFp16:
        case Algo::kWmmaBf16:
        case Algo::kMmaPtx:
        case Algo::kMmaLdm:
        case Algo::kMmaOpt:
            return 80;
        default:
            return 0;
    }
}

// Why this shape cannot take this algorithm, or an empty string when it can.
std::string why_not(Algo a, const GemmDesc& d) {
    if (!hand_kernel_shape(d)) {
        return "the hand written kernels take row major, no transpose, packed leading dimensions, "
               "one batch, and dimensions inside int range";
    }
    switch (a) {
        case Algo::kNaive:
        case Algo::kTiled:
            return is_fp32(d) ? "" : "this rung is FP32 in and FP32 out only";
        case Algo::kRegister:
        case Algo::kCpAsync:
            if (!is_fp32(d)) {
                return "this rung is FP32 in and FP32 out only";
            }
            return divides(d, 128, 128, 8)
                       ? ""
                       : "this rung needs m and n divisible by 128 and k divisible by 8";
        case Algo::kWmmaFp16:
            if (!is_fp16_in(d)) {
                return "this rung is FP16 in and FP32 out only";
            }
            return divides(d, 64, 64, 16)
                       ? ""
                       : "this rung needs m and n divisible by 64 and k divisible by 16";
        case Algo::kWmmaBf16:
            if (!is_bf16_in(d)) {
                return "this rung is BF16 in and FP32 out only";
            }
            return divides(d, 64, 64, 16)
                       ? ""
                       : "this rung needs m and n divisible by 64 and k divisible by 16";
        case Algo::kMmaPtx:
        case Algo::kMmaLdm:
            if (!is_fp16_in(d)) {
                return "this rung is FP16 in and FP32 out only";
            }
            return divides(d, 64, 64, 16)
                       ? ""
                       : "this rung needs m and n divisible by 64 and k divisible by 16";
        case Algo::kMmaOpt:
            if (!is_fp16_in(d)) {
                return "this rung is FP16 in and FP32 out only";
            }
            return divides(d, 128, 128, 32)
                       ? ""
                       : "this rung needs m and n divisible by 128 and k divisible by 32";
        default:
            return "";
    }
}

// ---------------------------------------------------------------------------
// The provisional kAuto heuristic
// ---------------------------------------------------------------------------

Algo pick_auto(const GemmDesc& d) {
    if (!hand_kernel_shape(d) || d.k <= 0) {
        return Algo::kCublas;
    }
    if (is_fp16_in(d)) {
        if (divides(d, 128, 128, 32)) {
            return Algo::kMmaOpt;
        }
        if (divides(d, 64, 64, 16)) {
            return Algo::kWmmaFp16;
        }
        return Algo::kCublas;
    }
    if (is_bf16_in(d)) {
        if (divides(d, 64, 64, 16)) {
            return Algo::kWmmaBf16;
        }
        return Algo::kCublas;
    }
    if (is_fp32(d)) {
        // Large enough that the double buffered kernel's staging pays for
        // itself; below that its 128 by 128 blocks quantize badly on 48 SMs.
        const bool large = d.m >= 512 && d.n >= 512 && d.k >= 512;
        if (large && divides(d, 128, 128, 8)) {
            return Algo::kCpAsync;
        }
        // One block per output tile beats a launch into cuBLAS's heuristics at
        // this size, and it keeps the whole call on one stream.
        const bool small = d.m <= 128 && d.n <= 128 && d.k <= 128;
        if (small) {
            return Algo::kNaive;
        }
        return Algo::kCublas;
    }
    return Algo::kCublas;
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

Status validate(const GemmDesc& d, const void* alpha, const void* a, const void* b,
                const void* beta, const void* c) {
    if (d.m < 0 || d.n < 0 || d.k < 0) {
        detail::set_last_error("gemm: m, n and k must all be non negative");
        return Status::kInvalidValue;
    }
    if (d.batch_count < 1) {
        detail::set_last_error("gemm: batch_count must be at least 1");
        return Status::kInvalidValue;
    }
    if (alpha == nullptr || beta == nullptr) {
        detail::set_last_error("gemm: alpha and beta are host pointers and must not be null");
        return Status::kInvalidValue;
    }
    const std::int64_t need_a = min_ld(d.layout, stored_a(d));
    const std::int64_t need_b = min_ld(d.layout, stored_b(d));
    const std::int64_t need_c = min_ld(d.layout, Stored{d.m, d.n});
    if (d.lda < need_a || d.ldb < need_b || d.ldc < need_c) {
        detail::set_last_error("gemm: leading dimensions too small for the described shape, need "
                               "lda at least " +
                               std::to_string(need_a) + ", ldb at least " + std::to_string(need_b) +
                               ", ldc at least " + std::to_string(need_c));
        return Status::kInvalidValue;
    }
    if (d.m > 0 && d.n > 0) {
        if (c == nullptr) {
            detail::set_last_error("gemm: c is null for a non empty output");
            return Status::kInvalidValue;
        }
        if (d.k > 0 && (a == nullptr || b == nullptr)) {
            detail::set_last_error("gemm: a or b is null for a non empty contraction");
            return Status::kInvalidValue;
        }
    }
    if (d.dt_c != DType::kR32F) {
        detail::set_last_error("gemm: the only output type in 1.1.0 is 32 bit float");
        return Status::kNotSupported;
    }
    if (d.dt_a != d.dt_b) {
        detail::set_last_error("gemm: mixed input types are not supported");
        return Status::kNotSupported;
    }
    return Status::kSuccess;
}

// ---------------------------------------------------------------------------
// cuBLAS path
// ---------------------------------------------------------------------------

cublasOperation_t to_cublas(Op o) {
    // kC is a plain transpose here: every type the library ships is real.
    return o == Op::kN ? CUBLAS_OP_N : CUBLAS_OP_T;
}

cudaDataType to_cuda_dtype(DType t) {
    switch (t) {
        case DType::kR16F:
            return CUDA_R_16F;
        case DType::kR16BF:
            return CUDA_R_16BF;
        case DType::kR32F:
        default:
            return CUDA_R_32F;
    }
}

void check_cublas(cublasStatus_t s, const char* expr) {
    if (s != CUBLAS_STATUS_SUCCESS) {
        Status mapped = Status::kExecutionFailed;
        switch (s) {
            case CUBLAS_STATUS_NOT_INITIALIZED:
                mapped = Status::kNotInitialized;
                break;
            case CUBLAS_STATUS_ALLOC_FAILED:
                mapped = Status::kAllocFailed;
                break;
            case CUBLAS_STATUS_INVALID_VALUE:
                mapped = Status::kInvalidValue;
                break;
            case CUBLAS_STATUS_ARCH_MISMATCH:
                mapped = Status::kArchMismatch;
                break;
            case CUBLAS_STATUS_NOT_SUPPORTED:
                mapped = Status::kNotSupported;
                break;
            default:
                break;
        }
        throw Error(mapped, std::string("cuBLAS error ") + cublasGetStatusName(s) + ": " + expr);
    }
}

// C = beta * C for an empty contraction. beta zero has to clear C rather than
// scale it, because a NaN in C is legal input when beta is zero and 0 * NaN is
// still NaN.
void scale_c(cublasHandle_t h, const GemmDesc& d, float beta, void* c, cudaStream_t stream) {
    if (beta == 1.0f) {
        return;
    }
    const std::int64_t lead = d.layout == Layout::kRowMajor ? d.n : d.m;
    const std::int64_t other = d.layout == Layout::kRowMajor ? d.m : d.n;
    auto* base = static_cast<float*>(c);
    if (beta == 0.0f) {
        CKL_CUDA_CHECK(cudaMemset2DAsync(base, static_cast<std::size_t>(d.ldc) * sizeof(float), 0,
                                         static_cast<std::size_t>(lead) * sizeof(float),
                                         static_cast<std::size_t>(other), stream));
        return;
    }
    for (std::int64_t i = 0; i < other; ++i) {
        check_cublas(cublasSscal_64(h, lead, &beta, base + i * d.ldc, 1), "cublasSscal_64");
    }
}

void run_cublas(const Context& ctx, const GemmDesc& d, const void* alpha, const void* a,
                const void* b, const void* beta, void* c) {
    auto* h = static_cast<cublasHandle_t>(ctx.cublas());
    check_cublas(cublasSetStream(h, ctx.stream()), "cublasSetStream");

    if (d.k == 0) {
        scale_c(h, d, *static_cast<const float*>(beta), c, ctx.stream());
        return;
    }

    const cublasOperation_t opa = to_cublas(d.op_a);
    const cublasOperation_t opb = to_cublas(d.op_b);
    const cudaDataType ta = to_cuda_dtype(d.dt_a);
    const cudaDataType tb = to_cuda_dtype(d.dt_b);
    const cudaDataType tc = to_cuda_dtype(d.dt_c);

    if (d.batch_count > 1) {
        if (d.layout == Layout::kRowMajor) {
            check_cublas(cublasGemmStridedBatchedEx_64(
                             h, opb, opa, d.n, d.m, d.k, alpha, b, tb, d.ldb, d.stride_b, a, ta,
                             d.lda, d.stride_a, beta, c, tc, d.ldc, d.stride_c, d.batch_count,
                             CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
                         "cublasGemmStridedBatchedEx_64");
        } else {
            check_cublas(cublasGemmStridedBatchedEx_64(
                             h, opa, opb, d.m, d.n, d.k, alpha, a, ta, d.lda, d.stride_a, b, tb,
                             d.ldb, d.stride_b, beta, c, tc, d.ldc, d.stride_c, d.batch_count,
                             CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
                         "cublasGemmStridedBatchedEx_64");
        }
        return;
    }

    // Row major goes through the transpose identity: a row major buffer read
    // column major is its own transpose, so asking for C_t = op(B)_t * op(A)_t
    // with m and n swapped writes exactly the row major C we want, and each
    // operand keeps its own op flag.
    if (d.layout == Layout::kRowMajor) {
        check_cublas(cublasGemmEx_64(h, opb, opa, d.n, d.m, d.k, alpha, b, tb, d.ldb, a, ta, d.lda,
                                     beta, c, tc, d.ldc, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
                     "cublasGemmEx_64");
    } else {
        check_cublas(cublasGemmEx_64(h, opa, opb, d.m, d.n, d.k, alpha, a, ta, d.lda, b, tb, d.ldb,
                                     beta, c, tc, d.ldc, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
                     "cublasGemmEx_64");
    }
}

// ---------------------------------------------------------------------------
// Hand written kernel path
// ---------------------------------------------------------------------------

void run_hand_kernel(Algo a, const GemmDesc& d, float alpha, const void* pa, const void* pb,
                     float beta, void* pc, cudaStream_t stream) {
    const int m = static_cast<int>(d.m);
    const int n = static_cast<int>(d.n);
    const int k = static_cast<int>(d.k);
    const auto* f32a = static_cast<const float*>(pa);
    const auto* f32b = static_cast<const float*>(pb);
    const auto* f16a = static_cast<const __half*>(pa);
    const auto* f16b = static_cast<const __half*>(pb);
    const auto* bf16a = static_cast<const __nv_bfloat16*>(pa);
    const auto* bf16b = static_cast<const __nv_bfloat16*>(pb);
    auto* out = static_cast<float*>(pc);

    switch (a) {
        case Algo::kNaive:
            gemm_naive(f32a, f32b, out, m, n, k, alpha, beta, stream);
            return;
        case Algo::kTiled:
            gemm_tiled(f32a, f32b, out, m, n, k, alpha, beta, stream);
            return;
        case Algo::kRegister:
            gemm_register(f32a, f32b, out, m, n, k, alpha, beta, stream);
            return;
        case Algo::kCpAsync:
            gemm_cp_async(f32a, f32b, out, m, n, k, alpha, beta, stream);
            return;
        case Algo::kWmmaFp16:
            gemm_wmma_fp16(f16a, f16b, out, m, n, k, alpha, beta, stream);
            return;
        case Algo::kWmmaBf16:
            gemm_wmma_bf16(bf16a, bf16b, out, m, n, k, alpha, beta, stream);
            return;
        case Algo::kMmaPtx:
            gemm_mma_ptx(f16a, f16b, out, m, n, k, alpha, beta, stream);
            return;
        case Algo::kMmaLdm:
            gemm_mma_ldm(f16a, f16b, out, m, n, k, alpha, beta, stream);
            return;
        case Algo::kMmaOpt:
            gemm_mma_opt(f16a, f16b, out, m, n, k, alpha, beta, stream);
            return;
        default:
            throw Error(Status::kInternal, "dispatch reached a hand kernel case it does not own");
    }
}

}  // namespace

Algo gemm_query(const Context& ctx, const GemmDesc& desc) {
    // The heuristic is shape driven, but a device that cannot run the chosen
    // rung would make the answer a lie, so the architecture gate applies here
    // too and the answer degrades to the vendor path.
    const Algo picked = pick_auto(desc);
    if (ctx.compute_capability() < required_cc(picked)) {
        return Algo::kCublas;
    }
    return picked;
}

std::size_t gemm_workspace_size(const Context& ctx, const GemmDesc& desc) {
    (void)ctx;
    (void)desc;
    // Every path shipped in 1.1.0 runs out of registers and shared memory only.
    return 0;
}

Status gemm(Context& ctx, const GemmDesc& desc, const void* alpha, const void* a, const void* b,
            const void* beta, void* c, Algo* chosen) {
    detail::clear_last_error();

    // Written before any early return, so a caller reading it after a failure
    // still sees what the dispatcher was aiming at.
    Algo picked = desc.algo == Algo::kAuto ? pick_auto(desc) : desc.algo;
    if (chosen != nullptr) {
        *chosen = picked;
    }

    const Status valid = validate(desc, alpha, a, b, beta, c);
    if (valid != Status::kSuccess) {
        return valid;
    }

    switch (picked) {
        case Algo::kTileFamily:
        case Algo::kSplitK:
        case Algo::kStreamK:
        case Algo::kCutlass:
            detail::set_last_error(std::string("gemm: algorithm ") + algo_name(picked) +
                                   " is declared for ABI stability but not implemented in 1.1.0");
            return Status::kNotSupported;
        default:
            break;
    }

    // Strided batched work goes to cuBLAS. Batching the hand kernels is future
    // work; looping them here would be a silent reinterpretation of the call.
    if (desc.batch_count > 1 && picked != Algo::kCublas) {
        if (desc.algo == Algo::kAuto) {
            picked = Algo::kCublas;
            if (chosen != nullptr) {
                *chosen = picked;
            }
        } else {
            detail::set_last_error(std::string("gemm: algorithm ") + algo_name(picked) +
                                   " has no strided batched form; use kAuto or kCublas");
            return Status::kNotSupported;
        }
    }

    if (picked != Algo::kCublas) {
        const std::string reason = why_not(picked, desc);
        if (!reason.empty()) {
            detail::set_last_error(std::string("gemm: algorithm ") + algo_name(picked) +
                                   " cannot take this shape: " + reason);
            return Status::kNotSupported;
        }
        const int need = required_cc(picked);
        if (ctx.compute_capability() < need) {
            // kAuto is allowed to choose again, and chosen reports where it
            // landed. An explicitly named algorithm is not rerouted: the caller
            // asked for that path and gets told the device cannot run it.
            if (desc.algo == Algo::kAuto) {
                picked = Algo::kCublas;
                if (chosen != nullptr) {
                    *chosen = picked;
                }
            } else {
                detail::set_last_error(
                    std::string("gemm: algorithm ") + algo_name(picked) +
                    " needs compute capability " + std::to_string(need / 10) + "." +
                    std::to_string(need % 10) + "; this device reports " +
                    std::to_string(ctx.compute_capability() / 10) + "." +
                    std::to_string(ctx.compute_capability() % 10));
                return Status::kArchMismatch;
            }
        }
    }

    if (desc.m == 0 || desc.n == 0) {
        return Status::kSuccess;  // no output elements, nothing to write
    }

    try {
        if (picked == Algo::kCublas) {
            run_cublas(ctx, desc, alpha, a, b, beta, c);
        } else if (desc.k == 0) {
            // An empty contraction is C = beta * C. The hand kernels that carry
            // a beta only path handle it, but routing it through one place keeps
            // the zero beta clearing rule in a single implementation.
            auto* h = static_cast<cublasHandle_t>(ctx.cublas());
            check_cublas(cublasSetStream(h, ctx.stream()), "cublasSetStream");
            scale_c(h, desc, *static_cast<const float*>(beta), c, ctx.stream());
        } else {
            run_hand_kernel(picked, desc, *static_cast<const float*>(alpha), a, b,
                            *static_cast<const float*>(beta), c, ctx.stream());
            CKL_CUDA_LAST_ERROR(false);
        }
    } catch (const Error& e) {
        detail::set_last_error(e.what());
        return e.status();
    } catch (const std::exception& e) {
        detail::set_last_error(e.what());
        return Status::kInternal;
    }

    return Status::kSuccess;
}

}  // namespace ckl
