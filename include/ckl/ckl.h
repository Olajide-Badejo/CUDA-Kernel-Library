#ifndef CKL_CKL_H
#define CKL_CKL_H

/**
 * @file ckl.h
 * @brief The C ABI of the CUDA Kernel Library.
 *
 * Pure C99: it includes <stdint.h>, <stddef.h> and the generated ckl_export.h,
 * and no CUDA header at all. Device pointers and streams cross as void*, so a C
 * or Fortran consumer needs no toolkit headers to call the library.
 *
 * Nothing but C crosses this boundary: no exceptions, no std::string, no
 * templates, no default arguments. Every entry point catches everything and
 * turns it into a ckl_status_t; the detail of the last failure on the calling
 * thread is available through ckl_last_error.
 *
 * alpha and beta are HOST values. A device pointer mode is a documented non
 * goal until CUDA graph capture support lands. Where a routine takes them as
 * const void*, they point at a float, because the compute type is always
 * 32 bit float.
 */

#include <stddef.h>
#include <stdint.h>

#include "ckl/ckl_export.h"

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Outcome of a CKL call; mirrors ckl::Status one for one. */
typedef enum {
    CKL_STATUS_SUCCESS = 0,     /**< The call did what it was asked to do. */
    CKL_STATUS_NOT_INITIALIZED, /**< No usable device, or a vendor handle would not initialize. */
    CKL_STATUS_INVALID_VALUE,   /**< A malformed argument: bad dimensions, a too small leading
                                     dimension, a null buffer. */
    CKL_STATUS_ARCH_MISMATCH,   /**< The running device cannot execute the selected path. */
    CKL_STATUS_NOT_SUPPORTED, /**< The shape or type combination is outside what this call takes. */
    CKL_STATUS_ALLOC_FAILED,  /**< A device or host allocation failed. */
    CKL_STATUS_EXECUTION_FAILED, /**< A launch or a vendor call failed at run time. */
    CKL_STATUS_INTERNAL          /**< A defect in the library; ckl_last_error says what happened. */
} ckl_status_t;

/** @brief Opaque handle to a ckl::Context. Create it with ckl_create. */
typedef struct ckl_context* ckl_handle_t;

/** @brief Transpose flag applied to an operand. */
typedef enum {
    CKL_OP_N = 0, /**< No transpose. */
    CKL_OP_T = 1, /**< Transpose. */
    CKL_OP_C = 2  /**< Conjugate transpose, treated as CKL_OP_T; every type here is real. */
} ckl_operation_t;

/** @brief How a matrix is stored in memory. */
typedef enum {
    CKL_ROW_MAJOR = 0, /**< Row elements adjacent; the leading dimension counts columns. */
    CKL_COL_MAJOR = 1  /**< Column elements adjacent; the leading dimension counts rows. */
} ckl_layout_t;

/** @brief Element type of a matrix operand. */
typedef enum {
    CKL_R_32F = 0, /**< 32 bit float. */
    CKL_R_16F = 1, /**< 16 bit IEEE half. */
    CKL_R_16BF = 2 /**< 16 bit bfloat. */
} ckl_datatype_t;

/** @brief Direction of a ckl_memcpy transfer. */
typedef enum {
    CKL_MEMCPY_H2D = 0, /**< Host to device. */
    CKL_MEMCPY_D2H = 1, /**< Device to host. */
    CKL_MEMCPY_D2D = 2  /**< Device to device. */
} ckl_memcpy_kind_t;

/**
 * @brief Every rung of the GEMM ladder plus the vendor baseline.
 *
 * The values match ckl::Algo one for one and in the same order. The four
 * entries after CKL_ALGO_MMA_OPT are declared for ABI stability and return
 * CKL_STATUS_NOT_SUPPORTED in 1.1.0.
 */
typedef enum {
    CKL_ALGO_AUTO = 0,    /**< Let the dispatcher choose and report the choice through chosen. */
    CKL_ALGO_NAIVE,       /**< One thread per output element, FP32. */
    CKL_ALGO_TILED,       /**< Shared memory tiled, FP32. */
    CKL_ALGO_REGISTER,    /**< Register blocked with float4 loads, FP32. */
    CKL_ALGO_CP_ASYNC,    /**< cp.async double buffered, FP32. */
    CKL_ALGO_WMMA_FP16,   /**< WMMA fragments, FP16 in, FP32 accumulate. */
    CKL_ALGO_WMMA_BF16,   /**< WMMA fragments, BF16 in, FP32 accumulate. */
    CKL_ALGO_MMA_PTX,     /**< Raw mma.sync PTX, FP16 in. */
    CKL_ALGO_MMA_LDM,     /**< mma.sync with ldmatrix fragment loads, FP16 in. */
    CKL_ALGO_MMA_OPT,     /**< The top tensor kernel: 128x128 tile, ldmatrix, cp.async. */
    CKL_ALGO_TILE_FAMILY, /**< Declared for ABI stability; not implemented in 1.1.0. */
    CKL_ALGO_SPLITK,      /**< Declared for ABI stability; not implemented in 1.1.0. */
    CKL_ALGO_STREAMK,     /**< Declared for ABI stability; not implemented in 1.1.0. */
    CKL_ALGO_CUTLASS,     /**< Declared for ABI stability; not implemented in 1.1.0. */
    CKL_ALGO_CUBLAS       /**< The vendor path, which takes every shape the hand kernels refuse. */
} ckl_algo_t;

/**
 * @brief Every CSR SpMV path plus the two cuSPARSE algorithms.
 *
 * The values match ckl::SpmvAlgo one for one and in the same order.
 * CKL_SPMV_CUSPARSE_DEFAULT is CUSPARSE_SPMV_ALG_DEFAULT and
 * CKL_SPMV_CUSPARSE_ALG2 is CUSPARSE_SPMV_CSR_ALG2, which buys a deterministic
 * reduction order at a cost; both are measured on every matrix, so a percentage
 * always says which one it is against.
 */
typedef enum {
    CKL_SPMV_AUTO = 0,         /**< Let the plan choose and report it through chosen. */
    CKL_SPMV_CSR_NAIVE,        /**< One thread per row. */
    CKL_SPMV_CSR_WARP,         /**< One 32 lane warp per row. */
    CKL_SPMV_CSR_VECTOR,       /**< A lane group of 2 to 32 per row. */
    CKL_SPMV_MERGE,            /**< Merge path after Merrill and Garland. */
    CKL_SPMV_SELL_C_SIGMA,     /**< Sliced ELLPACK, C = 32. */
    CKL_SPMV_BSR,              /**< Block CSR over the detected block dimension. */
    CKL_SPMV_CUSPARSE_DEFAULT, /**< cusparseSpMV, CUSPARSE_SPMV_ALG_DEFAULT. */
    CKL_SPMV_CUSPARSE_ALG2     /**< cusparseSpMV, CUSPARSE_SPMV_CSR_ALG2. */
} ckl_spmv_algo_t;

/**
 * @brief Direction of a transform. Neither direction is scaled, as in cuFFT.
 *
 * The values match ckl::FftDirection one for one and in the same order.
 */
typedef enum {
    CKL_FFT_FORWARD = 0, /**< exp(-2 pi i k n / N). */
    CKL_FFT_INVERSE = 1  /**< exp(+2 pi i k n / N), unscaled. */
} ckl_fft_direction_t;

/**
 * @brief Every 1D transform path plus the cuFFT baseline.
 *
 * The values match ckl::FftAlgo one for one and in the same order. Every hand
 * written rung is Stockham autosort; Cooley-Tukey with a separate bit reversal
 * pass is not implemented, and docs/fft.md derives why.
 */
typedef enum {
    CKL_FFT_AUTO = 0,        /**< Let the plan choose and report it through chosen. */
    CKL_FFT_RADIX2_GLOBAL,   /**< One global memory pass per radix 2 stage. */
    CKL_FFT_SHARED_RESIDENT, /**< One block per transform, both buffers in shared memory. */
    CKL_FFT_RADIX4_GLOBAL,   /**< Radix 4 butterflies, half the stages. */
    CKL_FFT_RADIX8_GLOBAL,   /**< Radix 8 butterflies mixed with one narrower stage. */
    CKL_FFT_FOUR_STEP,       /**< N = N1 * N2, batched shared transforms and transposes. */
    CKL_FFT_CUFFT            /**< cuFFT, planned once per size and reused. */
} ckl_fft_algo_t;

/**
 * @brief Every convolution path plus the cuFFT baseline.
 *
 * The values match ckl::ConvAlgo one for one and in the same order.
 */
typedef enum {
    CKL_CONV_AUTO = 0,        /**< Let the plan choose and report it through chosen. */
    CKL_CONV_FFT_SEPARATE,    /**< Forward, a standalone pointwise multiply, inverse. */
    CKL_CONV_FFT_FUSED,       /**< The multiply and the 1/L folded into the transform epilogues. */
    CKL_CONV_DIRECT_SHARED,   /**< Tiled time domain with the taps in shared memory. */
    CKL_CONV_DIRECT_CONSTANT, /**< The same tiling with the taps in constant memory. */
    CKL_CONV_CUFFT            /**< cuFFT forward, a pointwise multiply, cuFFT inverse. */
} ckl_conv_algo_t;

/**
 * @brief Element type a reduction or scan works on.
 *
 * The values match ckl::ScanDType one for one and in the same order. FP32
 * carries every operator; the other three carry CKL_SCAN_OP_SUM, and anything
 * else returns CKL_STATUS_NOT_SUPPORTED with the reason in ckl_last_error.
 */
typedef enum {
    CKL_SCAN_F32 = 0, /**< 32 bit float. */
    CKL_SCAN_F64 = 1, /**< 64 bit float. */
    CKL_SCAN_I32 = 2, /**< 32 bit signed integer. */
    CKL_SCAN_I64 = 3  /**< 64 bit signed integer. */
} ckl_scan_dtype_t;

/**
 * @brief The binary operator a reduction or scan combines with.
 *
 * The values match ckl::ScanOp one for one and in the same order.
 * CKL_SCAN_OP_LAST_NONZERO is associative and not commutative: `a op b` is `b`
 * unless `b` is zero, in which case it is `a`. It runs on every rung of the scan
 * ladder and on no rung of the reduction ladder, which returns
 * CKL_STATUS_NOT_SUPPORTED and says why.
 */
typedef enum {
    CKL_SCAN_OP_SUM = 0,         /**< Addition; identity zero. */
    CKL_SCAN_OP_MAX = 1,         /**< Maximum. */
    CKL_SCAN_OP_MIN = 2,         /**< Minimum. */
    CKL_SCAN_OP_LAST_NONZERO = 3 /**< The right operand unless it is zero. */
} ckl_scan_op_t;

/**
 * @brief Every rung of the reduction ladder plus the CUB baseline.
 *
 * The values match ckl::ReduceAlgo one for one and in the same order.
 */
typedef enum {
    CKL_REDUCE_AUTO = 0,      /**< Let the plan choose and report it through chosen. */
    CKL_REDUCE_ATOMIC,        /**< One atomic per element onto one global accumulator. */
    CKL_REDUCE_SHARED_TREE,   /**< Shared memory tree per block. */
    CKL_REDUCE_SHUFFLE,       /**< Warp shuffle, one atomic per block. */
    CKL_REDUCE_VEC4,          /**< 16 byte loads over persistent blocks. */
    CKL_REDUCE_SINGLE_PASS,   /**< One launch; the last block combines the partials. */
    CKL_REDUCE_TWO_PASS,      /**< The same partials, combined by a second launch. */
    CKL_REDUCE_DETERMINISTIC, /**< Fixed tree, fixed block count, no atomic accumulation. */
    CKL_REDUCE_KAHAN,         /**< Compensated summation; sum only. */
    CKL_REDUCE_CUB            /**< cub::DeviceReduce, the baseline. */
} ckl_reduce_algo_t;

/**
 * @brief Every rung of the scan ladder plus the CUB baseline.
 *
 * The values match ckl::ScanAlgo one for one and in the same order.
 */
typedef enum {
    CKL_SCAN_AUTO = 0,         /**< Let the plan choose and report it through chosen. */
    CKL_SCAN_HILLIS_STEELE,    /**< Hillis-Steele block scan, one item per thread. */
    CKL_SCAN_BLELLOCH,         /**< Work efficient upsweep and downsweep. */
    CKL_SCAN_THREE_KERNEL,     /**< Scan then propagate, about four passes. */
    CKL_SCAN_REDUCE_THEN_SCAN, /**< Reduce then scan, about three passes. */
    CKL_SCAN_LOOKBACK,         /**< Single pass decoupled look-back, about two passes. */
    CKL_SCAN_DETERMINISTIC,    /**< Fixed tree and fixed geometry; bit identical across runs. */
    CKL_SCAN_CUB               /**< cub::DeviceScan, the baseline. */
} ckl_scan_algo_t;

/**
 * @brief Numeric library version, 10000 * major + 100 * minor + patch.
 * @return 10100 for release 1.1.0.
 */
CKL_EXPORT int ckl_get_version(void);

/**
 * @brief Library version as text.
 * @return A static string such as "1.1.0", never null.
 */
CKL_EXPORT const char* ckl_get_version_string(void);

/**
 * @brief Symbolic name of a status code.
 * @param s Status to name.
 * @return A static string such as "CKL_STATUS_INVALID_VALUE", never null.
 */
CKL_EXPORT const char* ckl_status_string(ckl_status_t s);

/**
 * @brief Copies the detail of the last failure on this thread into buf.
 * @param buf Destination buffer; the message is always NUL terminated.
 * @param len Capacity of buf in bytes; the message is truncated to fit.
 * @return CKL_STATUS_SUCCESS, or CKL_STATUS_INVALID_VALUE when buf is NULL or len is zero.
 * @note The message is thread local and valid until the next CKL call on the
 *       same thread. Read it before you make another call.
 */
CKL_EXPORT ckl_status_t ckl_last_error(char* buf, size_t len);

/**
 * @brief Allocates device memory, so a consumer without the toolkit can hold operands.
 * @param dptr Out-param receiving the device pointer; NULL is written on failure.
 * @param bytes Size of the request. Zero is valid and yields a NULL pointer with success.
 * @return CKL_STATUS_SUCCESS, CKL_STATUS_ALLOC_FAILED, or CKL_STATUS_INVALID_VALUE
 *         when dptr is NULL.
 * @note A Fortran or C program that links only libckl has no cudaMalloc to call,
 *       and these four entry points are the whole of what it needs. The memory
 *       comes from cudaMalloc, so it is at least 256 byte aligned.
 */
CKL_EXPORT ckl_status_t ckl_device_malloc(void** dptr, size_t bytes);

/**
 * @brief Frees memory from ckl_device_malloc.
 * @param dptr Pointer to free; NULL is accepted and does nothing.
 * @return CKL_STATUS_SUCCESS, or a failure status if the runtime rejects the pointer.
 */
CKL_EXPORT ckl_status_t ckl_device_free(void* dptr);

/**
 * @brief Copies bytes between host and device memory.
 * @param dst Destination pointer, in the space the kind implies.
 * @param src Source pointer.
 * @param bytes Number of bytes to copy; zero succeeds and does nothing.
 * @param kind Direction of the transfer.
 * @return CKL_STATUS_SUCCESS, CKL_STATUS_INVALID_VALUE for a null pointer or an
 *         out of range kind, or CKL_STATUS_EXECUTION_FAILED.
 * @note Synchronous with respect to the host on the default stream, the same
 *       contract cudaMemcpy carries. It does not observe a handle's stream.
 */
CKL_EXPORT ckl_status_t ckl_memcpy(void* dst, const void* src, size_t bytes,
                                   ckl_memcpy_kind_t kind);

/**
 * @brief Blocks until every stream on the current device has finished.
 * @return CKL_STATUS_SUCCESS, or the mapped failure the runtime reported.
 */
CKL_EXPORT ckl_status_t ckl_device_synchronize(void);

/**
 * @brief Creates a handle; the C face of the ckl::Context constructor.
 * @param h Out-param receiving the handle; NULL is written on failure.
 * @return CKL_STATUS_SUCCESS, CKL_STATUS_NOT_INITIALIZED when there is no usable
 *         device, or CKL_STATUS_INVALID_VALUE when h is NULL.
 * @note A handle is externally synchronized: one handle per thread, or a lock of
 *       your own around every call that touches a shared one.
 */
CKL_EXPORT ckl_status_t ckl_create(ckl_handle_t* h);

/**
 * @brief Destroys a handle and its vendor handles.
 * @param h Handle to destroy.
 * @return CKL_STATUS_SUCCESS, or CKL_STATUS_INVALID_VALUE when h is NULL.
 */
CKL_EXPORT ckl_status_t ckl_destroy(ckl_handle_t h);

/**
 * @brief Sets the stream the handle enqueues on.
 * @param h Handle to configure.
 * @param cuda_stream A cudaStream_t passed as void*; NULL means the default stream.
 * @return CKL_STATUS_SUCCESS, or CKL_STATUS_INVALID_VALUE when h is NULL.
 * @note This does not synchronize. Work already enqueued stays on its old stream.
 */
CKL_EXPORT ckl_status_t ckl_set_stream(ckl_handle_t h, void* cuda_stream);

/**
 * @brief Hands the handle a caller owned scratch buffer.
 * @param h Handle to configure.
 * @param ws Device pointer to the scratch, or NULL for none.
 * @param bytes Size of the buffer.
 * @return CKL_STATUS_SUCCESS, or CKL_STATUS_INVALID_VALUE when h is NULL.
 * @note The library never frees this buffer. Nothing in 1.1.0 needs one.
 */
CKL_EXPORT ckl_status_t ckl_set_workspace(ckl_handle_t h, void* ws, size_t bytes);

/**
 * @brief Bytes of scratch the described GEMM needs.
 * @param h Handle the call would run on.
 * @param layout Storage order of all three matrices.
 * @param opa Transpose flag for A.
 * @param opb Transpose flag for B.
 * @param m Rows of op(A) and of C.
 * @param n Columns of op(B) and of C.
 * @param k Contraction extent.
 * @param dta Element type of A.
 * @param lda Leading dimension of A.
 * @param dtb Element type of B.
 * @param ldb Leading dimension of B.
 * @param dtc Element type of C.
 * @param ldc Leading dimension of C.
 * @param algo Requested path, or CKL_ALGO_AUTO.
 * @param bytes Out-param receiving the answer.
 * @return CKL_STATUS_SUCCESS, or CKL_STATUS_INVALID_VALUE for a null argument or
 *         an out of range enum.
 * @note Every path shipped in 1.1.0 answers zero; the entry point exists so a
 *       later split-K or stream-K rung can ask for scratch without an ABI break.
 */
CKL_EXPORT ckl_status_t ckl_gemm_workspace_size(ckl_handle_t h, ckl_layout_t layout,
                                                ckl_operation_t opa, ckl_operation_t opb, int64_t m,
                                                int64_t n, int64_t k, ckl_datatype_t dta,
                                                int64_t lda, ckl_datatype_t dtb, int64_t ldb,
                                                ckl_datatype_t dtc, int64_t ldc, ckl_algo_t algo,
                                                size_t* bytes);

/**
 * @brief What CKL_ALGO_AUTO would pick for the described GEMM.
 * @param h Handle whose device capability gates the answer.
 * @param layout Storage order of all three matrices.
 * @param opa Transpose flag for A.
 * @param opb Transpose flag for B.
 * @param m Rows of op(A) and of C.
 * @param n Columns of op(B) and of C.
 * @param k Contraction extent.
 * @param dta Element type of A.
 * @param lda Leading dimension of A.
 * @param dtb Element type of B.
 * @param ldb Leading dimension of B.
 * @param dtc Element type of C.
 * @param ldc Leading dimension of C.
 * @param chosen Out-param receiving the algorithm the dispatcher would run.
 * @return CKL_STATUS_SUCCESS, or CKL_STATUS_INVALID_VALUE for a null argument or
 *         an out of range enum.
 * @note Writes *chosen and launches nothing.
 */
CKL_EXPORT ckl_status_t ckl_gemm_query(ckl_handle_t h, ckl_layout_t layout, ckl_operation_t opa,
                                       ckl_operation_t opb, int64_t m, int64_t n, int64_t k,
                                       ckl_datatype_t dta, int64_t lda, ckl_datatype_t dtb,
                                       int64_t ldb, ckl_datatype_t dtc, int64_t ldc,
                                       ckl_algo_t* chosen);

/**
 * @brief Single precision GEMM: C = alpha * op(A) * op(B) + beta * C.
 * @param h Handle supplying the stream and the vendor handles.
 * @param layout Storage order of all three matrices.
 * @param opa Transpose flag for A.
 * @param opb Transpose flag for B.
 * @param m Rows of op(A) and of C.
 * @param n Columns of op(B) and of C.
 * @param k Contraction extent.
 * @param alpha Host scale on the product.
 * @param a Device pointer to A.
 * @param lda Leading dimension of A.
 * @param b Device pointer to B.
 * @param ldb Leading dimension of B.
 * @param beta Host scale on the incoming C; when zero, C is not read.
 * @param c Device pointer to C, written in place.
 * @param ldc Leading dimension of C.
 * @return CKL_STATUS_SUCCESS, or the failure status; ckl_last_error carries the detail.
 * @note Asynchronous on the handle's stream. Always dispatches through
 *       CKL_ALGO_AUTO; use ckl_gemm_ex to name a path or to learn which one ran.
 */
CKL_EXPORT ckl_status_t ckl_sgemm(ckl_handle_t h, ckl_layout_t layout, ckl_operation_t opa,
                                  ckl_operation_t opb, int64_t m, int64_t n, int64_t k, float alpha,
                                  const float* a, int64_t lda, const float* b, int64_t ldb,
                                  float beta, float* c, int64_t ldc);

/**
 * @brief Mixed precision GEMM with an explicit algorithm and a chosen out-param.
 * @param h Handle supplying the stream and the vendor handles.
 * @param layout Storage order of all three matrices.
 * @param opa Transpose flag for A.
 * @param opb Transpose flag for B.
 * @param m Rows of op(A) and of C.
 * @param n Columns of op(B) and of C.
 * @param k Contraction extent.
 * @param alpha Host pointer to a float scale on the product.
 * @param a Device pointer to A.
 * @param dta Element type of A.
 * @param lda Leading dimension of A.
 * @param b Device pointer to B.
 * @param dtb Element type of B; must equal dta.
 * @param ldb Leading dimension of B.
 * @param beta Host pointer to a float scale on the incoming C.
 * @param c Device pointer to C, written in place.
 * @param dtc Element type of C; CKL_R_32F is the only one in 1.1.0.
 * @param ldc Leading dimension of C.
 * @param algo Requested path, or CKL_ALGO_AUTO.
 * @param chosen Optional out-param receiving the path taken; may be NULL. When
 *        non-NULL it is written on success and on failure alike.
 * @return CKL_STATUS_SUCCESS, or the failure status; ckl_last_error carries the detail.
 * @note An explicitly named algo is never rerouted. Only CKL_ALGO_AUTO chooses
 *       again, and chosen reports where it landed.
 * @note Asynchronous on the handle's stream.
 */
CKL_EXPORT ckl_status_t ckl_gemm_ex(ckl_handle_t h, ckl_layout_t layout, ckl_operation_t opa,
                                    ckl_operation_t opb, int64_t m, int64_t n, int64_t k,
                                    const void* alpha, const void* a, ckl_datatype_t dta,
                                    int64_t lda, const void* b, ckl_datatype_t dtb, int64_t ldb,
                                    const void* beta, void* c, ckl_datatype_t dtc, int64_t ldc,
                                    ckl_algo_t algo, ckl_algo_t* chosen);

/**
 * @brief Strided batched GEMM: batch i of A starts at a + i * stride_a, and so on.
 * @param h Handle supplying the stream and the vendor handles.
 * @param layout Storage order of all three matrices.
 * @param opa Transpose flag for A.
 * @param opb Transpose flag for B.
 * @param m Rows of op(A) and of C.
 * @param n Columns of op(B) and of C.
 * @param k Contraction extent.
 * @param alpha Host pointer to a float scale on the product.
 * @param a Device pointer to the first A.
 * @param dta Element type of A.
 * @param lda Leading dimension of A.
 * @param stride_a Elements between consecutive batches of A.
 * @param b Device pointer to the first B.
 * @param dtb Element type of B; must equal dta.
 * @param ldb Leading dimension of B.
 * @param stride_b Elements between consecutive batches of B.
 * @param beta Host pointer to a float scale on the incoming C.
 * @param c Device pointer to the first C, written in place.
 * @param dtc Element type of C; CKL_R_32F is the only one in 1.1.0.
 * @param ldc Leading dimension of C.
 * @param stride_c Elements between consecutive batches of C.
 * @param batch_count Number of matrices in the batch; at least 1.
 * @param algo Requested path, or CKL_ALGO_AUTO.
 * @return CKL_STATUS_SUCCESS, or the failure status; ckl_last_error carries the detail.
 * @note Batched work runs on cuBLAS in 1.1.0. CKL_ALGO_AUTO reroutes to it;
 *       naming a hand written rung returns CKL_STATUS_NOT_SUPPORTED rather than
 *       looping the kernel behind the caller's back.
 */
CKL_EXPORT ckl_status_t ckl_gemm_strided_batched_ex(
    ckl_handle_t h, ckl_layout_t layout, ckl_operation_t opa, ckl_operation_t opb, int64_t m,
    int64_t n, int64_t k, const void* alpha, const void* a, ckl_datatype_t dta, int64_t lda,
    int64_t stride_a, const void* b, ckl_datatype_t dtb, int64_t ldb, int64_t stride_b,
    const void* beta, void* c, ckl_datatype_t dtc, int64_t ldc, int64_t stride_c,
    int32_t batch_count, ckl_algo_t algo);

/**
 * @brief Single precision CSR SpMV: y = alpha * (A * x) + beta * y.
 * @param h Handle supplying the stream; the plan keeps its own cuSPARSE handle.
 * @param m Rows of A and length of y.
 * @param n Columns of A and length of x.
 * @param nnz Number of stored nonzeros.
 * @param alpha Host scale on the product.
 * @param row_ptr Device pointer to the CSR row offsets, length m+1.
 * @param col_idx Device pointer to the CSR column indices, length nnz.
 * @param values Device pointer to the CSR values, length nnz.
 * @param x Device pointer to x, length n.
 * @param beta Host scale on the incoming y; when zero, y is not read.
 * @param y Device pointer to y, length m, written in place.
 * @param algo Requested path, or CKL_SPMV_AUTO.
 * @param chosen Optional out-param receiving the path taken; may be NULL. When
 *        non-NULL it is written on success and on failure alike.
 * @return CKL_STATUS_SUCCESS, or the failure status; ckl_last_error carries the detail.
 * @note An explicitly named algo is never rerouted. Only CKL_SPMV_AUTO chooses,
 *       and chosen reports where it landed.
 * @note This entry point keeps a one entry plan cache keyed on the three device
 *       buffers and the shape, because building a plan per call is exactly the
 *       defect the plan exists to remove. A repeated call on the same matrix
 *       reuses the descriptors, both workspaces and both converted layouts. A
 *       benchmark should build a ckl::SpmvPlan of its own instead: the cache
 *       holds one matrix, so alternating between two rebuilds on every call.
 * @note Asynchronous on the handle's stream.
 */
CKL_EXPORT ckl_status_t ckl_spmv_csr(ckl_handle_t h, int64_t m, int64_t n, int64_t nnz, float alpha,
                                     const int* row_ptr, const int* col_idx, const float* values,
                                     const float* x, float beta, float* y, ckl_spmv_algo_t algo,
                                     ckl_spmv_algo_t* chosen);

/**
 * @brief Single precision complex to complex FFT of a batch of transforms.
 * @param h Handle supplying the stream; the plan keeps its own tables and cuFFT handle.
 * @param n Transform length; a power of two from 2 to 16777216.
 * @param batch Transforms laid out back to back; at least 1.
 * @param in Device pointer to n * batch interleaved complex FP32 inputs.
 * @param out Device pointer to n * batch interleaved complex FP32 outputs; must
 *        differ from in, because every path here is out of place.
 * @param dir Forward or inverse; neither is scaled.
 * @param algo Requested path, or CKL_FFT_AUTO.
 * @param chosen Optional out-param receiving the path taken; may be NULL. When
 *        non-NULL it is written on success and on failure alike.
 * @return CKL_STATUS_SUCCESS, or the failure status; ckl_last_error carries the detail.
 * @note An explicitly named algo is never rerouted. Only CKL_FFT_AUTO chooses,
 *       and chosen reports where it landed.
 * @note This entry point keeps a one entry plan cache keyed on the length and the
 *       batch, because building a plan per call would put the twiddle tables and
 *       a cuFFT handle inside the caller's timed region. A benchmark should build
 *       a ckl::FftPlan of its own: the cache holds one plan, so alternating
 *       between two sizes rebuilds on every call.
 * @note Asynchronous on the handle's stream.
 */
CKL_EXPORT ckl_status_t ckl_fft_c2c(ckl_handle_t h, int64_t n, int64_t batch, const void* in,
                                    void* out, ckl_fft_direction_t dir, ckl_fft_algo_t algo,
                                    ckl_fft_algo_t* chosen);

/**
 * @brief Linear convolution of a real signal with a real filter.
 * @param h Handle supplying the stream.
 * @param signal_length N, the signal length; at least 1.
 * @param signal Device pointer to N reals.
 * @param filter_length M, the filter length; at least 1.
 * @param filter Device pointer to M reals.
 * @param out Device pointer to N + M - 1 reals.
 * @param algo Requested path, or CKL_CONV_AUTO.
 * @param chosen Optional out-param receiving the path taken; may be NULL. When
 *        non-NULL it is written on success and on failure alike.
 * @return CKL_STATUS_SUCCESS, or the failure status; ckl_last_error carries the detail.
 * @note An explicitly named algo is never rerouted. Only CKL_CONV_AUTO chooses,
 *       and chosen reports where it landed.
 * @note This entry point keeps a one entry plan cache keyed on the signal length,
 *       the filter pointer and the filter length. Building the plan transforms
 *       the filter, and doing that on every call is exactly what the cached
 *       filter spectrum exists to prevent.
 * @note Asynchronous on the handle's stream.
 */
CKL_EXPORT ckl_status_t ckl_conv_r2r(ckl_handle_t h, int64_t signal_length, const float* signal,
                                     int64_t filter_length, const float* filter, float* out,
                                     ckl_conv_algo_t algo, ckl_conv_algo_t* chosen);

/**
 * @brief Device wide reduction of n elements to one.
 * @param h Handle supplying the stream; the plan keeps its own workspace.
 * @param n Number of elements; zero writes the operator's identity and succeeds.
 * @param dtype Element type of in and out.
 * @param op Binary operator to combine with.
 * @param in Device pointer to n elements.
 * @param out Device pointer to one element.
 * @param algo Requested rung, or CKL_REDUCE_AUTO.
 * @param chosen Optional out-param receiving the rung taken; may be NULL. When
 *        non-NULL it is written on success and on failure alike.
 * @return CKL_STATUS_SUCCESS, or the failure status; ckl_last_error carries the detail.
 * @note An explicitly named algo is never rerouted. Only CKL_REDUCE_AUTO
 *       chooses, and chosen reports where it landed. Setting
 *       CKL_REDUCE_DETERMINISTIC in the environment makes it choose the
 *       deterministic rung, which chosen then reports.
 * @note This entry point keeps a one entry plan cache keyed on the length and
 *       the element type, because building a plan per call would put the
 *       workspace allocation and the CUB storage query inside whatever the
 *       caller was timing. A benchmark should build a ckl::ScanPlan of its own:
 *       the cache holds one plan, so alternating between two lengths rebuilds on
 *       every call.
 * @note Asynchronous on the handle's stream.
 */
CKL_EXPORT ckl_status_t ckl_reduce(ckl_handle_t h, int64_t n, ckl_scan_dtype_t dtype,
                                   ckl_scan_op_t op, const void* in, void* out,
                                   ckl_reduce_algo_t algo, ckl_reduce_algo_t* chosen);

/**
 * @brief Device wide prefix scan of n elements.
 * @param h Handle supplying the stream; the plan keeps its own workspace.
 * @param n Number of elements; zero succeeds and writes nothing.
 * @param dtype Element type of in and out.
 * @param op Binary operator to combine with.
 * @param exclusive Non zero for an exclusive scan, zero for an inclusive one.
 * @param in Device pointer to n elements.
 * @param out Device pointer to n elements; it may equal in, and every rung runs
 *        in place when it does.
 * @param algo Requested rung, or CKL_SCAN_AUTO.
 * @param chosen Optional out-param receiving the rung taken; may be NULL. When
 *        non-NULL it is written on success and on failure alike.
 * @return CKL_STATUS_SUCCESS, or the failure status; ckl_last_error carries the detail.
 * @note An explicitly named algo is never rerouted. Only CKL_SCAN_AUTO chooses,
 *       and chosen reports where it landed.
 * @note The same one entry plan cache as ckl_reduce, and for the same reason.
 * @note Asynchronous on the handle's stream.
 */
CKL_EXPORT ckl_status_t ckl_scan(ckl_handle_t h, int64_t n, ckl_scan_dtype_t dtype,
                                 ckl_scan_op_t op, int exclusive, const void* in, void* out,
                                 ckl_scan_algo_t algo, ckl_scan_algo_t* chosen);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CKL_CKL_H */
