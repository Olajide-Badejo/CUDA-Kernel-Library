#ifndef CKL_CKL_H
#define CKL_CKL_H

/* The C ABI of the CUDA Kernel Library. Pure C99: it includes <stdint.h>,
 * <stddef.h> and the generated ckl_export.h, and no CUDA header at all. Device
 * pointers and streams cross as void*, so a C or Fortran consumer needs no
 * toolkit headers to call the library.
 *
 * Nothing but C crosses this boundary: no exceptions, no std::string, no
 * templates, no default arguments. Every entry point catches everything and
 * turns it into a ckl_status_t; the detail of the last failure on the calling
 * thread is available through ckl_last_error.
 *
 * alpha and beta are HOST values. A device pointer mode is a documented non
 * goal until CUDA graph capture support lands. Where a routine takes them as
 * const void*, they point at a float, because the compute type is always
 * 32 bit float. */

#include <stddef.h>
#include <stdint.h>

#include "ckl/ckl_export.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    CKL_STATUS_SUCCESS = 0,
    CKL_STATUS_NOT_INITIALIZED,
    CKL_STATUS_INVALID_VALUE,
    CKL_STATUS_ARCH_MISMATCH,
    CKL_STATUS_NOT_SUPPORTED,
    CKL_STATUS_ALLOC_FAILED,
    CKL_STATUS_EXECUTION_FAILED,
    CKL_STATUS_INTERNAL
} ckl_status_t;

typedef struct ckl_context* ckl_handle_t;

typedef enum { CKL_OP_N = 0, CKL_OP_T = 1, CKL_OP_C = 2 } ckl_operation_t;

typedef enum { CKL_ROW_MAJOR = 0, CKL_COL_MAJOR = 1 } ckl_layout_t;

typedef enum { CKL_R_32F = 0, CKL_R_16F = 1, CKL_R_16BF = 2 } ckl_datatype_t;

typedef enum {
    CKL_ALGO_AUTO = 0,
    CKL_ALGO_NAIVE,
    CKL_ALGO_TILED,
    CKL_ALGO_REGISTER,
    CKL_ALGO_CP_ASYNC,
    CKL_ALGO_WMMA_FP16,
    CKL_ALGO_WMMA_BF16,
    CKL_ALGO_MMA_PTX,
    CKL_ALGO_MMA_LDM,
    CKL_ALGO_MMA_OPT,
    CKL_ALGO_TILE_FAMILY,
    CKL_ALGO_SPLITK,
    CKL_ALGO_STREAMK,
    CKL_ALGO_CUTLASS,
    CKL_ALGO_CUBLAS
} ckl_algo_t;

/* version: 10000 * major + 100 * minor + patch */
CKL_EXPORT int ckl_get_version(void);
CKL_EXPORT const char* ckl_get_version_string(void);
CKL_EXPORT const char* ckl_status_string(ckl_status_t s);

/* Copies the thread local detail of the last failure into buf, always NUL
 * terminated, and valid until the next CKL call on the same thread. Returns
 * CKL_STATUS_INVALID_VALUE when buf is NULL or len is zero. */
CKL_EXPORT ckl_status_t ckl_last_error(char* buf, size_t len);

CKL_EXPORT ckl_status_t ckl_create(ckl_handle_t* h);
CKL_EXPORT ckl_status_t ckl_destroy(ckl_handle_t h);
CKL_EXPORT ckl_status_t ckl_set_stream(ckl_handle_t h, void* cuda_stream);
CKL_EXPORT ckl_status_t ckl_set_workspace(ckl_handle_t h, void* ws, size_t bytes);

/* Bytes of scratch the described GEMM needs. Every path shipped in 1.1.0
 * answers zero; the entry point exists so a later split-K or stream-K rung can
 * ask for scratch without an ABI break. */
CKL_EXPORT ckl_status_t ckl_gemm_workspace_size(ckl_handle_t h, ckl_layout_t layout,
                                                ckl_operation_t opa, ckl_operation_t opb, int64_t m,
                                                int64_t n, int64_t k, ckl_datatype_t dta,
                                                int64_t lda, ckl_datatype_t dtb, int64_t ldb,
                                                ckl_datatype_t dtc, int64_t ldc, ckl_algo_t algo,
                                                size_t* bytes);

/* What CKL_ALGO_AUTO would pick for the described GEMM. Writes *chosen and
 * launches nothing. */
CKL_EXPORT ckl_status_t ckl_gemm_query(ckl_handle_t h, ckl_layout_t layout, ckl_operation_t opa,
                                       ckl_operation_t opb, int64_t m, int64_t n, int64_t k,
                                       ckl_datatype_t dta, int64_t lda, ckl_datatype_t dtb,
                                       int64_t ldb, ckl_datatype_t dtc, int64_t ldc,
                                       ckl_algo_t* chosen);

CKL_EXPORT ckl_status_t ckl_sgemm(ckl_handle_t h, ckl_layout_t layout, ckl_operation_t opa,
                                  ckl_operation_t opb, int64_t m, int64_t n, int64_t k, float alpha,
                                  const float* a, int64_t lda, const float* b, int64_t ldb,
                                  float beta, float* c, int64_t ldc);

CKL_EXPORT ckl_status_t ckl_gemm_ex(ckl_handle_t h, ckl_layout_t layout, ckl_operation_t opa,
                                    ckl_operation_t opb, int64_t m, int64_t n, int64_t k,
                                    const void* alpha, const void* a, ckl_datatype_t dta,
                                    int64_t lda, const void* b, ckl_datatype_t dtb, int64_t ldb,
                                    const void* beta, void* c, ckl_datatype_t dtc, int64_t ldc,
                                    ckl_algo_t algo, ckl_algo_t* chosen);

CKL_EXPORT ckl_status_t ckl_gemm_strided_batched_ex(
    ckl_handle_t h, ckl_layout_t layout, ckl_operation_t opa, ckl_operation_t opb, int64_t m,
    int64_t n, int64_t k, const void* alpha, const void* a, ckl_datatype_t dta, int64_t lda,
    int64_t stride_a, const void* b, ckl_datatype_t dtb, int64_t ldb, int64_t stride_b,
    const void* beta, void* c, ckl_datatype_t dtc, int64_t ldc, int64_t stride_c,
    int32_t batch_count, ckl_algo_t algo);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* CKL_CKL_H */
