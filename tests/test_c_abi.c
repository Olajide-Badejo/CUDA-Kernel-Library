/* C ABI test, compiled as C99. It includes ckl/ckl.h and nothing from the CUDA
 * toolkit, which is the property under test: a C consumer of this library needs
 * no toolkit headers. Device allocations come through the small shim in
 * test_c_abi_support.h, whose implementation is the only translation unit here
 * that sees cuda_runtime.h.
 *
 * Exit code 0 means every check passed. */

#include <math.h>
#include <stdio.h>
#include <string.h>

#include "ckl/ckl.h"
#include "test_c_abi_support.h"

#define N 8

static int failures = 0;

static void check(int condition, const char* what) {
    printf("  %-46s %s\n", what, condition ? "PASS" : "FAIL");
    if (!condition) {
        failures++;
    }
}

/* Row major square matrices of ones times each other: every output element is
 * N, small enough that a wrong answer is obvious. */
static void run_sgemm_case(ckl_handle_t h) {
    float host_a[N * N];
    float host_c[N * N];
    void* da = NULL;
    void* db = NULL;
    void* dc = NULL;
    ckl_status_t s;
    int i;
    int correct;

    for (i = 0; i < N * N; ++i) {
        host_a[i] = 1.0f;
        host_c[i] = 0.0f;
    }

    check(ckl_test_alloc(&da, sizeof(host_a)) == 0, "cudaMalloc a through the shim");
    check(ckl_test_alloc(&db, sizeof(host_a)) == 0, "cudaMalloc b through the shim");
    check(ckl_test_alloc(&dc, sizeof(host_c)) == 0, "cudaMalloc c through the shim");
    check(ckl_test_upload(da, host_a, sizeof(host_a)) == 0, "upload a");
    check(ckl_test_upload(db, host_a, sizeof(host_a)) == 0, "upload b");
    check(ckl_test_upload(dc, host_c, sizeof(host_c)) == 0, "upload c");

    s = ckl_sgemm(h, CKL_ROW_MAJOR, CKL_OP_N, CKL_OP_N, N, N, N, 1.0f, (const float*)da, N,
                  (const float*)db, N, 0.0f, (float*)dc, N);
    check(s == CKL_STATUS_SUCCESS, "ckl_sgemm returns CKL_STATUS_SUCCESS");
    check(ckl_test_sync() == 0, "device synchronize after ckl_sgemm");
    check(ckl_test_download(host_c, dc, sizeof(host_c)) == 0, "download c");

    correct = 1;
    for (i = 0; i < N * N; ++i) {
        if (fabs((double)host_c[i] - (double)N) > 1e-4) {
            correct = 0;
        }
    }
    check(correct, "ckl_sgemm result is A times B");

    /* Deliberately invalid: lda below the row length of a row major A. */
    s = ckl_sgemm(h, CKL_ROW_MAJOR, CKL_OP_N, CKL_OP_N, N, N, N, 1.0f, (const float*)da, 1,
                  (const float*)db, N, 0.0f, (float*)dc, N);
    check(s == CKL_STATUS_INVALID_VALUE, "bad lda gives CKL_STATUS_INVALID_VALUE");

    {
        char message[256];
        ckl_status_t got = ckl_last_error(message, sizeof(message));
        check(got == CKL_STATUS_SUCCESS, "ckl_last_error succeeds");
        check(strlen(message) > 0, "ckl_last_error message is not empty");
        printf("      last error: %s\n", message);
        check(ckl_last_error(NULL, sizeof(message)) == CKL_STATUS_INVALID_VALUE,
              "ckl_last_error rejects a null buffer");
    }

    /* An algorithm the shape cannot take is refused, never rerouted. */
    {
        ckl_algo_t chosen = CKL_ALGO_AUTO;
        float alpha = 1.0f;
        float beta = 0.0f;
        s = ckl_gemm_ex(h, CKL_ROW_MAJOR, CKL_OP_N, CKL_OP_N, N, N, N, &alpha, da, CKL_R_32F, N, db,
                        CKL_R_32F, N, &beta, dc, CKL_R_32F, N, CKL_ALGO_MMA_OPT, &chosen);
        check(s == CKL_STATUS_NOT_SUPPORTED,
              "explicit mma_opt on 8x8x8 is CKL_STATUS_NOT_SUPPORTED");
        check(chosen == CKL_ALGO_MMA_OPT, "chosen names the algorithm that was asked for");
    }

    /* kAuto writes chosen and reports the path it actually took. */
    {
        ckl_algo_t chosen = CKL_ALGO_MMA_OPT;
        float alpha = 1.0f;
        float beta = 0.0f;
        s = ckl_gemm_ex(h, CKL_ROW_MAJOR, CKL_OP_N, CKL_OP_N, N, N, N, &alpha, da, CKL_R_32F, N, db,
                        CKL_R_32F, N, &beta, dc, CKL_R_32F, N, CKL_ALGO_AUTO, &chosen);
        check(s == CKL_STATUS_SUCCESS, "kAuto FP32 8x8x8 succeeds");
        check(chosen == CKL_ALGO_NAIVE, "kAuto reports the naive rung on a tiny FP32 shape");
    }

    {
        size_t bytes = 12345;
        s = ckl_gemm_workspace_size(h, CKL_ROW_MAJOR, CKL_OP_N, CKL_OP_N, N, N, N, CKL_R_32F, N,
                                    CKL_R_32F, N, CKL_R_32F, N, CKL_ALGO_AUTO, &bytes);
        check(s == CKL_STATUS_SUCCESS && bytes == 0, "ckl_gemm_workspace_size answers zero");
    }

    {
        ckl_algo_t chosen = CKL_ALGO_AUTO;
        s = ckl_gemm_query(h, CKL_ROW_MAJOR, CKL_OP_N, CKL_OP_N, N, N, N, CKL_R_32F, N, CKL_R_32F,
                           N, CKL_R_32F, N, &chosen);
        check(s == CKL_STATUS_SUCCESS && chosen == CKL_ALGO_NAIVE,
              "ckl_gemm_query agrees with what kAuto ran");
    }

    check(ckl_test_free(da) == 0, "free a");
    check(ckl_test_free(db) == 0, "free b");
    check(ckl_test_free(dc) == 0, "free c");
}

int main(void) {
    ckl_handle_t h = NULL;
    ckl_status_t s;

    printf("C ABI (compiled as C99, no CUDA headers in this translation unit)\n");

    check(ckl_get_version() == 10100, "ckl_get_version is 10100");
    check(strcmp(ckl_get_version_string(), "1.1.0") == 0, "ckl_get_version_string is 1.1.0");
    check(strcmp(ckl_status_string(CKL_STATUS_ARCH_MISMATCH), "CKL_STATUS_ARCH_MISMATCH") == 0,
          "ckl_status_string names its code");
    check(ckl_create(NULL) == CKL_STATUS_INVALID_VALUE, "ckl_create rejects a null out-param");
    check(ckl_destroy(NULL) == CKL_STATUS_INVALID_VALUE, "ckl_destroy rejects a null handle");

    if (!ckl_test_has_device()) {
        printf("  no CUDA device visible; ckl_create must report, not abort\n");
        s = ckl_create(&h);
        check(s == CKL_STATUS_NOT_INITIALIZED, "ckl_create without a device reports a status");
        printf("%s\n", failures == 0 ? "C ABI checks passed" : "C ABI checks FAILED");
        return failures == 0 ? 0 : 1;
    }

    s = ckl_create(&h);
    check(s == CKL_STATUS_SUCCESS && h != NULL, "ckl_create");
    if (s != CKL_STATUS_SUCCESS) {
        printf("C ABI checks FAILED\n");
        return 1;
    }

    check(ckl_set_stream(h, NULL) == CKL_STATUS_SUCCESS, "ckl_set_stream on the default stream");
    check(ckl_set_workspace(h, NULL, 0) == CKL_STATUS_SUCCESS, "ckl_set_workspace with no buffer");

    run_sgemm_case(h);

    check(ckl_destroy(h) == CKL_STATUS_SUCCESS, "ckl_destroy");

    printf("%s\n", failures == 0 ? "C ABI checks passed" : "C ABI checks FAILED");
    return failures == 0 ? 0 : 1;
}
