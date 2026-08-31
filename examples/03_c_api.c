/* The C ABI end to end, in C99, with no CUDA header anywhere in this file.
 *
 * ckl.h includes <stddef.h>, <stdint.h> and the generated ckl_export.h and
 * nothing else, and the device memory helpers cover allocate, copy, free and
 * synchronize, so a C program that links libckl needs no toolkit headers of its
 * own. Compile it with a C compiler and the link line stays the same.
 *
 * The error path is shown deliberately: a deliberately bad leading dimension is
 * rejected, and ckl_last_error says what was wrong with it. */

#include <math.h>
#include <stdio.h>

#include "ckl/ckl.h"

enum { kM = 128, kN = 128, kK = 64 };

static void print_last_error(const char* where) {
    char detail[512];
    if (ckl_last_error(detail, sizeof detail) == CKL_STATUS_SUCCESS) {
        printf("  %s: %s\n", where, detail);
    }
}

int main(void) {
    static float host_a[kM * kK];
    static float host_b[kK * kN];
    static float host_c[kM * kN];

    ckl_handle_t h = NULL;
    void* da = NULL;
    void* db = NULL;
    void* dc = NULL;
    ckl_status_t s;
    ckl_algo_t chosen = CKL_ALGO_AUTO;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const float want = (float)kK;
    int i;
    int failed = 0;

    for (i = 0; i < kM * kK; ++i) {
        host_a[i] = 1.0f;
    }
    for (i = 0; i < kK * kN; ++i) {
        host_b[i] = 1.0f;
    }

    printf("ckl %s (version %d)\n", ckl_get_version_string(), ckl_get_version());

    s = ckl_create(&h);
    if (s != CKL_STATUS_SUCCESS) {
        printf("FAIL: ckl_create returned %s\n", ckl_status_string(s));
        print_last_error("ckl_create");
        return 1;
    }

    if (ckl_device_malloc(&da, sizeof host_a) != CKL_STATUS_SUCCESS ||
        ckl_device_malloc(&db, sizeof host_b) != CKL_STATUS_SUCCESS ||
        ckl_device_malloc(&dc, sizeof host_c) != CKL_STATUS_SUCCESS) {
        printf("FAIL: device allocation\n");
        print_last_error("ckl_device_malloc");
        failed = 1;
        goto cleanup;
    }
    ckl_memcpy(da, host_a, sizeof host_a, CKL_MEMCPY_H2D);
    ckl_memcpy(db, host_b, sizeof host_b, CKL_MEMCPY_H2D);

    /* An lda smaller than k cannot describe a packed row major A. The call is
     * rejected before anything launches, and the detail names the bound. */
    s = ckl_gemm_ex(h, CKL_ROW_MAJOR, CKL_OP_N, CKL_OP_N, kM, kN, kK, &alpha, da, CKL_R_32F, 1, db,
                    CKL_R_32F, kN, &beta, dc, CKL_R_32F, kN, CKL_ALGO_AUTO, &chosen);
    if (s != CKL_STATUS_INVALID_VALUE) {
        printf("FAIL: a bad lda returned %s, expected CKL_STATUS_INVALID_VALUE\n",
               ckl_status_string(s));
        failed = 1;
        goto cleanup;
    }
    printf("rejected a bad lda as expected\n");
    print_last_error("ckl_gemm_ex");

    s = ckl_gemm_ex(h, CKL_ROW_MAJOR, CKL_OP_N, CKL_OP_N, kM, kN, kK, &alpha, da, CKL_R_32F, kK, db,
                    CKL_R_32F, kN, &beta, dc, CKL_R_32F, kN, CKL_ALGO_AUTO, &chosen);
    if (s != CKL_STATUS_SUCCESS) {
        printf("FAIL: ckl_gemm_ex returned %s\n", ckl_status_string(s));
        print_last_error("ckl_gemm_ex");
        failed = 1;
        goto cleanup;
    }

    /* The GEMM is asynchronous on the handle's stream; ckl_memcpy is
     * synchronous on the default stream, which is where the handle starts. */
    ckl_device_synchronize();
    ckl_memcpy(host_c, dc, sizeof host_c, CKL_MEMCPY_D2H);

    for (i = 0; i < kM * kN; ++i) {
        if (fabsf(host_c[i] - want) > 1e-3f) {
            printf("FAIL: c[%d] = %f, expected %f\n", i, (double)host_c[i], (double)want);
            failed = 1;
            goto cleanup;
        }
    }
    printf("PASS: %dx%dx%d fp32 gemm through the C ABI, algo id %d, every element %.0f\n", kM, kN,
           kK, (int)chosen, (double)want);

cleanup:
    ckl_device_free(da);
    ckl_device_free(db);
    ckl_device_free(dc);
    ckl_destroy(h);
    return failed;
}
