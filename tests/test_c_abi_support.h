#ifndef CKL_TEST_C_ABI_SUPPORT_H
#define CKL_TEST_C_ABI_SUPPORT_H

/* Device memory helpers for the C ABI test. The test itself is compiled as C99
 * and must not include a CUDA header, because the whole point of ckl.h is that a
 * C consumer needs none; the CUDA calls live in a second translation unit that
 * is compiled as C++ and exposes this plain C interface.
 *
 * Every function returns 0 on success and non zero on failure. */

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Non zero when the machine has at least one usable CUDA device. */
int ckl_test_has_device(void);

int ckl_test_alloc(void** out, size_t bytes);
int ckl_test_free(void* p);
int ckl_test_upload(void* device_dst, const void* host_src, size_t bytes);
int ckl_test_download(void* host_dst, const void* device_src, size_t bytes);
int ckl_test_sync(void);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CKL_TEST_C_ABI_SUPPORT_H */
