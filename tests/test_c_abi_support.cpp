// The CUDA side of the C ABI test. Compiled as C++ so it can include
// cuda_runtime.h; everything it exports is plain C, so test_c_abi.c stays a C99
// translation unit that sees no toolkit header at all.

#include "test_c_abi_support.h"

#include <cuda_runtime.h>

extern "C" {

int ckl_test_has_device(void) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) {
        return 0;
    }
    return count > 0 ? 1 : 0;
}

int ckl_test_alloc(void** out, size_t bytes) {
    *out = nullptr;
    return cudaMalloc(out, bytes) == cudaSuccess ? 0 : 1;
}

int ckl_test_free(void* p) {
    return cudaFree(p) == cudaSuccess ? 0 : 1;
}

int ckl_test_upload(void* device_dst, const void* host_src, size_t bytes) {
    return cudaMemcpy(device_dst, host_src, bytes, cudaMemcpyHostToDevice) == cudaSuccess ? 0 : 1;
}

int ckl_test_download(void* host_dst, const void* device_src, size_t bytes) {
    return cudaMemcpy(host_dst, device_src, bytes, cudaMemcpyDeviceToHost) == cudaSuccess ? 0 : 1;
}

int ckl_test_sync(void) {
    return cudaDeviceSynchronize() == cudaSuccess ? 0 : 1;
}

}  // extern "C"
