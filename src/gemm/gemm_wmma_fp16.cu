// WMMA FP16 GEMM: FP16 storage, FP32 accumulate. The kernel body is the shared
// templated implementation; this file instantiates it for __half.

#include "ckl/detail/wmma_gemm.cuh"

namespace ckl {

void gemm_wmma_fp16(const __half* a, const __half* b, float* c, int m, int n, int k, float alpha,
                    float beta, cudaStream_t stream) {
    detail::launch_wmma<__half>(a, b, c, m, n, k, alpha, beta, stream);
}

}  // namespace ckl
