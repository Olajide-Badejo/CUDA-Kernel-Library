// WMMA BF16 GEMM: BF16 storage, FP32 accumulate. Same templated body as the FP16
// rung, instantiated for __nv_bfloat16. BF16 keeps the FP32 exponent range with a
// shorter mantissa, so its error distribution differs from FP16; the tolerance is
// derived per type in docs/gemm.md.

#include "ckl/detail/wmma_gemm.cuh"

namespace ckl {

void gemm_wmma_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b, float* c, int m, int n, int k,
                    float alpha, float beta, cudaStream_t stream) {
    detail::launch_wmma<__nv_bfloat16>(a, b, c, m, n, k, alpha, beta, stream);
}

}  // namespace ckl
