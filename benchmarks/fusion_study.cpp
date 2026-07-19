// Epilogue fusion study. A GEMM followed by a bias add and ReLU is a common
// pattern. The question the roofline asks: is it worth folding the bias and ReLU
// into the GEMM epilogue, or is a separate elementwise pass fine?
//
// This times two ways to compute C = relu(alpha * A*B + bias):
//   unfused: the top GEMM writes C, then a separate kernel reads C, adds the
//            column bias, applies ReLU, and writes C again.
//   fused:   the top GEMM adds the bias and applies ReLU while C is still in
//            registers, so C is written exactly once.
//
// It also reports the arithmetic intensity of the standalone epilogue, which is
// what makes the decision: the epilogue is memory bound (about 2 FLOPs per C
// element over a read plus a write of C), so when it runs as a separate pass it is
// pure bandwidth overhead, and folding it into a compute bound GEMM removes that
// pass at almost no cost. The measured delta quantifies the saving.

#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_fp16.h>

#include "ckl/cuda_check.hpp"
#include "ckl/device_buffer.hpp"
#include "ckl/event_timer.hpp"
#include "ckl/gemm.hpp"
#include "ckl/reference.hpp"

int main(int argc, char** argv) {
    const int n = argc > 1 ? std::atoi(argv[1]) : 4096;

    const auto fa = ckl::random_matrix(n, n, 1);
    const auto fb = ckl::random_matrix(n, n, 2);
    const auto fbias = ckl::random_matrix(n, 1, 3);
    std::vector<__half> a(fa.size());
    std::vector<__half> b(fb.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        a[i] = __float2half(fa[i]);
        b[i] = __float2half(fb[i]);
    }

    ckl::DeviceBuffer<__half> da(a.size());
    ckl::DeviceBuffer<__half> db(b.size());
    ckl::DeviceBuffer<float> dc(static_cast<std::size_t>(n) * n);
    ckl::DeviceBuffer<float> dbias(fbias.size());
    da.copy_from_host(a);
    db.copy_from_host(b);
    dbias.copy_from_host(fbias);
    dc.zero();

    ckl::TimingStats unfused = ckl::time_stream([&](cudaStream_t s) {
        ckl::gemm_mma_opt(da.data(), db.data(), dc.data(), n, n, n, 1.0f, 0.0f, s);
        ckl::gemm_bias_relu(dc.data(), dbias.data(), n, n, s);
    });
    ckl::TimingStats fused = ckl::time_stream([&](cudaStream_t s) {
        ckl::gemm_mma_opt_bias(da.data(), db.data(), dc.data(), dbias.data(), n, n, n, 1.0f, s);
    });
    ckl::TimingStats gemm_only = ckl::time_stream([&](cudaStream_t s) {
        ckl::gemm_mma_opt(da.data(), db.data(), dc.data(), n, n, n, 1.0f, 0.0f, s);
    });

    const double epilogue_flops = 2.0 * n * n;  // add + relu compare
    const double epilogue_bytes =
        3.0 * static_cast<double>(n) * n * sizeof(float);  // read C, write C, read bias approx
    const double epilogue_intensity = epilogue_flops / epilogue_bytes;
    const double saved_ms = unfused.median_ms - fused.median_ms;
    const double sep_pass_ms = unfused.median_ms - gemm_only.median_ms;

    std::printf("Epilogue fusion study (C = relu(alpha*A*B + bias)), n = %d\n", n);
    std::printf("  GEMM only            : %.4f ms\n", gemm_only.median_ms);
    std::printf("  unfused (GEMM + pass): %.4f ms  (separate epilogue pass %.4f ms)\n",
                unfused.median_ms, sep_pass_ms);
    std::printf("  fused epilogue       : %.4f ms\n", fused.median_ms);
    std::printf("  saved by fusion      : %.4f ms  (%.1f percent of unfused)\n", saved_ms,
                unfused.median_ms > 0.0 ? 100.0 * saved_ms / unfused.median_ms : 0.0);
    std::printf("  epilogue intensity   : %.3f FLOP/byte (memory bound; ridge is about 113)\n",
                epilogue_intensity);
    std::printf("Decision: the standalone epilogue is memory bound, so as a separate\n");
    std::printf("pass it is pure bandwidth cost; folding it into the compute bound GEMM\n");
    std::printf("removes that pass. Fusion is the right call here.\n");
    return 0;
}
