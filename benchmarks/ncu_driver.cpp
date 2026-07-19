// Single kernel driver for Nsight Compute rounds. It runs exactly one GEMM
// variant on one square shape a fixed number of times so ncu can profile one
// settled launch with `--launch-skip N -c 1`. Keeping the driver this small
// means the ncu report contains just the kernel under study, not the harness.
//
// Usage: ncu_driver <variant> <size> [launches]
//   variant in {naive, tiled, register, cp_async, wmma_fp16, wmma_bf16, mma_ptx}
//   size    square dimension
//   launches how many times to run (default 5; profile the last with skip 4)

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "ckl/cuda_check.hpp"
#include "ckl/device_buffer.hpp"
#include "ckl/gemm.hpp"
#include "ckl/reference.hpp"

namespace {

template <typename T>
T from_float(float f);
template <>
float from_float<float>(float f) { return f; }
template <>
__half from_float<__half>(float f) { return __float2half(f); }
template <>
__nv_bfloat16 from_float<__nv_bfloat16>(float f) { return __float2bfloat16(f); }

template <typename T, typename LaunchFn>
void run(LaunchFn launch, int n, int launches) {
    std::vector<T> ha(static_cast<std::size_t>(n) * n);
    std::vector<T> hb(static_cast<std::size_t>(n) * n);
    const auto fa = ckl::random_matrix(n, n, 101);
    const auto fb = ckl::random_matrix(n, n, 202);
    for (std::size_t i = 0; i < ha.size(); ++i) {
        ha[i] = from_float<T>(fa[i]);
        hb[i] = from_float<T>(fb[i]);
    }
    ckl::DeviceBuffer<T> da(ha.size());
    ckl::DeviceBuffer<T> db(hb.size());
    ckl::DeviceBuffer<float> dc(static_cast<std::size_t>(n) * n);
    da.copy_from_host(ha);
    db.copy_from_host(hb);
    dc.zero();
    for (int i = 0; i < launches; ++i) {
        launch(da.data(), db.data(), dc.data(), n, n, n, 1.0f, 0.0f, nullptr);
    }
    CKL_CUDA_CHECK(cudaDeviceSynchronize());
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <variant> <size> [launches]\n", argv[0]);
        return 2;
    }
    const std::string variant = argv[1];
    const int n = std::atoi(argv[2]);
    const int launches = argc > 3 ? std::atoi(argv[3]) : 5;

    if (variant == "naive") {
        run<float>(ckl::gemm_naive, n, launches);
    } else if (variant == "tiled") {
        run<float>(ckl::gemm_tiled, n, launches);
    } else if (variant == "register") {
        run<float>(ckl::gemm_register, n, launches);
    } else if (variant == "cp_async") {
        run<float>(ckl::gemm_cp_async, n, launches);
    } else if (variant == "wmma_fp16") {
        run<__half>(ckl::gemm_wmma_fp16, n, launches);
    } else if (variant == "wmma_bf16") {
        run<__nv_bfloat16>(ckl::gemm_wmma_bf16, n, launches);
    } else if (variant == "mma_ptx") {
        run<__half>(ckl::gemm_mma_ptx, n, launches);
    } else {
        std::fprintf(stderr, "unknown variant: %s\n", variant.c_str());
        return 2;
    }
    std::printf("ran %s at %d cubed, %d launches\n", variant.c_str(), n, launches);
    return 0;
}
