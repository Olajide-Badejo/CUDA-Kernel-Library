// Single kernel driver for Nsight Compute rounds. It runs exactly one GEMM
// variant on one square shape a fixed number of times so ncu can profile one
// settled launch with `--launch-skip N -c 1`. Keeping the driver this small
// means the ncu report contains just the kernel under study, not the harness.
//
// Usage: ncu_driver <variant> <size> [launches]
//   variant in {naive, tiled}
//   size    square dimension
//   launches how many times to run (default 5; profile the last with skip 4)

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "ckl/cuda_check.hpp"
#include "ckl/device_buffer.hpp"
#include "ckl/gemm.hpp"
#include "ckl/reference.hpp"

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <variant> <size> [launches]\n", argv[0]);
        return 2;
    }
    const std::string variant = argv[1];
    const int n = std::atoi(argv[2]);
    const int launches = argc > 3 ? std::atoi(argv[3]) : 5;

    void (*launch)(const float*, const float*, float*, int, int, int, float, float,
                   cudaStream_t) = nullptr;
    if (variant == "naive") {
        launch = ckl::gemm_naive;
    } else if (variant == "tiled") {
        launch = ckl::gemm_tiled;
    } else {
        std::fprintf(stderr, "unknown variant: %s\n", variant.c_str());
        return 2;
    }

    const auto ha = ckl::random_matrix(n, n, 101);
    const auto hb = ckl::random_matrix(n, n, 202);
    ckl::DeviceBuffer<float> da(ha.size());
    ckl::DeviceBuffer<float> db(hb.size());
    ckl::DeviceBuffer<float> dc(static_cast<std::size_t>(n) * n);
    da.copy_from_host(ha);
    db.copy_from_host(hb);
    dc.zero();

    for (int i = 0; i < launches; ++i) {
        launch(da.data(), db.data(), dc.data(), n, n, n, 1.0f, 0.0f, nullptr);
    }
    CKL_CUDA_CHECK(cudaDeviceSynchronize());
    std::printf("ran %s at %d cubed, %d launches\n", variant.c_str(), n, launches);
    return 0;
}
