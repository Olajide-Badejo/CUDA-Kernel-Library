// Smoke test for the installed package. It calls one entry point from two of the
// six libraries and checks the arithmetic, so a link that resolves but produces
// nothing still fails. Built only against an install prefix, never inside the
// project tree.

#include <cmath>
#include <cstdio>
#include <vector>

#include <cuda_runtime.h>

#include "ckl/device_buffer.hpp"
#include "ckl/gemm.hpp"
#include "ckl/gemv.hpp"
#include "ckl/version.hpp"

namespace {

constexpr int kN = 8;
constexpr float kTolerance = 1e-4F;

// Row major square matrices of ones times a column of ones: every output element
// is kN, and the GEMV result is kN as well. Small enough that a wrong answer is
// obvious and no reference kernel is needed.
int run() {
    std::vector<float> a(kN * kN, 1.0F);
    std::vector<float> b(kN * kN, 1.0F);
    std::vector<float> x(kN, 1.0F);

    ckl::DeviceBuffer<float> da(a.size());
    ckl::DeviceBuffer<float> db(b.size());
    ckl::DeviceBuffer<float> dc(a.size());
    ckl::DeviceBuffer<float> dx(x.size());
    ckl::DeviceBuffer<float> dy(kN);

    da.copy_from_host(a);
    db.copy_from_host(b);
    dx.copy_from_host(x);
    dc.zero();
    dy.zero();

    ckl::gemm_naive(da.data(), db.data(), dc.data(), kN, kN, kN, 1.0F, 0.0F);
    ckl::gemv_naive(da.data(), dx.data(), dy.data(), kN, kN, 1.0F, 0.0F);

    cudaError_t sync = cudaDeviceSynchronize();
    if (sync != cudaSuccess) {
        std::printf("consume: cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(sync));
        return 1;
    }

    const std::vector<float> c = dc.to_host();
    const std::vector<float> y = dy.to_host();

    for (float v : c) {
        if (std::fabs(v - static_cast<float>(kN)) > kTolerance) {
            std::printf("consume: gemm_naive gave %f, expected %d\n", static_cast<double>(v), kN);
            return 1;
        }
    }
    for (float v : y) {
        if (std::fabs(v - static_cast<float>(kN)) > kTolerance) {
            std::printf("consume: gemv_naive gave %f, expected %d\n", static_cast<double>(v), kN);
            return 1;
        }
    }

    std::printf("consume: ckl %s (version macro %d), gemm_naive and gemv_naive both correct\n",
                CKL_VERSION_STRING, CKL_VERSION);
    return 0;
}

}  // namespace

int main() { return run(); }
