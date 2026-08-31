// Strided batched GEMM: one call covers a stack of matrices laid out back to
// back in one allocation. Batch b of A starts at a + b * stride_a, and likewise
// for B and C.
//
// Batch b of A is filled with the value (b + 1) and B is all ones, so element
// of C in batch b is k * (b + 1). That makes a batch handled in the wrong order
// or a stride off by one visible in the check rather than merely plausible.
//
// Batched work runs on cuBLAS in 1.1.0; the hand written rungs have no batched
// form and the dispatcher says so instead of looping them behind your back.

#include <cmath>
#include <cstdio>
#include <vector>

#include "ckl/ckl.hpp"

namespace {

constexpr int kM = 64;
constexpr int kN = 64;
constexpr int kK = 32;
constexpr int kBatch = 8;

}  // namespace

int main() {
    try {
        ckl::Context ctx;

        const std::size_t elems_a = static_cast<std::size_t>(kM) * kK;
        const std::size_t elems_b = static_cast<std::size_t>(kK) * kN;
        const std::size_t elems_c = static_cast<std::size_t>(kM) * kN;

        std::vector<float> host_a(elems_a * kBatch);
        std::vector<float> host_b(elems_b * kBatch, 1.0F);
        for (int batch = 0; batch < kBatch; ++batch) {
            const float value = static_cast<float>(batch + 1);
            for (std::size_t i = 0; i < elems_a; ++i) {
                host_a[batch * elems_a + i] = value;
            }
        }

        ckl::DeviceBuffer<float> a(host_a.size());
        ckl::DeviceBuffer<float> b(host_b.size());
        ckl::DeviceBuffer<float> c(elems_c * kBatch);
        a.copy_from_host(host_a);
        b.copy_from_host(host_b);
        c.zero();

        ckl::GemmDesc desc;
        desc.layout = ckl::Layout::kRowMajor;
        desc.m = kM;
        desc.n = kN;
        desc.k = kK;
        desc.lda = kK;
        desc.ldb = kN;
        desc.ldc = kN;
        desc.stride_a = static_cast<std::int64_t>(elems_a);
        desc.stride_b = static_cast<std::int64_t>(elems_b);
        desc.stride_c = static_cast<std::int64_t>(elems_c);
        desc.batch_count = kBatch;

        const float alpha = 1.0F;
        const float beta = 0.0F;
        ckl::Algo chosen = ckl::Algo::kAuto;

        const ckl::Status s =
            ckl::gemm(ctx, desc, &alpha, a.data(), b.data(), &beta, c.data(), &chosen);
        if (s != ckl::Status::kSuccess) {
            std::printf("FAIL: gemm returned %s\n", ckl::status_string(s));
            return 1;
        }
        CKL_CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));

        const std::vector<float> result = c.to_host();
        for (int batch = 0; batch < kBatch; ++batch) {
            const float want = static_cast<float>(kK) * static_cast<float>(batch + 1);
            for (std::size_t i = 0; i < elems_c; ++i) {
                const float got = result[batch * elems_c + i];
                if (std::fabs(got - want) > 1e-3F) {
                    std::printf("FAIL: batch %d element %zu = %f, expected %f\n", batch, i,
                                static_cast<double>(got), static_cast<double>(want));
                    return 1;
                }
            }
        }

        std::printf("PASS: %d batches of %dx%dx%d via %s, every batch scaled correctly\n", kBatch,
                    kM, kN, kK, ckl::algo_name(chosen));
        return 0;
    } catch (const ckl::Error& e) {
        std::printf("FAIL: ckl::Error(%s): %s\n", ckl::status_string(e.status()), e.what());
        return 1;
    } catch (const std::exception& e) {
        std::printf("FAIL: %s\n", e.what());
        return 1;
    }
}
