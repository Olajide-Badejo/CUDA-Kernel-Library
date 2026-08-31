// The shortest useful CKL program: build a Context, describe an FP32 GEMM,
// run it, and check the answer against a value I can compute by hand.
//
// A is all ones (m by k), B is all ones (k by n), so every element of C is k.

#include <cmath>
#include <cstdio>
#include <vector>

#include "ckl/ckl.hpp"

namespace {

constexpr int kM = 512;
constexpr int kN = 512;
constexpr int kK = 512;

}  // namespace

int main() {
    try {
        ckl::Context ctx;

        std::vector<float> host_a(static_cast<std::size_t>(kM) * kK, 1.0F);
        std::vector<float> host_b(static_cast<std::size_t>(kK) * kN, 1.0F);

        ckl::DeviceBuffer<float> a(host_a.size());
        ckl::DeviceBuffer<float> b(host_b.size());
        ckl::DeviceBuffer<float> c(static_cast<std::size_t>(kM) * kN);
        a.copy_from_host(host_a);
        b.copy_from_host(host_b);
        c.zero();

        ckl::GemmDesc desc;
        desc.layout = ckl::Layout::kRowMajor;
        desc.m = kM;
        desc.n = kN;
        desc.k = kK;
        // Row major and packed: the leading dimension is the row length.
        desc.lda = kK;
        desc.ldb = kN;
        desc.ldc = kN;
        desc.algo = ckl::Algo::kAuto;

        const float alpha = 1.0F;
        const float beta = 0.0F;
        ckl::Algo chosen = ckl::Algo::kAuto;

        const ckl::Status s =
            ckl::gemm(ctx, desc, &alpha, a.data(), b.data(), &beta, c.data(), &chosen);
        if (s != ckl::Status::kSuccess) {
            std::printf("FAIL: gemm returned %s\n", ckl::status_string(s));
            return 1;
        }

        // gemm is asynchronous on the Context stream, so the copy back has to
        // wait for it. to_host uses the default stream, which synchronizes here,
        // but being explicit is what a real program would do.
        CKL_CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));

        const std::vector<float> result = c.to_host();
        const float want = static_cast<float>(kK);
        for (std::size_t i = 0; i < result.size(); ++i) {
            if (std::fabs(result[i] - want) > 1e-2F) {
                std::printf("FAIL: c[%zu] = %f, expected %f\n", i, static_cast<double>(result[i]),
                            static_cast<double>(want));
                return 1;
            }
        }

        std::printf("PASS: %dx%dx%d fp32 gemm via %s, every element %.0f\n", kM, kN, kK,
                    ckl::algo_name(chosen), static_cast<double>(want));
        return 0;
    } catch (const ckl::Error& e) {
        std::printf("FAIL: ckl::Error(%s): %s\n", ckl::status_string(e.status()), e.what());
        return 1;
    } catch (const std::exception& e) {
        std::printf("FAIL: %s\n", e.what());
        return 1;
    }
}
