// Running on a stream of my own, with a workspace of my own.
//
// Two things are worth taking from this one. First, a Context is externally
// synchronized: one Context per stream and per thread, never one shared handle
// whose stream two threads reset from under each other. Second, ask
// gemm_workspace_size what the shape needs rather than guessing. The FP32 shape
// below answers zero; an FP16 shape the dispatcher splits along K answers with
// the partial planes it needs, and handing that buffer over is what keeps the
// allocation out of the call.

#include <cmath>
#include <cstdio>
#include <vector>

#include "ckl/ckl.hpp"

namespace {

constexpr int kM = 512;
constexpr int kN = 512;
constexpr int kK = 256;

}  // namespace

int main() {
    cudaStream_t stream = nullptr;
    void* workspace = nullptr;

    try {
        ckl::Context ctx;
        CKL_CUDA_CHECK(cudaStreamCreate(&stream));
        ctx.set_stream(stream);

        ckl::GemmDesc desc;
        desc.layout = ckl::Layout::kRowMajor;
        desc.m = kM;
        desc.n = kN;
        desc.k = kK;
        desc.lda = kK;
        desc.ldb = kN;
        desc.ldc = kN;

        const std::size_t need = ckl::gemm_workspace_size(ctx, desc);
        std::printf("workspace this shape asks for: %zu bytes\n", need);
        // A workspace is handed over whether or not this shape wants one, so the
        // call pattern is the same once a rung starts asking for scratch. Zero
        // bytes stays a null pointer.
        const std::size_t give = need > 0 ? need : 0;
        if (give > 0) {
            CKL_CUDA_CHECK(cudaMalloc(&workspace, give));
        }
        ctx.set_workspace(workspace, give);

        std::vector<float> host_a(static_cast<std::size_t>(kM) * kK, 1.0F);
        std::vector<float> host_b(static_cast<std::size_t>(kK) * kN, 1.0F);
        ckl::DeviceBuffer<float> a(host_a.size());
        ckl::DeviceBuffer<float> b(host_b.size());
        ckl::DeviceBuffer<float> c(static_cast<std::size_t>(kM) * kN);
        a.copy_from_host(host_a);
        b.copy_from_host(host_b);
        c.zero();

        const float alpha = 1.0F;
        const float beta = 0.0F;
        ckl::Algo chosen = ckl::Algo::kAuto;

        const ckl::Status s =
            ckl::gemm(ctx, desc, &alpha, a.data(), b.data(), &beta, c.data(), &chosen);
        if (s != ckl::Status::kSuccess) {
            std::printf("FAIL: gemm returned %s\n", ckl::status_string(s));
            return 1;
        }

        // Nothing has run yet as far as the host knows. The work was enqueued on
        // my stream, so my stream is what I wait on.
        CKL_CUDA_CHECK(cudaStreamSynchronize(stream));

        const std::vector<float> result = c.to_host();
        const float want = static_cast<float>(kK);
        for (std::size_t i = 0; i < result.size(); ++i) {
            if (std::fabs(result[i] - want) > 1e-2F) {
                std::printf("FAIL: c[%zu] = %f, expected %f\n", i, static_cast<double>(result[i]),
                            static_cast<double>(want));
                return 1;
            }
        }

        std::printf("PASS: %dx%dx%d fp32 gemm via %s on a non default stream\n", kM, kN, kK,
                    ckl::algo_name(chosen));
        cudaFree(workspace);
        cudaStreamDestroy(stream);
        return 0;
    } catch (const ckl::Error& e) {
        std::printf("FAIL: ckl::Error(%s): %s\n", ckl::status_string(e.status()), e.what());
        cudaFree(workspace);
        cudaStreamDestroy(stream);
        return 1;
    } catch (const std::exception& e) {
        std::printf("FAIL: %s\n", e.what());
        cudaFree(workspace);
        cudaStreamDestroy(stream);
        return 1;
    }
}
