// The vendor fused convolution baseline, built with cufftXtSetCallback.
//
// Why this is a binary of its own. cufftXtSetCallback needs static linking
// against libcufft_static and separate device compilation, and libckl links the
// shared cuFFT; putting both in one process would mean two copies of cuFFT.
// So this target links libcufft_static, turns on separable compilation, and links
// nothing from libckl. It is a benchmark, not part of the library, and no library
// target depends on it.
//
// Why it exists at all. Section 12.2 of the build spec says: spend thirty minutes
// building one callback example against the installed toolkit; if it links and
// runs it is the vendor fused baseline, and if it does not, the failure and the
// toolkit version go in docs/fft.md and every comparing table says the vendor
// fused path is plan plus a separate pointwise kernel. On this toolkit the probe
// linked and ran, so this is that baseline. docs/fft.md records the probe.
//
// What it measures. Two vendor paths that compute the same convolution:
//
//   separate: cufftExecC2C forward, a standalone pointwise multiply kernel that
//             reads two spectra and writes one, cufftExecC2C inverse.
//   fused:    cufftExecC2C forward with a store callback that multiplies by the
//             cached filter spectrum as each output element is written, then
//             cufftExecC2C inverse with a store callback that applies the 1/L.
//
// The difference between them is the pointwise pass, on the vendor's side of the
// comparison, measured the same way the library's own kFftSeparate and kFftFused
// rows are.
//
// Every number this prints is at whatever clocks the machine happened to be at.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <cufft.h>
#include <cufftXt.h>

namespace {

constexpr int kBlock = 256;

// What the callback needs: the filter spectrum and the scale. Passed as the
// caller info pointer, so one callback serves both directions.
struct CallbackInfo {
    const cufftComplex* filter;
    float scale;
};

__device__ void store_fused(void* data_out, size_t offset, cufftComplex element, void* caller_info,
                            void* shared_ptr) {
    (void)shared_ptr;
    const CallbackInfo* info = static_cast<const CallbackInfo*>(caller_info);
    cufftComplex v = element;
    if (info->filter != nullptr) {
        const cufftComplex f = info->filter[offset];
        v = make_cuComplex(element.x * f.x - element.y * f.y, element.x * f.y + element.y * f.x);
    }
    v.x *= info->scale;
    v.y *= info->scale;
    static_cast<cufftComplex*>(data_out)[offset] = v;
}

__device__ cufftCallbackStoreC d_store_fused = store_fused;

__global__ void pointwise(const cufftComplex* a, const cufftComplex* b, cufftComplex* out, int l) {
    const int t =
        static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (t >= l) {
        return;
    }
    out[t] = make_cuComplex(a[t].x * b[t].x - a[t].y * b[t].y, a[t].x * b[t].y + a[t].y * b[t].x);
}

__global__ void scale_kernel(cufftComplex* a, int l, float s) {
    const int t =
        static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (t >= l) {
        return;
    }
    a[t].x *= s;
    a[t].y *= s;
}

__global__ void pack(const float* signal, cufftComplex* padded, int n, int l) {
    const int t =
        static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (t >= l) {
        return;
    }
    padded[t] = t < n ? make_cuComplex(signal[t], 0.0f) : make_cuComplex(0.0f, 0.0f);
}

__global__ void extract(const cufftComplex* spectrum, float* out, int out_len) {
    const int t =
        static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (t >= out_len) {
        return;
    }
    out[t] = spectrum[t].x;
}

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s failed: %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}

void check(cufftResult r, const char* what) {
    if (r != CUFFT_SUCCESS) {
        std::fprintf(stderr, "%s failed: cuFFT result %d\n", what, static_cast<int>(r));
        std::exit(3);
    }
}

int grid_of(int total) {
    return (total + kBlock - 1) / kBlock;
}

int next_power_of_two(long long v) {
    int b = 1;
    while (static_cast<long long>(b) < v) {
        b <<= 1;
    }
    return b;
}

// mt19937_64 mapped to a float by explicit bit arithmetic, the same fill the rest
// of the harness uses, so the data is the same on every standard library.
std::vector<float> random_signal(int count, unsigned long long seed) {
    std::vector<float> out(static_cast<std::size_t>(count));
    unsigned long long x = seed;
    for (float& v : out) {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        const unsigned long long m = (x >> 40) & ((1ULL << 24) - 1ULL);
        v = static_cast<float>(static_cast<double>(m) * (2.0 / 16777215.0) - 1.0);
    }
    return out;
}

double median_of(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// CUDA events, warmups then reps, the same shape as the harness timer. The timer
// header is not reused here on purpose: this target links libcufft_static and
// nothing from libckl, which is what keeps the two cuFFT copies apart.
template <typename F>
double time_ms(F&& launch, cudaStream_t stream, int warmups, int reps) {
    cudaEvent_t start;
    cudaEvent_t stop;
    check(cudaEventCreate(&start), "cudaEventCreate");
    check(cudaEventCreate(&stop), "cudaEventCreate");
    for (int i = 0; i < warmups; ++i) {
        launch(stream);
    }
    check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(reps));
    for (int i = 0; i < reps; ++i) {
        check(cudaEventRecord(start, stream), "cudaEventRecord");
        launch(stream);
        check(cudaEventRecord(stop, stream), "cudaEventRecord");
        check(cudaEventSynchronize(stop), "cudaEventSynchronize");
        float ms = 0.0f;
        check(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime");
        samples.push_back(static_cast<double>(ms));
    }
    check(cudaEventDestroy(start), "cudaEventDestroy");
    check(cudaEventDestroy(stop), "cudaEventDestroy");
    return median_of(samples);
}

}  // namespace

int main(int argc, char** argv) {
    int log2n = 20;
    int m = 256;
    int warmups = 5;
    int reps = 50;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            std::printf(
                "usage: %s [log2n] [--filter M] [--warmups N] [--reps N]\n"
                "  The vendor fused convolution baseline, built against libcufft_static with\n"
                "  cufftXtSetCallback. See docs/fft.md for the callback probe result.\n",
                argv[0]);
            return 0;
        }
        if (arg == "--filter" && i + 1 < argc) {
            m = std::atoi(argv[++i]);
        } else if (arg == "--warmups" && i + 1 < argc) {
            warmups = std::atoi(argv[++i]);
        } else if (arg == "--reps" && i + 1 < argc) {
            reps = std::atoi(argv[++i]);
        } else if (arg.rfind("--", 0) != 0) {
            log2n = std::atoi(arg.c_str());
        } else {
            std::fprintf(stderr, "unknown flag %s\n", arg.c_str());
            return 2;
        }
    }
    const int n = 1 << log2n;
    const int out_len = n + m - 1;
    const int l = next_power_of_two(out_len);

    int runtime = 0;
    cudaRuntimeGetVersion(&runtime);
    std::printf("cuda runtime %d, cufft %d.%d.%d, static link with cufftXtSetCallback\n", runtime,
                CUFFT_VER_MAJOR, CUFFT_VER_MINOR, CUFFT_VER_PATCH);
    std::printf("N = %d (2^%d), M = %d, output %d, padded L = %d\n", n, log2n, m, out_len, l);

    cudaStream_t stream = nullptr;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamDefault), "cudaStreamCreateWithFlags");

    const std::vector<float> signal = random_signal(n, 5);
    const std::vector<float> filter = random_signal(m, 6);
    float* d_signal = nullptr;
    float* d_filter = nullptr;
    float* d_out = nullptr;
    cufftComplex* d_a = nullptr;
    cufftComplex* d_b = nullptr;
    cufftComplex* d_filter_spec = nullptr;
    CallbackInfo* d_info_fwd = nullptr;
    CallbackInfo* d_info_inv = nullptr;
    check(cudaMalloc(&d_signal, sizeof(float) * signal.size()), "cudaMalloc");
    check(cudaMalloc(&d_filter, sizeof(float) * filter.size()), "cudaMalloc");
    check(cudaMalloc(&d_out, sizeof(float) * static_cast<std::size_t>(out_len)), "cudaMalloc");
    check(cudaMalloc(&d_a, sizeof(cufftComplex) * static_cast<std::size_t>(l)), "cudaMalloc");
    check(cudaMalloc(&d_b, sizeof(cufftComplex) * static_cast<std::size_t>(l)), "cudaMalloc");
    check(cudaMalloc(&d_filter_spec, sizeof(cufftComplex) * static_cast<std::size_t>(l)),
          "cudaMalloc");
    check(cudaMalloc(&d_info_fwd, sizeof(CallbackInfo)), "cudaMalloc");
    check(cudaMalloc(&d_info_inv, sizeof(CallbackInfo)), "cudaMalloc");
    check(
        cudaMemcpy(d_signal, signal.data(), sizeof(float) * signal.size(), cudaMemcpyHostToDevice),
        "cudaMemcpy");
    check(
        cudaMemcpy(d_filter, filter.data(), sizeof(float) * filter.size(), cudaMemcpyHostToDevice),
        "cudaMemcpy");

    // Three plans, all created here, outside every timed region: one plain plan
    // for the separate path and for building the filter spectrum, and one per
    // direction for the fused path, because a callback is set on a plan.
    cufftHandle plain;
    cufftHandle fused_fwd;
    cufftHandle fused_inv;
    check(cufftPlan1d(&plain, l, CUFFT_C2C, 1), "cufftPlan1d");
    check(cufftPlan1d(&fused_fwd, l, CUFFT_C2C, 1), "cufftPlan1d");
    check(cufftPlan1d(&fused_inv, l, CUFFT_C2C, 1), "cufftPlan1d");
    check(cufftSetStream(plain, stream), "cufftSetStream");
    check(cufftSetStream(fused_fwd, stream), "cufftSetStream");
    check(cufftSetStream(fused_inv, stream), "cufftSetStream");

    // The filter spectrum, once. A benchmark that transformed the filter inside
    // its timed region would be timing the wrong thing on both sides.
    pack<<<grid_of(l), kBlock, 0, stream>>>(d_filter, d_a, m, l);
    check(cudaGetLastError(), "pack");
    check(cufftExecC2C(plain, d_a, d_filter_spec, CUFFT_FORWARD), "cufftExecC2C");
    check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

    const CallbackInfo host_fwd{d_filter_spec, 1.0f};
    const CallbackInfo host_inv{nullptr, 1.0f / static_cast<float>(l)};
    check(cudaMemcpy(d_info_fwd, &host_fwd, sizeof(CallbackInfo), cudaMemcpyHostToDevice),
          "cudaMemcpy");
    check(cudaMemcpy(d_info_inv, &host_inv, sizeof(CallbackInfo), cudaMemcpyHostToDevice),
          "cudaMemcpy");

    cufftCallbackStoreC h_store = nullptr;
    check(cudaMemcpyFromSymbol(&h_store, d_store_fused, sizeof(h_store)), "cudaMemcpyFromSymbol");
    check(cufftXtSetCallback(fused_fwd, reinterpret_cast<void**>(&h_store), CUFFT_CB_ST_COMPLEX,
                             reinterpret_cast<void**>(&d_info_fwd)),
          "cufftXtSetCallback forward");
    check(cufftXtSetCallback(fused_inv, reinterpret_cast<void**>(&h_store), CUFFT_CB_ST_COMPLEX,
                             reinterpret_cast<void**>(&d_info_inv)),
          "cufftXtSetCallback inverse");

    auto run_separate = [&](cudaStream_t s) {
        pack<<<grid_of(l), kBlock, 0, s>>>(d_signal, d_a, n, l);
        cufftExecC2C(plain, d_a, d_b, CUFFT_FORWARD);
        pointwise<<<grid_of(l), kBlock, 0, s>>>(d_b, d_filter_spec, d_b, l);
        cufftExecC2C(plain, d_b, d_a, CUFFT_INVERSE);
        scale_kernel<<<grid_of(l), kBlock, 0, s>>>(d_a, l, 1.0f / static_cast<float>(l));
        extract<<<grid_of(out_len), kBlock, 0, s>>>(d_a, d_out, out_len);
    };
    auto run_fused = [&](cudaStream_t s) {
        pack<<<grid_of(l), kBlock, 0, s>>>(d_signal, d_a, n, l);
        cufftExecC2C(fused_fwd, d_a, d_b, CUFFT_FORWARD);
        cufftExecC2C(fused_inv, d_b, d_a, CUFFT_INVERSE);
        extract<<<grid_of(out_len), kBlock, 0, s>>>(d_a, d_out, out_len);
    };

    // Both paths have to agree before either is timed, which is also the proof
    // that the callback ran: without it the fused answer would be unscaled and
    // unmultiplied, and the comparison would fail by orders of magnitude.
    std::vector<float> from_separate(static_cast<std::size_t>(out_len));
    std::vector<float> from_fused(static_cast<std::size_t>(out_len));
    run_separate(stream);
    check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    check(cudaMemcpy(from_separate.data(), d_out, sizeof(float) * from_separate.size(),
                     cudaMemcpyDeviceToHost),
          "cudaMemcpy");
    run_fused(stream);
    check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    check(cudaMemcpy(from_fused.data(), d_out, sizeof(float) * from_fused.size(),
                     cudaMemcpyDeviceToHost),
          "cudaMemcpy");

    double num = 0.0;
    double den = 0.0;
    for (std::size_t i = 0; i < from_separate.size(); ++i) {
        const double d = static_cast<double>(from_fused[i]) - static_cast<double>(from_separate[i]);
        num += d * d;
        den += static_cast<double>(from_separate[i]) * static_cast<double>(from_separate[i]);
    }
    const double agreement = den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
    // 8 log2(L) FLT_EPSILON, the family's derived bound, evaluated here rather
    // than pulled from libckl because this target links none of it.
    const double bound = 8.0 * std::log2(static_cast<double>(l)) * 1.1920929e-7;
    std::printf("fused against separate: relative RMS %.4e, bound %.4e\n", agreement, bound);
    if (!(agreement <= bound)) {
        std::fprintf(stderr,
                     "the two vendor paths disagree, so the callback did not do what the separate "
                     "pointwise kernel does; nothing timed\n");
        return 4;
    }

    const double separate_ms = time_ms(run_separate, stream, warmups, reps);
    const double fused_ms = time_ms(run_fused, stream, warmups, reps);
    std::printf("\n%-28s %10s %10s\n", "variant", "median_ms", "pct_sep");
    std::printf("%-28s %10.4f %10s\n", "baseline_cufft_separate", separate_ms, "100.0%");
    std::printf("%-28s %10.4f %9.1f%%\n", "baseline_cufft_callback", fused_ms,
                fused_ms > 0.0 ? 100.0 * separate_ms / fused_ms : 0.0);
    std::printf(
        "\nbaseline_cufft_callback is the vendor fused path: cufftXtSetCallback linked and ran "
        "on this\ntoolkit, so a table comparing against a vendor fused convolution means this "
        "row.\n");

    check(cufftDestroy(plain), "cufftDestroy");
    check(cufftDestroy(fused_fwd), "cufftDestroy");
    check(cufftDestroy(fused_inv), "cufftDestroy");
    check(cudaFree(d_signal), "cudaFree");
    check(cudaFree(d_filter), "cudaFree");
    check(cudaFree(d_out), "cudaFree");
    check(cudaFree(d_a), "cudaFree");
    check(cudaFree(d_b), "cudaFree");
    check(cudaFree(d_filter_spec), "cudaFree");
    check(cudaFree(d_info_fwd), "cudaFree");
    check(cudaFree(d_info_inv), "cudaFree");
    check(cudaStreamDestroy(stream), "cudaStreamDestroy");
    return 0;
}
