// FftPlan, Fft2dPlan and ConvPlan: everything an FFT or convolution variant needs
// that does not change between calls.
//
// The contract is the one ckl::SpmvPlan already carries, for the same reason.
// Building a plan computes the twiddle tables in double precision and rounds them
// to FP32, factors n for the four step form, allocates the ping pong workspace,
// creates the cuFFT handle, and for a convolution transforms the filter once and
// caches its spectrum. A call then enqueues launches and nothing else. A
// benchmark that created a cuFFT plan inside its timed region would be measuring
// plan creation and calling it cuFFT, which is defect A2 in another family's
// clothes.
//
// Three decisions in this file are mine and none of them is a measurement.
//
//   The twiddle tables are factored: t = q * 2^lo_bits + r, one table of the
//   coarse factors and one of the fine ones, multiplied at lookup. A direct table
//   costs 8n bytes, which is 128 MB at 2^24 and is read from DRAM on the late
//   stages, where it would be a quarter of the traffic the model declares. The
//   factored pair is under 64 KB at every size in scope and stays in cache. The
//   price is one complex multiply and one extra rounding per twiddle, which is
//   far inside the 8 log2(n) FLT_EPSILON bound.
//
//   The four step factorization splits log2(n) as evenly as it can, with the
//   larger half first. Both factors then sit at or below 2^12 for every n up to
//   2^24, which is the condition the shared resident sub transform needs, and an
//   even split gives both batched passes the most blocks to fill 48 SMs with.
//
//   kAuto runs the shared resident kernel at or below 2^12 and the four step form
//   above it. That is the shared memory budget and nothing else. The radix 4 and
//   radix 8 rungs are never chosen automatically because whether a wider butterfly
//   wins depends on the occupancy its register count leaves, and there is no
//   committed sweep yet.
//
// All three are provisional and docs/fft.md says so.

#include "ckl/fft.hpp"

#include <cfloat>
#include <cmath>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include <cufft.h>

#include "ckl/cuda_check.hpp"
#include "ckl/device_buffer.hpp"
#include "ckl/status.hpp"

#include "fft_internal.hpp"

namespace ckl {

namespace {

constexpr int kMaxLength = 1 << 24;
constexpr int kSharedResidentMax = 4096;
constexpr int kAlgoCount = 7;
constexpr int kConvAlgoCount = 6;
constexpr double kPi = 3.14159265358979323846;

// The two ceilings the provisional convolution rule compares against. Both are
// already on the record for this part: 579 GB/s is the measured DRAM roof from
// the device probe, and 30.7 TFLOP/s is 6144 FMA lanes times 2 FLOP at the 2.5
// GHz lock. Neither is a measurement of this family, and ConvPlan::query says so.
constexpr double kDramRoofBytesPerSecond = 5.79e11;
constexpr double kFp32RoofFlopsPerSecond = 3.07e13;

void check(cufftResult r, const char* expr) {
    if (r == CUFFT_SUCCESS) {
        return;
    }
    Status s = Status::kExecutionFailed;
    if (r == CUFFT_ALLOC_FAILED) {
        s = Status::kAllocFailed;
    } else if (r == CUFFT_INVALID_SIZE || r == CUFFT_INVALID_VALUE || r == CUFFT_INVALID_PLAN) {
        s = Status::kInvalidValue;
    } else if (r == CUFFT_NOT_SUPPORTED) {
        s = Status::kNotSupported;
    }
    throw Error(s, std::string("cuFFT error ") + std::to_string(static_cast<int>(r)) + ": " + expr);
}

bool power_of_two(int v) {
    return v > 0 && (v & (v - 1)) == 0;
}

int log2_exact(int v) {
    int b = 0;
    while ((1 << b) < v) {
        ++b;
    }
    return b;
}

int next_power_of_two(long long v) {
    int b = 1;
    while (static_cast<long long>(b) < v) {
        b <<= 1;
    }
    return b;
}

}  // namespace

namespace detail {

// A factored twiddle table. See ckl::FftTwiddles for why it is factored. Named
// rather than anonymous because the plan Impl structs hold one and a class with
// external linkage must not have a member whose type has internal linkage.
struct TwiddleTable {
    DeviceBuffer<float2> hi;
    DeviceBuffer<float2> lo;
    int hi_len = 0;
    int lo_len = 0;
    int lo_bits = 0;
    int n = 0;

    void build(int length, bool fast) {
        n = length;
        const int bits = log2_exact(length);
        lo_bits = (bits + 1) / 2;
        lo_len = 1 << lo_bits;
        hi_len = length >> lo_bits;
        hi = DeviceBuffer<float2>(static_cast<std::size_t>(hi_len));
        lo = DeviceBuffer<float2>(static_cast<std::size_t>(lo_len));
        if (fast) {
            detail::fft_fill_twiddles_fast(hi.data(), lo.data(), hi_len, lo_len, length, nullptr);
            CKL_CUDA_CHECK(cudaStreamSynchronize(nullptr));
            return;
        }
        // Double precision, rounded once. A recurrence would be cheaper and its
        // error would grow with the stage index, which is exactly the failure the
        // tolerance model would then have to absorb, so it is banned outright.
        std::vector<float2> h(static_cast<std::size_t>(hi_len));
        std::vector<float2> l(static_cast<std::size_t>(lo_len));
        for (int q = 0; q < hi_len; ++q) {
            const double angle = -2.0 * kPi * static_cast<double>(q) / static_cast<double>(hi_len);
            h[static_cast<std::size_t>(q)] = make_float2(static_cast<float>(std::cos(angle)),
                                                         static_cast<float>(std::sin(angle)));
        }
        for (int r = 0; r < lo_len; ++r) {
            const double angle = -2.0 * kPi * static_cast<double>(r) / static_cast<double>(length);
            l[static_cast<std::size_t>(r)] = make_float2(static_cast<float>(std::cos(angle)),
                                                         static_cast<float>(std::sin(angle)));
        }
        hi.copy_from_host(h);
        lo.copy_from_host(l);
    }

    FftTwiddles view() const {
        FftTwiddles t;
        t.hi = hi.data();
        t.lo = lo.data();
        t.lo_bits = lo_bits;
        t.n = n;
        return t;
    }

    long long bytes() const {
        return 8LL * (static_cast<long long>(hi_len) + static_cast<long long>(lo_len));
    }
};

}  // namespace detail

const char* fft_algo_name(FftAlgo a) {
    switch (a) {
        case FftAlgo::kAuto:
            return "kAuto";
        case FftAlgo::kRadix2Global:
            return "kRadix2Global";
        case FftAlgo::kSharedResident:
            return "kSharedResident";
        case FftAlgo::kRadix4Global:
            return "kRadix4Global";
        case FftAlgo::kRadix8Global:
            return "kRadix8Global";
        case FftAlgo::kFourStep:
            return "kFourStep";
        case FftAlgo::kCufft:
            return "kCufft";
    }
    return "unknown";
}

const char* fft_transpose_name(FftTranspose t) {
    switch (t) {
        case FftTranspose::kNaive:
            return "kNaive";
        case FftTranspose::kTiledPadded:
            return "kTiledPadded";
        case FftTranspose::kStridedShared:
            return "kStridedShared";
    }
    return "unknown";
}

const char* conv_algo_name(ConvAlgo a) {
    switch (a) {
        case ConvAlgo::kAuto:
            return "kAuto";
        case ConvAlgo::kFftSeparate:
            return "kFftSeparate";
        case ConvAlgo::kFftFused:
            return "kFftFused";
        case ConvAlgo::kDirectShared:
            return "kDirectShared";
        case ConvAlgo::kDirectConstant:
            return "kDirectConstant";
        case ConvAlgo::kCufft:
            return "kCufft";
    }
    return "unknown";
}

const char* fft_direction_name(FftDirection d) {
    return d == FftDirection::kForward ? "forward" : "inverse";
}

double fft_tolerance(int n) {
    const int nn = n > 2 ? n : 2;
    return 8.0 * std::log2(static_cast<double>(nn)) * static_cast<double>(FLT_EPSILON);
}

// ---------------------------------------------------------------------------

struct FftPlan::Impl {
    int n = 0;
    int batch = 1;
    int log2n = 0;
    FftPlanOptions opt;

    detail::TwiddleTable tw;
    detail::TwiddleTable tw_double;
    detail::TwiddleTable tw_n1;
    detail::TwiddleTable tw_n2;
    int n1 = 0;
    int n2 = 0;

    DeviceBuffer<float2> work;
    DeviceBuffer<float2> real_a;
    DeviceBuffer<float2> real_b;

    cufftHandle cufft_plan = 0;
    bool cufft_ready = false;
    cudaStream_t cufft_stream = nullptr;

    std::string refusal[kAlgoCount];

    ~Impl() {
        if (cufft_ready) {
            cufftDestroy(cufft_plan);
            cufft_ready = false;
        }
    }

    void build() {
        if (!power_of_two(n) || n < 2 || n > kMaxLength) {
            throw Error(Status::kInvalidValue,
                        "FftPlan: n must be a power of two between 2 and 16777216, and is " +
                            std::to_string(n));
        }
        if (batch < 1) {
            throw Error(Status::kInvalidValue,
                        "FftPlan: batch must be at least 1, and is " + std::to_string(batch));
        }
        log2n = log2_exact(n);
        tw.build(n, opt.fast_twiddles);
        work = DeviceBuffer<float2>(static_cast<std::size_t>(n) * static_cast<std::size_t>(batch));

        factor_four_step();

        if (n <= kSharedResidentMax) {
            detail::fft_shared_prepare(2 * n * static_cast<int>(sizeof(float2)));
        }
        if (n1 > 0) {
            const int wider = n1 > n2 ? n1 : n2;
            detail::fft_shared_prepare(2 * wider * static_cast<int>(sizeof(float2)));
        }

        if (opt.build_real) {
            if (n > kMaxLength / 2) {
                throw Error(Status::kNotSupported,
                            "FftPlan: build_real needs the twiddles of length 2n, and 2n would be "
                            "above the 2^24 ceiling for n = " +
                                std::to_string(n));
            }
            tw_double.build(2 * n, opt.fast_twiddles);
            real_a =
                DeviceBuffer<float2>(static_cast<std::size_t>(n) * static_cast<std::size_t>(batch));
            real_b =
                DeviceBuffer<float2>(static_cast<std::size_t>(n) * static_cast<std::size_t>(batch));
        }

        int length = n;
        check(
            cufftPlanMany(&cufft_plan, 1, &length, nullptr, 1, n, nullptr, 1, n, CUFFT_C2C, batch),
            "cufftPlanMany");
        cufft_ready = true;

        if (n > kSharedResidentMax) {
            refusal[static_cast<int>(FftAlgo::kSharedResident)] =
                "the single block kernel holds two ping pong buffers, which is 16n bytes, and " +
                std::to_string(16LL * n) +
                " bytes is above the 101376 byte per block shared memory limit of this part. The "
                "bound is 4096 points.";
        }
    }

    void factor_four_step() {
        if (log2n < 4) {
            refusal[static_cast<int>(FftAlgo::kFourStep)] =
                "the four step form needs two factors of at least 2 each, so at least 2^4 points, "
                "and n is " +
                std::to_string(n);
            return;
        }
        int p1 = (log2n + 1) / 2;
        if (opt.four_step_n1 != 0) {
            if (!power_of_two(opt.four_step_n1) || n % opt.four_step_n1 != 0) {
                throw Error(Status::kInvalidValue,
                            "FftPlan: four_step_n1 must be a power of two that divides n, and is " +
                                std::to_string(opt.four_step_n1));
            }
            p1 = log2_exact(opt.four_step_n1);
        }
        const int p2 = log2n - p1;
        if (p1 < 1 || p2 < 1) {
            throw Error(Status::kInvalidValue,
                        "FftPlan: the four step factors must both be at least 2 points");
        }
        const int f1 = 1 << p1;
        const int f2 = 1 << p2;
        if (f1 > kSharedResidentMax || f2 > kSharedResidentMax) {
            refusal[static_cast<int>(FftAlgo::kFourStep)] =
                "the four step sub transforms are shared resident, so both factors have to sit at "
                "or below 4096; this factorization is " +
                std::to_string(f1) + " by " + std::to_string(f2);
            return;
        }
        n1 = f1;
        n2 = f2;
        tw_n1.build(n1, opt.fast_twiddles);
        tw_n2.build(n2, opt.fast_twiddles);
    }

    int passes(FftAlgo a) const {
        switch (a) {
            case FftAlgo::kRadix2Global:
                return log2n;
            case FftAlgo::kRadix4Global:
                return detail::fft_ladder(log2n, 4, nullptr);
            case FftAlgo::kRadix8Global:
                return detail::fft_ladder(log2n, 8, nullptr);
            case FftAlgo::kSharedResident:
                return 1;
            case FftAlgo::kFourStep:
                return 5;
            case FftAlgo::kCufft:
            case FftAlgo::kAuto:
            default:
                return 0;
        }
    }

    // Twiddle lookups one call performs. Each one is two loads, because the table
    // is factored.
    long long twiddle_reads(FftAlgo a) const {
        const long long b = batch;
        switch (a) {
            case FftAlgo::kRadix2Global:
            case FftAlgo::kSharedResident:
                return static_cast<long long>(n / 2) * log2n * b;
            case FftAlgo::kRadix4Global:
            case FftAlgo::kRadix8Global: {
                int radix[32];
                const int stages =
                    detail::fft_ladder(log2n, a == FftAlgo::kRadix4Global ? 4 : 8, radix);
                long long reads = 0;
                for (int i = 0; i < stages; ++i) {
                    reads += static_cast<long long>(radix[i] - 1) * (n / radix[i]);
                }
                return reads * b;
            }
            case FftAlgo::kFourStep: {
                if (n1 == 0) {
                    return 0;
                }
                const long long sub2 = static_cast<long long>(n1) * (n2 / 2) * log2_exact(n2);
                const long long sub1 = static_cast<long long>(n2) * (n1 / 2) * log2_exact(n1);
                const long long cross = static_cast<long long>(n1) * n2;
                return (sub2 + sub1 + cross) * b;
            }
            case FftAlgo::kCufft:
            case FftAlgo::kAuto:
            default:
                return 0;
        }
    }

    FftAlgo query() const {
        return n <= kSharedResidentMax ? FftAlgo::kSharedResident : FftAlgo::kFourStep;
    }

    void run_cufft(FftDirection dir, const float2* in, float2* out, cudaStream_t stream) {
        if (cufft_stream != stream) {
            check(cufftSetStream(cufft_plan, stream), "cufftSetStream");
            cufft_stream = stream;
        }
        check(cufftExecC2C(cufft_plan,
                           const_cast<cufftComplex*>(reinterpret_cast<const cufftComplex*>(in)),
                           reinterpret_cast<cufftComplex*>(out),
                           dir == FftDirection::kForward ? CUFFT_FORWARD : CUFFT_INVERSE),
              "cufftExecC2C");
    }

    void require(FftAlgo a) const {
        const std::string& why = refusal[static_cast<int>(a)];
        if (!why.empty()) {
            throw Error(Status::kNotSupported,
                        std::string("FftPlan refuses ") + fft_algo_name(a) + ": " + why);
        }
    }

    void run(FftAlgo a, FftDirection dir, const float2* in, float2* out, cudaStream_t stream) {
        run_epilogue(a, dir, in, out, nullptr, 1.0f, stream);
    }

    void run_epilogue(FftAlgo a, FftDirection dir, const float2* in, float2* out, const float2* mul,
                      float scale, cudaStream_t stream) {
        require(a);
        const bool plain = mul == nullptr && scale == 1.0f;
        switch (a) {
            case FftAlgo::kRadix2Global:
            case FftAlgo::kRadix4Global:
            case FftAlgo::kRadix8Global:
                if (!plain) {
                    throw Error(Status::kNotSupported,
                                std::string("FftPlan: ") + fft_algo_name(a) +
                                    " carries no store epilogue; the fused paths run on the shared "
                                    "resident and four step forms.");
                }
                if (a == FftAlgo::kRadix2Global) {
                    fft_radix2_global(in, out, work.data(), tw.view(), n, batch, dir, stream);
                } else if (a == FftAlgo::kRadix4Global) {
                    fft_radix4_global(in, out, work.data(), tw.view(), n, batch, dir, stream);
                } else {
                    fft_radix8_global(in, out, work.data(), tw.view(), n, batch, dir, stream);
                }
                return;
            case FftAlgo::kSharedResident:
                fft_shared_resident(in, out, tw.view(), n, batch, dir, mul, scale, stream);
                return;
            case FftAlgo::kFourStep:
                fft_four_step(in, out, work.data(), tw_n1.view(), tw_n2.view(), tw.view(), n1, n2,
                              batch, dir, opt.transpose, mul, scale, stream);
                return;
            case FftAlgo::kCufft:
                if (!plain) {
                    throw Error(Status::kNotSupported,
                                "FftPlan: the cuFFT path carries no store epilogue here. The "
                                "callback probe in docs/fft.md records whether cufftXtSetCallback "
                                "is available on this toolkit; the vendor fused row is built "
                                "outside the library, in benchmarks/bench_conv_callback.cu.");
                }
                run_cufft(dir, in, out, stream);
                return;
            case FftAlgo::kAuto:
            default:
                throw Error(Status::kInternal, "FftPlan::run reached kAuto; dispatch resolves it");
        }
    }

    void require_real() const {
        if (!opt.build_real) {
            throw Error(Status::kNotSupported,
                        "FftPlan: this plan was built without FftPlanOptions::build_real, so it "
                        "holds neither the twiddles of the doubled length nor the packing scratch "
                        "the real transforms need.");
        }
    }

    void run_r2c(FftAlgo a, const float* in, float2* out, cudaStream_t stream) {
        require_real();
        fft_pack_real(in, real_a.data(), n, batch, stream);
        run(a, FftDirection::kForward, real_a.data(), real_b.data(), stream);
        fft_r2c_untangle(real_b.data(), out, tw_double.view(), n, batch, stream);
    }

    void run_c2r(FftAlgo a, const float2* in, float* out, cudaStream_t stream) {
        require_real();
        fft_c2r_pack(in, real_a.data(), tw_double.view(), n, batch, stream);
        run(a, FftDirection::kInverse, real_a.data(), real_b.data(), stream);
        fft_unpack_real(real_b.data(), out, n, batch, 1.0f, stream);
    }
};

FftPlan::FftPlan(int n, const FftPlanOptions& opt) : impl_(std::make_unique<Impl>()) {
    impl_->n = n;
    impl_->batch = opt.batch;
    impl_->opt = opt;
    impl_->build();
}

FftPlan::~FftPlan() = default;
FftPlan::FftPlan(FftPlan&&) noexcept = default;
FftPlan& FftPlan::operator=(FftPlan&&) noexcept = default;

FftPlan::Impl& FftPlan::live() const {
    if (!impl_) {
        throw Error(Status::kInvalidValue, "FftPlan: this plan has been moved from");
    }
    return *impl_;
}

int FftPlan::n() const {
    return live().n;
}

int FftPlan::batch() const {
    return live().batch;
}

int FftPlan::log2n() const {
    return live().log2n;
}

bool FftPlan::fast_twiddles() const {
    return live().opt.fast_twiddles;
}

int FftPlan::shared_resident_max() {
    return kSharedResidentMax;
}

int FftPlan::four_step_n1() const {
    return live().n1;
}

int FftPlan::four_step_n2() const {
    return live().n2;
}

int FftPlan::passes(FftAlgo a) const {
    const Impl& p = live();
    return p.passes(a == FftAlgo::kAuto ? p.query() : a);
}

long long FftPlan::model_bytes(FftAlgo a) const {
    const Impl& p = live();
    const FftAlgo resolved = a == FftAlgo::kAuto ? p.query() : a;
    return static_cast<long long>(p.passes(resolved)) * 16LL * p.n * p.batch;
}

long long FftPlan::twiddle_bytes_low(FftAlgo a) const {
    const Impl& p = live();
    const FftAlgo resolved = a == FftAlgo::kAuto ? p.query() : a;
    if (p.twiddle_reads(resolved) == 0) {
        return 0;
    }
    long long bytes = p.tw.bytes();
    if (resolved == FftAlgo::kFourStep && p.n1 > 0) {
        bytes += p.tw_n1.bytes() + p.tw_n2.bytes();
    }
    return bytes;
}

long long FftPlan::twiddle_bytes_high(FftAlgo a) const {
    const Impl& p = live();
    const FftAlgo resolved = a == FftAlgo::kAuto ? p.query() : a;
    return 16LL * p.twiddle_reads(resolved);
}

double FftPlan::flop_model() const {
    const Impl& p = live();
    return 5.0 * static_cast<double>(p.n) * static_cast<double>(p.log2n) *
           static_cast<double>(p.batch);
}

bool FftPlan::supports(FftAlgo a) const {
    const Impl& p = live();
    const FftAlgo resolved = a == FftAlgo::kAuto ? p.query() : a;
    return p.refusal[static_cast<int>(resolved)].empty();
}

const char* FftPlan::refusal(FftAlgo a) const {
    const Impl& p = live();
    const FftAlgo resolved = a == FftAlgo::kAuto ? p.query() : a;
    return p.refusal[static_cast<int>(resolved)].c_str();
}

FftAlgo FftPlan::query() const {
    return live().query();
}

// ---------------------------------------------------------------------------

struct Fft2dPlan::Impl {
    int rows = 0;
    int cols = 0;
    Fft2dPlanOptions opt;
    detail::TwiddleTable tw_rows;
    detail::TwiddleTable tw_cols;
    DeviceBuffer<float2> work;

    void build() {
        if (!power_of_two(rows) || !power_of_two(cols) || rows < 2 || cols < 2 ||
            rows > kSharedResidentMax || cols > kSharedResidentMax) {
            throw Error(Status::kInvalidValue,
                        "Fft2dPlan: both extents must be powers of two between 2 and 4096, the "
                        "shared resident bound the row and column passes run inside; got " +
                            std::to_string(rows) + " by " + std::to_string(cols));
        }
        tw_rows.build(rows, opt.fast_twiddles);
        tw_cols.build(cols, opt.fast_twiddles);
        work =
            DeviceBuffer<float2>(static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols));
        const int wider = rows > cols ? rows : cols;
        detail::fft_shared_prepare(2 * wider * static_cast<int>(sizeof(float2)));
    }

    int passes() const { return opt.transpose == FftTranspose::kStridedShared ? 2 : 4; }

    void run(FftDirection dir, const float2* in, float2* out, cudaStream_t stream) {
        // Row pass: one shared resident transform of length cols per row.
        fft_shared_resident(in, work.data(), tw_cols.view(), cols, rows, dir, nullptr, 1.0f,
                            stream);
        if (opt.transpose == FftTranspose::kStridedShared) {
            // The column pass reads its column straight into the shared transform
            // buffer, so there is no transpose at all and the whole 2D transform
            // costs two passes instead of four.
            detail::fft_shared_strided(work.data(), out, tw_rows.view(), rows, cols, dir, cols, 1,
                                       cols, 1, stream);
            return;
        }
        fft_transpose(work.data(), out, rows, cols, 1, opt.transpose, nullptr, 1.0f, stream);
        fft_shared_resident(out, work.data(), tw_rows.view(), rows, cols, dir, nullptr, 1.0f,
                            stream);
        fft_transpose(work.data(), out, cols, rows, 1, opt.transpose, nullptr, 1.0f, stream);
    }
};

Fft2dPlan::Fft2dPlan(int rows, int cols, const Fft2dPlanOptions& opt)
    : impl_(std::make_unique<Impl>()) {
    impl_->rows = rows;
    impl_->cols = cols;
    impl_->opt = opt;
    impl_->build();
}

Fft2dPlan::~Fft2dPlan() = default;
Fft2dPlan::Fft2dPlan(Fft2dPlan&&) noexcept = default;
Fft2dPlan& Fft2dPlan::operator=(Fft2dPlan&&) noexcept = default;

Fft2dPlan::Impl& Fft2dPlan::live() const {
    if (!impl_) {
        throw Error(Status::kInvalidValue, "Fft2dPlan: this plan has been moved from");
    }
    return *impl_;
}

int Fft2dPlan::rows() const {
    return live().rows;
}

int Fft2dPlan::cols() const {
    return live().cols;
}

FftTranspose Fft2dPlan::transpose() const {
    return live().opt.transpose;
}

int Fft2dPlan::passes() const {
    return live().passes();
}

long long Fft2dPlan::model_bytes() const {
    const Impl& p = live();
    return static_cast<long long>(p.passes()) * 16LL * p.rows * p.cols;
}

long long Fft2dPlan::transpose_model_bytes() const {
    const Impl& p = live();
    if (p.opt.transpose == FftTranspose::kStridedShared) {
        return 0;
    }
    return 2LL * 16LL * p.rows * p.cols;
}

// ---------------------------------------------------------------------------

struct ConvPlan::Impl {
    int n = 0;
    int m = 0;
    int out_len = 0;
    int l = 0;
    ConvPlanOptions opt;
    FftAlgo algo = FftAlgo::kAuto;

    std::unique_ptr<FftPlan> transform;
    DeviceBuffer<float2> a;
    DeviceBuffer<float2> b;
    DeviceBuffer<float2> filter_spec;
    DeviceBuffer<float> taps;

    std::string refusal[kConvAlgoCount];

    void build(const float* filter) {
        if (n < 1 || m < 1) {
            throw Error(Status::kInvalidValue,
                        "ConvPlan: the signal and the filter must both hold at least one sample");
        }
        if (filter == nullptr) {
            throw Error(Status::kInvalidValue, "ConvPlan: the filter pointer is null");
        }
        out_len = n + m - 1;
        const int padded = next_power_of_two(out_len);
        l = padded < 16 ? 16 : padded;
        if (l > kMaxLength) {
            throw Error(Status::kNotSupported, "ConvPlan: the padded transform length would be " +
                                                   std::to_string(l) +
                                                   ", above the 2^24 ceiling of this family");
        }

        FftPlanOptions fopt;
        fopt.batch = 1;
        fopt.fast_twiddles = opt.fast_twiddles;
        transform = std::make_unique<FftPlan>(l, fopt);
        algo = opt.fft_algo == FftAlgo::kAuto ? transform->query() : opt.fft_algo;
        transform->live().require(algo);

        a = DeviceBuffer<float2>(static_cast<std::size_t>(l));
        b = DeviceBuffer<float2>(static_cast<std::size_t>(l));
        filter_spec = DeviceBuffer<float2>(static_cast<std::size_t>(l));
        taps = DeviceBuffer<float>(static_cast<std::size_t>(m));
        CKL_CUDA_CHECK(cudaMemcpy(taps.data(), filter, sizeof(float) * static_cast<std::size_t>(m),
                                  cudaMemcpyDeviceToDevice));

        // The filter spectrum, once. This is the whole reason a convolution has a
        // plan: a timed call must never transform the filter.
        conv_pack(taps.data(), a.data(), m, l, nullptr);
        transform->live().run(algo, FftDirection::kForward, a.data(), filter_spec.data(), nullptr);
        CKL_CUDA_CHECK(cudaStreamSynchronize(nullptr));

        if (algo == FftAlgo::kRadix2Global || algo == FftAlgo::kRadix4Global ||
            algo == FftAlgo::kRadix8Global || algo == FftAlgo::kCufft) {
            refusal[static_cast<int>(ConvAlgo::kFftFused)] =
                std::string(
                    "the fused variant folds the filter multiply into a store epilogue, "
                    "and ") +
                fft_algo_name(algo) +
                " carries none. Build the plan on the shared resident or the "
                "four step form.";
        }
        if (m > conv_direct_constant_max_taps()) {
            refusal[static_cast<int>(ConvAlgo::kDirectConstant)] =
                "the constant memory rung holds at most " +
                std::to_string(conv_direct_constant_max_taps()) + " taps and this filter has " +
                std::to_string(m);
        } else {
            conv_direct_constant_bind(taps.data(), m, this, nullptr);
            CKL_CUDA_CHECK(cudaStreamSynchronize(nullptr));
        }
    }

    int passes(ConvAlgo c) const {
        switch (c) {
            case ConvAlgo::kFftSeparate:
            case ConvAlgo::kFftFused:
                return 2 * transform->passes(algo);
            case ConvAlgo::kDirectShared:
            case ConvAlgo::kDirectConstant:
            case ConvAlgo::kCufft:
            case ConvAlgo::kAuto:
            default:
                return 0;
        }
    }

    // The blocks and chunks the direct kernels run, which is what their traffic
    // model is built out of. Both kernels use a 256 output tile and a 256 tap
    // chunk, and each chunk reads a window of 511 signal samples per block.
    long long direct_signal_reads() const {
        const long long blocks = (out_len + 255) / 256;
        const long long chunks = (m + 255) / 256;
        return blocks * chunks * 511;
    }

    long long model_bytes(ConvAlgo c) const {
        const long long ll = l;
        const long long transform_bytes =
            2LL * transform->passes(algo) * 16LL * ll;  // forward and inverse
        switch (c) {
            case ConvAlgo::kFftSeparate:
                // pack: 4N read plus 8L written. pointwise: 8L twice read, 8L
                // written. extract: 8 read plus 4 written per output sample.
                return 4LL * n + 8LL * ll + transform_bytes + 24LL * ll + 12LL * out_len;
            case ConvAlgo::kFftFused:
                // The pointwise pass is gone; the filter spectrum is read inside
                // the forward transform's store epilogue instead, which is 8L and
                // not 24L, and the 1/L rides on the inverse epilogue for nothing.
                return 4LL * n + 8LL * ll + transform_bytes + 8LL * ll + 12LL * out_len;
            case ConvAlgo::kDirectShared:
                return 4LL * (direct_signal_reads() +
                              static_cast<long long>((out_len + 255) / 256) * m + out_len);
            case ConvAlgo::kDirectConstant:
                // The taps come out of the constant bank, which is not DRAM after
                // the first touch, so only the signal windows and the output are
                // counted.
                return 4LL * (direct_signal_reads() + out_len);
            case ConvAlgo::kCufft:
            case ConvAlgo::kAuto:
            default:
                return 0;
        }
    }

    double flop_model(ConvAlgo c) const {
        if (c == ConvAlgo::kDirectShared || c == ConvAlgo::kDirectConstant) {
            return 2.0 * static_cast<double>(n) * static_cast<double>(m);
        }
        const double ll = static_cast<double>(l);
        return 2.0 * 5.0 * ll * std::log2(ll) + 6.0 * ll;
    }

    ConvAlgo query() const {
        // Arithmetic, not a measurement. The direct path is compute bound and the
        // FFT path is bandwidth bound, so the two models are divided by the two
        // roofs of this part and compared. The crossover this predicts is exactly
        // what the committed sweep exists to measure, and docs/fft.md says the
        // rule is provisional until it runs.
        const double t_direct = flop_model(ConvAlgo::kDirectShared) / kFp32RoofFlopsPerSecond;
        const double t_fft =
            static_cast<double>(model_bytes(ConvAlgo::kFftFused)) / kDramRoofBytesPerSecond;
        if (t_direct < t_fft) {
            return refusal[static_cast<int>(ConvAlgo::kDirectConstant)].empty()
                       ? ConvAlgo::kDirectConstant
                       : ConvAlgo::kDirectShared;
        }
        return refusal[static_cast<int>(ConvAlgo::kFftFused)].empty() ? ConvAlgo::kFftFused
                                                                      : ConvAlgo::kFftSeparate;
    }

    void require(ConvAlgo c) const {
        const std::string& why = refusal[static_cast<int>(c)];
        if (!why.empty()) {
            throw Error(Status::kNotSupported,
                        std::string("ConvPlan refuses ") + conv_algo_name(c) + ": " + why);
        }
    }

    void run(ConvAlgo c, const float* signal, float* out, cudaStream_t stream) {
        require(c);
        FftPlan::Impl& t = transform->live();
        const float inv_l = 1.0f / static_cast<float>(l);
        switch (c) {
            case ConvAlgo::kFftSeparate:
                conv_pack(signal, a.data(), n, l, stream);
                t.run(algo, FftDirection::kForward, a.data(), b.data(), stream);
                conv_pointwise(b.data(), filter_spec.data(), b.data(), l, 1.0f, stream);
                t.run(algo, FftDirection::kInverse, b.data(), a.data(), stream);
                conv_extract(a.data(), out, out_len, inv_l, stream);
                return;
            case ConvAlgo::kFftFused:
                conv_pack(signal, a.data(), n, l, stream);
                t.run_epilogue(algo, FftDirection::kForward, a.data(), b.data(), filter_spec.data(),
                               1.0f, stream);
                t.run_epilogue(algo, FftDirection::kInverse, b.data(), a.data(), nullptr, inv_l,
                               stream);
                conv_extract(a.data(), out, out_len, 1.0f, stream);
                return;
            case ConvAlgo::kDirectShared:
                conv_direct_shared(signal, taps.data(), out, n, m, stream);
                return;
            case ConvAlgo::kDirectConstant:
                if (!conv_direct_constant_bound(this)) {
                    // Another plan took the bank. Rebinding is a 16 KB device to
                    // device copy on this stream, and it only happens when two
                    // plans alternate; the steady state costs nothing.
                    conv_direct_constant_bind(taps.data(), m, this, stream);
                }
                conv_direct_constant(signal, out, n, m, this, stream);
                return;
            case ConvAlgo::kCufft:
                conv_pack(signal, a.data(), n, l, stream);
                t.run(FftAlgo::kCufft, FftDirection::kForward, a.data(), b.data(), stream);
                conv_pointwise(b.data(), filter_spec.data(), b.data(), l, 1.0f, stream);
                t.run(FftAlgo::kCufft, FftDirection::kInverse, b.data(), a.data(), stream);
                conv_extract(a.data(), out, out_len, inv_l, stream);
                return;
            case ConvAlgo::kAuto:
            default:
                throw Error(Status::kInternal, "ConvPlan::run reached kAuto; dispatch resolves it");
        }
    }
};

ConvPlan::ConvPlan(int signal_length, const float* filter, int filter_length,
                   const ConvPlanOptions& opt)
    : impl_(std::make_unique<Impl>()) {
    impl_->n = signal_length;
    impl_->m = filter_length;
    impl_->opt = opt;
    impl_->build(filter);
}

ConvPlan::~ConvPlan() = default;
ConvPlan::ConvPlan(ConvPlan&&) noexcept = default;
ConvPlan& ConvPlan::operator=(ConvPlan&&) noexcept = default;

ConvPlan::Impl& ConvPlan::live() const {
    if (!impl_) {
        throw Error(Status::kInvalidValue, "ConvPlan: this plan has been moved from");
    }
    return *impl_;
}

int ConvPlan::signal_length() const {
    return live().n;
}

int ConvPlan::filter_length() const {
    return live().m;
}

int ConvPlan::output_length() const {
    return live().out_len;
}

int ConvPlan::transform_length() const {
    return live().l;
}

FftAlgo ConvPlan::fft_algo() const {
    return live().algo;
}

int ConvPlan::passes(ConvAlgo a) const {
    const Impl& p = live();
    return p.passes(a == ConvAlgo::kAuto ? p.query() : a);
}

long long ConvPlan::model_bytes(ConvAlgo a) const {
    const Impl& p = live();
    return p.model_bytes(a == ConvAlgo::kAuto ? p.query() : a);
}

double ConvPlan::flop_model(ConvAlgo a) const {
    const Impl& p = live();
    return p.flop_model(a == ConvAlgo::kAuto ? p.query() : a);
}

bool ConvPlan::supports(ConvAlgo a) const {
    const Impl& p = live();
    const ConvAlgo resolved = a == ConvAlgo::kAuto ? p.query() : a;
    return p.refusal[static_cast<int>(resolved)].empty();
}

const char* ConvPlan::refusal(ConvAlgo a) const {
    const Impl& p = live();
    const ConvAlgo resolved = a == ConvAlgo::kAuto ? p.query() : a;
    return p.refusal[static_cast<int>(resolved)].c_str();
}

ConvAlgo ConvPlan::query() const {
    return live().query();
}

// ---------------------------------------------------------------------------

namespace detail {

void fft_launch(FftPlan& plan, FftAlgo algo, FftDirection dir, const float2* in, float2* out,
                cudaStream_t stream) {
    plan.live().run(algo, dir, in, out, stream);
}

void fft_r2c_launch(FftPlan& plan, FftAlgo algo, const float* in, float2* out,
                    cudaStream_t stream) {
    plan.live().run_r2c(algo, in, out, stream);
}

void fft_c2r_launch(FftPlan& plan, FftAlgo algo, const float2* in, float* out,
                    cudaStream_t stream) {
    plan.live().run_c2r(algo, in, out, stream);
}

void fft2d_launch(Fft2dPlan& plan, FftDirection dir, const float2* in, float2* out,
                  cudaStream_t stream) {
    plan.live().run(dir, in, out, stream);
}

void conv_launch(ConvPlan& plan, ConvAlgo algo, const float* signal, float* out,
                 cudaStream_t stream) {
    plan.live().run(algo, signal, out, stream);
}

}  // namespace detail

}  // namespace ckl
