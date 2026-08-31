// The entry point every ckl test binary shares.
//
// It exists for three reasons GoogleTest's default main cannot cover. It parses
// --seed before InitGoogleTest and prints the value, so a failing run is
// reproducible from its own first line of output. It probes for a device once
// and turns "no device" into exit code 77, which the CMake side declares as
// SKIP_RETURN_CODE, so a machine without a GPU reports skips rather than a wall
// of aborted binaries. And it wraps RUN_ALL_TESTS in a catch all: an exception
// escaping the framework (a CUDA error thrown out of a fixture constructor, say)
// prints what it was instead of terminating the process with SIGABRT and no
// message.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>

#include <cuda_runtime.h>

#include <gtest/gtest.h>

#include "gpu_environment.hpp"

namespace ckl::test {

std::uint64_t& seed_ref() {
    // Fixed by default: an unseeded run has to be reproducible, and CI varies it
    // explicitly rather than the suite varying itself.
    static std::uint64_t value = 20250901ULL;
    return value;
}

bool& gpu_present() {
    static bool value = false;
    return value;
}

std::string& gpu_note() {
    static std::string value = "not probed";
    return value;
}

}  // namespace ckl::test

namespace {

// Pulls --seed=N or --seed N out of argv before GoogleTest sees them, since
// GoogleTest treats an unknown flag as a positional argument and would leave it
// for a main that did not expect one.
std::uint64_t extract_seed(int& argc, char** argv, bool* found) {
    std::uint64_t value = 0;
    *found = false;
    int out = 1;
    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        const char* digits = nullptr;
        if (std::strncmp(arg, "--seed=", 7) == 0) {
            digits = arg + 7;
        } else if (std::strcmp(arg, "--seed") == 0 && i + 1 < argc) {
            digits = argv[i + 1];
            ++i;
        }
        if (digits != nullptr) {
            value = std::strtoull(digits, nullptr, 10);
            *found = true;
            continue;
        }
        argv[out] = argv[i];
        ++out;
    }
    argc = out;
    argv[out] = nullptr;
    return value;
}

// Runs once before the first test. Anything a suite wants to know about the
// device it is running on is printed here, not per test.
class GpuEnvironment : public ::testing::Environment {
public:
    void SetUp() override {
        if (!::ckl::test::gpu_present()) {
            std::printf("[ ckl      ] no usable CUDA device (%s); every test will skip\n",
                        ::ckl::test::gpu_note().c_str());
            return;
        }
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
            std::printf("[ ckl      ] device 0: %s, compute capability %d.%d, %d SMs\n", prop.name,
                        prop.major, prop.minor, prop.multiProcessorCount);
        }
    }
};

// Probes for a device without letting a failure escape. cudaGetDeviceCount is
// the cheapest call that answers the question and it does not create a context,
// so a listing run (--gtest_list_tests) pays almost nothing for it.
void probe_device() {
    int count = 0;
    const cudaError_t e = cudaGetDeviceCount(&count);
    if (e != cudaSuccess) {
        ::ckl::test::gpu_present() = false;
        ::ckl::test::gpu_note() = std::string(cudaGetErrorName(e)) + ": " + cudaGetErrorString(e);
        cudaGetLastError();  // do not leave the probe's error pending
        return;
    }
    if (count <= 0) {
        ::ckl::test::gpu_present() = false;
        ::ckl::test::gpu_note() = "cudaGetDeviceCount reported 0 devices";
        return;
    }
    ::ckl::test::gpu_present() = true;
    ::ckl::test::gpu_note() = "ok";
}

}  // namespace

int main(int argc, char** argv) {
    bool seed_given = false;
    const std::uint64_t parsed = extract_seed(argc, argv, &seed_given);
    if (seed_given) {
        ::ckl::test::seed_ref() = parsed;
    }

    ::testing::InitGoogleTest(&argc, argv);

    const bool listing = GTEST_FLAG_GET(list_tests);
    if (!listing) {
        std::printf("[ ckl      ] seed %llu%s\n",
                    static_cast<unsigned long long>(::ckl::test::seed()),
                    seed_given ? " (--seed)" : " (default)");
        probe_device();
    }

    ::testing::AddGlobalTestEnvironment(new GpuEnvironment);

    int rc = 1;
    try {
        rc = RUN_ALL_TESTS();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[ ckl      ] unhandled exception escaped the test run: %s\n",
                     e.what());
        return 1;
    } catch (...) {
        std::fprintf(stderr,
                     "[ ckl      ] unhandled non standard exception escaped the test run\n");
        return 1;
    }

    if (!listing && !::ckl::test::gpu_present()) {
        // ctest maps this to a skip through SKIP_RETURN_CODE.
        return 77;
    }
    return rc;
}
