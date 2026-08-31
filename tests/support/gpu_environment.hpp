#pragma once

// Shared scaffolding for every GoogleTest binary in this tree.
//
// Three jobs. It decides once whether there is a usable CUDA device and turns
// "no device" into a skip that reaches ctest as exit code 77 instead of a
// constructor throwing out of a fixture. It carries the seed that --seed sets,
// so one flag moves every dataset in a suite and CI can run the same suite under
// several seeds. And it puts a CUDA error check in TearDown, so a kernel that
// left a sticky error behind fails the test that caused it rather than the next
// unrelated one.

#include <cstdint>
#include <cstdio>
#include <string>

#include <cuda_runtime.h>

#include <gtest/gtest.h>

namespace ckl::test {

// Defined in test_main.cpp, which every test binary links.
std::uint64_t& seed_ref();
bool& gpu_present();
std::string& gpu_note();

inline std::uint64_t seed() {
    return seed_ref();
}

// A distinct, reproducible seed per dataset. Datasets ask for a stream number
// rather than deriving a seed from their shape: deriving from the shape meant
// two shapes that happened to share a dimension shared their data, and no seed
// on the command line could separate them. The mix is splitmix64, so nearby
// stream numbers give unrelated states.
inline std::uint64_t seed_stream(std::uint64_t stream) {
    std::uint64_t x = seed() + stream * 0x9E3779B97F4A7C15ULL;
    x ^= x >> 30;
    x *= 0xBF58476D1CE4E5B9ULL;
    x ^= x >> 27;
    x *= 0x94D049BB133111EBULL;
    x ^= x >> 31;
    return x;
}

// Scientific notation for the measured numbers a test records. std::to_string
// on a double gives six digits after the point, which prints every error this
// suite cares about as 0.000000.
inline std::string sci(double value) {
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%.6e", value);
    return std::string(buffer);
}

}  // namespace ckl::test

// Skips the current test when the machine has no usable device. Placed in a
// fixture SetUp, so the skip is recorded before any fixture member touches CUDA.
#define CKL_SKIP_WITHOUT_GPU()                                                    \
    do {                                                                          \
        if (!::ckl::test::gpu_present()) {                                        \
            GTEST_SKIP() << "no usable CUDA device: " << ::ckl::test::gpu_note(); \
        }                                                                         \
    } while (0)

namespace ckl::test {

// Fails the current test if a CUDA error is still pending. Anything that left
// one behind produced its result on a poisoned context.
inline void expect_no_pending_cuda_error() {
    if (!gpu_present()) {
        return;
    }
    const cudaError_t e = cudaGetLastError();
    EXPECT_EQ(e, cudaSuccess) << "a CUDA error was left pending at the end of this test: "
                              << cudaGetErrorName(e) << " (" << cudaGetErrorString(e) << ")";
}

// Base fixture for a test that needs a device.
class GpuTest : public ::testing::Test {
protected:
    void SetUp() override { CKL_SKIP_WITHOUT_GPU(); }
    void TearDown() override { expect_no_pending_cuda_error(); }
};

// The same, for value parameterized tests.
template <typename T>
class GpuTestWithParam : public ::testing::TestWithParam<T> {
protected:
    void SetUp() override { CKL_SKIP_WITHOUT_GPU(); }
    void TearDown() override { expect_no_pending_cuda_error(); }
};

}  // namespace ckl::test
