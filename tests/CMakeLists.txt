# Correctness tests. These need a GPU to run; CI marks them hardware only and
# skips them in the cloud (Section 15). ctest runs them locally.
#
# Section 14 replaced the seven hand rolled main() programs with GoogleTest
# suites, one binary per family. The reason is filterability: the old programs
# collapsed roughly 200 logical checks into 7 ctest results, so a single wrong
# element anywhere reported as "test_gemm failed" and the only way to find out
# which case broke was to read the printed table.
#
# support/ holds reference.hpp, gpu_environment.hpp and the shared main. It is
# test scaffolding, so it is not installed and not visible to consumers.
# tests/consume/ is a standalone project that builds against an install prefix;
# it is deliberately not added here.
#
# Gate C runs all four compute-sanitizer tools over the two GEMM suites:
#
#   compute-sanitizer --tool memcheck   ./ckl_test_gemm
#   compute-sanitizer --tool racecheck  ./ckl_test_gemm
#   compute-sanitizer --tool initcheck  ./ckl_test_gemm
#   compute-sanitizer --tool synccheck  ./ckl_test_gemm
#   (the same four for ./ckl_test_gemm_tensor)
#
# One flag is needed: synccheck on the tensor suite wants
# `--num-cuda-barriers 65536`. Its default budget for tracked cuda::barrier
# objects is too small for the cp.async pipeline in gemm_mma_opt running
# alongside the cuBLAS tensor oracle, and the tool then prints an overflow
# warning and fails the run. That is the tool's tracking limit, not a kernel
# defect; with the larger budget it reports zero errors.

include(FetchContent)

# v1.17.0 is the newest tagged release that builds clean under this tree's
# toolchain (nvcc 13.3 host g++-14, C++23). It is pinned to the tag rather than
# to a moving branch so a rebuild months from now discovers the same tests; the
# commit hash for the tag is recorded by FetchContent in the build tree.
FetchContent_Declare(googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.17.0
    GIT_SHALLOW TRUE
    SYSTEM
    EXCLUDE_FROM_ALL)

# Only gtest is used; gmock would double the build for nothing.
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googletest)
include(GoogleTest)

# The shared scaffolding: reference.hpp, gpu_environment.hpp, and the main()
# every suite links. It is a real library rather than an INTERFACE target
# because test_main.cpp has to be compiled once per binary.
add_library(ckl_test_support STATIC support/test_main.cpp)
target_include_directories(ckl_test_support PUBLIC "${CMAKE_CURRENT_LIST_DIR}/support")
target_link_libraries(ckl_test_support PUBLIC GTest::gtest ckl_headers CUDA::cudart)
ckl_set_warnings(ckl_test_support)

# Every discovered test gets the gpu label, the 77 skip contract that
# support/test_main.cpp returns when there is no device, and a timeout. A test
# with no timeout is a hung CI job waiting to happen.
#
# DISCOVERY_MODE PRE_TEST runs the enumeration at ctest time rather than at
# build time, so cross compiling or building on a machine without a GPU still
# configures; the binary only has to be runnable when the tests are.
function(ckl_register_gtest target)
    gtest_discover_tests(${target}
        DISCOVERY_MODE PRE_TEST
        DISCOVERY_TIMEOUT 120
        PROPERTIES
            LABELS "gpu"
            SKIP_RETURN_CODE 77
            TIMEOUT 600)
endfunction()

add_executable(ckl_test_gemm test_gemm.cpp)
target_link_libraries(ckl_test_gemm PRIVATE ckl_gemm ckl_headers ckl_test_support)
ckl_set_warnings(ckl_test_gemm)
ckl_register_gtest(ckl_test_gemm)

add_executable(ckl_test_gemm_tensor test_gemm_tensor.cpp)
target_link_libraries(ckl_test_gemm_tensor PRIVATE ckl_gemm ckl_headers ckl_test_support)
# The shared memory swizzles of the mma_opt mainloop are integer maps, and their
# properties (bijection, chunk preserving, conflict free) are cheaper to check on
# the host than to read off a profiler counter. src/ is on the include path for
# that one private header; nothing else here reaches into the implementation.
target_include_directories(ckl_test_gemm_tensor PRIVATE "${PROJECT_SOURCE_DIR}/src")
ckl_set_warnings(ckl_test_gemm_tensor)
ckl_register_gtest(ckl_test_gemm_tensor)

if(TARGET ckl_api)
    add_executable(ckl_test_dispatch test_dispatch.cpp)
    target_link_libraries(ckl_test_dispatch PRIVATE ckl_api ckl_gemm ckl_headers ckl_test_support)
    # The 9.6 heuristic reads a committed tile sweep when there is one, and the
    # tree deliberately has none: a sweep is owner work at locked clocks, and a
    # made up one would be a measurement claim. support/tile_sweep_fixture.csv is
    # a hand written stand in with obviously synthetic throughput numbers, used
    # only to prove that a sweep overrides the quantization only rule. It is
    # fixture data, not a result, which is why it lives here and not under
    # experiments/results.
    target_compile_definitions(ckl_test_dispatch PRIVATE
        CKL_TILE_SWEEP_FIXTURE="${CMAKE_CURRENT_LIST_DIR}/support/tile_sweep_fixture.csv")
    ckl_set_warnings(ckl_test_dispatch)
    ckl_register_gtest(ckl_test_dispatch)
endif()

if(TARGET ckl_gemv)
    add_executable(ckl_test_gemv test_gemv.cpp)
    target_link_libraries(ckl_test_gemv PRIVATE ckl_gemv ckl_headers ckl_test_support)
    ckl_set_warnings(ckl_test_gemv)
    ckl_register_gtest(ckl_test_gemv)
endif()

# ckl_api for ckl::spmv, which is the dispatcher; ckl_gemm and ckl_gemv for the
# symmetric positive definite case, which builds A = B^T B + n I on the device
# and cross checks the sparse answer against the dense one.
if(TARGET ckl_sparse AND TARGET ckl_api AND TARGET ckl_gemv)
    add_executable(ckl_test_spmv test_spmv.cpp)
    target_link_libraries(ckl_test_spmv PRIVATE
        ckl_api ckl_sparse ckl_gemm ckl_gemv ckl_headers ckl_test_support)
    ckl_set_warnings(ckl_test_spmv)
    ckl_register_gtest(ckl_test_spmv)
endif()

# ckl_api for ckl::fft, ckl::fft2d and ckl::conv, which are the dispatchers, and
# ckl_fft for the rungs and the plans they run out of.
if(TARGET ckl_fft AND TARGET ckl_api)
    add_executable(ckl_test_fft test_fft.cpp)
    target_link_libraries(ckl_test_fft PRIVATE ckl_api ckl_fft ckl_headers ckl_test_support)
    ckl_set_warnings(ckl_test_fft)
    ckl_register_gtest(ckl_test_fft)

    add_executable(ckl_test_conv test_conv.cpp)
    target_link_libraries(ckl_test_conv PRIVATE ckl_api ckl_fft ckl_headers ckl_test_support)
    ckl_set_warnings(ckl_test_conv)
    ckl_register_gtest(ckl_test_conv)
endif()

# ckl_api for ckl::reduce and ckl::scan, which are the dispatchers, and ckl_scan
# for the rungs and the plan they run out of.
#
# This suite is registered by hand rather than through ckl_register_gtest,
# because the 2^28 cases need a label and a timeout of their own. Discovery
# excludes them, and one extra ctest entry runs the whole Slow subset under the
# labels "gpu;slow". A run of `ctest -L gpu -LE slow` is then the fast suite and
# `ctest -L slow` is the large one, and neither measures the other twice.
if(TARGET ckl_scan AND TARGET ckl_api)
    add_executable(ckl_test_scan test_scan.cpp)
    target_link_libraries(ckl_test_scan PRIVATE ckl_api ckl_scan ckl_headers ckl_test_support)
    ckl_set_warnings(ckl_test_scan)
    # TEST_FILTER, not EXTRA_ARGS. EXTRA_ARGS is appended to every generated test
    # command, so a --gtest_filter there lands after the one naming the single
    # test and, last filter winning, every ctest entry would re-run the whole
    # suite. That is not a hypothetical: it cost 9.9 seconds per entry and 670
    # seconds of wall clock before it was caught. TEST_FILTER goes to the
    # discovery invocation only, which is what excluding a suite from discovery
    # actually means.
    gtest_discover_tests(ckl_test_scan
        TEST_FILTER -ScanSlow.*
        DISCOVERY_MODE PRE_TEST
        DISCOVERY_TIMEOUT 120
        PROPERTIES
            LABELS "gpu"
            SKIP_RETURN_CODE 77
            TIMEOUT 900)
    add_test(NAME scan_slow COMMAND ckl_test_scan --gtest_filter=ScanSlow.*)
    set_tests_properties(scan_slow PROPERTIES
        LABELS "gpu;slow"
        SKIP_RETURN_CODE 77
        TIMEOUT 3600)
endif()

if(TARGET ckl_trsm)
    add_executable(ckl_test_trsm test_trsm.cpp)
    target_link_libraries(ckl_test_trsm PRIVATE ckl_trsm ckl_headers ckl_test_support)
    ckl_set_warnings(ckl_test_trsm)
    ckl_register_gtest(ckl_test_trsm)
endif()

if(TARGET ckl_solver)
    add_executable(ckl_test_solver test_solver.cpp)
    target_link_libraries(ckl_test_solver PRIVATE ckl_solver ckl_headers ckl_test_support)
    ckl_set_warnings(ckl_test_solver)
    ckl_register_gtest(ckl_test_solver)
endif()

if(TARGET ckl_telemetry)
    add_executable(ckl_test_nvml test_nvml.cpp)
    target_link_libraries(ckl_test_nvml PRIVATE
        ckl_telemetry ckl_gemm ckl_headers ckl_test_support)
    ckl_set_warnings(ckl_test_nvml)
    ckl_register_gtest(ckl_test_nvml)
endif()

# The C ABI test is two translation units on purpose. test_c_abi.c is C99 and
# includes no CUDA header, which is the property being tested; the device memory
# it needs comes from a C++ shim that exposes a plain C interface. It stays a
# plain C program with its own main: linking it against a C++ test framework
# would defeat what it checks.
if(TARGET ckl_api)
    add_executable(test_c_abi test_c_abi.c test_c_abi_support.cpp)
    set_source_files_properties(test_c_abi.c PROPERTIES LANGUAGE C)
    set_target_properties(test_c_abi PROPERTIES C_STANDARD 99 C_STANDARD_REQUIRED ON)
    target_include_directories(test_c_abi PRIVATE "${CMAKE_CURRENT_LIST_DIR}")
    target_link_libraries(test_c_abi PRIVATE ckl_api ckl_headers CUDA::cudart)
    target_compile_options(test_c_abi PRIVATE
        $<$<COMPILE_LANGUAGE:C>:-Wall -Wextra -Wpedantic -Wshadow -Wconversion>)
    ckl_set_warnings(test_c_abi)
    add_test(NAME c_abi COMMAND test_c_abi)
    set_tests_properties(c_abi PROPERTIES LABELS "gpu" SKIP_RETURN_CODE 77 TIMEOUT 300)
endif()
