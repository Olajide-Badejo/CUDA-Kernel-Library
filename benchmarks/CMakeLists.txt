# Benchmark binaries. GPU only at run time; built in CI to keep them warning
# clean but executed only on real hardware.
#
# support/ holds event_timer.hpp and roofline.hpp, which used to sit in the
# public include tree. They are harness code, so they are not installed. The
# drivers also use tests/support/reference.hpp to fill and check buffers.

add_library(ckl_bench_support INTERFACE)
target_include_directories(ckl_bench_support INTERFACE
    "${CMAKE_CURRENT_LIST_DIR}/support"
    "${PROJECT_SOURCE_DIR}/tests/support")

add_executable(bench_gemm bench_gemm.cpp)
target_link_libraries(bench_gemm PRIVATE ckl_gemm ckl_headers ckl_bench_support)
ckl_set_warnings(bench_gemm)

add_executable(bench_gemm_tensor bench_gemm_tensor.cpp)
target_link_libraries(bench_gemm_tensor PRIVATE ckl_gemm ckl_headers ckl_bench_support)
ckl_set_warnings(bench_gemm_tensor)

add_executable(fusion_study fusion_study.cpp)
target_link_libraries(fusion_study PRIVATE ckl_gemm ckl_headers ckl_bench_support)
ckl_set_warnings(fusion_study)

add_executable(ncu_driver ncu_driver.cpp)
target_link_libraries(ncu_driver PRIVATE ckl_gemm ckl_headers ckl_bench_support)
ckl_set_warnings(ncu_driver)

if(TARGET ckl_gemv)
    add_executable(bench_gemv bench_gemv.cpp)
    target_link_libraries(bench_gemv PRIVATE ckl_gemv ckl_headers ckl_bench_support)
    ckl_set_warnings(bench_gemv)
endif()

# bench_spmv is the only binary in the tree that links CUPTI. Gate S needs an
# allocation counter around the timed region, and CUPTI is the only way to see a
# cudaMalloc issued from inside a vendor call. It is a profiling dependency of
# one benchmark, never of a library target: nothing under src/ links it.
if(TARGET ckl_sparse AND TARGET ckl_api)
    add_executable(bench_spmv bench_spmv.cpp)
    target_link_libraries(bench_spmv PRIVATE
        ckl_api ckl_sparse ckl_headers ckl_bench_support)
    if(TARGET CUDA::cupti)
        target_link_libraries(bench_spmv PRIVATE CUDA::cupti)
    else()
        message(FATAL_ERROR
            "CUDA::cupti was not found in this toolkit. bench_spmv needs it for the Gate S "
            "allocation counter; a build without it would ship a gate that cannot fail.")
    endif()
    ckl_set_warnings(bench_spmv)
endif()

# bench_fft and bench_conv arm the same CUPTI allocation counter bench_spmv does.
# The claim a plan makes is that a timed call enqueues launches and nothing else,
# and for this family that claim covers a cuFFT handle and a cached filter
# spectrum as well as the twiddle tables.
if(TARGET ckl_fft AND TARGET ckl_api)
    if(NOT TARGET CUDA::cupti)
        message(FATAL_ERROR
            "CUDA::cupti was not found in this toolkit. bench_fft and bench_conv need it for "
            "the allocation counter around their timed regions; a build without it would ship a "
            "gate that cannot fail.")
    endif()

    add_executable(bench_fft bench_fft.cpp)
    target_link_libraries(bench_fft PRIVATE
        ckl_api ckl_fft ckl_headers ckl_bench_support CUDA::cupti)
    ckl_set_warnings(bench_fft)

    add_executable(bench_conv bench_conv.cpp)
    target_link_libraries(bench_conv PRIVATE
        ckl_api ckl_fft ckl_headers ckl_bench_support CUDA::cupti)
    ckl_set_warnings(bench_conv)
endif()

# The vendor fused convolution baseline. cufftXtSetCallback needs static linking
# against libcufft_static and separate device compilation, and libckl links the
# shared cuFFT, so this is a target of its own that links nothing from libckl.
# Section 12.2 of the build spec asks for exactly one such example; the probe
# result and the toolkit version are recorded in docs/fft.md.
if(TARGET CUDA::cufft_static)
    add_executable(bench_conv_callback bench_conv_callback.cu)
    set_target_properties(bench_conv_callback PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(bench_conv_callback PRIVATE CUDA::cufft_static CUDA::culibos)
    ckl_set_warnings(bench_conv_callback)
else()
    message(STATUS
        "CUDA::cufft_static was not found, so the cufftXtSetCallback baseline is not built. "
        "docs/fft.md records the probe result for the toolkit this tree was developed against.")
endif()

# The unified sweep binary needs every family plus telemetry. It also links
# cuBLAS directly, which no other benchmark does: the fairness protocol asks for
# the autotuned selection as a named baseline variant, and the library never
# passes anything but CUBLAS_GEMM_DEFAULT, so that one call is made here.
if(TARGET ckl_gemm AND TARGET ckl_gemv AND TARGET ckl_sparse AND TARGET ckl_fft
        AND TARGET ckl_trsm AND TARGET ckl_telemetry AND TARGET ckl_api)
    add_executable(bench_all bench_all.cpp)
    target_link_libraries(bench_all PRIVATE
        ckl_api ckl_gemm ckl_gemv ckl_sparse ckl_fft ckl_trsm ckl_telemetry ckl_headers
        ckl_bench_support CUDA::cublas)
    ckl_set_warnings(bench_all)
endif()
