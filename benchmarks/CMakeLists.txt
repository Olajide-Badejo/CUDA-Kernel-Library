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

# The unified sweep binary needs every family plus telemetry. It also links
# cuBLAS directly, which no other benchmark does: the fairness protocol asks for
# the autotuned selection as a named baseline variant, and the library never
# passes anything but CUBLAS_GEMM_DEFAULT, so that one call is made here.
if(TARGET ckl_gemm AND TARGET ckl_gemv AND TARGET ckl_sparse AND TARGET ckl_trsm
        AND TARGET ckl_telemetry AND TARGET ckl_api)
    add_executable(bench_all bench_all.cpp)
    target_link_libraries(bench_all PRIVATE
        ckl_api ckl_gemm ckl_gemv ckl_sparse ckl_trsm ckl_telemetry ckl_headers
        ckl_bench_support CUDA::cublas)
    ckl_set_warnings(bench_all)
endif()
