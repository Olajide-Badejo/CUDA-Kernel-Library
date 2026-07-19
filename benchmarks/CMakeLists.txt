# Benchmark binaries. GPU only at run time; built in CI to keep them warning
# clean but executed only on real hardware.

add_executable(bench_gemm bench_gemm.cpp)
target_link_libraries(bench_gemm PRIVATE ckl_gemm ckl_headers)
ckl_set_warnings(bench_gemm)

add_executable(bench_gemm_tensor bench_gemm_tensor.cpp)
target_link_libraries(bench_gemm_tensor PRIVATE ckl_gemm ckl_headers)
ckl_set_warnings(bench_gemm_tensor)

add_executable(fusion_study fusion_study.cpp)
target_link_libraries(fusion_study PRIVATE ckl_gemm ckl_headers)
ckl_set_warnings(fusion_study)

add_executable(ncu_driver ncu_driver.cpp)
target_link_libraries(ncu_driver PRIVATE ckl_gemm ckl_headers)
ckl_set_warnings(ncu_driver)

if(TARGET ckl_gemv)
    add_executable(bench_gemv bench_gemv.cpp)
    target_link_libraries(bench_gemv PRIVATE ckl_gemv ckl_headers)
    ckl_set_warnings(bench_gemv)
endif()

if(TARGET ckl_sparse)
    add_executable(bench_spmv bench_spmv.cpp)
    target_link_libraries(bench_spmv PRIVATE ckl_sparse ckl_headers)
    ckl_set_warnings(bench_spmv)
endif()

# The unified sweep binary needs every family plus telemetry.
if(TARGET ckl_gemm AND TARGET ckl_gemv AND TARGET ckl_sparse AND TARGET ckl_trsm
        AND TARGET ckl_telemetry)
    add_executable(bench_all bench_all.cpp)
    target_link_libraries(bench_all PRIVATE
        ckl_gemm ckl_gemv ckl_sparse ckl_trsm ckl_telemetry ckl_headers)
    ckl_set_warnings(bench_all)
endif()
