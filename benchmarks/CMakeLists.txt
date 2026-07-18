# Benchmark binaries. GPU only at run time; built in CI to keep them warning
# clean but executed only on real hardware.

add_executable(bench_gemm bench_gemm.cpp)
target_link_libraries(bench_gemm PRIVATE ckl_gemm ckl_headers)
ckl_set_warnings(bench_gemm)

add_executable(ncu_driver ncu_driver.cpp)
target_link_libraries(ncu_driver PRIVATE ckl_gemm ckl_headers)
ckl_set_warnings(ncu_driver)
