# Benchmark binaries. GPU only at run time; built in CI to keep them warning
# clean but executed only on real hardware.

add_executable(bench_gemm bench_gemm.cpp)
target_link_libraries(bench_gemm PRIVATE ckl_gemm ckl_headers)
ckl_set_warnings(bench_gemm)
