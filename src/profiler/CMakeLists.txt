# Roofline profiler executable.

add_executable(roofline roofline.cpp)
target_link_libraries(roofline PRIVATE ckl_gemm ckl_gemv ckl_headers)
ckl_set_warnings(roofline)
