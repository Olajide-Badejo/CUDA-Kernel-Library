# Roofline profiler executable. ckl_telemetry is linked because the compute roofs
# are built from the NVML graphics clock sampled during the measurement window,
# not from the boost clock in the device properties.

add_executable(roofline roofline.cpp)
target_link_libraries(roofline PRIVATE ckl_gemm ckl_gemv ckl_telemetry ckl_headers)
ckl_set_warnings(roofline)
