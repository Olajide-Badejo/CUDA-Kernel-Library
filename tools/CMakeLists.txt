# Command line tools. They are part of the workflow, not part of the library, so
# they carry the ckl_ prefix and live outside src/. Gated on CKL_BUILD_TOOLS.

add_executable(ckl_device_probe device_probe.cu)
target_link_libraries(ckl_device_probe PRIVATE ckl_headers)
ckl_set_warnings(ckl_device_probe)
ckl_set_lineinfo(ckl_device_probe)

# The roofline profiler needs the ladder kernels it measures, and ckl_telemetry
# because the compute roofs are built from the NVML graphics clock sampled during
# the measurement window, not from the boost clock in the device properties.
if(TARGET ckl_gemm AND TARGET ckl_gemv AND TARGET ckl_telemetry)
    add_executable(ckl_roofline roofline.cpp)
    target_link_libraries(ckl_roofline PRIVATE
        ckl_gemm ckl_gemv ckl_telemetry ckl_headers)
    # The timing and analytical roofline helpers left the public include tree
    # (Section 6); the tool reaches them where they now live.
    target_include_directories(ckl_roofline PRIVATE
        "${PROJECT_SOURCE_DIR}/benchmarks/support"
        "${PROJECT_SOURCE_DIR}/tests/support")
    ckl_set_warnings(ckl_roofline)
    ckl_set_lineinfo(ckl_roofline)
endif()
