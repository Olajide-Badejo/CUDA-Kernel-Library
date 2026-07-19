# NVML telemetry monitor. Uses the CUDAToolkit NVML target when present, else the
# FindNVML fallback (stub plus the WSL runtime library).

add_library(ckl_telemetry STATIC
    nvml_monitor.cpp
)

target_link_libraries(ckl_telemetry PUBLIC ckl_headers)

if(TARGET CUDA::nvml)
    target_link_libraries(ckl_telemetry PRIVATE CUDA::nvml)
else()
    find_package(NVML)
    if(TARGET ckl::nvml)
        target_link_libraries(ckl_telemetry PRIVATE ckl::nvml)
    else()
        message(WARNING "NVML not found; telemetry will report unavailable at runtime")
    endif()
endif()

find_package(Threads REQUIRED)
target_link_libraries(ckl_telemetry PRIVATE Threads::Threads)

set_target_properties(ckl_telemetry PROPERTIES POSITION_INDEPENDENT_CODE ON)
ckl_set_warnings(ckl_telemetry)
