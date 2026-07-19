# Locate NVML (the NVIDIA Management Library). Prefer the CUDAToolkit provided
# target; fall back to the stub in the toolkit and, under WSL2, the runtime
# library the Windows driver installs under /usr/lib/wsl/lib. The header ships
# with the toolkit. Defines an imported target ckl::nvml when found.

if(TARGET ckl::nvml)
    return()
endif()

find_path(NVML_INCLUDE_DIR nvml.h
    HINTS ${CUDAToolkit_INCLUDE_DIRS} /usr/local/cuda/include)

find_library(NVML_LIBRARY
    NAMES nvidia-ml libnvidia-ml.so.1
    HINTS ${CUDAToolkit_LIBRARY_DIR}
          /usr/local/cuda/lib64/stubs
          /usr/lib/wsl/lib
          /usr/lib/x86_64-linux-gnu)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NVML DEFAULT_MSG NVML_LIBRARY NVML_INCLUDE_DIR)

if(NVML_FOUND)
    add_library(ckl::nvml UNKNOWN IMPORTED)
    set_target_properties(ckl::nvml PROPERTIES
        IMPORTED_LOCATION "${NVML_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${NVML_INCLUDE_DIR}")
endif()
