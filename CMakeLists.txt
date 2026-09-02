cmake_minimum_required(VERSION 3.28)

# Default to this machine only: RTX 5070, Blackwell GB205, compute capability
# 12.0. Set before project() so enable_language(CUDA) picks it up; project()
# would otherwise seed a default (sm_75 here) that a later set() cannot dislodge.
# The release and multiarch presets override this with the full fat binary list.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 120)
endif()

# CXX is enabled alongside CUDA because the math library wrappers and some tests
# are plain host translation units. C is enabled so the C ABI test compiles as
# C99 rather than as C++ that happens to parse.
project(cuda_kernel_lab
    VERSION 1.1.0
    DESCRIPTION "CUDA kernel library driven from naive to compute bound tensor core GEMM"
    LANGUAGES C CXX CUDA)

# ---------------------------------------------------------------------------
# Language standards
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Device code standard: the newest the installed nvcc accepts. CUDA 13.3
# accepts C++20 for device code.
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# The C ABI is C99 and the test that exercises it is compiled as C99.
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
include(CompilerWarnings)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(GenerateExportHeader)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(CKL_BUILD_TESTS "Build correctness tests (need a GPU to run)" ON)
option(CKL_BUILD_BENCH "Build benchmark binaries (need a GPU to run)" ON)
option(CKL_BUILD_TOOLS "Build the device probe and roofline command line tools" ON)
option(CKL_BUILD_FORTRAN "Build the Fortran interface and driver" OFF)
option(CKL_BUILD_EXAMPLES "Build the standalone examples" ON)
# Warnings are errors locally (a phase gate). CI compiles sm_120 with a different
# host GCC than this machine, whose warning set differs, so CI turns this off to
# get a real compile check without failing on cross toolchain warning noise.
option(CKL_WERROR "Treat compiler warnings as errors" ON)
# Line info lets ncu attribute stalls to source lines without paying the -G debug
# penalty. It is a profiling aid, so it is a private per target flag rather than a
# global CMAKE_CUDA_FLAGS append that would also reach anyone embedding this tree.
option(CKL_LINEINFO "Emit -lineinfo for CUDA sources in Release builds" ON)

# ---------------------------------------------------------------------------
# Toolkit
# ---------------------------------------------------------------------------
find_package(CUDAToolkit REQUIRED)

# 12.8 is the first toolkit that can emit sm_120 SASS. Anything older cannot
# build the default architecture, and the failure it produces on its own is a
# ptxas message about an unknown target that says nothing useful.
if(CUDAToolkit_VERSION VERSION_LESS 12.8)
    message(FATAL_ERROR
        "CUDA Toolkit 12.8 or newer is required: 12.8 is the first release with "
        "sm_120 codegen, which this project targets by default. "
        "Found ${CUDAToolkit_VERSION}.")
endif()

# ---------------------------------------------------------------------------
# Generated headers
# ---------------------------------------------------------------------------
configure_file(
    "${PROJECT_SOURCE_DIR}/cmake/version.hpp.in"
    "${PROJECT_BINARY_DIR}/include/ckl/version.hpp"
    @ONLY)

# ---------------------------------------------------------------------------
# Header-only interface for the ckl public headers
# ---------------------------------------------------------------------------
add_library(ckl_headers INTERFACE)
add_library(ckl::headers ALIAS ckl_headers)
set_target_properties(ckl_headers PROPERTIES EXPORT_NAME headers)

target_include_directories(ckl_headers INTERFACE
    "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
    "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")

# Nothing in the public headers needs more than C++17; individual translation
# units ask for a higher standard themselves.
target_compile_features(ckl_headers INTERFACE cxx_std_17)
target_link_libraries(ckl_headers INTERFACE CUDA::cudart)

# Without this a static consumer on MSVC reads __declspec(dllimport) off an
# archive and fails to link.
target_compile_definitions(ckl_headers INTERFACE
    "$<$<NOT:$<BOOL:${BUILD_SHARED_LIBS}>>:CKL_STATIC_DEFINE>")

# ---------------------------------------------------------------------------
# Shared per target settings
# ---------------------------------------------------------------------------

# Profiling line info, per target and only where it is wanted.
function(ckl_set_lineinfo target)
    target_compile_options(${target} PRIVATE
        "$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>,$<BOOL:${CKL_LINEINFO}>>:-lineinfo>")
endfunction()

# Everything a shipped library needs: soname, hidden visibility, the shared
# export symbol the single generated ckl_export.h keys off, and membership in the
# CKLTargets export set. `component` becomes the short exported name, so
# ckl_gemm is reachable as ckl::gemm from an install tree.
function(ckl_configure_library target component)
    set_target_properties(${target} PROPERTIES
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}
        POSITION_INDEPENDENT_CODE ON
        CXX_VISIBILITY_PRESET hidden
        CUDA_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
        DEFINE_SYMBOL CKL_EXPORTS
        EXPORT_NAME ${component})
    ckl_set_lineinfo(${target})
    ckl_set_warnings(${target})
    add_library(ckl::${component} ALIAS ${target})
    install(TARGETS ${target}
        EXPORT CKLTargets
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
endfunction()

# ---------------------------------------------------------------------------
# Libraries
# ---------------------------------------------------------------------------
# Guarded so a partial tree still configures.
if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/src/gemm/CMakeLists.txt")
    add_subdirectory(src/gemm)
endif()

if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/src/gemv/CMakeLists.txt")
    add_subdirectory(src/gemv)
endif()

if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/src/sparse/CMakeLists.txt")
    add_subdirectory(src/sparse)
endif()

if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/src/fft/CMakeLists.txt")
    add_subdirectory(src/fft)
endif()

if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/src/scan/CMakeLists.txt")
    add_subdirectory(src/scan)
endif()

if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/src/trsm/CMakeLists.txt")
    add_subdirectory(src/trsm)
endif()

if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/src/solver/CMakeLists.txt")
    add_subdirectory(src/solver)
endif()

if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/src/telemetry/CMakeLists.txt")
    add_subdirectory(src/telemetry)
endif()

# Added last because ckl_api links the family libraries. ckl_core, defined in the
# same directory, sits underneath them instead: the families reference it by name
# above and CMake resolves that at generate time.
if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/src/api/CMakeLists.txt")
    add_subdirectory(src/api)
endif()

# ---------------------------------------------------------------------------
# Aggregate target: one link line for a consumer that wants the whole library
# ---------------------------------------------------------------------------
add_library(ckl INTERFACE)
add_library(ckl::ckl ALIAS ckl)
target_link_libraries(ckl INTERFACE
    ckl_headers ckl_core ckl_api
    ckl_gemm ckl_gemv ckl_sparse ckl_fft ckl_scan ckl_trsm ckl_solver ckl_telemetry)

# ---------------------------------------------------------------------------
# Tools, tests, benchmarks, and the later phase hooks
# ---------------------------------------------------------------------------
if(CKL_BUILD_TOOLS AND EXISTS "${CMAKE_CURRENT_LIST_DIR}/tools/CMakeLists.txt")
    add_subdirectory(tools)
endif()

if(CKL_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_LIST_DIR}/tests/CMakeLists.txt")
    enable_testing()
    add_subdirectory(tests)
endif()

if(CKL_BUILD_BENCH AND EXISTS "${CMAKE_CURRENT_LIST_DIR}/benchmarks/CMakeLists.txt")
    add_subdirectory(benchmarks)
endif()

if(CKL_BUILD_FORTRAN AND EXISTS "${CMAKE_CURRENT_LIST_DIR}/fortran/CMakeLists.txt")
    add_subdirectory(fortran)
endif()

if(CKL_BUILD_EXAMPLES AND EXISTS "${CMAKE_CURRENT_LIST_DIR}/examples/CMakeLists.txt")
    add_subdirectory(examples)
endif()

# ---------------------------------------------------------------------------
# Install and export
# ---------------------------------------------------------------------------
install(TARGETS ckl_headers ckl
    EXPORT CKLTargets
    INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

install(DIRECTORY "${PROJECT_SOURCE_DIR}/include/ckl"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    FILES_MATCHING
        PATTERN "*.h"
        PATTERN "*.hpp")

install(FILES
        "${PROJECT_BINARY_DIR}/include/ckl/version.hpp"
        "${PROJECT_BINARY_DIR}/include/ckl/ckl_export.h"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/ckl")

# The vendor libraries are PRIVATE, so on a static build they reach the export as
# $<LINK_ONLY:...> and a consumer still has to resolve them. NVML comes from our
# own find module, which therefore has to ship next to the config.
install(FILES "${PROJECT_SOURCE_DIR}/cmake/FindNVML.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/CKL")

install(EXPORT CKLTargets
    FILE CKLTargets.cmake
    NAMESPACE ckl::
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/CKL")

configure_package_config_file(
    "${PROJECT_SOURCE_DIR}/cmake/CKLConfig.cmake.in"
    "${PROJECT_BINARY_DIR}/CKLConfig.cmake"
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/CKL")

write_basic_package_version_file(
    "${PROJECT_BINARY_DIR}/CKLConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)

install(FILES
        "${PROJECT_BINARY_DIR}/CKLConfig.cmake"
        "${PROJECT_BINARY_DIR}/CKLConfigVersion.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/CKL")

# pkg-config, because the Fortran side of the world consumes through it more
# often than through a CMake package. Written by an install time script so the
# paths follow `cmake --install --prefix` rather than the configure time prefix.
set(CKL_PC_RELATIVE_LIBDIR "${CMAKE_INSTALL_LIBDIR}")
set(CKL_PC_RELATIVE_INCLUDEDIR "${CMAKE_INSTALL_INCLUDEDIR}")
set(CKL_PC_CUDA_LIBDIR "${CUDAToolkit_LIBRARY_DIR}")
configure_file(
    "${PROJECT_SOURCE_DIR}/cmake/install_pkgconfig.cmake.in"
    "${PROJECT_BINARY_DIR}/install_pkgconfig.cmake"
    @ONLY)
install(SCRIPT "${PROJECT_BINARY_DIR}/install_pkgconfig.cmake")
