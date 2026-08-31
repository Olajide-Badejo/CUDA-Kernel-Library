# The examples build two ways, and both are checked.
#
#   In tree:     configure the root project with -DCKL_BUILD_EXAMPLES=ON.
#   Standalone:  install the project, then point this directory at the prefix.
#
#     cmake --install <build> --prefix /tmp/ckl-prefix
#     cmake -S examples -B /tmp/ckl-examples -DCMAKE_PREFIX_PATH=/tmp/ckl-prefix
#     cmake --build /tmp/ckl-examples
#
# The standalone path is the one that matters to a stranger: it proves the
# examples compile against nothing but the installed headers and the exported
# ckl::ckl target, with no access to the source tree.

if(NOT DEFINED PROJECT_NAME)
    cmake_minimum_required(VERSION 3.28)
    # C is enabled for 03_c_api.c, which has to compile as C99 rather than as
    # C++ that happens to parse.
    project(ckl_examples LANGUAGES C CXX)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_C_STANDARD 99)
    set(CMAKE_C_STANDARD_REQUIRED ON)
    find_package(CKL REQUIRED)
endif()

set(CKL_EXAMPLE_CXX
    01_gemm_minimal
    02_gemm_strided_batched
    04_custom_stream_and_workspace)

foreach(name IN LISTS CKL_EXAMPLE_CXX)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE ckl::ckl)
    if(COMMAND ckl_set_warnings)
        ckl_set_warnings(${name})
    endif()
endforeach()

# The C example links the same target. ckl.h pulls in no CUDA header, so this
# translation unit never sees the toolkit include directories it would need if
# it did. libm comes in explicitly for fabsf: a C link line does not get it for
# free the way a C++ one does.
add_executable(03_c_api 03_c_api.c)
target_link_libraries(03_c_api PRIVATE ckl::ckl)
if(UNIX)
    target_link_libraries(03_c_api PRIVATE m)
endif()
if(COMMAND ckl_set_warnings)
    ckl_set_warnings(03_c_api)
endif()

list(APPEND CKL_EXAMPLE_ALL ${CKL_EXAMPLE_CXX} 03_c_api)

foreach(name IN LISTS CKL_EXAMPLE_ALL)
    set_target_properties(${name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
endforeach()

# Every example runs to completion and prints PASS, so they are cheap end to end
# tests of the public surface. They need a GPU, hence the gpu label.
if(BUILD_TESTING OR CKL_BUILD_TESTS)
    foreach(name IN LISTS CKL_EXAMPLE_ALL)
        add_test(NAME example.${name} COMMAND ${name})
        set_tests_properties(example.${name} PROPERTIES
            LABELS "gpu;example"
            PASS_REGULAR_EXPRESSION "PASS")
    endforeach()
endif()
