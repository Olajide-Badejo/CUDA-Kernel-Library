# Correctness tests. These need a GPU to run; CI marks them hardware only and
# skips them in the cloud (Section 15). ctest runs them locally.
#
# support/ holds reference.hpp, which used to sit in the public include tree. It
# is test scaffolding, so it is not installed and not visible to consumers.
# tests/consume/ is a standalone project that builds against an install prefix;
# it is deliberately not added here.

add_library(ckl_test_support INTERFACE)
target_include_directories(ckl_test_support INTERFACE "${CMAKE_CURRENT_LIST_DIR}/support")

add_executable(test_gemm test_gemm.cpp)
target_link_libraries(test_gemm PRIVATE ckl_api ckl_gemm ckl_headers ckl_test_support)
ckl_set_warnings(test_gemm)
add_test(NAME gemm_correctness COMMAND test_gemm)
set_tests_properties(gemm_correctness PROPERTIES LABELS "gpu")

# The C ABI test is two translation units on purpose. test_c_abi.c is C99 and
# includes no CUDA header, which is the property being tested; the device memory
# it needs comes from a C++ shim that exposes a plain C interface.
if(TARGET ckl_api)
    add_executable(test_c_abi test_c_abi.c test_c_abi_support.cpp)
    set_source_files_properties(test_c_abi.c PROPERTIES LANGUAGE C)
    set_target_properties(test_c_abi PROPERTIES C_STANDARD 99 C_STANDARD_REQUIRED ON)
    target_include_directories(test_c_abi PRIVATE "${CMAKE_CURRENT_LIST_DIR}")
    target_link_libraries(test_c_abi PRIVATE ckl_api ckl_headers CUDA::cudart)
    target_compile_options(test_c_abi PRIVATE
        $<$<COMPILE_LANGUAGE:C>:-Wall -Wextra -Wpedantic -Wshadow -Wconversion>)
    ckl_set_warnings(test_c_abi)
    add_test(NAME c_abi COMMAND test_c_abi)
    set_tests_properties(c_abi PROPERTIES LABELS "gpu")
endif()

add_executable(test_gemm_tensor test_gemm_tensor.cpp)
target_link_libraries(test_gemm_tensor PRIVATE ckl_gemm ckl_headers ckl_test_support)
ckl_set_warnings(test_gemm_tensor)
add_test(NAME gemm_tensor_correctness COMMAND test_gemm_tensor)
set_tests_properties(gemm_tensor_correctness PROPERTIES LABELS "gpu")

if(TARGET ckl_gemv)
    add_executable(test_gemv test_gemv.cpp)
    target_link_libraries(test_gemv PRIVATE ckl_gemv ckl_headers ckl_test_support)
    ckl_set_warnings(test_gemv)
    add_test(NAME gemv_correctness COMMAND test_gemv)
    set_tests_properties(gemv_correctness PROPERTIES LABELS "gpu")
endif()

if(TARGET ckl_sparse)
    add_executable(test_spmv test_spmv.cpp)
    target_link_libraries(test_spmv PRIVATE ckl_sparse ckl_headers ckl_test_support)
    ckl_set_warnings(test_spmv)
    add_test(NAME spmv_correctness COMMAND test_spmv)
    set_tests_properties(spmv_correctness PROPERTIES LABELS "gpu")
endif()

if(TARGET ckl_trsm)
    add_executable(test_trsm test_trsm.cpp)
    target_link_libraries(test_trsm PRIVATE ckl_trsm ckl_headers ckl_test_support)
    ckl_set_warnings(test_trsm)
    add_test(NAME trsm_correctness COMMAND test_trsm)
    set_tests_properties(trsm_correctness PROPERTIES LABELS "gpu")
endif()

if(TARGET ckl_solver)
    add_executable(test_solver test_solver.cpp)
    target_link_libraries(test_solver PRIVATE ckl_solver ckl_headers ckl_test_support)
    ckl_set_warnings(test_solver)
    add_test(NAME solver_residual COMMAND test_solver)
    set_tests_properties(solver_residual PROPERTIES LABELS "gpu")
endif()

if(TARGET ckl_telemetry)
    add_executable(test_nvml test_nvml.cpp)
    target_link_libraries(test_nvml PRIVATE
        ckl_telemetry ckl_gemm ckl_headers ckl_test_support)
    ckl_set_warnings(test_nvml)
    add_test(NAME nvml_monitor COMMAND test_nvml)
    set_tests_properties(nvml_monitor PROPERTIES LABELS "gpu")
endif()
