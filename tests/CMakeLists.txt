# Correctness tests. These need a GPU to run; CI marks them hardware only and
# skips them in the cloud (Section 15). ctest runs them locally.

add_executable(test_gemm test_gemm.cpp)
target_link_libraries(test_gemm PRIVATE ckl_gemm ckl_headers)
ckl_set_warnings(test_gemm)
add_test(NAME gemm_correctness COMMAND test_gemm)
set_tests_properties(gemm_correctness PROPERTIES LABELS "gpu")

add_executable(test_gemm_tensor test_gemm_tensor.cpp)
target_link_libraries(test_gemm_tensor PRIVATE ckl_gemm ckl_headers)
ckl_set_warnings(test_gemm_tensor)
add_test(NAME gemm_tensor_correctness COMMAND test_gemm_tensor)
set_tests_properties(gemm_tensor_correctness PROPERTIES LABELS "gpu")
