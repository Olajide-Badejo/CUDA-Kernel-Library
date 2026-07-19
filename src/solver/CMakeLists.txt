# cuSOLVER dense solver wrapper.

add_library(ckl_solver STATIC
    dense_solver.cpp
)

target_link_libraries(ckl_solver
    PUBLIC ckl_headers
    PRIVATE CUDA::cusolver
)

set_target_properties(ckl_solver PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

ckl_set_warnings(ckl_solver)
