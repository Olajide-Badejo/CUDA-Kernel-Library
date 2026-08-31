# cuSOLVER dense solver wrapper.

add_library(ckl_solver
    dense_solver.cpp
)

target_link_libraries(ckl_solver
    PUBLIC ckl_headers
    PRIVATE CUDA::cusolver
)

ckl_configure_library(ckl_solver solver)
