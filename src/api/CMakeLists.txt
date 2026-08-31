# Two targets, because the dependency runs in two directions.
#
# ckl_core sits below the six family libraries: it owns ckl::Context, the
# process wide default context the v1 free functions route through, the cached
# device capability the launchers gate on, and the status names. Every family
# library links it.
#
# ckl_api sits above them: descriptor validation, the kAuto heuristic, and the C
# ABI, all of which call into the family kernels. Splitting the two is what keeps
# the link graph acyclic, which a shared build requires.

add_library(ckl_core
    context.cpp
    status.cpp
)

target_link_libraries(ckl_core
    PUBLIC ckl_headers
    PRIVATE CUDA::cublas CUDA::cusparse CUDA::cusolver
)

ckl_configure_library(ckl_core core)

add_library(ckl_api
    dispatch.cpp
    c_api.cpp
)

# ckl_core is PUBLIC because the C++ entry points in ckl/gemm.hpp take a
# ckl::Context by reference, so a consumer of ckl_api constructs one itself.
target_link_libraries(ckl_api
    PUBLIC ckl_headers ckl_core
    PRIVATE ckl_gemm ckl_gemv ckl_sparse ckl_trsm CUDA::cublas
)

ckl_configure_library(ckl_api api)
