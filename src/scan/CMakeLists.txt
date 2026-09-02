# Reduction and prefix scan family library.
#
# scan_plan.cpp is host only: it owns the partials, the level buffers, the tile
# status array, the two tile counters and the CUB temporary storage, and it turns
# the two runtime enums into the template arguments the rungs are written over.
# It calls the launchers in the .cu files rather than containing a kernel of its
# own. cub_ref.cu is the vendor baseline and the only file that includes CUB.
#
# CUB ships inside the toolkit as part of CCCL and needs no link library, so this
# target links nothing the other families do not.

add_library(ckl_scan
    reduce_atomic.cu
    reduce_shared_tree.cu
    reduce_shuffle.cu
    reduce_vec4.cu
    reduce_single_pass.cu
    reduce_deterministic.cu
    reduce_kahan.cu
    scan_hillis_steele.cu
    scan_blelloch.cu
    scan_three_kernel.cu
    scan_reduce_then_scan.cu
    scan_lookback.cu
    cub_ref.cu
    scan_plan.cpp
)

target_link_libraries(ckl_scan
    PUBLIC ckl_headers
    PRIVATE ckl_core
)

# scan_internal.hpp, scan_rungs.hpp, scan_device.cuh and scan_levels.cuh sit next
# to the sources and are found relative to them; nothing outside src/scan
# includes any of them, except src/api, which puts src/ on its own private
# include path for scan_internal.hpp.

ckl_configure_library(ckl_scan scan)
