# FFT and convolution family library.
#
# fft_plan.cpp is host only: it owns the twiddle tables, the ping pong workspace,
# the four step factorization, the cuFFT handle and the cached filter spectrum,
# and it calls the launchers in the .cu files rather than containing a kernel of
# its own. cufft_ref.cpp is the free function face of the vendor baseline and
# holds the one entry plan cache that keeps it from planning per call.

add_library(ckl_fft
    fft_radix2.cu
    fft_radix4.cu
    fft_shared.cu
    fft_transpose.cu
    conv_fft.cu
    conv_direct.cu
    fft_plan.cpp
    cufft_ref.cpp
)

target_link_libraries(ckl_fft
    PUBLIC ckl_headers
    PRIVATE ckl_core CUDA::cufft
)

# fft_internal.hpp and fft_device.cuh sit next to the sources and are found
# relative to them; nothing outside src/fft includes either, except src/api,
# which puts src/ on its own private include path for fft_internal.hpp.

ckl_configure_library(ckl_fft fft)
