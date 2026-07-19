# Warning flags applied to every target. Zero warnings is a phase gate
# (Section 11), so these are added as PRIVATE compile options on each target
# through ckl_set_warnings(). Host C++ and nvcc host passthrough are handled
# separately because nvcc forwards host flags via -Xcompiler.

function(ckl_set_warnings target)
    # Pure host C++ gets the full pedantic set. The CUDA host passthrough drops
    # -Wpedantic on purpose: nvcc with separable compilation emits generated
    # .stub.c files using GCC-style line directives that -Wpedantic rejects, and
    # those files are not ours to fix. -Wall -Wextra still apply to our device
    # translation units through -Xcompiler, and --Werror=all-warnings makes any
    # real nvcc warning fatal.
    target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic -Wshadow -Wconversion>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>
    )
    # Warnings as errors on device code, gated so CI can compile with a different
    # toolchain without the zero warning gate tripping on cross compiler noise.
    if(CKL_WERROR)
        target_compile_options(${target} PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--Werror=all-warnings>
        )
    endif()
endfunction()
