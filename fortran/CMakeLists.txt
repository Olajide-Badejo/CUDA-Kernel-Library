# The Fortran interface. enable_language(Fortran) lives here rather than in the
# root project() call so a build that does not ask for CKL_BUILD_FORTRAN never
# looks for a Fortran compiler.

enable_language(Fortran)

# .mod files are compiler and version specific, so they get their own directory
# and the .f90 sources are installed next to them. A consumer on a different
# gfortran recompiles the sources; a consumer on this one uses the .mod.
set(CKL_FORTRAN_MODULE_DIR "${CMAKE_CURRENT_BINARY_DIR}/finclude")

option(CKL_BLAS_ALIASES
    "Build libckl_blas, the archive exporting sgemm_, sgemv_ and strsm_. \
Link it INSTEAD of a CPU BLAS, never alongside one."
    OFF)

# ---------------------------------------------------------------------------
# ckl_fortran: the module layer. Always built when CKL_BUILD_FORTRAN is on.
# ---------------------------------------------------------------------------
add_library(ckl_fortran
    ckl_mod.f90
    ckl_blas_core.f90
)

set_target_properties(ckl_fortran PROPERTIES
    Fortran_MODULE_DIRECTORY "${CKL_FORTRAN_MODULE_DIR}")

target_include_directories(ckl_fortran PUBLIC
    "$<BUILD_INTERFACE:${CKL_FORTRAN_MODULE_DIR}>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ckl/finclude>")

# ckl_api carries the C entry points the module binds; cudart comes along
# because ckl_api needs the runtime at link time and a gfortran driver has no
# idea where it lives.
target_link_libraries(ckl_fortran PUBLIC ckl_api CUDA::cudart)

ckl_configure_library(ckl_fortran fortran)

# ---------------------------------------------------------------------------
# ckl_blas: the standard mangled aliases, on their own, behind an option.
# ---------------------------------------------------------------------------
if(CKL_BLAS_ALIASES)
    add_library(ckl_blas ckl_blas.f90)
    set_target_properties(ckl_blas PROPERTIES
        Fortran_MODULE_DIRECTORY "${CKL_FORTRAN_MODULE_DIR}")
    target_link_libraries(ckl_blas PUBLIC ckl_fortran)
    ckl_configure_library(ckl_blas blas)
endif()

# ---------------------------------------------------------------------------
# Install: .mod files for this compiler, .f90 sources for every other one.
# ---------------------------------------------------------------------------
install(DIRECTORY "${CKL_FORTRAN_MODULE_DIR}/"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/ckl/finclude"
    FILES_MATCHING PATTERN "*.mod")

install(FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/ckl_mod.f90"
        "${CMAKE_CURRENT_SOURCE_DIR}/ckl_blas_core.f90"
        "${CMAKE_CURRENT_SOURCE_DIR}/ckl_blas.f90"
    DESTINATION "${CMAKE_INSTALL_DATADIR}/ckl/fortran")

# Reaches cmake/ckl.pc.in through the root's install time pkg-config script,
# which is configured after this directory is processed. ckl_blas is left out on
# purpose: it must never appear on a default link line.
set(CKL_PC_FORTRAN_LIBS " -lckl_fortran" PARENT_SCOPE)

# ---------------------------------------------------------------------------
# Tests and the example driver
# ---------------------------------------------------------------------------
# Both drivers exit 77 when no CUDA device answers, which ctest reads as a skip
# rather than a failure.
if(CKL_BUILD_TESTS)
    add_executable(test_fortran_module tests/test_ckl_mod.f90)
    target_link_libraries(test_fortran_module PRIVATE ckl_fortran)
    add_test(NAME fortran_module COMMAND test_fortran_module)
    set_tests_properties(fortran_module PROPERTIES
        LABELS "gpu" SKIP_RETURN_CODE 77)

    # Gate E's alias test: the standard BLAS signature with host arrays. It only
    # exists when the alias archive does, because that archive is what defines
    # the symbol it calls.
    if(CKL_BLAS_ALIASES)
        add_executable(test_fortran_blas tests/test_ckl_blas.f90)
        target_link_libraries(test_fortran_blas PRIVATE ckl_blas ckl_fortran)
        add_test(NAME fortran_blas_alias COMMAND test_fortran_blas)
        set_tests_properties(fortran_blas_alias PROPERTIES
            LABELS "gpu" SKIP_RETURN_CODE 77)
    endif()
endif()

# The conjugate gradient driver validates against LAPACK and times against the
# CPU BLAS, so it needs both. Without them the example is simply not built; the
# module and the tests do not depend on it.
find_library(CKL_LAPACK_LIBRARY NAMES lapack)
find_library(CKL_BLAS_LIBRARY NAMES blas)

if(CKL_LAPACK_LIBRARY AND CKL_BLAS_LIBRARY)
    add_executable(cg_driver examples/cg_driver.f90)
    target_link_libraries(cg_driver PRIVATE
        ckl_fortran "${CKL_LAPACK_LIBRARY}" "${CKL_BLAS_LIBRARY}")

    if(CKL_BUILD_TESTS)
        add_test(NAME fortran_cg_example COMMAND cg_driver)
        set_tests_properties(fortran_cg_example PROPERTIES
            LABELS "gpu" SKIP_RETURN_CODE 77)
    endif()
else()
    message(STATUS
        "ckl: LAPACK or the reference BLAS was not found, so the Fortran CG "
        "example is skipped. Install liblapack-dev and libblas-dev to build it.")
endif()
