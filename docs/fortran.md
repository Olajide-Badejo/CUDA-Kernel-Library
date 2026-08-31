# Fortran

Two layers, one archive each.

- `libckl_fortran` is the module layer: module `ckl` binds every symbol in
  `include/ckl/ckl.h`, and module `ckl_blas_core` adds the device pointer BLAS
  routines and the host array engines behind them. Always built when
  `CKL_BUILD_FORTRAN=ON`.
- `libckl_blas` is three symbols and nothing else: `sgemm_`, `sgemv_`, `strsm_`.
  It exists only when `CKL_BLAS_ALIASES=ON`, and it is meant to replace a CPU
  BLAS on the link line, never to sit beside one.

Fortran is column major and so is the reference BLAS, so everything below goes
straight to the CKL column major path. No transposes are synthesised in the
wrappers.

## Build

```
cmake -S . -B build -G Ninja \
  -DCMAKE_CXX_COMPILER=g++-14 -DCMAKE_CUDA_HOST_COMPILER=g++-14 \
  -DCKL_BUILD_FORTRAN=ON -DCKL_BLAS_ALIASES=ON
cmake --build build
```

`enable_language(Fortran)` lives in `fortran/CMakeLists.txt`, so a build that
leaves `CKL_BUILD_FORTRAN` at its default never looks for a Fortran compiler.

## Supported compilers and name mangling

Verified on gfortran 15.2. The ABI is also what nvfortran produces, and
`docs/fortran.md` is the place that record lives:

| compiler  | symbol name | scalar and array arguments | hidden character length |
|-----------|-------------|----------------------------|-------------------------|
| gfortran  | `sgemm_`    | by reference               | `size_t` by value, appended after the whole argument list |
| nvfortran | `sgemm_`    | by reference               | 64 bit integer by value, appended after the whole argument list |

`fortran/ckl_blas.f90` declares those trailing lengths explicitly, so the
`bind(C)` prototype matches the real call frame instead of relying on the
compiler to synthesise them. They are never read: a BLAS option character is
length 1 and the value only has to occupy the right register slot, which is why
the gfortran and nvfortran widths can differ without breaking the call.

**Call side implication.** Pass the option characters exactly as you already do,
`call sgemm('N', 'N', ...)`. Do not pass a `character(len=*)` longer than one,
and do not pass a null terminated C string: the routines read the first
character only.

Only gfortran is tested in CI, because it is the compiler on the machine this
project is built on. An nvfortran build is expected to work from the ABI above
and is not claimed as verified.

## Host arrays versus device pointers

The standard names take **host** arrays and manage the transfers themselves.
That is what an existing Fortran program passes, and handing a host address to a
routine that expects a device pointer is the silent wrong answer the project's
ground rules ban.

| routine | arrays | where the data lives |
|---------|--------|----------------------|
| `sgemm_`, `sgemv_`, `strsm_` | host | copied in, computed on the device, copied back |
| `ckl_sgemm_d`, `ckl_sgemv_d`, `ckl_strsm_d` | device | caller owns residency |
| `ckl_sgemm_arrays` (module `ckl`) | host, assumed shape | copied in and back, dimensions read off the array shapes |

The `_d` routines take an `info` argument and return a `ckl_status_t` through
it. The standard named routines have nowhere to put a status, so a CKL failure
prints the detail from `ckl_last_error` and stops, the same exit the reference
BLAS takes through XERBLA for a bad argument.

`ckl_sgemv_d` accepts unit increments only and returns `CKL_STATUS_NOT_SUPPORTED`
for anything else. Packing a strided vector needs a host round trip, and doing
that quietly under a device pointer name would be a fallback rather than a
result.

## How each routine reaches the GPU

- **SGEMM** maps one to one onto `ckl_sgemm` on the column major path, each
  operand keeping its own op flag.
- **SGEMV** is a GEMM with `n = 1`. Same arithmetic, no special case:
  `y = alpha op(A) x + beta y` is `C = alpha op(A) B + beta C` with B and C single
  columns. Non unit increments are packed into unit stride buffers on the host
  first, which is O(len) work outside the multiply.
- **STRSM** is blocked, with a 64 wide diagonal block. The trailing updates,
  which hold almost all the flops, are GEMMs and run on the device. The diagonal
  blocks are small and inherently sequential, so each block step pulls a 64 by 64
  corner of A and the matching panel of B down to the host, solves there, and
  pushes the panel back. All sixteen combinations of side, uplo, transa and diag
  go through the same code path; `fortran/tests/test_ckl_blas.f90` checks four of
  them against a dense residual.

## Link strategy

`libckl_blas` defines `sgemm_`, `sgemv_` and `strsm_`. Linking it next to a CPU
BLAS is a duplicate symbol error, and that is deliberate: it replaces the CPU
BLAS, it does not augment it.

```
# accelerated
gfortran myprog.f90 -o myprog -lckl_blas -lckl_fortran $(pkg-config --libs ckl)

# NOT this: sgemm_ is defined twice
gfortran myprog.f90 -o myprog -lckl_blas -lckl_fortran -lblas
```

A program that also calls LAPACK is the awkward case, because LAPACK pulls in a
BLAS of its own. That is why the CG example in `fortran/examples/cg_driver.f90`
links LAPACK and the reference BLAS and calls CKL through the module instead of
through the aliases; the alias archive is exercised on its own in
`fortran/tests/test_ckl_blas.f90`.

## Install layout

```
<prefix>/include/ckl/ckl.h              the C ABI
<prefix>/include/ckl/finclude/*.mod     compiled modules for this gfortran
<prefix>/share/ckl/fortran/*.f90        the module sources
<prefix>/lib/libckl_fortran.a
<prefix>/lib/libckl_blas.a              only when CKL_BLAS_ALIASES was on
<prefix>/lib/pkgconfig/ckl.pc
```

`.mod` files are compiler and version specific. On any compiler other than the
one that built the install, recompile the sources under `share/ckl/fortran`; they
are shipped for exactly that reason.

`ckl.pc` carries both include directories and the whole static link order, so a
plain gfortran command line needs nothing else:

```
export PKG_CONFIG_PATH=<prefix>/lib/pkgconfig
gfortran $(pkg-config --cflags ckl) myprog.f90 -o myprog $(pkg-config --libs ckl)
```

## Minimal usage

```fortran
program demo
    use ckl
    implicit none
    integer, parameter :: n = 512
    type(c_ptr) :: h
    integer(c_int) :: rc
    real(c_float) :: a(n, n), b(n, n), c(n, n)

    a = 1.0_c_float
    b = 2.0_c_float
    c = 0.0_c_float

    rc = ckl_create(h)
    call ckl_check(rc, 'ckl_create')
    call ckl_sgemm_arrays(h, CKL_OP_N, CKL_OP_N, 1.0_c_float, a, b, 0.0_c_float, c, rc)
    call ckl_check(rc, 'ckl_sgemm_arrays')
    rc = ckl_destroy(h)

    write (*, *) c(1, 1)
end program demo
```

For the device pointer path, `ckl_alloc_floats`, `ckl_put`, `ckl_put2`,
`ckl_get`, `ckl_get2` and `ckl_free` wrap the four device memory entry points
that Part 09 added to the C ABI (`ckl_device_malloc`, `ckl_device_free`,
`ckl_memcpy`, `ckl_device_synchronize`). A Fortran or C consumer has no CUDA
toolkit headers, so without those four it could not hold an operand at all.

## What runs here

- `fortran_module`: the module layer on the device pointer path, GEMM against a
  double precision Fortran oracle at `8 sqrt(k) eps`.
- `fortran_blas_alias`: SGEMM, SGEMV and STRSM through the standard signatures
  with host arrays, same oracle. Needs `CKL_BLAS_ALIASES=ON`.
- `fortran_cg_example`: conjugate gradient on a synthetic SPD system, validated
  against an LAPACK `SPOTRF` and `SPOTRS` solve, with the Gate E residual
  `norm(Ax - b, inf) / (norm(A, inf) norm(x, inf) n eps) <= 10` printed and
  asserted, and an SGEMV timing comparison against the CPU reference BLAS
  measured at run time.

All three are labelled `gpu` and exit 77, which ctest reads as a skip, when no
CUDA device answers.
