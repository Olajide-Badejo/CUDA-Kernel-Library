# Building

Everything below is what I actually run. Where a requirement looks arbitrary,
the reason it exists is written next to it, because a version floor with no
explanation is a version floor nobody dares to move.

## Requirements

| Component | Minimum | What I build with | Why the floor is where it is |
|---|---|---|---|
| CUDA Toolkit | 12.8 | 13.3.73 | 12.8 is the first release that can emit `sm_120` SASS, which is the default architecture here. An older toolkit fails with a ptxas message about an unknown target that explains nothing, so CMake checks the version itself and says why. |
| CMake | 3.28 | 4.4.0 | `cmake_minimum_required(VERSION 3.28)`. The presets file is schema version 6. |
| Host C++ compiler | not pinned by CMake | **g++-14.3**, verified | See below. The build asks for C++23 on host translation units, C++20 on device code, and C99 for the ABI test and the C example, so the compiler has to have all three. |
| Generator | any | Ninja | Every preset uses it. Make works; it is just slower. |
| GPU | whatever your `CMAKE_CUDA_ARCHITECTURES` names | RTX 5070, sm_120 | The release fat binary carries SASS from `sm_80` up. Individual rungs have their own floors: `cp.async`, WMMA, `mma.sync` and `ldmatrix` all need compute capability 8.0, and the launcher refuses below it rather than running a kernel whose body was compiled out. |
| Driver | one that supports the toolkit | 610.62 | Under WSL2 this is the Windows side driver, not a Linux one. |
| gfortran | optional | 15.2 | Only when `CKL_BUILD_FORTRAN=ON`. See [`fortran.md`](fortran.md). |
| Doxygen | optional | 1.15.0 | Only to regenerate `docs/api`. |

### The host compiler is g++-14, and that is not a preference

The system default on my machine is GCC 15.2, and nvcc 13.3 cannot parse its
libstdc++: the headers use `if consteval`, which the nvcc frontend rejects. The
failure arrives as a wall of errors inside `<bits/...>` with nothing in it that
points at this project. GCC 14.3 is verified working and is what every command
in this document uses:

```sh
-DCMAKE_CXX_COMPILER=g++-14 -DCMAKE_CUDA_HOST_COMPILER=g++-14
```

If you configure without those and get template errors from inside libstdc++,
this is the cause.

## WSL2, which is where this project is developed

The card is in a Windows machine and the toolchain lives in an Ubuntu WSL2
instance. Two things about that setup are worth writing down.

The CUDA toolkit goes in the WSL distribution; the driver stays on Windows. Do
not install a Linux display driver inside WSL. `nvidia-smi` inside WSL reports
the Windows driver version, which is the number to quote in a bug report.

Build out of tree in the Linux filesystem rather than on the mounted `/mnt/c`
path. The 9p mount is slow enough that a full rebuild takes several times
longer:

```sh
cmake -S . -B /tmp/ckl-build -G Ninja \
  -DCMAKE_CXX_COMPILER=g++-14 -DCMAKE_CUDA_HOST_COMPILER=g++-14
cmake --build /tmp/ckl-build
```

Profiling has one more requirement: `ncu` needs GPU performance counter access,
which under WSL2 is a Windows side registry setting
(`RmProfilingAdminOnly = 0`) followed by a reboot. The whole fight is recorded
in [`ENGINEERING_LOG.md`](ENGINEERING_LOG.md).

## The presets

```sh
cmake --preset default        # then: cmake --build --preset default
```

| Preset | Build type | Architectures | For |
|---|---|---|---|
| `default` | Release | `native` | Day to day work on the machine you are sitting at. Binary dir `build/`. |
| `release` | Release | `80-real;86-real;89-real;90-real;120-real;120-virtual` | The shipping fat binary: SASS for Ampere through Blackwell, plus PTX so a future architecture can JIT. Binary dir `out/release/`. |
| `multiarch` | Release | the same list | The release list under `CKL_WERROR=ON`. This is the gate that catches a warning that only appears when compiling for an older architecture. Binary dir `out/multiarch/`. |
| `asan` | Debug | `120` | Host side AddressSanitizer. Device code is not instrumented. The architecture is pinned rather than `native` because CMake's native detection compiles and runs a probe binary, and the sanitizer link flags make that probe fail. `CKL_LINEINFO` is off. Binary dir `out/asan/`. |

The presets do not set the host compiler, so pass `-DCMAKE_CXX_COMPILER=g++-14
-DCMAKE_CUDA_HOST_COMPILER=g++-14` alongside them, or set `CXX` and
`CUDAHOSTCXX` in the environment.

## Options

All of these are plain `cmake -D` cache variables.

| Option | Default | What it does |
|---|---|---|
| `CKL_BUILD_TESTS` | `ON` | Builds the GoogleTest suites and calls `enable_testing()`. GoogleTest 1.17.0 is fetched at configure time, so the first configure needs network access. The tests need a GPU to run. |
| `CKL_BUILD_BENCH` | `ON` | Builds the benchmark binaries under `benchmarks/`. They need a GPU to run. |
| `CKL_BUILD_TOOLS` | `ON` | Builds `ckl_device_probe` and `ckl_roofline` in `tools/`. |
| `CKL_BUILD_EXAMPLES` | `ON` | Builds the four programs in `examples/`. Each one runs to completion and prints `PASS`, and they register as ctest cases labeled `gpu;example`. |
| `CKL_BUILD_FORTRAN` | `OFF` | Builds `libckl_fortran` and the Fortran drivers. `enable_language(Fortran)` lives inside `fortran/CMakeLists.txt`, so leaving this off means the build never looks for a Fortran compiler. |
| `CKL_BLAS_ALIASES` | `OFF` | Adds `libckl_blas`, an archive holding exactly `sgemm_`, `sgemv_` and `strsm_`. It is meant to *replace* a CPU BLAS on a link line, never to sit beside one, so it is never in the default link line. Requires `CKL_BUILD_FORTRAN=ON`; see [`fortran.md`](fortran.md). |
| `CKL_WERROR` | `ON` | Warnings are errors on device code (`--Werror=all-warnings`). Local builds keep this on as a phase gate. CI turns it off because it compiles `sm_120` with a different host GCC whose warning set differs, and cross toolchain warning noise is not a reason to fail a build. |
| `CKL_LINEINFO` | `ON` | Emits `-lineinfo` for CUDA sources in Release builds so `ncu` can attribute stalls to source lines without paying the `-G` debug penalty. It is a private per target flag, not a global `CMAKE_CUDA_FLAGS` append, so it does not leak into a tree that embeds this one. |
| `BUILD_SHARED_LIBS` | `OFF` | Standard CMake. `ON` builds the six family libraries plus `ckl_core` and `ckl_api` as shared objects with hidden visibility and a soname; `OFF` builds static archives, and `ckl_headers` then defines `CKL_STATIC_DEFINE` for consumers. |
| `CMAKE_CUDA_ARCHITECTURES` | `120` | Set before `project()` so `enable_language(CUDA)` picks it up. `project()` would otherwise seed a default that a later `set()` cannot dislodge. |

## Make targets

The `Makefile` is a thin front end over the CMake build, meant to be run inside
WSL. `make setup` followed by `make all` is the reproduction path from a clean
clone.

| Target | What it does |
|---|---|
| `setup`, `configure` | Configure into `build/` (`BUILD_DIR`, `BUILD_TYPE` and `GENERATOR` are overridable). |
| `build` | Configure then compile every target. Zero warnings is a gate. |
| `test` | Build then `ctest --output-on-failure`. Needs a GPU. |
| `bench` | Run the default GEMM benchmark. |
| `roofline` | Measure the hardware ceilings with `ckl_roofline` and render the figure. |
| `sweep`, `sweep-quick` | The full resumable benchmark sweep, refreshing `experiments/results/summary.csv`. `sweep` locks the graphics clock first and refuses to run if it cannot. |
| `report` | Regenerate every figure and table from the committed results and build both PDFs. Does not need a GPU or a toolkit, so it runs in CI. |
| `check-style` | The dash gate and the provenance gate. |
| `all` | `build`, `test`, `sweep`, `report`, `check-style`. |
| `clean` | Remove the build tree and the LaTeX output. |

## Installing

```sh
cmake --install /tmp/ckl-build --prefix /usr/local
```

That writes the headers, the archives or shared objects, a CMake package
(`lib/cmake/CKL/`) and a pkg-config file (`lib/pkgconfig/ckl.pc`). Consuming any
of those three ways is [`using.md`](using.md).

## Regenerating the API documentation

```sh
doxygen Doxyfile
```

Output lands in `docs/api/`, which is gitignored: the headers are the source of
truth and the render is rebuilt on demand. The run has to finish with zero
warnings. `WARN_IF_UNDOCUMENTED` is on and `EXTRACT_ALL` is off, so a public
declaration added to `include/ckl/` without a doc comment shows up as a warning
rather than silently missing from the site.
