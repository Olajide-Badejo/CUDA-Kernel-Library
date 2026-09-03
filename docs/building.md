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
| `CKL_WERROR` | `ON` | Warnings are errors on device code (`--Werror=all-warnings`). On here, on in every build matrix cell, and on in the GPU job. The v1 CI turned it off on the grounds that the container's host GCC differed; it does differ (13.3 in the container against 14.3 here) and the tree compiles clean under both, so the exemption was not needed and is gone. |
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
| `report` | Regenerate every figure and table from the committed results and build both PDFs. Does not need a GPU or a toolkit, so it runs in CI. Exports `SOURCE_DATE_EPOCH` so matplotlib does not stamp the wall clock into the PDFs. |
| `format` | Rewrite every source file the way clang-format wants it. |
| `format-check` | clang-format, read only, non zero on the first difference. |
| `tidy` | Configure a scanning free analysis tree and run clang-tidy over the host translation units. |
| `doxygen` | Build the API documentation. Zero warnings, because the Doxyfile sets `WARN_AS_ERROR`. |
| `perf-regression` | Tonight's quick sweep against `experiments/results/perf_baseline.csv`. Needs a GPU. |
| `check-style`, `style` | The whole style gate: format, ruff, yamllint, dashes, provenance, clang-tidy. Exactly what the CI style job runs. |
| `all` | `build`, `test`, `sweep`, `report`, `check-style`. |
| `clean` | Remove the build tree and the LaTeX output. |

The style tools are pinned, and `check-style` uses whatever `clang-format` and
`clang-tidy` are on `PATH`, so point it at the pinned ones:

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
make check-style CLANG_FORMAT=.venv/bin/clang-format \
                 CLANG_TIDY=.venv/bin/clang-tidy \
                 RUFF=.venv/bin/ruff YAMLLINT=.venv/bin/yamllint
```

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

## Continuous integration

Three workflow files. `ci.yml` holds everything that can be checked without a
GPU and runs on every push and pull request; `gpu.yml` and
`perf-regression.yml` need a card in the machine and run on a self hosted
runner.

| Job | Where | When | What has to hold |
|---|---|---|---|
| `style` | CUDA 13.3.1 container | every push | clang-format clean tree wide, ruff, yamllint, the dash gate, the provenance gate, clang-tidy with a non empty `WarningsAsErrors` |
| `build-matrix` | CUDA 12.8.1 and 13.3.1 containers, five architectures each | every push | every cell compiles with `CKL_WERROR=ON` |
| `install-consume` | CUDA 13.3.1 container | every push | install to a prefix, then `tests/consume/` through `find_package`, `examples/` standalone, and `pkg-config --modversion ckl` |
| `sass-diff` | CUDA 13.3.1 container | every push | the top kernel's instruction stream matches `experiments/sass/golden/mma_opt.sass` |
| `report` | Ubuntu 24.04 container with TeX Live | every push | `make report` reproduces `report/figures` and `report/tables` byte for byte, both PDFs compile and are uploaded |
| `docs` | `ubuntu-latest` | push to main | Doxygen with zero warnings, published to Pages |
| `gpu` | self hosted, `[self-hosted, gpu]` | push to main, manual | `ctest -L gpu`, then memcheck, racecheck, initcheck and synccheck on both GEMM suites |
| `perf-regression` | self hosted, `[self-hosted, gpu]` | nightly at 04:00 UTC, manual | tonight's quick sweep is not more than 5 percent slower than the committed baseline on any row where the bootstrap intervals also separate |

No step anywhere carries `continue-on-error`. The v1 workflow marked
clang-format best effort because its output moves between major versions, which
made the only formatting check in the repository unable to fail. The version is
pinned instead.

### What is pinned, and why each one

| Pin | Value | Why |
|---|---|---|
| clang-format | `20.1.7`, PyPI wheel | 13 files that clang-format 19 accepts, 20.1.7 rewrites. Without a pin the gate is whatever the runner image happens to ship. |
| clang-tidy | `20.1.0`, PyPI wheel | Same reason: the check set and its diagnostics move between releases. |
| ruff | `0.15.22` | New rules arrive in new releases and would fail a tree nobody changed. |
| yamllint | `1.35.1` | Same. |
| matplotlib, numpy | `3.10.7`, `2.3.5` | The report job diffs the figures byte for byte, and matplotlib writes its own rasteriser output into the PNG. Debian's `3.10.7+dfsg1` and PyPI's `3.10.7` disagree on 3068 bytes of `ladder.png`. |
| CUDA images | `nvidia/cuda:12.8.1-devel-ubuntu24.04` and `13.3.1-devel-ubuntu24.04`, by digest | A tag is rebuilt whenever a base package changes. The SASS golden was assembled by nvcc 13.3.73, which is what the 13.3.1 image carries, and a rebuild that moved ptxas would turn that gate red for no reason of ours. |
| Ubuntu image | `ubuntu:24.04`, by digest | Pins TeX Live to the 24.04 packages. |
| Actions | full commit SHA | A tag on a third party action is a name its owner can move. |

`Dockerfile` at the repository root is the same 13.3.1 image with the build
tools added. It is what to use for regenerating the SASS golden or for
reproducing a matrix cell locally:

```sh
docker build -t ckl-build .
docker run --rm -v "$PWD":/src -w /src ckl-build \
    cmake -S . -B build-container -G Ninja -DCMAKE_CUDA_ARCHITECTURES=120-real
docker run --rm -v "$PWD":/src -w /src ckl-build cmake --build build-container
```

### The 12.8 cells build less than the 13.3 cells

CUDA 12.8 is the toolkit floor because it is the first release that emits
`sm_120` SASS, and the root `CMakeLists.txt` refuses anything older. The
library, the tools, the tests and the examples all compile there with warnings
as errors. The benchmarks do not: `benchmarks/bench_all.cpp` names
`CUBLAS_GEMM_AUTOTUNE`, which cuBLAS added in CUDA 13, and 12.8 answers

```text
error: 'CUBLAS_GEMM_AUTOTUNE' was not declared in this scope
```

So the 12.8 cells configure with `-DCKL_BUILD_BENCH=OFF`. What ships builds on
the floor; the measurement driver needs 13.

### The GPU runner does not exist yet

`gpu.yml` and `perf-regression.yml` both ask for `runs-on: [self-hosted, gpu]`.
Nothing answers to that today, so both queue and eventually expire. That is the
truthful state and it is not disguised: neither workflow has a fallback to a
hosted runner, and nothing in the checks list turns green on the strength of a
job that did not run.

To register the machine with the RTX 5070 in it:

1. On GitHub: repository, Settings, Actions, Runners, New self hosted runner,
   Linux x64. The page hands out a download URL and a registration token; the
   token expires in an hour.
2. On the machine, inside WSL2 (the toolkit and the driver are reachable there):

   ```sh
   mkdir -p ~/actions-runner && cd ~/actions-runner
   curl -o runner.tar.gz -L <the URL the page gives>
   tar xzf runner.tar.gz
   ./config.sh --url https://github.com/<owner>/cuda-kernel-lab \
       --token <the token the page gives> \
       --labels gpu --name ckl-rtx5070 --unattended
   ```

   The `gpu` label is what the two workflows match on. `self-hosted` is added
   automatically.
3. Run it. As a foreground process, `./run.sh`. As a service that survives a
   reboot, `sudo ./svc.sh install && sudo ./svc.sh start`, which needs systemd
   in WSL (`systemd=true` under `[boot]` in `/etc/wsl.conf`).
4. Check that the runner sees the card: `nvidia-smi` and `compute-sanitizer
   --version` both have to work as the user the service runs as. The sanitizer
   comes with the toolkit; if `compute-sanitizer` is not on `PATH`, add
   `/usr/local/cuda/bin`.
5. A self hosted runner executes whatever a pull request contains. Keep this one
   on `pull_request` from branches in the repository only, which is what the two
   workflows do: `gpu.yml` triggers on push to main and manual dispatch, and
   `perf-regression.yml` on a schedule and manual dispatch. Neither runs on a
   fork's pull request.

### The nightly comparison is skipped until a baseline is committed

`perf-regression.yml` has a first job that looks for
`experiments/results/perf_baseline.csv` and a second that only runs when the
first found it. Until the owner commits a sweep summary as that file, the
comparison job shows as skipped. Skipped is not passed, and that distinction is
the point: a green job with no comparison in it would be worse than no job.

The baseline is a `benchmarks/sweep.py` summary from a run at locked clocks.
Once it is on file, the comparison can be rehearsed by hand:

```sh
python3 scripts/perf_regression.py --baseline experiments/results/perf_baseline.csv \
                                   --current /tmp/tonight.csv
```

A row fails only when both halves are true: the median is more than 5 percent
slower, and tonight's `ci95_lo_ms` is above the baseline's `ci95_hi_ms`. Either
half alone produces a job that cries wolf.

### Pages

The `docs` job publishes Doxygen output to GitHub Pages on every push to main.
It needs Pages switched on once: repository, Settings, Pages, Source, GitHub
Actions. Until that is done the deploy step fails, which is the right failure:
the job is not pretending to have published anything.
