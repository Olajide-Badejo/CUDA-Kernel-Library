# Contributing

This is a single author project (Olajide Badejo), but the workflow is written down
so a clean clone reproduces everything and so the checks are unambiguous.

## What needs real hardware

The correctness tests, the benchmarks, the diagnostic rounds, and the sweep all
run on the GPU and need an NVIDIA card with a driver for CUDA 13.3 (this repo
targets one RTX 5070, sm_120). CI on GitHub compiles `sm_120` without a GPU and
runs only the host checks; anything labeled `gpu` in ctest is skipped there. To run
the GPU tests: `ctest --test-dir build` on a machine with the card.

For the diagnostic rounds, `ncu` needs GPU performance counter access. Under WSL2
that is a Windows side setting (`RmProfilingAdminOnly = 0`, then reboot); see
`docs/ENGINEERING_LOG.md`.

## Build and check

The host compiler is g++-14, not the system default. GCC 15's libstdc++ uses
`if consteval`, which the nvcc 13.3 frontend cannot parse, and the failure is a
wall of errors inside `<bits/...>` that points at nothing in this project. The
full toolchain story, the presets and every option are in `docs/building.md`.

```sh
make setup       # configure into build/
make build       # compile, zero warnings is a gate
make test        # GPU correctness tests
make check-style # dash gate and provenance gate
make sweep       # full benchmark sweep, refreshes summary.csv
make roofline    # roofline data and figure
make report      # regenerate figures and tables, build both PDFs
make all         # build, test, sweep, report, check-style
```

Configuring by hand rather than through the Makefile, which is what the presets
and an out of tree build want:

```sh
cmake --preset default \
  -DCMAKE_CXX_COMPILER=g++-14 -DCMAKE_CUDA_HOST_COMPILER=g++-14
cmake --build --preset default
```

The presets are `default` (native architecture), `release` and `multiarch` (the
full fat binary list, `multiarch` with warnings as errors), and `asan` (host side
AddressSanitizer). Options worth knowing: `CKL_BUILD_TESTS`, `CKL_BUILD_BENCH`,
`CKL_BUILD_TOOLS`, `CKL_BUILD_EXAMPLES`, `CKL_BUILD_FORTRAN`, `CKL_BLAS_ALIASES`,
`CKL_WERROR`, `CKL_LINEINFO`.

If you touched a public header, `doxygen Doxyfile` has to finish with zero
warnings. `WARN_IF_UNDOCUMENTED` is on and `EXTRACT_ALL` is off, so an
undocumented public declaration is a visible failure rather than a silent gap.

## The gates

Three of them, all runnable locally, none with `continue-on-error`.

- **Dash gate.** No en dash and no em dash anywhere, in any file, in any
  language: `python3 scripts/check_no_dashes.py .`. Write a range as "256 to
  8192". For `.tex` files it also flags literal `--` and `---` in prose, since
  those typeset as dashes. Enforced in CI and again inside the report build.
- **Provenance gate.** `python3 scripts/check_provenance.py`. Every commit hash
  recorded in a results file has to resolve in git, or be listed in
  `experiments/results/legacy_hashes.txt` as a known dead v1 hash.
- **Zero warnings.** `CKL_WERROR` is on locally, so a warning fails the build.

## Every result carries a commit, and the exceptions are written down

Every performance number comes from a run on this hardware and is traceable to a
results file and a commit that produced it. No number without a run: an estimate
is labeled a hypothesis, an unmeasured value reads "pending", and nothing is hand
copied into a report.

That rule has a known standing breach, and it is recorded rather than papered
over. The v1 results in `experiments/results/summary.csv` carry commit hashes
that no longer resolve in this repository, because the history was rewritten
after they were measured. They are listed in
`experiments/results/legacy_hashes.txt` so the provenance gate can still be a
real gate, and the whole defect, along with everything else I found auditing my
own published claims, is in [`docs/CORRECTIONS.md`](docs/CORRECTIONS.md). Read
that file before quoting a percent from this repository.

## Style

- No en dash or em dash anywhere (`scripts/check_no_dashes.py`, enforced in CI and
  the report build).
- C++ and CUDA: `snake_case` functions and files, `PascalCase` types, `UPPER_SNAKE`
  constants and template tile parameters, trailing underscore members; kernels
  named `<op>_<variant>`; `.cu`/`.cuh` device, `.cpp`/`.hpp` host. Formatted with
  the committed `.clang-format`, checked by `.clang-tidy`. `.editorconfig` carries
  the same whitespace rules for editors that read neither.
- Public headers under `include/ckl/` carry Doxygen tags: `@brief` on everything,
  `@param` and `@return` on every function, and `@note` for a contract a caller
  can violate.
- Python: `snake_case`, ruff clean against the committed `ruff.toml`.
- Comments explain what the code cannot say on its own. No TODOs and no stubs in
  shipped files.
- A silent fallback is a bug: either the fast path runs, or a status comes back,
  or `chosen` reports the path that was taken.
