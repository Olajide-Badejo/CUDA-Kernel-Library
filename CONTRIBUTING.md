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
`PROGRESS.md`.

## Build and check

```sh
make setup       # configure
make build       # compile, zero warnings is a gate
make test        # GPU correctness tests
make check-style # dash gate (clang-format, clang-tidy, ruff in CI)
make sweep       # full benchmark sweep, refreshes summary.csv
make roofline    # roofline data and figure
```

## Style

- No en dash or em dash anywhere (`scripts/check_no_dashes.py`, enforced in CI and
  the report build).
- C++ and CUDA: `snake_case` functions and files, `PascalCase` types, `UPPER_SNAKE`
  constants and template tile parameters, trailing underscore members; kernels
  named `<op>_<variant>`; `.cu`/`.cuh` device, `.cpp`/`.hpp` host. Formatted with
  the committed `.clang-format`, checked by `.clang-tidy`.
- Python: `snake_case`, ruff clean against the committed `ruff.toml`.
- Every performance number comes from a run on this machine and is traceable to a
  results file and a commit. No number without a run.
