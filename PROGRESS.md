# PROGRESS

One entry per phase: what got built, what was verified with real output, what is
still open. Kept current from the first commit per Section 0 of the spec.

## Phase 0: toolchain, GPU probe, dash lint, skeleton

Status: in progress. Repo skeleton, build system, dash checker, and device probe
are built and verified on the target machine. One blocker remains (ncu counter
permission) that needs a Windows-side action before the Phase 2/3 diagnostic
rounds; it does not block Phases 1 and 2 kernel work.

### Verified on the machine

- GPU probe built and run. Measured ground truth used everywhere the roofline
  needs a ceiling, replacing the datasheet:
  - RTX 5070, compute capability 12.0 (sm_120), 48 SMs, 12227 MiB.
  - Memory: 192 bit bus, 14001 MHz, theoretical 672.0 GB/s (matches datasheet).
  - Measured stream copy bandwidth: about 540 GB/s (roughly 80 percent of
    theoretical, typical for a plain streaming copy).
  - L2 cache 48 MiB, shared memory 100 KiB per SM (48 KiB per block default),
    65536 registers per SM, 1536 max threads per SM.
  - Peak FP32 estimate about 32.3 TFLOP/s (non tensor core).
- CMake configure and Ninja build clean, targeting sm_120.
- Dash checker (`scripts/check_no_dashes.py`) runs clean over the tree.

### Toolchain, with substitutions against the spec floors

| Component | Spec floor | Installed | Action |
|---|---|---|---|
| CUDA Toolkit | 13.3 | 13.3.73 | matches |
| GPU driver (CUDA UMD) | current | 610.62 / CUDA 13.3 | matches |
| GCC / G++ | 16.1 | 15.2.0 | 16.1 not yet released; 15.2 is newest available, recorded here |
| CMake | 4.4 | 4.2.3 | 4.2.3 configures and builds cleanly; upgrade only if a 4.4 feature is needed |
| CUDA C++ device standard | newest accepted, target 23 | C++20 | nvcc 13.3 accepts C++20 for device code; host code stays C++23. Recorded per Section 3 rather than downgrading host silently |
| Ubuntu (WSL2) | Ubuntu | 26.04 LTS | matches intent |
| Python | (unspecified) | 3.14.4 | fine |
| ncu / nsys | on PATH | present in toolkit | present |
| cuBLAS / cuSPARSE / cuSOLVER | present | 13.6 / 12.8 / 12.2 | present |
| NVML | present | header plus runtime `libnvidia-ml.so.1` plus stub | present |

Still to install when their phase arrives (not blocking kernel work):
clang-format, clang-tidy (style gate, Phase 9), ruff (Python lint), and
texlive-latex-extra plus latexmk (reports, Phase 10).

### Open blocker: ncu GPU performance counter permission (WSL2)

`ncu` returns `ERR_NVGPUCTRPERM` for both the normal user and root. Under WSL2 the
counter permission is enforced by the Windows NVIDIA driver, so a Linux side or
`sudo` fix does not apply. The diagnostic rounds (Phases 2 to 5) are the core of
the project and depend on this, so it must be cleared before Phase 2's round.

Fix (needs Windows administrator and a reboot or driver restart); either path:

- GUI: NVIDIA Control Panel then Desktop or Developer menu then Manage GPU
  Performance Counters then "Allow access to the GPU performance counters to all
  users", apply.
- Registry (elevated): set DWORD `RmProfilingAdminOnly = 0` under
  `HKLM\SYSTEM\CurrentControlSet\Services\nvlddmkm\Global\NVTweak`, then reboot.
  The key currently has no such value.

Verification after the fix: rerun the ncu probe on `stream_copy` and confirm no
`ERR_NVGPUCTRPERM` and a populated metric table. See ENGINEERING_LOG entry 2026-07-19.

### Decisions taken

- Repo lives on the Windows filesystem (the working directory) for IDE
  visibility; builds run in WSL2 against the `/mnt/c` path through a
  `~/ckl` symlink. Revisit only if the CUTLASS fetch in Phase 5 makes `/mnt/c`
  I/O too slow, in which case relocate to a WSL native path.
- `--Werror=all-warnings` on device code; `-Wpedantic` applied to hand written
  host C++ only, not the nvcc host passthrough, because nvcc's generated
  `.stub.c` files under separable compilation use line directives that
  `-Wpedantic` rejects and those files are not ours.

## Phase 1: naive GEMM, harness, cuBLAS oracle

Status: pending.
