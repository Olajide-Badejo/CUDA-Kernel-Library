# Changelog

Notable changes per phase. Dates are ISO. This project follows the phase
roadmap in the build specification; versions are tagged at the definition of
done milestones.

## Unreleased

### Phase 0 (2026-07-19)

- Repository skeleton, MIT license, `.gitignore`, attribution disabled.
- `scripts/check_no_dashes.py` dash gate, verified clean over the tree.
- CMake build targeting sm_120 with a zero warning policy; Ninja generator.
- `src/tools/device_probe.cu`: prints device properties and a measured
  streaming bandwidth used as the roofline ceiling.
- Public headers `ckl/cuda_check.hpp` and `ckl/device_buffer.hpp`.
- Documented toolchain substitutions and the WSL2 ncu counter permission
  blocker in `PROGRESS.md` and `docs/ENGINEERING_LOG.md`.
