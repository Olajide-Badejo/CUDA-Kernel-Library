# What this changes

Say what changed and why in a sentence or two. If it changes a kernel, name the
limiter you were aiming at.

## Checklist

- [ ] `make build` compiles with zero warnings (`CKL_WERROR` on).
- [ ] `ctest --test-dir build` is green, including the `gpu` labeled tests, on a
      machine with an NVIDIA card.
- [ ] `make check-style` passes: the dash gate (`scripts/check_no_dashes.py`) and
      the provenance gate (`scripts/check_provenance.py`).
- [ ] `doxygen Doxyfile` finishes with zero warnings, if a public header changed.
- [ ] No new performance number without a run. Every figure quoted here comes
      from a results file in `experiments/results/` and names the commit that
      produced it. An estimate is labeled as a hypothesis, and an unmeasured
      value reads "pending".
- [ ] No en dash or em dash anywhere in the diff, including the commit messages.
- [ ] No TODOs, stubs, or silent fallbacks: either the fast path runs, or a
      status comes back, or `chosen` reports the path that was taken.

## Measurements, if this touches performance

| shape | before | after | results file | commit |
|-------|--------|-------|--------------|--------|
|       |        |       |              |        |

Delete the table if the change cannot move a number.
