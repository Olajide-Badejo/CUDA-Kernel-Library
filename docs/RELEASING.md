# Releasing, and the measurement campaign that gates it

This is the maintainer runbook for turning the committed 1.1.0 tree into a
published release. The code work is done; what remains is the measurement
campaign that every "pending" in the docs and the report waits on, and the
release mechanics. Nothing here is optional: ground rule 3 says no number
without a run, so the README and report keep their v1-era numbers and caveats
until this campaign replaces them.

## 0. One-time machine setup

- **Clock locking needs Windows admin.** WSL cannot lock clocks on this driver.
  In an elevated (administrator) Windows terminal:

  ```
  nvidia-smi -pm 1
  nvidia-smi -lgc 2500
  ```

  Verify from WSL: `nvidia-smi --query-gpu=clocks.gr --format=csv`. Release the
  lock afterwards with `nvidia-smi -rgc`. The sweep refuses to run unlocked
  unless `--allow-unlocked` is passed, and unlocked rows can never carry a
  locked-clock claim.
- **Pinned plotting wheels.** Figures are byte-diffed in CI, so regenerate them
  only with the PyPI wheels in `requirements.txt` (`pip install --user
  --force-reinstall --no-deps -r requirements.txt`). Debian's matplotlib
  produces different bytes from the same numbers.
- **ncu counters.** `RmProfilingAdminOnly = 0` on the Windows side, then reboot
  (see `docs/ENGINEERING_LOG.md`).
- **GPU CI runner** (can wait until after the campaign): repository Settings,
  Actions, Runners, New self-hosted runner, Linux x64, then on this machine
  `./config.sh --url <repo> --token <token> --labels gpu --name ckl-rtx5070
  --unattended` and `./run.sh`. Details in `docs/building.md`.
- **GitHub Pages**: Settings, Pages, Source: GitHub Actions, or the docs job's
  deploy step fails (which is the correct failure until then).

## 1. The measurement campaign (clocks locked throughout)

Budget a day of machine time; the v1 estimate of 4 to 7 hours predates the
five-process repeats and the paired L2 and launch-mode rows.

1. **Full sweep.** `make sweep` (that is `python3 benchmarks/sweep.py`,
   94 baseline plus 162 variant configurations, 5 processes each). The preamble
   verifies the lock and every row records it. A throttle failure stops the
   sweep; cool the machine and re-run, it resumes.
2. **Tile sweep.** `python3 benchmarks/tile_sweep.py` writes
   `experiments/results/tile_sweep.csv` (84 configurations). Committing that CSV
   switches `Algo::kAuto` from the untuned guard to the measured 9.6 heuristic;
   re-run `ctest -L gpu` afterwards, since the dispatch tests assert on the
   plan.
3. **SpMV suite.** Matrices are already cached and verified
   (`scripts/fetch_matrices.py --verify`); the sweep's spmv family writes the
   suite rows. Gate S risk recorded on the board: merge versus warp on
   soc-LiveJournal1 measured 1.33x unlocked against the 1.5x bar; if the locked
   run confirms it, either the merge kernel gains block-level tile staging or
   the gate is amended in writing in `docs/CORRECTIONS.md`.
4. **ncu rounds.** `benchmarks/run_ncu_round.sh` per family: re-run GEMM rounds
   7 and 9 (the bank conflict metric pages A6 owes), one round per Section 9.4
   item that landed, the cuBLAS profile round, and the named rounds for SpMV,
   FFT, and scan (`dram__bytes.sum` against declared model bytes is Gate X's
   15 percent check). Commit the text pages and `round_meta.txt`.
5. **Provenance closure.** Delete `experiments/results/legacy_hashes.txt`,
   re-run `make check-style`; it must stay green on the new rows alone. Update
   `docs/CORRECTIONS.md`: A1 and A4 close, A6 closes or records what could not
   be reconstructed.
6. **Perf baseline.** Copy the fresh summary to
   `experiments/results/perf_baseline.csv` and commit; the nightly
   perf-regression job switches on.
7. **Numbers into prose.** Regenerate everything (`make report`), replace the
   README's provisional table and caveat sentence, fill the register study
   throughput column (`benchmarks/register_study.py --measure`), and check the
   Gate D, S, X, R legs (the mechanically checkable criteria are in the V2
   spec sections 9.6, 12.1, 12.2, 12.3). A missed target is recorded with its
   cause, never quietly lowered.
8. **Full gate.** `make all` from a clean clone reproduces every number; ctest,
   check-style, doxygen, sass-diff all green; `git diff --exit-code
   report/figures report/tables` after a rebuild.

## 2. Release mechanics

1. Push `main`; let CI run every hosted job green and the GPU job green on the
   runner.
2. Tag: `git tag -a v1.1.0 -m "1.1.0"` and push the tag.
3. Create the GitHub release; attach `report/build/main.pdf` and
   `report/build/debug.pdf` (the README's links point at the latest release)
   plus an artifact bundle of `experiments/results/`.
4. Mint the Zenodo DOI (owner account), then add it to `CITATION.cff` and the
   README badge row in a follow-up commit.

## 3. What 1.1.0 explicitly does not claim

Until step 1.7 lands: every performance percent in the README and report is
v1-era with the provenance caveat, the new families' tables read pending, the
fused convolution gain and the split-K and stream-K speedups are hypotheses,
and `docs/CORRECTIONS.md` is the authoritative list. That is by design; the
gates are real and a gate that cannot fail is not a gate.
