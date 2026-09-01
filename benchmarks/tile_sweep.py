#!/usr/bin/env python3
"""Sweep every tile family instantiation across the Gate D shape matrix.

Writes experiments/results/tile_sweep.csv, which is the file the Section 9.6
dispatch heuristic reads: for each shape it keeps the throughput of every block
tile, and kAuto picks the tile that quantizes best among those within five
percent of the fastest at the nearest swept shape. Without this file the
heuristic has no throughput term at all and falls back to wave quantization
alone, which is a hypothesis about which shape wins and not a measurement.
ckl::GemmPlan::tuned reports which of the two happened.

This is a separate driver from sweep.py rather than a family inside it because
the two answer different questions. sweep.py measures the ladder: one row per
rung, to show the progression. This measures one rung's parameter space: six
tiles times fourteen shapes, all of them the same kernel, and the output is a
decision table rather than a narrative. It reuses sweep.py's clock preamble
outright, so the locking rule is the same code and cannot drift.

The matrix is Gate D's: 128, 256, 512, 768, 1024, 2048, 4096 and 8192 cubed, the
skinny pair (m 8192, n 64, k 4096) and its transpose, and the unaligned rows
that used to fall off the ladder entirely. Unaligned shapes are first class here
because the whole point of the predicated tails is that they stop being a
different code path.

The full matrix is hours of GPU time at locked clocks. --limit and --only run a
subset, --list prints the matrix without running anything, and --out writes
somewhere other than the canonical results file, which is what a smoke run
should use.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

# The clock preamble, the drift check and the commit stamp are sweep.py's. This
# file is next to it, so a plain import finds it however the script was invoked.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sweep import (  # noqa: E402
    CLOCK_TOLERANCE,
    DEFAULT_LOCK_MHZ,
    check_clock_drift,
    git_commit,
    lock_clocks,
    unlock_clocks,
)

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "experiments" / "results"
DEFAULT_OUT = RESULTS / "tile_sweep.csv"
DEFAULT_BENCH = REPO / "build" / "benchmarks" / "bench_all"

# Gate D, in the order the report reads them.
CUBE_SIZES = [128, 256, 512, 768, 1024, 2048, 4096, 8192]
SKINNY = [(8192, 64, 4096), (64, 8192, 4096)]
# The shapes Section 9.5 named as the fallback cliff. They are rows of the sweep
# and not a footnote to it.
UNALIGNED = [(127, 127, 127), (1000, 1000, 1000), (4096, 4090, 4096), (129, 257, 193)]

# The roster, as bench_all names it. Kept in step with
# src/gemm/detail/gemm_tile_registry.hpp by the shape in the name: a row means
# the same tile whatever order the roster is in.
TILES = [
    "tile_128x128x32",
    "tile_128x256x32",
    "tile_256x128x32",
    "tile_128x64x64",
    "tile_64x128x64",
    "tile_128x128x64",
]

# Section 13 fields this driver can honor today. Anything it cannot measure is
# absent rather than blank guessed: the ncu derived columns belong to the
# diagnostic rounds, not here.
COLUMNS = [
    "family", "variant", "dtype", "m", "n", "k",
    "median_ms", "iqr_ms", "gflops", "baseline_gflops", "pct_baseline",
    "throttled", "median_sm_clock_mhz", "clock_locked", "locked_clock_mhz",
    "max_temp_c", "max_power_w", "cuda_runtime", "cuda_driver", "commit",
]


def shapes() -> list[tuple[int, int, int]]:
    out = [(s, s, s) for s in CUBE_SIZES]
    out.extend(SKINNY)
    out.extend(UNALIGNED)
    return out


def matrix() -> list[tuple[str, tuple[int, int, int]]]:
    return [(tile, shape) for shape in shapes() for tile in TILES]


def shape_label(shape: tuple[int, int, int]) -> str:
    return f"{shape[0]}x{shape[1]}x{shape[2]}"


def existing_keys(path: Path, commit: str) -> set:
    """Rows already measured at this commit, so a killed sweep resumes."""
    keys = set()
    if not path.exists():
        return keys
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("commit") == commit:
                keys.add((row["variant"], row["m"], row["n"], row["k"]))
    return keys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help=f"CSV to append to (default {DEFAULT_OUT.relative_to(REPO)})")
    ap.add_argument("--bench", type=Path, default=DEFAULT_BENCH,
                    help="path to the bench_all binary")
    ap.add_argument("--list", action="store_true", help="print the matrix and exit")
    ap.add_argument("--limit", type=int, default=0, metavar="N",
                    help="run only the first N configurations (0 means all)")
    ap.add_argument("--only", action="append", default=[], metavar="MxNxK",
                    help="restrict to these shapes; repeatable")
    ap.add_argument("--force", action="store_true", help="rerun configs that already have a row")
    ap.add_argument("--lock-clock", type=int, default=DEFAULT_LOCK_MHZ, metavar="MHZ",
                    help=f"graphics clock to lock before measuring (default {DEFAULT_LOCK_MHZ})")
    ap.add_argument("--allow-unlocked", action="store_true",
                    help="run even though the clock could not be locked; every row is "
                         "stamped clock_locked=false and carries no locked clock")
    args = ap.parse_args()

    configs = matrix()
    if args.only:
        wanted = set(args.only)
        configs = [c for c in configs if shape_label(c[1]) in wanted]
        if not configs:
            print(f"no shape in the matrix matches {sorted(wanted)}", file=sys.stderr)
            return 2

    if args.list:
        for tile, shape in configs:
            print(f"{tile:18s} {shape_label(shape)}")
        print(f"{len(configs)} configurations")
        return 0

    if not args.bench.exists():
        print(f"bench_all not found at {args.bench}; build first (make build)", file=sys.stderr)
        return 1

    # Preamble: lock before a single configuration runs, so no row in the sweep
    # was measured at a different clock from any other. Same rule and same code
    # as the ladder sweep.
    locked, why = lock_clocks(args.lock_clock)
    if locked:
        print(f"clock: {why}")
        locked_mhz = args.lock_clock
    else:
        print(f"clock: NOT locked. {why}", file=sys.stderr)
        if not args.allow_unlocked:
            print("refusing to run. Locking needs persistence mode and clock control, which "
                  "under WSL2 means an elevated Windows side nvidia-smi, or run with "
                  "--allow-unlocked and accept that the rows carry no locked clock.",
                  file=sys.stderr)
            return 2
        print("running unlocked because --allow-unlocked was passed; every row will be "
              "stamped clock_locked=false and no locked clock claim can be made from this "
              "data, which also means the dispatch heuristic must not be tuned from it.",
              file=sys.stderr)
        locked_mhz = None

    commit = git_commit()
    done = set() if args.force else existing_keys(args.out, commit)
    todo = [c for c in configs
            if (c[0], str(c[1][0]), str(c[1][1]), str(c[1][2])) not in done]
    resumed = len(configs) - len(todo)
    if args.limit > 0:
        todo = todo[:args.limit]
    print(f"tile sweep: {len(configs)} configs, {resumed} already measured at this commit, "
          f"{len(todo)} to run (commit {commit[:8]})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fresh = not args.out.exists()
    measured: list[dict] = []
    start = time.time()
    with args.out.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        if fresh:
            writer.writeheader()
        for i, (tile, (m, n, k)) in enumerate(todo, start=1):
            proc = subprocess.run(
                [str(args.bench), "gemm", tile, "fp16", str(m), str(n), str(k), commit],
                capture_output=True, text=True, check=False)
            if proc.returncode != 0:
                print(f"  FAILED {tile} {shape_label((m, n, k))}: {proc.stderr.strip()}",
                      file=sys.stderr)
                continue
            row = json.loads(proc.stdout.strip())
            row["clock_locked"] = locked
            row["locked_clock_mhz"] = locked_mhz if locked else ""
            writer.writerow(row)
            f.flush()
            measured.append(row)
            elapsed = time.time() - start
            eta = elapsed / i * (len(todo) - i)
            print(f"  [{i}/{len(todo)}] {tile} {shape_label((m, n, k))} "
                  f"{row['gflops']:.0f} GFLOP/s ({row['pct_baseline']:.0f}% base) eta {eta:.0f}s")

    if locked:
        unlock_clocks()
        print("clock: lock released (nvidia-smi -rgc)")
    print(f"wrote {args.out} ({len(measured)} new rows)")

    throttled = sum(1 for r in measured if r.get("throttled"))
    if throttled:
        print(f"WARNING: {throttled} row(s) flagged throttled; rerun after cooldown before "
              f"trusting them")

    if locked:
        drifted = check_clock_drift(measured, locked_mhz)
        if drifted:
            print(f"clock drift: {len(drifted)} row(s) measured more than "
                  f"{CLOCK_TOLERANCE * 100:.0f} percent off the {locked_mhz} MHz lock. "
                  f"These are not comparable with the rest of the sweep:", file=sys.stderr)
            for line in drifted:
                print("  " + line, file=sys.stderr)
            return 1
        print(f"clock drift: every row within {CLOCK_TOLERANCE * 100:.0f} percent of the "
              f"{locked_mhz} MHz lock")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
