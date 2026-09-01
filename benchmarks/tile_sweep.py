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
decision table rather than a narrative.

Everything about the measurement protocol comes from sweep.py by import, not by
copy: the clock preamble, the independent process repeats with their bootstrap
interval, the shuffled order with cooldowns, the throttle gate, and the vendor
baseline measured once per shape. A decision table built under a weaker protocol
than the ladder would be a decision table nobody should trust, and two copies of
the rules would drift.

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
import random
import sys
import time
from pathlib import Path

# The measurement protocol is sweep.py's. This file is next to it, so a plain
# import finds it however the script was invoked.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sweep import (  # noqa: E402
    BASELINE_VARIANT,
    BOOTSTRAP_RESAMPLES,
    CLOCK_TOLERANCE,
    DEFAULT_COOLDOWN_S,
    DEFAULT_LOCK_MHZ,
    DEFAULT_PROCESS_REPS,
    SWEEP_SCHEMA_VERSION,
    BenchFailure,
    Config,
    check_clock_drift,
    clock_preamble,
    git_commit,
    measure_config,
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

# The dispatcher reads variant, m, n, k and gflops out of this file and ignores
# the rest. Everything else is provenance: without it a reader cannot tell a row
# measured at a locked clock from one measured on a throttling GPU.
COLUMNS = [
    "family", "variant", "dtype", "m", "n", "k",
    "median_ms", "iqr_ms", "min_ms", "reps",
    "ci95_lo_ms", "ci95_hi_ms", "process_reps",
    "gflops", "baseline_variant", "baseline_gflops", "pct_baseline",
    "l2_flushed", "launch_mode", "inner_launches",
    "chosen", "entry_point", "tile", "verify_residual", "verify_tol",
    "cublas_math_mode",
    "throttled", "throttle_reruns", "median_sm_clock_mhz", "clock_locked",
    "locked_clock_mhz", "max_temp_c", "max_power_w",
    "order_seed", "order_index", "sweep_schema_version",
    "cuda_runtime", "cuda_driver", "commit",
]


def shapes() -> list[tuple[int, int, int]]:
    out = [(s, s, s) for s in CUBE_SIZES]
    out.extend(SKINNY)
    out.extend(UNALIGNED)
    return out


def matrix() -> list[Config]:
    return [Config("gemm", tile, "fp16", shape) for shape in shapes() for tile in TILES]


def shape_label(cfg: Config) -> str:
    return f"{cfg.m}x{cfg.n}x{cfg.k}"


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


def parse_args() -> argparse.Namespace:
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
    ap.add_argument("--process-reps", type=int, default=DEFAULT_PROCESS_REPS, metavar="N",
                    help=f"independent bench_all processes per configuration "
                         f"(default {DEFAULT_PROCESS_REPS})")
    ap.add_argument("--cooldown-s", type=float, default=DEFAULT_COOLDOWN_S, metavar="S",
                    help=f"sleep between runs (default {DEFAULT_COOLDOWN_S})")
    ap.add_argument("--throttle-cooldown-s", type=float, default=60.0, metavar="S")
    ap.add_argument("--order-seed", type=int, default=None, metavar="N",
                    help="seed for the configuration order; a fresh random one by default, "
                         "and whichever is used is recorded in every row")
    ap.add_argument("--bootstrap-seed", type=int, default=20250901, metavar="N")
    ap.add_argument("--bootstrap-resamples", type=int, default=BOOTSTRAP_RESAMPLES, metavar="N")
    ap.add_argument("--commit", default=None, metavar="HASH")
    ap.add_argument("--bench-flag", action="append", default=[], dest="bench_flags",
                    metavar="FLAG", help="extra argument passed through to bench_all")
    return ap.parse_args()


# Digits worth keeping in the decision table. Six significant figures on a
# throughput measured to three is noise dressed as precision.
ROUNDING = {"gflops": 3, "baseline_gflops": 3, "pct_baseline": 2, "median_ms": 6,
            "iqr_ms": 6, "min_ms": 6, "ci95_lo_ms": 6, "ci95_hi_ms": 6,
            "median_sm_clock_mhz": 0, "max_power_w": 1}


def write_row(writer, row: dict) -> None:
    out = {}
    for k in COLUMNS:
        v = row.get(k, "")
        if k in ROUNDING and isinstance(v, float):
            v = round(v, ROUNDING[k])
        out[k] = v
    writer.writerow(out)


def main() -> int:
    args = parse_args()
    configs = matrix()
    if args.only:
        wanted = set(args.only)
        configs = [c for c in configs if shape_label(c) in wanted]
        if not configs:
            print(f"no shape in the matrix matches {sorted(args.only)}", file=sys.stderr)
            return 2

    if args.list:
        for cfg in configs:
            print(f"{cfg.variant:18s} {shape_label(cfg)}")
        print(f"{len(configs)} configurations, {args.process_reps} processes each")
        return 0

    if not args.bench.exists():
        print(f"bench_all not found at {args.bench}; build first (make build)", file=sys.stderr)
        return 1
    try:
        import numpy  # noqa: F401
    except ImportError:
        print("numpy is needed for the bootstrap confidence interval", file=sys.stderr)
        return 1

    commit = args.commit or git_commit()
    done = set() if args.force else existing_keys(args.out, commit)
    todo = [c for c in configs
            if (c.variant, str(c.m), str(c.n), str(c.k)) not in done]
    resumed = len(configs) - len(todo)

    # One vendor baseline per shape, measured before the tiles at that shape and
    # reused by all six of them. Six tiles re-measuring cuBLAS six times is
    # exactly the mistake that produced the impossible percentages of defect A2.
    base_name = BASELINE_VARIANT["gemm"]
    base_shapes: list[tuple[int, int, int]] = []
    for cfg in todo:
        shape = (cfg.m, cfg.n, cfg.k)
        if shape not in base_shapes:
            base_shapes.append(shape)
    baseline_cfgs = [Config("gemm", base_name, "fp16", shape) for shape in base_shapes]

    # Preamble: lock before a single configuration runs, so no row in the sweep
    # was measured at a different clock from any other. Same rule and same code
    # as the ladder sweep.
    locked, locked_mhz, code = clock_preamble(args.lock_clock, args.allow_unlocked)
    if code != 0:
        return code

    if args.order_seed is None:
        args.order_seed = random.randrange(1 << 30)
    rng = random.Random(args.order_seed)
    rng.shuffle(baseline_cfgs)
    rng.shuffle(todo)
    if args.limit > 0:
        todo = todo[:args.limit]
        keep = {(c.m, c.n, c.k) for c in todo}
        baseline_cfgs = [b for b in baseline_cfgs if (b.m, b.n, b.k) in keep]
    print(f"tile sweep: {len(configs)} configs, {resumed} already measured at this commit, "
          f"{len(baseline_cfgs)} baselines and {len(todo)} tiles to run (commit {commit[:8]}), "
          f"order seed {args.order_seed}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fresh = not args.out.exists()
    measured: list[dict] = []
    throttled_rows: list[str] = []
    failures = 0
    baselines: dict[tuple[int, int, int], dict] = {}
    start = time.time()

    with args.out.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        if fresh:
            writer.writeheader()
        queue = [("baseline", c) for c in baseline_cfgs] + [("tile", c) for c in todo]
        for i, (phase, cfg) in enumerate(queue, start=1):
            if i > 1 and args.cooldown_s > 0:
                time.sleep(args.cooldown_s)
            try:
                row = measure_config(args.bench, cfg, commit, args)
                reruns = 0
                if row.get("throttled"):
                    reruns = 1
                    print(f"  throttled: {cfg.label()}; cooling for "
                          f"{args.throttle_cooldown_s:.0f}s and re-running", file=sys.stderr)
                    time.sleep(args.throttle_cooldown_s)
                    row = measure_config(args.bench, cfg, commit, args)
            except BenchFailure as exc:
                print(f"  FAILED {exc}", file=sys.stderr)
                failures += 1
                continue
            row["throttle_reruns"] = reruns
            if row.get("throttled"):
                throttled_rows.append(cfg.label())
            row["clock_locked"] = locked
            row["locked_clock_mhz"] = locked_mhz if locked else ""
            row["order_seed"] = args.order_seed
            row["order_index"] = i
            row["sweep_schema_version"] = SWEEP_SCHEMA_VERSION

            shape = (cfg.m, cfg.n, cfg.k)
            if phase == "baseline":
                baselines[shape] = row
            base = baselines.get(shape)
            if base is not None:
                row["baseline_variant"] = base["variant"]
                row["baseline_gflops"] = round(float(base["gflops"]), 3)
                row["pct_baseline"] = (round(100.0 * float(row["gflops"]) /
                                             float(base["gflops"]), 2)
                                       if float(base["gflops"]) > 0 else "")
            write_row(writer, row)
            f.flush()
            measured.append(row)
            elapsed = time.time() - start
            eta = elapsed / i * (len(queue) - i)
            pct = row.get("pct_baseline")
            pct_text = f" ({float(pct):.0f}% base)" if isinstance(pct, (int, float)) else ""
            print(f"  [{i}/{len(queue)}] {cfg.variant:24s} {shape_label(cfg)} "
                  f"{row['gflops']:.0f} GFLOP/s{pct_text} eta {eta:.0f}s")

    if locked:
        unlock_clocks()
        print("clock: lock released (nvidia-smi -rgc)")
    print(f"wrote {args.out} ({len(measured)} new rows)")

    status = 0
    if failures:
        print(f"{failures} configuration(s) failed to produce a row", file=sys.stderr)
        status = 1
    if throttled_rows:
        print(f"throttle gate FAILED: {len(throttled_rows)} configuration(s) were still "
              f"throttled after a cooldown and a re-run:", file=sys.stderr)
        for line in throttled_rows:
            print("  " + line, file=sys.stderr)
        status = 1

    if locked:
        drifted = check_clock_drift(measured, locked_mhz)
        if drifted:
            print(f"clock drift: {len(drifted)} row(s) measured more than "
                  f"{CLOCK_TOLERANCE * 100:.0f} percent off the {locked_mhz} MHz lock. "
                  f"These are not comparable with the rest of the sweep:", file=sys.stderr)
            for line in drifted:
                print("  " + line, file=sys.stderr)
            status = 1
        else:
            print(f"clock drift: every row within {CLOCK_TOLERANCE * 100:.0f} percent of the "
                  f"{locked_mhz} MHz lock")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
