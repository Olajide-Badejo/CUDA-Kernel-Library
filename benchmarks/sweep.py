#!/usr/bin/env python3
"""Drive the full benchmark sweep and write the canonical results.

Resolves the sweep matrix (families, variants, dtypes, shapes), runs bench_all
once per configuration with NVML sampling on, and appends one JSONL row per run to
experiments/results/sweep.jsonl. The sweep is resumable: a configuration whose
(family, variant, dtype, m, n, k, commit) row already exists is skipped, so a
killed run resumes without repeating finished work. Pass --force to redo, --quick
for a representative subset, --list to print the matrix without running.

After the sweep it refreshes experiments/results/summary.csv (the one committed
canonical summary) and reports any rows flagged as thermally or power throttled;
per the build spec those are rerun after cooldown rather than trusted.

Progress uses tqdm when available and falls back to plain lines otherwise, so it
reads cleanly in a CI log.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "experiments" / "results"
JSONL = RESULTS / "sweep.jsonl"
SUMMARY = RESULTS / "summary.csv"
BENCH_ALL = REPO / "build" / "benchmarks" / "bench_all"

# Sweep matrix. Each entry: (family, variant, dtype, [shape (m, n, k), ...]).
GEMM_FP32_SIZES = [512, 1024, 2048, 4096]
GEMM_TENSOR_SIZES = [1024, 2048, 4096, 8192]
GEMV_SIZES = [1024, 2048, 4096, 8192]
SPMV_SIZES = [16384, 65536, 131072]
TRSM_SIZES = [512, 1024, 2048]


def cube(sizes):
    return [(s, s, s) for s in sizes]


def matrix(quick: bool):
    rows = []
    fp32 = ["naive", "tiled", "register", "cp_async"] if not quick else ["naive", "cp_async"]
    for v in fp32:
        for shape in cube(GEMM_FP32_SIZES if not quick else [1024, 2048]):
            rows.append(("gemm", v, "fp32", shape))
    fp16 = ["wmma", "mma_ptx", "mma_ldm", "mma_opt"] if not quick else ["wmma", "mma_opt"]
    for v in fp16:
        for shape in cube(GEMM_TENSOR_SIZES if not quick else [2048, 4096]):
            rows.append(("gemm", v, "fp16", shape))
    for shape in cube(GEMM_TENSOR_SIZES if not quick else [2048]):
        rows.append(("gemm", "wmma", "bf16", shape))
    for v in ["naive", "warp", "vectorized"]:
        for shape in [(s, s, 0) for s in (GEMV_SIZES if not quick else [2048, 4096])]:
            rows.append(("gemv", v, "fp32", shape))
    for v in ["naive", "warp"]:
        for shape in [(s, s, 0) for s in (SPMV_SIZES if not quick else [65536])]:
            rows.append(("spmv", v, "fp32", shape))
    for v in ["naive", "blocked"]:
        for shape in [(s, 256, 0) for s in (TRSM_SIZES if not quick else [1024])]:
            rows.append(("trsm", v, "fp32", shape))
    return rows


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                              capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def done_keys() -> set:
    keys = set()
    if JSONL.exists():
        with JSONL.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                keys.add((r["family"], r["variant"], r["dtype"], r["m"], r["n"], r["k"],
                          r.get("commit", "")))
    return keys


def refresh_summary() -> int:
    if not JSONL.exists():
        return 0
    rows = []
    with JSONL.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        return 0
    cols = ["family", "variant", "dtype", "m", "n", "k", "median_ms", "iqr_ms",
            "gflops", "baseline_gflops", "pct_baseline", "throttled",
            "median_sm_clock_mhz", "max_temp_c", "max_power_w", "cuda_runtime",
            "cuda_driver", "commit"]
    with SUMMARY.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return len(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="rerun configs that already have a row")
    ap.add_argument("--quick", action="store_true", help="representative subset")
    ap.add_argument("--list", action="store_true", help="print the matrix and exit")
    args = ap.parse_args()

    configs = matrix(args.quick)
    if args.list:
        for fam, var, dt, (m, n, k) in configs:
            print(f"{fam:5s} {var:12s} {dt:5s} {m}x{n}x{k}")
        print(f"{len(configs)} configurations")
        return 0

    if not BENCH_ALL.exists():
        print(f"bench_all not found at {BENCH_ALL}; build first (make build)", file=sys.stderr)
        return 1

    RESULTS.mkdir(parents=True, exist_ok=True)
    commit = git_commit()
    done = set() if args.force else done_keys()

    todo = [c for c in configs
            if (c[0], c[1], c[2], c[3][0], c[3][1], c[3][2], commit) not in done]
    skipped = len(configs) - len(todo)
    print(f"sweep: {len(configs)} configs, {skipped} already done, {len(todo)} to run "
          f"(commit {commit[:8]})")

    try:
        from tqdm import tqdm
        bar = tqdm(todo, unit="cfg")
    except ImportError:
        bar = todo

    throttled = 0
    start = time.time()
    with JSONL.open("a") as out:
        for i, (fam, var, dt, (m, n, k)) in enumerate(bar, start=1):
            proc = subprocess.run(
                [str(BENCH_ALL), fam, var, dt, str(m), str(n), str(k), commit],
                capture_output=True, text=True)
            if proc.returncode != 0:
                print(f"  FAILED {fam} {var} {dt} {m}x{n}x{k}: {proc.stderr.strip()}",
                      file=sys.stderr)
                continue
            row = proc.stdout.strip()
            out.write(row + "\n")
            out.flush()
            r = json.loads(row)
            if r.get("throttled"):
                throttled += 1
            if not hasattr(bar, "set_postfix"):
                elapsed = time.time() - start
                eta = elapsed / i * (len(todo) - i)
                print(f"  [{i}/{len(todo)}] {fam} {var} {dt} {m}x{n}x{k} "
                      f"{r['gflops']:.0f} GFLOP/s ({r['pct_baseline']:.0f}% base) "
                      f"eta {eta:.0f}s")

    n_rows = refresh_summary()
    print(f"wrote {SUMMARY.relative_to(REPO)} ({n_rows} rows)")
    if throttled:
        print(f"WARNING: {throttled} row(s) flagged throttled; rerun after cooldown "
              f"before trusting them")
    else:
        print("no throttled rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
