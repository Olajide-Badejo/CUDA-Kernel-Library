#!/usr/bin/env python3
"""Compare a quick sweep against the committed performance baseline.

The nightly perf-regression job runs this. It answers one question per
(family, variant, dtype, shape): is tonight's median more than a tolerance
slower than the baseline's, by more than the measurement noise?

Both halves have to be true before a row counts as a regression:

  1. The median is more than --tolerance-pct slower (5 percent by default).
  2. The two bootstrap intervals do not overlap: tonight's ci95_lo_ms is above
     the baseline's ci95_hi_ms.

Either half on its own produces a job that cries wolf. A 6 percent move on a
kernel whose interval spans 12 percent is noise, and the sweep already records
the interval that says so; a non overlapping pair of intervals 1 percent apart
is a real but uninteresting drift on a kernel measured very precisely. Both
together is a change worth waking someone for.

The baseline is a sweep summary, in the same schema benchmarks/sweep.py writes,
copied to experiments/results/perf_baseline.csv by the owner from a run at
locked clocks. Until that file is committed there is nothing to compare
against, and this script says so and exits non-zero rather than passing: the
workflow keeps the job from starting at all (a real skipped conclusion) until
the file lands.

A row measured while the GPU was throttled is not evidence either way, so a
throttled row in tonight's sweep fails the run on its own.

Usage:
  scripts/perf_regression.py                     run the quick sweep and compare
  scripts/perf_regression.py --current out.csv   compare a summary already on disk
  scripts/perf_regression.py --baseline b.csv --current c.csv
  scripts/perf_regression.py --tolerance-pct 10

Exit codes: 0 no regression, 1 at least one regression, 2 nothing to compare.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "experiments" / "results"
BASELINE = RESULTS / "perf_baseline.csv"
SWEEP = REPO / "benchmarks" / "sweep.py"

DEFAULT_TOLERANCE_PCT = 5.0


def shape_of(row: dict) -> str:
    """The shape column of the key, whichever of the three forms the family uses."""
    matrix = (row.get("matrix") or "").strip()
    if matrix:
        return f"matrix={matrix}"
    elements = (row.get("elements") or "").strip()
    if elements and elements not in ("0", ""):
        return f"n={elements}"
    m = (row.get("m") or "0").strip()
    n = (row.get("n") or "0").strip()
    k = (row.get("k") or "0").strip()
    return f"{m}x{n}x{k}"


def key_of(row: dict) -> tuple[str, str, str, str]:
    return (
        (row.get("family") or "").strip(),
        (row.get("variant") or "").strip(),
        (row.get("dtype") or "").strip(),
        shape_of(row),
    )


def as_float(row: dict, column: str) -> float | None:
    text = (row.get(column) or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def truthy(text: str | None) -> bool:
    return (text or "").strip().lower() in ("1", "true", "yes")


def read_summary(path: Path) -> dict[tuple[str, str, str, str], dict]:
    if not path.is_file():
        print(
            f"perf_regression: no summary at {path}.\n"
            "The baseline is the owner's locked clock sweep, copied to "
            f"{BASELINE.relative_to(REPO)}. Until it is committed there is "
            "nothing to compare against and this job has no verdict to give.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print(f"perf_regression: {path} has a header and no rows.", file=sys.stderr)
        raise SystemExit(2)
    out: dict[tuple[str, str, str, str], dict] = {}
    for row in rows:
        # A shape can carry a flushed and an unflushed row, and a stream and a
        # graph launched one. Only the canonical row answers the plain question,
        # and comparing a graph row against a stream row is a category error.
        if "canonical" in row and not truthy(row.get("canonical")):
            continue
        key = key_of(row)
        if key in out:
            raise SystemExit(f"perf_regression: {path} has two canonical rows for {key}.")
        out[key] = row
    return out


def run_quick_sweep(bench_dir: Path, jsonl: Path, summary: Path, extra: list[str]) -> None:
    cmd = [
        sys.executable,
        str(SWEEP),
        "--quick",
        "--jsonl",
        str(jsonl),
        "--summary",
        str(summary),
        *extra,
    ]
    print("running:", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(bench_dir), check=False)
    if proc.returncode != 0:
        raise SystemExit(
            f"perf_regression: the quick sweep exited {proc.returncode}; "
            "there is no measurement to compare."
        )


def compare(
    baseline: dict[tuple[str, str, str, str], dict],
    current: dict[tuple[str, str, str, str], dict],
    tolerance_pct: float,
) -> int:
    regressions: list[str] = []
    throttled: list[str] = []
    missing: list[str] = []
    compared = 0

    for key, base_row in sorted(baseline.items()):
        cur_row = current.get(key)
        if cur_row is None:
            missing.append(" ".join(key))
            continue

        base_ms = as_float(base_row, "median_ms")
        cur_ms = as_float(cur_row, "median_ms")
        if base_ms is None or cur_ms is None or base_ms <= 0.0:
            missing.append(" ".join(key) + " (no median_ms)")
            continue

        if truthy(cur_row.get("throttled")):
            throttled.append(" ".join(key))
            continue

        compared += 1
        slower_pct = 100.0 * (cur_ms - base_ms) / base_ms

        # The intervals. A summary written before the bootstrap columns existed
        # has none, and then the interval half of the rule cannot be evaluated;
        # treat a missing interval as a zero width one at the median, which is
        # the strictest reading and never hides a regression.
        base_hi = as_float(base_row, "ci95_hi_ms")
        cur_lo = as_float(cur_row, "ci95_lo_ms")
        base_hi = base_ms if base_hi is None else base_hi
        cur_lo = cur_ms if cur_lo is None else cur_lo
        separated = cur_lo > base_hi

        verdict = "ok"
        if slower_pct > tolerance_pct and separated:
            verdict = "REGRESSION"
            regressions.append(
                f"{' '.join(key)}: {base_ms:.6f} ms to {cur_ms:.6f} ms, "
                f"{slower_pct:+.2f} percent, tonight's ci95 low {cur_lo:.6f} ms "
                f"is above the baseline's ci95 high {base_hi:.6f} ms"
            )
        elif slower_pct > tolerance_pct:
            verdict = "slower, intervals overlap"
        print(
            f"{key[0]:>7} {key[1]:>14} {key[2]:>5} {key[3]:>16}  "
            f"{base_ms:12.6f} -> {cur_ms:12.6f} ms  {slower_pct:+7.2f}%  {verdict}"
        )

    extra = sorted(set(current) - set(baseline))
    print()
    print(f"compared {compared} of {len(baseline)} baseline rows")
    if extra:
        print(f"{len(extra)} row(s) measured tonight are not in the baseline (not an error):")
        for name in extra:
            print(f"  {' '.join(name)}")

    failed = False
    if missing:
        print(f"\n{len(missing)} baseline row(s) have no measurement tonight:", file=sys.stderr)
        for name in missing:
            print(f"  {name}", file=sys.stderr)
        failed = True
    if throttled:
        print(f"\n{len(throttled)} row(s) were throttled and prove nothing:", file=sys.stderr)
        for name in throttled:
            print(f"  {name}", file=sys.stderr)
        failed = True
    if regressions:
        print(f"\n{len(regressions)} regression(s):", file=sys.stderr)
        for line in regressions:
            print(f"  {line}", file=sys.stderr)
        failed = True

    if failed:
        return 1
    print("no regression: every baseline row is within tolerance or inside the interval.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--baseline", type=Path, default=BASELINE)
    ap.add_argument(
        "--current",
        type=Path,
        default=None,
        help="a summary already on disk; without it the quick sweep is run",
    )
    ap.add_argument("--tolerance-pct", type=float, default=DEFAULT_TOLERANCE_PCT)
    ap.add_argument(
        "--sweep-arg",
        action="append",
        default=[],
        dest="sweep_args",
        metavar="ARG",
        help="passed through to benchmarks/sweep.py, repeatable",
    )
    args = ap.parse_args()

    baseline = read_summary(args.baseline)
    print(f"baseline: {args.baseline} ({len(baseline)} canonical rows)")

    if args.current is not None:
        current = read_summary(args.current)
        print(f"current:  {args.current} ({len(current)} canonical rows)")
    else:
        with tempfile.TemporaryDirectory(prefix="ckl-perf-") as tmp:
            jsonl = Path(tmp) / "sweep.jsonl"
            summary = Path(tmp) / "summary.csv"
            run_quick_sweep(REPO, jsonl, summary, args.sweep_args)
            current = read_summary(summary)
        print(f"current:  quick sweep ({len(current)} canonical rows)")

    print(f"tolerance: more than {args.tolerance_pct} percent slower AND intervals separated\n")
    return compare(baseline, current, args.tolerance_pct)


if __name__ == "__main__":
    raise SystemExit(main())
