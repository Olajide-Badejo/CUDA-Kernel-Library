#!/usr/bin/env python3
"""Fail if any recorded result carries a commit hash that git cannot resolve.

The README and CONTRIBUTING both promise that every number is traceable to the
commit that produced it. That promise is only worth something if the hash is
still reachable, so this gate walks every results file, collects every commit
hash it finds, and runs `git cat-file -t` on each one.

Sources walked:
  experiments/results/summary.csv               column `commit`
  experiments/results/sweep.jsonl               field  `commit`
  experiments/results/ncu/round*/round_meta.txt line   `commit: <hash>`

Hashes listed in experiments/results/legacy_hashes.txt are allowed through.
That file records the v1 data whose history was rewritten before this gate
existed (defect A1). The re-run happened: summary.csv and the ncu rounds from 12
on carry hashes that resolve, so the allowlist no longer covers anything this
release claims. It is kept, scoped, because the artifacts those dead hashes
stamp are still in the tree as history: the archived v1 rows in sweep.jsonl and
the round 01 to 09 meta pages. Its own header carries the reasoning. A hash that
is not already in it is a real failure and is never added to make a build pass.

Exit code 0 means every hash outside the allowlist resolves, 1 means at least
one does not. Wired into `make check-style`.
"""

from __future__ import annotations

import csv
import json
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "experiments" / "results"
SUMMARY = RESULTS / "summary.csv"
JSONL = RESULTS / "sweep.jsonl"
NCU = RESULTS / "ncu"
ALLOWLIST = RESULTS / "legacy_hashes.txt"

HASH_RE = re.compile(r"^[0-9a-f]{7,40}$")
META_RE = re.compile(r"^\s*commit\s*[:=]?\s*([0-9a-fA-F]{7,40})\s*$")

# Placeholders bench_all and sweep.py write when git is not reachable. They are
# not hashes and are reported separately so the message is not confusing.
PLACEHOLDERS = {"", "unknown", "none", "n/a"}


def load_allowlist() -> set[str]:
    if not ALLOWLIST.exists():
        return set()
    allowed = set()
    for line in ALLOWLIST.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip().lower()
        if line:
            allowed.add(line)
    return allowed


def from_summary() -> list[tuple[str, str]]:
    """Return (hash, source) pairs from the canonical CSV summary."""
    if not SUMMARY.exists():
        return []
    found = []
    with SUMMARY.open(newline="") as f:
        for lineno, row in enumerate(csv.DictReader(f), start=2):
            value = (row.get("commit") or "").strip()
            found.append((value, f"{SUMMARY.relative_to(REPO)}:{lineno}"))
    return found


def from_jsonl() -> list[tuple[str, str]]:
    if not JSONL.exists():
        return []
    found = []
    with JSONL.open() as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                found.append(("", f"{JSONL.relative_to(REPO)}:{lineno} (unparseable)"))
                continue
            found.append((str(row.get("commit", "")).strip(),
                          f"{JSONL.relative_to(REPO)}:{lineno}"))
    return found


def from_round_meta() -> list[tuple[str, str]]:
    found = []
    for meta in sorted(NCU.glob("round*/round_meta.txt")):
        hits = 0
        for lineno, line in enumerate(meta.read_text(encoding="utf-8").splitlines(), start=1):
            m = META_RE.match(line)
            if m:
                hits += 1
                found.append((m.group(1).strip(),
                              f"{meta.relative_to(REPO)}:{lineno}"))
        if hits == 0:
            found.append(("", f"{meta.relative_to(REPO)} (no commit line)"))
    return found


def resolves(commit: str) -> bool:
    proc = subprocess.run(["git", "cat-file", "-t", commit], cwd=REPO,
                          capture_output=True, text=True, check=False)
    return proc.returncode == 0 and proc.stdout.strip() == "commit"


def main() -> int:
    allowed = load_allowlist()
    records = from_summary() + from_jsonl() + from_round_meta()

    if not records:
        print("provenance check: no results files found; nothing to verify.")
        return 0

    # One git call per distinct hash, not one per row.
    verdict: dict[str, bool] = {}
    offenders: list[str] = []
    legacy_seen: set[str] = set()
    checked = 0

    for value, source in records:
        key = value.lower()
        if key in PLACEHOLDERS:
            offenders.append(f"{source}: no commit recorded (value {value!r})")
            continue
        if not HASH_RE.match(key):
            offenders.append(f"{source}: {value!r} is not a commit hash")
            continue
        if key in allowed:
            legacy_seen.add(key)
            continue
        if key not in verdict:
            verdict[key] = resolves(key)
            checked += 1
        if not verdict[key]:
            offenders.append(f"{source}: commit {value} does not resolve "
                             f"(git cat-file -t {value} failed)")

    if offenders:
        print("provenance check FAILED:")
        for line in offenders:
            print("  " + line)
        print(f"\n{len(offenders)} offending row(s). Either the result predates a history "
              f"rewrite, in which case re-run it on current history, or add the hash to "
              f"{ALLOWLIST.relative_to(REPO)} with a reason.")
        return 1

    print(f"provenance check clean: {len(records)} recorded row(s), "
          f"{checked} distinct hash(es) resolved in git, "
          f"{len(legacy_seen)} known dead v1 hash(es) allowed by "
          f"{ALLOWLIST.relative_to(REPO)}.")
    if legacy_seen:
        print("  those hashes guard archived v1 artifacts only (the superseded rows in "
              "sweep.jsonl and the round 01 to 09 meta pages). No current claim rests "
              "on them; see the header of the allowlist.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
