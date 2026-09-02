#!/usr/bin/env python3
"""Fail if the top kernel's SASS has moved away from the committed golden file.

The top tensor kernel is the one every performance claim on the FP16 ladder
rests on, and what it compiles to is a compile time fact: no GPU, no locked
clocks, no measurement protocol. So it can be pinned. This script captures the
kernel from a fresh build through benchmarks/capture_sass.sh, compares it with
experiments/sass/golden/<rung>.sass, and exits non-zero on any difference with
a unified diff on stdout.

The gate is only worth having if it can fail, so the normalization in
capture_sass.sh is deliberately narrow: it removes instruction addresses,
encoded words, column padding and the unnamed namespace module hash, and keeps
every mnemonic, every operand and their order. A rebuild of an unchanged tree
passes; one changed instruction in the mainloop does not.

When a change to the kernel is intended, regenerate the golden with --update
and commit it in the same change as the kernel, so the diff in review shows what
moved in the instruction stream.

Usage:
  scripts/sass_diff.py                          diff the top kernel
  scripts/sass_diff.py --rung tile_128x128x32   diff another rung
  scripts/sass_diff.py --update                 rewrite the golden
  scripts/sass_diff.py --build-dir /tmp/b       read a build tree elsewhere

Exit codes: 0 identical, 1 different, 2 the capture could not be produced.
"""

from __future__ import annotations

import argparse
import difflib
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CAPTURE = REPO / "benchmarks" / "capture_sass.sh"
GOLDEN_DIR = REPO / "experiments" / "sass" / "golden"

# The kernel the ladder's headline number comes from.
DEFAULT_RUNG = "mma_opt"


def capture(rung: str, build_dir: Path) -> str:
    """One normalized capture of a rung, straight from the built objects."""
    cmd = [
        "bash",
        str(CAPTURE),
        "--rung",
        rung,
        "--stdout",
        "--build-dir",
        str(build_dir),
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if out.returncode != 0 or not out.stdout.strip():
        print(f"sass_diff: could not capture rung '{rung}'.", file=sys.stderr)
        print(out.stderr.rstrip(), file=sys.stderr)
        print(
            "Build the tree first, or point --build-dir at a build directory that "
            "has the ckl_gemm objects.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return out.stdout


def toolkit_line(lines: list[str]) -> str:
    """The assembler a capture names in its header, or a stand in."""
    for line in lines[:8]:
        if line.startswith("// toolkit:"):
            return line.split(":", 1)[1].strip()
    return "an unrecorded toolkit"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rung", default=DEFAULT_RUNG, help="which ladder rung to diff")
    parser.add_argument(
        "--build-dir",
        default=None,
        help="build tree holding the objects (default: the repository's build directory)",
    )
    parser.add_argument("--golden", default=None, help="path to the golden file")
    parser.add_argument(
        "--update",
        action="store_true",
        help="rewrite the golden from this build instead of comparing against it",
    )
    args = parser.parse_args()

    build_dir = Path(args.build_dir).resolve() if args.build_dir else REPO / "build"
    golden = Path(args.golden) if args.golden else GOLDEN_DIR / f"{args.rung}.sass"

    fresh = capture(args.rung, build_dir)

    if args.update:
        golden.parent.mkdir(parents=True, exist_ok=True)
        golden.write_text(fresh, encoding="utf-8", newline="\n")
        lines = len(fresh.splitlines())
        print(f"golden updated: {golden.relative_to(REPO)} ({lines} lines)")
        return 0

    if not golden.exists():
        print(f"sass_diff: no golden at {golden}.", file=sys.stderr)
        print("Create it with --update and commit it alongside the kernel.", file=sys.stderr)
        return 2

    old = golden.read_text(encoding="utf-8").splitlines(keepends=True)
    new = fresh.splitlines(keepends=True)

    # SASS is a function of the source, the architecture and the assembler. When
    # the assembler is the thing that moved, say so, because the unified diff
    # below would otherwise read as a kernel change and send a reader looking for
    # an edit that is not there.
    old_toolkit = toolkit_line(old)
    new_toolkit = toolkit_line(new)
    if old_toolkit != new_toolkit:
        print(
            f"sass_diff: the golden was assembled by {old_toolkit} and this tree "
            f"has {new_toolkit}.",
            file=sys.stderr,
        )
        print(
            "Assemble on the toolkit the golden names, or regenerate the golden "
            "with --update on this one and commit the churn as a toolkit bump.",
            file=sys.stderr,
        )
        return 1

    if old == new:
        print(
            f"sass diff clean: {args.rung} matches "
            f"{golden.relative_to(REPO)} ({len(new)} lines)"
        )
        return 0

    diff = difflib.unified_diff(
        old,
        new,
        fromfile=str(golden.relative_to(REPO)),
        tofile=f"fresh capture of {args.rung}",
        n=3,
    )
    sys.stdout.writelines(diff)
    common = min(len(old), len(new))
    changed = sum(1 for i in range(common) if old[i] != new[i]) + abs(len(old) - len(new))
    print(
        f"\nsass diff FAILED: {args.rung} differs from the golden "
        f"(at least {changed} line(s)).",
        file=sys.stderr,
    )
    print(
        "If the change is intended, rerun with --update and commit the golden "
        "with the kernel change.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
