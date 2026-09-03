#!/usr/bin/env python3
"""Run clang-tidy over the host translation units in a compile database.

Two things stop `clang-tidy -p build` from working straight out of the tree, and
both are handled here rather than in a CI script that nobody can run locally.

The database has to be free of GCC's module scanning flags. CMake's Ninja
generator puts -fmodules-ts, -fdeps-format=p1689r5 and -fmodule-mapper=... into
every C++ command line, clang rejects all three, and the result is 126 parse
errors that hide every real finding. Configure the analysis build with
-DCMAKE_CXX_SCAN_FOR_MODULES=OFF and they are gone; `make tidy` does that.

Device translation units are not analysed. clang would have to be a CUDA
compiler to parse them, and it would disagree with nvcc about what sm_120
accepts. nvcc's own --Werror=all-warnings covers those under CKL_WERROR.

Findings are errors, not warnings: .clang-tidy sets WarningsAsErrors to the
enabled check set, so this exits non-zero the moment one fires.

Usage:
  scripts/run_clang_tidy.py -p build            analyse every host TU
  scripts/run_clang_tidy.py -p build --jobs 4   with a different parallelism
  scripts/run_clang_tidy.py -p build --list     print the file list and exit
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# nvcc drives these; clang has no business parsing them.
DEVICE_SUFFIXES = (".cu", ".cuh")
HOST_SUFFIXES = (".c", ".cc", ".cpp", ".cxx")

# Anything FetchContent dropped into the build tree is not ours to clean up.
EXCLUDE_PARTS = ("_deps", "CMakeFiles")


def host_translation_units(build_dir: Path) -> list[Path]:
    """Every host TU in the compile database that lives in this repository."""
    db = build_dir / "compile_commands.json"
    if not db.is_file():
        raise SystemExit(
            f"no compile database at {db}. Configure with "
            "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON and "
            "-DCMAKE_CXX_SCAN_FOR_MODULES=OFF first, or run `make tidy`."
        )
    entries = json.loads(db.read_text(encoding="utf-8"))
    out: set[Path] = set()
    for entry in entries:
        path = Path(entry["file"]).resolve()
        if path.suffix in DEVICE_SUFFIXES:
            continue
        if path.suffix not in HOST_SUFFIXES:
            continue
        if any(part in EXCLUDE_PARTS for part in path.parts):
            continue
        try:
            path.relative_to(REPO)
        except ValueError:
            continue
        out.add(path)
    return sorted(out)


def scanning_flags_present(build_dir: Path) -> bool:
    """True when the database still carries GCC's module scanning flags."""
    text = (build_dir / "compile_commands.json").read_text(encoding="utf-8")
    return "-fmodules-ts" in text or "-fdeps-format" in text


def run_one(tidy: str, build_dir: Path, path: Path) -> tuple[Path, int, str]:
    proc = subprocess.run(
        [tidy, "-p", str(build_dir), "--quiet", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    return path, proc.returncode, proc.stdout + proc.stderr


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-p", "--build-dir", type=Path, default=REPO / "build")
    ap.add_argument("--tidy", default=os.environ.get("CLANG_TIDY", "clang-tidy"))
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--list", action="store_true", help="print the files and exit")
    args = ap.parse_args()

    build_dir = args.build_dir.resolve()
    tidy = shutil.which(args.tidy)
    if tidy is None:
        raise SystemExit(
            f"clang-tidy not found as '{args.tidy}'. "
            "Install it with `pip install -r requirements-dev.txt`."
        )

    files = host_translation_units(build_dir)
    if args.list:
        for path in files:
            print(path.relative_to(REPO))
        return 0
    if not files:
        raise SystemExit(f"no host translation units in {build_dir}/compile_commands.json")

    version = subprocess.run([tidy, "--version"], capture_output=True, text=True, check=False)
    print(version.stdout.strip())
    if scanning_flags_present(build_dir):
        print(
            "warning: the compile database still has GCC module scanning flags; "
            "expect clang-diagnostic-error on every file. Reconfigure with "
            "-DCMAKE_CXX_SCAN_FOR_MODULES=OFF.",
            file=sys.stderr,
        )
    print(f"clang-tidy over {len(files)} host translation units in {build_dir}")

    failed: list[Path] = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for path, code, output in pool.map(lambda p: run_one(tidy, build_dir, p), files):
            if output.strip():
                print(output.rstrip())
            if code != 0:
                failed.append(path)

    if failed:
        print(f"\nclang-tidy: {len(failed)} translation unit(s) with findings:", file=sys.stderr)
        for path in failed:
            print(f"  {path.relative_to(REPO)}", file=sys.stderr)
        return 1
    print("clang-tidy clean.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
