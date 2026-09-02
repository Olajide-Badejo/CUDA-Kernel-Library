#!/usr/bin/env python3
"""Reassemble the top tensor kernel under register ceilings and tabulate the cost.

The kernel is gemm_mma_opt_kernel<false>, the 128 by 128 by 32 mainloop the FP16
ladder's headline number comes from. ptxas gives it 122 registers per thread when
nothing constrains it. This asks what a constraint costs, along two axes:

  maxrregcount    a hard per thread register ceiling
  launch_bounds   a floor on blocks per SM, the second parameter of
                  __launch_bounds__, which ptxas has to reach by spilling

How, and why it is done this way. `nvcc -maxrregcount` is silently ignored for a
kernel that carries __launch_bounds__, and so is `ptxas -maxrregcount` when the
PTX carries .maxntid; this was confirmed on the toolkit in the toolkit column
before the study was written this way. Launch bounds win, by design. So the study
compiles the translation unit to PTX once with the build's own flags, edits the
one performance tuning directive in a copy of that PTX, and reassembles:

  the maxrregcount axis  drops .maxntid, so the ceiling is the only constraint
  the launch_bounds axis keeps .maxntid and adds .minnctapersm N

Dropping .maxntid does not move the baseline: unconstrained, both PTX variants
still assemble to 122 registers with no spill, which is the row the table opens
with. Nothing here changes the shipped kernel; every edit is to a copy of the
PTX in a temporary directory.

Every column except throughput is a compile time fact. Registers, stack frame and
spill traffic come from ptxas -v. Blocks per SM is the occupancy arithmetic over
the sm_120 limits, written out below so a reader can check it rather than trust
it. Throughput stays "pending": it needs locked clocks and the protocol in
docs/benchmarking.md, and that is owner work.

Usage:
  benchmarks/register_study.py
  benchmarks/register_study.py --build-dir /var/tmp/ckl-build --out somewhere.csv

Exit codes: 0 on success, 2 when the build tree or the compile command is missing.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO / "experiments" / "results" / "register_study.csv"
SOURCE_NAME = "gemm_mma_opt.cu"

# The kernel under study: the plain, non fused instantiation. ILb1EE is the fused
# bias and ReLU epilogue, a different kernel with a different budget.
KERNEL_MARK = "gemm_mma_opt_kernelILb0EE"

# sm_120 occupancy limits, all of them verified hardware facts for this part in
# tasks/COMMON.md.
REGS_PER_SM = 65536
SHARED_PER_SM = 100 * 1024  # 102400 bytes usable of the 128 KB unified block
WARPS_PER_SM = 48
MAX_BLOCKS_PER_SM = 24
# Registers are allocated per warp in units of 256, warps in groups of 4.
REG_ALLOC_UNIT = 256
WARP_ALLOC_GRANULARITY = 4

# What the kernel asks for at launch, from gemm_mma_opt.cu: three pipeline stages
# of a 128 by 32 A tile plus a 32 by 128 B tile, in halves.
THREADS_PER_BLOCK = 256
SHARED_BYTES = 3 * (128 * 32 + 32 * 128) * 2  # 49152

# The register ceilings the study walks. 111 is where the v1 kernel landed and is
# kept as a reference point; None is the assembler's own answer.
MAXRREG_VALUES = [None, 64, 80, 96, 111, 128, 168]
# Blocks per SM floors. 1 is the kernel as it ships, which declares a thread
# ceiling and no block floor.
MINCTA_VALUES = [1, 2, 3, 4]

MAXNTID_RE = re.compile(r"^\.maxntid 256(,.*)?$", re.MULTILINE)
USED_RE = re.compile(r"Used (\d+) registers")
BARRIER_RE = re.compile(r"used (\d+) barriers")
PROPS_RE = re.compile(
    r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads"
)


def compile_command(build_dir: Path) -> tuple[list[str], Path]:
    """The exact nvcc line the build uses for the kernel's translation unit."""
    db = build_dir / "compile_commands.json"
    if not db.exists():
        print(f"register_study: no compile_commands.json in {build_dir}.", file=sys.stderr)
        print("Configure and build first; the study reuses the build's own flags.", file=sys.stderr)
        raise SystemExit(2)
    entries = json.loads(db.read_text(encoding="utf-8"))
    for entry in entries:
        if entry["file"].endswith(SOURCE_NAME):
            return shlex.split(entry["command"]), Path(entry["directory"])
    print(f"register_study: {SOURCE_NAME} is not in {db}.", file=sys.stderr)
    raise SystemExit(2)


def ptx_command(cmd: list[str], virtual_arch: str, out: Path) -> list[str]:
    """The build's own compile line, retargeted at PTX for one virtual arch.

    -o and its argument go, -c goes, and every code generation flag goes: a PTX
    emission names a virtual architecture and nothing else. The warnings as
    errors flag goes too, because a ceiling that forces a spill turns an
    informational ptxas message into a failed compile, and a study that cannot
    assemble its interesting rows is not a study.
    """
    kept: list[str] = []
    skip = False
    for arg in cmd:
        if skip:
            skip = False
            continue
        if arg == "-o":
            skip = True
            continue
        if arg in ("-c", "-dc"):
            continue
        if arg.startswith(("--generate-code", "-gencode", "-arch=", "--gpu-architecture")):
            continue
        if arg.startswith("--Werror"):
            continue
        kept.append(arg)
    return kept + [f"-arch={virtual_arch}", "-ptx", "-o", str(out)]


def blocks_per_sm(registers: int) -> tuple[int, int, int, int, str]:
    """Blocks resident per SM, and which resource is the one that binds."""
    warps = THREADS_PER_BLOCK // 32
    warps_rounded = math.ceil(warps / WARP_ALLOC_GRANULARITY) * WARP_ALLOC_GRANULARITY
    regs_per_warp = math.ceil(registers * 32 / REG_ALLOC_UNIT) * REG_ALLOC_UNIT
    regs_per_block = regs_per_warp * warps_rounded

    by_registers = REGS_PER_SM // regs_per_block if regs_per_block else MAX_BLOCKS_PER_SM
    by_shared = SHARED_PER_SM // SHARED_BYTES if SHARED_BYTES else MAX_BLOCKS_PER_SM
    by_warps = WARPS_PER_SM // warps

    limits = {
        "registers": by_registers,
        "shared_memory": by_shared,
        "warps": by_warps,
        "blocks_per_sm": MAX_BLOCKS_PER_SM,
    }
    blocks = min(limits.values())
    # Ties are reported as one name in a fixed order, so two runs of the same
    # numbers never disagree about which resource to blame.
    limiter = next(k for k in ("shared_memory", "registers", "warps", "blocks_per_sm")
                   if limits[k] == blocks)
    return by_registers, by_shared, by_warps, blocks, limiter


def parse_ptxas(text: str) -> dict[str, int] | None:
    """The resource report for the one kernel under study, or None."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if KERNEL_MARK not in line or "Function properties" not in line:
            continue
        result: dict[str, int] = {}
        props = PROPS_RE.search(lines[i + 1]) if i + 1 < len(lines) else None
        if props:
            result["stack_frame_bytes"] = int(props.group(1))
            result["spill_store_bytes"] = int(props.group(2))
            result["spill_load_bytes"] = int(props.group(3))
        for later in lines[i + 1 : i + 6]:
            used = USED_RE.search(later)
            if used:
                result["registers"] = int(used.group(1))
                barriers = BARRIER_RE.search(later)
                result["barriers"] = int(barriers.group(1)) if barriers else 0
                break
        if "registers" in result:
            return result
    return None


def toolkit_version() -> str:
    out = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, check=False)
    match = re.search(r"release [0-9.]+, V([0-9.]+)", out.stdout)
    return match.group(1) if match else "unknown"


def assemble(ptx: Path, arch: str, maxrreg: int | None) -> dict[str, int]:
    cmd = ["ptxas", f"-arch={arch}", "-O3", "-v", str(ptx), "-o", "/dev/null"]
    if maxrreg is not None:
        cmd.append(f"-maxrregcount={maxrreg}")
    run = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if run.returncode != 0:
        print(
            f"register_study: ptxas failed for {ptx.name} maxrregcount={maxrreg}",
            file=sys.stderr,
        )
        print(run.stderr.rstrip()[-2000:], file=sys.stderr)
        raise SystemExit(2)
    report = parse_ptxas(run.stderr)
    if report is None:
        print(f"register_study: ptxas said nothing about {KERNEL_MARK}", file=sys.stderr)
        raise SystemExit(2)
    return report


def row(axis: str, maxrreg: int | None, mincta: int, report: dict[str, int],
        arch: str, toolkit: str) -> dict[str, object]:
    regs = report["registers"]
    by_reg, by_shared, by_warps, blocks, limiter = blocks_per_sm(regs)
    return {
        "kernel": "gemm_mma_opt_kernel<false>",
        "study_axis": axis,
        "arch": arch,
        "toolkit": toolkit,
        "maxrregcount": "none" if maxrreg is None else maxrreg,
        "minnctapersm": mincta,
        "registers": regs,
        "barriers": report.get("barriers", 0),
        "stack_frame_bytes": report.get("stack_frame_bytes", 0),
        "spill_store_bytes": report.get("spill_store_bytes", 0),
        "spill_load_bytes": report.get("spill_load_bytes", 0),
        "threads_per_block": THREADS_PER_BLOCK,
        "shared_bytes_per_block": SHARED_BYTES,
        "blocks_by_registers": by_reg,
        "blocks_by_shared": by_shared,
        "blocks_by_warps": by_warps,
        "blocks_per_sm": blocks,
        "limiter": limiter,
        "theoretical_occupancy_pct": round(
            100.0 * blocks * (THREADS_PER_BLOCK // 32) / WARPS_PER_SM, 1
        ),
        "gflops": "pending",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--build-dir",
        default=str(REPO / "build"),
        help="build tree whose compile_commands.json supplies the flags",
    )
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="where the CSV goes")
    parser.add_argument("--arch", default="sm_120", help="target ptxas assembles for")
    parser.add_argument("--virtual-arch", default="compute_120", help="PTX target")
    args = parser.parse_args()

    build_dir = Path(args.build_dir).resolve()
    base, cwd = compile_command(build_dir)
    toolkit = toolkit_version()

    rows: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        shipped = tmpdir / "shipped.ptx"
        run = subprocess.run(
            ptx_command(base, args.virtual_arch, shipped),
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
        )
        if run.returncode != 0 or not shipped.exists():
            print("register_study: could not emit PTX for the kernel.", file=sys.stderr)
            print(run.stderr.rstrip()[-2000:], file=sys.stderr)
            return 2

        text = shipped.read_text(encoding="utf-8")
        if not MAXNTID_RE.search(text):
            print(
                "register_study: the PTX carries no .maxntid 256 directive, so the "
                "two axes cannot be separated. The kernel's __launch_bounds__ has "
                "changed and this script needs to change with it.",
                file=sys.stderr,
            )
            return 2

        # Axis one: the register ceiling, with the thread ceiling out of the way
        # so ptxas honors it.
        unbound = tmpdir / "no_launch_bounds.ptx"
        unbound.write_text(MAXNTID_RE.sub("", text), encoding="utf-8")
        for maxrreg in MAXRREG_VALUES:
            report = assemble(unbound, args.arch, maxrreg)
            rows.append(row("maxrregcount", maxrreg, 1, report, args.arch, toolkit))

        # Axis two: the blocks per SM floor, which is what the kernel would carry
        # if its __launch_bounds__ named a second parameter.
        for mincta in MINCTA_VALUES:
            if mincta == 1:
                variant = shipped
            else:
                variant = tmpdir / f"min_cta_{mincta}.ptx"
                floor = f"\\g<0>\n.minnctapersm {mincta}"
                variant.write_text(MAXNTID_RE.sub(floor, text), encoding="utf-8")
            report = assemble(variant, args.arch, None)
            rows.append(row("launch_bounds", None, mincta, report, args.arch, toolkit))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"register study written to {out_path}: {len(rows)} rows")
    print(
        f"{'axis':>13} {'maxrreg':>8} {'minCTA':>7} {'regs':>5} {'spillst':>8} "
        f"{'spillld':>8} {'blocks':>7}  limiter"
    )
    for r in rows:
        print(
            f"{r['study_axis']:>13} {str(r['maxrregcount']):>8} {r['minnctapersm']:>7} "
            f"{r['registers']:>5} {r['spill_store_bytes']:>8} {r['spill_load_bytes']:>8} "
            f"{r['blocks_per_sm']:>7}  {r['limiter']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
