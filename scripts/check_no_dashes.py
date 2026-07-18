#!/usr/bin/env python3
"""Fail if any tracked text file contains an en dash or em dash.

Ground rule 1 of the build spec: no en dash (U+2013) and no em dash (U+2014)
anywhere in the repository, in any language or document. LaTeX has an extra
trap because "--" and "---" typeset as dashes, so for .tex files this script
also flags literal double and triple hyphen runs in prose (verbatim and
listing blocks are skipped).

Exit code 0 means clean, 1 means at least one hit was found. Wired into
`make check-style` and the report build.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

EN_DASH = chr(0x2013)
EM_DASH = chr(0x2014)

# Extensions treated as text we control and must keep dash free.
TEXT_SUFFIXES = {
    ".md", ".tex", ".bib", ".py", ".sh", ".cu", ".cuh", ".cpp", ".hpp",
    ".h", ".c", ".cmake", ".txt", ".yml", ".yaml", ".json", ".toml",
    ".cfg", ".ini", ".clang-format", ".clang-tidy",
}
# Files without a suffix that we still want scanned.
TEXT_NAMES = {"Makefile", "CMakeLists.txt", "LICENSE", "CHANGELOG"}

# LaTeX environments where a literal "--" is legitimate (code, urls).
TEX_VERBATIM_STARTS = ("\\begin{verbatim}", "\\begin{lstlisting}", "\\begin{minted}")
TEX_VERBATIM_ENDS = ("\\end{verbatim}", "\\end{lstlisting}", "\\end{minted}")


def tracked_files(root: Path) -> list[Path]:
    """Return files git tracks, or every candidate file when not in a repo yet."""
    try:
        out = subprocess.run(
            ["git", "ls-files"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        )
        names = [line for line in out.stdout.splitlines() if line.strip()]
        return [root / n for n in names]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return [p for p in root.rglob("*") if p.is_file()]


def is_text(path: Path) -> bool:
    if path.suffix in TEXT_SUFFIXES:
        return True
    return path.name in TEXT_NAMES


def scan_unicode(path: Path, text: str) -> list[str]:
    hits = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        for col, ch in enumerate(line, start=1):
            if ch == EN_DASH:
                hits.append(f"{path}:{lineno}:{col}: en dash (U+2013)")
            elif ch == EM_DASH:
                hits.append(f"{path}:{lineno}:{col}: em dash (U+2014)")
    return hits


def scan_tex_hyphen_runs(path: Path, text: str) -> list[str]:
    hits = []
    in_verbatim = False
    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = line.lstrip()
        if stripped.startswith(TEX_VERBATIM_STARTS):
            in_verbatim = True
            continue
        if stripped.startswith(TEX_VERBATIM_ENDS):
            in_verbatim = False
            continue
        if in_verbatim or stripped.startswith("%"):
            continue
        if "---" in line or "--" in line:
            hits.append(f"{path}:{lineno}: literal double or triple hyphen in .tex prose")
    return hits


def main() -> int:
    root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path.cwd()
    all_hits: list[str] = []
    for path in tracked_files(root):
        if not path.is_file() or not is_text(path):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        all_hits.extend(scan_unicode(path, text))
        if path.suffix == ".tex":
            all_hits.extend(scan_tex_hyphen_runs(path, text))

    if all_hits:
        print("dash check FAILED:")
        for hit in all_hits:
            print("  " + hit)
        print(f"\n{len(all_hits)} hit(s). Replace with ASCII or reword ranges as 'A to B'.")
        return 1
    print("dash check clean: no en dash, no em dash, no .tex hyphen runs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
