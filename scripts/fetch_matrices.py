#!/usr/bin/env python3
"""Fetch the SuiteSparse SpMV suite, convert it to binary CSR, and checksum it.

The suite is described in experiments/matrices.toml. Nothing it downloads enters
the repository: archives and converted matrices live under $CKL_MATRIX_CACHE,
which defaults to ~/.cache/ckl/matrices, and the only tracked artifact is
experiments/matrix_manifest.csv.

What a manifest row asserts is the whole point of the file. It records the
group, the confirmed shape read out of the MatrixMarket header, the confirmed
nonzero count after symmetric expansion and duplicate summing, and the SHA-256
of both the archive as downloaded and the binary CSR as written. `--verify`
rechecks every one of those and exits non-zero on a mismatch, so a matrix that
was silently re-downloaded from a changed mirror cannot quietly become the
matrix a committed benchmark row names.

The shapes in matrices.toml are the SuiteSparse index values and are treated as
a claim to be checked, not as data. A conversion whose confirmed shape or
nonzero count disagrees with the claim fails rather than writing the row.

The binary format, little endian throughout:

    char[8]   "CKLCSR01"
    int32     m, n, nnz
    int32[m+1] row_ptr
    int32[nnz] col_idx
    float32[nnz] values

Columns inside a row come out sorted and duplicate columns are summed, which is
what makes cuSPARSE a legal oracle on these matrices.

Usage:

    python3 scripts/fetch_matrices.py                 # the six small matrices
    python3 scripts/fetch_matrices.py --include-large # adds soc-LiveJournal1
    python3 scripts/fetch_matrices.py --only cant --only scircuit
    python3 scripts/fetch_matrices.py --verify
    python3 scripts/fetch_matrices.py --list
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import shutil
import sys
import tarfile
import tomllib
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
SPEC = REPO / "experiments" / "matrices.toml"
MANIFEST = REPO / "experiments" / "matrix_manifest.csv"

MAGIC = b"CKLCSR01"
PENDING = "pending"

MANIFEST_COLUMNS = [
    "name", "group", "class", "rows", "cols", "nnz",
    "archive_sha256", "csr_sha256", "archive_bytes", "csr_bytes", "status", "note",
]

# A matrix this large is worth a long wait but not an unbounded one. The row is
# recorded pending instead of stalling the tool, per the Part 06 instructions.
LARGE_TIMEOUT_S = 1800
SMALL_TIMEOUT_S = 300


def cache_dir() -> Path:
    env = os.environ.get("CKL_MATRIX_CACHE", "").strip()
    if env:
        return Path(env).expanduser()
    return Path.home() / ".cache" / "ckl" / "matrices"


def load_spec(path: Path = SPEC) -> dict:
    with path.open("rb") as f:
        return tomllib.load(f)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


def download(url: str, dest: Path, timeout: float) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "ckl-fetch-matrices/1.1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as response, tmp.open("wb") as out:
        shutil.copyfileobj(response, out, length=1 << 20)
    tmp.replace(dest)


def fetch_archive(entry: dict, source: dict, cache: Path, timeout: float,
                  force: bool) -> tuple[Path | None, str]:
    """Download the archive, trying the primary host then the mirror."""
    name = entry["name"]
    dest = cache / f"{name}.tar.gz"
    if dest.exists() and not force:
        return dest, "cached"
    errors = []
    for base in (source["base_url"], source["mirror_url"]):
        url = f"{base}/{entry['group']}/{name}.tar.gz"
        try:
            download(url, dest, timeout)
            return dest, f"downloaded from {base}"
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            errors.append(f"{base}: {exc}")
    return None, "; ".join(errors)


# ---------------------------------------------------------------------------
# MatrixMarket to CSR
# ---------------------------------------------------------------------------


class ConversionError(RuntimeError):
    pass


def extract_mtx(archive: Path, name: str, cache: Path) -> Path:
    """Pull the matrix .mtx out of the archive into the cache.

    Several SuiteSparse archives ship a right hand side beside the matrix, as
    <name>_b.mtx, so the member is picked by exact name rather than by suffix.
    """
    with tarfile.open(archive, "r:gz") as tar:
        members = [m for m in tar.getmembers() if m.isfile() and m.name.endswith(".mtx")]
        exact = [m for m in members if Path(m.name).name == f"{name}.mtx"]
        if len(exact) != 1:
            found = ", ".join(sorted(Path(m.name).name for m in members)) or "none"
            raise ConversionError(
                f"{archive.name} holds no single member named {name}.mtx; it holds {found}")
        member = exact[0]
        out = cache / Path(member.name).name
        source = tar.extractfile(member)
        if source is None:
            raise ConversionError(f"{archive.name}: {member.name} could not be read")
        with source, out.open("wb") as sink:
            shutil.copyfileobj(source, sink, length=1 << 20)
    return out


def read_header(handle) -> tuple[str, str, int, int, int]:
    """Banner field and symmetry, then the dimension line."""
    banner = handle.readline().decode("ascii", "replace").strip()
    if not banner.startswith("%%MatrixMarket"):
        raise ConversionError("not a MatrixMarket file: missing the %%MatrixMarket banner")
    parts = banner.split()
    if len(parts) < 5 or parts[1] != "matrix" or parts[2] != "coordinate":
        raise ConversionError(f"only coordinate matrices are supported, banner reads: {banner}")
    field, symmetry = parts[3].lower(), parts[4].lower()
    line = handle.readline()
    while line.startswith(b"%"):
        line = handle.readline()
    dims = line.split()
    if len(dims) != 3:
        raise ConversionError(f"malformed dimension line: {line!r}")
    return field, symmetry, int(dims[0]), int(dims[1]), int(dims[2])


def read_entries(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int, str]:
    """Row, column and value arrays of the stored triangle, plus the shape."""
    with path.open("rb") as handle:
        field, symmetry, rows, cols, stored = read_header(handle)
        columns = 2 if field == "pattern" else 3
        table = np.loadtxt(handle, dtype=np.float64, usecols=range(columns), ndmin=2)
    if table.shape[0] != stored:
        raise ConversionError(
            f"header claims {stored} stored entries, the body holds {table.shape[0]}")
    r = table[:, 0].astype(np.int64) - 1
    c = table[:, 1].astype(np.int64) - 1
    v = np.ones(r.size, dtype=np.float64) if columns == 2 else table[:, 2]
    return r, c, v, rows, cols, symmetry


def expand_symmetry(r: np.ndarray, c: np.ndarray, v: np.ndarray,
                    symmetry: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mirror the stored triangle, which is what the index nnz counts."""
    if symmetry in ("general", "unsymmetric"):
        return r, c, v
    off = r != c
    if symmetry in ("symmetric", "hermitian"):
        mirror_v = v[off]
    elif symmetry == "skew-symmetric":
        mirror_v = -v[off]
    else:
        raise ConversionError(f"unknown symmetry {symmetry}")
    return (np.concatenate([r, c[off]]),
            np.concatenate([c, r[off]]),
            np.concatenate([v, mirror_v]))


def to_csr(r: np.ndarray, c: np.ndarray, v: np.ndarray,
           rows: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sorted by (row, column) with duplicate columns summed."""
    order = np.lexsort((c, r))
    r, c, v = r[order], c[order], v[order]
    if r.size:
        first = np.empty(r.size, dtype=bool)
        first[0] = True
        np.not_equal(r[1:], r[:-1], out=first[1:])
        np.logical_or(first[1:], c[1:] != c[:-1], out=first[1:])
        groups = np.cumsum(first) - 1
        # bincount with weights is the vectorized segmented sum; np.add.at does
        # the same thing one element at a time and takes minutes on 4M entries.
        summed = np.bincount(groups, weights=v, minlength=int(groups[-1]) + 1)
        keep = np.flatnonzero(first)
        r, c, v = r[keep], c[keep], summed
    counts = np.bincount(r, minlength=rows)
    row_ptr = np.zeros(rows + 1, dtype=np.int64)
    np.cumsum(counts, out=row_ptr[1:])
    return row_ptr, c, v


def write_csr(path: Path, m: int, n: int, row_ptr: np.ndarray, col_idx: np.ndarray,
              values: np.ndarray) -> None:
    tmp = path.with_suffix(path.suffix + ".part")
    with tmp.open("wb") as f:
        f.write(MAGIC)
        np.asarray([m, n, int(col_idx.size)], dtype="<i4").tofile(f)
        row_ptr.astype("<i4").tofile(f)
        col_idx.astype("<i4").tofile(f)
        values.astype("<f4").tofile(f)
    tmp.replace(path)


def convert(archive: Path, cache: Path, entry: dict) -> tuple[Path, int, int, int]:
    """Archive to binary CSR, with the shape confirmed against matrices.toml."""
    mtx = extract_mtx(archive, entry["name"], cache)
    try:
        r, c, v, rows, cols, symmetry = read_entries(mtx)
        r, c, v = expand_symmetry(r, c, v, symmetry)
        row_ptr, col_idx, values = to_csr(r, c, v, rows)
    finally:
        mtx.unlink(missing_ok=True)
    nnz = int(col_idx.size)
    claimed = (int(entry["rows"]), int(entry["cols"]), int(entry["nnz"]))
    if (rows, cols) != claimed[:2]:
        raise ConversionError(
            f"matrices.toml claims {claimed[0]} by {claimed[1]}, the file holds {rows} by {cols}")
    if nnz != claimed[2]:
        raise ConversionError(
            f"matrices.toml claims {claimed[2]} nonzeros, the converted CSR holds {nnz}")
    out = cache / f"{entry['name']}.csr.bin"
    write_csr(out, rows, cols, row_ptr, col_idx, values)
    return out, rows, cols, nnz


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def read_manifest(path: Path = MANIFEST) -> dict[str, dict]:
    if not path.exists():
        return {}
    with path.open(newline="") as f:
        return {row["name"]: row for row in csv.DictReader(f)}


def write_manifest(rows: list[dict], path: Path = MANIFEST) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def pending_row(entry: dict, note: str) -> dict:
    """A row for a matrix that is in the suite but not on this machine."""
    return {
        "name": entry["name"],
        "group": entry["group"],
        "class": entry["klass"],
        "rows": entry["rows"],
        "cols": entry["cols"],
        "nnz": entry["nnz"],
        "archive_sha256": PENDING,
        "csr_sha256": PENDING,
        "archive_bytes": PENDING,
        "csr_bytes": PENDING,
        "status": PENDING,
        "note": note,
    }


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def do_fetch(spec: dict, cache: Path, args) -> int:
    entries = spec["matrix"]
    if args.only:
        entries = [e for e in entries if e["name"] in args.only]
        if not entries:
            print(f"no matrix in {rel(SPEC)} is named {', '.join(args.only)}", file=sys.stderr)
            return 2
    cache.mkdir(parents=True, exist_ok=True)
    existing = read_manifest(args.manifest)
    rows: list[dict] = []
    failures = 0

    for entry in spec["matrix"]:
        name = entry["name"]
        if entry not in entries:
            rows.append(existing.get(name, pending_row(entry, "not selected by --only")))
            continue
        large = bool(entry.get("large", False))
        if large and not args.include_large:
            rows.append(pending_row(
                entry, "large; run with --include-large to fetch it"))
            print(f"{name}: pending, {entry['nnz']} nonzeros is a long download; "
                  f"pass --include-large to fetch it")
            continue

        timeout = args.timeout if args.timeout else (LARGE_TIMEOUT_S if large
                                                     else SMALL_TIMEOUT_S)
        archive, how = fetch_archive(entry, spec["source"], cache, timeout, args.force)
        if archive is None:
            rows.append(pending_row(entry, f"download failed: {how}"[:200]))
            print(f"{name}: pending, every source refused: {how}", file=sys.stderr)
            if not large:
                failures += 1
            continue

        binary = cache / f"{name}.csr.bin"
        if binary.exists() and not args.force:
            print(f"{name}: {how}, converted CSR already in the cache")
            confirmed = (int(entry["rows"]), int(entry["cols"]), int(entry["nnz"]))
        else:
            print(f"{name}: {how}, converting")
            try:
                binary, *confirmed_list = convert(archive, cache, entry)
                confirmed = tuple(confirmed_list)
            except (ConversionError, OSError, ValueError) as exc:
                rows.append(pending_row(entry, f"conversion failed: {exc}"[:200]))
                print(f"{name}: conversion FAILED: {exc}", file=sys.stderr)
                failures += 1
                continue

        rows.append({
            "name": name,
            "group": entry["group"],
            "class": entry["klass"],
            "rows": confirmed[0],
            "cols": confirmed[1],
            "nnz": confirmed[2],
            "archive_sha256": sha256_of(archive),
            "csr_sha256": sha256_of(binary),
            "archive_bytes": archive.stat().st_size,
            "csr_bytes": binary.stat().st_size,
            "status": "ok",
            "note": "shape and nonzero count confirmed against the MatrixMarket file",
        })
        print(f"{name}: ok, {confirmed[0]} by {confirmed[1]}, {confirmed[2]} nonzeros")

    write_manifest(rows, args.manifest)
    print(f"wrote {rel(args.manifest)} ({len(rows)} rows)")
    return 1 if failures else 0


def do_verify(spec: dict, cache: Path, args) -> int:
    rows = read_manifest(args.manifest)
    if not rows:
        print(f"{rel(args.manifest)} has no rows; run the fetch first", file=sys.stderr)
        return 1
    known = {e["name"]: e for e in spec["matrix"]}
    bad: list[str] = []
    skipped: list[str] = []
    checked = 0

    for name, row in rows.items():
        if name not in known:
            bad.append(f"{name}: in the manifest but not in {rel(SPEC)}")
            continue
        if row.get("status") != "ok" or row.get("csr_sha256") == PENDING:
            skipped.append(f"{name}: {row.get('note') or row.get('status')}")
            continue
        for kind, path in (("archive", cache / f"{name}.tar.gz"),
                           ("csr", cache / f"{name}.csr.bin")):
            expected = row[f"{kind}_sha256"]
            if not path.exists():
                bad.append(f"{name}: the {kind} is not in {cache}")
                continue
            actual = sha256_of(path)
            if actual != expected:
                bad.append(f"{name}: the {kind} hashes {actual[:16]}, "
                           f"the manifest says {expected[:16]}")
            else:
                checked += 1
        entry = known[name]
        for field in ("rows", "cols", "nnz"):
            if int(row[field]) != int(entry[field]):
                bad.append(f"{name}: manifest {field} is {row[field]}, "
                           f"{rel(SPEC)} says {entry[field]}")

    for name in known:
        if name not in rows:
            bad.append(f"{name}: in {rel(SPEC)} but missing from the manifest")

    for line in skipped:
        print(f"warning: skipped {line}", file=sys.stderr)
    if bad:
        print("verify FAILED:", file=sys.stderr)
        for line in bad:
            print("  " + line, file=sys.stderr)
        return 1
    print(f"verify clean: {checked} checksum(s) match, {len(skipped)} row(s) pending and skipped")
    return 0


def do_list(spec: dict, cache: Path, args) -> int:
    rows = read_manifest(args.manifest)
    print(f"cache: {cache}")
    print(f"{'matrix':20s} {'group':12s} {'class':20s} {'rows':>10s} {'nnz':>10s}  status")
    for entry in spec["matrix"]:
        row = rows.get(entry["name"], {})
        status = row.get("status", "not in the manifest")
        cached = (cache / f"{entry['name']}.csr.bin").exists()
        print(f"{entry['name']:20s} {entry['group']:12s} {entry['klass']:20s} "
              f"{entry['rows']:>10d} {entry['nnz']:>10d}  {status}"
              f"{'' if cached else ' (not cached)'}")
    syn = spec["synthetic"]
    print(f"{syn['name']:20s} {'generated':12s} {syn['klass']:20s} "
          f"{syn['rows']:>10d} {'varies':>10s}  built in process, seeds "
          f"{', '.join(str(s) for s in syn['seeds'])}")
    return 0


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--verify", action="store_true",
                    help="recheck every manifest checksum and exit non-zero on a mismatch")
    ap.add_argument("--list", action="store_true", help="print the suite and its cache state")
    ap.add_argument("--only", action="append", default=[], metavar="NAME",
                    help="restrict the fetch to this matrix; repeatable")
    ap.add_argument("--include-large", action="store_true",
                    help="also fetch the matrices marked large in matrices.toml")
    ap.add_argument("--force", action="store_true",
                    help="re-download and re-convert even when the cache holds the file")
    ap.add_argument("--timeout", type=float, default=0.0, metavar="S",
                    help="per request timeout; the default is 300s, or 1800s for a large matrix")
    ap.add_argument("--cache", type=Path, default=None,
                    help="cache directory; the default is $CKL_MATRIX_CACHE or "
                         "~/.cache/ckl/matrices")
    ap.add_argument("--spec", type=Path, default=SPEC, help="path to matrices.toml")
    ap.add_argument("--manifest", type=Path, default=MANIFEST, help="path to the manifest")
    args = ap.parse_args()

    spec = load_spec(args.spec)
    cache = args.cache.expanduser() if args.cache else cache_dir()

    if args.list:
        return do_list(spec, cache, args)
    if args.verify:
        return do_verify(spec, cache, args)
    return do_fetch(spec, cache, args)


if __name__ == "__main__":
    raise SystemExit(main())
