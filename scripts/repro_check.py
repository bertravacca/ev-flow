#!/usr/bin/env python3
"""Reproducibility check (F28): same seed -> identical cache.

Regenerates a ``(region, profile_type)`` cache **twice** into two independent
data roots (via ``ev-flow bootstrap``, which honours ``--data-root``) and
asserts the data parquets are content-identical. Exits non-zero on any
mismatch, naming the artifact that differed.

NHTS raw microdata is anchored at ``<repo_root>/data`` regardless of the data
root, so it is downloaded/reused once and shared across both builds; only the
offline sales-mix CSVs and the generated cache live under each data root.

Usage:
    python scripts/repro_check.py --region bay_area --profile-type residential --n 150
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path

# Only the deterministic data artifacts are compared. meta.json (git hash),
# bic_sweep_*.json, and the validation reports carry build-time/provenance
# fields and are intentionally excluded.
DATA_ARTIFACTS = [
    "fleet.parquet",
    "mobility.parquet",
    "plug_in_events.parquet",
    "sessions.parquet",
    "plug_status_15min.parquet",
    "plug_status_hourly.parquet",
]


def _content_hash(parquet_path: Path) -> str:
    """Order-sensitive content hash of a parquet's rows (ignores file metadata)."""
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    row_hashes = pd.util.hash_pandas_object(df, index=True).to_numpy()
    return hashlib.sha256(row_hashes.tobytes()).hexdigest()


def _bootstrap(root: Path, region: str, profile_type: str, n: int, seed: int | None) -> Path:
    """Build the cache for one (region, profile_type) under ``root``; return its dir."""
    cmd = [
        sys.executable, "-m", "pev_synth.cli", "bootstrap",
        "--data-root", str(root),
        "--regions", region,
        "--profile-types", profile_type,
        "--n", str(n),
    ]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    print(f"  $ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    return root / "pev" / "processed" / region / f"{profile_type}_ev_synth"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="repro_check", description=__doc__)
    parser.add_argument("--region", default="bay_area")
    parser.add_argument("--profile-type", default="residential")
    parser.add_argument("--n", type=int, default=150)
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Seed for both builds. Default: the pipeline's MASTER_SEED.",
    )
    args = parser.parse_args(argv)

    with tempfile.TemporaryDirectory(prefix="evflow-repro-a-") as a, \
            tempfile.TemporaryDirectory(prefix="evflow-repro-b-") as b:
        print(f"[1/2] building {args.region}/{args.profile_type} (n={args.n}) -> run A")
        cache_a = _bootstrap(Path(a), args.region, args.profile_type, args.n, args.seed)
        print(f"[2/2] building {args.region}/{args.profile_type} (n={args.n}) -> run B")
        cache_b = _bootstrap(Path(b), args.region, args.profile_type, args.n, args.seed)

        print("\nComparing data artifacts:")
        mismatches: list[str] = []
        for name in DATA_ARTIFACTS:
            fa, fb = cache_a / name, cache_b / name
            if not fa.is_file() or not fb.is_file():
                print(f"  MISSING  {name} (a={fa.is_file()} b={fb.is_file()})")
                mismatches.append(name)
                continue
            ha, hb = _content_hash(fa), _content_hash(fb)
            ok = ha == hb
            status = "OK     " if ok else "DIFFER "
            op = "==" if ok else "!="
            print(f"  {status} {name}  {ha[:12]} {op} {hb[:12]}")
            if not ok:
                mismatches.append(name)

    if mismatches:
        print(f"\nNOT REPRODUCIBLE: {len(mismatches)} artifact(s) differ: {', '.join(mismatches)}")
        return 1
    print(f"\nREPRODUCIBLE: all {len(DATA_ARTIFACTS)} data artifacts identical across two builds.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
