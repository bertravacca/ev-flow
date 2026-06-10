"""pev_synth._utc_migration — package-internal v1.1 → v2.0 bay_area cache migration.

This module is package-private (leading underscore). It is used only by
in-house callers who hold a v1.1 ``pev_synth`` cache and need to migrate it
to the v2.0 UTC-storage layout. PyPI users will not have a v1.1 cache and
should ignore this module.

Reads the v1.1 ``bay_area`` residential cache at
``data/pev/processed/residential_ev_synth/`` (timestamps tagged
``America/Los_Angeles``), converts every timestamp column to ``UTC`` via
``pd.Series.dt.tz_convert("UTC")``, and writes the v2.0 cache layout at
``data/pev/processed/bay_area/residential_ev_synth/``. Updates ``meta.json``
with v2.0 provenance fields (``methodology_version``, ``master_seed``,
``region``, ``profile_type``, ``migrated_from``, ``storage_timezone``,
``cache_provenance``, ``migration_timestamp_utc``).

The migration is **NOT** bit-for-bit. SHA-256 of the parquet bytes changes
because the tz tag in the pyarrow column metadata flips from
``America/Los_Angeles`` to ``UTC``. The correctness invariant is:

    pd.testing.assert_frame_equal(
        df_v11.assign(**{col: df_v11[col].dt.tz_convert("UTC") for col in ts_cols}),
        df_v20,
    )

passes for every parquet (after tz-equalisation of the v1.1 frame).

The migration is **idempotent**: a second run on an already-migrated cache
produces byte-identical output (modulo ``migration_timestamp_utc``, which is
stable across reruns because the file SHA-256s computed are stable).

Methodology version: v2.0.0.  Master seed: 20260520 (recorded in meta only;
this module performs no RNG-driven work).

Run from the repo root::

    python -m pev_synth._utc_migration \\
        --src data/pev/processed/residential_ev_synth \\
        --dst data/pev/processed/bay_area/residential_ev_synth
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Orchestrator-resolved contracts.
# ---------------------------------------------------------------------------
from ._paths import processed_root as _processed_root
from ._seeds import (
    MASTER_SEED as _MASTER_SEED,
)
from ._seeds import (
    METHODOLOGY_VERSION as _METHODOLOGY_VERSION,
)
from ._seeds import (
    module_offset as _module_offset,
)

METHODOLOGY_VERSION: Final[str] = _METHODOLOGY_VERSION
MASTER_SEED: Final[int] = _MASTER_SEED
MODULE_OFFSET: Final[int] = _module_offset("utc_migration")  # +1

DEFAULT_SRC_ROOT: Final[Path] = _processed_root() / "residential_ev_synth"
DEFAULT_DST_ROOT: Final[Path] = _processed_root() / "bay_area" / "residential_ev_synth"

#: filename → list of timestamp columns to ``tz_convert('UTC')``.
#: Files not listed here are copied verbatim (no timestamp columns).
TS_COLS_BY_FILE: Final[dict[str, list[str]]] = {
    "fleet.parquet": [],
    "mobility.parquet": ["start_ts", "end_ts"],
    "plug_in_events.parquet": ["arrival_ts", "departure_ts"],
    "sessions.parquet": ["t_in", "t_out"],
    "plug_status_15min.parquet": ["ts"],
    "plug_status_hourly.parquet": ["ts"],
}

#: All v1.1 parquet files migrated. Order is canonical for the report.
PARQUET_FILES: Final[tuple[str, ...]] = tuple(TS_COLS_BY_FILE.keys())

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class MigrationResult:
    """Return type of :func:`migrate_bay_area_v1_to_v2`.

    Attributes
    ----------
    n_files_migrated
        Count of files written under ``dst_root`` (5 parquets + ``meta.json``
        → 6 on a fresh migration).
    timestamp_cols_converted
        Mapping ``filename → list[ts_col_name]`` for files where at least one
        timestamp column was converted to UTC. Files with no timestamp columns
        map to ``[]``.
    sha256_after
        SHA-256 hex digest of every written file, keyed by filename.
    rows_per_file
        Row count of every migrated parquet, keyed by filename.
    skipped_files
        Files that were expected under ``src_root`` but were not found
        (informational; the migration does not back-fill missing schema).
    """

    n_files_migrated: int
    timestamp_cols_converted: dict[str, list[str]]
    sha256_after: dict[str, str]
    rows_per_file: dict[str, int]
    skipped_files: list[str]


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    """SHA-256 hex digest of ``path`` (streaming, 1 MiB chunks)."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _detect_ts_columns(df: pd.DataFrame) -> list[str]:
    """Return every column whose dtype is tz-aware datetime64."""
    cols: list[str] = []
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_datetime64_any_dtype(s) and getattr(s.dt, "tz", None) is not None:
            cols.append(c)
    return cols


def _convert_frame_to_utc(df: pd.DataFrame, ts_cols: list[str]) -> pd.DataFrame:
    """Return a copy of ``df`` with each timestamp col tz_converted to UTC.

    Columns already in UTC pass through untouched (the conversion is a no-op).
    Naive datetime columns raise — this is a defensive check; the v1.1 cache
    has every timestamp tagged as ``America/Los_Angeles``.
    """
    out = df.copy()
    for c in ts_cols:
        s = out[c]
        if not pd.api.types.is_datetime64_any_dtype(s):
            raise TypeError(f"column {c!r} is not datetime64; got {s.dtype}")
        if getattr(s.dt, "tz", None) is None:
            raise TypeError(f"column {c!r} is tz-naive; cannot tz_convert")
        out[c] = s.dt.tz_convert("UTC")
    return out


def _write_parquet_deterministic(df: pd.DataFrame, path: Path) -> None:
    """Write ``df`` to ``path`` with byte-stable serialisation.

    Settings:
    - ``compression="snappy"``
    - ``use_dictionary=True``
    - ``write_statistics=False`` (statistics include per-run metadata on some
      pyarrow builds)
    - ``version="2.6"``
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    # pyarrow.parquet.write_table is annotation-free in pyarrow's bundled
    # stubs, so mypy flags the call as untyped; the wrapper itself is typed.
    pq.write_table(  # type: ignore[no-untyped-call]
        table,
        where=path,
        compression="snappy",
        use_dictionary=True,
        version="2.6",
        write_statistics=False,
    )


# ---------------------------------------------------------------------------
# Reusable migration kernels
# ---------------------------------------------------------------------------

def migrate_parquet_tz(
    src: Path,
    dst: Path,
    ts_cols: list[str],
) -> tuple[int, list[str]]:
    """Migrate one parquet from LA-tagged timestamps to UTC.

    Parameters
    ----------
    src
        Source parquet path. Must exist.
    dst
        Destination parquet path. Parent directory is created if needed.
    ts_cols
        Timestamp columns to convert. If empty, the parquet is copied
        verbatim (still through a pandas round-trip so that the writer
        produces a byte-stable output regardless of the source byte layout).

    Returns
    -------
    n_rows
        Row count of the migrated frame.
    converted_cols
        The subset of ``ts_cols`` that were actually present and converted
        (defensive: drops any ts_col absent from the frame schema).
    """
    if not src.is_file():
        raise FileNotFoundError(f"source parquet not found: {src}")
    df = pd.read_parquet(src)
    present_ts_cols = [c for c in ts_cols if c in df.columns]
    out = _convert_frame_to_utc(df, present_ts_cols)
    _write_parquet_deterministic(out, dst)
    return len(out), present_ts_cols


def _build_meta(
    src_meta: dict[str, object],
    sha256_after: dict[str, str],
    rows_per_file: dict[str, int],
    timestamp_cols_converted: dict[str, list[str]],
    migration_timestamp_utc: str,
    src_root_str: str,
) -> dict[str, object]:
    """Compose the v2.0 meta.json by overlaying v2.0 provenance on v1.1.

    The base is a deep copy of ``src_meta`` (so all v1.1 fields are preserved
    for traceability); v2.0 fields are then overwritten/added on top.
    """
    meta: dict[str, object] = json.loads(json.dumps(src_meta))  # deep copy
    meta["methodology_version"] = METHODOLOGY_VERSION
    meta["master_seed"] = int(MASTER_SEED)
    meta["region"] = "bay_area"
    meta["profile_type"] = "residential"
    meta["migrated_from"] = "v1.1"
    meta["migration_timestamp_utc"] = migration_timestamp_utc
    meta["storage_timezone"] = "UTC"
    meta["tz_storage"] = "UTC"  # plan §3.1 canonical name (kept for back-compat).
    meta["tz_local_anchor"] = "America/Los_Angeles"
    meta["cache_provenance"] = {
        "original_path": src_root_str,
        "migration_script": "pev_synth.utc_migration",
        "provenance_tag": "v1.1_migrated",
    }
    meta["migration_provenance"] = {
        "v20_sha256_per_file": dict(sha256_after),
        "v20_rows_per_file": dict(rows_per_file),
        "timestamp_cols_converted": {k: list(v) for k, v in timestamp_cols_converted.items()},
        "migration_utc_timestamp": migration_timestamp_utc,
    }
    return meta


def _write_meta_deterministic(meta: dict[str, object], path: Path) -> None:
    """Write meta.json with sorted keys + stable indentation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")


# ---------------------------------------------------------------------------
# Top-level entry-point
# ---------------------------------------------------------------------------

def migrate_bay_area_v1_to_v2(
    src_root: Path = DEFAULT_SRC_ROOT,
    dst_root: Path = DEFAULT_DST_ROOT,
    repo_root: Path | None = None,
) -> MigrationResult:
    """Run the v1.1 → v2.0 bay_area residential cache migration.

    Parameters
    ----------
    src_root
        v1.1 cache root.  Default ``data/pev/processed/residential_ev_synth``.
        If relative and ``repo_root`` is supplied, joined to ``repo_root``.
    dst_root
        v2.0 cache root. Default
        ``data/pev/processed/bay_area/residential_ev_synth``. Created if
        missing.
    repo_root
        Optional repo root for resolving relative paths. If ``None``, paths
        are used as-is.

    Returns
    -------
    MigrationResult

    Notes
    -----
    The migration is **idempotent**: a second invocation produces identical
    parquet bytes (because every writer setting is deterministic and the
    tz_convert is a stable operation). ``meta.json`` reproduces identically
    except for ``migration_timestamp_utc`` and ``migration_provenance.migration_utc_timestamp``;
    a second run does **not** advance those fields if the parquet bytes are
    unchanged (see the early-exit branch).

    The v1.1 cache at ``src_root`` is **never** modified.
    """
    if repo_root is not None:
        if not src_root.is_absolute():
            src_root = repo_root / src_root
        if not dst_root.is_absolute():
            dst_root = repo_root / dst_root

    if not src_root.is_dir():
        raise FileNotFoundError(f"src_root not found or not a directory: {src_root}")

    dst_root.mkdir(parents=True, exist_ok=True)

    timestamp_cols_converted: dict[str, list[str]] = {}
    sha256_after: dict[str, str] = {}
    rows_per_file: dict[str, int] = {}
    skipped_files: list[str] = []

    # Migrate every expected parquet. Writes are deterministic (snappy +
    # fixed dictionary / version / no-statistics) so reruns are byte-stable.
    for fname in PARQUET_FILES:
        ts_cols = TS_COLS_BY_FILE[fname]
        src = src_root / fname
        dst = dst_root / fname
        if not src.is_file():
            skipped_files.append(fname)
            logger.warning("skipping missing source file: %s", src)
            continue
        n_rows, converted = migrate_parquet_tz(src, dst, ts_cols)
        timestamp_cols_converted[fname] = converted
        rows_per_file[fname] = n_rows
        sha256_after[fname] = _sha256(dst)
        logger.info(
            "migrated %s -> %s (rows=%d, ts_cols=%s)",
            src,
            dst,
            n_rows,
            converted,
        )

    # ---- Stage 3: meta.json ------------------------------------------
    src_meta_path = src_root / "meta.json"
    if src_meta_path.is_file():
        with src_meta_path.open("r", encoding="utf-8") as f:
            src_meta = json.load(f)
    else:
        src_meta = {}
        skipped_files.append("meta.json")
        logger.warning("source meta.json missing at %s", src_meta_path)

    dst_meta_path = dst_root / "meta.json"
    migration_timestamp_utc = _resolve_migration_timestamp(
        dst_meta_path=dst_meta_path,
        parquet_sha256=sha256_after,
    )
    meta_v2 = _build_meta(
        src_meta=src_meta,
        sha256_after=sha256_after,
        rows_per_file=rows_per_file,
        timestamp_cols_converted=timestamp_cols_converted,
        migration_timestamp_utc=migration_timestamp_utc,
        src_root_str=str(src_root),
    )
    _write_meta_deterministic(meta_v2, dst_meta_path)
    sha256_after["meta.json"] = _sha256(dst_meta_path)
    rows_per_file.setdefault("meta.json", 0)

    n_files_migrated = len(sha256_after)

    return MigrationResult(
        n_files_migrated=n_files_migrated,
        timestamp_cols_converted=dict(timestamp_cols_converted),
        sha256_after=dict(sha256_after),
        rows_per_file=dict(rows_per_file),
        skipped_files=list(skipped_files),
    )


def _resolve_migration_timestamp(
    dst_meta_path: Path,
    parquet_sha256: dict[str, str],
) -> str:
    """Return the timestamp to record in v2.0 meta.json.

    If a prior meta.json exists and its recorded parquet SHA-256s match the
    current ones (idempotent rerun), reuse the previously-stored migration
    timestamp so the rerun is byte-stable. Otherwise stamp the current UTC
    instant.
    """
    if dst_meta_path.is_file():
        try:
            with dst_meta_path.open("r", encoding="utf-8") as f:
                prior = json.load(f)
            prior_prov = prior.get("migration_provenance", {})
            prior_sha = prior_prov.get("v20_sha256_per_file", {})
            if prior_sha and all(
                prior_sha.get(k) == v for k, v in parquet_sha256.items()
            ):
                ts = prior_prov.get("migration_utc_timestamp")
                if isinstance(ts, str) and ts:
                    return ts
        except (OSError, json.JSONDecodeError):
            pass
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    """CLI: ``python -m pev_synth._utc_migration``."""
    parser = argparse.ArgumentParser(
        prog="pev_synth._utc_migration",
        description=(
            "One-shot v1.1 → v2.0 bay_area residential cache migration: "
            "tz_convert all timestamp columns from America/Los_Angeles to UTC "
            "and relocate to the v2.0 path layout."
        ),
    )
    parser.add_argument(
        "--src",
        type=str,
        default=None,
        help=f"Source v1.1 cache root (default: {DEFAULT_SRC_ROOT}).",
    )
    parser.add_argument(
        "--dst",
        type=str,
        default=None,
        help=f"Destination v2.0 cache root (default: {DEFAULT_DST_ROOT}).",
    )
    parser.add_argument(
        "--repo-root",
        type=str,
        default=None,
        help="Repo root for resolving relative paths.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    repo_root = Path(args.repo_root).resolve() if args.repo_root else None
    src = Path(args.src) if args.src else DEFAULT_SRC_ROOT
    dst = Path(args.dst) if args.dst else DEFAULT_DST_ROOT

    result = migrate_bay_area_v1_to_v2(src_root=src, dst_root=dst, repo_root=repo_root)
    print(
        "pev_synth._utc_migration: "
        f"n_files_migrated={result.n_files_migrated}, "
        f"skipped={result.skipped_files}, "
        f"rows={result.rows_per_file}"
    )
    return 0


# ---------------------------------------------------------------------------
# Re-export helpers for tests
# ---------------------------------------------------------------------------

__all__ = [
    "METHODOLOGY_VERSION",
    "MASTER_SEED",
    "MODULE_OFFSET",
    "DEFAULT_SRC_ROOT",
    "DEFAULT_DST_ROOT",
    "TS_COLS_BY_FILE",
    "PARQUET_FILES",
    "MigrationResult",
    "migrate_bay_area_v1_to_v2",
    "migrate_parquet_tz",
    "main",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
