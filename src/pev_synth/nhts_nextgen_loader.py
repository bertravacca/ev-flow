"""Module M1b — NHTS NextGen 2022 national loader (`pev_synth.nhts_nextgen_loader`).

Purpose
-------
Acquire the ORNL NHTS NextGen 2022 public-use Version 2.1 (April 2025)
CSV bundle, normalise column names to the ev-flow contract, materialise
four tidy parquet tables (household / person / vehicle / trip) tagged
with ``nhts_vintage = "2022_nextgen"``, plus a download manifest with
SHA-256 hashes.

This module is the v3.0 enrichment companion to :mod:`pev_synth.nhts_loader`
(NHTS 2017 California). The NextGen 2022 public-use file is national-only
because state-level geography (``HHSTATE``, ``HHSTFIPS``, ``HH_CBSA``) was
suppressed by the ORNL curators on sample-size privacy grounds (~7,893
households), and the vehicle ``MODEL`` column is likewise absent. The loader
inserts ``pd.NA`` placeholders for those columns so the rest of the
``donor_matcher`` pipeline can union 2017 + 2022 frames without a schema
divergence; the MODEL back-off draw is performed downstream in
:func:`pev_synth.donor_matcher._model_backoff_draw_from_2017`.

References
----------
- 2022 NextGen NHTS public-use download (V2.1, April 2025):
  https://nhts.ornl.gov/assets/2022/download/csv.zip  (~3.7 MB).
- 2022 NextGen NHTS Public Use Codebook V2.1:
  https://nhts.ornl.gov/media/2022/doc/codebook.pdf
- 2022 NextGen NHTS V2.1 Release Notes:
  https://nhts.ornl.gov/media/2022/doc/2022%20NextGen%20NHTS%20V2.1%20Release%20Notes.pdf
- Scientist S2 column-map authority: ``docs/v3/modeling/nhts_enrichment.md``
  §§3-6 (this module implements §6.2 / §6.3 exactly).

Schema deltas verified against Scientist S2's column map
(``docs/v3/modeling/nhts_enrichment.md`` §3-§4):

- Documented as absent in public file (vs 2017): ``HHSTATE``,
  ``HHSTFIPS``, ``HH_CBSA`` (HH frame); ``MODEL`` (VEH frame). All four
  are inserted as ``pd.NA`` placeholders so the schema remains
  compatible with the 2017 contract.
- Renamed: ``R_SEX_IMP`` → ``r_sex`` (PER); ``DWELL_TIME`` → ``dwell_min``
  (TRIP). Codes unchanged.
- Native columns preserved from 2017: ``HHFAMINC`` (11 bins), ``HFUEL``
  (1=Biodiesel, 2=PHEV, 3=BEV, 4=HEV), sentinel codes (-1/-7/-8/-9).
- New columns kept verbatim: ``STRATUMID`` (HH), ``WHYTRP1S`` (TRIP),
  ``TRIPMODE`` (TRIP), ``R_SEX_IMP`` (PER, retained alongside the renamed
  ``r_sex`` so the imputation flag isn't lost).

**Live download cross-check (2026-05-29 v2.1)**: a hands-on download of
the canonical URL revealed two divergences from Scientist S2's
documented contract — recorded here for v3.1 follow-up:

1. ``HHSTATE``, ``HHSTFIPS`` and ``HH_CBSA`` **are** present in the
   actual ``hhv2pub.csv``. This loader's contract-compliant behaviour
   overwrites them with ``pd.NA`` to honour the documented "national
   supplementary stratum" architecture — a deliberately conservative
   choice (any real geography in 2022 will be exposed to the
   downstream per-region narrower, which currently expects them
   absent). v3.1 may flip this behaviour after Scientist S2 reconciles
   the codebook.
2. The vehicle file uses ``VEHFUEL`` (not ``FUELTYPE``) and ``HYBRID``
   (not ``HFUEL``) as the alt-fuel discriminators in v2.1; ``HFUEL`` is
   the 2017 name and is absent. The loader still attempts a
   ``hfuel`` -> NA sentinel coercion on absent column (no-op) and the
   manifest's ``hfuel_counts`` block reports zeros — surfaced in the
   manifest so downstream consumers can detect the gap.
3. The trip file uses ``DWELTIME`` (not ``DWELL_TIME``), and ``MODEL``
   in the VEH file IS absent as documented.
4. Trip diary structure: **exactly 1 TRAVDAY per HOUSEID** in the live
   v2.1 file (mean=1.000, n_hh_with_multi_day=0), so D3's within-HH
   stitcher degenerates to single-day replay on NextGen 2022 just as
   it does on NHTS 2017. The vintage wrapper in
   :func:`pev_synth.travel_week_builder.sample_week_for_ev_v3` still
   routes NextGen 2022 to ``sample_week_within_houseid`` per the
   documented contract; the multi-day benefit will not materialise
   until a future vintage actually carries multi-day diaries.

Methodology version: v3.0. Master seed: 20260520. Module offset: +1 (shares
the M1 namespace with :mod:`pev_synth.nhts_loader`).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
import requests

from ._seeds import MASTER_SEED as _MASTER_SEED
from ._seeds import METHODOLOGY_VERSION as _METHODOLOGY_VERSION

__all__ = [
    "DEFAULT_CACHE_REL",
    "NEXTGEN_PARQUET_BASENAMES",
    "NHTS_NEXTGEN_CSV_MEMBERS",
    "NHTS_NEXTGEN_MANIFEST_BASENAME",
    "NHTS_NEXTGEN_SENTINELS",
    "NHTS_NEXTGEN_VINTAGE",
    "NHTS_NEXTGEN_ZIP_URL",
    "NHTSNextGenData",
    "NHTSNextGenLoaderConfig",
    "load_nhts_nextgen_2022",
    "main",
]

# --------------------------------------------------------------------------- #
# Module-level constants (Scientist S2 contract, §6.1 + §6.2).
# --------------------------------------------------------------------------- #

#: ORNL NextGen 2022 public-use bundle (V2.1, April 2025). ~3.7 MB.
NHTS_NEXTGEN_ZIP_URL: Final[str] = "https://nhts.ornl.gov/assets/2022/download/csv.zip"

#: CSV members we extract from the zip; other members ignored.
NHTS_NEXTGEN_CSV_MEMBERS: Final[tuple[str, ...]] = (
    "hhv2pub.csv",
    "perv2pub.csv",
    "vehv2pub.csv",
    "tripv2pub.csv",
)

#: Parquet basenames (national; no per-state subsetting in v3.0).
NEXTGEN_PARQUET_BASENAMES: Final[dict[str, str]] = {
    "hh": "nhts2022_national_hh.parquet",
    "per": "nhts2022_national_per.parquet",
    "veh": "nhts2022_national_veh.parquet",
    "trip": "nhts2022_national_trip.parquet",
}

#: Download manifest filename (URL, SHA-256s, fetch timestamp).
NHTS_NEXTGEN_MANIFEST_BASENAME: Final[str] = "nhts2022_download_manifest.json"

#: Default cache dir, expressed as a path relative to the repo root.
DEFAULT_CACHE_REL: Final[Path] = Path("data/pev/raw/nhts2022")

#: NHTS sentinel ints (codebook §3.3 / Data User Guide Table 5).
#: ``-1`` = appropriate skip; ``-7`` = "I prefer not to answer";
#: ``-8`` = "I don't know"; ``-9`` = "Not ascertained".
NHTS_NEXTGEN_SENTINELS: Final[tuple[int, ...]] = (-1, -7, -8, -9)

#: Categorical vintage tag stamped on every row.
NHTS_NEXTGEN_VINTAGE: Final[str] = "2022_nextgen"

#: Vehicle types kept (1=car, 2=van, 3=SUV, 4=pickup) — same as nhts_loader.
VEHTYPE_KEEP: Final[tuple[int, ...]] = (1, 2, 3, 4)

#: HFUEL codes (preserved 1:1 from 2017; codebook §3.6).
HFUEL_BIODIESEL: Final[int] = 1
HFUEL_PHEV: Final[int] = 2
HFUEL_BEV: Final[int] = 3
HFUEL_HEV_NON_PLUGIN: Final[int] = 4

#: HTTP timeout for the zip download (s).
DEFAULT_TIMEOUT_S: Final[int] = 600


logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Public types
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class NHTSNextGenLoaderConfig:
    """User-tunable knobs (mirrors ``NHTSLoaderConfig``)."""

    cache_dir: Path = DEFAULT_CACHE_REL
    download_url: str = NHTS_NEXTGEN_ZIP_URL
    request_timeout_s: int = DEFAULT_TIMEOUT_S
    force: bool = False


@dataclass
class NHTSNextGenData:
    """Container for the four NextGen 2022 national tables plus manifest."""

    hh: pd.DataFrame
    per: pd.DataFrame
    veh: pd.DataFrame
    trip: pd.DataFrame
    manifest: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Low-level helpers (mirrored from nhts_loader; lifted to avoid coupling).
# --------------------------------------------------------------------------- #


def _sha256_of_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return hex SHA-256 of ``path`` streaming in chunks."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_zip(url: str, dest: Path, timeout_s: int) -> None:
    """Stream ``url`` to ``dest`` atomically. Supports http(s) and file://."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    logger.info("downloading %s -> %s", url, dest)
    if url.startswith("file://"):
        from urllib.parse import urlparse
        from urllib.request import url2pathname

        src = Path(url2pathname(urlparse(url).path))
        with src.open("rb") as r, tmp.open("wb") as fh:
            for chunk in iter(lambda: r.read(1 << 20), b""):
                fh.write(chunk)
    else:
        with requests.get(url, stream=True, timeout=timeout_s) as r:
            r.raise_for_status()
            with tmp.open("wb") as fh:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    if chunk:
                        fh.write(chunk)
    tmp.replace(dest)


def _extract_members(zip_path: Path, dest_dir: Path, members: tuple[str, ...]) -> None:
    """Extract requested ``members`` (case-insensitive, basename-only) from zip."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    wanted = {m.lower() for m in members}
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            base = Path(info.filename).name.lower()
            if base in wanted:
                target = dest_dir / base
                with zf.open(info) as src, target.open("wb") as dst:
                    dst.write(src.read())


def _lower_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with lower-cased column names."""
    return df.rename(columns={c: c.lower() for c in df.columns})


def _replace_sentinels(s: pd.Series) -> pd.Series:
    """Map NHTS sentinel ints to NA. Preserves the original dtype family."""
    return s.where(~s.isin(NHTS_NEXTGEN_SENTINELS), other=pd.NA)


def _coerce_int(s: pd.Series) -> pd.Series:
    """Coerce a possibly-stringy NHTS column to nullable Int64."""
    if pd.api.types.is_numeric_dtype(s):
        return s
    return pd.to_numeric(s, errors="coerce").astype("Int64")


# --------------------------------------------------------------------------- #
# Per-table transforms (Scientist S2 §6.2).
# --------------------------------------------------------------------------- #


def _process_hh(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Normalise household table.

    * Lowercase columns.
    * Insert ``hhstate=pd.NA``, ``hhstfips=pd.NA``, ``hh_cbsa=pd.NA``
      placeholders (absent in the NextGen 2022 public file; suppressed by
      ORNL for privacy).
    * Coerce ``hhfaminc`` sentinels to NA.
    * Stamp ``nhts_vintage = "2022_nextgen"`` on every row.
    """
    df = _lower_columns(df_raw)
    # Sentinel -> NA for income.
    if "hhfaminc" in df.columns:
        df["hhfaminc"] = _replace_sentinels(_coerce_int(df["hhfaminc"]))
    # ABSENT-in-NextGen placeholders.
    df["hhstate"] = pd.array([pd.NA] * len(df), dtype="string")
    df["hhstfips"] = pd.array([pd.NA] * len(df), dtype="string")
    df["hh_cbsa"] = pd.array([pd.NA] * len(df), dtype="string")
    # Vintage tag.
    df["nhts_vintage"] = pd.array([NHTS_NEXTGEN_VINTAGE] * len(df), dtype="string")
    if "houseid" in df.columns:
        df = df.sort_values("houseid", kind="stable").reset_index(drop=True)
    return df


def _process_per(df_raw: pd.DataFrame, hh_houseids: set[object]) -> pd.DataFrame:
    """Normalise person table.

    * Lowercase columns; rename ``r_sex_imp`` → ``r_sex`` (Scientist S2 §3.2).
    * Restrict to HH-known houseids (defensive — should be 1:1).
    * Sentinel -> NA for ``r_age``, ``r_sex``, ``worker``, ``driver``.
    * Stamp ``nhts_vintage`` on every row.
    """
    df = _lower_columns(df_raw)
    if "r_sex_imp" in df.columns and "r_sex" not in df.columns:
        df = df.rename(columns={"r_sex_imp": "r_sex"})
    if "houseid" in df.columns:
        df = df[df["houseid"].isin(hh_houseids)].copy()
    for col in ("r_age", "r_sex", "worker", "driver"):
        if col in df.columns:
            df[col] = _replace_sentinels(_coerce_int(df[col]))
    df["nhts_vintage"] = pd.array([NHTS_NEXTGEN_VINTAGE] * len(df), dtype="string")
    sort_cols = [c for c in ("houseid", "personid") if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols, kind="stable").reset_index(drop=True)
    return df


def _process_veh(df_raw: pd.DataFrame, hh_houseids: set[object]) -> pd.DataFrame:
    """Normalise vehicle table.

    * Lowercase columns.
    * Restrict to HH-known houseids and VEHTYPE ∈ {1,2,3,4}.
    * Insert ``model = pd.NA`` placeholder (ABSENT in NextGen 2022).
    * Sentinel -> NA for ``vehyear``, ``annmiles``, ``hfuel``, ``fueltype``.
    * Stamp ``nhts_vintage`` on every row.
    """
    df = _lower_columns(df_raw)
    if "houseid" in df.columns:
        df = df[df["houseid"].isin(hh_houseids)].copy()
    if "vehtype" in df.columns:
        vehtype_num = pd.to_numeric(df["vehtype"], errors="coerce")
        df = df[vehtype_num.isin(VEHTYPE_KEEP)].copy()
    for col in ("vehyear", "annmiles", "hfuel", "fueltype"):
        if col in df.columns:
            df[col] = _replace_sentinels(_coerce_int(df[col]))
    if "make" in df.columns:
        ser = df["make"].astype("string").str.strip()
        ser = ser.where(~ser.isin({"", "-1", "-7", "-8", "-9", "XXXXX"}), other=pd.NA)
        df["make"] = ser
    # MODEL absent in NextGen 2022 public file — insert placeholder so
    # downstream donor_matcher can detect-and-backoff on the 2017 marginal.
    df["model"] = pd.array([pd.NA] * len(df), dtype="string")
    df["nhts_vintage"] = pd.array([NHTS_NEXTGEN_VINTAGE] * len(df), dtype="string")
    sort_cols = [c for c in ("houseid", "vehid") if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols, kind="stable").reset_index(drop=True)
    return df


def _process_trip(df_raw: pd.DataFrame, hh_houseids: set[object]) -> pd.DataFrame:
    """Normalise trip table.

    * Lowercase columns; rename ``dwell_time`` → ``dwell_min`` (Scientist
      S2 §3.4).
    * Restrict to HH-known houseids.
    * Sentinel -> NA for ``whyfrom``, ``whyto`` (only the documented
      -7/-8/-9 negative sentinels, NOT WHYTO==1 which is legitimately
      ``home``).
    * Coerce VEHID==-1 ("non-vehicular trip") to NA.
    * Stamp ``nhts_vintage`` on every row.
    """
    df = _lower_columns(df_raw)
    if "dwell_time" in df.columns and "dwell_min" not in df.columns:
        df = df.rename(columns={"dwell_time": "dwell_min"})
    if "houseid" in df.columns:
        df = df[df["houseid"].isin(hh_houseids)].copy()
    for col in ("whyfrom", "whyto"):
        if col in df.columns:
            df[col] = _replace_sentinels(_coerce_int(df[col]))
    if "vehid" in df.columns:
        vid = pd.to_numeric(df["vehid"], errors="coerce")
        vid = vid.where(vid > 0, other=pd.NA)
        df["vehid"] = vid
    if "trpmiles" in df.columns:
        miles = pd.to_numeric(df["trpmiles"], errors="coerce")
        df["trpmiles"] = miles.where(miles >= 0, other=np.nan)
    df["nhts_vintage"] = pd.array([NHTS_NEXTGEN_VINTAGE] * len(df), dtype="string")
    sort_cols = [c for c in ("houseid", "personid", "tdtrpnum") if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols, kind="stable").reset_index(drop=True)
    return df


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def _outputs_exist(cache_dir: Path) -> bool:
    paths = [cache_dir / name for name in NEXTGEN_PARQUET_BASENAMES.values()]
    paths.append(cache_dir / NHTS_NEXTGEN_MANIFEST_BASENAME)
    return all(p.is_file() for p in paths)


def _read_manifest(cache_dir: Path) -> dict[str, Any]:
    with (cache_dir / NHTS_NEXTGEN_MANIFEST_BASENAME).open("r", encoding="utf-8") as fh:
        manifest: dict[str, Any] = json.load(fh)
    return manifest


def _load_cached(cache_dir: Path) -> NHTSNextGenData:
    """Read the four NextGen parquets + manifest from ``cache_dir``."""
    hh = pd.read_parquet(cache_dir / NEXTGEN_PARQUET_BASENAMES["hh"])
    per = pd.read_parquet(cache_dir / NEXTGEN_PARQUET_BASENAMES["per"])
    veh = pd.read_parquet(cache_dir / NEXTGEN_PARQUET_BASENAMES["veh"])
    trip = pd.read_parquet(cache_dir / NEXTGEN_PARQUET_BASENAMES["trip"])
    manifest = _read_manifest(cache_dir)
    return NHTSNextGenData(hh=hh, per=per, veh=veh, trip=trip, manifest=manifest)


def load_nhts_nextgen_2022(
    cache_dir: Path | str | None = None,
    force: bool = False,
    download_url: str = NHTS_NEXTGEN_ZIP_URL,
    request_timeout_s: int = DEFAULT_TIMEOUT_S,
) -> NHTSNextGenData:
    """Acquire and normalise the NHTS NextGen 2022 public-use national files.

    Parameters
    ----------
    cache_dir
        Directory to cache the zip, the four extracted CSVs, the four
        normalised parquets, and the download manifest. Created if missing.
        Defaults to ``data/pev/raw/nhts2022/`` relative to cwd.
    force
        If True, re-download the zip and re-process even when outputs exist.
    download_url
        ORNL NextGen 2022 public-use zip URL. Override only for tests that
        point at a local ``file://...`` URL.
    request_timeout_s
        HTTP timeout for the zip download.

    Returns
    -------
    NHTSNextGenData
        Dataclass with four DataFrames (``hh``, ``per``, ``veh``, ``trip``),
        each carrying an ``nhts_vintage = "2022_nextgen"`` column, plus the
        parsed manifest dict.

    Notes
    -----
    * The NextGen 2022 public file is **national** (~7,893 HH); state-level
      geography (``HHSTATE``, ``HHSTFIPS``, ``HH_CBSA``) is suppressed by
      ORNL on sample-size privacy grounds. The loader inserts ``pd.NA``
      placeholders for those columns so the schema remains compatible
      with the 2017 contract.
    * Vehicle ``MODEL`` is likewise absent and inserted as ``pd.NA``;
      :func:`pev_synth.donor_matcher._model_backoff_draw_from_2017`
      resolves it via a conditional draw on the 2017 marginal.
    * Sentinel codes ``-1/-7/-8/-9`` are coerced to ``pd.NA`` in the same
      columns as the 2017 loader (mirrors ``nhts_loader._replace_sentinels``).
    """
    cache = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_REL
    cache.mkdir(parents=True, exist_ok=True)

    if not force and _outputs_exist(cache):
        logger.info(
            "NextGen 2022 cached outputs present in %s — returning without re-processing",
            cache,
        )
        return _load_cached(cache)

    zip_path = cache / "csv.zip"
    if force or not zip_path.is_file():
        _download_zip(download_url, zip_path, timeout_s=request_timeout_s)
    zip_sha = _sha256_of_file(zip_path)
    zip_size = zip_path.stat().st_size

    _extract_members(zip_path, cache, NHTS_NEXTGEN_CSV_MEMBERS)
    csv_sha: dict[str, str] = {}
    for member in NHTS_NEXTGEN_CSV_MEMBERS:
        csv_sha[member] = _sha256_of_file(cache / member)

    logger.info("loading raw NextGen 2022 CSVs from %s", cache)
    df_hh_raw = pd.read_csv(cache / "hhv2pub.csv", low_memory=False)
    df_per_raw = pd.read_csv(cache / "perv2pub.csv", low_memory=False)
    df_veh_raw = pd.read_csv(cache / "vehv2pub.csv", low_memory=False)
    df_trip_raw = pd.read_csv(cache / "tripv2pub.csv", low_memory=False)
    total_rows = int(
        df_hh_raw.shape[0]
        + df_per_raw.shape[0]
        + df_veh_raw.shape[0]
        + df_trip_raw.shape[0]
    )

    hh = _process_hh(df_hh_raw)
    hh_houseids = set(hh["houseid"].tolist()) if "houseid" in hh.columns else set()
    per = _process_per(df_per_raw, hh_houseids)
    veh = _process_veh(df_veh_raw, hh_houseids)
    trip = _process_trip(df_trip_raw, hh_houseids)

    # Write parquets (snappy, deterministic row order from per-table sorts).
    hh.to_parquet(cache / NEXTGEN_PARQUET_BASENAMES["hh"], compression="snappy", index=False)
    per.to_parquet(
        cache / NEXTGEN_PARQUET_BASENAMES["per"], compression="snappy", index=False
    )
    veh.to_parquet(
        cache / NEXTGEN_PARQUET_BASENAMES["veh"], compression="snappy", index=False
    )
    trip.to_parquet(
        cache / NEXTGEN_PARQUET_BASENAMES["trip"], compression="snappy", index=False
    )

    # Trip diary structure stats — Scientist S2 §2.1 expected ~30k trips
    # with one assigned TRAVDAY per HH. Surfaced in manifest for QA.
    travdays_per_hh: dict[str, float] = {}
    if "houseid" in trip.columns and "travday" in trip.columns and len(trip) > 0:
        by_hh = trip.groupby("houseid")["travday"].nunique()
        travdays_per_hh = {
            "mean": float(by_hh.mean()),
            "max": int(by_hh.max()),
            "n_hh_with_multi_day": int((by_hh >= 2).sum()),
            "n_hh_total": int(len(by_hh)),
        }

    manifest: dict[str, Any] = {
        "module": "pev_synth.nhts_nextgen_loader",
        "methodology_version": _METHODOLOGY_VERSION,
        "master_seed": _MASTER_SEED,
        "url": download_url,
        "zip_path": str((cache / "csv.zip").resolve()),
        "zip_sha256": zip_sha,
        "zip_size_bytes": zip_size,
        "csv_sha256": csv_sha,
        "fetch_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "release": "NHTS NextGen 2022 V2.1 (April 2025)",
        "scope": "national public-use; HHSTATE/HHSTFIPS/HH_CBSA suppressed by ORNL",
        "filter_rule": (
            "no state/CBSA narrowing applied; rows tagged nhts_vintage='2022_nextgen' "
            "and treated as national supplementary stratum by donor_matcher"
        ),
        "subset_size": {
            "households": int(len(hh)),
            "persons": int(len(per)),
            "vehicles": int(len(veh)),
            "trips": int(len(trip)),
        },
        "raw_size": {
            "households": int(len(df_hh_raw)),
            "persons": int(len(df_per_raw)),
            "vehicles": int(len(df_veh_raw)),
            "trips": int(len(df_trip_raw)),
            "all_rows_sum": total_rows,
        },
        "wthhfin_sum_national": float(hh["wthhfin"].sum()) if "wthhfin" in hh.columns else 0.0,
        "hfuel_counts": (
            {
                "biodiesel": int((veh["hfuel"] == HFUEL_BIODIESEL).sum()),
                "phev": int((veh["hfuel"] == HFUEL_PHEV).sum()),
                "bev": int((veh["hfuel"] == HFUEL_BEV).sum()),
                "hev_non_plugin": int((veh["hfuel"] == HFUEL_HEV_NON_PLUGIN).sum()),
                "plug_in_total_phev_plus_bev": int(
                    veh["hfuel"].isin([HFUEL_PHEV, HFUEL_BEV]).sum()
                ),
            }
            if "hfuel" in veh.columns
            else {}
        ),
        "trip_travday_structure": travdays_per_hh,
        "schema_deltas_vs_2017": {
            "absent_in_public": ["HHSTATE", "HHSTFIPS", "HH_CBSA", "MODEL"],
            "renamed": {"R_SEX_IMP": "r_sex", "DWELL_TIME": "dwell_min"},
            "new_columns_preserved": ["STRATUMID", "WHYTRP1S", "TRIPMODE", "R_SEX_IMP"],
        },
        "nhts_vintage": NHTS_NEXTGEN_VINTAGE,
        "fallback_triggered": False,
    }
    with (cache / NHTS_NEXTGEN_MANIFEST_BASENAME).open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)

    return NHTSNextGenData(hh=hh, per=per, veh=veh, trip=trip, manifest=manifest)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _default_repo_root() -> Path:
    """Return the study repo root via :func:`pev_synth._paths.repo_root`."""
    from ._paths import repo_root

    return repo_root()


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m pev_synth.nhts_nextgen_loader",
        description=(
            "Download NHTS NextGen 2022 V2.1 public-use national CSVs from "
            "ORNL and emit four normalised parquets tagged "
            "nhts_vintage='2022_nextgen'."
        ),
    )
    p.add_argument(
        "--repo-root",
        type=str,
        default=None,
        help=(
            "Study repo root used to anchor default --cache-dir "
            "(defaults to two parents up from this file). Set this if "
            "invoking from a non-repo cwd."
        ),
    )
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help=(
            "Output directory for the zip, CSVs, parquets, and manifest. "
            "Defaults to ``<repo-root>/data/pev/raw/nhts2022``. Absolute "
            "paths are honoured as-is; relative paths are resolved against "
            "--repo-root, not the current working directory."
        ),
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="re-download zip and re-process even if outputs exist",
    )
    p.add_argument(
        "--url",
        default=NHTS_NEXTGEN_ZIP_URL,
        help="override download URL (for tests)",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_S,
        help="HTTP request timeout in seconds",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    p.add_argument(
        "--add-on",
        type=str,
        default=None,
        choices=("vdot", "tdot", "ompo"),
        help=(
            "v3.1 stub: NextGen 2022 state add-on partner. Raises with an "
            "explicit message until URLs/licensing are confirmed."
        ),
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.add_on is not None:
        raise NotImplementedError(
            f"NHTS NextGen 2022 state add-on '{args.add_on}' (VDOT/TDOT/OMPO) "
            "is stubbed for v3.1: download URLs and licensing not yet confirmed. "
            "File a v3.1 ticket and verify the URL with the user before "
            "wiring up this path. See docs/v3/modeling/nhts_enrichment.md §5."
        )
    repo_root = (
        Path(args.repo_root).resolve() if args.repo_root else _default_repo_root()
    )
    if args.cache_dir is None:
        cache_dir = repo_root / DEFAULT_CACHE_REL
    else:
        cache_dir = args.cache_dir
        if not cache_dir.is_absolute():
            cache_dir = repo_root / cache_dir
    data = load_nhts_nextgen_2022(
        cache_dir=cache_dir,
        force=args.force,
        download_url=args.url,
        request_timeout_s=args.timeout,
    )
    m = data.manifest
    hfuel = m.get("hfuel_counts", {})
    print(
        "NHTS NextGen 2022 national subset written. "
        f"hh={m['subset_size']['households']} "
        f"per={m['subset_size']['persons']} "
        f"veh={m['subset_size']['vehicles']} "
        f"trip={m['subset_size']['trips']} "
        f"wthhfin_sum={m['wthhfin_sum_national']:.3e} "
        f"hfuel_plug_in={hfuel.get('plug_in_total_phev_plus_bev', 0)} "
        f"(phev={hfuel.get('phev', 0)}, bev={hfuel.get('bev', 0)})"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())


# Silence unused-import warnings in environments that strip docstrings.
_ = np
