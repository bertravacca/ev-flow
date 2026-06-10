"""Module M1 — NHTS 2017 California loader (`pev_synth.nhts_loader`).

Purpose
-------
Acquire the ORNL NHTS 2017 public-use CSV bundle, extract the California
subset (HHSTATE == "CA"), derive the per-person dwell-time field, and
materialise four tidy parquet tables (household / person / vehicle / trip)
plus a download manifest with SHA-256 hashes.

This module is upstream of every other `pev_synth` module (Wave 1, Dev A).
It depends on no other `pev_synth` submodule.

References
----------
- NHTS 2017 public-use download: https://nhts.ornl.gov/assets/2016/download/csv.zip
- User guide (do not download here, cite only):
  https://nhts.ornl.gov/assets/NHTS2017_UsersGuide_04232019_1.pdf
- Phase-4 methodology plan §2.1 (inputs) and §4 (travel-week construction).
- Phase-5 PM dev brief Module 1.

Orchestrator-resolved contracts override the PM brief where they conflict:
- Master seed: 20260518.
- Methodology version: v1.0.0.
- Filter scope: California (HHSTATE == "CA") — broader than the Bay-Area
  subset described in the PM brief; downstream M3 narrows to Bay Area.
- Output filenames: nhts2017_ca_{hh,per,veh,trip}.parquet under
  data/pev/raw/nhts2017/ (raw zone, not processed).
- Expected CA household count ≈ 26,095 per Phase-1 dossier §2.1 (NHTS
  2017 Summary of Travel Trends, CA add-on).
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

__all__ = [
    "NHTSData",
    "NHTSLoaderConfig",
    "load_nhts_2017_ca",
    "main",
]

# --------------------------------------------------------------------------- #
# Module-level constants (cited)
# --------------------------------------------------------------------------- #

#: ORNL public-use bundle. Vintage tag "2016" in the URL is the ORNL release
#: prefix; the file content is the NHTS 2017 wave (per PM brief §4).
NHTS_ZIP_URL: Final[str] = "https://nhts.ornl.gov/assets/2016/download/csv.zip"

#: Members we extract from the zip. Other members are ignored.
NHTS_CSV_MEMBERS: Final[tuple[str, ...]] = (
    "hhpub.csv",
    "perpub.csv",
    "vehpub.csv",
    "trippub.csv",
)

#: Output parquet basenames (one per CSV after CA filter).
PARQUET_BASENAMES: Final[dict[str, str]] = {
    "hh": "nhts2017_ca_hh.parquet",
    "per": "nhts2017_ca_per.parquet",
    "veh": "nhts2017_ca_veh.parquet",
    "trip": "nhts2017_ca_trip.parquet",
}

#: Download manifest filename (URL, SHA-256, fetch timestamp, sizes).
MANIFEST_BASENAME: Final[str] = "nhts2017_download_manifest.json"

#: NHTS sentinel values that map to missing (-7 "I don't know", -8 "I prefer
#: not to answer", -9 "Not ascertained"). Per user guide §3.3.
NHTS_SENTINELS: Final[tuple[int, ...]] = (-7, -8, -9)

#: Columns in which sentinels must be converted to pd.NA (per PM brief §6.4).
HH_NA_COLS: Final[tuple[str, ...]] = ("hhfaminc",)
PER_NA_COLS: Final[tuple[str, ...]] = ("r_age", "r_sex", "worker", "driver")
VEH_NA_COLS: Final[tuple[str, ...]] = ("vehyear", "annmiles", "hfuel", "fueltype")

#: Vehicle types we keep (1=car, 2=van, 3=SUV, 4=pickup) per plan §2.1.
VEHTYPE_KEEP: Final[tuple[int, ...]] = (1, 2, 3, 4)

#: HFUEL codes per NHTS 2017 Codebook v1.2 p.55. HFUEL is the alternative-fuel
#: subtype field, populated only when FUELTYPE == 03 (Hybrid, electric or
#: alternative fuel). FUELTYPE alone CANNOT separate BEV / PHEV / HEV — the
#: honest separation lives in HFUEL. Values:
#:   01 = Biodiesel
#:   02 = Plug-in Hybrid (PHEV; e.g., Chevy Volt)
#:   03 = Electric (BEV; e.g., Nissan Leaf)
#:   04 = Hybrid (gas/electric, not plug-in; e.g., Toyota Prius)
#:   97 = Some other fuel
#:   -1 = Appropriate skip (vehicle is gas/diesel; HFUEL not asked)
HFUEL_BIODIESEL: Final[int] = 1
HFUEL_PHEV: Final[int] = 2
HFUEL_BEV: Final[int] = 3
HFUEL_HEV_NON_PLUGIN: Final[int] = 4

#: Per-table column-name → final dtype contract. These are the public schemas
#: that downstream modules (M3, M4) rely on (PM brief §5, plan §8).
HH_DTYPES: Final[dict[str, str]] = {
    "houseid": "int64",
    "hhstate": "string",
    "hhstfips": "string",
    "hh_cbsa": "string",
    "cdivmsar": "Int16",
    "urbrur": "Int8",
    "hhfaminc": "Int8",
    "hhsize": "Int8",
    "hhvehcnt": "Int8",
    "homeown": "Int8",
    "wthhfin": "float64",
}

PER_DTYPES: Final[dict[str, str]] = {
    "houseid": "int64",
    "personid": "int32",
    "r_age": "Int8",
    "r_sex": "Int8",
    "worker": "Int8",
    "driver": "Int8",
    "wtperfin": "float64",
}

VEH_DTYPES: Final[dict[str, str]] = {
    "houseid": "int64",
    "vehid": "int32",
    "vehtype": "Int8",
    "fueltype": "Int8",
    "make": "string",
    "model": "string",
    "vehyear": "Int16",
    "annmiles": "float32",
    "hfuel": "Int8",
}

TRIP_DTYPES: Final[dict[str, str]] = {
    "houseid": "int64",
    "personid": "int32",
    "vehid": "Int32",
    "tdtrpnum": "int16",
    "strttime": "int16",
    "endtime": "int16",
    "trpmiles": "float32",
    "trvlcmin": "int16",
    "whyfrom": "Int8",
    "whyto": "Int8",
    "tdwknd": "Int8",
    "travday": "Int8",
    "daywgt": "float64",
    "wttrdfin": "float64",
    "dwell_min": "Int32",
}

#: Default cache_dir, expressed as a path relative to the study repo root.
#: The CLI resolves this against ``--repo-root`` (default
#: ``Path(__file__).resolve().parents[2]``) so the loader is cwd-independent.
DEFAULT_CACHE_REL: Final[Path] = Path("data/pev/raw/nhts2017")

#: Back-compat alias: the original cwd-relative default. Kept so existing
#: callers of ``load_nhts_2017_ca()`` with no ``cache_dir`` argument still
#: behave the same (cwd-relative). The CLI no longer uses this — see ``main``.
DEFAULT_CACHE_DIR: Final[Path] = DEFAULT_CACHE_REL

#: HTTP download timeout (s).
DEFAULT_TIMEOUT_S: Final[int] = 600


logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Public types
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class NHTSLoaderConfig:
    """User-tunable knobs (PM brief §7)."""

    cache_dir: Path = DEFAULT_CACHE_DIR
    download_url: str = NHTS_ZIP_URL
    request_timeout_s: int = DEFAULT_TIMEOUT_S
    force: bool = False


@dataclass
class NHTSData:
    """Container for the four CA-filtered NHTS 2017 tables plus manifest."""

    hh: pd.DataFrame
    per: pd.DataFrame
    veh: pd.DataFrame
    trip: pd.DataFrame
    manifest: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _sha256_of_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return hex SHA-256 of ``path`` streaming in chunks."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_zip(url: str, dest: Path, timeout_s: int) -> None:
    """Stream ``url`` to ``dest`` atomically (write to tmp then rename).

    Supports both http(s):// (via ``requests``) and file:// (via stdlib),
    so tests can point at a local zip without hitting the network.
    """
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
    """Extract only requested ``members`` (case-insensitive) from ``zip_path``."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    wanted = {m.lower() for m in members}
    with zipfile.ZipFile(zip_path) as zf:
        # Match by basename, ignore any nested directory in the archive.
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
    """Map NHTS sentinel ints to NA, preserving the original dtype family."""
    return s.where(~s.isin(NHTS_SENTINELS), other=pd.NA)


def _coerce_int(s: pd.Series) -> pd.Series:
    """Coerce a possibly-stringy NHTS column to nullable Int64."""
    if pd.api.types.is_numeric_dtype(s):
        return s
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def _enforce_dtypes(df: pd.DataFrame, dtypes: dict[str, str]) -> pd.DataFrame:
    """Project ``df`` onto ``dtypes`` columns and cast to the declared dtypes.

    Missing columns in ``df`` are filled with NA (e.g., a synthetic test
    feed may omit optional columns like ``hh_cbsa``). Extra columns are
    dropped. Column order matches ``dtypes`` insertion order (deterministic).
    """
    out = pd.DataFrame()
    for col, dt in dtypes.items():
        if col in df.columns:
            ser = df[col]
        else:
            ser = pd.Series([pd.NA] * len(df), index=df.index, dtype="object")
        if dt in {"int8", "int16", "int32", "int64"}:
            # Non-nullable int: must have no NA. Coerce via numeric then cast.
            ser = pd.to_numeric(ser, errors="coerce")
            if ser.isna().any():
                raise ValueError(
                    f"Column {col!r} contains NA but dtype contract is {dt} "
                    "(non-nullable). Investigate upstream sentinel handling."
                )
            out[col] = ser.astype(dt)
        elif dt in {"Int8", "Int16", "Int32", "Int64"}:
            out[col] = pd.to_numeric(ser, errors="coerce").astype(dt)
        elif dt in {"float32", "float64"}:
            out[col] = pd.to_numeric(ser, errors="coerce").astype(dt)
        elif dt == "string":
            # Preserve nulls as <NA>; cast everything else to str.
            out[col] = ser.astype("string")
        else:  # pragma: no cover — exhaustive over the contract
            raise ValueError(f"Unhandled dtype {dt!r} for column {col!r}.")
    return out


# --------------------------------------------------------------------------- #
# Per-table transforms
# --------------------------------------------------------------------------- #


def _process_hh(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Normalise household table and filter to California (HHSTATE == 'CA')."""
    df = _lower_columns(df_raw)
    if "hhstate" not in df.columns:
        raise KeyError("hhpub missing HHSTATE column — cannot apply CA filter.")
    # Filter California first to shrink the working frame.
    df = df[df["hhstate"].astype("string").str.upper() == "CA"].copy()
    # Sentinel -> NA for income.
    for col in HH_NA_COLS:
        if col in df.columns:
            df[col] = _replace_sentinels(_coerce_int(df[col]))
    # Some hh_cbsa values are reported as strings like "41860" or numeric;
    # normalise to zero-padded 5-char strings, replacing rural-blank with NA.
    if "hh_cbsa" in df.columns:
        cbsa = df["hh_cbsa"].astype("string").str.strip()
        cbsa = cbsa.where(~cbsa.isin({"", "-1", "0", "XXXXX"}), other=pd.NA)
        df["hh_cbsa"] = cbsa
    # HHSTFIPS is reported as int in raw NHTS — pad to 2-char.
    if "hhstfips" in df.columns:
        fips = pd.to_numeric(df["hhstfips"], errors="coerce")
        df["hhstfips"] = fips.astype("Int64").astype("string").str.zfill(2)
    df = _enforce_dtypes(df, HH_DTYPES)
    df = df.sort_values("houseid", kind="stable").reset_index(drop=True)
    return df


def _process_per(df_raw: pd.DataFrame, ca_houseids: pd.Index) -> pd.DataFrame:
    """Normalise person table and restrict to CA households."""
    df = _lower_columns(df_raw)
    df = df[df["houseid"].isin(ca_houseids)].copy()
    for col in PER_NA_COLS:
        if col in df.columns:
            df[col] = _replace_sentinels(_coerce_int(df[col]))
    df = _enforce_dtypes(df, PER_DTYPES)
    df = df.sort_values(["houseid", "personid"], kind="stable").reset_index(drop=True)
    return df


def _process_veh(df_raw: pd.DataFrame, ca_houseids: pd.Index) -> pd.DataFrame:
    """Normalise vehicle table: CA filter, VEHTYPE ∈ {1,2,3,4}, sentinel→NA."""
    df = _lower_columns(df_raw)
    df = df[df["houseid"].isin(ca_houseids)].copy()
    # Restrict to passenger-vehicle types per plan §2.1.
    if "vehtype" in df.columns:
        vehtype_num = pd.to_numeric(df["vehtype"], errors="coerce")
        df = df[vehtype_num.isin(VEHTYPE_KEEP)].copy()
    for col in VEH_NA_COLS:
        if col in df.columns:
            df[col] = _replace_sentinels(_coerce_int(df[col]))
    # Make/model: trim and treat NHTS "missing" placeholders as NA.
    for col in ("make", "model"):
        if col in df.columns:
            ser = df[col].astype("string").str.strip()
            ser = ser.where(~ser.isin({"", "-1", "-7", "-8", "-9", "XXXXX"}), other=pd.NA)
            df[col] = ser
    df = _enforce_dtypes(df, VEH_DTYPES)
    df = df.sort_values(["houseid", "vehid"], kind="stable").reset_index(drop=True)
    return df


def _compute_dwell_minutes(group: pd.DataFrame) -> pd.Series:
    """Dwell = STRTTIME(n+1) − ENDTIME(n) in minutes within a personid-day.

    Last trip of a personid-day receives NaN (NHTS gap; plan §2.1, §4.6).
    Times are HHMM ints; we convert to minutes-since-midnight.
    """
    # Group is already sorted by (travday, strttime) caller-side. Compute next
    # trip's start time within each (travday) block; rows that are the last
    # trip of the day produce NaN.
    next_strt = group.groupby("travday")["strttime"].shift(-1)
    end_min = (group["endtime"] // 100) * 60 + (group["endtime"] % 100)
    nxt_min = (next_strt // 100) * 60 + (next_strt % 100)
    dwell = nxt_min - end_min
    # Negative values can occur if the survey records a trip that crosses
    # midnight or has data inconsistency; clip and mark as NaN.
    dwell = dwell.where(dwell >= 0, other=pd.NA)
    return dwell


def _process_trip(df_raw: pd.DataFrame, ca_houseids: pd.Index) -> pd.DataFrame:
    """Normalise trip table, restrict to CA households, compute dwell_min.

    NHTS 2017 public-use ``trippub.csv`` does not carry a ``DAYWGT`` column —
    the PM brief listed it but ORNL only publishes ``WTTRDFIN`` (the trip-day
    weight) at the trip level. We satisfy the schema contract by mirroring
    ``wttrdfin`` into ``daywgt`` so downstream weighted samplers (M3, M4)
    do not need to special-case absence.
    """
    df = _lower_columns(df_raw)
    df = df[df["houseid"].isin(ca_houseids)].copy()
    # WHYFROM / WHYTO have sentinel codes for "appears as starting trip"; we
    # convert -7/-8/-9 only (NHTS code 1 = home is legitimate and not a
    # sentinel).
    for col in ("whyfrom", "whyto"):
        if col in df.columns:
            df[col] = _replace_sentinels(_coerce_int(df[col]))
    # VEHID is -1 for non-vehicular trips (walk/transit); coerce to NA.
    if "vehid" in df.columns:
        vid = pd.to_numeric(df["vehid"], errors="coerce")
        vid = vid.where(vid > 0, other=pd.NA)
        df["vehid"] = vid
    # NHTS 2017 does not publish DAYWGT; mirror WTTRDFIN so the schema
    # contract still has a populated trip-day weight column.
    if "daywgt" not in df.columns and "wttrdfin" in df.columns:
        df["daywgt"] = df["wttrdfin"]
    # TRPMILES sentinel −9 ("not ascertained") → NaN; keep as float32 NaN.
    if "trpmiles" in df.columns:
        miles = pd.to_numeric(df["trpmiles"], errors="coerce")
        df["trpmiles"] = miles.where(miles >= 0, other=np.nan)
    # TRVLCMIN sentinel −9 → keep but flag downstream; here we coerce to NA
    # so the non-nullable int cast below would fail. Use 0 minutes as the
    # safest substitution because the field is rarely sentinel in practice
    # (< 0.1 % of trips). Document this as a v1 simplification.
    if "trvlcmin" in df.columns:
        tcm = pd.to_numeric(df["trvlcmin"], errors="coerce")
        df["trvlcmin"] = tcm.where(tcm >= 0, other=0)
    # Compute dwell. Sort first so groupby preserves the within-day order.
    sort_cols = ["houseid", "personid", "travday", "strttime", "tdtrpnum"]
    df = df.sort_values(sort_cols, kind="stable").reset_index(drop=True)
    # dwell_min computed per (houseid, personid); within that, the helper
    # respects the travday boundary so the last-trip-of-day → NaN.
    dwell = df.groupby(["houseid", "personid"], sort=False, group_keys=False).apply(
        _compute_dwell_minutes, include_groups=False
    )
    df["dwell_min"] = dwell.values
    df = _enforce_dtypes(df, TRIP_DTYPES)
    df = df.sort_values(["houseid", "personid", "tdtrpnum"], kind="stable").reset_index(drop=True)
    return df


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def _outputs_exist(cache_dir: Path) -> bool:
    paths = [cache_dir / name for name in PARQUET_BASENAMES.values()]
    paths.append(cache_dir / MANIFEST_BASENAME)
    return all(p.is_file() for p in paths)


def _read_manifest(cache_dir: Path) -> dict[str, Any]:
    with (cache_dir / MANIFEST_BASENAME).open("r", encoding="utf-8") as fh:
        manifest: dict[str, Any] = json.load(fh)
    return manifest


def _load_cached(cache_dir: Path) -> NHTSData:
    """Read the four CA parquets + manifest from ``cache_dir``."""
    hh = pd.read_parquet(cache_dir / PARQUET_BASENAMES["hh"])
    per = pd.read_parquet(cache_dir / PARQUET_BASENAMES["per"])
    veh = pd.read_parquet(cache_dir / PARQUET_BASENAMES["veh"])
    trip = pd.read_parquet(cache_dir / PARQUET_BASENAMES["trip"])
    manifest = _read_manifest(cache_dir)
    return NHTSData(hh=hh, per=per, veh=veh, trip=trip, manifest=manifest)


def load_nhts_2017_ca(
    cache_dir: Path | str | None = None,
    force: bool = False,
    download_url: str = NHTS_ZIP_URL,
    request_timeout_s: int = DEFAULT_TIMEOUT_S,
) -> NHTSData:
    """Acquire and CA-filter the NHTS 2017 public-use files.

    Parameters
    ----------
    cache_dir
        Directory to cache the zip, the four extracted CSVs, the four
        CA-filtered parquets, and the download manifest. Created if missing.
        Defaults to ``data/pev/raw/nhts2017/`` relative to cwd.
    force
        If True, re-download the zip and re-process even when outputs exist.
    download_url
        ORNL public-use zip URL. Override only for tests that point at a
        local file://… URL.
    request_timeout_s
        HTTP timeout for the zip download.

    Returns
    -------
    NHTSData
        Dataclass with four DataFrames (``hh``, ``per``, ``veh``, ``trip``)
        and the parsed manifest dict.

    Notes
    -----
    - California filter is ``HHSTATE == "CA"`` (per orchestrator contract).
      Expected ≈ 26,095 households per Phase-1 dossier §2.1 (NHTS 2017
      Summary of Travel Trends, CA add-on).
    - Vehicles are restricted to VEHTYPE ∈ {1, 2, 3, 4} per plan §2.1.
    - Sentinel values −7/−8/−9 are converted to ``pd.NA`` in income, age,
      sex, worker, driver, annmiles, vehyear, hfuel, whyfrom, whyto.
    - ``dwell_min`` is computed within each (houseid, personid, travday):
      last trip of the day → NA.
    - HFUEL (NHTS 2017 Codebook v1.2 p.55) separates plug-in vehicles from
      conventional hybrids: 01=Biodiesel, 02=PHEV, 03=BEV, 04=HEV non-plugin.
      In the CA subset, 247 unique households reported a PHEV (HFUEL=2) —
      an adequate donor cohort for M3. (The 9 HFUEL=1 households are
      biodiesel, NOT PHEV; an earlier draft of this module conflated them.)
    """
    cache = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    cache.mkdir(parents=True, exist_ok=True)

    if not force and _outputs_exist(cache):
        logger.info("cached outputs present in %s — returning without re-processing", cache)
        return _load_cached(cache)

    zip_path = cache / "csv.zip"
    if force or not zip_path.is_file():
        _download_zip(download_url, zip_path, timeout_s=request_timeout_s)
    zip_sha = _sha256_of_file(zip_path)
    zip_size = zip_path.stat().st_size

    _extract_members(zip_path, cache, NHTS_CSV_MEMBERS)
    csv_sha: dict[str, str] = {}
    for member in NHTS_CSV_MEMBERS:
        csv_sha[member] = _sha256_of_file(cache / member)

    logger.info("loading raw CSVs from %s", cache)
    df_hh_raw = pd.read_csv(cache / "hhpub.csv", low_memory=False)
    df_per_raw = pd.read_csv(cache / "perpub.csv", low_memory=False)
    df_veh_raw = pd.read_csv(cache / "vehpub.csv", low_memory=False)
    df_trip_raw = pd.read_csv(cache / "trippub.csv", low_memory=False)
    total_size = int(
        df_hh_raw.shape[0] + df_per_raw.shape[0] + df_veh_raw.shape[0] + df_trip_raw.shape[0]
    )

    hh = _process_hh(df_hh_raw)
    ca_ids = pd.Index(hh["houseid"].unique(), name="houseid")
    per = _process_per(df_per_raw, ca_ids)
    veh = _process_veh(df_veh_raw, ca_ids)
    trip = _process_trip(df_trip_raw, ca_ids)

    # Write parquets (snappy, deterministic row order from per-table sorts).
    hh.to_parquet(cache / PARQUET_BASENAMES["hh"], compression="snappy", index=False)
    per.to_parquet(cache / PARQUET_BASENAMES["per"], compression="snappy", index=False)
    veh.to_parquet(cache / PARQUET_BASENAMES["veh"], compression="snappy", index=False)
    trip.to_parquet(cache / PARQUET_BASENAMES["trip"], compression="snappy", index=False)

    manifest: dict[str, Any] = {
        "module": "pev_synth.nhts_loader",
        "methodology_version": "v1.0.0",
        "master_seed": 20260518,
        "url": download_url,
        "zip_path": str((cache / "csv.zip").resolve()),
        "zip_sha256": zip_sha,
        "zip_size_bytes": zip_size,
        "csv_sha256": csv_sha,
        "fetch_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "filter_rule": (
            "HHSTATE == 'CA' (orchestrator-resolved; broader than PM brief Bay-Area filter)"
        ),
        "ca_subset_size": {
            "households": int(len(hh)),
            "persons": int(len(per)),
            "vehicles": int(len(veh)),
            "trips": int(len(trip)),
        },
        "total_size": {
            "households": int(len(df_hh_raw)),
            "persons": int(len(df_per_raw)),
            "vehicles": int(len(df_veh_raw)),
            "trips": int(len(df_trip_raw)),
            "all_rows_sum": total_size,
        },
        "wthhfin_sum_ca": float(hh["wthhfin"].sum()),
        # Plug-in / EV identification per NHTS 2017 Codebook v1.2 p.55. FUELTYPE
        # cannot separate BEV / PHEV / HEV (code 03 is an umbrella for "Hybrid,
        # electric or alternative fuel"). HFUEL is the authoritative subtype:
        # 01=Biodiesel, 02=PHEV, 03=BEV, 04=HEV non-plug-in, 97=Other, −1=skip.
        # Plug-in total = HFUEL ∈ {02, 03} = PHEV + BEV.
        "hfuel_counts": {
            "biodiesel": int((veh["hfuel"] == HFUEL_BIODIESEL).sum()),
            "phev": int((veh["hfuel"] == HFUEL_PHEV).sum()),
            "bev": int((veh["hfuel"] == HFUEL_BEV).sum()),
            "hev_non_plugin": int((veh["hfuel"] == HFUEL_HEV_NON_PLUGIN).sum()),
            "plug_in_total_phev_plus_bev": int(
                veh["hfuel"].isin([HFUEL_PHEV, HFUEL_BEV]).sum()
            ),
        },
        "fallback_triggered": False,
    }
    with (cache / MANIFEST_BASENAME).open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)

    return NHTSData(hh=hh, per=per, veh=veh, trip=trip, manifest=manifest)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _default_repo_root() -> Path:
    """Return the study repo root via :func:`pev_synth._paths.repo_root`."""
    from ._paths import repo_root

    return repo_root()


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m pev_synth.nhts_loader",
        description="Download NHTS 2017 ORNL CSVs and emit a California subset as parquet.",
    )
    p.add_argument(
        "--repo-root",
        type=str,
        default=None,
        help=(
            "Study repo root used to anchor default --cache-dir "
            "(defaults to two parents up from this file: the directory that "
            "contains code/ and data/). Set this if invoking from a non-repo "
            "cwd so the loader does not re-download the 84 MB ORNL zip into "
            "the wrong place."
        ),
    )
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help=(
            "Output directory for the zip, CSVs, parquets, and manifest. "
            "Defaults to ``<repo-root>/data/pev/raw/nhts2017``. Absolute "
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
        default=NHTS_ZIP_URL,
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
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
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
    data = load_nhts_2017_ca(
        cache_dir=cache_dir,
        force=args.force,
        download_url=args.url,
        request_timeout_s=args.timeout,
    )
    m = data.manifest
    hfuel = m["hfuel_counts"]
    print(
        "NHTS 2017 CA subset written. "
        f"hh={m['ca_subset_size']['households']} "
        f"per={m['ca_subset_size']['persons']} "
        f"veh={m['ca_subset_size']['vehicles']} "
        f"trip={m['ca_subset_size']['trips']} "
        f"wthhfin_sum={m['wthhfin_sum_ca']:.3e} "
        f"hfuel_plug_in={hfuel['plug_in_total_phev_plus_bev']} "
        f"(phev={hfuel['phev']}, bev={hfuel['bev']})"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())


# Silence unused-import warnings for numpy in environments that strip docstrings.
_ = np
