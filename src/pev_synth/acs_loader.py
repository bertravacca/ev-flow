"""ACS PUMS 5-year household microdata loader for ev-flow v3.0 (M3).

The American Community Survey (ACS) Public Use Microdata Sample (PUMS) provides
up-to-date household-mix priors (income / size / vehicle availability) at the
sub-state PUMA level. ev-flow consumes ACS PUMS strictly as a **calibration
target** for raking the NHTS household weights so the realised donor pool's
joint distribution over ``(income_tier, hhsize_band, hhvehcnt_band)`` matches
the region's current ACS-observed distribution. ACS PUMS is NOT a travel-diary
dataset and never replaces NHTS trip records.

Vintage default: 2020-2024 5-year PUMS (released 2026-03-05). Fallback:
2019-2023 5-year. Source: Census API
``https://api.census.gov/data/<end_year>/acs/acs5/pums``.

Variables (9, per Scientist S4 confirmation in
``docs/v3/modeling/nhts_enrichment.md`` §9.4):

- ``SERIALNO``  (string, year-prefixed; NOT int64)
- ``ST``        (state FIPS, Int8)
- ``PUMA``      (PUMA code, Int32)
- ``HINCP``     (household income, float64; recodes to NHTS hhfaminc 1-11)
- ``NP``        (number of persons, Int8 — stored as ``np_size`` to avoid
                 collision with the ``numpy`` import name)
- ``VEH``       (vehicle count, Int8; top-coded at 6)
- ``TEN``       (tenure, Int8)
- ``WGTP``      (household weight, float64)
- ``ADJINC``    (income inflation adjustment, 6-implied-decimals integer →
                 multiply HINCP)

Income binning: $35K / $100K / $150K (NHTS-2017-nominal cuts, aligned to the
``donor_matcher.INCOME_TIER_MAP``).

Output parquet (per ``(vintage, state)``):

``data/pev/raw/acs_pums/<years>/<state>/acs_pums_hh.parquet``

Manifest sidecar:

``data/pev/raw/acs_pums/<years>/<state>/acs_pums_manifest.json``

References
----------
- ACS 5-Year PUMS API page:
  https://www.census.gov/data/developers/data-sets/census-microdata-api/acs-5y-pums.html
- 2020-2024 ACS 5-year PUMS Data Dictionary (released 2026-03-05):
  https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMS_Data_Dictionary_2020-2024.pdf
- Scientist S4 dossier: ``docs/v3/modeling/nhts_enrichment.md`` §9.

Methodology version: v3.0.0.  Master seed: 20260520.  Module offset: see
``_seeds.py`` (the ACS loader itself is deterministic; no RNG is consumed).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
import requests

from ._seeds import METHODOLOGY_VERSION as _METHODOLOGY_VERSION

__all__ = [
    "ACS_API_BASE",
    "ACS_PUMS_VARIABLES",
    "DEFAULT_ACS_RAW_REL",
    "DEFAULT_VINTAGE",
    "FALLBACK_VINTAGE",
    "ACSLoaderConfig",
    "ACSPUMSData",
    "compute_household_mix_pmf",
    "load_acs_pums",
    "main",
    "recode_acs_income_to_nhts_tier",
    "STATE_FIPS",
]

METHODOLOGY_VERSION: Final[str] = _METHODOLOGY_VERSION

# --------------------------------------------------------------------------- #
# Module-level constants.
# --------------------------------------------------------------------------- #

ACS_API_BASE: Final[str] = "https://api.census.gov/data"

#: Variables fetched per household. Order is preserved into the URL.
ACS_PUMS_VARIABLES: Final[tuple[str, ...]] = (
    "SERIALNO", "ST", "PUMA", "HINCP", "NP", "VEH", "TEN", "WGTP", "ADJINC",
)

#: Default 5-year vintage label (per S4 §9.2).
DEFAULT_VINTAGE: Final[str] = "2020-2024"
#: Fallback 5-year vintage if the default endpoint fails.
FALLBACK_VINTAGE: Final[str] = "2019-2023"

#: Raw-zone subpath relative to ``data_root()``.
DEFAULT_ACS_RAW_REL: Final[Path] = Path("pev/raw/acs_pums")

#: Income tier cut-points in nominal dollars (S4 §9.5).
INCOME_TIER_CUTS: Final[tuple[float, float, float]] = (35_000.0, 100_000.0, 150_000.0)
INCOME_TIER_LABELS: Final[tuple[str, ...]] = ("t1", "t2", "t3", "t4")

#: Household-size band labels (S4 §9.7).
HHSIZE_BAND_LABELS: Final[tuple[str, ...]] = ("1", "2", "3", "4", "5+")
#: Vehicle-count band labels (S4 §9.7).
HHVEHCNT_BAND_LABELS: Final[tuple[str, ...]] = ("0", "1", "2", "3+")

#: U.S. state + DC + 5 island/PR FIPS codes (Census 2-digit). Keyed by USPS
#: code so callers can pass ``"CA"`` rather than ``"06"``.
STATE_FIPS: Final[dict[str, str]] = {
    "AL": "01", "AK": "02", "AZ": "04", "AR": "05", "CA": "06",
    "CO": "08", "CT": "09", "DE": "10", "DC": "11", "FL": "12",
    "GA": "13", "HI": "15", "ID": "16", "IL": "17", "IN": "18",
    "IA": "19", "KS": "20", "KY": "21", "LA": "22", "ME": "23",
    "MD": "24", "MA": "25", "MI": "26", "MN": "27", "MS": "28",
    "MO": "29", "MT": "30", "NE": "31", "NV": "32", "NH": "33",
    "NJ": "34", "NM": "35", "NY": "36", "NC": "37", "ND": "38",
    "OH": "39", "OK": "40", "OR": "41", "PA": "42", "RI": "44",
    "SC": "45", "SD": "46", "TN": "47", "TX": "48", "UT": "49",
    "VT": "50", "VA": "51", "WA": "53", "WV": "54", "WI": "55",
    "WY": "56",
    # PR + island territories.
    "PR": "72", "AS": "60", "GU": "66", "MP": "69", "VI": "78",
}

#: Persisted dtype contract (per S4 §9.6, with rename ``NP`` → ``np_size``).
HH_PARQUET_DTYPES: Final[dict[str, str]] = {
    "serialno": "string",
    "st": "Int8",
    "puma": "Int32",
    "hincp": "float64",
    "hincp_adj": "float64",
    "np_size": "Int8",
    "veh": "Int8",
    "ten": "Int8",
    "wgtp": "float64",
    "adjinc": "float64",
    "income_tier": "string",
    "hhsize_band": "string",
    "hhvehcnt_band": "string",
    "vintage": "string",
}

#: Output filenames inside ``<years>/<state>/``.
HH_PARQUET_BASENAME: Final[str] = "acs_pums_hh.parquet"
MANIFEST_BASENAME: Final[str] = "acs_pums_manifest.json"

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Config / data containers.
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ACSLoaderConfig:
    """User-tunable knobs for :func:`load_acs_pums`.

    Parameters
    ----------
    years:
        Vintage label, e.g. ``"2020-2024"``. Maps to the API ``<end_year>``
        slot by taking the right-hand half (``"2024"`` for the default).
    states:
        Iterable of USPS state codes (``"CA"``, ``"NY"``, ...). Empty tuple
        triggers an explicit failure rather than fetching the full US.
    cache_dir:
        Root under which ``<years>/<state>/`` parquet + manifest are written.
        ``None`` defers to :func:`_default_cache_root`.
    api_key:
        Optional Census API key (honoured for >500 calls/day). When ``None``
        the loader reads the ``CENSUS_API_KEY`` env var.
    timeout_s:
        Per-call HTTP timeout passed to :mod:`requests`.
    """

    years: str = DEFAULT_VINTAGE
    states: tuple[str, ...] = field(default_factory=tuple)
    cache_dir: Path | None = None
    api_key: str | None = None
    timeout_s: int = 60


@dataclass
class ACSPUMSData:
    """Container returned by :func:`load_acs_pums`.

    ``hh`` is the concatenated household-level frame across all requested
    states; ``manifest`` is the per-state manifest dict-of-dicts keyed by
    USPS code.
    """

    hh: pd.DataFrame
    manifest: dict[str, dict[str, object]]


# --------------------------------------------------------------------------- #
# Pure helpers (no I/O).
# --------------------------------------------------------------------------- #

def _end_year_from_vintage(years: str) -> int:
    """Return the API ``<end_year>`` integer for a vintage label.

    ``"2020-2024"`` → 2024.  ``"2024"`` → 2024 (1-year tolerated for
    forward-compat though v3.0 only ships 5-year).
    """
    if "-" in years:
        end = years.split("-", 1)[1]
    else:
        end = years
    try:
        return int(end)
    except ValueError as exc:
        raise ValueError(f"cannot parse end year from vintage label {years!r}") from exc


def _state_fips(usps_code: str) -> str:
    """Map a USPS state code to its 2-digit Census FIPS string."""
    code = usps_code.upper()
    if code not in STATE_FIPS:
        raise ValueError(
            f"unknown USPS state code {usps_code!r}; valid codes: "
            f"{sorted(STATE_FIPS.keys())}"
        )
    return STATE_FIPS[code]


def recode_acs_income_to_nhts_tier(hincp_adjusted: float) -> str:
    """Bin an inflation-adjusted ACS HINCP value to NHTS tier ``t1..t4``.

    Cuts at $35K / $100K / $150K (nominal, NHTS-2017 bracket edges; per
    S4 §9.5). Sentinel ``NaN`` returns the t1 floor — the loader drops
    rows with sentinel HINCP before binning, so NaN here means the cell
    was missing for non-GQ reasons (rare).

    Parameters
    ----------
    hincp_adjusted:
        ``HINCP * ADJINC / 1e6``.  Negative values (reported losses) bin
        to ``t1`` per the NHTS convention.
    """
    if hincp_adjusted is None or (isinstance(hincp_adjusted, float) and np.isnan(hincp_adjusted)):
        return INCOME_TIER_LABELS[0]
    val = float(hincp_adjusted)
    c1, c2, c3 = INCOME_TIER_CUTS
    if val < c1:
        return INCOME_TIER_LABELS[0]
    if val < c2:
        return INCOME_TIER_LABELS[1]
    if val < c3:
        return INCOME_TIER_LABELS[2]
    return INCOME_TIER_LABELS[3]


def _bin_hhsize(np_size: pd.Series) -> pd.Series:
    """Bin NP (persons in HH) into the v3 hhsize_band labels."""
    arr = pd.to_numeric(np_size, errors="coerce").to_numpy(dtype="float64")
    out = np.empty(arr.shape, dtype=object)
    out[:] = HHSIZE_BAND_LABELS[0]  # default placeholder
    out[arr >= 1] = HHSIZE_BAND_LABELS[0]
    out[arr >= 2] = HHSIZE_BAND_LABELS[1]
    out[arr >= 3] = HHSIZE_BAND_LABELS[2]
    out[arr >= 4] = HHSIZE_BAND_LABELS[3]
    out[arr >= 5] = HHSIZE_BAND_LABELS[4]
    # NaN handling: leave as NA in the output Series.
    s = pd.Series(out, index=np_size.index, dtype="string")
    s[~np.isfinite(arr)] = pd.NA
    return s


def _bin_hhvehcnt(veh: pd.Series) -> pd.Series:
    """Bin VEH into the v3 hhvehcnt_band labels (0/1/2/3+)."""
    arr = pd.to_numeric(veh, errors="coerce").to_numpy(dtype="float64")
    out = np.empty(arr.shape, dtype=object)
    out[:] = HHVEHCNT_BAND_LABELS[0]
    out[arr >= 0] = HHVEHCNT_BAND_LABELS[0]
    out[arr >= 1] = HHVEHCNT_BAND_LABELS[1]
    out[arr >= 2] = HHVEHCNT_BAND_LABELS[2]
    out[arr >= 3] = HHVEHCNT_BAND_LABELS[3]
    s = pd.Series(out, index=veh.index, dtype="string")
    s[~np.isfinite(arr)] = pd.NA
    return s


# --------------------------------------------------------------------------- #
# Census API fetch + parse.
# --------------------------------------------------------------------------- #

def _build_api_url(end_year: int, state_fips: str, api_key: str | None) -> str:
    """Compose the ACS PUMS API URL for one state."""
    vars_param = ",".join(ACS_PUMS_VARIABLES)
    url = (
        f"{ACS_API_BASE}/{end_year}/acs/acs5/pums"
        f"?get={vars_param}&for=state:{state_fips}"
    )
    if api_key:
        url += f"&key={api_key}"
    return url


def _fetch_state(url: str, timeout_s: int) -> list[list[str]]:
    """GET the Census API and parse the JSON body.

    The API returns a JSON array of arrays. Row 0 = header, rows 1.. = data.
    All values are strings (Census quirk). The caller dtypes-cast.

    Raises
    ------
    RuntimeError
        If the Census API redirects to the "missing key" HTML stub (the
        ACS PUMS endpoint enforces this even for single-state calls as
        of 2026-05; the older "≤500 calls/day key-optional" policy no
        longer applies to this dataset). The error message explicitly
        points callers to set ``CENSUS_API_KEY``.
    """
    resp = requests.get(url, timeout=timeout_s, allow_redirects=True)
    resp.raise_for_status()
    # Detect the "Missing Key" HTML stub that the API returns via a 302
    # redirect when no key is supplied (Content-Type: text/html).
    ctype = resp.headers.get("Content-Type", "")
    if "html" in ctype.lower() or resp.text.lstrip().startswith("<"):
        if "missing" in resp.text.lower() and "key" in resp.text.lower():
            raise RuntimeError(
                "Census API requires an API key for ACS PUMS access. "
                "Set the CENSUS_API_KEY environment variable to a valid "
                "key from https://api.census.gov/data/key_signup.html "
                "and retry. "
                f"(Effective URL: {resp.url})"
            )
        raise RuntimeError(
            f"Census API returned non-JSON ({ctype!r}); response starts: "
            f"{resp.text[:200]!r}"
        )
    body = resp.json()
    if not isinstance(body, list) or not body:
        raise RuntimeError(f"unexpected Census API body shape: {type(body).__name__}")
    return body


def _parse_state_payload(payload: list[list[str]], vintage: str) -> pd.DataFrame:
    """Convert raw Census payload into the persisted parquet schema.

    Steps
    -----
    1. Promote row 0 to column names; remaining rows form the data frame.
    2. Deduplicate to one record per ``SERIALNO`` (the API returns one row
       per person; HH-level columns are constant within a SERIALNO).
    3. Drop GQ / vacant / sentinel rows per S4 §9.6.
    4. Cast dtypes; compute ``hincp_adj``, ``income_tier``, ``hhsize_band``,
       ``hhvehcnt_band``.
    """
    header = payload[0]
    rows = payload[1:]
    df = pd.DataFrame(rows, columns=header)

    # Census payloads include the ``state`` echo column (the filter value);
    # drop it if present and not in our authoritative variable list.
    extras = [c for c in df.columns if c not in ACS_PUMS_VARIABLES]
    if extras:
        df = df.drop(columns=extras)

    # Deduplicate to HH level: keep first record per SERIALNO. (All requested
    # variables are HH-level so first==any.)
    df = df.drop_duplicates(subset="SERIALNO", keep="first").reset_index(drop=True)

    # Replace empty strings (Census sentinel for N/A on character columns)
    # with NA before casting.
    for col in df.columns:
        df[col] = df[col].replace({"": pd.NA})

    # Cast numerics. Coerce errors to NA (which then drops out via the
    # mandatory-non-null filter below).
    hincp = pd.to_numeric(df["HINCP"], errors="coerce")
    np_size = pd.to_numeric(df["NP"], errors="coerce")
    veh = pd.to_numeric(df["VEH"], errors="coerce")
    ten = pd.to_numeric(df["TEN"], errors="coerce")
    wgtp = pd.to_numeric(df["WGTP"], errors="coerce")
    adjinc_raw = pd.to_numeric(df["ADJINC"], errors="coerce")
    st = pd.to_numeric(df["ST"], errors="coerce")
    puma = pd.to_numeric(df["PUMA"], errors="coerce")

    # ADJINC has 6 implied decimals (e.g. 1015250 ≡ 1.015250).
    adjinc = adjinc_raw / 1_000_000.0
    hincp_adj = hincp * adjinc

    # Build the working frame.
    out = pd.DataFrame(
        {
            "serialno": df["SERIALNO"].astype("string"),
            "st": st.astype("Int8"),
            "puma": puma.astype("Int32"),
            "hincp": hincp.astype("float64"),
            "hincp_adj": hincp_adj.astype("float64"),
            "np_size": np_size.astype("Int8"),
            "veh": veh.astype("Int8"),
            "ten": ten.astype("Int8"),
            "wgtp": wgtp.astype("float64"),
            "adjinc": adjinc.astype("float64"),
        }
    )

    # Vacancy / GQ filter (S4 §9.6).
    n_pre = len(out)
    mask = (
        out["np_size"].notna() & (out["np_size"] > 0)
        & out["wgtp"].notna() & (out["wgtp"] > 0)
        & out["hincp"].notna()
        & out["veh"].notna()
        & out["ten"].isin([1, 2, 3, 4])
    )
    out = out.loc[mask].reset_index(drop=True)
    n_post = len(out)
    logger.info(
        "ACS PUMS parse: dropped %d / %d rows as GQ/vacant/sentinel; kept %d.",
        n_pre - n_post, n_pre, n_post,
    )

    # Derived columns.
    out["income_tier"] = (
        out["hincp_adj"].map(recode_acs_income_to_nhts_tier).astype("string")
    )
    out["hhsize_band"] = _bin_hhsize(out["np_size"])
    out["hhvehcnt_band"] = _bin_hhvehcnt(out["veh"])
    out["vintage"] = pd.array([vintage] * len(out), dtype="string")

    # Column order matches HH_PARQUET_DTYPES.
    out = out[list(HH_PARQUET_DTYPES.keys())]
    return out


# --------------------------------------------------------------------------- #
# Top-level loader + persistence.
# --------------------------------------------------------------------------- #

def _default_cache_root() -> Path:
    """Return the default raw-zone root for ACS PUMS parquets."""
    from ._paths import data_root

    return data_root() / DEFAULT_ACS_RAW_REL


def _state_dir(cache_root: Path, vintage: str, usps_code: str) -> Path:
    return cache_root / vintage / usps_code.upper()


def _resolve_api_key(explicit: str | None) -> tuple[str | None, bool]:
    """Return (effective_key_or_None, key_used_bool).

    Reads ``CENSUS_API_KEY`` env var when ``explicit`` is None.
    """
    key = explicit if explicit is not None else os.environ.get("CENSUS_API_KEY")
    return key, bool(key)


def _write_manifest(
    out_dir: Path,
    *,
    api_url: str,
    n_rows: int,
    key_used: bool,
    vintage: str,
    usps_code: str,
) -> Path:
    """Write the per-state manifest JSON sidecar; return its path."""
    manifest: dict[str, object] = {
        "module": "pev_synth.acs_loader",
        "methodology_version": METHODOLOGY_VERSION,
        "vintage": vintage,
        "state": usps_code.upper(),
        "api_endpoint": api_url.split("&key=", 1)[0],  # strip key
        "api_key_used": bool(key_used),
        "row_count": int(n_rows),
        "retrieval_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "variables": list(ACS_PUMS_VARIABLES),
    }
    out_path = out_dir / MANIFEST_BASENAME
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, out_path)
    return out_path


def _write_parquet(df: pd.DataFrame, out_dir: Path) -> Path:
    """Persist the HH frame with the canonical dtype contract."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / HH_PARQUET_BASENAME
    cast = df.copy()
    for col, dtype in HH_PARQUET_DTYPES.items():
        if col not in cast.columns:
            raise KeyError(f"acs_pums frame missing required column {col!r}")
        cast[col] = cast[col].astype(dtype)
    cast = cast[list(HH_PARQUET_DTYPES.keys())]
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    cast.to_parquet(tmp, engine="pyarrow", index=False, compression="snappy")
    os.replace(tmp, out_path)
    return out_path


def load_acs_pums(
    states: tuple[str, ...] | list[str],
    years: str = DEFAULT_VINTAGE,
    cache_dir: Path | None = None,
    *,
    force: bool = False,
    api_key: str | None = None,
    timeout_s: int = 60,
) -> ACSPUMSData:
    """Fetch and cache ACS PUMS 5-year household microdata.

    One Census API call is made per state. The per-state parquet +
    manifest are written under
    ``<cache_dir>/<years>/<state>/``. Cached parquets are re-used unless
    ``force=True``.

    Parameters
    ----------
    states:
        USPS state codes to fetch (e.g. ``("CA", "NY")``). Empty input
        raises ``ValueError`` — the loader refuses to fetch the full US
        implicitly to keep accidental API-key-less floods rare.
    years:
        Vintage label; default ``"2020-2024"``. Mapped to API
        ``<end_year>`` via :func:`_end_year_from_vintage`.
    cache_dir:
        Override for the raw-zone root; defaults to
        ``<data_root>/pev/raw/acs_pums``.
    force:
        When True, ignore on-disk cache and refetch.
    api_key:
        Override for ``CENSUS_API_KEY`` env var.
    timeout_s:
        Per-call HTTP timeout.

    Returns
    -------
    ACSPUMSData
        ``hh`` is the concatenated frame across all requested states;
        ``manifest`` is keyed by USPS code with per-state metadata.
    """
    if not states:
        raise ValueError(
            "load_acs_pums requires at least one state code; pass e.g. states=('CA',)"
        )
    states_list = tuple(s.upper() for s in states)
    cache_root = Path(cache_dir) if cache_dir is not None else _default_cache_root()
    end_year = _end_year_from_vintage(years)
    effective_key, key_used = _resolve_api_key(api_key)

    frames: list[pd.DataFrame] = []
    manifest: dict[str, dict[str, object]] = {}
    for usps_code in states_list:
        state_fips = _state_fips(usps_code)
        out_dir = _state_dir(cache_root, years, usps_code)
        parquet_path = out_dir / HH_PARQUET_BASENAME
        manifest_path = out_dir / MANIFEST_BASENAME
        api_url = _build_api_url(end_year, state_fips, effective_key)

        if parquet_path.is_file() and manifest_path.is_file() and not force:
            logger.info(
                "ACS PUMS: %s %s already cached at %s (use force=True to refetch).",
                usps_code, years, parquet_path,
            )
            df_state = pd.read_parquet(parquet_path)
            manifest[usps_code] = json.loads(manifest_path.read_text(encoding="utf-8"))
        else:
            logger.info("ACS PUMS: fetching %s (%s) → %s", usps_code, years, api_url)
            payload = _fetch_state(api_url, timeout_s=timeout_s)
            df_state = _parse_state_payload(payload, vintage=years)
            _write_parquet(df_state, out_dir)
            _write_manifest(
                out_dir,
                api_url=api_url,
                n_rows=len(df_state),
                key_used=key_used,
                vintage=years,
                usps_code=usps_code,
            )
            manifest[usps_code] = json.loads(manifest_path.read_text(encoding="utf-8"))

        frames.append(df_state)

    hh = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return ACSPUMSData(hh=hh, manifest=manifest)


# --------------------------------------------------------------------------- #
# Household-mix PMF (consumed by M3 raking).
# --------------------------------------------------------------------------- #

def compute_household_mix_pmf(
    acs_hh: pd.DataFrame,
    region: str | None = None,
    *,
    income_tier_scheme: str = "nhts_hhfaminc",
    hhsize_bins: str = "1,2,3,4,5+",
    veh_bins: str = "0,1,2,3+",
) -> pd.DataFrame:
    """Return the WGTP-weighted expected joint PMF over the M3 match keys.

    The output frame is consumable by
    :func:`pev_synth.donor_matcher.apply_acs_calibration_weights`.

    Parameters
    ----------
    acs_hh:
        Household frame from :func:`load_acs_pums` (carries ``income_tier``,
        ``hhsize_band``, ``hhvehcnt_band``, ``wgtp``).
    region:
        Optional name (purely a label echoed for traceability — actual
        per-region filtering happens upstream when the caller subsets
        ``acs_hh`` to the region's state(s)).
    income_tier_scheme, hhsize_bins, veh_bins:
        Knobs reserved for future binning variants. v3.0 supports the
        single ``"nhts_hhfaminc"`` scheme with NHTS-2017-nominal cuts;
        the bin-string arguments must match the module defaults or a
        :class:`ValueError` is raised.

    Returns
    -------
    pd.DataFrame
        Columns ``(income_tier, hhsize_band, hhvehcnt_band, p_expected)``,
        where ``p_expected`` sums to 1.0 across rows (within float
        tolerance, after dropping zero-weight cells). The ``region`` label
        is attached via ``out.attrs["region"]`` when provided.
    """
    if income_tier_scheme != "nhts_hhfaminc":
        raise ValueError(
            f"income_tier_scheme={income_tier_scheme!r} not supported in v3.0; "
            "only 'nhts_hhfaminc' is implemented"
        )
    if hhsize_bins != "1,2,3,4,5+" or veh_bins != "0,1,2,3+":
        raise ValueError(
            "v3.0 supports only the default bin schemes "
            "(hhsize_bins='1,2,3,4,5+', veh_bins='0,1,2,3+')"
        )

    required = {"income_tier", "hhsize_band", "hhvehcnt_band", "wgtp"}
    missing = required - set(acs_hh.columns)
    if missing:
        raise KeyError(f"acs_hh missing required columns: {sorted(missing)}")
    if acs_hh.empty:
        raise ValueError("compute_household_mix_pmf received an empty acs_hh frame")

    keys = ["income_tier", "hhsize_band", "hhvehcnt_band"]
    work = acs_hh[keys + ["wgtp"]].copy()
    # Drop rows with NA on any key (band binners propagate NA on bad inputs).
    work = work.dropna(subset=keys + ["wgtp"])

    cell_w = work.groupby(keys, observed=True)["wgtp"].sum()
    total = float(cell_w.sum())
    if total <= 0:
        raise ValueError("compute_household_mix_pmf: total WGTP is non-positive")
    pmf = (cell_w / total).rename("p_expected").reset_index()

    # Drop zero-weight cells for a clean PMF (sum tends to 1.0 exactly).
    pmf = pmf[pmf["p_expected"] > 0].reset_index(drop=True)
    if region is not None:
        pmf.attrs["region"] = str(region)
    return pmf


# --------------------------------------------------------------------------- #
# CLI.
# --------------------------------------------------------------------------- #

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m pev_synth.acs_loader",
        description=(
            "Fetch and cache Census ACS PUMS 5-year household microdata "
            "for one or more states. Writes parquet + manifest under "
            "<out>/<years>/<state>/."
        ),
    )
    p.add_argument(
        "--states",
        type=str,
        required=True,
        help="Comma-separated USPS state codes, e.g. 'CA,NY,TX'.",
    )
    p.add_argument(
        "--years",
        type=str,
        default=DEFAULT_VINTAGE,
        help=f"ACS PUMS 5-year vintage label (default: {DEFAULT_VINTAGE}).",
    )
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help=(
            "Output raw-zone root (default: <data_root>/pev/raw/acs_pums). "
            "Per-state files land at <out>/<years>/<state>/."
        ),
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-fetch even if cache is present.",
    )
    p.add_argument(
        "--timeout-s",
        type=int,
        default=60,
        help="Per-state HTTP timeout (seconds).",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return p


def main(argv: list[str] | None = None) -> int:
    """CLI entry-point for ``python -m pev_synth.acs_loader``."""
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    states = tuple(s.strip().upper() for s in args.states.split(",") if s.strip())
    if not states:
        print("--states must list at least one USPS code", file=sys.stderr)
        return 2

    out = Path(args.out) if args.out else None
    result = load_acs_pums(
        states=states,
        years=args.years,
        cache_dir=out,
        force=args.force,
        timeout_s=args.timeout_s,
    )

    for usps_code, mani in result.manifest.items():
        print(
            f"ACS PUMS {args.years} {usps_code}: "
            f"rows={mani.get('row_count')} "
            f"key_used={mani.get('api_key_used')} "
            f"endpoint={mani.get('api_endpoint')}",
            file=sys.stderr,
        )
    print(
        f"ACS PUMS loader: wrote {len(result.manifest)} state(s); "
        f"total HH rows={len(result.hh)}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
