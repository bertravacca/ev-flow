"""M3 -- pev_synth.donor_matcher (v2.0 refactor).

For each synthetic EV in the M2 fleet, select an NHTS-2017 donor household
and donor vehicle by stratified matching on (income_tier, hhsize_band,
urbrur, hhvehcnt_band), with a soft preference for vehicles whose HFUEL
code matches the synthetic powertrain (PHEV -> 2, BEV -> 3).

v2.0 lifts the v1.1 single-Bay-Area assumption:

* The donor pool is narrowed by ``region.states`` (state filter) and
  ``region.cbsas`` (CBSA filter); the ``us_national`` region uses no
  CBSA filter so the full national pool participates.
* For ``profile_type == 'workplace'``: an extra strict ANNMILES < 3600
  filter applies, after which a borrow-from-neighbours rule augments
  thin pools from ``region.workplace_donor_borrow_states`` (in listed
  order) until the pool reaches the configured threshold.  Borrow
  share is capped at 25% of the total pool (W10 in plan §4.8).
* Output rows carry two new provenance columns: ``donor_origin_state``
  (the donor's actual HHSTATE; non-null for borrowed rows, null for
  in-region rows) and ``borrow_used`` (bool).

References
----------
* v2.0 methodology: workplace donor borrow-from-neighbours (§2.6),
  donor selection (§4.3), ANNMILES workplace-only-feasible filter
  (§4.6), borrow logic (§4.7). Module 5 (donor_matcher).
* NHTS 2017 Codebook v1.2 p.55 -- HFUEL semantics
  (1=Biodiesel, 2=PHEV, 3=BEV, 4=HEV non-plug-in).

Methodology version: v2.0.0.  Master seed: 20260520.  Module offset: +2.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from ._seeds import (
    MASTER_SEED as _MASTER_SEED,
)
from ._seeds import (
    METHODOLOGY_VERSION as _METHODOLOGY_VERSION,
)
from ._seeds import (
    module_offset as _module_offset,
)
from .regions import PROFILE_TYPES, Region

__all__ = [
    "ANNMILES_WORKPLACE_MAX",
    "BORROW_POOL_THRESHOLD",
    "BORROW_SHARE_CAP",
    "DonorMatchConfig",
    "DonorMatchResult",
    "INCOME_TIER_MAP",
    "NHTS_VINTAGE_2017",
    "NHTS_VINTAGE_2022_NEXTGEN",
    "apply_acs_calibration_weights",
    "build_match_keys",
    "load_region_nhts",
    "match_donors",
    "narrow_to_region",
    "main",
    # v3.0 D5 NHTS NextGen helpers (private — exposed for tests + sibling modules).
    "_apply_vintage_mix",
    "_model_backoff_draw_from_2017",
]

# v3.0 NHTS vintage tags. These literals are also written by
# :mod:`pev_synth.nhts_loader` (back-compat shim, see ``_read_ca_parquets``)
# and :mod:`pev_synth.nhts_nextgen_loader`.
NHTS_VINTAGE_2017: Final[str] = "2017"
NHTS_VINTAGE_2022_NEXTGEN: Final[str] = "2022_nextgen"

# --------------------------------------------------------------------------- #
# Orchestrator-resolved constants.
# --------------------------------------------------------------------------- #

METHODOLOGY_VERSION: Final[str] = _METHODOLOGY_VERSION
MASTER_SEED_DEFAULT: Final[int] = _MASTER_SEED
MODULE_OFFSET: Final[int] = _module_offset("donor_matcher")  # +2

# NHTS Codebook v1.2 p.55.
HFUEL_BIODIESEL: Final[int] = 1
HFUEL_PHEV: Final[int] = 2
HFUEL_BEV: Final[int] = 3
HFUEL_HEV_NON_PLUGIN: Final[int] = 4

# v2.0 workplace constraints (orchestrator-resolved per PM brief + plan §4.6).
#: Strict cap: ANNMILES < 3600 mi/yr is the workplace-only-feasible cohort.
ANNMILES_WORKPLACE_MAX: Final[float] = 3600.0
#: Borrow trigger: if post-filter pool size is below this threshold and
#: the region has non-empty ``workplace_donor_borrow_states``, augment.
BORROW_POOL_THRESHOLD: Final[int] = 1500
#: Borrow share cap (W10): drop borrowed rows once their share would
#: exceed this fraction of total pool.
BORROW_SHARE_CAP: Final[float] = 0.25
#: Soft warning floor: if post-borrow pool is below this, log a warning
#: but proceed (v2.1 concern).
POOL_WARNING_FLOOR: Final[int] = 500

# HHFAMINC tier mapping (PM brief §6, plan §3.2 step 10).
INCOME_TIER_MAP: Final[dict[int, str]] = {
    1: "t1", 2: "t1", 3: "t1", 4: "t1",
    5: "t2", 6: "t2", 7: "t2",
    8: "t3", 9: "t3",
    10: "t4", 11: "t4",
}
INCOME_TIER_NULL_DEFAULT: Final[str] = "t2"
INCOME_TIER_CATEGORIES: Final[tuple[str, ...]] = ("t1", "t2", "t3", "t4")

# HHSIZE / HHVEHCNT band labels (plan §4.2; PM brief).
HHSIZE_BAND_LABELS: Final[tuple[str, ...]] = ("1", "2", "3", "4+")
HHVEHCNT_BAND_LABELS: Final[tuple[str, ...]] = ("1", "2", "3", "4+")

# Relaxation ladder (0 = strictest, 4 = fallback).
RELAXATION_LEVELS: Final[tuple[str, ...]] = (
    "income+hhsize+urbrur+hhvehcnt",
    "income+hhsize+urbrur",
    "income+urbrur",
    "income",
    "any",
)

# Default NHTS data root (per-region intermediate parquets land in
# data/pev/intermediate/nhts/<region>/{hh,veh}.parquet eventually; until
# Dev B/E finalise that, we fall back to the M1 outputs in
# data/pev/raw/nhts2017/).
DEFAULT_NHTS_DATA_DIR: Final[Path] = Path("data/pev/raw/nhts2017")
DEFAULT_FLEET_REL: Final[Path] = Path("data/pev/processed/residential_ev_synth/fleet.parquet")
DEFAULT_META_REL: Final[Path] = Path("data/pev/processed/residential_ev_synth/meta.json")


logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Config / result types.
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class DonorMatchConfig:
    """User-tunable knobs (PM brief §7)."""

    master_seed: int = MASTER_SEED_DEFAULT
    nhts_data_dir: Path = DEFAULT_NHTS_DATA_DIR
    annmiles_max_workplace: float = ANNMILES_WORKPLACE_MAX
    borrow_pool_threshold: int = BORROW_POOL_THRESHOLD
    borrow_share_cap: float = BORROW_SHARE_CAP


@dataclass
class DonorMatchResult:
    """Outcome of one M3 run; carried into meta.json provenance."""

    fleet: pd.DataFrame
    region_name: str
    profile_type: str
    n_hh_in_region: int
    n_hh_after_annmiles_filter: int
    n_hh_borrowed: int
    n_hh_total_pool: int
    borrow_share: float
    relaxation_counts: dict[int, int]
    unique_donor_houseids: int


# --------------------------------------------------------------------------- #
# Pure helpers.
# --------------------------------------------------------------------------- #

def _band_4plus(values: pd.Series, labels: tuple[str, ...] = HHSIZE_BAND_LABELS) -> pd.Series:
    """Bin an integer Series into {"1","2","3","4+"} by clipping at >=4."""
    if values.isna().any():
        raise ValueError("band_4plus received NA values; upstream contract violated")
    arr = values.astype("int64").to_numpy()
    bands = np.where(arr >= 4, labels[3], arr.astype(str))
    bands = np.where(bands == "0", labels[0], bands)
    return pd.Series(bands, index=values.index, dtype="string")


def _map_income_tier(hhfaminc: pd.Series) -> pd.Series:
    """Map NHTS HHFAMINC 1..11 -> income tier {t1, t2, t3, t4}."""
    arr = pd.to_numeric(hhfaminc, errors="coerce")
    tiers = arr.map(INCOME_TIER_MAP)
    tiers = tiers.fillna(INCOME_TIER_NULL_DEFAULT)
    return tiers.astype("string")


def build_match_keys(hh: pd.DataFrame) -> pd.DataFrame:
    """Augment a household frame with the four match-key columns."""
    out = hh.copy()
    out["income_tier"] = _map_income_tier(out["hhfaminc"])
    out["hhsize_band"] = _band_4plus(out["hhsize"])
    out["hhvehcnt_band"] = _band_4plus(out["hhvehcnt"])
    out["urbrur_int"] = pd.to_numeric(out["urbrur"], errors="coerce").astype("Int8")
    return out


# --------------------------------------------------------------------------- #
# NHTS loader (CA parquet + national HHPUB fallback).
# --------------------------------------------------------------------------- #

_HH_PARQUET_COLUMNS: Final[tuple[str, ...]] = (
    "houseid", "hhstate", "hh_cbsa", "urbrur", "hhfaminc", "hhsize", "hhvehcnt", "wthhfin",
)
_VEH_PARQUET_COLUMNS: Final[tuple[str, ...]] = (
    "houseid", "vehid", "vehtype", "make", "model", "vehyear", "annmiles", "hfuel",
)
_NATIONAL_HHPUB_COLUMNS: Final[tuple[str, ...]] = (
    "HOUSEID", "HHSTATE", "HH_CBSA", "URBRUR", "HHFAMINC", "HHSIZE", "HHVEHCNT", "WTHHFIN",
)
_NATIONAL_VEHPUB_COLUMNS: Final[tuple[str, ...]] = (
    "HOUSEID", "VEHID", "VEHTYPE", "MAKE", "MODEL", "VEHYEAR", "ANNMILES", "HFUEL",
)


def _ensure_vintage_column(df: pd.DataFrame, vintage: str) -> pd.DataFrame:
    """Stamp ``nhts_vintage = vintage`` on every row if the column is absent.

    v3.0 back-compat shim: 2017 parquets written by :mod:`nhts_loader`
    pre-date the vintage column. We stamp at read time so downstream
    partition logic can rely on the column being present without needing
    to touch ``nhts_loader``.
    """
    if "nhts_vintage" not in df.columns:
        df = df.copy()
        df["nhts_vintage"] = pd.array([vintage] * len(df), dtype="string")
    else:
        # Already present (NextGen loader or future re-emit) — leave intact.
        df["nhts_vintage"] = df["nhts_vintage"].astype("string")
    return df


def _read_ca_parquets(nhts_data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    hh = pd.read_parquet(nhts_data_dir / "nhts2017_ca_hh.parquet")
    veh = pd.read_parquet(nhts_data_dir / "nhts2017_ca_veh.parquet")
    hh = _ensure_vintage_column(hh, NHTS_VINTAGE_2017)
    veh = _ensure_vintage_column(veh, NHTS_VINTAGE_2017)
    return hh, veh


def _read_national_csvs(nhts_data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    hh = pd.read_csv(
        nhts_data_dir / "hhpub.csv",
        usecols=list(_NATIONAL_HHPUB_COLUMNS),
        dtype={"HH_CBSA": "string", "HHSTATE": "string"},
    )
    veh = pd.read_csv(
        nhts_data_dir / "vehpub.csv",
        usecols=list(_NATIONAL_VEHPUB_COLUMNS),
        dtype={"MAKE": "string", "MODEL": "string"},
    )
    # Lowercase columns + cast dtypes consistent with the CA parquet.
    hh = hh.rename(columns={c: c.lower() for c in hh.columns})
    veh = veh.rename(columns={c: c.lower() for c in veh.columns})
    hh["hhstate"] = hh["hhstate"].astype("string")
    hh["hh_cbsa"] = hh["hh_cbsa"].astype("string").str.strip().replace("", pd.NA)
    hh["urbrur"] = pd.to_numeric(hh["urbrur"], errors="coerce").astype("Int8")
    hh["hhfaminc"] = pd.to_numeric(hh["hhfaminc"], errors="coerce").astype("Int8")
    hh["hhsize"] = pd.to_numeric(hh["hhsize"], errors="coerce").astype("Int8")
    hh["hhvehcnt"] = pd.to_numeric(hh["hhvehcnt"], errors="coerce").astype("Int8")
    hh["wthhfin"] = pd.to_numeric(hh["wthhfin"], errors="coerce").astype("float64")

    veh["houseid"] = veh["houseid"].astype("int64")
    veh["vehid"] = pd.to_numeric(veh["vehid"], errors="coerce").astype("int32")
    veh["vehtype"] = pd.to_numeric(veh["vehtype"], errors="coerce").astype("Int8")
    veh["vehyear"] = pd.to_numeric(veh["vehyear"], errors="coerce").astype("Int16")
    veh["annmiles"] = pd.to_numeric(veh["annmiles"], errors="coerce").astype("float32")
    veh["hfuel"] = pd.to_numeric(veh["hfuel"], errors="coerce").astype("Int8")
    hh = _ensure_vintage_column(hh, NHTS_VINTAGE_2017)
    veh = _ensure_vintage_column(veh, NHTS_VINTAGE_2017)
    return hh, veh


# --------------------------------------------------------------------------- #
# v3.0 NHTS NextGen 2022 supplementary stratum loader.
# --------------------------------------------------------------------------- #

#: NextGen 2022 parquet basenames (must match
#: :data:`pev_synth.nhts_nextgen_loader.NEXTGEN_PARQUET_BASENAMES`; duplicated
#: here so this module does not import the loader at import time).
_NEXTGEN_HH_PARQUET: Final[str] = "nhts2022_national_hh.parquet"
_NEXTGEN_VEH_PARQUET: Final[str] = "nhts2022_national_veh.parquet"


def _read_nextgen_parquets(
    nhts_nextgen_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Return (hh, veh) NextGen 2022 frames or ``None`` if parquets absent."""
    hh_path = nhts_nextgen_dir / _NEXTGEN_HH_PARQUET
    veh_path = nhts_nextgen_dir / _NEXTGEN_VEH_PARQUET
    if not (hh_path.is_file() and veh_path.is_file()):
        return None
    hh = pd.read_parquet(hh_path)
    veh = pd.read_parquet(veh_path)
    hh = _ensure_vintage_column(hh, NHTS_VINTAGE_2022_NEXTGEN)
    veh = _ensure_vintage_column(veh, NHTS_VINTAGE_2022_NEXTGEN)
    return hh, veh


def load_region_nhts(
    region: Region,
    nhts_data_dir: Path,
    *,
    extra_vintages: tuple[str, ...] = (),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load NHTS HH + VEH frames filtered by ``region.states``.

    Strategy:
        * If the region's only state is CA, load the curated CA parquet
          (cleaner, smaller, faster).
        * Otherwise read the national HHPUB / VEHPUB CSVs and filter by
          ``region.states``.
        * v3.0: when ``extra_vintages`` includes ``"2022_nextgen"``, also
          load NHTS NextGen 2022 national parquets from
          ``nhts_data_dir.parent / "nhts2022/"`` (when present) and union
          them onto the 2017 frames. Each row carries an ``nhts_vintage``
          column so :func:`narrow_to_region` can partition behaviour.
          When the NextGen parquets are absent, a warning is logged and
          the 2017-only result is returned (no crash).

    With ``extra_vintages = ()`` (the default), the v2.1 byte-stable
    behaviour is preserved, with the only addition being the
    ``nhts_vintage = "2017"`` back-compat column stamped on every row at
    read time (see :func:`_ensure_vintage_column`).

    Returned frames carry lowercase column names matching the v1.1
    CA-parquet schema, plus the ``nhts_vintage`` column. NextGen-2022 rows
    carry ``hhstate = hh_cbsa = hhstfips = <NA>`` (suppressed in the public
    file) and vehicle rows carry ``model = <NA>`` —
    :func:`_model_backoff_draw_from_2017` fills the missing model from
    the 2017 marginal when the caller requests it.
    """
    nhts_data_dir = Path(nhts_data_dir)
    needs_national = set(region.states) - {"CA"}
    if not needs_national:
        hh, veh = _read_ca_parquets(nhts_data_dir)
    else:
        hh, veh = _read_national_csvs(nhts_data_dir)

    if region.states:
        hh = hh[hh["hhstate"].isin(list(region.states))].reset_index(drop=True)
        veh = veh[veh["houseid"].isin(hh["houseid"])].reset_index(drop=True)

    if NHTS_VINTAGE_2022_NEXTGEN in extra_vintages:
        nextgen_dir = nhts_data_dir.parent / "nhts2022"
        loaded = _read_nextgen_parquets(nextgen_dir)
        if loaded is None:
            logger.warning(
                "load_region_nhts: extra_vintages requested '%s' but parquets "
                "missing at %s; returning 2017-only frames",
                NHTS_VINTAGE_2022_NEXTGEN,
                nextgen_dir,
            )
        else:
            ng_hh, ng_veh = loaded
            hh = pd.concat([hh, ng_hh], ignore_index=True, sort=False)
            veh = pd.concat([veh, ng_veh], ignore_index=True, sort=False)
    return hh, veh


# --------------------------------------------------------------------------- #
# Region narrowing.
# --------------------------------------------------------------------------- #

def narrow_to_region(hh: pd.DataFrame, region: Region) -> pd.DataFrame:
    """Filter ``hh`` to ``region.states`` and ``region.cbsas``.

    Households whose ``hh_cbsa`` is null are dropped when CBSA narrowing
    is in effect.  For ``us_national`` (no CBSAs), the function is a
    state-filter pass-through.

    v3.0 vintage-aware partitioning:
        * 2017 rows go through the v2.1 state + CBSA narrowing.
        * 2022_nextgen rows bypass state/CBSA narrowing entirely (their
          ``hhstate`` / ``hh_cbsa`` are suppressed in the public file;
          they participate as a **national supplementary stratum** for
          every region — see ``docs/v3/modeling/nhts_enrichment.md`` §6.5).
        * The two slices are concatenated back so callers always see a
          single frame.

    When the input frame has no ``nhts_vintage`` column (legacy callers
    on the v2.1 path) or every row is tagged ``"2017"``, the v2.1
    narrowing logic runs unchanged — byte-stable.
    """
    if "hh_cbsa" not in hh.columns:
        raise KeyError("hh frame missing required column hh_cbsa")

    has_vintage_col = "nhts_vintage" in hh.columns
    if not has_vintage_col:
        # v2.1 legacy path — preserved bit-for-bit.
        out = hh
        if region.states:
            out = out[out["hhstate"].isin(list(region.states))]
        if region.cbsas:
            cbsa_strs = [str(c) for c in region.cbsas]
            out = out[out["hh_cbsa"].isin(cbsa_strs)]
        return out.reset_index(drop=True)

    vintage_series = hh["nhts_vintage"].astype("string")
    is_nextgen = (vintage_series == NHTS_VINTAGE_2022_NEXTGEN).to_numpy()
    # Anything that is not explicitly nextgen is treated as legacy 2017
    # (covers blank / NA vintage tags from legacy parquets too — defensive).
    is_legacy = ~is_nextgen

    legacy = hh[is_legacy]
    if region.states:
        legacy = legacy[legacy["hhstate"].isin(list(region.states))]
    if region.cbsas:
        cbsa_strs = [str(c) for c in region.cbsas]
        legacy = legacy[legacy["hh_cbsa"].isin(cbsa_strs)]

    if not is_nextgen.any():
        return legacy.reset_index(drop=True)

    nextgen = hh[is_nextgen]
    out = pd.concat([legacy, nextgen], ignore_index=True, sort=False)
    return out


# --------------------------------------------------------------------------- #
# v3.0 vintage-mix re-weighting + NextGen MODEL back-off.
# --------------------------------------------------------------------------- #

def _apply_vintage_mix(
    hh: pd.DataFrame,
    vintage_mix: Mapping[str, float],
) -> pd.DataFrame:
    """Re-weight ``hh.wthhfin`` so each vintage's share matches ``vintage_mix``.

    Algorithm:
        1. Within each vintage present in ``hh``, normalise ``wthhfin`` to
           sum to ``vintage_mix[vintage]``. Vintages missing from
           ``vintage_mix`` are dropped (zero weight).
        2. The combined ``wthhfin`` then sums to ``sum(vintage_mix.values())``
           — for the canonical case ``{"2017": p, "2022_nextgen": 1-p}``
           this is 1.0.

    Notes
    -----
    * Deterministic: pure pandas, no RNG. The per-EV vintage Bernoulli
      that **selects** a vintage uses the
      ``NHTS_VINTAGE_SELECT_SUBSEED_OFFSET`` stream; that selection
      happens downstream in
      :func:`pev_synth.travel_week_builder.sample_week_for_ev_v3`. This
      helper only rescales the donor pool so weighted sampling respects
      the configured mix.
    * When ``vintage_mix == {"2017": 1.0}`` (the default mix on every
      shipped Region), the function is a normalisation-to-1.0 pass on the
      2017 slice and drops anything else — byte-stable when the input
      already contained only 2017 rows.
    """
    if "nhts_vintage" not in hh.columns:
        raise KeyError("_apply_vintage_mix requires an 'nhts_vintage' column")
    if "wthhfin" not in hh.columns:
        raise KeyError("_apply_vintage_mix requires a 'wthhfin' column")

    mix = dict(vintage_mix)
    if not mix:
        raise ValueError("_apply_vintage_mix requires a non-empty vintage_mix")

    out = hh.copy()
    out["wthhfin"] = pd.to_numeric(out["wthhfin"], errors="coerce").astype("float64")
    vintage_series = out["nhts_vintage"].astype("string")

    # Drop vintages not in the mix.
    keep_mask = vintage_series.isin(list(mix.keys()))
    out = out[keep_mask.to_numpy()].copy().reset_index(drop=True)
    vintage_series = out["nhts_vintage"].astype("string")

    new_weights = out["wthhfin"].to_numpy(dtype="float64", copy=True)
    for vintage, target_share in mix.items():
        sub_mask = (vintage_series == vintage).to_numpy()
        if not sub_mask.any():
            continue
        sub_w = out.loc[sub_mask, "wthhfin"].to_numpy(dtype="float64")
        total = float(sub_w.sum())
        if not np.isfinite(total) or total <= 0:
            # Degenerate: assign uniform within the vintage so its slice
            # still sums to the target share.
            n = int(sub_mask.sum())
            new_weights[sub_mask] = float(target_share) / max(n, 1)
        else:
            scale = float(target_share) / total
            new_weights[sub_mask] = sub_w * scale

    out["wthhfin"] = new_weights.astype("float64")
    return out


def _model_backoff_draw_from_2017(
    veh: pd.DataFrame,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Fill ``veh.model`` for NextGen 2022 rows from the 2017 conditional draw.

    NextGen 2022 has no ``MODEL`` column (ORNL privacy suppression on
    ~14k vehicles). For each ``nhts_vintage == "2022_nextgen"`` row with
    ``model`` missing, sample one ``model`` value from the 2017
    conditional distribution given the matched tuple, with a 4-level
    back-off ladder:

        1. ``(make, vehyear, hfuel, fueltype)``
        2. ``(make, vehyear, hfuel)``
        3. ``(make, hfuel)``
        4. ``(make,)``

    If the make is itself absent from the 2017 marginal, ``model`` is
    set to ``"UNKNOWN"`` (downstream archetype lookups fall back to a
    make-level mean).

    Parameters
    ----------
    veh
        VEH frame already unioned via :func:`load_region_nhts` (must
        carry ``nhts_vintage``).
    rng
        Optional numpy generator for the per-row sample. Defaults to a
        fresh ``default_rng(0)`` so callers without a seed get a
        deterministic fill.

    Returns
    -------
    pd.DataFrame
        A copy of ``veh`` with ``model`` populated on every
        ``2022_nextgen`` row (the 2017 rows are untouched).
    """
    if "nhts_vintage" not in veh.columns:
        raise KeyError("_model_backoff_draw_from_2017 requires 'nhts_vintage' column")
    if "model" not in veh.columns:
        raise KeyError("_model_backoff_draw_from_2017 requires 'model' column")
    if rng is None:
        rng = np.random.default_rng(0)

    vintage_series = veh["nhts_vintage"].astype("string")
    is_2017 = (vintage_series == NHTS_VINTAGE_2017).to_numpy()
    is_nextgen = (vintage_series == NHTS_VINTAGE_2022_NEXTGEN).to_numpy()
    needs_fill_mask = is_nextgen & veh["model"].isna().to_numpy()
    if not needs_fill_mask.any():
        return veh.copy()

    legacy_2017 = veh[is_2017]

    def _pool(keys: tuple[str, ...]) -> dict[tuple, list[str]]:
        """Build a {key_tuple -> list[model]} conditional pool on 2017 rows."""
        sub = legacy_2017.dropna(subset=["model", *keys])
        out: dict[tuple, list[str]] = {}
        if len(sub) == 0:
            return out
        for key_tuple, grp in sub.groupby(list(keys), sort=False):
            if not isinstance(key_tuple, tuple):
                key_tuple = (key_tuple,)
            out[key_tuple] = grp["model"].astype("string").tolist()
        return out

    pool_full = _pool(("make", "vehyear", "hfuel", "fueltype"))
    pool_3 = _pool(("make", "vehyear", "hfuel"))
    pool_mk_hf = _pool(("make", "hfuel"))
    pool_mk = _pool(("make",))

    out = veh.copy()
    rows = out[needs_fill_mask]

    def _norm(v):
        if v is None or (isinstance(v, float) and np.isnan(v)) or pd.isna(v):
            return None
        return v

    new_models: list[str] = []
    for _, row in rows.iterrows():
        mk_n = _norm(row.get("make"))
        vy_n = _norm(row.get("vehyear"))
        hf_n = _norm(row.get("hfuel"))
        ft_n = _norm(row.get("fueltype"))

        candidates: list[str] | None = None
        for key, pool in (
            ((mk_n, vy_n, hf_n, ft_n), pool_full),
            ((mk_n, vy_n, hf_n), pool_3),
            ((mk_n, hf_n), pool_mk_hf),
            ((mk_n,), pool_mk),
        ):
            if None in key:
                continue
            candidates = pool.get(key)
            if candidates:
                break

        if candidates:
            new_models.append(str(rng.choice(candidates)))
        else:
            new_models.append("UNKNOWN")

    out.loc[needs_fill_mask, "model"] = pd.array(new_models, dtype="string")
    return out


def apply_workplace_annmiles_filter(
    hh: pd.DataFrame,
    veh: pd.DataFrame,
    annmiles_max: float = ANNMILES_WORKPLACE_MAX,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep only HHs with at least one vehicle whose ``annmiles < cap``.

    Strict ``<`` per PM gap #3.  Negative / NA ANNMILES is treated as
    unknown and excluded from the workplace-feasible cohort.
    """
    veh_ok = veh[(veh["annmiles"].astype("float32") < annmiles_max) & (veh["annmiles"] > 0)]
    keep_houseids = veh_ok["houseid"].unique()
    hh_keep = hh[hh["houseid"].isin(keep_houseids)].reset_index(drop=True)
    veh_keep = veh[veh["houseid"].isin(hh_keep["houseid"])].reset_index(drop=True)
    return hh_keep, veh_keep


# --------------------------------------------------------------------------- #
# Borrow-from-neighbours.
# --------------------------------------------------------------------------- #

def borrow_from_neighbours(
    in_region_hh: pd.DataFrame,
    in_region_veh: pd.DataFrame,
    region: Region,
    profile_type: str,
    nhts_data_dir: Path,
    *,
    pool_threshold: int = BORROW_POOL_THRESHOLD,
    share_cap: float = BORROW_SHARE_CAP,
    annmiles_max: float = ANNMILES_WORKPLACE_MAX,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Augment the donor pool from ``region.workplace_donor_borrow_states``.

    Returns
    -------
    (combined_hh, combined_veh, n_borrowed)
        Where ``combined_hh`` carries a ``donor_origin_state`` column
        populated with the actual HHSTATE for borrowed rows and a
        ``borrow_used`` bool column.  In-region rows carry
        ``borrow_used == False`` and ``donor_origin_state`` set to the
        empty string (downstream we treat empty as "in-region native").
    """
    n_in = len(in_region_hh)
    in_region_hh = in_region_hh.copy()
    in_region_hh["donor_origin_state"] = ""
    in_region_hh["borrow_used"] = False

    if (
        profile_type != "workplace"
        or n_in >= pool_threshold
        or not region.workplace_donor_borrow_states
    ):
        return in_region_hh, in_region_veh, 0

    # Read the national NHTS to fetch the borrow-state donors.
    nat_hh, nat_veh = _read_national_csvs(nhts_data_dir)
    # Try each borrow state in listed order, accumulating until either:
    #   - total pool >= threshold (with the share-cap headroom), or
    #   - the share-cap would be exceeded.
    borrowed_hh_frames: list[pd.DataFrame] = []
    borrowed_veh_frames: list[pd.DataFrame] = []
    n_borrowed_total = 0

    for state in region.workplace_donor_borrow_states:
        if state in region.states:
            continue  # already counted in in_region_hh
        # Maximum allowed total borrow given the share cap.
        if share_cap >= 1.0:
            max_borrow_total = 10**12
        else:
            # n_borrowed / (n_in + n_borrowed) <= share_cap
            # => n_borrowed <= share_cap * n_in / (1 - share_cap)
            max_borrow_total = int(share_cap * n_in / (1.0 - share_cap))
        remaining_capacity = max_borrow_total - n_borrowed_total
        if remaining_capacity <= 0:
            break

        sub_hh = nat_hh[nat_hh["hhstate"] == state].reset_index(drop=True)
        sub_veh = nat_veh[nat_veh["houseid"].isin(sub_hh["houseid"])].reset_index(drop=True)
        if profile_type == "workplace":
            sub_hh, sub_veh = apply_workplace_annmiles_filter(
                sub_hh, sub_veh, annmiles_max=annmiles_max
            )
        if sub_hh.empty:
            continue

        # Truncate to remaining capacity (deterministic: keep lowest houseids).
        if len(sub_hh) > remaining_capacity:
            sub_hh = sub_hh.sort_values("houseid", kind="stable").head(
                remaining_capacity
            ).reset_index(drop=True)
            sub_veh = sub_veh[sub_veh["houseid"].isin(sub_hh["houseid"])].reset_index(
                drop=True
            )

        sub_hh["donor_origin_state"] = state
        sub_hh["borrow_used"] = True
        borrowed_hh_frames.append(sub_hh)
        borrowed_veh_frames.append(sub_veh)
        n_borrowed_total += len(sub_hh)

        if n_in + n_borrowed_total >= pool_threshold:
            break

    if borrowed_hh_frames:
        all_hh = pd.concat([in_region_hh, *borrowed_hh_frames], ignore_index=True)
        all_veh = pd.concat([in_region_veh, *borrowed_veh_frames], ignore_index=True)
    else:
        all_hh = in_region_hh
        all_veh = in_region_veh
    return all_hh, all_veh, n_borrowed_total


# --------------------------------------------------------------------------- #
# Core matcher.
# --------------------------------------------------------------------------- #

def _vehicle_lookup(veh: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """Return houseid -> DataFrame of that HH's vehicles (sorted by vehid)."""
    cols = ["houseid", "vehid", "hfuel"]
    v = veh[cols].sort_values(["houseid", "vehid"], kind="stable").reset_index(drop=True)
    return {int(hid): grp.reset_index(drop=True) for hid, grp in v.groupby("houseid", sort=True)}


def _pick_vehicle(
    veh_pool: pd.DataFrame,
    powertrain: str,
    rng: np.random.Generator,
) -> int:
    """Pick one vehid from a non-empty per-HH vehicle frame."""
    if veh_pool.empty:
        raise ValueError("_pick_vehicle called on empty HH vehicle pool")
    if powertrain == "BEV":
        preferred = veh_pool[veh_pool["hfuel"] == HFUEL_BEV]
    elif powertrain == "PHEV":
        preferred = veh_pool[veh_pool["hfuel"] == HFUEL_PHEV]
    else:  # pragma: no cover - exhaustive
        preferred = veh_pool.iloc[0:0]
    chosen_pool = preferred if not preferred.empty else veh_pool
    idx = int(rng.integers(0, len(chosen_pool)))
    return int(chosen_pool.iloc[idx]["vehid"])


def _draw_houseid_weighted(
    rng: np.random.Generator,
    pool: pd.DataFrame,
) -> tuple[int, int]:
    """Sample one houseid from ``pool`` weighted by WTHHFIN."""
    if pool.empty:
        raise ValueError("_draw_houseid_weighted called on empty pool")
    w = pool["wthhfin"].to_numpy(dtype=np.float64)
    if not np.isfinite(w).all() or w.sum() <= 0:
        probs = np.full(len(pool), 1.0 / len(pool))
    else:
        probs = w / w.sum()
    idx = int(rng.choice(len(pool), p=probs))
    return int(pool.iloc[idx]["houseid"]), int(len(pool))


def _filter_pool(
    pool: pd.DataFrame,
    *,
    income_tier: str | None = None,
    hhsize_band: str | None = None,
    urbrur: int | None = None,
    hhvehcnt_band: str | None = None,
) -> pd.DataFrame:
    """Apply optional equality filters; absent keys = no constraint."""
    mask = np.ones(len(pool), dtype=bool)
    if income_tier is not None:
        mask &= pool["income_tier"].to_numpy() == income_tier
    if hhsize_band is not None:
        mask &= pool["hhsize_band"].to_numpy() == hhsize_band
    if urbrur is not None:
        mask &= pool["urbrur_int"].to_numpy() == urbrur
    if hhvehcnt_band is not None:
        mask &= pool["hhvehcnt_band"].to_numpy() == hhvehcnt_band
    return pool.loc[mask]


def _candidate_pool_with_relaxation(
    pool: pd.DataFrame,
    income_tier: str,
    hhsize_band: str,
    urbrur: int,
    hhvehcnt_band: str,
) -> tuple[pd.DataFrame, int]:
    """Walk the relaxation ladder; return the first non-empty pool + level."""
    levels = [
        dict(income_tier=income_tier, hhsize_band=hhsize_band, urbrur=urbrur,
             hhvehcnt_band=hhvehcnt_band),
        dict(income_tier=income_tier, hhsize_band=hhsize_band, urbrur=urbrur,
             hhvehcnt_band=None),
        dict(income_tier=income_tier, hhsize_band=None, urbrur=urbrur,
             hhvehcnt_band=None),
        dict(income_tier=income_tier, hhsize_band=None, urbrur=None,
             hhvehcnt_band=None),
        dict(income_tier=None, hhsize_band=None, urbrur=None, hhvehcnt_band=None),
    ]
    for level, kw in enumerate(levels):
        sub = _filter_pool(pool, **kw)
        if not sub.empty:
            return sub, level
    raise RuntimeError("donor pool is empty - no households to match against")


def _validate_profile_type(profile_type: str) -> None:
    if profile_type not in PROFILE_TYPES:
        raise ValueError(
            f"profile_type {profile_type!r} not supported; "
            f"valid: {sorted(PROFILE_TYPES)}"
        )


def match_donors(
    fleet: pd.DataFrame,
    region: Region,
    profile_type: str,
    *,
    seed: int = MASTER_SEED_DEFAULT,
    nhts_data_dir: str | os.PathLike[str] | None = None,
    annmiles_max: float = ANNMILES_WORKPLACE_MAX,
    borrow_pool_threshold: int = BORROW_POOL_THRESHOLD,
    borrow_share_cap: float = BORROW_SHARE_CAP,
) -> pd.DataFrame:
    """Match each synthetic EV in ``fleet`` to a donor (houseid, vehid).

    Parameters
    ----------
    fleet : pd.DataFrame
        M2 output frame.  Must include ``ev_id``, ``powertrain``,
        ``income_tier`` columns.
    region : Region
        v2.0 region object; drives state / CBSA narrowing and borrow
        targeting.
    profile_type : str
        ``"residential"`` or ``"workplace"``.  Workplace activates the
        ANNMILES < 3600 filter and the borrow-from-neighbours rule.
    seed : int
        Master seed.  Per-EV RNG uses ``seed + 2_000_000 + ev_id``.
    nhts_data_dir : Path | None
        Directory containing ``nhts2017_ca_{hh,veh}.parquet`` and / or
        ``{hh,veh}pub.csv``.  If ``None``, defaults to
        ``data/pev/raw/nhts2017/`` under the repo root.
    annmiles_max : float
        Strict workplace ANNMILES cap (default 3600).
    borrow_pool_threshold : int
        If the in-region post-filter pool falls below this, borrow.
    borrow_share_cap : float
        Maximum borrow share of the combined pool (W10; default 0.25).

    Returns
    -------
    pd.DataFrame
        ``fleet`` augmented with ``nhts_donor_houseid``,
        ``nhts_donor_vehid``, ``match_relaxation_level``,
        ``match_n_candidates``, ``donor_origin_state``, ``borrow_used``.
    """
    required_fleet_cols = {"ev_id", "powertrain", "income_tier"}
    missing = required_fleet_cols - set(fleet.columns)
    if missing:
        raise KeyError(f"fleet missing required columns: {sorted(missing)}")
    _validate_profile_type(profile_type)

    nhts_dir = Path(nhts_data_dir) if nhts_data_dir is not None else _resolve_default_nhts_dir()

    # 1. Load region-scoped NHTS.
    hh_raw, veh_raw = load_region_nhts(region, nhts_dir)

    # 2. Apply CBSA narrowing (if any).
    hh_narrow = narrow_to_region(hh_raw, region)
    veh_narrow = veh_raw[veh_raw["houseid"].isin(hh_narrow["houseid"])].reset_index(drop=True)
    n_pre = len(hh_narrow)

    # 3. Profile-specific filters.
    if profile_type == "workplace":
        hh_post, veh_post = apply_workplace_annmiles_filter(
            hh_narrow, veh_narrow, annmiles_max=annmiles_max
        )
    else:
        hh_post, veh_post = hh_narrow, veh_narrow
    n_after_annmiles = len(hh_post)

    # 4. Borrow-from-neighbours (workplace only; no-op for residential).
    hh_all, veh_all, n_borrowed = borrow_from_neighbours(
        hh_post,
        veh_post,
        region,
        profile_type,
        nhts_dir,
        pool_threshold=borrow_pool_threshold,
        share_cap=borrow_share_cap,
        annmiles_max=annmiles_max,
    )
    n_total = len(hh_all)
    if n_total < POOL_WARNING_FLOOR:
        logger.warning(
            "M3 donor pool for region=%s profile=%s is %d (< %d) after borrow; "
            "results may be poor (v2.1 concern)",
            region.name, profile_type, n_total, POOL_WARNING_FLOOR,
        )

    # 5. Keep only HHs that have at least one vehicle remaining.
    veh_houseids = set(veh_all["houseid"].to_numpy().tolist())
    hh_all = hh_all[hh_all["houseid"].isin(veh_houseids)].reset_index(drop=True)
    if hh_all.empty:
        raise RuntimeError(
            f"donor pool for region={region.name!r} profile={profile_type!r} is "
            "empty after vehicle-coverage filter; cannot match"
        )

    # 6. Build match keys.
    hh_keys = build_match_keys(hh_all)
    veh_by_hh = _vehicle_lookup(veh_all[veh_all["houseid"].isin(hh_keys["houseid"].to_numpy())])

    # 7. Match each EV.
    fleet_sorted = fleet.sort_values("ev_id", kind="stable").reset_index(drop=True)
    n = len(fleet_sorted)
    out_house = np.empty(n, dtype=np.int64)
    out_veh = np.empty(n, dtype=np.int32)
    out_level = np.empty(n, dtype=np.int8)
    out_ncand = np.empty(n, dtype=np.int32)
    out_donor_state = np.empty(n, dtype=object)
    out_borrow = np.zeros(n, dtype=bool)

    donor_origin_lookup = (
        hh_keys.set_index("houseid")["donor_origin_state"].to_dict()
    )
    borrow_lookup = hh_keys.set_index("houseid")["borrow_used"].to_dict()

    relaxation_counts: dict[int, int] = {i: 0 for i in range(5)}

    # RFC-015: pre-extract per-EV columns once to numpy and iterate with
    # ``.itertuples(name=None)`` (5-10× faster than ``.iterrows()`` at
    # N=50_000). The loop body still indexes by integer position ``i``
    # for output assignment so the existing structure is preserved.
    ev_id_arr = fleet_sorted["ev_id"].to_numpy(dtype=np.int64, copy=False)
    powertrain_arr = fleet_sorted["powertrain"].astype(str).to_numpy(copy=False)
    income_tier_arr = fleet_sorted["income_tier"].astype(str).to_numpy(copy=False)
    for i in range(len(fleet_sorted)):
        ev_id_int = int(ev_id_arr[i])
        rng = np.random.default_rng(seed + 2_000_000 + ev_id_int)
        powertrain = str(powertrain_arr[i])
        income_tier = str(income_tier_arr[i])

        # Sample target (hhsize, urbrur, hhvehcnt) bands conditional on the
        # EV's income tier from the WTHHFIN-weighted pool.
        tier_pool = hh_keys[hh_keys["income_tier"] == income_tier]
        if tier_pool.empty:
            tier_pool = hh_keys
        w = tier_pool["wthhfin"].to_numpy(dtype=np.float64)
        if not np.isfinite(w).all() or w.sum() <= 0:
            probs = np.full(len(tier_pool), 1.0 / len(tier_pool))
        else:
            probs = w / w.sum()
        target_hh_idx = int(rng.choice(len(tier_pool), p=probs))
        target_row = tier_pool.iloc[target_hh_idx]
        target_hhsize_band = str(target_row["hhsize_band"])
        target_urbrur = int(target_row["urbrur_int"])
        target_hhvehcnt_band = str(target_row["hhvehcnt_band"])

        candidates, level = _candidate_pool_with_relaxation(
            hh_keys,
            income_tier=income_tier,
            hhsize_band=target_hhsize_band,
            urbrur=target_urbrur,
            hhvehcnt_band=target_hhvehcnt_band,
        )
        houseid, n_cand = _draw_houseid_weighted(rng, candidates)

        veh_pool = veh_by_hh.get(houseid)
        if veh_pool is None or veh_pool.empty:
            raise RuntimeError(
                f"donor houseid {houseid} has no vehicles after pre-filter"
            )
        vehid = _pick_vehicle(veh_pool, powertrain, rng)

        out_house[i] = houseid
        out_veh[i] = vehid
        out_level[i] = level
        out_ncand[i] = n_cand
        origin = donor_origin_lookup.get(int(houseid), "")
        out_donor_state[i] = origin if origin else None
        out_borrow[i] = bool(borrow_lookup.get(int(houseid), False))
        relaxation_counts[level] += 1

    enriched = fleet_sorted.copy()
    enriched["nhts_donor_houseid"] = out_house
    enriched["nhts_donor_vehid"] = out_veh
    enriched["match_relaxation_level"] = out_level
    enriched["match_n_candidates"] = out_ncand
    enriched["donor_origin_state"] = pd.array(out_donor_state, dtype="string")
    enriched["borrow_used"] = out_borrow

    # Attach result metadata as DataFrame attribute for downstream callers.
    borrow_share = float(n_borrowed) / float(n_total) if n_total > 0 else 0.0
    enriched.attrs["donor_match"] = {
        "region_name": region.name,
        "profile_type": profile_type,
        "n_hh_in_region": int(n_pre),
        "n_hh_after_annmiles_filter": int(n_after_annmiles),
        "n_hh_borrowed": int(n_borrowed),
        "n_hh_total_pool": int(n_total),
        "borrow_share": float(borrow_share),
        "relaxation_counts": {int(k): int(v) for k, v in relaxation_counts.items()},
    }
    return enriched


def _resolve_default_nhts_dir() -> Path:
    """Return the absolute default NHTS dir under the repo root.

    Anchored on :func:`pev_synth._paths.repo_root` (RFC-016) but the
    historical default subpath (``DEFAULT_NHTS_DATA_DIR``) is preserved so
    existing M3 callers reading the raw CA parquets still find them. M3
    and M4 callers from ``cache_regen`` pass the explicit
    :func:`nhts_intermediate_root` via RFC-017 once RFC-014 emits per-region
    parquets.
    """
    from ._paths import repo_root

    return repo_root() / DEFAULT_NHTS_DATA_DIR


# --------------------------------------------------------------------------- #
# Persistence + CLI.
# --------------------------------------------------------------------------- #

def _write_fleet_inplace(df: pd.DataFrame, out_path: Path) -> None:
    """Write the enriched fleet to ``out_path`` with deterministic dtypes."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cast = df.copy()
    cast["nhts_donor_houseid"] = cast["nhts_donor_houseid"].astype("int64")
    cast["nhts_donor_vehid"] = cast["nhts_donor_vehid"].astype("int32")
    cast["match_relaxation_level"] = cast["match_relaxation_level"].astype("int8")
    cast["match_n_candidates"] = cast["match_n_candidates"].astype("int32")
    cast["borrow_used"] = cast["borrow_used"].astype("bool")
    # donor_origin_state is a pandas StringDtype; preserve as-is (parquet OK).
    cast = cast.sort_values("ev_id", kind="stable").reset_index(drop=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    cast.to_parquet(tmp, engine="pyarrow", index=False, compression="snappy")
    os.replace(tmp, out_path)


def _update_meta(
    meta_path: Path,
    enriched: pd.DataFrame,
) -> None:
    """Read meta.json, append donor_matcher provenance, write back.

    RFC-021.r: delegates the atomic write to
    :meth:`pev_synth._meta.MetaWriter.update` so the donor_matcher RMW
    site shares the single-writer code path with M6 / M7.
    """
    from ._meta import MetaWriter

    match_attrs = dict(enriched.attrs.get("donor_match", {}))
    payload = {
        "module": "pev_synth.donor_matcher",
        "methodology_version": METHODOLOGY_VERSION,
        **match_attrs,
    }
    top_level: dict[str, object] = {}
    # Borrow rate from the per-EV column (W10 metric).
    if "borrow_used" in enriched.columns:
        top_level["donor_borrow_rate"] = float(enriched["borrow_used"].mean())

    MetaWriter.update(
        meta_path,
        stage="donor_matcher",
        payload=payload,
        methodology_version=METHODOLOGY_VERSION,
        top_level=top_level,
    )


def _default_repo_root() -> Path:
    from ._paths import repo_root

    return repo_root()


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m pev_synth.donor_matcher",
        description=(
            "M3 v2.0: match each synthetic EV in fleet.parquet to an NHTS "
            "donor (houseid, vehid) for the given (region, profile_type)."
        ),
    )
    p.add_argument("--seed", type=int, default=MASTER_SEED_DEFAULT)
    p.add_argument("--region", type=str, default="bay_area")
    p.add_argument(
        "--profile-type", type=str, default="residential",
        choices=list(PROFILE_TYPES),
    )
    p.add_argument(
        "--in-fleet", type=str, default=None,
        help="Input fleet.parquet (defaults to v2.0 path for region/profile).",
    )
    p.add_argument(
        "--out-fleet", type=str, default=None,
        help="Output fleet.parquet (defaults to in-place overwrite of --in-fleet).",
    )
    p.add_argument(
        "--meta", type=str, default=None,
        help="meta.json output path (defaults next to --out-fleet).",
    )
    p.add_argument(
        "--nhts-data-dir", type=str, default=None,
        help="Override NHTS data directory.",
    )
    p.add_argument(
        "--repo-root", type=str, default=None,
        help="Repo root (defaults to two parents up from this file).",
    )
    p.add_argument(
        "--log-level", default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return p


def main(argv: list[str] | None = None) -> int:
    """CLI entry-point."""
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    repo_root = (
        Path(args.repo_root).resolve() if args.repo_root else _default_repo_root()
    )
    region = Region.from_name(args.region)

    cache_dir = (
        repo_root / "data" / "pev" / "processed" / region.name
        / f"{args.profile_type}_ev_synth"
    )
    in_fleet = (
        Path(args.in_fleet)
        if args.in_fleet
        else cache_dir / "fleet.parquet"
    )
    out_fleet = Path(args.out_fleet) if args.out_fleet else in_fleet
    meta_path = Path(args.meta) if args.meta else cache_dir / "meta.json"
    nhts_dir = (
        Path(args.nhts_data_dir).resolve()
        if args.nhts_data_dir
        else repo_root / DEFAULT_NHTS_DATA_DIR
    )

    logger.info("M3 v2.0: reading fleet from %s", in_fleet)
    fleet = pd.read_parquet(in_fleet)
    enriched = match_donors(
        fleet=fleet,
        region=region,
        profile_type=args.profile_type,
        seed=args.seed,
        nhts_data_dir=nhts_dir,
    )
    _write_fleet_inplace(enriched, out_fleet)
    _update_meta(meta_path, enriched)
    attrs = enriched.attrs.get("donor_match", {})
    print(
        "M3 donor_matcher v2.0: "
        f"region={region.name} profile={args.profile_type} "
        f"hh_in_region={attrs.get('n_hh_in_region')} "
        f"hh_post_annmiles={attrs.get('n_hh_after_annmiles_filter')} "
        f"hh_borrowed={attrs.get('n_hh_borrowed')} "
        f"hh_total={attrs.get('n_hh_total_pool')} "
        f"borrow_share={attrs.get('borrow_share', 0.0):.3f} "
        f"relax={attrs.get('relaxation_counts')}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


# Silence unused-import for hashlib in stripped-docstring environments.
_ = hashlib


# --------------------------------------------------------------------------- #
# v3.0 — ACS PUMS demographic raking helpers (Dev D6).
# --------------------------------------------------------------------------- #
#
# These helpers append a single-pass ratio raking pass to the M3 donor-pool
# weights so the realised NHTS joint distribution over
# ``(income_tier, hhsize_band, hhvehcnt_band)`` matches the ACS PUMS expected
# PMF for the region (per Scientist S4, docs/v3/modeling/nhts_enrichment.md
# §9.7–9.8). Calibration is opt-in per region
# (``Region.acs_calibration_enabled``) and writes the raked weights back into
# ``hh.wthhfin`` while preserving a per-row ``acs_cal_factor`` for audit.
#
# Determinism: pure pandas + numpy, no RNG consumed.
# --------------------------------------------------------------------------- #

#: PMF columns expected by :func:`apply_acs_calibration_weights`.
_ACS_PMF_REQUIRED_COLS: Final[tuple[str, ...]] = (
    "income_tier", "hhsize_band", "hhvehcnt_band", "p_expected",
)

#: Default strata for ACS raking. Matches S4 §9.7.
_ACS_RAKE_STRATA: Final[tuple[str, ...]] = (
    "income_tier", "hhsize_band", "hhvehcnt_band",
)

#: ACS strata bin labels (independent copy of acs_loader's tables — both
#: modules must agree). hhsize_band has FIVE bins for ACS (1/2/3/4/5+);
#: hhvehcnt_band has FOUR (0/1/2/3+).
_ACS_HHSIZE_BAND_LABELS: Final[tuple[str, ...]] = ("1", "2", "3", "4", "5+")
_ACS_HHVEHCNT_BAND_LABELS: Final[tuple[str, ...]] = ("0", "1", "2", "3+")

#: Fraction of cells permitted to clip before
#: :func:`apply_acs_calibration_weights` raises.
_ACS_CAP_FAIL_THRESHOLD: Final[float] = 0.05


def _assign_strata(
    hh: pd.DataFrame,
    *,
    hhsize_col: str = "hhsize",
    hhvehcnt_col: str = "hhvehcnt",
    overwrite: bool = False,
) -> pd.DataFrame:
    """Derive ``hhsize_band`` + ``hhvehcnt_band`` on an NHTS HH frame.

    Adds the ACS-compatible 5-bin ``hhsize_band`` (``{"1","2","3","4","5+"}``)
    and 4-bin ``hhvehcnt_band`` (``{"0","1","2","3+"}``) columns so the frame
    is directly joinable with a PMF from
    :func:`pev_synth.acs_loader.compute_household_mix_pmf`.

    The ``income_tier`` column is assumed present on the NHTS frame (per the
    M3 contract; produced upstream by :func:`build_match_keys`). When
    ``overwrite=False`` (default) the helper leaves any existing
    ``hhsize_band`` / ``hhvehcnt_band`` columns untouched.

    Parameters
    ----------
    hh:
        NHTS household frame with at least ``hhsize_col`` and
        ``hhvehcnt_col`` integer columns.
    hhsize_col, hhvehcnt_col:
        Source column names.
    overwrite:
        When True, recompute even if the destination columns already exist.

    Returns
    -------
    pd.DataFrame
        Copy of ``hh`` with the two band columns appended / refreshed.
    """
    if hhsize_col not in hh.columns:
        raise KeyError(f"_assign_strata: hh missing required column {hhsize_col!r}")
    if hhvehcnt_col not in hh.columns:
        raise KeyError(f"_assign_strata: hh missing required column {hhvehcnt_col!r}")

    out = hh.copy()

    def _bin_size(series: pd.Series) -> pd.Series:
        arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64")
        bands = np.empty(arr.shape, dtype=object)
        bands[:] = _ACS_HHSIZE_BAND_LABELS[0]
        bands[arr >= 1] = _ACS_HHSIZE_BAND_LABELS[0]
        bands[arr >= 2] = _ACS_HHSIZE_BAND_LABELS[1]
        bands[arr >= 3] = _ACS_HHSIZE_BAND_LABELS[2]
        bands[arr >= 4] = _ACS_HHSIZE_BAND_LABELS[3]
        bands[arr >= 5] = _ACS_HHSIZE_BAND_LABELS[4]
        s = pd.Series(bands, index=series.index, dtype="string")
        s[~np.isfinite(arr)] = pd.NA
        return s

    def _bin_veh(series: pd.Series) -> pd.Series:
        arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64")
        bands = np.empty(arr.shape, dtype=object)
        bands[:] = _ACS_HHVEHCNT_BAND_LABELS[0]
        bands[arr >= 0] = _ACS_HHVEHCNT_BAND_LABELS[0]
        bands[arr >= 1] = _ACS_HHVEHCNT_BAND_LABELS[1]
        bands[arr >= 2] = _ACS_HHVEHCNT_BAND_LABELS[2]
        bands[arr >= 3] = _ACS_HHVEHCNT_BAND_LABELS[3]
        s = pd.Series(bands, index=series.index, dtype="string")
        s[~np.isfinite(arr)] = pd.NA
        return s

    if overwrite or "hhsize_band" not in out.columns:
        out["hhsize_band"] = _bin_size(out[hhsize_col])
    if overwrite or "hhvehcnt_band" not in out.columns:
        out["hhvehcnt_band"] = _bin_veh(out[hhvehcnt_col])
    return out


def apply_acs_calibration_weights(
    hh: pd.DataFrame,
    expected_pmf: pd.DataFrame,
    *,
    strata_cols: tuple[str, ...] = _ACS_RAKE_STRATA,
    cap_min: float = 0.2,
    cap_max: float = 5.0,
    weight_col: str = "wthhfin",
) -> pd.DataFrame:
    """Single-pass ratio raking of NHTS HH weights to an ACS expected PMF.

    Scales ``hh[weight_col]`` cell-by-cell so the realised PMF over
    ``strata_cols`` matches ``expected_pmf.p_expected``. Per-row scale factor
    is also exposed as a new ``acs_cal_factor`` column for downstream
    auditability.

    Algorithm (per S4 §9.8):

    1. Realised share for cell ``c`` =
       ``sum(hh[weight_col] where strata==c) / sum(hh[weight_col])``.
    2. Expected share = ``expected_pmf.p_expected[c]``.
    3. Cell scale factor = ``expected / realised`` (0 if realised==0).
    4. Clip to ``[cap_min, cap_max]``.
    5. Multiply ``hh[weight_col]`` by the cell factor; record per-row factor.
    6. If more than 5% of *non-trivial* cells require clipping, raise
       :class:`ValueError` — the NHTS and ACS strata distributions are
       pathologically far apart and raking would dominate the donor signal.

    Parameters
    ----------
    hh:
        NHTS household frame. Must include ``weight_col`` (default
        ``wthhfin``) plus every column listed in ``strata_cols``. Bands can
        be assigned upstream via :func:`_assign_strata`.
    expected_pmf:
        Output of
        :func:`pev_synth.acs_loader.compute_household_mix_pmf`. Must have
        columns ``strata_cols + ('p_expected',)`` summing to 1.0.
    strata_cols:
        Joint match keys; defaults to
        ``("income_tier", "hhsize_band", "hhvehcnt_band")``.
    cap_min, cap_max:
        Scale-factor clip band. Defaults ``(0.2, 5.0)`` per S4.
    weight_col:
        Column name carrying the input weight (default ``"wthhfin"``).

    Returns
    -------
    pd.DataFrame
        A **new** frame (input is not mutated) with ``weight_col`` raked
        in place and an ``acs_cal_factor`` column added. The cap report
        is exposed via ``out.attrs["acs_calibration"]`` with keys
        ``("n_cells_total", "n_cells_capped", "cap_fraction",
        "cap_min", "cap_max", "n_strata_cols", "tolerance_threshold")``.

    Raises
    ------
    KeyError
        If ``hh`` or ``expected_pmf`` are missing required columns.
    ValueError
        If more than 5% of non-trivial cells require clipping, or if any
        input frame is empty.
    """
    if hh is None or hh.empty:
        raise ValueError("apply_acs_calibration_weights: hh is empty")
    if expected_pmf is None or expected_pmf.empty:
        raise ValueError("apply_acs_calibration_weights: expected_pmf is empty")

    keys = list(strata_cols)
    missing_hh = (set(keys) | {weight_col}) - set(hh.columns)
    if missing_hh:
        raise KeyError(
            f"apply_acs_calibration_weights: hh missing required columns "
            f"{sorted(missing_hh)}"
        )
    missing_pmf = (set(keys) | {"p_expected"}) - set(expected_pmf.columns)
    if missing_pmf:
        raise KeyError(
            f"apply_acs_calibration_weights: expected_pmf missing required "
            f"columns {sorted(missing_pmf)}"
        )
    if not (cap_min > 0.0 and cap_max > cap_min):
        raise ValueError(
            f"cap_min/cap_max invalid: {cap_min!r} / {cap_max!r} "
            "(require 0 < cap_min < cap_max)"
        )

    # --- Realised cell PMF on the NHTS frame ----------------------------- #
    work = hh.copy()
    # Normalise band columns to plain ``string`` dtype so the merge keys
    # match the PMF dtype regardless of any upstream categorical encoding.
    for col in keys:
        work[col] = work[col].astype("string")
    realised = (
        work.groupby(keys, observed=True)[weight_col].sum().rename("w_realised")
    )
    total = float(realised.sum())
    if total <= 0:
        raise ValueError(
            f"apply_acs_calibration_weights: total {weight_col} is non-positive"
        )
    realised_pmf = (realised / total).rename("p_realised").reset_index()

    # --- Cell-level scale factor ----------------------------------------- #
    pmf = expected_pmf[keys + ["p_expected"]].copy()
    for col in keys:
        pmf[col] = pmf[col].astype("string")
    merged = pmf.merge(realised_pmf, on=keys, how="outer").fillna({"p_expected": 0.0,
                                                                    "p_realised": 0.0})
    merged["scale_raw"] = np.where(
        merged["p_realised"] > 0,
        merged["p_expected"] / merged["p_realised"],
        # Cells present in ACS but absent in NHTS get scale 0 (no weight to
        # rake), and likewise NHTS-only cells fall through to the default 1.0
        # below. Both cases are recorded but neither counts toward the cap
        # threshold since they cannot be fixed by raking.
        0.0,
    )
    # Cells with no NHTS support are not capped (no row to scale).
    has_support = merged["p_realised"] > 0
    merged["scale"] = merged["scale_raw"].copy()
    merged.loc[has_support, "scale"] = merged.loc[has_support, "scale_raw"].clip(
        lower=cap_min, upper=cap_max,
    )
    merged["capped"] = has_support & (merged["scale"] != merged["scale_raw"])

    n_cells_total = int(has_support.sum())
    n_cells_capped = int(merged["capped"].sum())
    cap_fraction = (
        float(n_cells_capped) / float(n_cells_total) if n_cells_total > 0 else 0.0
    )
    if cap_fraction > _ACS_CAP_FAIL_THRESHOLD:
        sample = (
            merged.loc[merged["capped"], keys + ["p_realised", "p_expected",
                                                  "scale_raw", "scale"]]
            .head(10).to_dict(orient="records")
        )
        raise ValueError(
            "apply_acs_calibration_weights: "
            f"{n_cells_capped}/{n_cells_total} cells ({cap_fraction:.1%}) "
            f"required scale-cap clipping at [{cap_min}, {cap_max}]; "
            f"exceeds the {_ACS_CAP_FAIL_THRESHOLD:.0%} tolerance. "
            f"Sample capped cells: {sample}"
        )
    if n_cells_capped > 0:
        logger.warning(
            "M3 ACS raking: %d / %d cells (%.1f%%) capped at [%.2f, %.2f]",
            n_cells_capped, n_cells_total, cap_fraction * 100.0, cap_min, cap_max,
        )

    # --- Apply factor back to per-row weights ---------------------------- #
    factor_lookup = merged[keys + ["scale"]].copy()
    out = work.merge(factor_lookup, on=keys, how="left")
    out["scale"] = out["scale"].fillna(1.0)
    out[weight_col] = out[weight_col].astype("float64") * out["scale"].astype("float64")
    out = out.rename(columns={"scale": "acs_cal_factor"})

    out.attrs["acs_calibration"] = {
        "n_cells_total": int(n_cells_total),
        "n_cells_capped": int(n_cells_capped),
        "cap_fraction": float(cap_fraction),
        "cap_min": float(cap_min),
        "cap_max": float(cap_max),
        "n_strata_cols": int(len(keys)),
        "tolerance_threshold": float(_ACS_CAP_FAIL_THRESHOLD),
    }
    return out


# Silence unused-import for Mapping in stripped-docstring environments.
_ = Mapping
