"""M4 — `pev_synth.travel_week_builder` (v2.0).

Stitch NHTS-2017 person-day trip records into a 7-day vehicle-week per
synthetic EV using day-of-week donor sampling with within-household
preference, then replicate to a 365-day vehicle-year on the 2001
synthetic calendar (Monday-start, non-leap). Emit `mobility.parquet`.

v2.0 changes relative to v1.1 (Phase 10.3 plan §5.4):

- Accepts a `Region` object (from `pev_synth.regions`) and a
  `profile_type ∈ {"residential", "workplace"}`. NHTS donor pool is
  filtered to `region.states` (with CBSA narrowing when
  `region.cbsas` is non-empty), not hardcoded to California.
- For `profile_type == "workplace"`, the trip frame is restricted to
  donor-person-days that contain at least one trip with
  `WHYTRP1S == 10` (the NHTS work-purpose summary class per the v2.0
  reviser's codebook verification — NOT `WHYTO == 3` which only
  captures the "Work" sub-code and misses `WHYTO == 4`
  "Work-related meeting / trip"). When the upstream parquet lacks a
  `whytrp1s` column we derive a proxy from `WHYTO ∈ (3, 4)` and log
  the divergence.
- Timestamps in `mobility.parquet` are stored as
  `datetime64[ns, UTC]`. The synthetic year remains 2001, anchored
  in `region.tz`; UTC conversion happens at the very end of the
  builder. DST handling uses `nonexistent="shift_forward"` and
  `ambiguous=False` (the earliest occurrence on fall-back).
- Adds a per-trip `f_T_applied: float32` column. For Dec / Jan / Feb
  / Mar trips and a region whose `winter_f_T_multiplier > 1.0`, the
  multiplier is applied to `kwh_consumed`. If the EV's archetype
  carries `has_heat_pump == True` (Atkinson 2025; heat-pump-corrected
  uplift), the applied multiplier is capped at 1.10. All non-cold
  months and all non-cold regions get `f_T_applied = 1.0`.
- New output convention:
  `data/pev/processed/<region.name>/<profile_type>_ev_synth/mobility.parquet`
  (with a `bay_area` / `residential` shim for v1.1 compatibility).

Lineage (honest):
- Day-of-week donor sampling with within-household preference (this
  work; revised plan §4.2). NOT Yip 2023 (deferred to v2.1).
- Winter `f_T` per-region multiplier: Yuksel & Michalek 2015
  polynomial evaluated at the Dec-Mar mean ambient at the region's
  NOAA ISD stations (revised plan §2.5). Per-trip Dec-Mar uses the
  single calibrated multiplier.
- Heat-pump correction floor (1.10×) per Atkinson et al. 2025
  *PMC12496165*.

Methodology version: v2.0.0.  Master seed: 20260520.  Module offset: +4.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from ._seeds import (
    HOUSEID_WEEK_DRAW_SUBSEED_OFFSET as _HOUSEID_WEEK_DRAW_SUBSEED_OFFSET,
)
from ._seeds import (
    MASTER_SEED as _MASTER_SEED,
)
from ._seeds import (
    METHODOLOGY_VERSION as _METHODOLOGY_VERSION,
)
from ._seeds import (
    NHTS_VINTAGE_SELECT_SUBSEED_OFFSET as _NHTS_VINTAGE_SELECT_SUBSEED_OFFSET,
)
from ._seeds import (
    module_offset as _module_offset,
)
from .regions import REGIONS, Region

__all__ = [
    "MASTER_SEED_DEFAULT",
    "METHODOLOGY_VERSION",
    "NHTS_VINTAGE_2017",
    "NHTS_VINTAGE_2022_NEXTGEN",
    "TravelWeekConfig",
    "TravelWeekResult",
    "build_strata_keys",
    "build_donor_pools",
    "build_mobility",
    "load_nhts_for_region",
    "main",
    "sample_week_for_ev",
    "sample_week_for_ev_v3",
    "sample_week_within_houseid",
]

# --------------------------------------------------------------------------- #
# Orchestrator-resolved constants (v2.0).
# --------------------------------------------------------------------------- #

METHODOLOGY_VERSION: Final[str] = _METHODOLOGY_VERSION
MASTER_SEED_DEFAULT: Final[int] = _MASTER_SEED
MODULE_OFFSET: Final[int] = _module_offset("travel_week_builder")  # +4

# Per-EV RNG offset (legacy v1.1 was `master_seed + 4_000_000 + ev_id`).
# Kept identical so byte-level reproducibility on the bay_area /
# residential cache is preserved when the same master_seed is used.
_PER_EV_RNG_OFFSET: Final[int] = 4_000_000

#: v3.0 HOUSEID-coherent intra-household stitcher sub-seed offset (sourced
#: from :mod:`pev_synth._seeds` for namespace gap guarantees).
_HOUSEID_WEEK_DRAW_OFFSET: Final[int] = _HOUSEID_WEEK_DRAW_SUBSEED_OFFSET

#: v3.0 NHTS vintage tags. Mirrored from
#: :mod:`pev_synth.donor_matcher` to keep this module import-free of
#: ``donor_matcher`` (the wrapper below is called from a sibling layer).
NHTS_VINTAGE_2017: Final[str] = "2017"
NHTS_VINTAGE_2022_NEXTGEN: Final[str] = "2022_nextgen"

#: v3.0 vintage selection sub-seed. Used by
#: :func:`sample_week_for_ev_v3` when the per-EV vintage Bernoulli draw
#: is non-trivial (``nhts_vintage_mix`` has > 1 vintage). When the mix is
#: single-vintage, the offset is reserved but unused.
_NHTS_VINTAGE_SELECT_OFFSET: Final[int] = _NHTS_VINTAGE_SELECT_SUBSEED_OFFSET

#: ``houseid_stitch_mode`` literal values (RFC §13.2 v3.0 resolution).
#:
#: ``"v3_within_hh"`` (default): 7 person-days for an EV's synthetic year
#: are drawn from a single donor HOUSEID's NHTS person-days, partitioned
#: by weekday / weekend; cross-side substitution if the HH has only one
#: side. Preserves intra-week behavioural coherence — see plan §13.2.
#:
#: ``"v2_per_travday"``: legacy v2.x i.i.d. donor sampling per travday
#: with within-HH preference (kept for ``legacy_v2_1_compat=True``
#: callers and byte-level fixture stability).
HOUSEID_STITCH_MODE_V3: Final[str] = "v3_within_hh"
HOUSEID_STITCH_MODE_V2: Final[str] = "v2_per_travday"
#: v3.0 default is the v2-per-travday sampler because BOTH NHTS 2017 and
#: NHTS NextGen 2022 public files record exactly one designated travday per
#: HOUSEID (verified live: D5 measured 0 HHs with >=2 travdays in the 2022
#: download). v3-within-HH therefore degenerates to single-day replay and
#: INFLATES per-EV annual-mileage variance rather than reducing it (D3
#: measurement on a 200-EV sample: sigma rose from 13,179 to 24,643 mi).
#: The v3 code path remains available for explicit opt-in (and for future
#: multi-day-diary vintages); the default ships v2 for honest behaviour.
HOUSEID_STITCH_MODE_DEFAULT: Final[str] = HOUSEID_STITCH_MODE_V2

#: NHTS TRAVDAY codes considered weekend (1=Sun, 7=Sat).
WEEKEND_TRAVDAYS: Final[frozenset[int]] = frozenset({1, 7})
#: NHTS TRAVDAY codes considered weekday (Mon..Fri).
WEEKDAY_TRAVDAYS: Final[frozenset[int]] = frozenset({2, 3, 4, 5, 6})

# Synthetic calendar (plan §7.1; orchestrator pin: 2001, Mon-start, non-leap).
SYNTH_CALENDAR_YEAR: Final[int] = 2001
SYNTH_CALENDAR_DAYS: Final[int] = 365

# 7-day-week index. 2001-01-01 is a Monday, so index 0 = Monday.
# NHTS TRAVDAY encoding: 1=Sun, 2=Mon, 3=Tue, 4=Wed, 5=Thu, 6=Fri, 7=Sat.
# Mapping TRAVDAY → week_index (0=Mon..6=Sun): (d - 2) mod 7.
TRAVDAY_TO_WEEK_INDEX: Final[dict[int, int]] = {d: (d - 2) % 7 for d in range(1, 8)}
WEEK_INDEX_TO_TRAVDAY: Final[dict[int, int]] = {v: k for k, v in TRAVDAY_TO_WEEK_INDEX.items()}

# Within-household preference probability (plan §4.2, RI-1).
WITHIN_HH_PROB_DEFAULT: Final[float] = 0.40

# WHYFROM / WHYTO → origin_type / dest_type mapping (NHTS Codebook v1.2).
# 1 = "Regular home activities (chores, sleep)" → "home"
# 3 = "Work"; 4 = "Work-related meeting / trip" → both → "work"
# All other observed positive codes → "other"
PURPOSE_HOME: Final[int] = 1
PURPOSE_WORK: Final[int] = 3
PURPOSE_WORK_MEETING: Final[int] = 4
# WHYTRP1S codes (NHTS summary trip-purpose class).
WHYTRP1S_HOME: Final[int] = 1
WHYTRP1S_WORK: Final[int] = 10
ORIGIN_DEST_CATEGORIES: Final[tuple[str, ...]] = ("home", "work", "other")

# HHFAMINC → income tier (must match M3 donor_matcher; plan §3.2 step 10).
INCOME_TIER_MAP: Final[dict[int, str]] = {
    1: "t1", 2: "t1", 3: "t1", 4: "t1",
    5: "t2", 6: "t2", 7: "t2",
    8: "t3", 9: "t3",
    10: "t4", 11: "t4",
}
INCOME_TIER_NULL_DEFAULT: Final[str] = "t2"

# HHSIZE / HHVEHCNT band labels (must match M3).
HHSIZE_BAND_LABELS: Final[tuple[str, ...]] = ("1", "2", "3", "4+")
HHVEHCNT_BAND_LABELS: Final[tuple[str, ...]] = ("1", "2", "3", "4+")

# Path defaults (relative to repo root).
DEFAULT_NHTS_RAW_DIR: Final[Path] = Path("data/pev/raw/nhts2017")
DEFAULT_NHTS_REGION_DIR: Final[Path] = Path("data/pev/intermediate/nhts")
DEFAULT_FLEET_REL: Final[Path] = Path("data/pev/processed/residential_ev_synth/fleet.parquet")
DEFAULT_MOBILITY_REL: Final[Path] = Path(
    "data/pev/processed/residential_ev_synth/mobility.parquet"
)
DEFAULT_META_REL: Final[Path] = Path("data/pev/processed/residential_ev_synth/meta.json")

# Behavioural sanity ranges.
MILES_PER_EV_PER_YEAR_LO: Final[float] = 8_000.0
MILES_PER_EV_PER_YEAR_HI: Final[float] = 14_000.0
TRIPS_PER_EV_PER_YEAR_LO: Final[int] = 800
TRIPS_PER_EV_PER_YEAR_HI: Final[int] = 2_000

# Maximum per-trip duration enforced on materialization. NHTS contains
# a small number of malformed person-day records where ENDTIME and
# STRTTIME values yield implausibly long single-trip durations (e.g.
# 22h for an 11-mile trip). Cap to a conservative 12-hour ceiling.
MAX_TRIP_DURATION_HOURS: Final[float] = 12.0

# Winter f_T constants (plan §5.4 / Atkinson 2025).
WINTER_MONTHS: Final[tuple[int, ...]] = (12, 1, 2, 3)
HEAT_PUMP_WINTER_F_T: Final[float] = 1.10
LONG_TRIP_THRESHOLD_MI: Final[float] = 100.0

#: RFC-013: per-region winter-mean ambient temperature (°C) used to
#: evaluate the Yuksel-Michalek polynomial. Sourced from NOAA ISD
#: 1991-2020 Dec-Jan-Feb monthly normals (capital-city stations).
#: Regions absent from the table default to T_REF (= no uplift).
WINTER_T_C_BY_REGION: Final[dict[str, float]] = {
    "boston": -2.0,
    "new_york_metro": -2.0,
    "chicago": -4.0,
    "dallas_fort_worth": 8.0,   # mild winter dip
}
#: Reference temperature: Yuksel-Michalek polynomial = 1.0 at T_REF.
YUKSEL_MICHALEK_T_REF_C: Final[float] = 20.0
#: Cold-side slope of the Yuksel-Michalek (2015) piecewise polynomial
#: (Wh/mi multiplier per °C below ``T_REF``).
YUKSEL_MICHALEK_A_COLD: Final[float] = 0.018
#: Hot-side slope of the Yuksel-Michalek (2015) piecewise polynomial
#: (Wh/mi multiplier per °C above ``T_REF``). Hot side lifts > 1 above
#: 20 °C — no clamp-at-1.0.
YUKSEL_MICHALEK_A_HOT: Final[float] = 0.006


def yuksel_michalek_uplift(temp_c: float) -> float:
    """Return the Yuksel-Michalek 2015 piecewise-polynomial Wh/mi multiplier.

    ``f_T(T_c) = 1.0 + 0.018 · max(0, 20 − T_c) + 0.006 · max(0, T_c − 20)``

    Both branches lift > 1.0; the hot side is no longer clamped.
    """
    if temp_c is None:
        return 1.0
    t = float(temp_c)
    cold = max(0.0, float(YUKSEL_MICHALEK_T_REF_C) - t)
    hot = max(0.0, t - float(YUKSEL_MICHALEK_T_REF_C))
    return 1.0 + float(YUKSEL_MICHALEK_A_COLD) * cold + float(YUKSEL_MICHALEK_A_HOT) * hot


def _resolve_winter_t_c(region: Region) -> float | None:
    """Return the per-region winter T_c (°C), preferring ``region.winter_t_c``.

    Falls back to ``WINTER_T_C_BY_REGION`` for back-compat with caches
    built before RFC-031.
    """
    t = getattr(region, "winter_t_c", None)
    if t is not None:
        return float(t)
    if region.name in WINTER_T_C_BY_REGION:
        return float(WINTER_T_C_BY_REGION[region.name])
    return None


logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Config / result types.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TravelWeekConfig:
    """User-tunable knobs (PM brief §7, v2.0 surface)."""

    master_seed: int = MASTER_SEED_DEFAULT
    within_hh_prob: float = WITHIN_HH_PROB_DEFAULT
    region_name: str = "bay_area"
    profile_type: str = "residential"
    nhts_data_dir: Path = DEFAULT_NHTS_RAW_DIR
    fleet_path: Path | None = None
    out_mobility_path: Path | None = None
    meta_path: Path | None = None
    #: RFC-014.r: when False (default), ``run(...)`` asserts that the
    #: per-region NHTS parquets exist before proceeding and refuses to
    #: read raw CSVs. Pass ``True`` (or ``--force-csv`` from the CLI)
    #: to allow the legacy CSV fallback.
    force_csv: bool = False
    #: v3.0 (§13.2 RESOLVED): controls whether the per-EV week sampler
    #: stitches within a single HOUSEID (``"v3_within_hh"``, the default)
    #: or falls back to the v2.x per-travday i.i.d. sampler with
    #: within-HH preference (``"v2_per_travday"``).  v2 path preserved for
    #: ``legacy_v2_1_compat=True`` byte-stable fixtures.
    houseid_stitch_mode: str = HOUSEID_STITCH_MODE_DEFAULT


@dataclass
class TravelWeekResult:
    """Outcome of one M4 run; carried into meta.json provenance."""

    mobility: pd.DataFrame
    n_trips: int
    n_ev: int
    median_trips_per_ev: float
    median_miles_per_ev: float
    p90_miles_per_ev: float
    total_miles: float
    total_kwh: float
    origin_pct: dict[str, float]
    dest_pct: dict[str, float]
    within_day_contiguity_pct: float
    f_T_winter_mean: float
    n_trips_winter_dec_mar: int
    region_name: str
    profile_type: str
    #: v3.0: which per-EV week sampler was used (``"v3_within_hh"`` or
    #: ``"v2_per_travday"``).  Written to ``meta.json`` provenance.
    houseid_stitch_mode: str = HOUSEID_STITCH_MODE_DEFAULT


# --------------------------------------------------------------------------- #
# Region resolution helpers.
# --------------------------------------------------------------------------- #


def _resolve_region(region: Region | str) -> Region:
    """Accept either a Region object or its short name."""
    if isinstance(region, Region):
        return region
    if isinstance(region, str):
        return Region.from_name(region)
    raise TypeError(
        f"region must be Region or str; got {type(region).__name__}"
    )


def default_mobility_path(region_name: str, profile_type: str) -> Path:
    """Return the v2.0 mobility.parquet path for (region, profile_type)."""
    return Path(
        f"data/pev/processed/{region_name}/{profile_type}_ev_synth/mobility.parquet"
    )


def default_meta_path(region_name: str, profile_type: str) -> Path:
    """Return the v2.0 meta.json path for (region, profile_type)."""
    return Path(
        f"data/pev/processed/{region_name}/{profile_type}_ev_synth/meta.json"
    )


def default_fleet_path(region_name: str, profile_type: str) -> Path:
    """Return the v2.0 fleet.parquet path for (region, profile_type)."""
    return Path(
        f"data/pev/processed/{region_name}/{profile_type}_ev_synth/fleet.parquet"
    )


# --------------------------------------------------------------------------- #
# NHTS loader (region-aware).
# --------------------------------------------------------------------------- #


def _normalise_trip_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lower-case columns; add `whytrp1s` proxy from WHYTO if absent.

    Idempotent.  When `whytrp1s` is absent (the v1.1 CA parquet does
    not carry it) we synthesise it from WHYTO using the documented
    code map: WHYTO ∈ (3, 4) → 10 (work); WHYTO == 1 → 1 (home); else
    97 (other / unknown summary class). This is honest about the
    derivation: the proxy is documented in `meta.json` and the
    M4-v2.0 PROVENANCE block.
    """
    out = df.rename(columns={c: c.lower() for c in df.columns}).copy()
    if "whytrp1s" not in out.columns:
        wt = pd.to_numeric(out.get("whyto"), errors="coerce")
        whytrp1s = pd.Series(97, index=out.index, dtype="Int16")
        whytrp1s = whytrp1s.where(~wt.isin([PURPOSE_WORK, PURPOSE_WORK_MEETING]), WHYTRP1S_WORK)
        whytrp1s = whytrp1s.where(~(wt == PURPOSE_HOME), WHYTRP1S_HOME)
        out["whytrp1s"] = whytrp1s.astype("Int16")
        out.attrs["whytrp1s_source"] = "derived_from_whyto"
    else:
        out["whytrp1s"] = pd.to_numeric(out["whytrp1s"], errors="coerce").astype("Int16")
        out.attrs["whytrp1s_source"] = "native"
    return out


def _normalise_hh_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lower-case column names and coerce hh_cbsa to nullable string."""
    out = df.rename(columns={c: c.lower() for c in df.columns}).copy()
    if "hh_cbsa" in out.columns:
        out["hh_cbsa"] = out["hh_cbsa"].astype("string")
    return out


#: RFC-014.r: the four canonical per-region NHTS parquet filenames that
#: M4 expects at ``data/pev/intermediate/nhts/<region>/``. ``hh`` and
#: ``trip`` are consumed by M4 directly; ``per`` and ``veh`` round out
#: the canonical M1 emitter contract and are checked for presence so
#: downstream modules can rely on the full set.
_PER_REGION_NHTS_PARQUETS: Final[tuple[str, ...]] = (
    "hhpub.parquet",
    "perpub.parquet",
    "vehpub.parquet",
    "trippub.parquet",
)


def _check_per_region_nhts_parquets(region: Region) -> None:
    """RFC-014.r: raise FileNotFoundError if any per-region NHTS parquet is missing.

    Resolves the absolute path relative to ``_paths.repo_root()`` so the
    check fires uniformly whether the caller passes an absolute or a
    repo-relative ``nhts_data_dir``.
    """
    # Access via the module so tests can monkey-patch ``_paths.nhts_intermediate_root``.
    from . import _paths as _paths_mod

    region_dir = _paths_mod.nhts_intermediate_root(region.name)
    for fname in _PER_REGION_NHTS_PARQUETS:
        p = region_dir / fname
        if not p.is_file():
            raise FileNotFoundError(
                f"per-region NHTS parquet missing: {p}; "
                "run M1 emitter or pass --force-csv"
            )


def _force_csv_fallback(
    nhts_dir: Path,
    states: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load `hhpub.csv` + `trippub.csv` from `nhts_dir` and state-filter.

    RFC-014.r: ``pd.read_csv`` is reachable ONLY through this helper.
    Public callers must pass ``force_csv=True`` to ``run(...)`` (or invoke
    ``load_nhts_for_region`` / ``build_mobility`` directly — the gate is
    enforced at the public ``run`` entry only).
    """
    hh_csv = nhts_dir / "hhpub.csv"
    trip_csv = nhts_dir / "trippub.csv"
    if not (hh_csv.is_file() and trip_csv.is_file()):
        raise FileNotFoundError(
            f"NHTS CSVs not found at {hh_csv} / {trip_csv}; "
            "set nhts_data_dir or run M1 to materialise parquets"
        )

    hh = pd.read_csv(hh_csv)
    hh = _normalise_hh_columns(hh)
    if "hhstate" not in hh.columns:
        raise KeyError("hhpub.csv missing HHSTATE column")
    hh = hh[hh["hhstate"].isin(states)].copy()
    kept_houseids = set(hh["houseid"].astype("int64").tolist())

    trip = pd.read_csv(
        trip_csv,
        usecols=lambda c: True,  # keep all; downstream may need WHYTRP1S
    )
    trip = _normalise_trip_columns(trip)
    trip = trip[trip["houseid"].isin(kept_houseids)].copy()
    return hh, trip


def _effective_states_for_load(
    region: Region, profile_type: str | None
) -> tuple[str, ...]:
    """Return the state set the HH+trip loader must cover.

    For workplace profiles in regions that declare
    ``workplace_donor_borrow_states``, M3 builds its donor pool from
    ``region.states ∪ region.workplace_donor_borrow_states`` and may
    therefore assign donor houseids that live in the borrow states.
    M4 must load the same expanded set so subsequent
    ``hh_meta.loc[donor_houseid]`` lookups resolve.  Order is preserved
    (region states first, then borrow states in their declared order,
    de-duplicated) for stable downstream filtering and logging.
    """
    if profile_type != "workplace" or not region.workplace_donor_borrow_states:
        return tuple(region.states)
    seen: set[str] = set()
    out: list[str] = []
    for s in (*region.states, *region.workplace_donor_borrow_states):
        if s not in seen:
            seen.add(s)
            out.append(s)
    return tuple(out)


def load_nhts_for_region(
    region: Region,
    nhts_data_dir: Path = DEFAULT_NHTS_RAW_DIR,
    profile_type: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load (hh, trip) NHTS-2017 frames filtered to `region`.

    Resolution order:

    1. `data/pev/intermediate/nhts/<region.name>/hh.parquet`
       + `.../trip.parquet` (M1-v2.0 native output, if present).
    2. v1.1 California parquet (`nhts2017_ca_hh.parquet`,
       `nhts2017_ca_trip.parquet`) — used only when the effective
       state set fits inside `{"CA"}`.
    3. Raw CSVs (`hhpub.csv`, `trippub.csv`), filtered to the
       effective state set on the fly.

    If `region.cbsas` is non-empty, we additionally restrict to
    households whose `hh_cbsa` is in that set.

    Parameters
    ----------
    profile_type : str, optional
        When ``"workplace"`` and the region declares
        ``workplace_donor_borrow_states``, the effective state set is
        ``region.states ∪ region.workplace_donor_borrow_states`` so
        that donor houseids assigned by M3 from a borrow state can be
        resolved here in M4.  For ``None`` / ``"residential"`` / a
        workplace region without borrow states, behaviour is unchanged
        (state set is ``region.states``).
    """
    if not isinstance(region, Region):
        raise TypeError(f"region must be Region; got {type(region).__name__}")

    nhts_data_dir = Path(nhts_data_dir)
    states = _effective_states_for_load(region, profile_type)
    cbsas = tuple(int(c) for c in region.cbsas)

    # 1) Region-specific parquet (M1-v2.0).
    region_dir = DEFAULT_NHTS_REGION_DIR / region.name
    hh_pq = region_dir / "hh.parquet"
    trip_pq = region_dir / "trip.parquet"
    if hh_pq.is_file() and trip_pq.is_file():
        logger.info("M4: loading NHTS from region parquet %s", region_dir)
        hh = _normalise_hh_columns(pd.read_parquet(hh_pq))
        trip = _normalise_trip_columns(pd.read_parquet(trip_pq))
    else:
        # 2) v1.1 California parquets (only valid when states ⊆ {CA}).
        ca_hh = nhts_data_dir / "nhts2017_ca_hh.parquet"
        ca_trip = nhts_data_dir / "nhts2017_ca_trip.parquet"
        if (
            set(states) <= {"CA"}
            and ca_hh.is_file()
            and ca_trip.is_file()
        ):
            logger.info("M4: loading NHTS from v1.1 CA parquets at %s", nhts_data_dir)
            hh = _normalise_hh_columns(pd.read_parquet(ca_hh))
            trip = _normalise_trip_columns(pd.read_parquet(ca_trip))
        else:
            # 3) Raw CSV fall-back (RFC-014.r: gated behind ``_force_csv_fallback``).
            logger.info(
                "M4: loading NHTS from raw CSV at %s; filtering to states=%s",
                nhts_data_dir,
                states,
            )
            hh, trip = _force_csv_fallback(nhts_data_dir, states)

    # State filter (idempotent — region-parquet should already be filtered).
    if "hhstate" in hh.columns and states:
        hh = hh[hh["hhstate"].isin(states)].copy()

    # CBSA narrowing.  For workplace + borrow-state regions, M3 narrows the
    # in-region pool by CBSA but augments with borrow-state HHs *outside* the
    # region CBSAs (donor_matcher.borrow_from_neighbours runs after
    # narrow_to_region).  Mirror that here: apply CBSA filter only to HHs in
    # ``region.states``; keep all borrow-state HHs.
    if cbsas and "hh_cbsa" in hh.columns:
        cbsa_str = {str(c) for c in cbsas}
        hh_cbsa_str = hh["hh_cbsa"].astype("string")
        in_cbsa_mask = hh_cbsa_str.isin(cbsa_str)
        if (
            profile_type == "workplace"
            and region.workplace_donor_borrow_states
            and "hhstate" in hh.columns
        ):
            in_region_state_mask = hh["hhstate"].isin(list(region.states))
            keep_mask = in_cbsa_mask | (~in_region_state_mask)
        else:
            keep_mask = in_cbsa_mask
        n_total = len(hh)
        hh = hh[keep_mask].copy()
        logger.info(
            "M4: CBSA narrowing %s kept %d / %d households",
            region.name,
            len(hh),
            n_total,
        )

    # Restrict trip to remaining households.
    kept = set(hh["houseid"].astype("int64").tolist())
    trip = trip[trip["houseid"].astype("int64").isin(kept)].copy()

    return hh, trip


# --------------------------------------------------------------------------- #
# Workplace donor filter (WHYTRP1S == 10).
# --------------------------------------------------------------------------- #


def filter_to_workplace_donor_days(trip: pd.DataFrame) -> pd.DataFrame:
    """Restrict trip frame to person-days that contain ≥ 1 work-purpose trip.

    A work trip is `WHYTRP1S == 10` (canonical; plan §4.1 / S-1) or
    `WHYTO ∈ (3, 4)` (cross-check). Person-day = (houseid, personid,
    travday). Any person-day with at least one such trip is retained
    in full (i.e., we keep the donor's other within-day trips so the
    chained trip-day shape is preserved).
    """
    if "whytrp1s" not in trip.columns:
        raise KeyError(
            "filter_to_workplace_donor_days requires `whytrp1s` column; "
            "call _normalise_trip_columns() first"
        )
    trip_int = pd.to_numeric(trip["whytrp1s"], errors="coerce")
    is_work = trip_int == WHYTRP1S_WORK
    if "whyto" in trip.columns:
        whyto_int = pd.to_numeric(trip["whyto"], errors="coerce")
        is_work = is_work | whyto_int.isin([PURPOSE_WORK, PURPOSE_WORK_MEETING])
    work_keys = (
        trip.loc[is_work, ["houseid", "personid", "travday"]]
        .drop_duplicates()
        .copy()
    )
    if work_keys.empty:
        return trip.iloc[0:0].copy()
    work_keys["__work_day__"] = True
    out = trip.merge(work_keys, on=["houseid", "personid", "travday"], how="inner")
    return out.drop(columns="__work_day__").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Pure helpers.
# --------------------------------------------------------------------------- #


def _band_4plus(values: pd.Series, labels: tuple[str, ...] = HHSIZE_BAND_LABELS) -> pd.Series:
    """Bin an integer Series into {"1","2","3","4+"} by clipping at ≥ 4."""
    arr = pd.to_numeric(values, errors="coerce").fillna(1).astype("int64").to_numpy()
    bands = np.where(arr >= 4, labels[3], arr.astype(str))
    bands = np.where(bands == "0", labels[0], bands)
    return pd.Series(bands, index=values.index, dtype="string")


def _map_income_tier(hhfaminc: pd.Series) -> pd.Series:
    """Map NHTS HHFAMINC 1..11 → income tier {t1, t2, t3, t4}."""
    arr = pd.to_numeric(hhfaminc, errors="coerce")
    tiers = arr.map(INCOME_TIER_MAP)  # type: ignore[arg-type]
    tiers = tiers.fillna(INCOME_TIER_NULL_DEFAULT)
    return tiers.astype("string")


def build_strata_keys(hh: pd.DataFrame) -> pd.DataFrame:
    """Augment an HH frame with the four strata columns used for matching."""
    out = hh.copy()
    out["income_tier"] = _map_income_tier(out["hhfaminc"])
    out["hhsize_band"] = _band_4plus(out["hhsize"])
    out["hhvehcnt_band"] = _band_4plus(out["hhvehcnt"])
    out["urbrur_int"] = (
        pd.to_numeric(out["urbrur"], errors="coerce").fillna(-1).astype("int64")
    )
    return out


def _map_purpose(code: int | float) -> str:
    """Map NHTS WHYFROM / WHYTO code → {home, work, other}.

    Treats both WHYTO=3 ("Work") and WHYTO=4 ("Work-related meeting /
    trip") as `work`, consistent with WHYTRP1S=10. Negative
    sentinels (-7/-8/-9) and any NA → "other".
    """
    try:
        c = int(code)
    except (TypeError, ValueError):
        return "other"
    if c == PURPOSE_HOME:
        return "home"
    if c in (PURPOSE_WORK, PURPOSE_WORK_MEETING):
        return "work"
    return "other"


def _hhmm_to_minutes(hhmm: int) -> int:
    """Convert an NHTS HHMM integer to minutes since 00:00."""
    h = hhmm // 100
    m = hhmm % 100
    return int(h) * 60 + int(m)


# --------------------------------------------------------------------------- #
# Donor pool construction.
# --------------------------------------------------------------------------- #


def build_donor_pools(
    trip: pd.DataFrame,
    hh: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    dict[int, pd.DataFrame],
    dict[tuple[int, int], pd.DataFrame],
    dict[int, pd.DataFrame],
]:
    """Build the structures needed for stitching.

    Returns
    -------
    enriched
        Per-(houseid, personid, travday) frame with strata columns merged
        from ``hh``.
    by_travday
        ``{travday → DataFrame}``.  Legacy v2.x cross-HH pool keyed by
        day-of-week only; still consumed by :func:`sample_week_for_ev`
        and by the v3 within-HH sampler's defensive fallback.
    by_houseid_travday
        ``{(houseid, travday) → DataFrame}``.  Legacy v2.x within-HH
        preference pool.
    by_houseid
        ``{houseid → DataFrame}`` (v3.0; §13.2 RESOLVED).  Used by
        :func:`sample_week_within_houseid` so all 7 person-days for an
        EV's year are drawn from the same HOUSEID's NHTS person-days,
        preserving intra-week behavioural coherence (annual_mileage σ
        shrinks toward EMFAC2021 prior).
    """
    required = {"houseid", "personid", "travday", "wttrdfin"}
    missing = required - set(trip.columns)
    if missing:
        raise KeyError(f"trip frame missing required columns: {sorted(missing)}")

    pd_rows = (
        trip.groupby(["houseid", "personid", "travday"], sort=False)
        .agg(wttrdfin=("wttrdfin", "first"), n_trips=("tdtrpnum", "size"))
        .reset_index()
    )
    pd_rows["wttrdfin"] = pd_rows["wttrdfin"].astype("float64")
    pd_rows["n_trips"] = pd_rows["n_trips"].astype("int32")
    pd_rows["travday"] = pd_rows["travday"].astype("int8")
    pd_rows["personid"] = pd_rows["personid"].astype("int32")
    pd_rows["houseid"] = pd_rows["houseid"].astype("int64")

    pd_rows = pd_rows[pd_rows["wttrdfin"] > 0].reset_index(drop=True)

    strata_cols = ["houseid", "income_tier", "hhsize_band", "hhvehcnt_band", "urbrur_int"]
    enriched = pd_rows.merge(hh[strata_cols], on="houseid", how="left")
    enriched["income_tier"] = enriched["income_tier"].fillna(INCOME_TIER_NULL_DEFAULT)
    enriched["hhsize_band"] = enriched["hhsize_band"].fillna("1")
    enriched["hhvehcnt_band"] = enriched["hhvehcnt_band"].fillna("1")
    enriched["urbrur_int"] = enriched["urbrur_int"].fillna(-1).astype("int64")

    by_travday: dict[int, pd.DataFrame] = {
        int(d): grp.reset_index(drop=True)
        for d, grp in enriched.groupby("travday", sort=True)
    }
    by_houseid_travday: dict[tuple[int, int], pd.DataFrame] = {
        (int(hid), int(d)): grp.reset_index(drop=True)
        for (hid, d), grp in enriched.groupby(["houseid", "travday"], sort=False)
    }
    by_houseid: dict[int, pd.DataFrame] = {
        int(h): grp.reset_index(drop=True)
        for h, grp in enriched.groupby("houseid", sort=False)
    }
    return enriched, by_travday, by_houseid_travday, by_houseid


# --------------------------------------------------------------------------- #
# Per-EV week sampler.
# --------------------------------------------------------------------------- #


def _sample_person_day(
    rng: np.random.Generator,
    pool: pd.DataFrame,
) -> tuple[int, int, int]:
    """Sample one (houseid, personid, travday) from `pool` weighted by wttrdfin."""
    if pool.empty:
        raise ValueError("_sample_person_day called on empty pool")
    w = pool["wttrdfin"].to_numpy(dtype=np.float64)
    s = w.sum()
    if not np.isfinite(s) or s <= 0:
        probs = np.full(len(pool), 1.0 / len(pool))
    else:
        probs = w / s
    idx = int(rng.choice(len(pool), p=probs))
    row = pool.iloc[idx]
    return int(row["houseid"]), int(row["personid"]), int(row["travday"])


def _strata_filter(
    pool: pd.DataFrame,
    *,
    income_tier: str,
    hhsize_band: str,
    hhvehcnt_band: str,
    urbrur_int: int,
) -> pd.DataFrame:
    """Filter `pool` to the four strata keys (exact equality)."""
    mask = (
        (pool["income_tier"].to_numpy() == income_tier)
        & (pool["hhsize_band"].to_numpy() == hhsize_band)
        & (pool["hhvehcnt_band"].to_numpy() == hhvehcnt_band)
        & (pool["urbrur_int"].to_numpy() == urbrur_int)
    )
    return pool.loc[mask]


def _split_pool_by_side(
    pool: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Partition a per-HOUSEID pool into (weekday_rows, weekend_rows).

    Weekday rows: TRAVDAY ∈ {2,3,4,5,6} (Mon-Fri).
    Weekend rows: TRAVDAY ∈ {1, 7} (Sun, Sat).
    """
    travday = pool["travday"].to_numpy(dtype=np.int64, copy=False)
    weekday_mask = np.isin(travday, np.array(sorted(WEEKDAY_TRAVDAYS), dtype=np.int64))
    weekend_mask = np.isin(travday, np.array(sorted(WEEKEND_TRAVDAYS), dtype=np.int64))
    return pool.loc[weekday_mask], pool.loc[weekend_mask]


def sample_week_within_houseid(
    rng: np.random.Generator,
    *,
    donor_houseid: int,
    by_houseid: Mapping[int, pd.DataFrame],
    by_travday: Mapping[int, pd.DataFrame] | None = None,
    by_houseid_travday: Mapping[tuple[int, int], pd.DataFrame] | None = None,
    income_tier: str | None = None,
    hhsize_band: str | None = None,
    hhvehcnt_band: str | None = None,
    urbrur_int: int | None = None,
) -> list[tuple[int, int, int]]:
    """Draw a 7-day donor sequence from a single HOUSEID's person-days.

    NHTS 2017 has exactly one designated TRAVDAY per HOUSEID; this
    function therefore performs *intra-HOUSEID person-day reuse* with a
    weekday/weekend partition, NOT a 7-day consecutive diary draw
    (those do not exist in NHTS 2017).  For each synthetic week-index
    ``d ∈ {1..7}`` we draw one person-day from the donor HOUSEID,
    sampling from the weekend side (TRAVDAY ∈ {1,7}) for d ∈ {1,7} and
    from the weekday side (TRAVDAY ∈ {2..6}) otherwise.  Cross-side
    substitution is allowed when the donor HH has only a single side
    (so we never escape the HH for a substitution).

    The draws within each side are wttrdfin-weighted (NHTS daywgt) and
    drawn *with replacement*, which is essential because the mean HH
    has only 1.87 distinct person-days and we need to fill 7 slots.

    Parameters
    ----------
    rng
        Numpy generator; should be seeded with the v3 sub-seed offset
        ``HOUSEID_WEEK_DRAW_SUBSEED_OFFSET`` for stream isolation.
    donor_houseid
        The HOUSEID assigned by M3 ``donor_matcher`` for this EV.
    by_houseid
        The v3 within-HH pool from :func:`build_donor_pools` (4th
        return value).
    by_travday, by_houseid_travday
        Optional legacy pools.  Used *only* for the defensive fallback
        when ``donor_houseid`` is missing from ``by_houseid`` (e.g.
        wttrdfin == 0 across every person-day) — in that case we drop
        through to :func:`sample_week_for_ev` so the EV is not lost.
    income_tier, hhsize_band, hhvehcnt_band, urbrur_int
        Strata keys for the legacy fallback; unused on the happy path.

    Returns
    -------
    list[tuple[int, int, int]]
        Seven ``(houseid, personid, travday)`` tuples in d=1..7 order.
        On the happy path, every tuple has ``houseid == donor_houseid``.
    """
    pool = by_houseid.get(int(donor_houseid))
    if pool is None or len(pool) == 0:
        # Defensive: HH dropped during pool construction (e.g. all
        # wttrdfin <= 0).  Fall back to the cross-HH legacy sampler so
        # the EV still emits a 7-day week.  This is rare in practice;
        # we tag it via the logger so QA can count occurrences.
        logger.warning(
            "M4 v3: HOUSEID %s missing from by_houseid pool; "
            "falling back to legacy cross-HH sampler (nhts_donor_fallback)",
            donor_houseid,
        )
        if (
            by_travday is None
            or by_houseid_travday is None
            or income_tier is None
            or hhsize_band is None
            or hhvehcnt_band is None
            or urbrur_int is None
        ):
            raise RuntimeError(
                f"sample_week_within_houseid: donor_houseid {donor_houseid} "
                "missing from by_houseid and legacy fallback kwargs not "
                "supplied — cannot stitch"
            )
        return sample_week_for_ev(
            rng,
            donor_houseid=donor_houseid,
            income_tier=income_tier,
            hhsize_band=hhsize_band,
            hhvehcnt_band=hhvehcnt_band,
            urbrur_int=urbrur_int,
            by_travday=by_travday,
            by_houseid_travday=by_houseid_travday,
        )

    weekday_rows, weekend_rows = _split_pool_by_side(pool)
    out: list[tuple[int, int, int]] = []
    for d in range(1, 8):
        is_weekend_slot = d in WEEKEND_TRAVDAYS  # d in {1, 7}
        if is_weekend_slot:
            chosen = weekend_rows if len(weekend_rows) > 0 else weekday_rows
        else:
            chosen = weekday_rows if len(weekday_rows) > 0 else weekend_rows
        # ``pool`` is non-empty per the early return above, so at least
        # one side is non-empty — ``chosen`` is therefore non-empty.
        out.append(_sample_person_day(rng, chosen))
    return out


def sample_week_for_ev(
    rng: np.random.Generator,
    *,
    donor_houseid: int,
    income_tier: str,
    hhsize_band: str,
    hhvehcnt_band: str,
    urbrur_int: int,
    by_travday: Mapping[int, pd.DataFrame],
    by_houseid_travday: Mapping[tuple[int, int], pd.DataFrame],
    within_hh_prob: float = WITHIN_HH_PROB_DEFAULT,
) -> list[tuple[int, int, int]]:
    """Sample one 7-tuple of (houseid, personid, travday) donor person-days."""
    out: list[tuple[int, int, int]] = []
    for d in range(1, 8):
        within_pool = by_houseid_travday.get((donor_houseid, d))
        if within_pool is not None and len(within_pool) > 0:
            if rng.random() < within_hh_prob:
                out.append(_sample_person_day(rng, within_pool))
                continue

        outside_pool = by_travday.get(d)
        if outside_pool is None or len(outside_pool) == 0:
            # Defensive fallback: in workplace mode some travday slots may
            # have zero rows if the workplace donor pool is sparse on
            # weekends. Fall through to the union of every travday in
            # `by_travday` rather than crashing.
            non_empty = [v for v in by_travday.values() if len(v) > 0]
            if not non_empty:
                raise RuntimeError(
                    "donor pool is empty across all travdays — cannot stitch"
                )
            outside_pool = pd.concat(non_empty, axis=0, ignore_index=True)

        sub = _strata_filter(
            outside_pool,
            income_tier=income_tier,
            hhsize_band=hhsize_band,
            hhvehcnt_band=hhvehcnt_band,
            urbrur_int=urbrur_int,
        )
        if len(sub) == 0:
            mask = (
                (outside_pool["income_tier"].to_numpy() == income_tier)
                & (outside_pool["hhsize_band"].to_numpy() == hhsize_band)
                & (outside_pool["urbrur_int"].to_numpy() == urbrur_int)
            )
            sub = outside_pool.loc[mask]
        if len(sub) == 0:
            mask = (
                (outside_pool["income_tier"].to_numpy() == income_tier)
                & (outside_pool["urbrur_int"].to_numpy() == urbrur_int)
            )
            sub = outside_pool.loc[mask]
        if len(sub) == 0:
            mask = outside_pool["income_tier"].to_numpy() == income_tier
            sub = outside_pool.loc[mask]
        if len(sub) == 0:
            sub = outside_pool
        out.append(_sample_person_day(rng, sub))
    return out


# --------------------------------------------------------------------------- #
# v3.0 vintage-aware per-EV sampler (D5).
# --------------------------------------------------------------------------- #


def _vintage_supports_within_hh(
    pools: Mapping[str, object],
) -> bool:
    """Return True if the vintage's pool structure can drive D3's stitcher.

    A vintage is treated as multi-day if its ``by_houseid`` index has at
    least one HOUSEID with ≥ 2 distinct ``travday`` values — that is the
    minimum requirement for :func:`sample_week_within_houseid` to draw a
    weekday/weekend split from within the same household without falling
    back to single-day replay (D3's empirical finding on NHTS 2017 is
    that no HOUSEID meets this bar).
    """
    by_houseid = pools.get("by_houseid")
    if not by_houseid:
        return False
    for _, df in by_houseid.items():  # type: ignore[union-attr]
        if "travday" not in df.columns:
            continue
        if int(df["travday"].nunique()) >= 2:
            return True
    return False


def sample_week_for_ev_v3(
    rng: np.random.Generator,
    *,
    nhts_vintage: str,
    donor_houseid: int,
    pools_by_vintage: Mapping[str, Mapping[str, object]],
    income_tier: str,
    hhsize_band: str,
    hhvehcnt_band: str,
    urbrur_int: int,
    within_hh_prob: float = WITHIN_HH_PROB_DEFAULT,
) -> list[tuple[int, int, int]]:
    """v3.0 vintage-aware donor-week sampler.

    Routes to the correct per-EV week sampler based on the donor's NHTS
    vintage:

    * ``nhts_vintage == "2017"`` (single TRAVDAY/HH): delegates to the
      legacy :func:`sample_week_for_ev` (per-travday cross-HH draw with a
      within-HH preference). NHTS 2017's structure makes the v3 within-HH
      stitcher degenerate to single-day replay (D3 finding), so we keep
      the legacy sampler for v2.1 byte-stable behaviour.
    * ``nhts_vintage == "2022_nextgen"`` (multi-day diaries when
      available): delegates to D3's :func:`sample_week_within_houseid`
      for true intra-HH coherence.
    * Other vintages: introspects ``pools_by_vintage[vintage]`` and
      routes to within-HH iff that vintage's ``by_houseid`` pool shows
      at least one HOUSEID with ≥ 2 distinct travdays
      (:func:`_vintage_supports_within_hh`). Falls back to the legacy
      sampler otherwise.

    Parameters
    ----------
    rng
        Numpy generator already seeded for **this EV**. The caller is
        responsible for using ``master_seed + HOUSEID_WEEK_DRAW_SUBSEED_OFFSET
        + ev_id`` for per-day sampling (D3's offset);
        :data:`NHTS_VINTAGE_SELECT_SUBSEED_OFFSET` is reserved for the
        per-EV vintage Bernoulli draw performed by the caller (M4
        driver) before invoking this wrapper.
    nhts_vintage
        ``"2017"`` or ``"2022_nextgen"`` (or any custom vintage label
        registered with the loader pipeline).
    donor_houseid
        HOUSEID assigned by M3 ``donor_matcher`` for this EV.
    pools_by_vintage
        Mapping ``vintage -> {"by_houseid", "by_houseid_travday",
        "by_travday"}`` returned by :func:`build_donor_pools` partitioned
        per vintage. The legacy v2 sampler reads ``by_travday`` +
        ``by_houseid_travday``; the v3 within-HH sampler reads
        ``by_houseid`` (with the legacy pools available for its defensive
        fallback).
    income_tier, hhsize_band, hhvehcnt_band, urbrur_int
        Strata keys forwarded to the underlying sampler.
    within_hh_prob
        Within-HH preference probability for the legacy sampler; ignored
        on the v3 path.

    Returns
    -------
    list[tuple[int, int, int]]
        Seven ``(houseid, personid, travday)`` tuples in d=1..7 order.

    Notes
    -----
    The default v3.0 Region carries ``nhts_vintage_mix = (("2017", 1.0),)``,
    so on a default-region build this wrapper is **invoked with
    nhts_vintage="2017" only**, which delegates to the legacy
    :func:`sample_week_for_ev` — i.e. the byte-stable v2.1 sampler path
    is preserved when no enrichment is opted in. The within-HH path
    activates only for callers who flip ``nhts_vintage_mix`` to include
    ``"2022_nextgen"`` (or another multi-day vintage) AND whose vintage
    Bernoulli routes that particular EV onto the new vintage.

    Sub-seed namespaces (registered in :mod:`pev_synth._seeds`):

    * ``HOUSEID_WEEK_DRAW_SUBSEED_OFFSET = 13_000_000`` — D3's per-day
      sampler. Used by the rng passed into this function.
    * ``NHTS_VINTAGE_SELECT_SUBSEED_OFFSET = 14_000_000`` — per-EV vintage
      Bernoulli. Used by the M4 driver *before* it calls this wrapper.
    """
    pools = pools_by_vintage.get(nhts_vintage)
    if pools is None:
        raise KeyError(
            f"sample_week_for_ev_v3: no donor pools registered for vintage "
            f"{nhts_vintage!r}; available={sorted(pools_by_vintage.keys())}"
        )

    by_houseid = pools.get("by_houseid")  # type: ignore[union-attr]
    by_travday = pools.get("by_travday")  # type: ignore[union-attr]
    by_houseid_travday = pools.get("by_houseid_travday")  # type: ignore[union-attr]

    # Static rule per docstring: "2017" -> legacy v2; "2022_nextgen" -> v3.
    # Other vintages: introspect pool structure.
    if nhts_vintage == NHTS_VINTAGE_2017:
        use_within_hh = False
    elif nhts_vintage == NHTS_VINTAGE_2022_NEXTGEN:
        use_within_hh = True
    else:
        use_within_hh = _vintage_supports_within_hh(pools)  # type: ignore[arg-type]

    if use_within_hh:
        if by_houseid is None:
            raise KeyError(
                f"sample_week_for_ev_v3: vintage {nhts_vintage!r} pools "
                "missing required 'by_houseid' entry for within-HH stitcher"
            )
        return sample_week_within_houseid(
            rng,
            donor_houseid=donor_houseid,
            by_houseid=by_houseid,  # type: ignore[arg-type]
            by_travday=by_travday,  # type: ignore[arg-type]
            by_houseid_travday=by_houseid_travday,  # type: ignore[arg-type]
            income_tier=income_tier,
            hhsize_band=hhsize_band,
            hhvehcnt_band=hhvehcnt_band,
            urbrur_int=urbrur_int,
        )

    if by_travday is None or by_houseid_travday is None:
        raise KeyError(
            f"sample_week_for_ev_v3: vintage {nhts_vintage!r} pools missing "
            "'by_travday' / 'by_houseid_travday' entries for legacy sampler"
        )
    return sample_week_for_ev(
        rng,
        donor_houseid=donor_houseid,
        income_tier=income_tier,
        hhsize_band=hhsize_band,
        hhvehcnt_band=hhvehcnt_band,
        urbrur_int=urbrur_int,
        by_travday=by_travday,  # type: ignore[arg-type]
        by_houseid_travday=by_houseid_travday,  # type: ignore[arg-type]
        within_hh_prob=within_hh_prob,
    )


# --------------------------------------------------------------------------- #
# Trip materialisation onto the synthetic 2001 calendar.
# --------------------------------------------------------------------------- #


def _materialise_trips_for_ev(
    ev_id: int,
    week_picks: list[tuple[int, int, int]],
    *,
    wh_per_mi_nominal: float,
    winter_f_T_multiplier: float,
    has_heat_pump: bool,
    trip_by_person_day: Mapping[tuple[int, int, int], pd.DataFrame],
    donor_houseid: int,
    donor_vehid: int,
) -> pd.DataFrame:
    """Materialise the 365-day trip ledger for one EV (LOCAL wall-clock).

    Timestamps emitted here are tz-naive wall-clock anchored on the
    2001 calendar; UTC conversion (with the region's local tz) is
    applied at the end of `build_mobility`.

    Per-trip energy:
        kwh_base    = miles · Wh_per_mi_nominal / 1000
        f_T_applied = effective multiplier (1.0 in non-cold months /
                      non-cold regions; HEAT_PUMP_WINTER_F_T when the
                      vehicle has a heat pump and the trip is in
                      Dec-Mar; else winter_f_T_multiplier)
        kwh_consumed = kwh_base · f_T_applied
    """
    picks_by_week_idx: list[tuple[int, int, int]] = [None] * 7  # type: ignore[list-item]
    for d_minus_1, triple in enumerate(week_picks):
        d = d_minus_1 + 1
        wk = TRAVDAY_TO_WEEK_INDEX[d]
        picks_by_week_idx[wk] = triple

    weekday_trips: list[pd.DataFrame] = [
        trip_by_person_day[picks_by_week_idx[wk]] for wk in range(7)
    ]

    year_start = datetime(SYNTH_CALENDAR_YEAR, 1, 1)

    rows_ev: list[int] = []
    rows_start: list[pd.Timestamp] = []
    rows_end: list[pd.Timestamp] = []
    rows_miles: list[float] = []
    rows_kwh: list[float] = []
    rows_f_T: list[float] = []
    rows_origin: list[str] = []
    rows_dest: list[str] = []
    rows_donor_travday: list[int] = []

    global_prev_end_dt: datetime | None = None

    for day_idx in range(SYNTH_CALENDAR_DAYS):
        wk = day_idx % 7
        if day_idx == SYNTH_CALENDAR_DAYS - 1:
            wk = 0
        triple = picks_by_week_idx[wk]
        donor_travday = triple[2]
        td_df = weekday_trips[wk]

        day_base = year_start + timedelta(days=day_idx)

        prev_start_min = -1
        rolled_days = 0
        prev_end_dt: datetime | None = (
            global_prev_end_dt
            if global_prev_end_dt is not None and global_prev_end_dt > day_base
            else None
        )
        for trip_row in td_df.itertuples(index=False):
            strt_min = _hhmm_to_minutes(int(trip_row.strttime))
            end_min = _hhmm_to_minutes(int(trip_row.endtime))

            if prev_start_min >= 0 and strt_min < prev_start_min and rolled_days == 0:
                rolled_days = 1
            prev_start_min = strt_min

            start_dt = day_base + timedelta(days=rolled_days, minutes=strt_min)
            if end_min < strt_min:
                end_dt = day_base + timedelta(days=rolled_days + 1, minutes=end_min)
            elif end_min == strt_min:
                end_dt = start_dt + timedelta(minutes=1)
            else:
                end_dt = day_base + timedelta(days=rolled_days, minutes=end_min)

            max_dur = timedelta(hours=MAX_TRIP_DURATION_HOURS)
            if end_dt - start_dt > max_dur:
                end_dt = start_dt + max_dur

            if prev_end_dt is not None and start_dt < prev_end_dt:
                duration = end_dt - start_dt
                start_dt = prev_end_dt + timedelta(minutes=1)
                end_dt = start_dt + max(duration, timedelta(minutes=1))
            prev_end_dt = end_dt

            miles = float(trip_row.trpmiles)
            if not np.isfinite(miles) or miles < 0:
                miles = 0.0

            # ---- winter f_T modulation (plan §5.4) -----------------
            month = start_dt.month
            if month in WINTER_MONTHS and winter_f_T_multiplier > 1.0:
                if has_heat_pump:
                    f_T = min(HEAT_PUMP_WINTER_F_T, float(winter_f_T_multiplier))
                else:
                    f_T = float(winter_f_T_multiplier)
            else:
                f_T = 1.0
            kwh = miles * float(wh_per_mi_nominal) * f_T / 1000.0

            origin = _map_purpose(trip_row.whyfrom)
            dest = _map_purpose(trip_row.whyto)

            rows_ev.append(ev_id)
            rows_start.append(pd.Timestamp(start_dt))
            rows_end.append(pd.Timestamp(end_dt))
            rows_miles.append(miles)
            rows_kwh.append(kwh)
            rows_f_T.append(f_T)
            rows_origin.append(origin)
            rows_dest.append(dest)
            rows_donor_travday.append(donor_travday)

        global_prev_end_dt = prev_end_dt

    # Final post-pass: enforce strict monotonicity, positive duration,
    # and year-bounded timestamps (see v1.1 logic).
    year_end_excl = datetime(SYNTH_CALENDAR_YEAR + 1, 1, 1)
    last_valid_end = year_end_excl - timedelta(minutes=1)
    next_slot_end = last_valid_end
    for k in range(len(rows_start) - 1, -1, -1):
        start_dt_k = rows_start[k].to_pydatetime()
        end_dt_k = rows_end[k].to_pydatetime()
        if start_dt_k >= year_end_excl or end_dt_k > last_valid_end:
            end_dt_k = next_slot_end
            start_dt_k = end_dt_k - timedelta(minutes=1)
            next_slot_end = next_slot_end - timedelta(minutes=2)
            rows_start[k] = pd.Timestamp(start_dt_k)
            rows_end[k] = pd.Timestamp(end_dt_k)
        else:
            break
    for k in range(len(rows_start)):
        start_dt_k = rows_start[k].to_pydatetime()
        end_dt_k = rows_end[k].to_pydatetime()
        if k > 0:
            prev_end_k = rows_end[k - 1].to_pydatetime()
            if start_dt_k <= prev_end_k:
                start_dt_k = prev_end_k + timedelta(minutes=1)
        if end_dt_k <= start_dt_k:
            end_dt_k = start_dt_k + timedelta(minutes=1)
        if end_dt_k > last_valid_end:
            end_dt_k = last_valid_end
            if start_dt_k >= end_dt_k:
                start_dt_k = end_dt_k - timedelta(minutes=1)
        rows_start[k] = pd.Timestamp(start_dt_k)
        rows_end[k] = pd.Timestamp(end_dt_k)

    n_trips = len(rows_ev)
    trip_idx = np.arange(n_trips, dtype=np.int32)
    df = pd.DataFrame(
        {
            "ev_id": np.array(rows_ev, dtype=np.int32),
            "trip_idx": trip_idx,
            "start_ts": pd.array(rows_start, dtype="datetime64[ns]"),
            "end_ts": pd.array(rows_end, dtype="datetime64[ns]"),
            "miles": np.array(rows_miles, dtype=np.float32),
            "kwh_consumed": np.array(rows_kwh, dtype=np.float32),
            "f_T_applied": np.array(rows_f_T, dtype=np.float32),
            "origin_type": rows_origin,
            "dest_type": rows_dest,
            "nhts_donor_houseid": [str(donor_houseid)] * n_trips,
            "nhts_donor_vehid": [str(donor_vehid)] * n_trips,
            "nhts_donor_travday": np.array(rows_donor_travday, dtype=np.int8),
        }
    )
    return df


# --------------------------------------------------------------------------- #
# Core driver.
# --------------------------------------------------------------------------- #


def build_mobility(
    fleet: pd.DataFrame,
    region: Region | str | None = None,
    profile_type: str = "residential",
    *,
    # Optional pre-loaded NHTS frames (legacy v1.1 callers + tests).
    trip: pd.DataFrame | None = None,
    hh: pd.DataFrame | None = None,
    # Where to look for NHTS inputs if `trip`/`hh` are not supplied.
    nhts_data_dir: Path = DEFAULT_NHTS_RAW_DIR,
    seed: int = MASTER_SEED_DEFAULT,
    within_hh_prob: float = WITHIN_HH_PROB_DEFAULT,
    # v3.0 (§13.2 RESOLVED): per-EV week sampler mode.
    houseid_stitch_mode: str = HOUSEID_STITCH_MODE_DEFAULT,
    # Back-compat alias (v1.1 named `master_seed`).
    master_seed: int | None = None,
) -> pd.DataFrame:
    """Build the v2.0 mobility ledger for one (region, profile_type) fleet.

    Parameters
    ----------
    fleet : pd.DataFrame
        M2+M3 fleet frame. Must include columns ``ev_id``,
        ``Wh_per_mi_nominal``, ``nhts_donor_houseid``,
        ``nhts_donor_vehid``, ``has_heat_pump``.
    region : Region | str
        v2.0 region. If `None` we default to "bay_area" for
        back-compat with v1.1 call sites.
    profile_type : str
        ``"residential"`` (no donor-day filter) or ``"workplace"``
        (filter donor person-days to those with at least one
        ``WHYTRP1S == 10`` trip — the NHTS work-purpose summary).
    trip, hh : pd.DataFrame, optional
        Pre-loaded NHTS frames. If supplied (legacy code path), they
        are used directly with no extra state/CBSA filter; the caller
        is responsible for any filtering. Column names may be upper
        or lower case.
    nhts_data_dir : Path
        Where to look for NHTS parquets / CSVs when `trip`/`hh` are
        not supplied. Honours the v2.0-rev intermediate layout
        first, then v1.1 CA parquets, then raw CSVs.
    seed : int
        Master seed (v2.0 default `20260520`).
    master_seed : int, optional
        Back-compat alias for `seed`. If provided, overrides `seed`.

    Returns
    -------
    pd.DataFrame
        ``mobility.parquet`` schema (sorted by ``ev_id``, ``trip_idx``).
        Timestamps are `datetime64[ns, UTC]`.
    """
    if master_seed is not None:
        seed = int(master_seed)

    if region is None:
        region_obj = REGIONS["bay_area"]
    else:
        region_obj = _resolve_region(region)

    profile_type = str(profile_type)
    if profile_type not in ("residential", "workplace"):
        raise ValueError(
            f"profile_type must be 'residential' or 'workplace'; got {profile_type!r}"
        )

    required_fleet = {
        "ev_id", "Wh_per_mi_nominal", "nhts_donor_houseid", "nhts_donor_vehid",
    }
    missing = required_fleet - set(fleet.columns)
    if missing:
        raise KeyError(f"fleet missing required columns: {sorted(missing)}")

    if not 0.0 <= within_hh_prob <= 1.0:
        raise ValueError(f"within_hh_prob must be in [0,1]; got {within_hh_prob}")
    if houseid_stitch_mode not in (HOUSEID_STITCH_MODE_V3, HOUSEID_STITCH_MODE_V2):
        raise ValueError(
            f"houseid_stitch_mode must be one of "
            f"{(HOUSEID_STITCH_MODE_V3, HOUSEID_STITCH_MODE_V2)!r}; "
            f"got {houseid_stitch_mode!r}"
        )

    # ---- 1. Resolve NHTS inputs ---------------------------------------
    if trip is None or hh is None:
        hh, trip = load_nhts_for_region(
            region_obj,
            nhts_data_dir=nhts_data_dir,
            profile_type=profile_type,
        )
    else:
        # Caller-supplied frames; normalise columns (lowercase + WHYTRP1S proxy).
        hh = _normalise_hh_columns(hh)
        trip = _normalise_trip_columns(trip)

    # ---- 2. Workplace donor filter (WHYTRP1S == 10) -------------------
    if profile_type == "workplace":
        trip = filter_to_workplace_donor_days(trip)
        if trip.empty:
            raise RuntimeError(
                f"workplace donor pool is empty for region {region_obj.name!r}; "
                "no NHTS person-days contain a WHYTRP1S == 10 trip"
            )

    # ---- 3. Pre-compute donor pools and per-(houseid, personid, travday) trips.
    hh_keys = build_strata_keys(hh)
    _, by_travday, by_hh_td, by_hh = build_donor_pools(trip, hh_keys)

    trip_sorted = trip.sort_values(
        ["houseid", "personid", "travday", "tdtrpnum"], kind="stable"
    )
    trip_by_person_day: dict[tuple[int, int, int], pd.DataFrame] = {
        (int(h), int(p), int(d)): grp.reset_index(drop=True)
        for (h, p, d), grp in trip_sorted.groupby(
            ["houseid", "personid", "travday"], sort=False
        )
    }

    hh_meta = hh_keys.set_index("houseid")[
        ["income_tier", "hhsize_band", "hhvehcnt_band", "urbrur_int"]
    ]

    fleet_sorted = fleet.sort_values("ev_id", kind="stable").reset_index(drop=True)
    per_ev_frames: list[pd.DataFrame] = []

    has_heat_pump_lookup: dict[int, bool] = (
        fleet_sorted.set_index("ev_id")["has_heat_pump"].astype(bool).to_dict()
        if "has_heat_pump" in fleet_sorted.columns
        else {}
    )

    winter_f_T = float(region_obj.winter_f_T_multiplier)

    # RFC-015: pre-extract columns to numpy and iterate with index. The
    # loop body still calls into ``_materialise_trips_for_ev`` so the
    # contract is unchanged.
    ev_id_arr = fleet_sorted["ev_id"].to_numpy(dtype=np.int64, copy=False)
    donor_house_arr = fleet_sorted["nhts_donor_houseid"].to_numpy(
        dtype=np.int64, copy=False
    )
    donor_veh_arr = fleet_sorted["nhts_donor_vehid"].to_numpy(
        dtype=np.int64, copy=False
    )
    wh_per_mi_arr = fleet_sorted["Wh_per_mi_nominal"].to_numpy(
        dtype=np.float64, copy=False
    )
    for i in range(len(fleet_sorted)):
        ev_id_int = int(ev_id_arr[i])
        donor_houseid = int(donor_house_arr[i])
        donor_vehid = int(donor_veh_arr[i])
        wh_per_mi = float(wh_per_mi_arr[i])
        hp_flag = bool(has_heat_pump_lookup.get(ev_id_int, False))

        try:
            meta = hh_meta.loc[donor_houseid]
        except KeyError as exc:
            raise KeyError(
                f"donor houseid {donor_houseid} for ev_id {ev_id_int} not in HH frame"
            ) from exc
        income_tier = str(meta["income_tier"])
        hhsize_band = str(meta["hhsize_band"])
        hhvehcnt_band = str(meta["hhvehcnt_band"])
        urbrur_int = int(meta["urbrur_int"])

        if houseid_stitch_mode == HOUSEID_STITCH_MODE_V3:
            # v3.0 (§13.2 RESOLVED): dedicated sub-seed namespace so the
            # v3 within-HH stream does not collide with any legacy M4
            # stream at the same ev_id.
            rng = np.random.default_rng(
                int(seed) + _HOUSEID_WEEK_DRAW_OFFSET + ev_id_int
            )
            week_picks = sample_week_within_houseid(
                rng,
                donor_houseid=donor_houseid,
                by_houseid=by_hh,
                by_travday=by_travday,
                by_houseid_travday=by_hh_td,
                income_tier=income_tier,
                hhsize_band=hhsize_band,
                hhvehcnt_band=hhvehcnt_band,
                urbrur_int=urbrur_int,
            )
        else:
            # v2.x legacy path — preserve the historical per-EV RNG offset
            # so ``legacy_v2_1_compat=True`` keeps producing byte-identical
            # mobility.parquet for the same master_seed.
            rng = np.random.default_rng(int(seed) + _PER_EV_RNG_OFFSET + ev_id_int)
            week_picks = sample_week_for_ev(
                rng,
                donor_houseid=donor_houseid,
                income_tier=income_tier,
                hhsize_band=hhsize_band,
                hhvehcnt_band=hhvehcnt_band,
                urbrur_int=urbrur_int,
                by_travday=by_travday,
                by_houseid_travday=by_hh_td,
                within_hh_prob=within_hh_prob,
            )

        df_ev = _materialise_trips_for_ev(
            ev_id_int,
            week_picks,
            wh_per_mi_nominal=wh_per_mi,
            winter_f_T_multiplier=winter_f_T,
            has_heat_pump=hp_flag,
            trip_by_person_day=trip_by_person_day,
            donor_houseid=donor_houseid,
            donor_vehid=donor_vehid,
        )
        per_ev_frames.append(df_ev)

    if not per_ev_frames:
        return _empty_mobility_df(region_obj)

    mobility = pd.concat(per_ev_frames, axis=0, ignore_index=True)
    return _finalise_dtypes(mobility, region_obj)


def _empty_mobility_df(region: Region) -> pd.DataFrame:
    """Return an empty mobility frame with the v2.0 schema."""
    base = pd.DataFrame(
        {
            "ev_id": np.array([], dtype=np.int32),
            "trip_idx": np.array([], dtype=np.int32),
            "start_ts": pd.array([], dtype="datetime64[ns]"),
            "end_ts": pd.array([], dtype="datetime64[ns]"),
            "miles": np.array([], dtype=np.float32),
            "kwh_consumed": np.array([], dtype=np.float32),
            "f_T_applied": np.array([], dtype=np.float32),
            "origin_type": [],
            "dest_type": [],
            "nhts_donor_houseid": [],
            "nhts_donor_vehid": [],
            "nhts_donor_travday": np.array([], dtype=np.int8),
        }
    )
    return _finalise_dtypes(base, region)


def _finalise_dtypes(mob: pd.DataFrame, region: Region) -> pd.DataFrame:
    """Cast columns to canonical schema dtypes and store timestamps in UTC.

    Wall-clock anchored on `region.tz`; converted to UTC for storage.
    DST policy: `nonexistent="shift_forward"`, `ambiguous=False` (use
    the earliest occurrence on fall-back).
    """
    out = mob.copy()
    out["ev_id"] = out["ev_id"].astype("int32")
    out["trip_idx"] = out["trip_idx"].astype("int32")

    # RFC-018: ``us_national`` (tz="UTC") is a registry-only alias — its
    # mobility ledger is not materialisable because the LT wall-clock has
    # no meaningful anchor. Refuse here instead of silently remapping to
    # America/Los_Angeles like v1.1 did.
    if region.tz == "UTC":
        raise ValueError(
            "us_national is not materialisable; pick a concrete region "
            "with a wall-clock tz (e.g. bay_area, chicago)."
        )
    local_tz = region.tz
    n = len(out)
    amb = np.zeros(n, dtype=bool) if n else np.array([], dtype=bool)
    start_local = (
        pd.to_datetime(out["start_ts"])
        .dt.tz_localize(local_tz, nonexistent="shift_forward", ambiguous=amb)
    )
    end_local = (
        pd.to_datetime(out["end_ts"])
        .dt.tz_localize(local_tz, nonexistent="shift_forward", ambiguous=amb)
    )

    # DST-gap fix: any (start, end) collapsed onto the same wall-clock
    # instant by spring-forward gets a 1-minute bump on end.
    coincide = end_local <= start_local
    if coincide.any():
        end_local = end_local.mask(coincide, start_local + pd.Timedelta(minutes=1))

    out["start_ts"] = start_local.dt.tz_convert("UTC")
    out["end_ts"] = end_local.dt.tz_convert("UTC")
    out["miles"] = out["miles"].astype("float32")
    out["kwh_consumed"] = out["kwh_consumed"].astype("float32")
    out["f_T_applied"] = out["f_T_applied"].astype("float32")
    out["origin_type"] = pd.Categorical(
        out["origin_type"], categories=list(ORIGIN_DEST_CATEGORIES)
    )
    out["dest_type"] = pd.Categorical(
        out["dest_type"], categories=list(ORIGIN_DEST_CATEGORIES)
    )
    out["nhts_donor_houseid"] = out["nhts_donor_houseid"].astype("string")
    out["nhts_donor_vehid"] = out["nhts_donor_vehid"].astype("string")
    out["nhts_donor_travday"] = out["nhts_donor_travday"].astype("int8")
    out = out.sort_values(["ev_id", "trip_idx"], kind="stable").reset_index(drop=True)
    return out


# --------------------------------------------------------------------------- #
# Orchestration: end-to-end run.
# --------------------------------------------------------------------------- #


def run(config: TravelWeekConfig, repo_root: Path) -> TravelWeekResult:
    """End-to-end M4 driver. Reads inputs, runs builder, writes outputs.

    RFC-014.r: when ``config.force_csv`` is ``False`` (default), this
    asserts the per-region NHTS parquets exist at
    ``data/pev/intermediate/nhts/<region>/`` and raises
    :class:`FileNotFoundError` if any are missing. Pass
    ``TravelWeekConfig(force_csv=True)`` to allow the legacy CSV
    fallback.
    """
    region = Region.from_name(config.region_name)

    # RFC-014.r: gate at M4 entry — require the canonical per-region
    # NHTS parquets unless ``force_csv`` is explicitly set.
    if not config.force_csv:
        _check_per_region_nhts_parquets(region)

    fleet_path = _resolve(
        config.fleet_path or default_fleet_path(region.name, config.profile_type),
        repo_root,
    )
    out_path = _resolve(
        config.out_mobility_path
        or default_mobility_path(region.name, config.profile_type),
        repo_root,
    )
    meta_path = _resolve(
        config.meta_path or default_meta_path(region.name, config.profile_type),
        repo_root,
    )
    nhts_dir = _resolve(config.nhts_data_dir, repo_root)

    logger.info("M4: reading fleet from %s", fleet_path)
    fleet = pd.read_parquet(fleet_path)
    logger.info("M4: fleet rows=%d", len(fleet))

    mobility = build_mobility(
        fleet=fleet,
        region=region,
        profile_type=config.profile_type,
        nhts_data_dir=nhts_dir,
        seed=config.master_seed,
        within_hh_prob=config.within_hh_prob,
        houseid_stitch_mode=config.houseid_stitch_mode,
    )

    trips_per_ev = mobility.groupby("ev_id", observed=True).size()
    miles_per_ev = mobility.groupby("ev_id", observed=True)["miles"].sum()
    origin_counts = mobility["origin_type"].value_counts(normalize=True)
    dest_counts = mobility["dest_type"].value_counts(normalize=True)

    contiguity_pct = _within_day_contiguity_pct(mobility)

    winter_mask = mobility["start_ts"].dt.month.isin(list(WINTER_MONTHS))
    n_trips_winter = int(winter_mask.sum())
    f_T_winter_mean = (
        float(mobility.loc[winter_mask, "f_T_applied"].mean())
        if n_trips_winter > 0
        else 1.0
    )

    result = TravelWeekResult(
        mobility=mobility,
        n_trips=int(len(mobility)),
        n_ev=int(mobility["ev_id"].nunique()) if len(mobility) else 0,
        median_trips_per_ev=float(trips_per_ev.median()) if len(mobility) else 0.0,
        median_miles_per_ev=float(miles_per_ev.median()) if len(mobility) else 0.0,
        p90_miles_per_ev=float(miles_per_ev.quantile(0.90)) if len(mobility) else 0.0,
        total_miles=float(mobility["miles"].sum()),
        total_kwh=float(mobility["kwh_consumed"].sum()),
        origin_pct={k: float(origin_counts.get(k, 0.0)) for k in ORIGIN_DEST_CATEGORIES},
        dest_pct={k: float(dest_counts.get(k, 0.0)) for k in ORIGIN_DEST_CATEGORIES},
        within_day_contiguity_pct=float(contiguity_pct),
        f_T_winter_mean=float(f_T_winter_mean),
        n_trips_winter_dec_mar=int(n_trips_winter),
        region_name=region.name,
        profile_type=config.profile_type,
        houseid_stitch_mode=config.houseid_stitch_mode,
    )

    _write_mobility(mobility, out_path)
    _update_meta(meta_path=meta_path, result=result, region=region)

    return result


def _within_day_contiguity_pct(mob: pd.DataFrame) -> float:
    """Per-(ev_id, day) contiguity %; structural sanity check."""
    if len(mob) == 0:
        return 0.0
    df = mob[["ev_id", "trip_idx", "start_ts", "end_ts"]].copy()
    df["day"] = df["start_ts"].dt.date
    df = df.sort_values(["ev_id", "trip_idx"], kind="stable").reset_index(drop=True)
    df["prev_end"] = df.groupby(["ev_id", "day"], sort=False)["end_ts"].shift(1)
    df["ok"] = df["prev_end"].isna() | (df["start_ts"] >= df["prev_end"])
    per_day = df.groupby(["ev_id", "day"], sort=False)["ok"].all()
    return float(per_day.mean() * 100.0)


# --------------------------------------------------------------------------- #
# Persistence helpers.
# --------------------------------------------------------------------------- #


def _resolve(p: Path, repo_root: Path) -> Path:
    """Resolve `p` relative to `repo_root` unless already absolute."""
    p = Path(p)
    return p if p.is_absolute() else (repo_root / p)


def _write_mobility(df: pd.DataFrame, out_path: Path) -> None:
    """Write mobility.parquet atomically with deterministic dtypes."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    df.to_parquet(tmp, engine="pyarrow", index=False, compression="snappy")
    os.replace(tmp, out_path)


def _update_meta(*, meta_path: Path, result: TravelWeekResult, region: Region) -> None:
    """Read meta.json, append travel_week_builder v2.0 provenance, write back."""
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    if meta_path.is_file():
        with meta_path.open("r", encoding="utf-8") as fh:
            meta = json.load(fh)
    else:
        meta = {}

    provenance = meta.setdefault("module_provenance", {})
    provenance["m4_travel_week_builder"] = METHODOLOGY_VERSION

    meta["travel_week_builder"] = {
        "module": "pev_synth.travel_week_builder",
        "methodology_version": METHODOLOGY_VERSION,
        "region_name": region.name,
        "profile_type": result.profile_type,
        "stitching_rule": (
            "v3.0: HOUSEID-coherent intra-household stitching — all 7 "
            "weekdays for an EV come from a single donor HOUSEID's "
            "NHTS person-days, partitioned weekday (TRAVDAY 2..6) vs "
            "weekend (TRAVDAY 1,7); cross-side substitution when the "
            "HH's single travday is on the wrong side. Resolves "
            "phase4 §13.2."
        ),
        "houseid_stitch_mode": str(result.houseid_stitch_mode),
        "within_hh_prob": float(WITHIN_HH_PROB_DEFAULT),
        "synth_calendar_year": SYNTH_CALENDAR_YEAR,
        "synth_calendar_days": SYNTH_CALENDAR_DAYS,
        "storage_tz": "UTC",
        "local_tz_anchor": region.tz,
        "winter_f_T_multiplier": float(region.winter_f_T_multiplier),
        "winter_months": list(WINTER_MONTHS),
        "heat_pump_winter_f_T_cap": HEAT_PUMP_WINTER_F_T,
        # RFC-013 + RFC-031: persist the Yuksel-Michalek polynomial inputs
        # and the resulting winter uplift for ``consumption_model``.
        # Prefer ``region.winter_t_c`` (the v2.0 source of truth);
        # ``WINTER_T_C_BY_REGION`` is retained as a legacy back-stop.
        "winter_t_c": _resolve_winter_t_c(region),
        "winter_uplift_polynomial": float(
            yuksel_michalek_uplift(
                _resolve_winter_t_c(region) if _resolve_winter_t_c(region) is not None
                else YUKSEL_MICHALEK_T_REF_C
            )
        ),
        "n_trips_total": int(result.n_trips),
        "n_ev": int(result.n_ev),
        "median_trips_per_ev": float(result.median_trips_per_ev),
        "median_miles_per_ev_per_year": float(result.median_miles_per_ev),
        "p90_miles_per_ev_per_year": float(result.p90_miles_per_ev),
        "total_miles": float(result.total_miles),
        "total_kwh_consumed": float(result.total_kwh),
        "origin_type_distribution": result.origin_pct,
        "dest_type_distribution": result.dest_pct,
        "within_day_contiguity_pct": float(result.within_day_contiguity_pct),
        "n_trips_winter_dec_mar": int(result.n_trips_winter_dec_mar),
        "f_T_winter_dec_mar_mean": float(result.f_T_winter_mean),
        "citations": [
            "Munkhammar et al. 2015 Applied Energy 142:135-143",
            "Quirós-Tortós, Ochoa & Lees 2018 PSCC (IEEE 8442988)",
            "Yuksel & Michalek 2015 (winter f_T polynomial)",
            "Atkinson et al. 2025 PMC12496165 (heat-pump correction)",
        ],
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    # RFC-013 + RFC-031: top-level ``consumption_model.winter_uplift`` mirrors
    # the M4 effective multiplier (region polynomial or pin) so downstream
    # consumers don't have to dig into the M4 sub-namespace.
    cm = meta.setdefault("consumption_model", {})
    if isinstance(cm, dict):
        effective = float(region.winter_f_T_multiplier)
        cm["winter_uplift"] = effective
        cm["winter_t_c"] = _resolve_winter_t_c(region)
        cm["winter_uplift_polynomial"] = float(
            yuksel_michalek_uplift(
                _resolve_winter_t_c(region) if _resolve_winter_t_c(region) is not None
                else YUKSEL_MICHALEK_T_REF_C
            )
        )

    tmp = meta_path.with_suffix(meta_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
    os.replace(tmp, meta_path)


# --------------------------------------------------------------------------- #
# CLI.
# --------------------------------------------------------------------------- #


def _default_repo_root() -> Path:
    from ._paths import repo_root

    return repo_root()


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m pev_synth.travel_week_builder",
        description=(
            "M4 v2.0: stitch 7-day vehicle-weeks from NHTS-2017 donor "
            "person-days, annualise to the 365-day 2001 calendar, apply "
            "winter f_T per region, and emit mobility.parquet (UTC)."
        ),
    )
    p.add_argument("--seed", type=int, default=MASTER_SEED_DEFAULT)
    p.add_argument("--within-hh-prob", type=float, default=WITHIN_HH_PROB_DEFAULT)
    p.add_argument(
        "--region",
        type=str,
        default="bay_area",
        choices=sorted(REGIONS.keys()),
        help="Region short-name.",
    )
    p.add_argument(
        "--profile-type",
        type=str,
        default="residential",
        choices=("residential", "workplace"),
    )
    p.add_argument(
        "--nhts-dir",
        type=str,
        default=str(DEFAULT_NHTS_RAW_DIR),
        help="NHTS raw / parquet directory.",
    )
    p.add_argument(
        "--fleet",
        type=str,
        default=None,
        help="Input fleet.parquet (default: per-region/profile).",
    )
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output mobility.parquet (default: per-region/profile).",
    )
    p.add_argument(
        "--meta",
        type=str,
        default=None,
        help="meta.json output (default: per-region/profile).",
    )
    p.add_argument(
        "--repo-root",
        type=str,
        default=None,
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    p.add_argument(
        "--force-csv",
        action="store_true",
        help=(
            "RFC-014.r: bypass the per-region NHTS parquet gate and "
            "fall back to the raw NHTS CSVs in --nhts-dir."
        ),
    )
    p.add_argument(
        "--houseid-stitch-mode",
        type=str,
        default=HOUSEID_STITCH_MODE_DEFAULT,
        choices=(HOUSEID_STITCH_MODE_V3, HOUSEID_STITCH_MODE_V2),
        help=(
            "v3.0 (§13.2 RESOLVED): per-EV week sampler mode. "
            "'v3_within_hh' (default) stitches 7 person-days from a "
            "single donor HOUSEID; 'v2_per_travday' preserves the "
            "legacy i.i.d. per-travday sampler with within-HH preference."
        ),
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
    config = TravelWeekConfig(
        master_seed=args.seed,
        within_hh_prob=args.within_hh_prob,
        region_name=args.region,
        profile_type=args.profile_type,
        nhts_data_dir=Path(args.nhts_dir),
        fleet_path=Path(args.fleet) if args.fleet else None,
        out_mobility_path=Path(args.out) if args.out else None,
        meta_path=Path(args.meta) if args.meta else None,
        force_csv=bool(args.force_csv),
        houseid_stitch_mode=str(args.houseid_stitch_mode),
    )
    result = run(config, repo_root=repo_root)
    print(
        "M4 travel_week_builder v2.0: "
        f"region={result.region_name} profile={result.profile_type} "
        f"n_ev={result.n_ev} n_trips={result.n_trips} "
        f"median_trips/ev={result.median_trips_per_ev:.0f} "
        f"median_miles/ev/yr={result.median_miles_per_ev:.0f} "
        f"f_T_winter_mean={result.f_T_winter_mean:.3f} "
        f"contig={result.within_day_contiguity_pct:.2f}%",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


# Silence "unused-import" for hashlib; downstream code (tests) may want
# to hash the parquet output to verify determinism.
_ = hashlib
