"""M7 — `pev_synth.hourly_resampler`.

Rasterise per-session plug events (from M6 `sessions.parquet`) into the
optimiser-facing **15-minute plug-status time series** (primary artifact) and
the **hourly plug-status time series** (derived; reporting).

Methodology references
----------------------
- Phase-4 methodology plan, §7 "Plug-status time-series construction"
  (15-min primary + hourly derived).
- Phase-5 PM dev brief, Module 7 (`pev_synth.hourly_resampler`).
- Phase-10.3 v2.0-rev §4.7 / §5.7 (storage tz, region routing).
- Phase-10.4 Wave 2 Dev F brief (region routing public API).

v2.0 additions on top of Dev B's storage-tz refactor
----------------------------------------------------
- New public helper :func:`build_plug_status` that consumes a
  ``Region`` + ``profile_type`` and resolves output paths to
  ``data/pev/processed/{region}/{profile_type}_ev_synth/``.
- :func:`resolve_v2_paths` returns an ``M7Paths`` for the v2.0 layout.
- CLI accepts ``--region`` and ``--profile-type`` for the v2.0 path.
- Workplace location ``"work"`` is propagated through the location
  categorical alongside ``"home"`` / ``"dcfc_inject"`` so plug-status
  rasterisation works for both profile types without re-mapping.

Scope (this module instance, per the user-prompt brief)
-------------------------------------------------------
- `plug_status_15min.parquet`  — primary, n_vehicles × 35,040 steps.
- `plug_status_hourly.parquet` — derived, n_vehicles × 8,760 steps.

`fleet_bounds.parquet` (curated from Argonne EV-FACTS, Quirós-Tortós et al.
(2018), INL/EXT-15-36317 (Smart & Salisbury 2015), Sears et al. (2014), EPA
fueleconomy.gov, SAE J1772-201710, NREL EVI-Pro Lite assumptions, and CARB
EMFAC2021) is intentionally OUT OF SCOPE for this Dev D / M7-v1 build per
the explicit user-prompt brief. It
will be produced by a follow-up pass (the broader phase-5 PM brief still
references it, but the user-prompt brief that this module is built against
restricts deliverables to the two plug-status parquets).

Schema (this build)
-------------------
`plug_status_15min.parquet` and `plug_status_hourly.parquet` share columns:
    ev_id    : int32              (FK to fleet.parquet)
    ts       : timestamp[ns, tz=UTC]   (v2.0 storage tz; was LA in v1.1)
    plugged  : bool               (True iff plugged-in for ≥ 7.5 min of the
                                   15-min slot, or for ≥ 2 of 4 15-min
                                   children of the hour)
    location : category, nullable ("home" / "dcfc_inject" / null when not
                                   plugged; ties resolved by counting which
                                   location dominates the slot's coverage —
                                   home wins on tie)
PK: (ev_id, ts). Sorted by (ev_id, ts).

Rounding policy (15-min)
------------------------
A slot ``[t, t + 15 min)`` is plugged iff at least one session covers ≥ 7.5
minutes of the slot's interval. Overlaps are unioned (logical OR). This is
the "majority of the slot" rule at 15-min resolution, equivalent to the
plan §7.2 "step centre falls inside [t_in, t_out)" formulation after
banker's rounding.

Rounding policy (hourly)
------------------------
An hour is plugged iff ≥ 2 of the 4 fifteen-minute slots within the hour
are plugged (plan §7.3 majority-rule downsampling).

Year-end wrap (§7.4)
--------------------
Sessions that begin before 2001-01-01 00:00 are clipped to that boundary;
sessions whose end exceeds 2002-01-01 00:00 are clipped to 2001-12-31
23:45 (the last 15-min slot start) + 15 min. Wrap-around onto early
January is NOT implemented in this build (the steady-state assumption is
handled upstream by M5/M6 producing in-year sessions).

DCFC sessions
-------------
DCFC sessions (`location == "dcfc_inject"`) ARE INCLUDED in the plug
status because the EV is drawing energy from the grid during DCFC. They
are short (≤ 30 min typical) so the ≥ 7.5-min threshold filters out
very-short DCFC fragments — which is the desired behaviour for a
residential L2-centric reporting artifact.

Determinism
-----------
The module is deterministic given a fixed input. No RNG is used. The
master seed (20260518) is recorded for provenance only.

Methodology version: v2.0-rev. Master seed: 20260520. Module offset: +6.

(Per RFC-016: constants imported from ``pev_synth._seeds``.)

Run from the repo root::

    python -m pev_synth.hourly_resampler \\
        --sessions data/pev/processed/residential_ev_synth/sessions.parquet \\
        --out-dir  data/pev/processed/residential_ev_synth
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import logging
import os
import platform
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

if TYPE_CHECKING:  # pragma: no cover
    from .regions import Region

# ---------------------------------------------------------------------------
# Orchestrator-resolved contracts.
# ---------------------------------------------------------------------------

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
MASTER_SEED_DEFAULT: Final[int] = _MASTER_SEED
MODULE_OFFSET: Final[int] = _module_offset("hourly_resampler")  # +6

#: v2.0 storage timezone (was ``"America/Los_Angeles"`` in v1.1).
#: Plan §3.1: every persisted timestamp column is ``pyarrow.timestamp("ns", tz="UTC")``.
#: The local wall-clock anchor (for year-boundary computation upstream) is
#: surfaced separately via :data:`LOCAL_ANCHOR_TZ_DEFAULT`; per the v2.0-rev
#: API contract (plan §3.3) the query-time tz override is handled at the
#: ``api.py`` boundary, not in this rasteriser.
PEV_TZ: Final[str] = "UTC"

#: Legacy alias retained for back-compat with v1.1 callers that
#: imported ``PEV_TZ`` expecting LA.  Prefer :data:`PEV_TZ` for new code.
LOCAL_ANCHOR_TZ_DEFAULT: Final[str] = "America/Los_Angeles"

SYNTH_YEAR: Final[int] = 2001  # non-leap, Monday-start.
DT_MINUTES_PRIMARY: Final[int] = 15
OVERLAP_THRESHOLD_MINUTES: Final[float] = 7.5  # majority of a 15-min slot.
HOURLY_MAJORITY_RULE: Final[int] = 2  # ≥ 2 of 4 children → hour plugged.

N_15MIN_PER_YEAR: Final[int] = 35_040  # 8760 × 4.
N_HOUR_PER_YEAR: Final[int] = 8_760
N_VEHICLES_DEFAULT: Final[int] = 1_000

#: Location categorical domain. v2.0 added ``"work"`` for workplace
#: caches; residential rows continue to use ``"home"`` / ``"dcfc_inject"``
#: exclusively, so the on-disk dictionary encoding stays back-compatible
#: with v1.1 plug_status parquets.
LOCATION_CATEGORIES: Final[tuple[str, ...]] = ("home", "dcfc_inject", "work")
LOC_HOME_IDX: Final[int] = 0
LOC_DCFC_IDX: Final[int] = 1
LOC_WORK_IDX: Final[int] = 2

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# pyarrow schemas (preserve dtypes on disk; required for tz round-trip).
# ---------------------------------------------------------------------------

PLUG_STATUS_SCHEMA: Final[pa.Schema] = pa.schema(
    [
        pa.field("ev_id", pa.int32(), nullable=False),
        pa.field("ts", pa.timestamp("ns", tz=PEV_TZ), nullable=False),
        pa.field("plugged", pa.bool_(), nullable=False),
        pa.field(
            "location",
            pa.dictionary(pa.int8(), pa.string()),
            nullable=True,
        ),
    ]
)


# ---------------------------------------------------------------------------
# Year-index construction (§7.1).
# ---------------------------------------------------------------------------

def build_year_index_15min(year: int = SYNTH_YEAR, tz: str = PEV_TZ) -> pd.DatetimeIndex:
    """Construct the 15-min, tz-aware, year-aligned slot-start index.

    Default ``tz="UTC"`` (v2.0 storage tz, plan §3.1).  The year boundary
    is ``[year-01-01 00:00 tz, (year+1)-01-01 00:00 tz)``.  35,040 slots in
    a non-leap year.

    When ``tz != "UTC"`` (e.g. a local zone for back-compat), DST is
    handled by constructing the series in UTC and converting — this
    avoids ambiguous/nonexistent-stamp foot-guns of ``tz_localize`` on a
    pre-built local-time range.
    """
    start_utc = pd.Timestamp(f"{year}-01-01", tz=tz).tz_convert("UTC")
    end_utc = pd.Timestamp(f"{year + 1}-01-01", tz=tz).tz_convert("UTC")
    idx_utc = pd.date_range(start=start_utc, end=end_utc, freq="15min", inclusive="left")
    idx = idx_utc.tz_convert(tz)
    if year == SYNTH_YEAR and len(idx) != N_15MIN_PER_YEAR:
        # 2001 is a non-leap year with no DST adjustment for the count
        # because the spring-forward + fall-back hours cancel within the
        # year. Defensive check (will trip on a year with DST asymmetry).
        raise AssertionError(
            f"expected {N_15MIN_PER_YEAR} 15-min slots in {year}, got {len(idx)}"
        )
    return idx


def build_year_index_hourly(year: int = SYNTH_YEAR, tz: str = PEV_TZ) -> pd.DatetimeIndex:
    """Construct the hourly, tz-aware, year-aligned slot-start index.

    Default ``tz="UTC"`` (v2.0 storage tz, plan §3.1).  8,760 slots in a
    non-leap year.
    """
    start_utc = pd.Timestamp(f"{year}-01-01", tz=tz).tz_convert("UTC")
    end_utc = pd.Timestamp(f"{year + 1}-01-01", tz=tz).tz_convert("UTC")
    idx_utc = pd.date_range(start=start_utc, end=end_utc, freq="1h", inclusive="left")
    idx = idx_utc.tz_convert(tz)
    if year == SYNTH_YEAR and len(idx) != N_HOUR_PER_YEAR:
        raise AssertionError(
            f"expected {N_HOUR_PER_YEAR} hourly slots in {year}, got {len(idx)}"
        )
    return idx


# ---------------------------------------------------------------------------
# Core rasterisation (per-EV).
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class RasterStats:
    """Per-EV stats returned by ``rasterise_one_ev``."""

    n_sessions: int
    n_sessions_clipped: int
    n_slots_plugged: int
    n_slots_dcfc: int
    n_slots_home: int


def rasterise_one_ev(
    sessions_ev: pd.DataFrame,
    index_15min: pd.DatetimeIndex,
    overlap_threshold_minutes: float = OVERLAP_THRESHOLD_MINUTES,
) -> tuple[np.ndarray, np.ndarray, RasterStats]:
    """Rasterise one EV's sessions onto the 15-min slot-start grid.

    Returns
    -------
    plugged : np.ndarray[bool], shape (35_040,)
        True where the slot has ≥ ``overlap_threshold_minutes`` of session
        coverage.
    location_codes : np.ndarray[int8], shape (35_040,)
        For each plugged slot: 0 → "home", 1 → "dcfc_inject", -1 → unplugged
        (the sentinel for the null category code).
        Ties between locations within a single slot are broken in favour of
        "home" (the dominant residential mode).
    stats : RasterStats
    """
    n = len(index_15min)
    if n == 0:
        return np.zeros(0, dtype=bool), np.full(0, -1, dtype=np.int8), RasterStats(0, 0, 0, 0, 0)

    # Slot-edge timestamps in UTC nanoseconds (avoid tz-arithmetic surprises).
    slot_starts_ns = index_15min.tz_convert("UTC").asi8  # int64 ns since epoch
    slot_step_ns = int(15 * 60 * 1_000_000_000)  # 15 min in ns
    year_start_ns = int(slot_starts_ns[0])
    year_end_ns = int(slot_starts_ns[-1] + slot_step_ns)  # exclusive end of last slot

    threshold_ns = int(overlap_threshold_minutes * 60 * 1_000_000_000)

    plugged_per_loc = np.zeros((len(LOCATION_CATEGORIES), n), dtype=bool)

    n_clipped = 0
    for row in sessions_ev.itertuples(index=False):
        t_in_ns = int(pd.Timestamp(row.t_in).tz_convert("UTC").value)
        t_out_ns = int(pd.Timestamp(row.t_out).tz_convert("UTC").value)
        if t_out_ns <= t_in_ns:
            continue  # zero-length or backwards; ignore.

        # Year clip.
        clipped = False
        if t_in_ns < year_start_ns:
            t_in_ns = year_start_ns
            clipped = True
        if t_out_ns > year_end_ns:
            t_out_ns = year_end_ns
            clipped = True
        if t_in_ns >= t_out_ns:
            if clipped:
                n_clipped += 1
            continue
        if clipped:
            n_clipped += 1

        loc = str(row.location) if row.location is not None else "home"
        loc_idx = LOCATION_CATEGORIES.index(loc) if loc in LOCATION_CATEGORIES else 0

        # Identify the range of slots that could overlap the session.
        # First slot whose start is < t_out_ns AND first slot whose end > t_in_ns.
        first_slot = int((t_in_ns - year_start_ns) // slot_step_ns)  # floor of slot start ≤ t_in
        # Convert to a slot whose [start, end) intersects the session.
        # The earliest such slot has start ≤ t_in_ns (so its end > t_in_ns iff
        # start + step > t_in_ns, which is true by construction of the floor).
        if first_slot < 0:
            first_slot = 0
        # the last slot whose start < t_out
        last_slot = int((t_out_ns - 1 - year_start_ns) // slot_step_ns)
        if last_slot >= n:
            last_slot = n - 1
        if last_slot < first_slot:
            continue

        for k in range(first_slot, last_slot + 1):
            slot_start = year_start_ns + k * slot_step_ns
            slot_end = slot_start + slot_step_ns
            overlap = min(slot_end, t_out_ns) - max(slot_start, t_in_ns)
            if overlap >= threshold_ns:
                plugged_per_loc[loc_idx, k] = True

    # Union of locations: a slot is plugged if any location flags it.
    plugged = plugged_per_loc.any(axis=0)
    # Tie-break priority for slots flagged by multiple sources:
    # home (0) > work (2) > dcfc_inject (1).
    # In practice residential and workplace caches never mix sessions
    # for the same EV so this ordering is only exercised when a session
    # row carries an unexpected location string; the priority preserves
    # v1.1 behaviour for the residential path (home > dcfc).
    location_codes = np.full(n, -1, dtype=np.int8)
    location_codes[plugged_per_loc[LOC_HOME_IDX]] = LOC_HOME_IDX
    only_work = plugged_per_loc[LOC_WORK_IDX] & ~plugged_per_loc[LOC_HOME_IDX]
    location_codes[only_work] = LOC_WORK_IDX
    only_dcfc = (
        plugged_per_loc[LOC_DCFC_IDX]
        & ~plugged_per_loc[LOC_HOME_IDX]
        & ~plugged_per_loc[LOC_WORK_IDX]
    )
    location_codes[only_dcfc] = LOC_DCFC_IDX

    stats = RasterStats(
        n_sessions=int(len(sessions_ev)),
        n_sessions_clipped=int(n_clipped),
        n_slots_plugged=int(plugged.sum()),
        n_slots_dcfc=int((location_codes == LOC_DCFC_IDX).sum()),
        n_slots_home=int((location_codes == LOC_HOME_IDX).sum()),
    )
    return plugged, location_codes, stats


def aggregate_hourly(plugged_15min: np.ndarray) -> np.ndarray:
    """Downsample 15-min plug status to hourly via the ≥ 2-of-4 majority rule.

    Parameters
    ----------
    plugged_15min : np.ndarray[bool], shape (35_040,)

    Returns
    -------
    plugged_hourly : np.ndarray[bool], shape (8_760,)
    """
    n = plugged_15min.shape[0]
    if n % 4 != 0:
        raise ValueError(f"plugged_15min length {n} not divisible by 4")
    reshaped = plugged_15min.reshape(n // 4, 4).astype(np.int8)
    return (reshaped.sum(axis=1) >= HOURLY_MAJORITY_RULE)


def aggregate_hourly_location(
    plugged_15min: np.ndarray,
    location_codes_15min: np.ndarray,
) -> np.ndarray:
    """Majority-vote location code per hour, conditional on the hour being plugged.

    For each hour the location code is the mode over its 4 children, with
    the v1.1 home > dcfc priority preserved and the v2.0 ``"work"``
    category folded in between (home > work > dcfc). For unplugged
    hours the code is ``-1``.
    """
    n = plugged_15min.shape[0]
    plugged_hourly = aggregate_hourly(plugged_15min)
    hourly_loc = np.full(n // 4, -1, dtype=np.int8)
    reshaped = location_codes_15min.reshape(n // 4, 4)
    count_home = (reshaped == LOC_HOME_IDX).sum(axis=1)
    count_dcfc = (reshaped == LOC_DCFC_IDX).sum(axis=1)
    count_work = (reshaped == LOC_WORK_IDX).sum(axis=1)
    # v1.1 priority for residential caches: home wins ties with dcfc.
    # v2.0 priority for workplace caches: any work child overrides dcfc.
    home_wins = (
        plugged_hourly & (count_home > 0) & (count_home >= count_dcfc) & (count_home >= count_work)
    )
    work_wins = (
        plugged_hourly
        & ~home_wins
        & (count_work > 0)
        & (count_work >= count_dcfc)
    )
    dcfc_wins = plugged_hourly & ~home_wins & ~work_wins & (count_dcfc > 0)
    hourly_loc[home_wins] = LOC_HOME_IDX
    hourly_loc[work_wins] = LOC_WORK_IDX
    hourly_loc[dcfc_wins] = LOC_DCFC_IDX
    return hourly_loc


# ---------------------------------------------------------------------------
# Pyarrow assembly + write.
# ---------------------------------------------------------------------------

def _build_plug_status_table(
    ev_ids: np.ndarray,
    ts_index: pd.DatetimeIndex,
    plugged_per_ev: np.ndarray,
    location_codes_per_ev: np.ndarray,
) -> pa.Table:
    """Assemble a long-format pyarrow Table with PK (ev_id, ts).

    Parameters
    ----------
    ev_ids : np.ndarray[int32], shape (E,)
    ts_index : pd.DatetimeIndex, length T (tz-aware LA)
    plugged_per_ev : np.ndarray[bool], shape (E, T)
    location_codes_per_ev : np.ndarray[int8], shape (E, T)
        Codes: 0 → "home", 1 → "dcfc_inject", -1 → null/unplugged.
    """
    E = ev_ids.shape[0]
    T = len(ts_index)
    if plugged_per_ev.shape != (E, T) or location_codes_per_ev.shape != (E, T):
        raise ValueError("shape mismatch between ev_ids/ts_index and plug arrays")

    # Repeat ev_id E times for each row; tile timestamps once per EV.
    ev_id_col = np.repeat(ev_ids.astype(np.int32), T)

    # ts column: tile the tz-aware index E times. We pass the raw int64 ns
    # representation and re-attach tz info via the pyarrow schema cast at
    # write time (timestamp[ns, tz=PEV_TZ] in the parquet schema). Using
    # ``ts_index.tz_convert("UTC").asi8`` produces the canonical UTC ns
    # values which pyarrow re-interprets in the target zone.
    ts_ns_utc = ts_index.tz_convert("UTC").asi8
    ts_col = np.tile(ts_ns_utc, E)
    ts_arr = pa.array(ts_col, type=pa.timestamp("ns", tz="UTC")).cast(
        pa.timestamp("ns", tz=PEV_TZ)
    )

    plugged_col = plugged_per_ev.reshape(-1).astype(bool)

    # Location: build a dictionary-encoded categorical with null where
    # location_codes == -1.
    loc_codes = location_codes_per_ev.reshape(-1)
    is_null = loc_codes == -1
    # The dictionary categories are fixed and ordered (LOCATION_CATEGORIES);
    # we attach a null mask so unplugged slots round-trip as nulls.
    indices_with_nulls = pa.array(
        np.where(is_null, 0, loc_codes).astype(np.int8),
        type=pa.int8(),
        from_pandas=False,
        mask=is_null,
    )
    dictionary = pa.array(list(LOCATION_CATEGORIES), type=pa.string())
    location_arr = pa.DictionaryArray.from_arrays(indices_with_nulls, dictionary)

    table = pa.table(
        {
            "ev_id": pa.array(ev_id_col, type=pa.int32()),
            "ts": ts_arr,
            "plugged": pa.array(plugged_col, type=pa.bool_()),
            "location": location_arr,
        },
        schema=PLUG_STATUS_SCHEMA,
    )
    return table


def write_plug_status_parquet(
    table: pa.Table,
    out_path: str | os.PathLike[str],
) -> Path:
    """Write a plug-status parquet (snappy, deterministic ordering)."""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # ``use_dictionary`` and ``compression`` are fixed for byte-stable output.
    pq.write_table(
        table,
        where=p,
        compression="snappy",
        use_dictionary=True,
        version="2.6",
        write_statistics=False,  # statistics include run timestamp on some pyarrow versions
    )
    return p


# ---------------------------------------------------------------------------
# End-to-end pipeline.
# ---------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclasses.dataclass(frozen=True)
class M7Paths:
    """Filesystem paths the M7 pipeline reads/writes."""

    sessions_in: Path
    plug_status_15min_out: Path
    plug_status_hourly_out: Path
    meta_out: Path


def _default_paths(repo_root: Path) -> M7Paths:
    base = repo_root / "data" / "pev" / "processed" / "residential_ev_synth"
    return M7Paths(
        sessions_in=base / "sessions.parquet",
        plug_status_15min_out=base / "plug_status_15min.parquet",
        plug_status_hourly_out=base / "plug_status_hourly.parquet",
        meta_out=base / "meta.json",
    )


#: Supported v2.0 profile types.
_PROFILE_TYPES_V2: Final[tuple[str, ...]] = ("residential", "workplace")


def resolve_v2_paths(
    repo_root: Path,
    region: Region,
    profile_type: str,
) -> M7Paths:
    """Resolve the v2.0 region-routed I/O paths for the M7 module.

    The on-disk layout is
    ``data/pev/processed/{region.name}/{profile_type}_ev_synth/`` per
    Phase-10.3 §1.
    """
    if profile_type not in _PROFILE_TYPES_V2:
        raise ValueError(
            f"profile_type must be one of {_PROFILE_TYPES_V2}; got {profile_type!r}"
        )
    base = (
        repo_root / "data" / "pev" / "processed"
        / region.name / f"{profile_type}_ev_synth"
    )
    return M7Paths(
        sessions_in=base / "sessions.parquet",
        plug_status_15min_out=base / "plug_status_15min.parquet",
        plug_status_hourly_out=base / "plug_status_hourly.parquet",
        meta_out=base / "meta.json",
    )


def build_plug_status(
    sessions_path: Path,
    region: Region,
    profile_type: str,
    seed: int = MASTER_SEED_DEFAULT,
    n_vehicles: int = N_VEHICLES_DEFAULT,
    repo_root: Path | None = None,
    out_dir: Path | None = None,
) -> dict[str, Path]:
    """v2.0 public API for the M7 rasteriser.

    Reads ``sessions.parquet`` from the v2.0 region-routed path
    (``data/pev/processed/{region}/{profile_type}_ev_synth/``) and writes
    the two plug-status parquets alongside it.

    Parameters
    ----------
    sessions_path:
        Absolute path to ``sessions.parquet``. Must live under
        ``data/pev/processed/{region.name}/{profile_type}_ev_synth/`` for
        the output paths to be derived correctly. If a different
        location is desired, set ``repo_root`` and let the function
        resolve the canonical layout; otherwise the parents of
        ``sessions_path`` are used directly.
    region:
        :class:`pev_synth.regions.Region` instance. Used to validate the
        region routing (the function does not implicitly trust the
        ``sessions_path`` parent's name).
    profile_type:
        ``"residential"`` or ``"workplace"``.
    seed:
        Master seed (provenance only — the module is deterministic).
    n_vehicles:
        Number of EVs to materialise rows for (default 1000).
    repo_root:
        Optional override for the *canonical repo layout* used to derive
        output paths when ``out_dir`` is not supplied. Note that this
        derives ``<repo_root>/data/pev/processed/...`` directly and does
        **not** consult ``PEV_SYNTH_DATA_ROOT``; callers that honour a
        data-root override (e.g. ``cache_regen``) should pass ``out_dir``
        explicitly instead so the rasters land in the same directory as
        the M2–M6 artifacts.
    out_dir:
        Optional explicit output directory. When provided it takes
        precedence over ``repo_root`` for the two plug-status parquets and
        ``meta.json`` — they are written directly under ``out_dir``. This
        is the data-root-honouring path the orchestrator already computed
        (and includes the ``replicates/r{id}/`` leaf when ``r_total > 1``),
        guaranteeing the M7 rasters co-locate with ``sessions.parquet``.

    Returns
    -------
    dict[str, Path]
        ``{"plug_status_15min": ..., "plug_status_hourly": ...,
        "meta": ...}``
    """
    if profile_type not in _PROFILE_TYPES_V2:
        raise ValueError(
            f"profile_type must be one of {_PROFILE_TYPES_V2}; got {profile_type!r}"
        )
    sp = Path(sessions_path)
    if out_dir is not None:
        # Explicit output directory wins: write the rasters + meta here,
        # keeping the caller-provided ``sessions_path`` for reading.
        base = Path(out_dir)
        paths = M7Paths(
            sessions_in=sp,
            plug_status_15min_out=base / "plug_status_15min.parquet",
            plug_status_hourly_out=base / "plug_status_hourly.parquet",
            meta_out=base / "meta.json",
        )
    elif repo_root is not None:
        paths = resolve_v2_paths(Path(repo_root), region, profile_type)
        # Allow caller to point at a different sessions parquet by
        # passing sessions_path explicitly; otherwise use the canonical
        # location.
        paths = dataclasses.replace(paths, sessions_in=sp)
    else:
        base = sp.parent
        paths = M7Paths(
            sessions_in=sp,
            plug_status_15min_out=base / "plug_status_15min.parquet",
            plug_status_hourly_out=base / "plug_status_hourly.parquet",
            meta_out=base / "meta.json",
        )

    run(paths, master_seed=int(seed), n_vehicles=int(n_vehicles))
    return {
        "plug_status_15min": paths.plug_status_15min_out,
        "plug_status_hourly": paths.plug_status_hourly_out,
        "meta": paths.meta_out,
    }


def rasterise_all_evs(
    sessions: pd.DataFrame,
    ev_ids: Iterable[int],
    index_15min: pd.DatetimeIndex,
    overlap_threshold_minutes: float = OVERLAP_THRESHOLD_MINUTES,
    log_every: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[RasterStats]]:
    """Rasterise sessions for a list of EVs.

    Returns
    -------
    ev_ids_arr : np.ndarray[int32], shape (E,)
    plugged_per_ev : np.ndarray[bool], shape (E, T)
    location_codes_per_ev : np.ndarray[int8], shape (E, T)
    stats : list[RasterStats], length E
    """
    ev_ids_arr = np.fromiter(ev_ids, dtype=np.int32)
    T = len(index_15min)
    E = ev_ids_arr.shape[0]
    plugged_per_ev = np.zeros((E, T), dtype=bool)
    location_codes_per_ev = np.full((E, T), -1, dtype=np.int8)
    stats: list[RasterStats] = []

    # Pre-group sessions by ev_id (a single scan; sessions table is small
    # relative to the 35M-row output).
    if "ev_id" not in sessions.columns:
        raise ValueError("sessions DataFrame missing 'ev_id' column")
    grouped = sessions.groupby("ev_id", sort=False)
    by_ev: dict[int, pd.DataFrame] = {int(k): g for k, g in grouped}

    for i, ev_id in enumerate(ev_ids_arr):
        sub = by_ev.get(int(ev_id))
        if sub is None or sub.empty:
            stats.append(RasterStats(0, 0, 0, 0, 0))
            continue
        plugged, locs, s = rasterise_one_ev(
            sub, index_15min, overlap_threshold_minutes=overlap_threshold_minutes
        )
        plugged_per_ev[i] = plugged
        location_codes_per_ev[i] = locs
        stats.append(s)
        if log_every and (i + 1) % log_every == 0:
            logger.info("rasterised %d / %d EVs", i + 1, E)

    return ev_ids_arr, plugged_per_ev, location_codes_per_ev, stats


def _validate_sessions(df: pd.DataFrame) -> None:
    """Soft schema check on the input sessions table."""
    required = {"ev_id", "t_in", "t_out", "location"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"sessions missing required columns: {sorted(missing)}")
    # Coerce timestamps to tz-aware LA.
    for col in ("t_in", "t_out"):
        ts = df[col]
        if not pd.api.types.is_datetime64_any_dtype(ts):
            raise TypeError(f"sessions.{col} must be datetime64; got {ts.dtype}")
        if getattr(ts.dt, "tz", None) is None:
            raise TypeError(f"sessions.{col} must be tz-aware (got naive)")


def run(
    paths: M7Paths,
    master_seed: int = MASTER_SEED_DEFAULT,
    n_vehicles: int = N_VEHICLES_DEFAULT,
) -> dict[str, object]:
    """End-to-end M7 pipeline.

    Returns a summary dict suitable for logging.
    """
    if not paths.sessions_in.is_file():
        raise FileNotFoundError(f"sessions.parquet not found at {paths.sessions_in}")

    sessions = pd.read_parquet(paths.sessions_in)
    _validate_sessions(sessions)

    # Coerce ev_id to int.
    sessions = sessions.copy()
    sessions["ev_id"] = sessions["ev_id"].astype(np.int32)

    # Coerce tz to the storage tz (v2.0: UTC). Upstream sessions may arrive
    # tagged as LA (v1.1) or UTC (v2.0-native) — both are accepted; the
    # numeric instant is preserved, only the tz tag is rewritten.
    for col in ("t_in", "t_out"):
        if str(sessions[col].dt.tz) != PEV_TZ:
            sessions[col] = sessions[col].dt.tz_convert(PEV_TZ)

    # Coerce location categories.
    if "location" in sessions.columns:
        sessions["location"] = sessions["location"].astype(str)

    index_15min = build_year_index_15min()
    index_hourly = build_year_index_hourly()

    ev_ids = np.arange(n_vehicles, dtype=np.int32)
    ev_ids_arr, plugged_per_ev, location_codes_per_ev, _ = rasterise_all_evs(
        sessions, ev_ids, index_15min
    )

    # 15-min table.
    table_15 = _build_plug_status_table(
        ev_ids_arr, index_15min, plugged_per_ev, location_codes_per_ev
    )
    write_plug_status_parquet(table_15, paths.plug_status_15min_out)

    # Hourly aggregation.
    plugged_hourly_per_ev = np.zeros((len(ev_ids_arr), N_HOUR_PER_YEAR), dtype=bool)
    loc_hourly_per_ev = np.full((len(ev_ids_arr), N_HOUR_PER_YEAR), -1, dtype=np.int8)
    for i in range(len(ev_ids_arr)):
        plugged_hourly_per_ev[i] = aggregate_hourly(plugged_per_ev[i])
        loc_hourly_per_ev[i] = aggregate_hourly_location(
            plugged_per_ev[i], location_codes_per_ev[i]
        )

    table_h = _build_plug_status_table(
        ev_ids_arr, index_hourly, plugged_hourly_per_ev, loc_hourly_per_ev
    )
    write_plug_status_parquet(table_h, paths.plug_status_hourly_out)

    # Fleet rates.
    fleet_rate_15 = float(plugged_per_ev.mean())
    fleet_rate_h = float(plugged_hourly_per_ev.mean())

    # Hour-of-day, day-of-week profiles (means across EVs).
    hour_of_day = index_hourly.hour
    dow = index_hourly.dayofweek
    plug_rate_by_hour = np.zeros(24, dtype=float)
    plug_rate_by_dow = np.zeros(7, dtype=float)
    fleet_hourly = plugged_hourly_per_ev.mean(axis=0)
    for h in range(24):
        plug_rate_by_hour[h] = float(fleet_hourly[hour_of_day == h].mean())
    for d in range(7):
        plug_rate_by_dow[d] = float(fleet_hourly[dow == d].mean())

    # File sizes.
    size_15_mb = paths.plug_status_15min_out.stat().st_size / (1024 ** 2)
    size_h_mb = paths.plug_status_hourly_out.stat().st_size / (1024 ** 2)

    summary: dict[str, object] = {
        "master_seed": int(master_seed),
        "methodology_version": METHODOLOGY_VERSION,
        "n_vehicles": int(len(ev_ids_arr)),
        "n_rows_15min": int(len(ev_ids_arr)) * N_15MIN_PER_YEAR,
        "n_rows_hourly": int(len(ev_ids_arr)) * N_HOUR_PER_YEAR,
        "fleet_plug_rate_15min": fleet_rate_15,
        "fleet_plug_rate_hourly": fleet_rate_h,
        "plug_rate_by_hour_of_day": [float(x) for x in plug_rate_by_hour],
        "plug_rate_by_day_of_week": [float(x) for x in plug_rate_by_dow],
        "size_15min_mb": float(size_15_mb),
        "size_hourly_mb": float(size_h_mb),
        "sha256_15min": _sha256(paths.plug_status_15min_out),
        "sha256_hourly": _sha256(paths.plug_status_hourly_out),
    }

    update_meta_with_m7(paths.meta_out, summary)

    return summary


def update_meta_with_m7(
    meta_path: str | os.PathLike[str],
    summary: dict[str, object],
) -> Path:
    """Append `hourly_resampler` provenance block to meta.json.

    RFC-021.r: routes through :meth:`pev_synth._meta.MetaWriter.update`
    so M7 shares the single-writer code path with M3 / M6.
    """
    from ._meta import MetaWriter, load_meta

    block = {
        "methodology_version": METHODOLOGY_VERSION,
        "module": "pev_synth.hourly_resampler",
        "master_seed": int(summary.get("master_seed", MASTER_SEED_DEFAULT)),
        "synth_year": SYNTH_YEAR,
        "tz": PEV_TZ,
        "dt_minutes_primary": DT_MINUTES_PRIMARY,
        "overlap_threshold_minutes_15min": OVERLAP_THRESHOLD_MINUTES,
        "hourly_majority_rule": HOURLY_MAJORITY_RULE,
        "n_rows_15min": summary.get("n_rows_15min"),
        "n_rows_hourly": summary.get("n_rows_hourly"),
        "fleet_plug_rate_15min": summary.get("fleet_plug_rate_15min"),
        "fleet_plug_rate_hourly": summary.get("fleet_plug_rate_hourly"),
        "plug_rate_by_hour_of_day": summary.get("plug_rate_by_hour_of_day"),
        "plug_rate_by_day_of_week": summary.get("plug_rate_by_day_of_week"),
        "size_15min_mb": summary.get("size_15min_mb"),
        "size_hourly_mb": summary.get("size_hourly_mb"),
        "sha256": {
            "plug_status_15min.parquet": summary.get("sha256_15min"),
            "plug_status_hourly.parquet": summary.get("sha256_hourly"),
        },
        "dcfc_in_plug_status": True,
        "notes": (
            "15-min slot plugged iff ≥ 7.5 min of session coverage; "
            "hour plugged iff ≥ 2 of 4 children plugged. "
            "Sessions clipped to [2001-01-01 00:00, 2002-01-01 00:00). "
            "fleet_bounds.parquet is OUT OF SCOPE for this v1 build."
        ),
    }
    existing = load_meta(meta_path)
    pkg_versions = dict(existing.get("package_versions", {}))
    pkg_versions.setdefault("numpy", np.__version__)
    pkg_versions.setdefault("pandas", pd.__version__)
    pkg_versions.setdefault("pyarrow", pa.__version__)
    pkg_versions.setdefault("python", platform.python_version())

    return MetaWriter.update(
        meta_path,
        stage="hourly_resampler",
        payload=block,
        methodology_version=METHODOLOGY_VERSION,
        top_level={
            "package_versions": pkg_versions,
            "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        },
        json_default=_json_default,
    )


def _json_default(o: object) -> object:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        v = float(o)
        if not np.isfinite(v):
            return None
        return v
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"unserialisable: {type(o)}")


# ---------------------------------------------------------------------------
# CLI entry-point.
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    """CLI: ``python -m pev_synth.hourly_resampler``.

    Without ``--region`` the legacy v1.1 flat layout is used. With
    ``--region`` (and optionally ``--profile-type``) the v2.0
    region-routed layout is resolved automatically.
    """
    parser = argparse.ArgumentParser(
        prog="pev_synth.hourly_resampler",
        description="M7 — rasterise sessions to 15-min and hourly plug-status parquets.",
    )
    parser.add_argument(
        "--repo-root",
        type=str,
        default=None,
        help="Repo root (defaults to two parents up from this file).",
    )
    parser.add_argument(
        "--region",
        type=str,
        default=None,
        help=(
            "v2.0 region name (e.g. 'bay_area'). When set, paths follow "
            "data/pev/processed/<region>/<profile_type>_ev_synth/."
        ),
    )
    parser.add_argument(
        "--profile-type",
        type=str,
        default="residential",
        choices=["residential", "workplace"],
        help="v2.0 profile type. Ignored when --region is unset.",
    )
    parser.add_argument("--sessions", type=str, default=None, help="Override sessions parquet.")
    parser.add_argument("--out-dir", type=str, default=None, help="Override output directory.")
    parser.add_argument("--seed", type=int, default=MASTER_SEED_DEFAULT)
    parser.add_argument("--n-vehicles", type=int, default=N_VEHICLES_DEFAULT)
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

    from ._paths import repo_root as _repo_root_fn  # noqa: PLC0415

    repo_root = (
        Path(args.repo_root).resolve() if args.repo_root else _repo_root_fn()
    )
    if args.region is not None:
        # Late import to avoid circulars on package init.
        from .regions import Region as _Region  # noqa: PLC0415

        region = _Region.from_name(args.region)
        paths = resolve_v2_paths(repo_root, region, args.profile_type)
    else:
        paths = _default_paths(repo_root)
    if args.sessions:
        paths = dataclasses.replace(paths, sessions_in=Path(args.sessions).resolve())
    if args.out_dir:
        out = Path(args.out_dir).resolve()
        paths = dataclasses.replace(
            paths,
            plug_status_15min_out=out / "plug_status_15min.parquet",
            plug_status_hourly_out=out / "plug_status_hourly.parquet",
            meta_out=out / "meta.json",
        )

    summary = run(paths, master_seed=int(args.seed), n_vehicles=int(args.n_vehicles))
    print(
        "M7 hourly_resampler: "
        f"n_vehicles={summary['n_vehicles']}, "
        f"rows_15min={summary['n_rows_15min']}, rows_hourly={summary['n_rows_hourly']}, "
        f"fleet_plug_rate_15min={summary['fleet_plug_rate_15min']:.4f}, "
        f"fleet_plug_rate_hourly={summary['fleet_plug_rate_hourly']:.4f}, "
        f"size_15min_MB={summary['size_15min_mb']:.2f}, "
        f"size_hourly_MB={summary['size_hourly_mb']:.2f}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
