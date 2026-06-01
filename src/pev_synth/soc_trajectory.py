"""M6 — `pev_synth.soc_trajectory`.

Continuous-time State-of-Charge ledger and per-session energy bookkeeping
for the synthetic EV fleet (Phase-4 §6; Phase-10.3 v2.0-rev §4.6 / §5.6;
Phase-10.4 Wave 2 Dev F brief).

v2.0 additions (this revision)
------------------------------
- Region routing: outputs land in
  ``data/pev/processed/{region.name}/{profile_type}_ev_synth/`` rather
  than the v1.1 flat ``data/pev/processed/residential_ev_synth/`` path.
- Storage timezone for v2 outputs is **UTC** (was
  ``America/Los_Angeles`` in v1.1; the legacy LA storage path is
  preserved on the v1 entry-points :func:`run` and :func:`_default_paths`
  for byte-identical replay against the v1.1 fixtures used by the
  pre-existing test suite — Dev B's UTC migration already converted the
  on-disk bay_area artifacts to UTC, so the v2 compute path operates on
  UTC inputs end-to-end).
- New public API :func:`compute_sessions` consumes a ``Region`` + a
  ``profile_type`` argument; returns the sessions DataFrame.
- For ``profile_type == "workplace"``:
    * Mobility / plug-in events are pre-filtered to workplace donor
      sessions upstream (Dev D's M3 applies ``ANNMILES < 3600``); this
      module enforces the workplace location guard locally and tags any
      residual ``home_charging_required`` cases per §4.6 (S-6).
    * No DCFC injection (workplace EVs that lack enough workplace energy
      receive the ``home_charging_required`` flag instead; their SoC is
      permitted to drop below ``soc_min`` per §4.6).
    * Sessions are at workplace dest only (``location == "work"``).
- Master seed default flipped to the v2.0 orchestrator-resolved value
  ``20260520`` (was 20260518 for v1.1).

Scope (v1 simplifications relative to the PM brief §6 — see "Hard rules"
in the orchestrator task spec):

- η is one-way (grid → battery). For a session delivering ``E_grid`` kWh
  of grid energy, the battery gains ``η · E_grid``. There is no
  temperature dependence on η in v1 (η_eff(T) deferred to v1.1; the
  Phase-4 RI-2 §2.5 correction is NOT applied in v1).
- DCFC injection (when a trip would drop SoC below ``soc_min``) is
  modelled as a single midpoint top-up that brings SoC up to
  ``soc_min_frac + 0.20`` (or 1.0 if smaller). This is the RI-3
  simplification (single midpoint injection only; no multi-step DCFC).
- BEVs and PHEVs are treated identically with the DCFC injection rule;
  the gasoline_extension PHEV branch of the brief §6.4 is NOT modelled
  in v1 (recorded in §13 known limitations).
- Per-vehicle event timeline interleaves two sources:
    (a) ``mobility.parquet`` trip events (consume SoC),
    (b) ``plug_in_events.parquet`` rows where ``plug_in == True``
        (charge SoC).
- The optimiser at ``pev_optim/`` requires per-session columns
  ``t_in``, ``t_out``, ``soc_in_kwh``, ``soc_target_out_kwh`` in kWh;
  we keep ``soc_in_pct`` and ``soc_out_pct`` as percentages for the
  user-facing report. Both sets are written to ``sessions.parquet``.

Session semantics (v1.0.1 — bug fix, see Notes below):
    For HOME sessions, ``t_in`` is the arrival/plug-in timestamp and
    ``t_out`` is the **departure deadline** — the next trip's start
    timestamp (or the end-of-window 2001-12-31 23:59 if no further
    trip). ``t_out`` is NEVER truncated to when the battery happens to
    reach 100 %. The optimiser will decide *when within ``[t_in,
    t_out)``* to actually deliver ``energy_kwh``. ``soc_target_out_kwh``
    is the SoC the optimiser must achieve by the ``t_out`` deadline;
    for v1 this equals ``soc_out_kwh`` (the SoC the EV will be at when
    it drives away).

    For DCFC injection rows (synthesised mid-trip when a leg would
    otherwise breach ``soc_min``), ``[t_in, t_out)`` is a short
    book-keeping footprint sized so ``energy_kwh ≤ max_kw · dwell``;
    the optimiser does NOT dispatch DCFC sessions.

Notes
-----

SoC ledger:
    The ledger is propagated chronologically per EV (trip events
    discharge; plug-in events charge up to the maximum the dwell
    permits, capped at 1.0). For each plug-in event the post-charge
    SoC fraction is::

        target_soc_frac = min(
            1.0,
            soc_in_frac + max_kw · dwell · η / B
        )

    where ``dwell = (departure_ts − arrival_ts).total_seconds() / 3600``
    and ``max_kw = min(EVSE_home_kw, OBC_kw)``. The grid-side energy
    delivered over the dwell is::

        energy_kwh = (soc_target_out_kwh − soc_in_kwh) / η

    which is the **total demand** the optimiser must meet by ``t_out``.

Bug fix (v1.0.1, 2026-05-19):
    Previous versions truncated ``t_out`` for home sessions to the
    moment the battery reached 100 %. This was wrong for two reasons:
    (i) the optimiser contract at ``pev_optim/optimizer.py`` reads
    ``t_out`` as the **departure deadline** (the latest time by which
    ``soc_target_out_kwh`` must be reached), and (ii) the M7 hourly
    resampler rasterises ``[t_in, t_out)`` into plug-status, which the
    truncation collapsed to a 0.47 h median (fleet plug-rate 6.6 %)
    instead of the expected residential 5–15 h overnight dwell.
    The fix: ``t_out = departure_ts`` (from M5) for every home session;
    the "completed_early" truncation has been removed.

Outputs (under ``data/pev/processed/residential_ev_synth/``):
    - ``sessions.parquet``  — primary; one row per charging session.
    - ``meta.json``         — extended with ``soc_trajectory`` block.

Methodology version: v2.0.0.  Master seed: 20260520.  Module offset: +6.
Legacy v1 entry-points (run / _default_paths / MASTER_SEED_DEFAULT) still
expose the v1.1 contract (LA storage tz, 20260518 seed) so the v1
fixture-based tests continue to pass byte-identically.
Math is in Unicode (η, Δ, ≤, ≥, ∑). Type hints throughout.
"""

from __future__ import annotations

import argparse
import logging
import platform
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np
import pandas as pd

from ._seeds import (
    MASTER_SEED as _MASTER_SEED_V2,
)
from ._seeds import (
    METHODOLOGY_VERSION as _METHODOLOGY_VERSION_V2,
)
from ._seeds import (
    module_offset as _module_offset,
)

if TYPE_CHECKING:  # pragma: no cover
    from .regions import Region

__all__ = [
    "_clamp_overlapping_sessions",
    "MASTER_SEED_DEFAULT",
    "MASTER_SEED_V2_DEFAULT",
    "METHODOLOGY_VERSION",
    "METHODOLOGY_VERSION_V2",
    "MODULE_OFFSET",
    "PEV_TZ",
    "STORAGE_TZ_V2",
    "SESSIONS_SCHEMA",
    "SOC_MIN_FRAC",
    "HOME_CHARGING_REQUIRED_MAX_SHARE",
    "SocTrajectoryConfig",
    "SocTrajectoryResult",
    "M6Paths",
    "build_event_timeline",
    "compute_initial_soc",
    "compute_sessions",
    "iterate_soc_ledger",
    "resolve_v2_paths",
    "run",
    "main",
]

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Orchestrator-resolved contracts.
# --------------------------------------------------------------------------- #

#: v1.1 contract preserved on the legacy :func:`run` entry-point.
METHODOLOGY_VERSION: Final[str] = "v1.0.1"
#: v1.1 master seed (kept so the v1 fixture-based tests stay byte-identical).
MASTER_SEED_DEFAULT: Final[int] = 20_260_518
MODULE_OFFSET: Final[int] = _module_offset("soc_trajectory")  # +6
#: Legacy v1.1 storage tz. The v2 entry-point uses :data:`STORAGE_TZ_V2`.
PEV_TZ: Final[str] = "America/Los_Angeles"

#: v2.0 orchestrator-resolved master seed (Phase-10.4 Wave 2 brief).
MASTER_SEED_V2_DEFAULT: Final[int] = _MASTER_SEED_V2
#: v2.0 methodology version (Phase-10.3 v2.0-rev §5.6).
METHODOLOGY_VERSION_V2: Final[str] = _METHODOLOGY_VERSION_V2
#: v2.0 storage tz for sessions.parquet timestamps (plan §3.1).
STORAGE_TZ_V2: Final[str] = "UTC"
#: W8 tightened bound per S-6 / §4.6 — share of workplace EVs with at least
#: one ``home_charging_required`` flag must stay at or below this fraction
#: per 4-week window. Used as a soft acceptance target in the meta block.
HOME_CHARGING_REQUIRED_MAX_SHARE: Final[float] = 0.15

# Sub-seed namespaces (per-EV independent streams, offset to avoid
# collision with other modules' streams). RFC-001/-016: imported from the
# central registry so cross-module collisions are detected at import time.
from ._seeds import SOC0_SUBSEED_OFFSET as _SOC0_SUBSEED_OFFSET  # noqa: E402

# Physics / config defaults (per the task spec).
SOC_MIN_FRAC: Final[float] = 0.10  # = soc_min_kwh / B_kwh (M2-pinned).
DCFC_TOPUP_DELTA: Final[float] = 0.20  # DCFC raises SoC to soc_min + 0.20.
#: Back-compat alias of :attr:`SocTrajectoryConfig.dcfc_nameplate_kw`'s
#: default. Default 50 kW reflects pre-2020 public-DCFC nameplates;
#: configure via ``SocTrajectoryConfig.dcfc_nameplate_kw`` for 2024-era
#: 150-350 kW fleets. Prefer the config-field form over reading this
#: module-level constant directly. Held at 50.0 to preserve regression
#: fixtures that pin session ``max_kw`` to the v2.0 default.
DCFC_NAMEPLATE_KW: Final[float] = 50.0
SOC0_BETA_ALPHA: Final[float] = 8.0  # Beta(8, 2) per the task spec.
SOC0_BETA_BETA: Final[float] = 2.0
SOC0_LO: Final[float] = SOC_MIN_FRAC  # scaled to [soc_min_frac, 1].

# Output schema (column → numpy dtype). Used both for assertion and to
# coerce dtypes before writing parquet.
#
# RFC-008: v2.0-native caches split ``energy_kwh`` into
# ``energy_kwh_grid`` (AC-side / DC-side wall energy = battery / eta) and
# ``energy_kwh_batt`` (battery-side = (soc_out - soc_in) · B). The
# legacy ``energy_kwh`` column is kept as an alias for ``_grid`` so v1.1
# consumers continue to work; ``bay_area`` v1.1_migrated caches keep only
# the single column.
SESSIONS_SCHEMA: Final[dict[str, str]] = {
    "ev_id": "int32",
    "session_idx": "int32",
    # t_in, t_out are datetime64[ns, America/Los_Angeles] — set
    # explicitly below (pandas dtype string is too brittle).
    "soc_in_pct": "float32",
    "soc_out_pct": "float32",
    "soc_in_kwh": "float32",
    "soc_target_out_kwh": "float32",
    "energy_kwh": "float32",
    "energy_kwh_grid": "float32",
    "energy_kwh_batt": "float32",
    # v3.0 (§13.6 RESOLVED) — PHEV gasoline-extension ledger.
    "gas_kwh_equivalent": "float32",
    # location is a pandas Categorical with two known values.
    "max_kw": "float32",
}

# Location categorical values.
LOC_HOME: Final[str] = "home"
LOC_DCFC: Final[str] = "dcfc_inject"
LOC_WORK: Final[str] = "work"
#: v1.1 (residential) location categories.
LOCATION_CATEGORIES: Final[tuple[str, ...]] = (LOC_HOME, LOC_DCFC)
#: v2.0 workplace location categories. No DCFC at workplace.
LOCATION_CATEGORIES_WORKPLACE: Final[tuple[str, ...]] = (LOC_WORK,)


# --------------------------------------------------------------------------- #
# Configuration dataclasses.
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class SocTrajectoryConfig:
    """Configuration for the M6 SoC-trajectory module.

    All numerical values default to the orchestrator-resolved contracts
    in the task spec. The dataclass is frozen so configs are hashable
    and seed-reproducible.
    """

    master_seed: int = MASTER_SEED_DEFAULT
    soc_min_frac: float = SOC_MIN_FRAC
    dcfc_topup_delta: float = DCFC_TOPUP_DELTA
    dcfc_nameplate_kw: float = DCFC_NAMEPLATE_KW
    soc0_alpha: float = SOC0_BETA_ALPHA
    soc0_beta: float = SOC0_BETA_BETA
    soc0_lo: float = SOC0_LO


@dataclass(frozen=True)
class M6Paths:
    """Resolved I/O paths for the M6 module."""

    fleet_in: Path
    mobility_in: Path
    plug_in_events_in: Path
    sessions_out: Path
    meta_out: Path


@dataclass
class SocTrajectoryResult:
    """End-to-end M6 summary (returned by :func:`run`)."""

    sessions: pd.DataFrame
    n_sessions: int
    n_dcfc_injections: int
    fleet_kwh_charged_grid: float
    fleet_kwh_driven: float
    sessions_per_ev: dict[str, float] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Path helpers.
# --------------------------------------------------------------------------- #

def _default_paths(repo_root: Path) -> M6Paths:
    base = repo_root / "data" / "pev"
    syn = base / "processed" / "residential_ev_synth"
    return M6Paths(
        fleet_in=syn / "fleet.parquet",
        mobility_in=syn / "mobility.parquet",
        plug_in_events_in=syn / "plug_in_events.parquet",
        sessions_out=syn / "sessions.parquet",
        meta_out=syn / "meta.json",
    )


# --------------------------------------------------------------------------- #
# Event-timeline construction.
# --------------------------------------------------------------------------- #

# Internal event-row column layout for the per-EV iteration. We use a
# structured ndarray rather than building per-event Python objects so
# the inner loop stays vectorisable per EV.

_EVENT_KIND_TRIP: Final[int] = 0
_EVENT_KIND_PLUG: Final[int] = 1


def build_event_timeline(
    mobility: pd.DataFrame,
    plug_in_events: pd.DataFrame,
    tz: str = PEV_TZ,
) -> dict[int, pd.DataFrame]:
    """Build per-EV chronological event timelines.

    Parameters
    ----------
    mobility : DataFrame
        M4 output. Required columns: ``ev_id, start_ts, end_ts, miles,
        kwh_consumed``.
    plug_in_events : DataFrame
        M5 output. Required columns: ``ev_id, arrival_ts, departure_ts,
        dwell_h, location, plug_in``. Only rows where ``plug_in == True``
        are kept (per the task spec).
    tz:
        Wall-clock zone for the returned event timestamps. Defaults to
        the legacy v1.1 ``PEV_TZ`` (LA); v2.0 callers pass ``"UTC"``.

    Returns
    -------
    dict[int, DataFrame]
        Mapping from ``ev_id`` to a DataFrame of merged events sorted
        chronologically by ``t_event`` (the trip start for trips, the
        plug-in arrival for sessions). Columns:
            kind        — int8: 0 = trip, 1 = plug.
            t_event     — pandas Timestamp (tz-aware): primary sort key.
            t_start     — Timestamp; trip start or arrival.
            t_end       — Timestamp; trip end or departure.
            miles       — float32 (NaN for plug events).
            kwh_consumed — float32 (NaN for plug events).
            dwell_h     — float32 (NaN for trip events).
    """
    # Filter to plug-in events that actually plug in (per the task spec).
    pi = plug_in_events.loc[plug_in_events["plug_in"].astype(bool)].copy()

    # Build a unified column layout, one DataFrame per source, then concat.
    trips = pd.DataFrame(
        {
            "ev_id": mobility["ev_id"].astype("int32").to_numpy(copy=False),
            "kind": np.full(len(mobility), _EVENT_KIND_TRIP, dtype="int8"),
            "t_event": mobility["start_ts"].to_numpy(copy=False),
            "t_start": mobility["start_ts"].to_numpy(copy=False),
            "t_end": mobility["end_ts"].to_numpy(copy=False),
            "miles": mobility["miles"].astype("float32").to_numpy(copy=False),
            "kwh_consumed": mobility["kwh_consumed"].astype("float32").to_numpy(copy=False),
            "dwell_h": np.full(len(mobility), np.nan, dtype="float32"),
        }
    )
    plugs = pd.DataFrame(
        {
            "ev_id": pi["ev_id"].astype("int32").to_numpy(copy=False),
            "kind": np.full(len(pi), _EVENT_KIND_PLUG, dtype="int8"),
            "t_event": pi["arrival_ts"].to_numpy(copy=False),
            "t_start": pi["arrival_ts"].to_numpy(copy=False),
            "t_end": pi["departure_ts"].to_numpy(copy=False),
            "miles": np.full(len(pi), np.nan, dtype="float32"),
            "kwh_consumed": np.full(len(pi), np.nan, dtype="float32"),
            "dwell_h": pi["dwell_h"].astype("float32").to_numpy(copy=False),
        }
    )
    # Skip empty frames before concat to avoid the pandas FutureWarning
    # about dtype inference on all-NA columns.
    frames = [f for f in (trips, plugs) if len(f) > 0]
    if not frames:
        events = trips  # both empty; trips has the right schema.
    elif len(frames) == 1:
        events = frames[0].copy()
    else:
        events = pd.concat(frames, ignore_index=True, copy=False)

    # Cast tz-aware timestamps. The concat keeps tz if both inputs are
    # tz-aware; assert just in case. ``tz`` controls the target zone
    # (v1.1 → LA, v2.0 → UTC).
    events["t_event"] = pd.to_datetime(events["t_event"], utc=False).dt.tz_convert(tz)
    events["t_start"] = pd.to_datetime(events["t_start"], utc=False).dt.tz_convert(tz)
    events["t_end"] = pd.to_datetime(events["t_end"], utc=False).dt.tz_convert(tz)

    # Sort by (ev_id, t_event, kind). Plug events tie-break after trip
    # events with the same t_event — this is the standard convention
    # (a trip that "ends" right when a session starts is consumed
    # before charging begins). Stable sort preserves intra-source order.
    events = events.sort_values(
        ["ev_id", "t_event", "kind"], kind="mergesort"
    ).reset_index(drop=True)

    # Group by ev_id and split into a dict of per-EV frames.
    return {int(k): grp.reset_index(drop=True) for k, grp in events.groupby("ev_id", sort=True)}


# --------------------------------------------------------------------------- #
# Initial SoC.
# --------------------------------------------------------------------------- #

def compute_initial_soc(
    ev_ids: np.ndarray,
    master_seed: int,
    config: SocTrajectoryConfig,
) -> dict[int, float]:
    """Sample per-EV initial SoC fraction at t=0 (2001-01-01 00:00).

    Uses Beta(α, β) on [soc0_lo, 1.0] with α=8, β=2 (mean = 8/(8+2) =
    0.80 on the unit interval → mean ≈ 0.82 on [0.10, 1.00]). This
    encodes the qualitative Pecan-Street observation that residential
    EVs start the year well-charged.

    Per-EV seed = ``master_seed + MODULE_OFFSET + _SOC0_SUBSEED_OFFSET +
    ev_id`` so that adding / removing an EV does not perturb the other
    draws (independent streams per EV).
    """
    soc0_by_ev: dict[int, float] = {}
    span = 1.0 - config.soc0_lo
    for ev in ev_ids:
        sub = int(master_seed) + MODULE_OFFSET + _SOC0_SUBSEED_OFFSET + int(ev)
        rng_ev = np.random.default_rng(sub)
        x = float(rng_ev.beta(config.soc0_alpha, config.soc0_beta))
        soc0_by_ev[int(ev)] = config.soc0_lo + span * x
    return soc0_by_ev


# --------------------------------------------------------------------------- #
# Per-EV SoC ledger.
# --------------------------------------------------------------------------- #

@dataclass
class _Session:
    """In-memory session record (one per row of the eventual parquet)."""

    ev_id: int
    session_idx: int
    t_in: pd.Timestamp
    t_out: pd.Timestamp
    soc_in_pct: float
    soc_out_pct: float
    soc_in_kwh: float
    soc_target_out_kwh: float
    energy_kwh: float  # grid-side; alias of ``energy_kwh_grid`` for v1.1 readers
    location: str
    max_kw: float
    #: v2.0 workplace flag (S-6 / §4.6). False on every residential row.
    home_charging_required: bool = False
    #: RFC-008: battery-side energy delivered = ``(soc_out - soc_in) · B``
    #: = ``eta · energy_kwh_grid``. Persisted alongside ``energy_kwh`` /
    #: ``energy_kwh_grid`` in v2.0-native sessions.parquet. Always
    #: ``energy_kwh_batt ≤ energy_kwh_grid``.
    energy_kwh_batt: float = 0.0
    #: v3.0 PHEV gasoline extension (§13.6 RESOLVED). When the running
    #: SoC ledger would drop a PHEV below ``soc_min``, the deficit is
    #: served by the ICE and accumulated until the next plug-in
    #: session, where it is attached as ``gas_kwh_equivalent``
    #: (battery-side kWh, i.e. the energy the ICE supplied to the
    #: drivetrain in lieu of the depleted battery). Always ``0.0`` for
    #: BEV rows and for PHEV sessions where no intervening trip
    #: breached the floor. Surfaced into the ``fleet_energy_conservation``
    #: numerator and excluded from the electricity-only
    #: ``energy_per_session_distribution`` histogram.
    gas_kwh_equivalent: float = 0.0


def _iterate_one_ev(
    ev_id: int,
    events: pd.DataFrame,
    soc0: float,
    B: float,
    eta: float,
    evse_home_kw: float,
    obc_kw: float,
    config: SocTrajectoryConfig,
    profile_type: str = "residential",
    is_phev: bool = False,
) -> list[_Session]:
    """Iterate the chronological event ledger for one EV.

    Parameters
    ----------
    profile_type:
        Either ``"residential"`` (default; v1.1 logic — home sessions +
        DCFC injection) or ``"workplace"`` (v2.0 — work-only sessions,
        no DCFC, ``home_charging_required`` tagging per §4.6 / S-6).
    is_phev:
        v3.0 — when True, trip-discharge below ``soc_min`` is served by
        the ICE rather than DCFC-injected (residential) / floated
        below floor (workplace). The deficit accumulates as
        ``pending_gas_kwh`` and is attached to the next charging
        session as ``gas_kwh_equivalent`` (battery-side kWh). On the
        end-of-window edge case (residual ``pending_gas_kwh > 0`` and
        no further plug-in session), the residual is attached to the
        last recorded session for the EV. BEV rows are unaffected and
        keep the v2.0 DCFC / float-below-floor semantics.

    Returns a list of ``_Session`` records. ``session_idx`` is assigned
    0..N-1 in chronological order (re-sorted at the bottom).
    """
    is_workplace = profile_type == "workplace"
    sessions: list[_Session] = []
    soc = float(soc0)
    soc_min = config.soc_min_frac
    max_home_kw = float(min(evse_home_kw, obc_kw))

    # v3.0 (§13.6 RESOLVED) — per-EV running gasoline-equivalent kWh
    # accumulator for PHEVs. Reset every time the accumulator is
    # discharged onto a session row. BEV rows leave this at 0.0.
    pending_gas_kwh: float = 0.0

    # Use plain numpy / Python loop — the per-EV event count is small
    # (~2k events/EV) and the logic is branchy enough that a Python
    # loop is clearer than vectorisation.
    kinds = events["kind"].to_numpy(copy=False)
    t_starts = events["t_start"].to_numpy(copy=False)
    t_ends = events["t_end"].to_numpy(copy=False)
    kwh_consumed = events["kwh_consumed"].to_numpy(copy=False)
    dwell_h = events["dwell_h"].to_numpy(copy=False)

    # session_idx is assigned AFTER we re-sort the recorded sessions by
    # t_in (a DCFC injection placed at a trip's midpoint can fall
    # later in clock time than a subsequent plug-in event whose
    # arrival_ts is earlier — see the re-sort at the bottom).
    next_idx = 0  # placeholder; final idx is set at the end of the loop.

    # Workplace path: tracks whether the EV's SoC dropped below soc_min
    # at any point between sessions. If so, the next workplace session
    # is tagged home_charging_required = True (per §4.6 / S-6). When no
    # session follows, the flag is recorded on the last session for the
    # EV so the indicator stays observable.
    home_required_pending = False

    for i in range(len(events)):
        kind = int(kinds[i])

        if kind == _EVENT_KIND_TRIP:
            kwh = float(kwh_consumed[i])
            # Battery delta — η does not apply to the discharge side
            # (driving energy is drawn directly from the battery).
            soc_after = soc - kwh / B
            # v3.0 (§13.6 RESOLVED): PHEV gasoline extension. When a
            # trip would drop a PHEV's SoC below the floor, the
            # deficit is served by the ICE and recorded as
            # ``gas_kwh_equivalent`` (battery-side kWh) on the next
            # plug-in session. The SoC clamps at ``soc_min`` — the
            # PHEV physically rides the floor until it next plugs
            # in. This branch is mutually exclusive with the BEV
            # DCFC-inject (residential) and workplace
            # float-below-floor branches that follow.
            if is_phev and soc_after < soc_min - 1e-9:
                deficit_kwh = (soc_min - soc_after) * B
                pending_gas_kwh += deficit_kwh
                soc = soc_min
                continue
            if is_workplace:
                # Workplace profile: NO DCFC injection. If the trip
                # would drop SoC below soc_min, tag the next session
                # as ``home_charging_required`` and let the SoC float
                # below the floor for bookkeeping. Per §4.6 (S-6), the
                # ANNMILES donor filter (Dev D's M3) keeps this rate
                # low (target ≤ 15 % per 4-week window).
                if soc_after < soc_min - 1e-9:
                    home_required_pending = True
                soc = float(soc_after)
                continue
            if soc_after < soc_min - 1e-9:
                # Infeasible trip — inject a DCFC top-up at the trip
                # midpoint. The DCFC raises the at-midpoint SoC to
                # ``min(1.0, soc_min + DCFC_TOPUP_DELTA)``.
                t_start = pd.Timestamp(t_starts[i])
                t_end = pd.Timestamp(t_ends[i])
                t_mid = t_start + (t_end - t_start) / 2

                # SoC at the midpoint (assume linear drive
                # consumption — pre-midpoint half eats kwh/2).
                soc_at_mid_pre = soc - 0.5 * kwh / B
                # If even the half-trip already breaches soc_min, the
                # driver "found a DCFC" just before reaching empty —
                # clamp at soc_min for the bookkeeping floor. The
                # raised soc_in_frac is the SoC the EV arrives at the
                # DCFC with.
                soc_in_frac_dc = float(max(soc_at_mid_pre, soc_min))
                # DCFC target frac: nominally soc_min + 0.20 (= 0.30),
                # but we must ALSO ensure the remaining half-trip
                # completes without re-breaching soc_min. The remaining
                # half consumes kwh/2 kWh → ΔSoC = kwh / (2·B). So the
                # post-DCFC SoC must be ≥ soc_min + kwh/(2·B). The
                # final target is the max of the v1 nominal and this
                # trip-completion requirement, capped at 1.0.
                soc_needed_post = soc_min + 0.5 * kwh / B
                soc_out_frac = float(min(
                    1.0,
                    max(soc_min + config.dcfc_topup_delta, soc_needed_post, soc_in_frac_dc),
                ))
                soc_in_pct = float(soc_in_frac_dc * 100.0)
                soc_in_kwh = float(soc_in_frac_dc * B)
                soc_out_pct = float(soc_out_frac * 100.0)
                soc_out_kwh = float(soc_out_frac * B)
                # Energy targeted to the grid side. DCFC is one-way
                # (η still applies AC → DC; place-holder factor).
                energy_kwh_target = float(max(0.0, (soc_out_frac - soc_in_frac_dc) * B / eta))
                # DCFC dwell footprint: physically sized to honour the
                # `energy_kwh ≤ max_kw · dwell` rule (criterion 6). A
                # 50 kW nameplate delivering E kWh needs E/50 hours,
                # plus a small safety margin so floating-point rounding
                # cannot tip a row over the constraint. The minimum
                # floor of 1 minute keeps the optimiser's
                # ``[t_in, t_out)`` interval positive when E ≈ 0.
                required_dwell_h = (
                    energy_kwh_target / config.dcfc_nameplate_kw
                    if config.dcfc_nameplate_kw > 0
                    else 0.0
                )
                dwell_minutes = max(1.0, required_dwell_h * 60.0 + 0.5)
                t_in_sess = t_mid
                t_out_sess = t_mid + pd.Timedelta(minutes=float(dwell_minutes))
                # The DCFC must fit within [t_mid, t_end] (it happens
                # in-trip). If the required dwell exceeds the trip's
                # remaining half (only for the absurd PHEV-mileage
                # tails injected by the upstream donor sampler — e.g.
                # a 1,400-mile, 3-hour trip on a 13.6-kWh battery), we
                # cap the dwell at t_end AND clamp the delivered
                # energy to what 50 kW can physically push through in
                # that window. The residual unmet consumption is
                # absorbed by the post-injection SoC clamp at soc_min
                # and is logged in `meta.json` /
                # `v1_simplifications` as a known limitation.
                if t_out_sess > t_end:
                    t_out_sess = t_end
                dwell_capped_h = (t_out_sess - t_in_sess).total_seconds() / 3600.0
                energy_max_at_dwell = config.dcfc_nameplate_kw * dwell_capped_h
                energy_kwh = float(min(energy_kwh_target, energy_max_at_dwell))
                # Recompute soc_out_frac from the (possibly clipped)
                # energy: ΔSoC = energy · η / B.
                soc_out_frac_realised = float(min(1.0, soc_in_frac_dc + energy_kwh * eta / B))
                # Enforce monotonicity for floating-point safety.
                soc_out_frac_realised = max(soc_out_frac_realised, soc_in_frac_dc)
                soc_out_frac = soc_out_frac_realised
                soc_out_pct = float(soc_out_frac * 100.0)
                soc_out_kwh = float(soc_out_frac * B)
                # RFC-008: battery-side energy = grid · eta (one-way DCFC).
                energy_kwh_batt = float(max(0.0, energy_kwh * eta))
                sessions.append(
                    _Session(
                        ev_id=ev_id,
                        session_idx=next_idx,
                        t_in=t_in_sess,
                        t_out=t_out_sess,
                        soc_in_pct=soc_in_pct,
                        soc_out_pct=soc_out_pct,
                        soc_in_kwh=soc_in_kwh,
                        soc_target_out_kwh=soc_out_kwh,
                        energy_kwh=energy_kwh,
                        location=LOC_DCFC,
                        max_kw=float(config.dcfc_nameplate_kw),
                        energy_kwh_batt=energy_kwh_batt,
                    )
                )
                next_idx += 1
                # Resume the trip from the post-DCFC SoC. The DCFC
                # top-up was sized so that the second half of the trip
                # (kwh/2 kWh) completes without re-breaching soc_min.
                # If the half-trip is so large that even
                # `soc_min + kwh/(2·B) > 1.0` (impossible to fit in
                # battery), we clamp at soc_min and document the
                # residual underconsumption in §13 known limitations.
                soc = float(max(soc_out_frac - 0.5 * kwh / B, soc_min))
            else:
                soc = float(soc_after)

        else:  # _EVENT_KIND_PLUG
            # Bug fix (v1.0.1): t_out is ALWAYS the departure deadline
            # (= the next trip's start, or end-of-window per M5). The
            # optimiser interprets t_out as the deadline by which
            # soc_target_out_kwh must be reached; the previous
            # truncation to "when SoC reaches 100 %" collapsed the
            # ``[t_in, t_out)`` plug-in interval that M7 rasterises
            # into plug-status, hence the 0.47 h median session
            # duration / 6.6 % plug-rate regression. See module
            # docstring's "Notes / Bug fix" section.
            t_in_sess = pd.Timestamp(t_starts[i])
            t_out_sess = pd.Timestamp(t_ends[i])
            dwell = float(dwell_h[i])

            soc_in_frac = float(max(soc, soc_min))
            # Maximum grid-side energy deliverable in the dwell window.
            max_session_kwh_grid = max_home_kw * dwell
            # Target SoC fraction: cap at 1.0; gain in battery = η · E_grid.
            # Most overnight sessions saturate at 1.0; the EV reaches
            # this earlier than t_out but the optimiser still owns the
            # decision of *when* within [t_in, t_out) to deliver the
            # energy — we do NOT truncate t_out.
            target_soc_frac = float(min(1.0, soc_in_frac + max_session_kwh_grid * eta / B))
            # Enforce monotonicity even in pathological cases where
            # numerical noise would put target slightly below in.
            if target_soc_frac < soc_in_frac:
                target_soc_frac = soc_in_frac

            soc_in_pct = float(soc_in_frac * 100.0)
            soc_out_pct = float(target_soc_frac * 100.0)
            soc_in_kwh = float(soc_in_frac * B)
            soc_out_kwh = float(target_soc_frac * B)
            # For v1, soc_target_out_kwh = soc_out_kwh: the deadline
            # value the optimiser sees equals the SoC the EV will be
            # at when it drives away. The grid-side energy demand
            # follows directly from the two endpoints.
            soc_target_out_kwh_val = soc_out_kwh
            energy_kwh = float((target_soc_frac - soc_in_frac) * B / eta) if eta > 0 else 0.0

            # Defensive guard for the degenerate (zero-dwell) case
            # only — keeps ``t_out > t_in`` for the optimiser. This is
            # NOT the deleted truncation; it triggers solely when M5
            # emits a plug-in with departure_ts == arrival_ts.
            if t_out_sess <= t_in_sess:
                t_out_sess = t_in_sess + pd.Timedelta(minutes=1)

            # Location: residential plug-ins are LOC_HOME; workplace
            # plug-ins are LOC_WORK (the upstream filter guarantees the
            # event is at the work dest). The home_charging_required
            # flag latches True for the next workplace session whenever
            # an intervening trip dropped SoC below soc_min.
            location_value = LOC_WORK if is_workplace else LOC_HOME
            session_home_required = False
            if is_workplace:
                # Allow the v2 contract to flag the session when the
                # CURRENT pre-charge SoC is itself below soc_min (i.e.
                # the EV is rolling into the workplace already in
                # deficit) OR when an intervening trip flagged the
                # pending bit.
                if soc < soc_min - 1e-9 or home_required_pending:
                    session_home_required = True
                home_required_pending = False
                # Workplace soc_in is the *actual* SoC (no clamp to
                # soc_min) so the home_charging_required deficit is
                # visible to downstream validators.
                soc_in_frac_workplace = float(soc)
                soc_in_pct = float(soc_in_frac_workplace * 100.0)
                soc_in_kwh = float(soc_in_frac_workplace * B)
                # Target gain is still bounded by what max_kw can push
                # over the dwell; SoC may finish below 1.0 (typical) or
                # below soc_min if the deficit is large (flagged).
                target_soc_frac = float(min(
                    1.0,
                    soc_in_frac_workplace + max_session_kwh_grid * eta / B,
                ))
                if target_soc_frac < soc_in_frac_workplace:
                    target_soc_frac = soc_in_frac_workplace
                soc_out_pct = float(target_soc_frac * 100.0)
                soc_out_kwh = float(target_soc_frac * B)
                soc_target_out_kwh_val = soc_out_kwh
                energy_kwh = float(
                    (target_soc_frac - soc_in_frac_workplace) * B / eta
                ) if eta > 0 else 0.0
            # RFC-008: battery-side energy = grid · eta_ac.
            energy_kwh_batt = float(max(0.0, energy_kwh * eta))
            # v3.0 (§13.6 RESOLVED): flush any accumulated PHEV
            # gasoline-extension deficit onto this plug-in row and
            # reset the per-EV accumulator. Always 0.0 on BEV rows
            # (``is_phev=False`` short-circuits the accumulator above).
            gas_kwh_equivalent = float(pending_gas_kwh)
            pending_gas_kwh = 0.0
            sessions.append(
                _Session(
                    ev_id=ev_id,
                    session_idx=next_idx,
                    t_in=t_in_sess,
                    t_out=t_out_sess,
                    soc_in_pct=soc_in_pct,
                    soc_out_pct=soc_out_pct,
                    soc_in_kwh=soc_in_kwh,
                    soc_target_out_kwh=soc_target_out_kwh_val,
                    energy_kwh=energy_kwh,
                    location=location_value,
                    max_kw=float(max_home_kw),
                    home_charging_required=session_home_required,
                    energy_kwh_batt=energy_kwh_batt,
                    gas_kwh_equivalent=gas_kwh_equivalent,
                )
            )
            next_idx += 1
            soc = float(target_soc_frac)

    # Residual pending flag: if a workplace EV's timeline ends with a
    # trip that breached soc_min but no further session followed, we
    # latch the flag on the most recent workplace session so the
    # indicator survives.
    if is_workplace and home_required_pending and sessions:
        sessions[-1].home_charging_required = True

    # v3.0 (§13.6 RESOLVED): residual PHEV gas-extension deficit. If
    # the EV's simulation window ends with ``pending_gas_kwh > 0`` and
    # no further plug-in session followed, attach the residual to the
    # last recorded session for that EV. Choice rationale: this is the
    # simpler path (no synthetic zero-energy gas-only marker row), and
    # it keeps the ``fleet_energy_conservation`` numerator complete
    # without inventing a new ``location`` category. PHEVs with no
    # sessions at all in the window are dropped — but those EVs also
    # contribute zero to the energy-conservation denominator (no
    # trips fed kwh_consumed) by construction.
    if is_phev and pending_gas_kwh > 0.0 and sessions:
        sessions[-1].gas_kwh_equivalent += float(pending_gas_kwh)
        pending_gas_kwh = 0.0

    # Re-sort sessions by t_in (chronological) and reassign session_idx
    # so the optimiser's left-to-right scan over (session_idx) is also
    # left-to-right in clock time. The SoC ledger itself was already
    # computed in event order — only the row ordering changes.
    sessions.sort(key=lambda s: (s.t_in, s.t_out))
    # RFC-026: clamp-overlap logic extracted into a free helper.
    _clamp_overlapping_sessions(sessions)
    for i, sess in enumerate(sessions):
        sess.session_idx = i

    return sessions


def _clamp_overlapping_sessions(sessions: list[_Session]) -> None:
    """Clamp consecutive ``[t_in, t_out)`` windows so they don't overlap.

    Handles the rare cases where (a) an upstream M5 plug-in event's
    ``departure_ts`` falls after the next M4 trip's ``start_ts`` and (b)
    a DCFC dwell capped at trip ``t_end`` still extends past a home
    plug-in's ``arrival_ts``. The clamp preserves criterion 6
    (``energy_kwh ≤ max_kw · dwell``) because shrinking the dwell only
    tightens the upper bound. A clamp that would zero the dwell is
    widened to a 1-second floor.

    Mutates ``sessions`` in place. Requires ``sessions`` to be sorted by
    ``t_in``. Idempotent — calling twice produces the same result.

    Per RFC-008 the battery-side energy is scaled in lock-step so
    ``energy_kwh_batt = eta · energy_kwh_grid`` survives the clamp.
    """
    for i in range(len(sessions) - 1):
        if sessions[i].t_out > sessions[i + 1].t_in:
            new_t_out = sessions[i + 1].t_in
            if new_t_out <= sessions[i].t_in:
                new_t_out = sessions[i].t_in + pd.Timedelta(seconds=1)
            sessions[i].t_out = new_t_out
            new_dwell_h = (
                sessions[i].t_out - sessions[i].t_in
            ).total_seconds() / 3600.0
            max_e = sessions[i].max_kw * new_dwell_h
            if sessions[i].energy_kwh > max_e:
                prev_grid = sessions[i].energy_kwh
                sessions[i].energy_kwh = float(max_e)
                if prev_grid > 0:
                    ratio = float(max_e) / prev_grid
                    sessions[i].energy_kwh_batt = float(
                        sessions[i].energy_kwh_batt * ratio
                    )


def iterate_soc_ledger(
    fleet: pd.DataFrame,
    timelines: dict[int, pd.DataFrame],
    soc0_by_ev: dict[int, float],
    config: SocTrajectoryConfig,
    profile_type: str = "residential",
    storage_tz: str = PEV_TZ,
) -> pd.DataFrame:
    """Iterate the SoC ledger for every EV and emit the sessions table.

    Parameters
    ----------
    fleet : DataFrame
        Must have ``ev_id, B_kwh, eta_charge_nominal, OBC_kw,
        EVSE_home_kw`` columns.
    timelines : dict[int, DataFrame]
        Output of :func:`build_event_timeline`.
    soc0_by_ev : dict[int, float]
        Output of :func:`compute_initial_soc`.
    config : SocTrajectoryConfig
        Module config.
    profile_type:
        ``"residential"`` (v1.1 default) or ``"workplace"`` (v2.0).
    storage_tz:
        Output timezone for the ``t_in`` / ``t_out`` columns. Defaults
        to v1.1 LA for back-compat; v2.0 callers should pass
        :data:`STORAGE_TZ_V2` (= ``"UTC"``).

    Returns
    -------
    DataFrame
        Long-format sessions, one row per session, sorted by
        ``(ev_id, session_idx)``.
    """
    # Index fleet by ev_id for O(1) lookup of physics parameters.
    fleet_idx = fleet.set_index("ev_id")
    # v3.0: PHEV identification for the §13.6 gasoline-extension branch
    # in :func:`_iterate_one_ev`. Fixtures that omit the ``powertrain``
    # column fall back to BEV-only semantics (no PHEV branch fires) so
    # the legacy synthetic test fleets keep their v1.1 behaviour.
    has_powertrain = "powertrain" in fleet_idx.columns
    all_sessions: list[_Session] = []

    for ev in sorted(fleet_idx.index.unique()):
        ev_int = int(ev)
        events = timelines.get(ev_int)
        if events is None or len(events) == 0:
            continue
        row = fleet_idx.loc[ev_int]
        B = float(row["B_kwh"])
        eta = float(row["eta_charge_nominal"])
        evse_home_kw = float(row["EVSE_home_kw"])
        obc_kw = float(row["OBC_kw"])
        soc0 = soc0_by_ev.get(ev_int, 0.70)
        is_phev = bool(
            has_powertrain and str(row["powertrain"]) == "PHEV"
        )

        ev_sessions = _iterate_one_ev(
            ev_id=ev_int,
            events=events,
            soc0=soc0,
            B=B,
            eta=eta,
            evse_home_kw=evse_home_kw,
            obc_kw=obc_kw,
            config=config,
            profile_type=profile_type,
            is_phev=is_phev,
        )
        all_sessions.extend(ev_sessions)

    return _sessions_to_dataframe(
        all_sessions, profile_type=profile_type, storage_tz=storage_tz
    )


def _location_categories(profile_type: str) -> tuple[str, ...]:
    """Return the categorical-domain for the given profile."""
    return (
        LOCATION_CATEGORIES_WORKPLACE
        if profile_type == "workplace"
        else LOCATION_CATEGORIES
    )


# --------------------------------------------------------------------------- #
# v2.0 public API: compute_sessions.
# --------------------------------------------------------------------------- #

#: Supported v2.0 profile types (see :mod:`pev_synth.regions`).
_PROFILE_TYPES_V2: Final[tuple[str, ...]] = ("residential", "workplace")


def compute_sessions(
    fleet: pd.DataFrame,
    mobility: pd.DataFrame,
    plug_in_events: pd.DataFrame,
    region: Region,
    profile_type: str,
    seed: int = MASTER_SEED_V2_DEFAULT,
) -> pd.DataFrame:
    """v2.0 public API for the M6 SoC ledger.

    Parameters
    ----------
    fleet:
        v2.0 ``fleet.parquet`` with the v1.1 + M3/M5 columns (``ev_id``,
        ``B_kwh``, ``eta_charge_nominal``, ``OBC_kw``, ``EVSE_home_kw``).
    mobility:
        v2.0 ``mobility.parquet``. Timestamps tz-aware UTC. For
        ``profile_type == "workplace"`` only trips with
        ``dest_type == "work"`` are charging-relevant, but ALL trips
        discharge SoC and are retained.
    plug_in_events:
        v2.0 ``plug_in_events.parquet`` from M5. Workplace events have
        ``location == "work"`` (filtered before this function is called
        by the upstream Dev E module). This function additionally
        enforces a local guard for ``profile_type == "workplace"`` that
        every plug-in carries ``location == "work"``.
    region:
        :class:`pev_synth.regions.Region` instance. The region's
        ``name`` controls the output path layout when persisting; this
        function does not write to disk — it returns the DataFrame.
    profile_type:
        ``"residential"`` or ``"workplace"``.
    seed:
        Master seed. Defaults to :data:`MASTER_SEED_V2_DEFAULT`
        (``20260520``).

    Returns
    -------
    DataFrame
        v1.1-shape sessions table with UTC ``t_in`` / ``t_out``.
        For workplace profile, a ``home_charging_required`` bool column
        is appended.
    """
    if profile_type not in _PROFILE_TYPES_V2:
        raise ValueError(
            f"profile_type must be one of {_PROFILE_TYPES_V2}; got {profile_type!r}"
        )

    cfg = SocTrajectoryConfig(master_seed=int(seed))

    # Soft schema checks (early failure with a useful message).
    _validate_v2_inputs(fleet, mobility, plug_in_events, profile_type)

    # For the workplace profile, defensively enforce the work-only
    # invariant on the plug-in events. We accept rows tagged
    # ``location == "work"``; anything else is dropped (no DCFC; no
    # home) and warned via the logger.
    if profile_type == "workplace" and "location" in plug_in_events.columns:
        loc_str = plug_in_events["location"].astype(str)
        is_work = loc_str == LOC_WORK
        if not bool(is_work.all()):
            n_dropped = int((~is_work).sum())
            if n_dropped:
                logger.warning(
                    "compute_sessions(workplace): dropping %d non-work "
                    "plug-in events (upstream filter contract violated)",
                    n_dropped,
                )
            plug_in_events = plug_in_events.loc[is_work].copy()

    timelines = build_event_timeline(
        mobility, plug_in_events, tz=STORAGE_TZ_V2
    )
    ev_ids = fleet["ev_id"].to_numpy()
    soc0_by_ev = compute_initial_soc(ev_ids, cfg.master_seed, cfg)

    sessions = iterate_soc_ledger(
        fleet,
        timelines,
        soc0_by_ev,
        cfg,
        profile_type=profile_type,
        storage_tz=STORAGE_TZ_V2,
    )
    return sessions


def _validate_v2_inputs(
    fleet: pd.DataFrame,
    mobility: pd.DataFrame,
    plug_in_events: pd.DataFrame,
    profile_type: str,
) -> None:
    """Early-fail schema guard for :func:`compute_sessions`."""
    required_fleet = {"ev_id", "B_kwh", "eta_charge_nominal", "OBC_kw", "EVSE_home_kw"}
    missing = required_fleet - set(fleet.columns)
    if missing:
        raise ValueError(f"fleet missing columns: {sorted(missing)}")
    required_mob = {"ev_id", "start_ts", "end_ts", "miles", "kwh_consumed"}
    missing = required_mob - set(mobility.columns)
    if missing:
        raise ValueError(f"mobility missing columns: {sorted(missing)}")
    required_pi = {"ev_id", "arrival_ts", "departure_ts", "dwell_h", "plug_in"}
    missing = required_pi - set(plug_in_events.columns)
    if missing:
        raise ValueError(f"plug_in_events missing columns: {sorted(missing)}")
    if profile_type == "workplace" and "location" not in plug_in_events.columns:
        raise ValueError(
            "workplace plug_in_events must include a 'location' column "
            "tagged as 'work' (upstream M5 contract)."
        )


def resolve_v2_paths(
    repo_root: Path,
    region: Region,
    profile_type: str,
) -> M6Paths:
    """Resolve the v2.0 region-routed I/O paths for the M6 module.

    The on-disk layout is
    ``data/pev/processed/{region.name}/{profile_type}_ev_synth/`` per
    Phase-10.3 §1.

    Parameters
    ----------
    repo_root:
        Path to the repository root (the directory containing ``data/``
        and ``code/``).
    region:
        :class:`pev_synth.regions.Region` instance.
    profile_type:
        ``"residential"`` or ``"workplace"``.
    """
    if profile_type not in _PROFILE_TYPES_V2:
        raise ValueError(
            f"profile_type must be one of {_PROFILE_TYPES_V2}; got {profile_type!r}"
        )
    base = repo_root / "data" / "pev" / "processed" / region.name / f"{profile_type}_ev_synth"
    return M6Paths(
        fleet_in=base / "fleet.parquet",
        mobility_in=base / "mobility.parquet",
        plug_in_events_in=base / "plug_in_events.parquet",
        sessions_out=base / "sessions.parquet",
        meta_out=base / "meta.json",
    )


def _sessions_to_dataframe(
    sessions: list[_Session],
    profile_type: str = "residential",
    storage_tz: str = PEV_TZ,
) -> pd.DataFrame:
    """Materialise a list of `_Session` into a parquet-ready DataFrame.

    The ``profile_type`` argument controls the categorical domain of the
    ``location`` column (residential adds ``"home" | "dcfc_inject"``;
    workplace uses ``"work"`` exclusively). The ``storage_tz`` argument
    determines the output tz of ``t_in`` / ``t_out``; v1 callers leave
    it at the LA default while v2 callers pass UTC.
    """
    location_domain = _location_categories(profile_type)
    if not sessions:
        return _empty_sessions_df(profile_type=profile_type, storage_tz=storage_tz)

    energy_grid_arr = np.array(
        [s.energy_kwh for s in sessions], dtype="float32"
    )
    energy_batt_arr = np.array(
        [s.energy_kwh_batt for s in sessions], dtype="float32"
    )
    gas_kwh_arr = np.array(
        [s.gas_kwh_equivalent for s in sessions], dtype="float32"
    )
    df = pd.DataFrame(
        {
            "ev_id": np.array([s.ev_id for s in sessions], dtype="int32"),
            "session_idx": np.array([s.session_idx for s in sessions], dtype="int32"),
            "t_in": pd.to_datetime([s.t_in for s in sessions]),
            "t_out": pd.to_datetime([s.t_out for s in sessions]),
            "soc_in_pct": np.array([s.soc_in_pct for s in sessions], dtype="float32"),
            "soc_out_pct": np.array([s.soc_out_pct for s in sessions], dtype="float32"),
            "soc_in_kwh": np.array([s.soc_in_kwh for s in sessions], dtype="float32"),
            "soc_target_out_kwh": np.array(
                [s.soc_target_out_kwh for s in sessions], dtype="float32"
            ),
            # RFC-008: v2.0-native caches emit both grid (=legacy
            # ``energy_kwh``) and batt sides; the ``energy_kwh`` alias
            # column keeps v1.1 readers working.
            "energy_kwh": energy_grid_arr,
            "energy_kwh_grid": energy_grid_arr,
            "energy_kwh_batt": energy_batt_arr,
            # v3.0 (§13.6 RESOLVED): PHEV gasoline-extension ledger.
            # 0.0 on every BEV row and on PHEV rows where no intervening
            # trip breached ``soc_min``. Validator picks this up in
            # ``check_fleet_energy_conservation`` (numerator) and
            # excludes nonzero rows from
            # ``check_energy_per_session_distribution`` (electricity-only).
            "gas_kwh_equivalent": gas_kwh_arr,
            "location": pd.Categorical(
                [s.location for s in sessions], categories=list(location_domain)
            ),
            "max_kw": np.array([s.max_kw for s in sessions], dtype="float32"),
        }
    )
    if profile_type == "workplace":
        df["home_charging_required"] = np.array(
            [bool(s.home_charging_required) for s in sessions], dtype=bool
        )
    # Ensure tz on timestamps. to_datetime above preserves tz, but we
    # tz_convert defensively to the target storage tz.
    if df["t_in"].dt.tz is None:
        df["t_in"] = df["t_in"].dt.tz_localize(storage_tz)
    else:
        df["t_in"] = df["t_in"].dt.tz_convert(storage_tz)
    if df["t_out"].dt.tz is None:
        df["t_out"] = df["t_out"].dt.tz_localize(storage_tz)
    else:
        df["t_out"] = df["t_out"].dt.tz_convert(storage_tz)

    df = df.sort_values(["ev_id", "session_idx"], kind="mergesort").reset_index(drop=True)
    return df


def _empty_sessions_df(
    profile_type: str = "residential",
    storage_tz: str = PEV_TZ,
) -> pd.DataFrame:
    """Return an empty sessions DataFrame with the documented schema."""
    location_domain = _location_categories(profile_type)
    cols = {
        "ev_id": pd.Series([], dtype="int32"),
        "session_idx": pd.Series([], dtype="int32"),
        "t_in": pd.Series([], dtype=f"datetime64[ns, {storage_tz}]"),
        "t_out": pd.Series([], dtype=f"datetime64[ns, {storage_tz}]"),
        "soc_in_pct": pd.Series([], dtype="float32"),
        "soc_out_pct": pd.Series([], dtype="float32"),
        "soc_in_kwh": pd.Series([], dtype="float32"),
        "soc_target_out_kwh": pd.Series([], dtype="float32"),
        "energy_kwh": pd.Series([], dtype="float32"),
        # RFC-008: empty frame also reflects the split for schema parity.
        "energy_kwh_grid": pd.Series([], dtype="float32"),
        "energy_kwh_batt": pd.Series([], dtype="float32"),
        # v3.0 (§13.6 RESOLVED) — PHEV gasoline-extension ledger column.
        "gas_kwh_equivalent": pd.Series([], dtype="float32"),
        "location": pd.Categorical([], categories=list(location_domain)),
        "max_kw": pd.Series([], dtype="float32"),
    }
    if profile_type == "workplace":
        cols["home_charging_required"] = pd.Series([], dtype=bool)
    return pd.DataFrame(cols)


# --------------------------------------------------------------------------- #
# meta.json update.
# --------------------------------------------------------------------------- #

def _update_meta(
    meta_path: Path,
    *,
    n_sessions: int,
    n_dcfc_injections: int,
    fleet_kwh_charged_grid: float,
    fleet_kwh_driven: float,
    sessions_per_ev_median: float,
    profile_type: str = "residential",
    storage_tz: str = PEV_TZ,
    home_charging_required_share: float | None = None,
    n_home_charging_required_sessions: int = 0,
) -> None:
    """Refresh ``meta.json`` with the M6 provenance block.

    RFC-021.r: the actual read-modify-write is routed through
    :meth:`pev_synth._meta.MetaWriter.update` so M6 shares the single
    writer code path with M3 / M7.
    """
    from ._meta import MetaWriter, load_meta

    is_v2 = storage_tz == STORAGE_TZ_V2
    version_label = METHODOLOGY_VERSION_V2 if is_v2 else METHODOLOGY_VERSION
    seed_label = MASTER_SEED_V2_DEFAULT if is_v2 else MASTER_SEED_DEFAULT

    block = {
        "module": "pev_synth.soc_trajectory",
        "methodology_version": version_label,
        "master_seed": int(seed_label),
        "module_offset": MODULE_OFFSET,
        "profile_type": profile_type,
        "storage_tz": storage_tz,
        "n_sessions": int(n_sessions),
        "n_dcfc_injections": int(n_dcfc_injections),
        "dcfc_fraction": float(n_dcfc_injections / n_sessions) if n_sessions > 0 else 0.0,
        "fleet_kwh_charged_grid": float(fleet_kwh_charged_grid),
        "fleet_kwh_driven": float(fleet_kwh_driven),
        "energy_ratio_charged_over_driven": (
            float(fleet_kwh_charged_grid / fleet_kwh_driven)
            if fleet_kwh_driven > 0 else float("nan")
        ),
        "sessions_per_ev_median": float(sessions_per_ev_median),
        "session_t_out_semantics": "departure_ts (full dwell)",
        "session_t_out_bug_fix_note": (
            "v1.0.1 (2026-05-19): t_out is now the departure deadline "
            "(next trip's start_ts from M5, or end-of-window if last). "
            "The previous 'completed_early' truncation to when SoC "
            "reached 100 % violated the optimiser contract and "
            "collapsed the M7 plug-status rasterisation to a 0.47 h "
            "median. soc_target_out_kwh == soc_out_kwh for every row."
        ),
        "soc0_distribution": {
            "alpha": SOC0_BETA_ALPHA,
            "beta": SOC0_BETA_BETA,
            "scaled_to_lo": SOC0_LO,
            "scaled_to_hi": 1.0,
        },
        "dcfc_policy_v1": {
            "topup_delta_frac": DCFC_TOPUP_DELTA,
            "topup_target_frac": min(1.0, SOC_MIN_FRAC + DCFC_TOPUP_DELTA),
            "nameplate_kw": DCFC_NAMEPLATE_KW,
            "injection_strategy": "single midpoint injection (RI-3 simplification)",
        },
        "v1_simplifications": [
            "η_eff(T) NOT modelled (RI-2 deferred to v1.1).",
            "PHEV gasoline_extension NOT modelled (§6.4 deferred to v1.1).",
            "DCFC injection: single midpoint top-up to soc_min + 0.20 (RI-3).",
        ],
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    if profile_type == "workplace":
        block["home_charging_required_share"] = (
            float(home_charging_required_share)
            if home_charging_required_share is not None
            else 0.0
        )
        block["home_charging_required_max_share"] = HOME_CHARGING_REQUIRED_MAX_SHARE
        block["n_home_charging_required_sessions"] = int(
            n_home_charging_required_sessions
        )

    # Refresh the existing package_versions block (shallow-merged at the
    # top level by MetaWriter.update).
    existing = load_meta(meta_path)
    pkg_versions = dict(existing.get("package_versions", {}))
    pkg_versions["python"] = platform.python_version()
    try:
        pkg_versions["numpy"] = np.__version__
        pkg_versions["pandas"] = pd.__version__
    except AttributeError:  # pragma: no cover
        pass

    MetaWriter.update(
        meta_path,
        stage="soc_trajectory",
        payload=block,
        methodology_version=version_label,
        top_level={"package_versions": pkg_versions},
    )


# --------------------------------------------------------------------------- #
# End-to-end driver.
# --------------------------------------------------------------------------- #

def run(
    paths: M6Paths,
    config: SocTrajectoryConfig | None = None,
    profile_type: str = "residential",
    storage_tz: str = PEV_TZ,
) -> SocTrajectoryResult:
    """Execute the full M6 pipeline end-to-end.

    Reads ``fleet.parquet``, ``mobility.parquet`` and
    ``plug_in_events.parquet``; writes ``sessions.parquet`` and updates
    ``meta.json``.

    Parameters
    ----------
    paths:
        Resolved I/O paths (see :func:`resolve_v2_paths` for the v2.0
        layout).
    config:
        Module config. Defaults to v1.1 contract (``master_seed`` =
        ``20260518``); v2.0 callers should pass
        ``SocTrajectoryConfig(master_seed=MASTER_SEED_V2_DEFAULT)``.
    profile_type:
        ``"residential"`` (v1.1 default) or ``"workplace"`` (v2.0).
    storage_tz:
        Output timezone for the sessions parquet. Defaults to the
        legacy LA tz; v2.0 callers should pass :data:`STORAGE_TZ_V2`.

    The output is byte-identical across re-runs with the same seed (the
    function is purely deterministic given its inputs).
    """
    cfg = config or SocTrajectoryConfig()

    if not paths.fleet_in.is_file():
        raise FileNotFoundError(f"fleet.parquet not found at {paths.fleet_in}")
    if not paths.mobility_in.is_file():
        raise FileNotFoundError(f"mobility.parquet not found at {paths.mobility_in}")
    if not paths.plug_in_events_in.is_file():
        raise FileNotFoundError(
            f"plug_in_events.parquet not found at {paths.plug_in_events_in}"
        )

    fleet = pd.read_parquet(paths.fleet_in)
    mobility = pd.read_parquet(paths.mobility_in)
    plug_in_events = pd.read_parquet(paths.plug_in_events_in)

    logger.info(
        "M6 inputs: fleet=%d, mobility=%d, plug_in_events=%d (%d plugged)",
        len(fleet),
        len(mobility),
        len(plug_in_events),
        int(plug_in_events["plug_in"].sum()),
    )

    # Workplace guard: keep only ``location == "work"`` plug-in events.
    if profile_type == "workplace" and "location" in plug_in_events.columns:
        loc_str = plug_in_events["location"].astype(str)
        plug_in_events = plug_in_events.loc[loc_str == LOC_WORK].copy()

    timelines = build_event_timeline(mobility, plug_in_events, tz=storage_tz)
    ev_ids = fleet["ev_id"].to_numpy()
    soc0_by_ev = compute_initial_soc(ev_ids, cfg.master_seed, cfg)
    sessions = iterate_soc_ledger(
        fleet,
        timelines,
        soc0_by_ev,
        cfg,
        profile_type=profile_type,
        storage_tz=storage_tz,
    )

    # Write parquet. Sort once more to guarantee deterministic row order
    # for byte-identical reruns.
    sessions = sessions.sort_values(["ev_id", "session_idx"], kind="mergesort").reset_index(
        drop=True
    )
    paths.sessions_out.parent.mkdir(parents=True, exist_ok=True)
    # row_group_size and compression are fixed for byte-identical writes.
    sessions.to_parquet(paths.sessions_out, index=False, compression="snappy")

    # Summary numbers.
    n_sessions = int(len(sessions))
    n_dcfc = (
        int((sessions["location"] == LOC_DCFC).sum())
        if profile_type == "residential"
        else 0
    )
    fleet_kwh_charged_grid = float(sessions["energy_kwh"].sum())
    fleet_kwh_driven = float(mobility["kwh_consumed"].sum())
    if n_sessions > 0:
        sess_per_ev = sessions.groupby("ev_id", observed=True).size()
        sess_med = float(sess_per_ev.median())
    else:
        sess_med = 0.0

    # Workplace-only: share of EVs with at least one
    # ``home_charging_required`` session.
    home_required_share: float | None = None
    n_home_required_sessions: int = 0
    if profile_type == "workplace" and "home_charging_required" in sessions.columns:
        n_home_required_sessions = int(sessions["home_charging_required"].sum())
        evs_flagged = sessions.loc[
            sessions["home_charging_required"], "ev_id"
        ].nunique()
        n_evs_total = int(fleet["ev_id"].nunique())
        home_required_share = (
            float(evs_flagged) / n_evs_total if n_evs_total > 0 else 0.0
        )

    _update_meta(
        paths.meta_out,
        n_sessions=n_sessions,
        n_dcfc_injections=n_dcfc,
        fleet_kwh_charged_grid=fleet_kwh_charged_grid,
        fleet_kwh_driven=fleet_kwh_driven,
        sessions_per_ev_median=sess_med,
        profile_type=profile_type,
        storage_tz=storage_tz,
        home_charging_required_share=home_required_share,
        n_home_charging_required_sessions=n_home_required_sessions,
    )

    return SocTrajectoryResult(
        sessions=sessions,
        n_sessions=n_sessions,
        n_dcfc_injections=n_dcfc,
        fleet_kwh_charged_grid=fleet_kwh_charged_grid,
        fleet_kwh_driven=fleet_kwh_driven,
        sessions_per_ev={"median": sess_med},
    )


# --------------------------------------------------------------------------- #
# CLI entry-point: `python -m pev_synth.soc_trajectory`.
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    """CLI: run the M6 pipeline against the default repo-root paths.

    Without ``--region`` the legacy v1.1 flat layout
    (``data/pev/processed/residential_ev_synth/``) is used. With
    ``--region`` (and optionally ``--profile-type``) the v2.0
    region-routed layout is used and the storage tz flips to UTC.
    """
    parser = argparse.ArgumentParser(
        prog="pev_synth.soc_trajectory",
        description="M6: SoC trajectory + per-session energy ledger.",
    )
    parser.add_argument("--seed", type=int, default=None)
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
            "v2.0 region name (e.g. 'bay_area'). When set, output paths "
            "follow data/pev/processed/<region>/<profile_type>_ev_synth/ "
            "and storage tz is UTC."
        ),
    )
    parser.add_argument(
        "--profile-type",
        type=str,
        default="residential",
        choices=["residential", "workplace"],
        help="v2.0 profile type. Ignored when --region is unset.",
    )
    args = parser.parse_args(argv)

    from ._paths import repo_root as _repo_root_fn  # noqa: PLC0415

    repo_root = (
        Path(args.repo_root).resolve() if args.repo_root else _repo_root_fn()
    )

    if args.region is not None:
        # v2.0 path: region-routed I/O, UTC storage tz, v2 seed default.
        # Late import to avoid circulars on package init.
        from .regions import Region as _Region  # noqa: PLC0415

        region = _Region.from_name(args.region)
        paths = resolve_v2_paths(repo_root, region, args.profile_type)
        seed_v = int(args.seed) if args.seed is not None else int(MASTER_SEED_V2_DEFAULT)
        cfg = SocTrajectoryConfig(master_seed=seed_v)
        summary = run(
            paths,
            config=cfg,
            profile_type=args.profile_type,
            storage_tz=STORAGE_TZ_V2,
        )
    else:
        # v1.1 legacy path.
        paths = _default_paths(repo_root)
        seed_v = int(args.seed) if args.seed is not None else int(MASTER_SEED_DEFAULT)
        cfg = SocTrajectoryConfig(master_seed=seed_v)
        summary = run(paths, config=cfg)

    print(
        "M6 soc_trajectory: n_sessions={n}, n_dcfc={d} ({pct:.2%}), "
        "fleet_kwh_charged={c:,.0f}, fleet_kwh_driven={dr:,.0f}, "
        "ratio={r:.3f}".format(
            n=summary.n_sessions,
            d=summary.n_dcfc_injections,
            pct=(summary.n_dcfc_injections / summary.n_sessions) if summary.n_sessions else 0.0,
            c=summary.fleet_kwh_charged_grid,
            dr=summary.fleet_kwh_driven,
            r=(
                summary.fleet_kwh_charged_grid / summary.fleet_kwh_driven
                if summary.fleet_kwh_driven > 0
                else float("nan")
            ),
        ),
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
