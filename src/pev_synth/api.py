"""Public API for the ``pev_synth`` package (v2.0).

This module wraps the v2.0 per-region cached pipeline outputs (under
``data/pev/processed/<region.name>/<profile_type>_ev_synth/``) in a small
library surface:

    >>> import pev_synth as ps
    >>> fleet = ps.generate_profiles('residential', n=1000, region='bay_area')
    >>> prof  = fleet[0]
    >>> pa    = prof.generate_presence_absence('2001-01-01', '2001-01-08')

Two profile types are registered in v2.0:

* ``residential`` — full M1-M7 residential pipeline.
* ``workplace``   — full M1-M7 workplace pipeline (Wave-2 modules).

The legacy ``fleet_depot`` profile type from v1.1 was hard-removed in v2.0
per plan §2.6 / Open Question 6 — ``generate_profiles('fleet_depot', ...)``
raises ``ValueError`` and ``list_profile_types()`` no longer includes it.

Time conventions (v2.0)
-----------------------
* Storage tz: every cached timestamp column is ``pyarrow.timestamp("ns",
  tz="UTC")`` (plan §3.1).
* Query tz: each time-window method accepts a ``tz`` keyword argument.
  ``tz=None`` (default) returns a UTC-indexed Series / DataFrame. Pass
  ``tz=region.tz`` (or any IANA zone) to receive the same data on a
  local-wall-clock index.
* User-supplied ``t_start`` / ``t_stop``: if tz-aware, used as-is (converted
  to UTC for the half-open ``[t0, t1)`` slice). If tz-naive, the timestamp
  is localised to ``region.tz`` first, then converted to UTC. This matches
  plan §3.3 and is backward-compatible with the v1.1 LA-implicit
  convention when the region is bay_area / la_basin (``America/Los_Angeles``).
* The cached data lives on the 2001 synthetic calendar. User-supplied
  timestamps in another year are re-mapped to 2001 by replacing only the
  year component (month/day/hour-of-day preserved). The remap happens
  on the local wall-clock before UTC conversion so the LT month-day-hour
  is preserved regardless of region.

Math notation: η for one-way charge efficiency, Δsoc for state-of-charge
changes, β / σ for distribution parameters. ASCII-only otherwise.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Literal,
    Protocol,
    runtime_checkable,
)

import numpy as np
import pandas as pd
from pytz.exceptions import AmbiguousTimeError, NonExistentTimeError

from ._paths import replicate_dir as _replicate_dir
from .regions import (
    _DESCOPED_PROFILE_TYPES,
    PROFILE_TYPES,
    Region,
    list_profile_types,
)

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------

_STORAGE_TZ = "UTC"
_CACHE_YEAR = 2001  # synthetic calendar year baked into the cache
_DEFAULT_SEED = 20260520  # v2.0 master seed
_KNOWN_TYPES: tuple[str, ...] = tuple(PROFILE_TYPES)

ProfileType = Literal["residential", "workplace"]


# ----------------------------------------------------------------------
# Helpers: profile-type validation, region resolution, tz / window helpers
# ----------------------------------------------------------------------

def _validate_profile_type(name: str) -> None:
    """Raise ``ValueError`` for descoped or unknown profile types.

    Mirrors :func:`pev_synth.regions._validate_profile_type` (the helper
    lives there too as the canonical registry; re-implementing here keeps
    the import surface minimal). Used by :func:`generate_profiles` and
    by any caller that needs the same error message.
    """
    if name in _DESCOPED_PROFILE_TYPES:
        raise ValueError(
            "fleet_depot was de-scoped in v2.0; see plan §2.6"
        )
    if name not in _KNOWN_TYPES:
        raise ValueError(
            f"unknown profile_type {name!r}; "
            f"valid: {list(_KNOWN_TYPES)}"
        )


def _resolve_region(region: Region | str) -> Region:
    """Coerce ``region`` (Region instance OR str name) to a Region.

    Raises ``ValueError`` (via ``Region.from_name``) if the string is not a
    registered region.
    """
    if isinstance(region, Region):
        return region
    if isinstance(region, str):
        return Region.from_name(region)
    raise TypeError(
        f"region must be a Region or str, got {type(region).__name__}"
    )


def _to_region_timestamp(
    ts: str | pd.Timestamp, region_tz: str
) -> pd.Timestamp:
    """Coerce ``ts`` to a tz-aware ``pd.Timestamp`` anchored in ``region_tz``.

    Tz-naive inputs are localised to ``region_tz``. Per RFC-002 the v1.1
    ``ambiguous="NaT"`` / ``nonexistent="shift_forward"`` behaviour silently
    let DST-fall-back wall-clocks become ``NaT``, which then made every
    ``>=`` comparison False and returned empty frames with no error. The
    v2.0 contract is: raise loudly with a clear message on either edge
    case so the caller can pick a side of the DST boundary.
    """
    out = pd.Timestamp(ts)
    if out.tzinfo is None:
        try:
            out = out.tz_localize(
                region_tz, ambiguous="raise", nonexistent="raise"
            )
        except AmbiguousTimeError as exc:
            raise ValueError(
                f"ambiguous wall-clock in DST fall-back window: "
                f"{ts!r} in tz {region_tz!r}"
            ) from exc
        except NonExistentTimeError as exc:
            raise ValueError(
                f"non-existent wall-clock in DST spring-forward window: "
                f"{ts!r} in tz {region_tz!r}"
            ) from exc
    else:
        out = out.tz_convert(region_tz)
    return out


def _remap_to_cache_year(
    ts: pd.Timestamp, cache_year: int = _CACHE_YEAR
) -> pd.Timestamp:
    """Replace the year component of ``ts`` with the cache year.

    Month, day, hour, minute, second, microsecond are preserved. February 29
    of a leap-year input is mapped to March 1 of a non-leap cache year (2001
    is non-leap) to avoid a localisation error.
    """
    try:
        return ts.replace(year=cache_year)
    except ValueError:
        # Feb 29 in a leap year -> Mar 1 in a non-leap cache year.
        return ts.replace(year=cache_year, month=3, day=1)


def _normalize_window(
    t_start: str | pd.Timestamp,
    t_stop: str | pd.Timestamp,
    region_tz: str = "America/Los_Angeles",
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Normalise ``(t_start, t_stop)`` for use against the UTC cache.

    Returned timestamps are tz-aware UTC and re-mapped to the cache year.

    Workflow per plan §3.3:

    1. Localise to ``region_tz`` (tz-naive input) or convert (tz-aware).
    2. Remap year to ``_CACHE_YEAR`` on the local wall-clock (so the LT
       month-day-hour-of-day is preserved across regions).
    3. ``tz_convert("UTC")`` so the slice against the cache's UTC-tagged
       parquet rows uses identical tz.
    """
    ts0_local = _to_region_timestamp(t_start, region_tz)
    ts1_local = _to_region_timestamp(t_stop, region_tz)
    if ts1_local <= ts0_local:
        raise ValueError(
            f"t_stop ({ts1_local}) must be strictly after t_start ({ts0_local})"
        )
    # RFC-003: remap each endpoint independently. The previous code derived
    # ``year_offset`` from ``ts0_local`` only, so a window crossing a
    # calendar boundary (e.g. 2001-12-31..2002-01-08) left ``ts1`` in 2002
    # outside the cache year — yielding silent empty/garbage results.
    pre_remap_weekday_0 = ts0_local.isoweekday()
    pre_remap_weekday_1 = ts1_local.isoweekday()
    remapped = False
    if ts0_local.year != _CACHE_YEAR:
        ts0_local = _remap_to_cache_year(ts0_local)
        remapped = True
    if ts1_local.year != _CACHE_YEAR:
        ts1_local = _remap_to_cache_year(ts1_local)
        remapped = True
    # M3: a window spanning the cache-year boundary (e.g.
    # ("2025-12-30", "2026-01-05")) remaps each endpoint into 2001
    # independently, which inverts the order. Catch it explicitly here so
    # the half-open slice does not silently return zero rows.
    if ts1_local <= ts0_local:
        raise ValueError(
            f"year-remap inverts the requested window: original endpoints "
            f"were {t_start!r}..{t_stop!r}; after remap to cache-year "
            f"{_CACHE_YEAR} the window collapses to "
            f"{ts0_local}..{ts1_local}. Split the window across the cache-year "
            "boundary (e.g. query [2025-12-30, 2026-01-01) and "
            "[2026-01-01, 2026-01-05) separately)."
        )
    # H3: the plug-in model is weekday-conditional, so a year-remap that
    # shifts the ISO weekday on either endpoint changes the
    # weekday/weekend mix relative to the original calendar week. Emit a
    # UserWarning once so the caller can snap to the same day-of-week
    # explicitly if it matters for their analysis.
    if remapped and (
        pre_remap_weekday_0 != ts0_local.isoweekday()
        or pre_remap_weekday_1 != ts1_local.isoweekday()
    ):
        warnings.warn(
            "year-remap to cache-year 2001 shifted day-of-week: "
            f"{t_start!r} mapped to {ts0_local.date()} ({ts0_local.day_name()}), "
            f"{t_stop!r} mapped to {ts1_local.date()} ({ts1_local.day_name()}). "
            "The plug-in model is weekday-conditional; weekday/weekend behavior "
            "may not match the original calendar week. Snap to the same day-of-week "
            "explicitly if this matters for your analysis.",
            UserWarning,
            stacklevel=3,
        )
    return (
        ts0_local.tz_convert(_STORAGE_TZ),
        ts1_local.tz_convert(_STORAGE_TZ),
    )


def _normalize_freq(freq: str) -> str:
    """Accept ``'15min'`` / ``'1h'`` (and case variants); return canonical."""
    f = freq.strip().lower()
    if f in ("15min", "15t", "15 min"):
        return "15min"
    if f in ("1h", "h", "60min"):
        return "1h"
    raise ValueError(f"Unsupported freq: {freq!r}. Use '15min' or '1h'.")


def _apply_query_tz(
    obj: pd.Series | pd.DataFrame, tz: str | None
) -> pd.Series | pd.DataFrame:
    """Return ``obj`` re-indexed in the requested query tz.

    ``tz=None`` keeps the storage tz (UTC). Any other value calls
    ``tz_convert`` on the DatetimeIndex.
    """
    if tz is None:
        return obj
    out = obj.copy()
    out.index = out.index.tz_convert(tz)
    return out


# ----------------------------------------------------------------------
# Profile
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class _ProfileRow:
    """Per-EV static row pulled from ``fleet.parquet``."""

    ev_id: int
    powertrain: str
    make: str
    model: str
    model_year: int
    archetype: str
    battery_kwh: float
    obc_kw: float
    evse_home_kw: float
    panel_cap_kw: float
    # v2.1 EVSE-enrichment fields (plan §A.3.1). Descriptive only -- they
    # do NOT alter the ``max_charge_kw`` bottleneck semantics.
    evse_brand: str
    evse_connector: str
    eta_charge: float
    has_heat_pump: bool
    wh_per_mi_nominal: float
    garage_type: str
    income_tier: str
    home_station: str
    seed: int
    soc_min_kwh: float
    donor_household_id: str
    donor_vehicle_id: str
    cluster_z: int

    @property
    def max_charge_kw(self) -> float:
        """min(on-board-charger kW, EVSE home kW). Panel cap is not applied
        because residential EVSEs are typically sized below panel cap.

        v2.1 note: ``evse_brand`` and ``evse_connector`` are descriptive
        only — they do NOT derate this quantity. The optimizer-input
        power remains ``min(OBC_kw, EVSE_home_kw, panel_cap_kw)``."""
        return float(min(self.obc_kw, self.evse_home_kw))


class Profile:
    """One synthetic EV. All data is loaded lazily from the cache.

    Public attributes / properties below cover the brief's API surface. The
    methods consult ``fleet.parquet`` (static), ``plug_status_15min.parquet``
    (presence/absence), ``sessions.parquet`` (charging sessions),
    ``mobility.parquet`` (trips). Filters are pushed down to pyarrow so we
    never load the full 35M-row table.

    All time-window methods accept a ``tz`` keyword: ``tz=None`` returns the
    UTC storage tz; pass ``region.tz`` (or any IANA zone) for a local
    wall-clock index instead.
    """

    def __init__(self, fleet: Fleet, ev_id: int) -> None:
        self._fleet = fleet
        self._ev_id = int(ev_id)
        self._row_cache: _ProfileRow | None = None

    # -- representation ------------------------------------------------- #

    def __repr__(self) -> str:
        return (
            f"Profile(ev_id={self._ev_id}, archetype={self.archetype!r}, "
            f"battery_kwh={self.battery_kwh:.1f}, "
            f"profile_type={self._fleet.profile_type!r}, "
            f"region={self._fleet.region.name!r})"
        )

    # -- static attributes --------------------------------------------- #

    @property
    def _row(self) -> _ProfileRow:
        if self._row_cache is None:
            self._row_cache = self._fleet._row_for(self._ev_id)
        return self._row_cache

    @property
    def ev_id(self) -> int:
        return self._ev_id

    @property
    def archetype(self) -> str:
        return self._row.archetype

    @property
    def battery_kwh(self) -> float:
        return self._row.battery_kwh

    @property
    def max_charge_kw(self) -> float:
        return self._row.max_charge_kw

    @property
    def eta_charge(self) -> float:
        return self._row.eta_charge

    @property
    def soc_min_kwh(self) -> float:
        return self._row.soc_min_kwh

    @property
    def cluster_z(self) -> int:
        return int(self._row.cluster_z)

    @property
    def donor_household_id(self) -> str:
        return str(self._row.donor_household_id)

    @property
    def donor_vehicle_id(self) -> str:
        return str(self._row.donor_vehicle_id)

    @property
    def profile_type(self) -> str:
        return self._fleet.profile_type

    @property
    def region(self) -> Region:
        return self._fleet.region

    # -- v2.1 EVSE-enrichment attributes ------------------------------ #

    @property
    def evse_brand(self) -> str:
        """EVSE brand string (e.g. ``'Tesla'``, ``'ChargePoint'``, ...).

        See ``docs/literature_review_evse_specs_us.md`` for the brand
        universe and prior sources.
        """
        return str(self._row.evse_brand)

    @property
    def evse_connector(self) -> str:
        """EVSE connector type (e.g. ``'J1772'``, ``'NACS'``,
        ``'Tesla_NACS_native'``, ``'NEMA_14_50'``, ...).
        """
        return str(self._row.evse_connector)

    # -- summary -------------------------------------------------------- #

    def summary(self) -> pd.Series:
        """Per-EV summary Series (one row of the fleet table)."""
        r = self._row
        return pd.Series(
            {
                "ev_id": r.ev_id,
                "profile_type": self.profile_type,
                "region": self._fleet.region.name,
                "archetype": r.archetype,
                "powertrain": r.powertrain,
                "make": r.make,
                "model": r.model,
                "model_year": r.model_year,
                "battery_kwh": r.battery_kwh,
                "max_charge_kw": r.max_charge_kw,
                "obc_kw": r.obc_kw,
                "evse_home_kw": r.evse_home_kw,
                "panel_cap_kw": r.panel_cap_kw,
                # v2.1 EVSE enrichment.
                "evse_brand": r.evse_brand,
                "evse_connector": r.evse_connector,
                "eta_charge": r.eta_charge,
                "soc_min_kwh": r.soc_min_kwh,
                "has_heat_pump": r.has_heat_pump,
                "wh_per_mi_nominal": r.wh_per_mi_nominal,
                "garage_type": r.garage_type,
                "income_tier": r.income_tier,
                "cluster_z": r.cluster_z,
                "donor_household_id": r.donor_household_id,
                "donor_vehicle_id": r.donor_vehicle_id,
            },
            name=f"ev_{r.ev_id:0{self._fleet._ev_id_pad}d}",
        )

    # -- dynamic queries ----------------------------------------------- #

    def generate_presence_absence(
        self,
        t_start: str | pd.Timestamp,
        t_stop: str | pd.Timestamp,
        freq: str = "15min",
        tz: str | None = None,
    ) -> pd.Series:
        """Bool plug-in series over ``[t_start, t_stop)`` at ``freq``.

        Returns a tz-aware ``pd.Series[bool]`` indexed by ``DatetimeIndex``.
        ``True`` means the EV is plugged-in at that timestep.

        Parameters
        ----------
        t_start, t_stop:
            Window endpoints. Tz-naive inputs are localised to
            ``self.region.tz``. Year is remapped to the cache year (2001).
        freq:
            ``'15min'`` (default) or ``'1h'``.
        tz:
            ``None`` (default) → UTC storage-tz index.
            Any IANA zone (e.g. ``self.region.tz``) → that-zone index.
            The cell values are identical; only the index labels change.
        """
        return self._fleet._presence_absence_one(
            self._ev_id, t_start, t_stop, freq, tz=tz
        )

    # User-facing alias for users who prefer the literature's term.
    def plug_status(
        self,
        t_start: str | pd.Timestamp,
        t_stop: str | pd.Timestamp,
        freq: str = "15min",
        tz: str | None = None,
    ) -> pd.Series:
        """Alias of :meth:`generate_presence_absence`."""
        return self.generate_presence_absence(t_start, t_stop, freq, tz=tz)

    def charging_sessions(
        self,
        t_start: str | pd.Timestamp,
        t_stop: str | pd.Timestamp,
        tz: str | None = None,
    ) -> pd.DataFrame:
        """Sessions whose ``[t_in, t_out)`` intersects the window."""
        return self._fleet._charging_sessions(
            [self._ev_id], t_start, t_stop, tz=tz
        )

    def soc_trajectory(
        self,
        t_start: str | pd.Timestamp,
        t_stop: str | pd.Timestamp,
        freq: str = "15min",
        tz: str | None = None,
    ) -> pd.Series:
        """Continuous SoC over ``[t_start, t_stop)`` in kWh.

        Inside a session, SoC ramps linearly from ``soc_in_kwh`` to
        ``soc_target_out_kwh`` over the session window (constant-power
        proxy consistent with the v1 ledger). Between sessions, SoC steps
        linearly from the previous ``soc_target_out`` down to the next
        ``soc_in`` (driving discharge).
        """
        return self._fleet._soc_trajectory_one(
            self._ev_id, t_start, t_stop, freq, tz=tz
        )

    def trips(
        self,
        t_start: str | pd.Timestamp,
        t_stop: str | pd.Timestamp,
        tz: str | None = None,
    ) -> pd.DataFrame:
        """Trips from ``mobility.parquet`` overlapping the window.

        Timestamp columns (``start_ts``, ``end_ts``) are returned in the
        requested ``tz`` (UTC by default).
        """
        return self._fleet._trips(
            [self._ev_id], t_start, t_stop, tz=tz
        )


# ----------------------------------------------------------------------
# FleetReader protocol (RFC-020)
# ----------------------------------------------------------------------


@runtime_checkable
class FleetReader(Protocol):
    """Public read-only surface of :class:`Fleet` consumed by the optimizer.

    RFC-020: the ``pev_optim`` adapter previously reached into
    ``fleet._ev_ids``, ``fleet._pos_by_evid``, ``fleet._row_for``,
    ``fleet._charging_sessions``, ``fleet._presence_absence_one`` and
    ``fleet._read_parquet`` — six private attributes. This protocol
    promotes the same calls to a typed public contract that the adapter
    depends on instead.
    """

    def ev_ids(self) -> list[int]: ...
    def has_ev(self, ev_id: int) -> bool: ...
    def row_for(self, ev_id: int) -> _ProfileRow: ...
    def charging_sessions_for(
        self,
        ev_ids: list[int],
        t_start: str | pd.Timestamp,
        t_stop: str | pd.Timestamp,
        tz: str | None = ...,
    ) -> pd.DataFrame: ...
    def presence_absence_one(
        self,
        ev_id: int,
        t_start: str | pd.Timestamp,
        t_stop: str | pd.Timestamp,
        freq: str = ...,
        tz: str | None = ...,
    ) -> pd.Series: ...
    def read_parquet(
        self,
        name: str,
        columns: list[str] | None = ...,
        ev_ids: list[int] | None = ...,
    ) -> pd.DataFrame: ...


# ----------------------------------------------------------------------
# aggregate-load kernel (module-level so it's testable in isolation)
# ----------------------------------------------------------------------

def _aggregate_kw_from_sessions(
    a_ns: np.ndarray,
    b_ns: np.ndarray,
    p_kw: np.ndarray,
    ts0_ns: int,
    step_ns: int,
    n_buckets: int,
) -> np.ndarray:
    """kW load per bucket given session timings + powers. Vectorised.

    Each session contributes ``p_kw[s] * (overlap_ns / step_ns)`` to each
    bucket it touches, where ``overlap_ns`` is the intersection in
    nanoseconds between the session window ``[a_ns[s], b_ns[s])`` and the
    bucket window ``[ts0_ns + j*step_ns, ts0_ns + (j+1)*step_ns)``.
    Numerically identical to the per-session inner loop modulo IEEE
    summation order (``np.add.at`` preserves session-wise add ordering).
    """
    load_kw = np.zeros(n_buckets, dtype=float)
    if a_ns.size == 0:
        return load_kw

    # Per-session first / last bucket index (clipped to the valid range).
    i0 = np.clip((a_ns - ts0_ns) // step_ns, 0, n_buckets).astype(np.int64)
    i1 = np.clip(
        (b_ns - 1 - ts0_ns) // step_ns + 1, 0, n_buckets
    ).astype(np.int64)
    counts = i1 - i0  # buckets touched per session (non-negative)
    total = int(counts.sum())
    if total == 0:
        return load_kw

    # Build flat (session_idx, bucket_idx) for every (session, bucket)
    # pair. ``np.repeat`` duplicates each session index by its bucket
    # count; ``arange - per-session offset`` gives the bucket index
    # within the session's range, which we shift by ``i0`` to get the
    # absolute bucket.
    session_idx = np.repeat(np.arange(a_ns.size), counts)
    cum_counts = np.concatenate(([0], counts.cumsum()))
    bucket_idx = (
        np.repeat(i0, counts)
        + np.arange(total)
        - np.repeat(cum_counts[:-1], counts)
    )

    bucket_lo = ts0_ns + bucket_idx * step_ns
    bucket_hi = bucket_lo + step_ns
    overlap_ns = (
        np.minimum(b_ns[session_idx], bucket_hi)
        - np.maximum(a_ns[session_idx], bucket_lo)
    )
    # overlap_ns >= 0 by the i0/i1 construction; clip defensively.
    overlap_ns = np.maximum(overlap_ns, 0)
    contrib = p_kw[session_idx] * (overlap_ns / step_ns)
    np.add.at(load_kw, bucket_idx, contrib)
    return load_kw


# ----------------------------------------------------------------------
# Fleet
# ----------------------------------------------------------------------

class Fleet:
    """A collection of :class:`Profile` instances backed by parquet files.

    ``Fleet`` is lazy: only ``fleet.parquet`` is read at construction time;
    bigger artifacts (plug_status, sessions, mobility) are read on demand
    via pyarrow predicate-pushdown by ``ev_id``.

    Attributes
    ----------
    region:
        The :class:`Region` this Fleet was generated for. The region's
        ``tz`` is the default local zone for tz-naive user input
        timestamps. Cache timestamps are always stored in UTC.
    profile_type:
        ``'residential'`` or ``'workplace'``.
    """

    # ---------------- construction ---------------- #

    def __init__(
        self,
        data_root: Path,
        ev_ids: list[int],
        profile_type: str,
        region: Region,
        meta: dict[str, Any] | None = None,
    ) -> None:
        self._data_root = Path(data_root)
        self._ev_ids: list[int] = sorted(int(e) for e in ev_ids)
        self._profile_type: str = str(profile_type)
        self._region: Region = region
        self._meta: dict[str, Any] = dict(meta or {})

        fleet_pq = self._data_root / "fleet.parquet"
        if not fleet_pq.exists():
            raise FileNotFoundError(
                f"fleet.parquet not found at {fleet_pq}. Did you point "
                "data_root at a valid processed bundle?"
            )
        full = pd.read_parquet(fleet_pq)
        # RFC-019: assert uniqueness once; ``.loc[self._ev_ids]`` silently
        # *expands* the frame when ev_id is duplicated, masking a corrupt
        # parquet as "merely large".
        if full["ev_id"].duplicated().any():
            dup_ct = int(full["ev_id"].duplicated().sum())
            raise ValueError(
                f"fleet.parquet at {fleet_pq} has {dup_ct} duplicated ev_id "
                "rows; the primary key must be unique."
            )
        sub = full.loc[full["ev_id"].isin(self._ev_ids)].copy()
        sub = (
            sub.set_index("ev_id", drop=False)
            .loc[self._ev_ids]
            .reset_index(drop=True)
        )
        self._fleet_df: pd.DataFrame = sub
        # Position lookup for __getitem__ by int and by ev_id string.
        self._pos_by_evid: dict[int, int] = {
            int(ev): i
            for i, ev in enumerate(self._fleet_df["ev_id"].tolist())
        }
        # L3: derive the ev_id zero-pad width from the largest ev_id in
        # the Fleet rather than hard-coding ``:04d``. Floor at 4 to
        # preserve the v1.1 ``ev_0042`` aesthetic on small fleets.
        max_id = max(self._ev_ids) if self._ev_ids else 0
        self._ev_id_pad: int = max(4, len(str(max_id)))

        # H6: surface the workplace-cohort caveat (105-vehicle EVWatts
        # cohort, plug-in median ~12:00 LT vs literature-canonical
        # ~09:00) at construction time so a downstream user cannot
        # silently mis-interpret the v2.0 workplace pipeline.
        if self._profile_type == "workplace":
            warnings.warn(
                "Workplace fleets in v2.0 are fit from the 105-vehicle public "
                "EVWatts cohort; plug-in median ~12:00 LT is ~3h later than "
                "the literature-canonical workplace median of ~09:00. "
                "Validator W1-W4 flag this divergence as EXPLAINED_FAIL. "
                "See pev_synth_api.md 'Workplace caveat' for context.",
                RuntimeWarning,
                stacklevel=2,
            )

    # ---------------- dunder ---------------- #

    def __len__(self) -> int:
        return len(self._ev_ids)

    def __iter__(self) -> Iterator[Profile]:
        for ev in self._ev_ids:
            yield Profile(self, ev)

    def __getitem__(self, key: int | str) -> Profile:
        """``fleet[42]`` (positional or ev_id int) or ``fleet['ev_0042']``.

        For unambiguous access prefer :meth:`by_ev_id` (by-id) or
        :meth:`by_position` (positional). ``__getitem__`` falls through
        from by-id to positional, which can flip semantics after
        ``filter(...)``.
        """
        if isinstance(key, str):
            if not key.startswith("ev_"):
                raise KeyError(
                    f"string keys must look like 'ev_0042', got {key!r}"
                )
            try:
                ev = int(key.removeprefix("ev_"))
            except ValueError as e:
                raise KeyError(f"unparseable ev_id string: {key!r}") from e
            if ev not in self._pos_by_evid:
                raise KeyError(f"ev_id {ev} not in this Fleet")
            return Profile(self, ev)
        if isinstance(key, bool):
            raise TypeError(
                f"Fleet keys must be int or str, not bool (got {key!r})"
            )
        if isinstance(key, (int, np.integer)):
            ev = int(key)
            # Prefer "by ev_id" semantics when the int IS one of the ids;
            # otherwise treat as positional. This makes both fleet[0] and
            # fleet[42] do something useful for a subset of size N.
            if ev in self._pos_by_evid:
                return Profile(self, ev)
            if 0 <= ev < len(self._ev_ids):
                return Profile(self, self._ev_ids[ev])
            raise IndexError(f"index out of range: {ev}")
        raise TypeError(
            f"Fleet keys must be int or str, got {type(key).__name__}"
        )

    def by_ev_id(self, ev_id: int) -> Profile:
        """Return the Profile for the given ev_id.

        Raises ``KeyError`` if ``ev_id`` is not in this Fleet. Use this
        when you want unambiguous by-id lookup; ``fleet[ev_id]`` works the
        same way only when ``ev_id`` is present.
        """
        if isinstance(ev_id, bool):
            raise TypeError(
                f"ev_id must be int, not bool (got {ev_id!r})"
            )
        if not isinstance(ev_id, (int, np.integer)):
            raise TypeError(
                f"ev_id must be int, got {type(ev_id).__name__}"
            )
        ev = int(ev_id)
        if ev not in self._pos_by_evid:
            raise KeyError(f"ev_id {ev} not in this Fleet")
        return Profile(self, ev)

    def by_position(self, i: int) -> Profile:
        """Return the i-th Profile in this Fleet (0-indexed positional).

        Raises ``IndexError`` if ``i`` is out of range. Use this when you
        want unambiguous positional access; ``fleet[i]`` works the same way
        only when ``i`` is NOT also a valid ev_id in this Fleet.
        """
        if isinstance(i, bool):
            raise TypeError(
                f"position must be int, not bool (got {i!r})"
            )
        if not isinstance(i, (int, np.integer)):
            raise TypeError(
                f"position must be int, got {type(i).__name__}"
            )
        pos = int(i)
        if not (0 <= pos < len(self._ev_ids)):
            raise IndexError(
                f"position out of range: {pos} (Fleet has {len(self._ev_ids)} EVs)"
            )
        return Profile(self, self._ev_ids[pos])

    def __repr__(self) -> str:
        return (
            f"Fleet(profile_type={self._profile_type!r}, "
            f"region={self._region.name!r}, n={len(self)}, "
            f"data_root={self._data_root!s})"
        )

    # ---------------- public properties ---------------- #

    @property
    def profile_type(self) -> str:
        return self._profile_type

    @property
    def region(self) -> Region:
        return self._region

    @property
    def data_root(self) -> Path:
        return self._data_root

    @property
    def meta(self) -> dict[str, Any]:
        """Read-only view of the bundle's meta.json (best effort)."""
        return dict(self._meta)

    @staticmethod
    def cache_path(
        region: Region | str,
        profile_type: str,
        *,
        replicate_id: int = 0,
        r_total: int = 1,
    ) -> Path:
        """Resolve the canonical cache directory for one replicate.

        RFC-022.r: when ``r_total > 1`` the returned path is
        ``<processed_root>/<region>/<profile_type>_ev_synth/replicates/r{replicate_id}/``.
        ``r_total == 1`` stays flat (alias ``r=0``) so legacy single-cache
        layouts continue to resolve.
        """
        region_obj = _resolve_region(region)
        return _replicate_dir(
            region_obj.name, str(profile_type), int(replicate_id), int(r_total)
        )

    # ---------------- public methods ---------------- #

    def profile_ids(self) -> list[int]:
        """List of ``ev_id`` ints in this Fleet."""
        return list(self._ev_ids)

    # ---- RFC-020 FleetReader public surface --------------------------

    def ev_ids(self) -> list[int]:
        """Public alias of :meth:`profile_ids` for :class:`FleetReader`."""
        return list(self._ev_ids)

    def has_ev(self, ev_id: int) -> bool:
        """Whether ``ev_id`` is present in this Fleet (public position lookup)."""
        return int(ev_id) in self._pos_by_evid

    def row_for(self, ev_id: int) -> _ProfileRow:
        """Public alias of the per-EV static row look-up."""
        return self._row_for(int(ev_id))

    def charging_sessions_for(
        self,
        ev_ids: list[int],
        t_start: str | pd.Timestamp,
        t_stop: str | pd.Timestamp,
        tz: str | None = None,
    ) -> pd.DataFrame:
        """Public alias of the per-EV-list session query."""
        return self._charging_sessions(list(ev_ids), t_start, t_stop, tz=tz)

    def presence_absence_one(
        self,
        ev_id: int,
        t_start: str | pd.Timestamp,
        t_stop: str | pd.Timestamp,
        freq: str = "15min",
        tz: str | None = None,
    ) -> pd.Series:
        """Public alias of :meth:`_presence_absence_one`."""
        return self._presence_absence_one(int(ev_id), t_start, t_stop, freq, tz=tz)

    def read_parquet(
        self,
        name: str,
        columns: list[str] | None = None,
        ev_ids: list[int] | None = None,
    ) -> pd.DataFrame:
        """Public alias of the per-EV-id parquet reader."""
        return self._read_parquet(name, columns=columns, ev_ids=ev_ids)

    def summary(self) -> pd.DataFrame:
        """One row per EV. Columns include the brief's requested set."""
        df = self._fleet_df.copy()
        # Compute the derived columns.
        df["max_charge_kw"] = np.minimum(
            df["OBC_kw"], df["EVSE_home_kw"]
        ).astype(float)
        # Per-EV mileage & session counts (from sessions + mobility).
        sess = self._read_parquet(
            "sessions.parquet",
            columns=["ev_id", "t_in"],
            ev_ids=self._ev_ids,
        )
        sess_count = sess.groupby("ev_id").size().rename("n_sessions_yr")
        # v2.0 caches always include the ``plug_in`` column; the v1.1
        # fallback was removed per the fail-loud policy.
        plug_events = self._read_parquet(
            "plug_in_events.parquet",
            columns=["ev_id", "plug_in"],
            ev_ids=self._ev_ids,
        )
        plug_rate = (
            plug_events.groupby("ev_id")["plug_in"]
            .mean()
            .rename("plug_in_rate_yr")
        )
        miles = self._read_parquet(
            "mobility.parquet",
            columns=["ev_id", "miles"],
            ev_ids=self._ev_ids,
        )
        miles_yr = miles.groupby("ev_id")["miles"].sum().rename("miles_yr")

        out = pd.DataFrame(
            {
                "ev_id": df["ev_id"].astype(int).values,
                "battery_kwh": df["B_kwh"].astype(float).values,
                "max_charge_kw": df["max_charge_kw"].astype(float).values,
                "eta_charge": df["eta_charge_nominal"].astype(float).values,
                "archetype": df["archetype"].astype(str).values,
                "cluster_z": df["cluster_z"].astype(int).values,
                # v2.1 EVSE enrichment -- cast to str for parity with the
                # ``archetype`` column above (Fleet.summary semantics).
                "evse_brand": df["EVSE_brand"].astype(str).values,
                "evse_connector": df["EVSE_connector"].astype(str).values,
            }
        )
        out = out.set_index("ev_id", drop=False)
        out["plug_in_rate_yr"] = (
            plug_rate.reindex(out.index).astype(float).fillna(0.0).values
        )
        out["n_sessions_yr"] = (
            sess_count.reindex(out.index)
            .astype("Int64")
            .fillna(0)
            .astype(int)
            .values
        )
        out["miles_yr"] = (
            miles_yr.reindex(out.index).astype(float).fillna(0.0).values
        )
        return out.reset_index(drop=True)

    def filter(self, **predicates: Any) -> Fleet:
        """Return a new ``Fleet`` containing EVs matching the predicates.

        Supported keys
        --------------
        * ``archetype``: equality (string).
        * ``powertrain``: equality (string, e.g. ``'BEV'``).
        * ``cluster_z``: equality (int).
        * ``evse_brand``: equality on the v2.1 ``EVSE_brand`` column
          (e.g. ``'Tesla'``, ``'ChargePoint'``).
        * ``evse_connector``: equality on the v2.1 ``EVSE_connector``
          column (e.g. ``'J1772'``, ``'NACS'``).
        * ``battery_kwh_gte`` / ``battery_kwh_lt`` / ``battery_kwh_gt`` /
          ``battery_kwh_lte``: numeric thresholds on ``B_kwh``.
        * ``max_charge_kw_gte`` / ``max_charge_kw_lt`` / ...

        Multiple predicates are AND-combined. Unknown keys raise
        ``KeyError`` with a list of supported keys.
        """
        df = self._fleet_df.copy()
        df["max_charge_kw"] = np.minimum(
            df["OBC_kw"], df["EVSE_home_kw"]
        ).astype(float)

        supported: set[str] = {
            "archetype",
            "powertrain",
            "cluster_z",
            # v2.1 — EVSE enrichment equality predicates.
            "evse_brand",
            "evse_connector",
            "battery_kwh_gte",
            "battery_kwh_gt",
            "battery_kwh_lte",
            "battery_kwh_lt",
            "max_charge_kw_gte",
            "max_charge_kw_gt",
            "max_charge_kw_lte",
            "max_charge_kw_lt",
        }
        for key in predicates:
            if key not in supported:
                raise KeyError(
                    f"unsupported filter key {key!r}. "
                    f"Supported: {sorted(supported)}"
                )

        mask = pd.Series(True, index=df.index)
        if "archetype" in predicates:
            mask &= df["archetype"].astype(str) == str(predicates["archetype"])
        if "powertrain" in predicates:
            mask &= df["powertrain"].astype(str) == str(
                predicates["powertrain"]
            )
        if "cluster_z" in predicates:
            mask &= df["cluster_z"].astype(int) == int(
                predicates["cluster_z"]
            )
        if "evse_brand" in predicates:
            mask &= df["EVSE_brand"].astype(str) == str(predicates["evse_brand"])
        if "evse_connector" in predicates:
            mask &= df["EVSE_connector"].astype(str) == str(
                predicates["evse_connector"]
            )

        for col, src in (
            ("battery_kwh", "B_kwh"),
            ("max_charge_kw", "max_charge_kw"),
        ):
            v = df[src].astype(float)
            if f"{col}_gte" in predicates:
                mask &= v >= float(predicates[f"{col}_gte"])
            if f"{col}_gt" in predicates:
                mask &= v > float(predicates[f"{col}_gt"])
            if f"{col}_lte" in predicates:
                mask &= v <= float(predicates[f"{col}_lte"])
            if f"{col}_lt" in predicates:
                mask &= v < float(predicates[f"{col}_lt"])

        kept = df.loc[mask, "ev_id"].astype(int).tolist()
        return Fleet(
            data_root=self._data_root,
            ev_ids=kept,
            profile_type=self._profile_type,
            region=self._region,
            meta=self._meta,
        )

    # ---------------- presence / absence ---------------- #

    def generate_presence_absence(
        self,
        t_start: str | pd.Timestamp,
        t_stop: str | pd.Timestamp,
        freq: str = "15min",
        tz: str | None = None,
    ) -> pd.DataFrame:
        """Bool wide matrix: index=timestamps, columns=ev_id.

        See :meth:`Profile.generate_presence_absence` for the time-window
        and year-remap rules. ``tz=None`` (default) yields a UTC index;
        any IANA zone yields the same data on a local-wall-clock index.
        """
        freq_n = _normalize_freq(freq)
        ts0, ts1 = _normalize_window(t_start, t_stop, self._region.tz)

        # Choose the right rasterisation.
        if freq_n == "15min":
            parquet = "plug_status_15min.parquet"
        else:
            parquet = "plug_status_hourly.parquet"

        df = self._read_parquet(
            parquet,
            columns=["ev_id", "ts", "plugged"],
            ev_ids=self._ev_ids,
        )
        # Time window (half-open).
        df = df.loc[(df["ts"] >= ts0) & (df["ts"] < ts1)]
        target_idx = self._target_index(ts0, ts1, freq_n)
        if df.empty:
            wide = pd.DataFrame(
                False,
                index=target_idx,
                columns=[int(e) for e in self._ev_ids],
            )
        else:
            wide = df.pivot(
                index="ts", columns="ev_id", values="plugged"
            ).astype(bool)
            wide = wide.reindex(
                index=target_idx, columns=self._ev_ids, fill_value=False
            )
        wide.index.name = "ts"
        wide.columns.name = "ev_id"
        return _apply_query_tz(wide, tz)

    def plug_status(
        self,
        t_start: str | pd.Timestamp,
        t_stop: str | pd.Timestamp,
        freq: str = "15min",
        tz: str | None = None,
    ) -> pd.DataFrame:
        """Alias of :meth:`generate_presence_absence`."""
        return self.generate_presence_absence(t_start, t_stop, freq, tz=tz)

    def _presence_absence_one(
        self,
        ev_id: int,
        t_start: str | pd.Timestamp,
        t_stop: str | pd.Timestamp,
        freq: str,
        tz: str | None = None,
    ) -> pd.Series:
        freq_n = _normalize_freq(freq)
        ts0, ts1 = _normalize_window(t_start, t_stop, self._region.tz)
        parquet = (
            "plug_status_15min.parquet"
            if freq_n == "15min"
            else "plug_status_hourly.parquet"
        )
        df = self._read_parquet(
            parquet,
            columns=["ev_id", "ts", "plugged"],
            ev_ids=[ev_id],
        )
        df = df.loc[(df["ts"] >= ts0) & (df["ts"] < ts1)]
        target_idx = self._target_index(ts0, ts1, freq_n)
        s = pd.Series(
            False,
            index=target_idx,
            dtype=bool,
            name=f"ev_{ev_id:0{self._ev_id_pad}d}",
        )
        if not df.empty:
            ts_idx = pd.DatetimeIndex(df["ts"]).tz_convert(_STORAGE_TZ)
            s.loc[ts_idx] = df["plugged"].astype(bool).values
        s.index.name = "ts"
        return _apply_query_tz(s, tz)

    # ---------------- charging sessions ---------------- #

    def charging_sessions(
        self,
        t_start: str | pd.Timestamp,
        t_stop: str | pd.Timestamp,
        tz: str | None = None,
    ) -> pd.DataFrame:
        """Sessions whose ``[t_in, t_out)`` intersects the window.

        Timestamp columns (``t_in``, ``t_out``) are returned in ``tz``
        (UTC by default).
        """
        return self._charging_sessions(self._ev_ids, t_start, t_stop, tz=tz)

    def _charging_sessions(
        self,
        ev_ids: list[int],
        t_start: str | pd.Timestamp,
        t_stop: str | pd.Timestamp,
        tz: str | None = None,
    ) -> pd.DataFrame:
        ts0, ts1 = _normalize_window(t_start, t_stop, self._region.tz)
        df = self._read_parquet(
            "sessions.parquet",
            columns=None,
            ev_ids=ev_ids,
        )
        # half-open intersection: [t_in, t_out) overlaps [ts0, ts1)
        mask = (df["t_out"] > ts0) & (df["t_in"] < ts1)
        out = df.loc[mask].reset_index(drop=True)
        if tz is not None and not out.empty:
            for col in ("t_in", "t_out"):
                if col in out.columns and pd.api.types.is_datetime64_any_dtype(
                    out[col]
                ):
                    out[col] = out[col].dt.tz_convert(tz)
        return out

    # ---------------- trips ---------------- #

    def _trips(
        self,
        ev_ids: list[int],
        t_start: str | pd.Timestamp,
        t_stop: str | pd.Timestamp,
        tz: str | None = None,
    ) -> pd.DataFrame:
        ts0, ts1 = _normalize_window(t_start, t_stop, self._region.tz)
        df = self._read_parquet(
            "mobility.parquet", columns=None, ev_ids=ev_ids
        )
        mask = (df["end_ts"] > ts0) & (df["start_ts"] < ts1)
        out = df.loc[mask].reset_index(drop=True)
        if tz is not None and not out.empty:
            for col in ("start_ts", "end_ts"):
                if col in out.columns and pd.api.types.is_datetime64_any_dtype(
                    out[col]
                ):
                    out[col] = out[col].dt.tz_convert(tz)
        return out

    # ---------------- soc trajectory ---------------- #

    def _soc_trajectory_one(
        self,
        ev_id: int,
        t_start: str | pd.Timestamp,
        t_stop: str | pd.Timestamp,
        freq: str,
        tz: str | None = None,
    ) -> pd.Series:
        """Continuous SoC for one EV. See :meth:`Profile.soc_trajectory`."""
        freq_n = _normalize_freq(freq)
        ts0, ts1 = _normalize_window(t_start, t_stop, self._region.tz)
        idx = self._target_index(ts0, ts1, freq_n)

        # Pull all sessions for this EV (we need pre/post anchors).
        sess = (
            self._read_parquet(
                "sessions.parquet",
                columns=[
                    "ev_id",
                    "t_in",
                    "t_out",
                    "soc_in_kwh",
                    "soc_target_out_kwh",
                ],
                ev_ids=[ev_id],
            )
            .sort_values("t_in")
            .reset_index(drop=True)
        )

        # If no sessions, default to soc_min for the whole window.
        row = self._row_for(ev_id)
        if sess.empty:
            s = pd.Series(
                row.soc_min_kwh,
                index=idx,
                dtype=float,
                name=f"soc_kwh_{ev_id:0{self._ev_id_pad}d}",
            )
            return _apply_query_tz(s, tz)

        # RFC-015: vectorise the per-session anchor interleave via numpy
        # slicing. Replaces a ``sess.iterrows()`` loop that was O(N) Python
        # per session in a user-facing hot path.
        t_in_ns = (
            sess["t_in"].dt.tz_convert("UTC")
            .dt.tz_localize(None)
            .astype("datetime64[ns]").astype("int64").values
        )
        t_out_ns = (
            sess["t_out"].dt.tz_convert("UTC")
            .dt.tz_localize(None)
            .astype("datetime64[ns]").astype("int64").values
        )
        soc_in = sess["soc_in_kwh"].astype(float).values
        soc_out = sess["soc_target_out_kwh"].astype(float).values
        n_sess = len(sess)
        x_anchor = np.empty(2 * n_sess, dtype="int64")
        y_anchor = np.empty(2 * n_sess, dtype=float)
        x_anchor[0::2] = t_in_ns
        x_anchor[1::2] = t_out_ns
        y_anchor[0::2] = soc_in
        y_anchor[1::2] = soc_out
        x_anchor = x_anchor.astype(float)
        order = np.argsort(x_anchor, kind="stable")
        x_anchor = x_anchor[order]
        y_anchor = y_anchor[order]

        x_query = np.array([t.value for t in idx], dtype="int64").astype(float)
        soc = np.interp(x_query, x_anchor, y_anchor)
        s = pd.Series(
            soc,
            index=idx,
            dtype=float,
            name=f"soc_kwh_{ev_id:0{self._ev_id_pad}d}",
        )
        return _apply_query_tz(s, tz)

    # ---------------- aggregate load ---------------- #

    def aggregate_load(
        self,
        t_start: str | pd.Timestamp,
        t_stop: str | pd.Timestamp,
        freq: str = "1h",
        tz: str | None = None,
    ) -> pd.Series:
        """Fleet-aggregate charging power (kW).

        Charge-asap baseline: each session draws constant ``max_kw`` from
        ``t_in`` until enough wall-energy has flowed to deliver
        ``energy_kwh`` at efficiency η, then idles. ``tz=None`` returns the
        UTC index; any IANA zone returns the equivalent local-wall-clock
        index.
        """
        freq_n = _normalize_freq(freq)
        ts0, ts1 = _normalize_window(t_start, t_stop, self._region.tz)
        idx = self._target_index(ts0, ts1, freq_n)
        dt_h = 0.25 if freq_n == "15min" else 1.0

        sess = self._charging_sessions(self._ev_ids, ts0, ts1)
        if sess.empty:
            s = pd.Series(
                0.0, index=idx, dtype=float, name="aggregate_load_kw"
            )
            return _apply_query_tz(s, tz)

        max_kw = sess["max_kw"].astype(float).values
        # Avoid divide-by-zero (some sessions may have max_kw==0 in
        # pathological data).
        safe_kw = np.where(max_kw > 0.0, max_kw, np.inf)
        # RFC-004: compute duration in integer ns via the exact
        # ``round(energy / kw * 3600)`` → seconds, then ×1e9. Float ×
        # 3.6e12 lost sub-µs precision and could flip a 15-min bucket
        # boundary on multi-hour sessions.
        energy_kwh = sess["energy_kwh"].astype(float).values
        dur_s = np.round(energy_kwh / safe_kw * 3600.0).astype("int64")
        dur_ns = dur_s * 1_000_000_000
        # RFC-005: tz-aware ``.astype("int64")`` emits FutureWarning in
        # pandas 2.x and raises in 3.x. Convert to UTC, drop the tz, then
        # cast to int64 ns since epoch.
        t_in_ns = (
            sess["t_in"]
            .dt.tz_convert("UTC")
            .dt.tz_localize(None)
            .astype("datetime64[ns]")
            .astype("int64")
            .values
        )
        t_end_ns = t_in_ns + dur_ns

        bucket_edges_ns = np.array([t.value for t in idx], dtype="int64")
        step_ns = int(dt_h * 3.6e12)
        n_buckets = len(idx)

        ts0_ns = int(bucket_edges_ns[0])
        ts1_ns = int(bucket_edges_ns[-1]) + step_ns
        a = np.maximum(t_in_ns, ts0_ns)
        b = np.minimum(t_end_ns, ts1_ns)
        valid = b > a
        a = a[valid]
        b = b[valid]
        p = max_kw[valid]
        # RFC-005-style vectorisation: scatter-add session contributions
        # into the bucket grid via ``np.add.at`` instead of an O(N_sess *
        # buckets_per_sess) Python loop.
        load_kw = _aggregate_kw_from_sessions(
            a, b, p, ts0_ns, step_ns, n_buckets
        )

        s = pd.Series(
            load_kw, index=idx, dtype=float, name="aggregate_load_kw"
        )
        return _apply_query_tz(s, tz)

    # ---------------- optimizer inputs ---------------- #

    def _to_optimizer_inputs(
        self,
        ev_ids: Iterable[int] | None = None,
        t_start: str | pd.Timestamp | None = None,
        t_stop: str | pd.Timestamp | None = None,
    ) -> dict[int, dict[str, Any]]:
        """Internal bridge to the in-house ``pev_optim`` optimizer; not part of the PyPI surface.

        Build dicts ready for :class:`pev_optim.EVOptimizer`.

        Each value dict has:

        * ``archetype``: ``VehicleArchetype`` built from cached per-EV
          parameters (battery, max-kW, η, soc_min).
        * ``plug_in``: ``pd.Series[bool]`` indexed by UTC 15-min
          timestamps. The optimizer is tz-agnostic; UTC matches the cache
          storage tz (v2.0).
        * ``sessions``: ``pd.DataFrame`` with ``t_in``, ``t_out``,
          ``soc_in_kwh``, ``soc_target_out_kwh`` (tz-aware UTC).

        If ``t_start`` / ``t_stop`` are not supplied, the full cache year
        is used (Jan 1 2001 → Jan 1 2002).

        Raises
        ------
        KeyError
            if any ``ev_id`` in ``ev_ids`` is not in this Fleet.
        ValueError
            on per-EV ``max_kw`` inconsistencies.
        ImportError
            if ``pev_optim`` is not installed.
        """
        try:
            from pev_optim.adapters.from_pev_synth import (
                to_optimizer_inputs as _impl,
            )
        except ImportError as exc:
            raise ImportError(
                "this entry point is the in-house bridge to the unpublished "
                "sibling package `pev_optim`; it is not installable from PyPI. "
                "If you have a private checkout of `pev_optim`, install it via "
                "`pip install -e <path/to/pev_optim>`."
            ) from exc

        return _impl(self, ev_ids=ev_ids, t_start=t_start, t_stop=t_stop)

    # ---------------- save / load ---------------- #

    def save(self, path: str | Path) -> Path:
        """Write a self-contained bundle under ``path/{profile_type}_fleet/``.

        Contents
        --------
        * ``fleet.parquet`` — the static rows for this Fleet's EV subset.
        * ``plug_status_15min.parquet`` — subsetted plug-status matrix.
        * ``sessions.parquet`` — subsetted sessions.
        * ``mobility.parquet`` — subsetted mobility.
        * ``plug_in_events.parquet`` — subsetted events (when present).
        * ``meta.json`` — bundle metadata: ``profile_type``, ``region``
          (name + tz), the subset's ``ev_ids``, and any inherited meta.
        """
        target = Path(path) / f"{self._profile_type}_fleet"
        target.mkdir(parents=True, exist_ok=True)

        # 1. fleet.parquet: just the subset rows.
        self._fleet_df.to_parquet(target / "fleet.parquet", index=False)

        # 2. plug_status_15min.parquet, sessions.parquet, mobility.parquet,
        #    plug_in_events.parquet, plug_status_hourly.parquet -- subset
        #    by ev_id.
        for fname in (
            "plug_status_15min.parquet",
            "plug_status_hourly.parquet",
            "sessions.parquet",
            "mobility.parquet",
            "plug_in_events.parquet",
        ):
            src = self._data_root / fname
            if not src.exists():
                continue
            df = pd.read_parquet(
                src, filters=[("ev_id", "in", self._ev_ids)]
            )
            df.to_parquet(target / fname, index=False)

        # 3. meta.json
        from . import __version__ as _api_version

        meta_out = {
            "profile_type": self._profile_type,
            "region": self._region.name,
            "region_tz": self._region.tz,
            "n": len(self._ev_ids),
            "ev_ids": self._ev_ids,
            "source_data_root": str(self._data_root),
            "inherited_meta": self._meta,
            "api_version": _api_version,
            "storage_timezone": _STORAGE_TZ,
        }
        with (target / "meta.json").open("w", encoding="utf-8") as f:
            json.dump(meta_out, f, indent=2, default=str)

        return target

    @classmethod
    def load(cls, path: str | Path) -> Fleet:
        """Inverse of :meth:`save`. Accepts either the inner bundle directory
        (``.../{type}_fleet``) or its parent (the latter is convenience).
        """
        p = Path(path)
        # If user passed the parent, look for an inner *_fleet directory.
        if not (p / "fleet.parquet").exists():
            candidates = list(p.glob("*_fleet"))
            if len(candidates) == 1:
                p = candidates[0]
            else:
                raise FileNotFoundError(
                    f"No fleet.parquet at {p} and ambiguous inner bundles: "
                    f"{candidates}"
                )
        meta_file = p / "meta.json"
        if not meta_file.exists():
            raise FileNotFoundError(
                f"meta.json is required to load a Fleet bundle; "
                f"expected at {meta_file}"
            )
        with meta_file.open(encoding="utf-8") as f:
            meta = json.load(f)
        profile_type = meta.get("profile_type")
        if not isinstance(profile_type, str) or not profile_type:
            raise ValueError(
                f"meta.json at {meta_file} is missing required key "
                f"'profile_type' (must be a non-empty string)"
            )
        region_name = meta.get("region")
        if not isinstance(region_name, str) or not region_name:
            raise ValueError(
                f"meta.json at {meta_file} is missing required key "
                f"'region' (must be a non-empty string)"
            )
        region = Region.from_name(region_name)
        fleet_df = pd.read_parquet(p / "fleet.parquet")
        ev_ids = fleet_df["ev_id"].astype(int).tolist()
        return cls(
            data_root=p,
            ev_ids=ev_ids,
            profile_type=profile_type,
            region=region,
            meta=meta,
        )

    # ---------------- private helpers ---------------- #

    def _row_for(self, ev_id: int) -> _ProfileRow:
        if ev_id not in self._pos_by_evid:
            raise KeyError(f"ev_id {ev_id} not in this Fleet")
        r = self._fleet_df.iloc[self._pos_by_evid[ev_id]]
        return _ProfileRow(
            ev_id=int(r["ev_id"]),
            powertrain=str(r["powertrain"]),
            make=str(r["make"]),
            model=str(r["model"]),
            model_year=int(r["model_year"]),
            archetype=str(r["archetype"]),
            battery_kwh=float(r["B_kwh"]),
            obc_kw=float(r["OBC_kw"]),
            evse_home_kw=float(r["EVSE_home_kw"]),
            panel_cap_kw=float(r["panel_cap_kw"]),
            evse_brand=str(r["EVSE_brand"]),
            evse_connector=str(r["EVSE_connector"]),
            eta_charge=float(r["eta_charge_nominal"]),
            has_heat_pump=bool(r["has_heat_pump"]),
            wh_per_mi_nominal=float(r["Wh_per_mi_nominal"]),
            garage_type=str(r["garage_type"]),
            income_tier=str(r["income_tier"]),
            home_station=str(r["home_station"]),
            seed=int(r["seed"]),
            soc_min_kwh=float(r["soc_min_kwh"]),
            donor_household_id=str(r["nhts_donor_houseid"]),
            donor_vehicle_id=str(r["nhts_donor_vehid"]),
            cluster_z=int(r["cluster_z"]),
        )

    def _read_parquet(
        self,
        fname: str,
        columns: list[str] | None,
        ev_ids: list[int] | None,
    ) -> pd.DataFrame:
        """Load a parquet from the bundle with ev_id pushdown."""
        path = self._data_root / fname
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found in bundle {self._data_root}"
            )
        filters = None
        if ev_ids is not None and len(ev_ids) > 0:
            filters = [("ev_id", "in", list(ev_ids))]
        return pd.read_parquet(path, columns=columns, filters=filters)

    @staticmethod
    def _target_index(
        ts0: pd.Timestamp, ts1: pd.Timestamp, freq: str
    ) -> pd.DatetimeIndex:
        """Build the canonical target index in storage tz (UTC).

        The returned ``DatetimeIndex`` is tz-aware UTC; callers wrap it via
        :func:`_apply_query_tz` to honour the user's ``tz`` kwarg.
        """
        return pd.date_range(
            start=ts0,
            end=ts1,
            freq=freq,
            inclusive="left",
            tz=_STORAGE_TZ,
        )


# ----------------------------------------------------------------------
# generate_profiles factory
# ----------------------------------------------------------------------

def generate_profiles(
    profile_type: ProfileType | str = "residential",
    n: int = 1000,
    region: Region | str = "bay_area",
    seed: int = _DEFAULT_SEED,
    data_root: Path | None = None,
    *,
    replicate_id: int = 0,
    r_total: int = 1,
) -> Fleet:
    """Return a :class:`Fleet` of ``n`` EVs of ``profile_type`` for ``region``.

    Parameters
    ----------
    profile_type:
        ``'residential'`` or ``'workplace'``. The legacy ``'fleet_depot'``
        from v1.1 is de-scoped in v2.0 (plan §2.6) and raises
        ``ValueError``.
    n:
        Number of EVs to return. Must be a positive integer not exceeding
        the cached fleet size for the requested ``(region, profile_type)``.
    region:
        Either a :class:`Region` instance or its short-name string (e.g.
        ``'bay_area'``, ``'dallas_fort_worth'``). The region drives
        (i) the cache path
        ``data/pev/processed/<region.name>/<profile_type>_ev_synth/`` and
        (ii) the default local tz for tz-naive ``t_start`` / ``t_stop``
        inputs on every Fleet / Profile time-window method.
    seed:
        Master seed for the random subset selection (default
        ``20260520`` — the v2.0 master seed).
    data_root:
        Optional override of the per-region cache directory. Useful for
        tests against ad-hoc bundles. Defaults to the canonical layout.

    Raises
    ------
    ValueError
        ``profile_type='fleet_depot'`` (de-scoped); unknown profile types;
        ``n`` not positive; ``n`` exceeds the cached fleet size.
    FileNotFoundError
        The ``(region, profile_type)`` cache does not exist on disk yet.
        No fleet caches ship in the wheel or in a fresh checkout; build
        one with a one-time ``python -m pev_synth.nhts_loader`` (NHTS 2017
        download + processing) followed by ``python -m
        pev_synth.cache_regen one --region <r> --profile-type <t>``. The
        error message points at that CLI.
    """
    _validate_profile_type(profile_type)

    if n is None or n < 1:
        raise ValueError(f"n must be a positive integer; got {n!r}")

    region_obj = _resolve_region(region)

    if data_root is not None:
        root = Path(data_root)
    else:
        # RFC-022.r: r_total > 1 resolves through replicate_dir so the
        # layout matches what cache_regen wrote.
        root = Fleet.cache_path(
            region_obj,
            profile_type,
            replicate_id=replicate_id,
            r_total=r_total,
        )
    fleet_pq = root / "fleet.parquet"
    if not fleet_pq.exists():
        raise FileNotFoundError(
            f"No cached fleet at {fleet_pq}. The cache for "
            f"region={region_obj.name!r}, profile_type={profile_type!r} "
            "has not been generated yet. No fleet caches ship in the wheel "
            "or in a fresh checkout; build one in two steps: (1) one-time "
            "`python -m pev_synth.nhts_loader` to fetch and process the "
            "NHTS 2017 microdata, then (2) "
            f"`python -m pev_synth.cache_regen one --region {region_obj.name} "
            f"--profile-type {profile_type}` (add `--n <N>` to size the "
            "fleet). Alternatively, if you pip-installed ev-flow and keep a "
            "prebuilt data tree elsewhere, set the PEV_SYNTH_DATA_ROOT "
            "environment variable to that directory (the default lookup "
            "path is <repo_root>/data, which only exists in a dev checkout)."
        )
    cached = pd.read_parquet(fleet_pq, columns=["ev_id"])
    cache_size = len(cached)
    if n > cache_size:
        raise ValueError(
            f"n={n} exceeds the cached fleet size ({cache_size}) for "
            f"(region={region_obj.name!r}, "
            f"profile_type={profile_type!r}). "
            "Use pev_synth.regenerate_fleet(...) to rebuild a bigger "
            "fleet (heavy pipeline; v2.0 stub)."
        )

    rng = np.random.default_rng(seed)
    all_ids = cached["ev_id"].astype(int).to_numpy()
    if n == cache_size:
        chosen = all_ids.tolist()
    else:
        chosen = sorted(
            int(x) for x in rng.choice(all_ids, size=n, replace=False)
        )

    meta_file = root / "meta.json"
    meta: dict[str, Any] = {}
    if meta_file.exists():
        try:
            with meta_file.open(encoding="utf-8") as f:
                meta = json.load(f)
        except json.JSONDecodeError:
            meta = {}

    return Fleet(
        data_root=root,
        ev_ids=chosen,
        profile_type=profile_type,
        region=region_obj,
        meta=meta,
    )


def regenerate_fleet(
    profile_type: ProfileType | str,
    n: int,
    region: Region | str = "bay_area",
    seed: int = _DEFAULT_SEED,
    data_root: Path | None = None,
) -> Fleet:
    """Run the heavy M1-M7 pipeline to produce a fresh ``n``-EV bundle.

    Stub: not implemented at the api level. The full pipeline runs via
    ``python -m pev_synth.cache_regen one``.
    """
    _validate_profile_type(profile_type)
    _ = (n, region, seed, data_root)  # acknowledged, unused
    raise NotImplementedError(
        "regenerate_fleet is a stub. Run "
        "`python -m pev_synth.cache_regen one --region <name> "
        "--profile-type <type> --n <N>` to rebuild a cache (after a "
        "one-time `python -m pev_synth.nhts_loader`), or use "
        "generate_profiles(...) with n <= cache_size for the already-"
        "generated cache."
    )


__all__ = [
    "Fleet",
    "Profile",
    "ProfileType",
    "Region",
    "generate_profiles",
    "list_profile_types",
    "regenerate_fleet",
]
