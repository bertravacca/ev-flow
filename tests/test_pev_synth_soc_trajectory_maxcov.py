"""Coverage-maximising tests for M6 — `pev_synth.soc_trajectory`.

This module complements ``tests/test_pev_synth_soc_trajectory.py`` (which
covers the eight orchestrator-named acceptance tests and the v2.0 /
v3.0 public API) by exercising the remaining branches that the existing
suite leaves uncovered:

- ``_default_paths`` (legacy v1.1 flat layout helper).
- ``build_event_timeline`` with both sources empty.
- The DCFC dwell-cap-at-trip-end branch and the post-clamp energy ceiling.
- ``_clamp_overlapping_sessions`` (overlap shrink + zero-dwell widen + the
  battery-energy lock-step rescale).
- ``iterate_soc_ledger`` skipping EVs that have no events.
- The workplace ``location`` drop-and-warn guard in ``compute_sessions``.
- Every ``_validate_v2_inputs`` raise path.
- ``resolve_v2_paths`` invalid-profile-type raise.
- ``_sessions_to_dataframe`` empty / tz-localize branches and
  ``_empty_sessions_df`` for both profile types.
- ``_update_meta`` for v1.1 and v2.0 + workplace.
- The end-to-end ``run`` driver (residential + workplace, including the
  missing-input ``FileNotFoundError`` guards), driven entirely off
  in-memory parquet written to ``tmp_path``.
- The ``main`` CLI for both the legacy and ``--region`` v2.0 code paths,
  plus ``--seed`` override.

Every fixture is synthetic and self-contained: no NHTS micro-data and no
prebuilt fleet caches are read. Fixed seeds are used everywhere a draw is
involved so the assertions are value-stable.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pev_synth import soc_trajectory as soc

PEV_TZ = "America/Los_Angeles"
UTC = "UTC"


# --------------------------------------------------------------------------- #
# Synthetic fixtures (mirroring the conventions in the sibling test file).
# --------------------------------------------------------------------------- #

def _make_fleet(
    n_evs: int = 3,
    B_kwh: float = 60.0,
    eta: float = 0.90,
    evse_home_kw: float = 7.2,
    obc_kw: float = 7.2,
    powertrain: list[str] | None = None,
) -> pd.DataFrame:
    cols = {
        "ev_id": np.arange(n_evs, dtype="int32"),
        "B_kwh": np.full(n_evs, B_kwh, dtype="float32"),
        "eta_charge_nominal": np.full(n_evs, eta, dtype="float32"),
        "EVSE_home_kw": np.full(n_evs, evse_home_kw, dtype="float32"),
        "OBC_kw": np.full(n_evs, obc_kw, dtype="float32"),
        "soc_min_kwh": np.full(n_evs, B_kwh * 0.10, dtype="float32"),
    }
    df = pd.DataFrame(cols)
    if powertrain is not None:
        df["powertrain"] = pd.Categorical(powertrain, categories=["BEV", "PHEV"])
    return df


def _trip(ev_id: int, start: str, end: str, miles: float, kwh: float, tz: str) -> dict:
    return {
        "ev_id": np.int32(ev_id),
        "start_ts": pd.Timestamp(start, tz=tz),
        "end_ts": pd.Timestamp(end, tz=tz),
        "miles": np.float32(miles),
        "kwh_consumed": np.float32(kwh),
    }


def _plug(ev_id: int, arr: str, dep: str, tz: str, location: str = "home") -> dict:
    arrival = pd.Timestamp(arr, tz=tz)
    departure = pd.Timestamp(dep, tz=tz)
    dwell_h = (departure - arrival).total_seconds() / 3600.0
    return {
        "ev_id": np.int32(ev_id),
        "arrival_ts": arrival,
        "departure_ts": departure,
        "dwell_h": np.float32(dwell_h),
        "location": location,
        "plug_in": True,
    }


def _trips_df(rows: list[dict], tz: str = PEV_TZ) -> pd.DataFrame:
    if rows:
        return pd.DataFrame(rows)
    return pd.DataFrame(
        {
            "ev_id": pd.Series([], dtype="int32"),
            "start_ts": pd.Series([], dtype=f"datetime64[ns, {tz}]"),
            "end_ts": pd.Series([], dtype=f"datetime64[ns, {tz}]"),
            "miles": pd.Series([], dtype="float32"),
            "kwh_consumed": pd.Series([], dtype="float32"),
        }
    )


def _plugs_df(
    rows: list[dict],
    tz: str = PEV_TZ,
    categories: list[str] | None = None,
) -> pd.DataFrame:
    cats = categories or ["home"]
    if rows:
        df = pd.DataFrame(rows)
        df["location"] = pd.Categorical(df["location"], categories=cats)
        return df
    return pd.DataFrame(
        {
            "ev_id": pd.Series([], dtype="int32"),
            "arrival_ts": pd.Series([], dtype=f"datetime64[ns, {tz}]"),
            "departure_ts": pd.Series([], dtype=f"datetime64[ns, {tz}]"),
            "dwell_h": pd.Series([], dtype="float32"),
            "location": pd.Categorical([], categories=cats),
            "plug_in": pd.Series([], dtype="bool"),
        }
    )


def _minimal_residential(n_evs: int = 2, tz: str = PEV_TZ):
    """One short daytime trip + one overnight plug-in per EV per day x 2 days."""
    fleet = _make_fleet(n_evs=n_evs)
    trips: list[dict] = []
    plugs: list[dict] = []
    for ev in range(n_evs):
        for d in range(2):
            day = pd.Timestamp(f"2001-01-0{d + 1} 09:00", tz=tz)
            trips.append(
                _trip(ev, day.isoformat(),
                      (day + pd.Timedelta(hours=1)).isoformat(),
                      miles=30.0, kwh=8.0, tz=tz)
            )
            plugs.append(
                _plug(ev, (day + pd.Timedelta(hours=9)).isoformat(),
                      (day + pd.Timedelta(hours=21)).isoformat(), tz=tz)
            )
    return fleet, _trips_df(trips, tz=tz), _plugs_df(plugs, tz=tz)


def _bay_area_region():
    from pev_synth.regions import Region

    return Region.from_name("bay_area")


# --------------------------------------------------------------------------- #
# Path helpers (299-301, 1099).
# --------------------------------------------------------------------------- #

def test_default_paths_legacy_flat_layout(tmp_path):
    """`_default_paths` returns the v1.1 flat residential layout."""
    paths = soc._default_paths(tmp_path)
    base = tmp_path / "data" / "pev" / "processed" / "residential_ev_synth"
    assert paths.fleet_in == base / "fleet.parquet"
    assert paths.mobility_in == base / "mobility.parquet"
    assert paths.plug_in_events_in == base / "plug_in_events.parquet"
    assert paths.sessions_out == base / "sessions.parquet"
    assert paths.meta_out == base / "meta.json"


def test_resolve_v2_paths_invalid_profile_type_raises(tmp_path):
    """resolve_v2_paths rejects an unknown profile_type (line 1099)."""
    region = _bay_area_region()
    with pytest.raises(ValueError, match="profile_type must be one of"):
        soc.resolve_v2_paths(tmp_path, region, "depot")


# --------------------------------------------------------------------------- #
# build_event_timeline — both sources empty (388).
# --------------------------------------------------------------------------- #

def test_build_event_timeline_both_sources_empty_hits_fallback():
    """No trips and no plug-ins exercises the `not frames` fallback (line 388).

    With both sources empty the function selects ``events = trips`` (the
    all-empty trips frame). The subsequent ``tz_convert`` cannot operate on
    the tz-naive empty column produced by ``to_numpy()``, so the empty-input
    case surfaces a ``TypeError`` rather than returning silently. We pin
    that observed contract here so the fallback branch is covered without
    pretending the degenerate input is supported.
    """
    trips = _trips_df([])
    plugs = _plugs_df([])
    with pytest.raises(TypeError, match="tz-naive"):
        soc.build_event_timeline(trips, plugs, tz=PEV_TZ)


def test_build_event_timeline_plugs_filtered_leaves_trips_only():
    """plug_in == False rows are dropped, exercising the single-frame branch."""
    trips = _trips_df([_trip(0, "2001-01-01 09:00", "2001-01-01 10:00", 30.0, 8.0, PEV_TZ)])
    plug = _plug(0, "2001-01-01 18:00", "2001-01-02 06:00", PEV_TZ)
    plug["plug_in"] = False
    plugs = _plugs_df([plug])
    timelines = soc.build_event_timeline(trips, plugs, tz=PEV_TZ)
    assert set(timelines) == {0}
    # Only the trip survives.
    assert (timelines[0]["kind"] == soc._EVENT_KIND_TRIP).all()


# --------------------------------------------------------------------------- #
# iterate_soc_ledger — EV with no events is skipped (917).
# --------------------------------------------------------------------------- #

def test_iterate_soc_ledger_skips_ev_without_events():
    """A fleet EV that has no timeline entry contributes zero sessions."""
    fleet = _make_fleet(n_evs=3)
    # Only EV 0 has events.
    trips = _trips_df([_trip(0, "2001-01-01 09:00", "2001-01-01 10:00", 30.0, 8.0, PEV_TZ)])
    plugs = _plugs_df([_plug(0, "2001-01-01 18:00", "2001-01-02 06:00", PEV_TZ)])
    cfg = soc.SocTrajectoryConfig()
    timelines = soc.build_event_timeline(trips, plugs)
    soc0 = soc.compute_initial_soc(fleet["ev_id"].to_numpy(), cfg.master_seed, cfg)
    df = soc.iterate_soc_ledger(fleet, timelines, soc0, cfg)
    assert set(df["ev_id"].unique()) == {0}, "EVs 1 and 2 (no events) must be skipped"


# --------------------------------------------------------------------------- #
# DCFC dwell capped at trip end + energy ceiling (645, and 852-865 via clamp).
# --------------------------------------------------------------------------- #

def test_dcfc_dwell_capped_at_trip_end():
    """A huge mid-trip DCFC need with a tiny battery caps dwell at trip end.

    A 13.6 kWh battery driving ~1400 mi of equivalent consumption in a
    3-hour window forces the DCFC ``required_dwell`` to exceed the trip's
    remaining half, so ``t_out_sess`` is clamped to ``t_end`` (line 645)
    and the delivered energy is clipped to ``nameplate · dwell``.
    """
    # A very short trip window (6 minutes → 3-minute remaining half) with a
    # large battery deficit means the DCFC ``required_dwell`` (energy / 50 kW)
    # exceeds the 3-minute window, so ``t_out_sess`` is clamped to ``t_end``.
    fleet = _make_fleet(n_evs=1, B_kwh=13.6, eta=0.90)
    trips = _trips_df([
        _trip(0, "2001-01-01 09:00", "2001-01-01 09:06", miles=120.0, kwh=380.0, tz=PEV_TZ),
    ])
    plugs = _plugs_df([])
    cfg = soc.SocTrajectoryConfig()
    timelines = soc.build_event_timeline(trips, plugs)
    soc0 = soc.compute_initial_soc(fleet["ev_id"].to_numpy(), cfg.master_seed, cfg)
    df = soc.iterate_soc_ledger(fleet, timelines, soc0, cfg)
    dcfc = df[df["location"].astype(str) == "dcfc_inject"]
    assert len(dcfc) == 1
    row = dcfc.iloc[0]
    # t_out is capped at the trip end (09:06), not t_mid + required dwell.
    trip_end = pd.Timestamp("2001-01-01 09:06", tz=PEV_TZ)
    assert pd.Timestamp(row["t_out"]) <= trip_end + pd.Timedelta(seconds=1)
    # Delivered energy honours the capped-dwell ceiling: energy ≤ 50 kW · dwell.
    dwell_h = (pd.Timestamp(row["t_out"]) - pd.Timestamp(row["t_in"])).total_seconds() / 3600.0
    assert float(row["energy_kwh"]) <= 50.0 * dwell_h + 1e-3


# --------------------------------------------------------------------------- #
# _clamp_overlapping_sessions (852-865) — direct unit tests.
# --------------------------------------------------------------------------- #

def _sess(ev_id, idx, t_in, t_out, energy_grid=10.0, energy_batt=9.0, max_kw=7.2):
    return soc._Session(
        ev_id=ev_id,
        session_idx=idx,
        t_in=pd.Timestamp(t_in, tz=PEV_TZ),
        t_out=pd.Timestamp(t_out, tz=PEV_TZ),
        soc_in_pct=20.0,
        soc_out_pct=80.0,
        soc_in_kwh=12.0,
        soc_target_out_kwh=48.0,
        energy_kwh=energy_grid,
        location=soc.LOC_HOME,
        max_kw=max_kw,
        energy_kwh_batt=energy_batt,
    )


def test_clamp_overlapping_shrinks_and_rescales_energy():
    """Overlap shrinks t_out and rescales grid+batt energy in lock-step."""
    # Session 0 nominally runs 4 h @ 7.2 kW = 28.8 kWh max but is set to a
    # large 50 kWh grid energy. After clamping to a 1-h window the ceiling
    # is 7.2 kWh, so both grid and batt energy shrink in the same ratio.
    s0 = _sess(0, 0, "2001-01-01 09:00", "2001-01-01 13:00",
               energy_grid=50.0, energy_batt=45.0, max_kw=7.2)
    s1 = _sess(0, 1, "2001-01-01 10:00", "2001-01-01 12:00")
    sessions = [s0, s1]
    soc._clamp_overlapping_sessions(sessions)
    # t_out clamped to next t_in.
    assert s0.t_out == pd.Timestamp("2001-01-01 10:00", tz=PEV_TZ)
    new_dwell_h = 1.0
    max_e = 7.2 * new_dwell_h
    assert s0.energy_kwh == pytest.approx(max_e, abs=1e-4)
    # Battery energy scaled by max_e / 50.0.
    assert s0.energy_kwh_batt == pytest.approx(45.0 * (max_e / 50.0), abs=1e-4)
    # Idempotent: a second call does not change anything.
    snapshot = (s0.t_out, s0.energy_kwh, s0.energy_kwh_batt)
    soc._clamp_overlapping_sessions(sessions)
    assert (s0.t_out, s0.energy_kwh, s0.energy_kwh_batt) == snapshot


def test_clamp_overlapping_zero_dwell_widened_to_one_second():
    """When the next t_in coincides with this t_in, dwell is widened to 1 s."""
    s0 = _sess(0, 0, "2001-01-01 09:00", "2001-01-01 13:00",
               energy_grid=0.0, energy_batt=0.0)
    s1 = _sess(0, 1, "2001-01-01 09:00", "2001-01-01 12:00")
    sessions = [s0, s1]
    soc._clamp_overlapping_sessions(sessions)
    assert s0.t_out == s0.t_in + pd.Timedelta(seconds=1)
    # prev_grid == 0 → the batt-rescale branch is skipped, energy stays 0.
    assert s0.energy_kwh == pytest.approx(0.0)
    assert s0.energy_kwh_batt == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Plug-event monotone guard (713) and workplace monotone guard (763).
# --------------------------------------------------------------------------- #

def test_plug_session_monotone_guard_zero_eta():
    """eta == 0 yields energy 0 and target == in; the monotone guard holds."""
    fleet = _make_fleet(n_evs=1, eta=0.0)
    trips = _trips_df([_trip(0, "2001-01-01 09:00", "2001-01-01 10:00", 30.0, 8.0, PEV_TZ)])
    plugs = _plugs_df([_plug(0, "2001-01-01 18:00", "2001-01-02 06:00", PEV_TZ)])
    cfg = soc.SocTrajectoryConfig()
    timelines = soc.build_event_timeline(trips, plugs)
    soc0 = soc.compute_initial_soc(fleet["ev_id"].to_numpy(), cfg.master_seed, cfg)
    df = soc.iterate_soc_ledger(fleet, timelines, soc0, cfg)
    home = df[df["location"].astype(str) == "home"]
    # With eta=0 no battery gain is possible: out == in and energy == 0.
    assert (home["soc_out_pct"] >= home["soc_in_pct"] - 1e-6).all()
    assert float(home["energy_kwh"].sum()) == pytest.approx(0.0)


def test_plug_zero_dwell_widened_to_one_minute():
    """A plug-in with departure == arrival is widened to t_in + 1 min (731)."""
    fleet = _make_fleet(n_evs=1, eta=0.90)
    trips = _trips_df([_trip(0, "2001-01-01 09:00", "2001-01-01 10:00", 30.0, 8.0, PEV_TZ)])
    # arrival_ts == departure_ts → zero dwell.
    plugs = _plugs_df([_plug(0, "2001-01-01 18:00", "2001-01-01 18:00", PEV_TZ)])
    cfg = soc.SocTrajectoryConfig()
    timelines = soc.build_event_timeline(trips, plugs)
    soc0 = soc.compute_initial_soc(fleet["ev_id"].to_numpy(), cfg.master_seed, cfg)
    df = soc.iterate_soc_ledger(fleet, timelines, soc0, cfg)
    home = df[df["location"].astype(str) == "home"].iloc[0]
    dwell = pd.Timestamp(home["t_out"]) - pd.Timestamp(home["t_in"])
    assert dwell == pd.Timedelta(minutes=1)


def test_workplace_monotone_guard_zero_eta():
    """Workplace path with eta == 0 also holds the monotone guard (line 763)."""
    fleet = _make_fleet(n_evs=1, eta=0.0)
    trips = _trips_df([
        _trip(0, "2001-01-01 08:00", "2001-01-01 09:00", 30.0, 8.0, UTC),
    ], tz=UTC)
    plugs = _plugs_df([
        _plug(0, "2001-01-01 09:00", "2001-01-01 17:00", UTC, location="work"),
    ], tz=UTC, categories=["work"])
    cfg = soc.SocTrajectoryConfig()
    timelines = soc.build_event_timeline(trips, plugs, tz=UTC)
    soc0 = soc.compute_initial_soc(fleet["ev_id"].to_numpy(), cfg.master_seed, cfg)
    df = soc.iterate_soc_ledger(
        fleet, timelines, soc0, cfg, profile_type="workplace", storage_tz=UTC
    )
    work = df[df["location"].astype(str) == "work"]
    assert (work["soc_out_pct"] >= work["soc_in_pct"] - 1e-6).all()
    assert float(work["energy_kwh"].sum()) == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Workplace residual home_charging_required latch on last session (804).
# --------------------------------------------------------------------------- #

def test_workplace_residual_home_required_latched_on_last_session():
    """A trip after the last workplace session that breaches the floor latches
    home_charging_required on the most recent session (line 803-804)."""
    fleet = _make_fleet(n_evs=1, B_kwh=20.0, eta=0.90)
    trips = _trips_df([
        # Morning commute (small).
        _trip(0, "2001-01-01 08:00", "2001-01-01 09:00", 20.0, 4.0, UTC),
        # Late trip AFTER the workplace plug-in that drains below floor; no
        # further session follows so the pending flag is latched on the
        # last recorded session.
        _trip(0, "2001-01-01 18:00", "2001-01-01 20:00", 200.0, 40.0, UTC),
    ], tz=UTC)
    plugs = _plugs_df([
        _plug(0, "2001-01-01 09:00", "2001-01-01 17:00", UTC, location="work"),
    ], tz=UTC, categories=["work"])
    cfg = soc.SocTrajectoryConfig()
    timelines = soc.build_event_timeline(trips, plugs, tz=UTC)
    soc0 = soc.compute_initial_soc(fleet["ev_id"].to_numpy(), cfg.master_seed, cfg)
    df = soc.iterate_soc_ledger(
        fleet, timelines, soc0, cfg, profile_type="workplace", storage_tz=UTC
    )
    assert df["home_charging_required"].any(), (
        "End-of-window workplace floor breach must latch home_charging_required"
    )
    last = df.sort_values("session_idx").iloc[-1]
    assert bool(last["home_charging_required"]) is True


# --------------------------------------------------------------------------- #
# compute_sessions workplace location drop-and-warn (1021-1032).
# --------------------------------------------------------------------------- #

def test_compute_sessions_workplace_drops_nonwork_locations(caplog):
    """Non-work plug-ins in the workplace profile are dropped with a warning."""
    fleet = _make_fleet(n_evs=1, B_kwh=40.0)
    trips = _trips_df([
        _trip(0, "2001-01-01 08:00", "2001-01-01 09:00", 30.0, 8.0, UTC),
    ], tz=UTC)
    # One "work" plug-in and one stray "home" plug-in that must be dropped.
    plugs = _plugs_df([
        _plug(0, "2001-01-01 09:00", "2001-01-01 17:00", UTC, location="work"),
        _plug(0, "2001-01-01 18:00", "2001-01-02 06:00", UTC, location="home"),
    ], tz=UTC, categories=["work", "home"])
    region = _bay_area_region()
    with caplog.at_level("WARNING"):
        df = soc.compute_sessions(
            fleet=fleet, mobility=trips, plug_in_events=plugs,
            region=region, profile_type="workplace", seed=20260520,
        )
    assert any("dropping" in rec.message for rec in caplog.records)
    # No "home" sessions survive.
    assert (df["location"].astype(str) == "work").all()


# --------------------------------------------------------------------------- #
# _validate_v2_inputs raise paths (1061, 1065, 1069, 1071).
# --------------------------------------------------------------------------- #

def test_validate_missing_fleet_column_raises():
    fleet, mob, pi = _minimal_residential(n_evs=1, tz=UTC)
    bad = fleet.drop(columns=["B_kwh"])
    region = _bay_area_region()
    with pytest.raises(ValueError, match="fleet missing columns"):
        soc.compute_sessions(bad, mob, pi, region, "residential", seed=20260520)


def test_validate_missing_mobility_column_raises():
    fleet, mob, pi = _minimal_residential(n_evs=1, tz=UTC)
    bad = mob.drop(columns=["kwh_consumed"])
    region = _bay_area_region()
    with pytest.raises(ValueError, match="mobility missing columns"):
        soc.compute_sessions(fleet, bad, pi, region, "residential", seed=20260520)


def test_validate_missing_plug_column_raises():
    fleet, mob, pi = _minimal_residential(n_evs=1, tz=UTC)
    bad = pi.drop(columns=["dwell_h"])
    region = _bay_area_region()
    with pytest.raises(ValueError, match="plug_in_events missing columns"):
        soc.compute_sessions(fleet, mob, bad, region, "residential", seed=20260520)


def test_validate_workplace_requires_location_column_raises():
    """Workplace profile needs a 'location' column (line 1070-1071)."""
    fleet = _make_fleet(n_evs=1, B_kwh=40.0)
    trips = _trips_df([_trip(0, "2001-01-01 08:00", "2001-01-01 09:00", 30.0, 8.0, UTC)], tz=UTC)
    # A plug-in frame with all required cols EXCEPT 'location'.
    plug = _plug(0, "2001-01-01 09:00", "2001-01-01 17:00", UTC, location="work")
    del plug["location"]
    pi = pd.DataFrame([plug])
    region = _bay_area_region()
    with pytest.raises(ValueError, match="must include a 'location' column"):
        soc.compute_sessions(fleet, trips, pi, region, "workplace", seed=20260520)


# --------------------------------------------------------------------------- #
# _sessions_to_dataframe empty path + _empty_sessions_df (1126-1127, 1193-1214).
# --------------------------------------------------------------------------- #

def test_empty_sessions_df_residential_schema():
    df = soc._empty_sessions_df(profile_type="residential", storage_tz=PEV_TZ)
    assert len(df) == 0
    assert "home_charging_required" not in df.columns
    assert str(df["t_in"].dtype) == f"datetime64[ns, {PEV_TZ}]"
    assert set(df["location"].cat.categories) == {"home", "dcfc_inject"}


def test_empty_sessions_df_workplace_schema():
    df = soc._empty_sessions_df(profile_type="workplace", storage_tz=UTC)
    assert "home_charging_required" in df.columns
    assert df["home_charging_required"].dtype == bool
    assert set(df["location"].cat.categories) == {"work"}
    assert str(df["t_in"].dtype) == f"datetime64[ns, {UTC}]"


def test_sessions_to_dataframe_empty_list_returns_empty_df():
    df = soc._sessions_to_dataframe([], profile_type="residential", storage_tz=UTC)
    assert len(df) == 0
    assert str(df["t_in"].dtype) == f"datetime64[ns, {UTC}]"


def test_iterate_soc_ledger_empty_timeline_returns_empty_df():
    """No EV has events → the empty-sessions branch of _sessions_to_dataframe."""
    fleet = _make_fleet(n_evs=2)
    cfg = soc.SocTrajectoryConfig()
    df = soc.iterate_soc_ledger(fleet, {}, {}, cfg)
    assert len(df) == 0


def test_sessions_to_dataframe_tz_localizes_naive_timestamps():
    """tz-naive _Session timestamps are localized to storage_tz (1176, 1180)."""
    s = soc._Session(
        ev_id=0, session_idx=0,
        t_in=pd.Timestamp("2001-01-01 09:00"),  # tz-naive
        t_out=pd.Timestamp("2001-01-01 17:00"),  # tz-naive
        soc_in_pct=20.0, soc_out_pct=80.0,
        soc_in_kwh=12.0, soc_target_out_kwh=48.0,
        energy_kwh=10.0, location=soc.LOC_HOME, max_kw=7.2,
        energy_kwh_batt=9.0,
    )
    df = soc._sessions_to_dataframe([s], profile_type="residential", storage_tz=UTC)
    assert str(df["t_in"].dt.tz) == "UTC"
    assert str(df["t_out"].dt.tz) == "UTC"


# --------------------------------------------------------------------------- #
# _update_meta (1240-1313) — both v1.1 and v2.0 + workplace.
# --------------------------------------------------------------------------- #

def _seed_meta(meta_path: Path) -> None:
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps({"package_versions": {"existing": "1.0"}}))


def test_update_meta_v11_residential(tmp_path):
    meta_path = tmp_path / "meta.json"
    _seed_meta(meta_path)
    soc._update_meta(
        meta_path,
        n_sessions=10,
        n_dcfc_injections=2,
        fleet_kwh_charged_grid=100.0,
        fleet_kwh_driven=90.0,
        sessions_per_ev_median=3.0,
        profile_type="residential",
        storage_tz=PEV_TZ,
    )
    meta = json.loads(meta_path.read_text())
    block = meta["soc_trajectory"]
    assert block["methodology_version"] == soc.METHODOLOGY_VERSION
    assert block["master_seed"] == soc.MASTER_SEED_DEFAULT
    assert block["n_dcfc_injections"] == 2
    assert block["dcfc_fraction"] == pytest.approx(0.2)
    assert block["energy_ratio_charged_over_driven"] == pytest.approx(100.0 / 90.0)
    assert "home_charging_required_share" not in block
    # Existing package_versions key is preserved and refreshed with python.
    assert "python" in meta["package_versions"]


def test_update_meta_v2_workplace(tmp_path):
    meta_path = tmp_path / "meta.json"
    _seed_meta(meta_path)
    soc._update_meta(
        meta_path,
        n_sessions=20,
        n_dcfc_injections=0,
        fleet_kwh_charged_grid=200.0,
        fleet_kwh_driven=0.0,  # forces the NaN energy-ratio branch.
        sessions_per_ev_median=5.0,
        profile_type="workplace",
        storage_tz=soc.STORAGE_TZ_V2,
        home_charging_required_share=0.1,
        n_home_charging_required_sessions=3,
    )
    meta = json.loads(meta_path.read_text())
    block = meta["soc_trajectory"]
    assert block["methodology_version"] == soc.METHODOLOGY_VERSION_V2
    assert block["master_seed"] == soc.MASTER_SEED_V2_DEFAULT
    assert block["storage_tz"] == "UTC"
    assert block["home_charging_required_share"] == pytest.approx(0.1)
    assert block["home_charging_required_max_share"] == soc.HOME_CHARGING_REQUIRED_MAX_SHARE
    assert block["n_home_charging_required_sessions"] == 3
    # n_sessions == 0 dcfc_fraction guard is the inverse here (n_sessions>0).
    assert block["dcfc_fraction"] == pytest.approx(0.0)


def test_update_meta_zero_sessions_dcfc_fraction_zero(tmp_path):
    """n_sessions == 0 → dcfc_fraction defaults to 0.0 (guard branch)."""
    meta_path = tmp_path / "meta.json"
    _seed_meta(meta_path)
    soc._update_meta(
        meta_path,
        n_sessions=0,
        n_dcfc_injections=0,
        fleet_kwh_charged_grid=0.0,
        fleet_kwh_driven=0.0,
        sessions_per_ev_median=0.0,
        profile_type="workplace",
        storage_tz=soc.STORAGE_TZ_V2,
        home_charging_required_share=None,  # None → 0.0 fallback branch.
    )
    meta = json.loads(meta_path.read_text())
    block = meta["soc_trajectory"]
    assert block["dcfc_fraction"] == 0.0
    assert block["home_charging_required_share"] == 0.0


# --------------------------------------------------------------------------- #
# run() end-to-end driver (1356-1454) + FileNotFoundError guards.
# --------------------------------------------------------------------------- #

def _write_inputs(base: Path, fleet, mob, pi) -> soc.M6Paths:
    base.mkdir(parents=True, exist_ok=True)
    paths = soc.M6Paths(
        fleet_in=base / "fleet.parquet",
        mobility_in=base / "mobility.parquet",
        plug_in_events_in=base / "plug_in_events.parquet",
        sessions_out=base / "sessions.parquet",
        meta_out=base / "meta.json",
    )
    fleet.to_parquet(paths.fleet_in, index=False)
    mob.to_parquet(paths.mobility_in, index=False)
    pi.to_parquet(paths.plug_in_events_in, index=False)
    _seed_meta(paths.meta_out)
    return paths


def test_run_residential_end_to_end(tmp_path):
    fleet, mob, pi = _minimal_residential(n_evs=2, tz=PEV_TZ)
    paths = _write_inputs(tmp_path / "syn", fleet, mob, pi)
    result = soc.run(paths)
    assert isinstance(result, soc.SocTrajectoryResult)
    assert result.n_sessions > 0
    assert paths.sessions_out.is_file()
    assert "median" in result.sessions_per_ev
    # meta.json was updated with the soc_trajectory block.
    meta = json.loads(paths.meta_out.read_text())
    assert meta["soc_trajectory"]["profile_type"] == "residential"
    assert result.fleet_kwh_driven == pytest.approx(float(mob["kwh_consumed"].sum()))


def test_run_workplace_end_to_end_utc(tmp_path):
    fleet = _make_fleet(n_evs=2, B_kwh=40.0)
    trips: list[dict] = []
    plugs: list[dict] = []
    for ev in range(2):
        day = pd.Timestamp("2001-01-01", tz=UTC)
        # High-mileage EV-1 morning to trigger home_charging_required.
        kwh = 50.0 if ev == 1 else 8.0
        trips.append(_trip(ev, (day + pd.Timedelta(hours=8)).isoformat(),
                           (day + pd.Timedelta(hours=9)).isoformat(),
                           180.0 if kwh > 40 else 30.0, kwh, UTC))
        plugs.append(_plug(ev, (day + pd.Timedelta(hours=9)).isoformat(),
                           (day + pd.Timedelta(hours=17)).isoformat(),
                           UTC, location="work"))
    mob = _trips_df(trips, tz=UTC)
    pi = _plugs_df(plugs, tz=UTC, categories=["work"])
    paths = _write_inputs(tmp_path / "wp", fleet, mob, pi)
    cfg = soc.SocTrajectoryConfig(master_seed=soc.MASTER_SEED_V2_DEFAULT)
    result = soc.run(paths, config=cfg, profile_type="workplace", storage_tz=soc.STORAGE_TZ_V2)
    assert result.n_sessions > 0
    assert result.n_dcfc_injections == 0  # no DCFC in workplace.
    meta = json.loads(paths.meta_out.read_text())
    block = meta["soc_trajectory"]
    assert block["profile_type"] == "workplace"
    assert "home_charging_required_share" in block
    # The written parquet has the workplace flag column.
    written = pd.read_parquet(paths.sessions_out)
    assert "home_charging_required" in written.columns


def test_run_zero_sessions_median_zero(tmp_path):
    """A fleet whose only trip never breaches the floor and never plugs in
    yields zero sessions, exercising the sess_med = 0.0 branch (line 1418)."""
    fleet = _make_fleet(n_evs=1, B_kwh=60.0, eta=0.90)
    # One short trip (no floor breach → no DCFC) and no plug-in events.
    mob = _trips_df([_trip(0, "2001-01-01 09:00", "2001-01-01 10:00", 5.0, 2.0, PEV_TZ)])
    pi = _plugs_df([])
    paths = _write_inputs(tmp_path / "zero", fleet, mob, pi)
    result = soc.run(paths)
    assert result.n_sessions == 0
    assert result.sessions_per_ev["median"] == 0.0
    meta = json.loads(paths.meta_out.read_text())
    assert meta["soc_trajectory"]["n_sessions"] == 0


def test_run_missing_fleet_raises(tmp_path):
    fleet, mob, pi = _minimal_residential(n_evs=1, tz=PEV_TZ)
    paths = _write_inputs(tmp_path / "m1", fleet, mob, pi)
    paths.fleet_in.unlink()
    with pytest.raises(FileNotFoundError, match="fleet.parquet"):
        soc.run(paths)


def test_run_missing_mobility_raises(tmp_path):
    fleet, mob, pi = _minimal_residential(n_evs=1, tz=PEV_TZ)
    paths = _write_inputs(tmp_path / "m2", fleet, mob, pi)
    paths.mobility_in.unlink()
    with pytest.raises(FileNotFoundError, match="mobility.parquet"):
        soc.run(paths)


def test_run_missing_plug_events_raises(tmp_path):
    fleet, mob, pi = _minimal_residential(n_evs=1, tz=PEV_TZ)
    paths = _write_inputs(tmp_path / "m3", fleet, mob, pi)
    paths.plug_in_events_in.unlink()
    with pytest.raises(FileNotFoundError, match="plug_in_events.parquet"):
        soc.run(paths)


# --------------------------------------------------------------------------- #
# main() CLI — legacy and v2.0 region paths (1469-1544).
# --------------------------------------------------------------------------- #

def test_main_legacy_path(tmp_path, capsys):
    """`main` with --repo-root (no --region) drives the v1.1 flat layout."""
    fleet, mob, pi = _minimal_residential(n_evs=2, tz=PEV_TZ)
    base = tmp_path / "data" / "pev" / "processed" / "residential_ev_synth"
    _write_inputs(base, fleet, mob, pi)
    rc = soc.main(["--repo-root", str(tmp_path), "--seed", "20260518"])
    assert rc == 0
    err = capsys.readouterr().err
    assert "M6 soc_trajectory" in err
    assert (base / "sessions.parquet").is_file()


def test_main_v2_region_workplace(tmp_path, capsys):
    """`main --region bay_area --profile-type workplace` drives the v2.0 path."""
    fleet = _make_fleet(n_evs=2, B_kwh=40.0)
    day = pd.Timestamp("2001-01-01", tz=UTC)
    trips = _trips_df([
        _trip(ev, (day + pd.Timedelta(hours=8)).isoformat(),
              (day + pd.Timedelta(hours=9)).isoformat(), 30.0, 8.0, UTC)
        for ev in range(2)
    ], tz=UTC)
    plugs = _plugs_df([
        _plug(ev, (day + pd.Timedelta(hours=9)).isoformat(),
              (day + pd.Timedelta(hours=17)).isoformat(), UTC, location="work")
        for ev in range(2)
    ], tz=UTC, categories=["work"])
    base = tmp_path / "data" / "pev" / "processed" / "bay_area" / "workplace_ev_synth"
    _write_inputs(base, fleet, trips, plugs)
    rc = soc.main([
        "--repo-root", str(tmp_path),
        "--region", "bay_area",
        "--profile-type", "workplace",
    ])
    assert rc == 0
    assert (base / "sessions.parquet").is_file()
    written = pd.read_parquet(base / "sessions.parquet")
    assert "home_charging_required" in written.columns


def test_main_v2_region_seed_override(tmp_path):
    """`main --region ... --seed N` threads the explicit seed into the config."""
    fleet, mob, pi = _minimal_residential(n_evs=2, tz=UTC)
    base = tmp_path / "data" / "pev" / "processed" / "bay_area" / "residential_ev_synth"
    _write_inputs(base, fleet, mob, pi)
    rc = soc.main([
        "--repo-root", str(tmp_path),
        "--region", "bay_area",
        "--profile-type", "residential",
        "--seed", "12345",
    ])
    assert rc == 0
    written = pd.read_parquet(base / "sessions.parquet")
    assert len(written) > 0
