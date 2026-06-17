"""Data-independent coverage tests for ``pev_synth.api``.

The bay_area residential cache (and every other heavy processed bundle) is
absent in CI, so the cache-gated suite in ``tests/test_pev_synth_api.py`` skips
there and the dynamic-query surface of ``Fleet`` / ``Profile`` goes untested.
This module synthesises a *complete* on-disk bundle under ``tmp_path`` -- one
that carries ``fleet.parquet`` plus the ``plug_status_15min`` /
``plug_status_hourly`` / ``sessions`` / ``mobility`` / ``plug_in_events``
artifacts the query methods read -- so the presence/absence, SoC-trajectory,
trip, aggregate-load, summary, filter, and save/load paths all execute without
the heavy NHTS-derived cache.

It complements (does not duplicate) the existing self-contained files
``test_pev_synth_api_p1.py`` (fail-loud edges), ``test_pev_synth_api_p4.py``
(getitem / by_ev_id / kernel) and ``test_pev_synth_api_red_team.py``
(M3/L3/C2/H6/H3 regressions): here we drive the *happy-path data paths* plus
the validation/helper branches not otherwise reached.

All tests are deterministic (no wall-clock, fixed RNG seeds) and run
unconditionally.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pev_synth.api import (
    Fleet,
    _apply_query_tz,
    _normalize_freq,
    _normalize_window,
    _remap_to_cache_year,
    _resolve_region,
    _validate_profile_type,
    generate_profiles,
    regenerate_fleet,
)
from pev_synth.regions import Region

STORAGE_TZ = "UTC"
REGION_TZ = "America/Los_Angeles"

# tz-naive query window = the LA-local day Jan 2 2001. Localized to LA (PST,
# UTC-8) it spans UTC [Jan2 08:00, Jan3 08:00). The fixture stores its session
# / trip / plug data inside that UTC bracket so these natural tz-naive queries
# exercise the LA-localize branch and still hit the data.
WINDOW_START = "2001-01-02"
WINDOW_STOP = "2001-01-03"
WINDOW_UTC0 = pd.Timestamp("2001-01-02 08:00", tz=STORAGE_TZ)
WINDOW_UTC1 = pd.Timestamp("2001-01-03 08:00", tz=STORAGE_TZ)
# A window with no fixture data of any kind.
EMPTY_START = "2001-01-20"
EMPTY_STOP = "2001-01-21"

# Static fleet columns required by ``Fleet._row_for`` / ``Fleet.summary``.
_FLEET_COLS_BASE: dict[str, object] = {
    "powertrain": "BEV",
    "make": "Test",
    "model": "TestModel",
    "model_year": 2024,
    "archetype": "commuter",
    "B_kwh": 60.0,
    "OBC_kw": 7.2,
    "EVSE_home_kw": 7.2,
    "panel_cap_kw": 20.0,
    "EVSE_brand": "ChargePoint",
    "EVSE_connector": "J1772",
    "eta_charge_nominal": 0.9,
    "has_heat_pump": True,
    "Wh_per_mi_nominal": 300.0,
    "garage_type": "attached",
    "income_tier": "mid",
    "home_station": "L2",
    "seed": 1,
    "soc_min_kwh": 6.0,
    "nhts_donor_houseid": "h1",
    "nhts_donor_vehid": "v1",
    "cluster_z": 0,
}


def _row_for(ev_id: int, **overrides: object) -> dict[str, object]:
    row = {"ev_id": int(ev_id), **_FLEET_COLS_BASE}
    row.update(overrides)
    return row


def _utc(ts: str) -> pd.Timestamp:
    return pd.Timestamp(ts, tz=STORAGE_TZ)


def _write_full_bundle(target: Path, ev_ids: list[int]) -> None:
    """Write a complete bundle: fleet + plug_status + sessions + mobility.

    Every timestamp column is tz-aware UTC on the 2001 synthetic calendar,
    matching the v2.0 storage contract so the Fleet query methods slice it
    correctly.
    """
    # Vary a couple of static fields per EV so ``filter`` predicates bite.
    rows = []
    for i, ev in enumerate(ev_ids):
        rows.append(
            _row_for(
                ev,
                archetype="commuter" if i % 2 == 0 else "errand",
                B_kwh=60.0 + 10.0 * i,
                OBC_kw=7.2 if i % 2 == 0 else 11.0,
                EVSE_home_kw=7.2 if i % 2 == 0 else 11.0,
                EVSE_brand="ChargePoint" if i % 2 == 0 else "Tesla",
                EVSE_connector="J1772" if i % 2 == 0 else "NACS",
                cluster_z=i,
            )
        )
    pd.DataFrame(rows).to_parquet(target / "fleet.parquet", index=False)

    # plug_status covers a UTC range wide enough that an LA-local-day query
    # (which localizes tz-naive input to America/Los_Angeles -> UTC and so
    # lands at +08:00/+07:00) falls entirely inside the plugged span. We mark
    # the whole stored span plugged=True for WINDOW_DAY's LA day.
    idx15 = pd.date_range(
        "2001-01-02 00:00", "2001-01-04 00:00", freq="15min", tz=STORAGE_TZ,
        inclusive="left",
    )
    ps_rows = []
    for ev in ev_ids:
        for t in idx15:
            ps_rows.append({"ev_id": int(ev), "ts": t, "plugged": True})
    pd.DataFrame(ps_rows).to_parquet(
        target / "plug_status_15min.parquet", index=False
    )

    idxh = pd.date_range(
        "2001-01-02 00:00", "2001-01-04 00:00", freq="1h", tz=STORAGE_TZ,
        inclusive="left",
    )
    psh_rows = []
    for ev in ev_ids:
        for t in idxh:
            psh_rows.append({"ev_id": int(ev), "ts": t, "plugged": True})
    pd.DataFrame(psh_rows).to_parquet(
        target / "plug_status_hourly.parquet", index=False
    )

    # sessions: one session per EV inside the LA-local WINDOW_DAY
    # ([Jan2 08:00, Jan3 08:00) UTC). Place it at 10:00-14:00 UTC.
    sess_rows = []
    for ev in ev_ids:
        sess_rows.append(
            {
                "ev_id": int(ev),
                "t_in": _utc("2001-01-02 10:00"),
                "t_out": _utc("2001-01-02 14:00"),
                "soc_in_kwh": 10.0,
                "soc_target_out_kwh": 40.0,
                "max_kw": 7.2,
                "energy_kwh": 28.8,  # 4 h * 7.2 kW
            }
        )
    pd.DataFrame(sess_rows).to_parquet(
        target / "sessions.parquet", index=False
    )

    # mobility: one trip per EV inside the same UTC bracket.
    mob_rows = []
    for ev in ev_ids:
        mob_rows.append(
            {
                "ev_id": int(ev),
                "start_ts": _utc("2001-01-02 15:00"),
                "end_ts": _utc("2001-01-02 16:00"),
                "miles": 12.5,
            }
        )
    pd.DataFrame(mob_rows).to_parquet(
        target / "mobility.parquet", index=False
    )

    # plug_in_events: needed by Fleet.summary's plug_in_rate computation.
    pie_rows = []
    for ev in ev_ids:
        pie_rows.extend(
            [
                {"ev_id": int(ev), "plug_in": True},
                {"ev_id": int(ev), "plug_in": False},
            ]
        )
    pd.DataFrame(pie_rows).to_parquet(
        target / "plug_in_events.parquet", index=False
    )


def _write_meta(target: Path) -> None:
    (target / "meta.json").write_text(
        json.dumps({"profile_type": "residential", "region": "bay_area"})
    )


@pytest.fixture
def full_fleet(tmp_path: Path) -> Fleet:
    """Three-EV Fleet backed by a complete synthetic on-disk bundle."""
    _write_full_bundle(tmp_path, [10, 20, 30])
    _write_meta(tmp_path)
    return Fleet.load(tmp_path)


# ----------------------------------------------------------------------
# Helper-level branches
# ----------------------------------------------------------------------

def test_validate_profile_type_accepts_known() -> None:
    # Known types do not raise.
    _validate_profile_type("residential")
    _validate_profile_type("workplace")


def test_validate_profile_type_descoped_message() -> None:
    with pytest.raises(ValueError, match=r"fleet_depot.*de-scoped"):
        _validate_profile_type("fleet_depot")


def test_validate_profile_type_unknown_lists_valid() -> None:
    with pytest.raises(ValueError, match="unknown profile_type"):
        _validate_profile_type("spaceship")


def test_resolve_region_passthrough_instance() -> None:
    bay = Region.from_name("bay_area")
    assert _resolve_region(bay) is bay


def test_resolve_region_from_string() -> None:
    out = _resolve_region("bay_area")
    assert isinstance(out, Region)
    assert out.name == "bay_area"


def test_resolve_region_rejects_non_region_non_str() -> None:
    with pytest.raises(TypeError, match="region must be a Region or str"):
        _resolve_region(42)  # type: ignore[arg-type]


def test_normalize_freq_canonical_variants() -> None:
    assert _normalize_freq("15min") == "15min"
    assert _normalize_freq("15T") == "15min"
    assert _normalize_freq(" 15 min ") == "15min"
    assert _normalize_freq("1H") == "1h"
    assert _normalize_freq("H") == "1h"
    assert _normalize_freq("60min") == "1h"


def test_normalize_freq_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="Unsupported freq"):
        _normalize_freq("5min")


def test_remap_to_cache_year_non_leap_passthrough() -> None:
    ts = pd.Timestamp("2025-06-15 13:45", tz=STORAGE_TZ)
    out = _remap_to_cache_year(ts)
    assert out.year == 2001
    assert (out.month, out.day, out.hour, out.minute) == (6, 15, 13, 45)


def test_remap_to_cache_year_leap_day_falls_to_mar_1() -> None:
    out = _remap_to_cache_year(pd.Timestamp("2024-02-29", tz=STORAGE_TZ))
    assert (out.year, out.month, out.day) == (2001, 3, 1)


def test_normalize_window_rejects_stop_before_start() -> None:
    with pytest.raises(ValueError, match="strictly after"):
        _normalize_window("2001-06-08", "2001-06-01", region_tz=REGION_TZ)


def test_normalize_window_tz_aware_input_converted() -> None:
    # tz-aware input takes the tz_convert branch in _to_region_timestamp.
    ts0, ts1 = _normalize_window(
        _utc("2001-06-01 12:00"), _utc("2001-06-02 12:00"), region_tz=REGION_TZ
    )
    assert (ts1 - ts0) == pd.Timedelta(days=1)
    assert str(ts0.tz) == STORAGE_TZ


def test_apply_query_tz_none_is_identity() -> None:
    s = pd.Series(
        [1.0, 2.0],
        index=pd.date_range("2001-01-01", periods=2, freq="1h", tz="UTC"),
    )
    out = _apply_query_tz(s, None)
    assert out is s


def test_apply_query_tz_converts_index() -> None:
    s = pd.Series(
        [1.0, 2.0],
        index=pd.date_range("2001-01-01", periods=2, freq="1h", tz="UTC"),
    )
    out = _apply_query_tz(s, REGION_TZ)
    assert str(out.index.tz) == REGION_TZ
    # Values unchanged; original untouched (copy semantics).
    np.testing.assert_array_equal(out.values, s.values)
    assert str(s.index.tz) == "UTC"


# ----------------------------------------------------------------------
# Fleet construction / dunder / properties
# ----------------------------------------------------------------------

def test_fleet_missing_fleet_parquet_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="fleet.parquet not found"):
        Fleet(
            data_root=tmp_path,
            ev_ids=[1],
            profile_type="residential",
            region=Region.from_name("bay_area"),
        )


def test_fleet_duplicate_ev_id_raises(tmp_path: Path) -> None:
    pd.DataFrame([_row_for(1), _row_for(1)]).to_parquet(
        tmp_path / "fleet.parquet", index=False
    )
    with pytest.raises(ValueError, match="duplicated ev_id"):
        Fleet(
            data_root=tmp_path,
            ev_ids=[1],
            profile_type="residential",
            region=Region.from_name("bay_area"),
        )


def test_fleet_repr_and_len(full_fleet: Fleet) -> None:
    r = repr(full_fleet)
    assert "Fleet(" in r
    assert "residential" in r
    assert "bay_area" in r
    assert "n=3" in r
    assert len(full_fleet) == 3


def test_fleet_public_properties(full_fleet: Fleet) -> None:
    assert full_fleet.profile_type == "residential"
    assert full_fleet.region.name == "bay_area"
    assert isinstance(full_fleet.data_root, Path)
    meta = full_fleet.meta
    assert meta["profile_type"] == "residential"
    # meta returns a copy: mutating it does not affect the Fleet.
    meta["profile_type"] = "mutated"
    assert full_fleet.meta["profile_type"] == "residential"


def test_fleet_iter_and_profile_ids(full_fleet: Fleet) -> None:
    ids = [p.ev_id for p in full_fleet]
    assert ids == full_fleet.profile_ids() == [10, 20, 30]


def test_fleet_getitem_string_key(full_fleet: Fleet) -> None:
    prof = full_fleet["ev_0020"]
    assert prof.ev_id == 20


def test_fleet_getitem_string_bad_prefix(full_fleet: Fleet) -> None:
    with pytest.raises(KeyError, match="must look like"):
        full_fleet["car_0020"]


def test_fleet_getitem_string_unparseable(full_fleet: Fleet) -> None:
    with pytest.raises(KeyError, match="unparseable"):
        full_fleet["ev_xx"]


def test_fleet_getitem_string_absent(full_fleet: Fleet) -> None:
    with pytest.raises(KeyError, match="not in this Fleet"):
        full_fleet["ev_9999"]


def test_fleet_getitem_positional_and_by_id(full_fleet: Fleet) -> None:
    # 10 is an ev_id -> by-id.
    assert full_fleet[10].ev_id == 10
    # 0 is not an ev_id but a valid position -> positional (ev_id 10).
    assert full_fleet[0].ev_id == 10


def test_fleet_getitem_int_out_of_range(full_fleet: Fleet) -> None:
    with pytest.raises(IndexError, match="index out of range"):
        full_fleet[999]


def test_fleet_getitem_rejects_bad_type(full_fleet: Fleet) -> None:
    with pytest.raises(TypeError, match="must be int or str"):
        full_fleet[1.5]  # type: ignore[index]


def test_fleet_public_reader_surface(full_fleet: Fleet) -> None:
    # RFC-020 FleetReader public aliases.
    assert full_fleet.ev_ids() == [10, 20, 30]
    assert full_fleet.has_ev(20) is True
    assert full_fleet.has_ev(99) is False
    row = full_fleet.row_for(20)
    assert row.ev_id == 20
    df = full_fleet.read_parquet(
        "sessions.parquet", columns=["ev_id"], ev_ids=[20]
    )
    assert (df["ev_id"] == 20).all()


def test_fleet_isinstance_fleetreader(full_fleet: Fleet) -> None:
    from pev_synth.api import FleetReader

    assert isinstance(full_fleet, FleetReader)


def test_cache_path_flat_layout() -> None:
    p = Fleet.cache_path("bay_area", "residential")
    assert p.name == "residential_ev_synth"
    assert "bay_area" in str(p)


def test_cache_path_replicate_layout() -> None:
    p = Fleet.cache_path(
        "bay_area", "residential", replicate_id=2, r_total=4
    )
    assert "replicates" in str(p)
    assert p.name == "r2"


# ----------------------------------------------------------------------
# Profile static surface
# ----------------------------------------------------------------------

def test_profile_repr_and_static_props(full_fleet: Fleet) -> None:
    prof = full_fleet[10]
    r = repr(prof)
    assert "Profile(ev_id=10" in r
    assert prof.ev_id == 10
    assert prof.archetype == "commuter"
    assert prof.battery_kwh == pytest.approx(60.0)
    assert prof.max_charge_kw == pytest.approx(7.2)
    assert prof.eta_charge == pytest.approx(0.9)
    assert prof.soc_min_kwh == pytest.approx(6.0)
    assert prof.cluster_z == 0
    assert prof.donor_household_id == "h1"
    assert prof.donor_vehicle_id == "v1"
    assert prof.profile_type == "residential"
    assert prof.region.name == "bay_area"
    assert prof.evse_brand == "ChargePoint"
    assert prof.evse_connector == "J1772"


def test_profile_row_is_cached(full_fleet: Fleet) -> None:
    prof = full_fleet[10]
    first = prof._row
    second = prof._row
    assert first is second


def test_profile_summary_series(full_fleet: Fleet) -> None:
    s = full_fleet[10].summary()
    assert isinstance(s, pd.Series)
    assert s["ev_id"] == 10
    assert s["region"] == "bay_area"
    assert s.name == "ev_0010"


# ----------------------------------------------------------------------
# Fleet.summary (912-971)
# ----------------------------------------------------------------------

def test_fleet_summary_columns_and_values(full_fleet: Fleet) -> None:
    df = full_fleet.summary()
    assert len(df) == 3
    needed = {
        "ev_id",
        "battery_kwh",
        "max_charge_kw",
        "eta_charge",
        "archetype",
        "cluster_z",
        "evse_brand",
        "evse_connector",
        "plug_in_rate_yr",
        "n_sessions_yr",
        "miles_yr",
    }
    assert needed <= set(df.columns)
    # one True / one False event per EV -> plug_in_rate 0.5.
    assert (df["plug_in_rate_yr"] == 0.5).all()
    # one session per EV.
    assert (df["n_sessions_yr"] == 1).all()
    # one 12.5-mile trip per EV.
    assert df["miles_yr"].tolist() == pytest.approx([12.5, 12.5, 12.5])


# ----------------------------------------------------------------------
# Presence / absence data paths (1076-1108, 1128-1152)
# ----------------------------------------------------------------------

def test_fleet_presence_absence_wide(full_fleet: Fleet) -> None:
    wide = full_fleet.generate_presence_absence(
        WINDOW_START, WINDOW_STOP, freq="15min"
    )
    assert isinstance(wide, pd.DataFrame)
    assert wide.shape == (96, 3)
    assert list(wide.columns) == [10, 20, 30]
    assert str(wide.index.tz) == STORAGE_TZ
    assert wide.index[0] == WINDOW_UTC0
    # The whole LA-local day is plugged in this fixture.
    assert wide.to_numpy().all()


def test_fleet_presence_absence_hourly_and_tz(full_fleet: Fleet) -> None:
    wide = full_fleet.plug_status(
        WINDOW_START, WINDOW_STOP, freq="1h", tz=REGION_TZ
    )
    assert wide.shape == (24, 3)
    assert str(wide.index.tz) == REGION_TZ


def test_fleet_presence_absence_empty_window(full_fleet: Fleet) -> None:
    # No plug_status rows in this window -> all-False reindexed frame.
    wide = full_fleet.generate_presence_absence(
        EMPTY_START, EMPTY_STOP, freq="15min"
    )
    assert wide.shape == (96, 3)
    assert not wide.to_numpy().any()


def test_profile_presence_absence_one(full_fleet: Fleet) -> None:
    prof = full_fleet[10]
    pa = prof.generate_presence_absence(
        WINDOW_START, WINDOW_STOP, freq="15min"
    )
    assert isinstance(pa, pd.Series)
    assert pa.dtype == bool
    assert len(pa) == 96
    assert pa.all()
    assert pa.name == "ev_0010"


def test_profile_presence_absence_one_empty(full_fleet: Fleet) -> None:
    prof = full_fleet[10]
    pa = prof.generate_presence_absence(EMPTY_START, EMPTY_STOP, freq="1h")
    assert len(pa) == 24
    assert not pa.any()


def test_profile_plug_status_alias(full_fleet: Fleet) -> None:
    prof = full_fleet[10]
    a = prof.generate_presence_absence(WINDOW_START, WINDOW_STOP)
    b = prof.plug_status(WINDOW_START, WINDOW_STOP)
    pd.testing.assert_series_equal(a, b)


def test_fleet_presence_absence_one_alias(full_fleet: Fleet) -> None:
    s = full_fleet.presence_absence_one(
        10, WINDOW_START, WINDOW_STOP, freq="15min"
    )
    assert s.name == "ev_0010"
    assert s.all()


# ----------------------------------------------------------------------
# charging_sessions / trips (1167-1191, 1202-1214)
# ----------------------------------------------------------------------

def test_fleet_charging_sessions_intersection(full_fleet: Fleet) -> None:
    sess = full_fleet.charging_sessions(WINDOW_START, WINDOW_STOP)
    assert len(sess) == 3
    assert str(sess["t_in"].dt.tz) == STORAGE_TZ


def test_fleet_charging_sessions_tz_kwarg(full_fleet: Fleet) -> None:
    sess = full_fleet.charging_sessions_for(
        [10], WINDOW_START, WINDOW_STOP, tz=REGION_TZ
    )
    assert len(sess) == 1
    assert str(sess["t_in"].dt.tz) == REGION_TZ
    assert str(sess["t_out"].dt.tz) == REGION_TZ


def test_profile_charging_sessions_no_overlap(full_fleet: Fleet) -> None:
    prof = full_fleet[10]
    sess = prof.charging_sessions(EMPTY_START, EMPTY_STOP)
    assert sess.empty


def test_profile_trips_intersection(full_fleet: Fleet) -> None:
    prof = full_fleet[10]
    trips = prof.trips(WINDOW_START, WINDOW_STOP)
    assert len(trips) == 1
    assert (trips["ev_id"] == 10).all()
    assert str(trips["start_ts"].dt.tz) == STORAGE_TZ


def test_profile_trips_tz_kwarg(full_fleet: Fleet) -> None:
    prof = full_fleet[10]
    trips = prof.trips(WINDOW_START, WINDOW_STOP, tz=REGION_TZ)
    assert str(trips["start_ts"].dt.tz) == REGION_TZ
    assert str(trips["end_ts"].dt.tz) == REGION_TZ


# ----------------------------------------------------------------------
# soc_trajectory (1227-1294)
# ----------------------------------------------------------------------

def test_soc_trajectory_with_session(full_fleet: Fleet) -> None:
    prof = full_fleet[10]
    soc = prof.soc_trajectory(WINDOW_START, WINDOW_STOP, freq="1h")
    assert isinstance(soc, pd.Series)
    assert soc.dtype == float
    assert soc.name == "soc_kwh_0010"
    # Inside the 10:00-14:00 UTC session SoC ramps 10 -> 40 kWh; check monotone
    # up across the session bracket.
    seg = soc.loc[_utc("2001-01-02 10:00"):_utc("2001-01-02 14:00")]
    assert (np.diff(seg.values) >= -1e-9).all()
    assert seg.iloc[0] == pytest.approx(10.0)
    assert seg.iloc[-1] == pytest.approx(40.0)


def test_soc_trajectory_no_sessions_defaults_to_soc_min(
    tmp_path: Path,
) -> None:
    _write_full_bundle(tmp_path, [10])
    # Overwrite sessions with an empty (but schema-correct) frame.
    empty = pd.DataFrame(
        {
            "ev_id": pd.Series([], dtype="int64"),
            "t_in": pd.Series([], dtype="datetime64[ns, UTC]"),
            "t_out": pd.Series([], dtype="datetime64[ns, UTC]"),
            "soc_in_kwh": pd.Series([], dtype="float64"),
            "soc_target_out_kwh": pd.Series([], dtype="float64"),
            "max_kw": pd.Series([], dtype="float64"),
            "energy_kwh": pd.Series([], dtype="float64"),
        }
    )
    empty.to_parquet(tmp_path / "sessions.parquet", index=False)
    _write_meta(tmp_path)
    fleet = Fleet.load(tmp_path)
    soc = fleet[10].soc_trajectory(WINDOW_START, WINDOW_STOP, freq="1h")
    # soc_min_kwh fixture value is 6.0.
    assert soc.tolist() == pytest.approx([6.0] * len(soc))


def test_soc_trajectory_tz_kwarg(full_fleet: Fleet) -> None:
    soc = full_fleet[10].soc_trajectory(
        WINDOW_START, WINDOW_STOP, freq="1h", tz=REGION_TZ
    )
    assert str(soc.index.tz) == REGION_TZ


# ----------------------------------------------------------------------
# aggregate_load (1313-1371)
# ----------------------------------------------------------------------

def test_aggregate_load_kw(full_fleet: Fleet) -> None:
    agg = full_fleet.aggregate_load(WINDOW_START, WINDOW_STOP, freq="1h")
    assert isinstance(agg, pd.Series)
    assert len(agg) == 24
    assert agg.name == "aggregate_load_kw"
    # 3 EVs * 7.2 kW each during the 10:00-14:00 UTC session windows.
    assert agg.loc[_utc("2001-01-02 11:00")] == pytest.approx(3 * 7.2)
    # Before the session window, no load.
    assert agg.loc[WINDOW_UTC0] == pytest.approx(0.0)
    assert str(agg.index.tz) == STORAGE_TZ


def test_aggregate_load_empty_window(full_fleet: Fleet) -> None:
    agg = full_fleet.aggregate_load(EMPTY_START, EMPTY_STOP, freq="1h")
    assert len(agg) == 24
    assert (agg == 0.0).all()


def test_aggregate_load_tz_kwarg(full_fleet: Fleet) -> None:
    agg = full_fleet.aggregate_load(
        WINDOW_START, WINDOW_STOP, freq="15min", tz=REGION_TZ
    )
    assert str(agg.index.tz) == REGION_TZ
    assert len(agg) == 96


# ----------------------------------------------------------------------
# filter (973-1059)
# ----------------------------------------------------------------------

def test_filter_archetype_equality(full_fleet: Fleet) -> None:
    sub = full_fleet.filter(archetype="commuter")
    assert isinstance(sub, Fleet)
    # ev 10 (i=0) and ev 30 (i=2) are commuters.
    assert sub.profile_ids() == [10, 30]
    assert sub.region.name == "bay_area"


def test_filter_powertrain_and_cluster(full_fleet: Fleet) -> None:
    sub = full_fleet.filter(powertrain="BEV", cluster_z=1)
    assert sub.profile_ids() == [20]


def test_filter_evse_predicates(full_fleet: Fleet) -> None:
    sub = full_fleet.filter(evse_brand="Tesla", evse_connector="NACS")
    assert sub.profile_ids() == [20]


def test_filter_numeric_thresholds(full_fleet: Fleet) -> None:
    # B_kwh = 60, 70, 80 for ev 10/20/30.
    sub = full_fleet.filter(battery_kwh_gte=70, battery_kwh_lt=80)
    assert sub.profile_ids() == [20]
    sub2 = full_fleet.filter(battery_kwh_gt=60, battery_kwh_lte=80)
    assert sub2.profile_ids() == [20, 30]
    sub3 = full_fleet.filter(max_charge_kw_gte=11.0)
    assert sub3.profile_ids() == [20]
    sub4 = full_fleet.filter(max_charge_kw_lt=11.0)
    assert sub4.profile_ids() == [10, 30]


def test_filter_unknown_key(full_fleet: Fleet) -> None:
    with pytest.raises(KeyError, match="unsupported filter key"):
        full_fleet.filter(bogus=1)


# ----------------------------------------------------------------------
# _row_for / _read_parquet edges (1535, 1570-1578)
# ----------------------------------------------------------------------

def test_row_for_absent_ev_id_raises(full_fleet: Fleet) -> None:
    with pytest.raises(KeyError, match="not in this Fleet"):
        full_fleet.row_for(9999)


def test_read_parquet_missing_file_raises(full_fleet: Fleet) -> None:
    with pytest.raises(FileNotFoundError, match="not found in bundle"):
        full_fleet.read_parquet("does_not_exist.parquet")


def test_read_parquet_no_ev_filter_returns_all(full_fleet: Fleet) -> None:
    df = full_fleet.read_parquet("sessions.parquet", ev_ids=None)
    assert set(df["ev_id"].unique()) == {10, 20, 30}


# ----------------------------------------------------------------------
# save / load (1441-1482, 1492-1529)
# ----------------------------------------------------------------------

def test_save_load_roundtrip(full_fleet: Fleet, tmp_path: Path) -> None:
    out = full_fleet.save(tmp_path / "out")
    assert out.name == "residential_fleet"
    for fname in (
        "fleet.parquet",
        "plug_status_15min.parquet",
        "sessions.parquet",
        "mobility.parquet",
        "plug_in_events.parquet",
        "meta.json",
    ):
        assert (out / fname).exists()
    meta = json.loads((out / "meta.json").read_text())
    assert meta["profile_type"] == "residential"
    assert meta["region"] == "bay_area"
    assert meta["ev_ids"] == [10, 20, 30]
    assert meta["storage_timezone"] == STORAGE_TZ

    loaded = Fleet.load(out)
    assert loaded.profile_ids() == full_fleet.profile_ids()
    assert loaded.region.name == "bay_area"
    pd.testing.assert_frame_equal(loaded.summary(), full_fleet.summary())


def test_save_skips_missing_optional_parquet(tmp_path: Path) -> None:
    """``save`` silently skips optional artifacts that are absent in the bundle.

    The bundle here has only ``fleet.parquet`` + ``sessions.parquet`` (no
    plug_status / mobility / events), exercising the ``continue`` branch.
    """
    src = tmp_path / "src"
    src.mkdir()
    pd.DataFrame([_row_for(10)]).to_parquet(src / "fleet.parquet", index=False)
    pd.DataFrame(
        [
            {
                "ev_id": 10,
                "t_in": _utc("2001-01-02 10:00"),
                "t_out": _utc("2001-01-02 14:00"),
                "soc_in_kwh": 10.0,
                "soc_target_out_kwh": 40.0,
                "max_kw": 7.2,
                "energy_kwh": 28.8,
            }
        ]
    ).to_parquet(src / "sessions.parquet", index=False)
    fleet = Fleet(
        data_root=src,
        ev_ids=[10],
        profile_type="residential",
        region=Region.from_name("bay_area"),
    )
    out = fleet.save(tmp_path / "out")
    assert (out / "fleet.parquet").exists()
    assert (out / "sessions.parquet").exists()
    # Absent-in-source artifacts are not written.
    assert not (out / "mobility.parquet").exists()
    assert not (out / "plug_status_15min.parquet").exists()


def test_load_via_parent_picks_single_inner(
    full_fleet: Fleet, tmp_path: Path
) -> None:
    out = full_fleet.save(tmp_path / "out")
    loaded = Fleet.load(out.parent)
    assert loaded.profile_ids() == full_fleet.profile_ids()


def test_load_via_parent_ambiguous_raises(
    full_fleet: Fleet, tmp_path: Path
) -> None:
    parent = tmp_path / "two"
    parent.mkdir()
    full_fleet.save(parent)
    # Create a second *_fleet sibling so the glob is ambiguous.
    (parent / "workplace_fleet").mkdir()
    with pytest.raises(FileNotFoundError, match="ambiguous inner bundles"):
        Fleet.load(parent)


# ----------------------------------------------------------------------
# generate_profiles factory (1650-1719) via data_root override
# ----------------------------------------------------------------------

def test_generate_profiles_with_data_root(tmp_path: Path) -> None:
    _write_full_bundle(tmp_path, [10, 20, 30])
    _write_meta(tmp_path)
    fleet = generate_profiles(
        "residential", n=2, region="bay_area", seed=42, data_root=tmp_path
    )
    assert isinstance(fleet, Fleet)
    assert len(fleet) == 2
    assert set(fleet.profile_ids()) <= {10, 20, 30}


def test_generate_profiles_full_cache_size(tmp_path: Path) -> None:
    _write_full_bundle(tmp_path, [10, 20, 30])
    _write_meta(tmp_path)
    fleet = generate_profiles(
        "residential", n=3, region="bay_area", data_root=tmp_path
    )
    # n == cache_size returns all ids in sorted order (no subsetting).
    assert fleet.profile_ids() == [10, 20, 30]


def test_generate_profiles_seed_reproducible(tmp_path: Path) -> None:
    _write_full_bundle(tmp_path, [10, 20, 30])
    _write_meta(tmp_path)
    a = generate_profiles(
        "residential", n=2, region="bay_area", seed=7, data_root=tmp_path
    )
    b = generate_profiles(
        "residential", n=2, region="bay_area", seed=7, data_root=tmp_path
    )
    assert a.profile_ids() == b.profile_ids()


def test_generate_profiles_n_exceeds_cache(tmp_path: Path) -> None:
    _write_full_bundle(tmp_path, [10, 20, 30])
    _write_meta(tmp_path)
    with pytest.raises(ValueError, match="exceeds the cached fleet size"):
        generate_profiles(
            "residential", n=99, region="bay_area", data_root=tmp_path
        )


def test_generate_profiles_n_not_positive(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        generate_profiles(
            "residential", n=0, region="bay_area", data_root=tmp_path
        )


def test_generate_profiles_fleet_depot(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"fleet_depot.*de-scoped"):
        generate_profiles(
            "fleet_depot", n=1, region="bay_area", data_root=tmp_path
        )


def test_generate_profiles_missing_cache_points_at_cli(
    tmp_path: Path,
) -> None:
    # data_root that has no fleet.parquet -> FileNotFoundError w/ pointer.
    with pytest.raises(FileNotFoundError, match="cache_regen"):
        generate_profiles(
            "residential", n=1, region="bay_area", data_root=tmp_path
        )


def test_generate_profiles_corrupt_meta_falls_back(tmp_path: Path) -> None:
    _write_full_bundle(tmp_path, [10, 20, 30])
    (tmp_path / "meta.json").write_text("{not valid json")
    fleet = generate_profiles(
        "residential", n=1, region="bay_area", seed=1, data_root=tmp_path
    )
    # Corrupt meta.json is tolerated -> empty meta dict.
    assert fleet.meta == {}


def test_generate_profiles_no_meta_file(tmp_path: Path) -> None:
    _write_full_bundle(tmp_path, [10, 20, 30])
    fleet = generate_profiles(
        "residential", n=1, region="bay_area", seed=1, data_root=tmp_path
    )
    assert fleet.meta == {}


# ----------------------------------------------------------------------
# regenerate_fleet stub
# ----------------------------------------------------------------------

def test_regenerate_fleet_is_stub() -> None:
    with pytest.raises(NotImplementedError, match="regenerate_fleet is a stub"):
        regenerate_fleet("residential", n=10, region="bay_area")


def test_regenerate_fleet_validates_type_first() -> None:
    with pytest.raises(ValueError, match=r"fleet_depot.*de-scoped"):
        regenerate_fleet("fleet_depot", n=10, region="bay_area")


# ----------------------------------------------------------------------
# workplace construction warning (data path through Fleet.__init__)
# ----------------------------------------------------------------------

def test_workplace_fleet_warns(tmp_path: Path) -> None:
    _write_full_bundle(tmp_path, [10])
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        Fleet(
            data_root=tmp_path,
            ev_ids=[10],
            profile_type="workplace",
            region=Region.from_name("bay_area"),
        )
    assert any(
        issubclass(w.category, RuntimeWarning)
        and "Workplace" in str(w.message)
        for w in rec
    )
