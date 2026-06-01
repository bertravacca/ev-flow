"""Unit tests for the `pev_synth` library API (v2.0).

Backed by the v2.0 UTC-storage bay_area cache at
``data/pev/processed/bay_area/residential_ev_synth/``.

v2.0 contract (vs v1.1):
  * Storage tz: UTC (every cached timestamp column).
  * Query tz: ``tz=None`` (default) → UTC index; pass ``tz=region.tz``
    (or any IANA zone) for a local wall-clock index.
  * Region routing: ``generate_profiles(profile_type, n, region=...)``
    accepts a string name or a Region instance.
  * ``fleet_depot`` was de-scoped (plan §2.6); calling
    ``generate_profiles('fleet_depot', ...)`` raises ``ValueError``.

Tests:
  * (≥ 8 new) — region routing, tz query parameter, fleet_depot ValueError,
    list_profile_types alignment, FileNotFoundError pointer.
  * (carry-forward from v1.1) — generate_profiles seed reproducibility,
    profile attributes, sessions intersection filter, soc trajectory
    monotonicity, aggregate load, filter, save/load round-trip,
    optimizer integration smoke, year remap.
  * (updated) — ``test_list_profile_types_returns_residential_and_workplace``
    replaces the stale v1.1 ``test_list_profile_types_registered`` that
    asserted ``fleet_depot`` was in the list.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make ``code/`` importable when tests are launched from anywhere.
_CODE_DIR = Path(__file__).resolve().parents[1]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import pev_synth as ps  # noqa: E402
from pev_synth.api import (  # noqa: E402
    _CACHE_YEAR,
    _DEFAULT_SEED,
    _normalize_window,
    _remap_to_cache_year,
    _to_region_timestamp,
)
from pev_synth.regions import Region  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
V20_BAY_AREA_ROOT = (
    REPO_ROOT
    / "data"
    / "pev"
    / "processed"
    / "bay_area"
    / "residential_ev_synth"
)
FLEET_PARQUET = V20_BAY_AREA_ROOT / "fleet.parquet"

pytestmark = pytest.mark.skipif(
    not FLEET_PARQUET.exists(),
    reason=f"v2.0 bay_area residential cache not present at {FLEET_PARQUET}",
)

REGION_TZ = "America/Los_Angeles"
STORAGE_TZ = "UTC"


# ---------------------------------------------------------------------------
# Module-scoped fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_fleet() -> ps.Fleet:
    """A reusable 25-EV subset (cheap enough for repeated method calls)."""
    return ps.generate_profiles(
        "residential", n=25, region="bay_area", seed=42
    )


@pytest.fixture(scope="module")
def fleet_full_meta() -> pd.DataFrame:
    """Raw fleet.parquet -- used to cross-check static attributes."""
    return pd.read_parquet(FLEET_PARQUET)


# ---------------------------------------------------------------------------
# Top-level API: list_*, generate_profiles signature, fleet_depot
# ---------------------------------------------------------------------------

def test_list_profile_types_returns_residential_and_workplace() -> None:
    """v2.0: ``fleet_depot`` was de-scoped per plan §2.6.

    Replaces the stale v1.1 ``test_list_profile_types_registered`` which
    asserted ``fleet_depot`` was in the list.
    """
    types = ps.list_profile_types()
    assert types == ["residential", "workplace"]
    assert "fleet_depot" not in types


def test_list_regions_returns_8_names() -> None:
    names = ps.list_regions()
    assert len(names) == 8
    assert "bay_area" in names
    assert "dallas_fort_worth" in names
    assert "us_national" in names


def test_region_top_level_export() -> None:
    """``Region`` and ``list_regions`` must be importable from ``pev_synth``."""
    from pev_synth import Region as ExportedRegion
    from pev_synth import list_regions as exported_list_regions

    assert ExportedRegion is Region
    assert "bay_area" in exported_list_regions()


def test_generate_profiles_residential_default_region() -> None:
    """Default region='bay_area' must resolve and load the v2.0 cache."""
    fleet = ps.generate_profiles("residential", n=10)
    assert isinstance(fleet, ps.Fleet)
    assert len(fleet) == 10
    assert fleet.profile_type == "residential"
    assert isinstance(fleet.region, Region)
    assert fleet.region.name == "bay_area"


def test_generate_profiles_region_string_or_instance() -> None:
    """region= accepts both a Region instance and a string short-name."""
    bay = Region.from_name("bay_area")
    by_str = ps.generate_profiles(
        "residential", n=5, region="bay_area", seed=42
    )
    by_inst = ps.generate_profiles(
        "residential", n=5, region=bay, seed=42
    )
    assert by_str.profile_ids() == by_inst.profile_ids()
    assert by_str.region.name == by_inst.region.name == "bay_area"


def test_generate_profiles_unknown_region_filenotfound(
    tmp_path: Path,
) -> None:
    """Missing (region, profile_type) cache → FileNotFoundError w/ pointer.

    The original v1.1 fixture used ``chicago`` as a proxy for "no cache on
    disk", but Dev I's M10 batch job has since generated chicago caches in
    the repo, invalidating the precondition. Use a ``tmp_path``-isolated
    ``data_root`` so no cache exists for ANY region. The error message must
    point at ``cache_regen`` so the caller can self-correct.
    """
    with pytest.raises(FileNotFoundError, match="cache_regen"):
        ps.generate_profiles(
            "residential", n=10, region="chicago", data_root=tmp_path
        )


def test_generate_profiles_fleet_depot_raises_valueerror() -> None:
    """fleet_depot is de-scoped in v2.0 (plan §2.6)."""
    with pytest.raises(ValueError, match=r"fleet_depot.*de-scoped"):
        ps.generate_profiles("fleet_depot", n=10, region="bay_area")


def test_generate_profiles_unknown_type_raises() -> None:
    with pytest.raises(ValueError, match="unknown profile_type"):
        ps.generate_profiles("foobar", n=10, region="bay_area")


def test_generate_profiles_seed_reproducible() -> None:
    a = ps.generate_profiles(
        "residential", n=50, region="bay_area", seed=2026
    )
    b = ps.generate_profiles(
        "residential", n=50, region="bay_area", seed=2026
    )
    assert a.profile_ids() == b.profile_ids()
    c = ps.generate_profiles(
        "residential", n=50, region="bay_area", seed=2027
    )
    assert a.profile_ids() != c.profile_ids()


def test_generate_profiles_subset_n_lt_cached() -> None:
    fleet = ps.generate_profiles(
        "residential", n=200, region="bay_area", seed=42
    )
    assert len(fleet) == 200
    assert len(set(fleet.profile_ids())) == 200


def test_generate_profiles_raises_on_n_gt_cached() -> None:
    with pytest.raises(ValueError, match="regenerate_fleet"):
        ps.generate_profiles(
            "residential", n=1001, region="bay_area", seed=42
        )


def test_n_too_small_raises() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        ps.generate_profiles("residential", n=0, region="bay_area")


def test_default_seed_is_v2_master_seed() -> None:
    """The v2.0 default master seed is 20260520 per the orchestrator."""
    assert _DEFAULT_SEED == 20260520


# ---------------------------------------------------------------------------
# Fleet region attribute
# ---------------------------------------------------------------------------

def test_fleet_region_attribute_is_region_instance(
    small_fleet: ps.Fleet,
) -> None:
    assert isinstance(small_fleet.region, Region)
    assert small_fleet.region.name == "bay_area"
    assert small_fleet.region.tz == "America/Los_Angeles"


def test_profile_inherits_region_from_fleet(small_fleet: ps.Fleet) -> None:
    prof = small_fleet[0]
    assert isinstance(prof.region, Region)
    assert prof.region.name == "bay_area"


# ---------------------------------------------------------------------------
# Profile static attributes
# ---------------------------------------------------------------------------

def test_profile_attributes_match_fleet_parquet(
    small_fleet: ps.Fleet, fleet_full_meta: pd.DataFrame
) -> None:
    prof = small_fleet[0]
    truth = fleet_full_meta.set_index("ev_id").loc[prof.ev_id]

    assert prof.archetype == str(truth["archetype"])
    assert prof.battery_kwh == pytest.approx(
        float(truth["B_kwh"]), rel=1e-5
    )
    assert prof.eta_charge == pytest.approx(
        float(truth["eta_charge_nominal"]), rel=1e-5
    )
    assert prof.soc_min_kwh == pytest.approx(
        float(truth["soc_min_kwh"]), rel=1e-5
    )
    assert prof.max_charge_kw == pytest.approx(
        float(min(truth["OBC_kw"], truth["EVSE_home_kw"])), rel=1e-5
    )
    assert prof.cluster_z == int(truth["cluster_z"])
    assert prof.profile_type == "residential"


def test_profile_summary_is_series(small_fleet: ps.Fleet) -> None:
    s = small_fleet[0].summary()
    assert isinstance(s, pd.Series)
    needed = {
        "ev_id",
        "archetype",
        "battery_kwh",
        "max_charge_kw",
        "eta_charge",
        "soc_min_kwh",
        "cluster_z",
        "donor_household_id",
        "donor_vehicle_id",
        "profile_type",
        "region",
    }
    assert needed <= set(s.index)
    assert s["region"] == "bay_area"


def test_fleet_summary_is_dataframe(small_fleet: ps.Fleet) -> None:
    df = small_fleet.summary()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == len(small_fleet)
    needed_cols = {
        "ev_id",
        "battery_kwh",
        "max_charge_kw",
        "eta_charge",
        "archetype",
        "cluster_z",
        "plug_in_rate_yr",
        "n_sessions_yr",
        "miles_yr",
    }
    assert needed_cols <= set(df.columns)
    assert (df["plug_in_rate_yr"] >= 0).all()
    assert (df["plug_in_rate_yr"] <= 1).all()
    assert (df["miles_yr"] >= 0).all()


# ---------------------------------------------------------------------------
# Presence / absence — tz contract
# ---------------------------------------------------------------------------

def test_presence_absence_default_tz_utc(small_fleet: ps.Fleet) -> None:
    """v2.0: ``tz=None`` (default) → UTC index."""
    prof = small_fleet[0]
    pa = prof.generate_presence_absence(
        "2001-01-01", "2001-01-08", freq="15min"
    )
    assert isinstance(pa, pd.Series)
    assert pa.dtype == bool
    assert pa.index.tz is not None
    assert str(pa.index.tz) == STORAGE_TZ


def test_presence_absence_tz_la_conversion(small_fleet: ps.Fleet) -> None:
    """Same data, LA-zone index when tz='America/Los_Angeles'."""
    prof = small_fleet[0]
    pa_utc = prof.generate_presence_absence(
        "2001-01-01", "2001-01-08", freq="15min", tz=None
    )
    pa_la = prof.generate_presence_absence(
        "2001-01-01", "2001-01-08", freq="15min", tz=REGION_TZ
    )
    # Same number of slots, same boolean values, different tz labels.
    assert len(pa_utc) == len(pa_la)
    assert str(pa_la.index.tz) == REGION_TZ
    assert str(pa_utc.index.tz) == STORAGE_TZ
    np.testing.assert_array_equal(pa_utc.values, pa_la.values)
    # Index instants are identical (just labelled differently).
    assert (pa_utc.index == pa_la.index).all()


def test_presence_absence_tz_region_attribute(
    small_fleet: ps.Fleet,
) -> None:
    """Passing ``tz=region.tz`` returns the region-local index."""
    prof = small_fleet[0]
    pa = prof.generate_presence_absence(
        "2001-01-01",
        "2001-01-02",
        freq="15min",
        tz=small_fleet.region.tz,
    )
    assert str(pa.index.tz) == small_fleet.region.tz


def test_presence_absence_window_length_correct(
    small_fleet: ps.Fleet,
) -> None:
    prof = small_fleet[0]
    pa_15 = prof.generate_presence_absence(
        "2001-01-01", "2001-01-08", freq="15min"
    )
    # 7 days * 24 h * 4 = 672 slots
    assert len(pa_15) == 7 * 24 * 4
    pa_h = prof.generate_presence_absence(
        "2001-01-01", "2001-01-08", freq="1h"
    )
    assert len(pa_h) == 7 * 24


def test_plug_status_alias_matches_canonical(
    small_fleet: ps.Fleet,
) -> None:
    prof = small_fleet[0]
    pa1 = prof.generate_presence_absence("2001-01-01", "2001-01-02")
    pa2 = prof.plug_status("2001-01-01", "2001-01-02")
    pd.testing.assert_series_equal(pa1, pa2)


def test_fleet_presence_absence_wide_dataframe(
    small_fleet: ps.Fleet,
) -> None:
    wide = small_fleet.generate_presence_absence(
        "2001-01-01", "2001-01-02", freq="15min"
    )
    assert isinstance(wide, pd.DataFrame)
    assert wide.shape == (96, len(small_fleet))
    assert all(dt == np.dtype(bool) for dt in wide.dtypes)
    assert wide.index.tz is not None
    assert str(wide.index.tz) == STORAGE_TZ
    assert list(wide.columns) == small_fleet.profile_ids()


def test_fleet_presence_absence_tz_kwarg(small_fleet: ps.Fleet) -> None:
    """Fleet-level wide DataFrame respects ``tz=`` like the Profile path."""
    wide_utc = small_fleet.generate_presence_absence(
        "2001-01-01", "2001-01-02", freq="15min"
    )
    wide_la = small_fleet.generate_presence_absence(
        "2001-01-01", "2001-01-02", freq="15min", tz=REGION_TZ
    )
    assert str(wide_la.index.tz) == REGION_TZ
    np.testing.assert_array_equal(wide_utc.values, wide_la.values)


# ---------------------------------------------------------------------------
# Sessions / SoC / trips
# ---------------------------------------------------------------------------

def test_charging_sessions_intersection_filter(
    small_fleet: ps.Fleet,
) -> None:
    prof = small_fleet[0]
    ts0 = pd.Timestamp("2001-06-01", tz=STORAGE_TZ)
    ts1 = pd.Timestamp("2001-06-08", tz=STORAGE_TZ)
    sess = prof.charging_sessions(ts0, ts1)
    assert isinstance(sess, pd.DataFrame)
    # Half-open intersection: t_out > ts0 AND t_in < ts1
    assert (sess["t_out"] > ts0).all()
    assert (sess["t_in"] < ts1).all()
    if not sess.empty:
        assert (sess["ev_id"] == prof.ev_id).all()
        # Default tz → UTC.
        assert str(sess["t_in"].dt.tz) == STORAGE_TZ


def test_charging_sessions_tz_kwarg_converts_columns(
    small_fleet: ps.Fleet,
) -> None:
    prof = small_fleet[0]
    sess_la = prof.charging_sessions(
        "2001-06-01", "2001-06-08", tz=REGION_TZ
    )
    if not sess_la.empty:
        assert str(sess_la["t_in"].dt.tz) == REGION_TZ
        assert str(sess_la["t_out"].dt.tz) == REGION_TZ


def test_soc_trajectory_monotone_within_session(
    small_fleet: ps.Fleet,
) -> None:
    prof = small_fleet[0]
    soc = prof.soc_trajectory("2001-01-01", "2001-01-08", freq="15min")
    assert isinstance(soc, pd.Series)
    assert soc.dtype == float
    sess = prof.charging_sessions("2001-01-01", "2001-01-08")
    assert not sess.empty, "Expected at least one session in test window"
    for _, row in sess.iterrows():
        t_in = row["t_in"]
        t_out = row["t_out"]
        if t_in < soc.index[0] or t_out > soc.index[-1]:
            continue
        in_pos = int(
            np.searchsorted(
                soc.index.values,
                np.datetime64(t_in.tz_convert("UTC").tz_localize(None)),
            )
        )
        out_pos = int(
            np.searchsorted(
                soc.index.values,
                np.datetime64(
                    t_out.tz_convert("UTC").tz_localize(None)
                ),
            )
        )
        if out_pos <= in_pos:
            continue
        seg = soc.values[in_pos : out_pos + 1]
        diffs = np.diff(seg)
        assert (diffs >= -1e-3).all(), (
            f"SoC decreased mid-session for {row.to_dict()}"
        )
        break


def test_trips_intersection(small_fleet: ps.Fleet) -> None:
    prof = small_fleet[0]
    trips = prof.trips("2001-01-01", "2001-01-08")
    assert isinstance(trips, pd.DataFrame)
    if not trips.empty:
        assert (trips["ev_id"] == prof.ev_id).all()
        # tz-naive user input is localised to the region tz (LA), then
        # converted to UTC for the slice. 2001-01-01 00:00 LA == UTC 08:00.
        ts0 = pd.Timestamp("2001-01-01", tz=REGION_TZ).tz_convert("UTC")
        ts1 = pd.Timestamp("2001-01-08", tz=REGION_TZ).tz_convert("UTC")
        assert (trips["end_ts"] > ts0).all()
        assert (trips["start_ts"] < ts1).all()


# ---------------------------------------------------------------------------
# Aggregate load
# ---------------------------------------------------------------------------

def test_aggregate_load_in_kw(small_fleet: ps.Fleet) -> None:
    agg = small_fleet.aggregate_load(
        "2001-01-01", "2001-01-02", freq="1h"
    )
    assert isinstance(agg, pd.Series)
    assert len(agg) == 24
    assert agg.dtype == float
    assert (agg >= -1e-9).all()
    total_cap = float(
        sum(small_fleet[i].max_charge_kw for i in range(len(small_fleet)))
    )
    assert (agg <= total_cap + 1e-6).all()
    assert agg.max() > 0.0
    # Default tz → UTC.
    assert str(agg.index.tz) == STORAGE_TZ


def test_aggregate_load_tz_kwarg(small_fleet: ps.Fleet) -> None:
    agg_la = small_fleet.aggregate_load(
        "2001-01-01", "2001-01-02", freq="1h", tz=REGION_TZ
    )
    assert str(agg_la.index.tz) == REGION_TZ


# ---------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------

def test_filter_archetype(small_fleet: ps.Fleet) -> None:
    df = small_fleet.summary()
    chosen_arch = df["archetype"].mode().iloc[0]
    sub = small_fleet.filter(archetype=chosen_arch)
    assert isinstance(sub, ps.Fleet)
    assert 0 < len(sub) <= len(small_fleet)
    sub_df = sub.summary()
    assert (sub_df["archetype"] == chosen_arch).all()
    # Filter preserves the region attribute.
    assert sub.region.name == small_fleet.region.name


def test_filter_battery_range(small_fleet: ps.Fleet) -> None:
    sub = small_fleet.filter(battery_kwh_gte=50, battery_kwh_lt=90)
    if len(sub) > 0:
        df = sub.summary()
        assert (df["battery_kwh"] >= 50).all()
        assert (df["battery_kwh"] < 90).all()


def test_filter_unknown_key_raises(small_fleet: ps.Fleet) -> None:
    with pytest.raises(KeyError, match="unsupported"):
        small_fleet.filter(nonexistent_key=1)


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def test_save_load_roundtrip(
    small_fleet: ps.Fleet, tmp_path: Path
) -> None:
    out = small_fleet.save(tmp_path)
    assert (out / "fleet.parquet").exists()
    assert (out / "sessions.parquet").exists()
    assert (out / "plug_status_15min.parquet").exists()
    assert (out / "meta.json").exists()

    loaded = ps.Fleet.load(out)
    assert isinstance(loaded, ps.Fleet)
    assert loaded.profile_type == small_fleet.profile_type
    assert loaded.profile_ids() == small_fleet.profile_ids()
    # Region round-trips via meta.json.
    assert loaded.region.name == small_fleet.region.name

    a = small_fleet.summary()
    b = loaded.summary()
    assert a.equals(b)

    pa_a = small_fleet[0].generate_presence_absence(
        "2001-01-01", "2001-01-02"
    )
    pa_b = loaded[small_fleet[0].ev_id].generate_presence_absence(
        "2001-01-01", "2001-01-02"
    )
    pd.testing.assert_series_equal(pa_a, pa_b, check_names=False)


def test_save_load_via_parent_dir(
    small_fleet: ps.Fleet, tmp_path: Path
) -> None:
    """``Fleet.load(parent)`` should pick the only ``*_fleet`` child."""
    out = small_fleet.save(tmp_path)
    loaded = ps.Fleet.load(out.parent)
    assert loaded.profile_ids() == small_fleet.profile_ids()


# ---------------------------------------------------------------------------
# Optimizer integration (smoke)
# ---------------------------------------------------------------------------

def test_to_optimizer_inputs_shape(small_fleet: ps.Fleet) -> None:
    pytest.importorskip("pev_optim", reason="in-house optimizer bridge")
    ev = small_fleet.profile_ids()[0]
    oi = small_fleet._to_optimizer_inputs(
        [ev], t_start="2001-01-01", t_stop="2001-01-08"
    )
    assert set(oi.keys()) == {ev}
    inp = oi[ev]
    assert set(inp.keys()) == {"archetype", "plug_in", "sessions"}
    assert isinstance(inp["plug_in"], pd.Series)
    assert inp["plug_in"].dtype == bool
    assert isinstance(inp["sessions"], pd.DataFrame)
    assert {"t_in", "t_out", "soc_in_kwh", "soc_target_out_kwh"} <= set(
        inp["sessions"].columns
    )


def test_to_optimizer_inputs_utc_plug_in(small_fleet: ps.Fleet) -> None:
    """v2.0 acceptance criterion #9: optimizer plug_in series is UTC-indexed."""
    pytest.importorskip("pev_optim", reason="in-house optimizer bridge")
    ev = small_fleet.profile_ids()[0]
    oi = small_fleet._to_optimizer_inputs(
        [ev], t_start="2001-01-01", t_stop="2001-01-08"
    )
    inp = oi[ev]
    assert str(inp["plug_in"].index.tz) == STORAGE_TZ


def test_to_optimizer_inputs_loads_in_evoptimizer(
    small_fleet: ps.Fleet,
) -> None:
    """Smoke test: ``EVOptimizer.__init__`` accepts our v2.0 UTC dicts."""
    pytest.importorskip("pev_optim", reason="in-house optimizer bridge")
    pytest.importorskip("caiso_data", reason="in-house price feed")
    from caiso_data.prices import CaisoPrices
    from pev_optim import EVOptimizer

    ev = small_fleet.profile_ids()[0]
    oi = small_fleet._to_optimizer_inputs(
        [ev], t_start="2001-01-01", t_stop="2001-01-08"
    )
    inp = oi[ev]
    idx = inp["plug_in"].index
    n = len(idx)
    prices = CaisoPrices(
        da_lmp=pd.Series(np.full(n, 30.0), index=idx),
        rt_lmp=pd.Series(np.full(n, 30.0), index=idx),
        regup_price=pd.Series(np.zeros(n), index=idx),
        spin_price=pd.Series(np.full(n, 5.0), index=idx),
        nonspin_price=pd.Series(np.full(n, 3.0), index=idx),
        regup_mileage_price=pd.Series(np.zeros(n), index=idx),
        regup_mileage_multiplier=pd.Series(np.ones(n), index=idx),
    )
    opt = EVOptimizer(
        archetype=inp["archetype"],
        plug_in=inp["plug_in"],
        sessions=inp["sessions"],
        prices=prices,
    )
    assert opt.archetype.battery_kwh > 0
    assert len(opt.plug_in) == n


# ---------------------------------------------------------------------------
# Year remap
# ---------------------------------------------------------------------------

def test_year_remap_2025_still_works_with_utc_storage(
    small_fleet: ps.Fleet,
) -> None:
    """Passing 2025-06-* timestamps still remaps onto the cache year (2001)
    even though storage tz is UTC."""
    prof = small_fleet[0]
    pa = prof.generate_presence_absence(
        "2025-06-01", "2025-06-08", freq="15min"
    )
    assert len(pa) == 7 * 24 * 4
    # Default tz is UTC; the LT June 1 maps to LT 2001-06-01 in LA, which
    # is 2001-06-01 07:00 UTC (PDT is UTC-7).
    assert pa.index[0].year == _CACHE_YEAR
    pa_native = prof.generate_presence_absence(
        f"{_CACHE_YEAR}-06-01", f"{_CACHE_YEAR}-06-08", freq="15min"
    )
    pd.testing.assert_series_equal(
        pa.reset_index(drop=True),
        pa_native.reset_index(drop=True),
        check_names=False,
    )


def test_remap_leap_day() -> None:
    """Feb 29 of a leap year is remapped to Mar 1 of the non-leap cache year."""
    leap = pd.Timestamp("2024-02-29", tz=STORAGE_TZ)
    remapped = _remap_to_cache_year(leap)
    assert remapped.year == _CACHE_YEAR
    assert (remapped.month, remapped.day) == (3, 1)


def test_normalize_window_preserves_duration() -> None:
    """A 7-day 2025 window should still be 7 days after remap."""
    ts0, ts1 = _normalize_window(
        "2025-06-01", "2025-06-08", region_tz=REGION_TZ
    )
    assert (ts1 - ts0) == pd.Timedelta(days=7)
    # Both endpoints land in cache year (UTC).
    assert ts0.year == _CACHE_YEAR
    assert ts1.year == _CACHE_YEAR
    assert str(ts0.tz) == STORAGE_TZ
    assert str(ts1.tz) == STORAGE_TZ


def test_normalize_window_tz_naive_in_region_tz() -> None:
    """tz-naive input → localised to region.tz → converted to UTC.

    2001-01-01 00:00 LT (PST, UTC-8) == 2001-01-01 08:00 UTC.
    """
    ts0, _ = _normalize_window(
        "2001-01-01", "2001-01-02", region_tz="America/Los_Angeles"
    )
    assert ts0 == pd.Timestamp("2001-01-01 08:00:00", tz="UTC")


def test_normalize_window_year_boundary() -> None:
    """RFC-003 / M3: a window crossing a year boundary must not silently
    return garbage.

    The independent per-endpoint year remap collapses
    ``(2001-12-31, 2002-01-08)`` to ``(2001-12-31, 2001-01-08)``, which
    inverts the window. v2.0 raises ``ValueError`` telling the caller to
    split the window across the calendar boundary (see the "Inverted
    post-remap window" note in docs/pev_synth_api.md), rather than the
    older behaviour of silently returning empty/garbage frames.
    """
    with pytest.raises(ValueError, match="inverts the requested window"):
        _normalize_window(
            "2001-12-31", "2002-01-08", region_tz="America/Los_Angeles"
        )


def test_to_region_timestamp_dst_ambiguous_raises_value_error() -> None:
    """RFC-002: a fall-back-window wall-clock must raise ``ValueError``.

    2001-10-28 01:30 in LA is ambiguous (PDT-end → PST). The v1.1 code
    used ``ambiguous='NaT'`` and silently returned ``NaT``; v2.0 must
    raise with a clear message.
    """
    with pytest.raises(ValueError, match="ambiguous"):
        _to_region_timestamp("2001-10-28 01:30:00", "America/Los_Angeles")


def test_to_region_timestamp_dst_nonexistent_raises_value_error() -> None:
    """RFC-002: a spring-forward gap wall-clock must raise ``ValueError``."""
    with pytest.raises(ValueError, match="non-existent"):
        _to_region_timestamp("2001-04-01 02:30:00", "America/Los_Angeles")


# ---------------------------------------------------------------------------
# Misc behaviour
# ---------------------------------------------------------------------------

def test_iteration_yields_profiles(small_fleet: ps.Fleet) -> None:
    found_ids = [p.ev_id for p in small_fleet]
    assert found_ids == small_fleet.profile_ids()
