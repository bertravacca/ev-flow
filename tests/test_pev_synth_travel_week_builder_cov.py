"""Coverage-focused unit tests for M4 — `pev_synth.travel_week_builder`.

These tests are deliberately DATA-INDEPENDENT: they fabricate tiny
NHTS-shaped trip / household frames in memory (or write minimal fixture
CSVs / parquets into ``tmp_path``) so they run and pass with the heavy
``data/`` tree absent.  They exercise the pure helpers, the donor-pool
builders, the per-EV week samplers, the trip-materialisation /
calendar-stitching logic, the end-to-end ``build_mobility`` driver, the
``run`` orchestrator, the persistence helpers, and the CLI ``main`` /
argument parser.

They complement (do not replace) ``test_pev_synth_travel_week_builder.py``,
whose acceptance tests are gated on the real NHTS inputs and skip on CI.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pev_synth import travel_week_builder as twb
from pev_synth.regions import REGIONS

SEED = twb.MASTER_SEED_DEFAULT


# --------------------------------------------------------------------------- #
# In-memory NHTS-shaped fixtures.
# --------------------------------------------------------------------------- #


def _make_trip_frame() -> pd.DataFrame:
    """A small NHTS-shaped trip frame: 3 HHs, several travdays each.

    Column names are UPPER-case (as in raw NHTS) to exercise the
    lower-casing path in ``_normalise_trip_columns``.
    """
    rows = []
    # HH 1: two travdays (weekday 3=Tue + weekend 7=Sat), one person.
    rows += [
        (1, 1, 3, 1, 800, 830, 5.0, 1, 3, 10, 200.0),   # home->work
        (1, 1, 3, 2, 1700, 1730, 5.0, 3, 1, 10, 200.0),  # work->home
        (1, 1, 7, 1, 1000, 1030, 8.0, 1, 6, 5, 200.0),   # home->shop (weekend)
    ]
    # HH 2: single weekday (4=Wed), no work trips.
    rows += [
        (2, 1, 4, 1, 900, 930, 3.0, 1, 6, 5, 150.0),
        (2, 1, 4, 2, 1200, 1230, 3.0, 6, 1, 5, 150.0),
    ]
    # HH 3: single weekend (1=Sun), with a work trip (WHYTO 4 = work meeting).
    rows += [
        (3, 1, 1, 1, 700, 800, 12.0, 1, 4, 10, 300.0),
    ]
    cols = [
        "HOUSEID", "PERSONID", "TRAVDAY", "TDTRPNUM", "STRTTIME", "ENDTIME",
        "TRPMILES", "WHYFROM", "WHYTO", "WHYTRP1S", "WTTRDFIN",
    ]
    return pd.DataFrame(rows, columns=cols)


def _make_hh_frame() -> pd.DataFrame:
    """Matching HH frame with NHTS strata columns (UPPER-case)."""
    return pd.DataFrame(
        {
            "HOUSEID": [1, 2, 3],
            "HHSTATE": ["CA", "CA", "CA"],
            "HHFAMINC": [6, 9, 11],
            "HHSIZE": [2, 4, 1],
            "HHVEHCNT": [2, 3, 1],
            "URBRUR": [1, 1, 2],
            # All in a bay_area CBSA (41860) so CBSA narrowing keeps every HH.
            "HH_CBSA": ["41860", "41860", "41860"],
        }
    )


def _make_fleet(n: int = 3, *, has_heat_pump: bool = False) -> pd.DataFrame:
    """A tiny fleet whose donor houseids exist in ``_make_hh_frame``."""
    houseids = np.array([1, 2, 3], dtype=np.int64)
    picks = houseids[np.arange(n) % len(houseids)]
    return pd.DataFrame(
        {
            "ev_id": np.arange(n, dtype=np.int64),
            "Wh_per_mi_nominal": np.full(n, 300.0),
            "nhts_donor_houseid": picks,
            "nhts_donor_vehid": np.ones(n, dtype=np.int64),
            "has_heat_pump": np.full(n, has_heat_pump, dtype=bool),
        }
    )


# --------------------------------------------------------------------------- #
# Pure helpers.
# --------------------------------------------------------------------------- #


def test_yuksel_michalek_uplift_branches() -> None:
    # At T_REF the polynomial is exactly 1.0.
    assert twb.yuksel_michalek_uplift(twb.YUKSEL_MICHALEK_T_REF_C) == pytest.approx(1.0)
    # None -> 1.0 (no temperature).
    assert twb.yuksel_michalek_uplift(None) == pytest.approx(1.0)
    # Cold side: 0 °C -> 1 + 0.018 * 20.
    assert twb.yuksel_michalek_uplift(0.0) == pytest.approx(1.0 + 0.018 * 20.0)
    # Hot side: 40 °C -> 1 + 0.006 * 20 (no clamp).
    assert twb.yuksel_michalek_uplift(40.0) == pytest.approx(1.0 + 0.006 * 20.0)
    assert twb.yuksel_michalek_uplift(40.0) > 1.0


def test_resolve_winter_t_c_prefers_region_attr_then_table_then_none() -> None:
    # boston carries winter_t_c = -2.0 directly.
    assert twb._resolve_winter_t_c(REGIONS["boston"]) == pytest.approx(-2.0)
    # bay_area has winter_t_c=None and is absent from WINTER_T_C_BY_REGION.
    assert twb._resolve_winter_t_c(REGIONS["bay_area"]) is None
    # dallas_fort_worth has winter_t_c=None but IS in the legacy table -> fallback.
    assert REGIONS["dallas_fort_worth"].winter_t_c is None
    assert "dallas_fort_worth" in twb.WINTER_T_C_BY_REGION
    assert twb._resolve_winter_t_c(REGIONS["dallas_fort_worth"]) == pytest.approx(
        twb.WINTER_T_C_BY_REGION["dallas_fort_worth"]
    )


def test_resolve_region_accepts_object_str_and_rejects_other() -> None:
    region = REGIONS["bay_area"]
    assert twb._resolve_region(region) is region
    assert twb._resolve_region("boston").name == "boston"
    with pytest.raises(TypeError, match="region must be Region or str"):
        twb._resolve_region(123)  # type: ignore[arg-type]


def test_default_path_helpers() -> None:
    mp = twb.default_mobility_path("bay_area", "residential")
    fp = twb.default_fleet_path("chicago", "workplace")
    metap = twb.default_meta_path("boston", "residential")
    assert mp.as_posix().endswith(
        "data/pev/processed/bay_area/residential_ev_synth/mobility.parquet"
    )
    assert fp.as_posix().endswith(
        "data/pev/processed/chicago/workplace_ev_synth/fleet.parquet"
    )
    assert metap.as_posix().endswith(
        "data/pev/processed/boston/residential_ev_synth/meta.json"
    )


def test_normalise_trip_columns_derives_whytrp1s_when_absent() -> None:
    df = pd.DataFrame(
        {"HOUSEID": [1, 2, 3], "WHYTO": [3, 1, 6]}
    )
    out = twb._normalise_trip_columns(df)
    assert "whytrp1s" in out.columns
    assert out.attrs["whytrp1s_source"] == "derived_from_whyto"
    # WHYTO 3 -> work(10); 1 -> home(1); 6 -> other(97).
    assert list(out["whytrp1s"].astype(int)) == [10, 1, 97]


def test_normalise_trip_columns_keeps_native_whytrp1s() -> None:
    df = pd.DataFrame({"HOUSEID": [1], "WHYTRP1S": [10]})
    out = twb._normalise_trip_columns(df)
    assert out.attrs["whytrp1s_source"] == "native"
    assert out["whytrp1s"].iloc[0] == 10


def test_normalise_hh_columns_coerces_cbsa_to_string() -> None:
    df = pd.DataFrame({"HOUSEID": [1], "HH_CBSA": [41860]})
    out = twb._normalise_hh_columns(df)
    assert "hh_cbsa" in out.columns
    assert out["hh_cbsa"].dtype == "string"


def test_band_4plus_and_income_tier_mapping() -> None:
    bands = twb._band_4plus(pd.Series([0, 1, 3, 4, 9]))
    assert list(bands) == ["1", "1", "3", "4+", "4+"]
    tiers = twb._map_income_tier(pd.Series([1, 6, 9, 11, np.nan]))
    assert list(tiers) == ["t1", "t2", "t3", "t4", twb.INCOME_TIER_NULL_DEFAULT]


def test_build_strata_keys_adds_four_columns() -> None:
    hh = twb._normalise_hh_columns(_make_hh_frame())
    keyed = twb.build_strata_keys(hh)
    for col in ("income_tier", "hhsize_band", "hhvehcnt_band", "urbrur_int"):
        assert col in keyed.columns
    assert keyed["urbrur_int"].dtype == "int64"


def test_map_purpose_and_hhmm_to_minutes() -> None:
    assert twb._map_purpose(1) == "home"
    assert twb._map_purpose(3) == "work"
    assert twb._map_purpose(4) == "work"
    assert twb._map_purpose(97) == "other"
    assert twb._map_purpose("not-an-int") == "other"
    assert twb._hhmm_to_minutes(830) == 8 * 60 + 30
    assert twb._hhmm_to_minutes(0) == 0


# --------------------------------------------------------------------------- #
# Loader helpers.
# --------------------------------------------------------------------------- #


def test_effective_states_for_load() -> None:
    boston = REGIONS["boston"]
    # Residential / None: region states only.
    assert twb._effective_states_for_load(boston, "residential") == tuple(boston.states)
    assert twb._effective_states_for_load(boston, None) == tuple(boston.states)
    # Workplace with borrow states: region ∪ borrow, de-duplicated, ordered.
    eff = twb._effective_states_for_load(boston, "workplace")
    assert set(eff) == set(boston.states) | set(boston.workplace_donor_borrow_states)
    assert eff[: len(boston.states)] == tuple(boston.states)
    # No duplicates.
    assert len(eff) == len(set(eff))


def test_check_per_region_nhts_parquets_raises_when_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pev_synth import _paths as paths_mod

    region_dir = tmp_path / "bay_area"
    region_dir.mkdir()
    monkeypatch.setattr(
        paths_mod, "nhts_intermediate_root", lambda name=None: region_dir
    )
    with pytest.raises(FileNotFoundError, match="per-region NHTS parquet missing"):
        twb._check_per_region_nhts_parquets(REGIONS["bay_area"])

    # Now create all four expected parquets — no raise.
    for fname in twb._PER_REGION_NHTS_PARQUETS:
        pd.DataFrame({"x": [1]}).to_parquet(region_dir / fname)
    twb._check_per_region_nhts_parquets(REGIONS["bay_area"])


def test_force_csv_fallback_reads_and_filters(tmp_path: Path) -> None:
    _make_hh_frame().to_csv(tmp_path / "hhpub.csv", index=False)
    _make_trip_frame().to_csv(tmp_path / "trippub.csv", index=False)
    hh, trip = twb._force_csv_fallback(tmp_path, ("CA",))
    assert set(hh["houseid"].astype(int)) == {1, 2, 3}
    assert "whytrp1s" in trip.columns
    # All trips belong to kept houseids.
    assert set(trip["houseid"].astype(int)).issubset({1, 2, 3})


def test_force_csv_fallback_missing_files_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="NHTS CSVs not found"):
        twb._force_csv_fallback(tmp_path, ("CA",))


def test_force_csv_fallback_missing_hhstate_raises(tmp_path: Path) -> None:
    hh = _make_hh_frame().drop(columns=["HHSTATE"])
    hh.to_csv(tmp_path / "hhpub.csv", index=False)
    _make_trip_frame().to_csv(tmp_path / "trippub.csv", index=False)
    with pytest.raises(KeyError, match="HHSTATE"):
        twb._force_csv_fallback(tmp_path, ("CA",))


def test_load_nhts_for_region_rejects_non_region() -> None:
    with pytest.raises(TypeError, match="region must be Region"):
        twb.load_nhts_for_region("bay_area")  # type: ignore[arg-type]


def test_load_nhts_for_region_via_region_parquet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resolution path #1: region-specific hh.parquet / trip.parquet."""
    region = REGIONS["bay_area"]
    region_dir = tmp_path / region.name
    region_dir.mkdir(parents=True)
    hh_df = _make_hh_frame()
    # HH3 gets a CBSA outside bay_area so the CBSA narrowing drops it.
    hh_df.loc[hh_df["HOUSEID"] == 3, "HH_CBSA"] = "12345"
    twb._normalise_hh_columns(hh_df).to_parquet(region_dir / "hh.parquet")
    twb._normalise_trip_columns(_make_trip_frame()).to_parquet(
        region_dir / "trip.parquet"
    )
    monkeypatch.setattr(twb, "DEFAULT_NHTS_REGION_DIR", tmp_path)
    hh, trip = twb.load_nhts_for_region(region, nhts_data_dir=tmp_path)
    assert len(hh) > 0
    assert "whytrp1s" in trip.columns
    # CBSA narrowing: bay_area declares CBSAs -> HH 3 (cbsa 12345) dropped.
    assert region.cbsas
    assert set(hh["houseid"].astype(int)).issubset({1, 2})


def test_load_nhts_for_region_via_ca_parquet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resolution path #2: v1.1 California parquets (states ⊆ {CA})."""
    empty_region_root = tmp_path / "intermediate"
    empty_region_root.mkdir()
    monkeypatch.setattr(twb, "DEFAULT_NHTS_REGION_DIR", empty_region_root)

    raw = tmp_path / "raw"
    raw.mkdir()
    twb._normalise_hh_columns(_make_hh_frame()).to_parquet(
        raw / "nhts2017_ca_hh.parquet"
    )
    twb._normalise_trip_columns(_make_trip_frame()).to_parquet(
        raw / "nhts2017_ca_trip.parquet"
    )
    region = REGIONS["bay_area"]  # states == ("CA",)
    hh, trip = twb.load_nhts_for_region(region, nhts_data_dir=raw)
    assert len(hh) > 0
    assert "whytrp1s" in trip.columns


def test_load_nhts_for_region_workplace_borrow_state_cbsa_branch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Workplace + borrow-state region: keep in-CBSA HHs OR borrow-state HHs."""
    region = REGIONS["boston"]
    assert region.workplace_donor_borrow_states
    region_dir = tmp_path / region.name
    region_dir.mkdir(parents=True)

    hh = _make_hh_frame()
    # HH1: in-region (MA) + in-CBSA; HH2: in-region but out-of-CBSA (dropped);
    # HH3: borrow-state (NY) out-of-CBSA (kept via borrow-state path).
    hh["HHSTATE"] = ["MA", "MA", "NY"]
    hh["HH_CBSA"] = ["14460", "99999", "99999"]
    twb._normalise_hh_columns(hh).to_parquet(region_dir / "hh.parquet")
    twb._normalise_trip_columns(_make_trip_frame()).to_parquet(
        region_dir / "trip.parquet"
    )
    monkeypatch.setattr(twb, "DEFAULT_NHTS_REGION_DIR", tmp_path)

    hh_out, _ = twb.load_nhts_for_region(
        region, nhts_data_dir=tmp_path, profile_type="workplace"
    )
    kept = set(hh_out["houseid"].astype(int))
    assert 1 in kept          # in-region + in-CBSA
    assert 2 not in kept      # in-region but out-of-CBSA -> dropped
    assert 3 in kept          # borrow-state -> kept despite out-of-CBSA


def test_load_nhts_for_region_via_csv_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resolution path #3: raw CSV fallback (no region parquet, non-CA-only)."""
    # Point region parquet dir at an empty dir so resolution falls through.
    empty_region_root = tmp_path / "intermediate"
    empty_region_root.mkdir()
    monkeypatch.setattr(twb, "DEFAULT_NHTS_REGION_DIR", empty_region_root)

    raw = tmp_path / "raw"
    raw.mkdir()
    hh_df = _make_hh_frame()
    hh_df["HHSTATE"] = "MA"  # boston region -> not CA-only -> CSV path
    hh_df["HH_CBSA"] = "14460"  # Boston CBSA so narrowing keeps every HH
    hh_df.to_csv(raw / "hhpub.csv", index=False)
    _make_trip_frame().to_csv(raw / "trippub.csv", index=False)

    region = REGIONS["boston"]
    hh, trip = twb.load_nhts_for_region(region, nhts_data_dir=raw)
    assert len(hh) == 3
    assert set(trip["houseid"].astype(int)).issubset({1, 2, 3})


# --------------------------------------------------------------------------- #
# Workplace donor filter.
# --------------------------------------------------------------------------- #


def test_filter_to_workplace_donor_days_requires_whytrp1s() -> None:
    trip = pd.DataFrame({"houseid": [1], "personid": [1], "travday": [2]})
    with pytest.raises(KeyError, match="requires `whytrp1s` column"):
        twb.filter_to_workplace_donor_days(trip)


def test_filter_to_workplace_donor_days_keeps_work_person_days() -> None:
    trip = twb._normalise_trip_columns(_make_trip_frame())
    out = twb.filter_to_workplace_donor_days(trip)
    # HH1 (travday 3) and HH3 (travday 1) have work trips; HH2 does not.
    kept = set(
        zip(out["houseid"].astype(int), out["travday"].astype(int), strict=True)
    )
    assert (1, 3) in kept
    assert (3, 1) in kept
    assert (2, 4) not in kept
    # HH1 weekend day (travday 7) has no work trip -> dropped.
    assert (1, 7) not in kept


def test_filter_to_workplace_donor_days_empty_when_no_work() -> None:
    trip = twb._normalise_trip_columns(
        pd.DataFrame(
            {
                "HOUSEID": [2, 2],
                "PERSONID": [1, 1],
                "TRAVDAY": [4, 4],
                "TDTRPNUM": [1, 2],
                "WHYTO": [6, 6],
                "WHYTRP1S": [5, 5],
            }
        )
    )
    out = twb.filter_to_workplace_donor_days(trip)
    assert out.empty


# --------------------------------------------------------------------------- #
# Donor pool construction.
# --------------------------------------------------------------------------- #


def test_build_donor_pools_missing_required_columns_raises() -> None:
    trip = pd.DataFrame({"houseid": [1], "personid": [1], "travday": [2]})
    with pytest.raises(KeyError, match="trip frame missing required columns"):
        twb.build_donor_pools(trip, pd.DataFrame())


def test_build_donor_pools_returns_four_keyed_structures() -> None:
    trip = twb._normalise_trip_columns(_make_trip_frame())
    hh = twb.build_strata_keys(twb._normalise_hh_columns(_make_hh_frame()))
    enriched, by_travday, by_hh_td, by_hh = twb.build_donor_pools(trip, hh)
    assert set(by_hh.keys()) == set(enriched["houseid"].astype(int).unique())
    # by_travday keyed by travday ints present in the frame.
    assert set(by_travday.keys()) == set(enriched["travday"].astype(int).unique())
    # Strata columns merged from hh.
    for col in ("income_tier", "hhsize_band", "hhvehcnt_band", "urbrur_int"):
        assert col in enriched.columns
    # by_houseid_travday keyed by (houseid, travday) tuples.
    for (hid, td) in by_hh_td:
        assert isinstance(hid, int)
        assert isinstance(td, int)


# --------------------------------------------------------------------------- #
# Per-EV samplers.
# --------------------------------------------------------------------------- #


def _within_pool(houseid: int, travdays: list[int]) -> pd.DataFrame:
    n = len(travdays)
    return pd.DataFrame(
        {
            "houseid": [houseid] * n,
            "personid": [1] * n,
            "travday": travdays,
            "wttrdfin": [1000.0] * n,
            "n_trips": [1] * n,
            "income_tier": ["t2"] * n,
            "hhsize_band": ["2"] * n,
            "hhvehcnt_band": ["2"] * n,
            "urbrur_int": [1] * n,
        }
    )


def test_sample_person_day_empty_raises() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="empty pool"):
        twb._sample_person_day(rng, _within_pool(1, []))


def test_sample_person_day_uniform_when_weights_nonpositive() -> None:
    rng = np.random.default_rng(0)
    pool = _within_pool(7, [2, 3])
    pool["wttrdfin"] = [0.0, 0.0]
    hid, pid, td = twb._sample_person_day(rng, pool)
    assert hid == 7
    assert td in {2, 3}


def test_sample_week_within_houseid_fallback_to_legacy() -> None:
    """Missing donor in by_houseid -> defensive fallback to legacy sampler."""
    by_travday = {d: _within_pool(50, [d]) for d in range(1, 8)}
    by_hh_td = {(50, d): _within_pool(50, [d]) for d in range(1, 8)}
    rng = np.random.default_rng(7)
    picks = twb.sample_week_within_houseid(
        rng,
        donor_houseid=999,  # absent from by_houseid
        by_houseid={},
        by_travday=by_travday,
        by_houseid_travday=by_hh_td,
        income_tier="t2",
        hhsize_band="2",
        hhvehcnt_band="2",
        urbrur_int=1,
    )
    assert len(picks) == 7


def test_sample_week_within_houseid_fallback_without_kwargs_raises() -> None:
    rng = np.random.default_rng(7)
    with pytest.raises(RuntimeError, match="legacy fallback kwargs not"):
        twb.sample_week_within_houseid(
            rng, donor_houseid=999, by_houseid={}
        )


def test_sample_week_for_ev_within_hh_preference_path() -> None:
    """within_hh_prob=1.0 forces the within-HH branch for every slot."""
    by_hh_td = {(3, d): _within_pool(3, [d]) for d in range(1, 8)}
    by_travday = {d: _within_pool(99, [d]) for d in range(1, 8)}
    rng = np.random.default_rng(11)
    picks = twb.sample_week_for_ev(
        rng,
        donor_houseid=3,
        income_tier="t2",
        hhsize_band="2",
        hhvehcnt_band="2",
        urbrur_int=1,
        by_travday=by_travday,
        by_houseid_travday=by_hh_td,
        within_hh_prob=1.0,
    )
    assert all(hid == 3 for hid, _, _ in picks)


def test_sample_week_for_ev_strata_fallback_chain() -> None:
    """No within-HH record + strata mismatch -> walks the relaxation chain."""
    # outside pool with a non-matching strata so progressive relaxation runs.
    by_travday = {}
    for d in range(1, 8):
        pool = _within_pool(99, [d])
        pool["hhvehcnt_band"] = "4+"  # mismatches requested "2"
        pool["hhsize_band"] = "3"     # mismatches requested "2"
        by_travday[d] = pool
    rng = np.random.default_rng(13)
    picks = twb.sample_week_for_ev(
        rng,
        donor_houseid=1,  # absent in by_houseid_travday -> outside path
        income_tier="t2",
        hhsize_band="2",
        hhvehcnt_band="2",
        urbrur_int=1,
        by_travday=by_travday,
        by_houseid_travday={},
        within_hh_prob=0.0,
    )
    assert len(picks) == 7
    assert all(hid == 99 for hid, _, _ in picks)


def test_sample_week_for_ev_empty_travday_union_fallback() -> None:
    """A missing travday slot falls back to the union of non-empty days."""
    by_travday = {3: _within_pool(99, [3])}  # only travday 3 present
    rng = np.random.default_rng(17)
    picks = twb.sample_week_for_ev(
        rng,
        donor_houseid=1,
        income_tier="t2",
        hhsize_band="2",
        hhvehcnt_band="2",
        urbrur_int=1,
        by_travday=by_travday,
        by_houseid_travday={},
        within_hh_prob=0.0,
    )
    assert len(picks) == 7


def test_sample_week_for_ev_all_empty_raises() -> None:
    rng = np.random.default_rng(19)
    with pytest.raises(RuntimeError, match="empty across all travdays"):
        twb.sample_week_for_ev(
            rng,
            donor_houseid=1,
            income_tier="t2",
            hhsize_band="2",
            hhvehcnt_band="2",
            urbrur_int=1,
            by_travday={},
            by_houseid_travday={},
        )


# --------------------------------------------------------------------------- #
# v3 vintage-aware sampler.
# --------------------------------------------------------------------------- #


def test_vintage_supports_within_hh() -> None:
    multi = {1: _within_pool(1, [2, 7])}
    single = {1: _within_pool(1, [2])}
    assert twb._vintage_supports_within_hh({"by_houseid": multi}) is True
    assert twb._vintage_supports_within_hh({"by_houseid": single}) is False
    assert twb._vintage_supports_within_hh({"by_houseid": {}}) is False
    assert twb._vintage_supports_within_hh({}) is False
    # A by_houseid frame lacking a `travday` column is skipped (continue).
    no_travday = {1: pd.DataFrame({"houseid": [1, 1], "personid": [1, 2]})}
    assert twb._vintage_supports_within_hh({"by_houseid": no_travday}) is False


def test_sample_week_for_ev_v3_custom_vintage_routes_via_introspection() -> None:
    """Unknown-but-registered vintage with a multi-travday HH -> within-HH."""
    multi = _within_pool(8, [2, 3, 4, 5, 6, 1, 7])
    pools = {
        "by_houseid": {8: multi},
        "by_houseid_travday": {(8, int(d)): _within_pool(8, [int(d)]) for d in multi["travday"]},
        "by_travday": {int(d): _within_pool(8, [int(d)]) for d in multi["travday"]},
    }
    rng = np.random.default_rng(23)
    picks = twb.sample_week_for_ev_v3(
        rng,
        nhts_vintage="custom_multi",
        donor_houseid=8,
        pools_by_vintage={"custom_multi": pools},
        income_tier="t2",
        hhsize_band="2",
        hhvehcnt_band="2",
        urbrur_int=1,
    )
    assert all(hid == 8 for hid, _, _ in picks)


def test_sample_week_for_ev_v3_2017_routes_to_legacy() -> None:
    """2017 vintage delegates to the legacy per-travday sampler (happy path)."""
    pools = {
        "by_houseid": {7: _within_pool(7, [3])},
        "by_houseid_travday": {(7, d): _within_pool(7, [d]) for d in range(1, 8)},
        "by_travday": {d: _within_pool(7, [d]) for d in range(1, 8)},
    }
    rng = np.random.default_rng(31)
    picks = twb.sample_week_for_ev_v3(
        rng,
        nhts_vintage="2017",
        donor_houseid=7,
        pools_by_vintage={"2017": pools},
        income_tier="t2",
        hhsize_band="2",
        hhvehcnt_band="2",
        urbrur_int=1,
    )
    assert len(picks) == 7
    assert all(hid == 7 for hid, _, _ in picks)


def test_sample_week_for_ev_v3_unknown_vintage_raises() -> None:
    pools = {"by_houseid": {7: _within_pool(7, [3])}}
    rng = np.random.default_rng(0)
    with pytest.raises(KeyError, match="no donor pools registered"):
        twb.sample_week_for_ev_v3(
            rng,
            nhts_vintage="2099_imaginary",
            donor_houseid=7,
            pools_by_vintage={"2017": pools},
            income_tier="t2",
            hhsize_band="2",
            hhvehcnt_band="2",
            urbrur_int=1,
        )


def test_sample_week_for_ev_v3_within_hh_missing_by_houseid_raises() -> None:
    pools = {"by_houseid_travday": {}, "by_travday": {}}
    rng = np.random.default_rng(0)
    with pytest.raises(KeyError, match="missing required 'by_houseid'"):
        twb.sample_week_for_ev_v3(
            rng,
            nhts_vintage="2022_nextgen",
            donor_houseid=1,
            pools_by_vintage={"2022_nextgen": pools},
            income_tier="t2",
            hhsize_band="2",
            hhvehcnt_band="2",
            urbrur_int=1,
        )


def test_sample_week_for_ev_v3_legacy_missing_pools_raises() -> None:
    pools = {"by_houseid": {1: _within_pool(1, [3])}}
    rng = np.random.default_rng(0)
    with pytest.raises(KeyError, match="missing\n?.*by_travday|by_travday"):
        twb.sample_week_for_ev_v3(
            rng,
            nhts_vintage="2017",
            donor_houseid=1,
            pools_by_vintage={"2017": pools},
            income_tier="t2",
            hhsize_band="2",
            hhvehcnt_band="2",
            urbrur_int=1,
        )


# --------------------------------------------------------------------------- #
# Trip materialisation + end-to-end build_mobility.
# --------------------------------------------------------------------------- #


def test_materialise_trips_for_ev_calendar_edge_cases() -> None:
    """Directly exercise the day-chaining edge branches in materialisation.

    Crafts one person-day with: an overnight trip (end < start), a
    zero-duration trip (end == start), an out-of-order start (rollover),
    a >12h trip (duration cap), and a negative-miles row (clamped to 0).
    """
    person_day = pd.DataFrame(
        {
            "houseid": [5] * 5,
            "personid": [1] * 5,
            "travday": [3] * 5,
            "tdtrpnum": [1, 2, 3, 4, 5],
            "strttime": [2330, 100, 200, 800, 1000],  # 200 < 100 not, 800 after
            "endtime": [30, 100, 1500, 805, 950],      # overnight; zero-dur; >12h
            "trpmiles": [10.0, 2.0, 200.0, -3.0, 5.0],  # negative -> clamp
            "whyfrom": [1, 1, 1, 3, 1],
            "whyto": [3, 1, 6, 1, 6],
        }
    )
    triple = (5, 1, 3)  # travday 3 -> weekday slot
    week_picks = [triple] * 7  # one triple per d=1..7
    out = twb._materialise_trips_for_ev(
        ev_id=0,
        week_picks=week_picks,
        wh_per_mi_nominal=300.0,
        winter_f_T_multiplier=1.4,
        has_heat_pump=False,
        trip_by_person_day={triple: person_day},
        donor_houseid=5,
        donor_vehid=1,
    )
    assert len(out) > 0
    # Negative miles are clamped to 0.
    assert (out["miles"] >= 0).all()
    # Strictly non-decreasing within each day after the monotonicity pass.
    assert (out["end_ts"] >= out["start_ts"]).all()
    # No trip exceeds the 12-hour duration ceiling.
    durs = (out["end_ts"] - out["start_ts"]).dt.total_seconds() / 3600.0
    assert durs.max() <= twb.MAX_TRIP_DURATION_HOURS + 1e-6


def test_build_mobility_in_memory_residential() -> None:
    fleet = _make_fleet(3)
    mob = twb.build_mobility(
        fleet=fleet,
        region="bay_area",
        profile_type="residential",
        trip=_make_trip_frame(),
        hh=_make_hh_frame(),
        seed=SEED,
    )
    assert len(mob) > 0
    assert mob["ev_id"].nunique() == 3
    assert str(mob["start_ts"].dtype) == "datetime64[ns, UTC]"
    assert str(mob["end_ts"].dtype) == "datetime64[ns, UTC]"
    # Schema columns present.
    for col in (
        "trip_idx", "miles", "kwh_consumed", "f_T_applied",
        "origin_type", "dest_type", "nhts_donor_houseid",
        "nhts_donor_vehid", "nhts_donor_travday",
    ):
        assert col in mob.columns
    # bay_area is non-cold -> every f_T_applied == 1.0.
    assert (mob["f_T_applied"] == 1.0).all()
    # Strictly increasing trip timestamps inside each EV.
    assert (mob["end_ts"] >= mob["start_ts"]).all()


def test_build_mobility_reproducible_same_seed() -> None:
    fleet = _make_fleet(3)
    kwargs = dict(
        fleet=fleet, region="bay_area", profile_type="residential",
        trip=_make_trip_frame(), hh=_make_hh_frame(), seed=SEED,
    )
    a = twb.build_mobility(**kwargs)
    b = twb.build_mobility(**kwargs)
    pd.testing.assert_frame_equal(a, b)


def test_build_mobility_master_seed_alias_overrides_seed() -> None:
    fleet = _make_fleet(2)
    a = twb.build_mobility(
        fleet=fleet, region="bay_area", profile_type="residential",
        trip=_make_trip_frame(), hh=_make_hh_frame(), master_seed=SEED,
    )
    b = twb.build_mobility(
        fleet=fleet, region="bay_area", profile_type="residential",
        trip=_make_trip_frame(), hh=_make_hh_frame(), seed=SEED,
    )
    pd.testing.assert_frame_equal(a, b)


def test_build_mobility_v2_stitch_mode() -> None:
    fleet = _make_fleet(3)
    mob = twb.build_mobility(
        fleet=fleet, region="bay_area", profile_type="residential",
        trip=_make_trip_frame(), hh=_make_hh_frame(), seed=SEED,
        houseid_stitch_mode=twb.HOUSEID_STITCH_MODE_V2,
    )
    assert len(mob) > 0


def test_build_mobility_v3_stitch_mode() -> None:
    """Explicit v3 within-HH mode: every EV carries exactly one donor HH."""
    fleet = _make_fleet(3)
    mob = twb.build_mobility(
        fleet=fleet, region="bay_area", profile_type="residential",
        trip=_make_trip_frame(), hh=_make_hh_frame(), seed=SEED,
        houseid_stitch_mode=twb.HOUSEID_STITCH_MODE_V3,
    )
    assert len(mob) > 0
    per_ev_unique_hh = mob.groupby("ev_id", observed=True)[
        "nhts_donor_houseid"
    ].nunique()
    assert (per_ev_unique_hh == 1).all()


def test_build_mobility_default_region_is_bay_area() -> None:
    fleet = _make_fleet(2)
    mob = twb.build_mobility(
        fleet=fleet, region=None, profile_type="residential",
        trip=_make_trip_frame(), hh=_make_hh_frame(), seed=SEED,
    )
    assert len(mob) > 0


def test_build_mobility_winter_f_T_applied_for_cold_region() -> None:
    fleet = _make_fleet(3, has_heat_pump=False)
    mob = twb.build_mobility(
        fleet=fleet, region="boston", profile_type="residential",
        trip=_make_trip_frame(), hh=_make_hh_frame(), seed=SEED,
    )
    months_local = mob["start_ts"].dt.tz_convert("America/New_York").dt.month
    dec_mar = mob.loc[months_local.isin([12, 1, 2, 3])]
    region_f_T = REGIONS["boston"].winter_f_T_multiplier
    assert len(dec_mar) > 0
    assert dec_mar["f_T_applied"].max() == pytest.approx(region_f_T, abs=1e-3)


def test_build_mobility_heat_pump_caps_f_T() -> None:
    fleet = _make_fleet(3, has_heat_pump=True)
    mob = twb.build_mobility(
        fleet=fleet, region="boston", profile_type="residential",
        trip=_make_trip_frame(), hh=_make_hh_frame(), seed=SEED,
    )
    months_local = mob["start_ts"].dt.tz_convert("America/New_York").dt.month
    dec_mar = mob.loc[months_local.isin([12, 1, 2, 3])]
    assert (dec_mar["f_T_applied"] == twb.HEAT_PUMP_WINTER_F_T).all()


def test_build_mobility_workplace_profile() -> None:
    # Restrict the fleet to donors that survive the workplace filter
    # (HH1 + HH3 contain work person-days; HH2 does not).
    fleet = _make_fleet(2)
    fleet["nhts_donor_houseid"] = np.array([1, 3], dtype=np.int64)
    mob = twb.build_mobility(
        fleet=fleet, region="bay_area", profile_type="workplace",
        trip=_make_trip_frame(), hh=_make_hh_frame(), seed=SEED,
    )
    assert len(mob) > 0
    # Workplace mode restricts the donor TRIP pool to work person-days, so the
    # work-trip share must exceed the residential build on the same fleet.
    mob_res = twb.build_mobility(
        fleet=fleet, region="bay_area", profile_type="residential",
        trip=_make_trip_frame(), hh=_make_hh_frame(), seed=SEED,
    )
    work_share_wp = float((mob["dest_type"] == "work").mean())
    work_share_res = float((mob_res["dest_type"] == "work").mean())
    assert work_share_wp >= work_share_res


def test_build_mobility_workplace_empty_pool_raises() -> None:
    # A trip frame with zero work person-days.
    trip = _make_trip_frame()
    trip["WHYTRP1S"] = 5
    trip["WHYTO"] = 6
    fleet = _make_fleet(1)
    with pytest.raises(RuntimeError, match="workplace donor pool is empty"):
        twb.build_mobility(
            fleet=fleet, region="bay_area", profile_type="workplace",
            trip=trip, hh=_make_hh_frame(), seed=SEED,
        )


def test_build_mobility_invalid_profile_type_raises() -> None:
    with pytest.raises(ValueError, match="profile_type must be"):
        twb.build_mobility(
            fleet=_make_fleet(1), region="bay_area", profile_type="bogus",
            trip=_make_trip_frame(), hh=_make_hh_frame(),
        )


def test_build_mobility_missing_fleet_columns_raises() -> None:
    fleet = _make_fleet(1).drop(columns=["Wh_per_mi_nominal"])
    with pytest.raises(KeyError, match="fleet missing required columns"):
        twb.build_mobility(
            fleet=fleet, region="bay_area", profile_type="residential",
            trip=_make_trip_frame(), hh=_make_hh_frame(),
        )


def test_build_mobility_bad_within_hh_prob_raises() -> None:
    with pytest.raises(ValueError, match="within_hh_prob must be in"):
        twb.build_mobility(
            fleet=_make_fleet(1), region="bay_area", profile_type="residential",
            trip=_make_trip_frame(), hh=_make_hh_frame(), within_hh_prob=1.5,
        )


def test_build_mobility_bad_stitch_mode_raises() -> None:
    with pytest.raises(ValueError, match="houseid_stitch_mode must be"):
        twb.build_mobility(
            fleet=_make_fleet(1), region="bay_area", profile_type="residential",
            trip=_make_trip_frame(), hh=_make_hh_frame(),
            houseid_stitch_mode="bogus",
        )


def test_build_mobility_unknown_donor_houseid_raises() -> None:
    fleet = _make_fleet(1)
    fleet["nhts_donor_houseid"] = 9999  # not in hh frame
    with pytest.raises(KeyError, match="not in HH frame"):
        twb.build_mobility(
            fleet=fleet, region="bay_area", profile_type="residential",
            trip=_make_trip_frame(), hh=_make_hh_frame(),
        )


def test_build_mobility_no_heat_pump_column_defaults_false() -> None:
    fleet = _make_fleet(2).drop(columns=["has_heat_pump"])
    mob = twb.build_mobility(
        fleet=fleet, region="boston", profile_type="residential",
        trip=_make_trip_frame(), hh=_make_hh_frame(), seed=SEED,
    )
    # Without heat pump, cold months use the full region multiplier.
    months_local = mob["start_ts"].dt.tz_convert("America/New_York").dt.month
    dec_mar = mob.loc[months_local.isin([12, 1, 2, 3])]
    assert dec_mar["f_T_applied"].max() > twb.HEAT_PUMP_WINTER_F_T


# --------------------------------------------------------------------------- #
# Empty frame + finalise + us_national refusal.
# --------------------------------------------------------------------------- #


def test_build_mobility_empty_fleet_returns_empty_frame() -> None:
    empty_fleet = _make_fleet(0)
    mob = twb.build_mobility(
        fleet=empty_fleet, region="bay_area", profile_type="residential",
        trip=_make_trip_frame(), hh=_make_hh_frame(), seed=SEED,
    )
    assert len(mob) == 0
    assert str(mob["start_ts"].dtype) == "datetime64[ns, UTC]"


def test_empty_mobility_df_schema() -> None:
    empty = twb._empty_mobility_df(REGIONS["bay_area"])
    assert len(empty) == 0
    assert str(empty["start_ts"].dtype) == "datetime64[ns, UTC]"


def test_finalise_dtypes_dst_gap_coincide_bump() -> None:
    """Spring-forward DST gap collapses start/end onto the same instant.

    2001-04-01 02:00 -> 03:00 is the US spring-forward gap. A trip with
    start 02:30 and end 02:45 both ``shift_forward`` to 03:00, making
    end <= start; the finaliser bumps end by 1 minute.
    """
    mob = pd.DataFrame(
        {
            "ev_id": np.array([0], dtype=np.int32),
            "trip_idx": np.array([0], dtype=np.int32),
            "start_ts": pd.array(
                [pd.Timestamp("2001-04-01 02:30")], dtype="datetime64[ns]"
            ),
            "end_ts": pd.array(
                [pd.Timestamp("2001-04-01 02:45")], dtype="datetime64[ns]"
            ),
            "miles": np.array([5.0], dtype=np.float32),
            "kwh_consumed": np.array([1.5], dtype=np.float32),
            "f_T_applied": np.array([1.0], dtype=np.float32),
            "origin_type": ["home"],
            "dest_type": ["work"],
            "nhts_donor_houseid": ["1"],
            "nhts_donor_vehid": ["1"],
            "nhts_donor_travday": np.array([3], dtype=np.int8),
        }
    )
    out = twb._finalise_dtypes(mob, REGIONS["boston"])  # America/New_York
    assert (out["end_ts"] > out["start_ts"]).all()


def test_finalise_dtypes_refuses_us_national() -> None:
    # Build a tiny mobility frame then try to finalise under us_national tz.
    mob = pd.DataFrame(
        {
            "ev_id": np.array([0], dtype=np.int32),
            "trip_idx": np.array([0], dtype=np.int32),
            "start_ts": pd.array(
                [pd.Timestamp("2001-01-01 08:00")], dtype="datetime64[ns]"
            ),
            "end_ts": pd.array(
                [pd.Timestamp("2001-01-01 08:30")], dtype="datetime64[ns]"
            ),
            "miles": np.array([5.0], dtype=np.float32),
            "kwh_consumed": np.array([1.5], dtype=np.float32),
            "f_T_applied": np.array([1.0], dtype=np.float32),
            "origin_type": ["home"],
            "dest_type": ["work"],
            "nhts_donor_houseid": ["1"],
            "nhts_donor_vehid": ["1"],
            "nhts_donor_travday": np.array([3], dtype=np.int8),
        }
    )
    with pytest.raises(ValueError, match="not materialisable"):
        twb._finalise_dtypes(mob, REGIONS["us_national"])


# --------------------------------------------------------------------------- #
# run() orchestrator + persistence + CLI.
# --------------------------------------------------------------------------- #


def _seed_region_parquets(region_name: str, root: Path) -> None:
    """Write the four canonical per-region NHTS parquets + hh/trip parquets."""
    region_dir = root / region_name
    region_dir.mkdir(parents=True, exist_ok=True)
    hh = _make_hh_frame()
    trip = _make_trip_frame()
    for fname in twb._PER_REGION_NHTS_PARQUETS:
        hh.to_parquet(region_dir / fname)
    twb._normalise_hh_columns(hh).to_parquet(region_dir / "hh.parquet")
    twb._normalise_trip_columns(trip).to_parquet(region_dir / "trip.parquet")


def test_run_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from pev_synth import _paths as paths_mod

    region_name = "bay_area"
    intermediate = tmp_path / "intermediate" / "nhts"
    intermediate.mkdir(parents=True)
    _seed_region_parquets(region_name, intermediate)

    monkeypatch.setattr(twb, "DEFAULT_NHTS_REGION_DIR", intermediate)
    monkeypatch.setattr(
        paths_mod,
        "nhts_intermediate_root",
        lambda name=None: (intermediate / name) if name else intermediate,
    )

    fleet_path = tmp_path / "fleet.parquet"
    _make_fleet(3).to_parquet(fleet_path)
    out_mob = tmp_path / "mobility.parquet"
    meta_path = tmp_path / "meta.json"

    config = twb.TravelWeekConfig(
        master_seed=SEED,
        region_name=region_name,
        profile_type="residential",
        nhts_data_dir=intermediate,
        fleet_path=fleet_path,
        out_mobility_path=out_mob,
        meta_path=meta_path,
    )
    result = twb.run(config, repo_root=tmp_path)
    assert result.n_ev == 3
    assert result.n_trips > 0
    assert out_mob.is_file()
    assert meta_path.is_file()
    # Meta json carries the v2.0 provenance block.
    meta = json.loads(meta_path.read_text())
    assert "travel_week_builder" in meta
    assert meta["travel_week_builder"]["region_name"] == region_name
    assert "consumption_model" in meta
    # within_day_contiguity is a percentage in [0, 100].
    assert 0.0 <= result.within_day_contiguity_pct <= 100.0


def test_run_requires_parquets_when_not_force_csv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pev_synth import _paths as paths_mod

    empty = tmp_path / "nhts"
    (empty / "bay_area").mkdir(parents=True)
    monkeypatch.setattr(
        paths_mod, "nhts_intermediate_root", lambda name=None: empty / (name or "")
    )
    config = twb.TravelWeekConfig(region_name="bay_area", profile_type="residential")
    with pytest.raises(FileNotFoundError, match="per-region NHTS parquet missing"):
        twb.run(config, repo_root=tmp_path)


def test_within_day_contiguity_pct_empty_and_nonempty() -> None:
    assert twb._within_day_contiguity_pct(pd.DataFrame()) == 0.0
    mob = pd.DataFrame(
        {
            "ev_id": [0, 0],
            "trip_idx": [0, 1],
            "start_ts": pd.to_datetime(
                ["2001-01-01 08:00", "2001-01-01 09:00"]
            ).tz_localize("UTC"),
            "end_ts": pd.to_datetime(
                ["2001-01-01 08:30", "2001-01-01 09:30"]
            ).tz_localize("UTC"),
        }
    )
    pct = twb._within_day_contiguity_pct(mob)
    assert pct == pytest.approx(100.0)


def test_resolve_relative_and_absolute(tmp_path: Path) -> None:
    rel = Path("sub/file.parquet")
    assert twb._resolve(rel, tmp_path) == tmp_path / rel
    absolute = tmp_path / "abs.parquet"
    assert twb._resolve(absolute, tmp_path) == absolute


def test_write_mobility_roundtrip(tmp_path: Path) -> None:
    mob = twb.build_mobility(
        fleet=_make_fleet(2), region="bay_area", profile_type="residential",
        trip=_make_trip_frame(), hh=_make_hh_frame(), seed=SEED,
    )
    out = tmp_path / "nested" / "mobility.parquet"
    twb._write_mobility(mob, out)
    assert out.is_file()
    back = pd.read_parquet(out)
    assert len(back) == len(mob)


def test_update_meta_merges_into_existing(tmp_path: Path) -> None:
    meta_path = tmp_path / "meta.json"
    meta_path.write_text(json.dumps({"existing_key": 1, "consumption_model": {}}))
    result = twb.TravelWeekResult(
        mobility=pd.DataFrame(),
        n_trips=10,
        n_ev=2,
        median_trips_per_ev=5.0,
        median_miles_per_ev=9000.0,
        p90_miles_per_ev=12000.0,
        total_miles=18000.0,
        total_kwh=5000.0,
        origin_pct={"home": 0.5, "work": 0.3, "other": 0.2},
        dest_pct={"home": 0.4, "work": 0.4, "other": 0.2},
        within_day_contiguity_pct=99.0,
        f_T_winter_mean=1.0,
        n_trips_winter_dec_mar=3,
        region_name="boston",
        profile_type="residential",
    )
    twb._update_meta(meta_path=meta_path, result=result, region=REGIONS["boston"])
    meta = json.loads(meta_path.read_text())
    assert meta["existing_key"] == 1
    assert meta["travel_week_builder"]["region_name"] == "boston"
    assert meta["consumption_model"]["winter_uplift"] == pytest.approx(
        REGIONS["boston"].winter_f_T_multiplier
    )


def test_default_repo_root_returns_path() -> None:
    root = twb._default_repo_root()
    assert isinstance(root, Path)
    assert root.is_absolute()


def test_build_arg_parser_defaults_and_overrides() -> None:
    parser = twb._build_arg_parser()
    args = parser.parse_args([])
    assert args.seed == twb.MASTER_SEED_DEFAULT
    assert args.region == "bay_area"
    assert args.profile_type == "residential"
    assert args.houseid_stitch_mode == twb.HOUSEID_STITCH_MODE_DEFAULT
    args2 = parser.parse_args(
        ["--region", "boston", "--profile-type", "workplace", "--force-csv"]
    )
    assert args2.region == "boston"
    assert args2.profile_type == "workplace"
    assert args2.force_csv is True


def test_main_invokes_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from pev_synth import _paths as paths_mod

    region_name = "bay_area"
    intermediate = tmp_path / "intermediate" / "nhts"
    intermediate.mkdir(parents=True)
    _seed_region_parquets(region_name, intermediate)
    monkeypatch.setattr(twb, "DEFAULT_NHTS_REGION_DIR", intermediate)
    monkeypatch.setattr(
        paths_mod,
        "nhts_intermediate_root",
        lambda name=None: (intermediate / name) if name else intermediate,
    )

    fleet_path = tmp_path / "fleet.parquet"
    _make_fleet(2).to_parquet(fleet_path)
    out_mob = tmp_path / "mobility.parquet"
    meta_path = tmp_path / "meta.json"

    rc = twb.main(
        [
            "--seed", str(SEED),
            "--region", region_name,
            "--profile-type", "residential",
            "--nhts-dir", str(intermediate),
            "--fleet", str(fleet_path),
            "--out", str(out_mob),
            "--meta", str(meta_path),
            "--repo-root", str(tmp_path),
            "--log-level", "ERROR",
        ]
    )
    assert rc == 0
    assert out_mob.is_file()
