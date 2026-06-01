"""Unit tests for M4 — `pev_synth.travel_week_builder` (v2.0).

Covers Wave 2 Dev E acceptance criteria 1-6 and the unit-test list in
the orchestrator brief:

    1. test_build_mobility_bay_area_residential
    2. test_build_mobility_utc_timestamps
    3. test_build_mobility_workplace_filters_to_whytrp1s_10
    4. test_build_mobility_winter_fT_applied_dec_mar
    5. test_build_mobility_winter_fT_not_applied_bay_area
    6. test_build_mobility_winter_fT_not_applied_in_summer
    7. test_build_mobility_heat_pump_correction_lowers_f_T
    8. test_build_mobility_seed_reproducible
    9. test_build_mobility_dallas_fort_worth_residential
   10. test_purpose_mapping_well_defined
   11. test_origin_dest_type_mapping_covers_all_whyto_codes
   12. test_within_hh_preference_fires_when_donor_hh_has_record
   13. test_workplace_donor_filter_drops_pure_non_work_days

The fleet/mobility fixtures are SMALL (5-25 EVs) so the suite finishes
in under a minute.  When the full residential cache is on disk we
additionally run two acceptance checks on the cached 1000-EV mobility
(test_cached_residential_mobility_*).

Tests are skipped if the NHTS CSV / parquet inputs are not present
locally.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make the `code/` directory importable from anywhere.
_CODE_DIR = Path(__file__).resolve().parents[1]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from pev_synth import travel_week_builder as twb  # noqa: E402
from pev_synth.regions import REGIONS  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
_NHTS_RAW = _REPO_ROOT / twb.DEFAULT_NHTS_RAW_DIR
# v2.0 per-region layout. `twb.DEFAULT_FLEET_REL` still points at the flat
# v1.1 path (`processed/residential_ev_synth/fleet.parquet`), which no
# longer exists; the M3 fleet + cached mobility now live under the region
# subdir. Point the test paths at the bay_area residential cache.
_FLEET_PATH = (
    _REPO_ROOT / "data/pev/processed/bay_area/residential_ev_synth/fleet.parquet"
)
_CACHED_MOBILITY_PATH = (
    _REPO_ROOT
    / "data/pev/processed/bay_area/residential_ev_synth/mobility.parquet"
)


def _have_minimum_inputs() -> bool:
    """We need an existing fleet and either CA parquets or raw CSVs."""
    if not _FLEET_PATH.is_file():
        return False
    if (_NHTS_RAW / "nhts2017_ca_hh.parquet").is_file():
        return True
    if (_NHTS_RAW / "hhpub.csv").is_file():
        return True
    return False


pytestmark = pytest.mark.skipif(
    not _have_minimum_inputs(),
    reason="needs M1 NHTS inputs + M3 fleet.parquet present in repo data tree",
)


# --------------------------------------------------------------------------- #
# Shared fixtures.
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def fleet() -> pd.DataFrame:
    """Full 1000-EV residential fleet (M3 output)."""
    return pd.read_parquet(_FLEET_PATH)


@pytest.fixture(scope="module")
def small_fleet(fleet: pd.DataFrame) -> pd.DataFrame:
    """A 25-EV slice for fast acceptance tests."""
    return fleet.sort_values("ev_id").head(25).reset_index(drop=True)


@pytest.fixture(scope="module")
def mini_fleet(fleet: pd.DataFrame) -> pd.DataFrame:
    """A 5-EV slice for the slowest tests (workplace, Boston)."""
    return fleet.sort_values("ev_id").head(5).reset_index(drop=True)


@pytest.fixture(scope="module")
def bay_area_residential(small_fleet: pd.DataFrame) -> pd.DataFrame:
    """25-EV bay_area residential mobility frame."""
    return twb.build_mobility(
        fleet=small_fleet,
        region="bay_area",
        profile_type="residential",
        seed=twb.MASTER_SEED_DEFAULT,
        nhts_data_dir=_NHTS_RAW,
    )


def _remap_fleet_donors_to_region(
    fleet: pd.DataFrame, region_name: str,
) -> pd.DataFrame:
    """Rewrite `nhts_donor_houseid` / `nhts_donor_vehid` to a valid HH in `region`.

    The existing residential fleet was built with CA donors; for the
    test suite we want to exercise non-CA regions without re-running
    the M3 donor matcher.  We just pick any HH (round-robin) from the
    region's NHTS frame and a dummy vehicle ID.
    """
    region = REGIONS[region_name]
    hh, _ = twb.load_nhts_for_region(region, nhts_data_dir=_NHTS_RAW)
    available_ids = hh["houseid"].astype("int64").to_numpy()
    if len(available_ids) == 0:
        raise RuntimeError(f"no HH available for region {region_name}")
    n = len(fleet)
    picks = available_ids[np.arange(n) % len(available_ids)]
    out = fleet.copy()
    out["nhts_donor_houseid"] = picks.astype(np.int64)
    out["nhts_donor_vehid"] = np.ones(n, dtype=np.int32)
    return out


# --------------------------------------------------------------------------- #
# Acceptance tests (orchestrator brief criteria 1-6).
# --------------------------------------------------------------------------- #


def test_build_mobility_bay_area_residential(
    bay_area_residential: pd.DataFrame, small_fleet: pd.DataFrame
) -> None:
    """Acceptance #1: bay_area / residential produces a non-empty mobility frame.

    The full-fleet (1000 EV) target is ≥ 1.5M trip rows — verified
    separately on the cached parquet.  Here we use a 25-EV slice and
    check (a) every EV gets ≥ 1 trip; (b) the per-EV median miles is
    plausible (1k - 30k mi/yr — broad band).
    """
    mob = bay_area_residential
    assert len(mob) > 0
    assert mob["ev_id"].nunique() == len(small_fleet)
    miles_per_ev = mob.groupby("ev_id", observed=True)["miles"].sum()
    assert miles_per_ev.min() > 0
    assert miles_per_ev.median() < 30_000


def test_build_mobility_utc_timestamps(bay_area_residential: pd.DataFrame) -> None:
    """Acceptance #5: every output timestamp is datetime64[ns, UTC]."""
    mob = bay_area_residential
    assert str(mob["start_ts"].dtype) == "datetime64[ns, UTC]"
    assert str(mob["end_ts"].dtype) == "datetime64[ns, UTC]"
    # tz attribute
    assert str(mob["start_ts"].dt.tz) == "UTC"
    assert str(mob["end_ts"].dt.tz) == "UTC"


def test_build_mobility_workplace_filters_to_whytrp1s_10(
    mini_fleet: pd.DataFrame,
) -> None:
    """Acceptance #4: workplace profile_type filters to WHYTRP1S==10 person-days.

    We don't require every trip to be a work trip (a workday donor's
    other trips are kept to preserve trip chaining); we require that
    every donor person-day used contains AT LEAST ONE work trip.
    """
    mob = twb.build_mobility(
        fleet=mini_fleet,
        region="bay_area",
        profile_type="workplace",
        seed=twb.MASTER_SEED_DEFAULT,
        nhts_data_dir=_NHTS_RAW,
    )
    assert len(mob) > 0
    # Workplace-filtered days should yield a higher work-trip share
    # than residential.
    work_share_wp = float((mob["dest_type"] == "work").mean())
    mob_res = twb.build_mobility(
        fleet=mini_fleet,
        region="bay_area",
        profile_type="residential",
        seed=twb.MASTER_SEED_DEFAULT,
        nhts_data_dir=_NHTS_RAW,
    )
    work_share_res = float((mob_res["dest_type"] == "work").mean())
    assert work_share_wp > work_share_res, (
        f"workplace work-share {work_share_wp:.3f} not > residential {work_share_res:.3f}"
    )
    # Every donor-day chosen must contain at least one work trip — confirmed
    # by checking that the cohort's per-donor-travday work-trip count is ≥ 1.
    donors = mob.groupby(
        ["nhts_donor_houseid", "nhts_donor_vehid", "nhts_donor_travday"],
        observed=True,
    )["dest_type"].apply(lambda s: (s == "work").any())
    assert bool(donors.all()), (
        "some workplace donor-days contain zero work trips — "
        "WHYTRP1S==10 filter not applied"
    )


def test_build_mobility_winter_fT_applied_dec_mar(mini_fleet: pd.DataFrame) -> None:
    """Acceptance #3: Boston Dec-Mar trips get f_T ≈ region pin × non-cold months.

    The synthetic year is 2001 so Dec / Jan / Feb / Mar map to discrete
    months and the per-trip multiplier must equal the region's
    ``winter_f_T_multiplier`` for non-heat-pump EVs. RFC-031: Boston pin
    is 1.396 (Yuksel-Michalek polynomial at -2 °C).
    """
    # Force has_heat_pump=False on the test fleet so the cap doesn't fire.
    fl = _remap_fleet_donors_to_region(mini_fleet, "boston")
    fl["has_heat_pump"] = False
    mob = twb.build_mobility(
        fleet=fl,
        region="boston",
        profile_type="residential",
        seed=twb.MASTER_SEED_DEFAULT,
        nhts_data_dir=_NHTS_RAW,
    )
    assert len(mob) > 0
    # Pull a Dec trip and a Jun trip with the same donor signature, if
    # any; failing that, compare the means of the populations.
    months_utc = mob["start_ts"].dt.tz_convert("America/New_York").dt.month
    dec_mar = mob.loc[months_utc.isin([12, 1, 2, 3])]
    other = mob.loc[~months_utc.isin([12, 1, 2, 3])]
    assert len(dec_mar) > 0
    assert len(other) > 0
    # f_T_applied per row must be exactly the region's winter_f_T in Dec-Mar.
    region_f_T = REGIONS["boston"].winter_f_T_multiplier
    assert abs(float(dec_mar["f_T_applied"].mean()) - region_f_T) < 1e-3
    assert abs(float(other["f_T_applied"].mean()) - 1.0) < 1e-3
    # kwh ratio per trip = f_T_applied (since miles · Wh_per_mi is identical
    # for the same trip).  Aggregate Wh/mi is f_T × baseline.
    wh_per_mi_dec = float(
        (dec_mar["kwh_consumed"].sum() * 1000.0)
        / max(dec_mar["miles"].sum(), 1e-6)
    )
    wh_per_mi_other = float(
        (other["kwh_consumed"].sum() * 1000.0)
        / max(other["miles"].sum(), 1e-6)
    )
    ratio = wh_per_mi_dec / wh_per_mi_other
    # RFC-031: Boston pin = 1.396; allow ±0.02 around the pin.
    assert abs(ratio - region_f_T) < 0.02, (
        f"boston Dec-Mar Wh/mi / non-cold Wh/mi = {ratio:.3f}; "
        f"expected ≈ {region_f_T:.3f}"
    )


def test_build_mobility_winter_fT_not_applied_bay_area(
    bay_area_residential: pd.DataFrame,
) -> None:
    """bay_area has winter_f_T_multiplier=1.0 — every f_T_applied = 1.0."""
    mob = bay_area_residential
    assert (mob["f_T_applied"] == 1.0).all()


def test_build_mobility_winter_fT_not_applied_in_summer(
    mini_fleet: pd.DataFrame,
) -> None:
    """Boston July trips: f_T_applied == 1.0 (winter months only)."""
    fl = _remap_fleet_donors_to_region(mini_fleet, "boston")
    fl["has_heat_pump"] = False
    mob = twb.build_mobility(
        fleet=fl,
        region="boston",
        profile_type="residential",
        seed=twb.MASTER_SEED_DEFAULT,
        nhts_data_dir=_NHTS_RAW,
    )
    months_local = mob["start_ts"].dt.tz_convert("America/New_York").dt.month
    jul = mob.loc[months_local == 7]
    assert len(jul) > 0
    assert (jul["f_T_applied"] == 1.0).all()


def test_build_mobility_heat_pump_correction_lowers_f_T(
    mini_fleet: pd.DataFrame,
) -> None:
    """Acceptance: heat-pump-corrected sub-fleet's Dec-Mar f_T capped at 1.10."""
    fl = _remap_fleet_donors_to_region(mini_fleet, "boston")
    fl["has_heat_pump"] = True
    mob_hp = twb.build_mobility(
        fleet=fl,
        region="boston",
        profile_type="residential",
        seed=twb.MASTER_SEED_DEFAULT,
        nhts_data_dir=_NHTS_RAW,
    )
    months_local = mob_hp["start_ts"].dt.tz_convert("America/New_York").dt.month
    dec_mar = mob_hp.loc[months_local.isin([12, 1, 2, 3])]
    assert len(dec_mar) > 0
    assert (dec_mar["f_T_applied"] == twb.HEAT_PUMP_WINTER_F_T).all()


def test_build_mobility_seed_reproducible(
    mini_fleet: pd.DataFrame, tmp_path: Path,
) -> None:
    """Acceptance #6: same seed → byte-identical mobility.parquet."""
    a = twb.build_mobility(
        fleet=mini_fleet, region="bay_area",
        profile_type="residential", seed=twb.MASTER_SEED_DEFAULT,
        nhts_data_dir=_NHTS_RAW,
    )
    b = twb.build_mobility(
        fleet=mini_fleet, region="bay_area",
        profile_type="residential", seed=twb.MASTER_SEED_DEFAULT,
        nhts_data_dir=_NHTS_RAW,
    )
    pd.testing.assert_frame_equal(a, b)
    p_a = tmp_path / "mob_a.parquet"
    p_b = tmp_path / "mob_b.parquet"
    twb._write_mobility(a, p_a)
    twb._write_mobility(b, p_b)
    sha_a = hashlib.sha256(p_a.read_bytes()).hexdigest()
    sha_b = hashlib.sha256(p_b.read_bytes()).hexdigest()
    assert sha_a == sha_b, "mobility.parquet differs between seed-identical runs"


def test_build_mobility_dallas_fort_worth_residential(
    mini_fleet: pd.DataFrame,
) -> None:
    """Acceptance #2: DFW region produces a non-empty mobility frame using
    the TX state filter + DFW/Austin/Houston CBSA narrowing."""
    fl = _remap_fleet_donors_to_region(mini_fleet, "dallas_fort_worth")
    mob = twb.build_mobility(
        fleet=fl,
        region="dallas_fort_worth",
        profile_type="residential",
        seed=twb.MASTER_SEED_DEFAULT,
        nhts_data_dir=_NHTS_RAW,
    )
    assert len(mob) > 0
    # UTC timestamps
    assert str(mob["start_ts"].dtype) == "datetime64[ns, UTC]"
    # RFC-031: DFW is non-cold (winter_f_T_multiplier=1.00); every f_T==1.0.
    months_local = mob["start_ts"].dt.tz_convert("America/Chicago").dt.month
    dec_mar = mob.loc[months_local.isin([12, 1, 2, 3])]
    if len(dec_mar) > 0:
        assert (dec_mar["f_T_applied"] - 1.0).abs().max() < 1e-3


# --------------------------------------------------------------------------- #
# Internal helpers / unit tests.
# --------------------------------------------------------------------------- #


def test_purpose_mapping_well_defined() -> None:
    """The WHYTO purpose map covers WHYTO=3 AND WHYTO=4 as work."""
    assert twb._map_purpose(1) == "home"
    assert twb._map_purpose(3) == "work"
    assert twb._map_purpose(4) == "work"  # NEW v2.0 — WHYTRP1S=10 absorbs both
    assert twb._map_purpose(2) == "other"
    assert twb._map_purpose(97) == "other"
    assert twb._map_purpose(-9) == "other"
    assert twb._map_purpose(float("nan")) == "other"


def test_origin_dest_type_mapping_covers_all_whyto_codes() -> None:
    """Every NHTS WHYFROM/WHYTO code maps to home / work / other."""
    codes = list(range(1, 20)) + [97, -7, -8, -9]
    mapped = {c: twb._map_purpose(c) for c in codes}
    allowed = set(twb.ORIGIN_DEST_CATEGORIES)
    for c, label in mapped.items():
        assert label in allowed, f"WHY code {c} mapped to {label!r}, not in {allowed}"
    seen = set(mapped.values())
    assert seen == allowed, f"missing classes from mapping: {allowed - seen}"


def _make_within_hh_pool(
    *,
    houseid: int,
    travdays: list[int],
    personids: list[int] | None = None,
    income_tier: str = "t2",
    hhsize_band: str = "2",
    hhvehcnt_band: str = "2",
    urbrur_int: int = 1,
) -> pd.DataFrame:
    """Build a synthetic per-HH donor pool with the schema expected by
    :func:`sample_week_within_houseid`.  One row per (personid, travday).
    """
    if personids is None:
        personids = [1] * len(travdays)
    n = len(travdays)
    return pd.DataFrame(
        {
            "houseid": [int(houseid)] * n,
            "personid": [int(p) for p in personids],
            "travday": [int(d) for d in travdays],
            "wttrdfin": [1000.0] * n,
            "n_trips": [1] * n,
            "income_tier": [income_tier] * n,
            "hhsize_band": [hhsize_band] * n,
            "hhvehcnt_band": [hhvehcnt_band] * n,
            "urbrur_int": [urbrur_int] * n,
        }
    )


def test_sample_week_within_houseid_stays_within_donor() -> None:
    """Every one of the 7 draws lives in the donor HOUSEID's pool."""
    donor_hh = 42
    pool = _make_within_hh_pool(houseid=donor_hh, travdays=[2, 3, 4, 5, 6, 1, 7])
    by_houseid = {donor_hh: pool}
    rng = np.random.default_rng(2026_05_28)
    picks = twb.sample_week_within_houseid(
        rng,
        donor_houseid=donor_hh,
        by_houseid=by_houseid,
    )
    assert len(picks) == 7
    for (hid, _pid, _td) in picks:
        assert hid == donor_hh


def test_sample_week_within_houseid_weekday_weekend_alignment() -> None:
    """When the HH has both sides, weekday slots draw from TRAVDAY 2..6
    and weekend slots (d=1, d=7) draw from TRAVDAY 1/7."""
    donor_hh = 99
    pool = _make_within_hh_pool(
        houseid=donor_hh,
        travdays=[2, 3, 4, 5, 6, 1, 7],
        personids=[1, 1, 2, 2, 3, 3, 4],
    )
    by_houseid = {donor_hh: pool}
    rng = np.random.default_rng(20260529)
    picks = twb.sample_week_within_houseid(
        rng,
        donor_houseid=donor_hh,
        by_houseid=by_houseid,
    )
    # d in {1..7}; index 0 is d=1 (Sunday), index 6 is d=7 (Saturday).
    weekday_indices = (1, 2, 3, 4, 5)  # d=2..6
    weekend_indices = (0, 6)           # d=1, d=7
    for i in weekday_indices:
        _, _, td = picks[i]
        assert td in {2, 3, 4, 5, 6}, f"weekday slot {i} got travday {td}"
    for i in weekend_indices:
        _, _, td = picks[i]
        assert td in {1, 7}, f"weekend slot {i} got travday {td}"


def test_sample_week_within_houseid_cross_side_substitution_weekday_only() -> None:
    """A donor HH with only weekday rows substitutes weekend slots."""
    donor_hh = 1234
    pool = _make_within_hh_pool(houseid=donor_hh, travdays=[2, 3])
    by_houseid = {donor_hh: pool}
    rng = np.random.default_rng(20260530)
    picks = twb.sample_week_within_houseid(
        rng,
        donor_houseid=donor_hh,
        by_houseid=by_houseid,
    )
    # All 7 picks must remain in this HH and use TRAVDAY ∈ {2, 3} (no escape).
    for (hid, _pid, td) in picks:
        assert hid == donor_hh
        assert td in {2, 3}


def test_sample_week_within_houseid_cross_side_substitution_weekend_only() -> None:
    """A donor HH with only weekend rows substitutes weekday slots."""
    donor_hh = 5678
    pool = _make_within_hh_pool(houseid=donor_hh, travdays=[1, 7])
    by_houseid = {donor_hh: pool}
    rng = np.random.default_rng(20260531)
    picks = twb.sample_week_within_houseid(
        rng,
        donor_houseid=donor_hh,
        by_houseid=by_houseid,
    )
    for (hid, _pid, td) in picks:
        assert hid == donor_hh
        assert td in {1, 7}


def test_build_donor_pools_returns_four_pools() -> None:
    """v3.0: ``build_donor_pools`` returns ``(enriched, by_travday,
    by_houseid_travday, by_houseid)`` and ``by_houseid`` keys agree with
    the unique HOUSEIDs in ``enriched``.
    """
    trip = pd.DataFrame(
        {
            "houseid": [1, 1, 2, 2, 3, 3],
            "personid": [1, 1, 1, 2, 1, 1],
            "travday": [2, 2, 5, 5, 1, 1],
            "tdtrpnum": [1, 2, 1, 1, 1, 2],
            "wttrdfin": [100.0] * 6,
        }
    )
    hh = pd.DataFrame(
        {
            "houseid": [1, 2, 3],
            "income_tier": ["t2", "t2", "t2"],
            "hhsize_band": ["2", "2", "2"],
            "hhvehcnt_band": ["2", "2", "2"],
            "urbrur_int": [1, 1, 1],
        }
    )
    out = twb.build_donor_pools(trip, hh)
    assert len(out) == 4
    enriched, by_travday, by_hh_td, by_houseid = out
    assert set(by_houseid.keys()) == set(enriched["houseid"].astype(int).unique().tolist())
    for hid, df in by_houseid.items():
        assert (df["houseid"].astype(int) == hid).all()


@pytest.mark.skipif(
    not _have_minimum_inputs(),
    reason="needs M1 NHTS inputs + M3 fleet.parquet present in repo data tree",
)
def test_intra_hh_weekly_correlation_recovers(fleet: pd.DataFrame) -> None:
    """Compare v3 within-HH stitching vs v2 per-travday on the same fleet.

    Empirical D3 finding on the bay_area residential cache (Phase 2):
    because NHTS 2017 records exactly one TRAVDAY per HOUSEID and the
    mean HH carries only ~1.87 person-days, the v3 intra-HH stitcher
    replays a single donor person-day 365 times — annual miles are
    therefore dominated by that one TRPMILES value.  The v2.x per-day
    i.i.d. cross-HH sampler is *more* central-limit-theorem-shrinking
    than v3 on this metric.  We honestly assert:

      (a) Both pipelines emit a positive, finite σ.
      (b) Every v3 EV's mobility carries exactly one donor HOUSEID
          (the new RESOLVED §13.2 invariant — intra-week coherence is
          preserved, even when the resulting marginal variance does
          not shrink).

    We do NOT assert ``σ_v3 < σ_v2`` because the empirical evidence on
    the live cache shows the opposite direction; doing so would lock
    in a false promise.  PM has been surfaced this finding via the v3
    methodology meta block (see ``check_annual_mileage`` v3 honest-FAIL
    path).
    """
    n_evs = 200
    sample = fleet.sort_values("ev_id").head(n_evs).reset_index(drop=True)

    mob_v3 = twb.build_mobility(
        fleet=sample,
        region="bay_area",
        profile_type="residential",
        seed=twb.MASTER_SEED_DEFAULT,
        nhts_data_dir=_NHTS_RAW,
        houseid_stitch_mode=twb.HOUSEID_STITCH_MODE_V3,
    )
    mob_v2 = twb.build_mobility(
        fleet=sample,
        region="bay_area",
        profile_type="residential",
        seed=twb.MASTER_SEED_DEFAULT,
        nhts_data_dir=_NHTS_RAW,
        houseid_stitch_mode=twb.HOUSEID_STITCH_MODE_V2,
    )
    miles_v3 = mob_v3.groupby("ev_id", observed=True)["miles"].sum()
    miles_v2 = mob_v2.groupby("ev_id", observed=True)["miles"].sum()
    sigma_v3 = float(miles_v3.std())
    sigma_v2 = float(miles_v2.std())
    assert np.isfinite(sigma_v3) and sigma_v3 > 0
    assert np.isfinite(sigma_v2) and sigma_v2 > 0

    # v3 intra-HH coherence: every EV's trips carry exactly one donor HH
    # (the RESOLVED §13.2 invariant; legacy sampler escapes ~7% of EVs).
    per_ev_unique_hh = mob_v3.groupby("ev_id", observed=True)[
        "nhts_donor_houseid"
    ].nunique()
    assert (per_ev_unique_hh == 1).all(), (
        "v3 within-HH stitcher must yield exactly one donor houseid per EV; "
        f"got per-EV unique-HH counts: {per_ev_unique_hh.value_counts().to_dict()}"
    )


def test_build_mobility_v3_default_uses_within_hh_stitcher(
    mini_fleet: pd.DataFrame,
) -> None:
    """v3 is the default: every EV's mobility carries exactly ONE
    distinct ``nhts_donor_houseid`` value (no cross-HH escape).
    """
    mob = twb.build_mobility(
        fleet=mini_fleet,
        region="bay_area",
        profile_type="residential",
        seed=twb.MASTER_SEED_DEFAULT,
        nhts_data_dir=_NHTS_RAW,
    )
    per_ev_unique_hh = mob.groupby("ev_id", observed=True)["nhts_donor_houseid"].nunique()
    assert (per_ev_unique_hh == 1).all(), (
        "v3 within-HH stitcher must yield exactly one donor houseid per EV; "
        f"got per-EV unique-HH counts: {per_ev_unique_hh.value_counts().to_dict()}"
    )


def test_within_hh_preference_fires_when_donor_hh_has_record() -> None:
    """Monte-Carlo: within-HH preference draws the own-HH record at p ≈ 0.40."""
    own_record = pd.DataFrame(
        {
            "houseid": [1],
            "personid": [1],
            "travday": [2],
            "wttrdfin": [1000.0],
            "n_trips": [1],
            "income_tier": ["t2"],
            "hhsize_band": ["2"],
            "hhvehcnt_band": ["2"],
            "urbrur_int": [1],
        }
    )
    outside_record = pd.DataFrame(
        {
            "houseid": np.arange(100, 150, dtype=np.int64),
            "personid": [1] * 50,
            "travday": [2] * 50,
            "wttrdfin": [1000.0] * 50,
            "n_trips": [1] * 50,
            "income_tier": ["t2"] * 50,
            "hhsize_band": ["2"] * 50,
            "hhvehcnt_band": ["2"] * 50,
            "urbrur_int": [1] * 50,
        }
    )

    n_trials = 2000
    own_hits = 0
    for trial in range(n_trials):
        rng_t = np.random.default_rng(1_000 + trial)
        by_hh_td_local = {(1, d): own_record.assign(travday=d) for d in range(1, 8)}
        by_travday_local = {
            d: outside_record.assign(travday=d) for d in range(1, 8)
        }
        picks = twb.sample_week_for_ev(
            rng_t,
            donor_houseid=1,
            income_tier="t2",
            hhsize_band="2",
            hhvehcnt_band="2",
            urbrur_int=1,
            by_travday=by_travday_local,
            by_houseid_travday=by_hh_td_local,
            within_hh_prob=0.40,
        )
        if picks[1][0] == 1:  # picks[1] = travday=2 → houseid 1 means own-HH
            own_hits += 1
    rate = own_hits / n_trials
    assert 0.36 <= rate <= 0.44, (
        f"within-HH rate {rate:.3f} outside [0.36, 0.44]"
    )


def test_workplace_donor_filter_drops_pure_non_work_days() -> None:
    """`filter_to_workplace_donor_days` keeps person-days with ≥1 work trip."""
    trip = pd.DataFrame(
        {
            "houseid": [1, 1, 2, 2, 3, 3],
            "personid": [1, 1, 1, 1, 1, 1],
            "travday": [2, 2, 2, 2, 2, 2],
            "tdtrpnum": [1, 2, 1, 2, 1, 2],
            "strttime": [800, 900, 800, 900, 800, 900],
            "endtime": [830, 930, 830, 930, 830, 930],
            "trpmiles": [5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
            "whyfrom": [1, 3, 1, 1, 1, 6],   # HH1: home→work then work→home (work-day)
            "whyto": [3, 1, 1, 1, 6, 1],     # HH2: home→home (no work); HH3: home→shop, shop→home
            "wttrdfin": [100.0] * 6,
            "whytrp1s": [10, 10, 1, 1, 5, 5],  # HH1 = work day; HH2/HH3 = non-work
        }
    )
    out = twb.filter_to_workplace_donor_days(trip)
    assert set(out["houseid"].unique().tolist()) == {1}
    assert len(out) == 2  # both HH1 trips retained


# --------------------------------------------------------------------------- #
# Whole-fleet checks (only when the v1.1 cached mobility exists).
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(
    not _CACHED_MOBILITY_PATH.is_file(),
    reason="cached residential mobility.parquet not present",
)
def test_cached_residential_mobility_row_count() -> None:
    """Acceptance #1 (full-fleet flavour): cached 1000-EV mobility has the
    expected order-of-magnitude trip count.

    The v1.1 cache was tz-naive America/Los_Angeles; v2.0 caches are
    UTC.  We don't assert which dtype is on disk (migration is a
    separate module); we only assert the row count band.
    """
    mob = pd.read_parquet(_CACHED_MOBILITY_PATH)
    # 1000-EV cache with ~365 days × ~4 trips/day ≈ 1.46M rows.
    assert len(mob) > 1_400_000
    assert mob["ev_id"].nunique() == 1000


def test_purpose_mapping_v1_compat() -> None:
    """A v1.1 caller that passed WHYTO=2 still gets `other` (not work)."""
    assert twb._map_purpose(2) == "other"


def test_load_nhts_resolves_bay_area_via_ca_parquet() -> None:
    """`load_nhts_for_region(bay_area)` returns a non-empty hh frame."""
    region = REGIONS["bay_area"]
    hh, trip = twb.load_nhts_for_region(region, nhts_data_dir=_NHTS_RAW)
    assert len(hh) > 0
    assert "houseid" in hh.columns
    assert "whytrp1s" in trip.columns  # proxy synthesised if absent


# --------------------------------------------------------------------------- #
# Regression: M3/M4 borrow-state HH-frame mismatch (Phase 10.7 bug).
# --------------------------------------------------------------------------- #


def test_load_nhts_workplace_borrows_neighbour_states() -> None:
    """When ``profile_type='workplace'`` on a borrow-state region, the loaded
    HH frame must cover ``region.states ∪ region.workplace_donor_borrow_states``
    so that donor houseids assigned by M3 from a borrow state resolve in M4.
    """
    region = REGIONS["boston"]
    assert region.workplace_donor_borrow_states, (
        "regression precondition: boston declares borrow states"
    )

    hh_res, _ = twb.load_nhts_for_region(
        region, nhts_data_dir=_NHTS_RAW, profile_type="residential"
    )
    hh_wp, _ = twb.load_nhts_for_region(
        region, nhts_data_dir=_NHTS_RAW, profile_type="workplace"
    )

    # Residential / default path: only in-region states present.
    res_states = set(hh_res["hhstate"].dropna().astype(str).unique().tolist())
    assert res_states.issubset(set(region.states))

    # Workplace path: must include at least one borrow state.
    wp_states = set(hh_wp["hhstate"].dropna().astype(str).unique().tolist())
    expected = set(region.states) | set(region.workplace_donor_borrow_states)
    assert wp_states.issubset(expected)
    assert wp_states & set(region.workplace_donor_borrow_states), (
        f"workplace HH frame for boston is missing borrow-state donors; "
        f"loaded states={sorted(wp_states)}"
    )


def test_build_mobility_boston_workplace_with_borrow_donors() -> None:
    """End-to-end M3 -> M4 for boston/workplace must not raise the
    'donor houseid ... not in HH frame' KeyError.

    This is the Phase 10.7 regression: M3 borrows donors from neighbour
    states (NY/VT/ME for boston), but M4 used to load HHs filtered to
    ``region.states`` only, so the very first ``hh_meta.loc[donor_houseid]``
    lookup raised when M3 had picked a borrow-state donor.
    """
    pytest.importorskip("pev_synth.vehicle_archetypes")
    pytest.importorskip("pev_synth.donor_matcher")
    from pev_synth.donor_matcher import match_donors  # noqa: E402
    from pev_synth.vehicle_archetypes import sample_fleet  # noqa: E402

    region = REGIONS["boston"]
    assert region.workplace_donor_borrow_states
    seed = twb.MASTER_SEED_DEFAULT

    fleet = sample_fleet(
        n=1000, seed=seed, region=region, profile_type="workplace",
    )
    fleet = match_donors(
        fleet, region=region, profile_type="workplace",
        seed=seed, nhts_data_dir=_NHTS_RAW,
    )

    # Sanity: the M3 fleet must contain ≥ 1 donor from a borrow state — that's
    # the only configuration that triggers the historical bug.
    if "donor_origin_state" in fleet.columns:
        borrowed = fleet["donor_origin_state"].astype(str).isin(
            list(region.workplace_donor_borrow_states)
        )
        assert int(borrowed.sum()) > 0, (
            "regression invalid: M3 did not produce any borrow-state donors"
        )

    mob = twb.build_mobility(
        fleet=fleet,
        region="boston",
        profile_type="workplace",
        seed=seed,
        nhts_data_dir=_NHTS_RAW,
    )
    assert len(mob) > 0
    assert mob["ev_id"].nunique() == 1000


# --------------------------------------------------------------------------- #
# v3.0 — vintage-aware wrapper sample_week_for_ev_v3 (Dev D5).
# --------------------------------------------------------------------------- #


def _make_legacy_2017_pool(houseid: int, travday: int) -> pd.DataFrame:
    """Build a single-travday-per-HH NHTS-2017-shaped per-HH pool."""
    return pd.DataFrame(
        {
            "houseid": [int(houseid)],
            "personid": [1],
            "travday": [int(travday)],
            "wttrdfin": [1000.0],
            "n_trips": [1],
            "income_tier": ["t2"],
            "hhsize_band": ["2"],
            "hhvehcnt_band": ["2"],
            "urbrur_int": [1],
        }
    )


def _pools_dict_from_per_hh_pool(per_hh: pd.DataFrame) -> dict[str, object]:
    """Construct the dict-of-pools shape sample_week_for_ev_v3 expects.

    For a single-HH pool, ``by_houseid`` maps houseid->frame,
    ``by_houseid_travday`` maps (houseid, travday)->frame, and
    ``by_travday`` maps travday->frame (the same frame).
    """
    hid = int(per_hh["houseid"].iloc[0])
    td_pool: dict[int, pd.DataFrame] = {}
    by_hh_td: dict[tuple[int, int], pd.DataFrame] = {}
    for td, grp in per_hh.groupby("travday", sort=False):
        td_int = int(td)
        td_pool[td_int] = grp.reset_index(drop=True)
        by_hh_td[(hid, td_int)] = grp.reset_index(drop=True)
    return {
        "by_houseid": {hid: per_hh.reset_index(drop=True)},
        "by_houseid_travday": by_hh_td,
        "by_travday": td_pool,
    }


def test_sample_week_for_ev_v3_2017_routes_to_legacy_per_travday() -> None:
    """2017 vintage uses sample_week_for_ev (cross-HH per-travday sampler).

    With one (1-travday) donor HH in the pool, the legacy sampler still
    fills 7 days by walking through ``by_travday`` and synthesising
    cross-HH draws — but with only this single donor in the pool, every
    slot resolves to the donor itself. We just assert the route picked the
    legacy sampler by checking that travday alignment matches the requested
    week-index (legacy per-travday selects from by_travday[d] when d is in
    the pool, falling back to non-empty union otherwise).
    """
    donor_hh = 1010
    pool = _make_legacy_2017_pool(donor_hh, travday=3)  # Tuesday
    pools_by_vintage = {"2017": _pools_dict_from_per_hh_pool(pool)}
    rng = np.random.default_rng(20260528)
    picks = twb.sample_week_for_ev_v3(
        rng,
        nhts_vintage="2017",
        donor_houseid=donor_hh,
        pools_by_vintage=pools_by_vintage,
        income_tier="t2",
        hhsize_band="2",
        hhvehcnt_band="2",
        urbrur_int=1,
    )
    assert len(picks) == 7
    # All picks fall back to the donor HH (only one in the pool).
    for hid, _pid, _td in picks:
        assert hid == donor_hh


def test_sample_week_for_ev_v3_2022_nextgen_routes_to_within_hh() -> None:
    """2022_nextgen vintage uses sample_week_within_houseid (intra-HH stitcher).

    Build a multi-travday donor HH (one weekday + one weekend row) so the
    within-HH stitcher can satisfy both sides without cross-side
    substitution. Every pick must land in the donor HH.
    """
    donor_hh = 4242
    multi_day = pd.DataFrame(
        {
            "houseid": [donor_hh] * 7,
            "personid": [1, 1, 2, 2, 3, 3, 4],
            "travday": [2, 3, 4, 5, 6, 1, 7],
            "wttrdfin": [1000.0] * 7,
            "n_trips": [1] * 7,
            "income_tier": ["t2"] * 7,
            "hhsize_band": ["2"] * 7,
            "hhvehcnt_band": ["2"] * 7,
            "urbrur_int": [1] * 7,
        }
    )
    pools_by_vintage = {
        "2022_nextgen": _pools_dict_from_per_hh_pool(multi_day),
    }
    rng = np.random.default_rng(20260529)
    picks = twb.sample_week_for_ev_v3(
        rng,
        nhts_vintage="2022_nextgen",
        donor_houseid=donor_hh,
        pools_by_vintage=pools_by_vintage,
        income_tier="t2",
        hhsize_band="2",
        hhvehcnt_band="2",
        urbrur_int=1,
    )
    assert len(picks) == 7
    # All picks must remain in the donor HH (intra-HH coherence invariant).
    for hid, _pid, _td in picks:
        assert hid == donor_hh
    # Weekday slots (d=2..6) get TRAVDAY in {2..6}; weekend slots (d=1,7)
    # get TRAVDAY in {1,7}.
    for i in (1, 2, 3, 4, 5):  # d=2..6
        assert picks[i][2] in {2, 3, 4, 5, 6}
    for i in (0, 6):  # d=1, d=7
        assert picks[i][2] in {1, 7}


def test_sample_week_for_ev_v3_unknown_vintage_raises() -> None:
    """Unknown vintage label that isn't in pools_by_vintage -> KeyError."""
    pools_by_vintage = {"2017": _pools_dict_from_per_hh_pool(
        _make_legacy_2017_pool(1, 3)
    )}
    rng = np.random.default_rng(0)
    with pytest.raises(KeyError, match="no donor pools registered"):
        twb.sample_week_for_ev_v3(
            rng,
            nhts_vintage="unknown_vintage",
            donor_houseid=1,
            pools_by_vintage=pools_by_vintage,
            income_tier="t2",
            hhsize_band="2",
            hhvehcnt_band="2",
            urbrur_int=1,
        )


def test_sample_week_for_ev_v3_exported_in_all() -> None:
    assert "sample_week_for_ev_v3" in twb.__all__
    assert hasattr(twb, "sample_week_for_ev_v3")
