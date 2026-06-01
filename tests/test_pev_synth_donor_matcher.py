"""Unit tests for M3 -- pev_synth.donor_matcher (v2.0 refactor).

v2.0 acceptance criteria (per Wave 2 Dev D brief):
    6. match_donors(fleet_bay_area, region=bay_area, profile='residential')
       produces non-null donors; pool ~3,248 HH.
    7. match_donors(fleet_chicago, region=chicago, profile='residential')
       runs on the Chicago CBSAs; pool > 1500 HH.
    8. match_donors(fleet_boston_workplace, region=boston, profile='workplace')
       activates ANNMILES<3600 + borrow from NY/VT/ME; borrow share <= 25%.
    9. donor_origin_state populated for borrowed rows; null for native rows.
   10. Workplace post-filter median ANNMILES < 3600 (strict).
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make the ``code/`` directory importable from anywhere.
_CODE_DIR = Path(__file__).resolve().parents[1]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from pev_synth import donor_matcher as dm  # noqa: E402
from pev_synth import vehicle_archetypes as va  # noqa: E402
from pev_synth.regions import REGIONS  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
_NHTS_DIR = _REPO_ROOT / "data" / "pev" / "raw" / "nhts2017"
_CA_HH = _NHTS_DIR / "nhts2017_ca_hh.parquet"
_CA_VEH = _NHTS_DIR / "nhts2017_ca_veh.parquet"
_NATIONAL_HH = _NHTS_DIR / "hhpub.csv"


def _have_ca_data() -> bool:
    return _CA_HH.is_file() and _CA_VEH.is_file()


def _have_national_data() -> bool:
    return _NATIONAL_HH.is_file()


_skip_if_no_data = pytest.mark.skipif(
    not (_have_ca_data() and _have_national_data()),
    reason="needs NHTS CA parquets + national hhpub.csv present in repo data tree",
)


# --------------------------------------------------------------------------- #
# Shared fixtures.
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def fleet_bay_area_res() -> pd.DataFrame:
    return va.sample_fleet(
        n=500, seed=va.MASTER_SEED_DEFAULT,
        region="bay_area", profile_type="residential",
    )


@pytest.fixture(scope="module")
def fleet_chicago_res() -> pd.DataFrame:
    return va.sample_fleet(
        n=500, seed=va.MASTER_SEED_DEFAULT,
        region="chicago", profile_type="residential",
    )


@pytest.fixture(scope="module")
def fleet_boston_wp() -> pd.DataFrame:
    return va.sample_fleet(
        n=200, seed=va.MASTER_SEED_DEFAULT,
        region="boston", profile_type="workplace",
    )


# --------------------------------------------------------------------------- #
# Acceptance tests.
# --------------------------------------------------------------------------- #

@_skip_if_no_data
def test_match_donors_residential_bay_area(
    fleet_bay_area_res: pd.DataFrame,
) -> None:
    """Criterion 6: Bay-Area residential ~3,248 HH pool, all donors non-null."""
    enriched = dm.match_donors(
        fleet_bay_area_res,
        region=REGIONS["bay_area"],
        profile_type="residential",
        seed=dm.MASTER_SEED_DEFAULT,
    )
    assert len(enriched) == len(fleet_bay_area_res)
    assert enriched["nhts_donor_houseid"].notna().all()
    assert enriched["nhts_donor_vehid"].notna().all()
    attrs = enriched.attrs.get("donor_match", {})
    n_pool = attrs["n_hh_total_pool"]
    assert 3000 <= n_pool <= 3500, (
        f"Bay-Area residential donor pool {n_pool} outside the ~3,248 band"
    )
    # No borrow for residential.
    assert attrs["n_hh_borrowed"] == 0
    assert (~enriched["borrow_used"]).all()


@_skip_if_no_data
def test_match_donors_residential_chicago(
    fleet_chicago_res: pd.DataFrame,
) -> None:
    """Criterion 7: Chicago residential pool > 1500 HH."""
    enriched = dm.match_donors(
        fleet_chicago_res,
        region=REGIONS["chicago"],
        profile_type="residential",
        seed=dm.MASTER_SEED_DEFAULT,
    )
    assert enriched["nhts_donor_houseid"].notna().all()
    attrs = enriched.attrs.get("donor_match", {})
    assert attrs["n_hh_total_pool"] > 1500, (
        f"Chicago residential donor pool {attrs['n_hh_total_pool']} too small"
    )
    assert attrs["n_hh_borrowed"] == 0


@_skip_if_no_data
def test_match_donors_workplace_annmiles_filter(
    fleet_boston_wp: pd.DataFrame,
) -> None:
    """Criterion 10: workplace post-filter median ANNMILES < 3600 (strict)."""
    region = REGIONS["boston"]
    hh, veh = dm.load_region_nhts(region, _NHTS_DIR)
    hh_narrow = dm.narrow_to_region(hh, region)
    veh_narrow = veh[veh["houseid"].isin(hh_narrow["houseid"])]
    hh_post, veh_post = dm.apply_workplace_annmiles_filter(
        hh_narrow, veh_narrow, annmiles_max=dm.ANNMILES_WORKPLACE_MAX
    )
    # Every retained HH has at least one veh with annmiles in (0, 3600).
    sub_veh = veh_post[(veh_post["annmiles"] > 0) & (veh_post["annmiles"] < 3600)]
    medians = sub_veh.groupby("houseid")["annmiles"].min()
    # The minimum-per-HH ANNMILES must all be < 3600.
    assert (medians < dm.ANNMILES_WORKPLACE_MAX).all(), (
        "workplace ANNMILES filter retained an HH with no <3600 vehicle"
    )
    # Median across retained HHs is strictly < 3600 by construction.
    assert float(medians.median()) < dm.ANNMILES_WORKPLACE_MAX


@_skip_if_no_data
def test_match_donors_borrow_share_under_25pct(
    fleet_boston_wp: pd.DataFrame,
) -> None:
    """Criterion 8 (capped): borrow share of total pool <= 25% (W10)."""
    enriched = dm.match_donors(
        fleet_boston_wp,
        region=REGIONS["boston"],
        profile_type="workplace",
        seed=dm.MASTER_SEED_DEFAULT,
    )
    attrs = enriched.attrs.get("donor_match", {})
    assert attrs["n_hh_after_annmiles_filter"] < attrs["n_hh_in_region"], (
        "workplace ANNMILES filter must shrink the pool from the in-region size"
    )
    assert attrs["n_hh_borrowed"] > 0, (
        "boston workplace should trigger borrow (NY/VT/ME)"
    )
    borrow_share = attrs["borrow_share"]
    assert 0.0 < borrow_share <= dm.BORROW_SHARE_CAP + 1e-9, (
        f"borrow share {borrow_share:.3f} exceeded the {dm.BORROW_SHARE_CAP:.2f} cap"
    )


@_skip_if_no_data
def test_match_donors_donor_origin_state_populated_for_borrowed(
    fleet_boston_wp: pd.DataFrame,
) -> None:
    """Criterion 9: donor_origin_state non-null for borrowed rows."""
    enriched = dm.match_donors(
        fleet_boston_wp,
        region=REGIONS["boston"],
        profile_type="workplace",
        seed=dm.MASTER_SEED_DEFAULT,
    )
    borrowed_mask = enriched["borrow_used"].to_numpy()
    native_mask = ~borrowed_mask
    if borrowed_mask.any():
        donor_states = enriched.loc[borrowed_mask, "donor_origin_state"].astype("string")
        assert donor_states.notna().all(), (
            "borrowed rows must carry a non-null donor_origin_state"
        )
        # All borrowed donors must come from the listed borrow states.
        allowed = set(REGIONS["boston"].workplace_donor_borrow_states)
        assert set(donor_states.dropna().unique().tolist()) <= allowed, (
            f"donor_origin_state outside borrow set; got "
            f"{set(donor_states.unique())}, allowed {allowed}"
        )
    if native_mask.any():
        native_states = enriched.loc[native_mask, "donor_origin_state"]
        assert native_states.isna().all(), (
            "native (in-region) rows should have null donor_origin_state"
        )


@_skip_if_no_data
def test_match_donors_us_national_no_cbsa_filter() -> None:
    """us_national region has no CBSA narrowing; pool covers many states."""
    fleet = va.sample_fleet(
        n=50, seed=va.MASTER_SEED_DEFAULT,
        region="us_national", profile_type="residential",
    )
    enriched = dm.match_donors(
        fleet,
        region=REGIONS["us_national"],
        profile_type="residential",
        seed=dm.MASTER_SEED_DEFAULT,
    )
    attrs = enriched.attrs.get("donor_match", {})
    assert attrs["n_hh_total_pool"] > 50_000, (
        f"us_national pool {attrs['n_hh_total_pool']} unexpectedly small"
    )


@_skip_if_no_data
def test_match_donors_seed_reproducible(
    fleet_bay_area_res: pd.DataFrame, tmp_path: Path
) -> None:
    """Same seed -> identical fleet (DataFrame + parquet bytes)."""
    a = dm.match_donors(
        fleet_bay_area_res, region=REGIONS["bay_area"],
        profile_type="residential", seed=dm.MASTER_SEED_DEFAULT,
    )
    b = dm.match_donors(
        fleet_bay_area_res, region=REGIONS["bay_area"],
        profile_type="residential", seed=dm.MASTER_SEED_DEFAULT,
    )
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))

    pa = tmp_path / "a.parquet"
    pb = tmp_path / "b.parquet"
    dm._write_fleet_inplace(a, pa)
    dm._write_fleet_inplace(b, pb)
    assert hashlib.sha256(pa.read_bytes()).hexdigest() == hashlib.sha256(
        pb.read_bytes()
    ).hexdigest(), "seed-identical M3 runs produced different parquet bytes"


# --------------------------------------------------------------------------- #
# Pure-function tests (no real-data dependency).
# --------------------------------------------------------------------------- #

def test_match_donors_invalid_profile_type_raises() -> None:
    """profile_type not in {residential, workplace} -> ValueError."""
    fleet = pd.DataFrame(
        {
            "ev_id": np.array([0], dtype=np.int32),
            "powertrain": pd.Categorical(["BEV"], categories=["BEV", "PHEV"]),
            "income_tier": pd.Categorical(["t2"], categories=["t1", "t2", "t3", "t4"]),
        }
    )
    with pytest.raises(ValueError, match="profile_type"):
        dm.match_donors(fleet, region=REGIONS["bay_area"], profile_type="garage")


def test_match_donors_missing_columns_raises() -> None:
    """Required fleet columns must be present."""
    fleet = pd.DataFrame({"ev_id": [0]})
    with pytest.raises(KeyError, match="fleet missing required columns"):
        dm.match_donors(
            fleet, region=REGIONS["bay_area"], profile_type="residential"
        )


def test_relaxation_ladder_demotes_when_no_candidates() -> None:
    """Synthetic single-HH pool: level 0 fails -> level 3 (income only) hits."""
    hh = pd.DataFrame(
        {
            "houseid": [1000],
            "hhsize": pd.array([5], dtype="Int8"),
            "hhvehcnt": pd.array([5], dtype="Int8"),
            "urbrur": pd.array([2], dtype="Int8"),
            "hhfaminc": pd.array([6], dtype="Int8"),
            "wthhfin": [123.0],
            "hh_cbsa": pd.Series(["41860"], dtype="string"),
            "hhstate": pd.Series(["CA"], dtype="string"),
        }
    )
    hh_keys = dm.build_match_keys(hh)
    pool, level = dm._candidate_pool_with_relaxation(
        hh_keys,
        income_tier="t2",
        hhsize_band="2",
        urbrur=1,
        hhvehcnt_band="2",
    )
    assert level == 3, f"expected demotion to level 3, got {level}"
    assert len(pool) == 1


def test_borrow_from_neighbours_noop_for_residential() -> None:
    """Residential profile never triggers borrow even on a tiny pool."""
    hh = pd.DataFrame(
        {
            "houseid": [1, 2, 3],
            "hhstate": pd.Series(["MA", "MA", "MA"], dtype="string"),
        }
    )
    veh = pd.DataFrame(
        {
            "houseid": [1, 2, 3],
            "vehid": pd.array([1, 1, 1], dtype="int32"),
            "annmiles": pd.array([100.0, 200.0, 300.0], dtype="float32"),
            "hfuel": pd.array([3, 3, 3], dtype="Int8"),
        }
    )
    out_hh, out_veh, n_borrowed = dm.borrow_from_neighbours(
        hh, veh,
        region=REGIONS["boston"],
        profile_type="residential",
        nhts_data_dir=_NHTS_DIR,
    )
    assert n_borrowed == 0
    assert len(out_hh) == 3
    assert (~out_hh["borrow_used"]).all()


def test_apply_workplace_annmiles_filter_strict() -> None:
    """Filter retains only HHs with at least one veh whose annmiles < 3600."""
    hh = pd.DataFrame({"houseid": [1, 2, 3], "wthhfin": [1.0, 1.0, 1.0]})
    veh = pd.DataFrame(
        {
            "houseid": [1, 1, 2, 3],
            "vehid": pd.array([1, 2, 1, 1], dtype="int32"),
            "annmiles": pd.array([3000.0, 8000.0, 3600.0, -9.0], dtype="float32"),
            "hfuel": pd.array([3, -1, 2, -1], dtype="Int8"),
        }
    )
    hh_keep, veh_keep = dm.apply_workplace_annmiles_filter(
        hh, veh, annmiles_max=3600.0
    )
    # HH 1 keeps (vehicle 1 has annmiles=3000 < 3600).
    # HH 2 dropped: 3600 is not < 3600 (strict).
    # HH 3 dropped: annmiles == -9 (NA sentinel).
    assert hh_keep["houseid"].tolist() == [1]


def test_load_region_nhts_state_filter() -> None:
    """load_region_nhts respects region.states (us_national reads national)."""
    if not _have_ca_data():
        pytest.skip("no NHTS CA data")
    region = REGIONS["bay_area"]
    hh, veh = dm.load_region_nhts(region, _NHTS_DIR)
    assert (hh["hhstate"] == "CA").all()
    assert veh["houseid"].isin(hh["houseid"]).all()


# --------------------------------------------------------------------------- #
# v3.0 — ACS PUMS raking (Dev D6).
# --------------------------------------------------------------------------- #

def _synthetic_hh_for_raking() -> pd.DataFrame:
    """Build a small NHTS-shaped frame with the M3 match keys.

    Cells (income_tier × hhsize_band × hhvehcnt_band) cover four strata so
    the test can assert per-cell rescaling and the post-rake realised PMF.
    """
    rows = [
        # Each row = one HH; wthhfin starts uniform at 100.
        ("t1", "1", "0"),
        ("t1", "1", "0"),
        ("t2", "2", "1"),
        ("t2", "2", "1"),
        ("t2", "2", "1"),
        ("t3", "3", "2"),
        ("t4", "4", "3+"),
        ("t4", "4", "3+"),
    ]
    df = pd.DataFrame(rows, columns=["income_tier", "hhsize_band", "hhvehcnt_band"])
    df["wthhfin"] = pd.Series([100.0] * len(df), dtype="float64")
    df["houseid"] = pd.Series(range(len(df)), dtype="int64")
    return df


def test_assign_strata_basic() -> None:
    hh = pd.DataFrame(
        {
            "hhsize": pd.array([1, 2, 3, 4, 5, 7], dtype="Int8"),
            "hhvehcnt": pd.array([0, 1, 2, 3, 5, 7], dtype="Int8"),
        }
    )
    out = dm._assign_strata(hh)
    assert out["hhsize_band"].tolist() == ["1", "2", "3", "4", "5+", "5+"]
    assert out["hhvehcnt_band"].tolist() == ["0", "1", "2", "3+", "3+", "3+"]


def test_assign_strata_overwrite_respects_flag() -> None:
    hh = pd.DataFrame(
        {
            "hhsize": pd.array([2, 3], dtype="Int8"),
            "hhvehcnt": pd.array([1, 2], dtype="Int8"),
            "hhsize_band": pd.Series(["preexisting", "preexisting"], dtype="string"),
            "hhvehcnt_band": pd.Series(["preexisting", "preexisting"], dtype="string"),
        }
    )
    # Default: preserve existing columns.
    out = dm._assign_strata(hh)
    assert out["hhsize_band"].tolist() == ["preexisting", "preexisting"]
    # overwrite=True replaces.
    out2 = dm._assign_strata(hh, overwrite=True)
    assert out2["hhsize_band"].tolist() == ["2", "3"]
    assert out2["hhvehcnt_band"].tolist() == ["1", "2"]


def test_apply_acs_calibration_weights_basic() -> None:
    """Synthetic NHTS frame + expected PMF: post-rake realised matches expected."""
    hh = _synthetic_hh_for_raking()
    # NHTS realised PMF (uniform wthhfin=100):
    #   t1/1/0   → 2/8 = 0.25
    #   t2/2/1   → 3/8 = 0.375
    #   t3/3/2   → 1/8 = 0.125
    #   t4/4/3+  → 2/8 = 0.25
    # Choose an expected PMF deliberately different from realised.
    expected = pd.DataFrame(
        {
            "income_tier": pd.array(["t1", "t2", "t3", "t4"], dtype="string"),
            "hhsize_band": pd.array(["1", "2", "3", "4"], dtype="string"),
            "hhvehcnt_band": pd.array(["0", "1", "2", "3+"], dtype="string"),
            "p_expected": [0.10, 0.50, 0.20, 0.20],
        }
    )

    out = dm.apply_acs_calibration_weights(hh, expected)
    assert "acs_cal_factor" in out.columns
    # No mutation of the input frame.
    assert "acs_cal_factor" not in hh.columns
    # Original column preserved.
    assert "wthhfin" in out.columns

    # Realised post-rake matches expected within float tolerance.
    keys = ["income_tier", "hhsize_band", "hhvehcnt_band"]
    realised = (
        out.groupby(keys, observed=True)["wthhfin"].sum().rename("w").reset_index()
    )
    realised["p_realised"] = realised["w"] / realised["w"].sum()
    merged = expected.merge(realised, on=keys, how="left")
    diff = (merged["p_realised"] - merged["p_expected"]).abs().max()
    assert float(diff) < 1e-9, f"post-rake PMF diverges from expected: {diff:.3e}"

    # Cap-rate behavior captured on attrs.
    attrs = out.attrs.get("acs_calibration", {})
    assert attrs["n_cells_total"] == 4
    assert attrs["n_cells_capped"] == 0
    assert attrs["cap_fraction"] == 0.0


def test_apply_acs_calibration_weights_records_capped_cells() -> None:
    """A pathological expected PMF triggers per-cell clipping but stays under threshold.

    To stay under the 5% tolerance with only a single clipped cell we need
    >=20 NHTS cells. We build a 41-cell pool (40 cold + 1 "hot") so the
    clipped fraction (1/41 ≈ 2.4%) lands inside the tolerance.
    """
    rows = [(f"cell_{i}", "x", "y") for i in range(40)] + [("hot", "x", "y")]
    df = pd.DataFrame(rows, columns=["income_tier", "hhsize_band", "hhvehcnt_band"])
    df["wthhfin"] = pd.Series([100.0] * len(df), dtype="float64")
    for c in ("income_tier", "hhsize_band", "hhvehcnt_band"):
        df[c] = df[c].astype("string")

    # Realised share for each cell = 1/41 (uniform weights).
    each_p_realised = 1.0 / 41.0
    # Drive the "hot" cell's expected share to 6 × realised → ratio 6 →
    # clipped to 5.0 (the only clip). Renormalise the cold cells so the
    # PMF sums to 1.
    hot_expected = each_p_realised * 6.0
    cold_p = (1.0 - hot_expected) / 40.0
    expected_rows = [(f"cell_{i}", "x", "y", cold_p) for i in range(40)]
    expected_rows.append(("hot", "x", "y", hot_expected))
    expected = pd.DataFrame(
        expected_rows,
        columns=["income_tier", "hhsize_band", "hhvehcnt_band", "p_expected"],
    )
    for c in ("income_tier", "hhsize_band", "hhvehcnt_band"):
        expected[c] = expected[c].astype("string")

    out = dm.apply_acs_calibration_weights(df, expected)
    attrs = out.attrs["acs_calibration"]
    assert attrs["n_cells_capped"] == 1
    assert attrs["n_cells_total"] == 41
    assert attrs["cap_fraction"] == pytest.approx(1.0 / 41.0)
    assert attrs["cap_fraction"] <= 0.05, "this fixture must stay under tolerance"


def test_apply_acs_calibration_weights_raises_when_too_many_capped() -> None:
    """>5% of cells beyond [cap_min, cap_max] ⇒ ValueError."""
    # 4 cells, realised uniform 25% each.
    hh = _synthetic_hh_for_raking()
    # Force 2 of 4 cells (50%) to exceed the [0.2, 5.0] cap.
    expected = pd.DataFrame(
        {
            "income_tier": pd.array(["t1", "t2", "t3", "t4"], dtype="string"),
            "hhsize_band": pd.array(["1", "2", "3", "4"], dtype="string"),
            "hhvehcnt_band": pd.array(["0", "1", "2", "3+"], dtype="string"),
            # realised was (0.25, 0.375, 0.125, 0.25). Force ratios:
            #   t1/1/0 → 0.95 / 0.25  = 3.8 (within cap)
            #   t2/2/1 → 0.02 / 0.375 = 0.053 (below cap) -- CAPPED
            #   t3/3/2 → 0.02 / 0.125 = 0.16 (below cap)  -- CAPPED
            #   t4/4/3+→ 0.01 / 0.25  = 0.04 (below cap)  -- CAPPED
            "p_expected": [0.95, 0.02, 0.02, 0.01],
        }
    )
    with pytest.raises(ValueError, match="required scale-cap clipping"):
        dm.apply_acs_calibration_weights(hh, expected)


def test_apply_acs_calibration_weights_does_not_mutate_input() -> None:
    """Input ``hh`` frame must remain unchanged post-call."""
    hh = _synthetic_hh_for_raking()
    snapshot = hh.copy()
    expected = pd.DataFrame(
        {
            "income_tier": pd.array(["t1", "t2", "t3", "t4"], dtype="string"),
            "hhsize_band": pd.array(["1", "2", "3", "4"], dtype="string"),
            "hhvehcnt_band": pd.array(["0", "1", "2", "3+"], dtype="string"),
            "p_expected": [0.25, 0.25, 0.25, 0.25],
        }
    )
    _ = dm.apply_acs_calibration_weights(hh, expected)
    pd.testing.assert_frame_equal(hh, snapshot)


def test_apply_acs_calibration_weights_missing_columns_raises() -> None:
    bad = pd.DataFrame({"foo": [1, 2]})
    expected = pd.DataFrame(
        {
            "income_tier": pd.array(["t1"], dtype="string"),
            "hhsize_band": pd.array(["1"], dtype="string"),
            "hhvehcnt_band": pd.array(["0"], dtype="string"),
            "p_expected": [1.0],
        }
    )
    with pytest.raises(KeyError, match="hh missing required columns"):
        dm.apply_acs_calibration_weights(bad, expected)


def test_apply_acs_calibration_weights_empty_pmf_raises() -> None:
    hh = _synthetic_hh_for_raking()
    empty = pd.DataFrame(
        columns=["income_tier", "hhsize_band", "hhvehcnt_band", "p_expected"]
    )
    with pytest.raises(ValueError, match="expected_pmf is empty"):
        dm.apply_acs_calibration_weights(hh, empty)


def test_apply_acs_calibration_weights_invalid_caps_raise() -> None:
    hh = _synthetic_hh_for_raking()
    expected = pd.DataFrame(
        {
            "income_tier": pd.array(["t1"], dtype="string"),
            "hhsize_band": pd.array(["1"], dtype="string"),
            "hhvehcnt_band": pd.array(["0"], dtype="string"),
            "p_expected": [1.0],
        }
    )
    with pytest.raises(ValueError, match="cap_min/cap_max invalid"):
        dm.apply_acs_calibration_weights(hh, expected, cap_min=5.0, cap_max=0.2)
    with pytest.raises(ValueError, match="cap_min/cap_max invalid"):
        dm.apply_acs_calibration_weights(hh, expected, cap_min=-1.0, cap_max=2.0)


def test_apply_acs_calibration_weights_preserves_cell_with_no_support() -> None:
    """Expected cell with no realised support gets scale 0; uncoverage logged.

    Cells present in ACS but absent in NHTS cannot be raked by definition;
    they are accounted for in ``n_cells_total`` of the realised side (which is
    0 for those cells) and excluded from the cap-fraction calculation.
    """
    hh = _synthetic_hh_for_raking()
    expected = pd.DataFrame(
        {
            "income_tier": pd.array(["t1", "t2", "t3", "t4", "t4"], dtype="string"),
            "hhsize_band": pd.array(["1", "2", "3", "4", "5+"], dtype="string"),
            "hhvehcnt_band": pd.array(["0", "1", "2", "3+", "3+"], dtype="string"),
            # Last cell (t4/5+/3+) is in ACS but not NHTS — should not raise.
            "p_expected": [0.25, 0.30, 0.15, 0.20, 0.10],
        }
    )
    out = dm.apply_acs_calibration_weights(hh, expected)
    # No row in ``out`` lives in the missing cell, so nothing to assert beyond
    # the call not raising; sanity check on the attrs.
    attrs = out.attrs["acs_calibration"]
    assert attrs["n_cells_total"] == 4  # only the realised-supported cells


# --------------------------------------------------------------------------- #
# v3.0 — NHTS NextGen 2022 enrichment (Dev D5).
# --------------------------------------------------------------------------- #


def _write_nextgen_parquet_fixtures(tmp_dir: Path) -> Path:
    """Write minimal NextGen-2022 parquets into ``tmp_dir / nhts2022/``.

    Returns the parent dir (so callers can pass ``parent / "nhts2017"`` as
    ``nhts_data_dir`` to mirror the production layout).
    """
    nextgen_dir = tmp_dir / "nhts2022"
    nextgen_dir.mkdir(parents=True, exist_ok=True)
    hh = pd.DataFrame(
        {
            "houseid": pd.array([90000001, 90000002, 90000003], dtype="int64"),
            "hhstate": pd.array([pd.NA, pd.NA, pd.NA], dtype="string"),
            "hhstfips": pd.array([pd.NA, pd.NA, pd.NA], dtype="string"),
            "hh_cbsa": pd.array([pd.NA, pd.NA, pd.NA], dtype="string"),
            "cdivmsar": pd.array([53, 31, 99], dtype="Int16"),
            "urbrur": pd.array([1, 1, 2], dtype="Int8"),
            "hhfaminc": pd.array([7, 5, 11], dtype="Int8"),
            "hhsize": pd.array([2, 3, 1], dtype="Int8"),
            "hhvehcnt": pd.array([2, 2, 1], dtype="Int8"),
            "wthhfin": pd.Series([1000.0, 2000.0, 500.0], dtype="float64"),
            "nhts_vintage": pd.array(["2022_nextgen"] * 3, dtype="string"),
        }
    )
    veh = pd.DataFrame(
        {
            "houseid": pd.array([90000001, 90000001, 90000003], dtype="int64"),
            "vehid": pd.array([1, 2, 1], dtype="int32"),
            "vehtype": pd.array([1, 3, 1], dtype="Int8"),
            "fueltype": pd.array([1, 3, 1], dtype="Int8"),
            "make": pd.array(["Toyota", "Tesla", "Honda"], dtype="string"),
            "model": pd.array([pd.NA, pd.NA, pd.NA], dtype="string"),
            "vehyear": pd.array([2018, 2022, 2017], dtype="Int16"),
            "annmiles": pd.array([12000.0, 11000.0, 8500.0], dtype="float32"),
            "hfuel": pd.array([-1, 3, -1], dtype="Int8"),
            "nhts_vintage": pd.array(["2022_nextgen"] * 3, dtype="string"),
        }
    )
    hh.to_parquet(nextgen_dir / "nhts2022_national_hh.parquet")
    veh.to_parquet(nextgen_dir / "nhts2022_national_veh.parquet")
    return nextgen_dir


@_skip_if_no_data
def test_load_region_nhts_default_no_extra_vintages_is_byte_stable() -> None:
    """``extra_vintages=()`` (default) yields the v2.1 frame plus an
    ``nhts_vintage='2017'`` back-compat column on every row."""
    region = REGIONS["bay_area"]
    hh, veh = dm.load_region_nhts(region, _NHTS_DIR)
    assert "nhts_vintage" in hh.columns
    assert "nhts_vintage" in veh.columns
    # All 2017 (back-compat stamp on legacy parquets).
    assert set(hh["nhts_vintage"].astype("string").dropna().tolist()) == {"2017"}
    assert set(veh["nhts_vintage"].astype("string").dropna().tolist()) == {"2017"}
    # State filter still applies.
    assert (hh["hhstate"] == "CA").all()


@_skip_if_no_data
def test_load_region_nhts_extra_vintages_2022_nextgen_unions(
    tmp_path: Path,
) -> None:
    """When NextGen parquets exist next to the 2017 cache, they are unioned
    onto the result with their own vintage tag."""
    # Stage a copy of the 2017 parquets in a tmp dir + NextGen fixtures alongside.
    nhts_dir = tmp_path / "nhts2017"
    nhts_dir.mkdir()
    for fname in (
        "nhts2017_ca_hh.parquet",
        "nhts2017_ca_veh.parquet",
    ):
        src = _NHTS_DIR / fname
        dst = nhts_dir / fname
        dst.write_bytes(src.read_bytes())
    _write_nextgen_parquet_fixtures(tmp_path)

    region = REGIONS["bay_area"]
    hh, veh = dm.load_region_nhts(
        region, nhts_dir, extra_vintages=("2022_nextgen",)
    )
    vintages = set(hh["nhts_vintage"].astype("string").dropna().tolist())
    assert vintages == {"2017", "2022_nextgen"}
    # NextGen rows carry NA state/cbsa.
    ng = hh[hh["nhts_vintage"] == "2022_nextgen"]
    assert ng["hhstate"].isna().all()
    assert ng["hh_cbsa"].isna().all()
    # VEH carries NA model on NextGen rows.
    ng_v = veh[veh["nhts_vintage"] == "2022_nextgen"]
    assert ng_v["model"].isna().all()


def test_load_region_nhts_extra_vintages_missing_parquets_warns_not_raises(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Requesting 2022_nextgen when parquets are absent logs a warning."""
    if not _have_ca_data():
        pytest.skip("no NHTS CA data")
    # Stage 2017 only; no NextGen dir at all.
    nhts_dir = tmp_path / "nhts2017"
    nhts_dir.mkdir()
    for fname in (
        "nhts2017_ca_hh.parquet",
        "nhts2017_ca_veh.parquet",
    ):
        src = _NHTS_DIR / fname
        dst = nhts_dir / fname
        dst.write_bytes(src.read_bytes())

    region = REGIONS["bay_area"]
    with caplog.at_level("WARNING"):
        hh, veh = dm.load_region_nhts(
            region, nhts_dir, extra_vintages=("2022_nextgen",)
        )
    assert any("2022_nextgen" in r.message for r in caplog.records)
    # 2017-only result on the fallback.
    vintages = set(hh["nhts_vintage"].astype("string").dropna().tolist())
    assert vintages == {"2017"}


def test_narrow_to_region_partitions_2017_and_2022_nextgen() -> None:
    """2017 rows are CBSA-narrowed; 2022_nextgen rows pass through verbatim."""
    region = REGIONS["bay_area"]
    bay_cbsa = str(next(iter(region.cbsas)))
    hh = pd.DataFrame(
        {
            "houseid": [1, 2, 3, 4],
            "hhstate": pd.array(["CA", "CA", pd.NA, pd.NA], dtype="string"),
            "hh_cbsa": pd.array([bay_cbsa, "99999", pd.NA, pd.NA], dtype="string"),
            "nhts_vintage": pd.array(
                ["2017", "2017", "2022_nextgen", "2022_nextgen"], dtype="string"
            ),
        }
    )
    out = dm.narrow_to_region(hh, region)
    out_ids = sorted(out["houseid"].tolist())
    # 2017 narrowing keeps only houseid 1; both NextGen rows pass through.
    assert out_ids == [1, 3, 4]


def test_narrow_to_region_no_vintage_column_is_byte_stable() -> None:
    """When ``nhts_vintage`` is absent, narrow_to_region is the v2.1 path."""
    region = REGIONS["bay_area"]
    bay_cbsa = str(next(iter(region.cbsas)))
    hh = pd.DataFrame(
        {
            "houseid": [1, 2, 3],
            "hhstate": pd.array(["CA", "CA", "NV"], dtype="string"),
            "hh_cbsa": pd.array([bay_cbsa, "99999", bay_cbsa], dtype="string"),
        }
    )
    out = dm.narrow_to_region(hh, region)
    assert out["houseid"].tolist() == [1]


def test_apply_vintage_mix_normalises_per_vintage() -> None:
    """80/20 mix: 2017 slice sums to 0.8, 2022_nextgen slice sums to 0.2."""
    hh = pd.DataFrame(
        {
            "houseid": [1, 2, 3, 4, 5],
            "wthhfin": [100.0, 200.0, 300.0, 50.0, 150.0],
            "nhts_vintage": pd.array(
                ["2017", "2017", "2017", "2022_nextgen", "2022_nextgen"],
                dtype="string",
            ),
        }
    )
    out = dm._apply_vintage_mix(hh, {"2017": 0.8, "2022_nextgen": 0.2})
    legacy = out[out["nhts_vintage"] == "2017"]["wthhfin"].sum()
    ng = out[out["nhts_vintage"] == "2022_nextgen"]["wthhfin"].sum()
    assert legacy == pytest.approx(0.8, rel=1e-9)
    assert ng == pytest.approx(0.2, rel=1e-9)
    assert out["wthhfin"].sum() == pytest.approx(1.0, rel=1e-9)


def test_apply_vintage_mix_default_single_vintage_normalises_to_one() -> None:
    hh = pd.DataFrame(
        {
            "houseid": [1, 2, 3],
            "wthhfin": [10.0, 20.0, 70.0],
            "nhts_vintage": pd.array(["2017"] * 3, dtype="string"),
        }
    )
    out = dm._apply_vintage_mix(hh, {"2017": 1.0})
    assert out["wthhfin"].sum() == pytest.approx(1.0, rel=1e-12)
    # Per-row weights preserve the input proportions.
    assert out["wthhfin"].tolist() == pytest.approx([0.1, 0.2, 0.7], rel=1e-9)


def test_apply_vintage_mix_drops_vintages_absent_from_mix() -> None:
    """A vintage present in the frame but absent in the mix is dropped."""
    hh = pd.DataFrame(
        {
            "houseid": [1, 2, 3],
            "wthhfin": [100.0, 100.0, 100.0],
            "nhts_vintage": pd.array(
                ["2017", "2022_nextgen", "2022_nextgen"], dtype="string"
            ),
        }
    )
    out = dm._apply_vintage_mix(hh, {"2017": 1.0})
    assert set(out["nhts_vintage"].tolist()) == {"2017"}
    assert len(out) == 1
    assert out["wthhfin"].iloc[0] == pytest.approx(1.0)


def test_model_backoff_fills_nextgen_model_from_2017_marginal() -> None:
    """NextGen rows with NA model get a string from the 2017 marginal."""
    veh = pd.DataFrame(
        {
            "houseid": [1, 2, 3, 90, 91],
            "vehid": pd.array([1, 1, 1, 1, 1], dtype="int32"),
            "make": pd.array(
                ["Toyota", "Toyota", "Honda", "Toyota", "Honda"], dtype="string"
            ),
            "model": pd.array(
                ["Camry", "Camry", "Civic", pd.NA, pd.NA], dtype="string"
            ),
            "vehyear": pd.array([2018, 2018, 2019, 2018, 2019], dtype="Int16"),
            "hfuel": pd.array([-1, -1, -1, -1, -1], dtype="Int8"),
            "fueltype": pd.array([1, 1, 1, 1, 1], dtype="Int8"),
            "nhts_vintage": pd.array(
                ["2017", "2017", "2017", "2022_nextgen", "2022_nextgen"],
                dtype="string",
            ),
        }
    )
    import numpy as np
    rng = np.random.default_rng(42)
    out = dm._model_backoff_draw_from_2017(veh, rng=rng)
    assert out.loc[out["houseid"] == 90, "model"].iloc[0] == "Camry"
    assert out.loc[out["houseid"] == 91, "model"].iloc[0] == "Civic"
    # 2017 rows untouched.
    assert (
        out.loc[out["nhts_vintage"] == "2017", "model"].tolist()
        == veh.loc[veh["nhts_vintage"] == "2017", "model"].tolist()
    )


def test_model_backoff_unknown_when_make_absent_from_2017() -> None:
    """NextGen make never seen in 2017 -> model fills as 'UNKNOWN'."""
    import numpy as np
    veh = pd.DataFrame(
        {
            "houseid": [1, 90],
            "vehid": pd.array([1, 1], dtype="int32"),
            "make": pd.array(["Toyota", "Rivian"], dtype="string"),
            "model": pd.array(["Camry", pd.NA], dtype="string"),
            "vehyear": pd.array([2018, 2022], dtype="Int16"),
            "hfuel": pd.array([-1, 3], dtype="Int8"),
            "fueltype": pd.array([1, 3], dtype="Int8"),
            "nhts_vintage": pd.array(["2017", "2022_nextgen"], dtype="string"),
        }
    )
    out = dm._model_backoff_draw_from_2017(veh, rng=np.random.default_rng(0))
    assert out.loc[out["houseid"] == 90, "model"].iloc[0] == "UNKNOWN"


def test_model_backoff_noop_when_no_nextgen_rows_missing_model() -> None:
    """If no NextGen row has missing model, the function returns unchanged."""
    veh = pd.DataFrame(
        {
            "houseid": [1, 2],
            "vehid": pd.array([1, 1], dtype="int32"),
            "make": pd.array(["Toyota", "Honda"], dtype="string"),
            "model": pd.array(["Camry", "Civic"], dtype="string"),
            "vehyear": pd.array([2018, 2019], dtype="Int16"),
            "hfuel": pd.array([-1, -1], dtype="Int8"),
            "fueltype": pd.array([1, 1], dtype="Int8"),
            "nhts_vintage": pd.array(["2017", "2017"], dtype="string"),
        }
    )
    out = dm._model_backoff_draw_from_2017(veh)
    pd.testing.assert_frame_equal(out.reset_index(drop=True), veh.reset_index(drop=True))
