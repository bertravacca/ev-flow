"""Unit tests for M2 -- pev_synth.vehicle_archetypes (v2.0 refactor).

v2.0 acceptance criteria (per Wave 2 Dev D brief):
    1. sample_fleet(n=1000, region='bay_area') returns 1000 rows matching
       v1.1 schema + new (profile_type, region_name) columns.
    2. sample_fleet(region='dallas_fort_worth') shows pickup share >= 22%
       after the +10 pp overlay (target band 22-26% per plan §2.7).
    3. sample_fleet(region='bay_area', profile_type='workplace') has EVSE
       distribution with >= 60% at 6.6-7.7 kW (J1772 workplace dominance).
    4. Re-running with the same seed produces byte-identical fleet.parquet.
    5. Calling with an unknown region raises ValueError.

The legacy v1.1 tests (PHEV share, eta bounds, soc_min_kwh, heat-pump
make/year rule, panel-cap <= EVSE) are kept and re-parameterised against
the v2.0 signature.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make the ``code/`` directory importable when run from anywhere.
_CODE_DIR = Path(__file__).resolve().parents[1]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from pev_synth import vehicle_archetypes as va  # noqa: E402

# ---------------------------------------------------------------------------
# Skip predicate for tests that exercise ``sample_fleet``: it transitively
# calls ``load_sales_mix(region.sales_mix_source)``, which reads raw CSVs
# that are NOT shipped in the repo (see .gitignore). The lone exception is
# ``test_sample_fleet_unknown_region_raises``, which fails the region
# lookup before any CSV is touched, so it does not need this guard.
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SALES_MIX_DIR = _REPO_ROOT / "data" / "pev" / "raw" / "literature" / "sales_mix"

_needs_sales_mix = pytest.mark.skipif(
    not _SALES_MIX_DIR.exists() or not any(_SALES_MIX_DIR.glob("*.csv")),
    reason=(
        f"sales-mix CSVs not present under {_SALES_MIX_DIR}; "
        "run `python -m pev_synth.sales_mix_data build-sales-mix` first"
    ),
)


# ---------------------------------------------------------------------------
# v2.0 acceptance tests (new).
# ---------------------------------------------------------------------------

@_needs_sales_mix
def test_sample_fleet_with_region_bay_area() -> None:
    """Acceptance #1: 1000 rows; v2.0 schema; profile_type / region_name set."""
    df = va.sample_fleet(n=1000, seed=va.MASTER_SEED_DEFAULT, region="bay_area")
    assert len(df) == 1000
    expected_cols = [c for c, _, _ in va.SCHEMA]
    assert list(df.columns) == expected_cols
    assert df["ev_id"].is_unique
    assert df["ev_id"].min() == 0 and df["ev_id"].max() == 999
    # v3.0 nullability: ``phev_fuel_economy_mpg`` is NaN on BEV rows by
    # design. Restrict the no-null assertion to non-nullable columns.
    nonnull_cols = [c for c, _, nullable in va.SCHEMA if not nullable]
    assert df[nonnull_cols].notna().all().all(), (
        "no nulls allowed in non-nullable M2-owned columns"
    )
    assert (df["profile_type"].astype(str) == "residential").all()
    assert (df["region_name"].astype(str) == "bay_area").all()


@_needs_sales_mix
def test_sample_fleet_dfw_pickup_overlay_above_22pct() -> None:
    """Acceptance #2: DFW pickup share >= 22% after +10 pp overlay."""
    df = va.sample_fleet(n=5000, seed=va.MASTER_SEED_DEFAULT, region="dallas_fort_worth")
    pickup_share = (df["archetype"].astype(str) == "pickup").mean()
    assert 0.22 <= pickup_share <= 0.30, (
        f"DFW pickup share {pickup_share:.3f} out of [0.22, 0.30]"
    )


@_needs_sales_mix
def test_sample_fleet_workplace_evse_shifted_to_lower_kw() -> None:
    """Acceptance #3: workplace EVSE distribution >= 60% at 6.6 - 7.7 kW."""
    df = va.sample_fleet(
        n=2000, seed=va.MASTER_SEED_DEFAULT,
        region="bay_area", profile_type="workplace",
    )
    evse = df["EVSE_home_kw"].to_numpy()
    j1772_share = ((evse == np.float32(6.6)) | (evse == np.float32(7.7))).mean()
    assert j1772_share >= 0.60, (
        f"workplace J1772 share {j1772_share:.3f} below 0.60 floor"
    )
    # Workplace lots: no panel cap, garage_type uniform.
    assert (df["panel_cap_kw"].to_numpy() == df["EVSE_home_kw"].to_numpy()).all()
    assert (df["garage_type"].astype(str) == "workplace_lot").all()


@_needs_sales_mix
def test_sample_fleet_seed_reproducible(tmp_path: Path) -> None:
    """Acceptance #4: identical seed -> SHA-256-identical parquet."""
    df_a = va.sample_fleet(n=500, seed=va.MASTER_SEED_DEFAULT, region="bay_area")
    df_b = va.sample_fleet(n=500, seed=va.MASTER_SEED_DEFAULT, region="bay_area")
    pd.testing.assert_frame_equal(df_a, df_b)

    p_a = tmp_path / "a.parquet"
    p_b = tmp_path / "b.parquet"
    va.write_fleet(df_a, p_a)
    va.write_fleet(df_b, p_b)
    sha_a = hashlib.sha256(p_a.read_bytes()).hexdigest()
    sha_b = hashlib.sha256(p_b.read_bytes()).hexdigest()
    assert sha_a == sha_b, "SHA-256 mismatch between two seed-identical runs"


def test_sample_fleet_unknown_region_raises() -> None:
    """Acceptance #5: unknown region name -> ValueError."""
    with pytest.raises(ValueError, match="Unknown region"):
        va.sample_fleet(n=10, region="atlantis")


@_needs_sales_mix
def test_sample_fleet_workplace_seed_reproducible(tmp_path: Path) -> None:
    """v2.0 §4 acceptance #6 (workplace SHA-256 determinism)."""
    df_a = va.sample_fleet(
        n=200, seed=va.MASTER_SEED_DEFAULT,
        region="boston", profile_type="workplace",
    )
    df_b = va.sample_fleet(
        n=200, seed=va.MASTER_SEED_DEFAULT,
        region="boston", profile_type="workplace",
    )
    pd.testing.assert_frame_equal(df_a, df_b)
    pa = tmp_path / "a.parquet"
    pb = tmp_path / "b.parquet"
    va.write_fleet(df_a, pa)
    va.write_fleet(df_b, pb)
    assert hashlib.sha256(pa.read_bytes()).hexdigest() == hashlib.sha256(
        pb.read_bytes()
    ).hexdigest()


# ---------------------------------------------------------------------------
# Legacy v1.1 contract tests (re-parameterised for v2.0).
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fleet_bay_area_default() -> pd.DataFrame:
    if not _SALES_MIX_DIR.exists() or not any(_SALES_MIX_DIR.glob("*.csv")):
        pytest.skip(
            f"sales-mix CSVs not present under {_SALES_MIX_DIR}; "
            "run `python -m pev_synth.sales_mix_data build-sales-mix` first"
        )
    return va.sample_fleet(n=1000, seed=va.MASTER_SEED_DEFAULT, region="bay_area")


def test_phev_share_within_band(fleet_bay_area_default: pd.DataFrame) -> None:
    """Bay-Area / cvrp_ca mix gives PHEV ~0.13 +/- 0.04 at n=1000."""
    phev_share = (fleet_bay_area_default["powertrain"].astype(str) == "PHEV").mean()
    assert 0.09 <= phev_share <= 0.17, (
        f"PHEV share {phev_share:.3f} far from cvrp_ca prior of 0.13"
    )


def test_battery_distribution_matches_prior(
    fleet_bay_area_default: pd.DataFrame,
) -> None:
    """BEV B_kwh median lands in the Argonne EV-FACTS 2025 band [65, 85]."""
    bev_b = fleet_bay_area_default.loc[
        fleet_bay_area_default["powertrain"].astype(str) == "BEV", "B_kwh"
    ].to_numpy()
    median = float(np.median(bev_b))
    p10 = float(np.quantile(bev_b, 0.10))
    p90 = float(np.quantile(bev_b, 0.90))
    assert 60.0 <= median <= 85.0, f"BEV median B {median:.1f} out of [60, 85]"
    assert p10 >= 50.0, f"BEV p10 B {p10:.1f} unreasonably low"
    assert p90 <= 140.0, f"BEV p90 B {p90:.1f} unreasonably high"


def test_p_max_le_obc_and_evse(fleet_bay_area_default: pd.DataFrame) -> None:
    """panel_cap <= EVSE_home; OBC on the SAE J1772 grid."""
    df = fleet_bay_area_default
    assert (df["panel_cap_kw"] <= df["EVSE_home_kw"] + 1e-6).all()
    valid_obc = np.array([3.3, 6.6, 7.7, 11.5, 19.2], dtype=np.float64)
    obc_vals = df["OBC_kw"].to_numpy(dtype=np.float64)
    distances = np.min(np.abs(obc_vals[:, None] - valid_obc[None, :]), axis=1)
    assert (distances < 1e-3).all(), (
        f"OBC contains values not on SAE J1772 grid; max dev "
        f"{float(distances.max()):.4f} kW"
    )


def test_phev_archetype_is_discrete_mixture(
    fleet_bay_area_default: pd.DataFrame,
) -> None:
    """PHEV B_kwh is a discrete mixture (<= 12 unique values), not continuous."""
    phev_b = fleet_bay_area_default.loc[
        fleet_bay_area_default["powertrain"].astype(str) == "PHEV", "B_kwh"
    ]
    n_unique = phev_b.nunique()
    assert n_unique <= 12, f"PHEV B has {n_unique} unique values; expected <= 12"


def test_heat_pump_flag_correct_by_model_year(
    fleet_bay_area_default: pd.DataFrame,
) -> None:
    """has_heat_pump = (make in HEATPUMP_MAKES) & (model_year >= 2021)."""
    df = fleet_bay_area_default
    make_arr = df["make"].astype(str).to_numpy()
    my_arr = df["model_year"].to_numpy()
    expected = np.array(
        [
            (m in va.HEATPUMP_MAKES) and (y >= va.HEATPUMP_MIN_MODEL_YEAR)
            for m, y in zip(make_arr, my_arr, strict=True)
        ]
    )
    assert (df["has_heat_pump"].to_numpy() == expected).all()


def test_eta_bounds_respected(fleet_bay_area_default: pd.DataFrame) -> None:
    """eta_charge_nominal in [0.85, 0.94]; median near the prior mu = 0.90."""
    eta = fleet_bay_area_default["eta_charge_nominal"].to_numpy()
    assert (eta >= 0.85).all() and (eta <= 0.94).all()
    assert 0.88 <= float(np.median(eta)) <= 0.92


def test_soc_min_kwh_is_ten_percent_of_b(
    fleet_bay_area_default: pd.DataFrame,
) -> None:
    """soc_min_kwh == 0.10 * B_kwh (exact)."""
    df = fleet_bay_area_default
    np.testing.assert_allclose(
        df["soc_min_kwh"].to_numpy(),
        0.10 * df["B_kwh"].to_numpy(),
        rtol=0,
        atol=1e-6,
    )


@_needs_sales_mix
def test_write_fleet_round_trip(tmp_path: Path) -> None:
    """write_fleet -> read_parquet round trip preserves the schema."""
    df = va.sample_fleet(n=100, seed=va.MASTER_SEED_DEFAULT, region="bay_area")
    out = tmp_path / "fleet_small.parquet"
    va.write_fleet(df, out)
    rt = pd.read_parquet(out)
    assert list(rt.columns) == [c for c, _, _ in va.SCHEMA]
    assert len(rt) == 100
    assert rt["ev_id"].is_unique


# ---------------------------------------------------------------------------
# v2.0 region routing tests.
# ---------------------------------------------------------------------------

@_needs_sales_mix
def test_region_name_column_populated_per_region() -> None:
    """Every region renders its own name in the region_name column."""
    for region_name in ("bay_area", "la_basin", "new_york_metro", "dallas_fort_worth"):
        df = va.sample_fleet(n=50, region=region_name)
        assert (df["region_name"].astype(str) == region_name).all()


@_needs_sales_mix
def test_sales_mix_overlay_zero_means_no_op() -> None:
    """A region with empty sales_mix_overlay keeps the source mix shape."""
    df = va.sample_fleet(n=2000, region="bay_area", seed=va.MASTER_SEED_DEFAULT)
    # cvrp_ca has 5.3% pickup share -> Bay Area should be in [0.02, 0.10].
    pickup_share = (df["archetype"].astype(str) == "pickup").mean()
    assert 0.02 <= pickup_share <= 0.10, (
        f"Bay-Area pickup share {pickup_share:.3f} drifted from cvrp_ca prior"
    )


@_needs_sales_mix
def test_bev_share_override() -> None:
    """User-supplied bev_share overrides the region's inferred share."""
    df = va.sample_fleet(n=2000, region="bay_area", bev_share=0.50, seed=42)
    bev_share = (df["powertrain"].astype(str) == "BEV").mean()
    assert 0.46 <= bev_share <= 0.54, (
        f"explicit bev_share=0.50 not honoured; observed {bev_share:.3f}"
    )


# ---------------------------------------------------------------------------
# v3.0 — PHEV gasoline extension (§13.6 RESOLVED).
# ---------------------------------------------------------------------------


def test_schema_has_phev_fuel_economy_mpg_column() -> None:
    """v3.0 SCHEMA contains the nullable ``phev_fuel_economy_mpg`` column."""
    cols = {c: (dt, nullable) for c, dt, nullable in va.SCHEMA}
    assert "phev_fuel_economy_mpg" in cols, (
        "SCHEMA missing v3.0 phev_fuel_economy_mpg column"
    )
    dtype, nullable = cols["phev_fuel_economy_mpg"]
    assert dtype == "float32", (
        f"phev_fuel_economy_mpg dtype must be float32, got {dtype}"
    )
    assert nullable is True, (
        "phev_fuel_economy_mpg must be nullable (NaN on BEV rows)"
    )


@_needs_sales_mix
def test_phev_fuel_economy_mpg_finite_for_phev_nan_for_bev() -> None:
    """v3.0 §13.6 — PHEV rows carry finite CS MPG; BEV rows carry NaN."""
    df = va.sample_fleet(
        n=2000, seed=va.MASTER_SEED_DEFAULT, region="bay_area",
    )
    assert "phev_fuel_economy_mpg" in df.columns
    phev_mask = df["powertrain"].astype(str) == "PHEV"
    bev_mask = df["powertrain"].astype(str) == "BEV"
    assert phev_mask.sum() > 0, "fixture must contain at least one PHEV"
    assert bev_mask.sum() > 0, "fixture must contain at least one BEV"
    # PHEV — finite, in a sane EPA combined-MPG range.
    phev_mpg = df.loc[phev_mask, "phev_fuel_economy_mpg"]
    assert phev_mpg.notna().all(), "PHEV rows must carry finite CS MPG"
    assert (phev_mpg > 10.0).all() and (phev_mpg < 100.0).all(), (
        f"PHEV CS MPG out of plausible EPA band: "
        f"min={float(phev_mpg.min())}, max={float(phev_mpg.max())}"
    )
    # BEV — NaN by design (the field is powertrain-specific).
    bev_mpg = df.loc[bev_mask, "phev_fuel_economy_mpg"]
    assert bev_mpg.isna().all(), "BEV rows must carry NaN CS MPG"


@_needs_sales_mix
def test_phev_fuel_economy_mpg_survives_parquet_roundtrip(tmp_path: Path) -> None:
    """phev_fuel_economy_mpg writes + reads back via parquet without dtype drift."""
    df = va.sample_fleet(
        n=300, seed=va.MASTER_SEED_DEFAULT, region="bay_area",
    )
    out = tmp_path / "fleet_v3.parquet"
    va.write_fleet(df, out)
    rt = pd.read_parquet(out)
    assert "phev_fuel_economy_mpg" in rt.columns
    assert rt["phev_fuel_economy_mpg"].dtype == np.float32
    # NaN preserved on BEV rows; finite preserved on PHEV rows.
    phev_mask = rt["powertrain"].astype(str) == "PHEV"
    bev_mask = rt["powertrain"].astype(str) == "BEV"
    if phev_mask.any():
        assert rt.loc[phev_mask, "phev_fuel_economy_mpg"].notna().all()
    if bev_mask.any():
        assert rt.loc[bev_mask, "phev_fuel_economy_mpg"].isna().all()
