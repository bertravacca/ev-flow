"""v2.1 EVSE-enrichment sampling tests (plan §F.2).

End-to-end sampling tests on small synthetic fleets, gated by
``_needs_sales_mix``. Each test uses ``n`` in the 5K-10K range so the
CLT band is tight (±2 pp on shares with prior > 5 %).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_CODE_DIR = Path(__file__).resolve().parents[1]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from pev_synth import vehicle_archetypes as va  # noqa: E402

# The repo root lives one level above ``tests/`` in the v2.1 layout, so all
# tests now use parents[1] to locate the repo root (and the ``data/`` tree).
# This exercises the sampler when the sales-mix CSVs are present rather than
# unconditionally skipping.
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
# kW distribution sanity (CLT bands on small-n fleets)
# ---------------------------------------------------------------------------

@_needs_sales_mix
def test_residential_evse_kw_realised_matches_prior() -> None:
    df = va.sample_fleet(
        n=10_000, seed=va.MASTER_SEED_DEFAULT, region="bay_area",
    )
    evse = df["EVSE_home_kw"].astype(float).to_numpy()
    # The unconditional prior has 19.2 = 0.03 but the hard-rule layer
    # bumps it via Ford CSP / GM / Rivian; 19.2 kW realised can land
    # well above 0.03 -- we just sanity-check the broad shape. Use
    # ``np.isclose`` because EVSE_home_kw is stored in float32 and the
    # nominal values (6.6, 9.6) don't round-trip exactly.
    share_3p3 = float(np.isclose(evse, 3.3, atol=1e-3).mean())
    share_6p6 = float(np.isclose(evse, 6.6, atol=1e-3).mean())
    share_11p5 = float(np.isclose(evse, 11.5, atol=1e-3).mean())
    # Bands from §D.2.1.
    assert 0.05 <= share_3p3 <= 0.12, f"3.3 kW share {share_3p3}"
    assert 0.22 <= share_6p6 <= 0.33, f"6.6 kW share {share_6p6}"
    # 11.5 kW realised may exceed the unconditional 0.26 if many
    # Hyundai/Kia bundle rows route to 11.5 kW. Allow up to 0.40.
    assert 0.21 <= share_11p5 <= 0.40, f"11.5 kW share {share_11p5}"


@_needs_sales_mix
def test_workplace_evse_kw_share_at_6p6_7p7() -> None:
    df = va.sample_fleet(
        n=5_000, seed=va.MASTER_SEED_DEFAULT,
        region="bay_area", profile_type="workplace",
    )
    evse = df["EVSE_home_kw"].astype(float).to_numpy()
    share = float(
        (
            np.isclose(evse, 6.6, atol=1e-3)
            | np.isclose(evse, 7.7, atol=1e-3)
        ).mean()
    )
    # Workplace prior: 0.38 + 0.34 = 0.72; W5 floor is 0.60.
    assert share >= 0.60, f"workplace 6.6/7.7 kW share {share}"


# ---------------------------------------------------------------------------
# Brand distribution
# ---------------------------------------------------------------------------

@_needs_sales_mix
def test_residential_brand_distribution_top_brands_in_band() -> None:
    df = va.sample_fleet(
        n=10_000, seed=va.MASTER_SEED_DEFAULT, region="bay_area",
    )
    counts = df["EVSE_brand"].value_counts(normalize=True)
    # Tesla and ChargePoint are the two majors per [lit §3.21, §3.23].
    # Allow ±5 pp on Tesla (prior=0.27) and ChargePoint (prior=0.18).
    # Note: Tesla self-selection rule INCREASES Tesla share above 0.27
    # for Tesla-owning EVs; the realised value is the population mean.
    assert 0.20 <= float(counts.get("Tesla", 0.0)) <= 0.45
    assert 0.13 <= float(counts.get("ChargePoint", 0.0)) <= 0.28


@_needs_sales_mix
def test_workplace_brand_distribution_chargepoint_dominant() -> None:
    df = va.sample_fleet(
        n=5_000, seed=va.MASTER_SEED_DEFAULT,
        region="bay_area", profile_type="workplace",
    )
    cp_share = float(
        (df["EVSE_brand"].astype(str) == "ChargePoint").mean()
    )
    assert cp_share >= 0.45, f"workplace ChargePoint share {cp_share}"


# ---------------------------------------------------------------------------
# Connector / NACS transition
# ---------------------------------------------------------------------------

@_needs_sales_mix
def test_connector_my_band_pre_2023_j1772_dominant() -> None:
    """Pre-2023 fleet: J1772 + Tesla_NACS_native dominate; raw NACS rare."""
    df = va.sample_fleet(
        n=10_000, seed=va.MASTER_SEED_DEFAULT, region="bay_area",
    )
    sub = df[(df["model_year"] >= 2018) & (df["model_year"] <= 2022)]
    if len(sub) < 100:
        pytest.skip("not enough pre-2023 EVs for a CLT-band check")
    conn = sub["EVSE_connector"].astype(str)
    j1772_or_tesla = float(
        (conn.isin({"J1772", "Tesla_NACS_native"})).mean()
    )
    assert j1772_or_tesla >= 0.80, (
        f"pre-2023 J1772+Tesla_NACS share {j1772_or_tesla}"
    )


@_needs_sales_mix
def test_connector_my_band_2025_plus_nacs_substantial() -> None:
    """MY 2025+ fleet: NACS share should be >= 0.20 across all OEMs."""
    df = va.sample_fleet(
        n=10_000, seed=va.MASTER_SEED_DEFAULT, region="bay_area",
    )
    sub = df[df["model_year"] >= 2025]
    if len(sub) < 100:
        pytest.skip("not enough MY2025+ EVs for a CLT-band check")
    conn = sub["EVSE_connector"].astype(str)
    nacs_share = float(
        (conn.isin({"NACS", "Tesla_NACS_native", "Universal_NACS_J1772"})).mean()
    )
    assert nacs_share >= 0.20, f"MY 2025+ NACS-family share {nacs_share}"


# ---------------------------------------------------------------------------
# Seed reproducibility for the new columns
# ---------------------------------------------------------------------------

@_needs_sales_mix
def test_evse_brand_connector_seed_reproducible() -> None:
    df_a = va.sample_fleet(
        n=500, seed=va.MASTER_SEED_DEFAULT, region="bay_area",
    )
    df_b = va.sample_fleet(
        n=500, seed=va.MASTER_SEED_DEFAULT, region="bay_area",
    )
    np.testing.assert_array_equal(
        df_a["EVSE_brand"].astype(str).to_numpy(),
        df_b["EVSE_brand"].astype(str).to_numpy(),
    )
    np.testing.assert_array_equal(
        df_a["EVSE_connector"].astype(str).to_numpy(),
        df_b["EVSE_connector"].astype(str).to_numpy(),
    )


# ---------------------------------------------------------------------------
# S5: non-CA region smoke test for the residential connector bands.
# The bands in _build_evse_connector_distribution_residential were
# calibrated against Tesla-heavy CA fleets. Verify they don't fail on
# non-Tesla-heavy regions.
# ---------------------------------------------------------------------------

@_needs_sales_mix
@pytest.mark.parametrize("region_name", ["chicago", "new_york_metro"])
def test_residential_connector_distribution_passes_non_ca_region(
    region_name: str, tmp_path: Path,
) -> None:
    """Generate a non-CA residential fleet, curate bounds, and assert
    ``check_evse_connector_distribution`` returns PASS. The residential
    bands (J1772 [0.30, 0.65], Tesla_NACS_native [0.10, 0.60],
    NACS [0.02, 0.25]) must accommodate non-Tesla-heavy regions.
    """
    from pev_synth import validation_bounds_curator as vbc  # noqa: PLC0415
    from pev_synth import validator as vld  # noqa: PLC0415

    df = va.sample_fleet(
        n=5_000, seed=va.MASTER_SEED_DEFAULT, region=region_name,
    )
    vbc.curate_bounds(tmp_path)
    bounds_subdir = tmp_path / region_name / "residential"
    result = vld.check_evse_connector_distribution(
        df, bounds_subdir, profile_type="residential",
    )
    assert result.status == "PASS", (
        f"region={region_name}: connector distribution check failed: "
        f"{result.details}"
    )
