"""v2.1 cross-module sanity tests (plan §F.7).

Covers the cross-module surface: the new seed offsets are registered,
the validation_bounds_curator builders produce the expected new check
parquets, and the post-recalibration W5 workplace EVSE band still
passes.
"""

from __future__ import annotations

import sys
from pathlib import Path

_CODE_DIR = Path(__file__).resolve().parents[1]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from pev_synth import _seeds as seeds  # noqa: E402
from pev_synth import validation_bounds_curator as vbc  # noqa: E402
from pev_synth import vehicle_archetypes as va  # noqa: E402

# ---------------------------------------------------------------------------
# Sub-seed registry
# ---------------------------------------------------------------------------

def test_seeds_subseed_offsets_include_new_evse() -> None:
    assert "evse_brand_sample" in seeds.SUBSEED_OFFSETS
    assert "evse_connector_sample" in seeds.SUBSEED_OFFSETS
    assert seeds.SUBSEED_OFFSETS["evse_brand_sample"] == 10_000_000
    assert seeds.SUBSEED_OFFSETS["evse_connector_sample"] == 11_000_000
    # And the pairwise-gap invariant must still hold.
    seeds._assert_subseed_gaps_ok()


# ---------------------------------------------------------------------------
# validation_bounds_curator registration
# ---------------------------------------------------------------------------

def test_validation_bounds_curator_builders_register_new_checks() -> None:
    """All 8 new v2.1 check IDs are registered in ALL_CHECK_IDS and _BUILDERS."""
    expected = {
        "evse_kw_distribution_residential",
        "evse_kw_distribution_workplace",
        "evse_brand_distribution_residential",
        "evse_brand_distribution_workplace",
        "evse_connector_distribution_residential",
        "evse_connector_distribution_workplace",
        "evse_connector_my_band_distribution",
        "bundle_rule_take_rates",
    }
    assert expected.issubset(set(vbc.ALL_CHECK_IDS))
    assert expected.issubset(set(vbc._BUILDERS.keys()))


def test_validation_bounds_curator_evse_enrichment_check_ids_grouped() -> None:
    """``EVSE_ENRICHMENT_CHECK_IDS`` enumerates exactly the 8 new IDs."""
    expected = {
        "evse_kw_distribution_residential",
        "evse_kw_distribution_workplace",
        "evse_brand_distribution_residential",
        "evse_brand_distribution_workplace",
        "evse_connector_distribution_residential",
        "evse_connector_distribution_workplace",
        "evse_connector_my_band_distribution",
        "bundle_rule_take_rates",
    }
    assert set(vbc.EVSE_ENRICHMENT_CHECK_IDS) == expected


def test_validation_bounds_curator_writes_new_parquets(tmp_path: Path) -> None:
    """Run ``curate_bounds`` and verify every new parquet appears in the
    right ``<region>/<profile_type>/`` subdir.
    """
    manifest = vbc.curate_bounds(tmp_path)
    by_id = {c["check_id"]: c for c in manifest["checks"]}

    # Residential-only checks land under residential/ only.
    for cid in (
        "evse_kw_distribution_residential",
        "evse_brand_distribution_residential",
        "evse_connector_distribution_residential",
        "bundle_rule_take_rates",
    ):
        assert by_id[cid]["applicable_profile_types"] == ["residential"]
        assert (tmp_path / "bay_area" / "residential" / f"{cid}.parquet").exists()
        assert not (
            tmp_path / "bay_area" / "workplace" / f"{cid}.parquet"
        ).exists()

    # Workplace-only.
    for cid in (
        "evse_kw_distribution_workplace",
        "evse_brand_distribution_workplace",
        "evse_connector_distribution_workplace",
    ):
        assert by_id[cid]["applicable_profile_types"] == ["workplace"]
        assert (tmp_path / "bay_area" / "workplace" / f"{cid}.parquet").exists()
        assert not (
            tmp_path / "bay_area" / "residential" / f"{cid}.parquet"
        ).exists()

    # Dual-applicable.
    for cid in ("evse_connector_my_band_distribution",):
        assert set(by_id[cid]["applicable_profile_types"]) == {
            "residential", "workplace",
        }
        for pt in ("residential", "workplace"):
            assert (tmp_path / "bay_area" / pt / f"{cid}.parquet").exists()


# ---------------------------------------------------------------------------
# Backwards-compat regression on the W5 workplace floor
# ---------------------------------------------------------------------------

def test_w5_workplace_evse_kw_distribution_still_passes_with_recalibrated_prior() -> None:
    """The W5 floor (>= 60 % at 6.6-7.7 kW) is still met by the
    recalibrated workplace prior (0.38 + 0.34 = 0.72).
    """
    workplace_combined = (
        va.EVSE_MIX_WORKPLACE.get(6.6, 0.0)
        + va.EVSE_MIX_WORKPLACE.get(7.7, 0.0)
    )
    assert workplace_combined >= 0.60, (
        f"workplace prior 6.6+7.7 mass {workplace_combined} below 0.60"
    )


def test_obc_overrides_consistent_with_obc_mix_bev_values() -> None:
    """Every OBC override value must be a member of the OBC_MIX_BEV
    categorical universe (so downstream OBC-bucketed code can still
    hash on it without a special case for an off-grid value).
    """
    valid_obc = set(va.OBC_MIX_BEV.keys())
    for make, model, my_min, my_max, target_kw in va.OBC_MODEL_YEAR_OVERRIDES:
        assert float(target_kw) in valid_obc, (
            f"override ({make}, {model}, {my_min}-{my_max}) "
            f"target_kw={target_kw} not in OBC_MIX_BEV"
        )
