"""v2.1 EVSE-enrichment unit tests on the prior tables (plan §F.1).

These tests do NOT call ``sample_fleet`` -- they only inspect the
in-memory constant tables in :mod:`pev_synth.vehicle_archetypes`. As a
result they run unconditionally (no sales-mix CSVs required).
"""

from __future__ import annotations

import sys
from pathlib import Path

_CODE_DIR = Path(__file__).resolve().parents[1]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from pev_synth import _seeds as seeds  # noqa: E402
from pev_synth import vehicle_archetypes as va  # noqa: E402

# ---------------------------------------------------------------------------
# Sum-to-one invariants
# ---------------------------------------------------------------------------

def test_evse_brand_mix_residential_sums_to_one() -> None:
    s = sum(va.EVSE_BRAND_MIX_RESIDENTIAL.values())
    assert abs(s - 1.0) < 1e-9, f"residential brand sum {s!r}"


def test_evse_brand_mix_workplace_sums_to_one() -> None:
    s = sum(va.EVSE_BRAND_MIX_WORKPLACE.values())
    assert abs(s - 1.0) < 1e-9, f"workplace brand sum {s!r}"


def test_evse_mix_residential_recalibrated_sums_to_one() -> None:
    """v2.1 — add the 9.6 kW bucket; total must still sum to 1.0."""
    s = sum(va.EVSE_MIX_RESIDENTIAL.values())
    assert abs(s - 1.0) < 1e-9, f"residential kw sum {s!r}"
    assert 9.6 in va.EVSE_MIX_RESIDENTIAL, "9.6 kW bucket missing"


def test_evse_mix_workplace_recalibrated_sums_to_one() -> None:
    s = sum(va.EVSE_MIX_WORKPLACE.values())
    assert abs(s - 1.0) < 1e-9, f"workplace kw sum {s!r}"


def test_nema_outlet_mix_sums_to_one() -> None:
    s = sum(va.NEMA_OUTLET_MIX.values())
    assert abs(s - 1.0) < 1e-9, f"NEMA outlet sum {s!r}"


def test_workplace_site_connector_mix_sums_to_one() -> None:
    s = sum(va.EVSE_CONNECTOR_MIX_WORKPLACE.values())
    assert abs(s - 1.0) < 1e-9, f"workplace site connector sum {s!r}"


def test_evse_connector_mix_by_make_my_bands_sum_to_one() -> None:
    """Every (make, MY-band) prior must sum to 1.0."""
    for make, bands in va.EVSE_CONNECTOR_MIX_BY_MAKE_MY.items():
        for lo, hi, prior in bands:
            s = sum(prior.values())
            assert abs(s - 1.0) < 1e-9, (
                f"{make} {lo}-{hi} connector mix sum {s!r}"
            )


# ---------------------------------------------------------------------------
# Category-universe invariants
# ---------------------------------------------------------------------------

def test_evse_brand_categories_superset_of_priors() -> None:
    for prior_name, prior in (
        ("residential", va.EVSE_BRAND_MIX_RESIDENTIAL),
        ("workplace", va.EVSE_BRAND_MIX_WORKPLACE),
    ):
        for k in prior:
            assert k in va.EVSE_BRAND_CATEGORIES, (
                f"{prior_name} prior key {k!r} not in EVSE_BRAND_CATEGORIES"
            )


def test_evse_connector_categories_superset_of_mix_by_make_my() -> None:
    for make, bands in va.EVSE_CONNECTOR_MIX_BY_MAKE_MY.items():
        for lo, hi, prior in bands:
            for k in prior:
                assert k in va.EVSE_CONNECTOR_CATEGORIES, (
                    f"{make} {lo}-{hi} connector key {k!r} not registered"
                )


def test_evse_connector_categories_superset_of_workplace_mix() -> None:
    for k in va.EVSE_CONNECTOR_MIX_WORKPLACE:
        assert k in va.EVSE_CONNECTOR_CATEGORIES, (
            f"workplace site connector key {k!r} not registered"
        )


def test_evse_connector_categories_superset_of_nema_outlet_mix() -> None:
    for k in va.NEMA_OUTLET_MIX:
        assert k in va.EVSE_CONNECTOR_CATEGORIES, (
            f"NEMA outlet key {k!r} not registered as a connector category"
        )


def test_obc_bucket_brand_multipliers_keys_subset_of_evse_brand_categories() -> None:
    for bucket, mults in va.OBC_BUCKET_BRAND_MULTIPLIERS.items():
        for k in mults:
            assert k in va.EVSE_BRAND_CATEGORIES, (
                f"OBC bucket {bucket!r} multiplier {k!r} not registered"
            )


# ---------------------------------------------------------------------------
# Bundle rules and OBC overrides
# ---------------------------------------------------------------------------

def test_ev_evse_bundle_rules_first_match_precedence_documented() -> None:
    """Every rule has a name. Tuple order encodes first-match precedence
    (plan §B.4.1) -- this test pins the documented order is preserved.
    """
    names = [r.name for r in va.EV_EVSE_BUNDLE_RULES]
    assert len(names) == len(set(names)), "duplicate rule name"
    # The CSP rule must precede the step-down rule so Lightning ER MY22-23
    # gets the brand+19.2kW override before the MY24+ clipper sees them.
    csp_idx = names.index("ford_lightning_er_my22_23_csp")
    step_idx = names.index("ford_lightning_er_my24p_step_down")
    assert csp_idx < step_idx, (
        "ford_lightning_er_my22_23_csp must precede ford_lightning_er_my24p"
    )


def test_obc_model_year_overrides_dispatches_to_known_make_model_or_wildcard() -> None:
    """Every override entry is either a known (make, model), a wildcard
    (``"*"`` model), or a documented placeholder pending the v2.2
    sales-mix expansion (Lucid, GMC Hummer_EV).
    """
    # Late import to avoid a top-level cycle.
    from pev_synth.sales_mix_data import _MODEL_ATTRS

    known_keys = set(_MODEL_ATTRS.keys())
    placeholder_makes = {"Lucid", "GMC"}
    for make, model, _, _, _ in va.OBC_MODEL_YEAR_OVERRIDES:
        if (make, model) in known_keys:
            continue
        if model == "*":
            continue
        if make in placeholder_makes:
            continue
        raise AssertionError(
            f"OBC override ({make!r}, {model!r}) unknown and not placeholder"
        )


def test_lightning_er_take_rate_is_user_locked_at_0_70() -> None:
    """Plan §I.3: user-locked CSP take-rate is 0.70 (not 0.85, not 1.0)."""
    rule = next(
        r for r in va.EV_EVSE_BUNDLE_RULES
        if r.name == "ford_lightning_er_my22_23_csp"
    )
    assert rule.take_rate == 0.70


def test_hummer_ev_and_lucid_rules_remain_placeholders() -> None:
    """Plan §I.6 option A: keep placeholder rules in EV_EVSE_BUNDLE_RULES
    but do NOT add Lucid / Hummer_EV to _MODEL_ATTRS in v2.1.
    """
    from pev_synth.sales_mix_data import _MODEL_ATTRS

    assert ("Lucid", "*") not in _MODEL_ATTRS  # only specific (make, model) keys
    assert ("GMC", "Hummer_EV") not in _MODEL_ATTRS
    # The rules themselves must still be present (documented intent).
    rule_names = {r.name for r in va.EV_EVSE_BUNDLE_RULES}
    assert "lucid_air_19_2kw_optional_wallbox" in rule_names
    assert "gmc_hummer_ev_powershift_charger" in rule_names


# ---------------------------------------------------------------------------
# Sub-seed offsets
# ---------------------------------------------------------------------------

def test_subseed_offsets_for_evse_disjoint_from_existing() -> None:
    """Pairwise gap >= N_EV_MAX (RFC-001 invariant) covers the new
    EVSE brand + connector offsets.
    """
    seeds._assert_subseed_gaps_ok()
    assert seeds.SUBSEED_OFFSETS["evse_brand_sample"] == 10_000_000
    assert seeds.SUBSEED_OFFSETS["evse_connector_sample"] == 11_000_000


def test_vehicle_archetypes_re_exports_subseed_offsets() -> None:
    """The M2 module also exposes the new offsets as constants for
    callers that key per-EV streams off the M2 sub-seed namespace.
    """
    assert va.EVSE_BRAND_SAMPLE_SUBSEED_OFFSET == 10_000_000
    assert va.EVSE_CONNECTOR_SAMPLE_SUBSEED_OFFSET == 11_000_000


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def test_schema_includes_new_columns() -> None:
    """``SCHEMA`` carries both new columns as non-nullable categorical."""
    schema_map = {name: (dtype, nullable) for name, dtype, nullable in va.SCHEMA}
    assert schema_map.get("EVSE_brand") == ("category", False)
    assert schema_map.get("EVSE_connector") == ("category", False)


def test_schema_columns_ordered_after_panel_cap() -> None:
    """``EVSE_brand`` / ``EVSE_connector`` cluster next to panel_cap_kw."""
    cols = [c for c, _, _ in va.SCHEMA]
    i_panel = cols.index("panel_cap_kw")
    i_brand = cols.index("EVSE_brand")
    i_conn = cols.index("EVSE_connector")
    assert i_brand == i_panel + 1
    assert i_conn == i_panel + 2
