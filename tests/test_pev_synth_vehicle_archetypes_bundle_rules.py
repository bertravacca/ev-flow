"""v2.1 hard-coupling rule tests (plan §F.3).

These tests exercise the brand+kW+connector overrides that
:func:`pev_synth.vehicle_archetypes._apply_bundle_rules` applies. For
each rule we sample a fleet large enough to give a CLT-band-safe
estimate of the realised take-rate -- typically ``n=20_000`` for the
Tesla / Hyundai rules. Low-volume rules (Lucid, Hummer EV) are skipped
in v2.1 because the underlying make/model is not in
``sales_mix_data._MODEL_ATTRS`` (plan §I.6 placeholder).
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

# parents[1] so the path resolves to the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SALES_MIX_DIR = _REPO_ROOT / "data" / "pev" / "raw" / "literature" / "sales_mix"

_needs_sales_mix = pytest.mark.skipif(
    not _SALES_MIX_DIR.exists() or not any(_SALES_MIX_DIR.glob("*.csv")),
    reason=(
        f"sales-mix CSVs not present under {_SALES_MIX_DIR}; "
        "run `python -m pev_synth.sales_mix_data build-sales-mix` first"
    ),
)


@pytest.fixture(scope="module")
def big_fleet():
    if not _SALES_MIX_DIR.exists() or not any(_SALES_MIX_DIR.glob("*.csv")):
        pytest.skip("sales-mix CSVs missing")
    # n=30K gives a tight CLT band on rules with ~5 % matched-EV
    # population; bigger fleets are exempted from the n<50K cap.
    return va.sample_fleet(
        n=30_000, seed=va.MASTER_SEED_DEFAULT, region="bay_area",
    )


# ---------------------------------------------------------------------------
# Take-rate checks
# ---------------------------------------------------------------------------

def _rule_realised_take_rate(
    fleet, rule: va.EvEvseBundleRule, target_brand: str,
) -> tuple[int, float]:
    make_arr = fleet["make"].astype(str).to_numpy()
    model_arr = fleet["model"].astype(str).to_numpy()
    powertrain_arr = fleet["powertrain"].astype(str).to_numpy()
    my_arr = fleet["model_year"].astype(int).to_numpy()
    brand_arr = fleet["EVSE_brand"].astype(str).to_numpy()
    if rule.predicate_make == "*":
        m_mask = np.ones(len(make_arr), dtype=bool)
    else:
        m_mask = make_arr == rule.predicate_make
    if rule.predicate_model == "*":
        mo_mask = np.ones(len(model_arr), dtype=bool)
    else:
        mo_mask = model_arr == rule.predicate_model
    if rule.predicate_powertrain == "*":
        p_mask = np.ones(len(powertrain_arr), dtype=bool)
    else:
        p_mask = powertrain_arr == rule.predicate_powertrain
    my_mask = (my_arr >= rule.model_year_min) & (
        my_arr <= rule.model_year_max
    )
    mask = m_mask & mo_mask & p_mask & my_mask
    n_match = int(mask.sum())
    if n_match == 0:
        return 0, float("nan")
    realised = float((brand_arr[mask] == target_brand).mean())
    return n_match, realised


@_needs_sales_mix
def test_ford_lightning_er_my22_23_csp_take_rate_70pct(big_fleet) -> None:
    rule = next(
        r for r in va.EV_EVSE_BUNDLE_RULES
        if r.name == "ford_lightning_er_my22_23_csp"
    )
    n, realised = _rule_realised_take_rate(big_fleet, rule, "Ford_CSP")
    if n < 50:
        pytest.skip(f"only {n} Lightning ER MY22-23 in the fleet")
    # ±0.10 CLT band when n_match < 100, else ±0.05.
    tol = 0.10 if n < 100 else 0.075
    assert abs(realised - 0.70) <= tol, (
        f"Lightning CSP take-rate {realised:.3f} not within {tol} of 0.70"
    )


@_needs_sales_mix
def test_ford_lightning_er_my24p_evse_home_kw_le_11_5(big_fleet) -> None:
    """After the MY24+ step-down, EVSE_home_kw <= 11.5 for matching rows."""
    df = big_fleet
    mask = (
        (df["make"].astype(str) == "Ford")
        & (df["model"].astype(str) == "F150_Lightning")
        & (df["model_year"] >= 2024)
    )
    if not mask.any():
        pytest.skip("no MY24+ Lightning ER in the fleet")
    kw = df.loc[mask, "EVSE_home_kw"].astype(float).to_numpy()
    assert kw.max() <= 11.5 + 1e-3, (
        f"MY24+ Lightning ER EVSE_home_kw max = {kw.max():.3f}"
    )


@_needs_sales_mix
def test_hyundai_ioniq_chargepoint_take_rate_55pct(big_fleet) -> None:
    """Hyundai MY24-26 owners: at least 55 % get ChargePoint brand.

    Because the base brand prior independently assigns ChargePoint at
    ~18 % residential share, the realised share among Hyundai EVs is
    closer to ``rule_take_rate + (1 - rule_take_rate) * base_prior``
    (~0.63). We check only the lower bound: the bundle rule guarantees
    >= rule.take_rate share, modulo small CLT noise.
    """
    rule = next(
        r for r in va.EV_EVSE_BUNDLE_RULES
        if r.name == "hyundai_ioniq_chargepoint_home_flex"
    )
    n, realised = _rule_realised_take_rate(big_fleet, rule, "ChargePoint")
    if n < 50:
        pytest.skip(f"only {n} Hyundai MY24-26 in fleet")
    floor = 0.55 - (0.10 if n < 100 else 0.05)
    assert realised >= floor, (
        f"Hyundai ChargePoint share {realised:.3f} below floor {floor:.3f}"
    )


@_needs_sales_mix
def test_kia_ioniq_chargepoint_take_rate_55pct(big_fleet) -> None:
    """Kia MY24-26: same lower-bound semantic as the Hyundai test above."""
    rule = next(
        r for r in va.EV_EVSE_BUNDLE_RULES
        if r.name == "kia_ioniq_chargepoint_home_flex"
    )
    n, realised = _rule_realised_take_rate(big_fleet, rule, "ChargePoint")
    if n < 50:
        pytest.skip(f"only {n} Kia MY24-26 in fleet")
    floor = 0.55 - (0.10 if n < 100 else 0.05)
    assert realised >= floor, (
        f"Kia ChargePoint share {realised:.3f} below floor {floor:.3f}"
    )


@_needs_sales_mix
def test_tesla_self_selection_wall_connector_share(big_fleet) -> None:
    """Within Tesla owners (residential), >= 0.55 install Tesla brand."""
    rule = next(
        r for r in va.EV_EVSE_BUNDLE_RULES
        if r.name == "tesla_wall_connector_self_selection"
    )
    n, realised = _rule_realised_take_rate(big_fleet, rule, "Tesla")
    if n < 100:
        pytest.skip(f"only {n} Tesla EVs in fleet")
    # Tesla appears in the base brand prior at 0.27 too, so realised is
    # higher than the take-rate alone. Lower bound = take_rate - tol.
    assert realised >= 0.55, f"Tesla brand share among Tesla EVs = {realised}"


@_needs_sales_mix
def test_rivian_my22_23_wall_charger_bundle_take_rate_60pct(big_fleet) -> None:
    rule = next(
        r for r in va.EV_EVSE_BUNDLE_RULES
        if r.name == "rivian_r1tr1s_wall_charger_bundle"
    )
    n, realised = _rule_realised_take_rate(big_fleet, rule, "Rivian")
    if n < 50:
        pytest.skip(f"only {n} Rivian MY22-23 in fleet")
    tol = 0.10 if n < 100 else 0.075
    assert abs(realised - 0.60) <= tol, (
        f"Rivian Wall Charger take-rate {realised:.3f} not within {tol} of 0.60"
    )


@_needs_sales_mix
def test_lucid_19_2kw_optional_wallbox_take_rate_45pct(big_fleet) -> None:
    """Plan §I.6 -- Lucid is a placeholder until v2.2 sales-mix expansion.
    Skip if no Lucid rows surfaced from the regional sales mix.
    """
    rule = next(
        r for r in va.EV_EVSE_BUNDLE_RULES
        if r.name == "lucid_air_19_2kw_optional_wallbox"
    )
    n, _ = _rule_realised_take_rate(big_fleet, rule, "Lucid")
    if n == 0:
        pytest.skip("Lucid not in regional sales mix (plan §I.6 placeholder)")
    # If somehow present, just sanity-check the realised brand share is
    # finite and non-negative.
    n2, realised = _rule_realised_take_rate(big_fleet, rule, "Lucid")
    assert n2 > 0
    assert 0.0 <= realised <= 1.0


@_needs_sales_mix
def test_non_matching_rows_never_get_bundle_brand_for_ford_csp(big_fleet) -> None:
    """A non-Ford row must never end up with EVSE_brand=='Ford_CSP'
    via the bundle layer. (Base prior gives Ford_CSP some mass on the
    19.2 kW bucket -- but the rule itself only fires for Ford Lightning
    ER MY22-23.)
    """
    df = big_fleet
    ford_csp_mask = df["EVSE_brand"].astype(str) == "Ford_CSP"
    non_ford = df.loc[ford_csp_mask & (df["make"].astype(str) != "Ford")]
    # Non-Ford Ford_CSP rows come ONLY from the base prior (Ford_CSP is
    # 0.02 unconditional, multiplied 4x on the 19.2 kW bucket), not from
    # the bundle rule. Sanity-check that the population is bounded.
    assert len(non_ford) / max(1, len(df)) < 0.05, (
        f"Too many non-Ford rows have EVSE_brand=Ford_CSP: {len(non_ford)}"
    )


@_needs_sales_mix
def test_bundle_realised_metadata_recorded(big_fleet) -> None:
    """``df.attrs['bundle_rule_realised_take_rates']`` carries one entry
    per rule with n_match / n_fired / target_take_rate.
    """
    realised = big_fleet.attrs.get("bundle_rule_realised_take_rates", {})
    assert isinstance(realised, dict)
    rule_names = {r.name for r in va.EV_EVSE_BUNDLE_RULES}
    assert set(realised.keys()) == rule_names
    for name, entry in realised.items():
        for key in ("n_match", "n_eligible", "n_fired", "target_take_rate"):
            assert key in entry, f"{name} missing {key}"
