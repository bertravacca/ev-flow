"""Max-coverage unit tests for M2 -- pev_synth.vehicle_archetypes.

This file targets the branches that the existing
``tests/test_pev_synth_vehicle_archetypes*.py`` suite leaves uncovered:
helper error paths (``_resolve_region``, ``_validate_profile_type``,
``_powertrain_subset``, ``_infer_bev_share``), the sales-mix overlay
arithmetic edge cases, the import-time EVSE brand-prior invariant, the
stratified heat-pump resampler (RFC-032), the OBC/brand/connector helper
branches, the persistence helpers (``_cast_to_schema``, ``write_fleet``,
``write_meta``, ``_hash_sales_mix``, ``_resolve_git_commit``), and the
``main`` CLI.

All tests are data-independent: every fleet-shaped frame is fabricated in
memory with fixed seeds, and the few tests that exercise the end-to-end
``sample_fleet`` / CLI / ``write_meta`` path are gated on the literature
sales-mix CSVs via ``_needs_sales_mix`` (the same guard the rest of the
suite uses). No heavy NHTS / fleet-cache tree is touched.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_CODE_DIR = Path(__file__).resolve().parents[1]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from pev_synth import vehicle_archetypes as va  # noqa: E402
from pev_synth.regions import REGIONS, Region  # noqa: E402

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
# Synthetic in-memory builders (no CSV / no NHTS dependency).
# ---------------------------------------------------------------------------

def _toy_mix() -> pd.DataFrame:
    """A minimal sales-mix frame with one BEV and one PHEV row."""
    return pd.DataFrame(
        {
            "archetype": ["sedan", "crossover", "suv", "pickup"],
            "make": ["Tesla", "Toyota", "Rivian", "Ford"],
            # Toyota Prius_Prime is in PHEV_MODEL_REGISTRY; rest are BEV.
            "model": ["Model_3_LR", "Prius_Prime", "R1S", "F150_Lightning"],
            "model_year_min": [2019, 2021, 2022, 2022],
            "model_year_max": [2025, 2025, 2025, 2025],
            "share": [0.4, 0.1, 0.2, 0.3],
            "battery_kwh_nominal": [75.0, 13.0, 135.0, 131.0],
            "obc_max_kw": [11.5, 6.6, 11.5, 19.2],
        }
    )


def _bay_area_clone(**overrides: object) -> Region:
    """Return a copy of the bay_area Region with field overrides applied."""
    import dataclasses

    return dataclasses.replace(REGIONS["bay_area"], **overrides)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _resolve_region / _validate_profile_type (lines 718, 721-724, 729-732).
# ---------------------------------------------------------------------------

def test_resolve_region_passthrough_instance() -> None:
    r = REGIONS["bay_area"]
    assert va._resolve_region(r) is r


def test_resolve_region_from_name_string() -> None:
    assert va._resolve_region("bay_area").name == "bay_area"


def test_resolve_region_bad_type_raises() -> None:
    with pytest.raises(TypeError, match="region must be a Region instance"):
        va._resolve_region(123)  # type: ignore[arg-type]


def test_validate_profile_type_ok() -> None:
    # No raise for a valid profile type.
    va._validate_profile_type("residential")
    va._validate_profile_type("workplace")


def test_validate_profile_type_bad_raises() -> None:
    with pytest.raises(ValueError, match="not supported"):
        va._validate_profile_type("garage_band")


# ---------------------------------------------------------------------------
# _assert_evse_brand_priors_ok import-time invariant (lines 283-297).
# ---------------------------------------------------------------------------

def test_brand_prior_sum_violation_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    bad = dict(va.EVSE_BRAND_MIX_RESIDENTIAL)
    bad["Other"] = bad["Other"] + 0.5  # break the sum-to-1 invariant
    monkeypatch.setattr(va, "EVSE_BRAND_MIX_RESIDENTIAL", bad)
    with pytest.raises(AssertionError, match="sums to"):
        va._assert_evse_brand_priors_ok()


def test_brand_prior_unknown_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    bad = {"NotARealBrand": 1.0}
    monkeypatch.setattr(va, "EVSE_BRAND_MIX_WORKPLACE", bad)
    with pytest.raises(AssertionError, match="not in EVSE_BRAND_CATEGORIES"):
        va._assert_evse_brand_priors_ok()


def test_obc_multiplier_unknown_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    bad = {19.2: {"Phantom_Brand": 2.0}}
    monkeypatch.setattr(va, "OBC_BUCKET_BRAND_MULTIPLIERS", bad)
    with pytest.raises(AssertionError, match="OBC_BUCKET_BRAND_MULTIPLIERS"):
        va._assert_evse_brand_priors_ok()


def test_brand_priors_currently_valid() -> None:
    # The real module constants must pass (re-running is idempotent).
    va._assert_evse_brand_priors_ok()


# ---------------------------------------------------------------------------
# _apply_sales_mix_overlay arithmetic (lines 770, 780, 795-800, 805-808).
# ---------------------------------------------------------------------------

def test_overlay_empty_is_noop() -> None:
    mix = _toy_mix()
    out = va._apply_sales_mix_overlay(mix, ())
    pd.testing.assert_frame_equal(out, mix)


def test_overlay_positive_pickup_increases_share() -> None:
    mix = _toy_mix()
    out = va._apply_sales_mix_overlay(mix, {"pickup_overlay": 0.10})
    assert np.isclose(out["share"].sum(), 1.0)
    pickup_before = float(mix.loc[mix["archetype"] == "pickup", "share"].sum())
    pickup_after = float(out.loc[out["archetype"] == "pickup", "share"].sum())
    assert pickup_after > pickup_before


def test_overlay_plain_key_without_suffix() -> None:
    # Line 780: key that does not end in "_overlay" maps to itself.
    mix = _toy_mix()
    out = va._apply_sales_mix_overlay(mix, {"pickup": 0.05})
    assert np.isclose(out["share"].sum(), 1.0)


def test_overlay_negative_delta_zeroes_archetype() -> None:
    # target_total <= 0 path (lines 795-796): drive pickup share to 0.
    mix = _toy_mix()
    out = va._apply_sales_mix_overlay(mix, {"pickup": -1.0})
    pickup_after = float(out.loc[out["archetype"] == "pickup", "share"].sum())
    assert np.isclose(pickup_after, 0.0)
    assert np.isclose(out["share"].sum(), 1.0)


def test_overlay_absent_archetype_is_noop() -> None:
    # Lines 797-800: positive delta on an archetype absent from the mix.
    mix = _toy_mix()
    out = va._apply_sales_mix_overlay(mix, {"motorcycle": 0.20})
    # No motorcycle rows to synthesise; shares unchanged after renorm.
    assert np.isclose(out["share"].sum(), 1.0)
    pd.testing.assert_series_equal(
        out["share"].reset_index(drop=True),
        mix["share"].reset_index(drop=True),
    )


def test_overlay_collapse_raises() -> None:
    # Lines 805-808: drive every archetype negative -> zero total.
    mix = _toy_mix()
    overlay = {"sedan": -1.0, "crossover": -1.0, "suv": -1.0, "pickup": -1.0}
    with pytest.raises(ValueError, match="collapsed the mix to a zero"):
        va._apply_sales_mix_overlay(mix, overlay)


# ---------------------------------------------------------------------------
# _powertrain_subset (lines 815-826) and _infer_bev_share (829-838).
# ---------------------------------------------------------------------------

def test_powertrain_subset_adds_powertrain_and_normalises() -> None:
    # Lines 816-817: mix without a 'powertrain' column triggers classify.
    mix = _toy_mix()
    sub = va._powertrain_subset(mix, "BEV")
    assert "powertrain" in sub.columns
    assert np.isclose(sub["weight_norm"].sum(), 1.0)
    assert (sub["powertrain"] == "BEV").all()


def test_powertrain_subset_empty_raises() -> None:
    # Line 820: a mix with no rows for the requested powertrain.
    mix = _toy_mix()
    # Drop the lone PHEV row.
    mix_no_phev = mix[mix["model"] != "Prius_Prime"].reset_index(drop=True)
    with pytest.raises(ValueError, match="no rows for powertrain 'PHEV'"):
        va._powertrain_subset(mix_no_phev, "PHEV")


def test_infer_bev_share_adds_powertrain() -> None:
    # Line 832: assigns 'powertrain' when missing.
    mix = _toy_mix()
    share = va._infer_bev_share(mix)
    bev_share = float(mix[mix["model"] != "Prius_Prime"]["share"].sum())
    assert np.isclose(share, bev_share)


def test_infer_bev_share_zero_total_fallback() -> None:
    # Line 837: zero total share -> legacy default.
    mix = _toy_mix()
    mix["share"] = 0.0
    assert va._infer_bev_share(mix) == va.BEV_SHARE_LEGACY_DEFAULT


# ---------------------------------------------------------------------------
# _resolve_home_stations (line 923) and _income_weights (line 937).
# ---------------------------------------------------------------------------

def test_home_stations_bay_area_fallback() -> None:
    # Line 923/929: bay_area with no temperature_stations -> fallback set.
    region = _bay_area_clone(temperature_stations=())
    assert region.name == "bay_area"
    assert va._resolve_home_stations(region) == va.HOME_STATIONS_BAY_AREA_FALLBACK


def test_home_stations_uses_region_stations_when_present() -> None:
    region = _bay_area_clone(temperature_stations=("KXYZ",))
    assert va._resolve_home_stations(region) == ("KXYZ",)


def test_income_weights_cvrp_ca_fallback() -> None:
    # Line 937: a non-acs_national scheme uses the CVRP-CA prior.
    region = _bay_area_clone(income_tier_scheme="cvrp_ca")
    assert va._income_weights(region) == va.INCOME_TIER_WEIGHTS_CVRP_CA


def test_income_weights_acs_national() -> None:
    region = _bay_area_clone(income_tier_scheme="acs_national")
    assert va._income_weights(region) == va.INCOME_TIER_WEIGHTS_ACS_NATIONAL


# ---------------------------------------------------------------------------
# _evse_mix_for_profile / _panel_cap_prob_for_profile.
# ---------------------------------------------------------------------------

def test_evse_mix_and_panel_cap_dispatch() -> None:
    assert va._evse_mix_for_profile("workplace") is va.EVSE_MIX_WORKPLACE
    assert va._evse_mix_for_profile("residential") is va.EVSE_MIX_RESIDENTIAL
    assert va._panel_cap_prob_for_profile("workplace") == va.PANEL_CAP_PROB_WORKPLACE
    assert (
        va._panel_cap_prob_for_profile("residential")
        == va.PANEL_CAP_PROB_RESIDENTIAL
    )


# ---------------------------------------------------------------------------
# _stratified_resample_heat_pump (RFC-032; lines 967-1021).
# ---------------------------------------------------------------------------

def test_resample_heat_pump_bad_rate_raises() -> None:
    arr = np.zeros(4, dtype=bool)
    arch = np.array(["sedan"] * 4, dtype=object)
    with pytest.raises(ValueError, match=r"must be in \[0, 1\]"):
        va._stratified_resample_heat_pump(arr, arch, target_rate=1.5, seed=7)


def test_resample_heat_pump_empty_input() -> None:
    arr = np.zeros(0, dtype=bool)
    arch = np.array([], dtype=object)
    out = va._stratified_resample_heat_pump(arr, arch, target_rate=0.5, seed=7)
    assert out.shape[0] == 0


def test_resample_heat_pump_hits_target_rate() -> None:
    rng = np.random.default_rng(0)
    n = 4000
    archs = np.array(
        rng.choice(list(va.ARCHETYPE_CATEGORIES), size=n), dtype=object
    )
    start = rng.random(n) < 0.10  # ~10 % start rate
    out = va._stratified_resample_heat_pump(
        start, archs, target_rate=0.50, seed=12345
    )
    achieved = float(out.mean())
    # Per-stratum proportional allocation hits the global target tightly.
    assert abs(achieved - 0.50) <= 0.01, achieved


def test_resample_heat_pump_flip_down() -> None:
    # current > want path (lines 1014-1020): start at 100 %, target low.
    n = 1000
    archs = np.array(["sedan"] * n, dtype=object)
    start = np.ones(n, dtype=bool)
    out = va._stratified_resample_heat_pump(
        start, archs, target_rate=0.20, seed=99
    )
    assert abs(float(out.mean()) - 0.20) <= 0.01


def test_resample_heat_pump_deterministic() -> None:
    n = 500
    rng = np.random.default_rng(3)
    archs = np.array(
        rng.choice(list(va.ARCHETYPE_CATEGORIES), size=n), dtype=object
    )
    start = rng.random(n) < 0.3
    a = va._stratified_resample_heat_pump(start, archs, 0.6, seed=7)
    b = va._stratified_resample_heat_pump(start, archs, 0.6, seed=7)
    assert np.array_equal(a, b)


def test_resample_heat_pump_no_change_when_already_at_target() -> None:
    # want == current short-circuit (line 1005-1006).
    n = 100
    archs = np.array(["suv"] * n, dtype=object)
    start = np.zeros(n, dtype=bool)
    out = va._stratified_resample_heat_pump(start, archs, 0.0, seed=1)
    assert not out.any()


# ---------------------------------------------------------------------------
# _apply_obc_overrides (line 1050 early-exit) and _sample_brand (1088-1103).
# ---------------------------------------------------------------------------

def test_apply_obc_overrides_lucid_wildcard_and_early_exit() -> None:
    # All rows match a wildcard rule -> `applied.all()` triggers the
    # early break at line 1050 on the next iteration.
    make = np.array(["Lucid", "Lucid"], dtype=object)
    model = np.array(["Air", "Sapphire"], dtype=object)
    year = np.array([2023, 2024], dtype=np.int16)
    obc = np.array([6.6, 6.6], dtype=np.float32)
    out = va._apply_obc_overrides(obc, make, model, year)
    assert np.allclose(out, 19.2)


def test_apply_obc_overrides_lightning_band_split() -> None:
    make = np.array(["Ford", "Ford"], dtype=object)
    model = np.array(["F150_Lightning", "F150_Lightning"], dtype=object)
    year = np.array([2022, 2024], dtype=np.int16)
    obc = np.array([6.6, 6.6], dtype=np.float32)
    out = va._apply_obc_overrides(obc, make, model, year)
    assert np.isclose(out[0], 19.2)  # MY22 ER bundle
    assert np.isclose(out[1], 11.5)  # MY24+ step-down


def test_sample_brand_all_masked_fallback() -> None:
    # Lines 1097-1098: a bucket multiplier that zeroes every brand falls
    # back to the identity (base) prior.
    rng = np.random.default_rng(0)
    base = {"Tesla": 0.5, "ChargePoint": 0.5}
    bucket = np.full(20, 6.6, dtype=np.float64)
    mults = {6.6: {"Tesla": 0.0, "ChargePoint": 0.0}}
    out = va._sample_brand(rng, base, bucket, mults, 20)
    assert set(np.unique(out)).issubset({"Tesla", "ChargePoint"})
    assert out.shape[0] == 20


def test_sample_brand_identity_when_no_multiplier() -> None:
    # Line 1092 (mult_vec = ones) branch: bucket with no multiplier entry.
    rng = np.random.default_rng(1)
    base = {"Tesla": 0.7, "Blink": 0.3}
    bucket = np.full(50, 11.5, dtype=np.float64)  # 11.5 -> identity {}
    out = va._sample_brand(rng, base, bucket, {11.5: {}}, 50)
    assert set(np.unique(out)).issubset({"Tesla", "Blink"})


# ---------------------------------------------------------------------------
# _match_bundle_predicate wildcard branches (lines 1120, 1128).
# ---------------------------------------------------------------------------

def test_match_bundle_predicate_make_and_powertrain_wildcards() -> None:
    rule = va.EvEvseBundleRule(
        name="wild",
        predicate_make="*",
        predicate_model="*",
        predicate_powertrain="*",
        model_year_min=2018,
        model_year_max=2099,
        overrides={},
        take_rate=1.0,
        citation="[lit §0.0]",
    )
    make = np.array(["Foo", "Bar"], dtype=object)
    model = np.array(["X", "Y"], dtype=object)
    year = np.array([2020, 2025], dtype=np.int16)
    pt = np.array(["BEV", "PHEV"], dtype=object)
    mask = va._match_bundle_predicate(rule, make, model, year, pt)
    assert mask.all()


# ---------------------------------------------------------------------------
# _connector_prior_for_make_my fallback (line 1236).
# ---------------------------------------------------------------------------

def test_connector_prior_unknown_make_uses_wildcard() -> None:
    prior = va._connector_prior_for_make_my("UnknownMake", 2023)
    assert prior == {"J1772": 0.97, "NACS": 0.03}


def test_connector_prior_out_of_band_year_fallback() -> None:
    # Line 1236: a year below every band falls to the wildcard first bucket.
    prior = va._connector_prior_for_make_my("Tesla", 1990)
    assert prior == va.EVSE_CONNECTOR_MIX_BY_MAKE_MY["*"][0][2]


def test_sample_connector_branches() -> None:
    # Exercises pending-override, Tesla pre/post-2024, Lectron, and the
    # generic make/MY lookup in a single residential call.
    rng = np.random.default_rng(2)
    make = np.array(["Ford", "Tesla", "Tesla", "Lectron_make", "Toyota"], dtype=object)
    year = np.array([2023, 2022, 2025, 2023, 2024], dtype=np.int16)
    brand = np.array(
        ["ChargePoint", "Tesla", "Tesla", "Lectron", "Blink"], dtype=object
    )
    pending = np.array([None, None, None, None, "CCS1"], dtype=object)
    out = va._sample_connector(rng, make, year, brand, pending, "residential")
    assert out[1] == "Tesla_NACS_native"  # Tesla pre-2024
    assert out[4] == "CCS1"  # pending override honoured
    assert out[3] in va.NEMA_OUTLET_MIX  # Lectron -> outlet draw


def test_sample_connector_workplace_short_circuit() -> None:
    rng = np.random.default_rng(5)
    make = np.array(["Tesla", "Ford"], dtype=object)
    year = np.array([2020, 2023], dtype=np.int16)
    brand = np.array(["Tesla", "ChargePoint"], dtype=object)
    pending = np.array([None, None], dtype=object)
    out = va._sample_connector(rng, make, year, brand, pending, "workplace")
    assert set(np.unique(out)).issubset(set(va.EVSE_CONNECTOR_MIX_WORKPLACE))


# ---------------------------------------------------------------------------
# sample_fleet validation error paths (lines 1348, 1353-1356, 1368).
# ---------------------------------------------------------------------------

def test_sample_fleet_bad_n_raises() -> None:
    with pytest.raises(ValueError, match=r"n must be in \[1, 50_000\]"):
        va.sample_fleet(n=0)
    with pytest.raises(ValueError, match=r"n must be in \[1, 50_000\]"):
        va.sample_fleet(n=50_001)


def test_sample_fleet_unknown_sales_mix_source_raises() -> None:
    # Line 1353-1356: region whose sales_mix_source isn't registered.
    region = _bay_area_clone(sales_mix_source="not_a_real_source")
    with pytest.raises(ValueError, match="unknown sales_mix_source"):
        va.sample_fleet(n=10, region=region)


@_needs_sales_mix
def test_sample_fleet_bad_bev_share_raises() -> None:
    # Line 1368: explicit out-of-range bev_share override.
    with pytest.raises(ValueError, match=r"bev_share must be in \[0, 1\]"):
        va.sample_fleet(n=10, region="bay_area", bev_share=1.5)


# ---------------------------------------------------------------------------
# Heat-pump-override path through sample_fleet (lines 1518-1523).
# ---------------------------------------------------------------------------

@_needs_sales_mix
def test_sample_fleet_heat_pump_override_path() -> None:
    region = _bay_area_clone(heat_pump_share_override=0.40)
    df = va.sample_fleet(
        n=3000, seed=va.MASTER_SEED_DEFAULT, region=region,
    )
    achieved = float(df["has_heat_pump"].mean())
    assert abs(achieved - 0.40) <= 0.02, achieved


# ---------------------------------------------------------------------------
# _cast_to_schema / write_fleet (lines 1731-1735, 1738-1745).
# ---------------------------------------------------------------------------

@_needs_sales_mix
def test_write_fleet_roundtrip(tmp_path: Path) -> None:
    df = va.sample_fleet(n=64, seed=va.MASTER_SEED_DEFAULT, region="bay_area")
    out = va.write_fleet(df, tmp_path / "sub" / "fleet.parquet")
    assert out.exists()
    back = pd.read_parquet(out)
    assert len(back) == 64
    assert list(back.columns) == [c for c, _, _ in va.SCHEMA]
    # ev_id is sorted ascending and contiguous after write.
    assert back["ev_id"].tolist() == list(range(64))


@_needs_sales_mix
def test_cast_to_schema_object_category_branch(tmp_path: Path) -> None:
    # Line 1731-1732: a categorical column passed as plain object is cast
    # to category by _cast_to_schema.
    df = va.sample_fleet(n=32, seed=va.MASTER_SEED_DEFAULT, region="bay_area")
    df = df.copy()
    df["make"] = df["make"].astype(str)  # demote category -> object
    cast = va._cast_to_schema(df)
    assert isinstance(cast["make"].dtype, pd.CategoricalDtype)


def test_default_out_dir() -> None:
    # Line 1749.
    p = va._default_out_dir(Path("/repo"), "bay_area", "workplace")
    assert p == Path(
        "/repo/data/pev/processed/bay_area/workplace_ev_synth"
    )


# ---------------------------------------------------------------------------
# write_meta / _hash_sales_mix / _resolve_git_commit (lines 1761-1902).
# ---------------------------------------------------------------------------

@_needs_sales_mix
def test_write_meta_full(tmp_path: Path) -> None:
    region = REGIONS["bay_area"]
    df = va.sample_fleet(
        n=200, seed=va.MASTER_SEED_DEFAULT, region=region,
    )
    bev_share = float((df["powertrain"].astype(str) == "BEV").mean())
    out = va.write_meta(
        df, tmp_path / "meta.json",
        master_seed=va.MASTER_SEED_DEFAULT, bev_share=bev_share,
        region=region, profile_type="residential",
    )
    assert out.exists()
    meta = json.loads(out.read_text())
    assert meta["methodology_version"] == va.METHODOLOGY_VERSION
    assert meta["master_seed"] == va.MASTER_SEED_DEFAULT
    assert meta["n_vehicles"] == 200
    assert meta["region"]["name"] == "bay_area"
    assert meta["profile_type"] == "residential"
    assert "evse_priors" in meta["raw_input_sha256"]
    assert "consumption_model" in meta
    vmeta = meta["vehicle_archetypes"]
    assert np.isclose(vmeta["bev_share"], bev_share)
    assert np.isclose(vmeta["phev_share"], 1.0 - bev_share)
    assert vmeta["evse_mix_profile"] == "residential"
    assert vmeta["evse_brand_realised_share"]


@_needs_sales_mix
def test_write_meta_workplace_with_hp_override(tmp_path: Path) -> None:
    region = _bay_area_clone(heat_pump_share_override=0.30)
    df = va.sample_fleet(
        n=200, seed=va.MASTER_SEED_DEFAULT,
        region=region, profile_type="workplace",
    )
    out = va.write_meta(
        df, tmp_path / "meta.json",
        master_seed=va.MASTER_SEED_DEFAULT, bev_share=0.85,
        region=region, profile_type="workplace",
    )
    meta = json.loads(out.read_text())
    assert meta["profile_type"] == "workplace"
    assert meta["vehicle_archetypes"]["evse_mix_profile"] == "workplace"
    assert meta["region"]["heat_pump_share_override"] == 0.30
    assert meta["consumption_model"]["heat_pump_share_override"] == 0.30


@_needs_sales_mix
def test_hash_sales_mix_stable() -> None:
    h1 = va._hash_sales_mix("cvrp_ca")
    h2 = va._hash_sales_mix("cvrp_ca")
    assert h1 == h2 and len(h1) == 64


def test_hash_sales_mix_unavailable() -> None:
    # Lines 1880-1881: load failure returns the sentinel.
    assert va._hash_sales_mix("definitely_not_a_source") == "unavailable"


def test_resolve_git_commit_returns_string() -> None:
    # Lines 1887-1899: best-effort; either a 40-char SHA or 'unknown'.
    sha = va._resolve_git_commit()
    assert isinstance(sha, str)
    assert sha == "unknown" or len(sha) == 40


def test_resolve_git_commit_unknown_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Line 1902: subprocess raising -> 'unknown' fallback.
    import subprocess

    def _boom(*_args: object, **_kw: object) -> None:
        raise OSError("git unavailable")

    monkeypatch.setattr(subprocess, "run", _boom)
    assert va._resolve_git_commit() == "unknown"


# ---------------------------------------------------------------------------
# main() CLI (lines 1909-1985).
# ---------------------------------------------------------------------------

@_needs_sales_mix
def test_main_cli_writes_outputs(tmp_path: Path) -> None:
    out_parquet = tmp_path / "fleet.parquet"
    out_meta = tmp_path / "meta.json"
    rc = va.main(
        [
            "--n", "50",
            "--seed", str(va.MASTER_SEED_DEFAULT),
            "--region", "bay_area",
            "--profile-type", "residential",
            "--out", str(out_parquet),
            "--meta-out", str(out_meta),
            "--repo-root", str(tmp_path),
        ]
    )
    assert rc == 0
    assert out_parquet.exists()
    assert out_meta.exists()
    back = pd.read_parquet(out_parquet)
    assert len(back) == 50


@_needs_sales_mix
def test_main_cli_with_bev_share_and_default_out(tmp_path: Path) -> None:
    # Exercises the --bev-share branch (eff_bev = float(args.bev_share))
    # and the default out_dir construction under --repo-root.
    rc = va.main(
        [
            "--n", "40",
            "--region", "bay_area",
            "--bev-share", "0.90",
            "--repo-root", str(tmp_path),
        ]
    )
    assert rc == 0
    default_dir = tmp_path / "data" / "pev" / "processed" / "bay_area" / (
        "residential_ev_synth"
    )
    assert (default_dir / "fleet.parquet").exists()
    assert (default_dir / "meta.json").exists()


@_needs_sales_mix
def test_main_cli_relative_out_paths_resolved_under_repo_root(
    tmp_path: Path,
) -> None:
    # Lines 1957-1961: relative --out / --meta-out are joined onto repo_root.
    rc = va.main(
        [
            "--n", "30",
            "--region", "bay_area",
            "--out", "rel/fleet.parquet",
            "--meta-out", "rel/meta.json",
            "--repo-root", str(tmp_path),
        ]
    )
    assert rc == 0
    assert (tmp_path / "rel" / "fleet.parquet").exists()
    assert (tmp_path / "rel" / "meta.json").exists()
