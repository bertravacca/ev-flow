"""Unit tests for M9 — `pev_synth.validator`.

Required tests per the M9 brief:

    1. test_validation_report_schema_complete
    2. test_summary_counts_match_total
    3. test_all_check_ids_present
    4. test_explained_fail_cites_plan_section_13
    5. test_optimizer_smoke_test_runs_on_sample_ev
    6. test_unbounded_v1_treated_as_explained_skip
    7. test_no_silent_pass_on_value_outside_bound

Tests use the actual on-disk pipeline artifacts (M1..M8 already complete).
The validator is deterministic; the only non-byte-identical field between
runs is `run_timestamp_utc`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make `code/` importable when tests are launched from anywhere.
_CODE_DIR = Path(__file__).resolve().parents[1]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from pev_synth import validator as v  # noqa: E402

# ---------------------------------------------------------------------------
# Paths (real artifacts on disk)
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
# v2.0 per-region layout (the v1.1 flat `processed/residential_ev_synth`
# path no longer exists). The legacy v1.1 validator path (no region/
# profile_type kwargs) resolves bounds to the `bay_area/residential`
# subtree, so we point OUTPUT_ROOT at the matching bay_area residential
# cache dir, which carries all the M1..M8 parquets.
OUTPUT_ROOT = (
    REPO_ROOT / "data" / "pev" / "processed" / "bay_area" / "residential_ev_synth"
)
BOUNDS_ROOT = REPO_ROOT / "data" / "pev" / "validation_bounds"

# Skip decorator for tests that exercise the validator against the real
# pipeline outputs. The `data/` tree is .gitignore-d, so on a clean checkout
# these directories do not exist; rebuild them by running pipeline modules
# M1..M8 (see plan §1) before exercising M9.
_needs_pipeline_artifacts = pytest.mark.skipif(
    not OUTPUT_ROOT.exists() or not BOUNDS_ROOT.exists(),
    reason=(
        f"validator inputs missing (need {OUTPUT_ROOT} and {BOUNDS_ROOT}); "
        "rebuild pipeline modules M1..M8 first"
    ),
)

EXPECTED_CHECK_IDS: tuple[str, ...] = (
    "battery_capacity_distribution",
    "obc_distribution",
    "weekday_plugin_cdf",
    "weekend_plugin_cdf",
    "session_duration_distribution",
    "energy_per_session_distribution",
    "annual_mileage",
    "wh_per_mi_at_70F",
    "eta_charging",
    "evi_pro_plug_in_split",
    "optimizer_smoke_test",
    "cross_artifact_integrity",
    "plug_rate_consistency",
    "fleet_energy_conservation",
)


@pytest.fixture(scope="module")
def report() -> v.Report:
    """Run the validator once for the module and return the in-memory Report.

    ``skip_optimizer=True``: the optimiser smoke check (#11) imports the
    in-house ``pev_optim`` / ``caiso_data`` / ``pev_data`` packages, which
    are NOT shipped with ev-flow and are absent here (and in CI). Without
    them the check records ERROR and the optimiser dependency would mask
    the data-fidelity checks this module exercises. Skipping the optimiser
    keeps the smoke row at ERROR (a recognised status, not a FAIL) and lets
    every M8-bound data-fidelity check run honestly. The dedicated optimiser
    test below runs its own un-skipped pass guarded by `importorskip`.
    """
    assert OUTPUT_ROOT.exists(), f"output_root missing: {OUTPUT_ROOT}"
    assert BOUNDS_ROOT.exists(), f"bounds_root missing: {BOUNDS_ROOT}"
    return v.run(output_root=OUTPUT_ROOT, bounds_root=BOUNDS_ROOT, skip_optimizer=True)


@pytest.fixture(scope="module")
def report_dict(report: v.Report) -> dict:
    return report.to_dict()


# ---------------------------------------------------------------------------
# Required tests
# ---------------------------------------------------------------------------

@_needs_pipeline_artifacts
def test_validation_report_schema_complete(report_dict: dict) -> None:
    """Acceptance criterion 1 — validation_report.json has the documented top-level
    schema."""
    required_keys = {
        "validator_version",
        "methodology_version",
        "run_timestamp_utc",
        "pipeline_artifacts_sha256",
        "checks",
        "summary",
    }
    assert set(report_dict.keys()) >= required_keys

    assert report_dict["validator_version"] == "v1.0.0"
    assert report_dict["methodology_version"] == "v1.0.0"
    assert isinstance(report_dict["pipeline_artifacts_sha256"], dict)
    assert all(
        isinstance(k, str) and isinstance(v_, str) and len(v_) == 64
        for k, v_ in report_dict["pipeline_artifacts_sha256"].items()
    )

    # Every check row has the documented sub-schema.
    required_check_keys = {
        "check_id", "status", "realised_value", "bound_source",
        "details", "explained_reason",
    }
    for c in report_dict["checks"]:
        assert required_check_keys <= set(c.keys()), c
        assert c["status"] in v.VALID_STATUSES, c["status"]

    # JSON file actually exists and is valid JSON.
    json_path = OUTPUT_ROOT / "validation_report.json"
    assert json_path.exists()
    with json_path.open() as f:
        loaded = json.load(f)
    assert loaded["validator_version"] == "v1.0.0"

    # Companion markdown exists.
    md_path = OUTPUT_ROOT / "validation_report.md"
    assert md_path.exists()
    text = md_path.read_text()
    assert "M9 validation report" in text
    # Every check_id appears as a row.
    for cid in EXPECTED_CHECK_IDS:
        assert cid in text


@_needs_pipeline_artifacts
def test_summary_counts_match_total(report_dict: dict) -> None:
    """Acceptance criterion 3 — summary counts add up to total (14)."""
    s = report_dict["summary"]
    counted = s["pass"] + s["fail"] + s["explained_fail"] + s["explained_skip"] + s["error"]
    assert counted == s["total"]
    assert s["total"] == 14
    assert s["total"] == len(report_dict["checks"])


@_needs_pipeline_artifacts
def test_all_check_ids_present(report_dict: dict) -> None:
    """Acceptance criterion 4 — every documented check_id appears exactly once."""
    ids = [c["check_id"] for c in report_dict["checks"]]
    assert sorted(ids) == sorted(EXPECTED_CHECK_IDS)
    # No dupes
    assert len(ids) == len(set(ids))
    # Every check has a status in the documented set.
    for c in report_dict["checks"]:
        assert c["status"] in v.VALID_STATUSES


@_needs_pipeline_artifacts
def test_explained_fail_cites_plan_section_13(report_dict: dict) -> None:
    """Acceptance criterion 5 — every EXPLAINED_FAIL cites §13.<n>."""
    explained_fails = [c for c in report_dict["checks"] if c["status"] == "EXPLAINED_FAIL"]
    # There must be at least one EXPLAINED_FAIL (the brief identifies several)
    # so this test exercises a real path, not a vacuous one.
    assert len(explained_fails) >= 1, "no EXPLAINED_FAIL rows produced"
    for c in explained_fails:
        reason = c.get("explained_reason", "")
        assert "§13." in reason, (
            f"EXPLAINED_FAIL row {c['check_id']} does not cite §13: {reason!r}"
        )


@_needs_pipeline_artifacts
def test_optimizer_smoke_test_runs_on_sample_ev() -> None:
    """Acceptance criterion 7 — optimizer smoke runs end-to-end without raising.

    The smoke check needs the in-house ``pev_optim`` package (plus
    ``caiso_data`` / ``pev_data``), which are not shipped with ev-flow.
    Skip cleanly when absent; the shared module fixture runs with
    ``skip_optimizer=True`` so the other data-fidelity checks aren't gated
    on the optimiser. Here we run a dedicated, un-skipped pass.
    """
    pytest.importorskip("pev_optim")
    rep = v.run(output_root=OUTPUT_ROOT, bounds_root=BOUNDS_ROOT, skip_optimizer=False)
    rep_dict = rep.to_dict()
    row = next(c for c in rep_dict["checks"] if c["check_id"] == "optimizer_smoke_test")
    # Must NOT raise (status would otherwise be FAIL with traceback text).
    assert row["status"] in ("PASS", "EXPLAINED_FAIL"), row
    if row["status"] == "PASS":
        rv = row["realised_value"]
        assert "statuses" in rv and len(rv["statuses"]) >= 1
        assert all(
            s in ("optimal", "feasible", "infeasible") for s in rv["statuses"]
        )


@_needs_pipeline_artifacts
def test_unbounded_v1_treated_as_explained_skip(report_dict: dict) -> None:
    """obc_distribution and evi_pro_plug_in_split are all unbounded_v1=true
    in the M8 bound parquets; the validator MUST emit EXPLAINED_SKIP for both."""
    for cid in ("obc_distribution", "evi_pro_plug_in_split"):
        row = next(c for c in report_dict["checks"] if c["check_id"] == cid)
        assert row["status"] == "EXPLAINED_SKIP", (
            f"{cid} should be EXPLAINED_SKIP, got {row['status']}"
        )


@_needs_pipeline_artifacts
def test_no_silent_pass_on_value_outside_bound() -> None:
    """For every PASS row, the realised value MUST sit inside the bound.

    We probe with a synthetic CheckResult-equivalent assertion: take each
    PASS row produced and verify it satisfies its own claim. This guards
    against an accidental ``return PASS`` on an out-of-band value.
    """
    rep_dict = json.loads((OUTPUT_ROOT / "validation_report.json").read_text())
    for c in rep_dict["checks"]:
        if c["status"] != "PASS":
            continue
        rv = c["realised_value"]
        cid = c["check_id"]
        if cid == "wh_per_mi_at_70F":
            # Per-archetype: ≥75% must be in band for the PASS to be honest.
            n_pass = rv["n_models_pass"]
            n_total = rv["n_models_total"]
            assert n_pass / n_total >= 0.75, c
        elif cid == "eta_charging":
            lo, hi = rv["bound_l2"]
            assert lo <= rv["mean"] <= hi, c
        elif cid == "cross_artifact_integrity":
            assert rv["sessions_dup_pk"] == 0 and not rv["issues"], c
        elif cid == "plug_rate_consistency":
            assert rv["abs_diff"] <= rv["tolerance"], c
        elif cid == "optimizer_smoke_test":
            assert len(rv["statuses"]) >= 1, c
            assert all(
                s in ("optimal", "feasible", "infeasible") for s in rv["statuses"]
            ), c
        elif cid == "battery_capacity_distribution":
            assert rv["tvd"] <= 0.05, c
        elif cid == "weekday_plugin_cdf" or cid == "weekend_plugin_cdf":
            for h in ("F_06", "F_12", "F_18", "F_22"):
                lo = rv["bounds"][h]["lo"]
                hi = rv["bounds"][h]["hi"]
                val = rv[h]
                assert lo <= val <= hi, (cid, h, val, lo, hi)
        elif cid == "session_duration_distribution":
            lo, hi = rv["median_bound"]
            assert lo <= rv["median_h"] <= hi, c
            lo, hi = rv["p95_bound"]
            assert lo <= rv["p95_h"] <= hi, c
        elif cid == "energy_per_session_distribution":
            lo, hi = rv["median_bound"]
            assert lo <= rv["median_kwh"] <= hi, c
            lo, hi = rv["p95_bound"]
            assert lo <= rv["p95_kwh"] <= hi, c
        elif cid == "annual_mileage":
            lo, hi = rv["mean_bound"]
            assert lo <= rv["median_miles_per_year"] <= hi, c
            lo, hi = rv["std_bound"]
            assert lo <= rv["std_miles_per_year"] <= hi, c
        elif cid == "fleet_energy_conservation":
            lo, hi = rv["tolerance_band"]
            assert lo <= rv["ratio"] <= hi, c


# ---------------------------------------------------------------------------
# Extra behavioural tests (beyond the brief's 7)
# ---------------------------------------------------------------------------

@_needs_pipeline_artifacts
def test_reproducibility_byte_identical_modulo_timestamp() -> None:
    """Acceptance criterion 8 — re-running M9 produces byte-identical JSON
    after stripping `run_timestamp_utc`."""
    r1 = v.run(output_root=OUTPUT_ROOT, bounds_root=BOUNDS_ROOT, skip_optimizer=True)
    r2 = v.run(output_root=OUTPUT_ROOT, bounds_root=BOUNDS_ROOT, skip_optimizer=True)
    d1 = r1.to_dict()
    d2 = r2.to_dict()
    d1.pop("run_timestamp_utc")
    d2.pop("run_timestamp_utc")
    assert d1 == d2


@_needs_pipeline_artifacts
def test_summary_majority_pass_or_skip(report_dict: dict) -> None:
    """Acceptance criterion 9 — no silent FAILs. EXPLAINED_FAILs are
    permitted but FAIL is reserved for un-explained breakage."""
    s = report_dict["summary"]
    assert s["fail"] == 0, (
        f"unexplained FAILs leak silently: {s}. Every failure must be "
        "EXPLAINED_FAIL with a §13 citation."
    )


def test_check_result_rejects_invalid_status() -> None:
    """`CheckResult` enforces the documented status enum at construction time."""
    with pytest.raises(ValueError, match="Invalid status"):
        v.CheckResult("dummy", "MAYBE", None, "", "")


def test_check_result_rejects_explained_fail_without_section_13() -> None:
    """EXPLAINED_FAIL without a §13 citation is rejected at construction."""
    with pytest.raises(ValueError, match="§13"):
        v.CheckResult(
            "dummy", "EXPLAINED_FAIL", 0.0, "", "details",
            explained_reason="no citation here",
        )


@_needs_pipeline_artifacts
def test_meta_json_updated_with_m9_provenance() -> None:
    """The validator writes an M9 provenance block into meta.json."""
    meta_path = OUTPUT_ROOT / "meta.json"
    assert meta_path.exists()
    with meta_path.open() as f:
        meta = json.load(f)
    assert "validator" in meta
    assert meta["validator"]["module"] == "pev_synth.validator"
    assert meta["validator"]["validator_version"] == "v1.0.0"
    assert meta["validator"]["methodology_version"] == "v1.0.0"
    assert meta["module_provenance"]["m9_validator"] == "v1.0.0"


def test_in_band_helper_handles_boundary() -> None:
    """`_in_band` includes both endpoints with a numerical-tolerance margin."""
    assert v._in_band(0.0, 0.0, 1.0)
    assert v._in_band(1.0, 0.0, 1.0)
    assert not v._in_band(-0.001, 0.0, 1.0)
    assert not v._in_band(1.001, 0.0, 1.0)


@_needs_pipeline_artifacts
def test_bounds_parquets_load_for_every_check_id() -> None:
    """Every M8 bound parquet referenced by the validator exists and is
    schema-conformant. Post RFC-009.r ``_load_bound`` reads from the
    strict nested layout ``<bounds_root>/<region>/<profile_type>/`` — we
    resolve the bay_area / residential subdir here (legacy v1.1 cache
    origin) so every residential check_id is reachable."""
    nested_root = BOUNDS_ROOT / "bay_area" / "residential"
    for cid in (
        "battery_capacity_distribution",
        "obc_distribution",
        "weekday_plugin_cdf",
        "weekend_plugin_cdf",
        "session_duration_distribution",
        "energy_per_session_distribution",
        "annual_mileage",
        "wh_per_mi_at_70F",
        "eta_charging",
        "evi_pro_plug_in_split",
        "optimizer_smoke_test",
    ):
        df = v._load_bound(nested_root, cid)
        assert {"check_id", "metric_name", "value", "lower_bound", "upper_bound",
                "source_url", "unbounded_v1"} <= set(df.columns), cid
        assert (df["check_id"] == cid).all(), cid


@_needs_pipeline_artifacts
def test_summary_counts_individual_checks_consistent(report_dict: dict) -> None:
    """Cross-verify summary counts match the per-check status tally."""
    s = report_dict["summary"]
    tally = {k.lower(): 0 for k in v.VALID_STATUSES}
    for c in report_dict["checks"]:
        tally[c["status"].lower()] += 1
    for k, expected in tally.items():
        assert s[k] == expected, (k, s[k], expected)


@_needs_pipeline_artifacts
def test_explained_skip_rows_have_no_realised_value(report_dict: dict) -> None:
    """`EXPLAINED_SKIP` rows must NOT report a realised value — the bound is
    informational and the validator does not gate the pipeline on them."""
    for c in report_dict["checks"]:
        if c["status"] == "EXPLAINED_SKIP":
            assert c["realised_value"] is None, c


# ---------------------------------------------------------------------------
# v3.0 — §13.2 RESOLVED: HOUSEID-coherent intra-household stitching
# ---------------------------------------------------------------------------


def _bounds_root_for_annual_mileage(tmp_path: Path) -> Path:
    """Build a tiny per-region/profile bounds_root with annual_mileage only.

    Mirrors the EMFAC2021 bay_area band so the unit tests below do not
    depend on the on-disk validation_bounds tree.
    """
    sub = tmp_path / "bay_area" / "residential"
    sub.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "check_id": ["annual_mileage", "annual_mileage"],
            "metric_name": ["mean_miles_per_year", "std_miles_per_year"],
            "value": [11800.0, 4300.0],
            "lower_bound": [10000.0, 3000.0],
            "upper_bound": [14000.0, 6000.0],
            "unit": ["miles", "miles"],
            "source_url": ["http://example.test/emfac"] * 2,
            "source_title": ["EMFAC2021"] * 2,
            "source_table_or_figure": ["Appendix B"] * 2,
            "digitisation_date": ["2026-05-19"] * 2,
            "extraction_method": ["table_lookup"] * 2,
            "provenance_note": ["test"] * 2,
            "unbounded_v1": [False, False],
        }
    )
    df.to_parquet(sub / "annual_mileage.parquet", index=False)
    return sub


def _make_mobility_with_sigma(target_sigma: float, n_evs: int = 200) -> pd.DataFrame:
    """Build a synthetic mobility frame with exactly the requested σ.

    Each EV emits a single trip whose miles deliver the per-EV total;
    miles are sampled from a normal centered on the EMFAC mean.
    """
    rng = np.random.default_rng(20260528)
    miles = rng.normal(loc=11800.0, scale=target_sigma, size=n_evs)
    miles = np.clip(miles, 100.0, None)  # keep positive
    return pd.DataFrame(
        {
            "ev_id": np.arange(n_evs, dtype=np.int32),
            "miles": miles.astype(np.float32),
        }
    )


def test_check_annual_mileage_v3_passes_when_sigma_in_band(tmp_path: Path) -> None:
    """v3.0 + σ in [3000, 6000] → PASS (§13.2 RESOLVED)."""
    bounds_sub = _bounds_root_for_annual_mileage(tmp_path)
    mob = _make_mobility_with_sigma(target_sigma=4500.0, n_evs=400)
    res = v.check_annual_mileage(mob, bounds_sub, methodology_version="v3.0")
    assert res.status == "PASS", res


def test_check_annual_mileage_v3_explained_fail_when_sigma_out_of_band(
    tmp_path: Path,
) -> None:
    """v3.0 + σ outside band → EXPLAINED_FAIL citing §13.2 as an ACKNOWLEDGED
    NHTS structural limitation (1 designated travday/HH in both 2017 and
    NextGen 2022 — neither v2_per_travday nor v3_within_hh can close the
    gap on the available US travel-survey vintages)."""
    bounds_sub = _bounds_root_for_annual_mileage(tmp_path)
    mob = _make_mobility_with_sigma(target_sigma=20_000.0, n_evs=400)
    res = v.check_annual_mileage(mob, bounds_sub, methodology_version="v3.0")
    assert res.status == "EXPLAINED_FAIL", res
    assert "§13.2" in res.explained_reason


def test_check_annual_mileage_legacy_still_explained_fail(tmp_path: Path) -> None:
    """v1.x / v2.x path with σ outside band → EXPLAINED_FAIL citing §13.2."""
    bounds_sub = _bounds_root_for_annual_mileage(tmp_path)
    mob = _make_mobility_with_sigma(target_sigma=20_000.0, n_evs=400)
    res = v.check_annual_mileage(mob, bounds_sub, methodology_version=None)
    assert res.status == "EXPLAINED_FAIL", res
    assert "§13.2" in res.explained_reason


def test_lim_13_2_carries_acknowledged_marker() -> None:
    """§13.2 limitation string announces v3.0 ACKNOWLEDGED — neither
    v2_per_travday nor v3_within_hh can close the gap on the available US
    travel-survey vintages (1 designated travday/HH in both 2017 and
    NextGen 2022). The text is the honest empirical reframe of D3's
    original "RESOLVED" claim after D5's NextGen 2022 measurement."""
    assert "ACKNOWLEDGED" in v.LIM_13_2 or "NHTS structural" in v.LIM_13_2
    assert "TRAVDAY" in v.LIM_13_2 or "travday" in v.LIM_13_2


def test_methodology_at_or_above_v3_helper() -> None:
    """The internal version-gate helper handles the formats we ship."""
    assert v._methodology_at_or_above_v3("v3.0")
    assert v._methodology_at_or_above_v3("v3.0.0")
    assert v._methodology_at_or_above_v3("v3.1-rev")
    assert v._methodology_at_or_above_v3("3.0")
    assert v._methodology_at_or_above_v3("v4.0")
    assert not v._methodology_at_or_above_v3("v2.0")
    assert not v._methodology_at_or_above_v3("v1.0.0")
    assert not v._methodology_at_or_above_v3(None)
    assert not v._methodology_at_or_above_v3("")
    assert not v._methodology_at_or_above_v3("garbage")


@_needs_pipeline_artifacts
def test_annual_mileage_not_in_explained_fail_set_when_v3_passes(
    tmp_path: Path,
) -> None:
    """Sanity invariant: when methodology >= v3 AND σ is in EMFAC band, the
    annual_mileage CheckResult is NOT EXPLAINED_FAIL.

    This is the headline behavioural promise of the v3 RFC §13.2
    resolution.  We use a synthetic mobility frame so the test does not
    depend on whether the on-disk cache has been regenerated yet.
    """
    bounds_sub = _bounds_root_for_annual_mileage(tmp_path)
    mob = _make_mobility_with_sigma(target_sigma=4500.0, n_evs=400)
    res = v.check_annual_mileage(mob, bounds_sub, methodology_version="v3.0")
    assert res.status != "EXPLAINED_FAIL"
    assert res.status == "PASS"


# ---------------------------------------------------------------------------
# v3.0 — §13.M2 (battery capacity): region-aware sales-mix bound
# ---------------------------------------------------------------------------


def _curate_bounds_for_battery(tmp_path: Path, region: str) -> Path:
    """Build a per-region/profile bounds tree containing the v3 region-aware
    ``battery_capacity_distribution.parquet`` for ``region``.

    Uses the real curator builder (so the parquet matches what
    `curate_bounds` writes) rather than a hand-rolled fixture — keeps
    the test honest if the schema ever evolves.
    """
    from pev_synth.validation_bounds_curator import (
        _build_battery_capacity_distribution,
        _rows_to_table,
        _write_parquet,
    )

    sub = tmp_path / region / "residential"
    sub.mkdir(parents=True, exist_ok=True)
    rows = _build_battery_capacity_distribution(region)
    table = _rows_to_table(rows, "battery_capacity_distribution")
    _write_parquet(table, sub / "battery_capacity_distribution.parquet")
    return sub


def _bay_area_bev_fleet_matching_cvrp_pmf() -> pd.DataFrame:
    """Build a synthetic BEV fleet whose B_kwh histogram matches the
    CVRP-CA PMF computed off ``data/pev/raw/literature/sales_mix/cvrp_ca.csv``.

    The realised TVD vs the bay_area bound should be ≈ 0; we keep an
    extra margin to absorb integer-bin-count rounding when n_bev = 1000.
    """
    from pev_synth.validation_bounds_curator import (
        _battery_pmf_from_sales_mix,
    )

    pmf_bins, _ = _battery_pmf_from_sales_mix("cvrp_ca")
    rows: list[dict[str, object]] = []
    ev_id = 0
    n_total = 1000
    for lo, hi, mass in pmf_bins:
        count = int(round(mass * n_total))
        # Use the bin midpoint for deterministic placement inside the bin.
        midpoint = (lo + hi) / 2.0
        for _ in range(count):
            rows.append({
                "ev_id": ev_id,
                "powertrain": "BEV",
                "B_kwh": float(midpoint),
            })
            ev_id += 1
    return pd.DataFrame(rows)


def test_lim_13_m2_archetype_mix_constant_present() -> None:
    """v3.0 — new §13.M2 (archetype mix) limitation constant is exported,
    cites §13.M2, and is distinct from the v2.1 ``LIM_13_13`` battery
    citation."""
    assert hasattr(v, "LIM_13_M2_ARCHETYPE_MIX")
    assert "§13.M2" in v.LIM_13_M2_ARCHETYPE_MIX
    assert "archetype" in v.LIM_13_M2_ARCHETYPE_MIX.lower()
    # Must not be the same string as the (now-RESOLVED) LIM_13_13.
    assert v.LIM_13_M2_ARCHETYPE_MIX != v.LIM_13_13


def test_check_battery_capacity_distribution_passes_bay_area_v3(
    tmp_path: Path,
) -> None:
    """v3.0 — when the bay_area cache is regenerated against the CVRP-CA
    regional bound, ``battery_capacity_distribution`` is PASS.

    Uses the real region-aware bound builder (cvrp_ca for bay_area) and
    a synthetic BEV fleet whose per-bin counts replicate the CVRP-CA
    PMF; the realised TVD against the bound should be ≈ 0 << 0.05.
    """
    bounds_sub = _curate_bounds_for_battery(tmp_path, "bay_area")
    fleet = _bay_area_bev_fleet_matching_cvrp_pmf()
    res = v.check_battery_capacity_distribution(fleet, bounds_sub)
    assert res.status == "PASS", res
    # The check should report the v3 metric_name we now publish.
    assert res.realised_value["tvd_metric_name"] == "tvd_vs_regional_sales_pmf"
    assert res.realised_value["tvd"] <= 0.05


def test_check_battery_capacity_distribution_emits_lim_13_m2_on_fail(
    tmp_path: Path,
) -> None:
    """v3.0 — when the realised fleet PMF drifts past 5 pp against the
    region-aware bound, the EXPLAINED_FAIL row cites
    ``LIM_13_M2_ARCHETYPE_MIX`` (NOT LIM_13_13 which D1 has repurposed
    for the SPEECh plug-in CDF resolution)."""
    bounds_sub = _curate_bounds_for_battery(tmp_path, "bay_area")
    # Force a degenerate fleet so TVD is large: every BEV at 30 kWh
    # (entirely in the [30, 50) bin, which the CVRP-CA PMF assigns ~0%).
    fleet = pd.DataFrame(
        {
            "ev_id": np.arange(500, dtype=np.int32),
            "powertrain": ["BEV"] * 500,
            "B_kwh": [40.0] * 500,
        }
    )
    res = v.check_battery_capacity_distribution(fleet, bounds_sub)
    assert res.status == "EXPLAINED_FAIL", res
    assert "§13.M2" in res.explained_reason
    # The retired LIM_13_13 string must NOT appear in the battery row's
    # explained_reason — D1 owns that citation now.
    assert res.explained_reason != v.LIM_13_13


def test_check_battery_capacity_distribution_back_compat_alias_ignored(
    tmp_path: Path,
) -> None:
    """v3.0 — the back-compat ``tvd_vs_argonne_pmf`` row carries
    ``unbounded_v1=True`` and must NOT shadow the live
    ``tvd_vs_regional_sales_pmf`` row when the validator picks a bound.

    Sanity-check that the live metric_name surfaces in realised_value.
    """
    bounds_sub = _curate_bounds_for_battery(tmp_path, "new_york_metro")
    fleet = _bay_area_bev_fleet_matching_cvrp_pmf()
    res = v.check_battery_capacity_distribution(fleet, bounds_sub)
    assert res.realised_value["tvd_metric_name"] == "tvd_vs_regional_sales_pmf", res


# ---------------------------------------------------------------------------
# v3.0 — §13.6 RESOLVED: PHEV gasoline extension (D2)
# ---------------------------------------------------------------------------


def _bounds_root_for_energy_per_session(tmp_path: Path) -> Path:
    """Stub bounds tree carrying just ``energy_per_session_distribution``.

    Mirrors a sane INL-derived band so the v3 check can be exercised
    without the on-disk validation_bounds tree.
    """
    sub = tmp_path / "bay_area" / "residential"
    sub.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "check_id": [
                "energy_per_session_distribution",
                "energy_per_session_distribution",
            ],
            "metric_name": [
                "median_kwh_per_session",
                "p95_kwh_per_session",
            ],
            "value": [8.0, 40.0],
            "lower_bound": [4.0, 25.0],
            "upper_bound": [15.0, 60.0],
            "unit": ["kWh", "kWh"],
            "source_url": ["http://example.test/inl"] * 2,
            "source_title": ["INL/EXT-15-36317"] * 2,
            "source_table_or_figure": ["Table 3"] * 2,
            "digitisation_date": ["2026-05-19"] * 2,
            "extraction_method": ["table_lookup"] * 2,
            "provenance_note": ["test"] * 2,
            "unbounded_v1": [False, False],
        }
    )
    df.to_parquet(sub / "energy_per_session_distribution.parquet", index=False)
    return sub


def _synth_sessions_for_d2(
    n_home: int = 50,
    home_energy_kwh: float = 10.0,
    n_phev_gas: int = 20,
    phev_gas_kwh: float = 12.0,
    phev_session_energy_kwh: float = 2.0,
    n_phev_no_gas: int = 10,
) -> pd.DataFrame:
    """Synthetic sessions frame for the v3 PHEV-aware validator tests.

    BEV home rows: 90 % uniform in [4, 12] kWh (median ~8 lands inside
    [4, 15]), 10 % uniform in [25, 35] kWh (lifts the p95 into [25, 60]).
    PHEV rows are constant.
    """
    rng = np.random.default_rng(20260528)
    n_tail = max(1, n_home // 10)
    n_body = n_home - n_tail
    body = rng.uniform(4.0, 12.0, size=n_body).astype(np.float32)
    tail = rng.uniform(25.0, 35.0, size=n_tail).astype(np.float32)
    bev_energies = np.concatenate([body, tail])
    rng.shuffle(bev_energies)
    rows = []
    for i in range(n_home):
        e = float(bev_energies[i])
        rows.append({
            "ev_id": np.int32(i),
            "session_idx": np.int32(0),
            "location": "home",
            "energy_kwh": np.float32(e),
            "energy_kwh_grid": np.float32(e),
            "energy_kwh_batt": np.float32(e * 0.9),
            "gas_kwh_equivalent": np.float32(0.0),
        })
    # PHEV home sessions WITH a gas deficit (excluded from histogram,
    # included in conservation numerator).
    for i in range(n_phev_gas):
        rows.append({
            "ev_id": np.int32(1000 + i),
            "session_idx": np.int32(0),
            "location": "home",
            "energy_kwh": np.float32(phev_session_energy_kwh),
            "energy_kwh_grid": np.float32(phev_session_energy_kwh),
            "energy_kwh_batt": np.float32(phev_session_energy_kwh * 0.9),
            "gas_kwh_equivalent": np.float32(phev_gas_kwh),
        })
    # PHEV home sessions with NO deficit — kept in the histogram.
    for i in range(n_phev_no_gas):
        rows.append({
            "ev_id": np.int32(2000 + i),
            "session_idx": np.int32(0),
            "location": "home",
            "energy_kwh": np.float32(home_energy_kwh),
            "energy_kwh_grid": np.float32(home_energy_kwh),
            "energy_kwh_batt": np.float32(home_energy_kwh * 0.9),
            "gas_kwh_equivalent": np.float32(0.0),
        })
    return pd.DataFrame(rows)


def test_lim_13_6_carries_resolved_marker() -> None:
    """§13.6 limitation string must announce v3.0 RESOLVED."""
    assert "RESOLVED in v3.0" in v.LIM_13_6
    assert "gas_kwh_equivalent" in v.LIM_13_6


def test_check_energy_per_session_filters_phev_gas_rows(tmp_path: Path) -> None:
    """Electricity-only histogram: PHEV sessions with gas > 0 are
    excluded BEFORE the median / p95 are computed.

    The fixture is built so that, if the small PHEV gas rows leaked
    into the histogram, they would drag the median below the INL lower
    bound. After filtering, median = 10 kWh sits comfortably inside
    [4, 15].
    """
    bounds_sub = _bounds_root_for_energy_per_session(tmp_path)
    sessions = _synth_sessions_for_d2(
        n_home=50, home_energy_kwh=10.0,
        n_phev_gas=80, phev_gas_kwh=12.0, phev_session_energy_kwh=2.0,
        n_phev_no_gas=10,
    )
    res = v.check_energy_per_session_distribution(sessions, bounds_sub)
    assert res.status == "PASS", res
    rv = res.realised_value
    # PHEV gas-extension rows surface in the realised_value block.
    assert rv["n_phev_gas_excluded"] == 80
    assert rv["n_home_sessions"] == 60  # 50 BEV + 10 PHEV electricity-only
    assert 4.0 <= rv["median_kwh"] <= 15.0


def test_check_fleet_energy_conservation_includes_gas_kwh() -> None:
    """v3.0 — the fleet_energy_conservation numerator must include
    ``gas_kwh_equivalent``."""
    # Build a deterministic sessions frame for arithmetic checks (no rng).
    n_bev, n_phev = 100, 20
    bev_e = 10.0  # kWh/session
    phev_e = 2.0
    phev_gas = 12.0
    rows = []
    for i in range(n_bev):
        rows.append({
            "ev_id": np.int32(i), "session_idx": np.int32(0),
            "location": "home",
            "energy_kwh": np.float32(bev_e),
            "energy_kwh_grid": np.float32(bev_e),
            "energy_kwh_batt": np.float32(bev_e * 0.9),
            "gas_kwh_equivalent": np.float32(0.0),
        })
    for i in range(n_phev):
        rows.append({
            "ev_id": np.int32(1000 + i), "session_idx": np.int32(0),
            "location": "home",
            "energy_kwh": np.float32(phev_e),
            "energy_kwh_grid": np.float32(phev_e),
            "energy_kwh_batt": np.float32(phev_e * 0.9),
            "gas_kwh_equivalent": np.float32(phev_gas),
        })
    sessions = pd.DataFrame(rows)
    # grid = 100·10 + 20·2 = 1040 kWh; gas = 20·12 = 240 kWh.
    # Pick total_drive such that (1040+240)/total_drive ≈ 1.111.
    total_drive = (1040.0 + 240.0) / 1.111
    mobility = pd.DataFrame({
        "ev_id": np.arange(120, dtype=np.int32),
        "kwh_consumed": np.full(
            120, total_drive / 120.0, dtype=np.float32,
        ),
    })
    res = v.check_fleet_energy_conservation(sessions, mobility)
    rv = res.realised_value
    assert "total_gas_kwh_equivalent" in rv
    assert rv["total_gas_kwh_equivalent"] == pytest.approx(240.0, rel=1e-2)
    assert rv["total_numerator_kwh"] == pytest.approx(
        rv["total_session_kwh"] + rv["total_gas_kwh_equivalent"], rel=1e-3,
    )
    assert res.status == "PASS", res


def test_check_fleet_energy_conservation_legacy_cache_zero_gas() -> None:
    """Pre-v3 sessions (no gas column) must degrade gracefully.

    Treating the missing column as zero preserves v1.1 byte-stability on
    untouched caches — the numerator collapses to the grid-only total
    and the realised block reports a zero gas number.
    """
    sessions = pd.DataFrame({
        "ev_id": np.arange(20, dtype=np.int32),
        "session_idx": np.zeros(20, dtype=np.int32),
        "location": ["home"] * 20,
        "energy_kwh": np.full(20, 10.0, dtype=np.float32),
        "energy_kwh_grid": np.full(20, 10.0, dtype=np.float32),
        "energy_kwh_batt": np.full(20, 9.0, dtype=np.float32),
    })
    mobility = pd.DataFrame({
        "ev_id": np.arange(20, dtype=np.int32),
        "kwh_consumed": np.full(20, 9.0, dtype=np.float32),
    })
    res = v.check_fleet_energy_conservation(sessions, mobility)
    assert res.realised_value["total_gas_kwh_equivalent"] == pytest.approx(0.0)
    # 200 / 180 = 1.111 → PASS.
    assert res.status == "PASS"


@_needs_pipeline_artifacts
def test_fleet_energy_conservation_surfaces_gas_field_in_report() -> None:
    """The v3 validator report must surface the new gas-side fields on
    the ``fleet_energy_conservation`` row regardless of whether the
    cache has been regenerated.

    The value may be 0.0 on pre-v3 caches that still ship sessions
    without ``gas_kwh_equivalent``; the column key itself, however,
    must always appear so downstream tooling can read the v3 schema
    unconditionally.
    """
    rep = v.run(
        output_root=OUTPUT_ROOT, bounds_root=BOUNDS_ROOT, skip_optimizer=True,
    )
    row = next(
        c for c in rep.to_dict()["checks"]
        if c["check_id"] == "fleet_energy_conservation"
    )
    rv = row["realised_value"]
    assert "total_gas_kwh_equivalent" in rv
    assert "total_numerator_kwh" in rv
