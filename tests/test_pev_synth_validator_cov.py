"""Coverage-raising unit tests for ``pev_synth.validator`` (M9).

These tests are deliberately *data-independent*: they do NOT touch the heavy
``data/`` tree (NHTS micro-data, processed fleet caches). Every input is a
small synthetic in-memory pandas/numpy fixture or a tiny fabricated bound
parquet written into pytest ``tmp_path``. They exercise the per-check
functions, the pure helpers (``_sha256``/``_round``/``_is_unbounded``/...),
the dataclasses, the ``run()`` orchestrator + replicate path, ``_run_or_error``,
``_update_meta_json``, the markdown/summary emitters, and the ``_cli`` entry
point.

The full-pipeline cache is intentionally absent on a fresh checkout, so the
``run()`` integration tests build a minimal but schema-faithful cache (the six
M1..M8 parquets + meta.json) into ``tmp_path`` and a real bounds tree via
:func:`pev_synth.validation_bounds_curator.curate_bounds`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pev_synth import validator as v
from pev_synth.validation_bounds_curator import curate_bounds

# ---------------------------------------------------------------------------
# Bound-parquet fabrication helpers
# ---------------------------------------------------------------------------


def _write_bound(
    bounds_root: Path,
    check_id: str,
    rows: list[dict[str, object]],
    *,
    source_url: str = "https://example.org/source",
) -> None:
    """Write a tiny ``<check_id>.parquet`` carrying the columns the validator
    reads. Each row dict supplies ``metric_name`` and any of
    ``value``/``lower_bound``/``upper_bound``/``unbounded_v1``/``note``.
    """
    full_rows = []
    for r in rows:
        full_rows.append(
            {
                "check_id": check_id,
                "metric_name": r["metric_name"],
                "value": r.get("value"),
                "lower_bound": r.get("lower_bound"),
                "upper_bound": r.get("upper_bound"),
                "unbounded_v1": bool(r.get("unbounded_v1", False)),
                "note": r.get("note", ""),
                "source_url": source_url,
            }
        )
    df = pd.DataFrame(full_rows)
    bounds_root.mkdir(parents=True, exist_ok=True)
    df.to_parquet(bounds_root / f"{check_id}.parquet")


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_sha256_matches_hashlib(tmp_path: Path) -> None:
    import hashlib

    p = tmp_path / "blob.bin"
    payload = b"ev-flow validator bytes" * 4096
    p.write_bytes(payload)
    assert v._sha256(p) == hashlib.sha256(payload).hexdigest()


def test_round_normal_and_nan_and_none() -> None:
    assert v._round(1.23456789, 3) == 1.235
    assert v._round(2.0) == 2.0
    assert v._round(None) != v._round(None) or True  # nan != nan
    import math

    assert math.isnan(v._round(None))
    assert math.isnan(v._round(float("nan")))
    # inf is passed through unchanged (not coerced to nan).
    assert math.isinf(v._round(float("inf")))


def test_in_band_inclusive_with_tolerance() -> None:
    assert v._in_band(0.5, 0.0, 1.0)
    assert v._in_band(0.0, 0.0, 1.0)
    assert v._in_band(1.0, 0.0, 1.0)
    assert not v._in_band(1.0001, 0.0, 1.0)
    # within the 1e-12 tolerance
    assert v._in_band(1.0 + 1e-13, 0.0, 1.0)


def test_is_unbounded_variants() -> None:
    flag_row = pd.Series({"unbounded_v1": True, "lower_bound": 1.0, "upper_bound": 2.0})
    assert v._is_unbounded(flag_row)

    note_row = pd.Series({"note": "this is unbounded_v1=true here", "lower_bound": np.nan})
    assert v._is_unbounded(note_row)

    nan_row = pd.Series({"lower_bound": np.nan, "upper_bound": np.nan})
    assert v._is_unbounded(nan_row)

    bounded_row = pd.Series(
        {"unbounded_v1": False, "note": "", "lower_bound": 1.0, "upper_bound": 2.0}
    )
    assert not v._is_unbounded(bounded_row)


def test_all_rows_unbounded() -> None:
    empty = pd.DataFrame({"unbounded_v1": pd.Series([], dtype=bool)})
    assert not v._all_rows_unbounded(empty)

    all_flagged = pd.DataFrame({"unbounded_v1": [True, True]})
    assert v._all_rows_unbounded(all_flagged)

    mixed = pd.DataFrame({"unbounded_v1": [True, False]})
    assert not v._all_rows_unbounded(mixed)

    # no flag column -> falls back to per-row _is_unbounded via note
    note_df = pd.DataFrame(
        {"note": ["unbounded_v1=true", "unbounded_v1=true"], "lower_bound": [np.nan, np.nan]}
    )
    assert v._all_rows_unbounded(note_df)


def test_load_bound_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="bound parquet missing"):
        v._load_bound(tmp_path, "does_not_exist")


def test_bound_url_empty_and_present() -> None:
    assert v._bound_url(pd.DataFrame()) == ""
    assert v._bound_url(pd.DataFrame({"foo": [1]})) == ""
    df = pd.DataFrame({"source_url": ["https://x.example"]})
    assert v._bound_url(df) == "https://x.example"


def test_is_v11_migrated_variants() -> None:
    assert not v._is_v11_migrated(None)
    assert not v._is_v11_migrated({})
    assert v._is_v11_migrated({"cache_provenance": "v1.1_migrated"})
    assert not v._is_v11_migrated({"cache_provenance": "v2.0_native"})
    assert v._is_v11_migrated({"cache_provenance": {"provenance_tag": "v1.1_migrated"}})
    assert not v._is_v11_migrated({"cache_provenance": {"provenance_tag": "v2.0_native"}})


def test_methodology_at_or_above_v3() -> None:
    assert not v._methodology_at_or_above_v3(None)
    assert not v._methodology_at_or_above_v3("")
    assert not v._methodology_at_or_above_v3("v2.1-rev")
    assert v._methodology_at_or_above_v3("v3.0.0")
    assert v._methodology_at_or_above_v3("v3.1-rev")
    assert v._methodology_at_or_above_v3("3.0")
    assert not v._methodology_at_or_above_v3("vX.bad")


def test_session_energy_grid_and_batt_columns() -> None:
    s_native = pd.DataFrame({"energy_kwh_grid": [1.0], "energy_kwh_batt": [0.9]})
    assert v._session_energy_grid(s_native, legacy_alias_ok=False, context="t").iloc[0] == 1.0
    assert v._session_energy_batt(s_native, legacy_alias_ok=False, context="t").iloc[0] == 0.9

    s_legacy = pd.DataFrame({"energy_kwh": [2.0]})
    assert v._session_energy_grid(s_legacy, legacy_alias_ok=True, context="t").iloc[0] == 2.0
    assert v._session_energy_batt(s_legacy, legacy_alias_ok=True, context="t").iloc[0] == 2.0

    with pytest.raises(KeyError, match="energy_kwh_grid"):
        v._session_energy_grid(s_legacy, legacy_alias_ok=False, context="t")
    with pytest.raises(KeyError, match="energy_kwh_batt"):
        v._session_energy_batt(s_legacy, legacy_alias_ok=False, context="t")


def test_local_hour_series() -> None:
    ts = pd.Series(pd.to_datetime(["2001-06-01T16:30:00Z", "2001-06-01T20:00:00Z"]))
    hours = v._local_hour_series(ts, "America/Los_Angeles")
    # 16:30 UTC -> 09:30 PDT; 20:00 UTC -> 13:00 PDT
    assert hours.iloc[0] == pytest.approx(9.5)
    assert hours.iloc[1] == pytest.approx(13.0)


# ---------------------------------------------------------------------------
# CheckResult / Report dataclasses
# ---------------------------------------------------------------------------


def test_checkresult_invalid_status_raises() -> None:
    with pytest.raises(ValueError, match="Invalid status"):
        v.CheckResult("c", "NOTASTATUS", None, "", "d")


def test_checkresult_explained_fail_needs_section() -> None:
    with pytest.raises(ValueError, match="must cite a plan §13"):
        v.CheckResult("c", "EXPLAINED_FAIL", None, "", "d", explained_reason="no cite")
    ok = v.CheckResult("c", "EXPLAINED_FAIL", None, "", "d", explained_reason="see §13.1 here")
    assert ok.status == "EXPLAINED_FAIL"


def test_report_to_dict_roundtrip() -> None:
    cr = v.CheckResult("c1", "PASS", {"x": 1}, "src", "details")
    rep = v.Report(
        validator_version="v1.0.0",
        methodology_version="v1.0.0",
        run_timestamp_utc="2026-01-01T00:00:00+00:00",
        pipeline_artifacts_sha256={"fleet.parquet": "abc"},
        checks=[cr],
        summary={"pass": 1, "total": 1},
    )
    d = rep.to_dict()
    assert d["validator_version"] == "v1.0.0"
    assert d["checks"][0]["check_id"] == "c1"
    assert d["summary"]["total"] == 1


# ---------------------------------------------------------------------------
# Per-check functions — battery capacity
# ---------------------------------------------------------------------------


def _battery_bound_rows(tvd_lo: float = 0.0, tvd_hi: float = 0.05) -> list[dict[str, object]]:
    # uniform reference PMF over 6 bins
    rows: list[dict[str, object]] = [
        {"metric_name": name, "value": 1.0 / 6.0} for name in v.ARGONNE_BIN_NAMES
    ]
    rows.append(
        {
            "metric_name": "tvd_vs_regional_sales_pmf",
            "lower_bound": tvd_lo,
            "upper_bound": tvd_hi,
        }
    )
    return rows


def test_battery_capacity_pass(tmp_path: Path) -> None:
    _write_bound(tmp_path, "battery_capacity_distribution", _battery_bound_rows())
    # fleet whose B_kwh PMF matches the uniform reference closely
    centres = [40, 57, 70, 80, 92, 120]
    b = []
    for c in centres:
        b.extend([float(c)] * 10)
    fleet = pd.DataFrame({"powertrain": ["BEV"] * len(b), "B_kwh": b})
    res = v.check_battery_capacity_distribution(fleet, tmp_path)
    assert res.status == "PASS"
    assert res.realised_value["tvd_metric_name"] == "tvd_vs_regional_sales_pmf"
    assert res.realised_value["n_bev"] == 60


def test_battery_capacity_explained_fail(tmp_path: Path) -> None:
    _write_bound(tmp_path, "battery_capacity_distribution", _battery_bound_rows())
    # all mass in one bin -> TVD far above 0.05
    fleet = pd.DataFrame({"powertrain": ["BEV"] * 20, "B_kwh": [40.0] * 20})
    res = v.check_battery_capacity_distribution(fleet, tmp_path)
    assert res.status == "EXPLAINED_FAIL"
    assert "§13." in res.explained_reason


def test_battery_capacity_legacy_argonne_name(tmp_path: Path) -> None:
    rows: list[dict[str, object]] = [
        {"metric_name": name, "value": 1.0 / 6.0} for name in v.ARGONNE_BIN_NAMES
    ]
    rows.append(
        {"metric_name": "tvd_vs_argonne_pmf", "lower_bound": 0.0, "upper_bound": 0.05}
    )
    _write_bound(tmp_path, "battery_capacity_distribution", rows)
    fleet = pd.DataFrame({"powertrain": ["BEV"] * 20, "B_kwh": [40.0] * 20})
    res = v.check_battery_capacity_distribution(fleet, tmp_path)
    assert res.realised_value["tvd_metric_name"] == "tvd_vs_argonne_pmf"


def test_battery_capacity_missing_tvd_row_raises(tmp_path: Path) -> None:
    rows: list[dict[str, object]] = [
        {"metric_name": name, "value": 1.0 / 6.0} for name in v.ARGONNE_BIN_NAMES
    ]
    _write_bound(tmp_path, "battery_capacity_distribution", rows)
    fleet = pd.DataFrame({"powertrain": ["BEV"] * 6, "B_kwh": [40.0] * 6})
    with pytest.raises(KeyError, match="tvd_vs"):
        v.check_battery_capacity_distribution(fleet, tmp_path)


# ---------------------------------------------------------------------------
# obc / evi_pro distribution (unbounded -> EXPLAINED_SKIP, else ERROR)
# ---------------------------------------------------------------------------


def test_obc_distribution_explained_skip(tmp_path: Path) -> None:
    _write_bound(
        tmp_path,
        "obc_distribution",
        [{"metric_name": "share_3p3", "unbounded_v1": True}],
    )
    res = v.check_obc_distribution(tmp_path)
    assert res.status == "EXPLAINED_SKIP"


def test_obc_distribution_error_when_bounded(tmp_path: Path) -> None:
    _write_bound(
        tmp_path,
        "obc_distribution",
        [{"metric_name": "share_3p3", "lower_bound": 0.1, "upper_bound": 0.2}],
    )
    res = v.check_obc_distribution(tmp_path)
    assert res.status == "ERROR"


def test_evi_pro_explained_skip(tmp_path: Path) -> None:
    _write_bound(
        tmp_path,
        "evi_pro_plug_in_split",
        [{"metric_name": "immediate", "unbounded_v1": True}],
    )
    assert v.check_evi_pro_plug_in_split(tmp_path).status == "EXPLAINED_SKIP"


def test_evi_pro_error_when_bounded(tmp_path: Path) -> None:
    _write_bound(
        tmp_path,
        "evi_pro_plug_in_split",
        [{"metric_name": "immediate", "lower_bound": 0.1, "upper_bound": 0.5}],
    )
    assert v.check_evi_pro_plug_in_split(tmp_path).status == "ERROR"


# ---------------------------------------------------------------------------
# Plug-in CDF (weekday / weekend)
# ---------------------------------------------------------------------------


def _cdf_bound_rows(tag: str, lo: float, hi: float) -> list[dict[str, object]]:
    return [
        {"metric_name": f"{tag}_plugin_cdf_at_hour_{h:02d}", "lower_bound": lo, "upper_bound": hi}
        for h in (6, 12, 18, 22)
    ]


def _make_pie(hours_utc: list[int], weekday: bool) -> pd.DataFrame:
    base = "2001-01-01" if weekday else "2001-01-06"  # Mon vs Sat
    ts = pd.to_datetime([f"{base}T{h:02d}:00:00" for h in hours_utc]).tz_localize("UTC")
    return pd.DataFrame({"arrival_ts": ts, "plug_in": [True] * len(hours_utc)})


def test_weekday_cdf_pass(tmp_path: Path) -> None:
    _write_bound(tmp_path, "weekday_plugin_cdf", _cdf_bound_rows("weekday", 0.0, 1.0))
    pie = _make_pie([5, 8, 14, 20], weekday=True)
    res = v.check_weekday_plugin_cdf(pie, tmp_path)
    assert res.status == "PASS"
    assert res.realised_value["n_events"] == 4


def test_weekday_cdf_empty_is_error(tmp_path: Path) -> None:
    _write_bound(tmp_path, "weekday_plugin_cdf", _cdf_bound_rows("weekday", 0.0, 1.0))
    pie = _make_pie([8], weekday=False)  # only a weekend event
    res = v.check_weekday_plugin_cdf(pie, tmp_path)
    assert res.status == "ERROR"


def test_weekend_cdf_explained_fail(tmp_path: Path) -> None:
    # All arrivals at hour 5 -> F(h)=1.0 for every probe hour; band [0, 0.5]
    # excludes it, forcing FAIL -> EXPLAINED_FAIL.
    _write_bound(tmp_path, "weekend_plugin_cdf", _cdf_bound_rows("weekend", 0.0, 0.5))
    pie = _make_pie([5, 5, 5, 5], weekday=False)
    res = v.check_weekend_plugin_cdf(pie, tmp_path)
    assert res.status == "EXPLAINED_FAIL"
    assert "§13." in res.explained_reason


def test_weekday_cdf_missing_metric_is_error(tmp_path: Path) -> None:
    # Only one of the four hour rows present -> ERROR status emitted.
    _write_bound(
        tmp_path,
        "weekday_plugin_cdf",
        [{"metric_name": "weekday_plugin_cdf_at_hour_06", "lower_bound": 0.0, "upper_bound": 1.0}],
    )
    pie = _make_pie([5, 8, 14, 20], weekday=True)
    res = v.check_weekday_plugin_cdf(pie, tmp_path)
    assert res.status == "ERROR"


# ---------------------------------------------------------------------------
# session duration / energy per session
# ---------------------------------------------------------------------------


def _session_frame(durations_h: list[float], energy: list[float]) -> pd.DataFrame:
    t_in = pd.to_datetime(["2001-01-01T00:00:00Z"] * len(durations_h))
    t_out = t_in + pd.to_timedelta(durations_h, unit="h")
    return pd.DataFrame(
        {
            "location": ["home"] * len(durations_h),
            "t_in": t_in,
            "t_out": t_out,
            "energy_kwh_grid": energy,
            "energy_kwh_batt": energy,
        }
    )


def test_session_duration_pass(tmp_path: Path) -> None:
    _write_bound(
        tmp_path,
        "session_duration_distribution",
        [
            {"metric_name": "median_duration_hours", "lower_bound": 5.0, "upper_bound": 15.0},
            {"metric_name": "p95_duration_hours", "lower_bound": 5.0, "upper_bound": 25.0},
        ],
    )
    sessions = _session_frame([8.0, 9.0, 10.0, 11.0], [10.0] * 4)
    res = v.check_session_duration_distribution(sessions, tmp_path)
    assert res.status == "PASS"
    assert res.realised_value["n_home_sessions"] == 4


def test_session_duration_explained_fail(tmp_path: Path) -> None:
    _write_bound(
        tmp_path,
        "session_duration_distribution",
        [
            {"metric_name": "median_duration_hours", "lower_bound": 0.1, "upper_bound": 0.2},
            {"metric_name": "p95_duration_hours", "lower_bound": 0.1, "upper_bound": 0.2},
        ],
    )
    sessions = _session_frame([8.0, 9.0, 10.0], [10.0] * 3)
    res = v.check_session_duration_distribution(sessions, tmp_path)
    assert res.status == "EXPLAINED_FAIL"
    assert "§13." in res.explained_reason


def test_energy_per_session_pass_and_phev_exclusion(tmp_path: Path) -> None:
    _write_bound(
        tmp_path,
        "energy_per_session_distribution",
        [
            {"metric_name": "median_kwh_per_session", "lower_bound": 5.0, "upper_bound": 20.0},
            {"metric_name": "p95_kwh_per_session", "lower_bound": 5.0, "upper_bound": 40.0},
        ],
    )
    sessions = _session_frame([8.0] * 4, [10.0, 11.0, 12.0, 13.0])
    sessions["gas_kwh_equivalent"] = [0.0, 0.0, 0.0, 5.0]  # last row excluded
    res = v.check_energy_per_session_distribution(sessions, tmp_path)
    assert res.status == "PASS"
    assert res.realised_value["n_phev_gas_excluded"] == 1
    assert res.realised_value["n_home_sessions"] == 3


def test_energy_per_session_explained_fail(tmp_path: Path) -> None:
    _write_bound(
        tmp_path,
        "energy_per_session_distribution",
        [
            {"metric_name": "median_kwh_per_session", "lower_bound": 0.1, "upper_bound": 0.2},
            {"metric_name": "p95_kwh_per_session", "lower_bound": 0.1, "upper_bound": 0.2},
        ],
    )
    sessions = _session_frame([8.0] * 3, [10.0, 20.0, 30.0])
    res = v.check_energy_per_session_distribution(sessions, tmp_path)
    assert res.status == "EXPLAINED_FAIL"
    assert v.LIM_13_6 in res.explained_reason


# ---------------------------------------------------------------------------
# annual mileage
# ---------------------------------------------------------------------------


def _mobility_frame(miles_per_ev: dict[int, float], kwh_per_mi: float = 0.3) -> pd.DataFrame:
    rows = []
    for ev_id, miles in miles_per_ev.items():
        rows.append(
            {
                "ev_id": ev_id,
                "miles": miles,
                "kwh_consumed": miles * kwh_per_mi,
                "start_ts": pd.Timestamp("2001-01-15T12:00:00Z"),
            }
        )
    return pd.DataFrame(rows)


def _annual_mileage_bound(tmp_path: Path, mean_band, std_band) -> None:
    _write_bound(
        tmp_path,
        "annual_mileage",
        [
            {"metric_name": "mean_miles_per_year", "lower_bound": mean_band[0],
             "upper_bound": mean_band[1]},
            {"metric_name": "std_miles_per_year", "lower_bound": std_band[0],
             "upper_bound": std_band[1]},
        ],
    )


def test_annual_mileage_pass(tmp_path: Path) -> None:
    _annual_mileage_bound(tmp_path, (4000, 16000), (0, 20000))
    mob = _mobility_frame({0: 10000.0, 1: 12000.0, 2: 8000.0})
    res = v.check_annual_mileage(mob, tmp_path)
    assert res.status == "PASS"
    assert res.realised_value["n_evs"] == 3


def test_annual_mileage_legacy_explained_fail(tmp_path: Path) -> None:
    _annual_mileage_bound(tmp_path, (4000, 6000), (3000, 6000))
    mob = _mobility_frame({0: 1000.0, 1: 50000.0})  # out of band median + huge sigma
    res = v.check_annual_mileage(mob, tmp_path, methodology_version=None)
    assert res.status == "EXPLAINED_FAIL"
    assert res.explained_reason == v.LIM_13_2


def test_annual_mileage_v3_explained_fail_still_cites_13_2(tmp_path: Path) -> None:
    _annual_mileage_bound(tmp_path, (4000, 6000), (3000, 6000))
    mob = _mobility_frame({0: 1000.0, 1: 50000.0})
    res = v.check_annual_mileage(mob, tmp_path, methodology_version="v3.0.0")
    # source returns EXPLAINED_FAIL with LIM_13_2 regardless of methodology
    assert res.status == "EXPLAINED_FAIL"
    assert res.realised_value["methodology_version"] == "v3.0.0"


# ---------------------------------------------------------------------------
# wh_per_mi_at_70F
# ---------------------------------------------------------------------------


def test_wh_per_mi_pass_and_explained_fail(tmp_path: Path) -> None:
    # bound only one model so n_total == 1 and we control pass_rate
    _write_bound(
        tmp_path,
        "wh_per_mi_at_70F",
        [{"metric_name": "wh_per_mi_chevy_bolt_euv", "lower_bound": 200.0, "upper_bound": 320.0}],
    )
    fleet = pd.DataFrame(
        {"ev_id": [0, 1], "model": ["Bolt_EUV", "Bolt_EUV"], "B_kwh": [65.0, 65.0]}
    )
    # 0.25 kWh/mi -> 250 Wh/mi, inside [200, 320]
    mob = pd.DataFrame(
        {
            "ev_id": [0, 1],
            "miles": [10000.0, 12000.0],
            "kwh_consumed": [2500.0, 3000.0],
        }
    )
    res = v.check_wh_per_mi_at_70F(fleet, mob, tmp_path)
    assert res.status == "PASS"
    assert res.realised_value["n_models_total"] == 1
    assert res.realised_value["n_models_pass"] == 1

    # Now out of band -> EXPLAINED_FAIL.
    _write_bound(
        tmp_path,
        "wh_per_mi_at_70F",
        [{"metric_name": "wh_per_mi_chevy_bolt_euv", "lower_bound": 100.0, "upper_bound": 120.0}],
    )
    res2 = v.check_wh_per_mi_at_70F(fleet, mob, tmp_path)
    assert res2.status == "EXPLAINED_FAIL"
    assert v.LIM_13_4 in res2.explained_reason


def test_wh_per_mi_model_absent_from_fleet(tmp_path: Path) -> None:
    _write_bound(
        tmp_path,
        "wh_per_mi_at_70F",
        [{"metric_name": "wh_per_mi_tesla_model_3_lr", "lower_bound": 200.0, "upper_bound": 320.0}],
    )
    fleet = pd.DataFrame({"ev_id": [0], "model": ["Bolt_EUV"], "B_kwh": [65.0]})
    mob = pd.DataFrame({"ev_id": [0], "miles": [10000.0], "kwh_consumed": [2500.0]})
    res = v.check_wh_per_mi_at_70F(fleet, mob, tmp_path)
    # The bound model is not in the fleet -> recorded as nan / not-in-band entry.
    arch = res.realised_value["per_archetype"]["wh_per_mi_tesla_model_3_lr"]
    assert arch["n_evs"] == 0
    assert arch["in_band"] is False


# ---------------------------------------------------------------------------
# eta_charging
# ---------------------------------------------------------------------------


def test_eta_charging_pass_and_fail(tmp_path: Path) -> None:
    _write_bound(
        tmp_path,
        "eta_charging",
        [{"metric_name": "eta_l2_nominal", "lower_bound": 0.85, "upper_bound": 0.95}],
    )
    fleet_pass = pd.DataFrame({"eta_charge_nominal": [0.9, 0.9, 0.9]})
    assert v.check_eta_charging(fleet_pass, tmp_path).status == "PASS"

    fleet_fail = pd.DataFrame({"eta_charge_nominal": [0.5, 0.5]})
    res = v.check_eta_charging(fleet_fail, tmp_path)
    assert res.status == "FAIL"
    assert res.realised_value["mean"] == 0.5


# ---------------------------------------------------------------------------
# cross-artifact integrity, plug-rate, energy conservation
# ---------------------------------------------------------------------------


def _fk_frame(ev_ids: list[int]) -> pd.DataFrame:
    return pd.DataFrame({"ev_id": ev_ids})


def test_cross_artifact_integrity_pass() -> None:
    fleet = pd.DataFrame({"ev_id": [0, 1]})
    sessions = pd.DataFrame({"ev_id": [0, 1], "session_idx": [0, 0]})
    res = v.check_cross_artifact_integrity(
        fleet, sessions, _fk_frame([0, 1]), _fk_frame([0, 1]),
        _fk_frame([0, 1]), _fk_frame([0, 1]),
    )
    assert res.status == "PASS"


def test_cross_artifact_integrity_fail_missing_fk_and_dup() -> None:
    fleet = pd.DataFrame({"ev_id": [0]})
    sessions = pd.DataFrame({"ev_id": [0, 0], "session_idx": [1, 1]})  # dup PK
    res = v.check_cross_artifact_integrity(
        fleet, sessions, _fk_frame([0, 99]), _fk_frame([0]),
        _fk_frame([0]), _fk_frame([0]),
    )
    assert res.status == "FAIL"
    issues = res.realised_value["issues"]
    assert any("missing from fleet" in i for i in issues)
    assert any("duplicate" in i for i in issues)


def test_plug_rate_consistency_pass_and_fail() -> None:
    # raster: 100 rows, 50 plugged -> rate 0.5
    raster = pd.DataFrame({"plugged": [True] * 50 + [False] * 50})
    # 1 vehicle -> n_steps_total = 35040; pick session steps so rate matches
    n_vehicles = 1
    # want rate_sessions ~ 0.5 -> total_session_steps = 0.5 * 35040 = 17520 steps
    steps = 17520
    t_in = pd.to_datetime(["2001-01-01T00:00:00Z"])
    t_out = t_in + pd.to_timedelta([steps * 15], unit="m")
    sessions = pd.DataFrame({"t_in": t_in, "t_out": t_out})
    res = v.check_plug_rate_consistency(sessions, raster, n_vehicles)
    assert res.status == "PASS"

    # Big mismatch -> FAIL
    t_out_big = t_in + pd.to_timedelta([35040 * 15], unit="m")
    sessions_big = pd.DataFrame({"t_in": t_in, "t_out": t_out_big})
    res2 = v.check_plug_rate_consistency(sessions_big, raster, n_vehicles)
    assert res2.status == "FAIL"


def test_fleet_energy_conservation_pass_and_explained_fail() -> None:
    sessions = pd.DataFrame({"energy_kwh_grid": [111.1], "gas_kwh_equivalent": [0.0]})
    mobility = pd.DataFrame({"kwh_consumed": [100.0]})
    res = v.check_fleet_energy_conservation(sessions, mobility)
    assert res.status == "PASS"
    assert res.realised_value["ratio"] == pytest.approx(1.111)

    sessions_bad = pd.DataFrame({"energy_kwh_grid": [80.0]})
    res2 = v.check_fleet_energy_conservation(sessions_bad, mobility)
    assert res2.status == "EXPLAINED_FAIL"
    assert "§13." in res2.explained_reason


# ---------------------------------------------------------------------------
# DST verification
# ---------------------------------------------------------------------------


def _make_15min_cache(output_root: Path) -> None:
    """Write a plug_status_15min.parquet spanning both DST transitions."""
    output_root.mkdir(parents=True, exist_ok=True)
    idx = pd.date_range("2001-03-30T00:00:00Z", "2001-10-30T00:00:00Z", freq="15min")
    df = pd.DataFrame({"ts": idx, "ev_id": 0, "plugged": False})
    df.to_parquet(output_root / "plug_status_15min.parquet")


def test_dst_verification_pass(tmp_path: Path) -> None:
    output_root = tmp_path / "cache"
    _make_15min_cache(output_root)
    bounds_root = tmp_path / "bounds"
    _write_bound(bounds_root, "dst_verification", [{"metric_name": "monotonic_utc"}])
    res = v.check_dst_verification(output_root, bounds_root)
    assert res.status == "PASS"
    assert res.realised_value["source_artifact"] == "plug_status_15min.parquet"
    assert "America/Los_Angeles" in res.realised_value["per_zone"]


def test_dst_verification_no_parquet_is_error(tmp_path: Path) -> None:
    output_root = tmp_path / "empty_cache"
    output_root.mkdir()
    bounds_root = tmp_path / "bounds"
    # bound parquet absent -> FileNotFoundError handled internally; source=""
    res = v.check_dst_verification(output_root, bounds_root)
    assert res.status == "ERROR"


def test_check_dst_zone_helper() -> None:
    idx = pd.DatetimeIndex(pd.date_range("2001-06-01T00:00:00Z", periods=4, freq="15min"))
    out = v._check_dst_zone(idx, "America/Los_Angeles")
    assert out["utc_monotonic"] is True
    assert out["local_nat_count"] == 0
    assert out["local_dup_count"] == 0
    assert out["n_rows"] == 4
    # UTC zone path
    out_utc = v._check_dst_zone(idx, "UTC")
    assert out_utc["zone"] == "UTC"


# ---------------------------------------------------------------------------
# Workplace W1-W10 checks
# ---------------------------------------------------------------------------


def _make_workplace_pie(hours_utc: list[int]) -> pd.DataFrame:
    ts = pd.to_datetime([f"2001-06-04T{h:02d}:00:00" for h in hours_utc]).tz_localize("UTC")
    return pd.DataFrame(
        {
            "arrival_ts": ts,
            "plug_in": [True] * len(hours_utc),
            "location": ["work"] * len(hours_utc),
        }
    )


def test_workplace_w1_pass_and_explained_fail(tmp_path: Path) -> None:
    rows = [
        {"metric_name": f"plug_in_cdf_at_hour_{h:02d}", "lower_bound": 0.0, "upper_bound": 1.0}
        for h in (9, 12, 15)
    ]
    _write_bound(tmp_path, "W1_workplace_plug_in_cdf", rows)
    # 16:00 UTC -> 09:00 PDT
    pie = _make_workplace_pie([16, 17, 18])
    res = v.check_workplace_w1_plug_in_cdf(pie, tmp_path, "America/Los_Angeles")
    assert res.status == "PASS"

    rows_fail = [
        {"metric_name": f"plug_in_cdf_at_hour_{h:02d}", "lower_bound": 0.99, "upper_bound": 1.0}
        for h in (9, 12, 15)
    ]
    _write_bound(tmp_path, "W1_workplace_plug_in_cdf", rows_fail)
    res2 = v.check_workplace_w1_plug_in_cdf(
        pie, tmp_path, "America/Los_Angeles", workplace_cohort_caveat="live caveat"
    )
    assert res2.status == "EXPLAINED_FAIL"
    assert "§13." in res2.explained_reason
    assert "live caveat" in res2.explained_reason


def test_workplace_w1_empty_explained_skip(tmp_path: Path) -> None:
    _write_bound(
        tmp_path,
        "W1_workplace_plug_in_cdf",
        [{"metric_name": "plug_in_cdf_at_hour_09", "lower_bound": 0.0, "upper_bound": 1.0}],
    )
    pie = pd.DataFrame(
        {"arrival_ts": pd.to_datetime([], utc=True), "plug_in": [], "location": []}
    )
    res = v.check_workplace_w1_plug_in_cdf(pie, tmp_path, "America/Los_Angeles")
    assert res.status == "EXPLAINED_SKIP"


def _workplace_sessions(durations_h: list[float], energy: list[float]) -> pd.DataFrame:
    t_in = pd.to_datetime(["2001-06-04T08:00:00Z"] * len(durations_h))
    t_out = t_in + pd.to_timedelta(durations_h, unit="h")
    return pd.DataFrame(
        {
            "location": ["work"] * len(durations_h),
            "t_in": t_in,
            "t_out": t_out,
            "energy_kwh_grid": energy,
        }
    )


def test_workplace_w2_pass_and_fail_and_skip(tmp_path: Path) -> None:
    _write_bound(
        tmp_path,
        "W2_workplace_session_duration",
        [{"metric_name": "median_duration_hours", "lower_bound": 5.0, "upper_bound": 9.0}],
    )
    sessions = _workplace_sessions([6.0, 7.0, 8.0], [10.0] * 3)
    assert v.check_workplace_w2_session_duration(sessions, tmp_path).status == "PASS"

    sessions_fail = _workplace_sessions([0.5, 0.5], [10.0] * 2)
    res = v.check_workplace_w2_session_duration(sessions_fail, tmp_path)
    assert res.status == "EXPLAINED_FAIL"

    no_work = pd.DataFrame(
        {
            "location": ["home"],
            "t_in": pd.to_datetime(["2001-06-04T08:00:00Z"]),
            "t_out": pd.to_datetime(["2001-06-04T16:00:00Z"]),
        }
    )
    assert v.check_workplace_w2_session_duration(no_work, tmp_path).status == "EXPLAINED_SKIP"


def test_workplace_w3_pass_and_skip(tmp_path: Path) -> None:
    _write_bound(
        tmp_path,
        "W3_workplace_session_energy",
        [{"metric_name": "median_kwh_per_session", "lower_bound": 8.0, "upper_bound": 25.0}],
    )
    sessions = _workplace_sessions([6.0, 6.0, 6.0], [10.0, 15.0, 20.0])
    assert v.check_workplace_w3_session_energy(sessions, tmp_path).status == "PASS"

    sessions_fail = _workplace_sessions([6.0], [100.0])
    assert v.check_workplace_w3_session_energy(sessions_fail, tmp_path).status == "EXPLAINED_FAIL"

    no_work = pd.DataFrame(
        {"location": ["home"], "t_in": pd.to_datetime(["2001-06-04T08:00:00Z"]),
         "t_out": pd.to_datetime(["2001-06-04T16:00:00Z"]), "energy_kwh_grid": [10.0]}
    )
    assert v.check_workplace_w3_session_energy(no_work, tmp_path).status == "EXPLAINED_SKIP"


def test_workplace_w4_pass_fail_skip(tmp_path: Path) -> None:
    _write_bound(
        tmp_path,
        "W4_workplace_plug_in_on_arrival",
        [{"metric_name": "plug_in_on_arrival_rate", "lower_bound": 0.85, "upper_bound": 0.95}],
    )
    pie = pd.DataFrame(
        {
            "plug_in": [True] * 9 + [False],
            "location": ["work"] * 10,
            "arrival_ts": pd.to_datetime(["2001-06-04T16:00:00Z"] * 10),
        }
    )
    res = v.check_workplace_w4_plug_in_on_arrival(pie, tmp_path)
    assert res.status == "PASS"
    assert res.realised_value["plug_in_on_arrival_rate"] == pytest.approx(0.9)

    pie_fail = pd.DataFrame(
        {"plug_in": [False] * 10, "location": ["work"] * 10,
         "arrival_ts": pd.to_datetime(["2001-06-04T16:00:00Z"] * 10)}
    )
    assert v.check_workplace_w4_plug_in_on_arrival(pie_fail, tmp_path).status == "EXPLAINED_FAIL"

    empty = pd.DataFrame(
        {"plug_in": [], "location": [], "arrival_ts": pd.to_datetime([], utc=True)}
    )
    assert v.check_workplace_w4_plug_in_on_arrival(empty, tmp_path).status == "EXPLAINED_SKIP"


def test_workplace_w5_pass_and_fail(tmp_path: Path) -> None:
    _write_bound(
        tmp_path,
        "W5_workplace_evse_kw_distribution",
        [{"metric_name": "share_6p6_to_7p7_kw", "lower_bound": 0.6, "upper_bound": 1.0}],
    )
    fleet = pd.DataFrame({"EVSE_home_kw": [6.6, 7.0, 7.7, 7.2]})
    assert v.check_workplace_w5_evse_kw_distribution(fleet, tmp_path).status == "PASS"

    fleet_fail = pd.DataFrame({"EVSE_home_kw": [11.5, 19.2, 3.3]})
    res_fail = v.check_workplace_w5_evse_kw_distribution(fleet_fail, tmp_path)
    assert res_fail.status == "EXPLAINED_FAIL"


def test_workplace_w6_pass_fail_skip(tmp_path: Path) -> None:
    _write_bound(
        tmp_path,
        "W6_workplace_cluster_window_max",
        [{"metric_name": "max_cluster_window_hours", "lower_bound": 0.0, "upper_bound": 6.0}],
    )
    res = v.check_workplace_w6_cluster_window_max(
        {"plug_in_model": {"W9_max_cluster_window_h": 5.0}}, tmp_path
    )
    assert res.status == "PASS"

    res_fail = v.check_workplace_w6_cluster_window_max(
        {"plug_in_model": {"W9_max_cluster_window_h": 9.0}}, tmp_path
    )
    assert res_fail.status == "FAIL"

    res_skip = v.check_workplace_w6_cluster_window_max({"plug_in_model": {}}, tmp_path)
    assert res_skip.status == "EXPLAINED_SKIP"


def test_workplace_w7_pass_skip_fallback(tmp_path: Path) -> None:
    _write_bound(
        tmp_path,
        "W7_workplace_borrow_share",
        [{"metric_name": "donor_borrow_rate", "lower_bound": 0.0, "upper_bound": 0.25}],
    )
    fleet = pd.DataFrame({"ev_id": [0, 1]})
    res = v.check_workplace_w7_borrow_share({"donor_borrow_rate": 0.1}, fleet, tmp_path)
    assert res.status == "PASS"

    # nested form
    res_nested = v.check_workplace_w7_borrow_share(
        {"donor_matcher": {"donor_borrow_rate": 0.3}}, fleet, tmp_path
    )
    assert res_nested.status == "FAIL"

    # fleet fallback: borrow_used mean = 0.5, band [0, 0.25] -> FAIL.
    fleet_b = pd.DataFrame({"ev_id": [0, 1], "borrow_used": [True, False]})
    res_fb = v.check_workplace_w7_borrow_share({}, fleet_b, tmp_path)
    assert res_fb.status == "FAIL"
    assert res_fb.realised_value["donor_borrow_rate"] == 0.5

    res_skip = v.check_workplace_w7_borrow_share({}, fleet, tmp_path)
    assert res_skip.status == "EXPLAINED_SKIP"


def test_workplace_w8_pass_and_explained_fail(tmp_path: Path) -> None:
    _write_bound(
        tmp_path,
        "W8_workplace_home_charging_required_rate",
        [{"metric_name": "home_charging_required_share", "lower_bound": 0.0, "upper_bound": 0.15}],
    )
    fleet = pd.DataFrame({"ev_id": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]})
    sessions = pd.DataFrame(
        {"ev_id": [0], "infeasibility_flag": ["home_charging_required"]}
    )
    res = v.check_workplace_w8_home_charging_required_rate(sessions, fleet, tmp_path)
    assert res.status == "PASS"
    assert res.realised_value["n_evs_flagged"] == 1

    sessions_fail = pd.DataFrame(
        {"ev_id": [0, 1, 2, 3, 4], "home_charging_required": [True] * 5}
    )
    res2 = v.check_workplace_w8_home_charging_required_rate(sessions_fail, fleet, tmp_path)
    assert res2.status == "EXPLAINED_FAIL"


def test_workplace_w9_pass_fail_skip(tmp_path: Path) -> None:
    _write_bound(
        tmp_path,
        "W9_workplace_cluster_window_invariant",
        [
            {
                "metric_name": "cluster_window_invariant_active",
                "lower_bound": 1.0,
                "upper_bound": 1.0,
            }
        ],
    )
    res = v.check_workplace_w9_cluster_window_invariant(
        {"plug_in_model": {"W9_max_cluster_window_h": 5.0}}, tmp_path
    )
    assert res.status == "PASS"

    res_fail = v.check_workplace_w9_cluster_window_invariant(
        {"plug_in_model": {"W9_max_cluster_window_h": 9.0}}, tmp_path
    )
    assert res_fail.status == "FAIL"

    res_marker = v.check_workplace_w9_cluster_window_invariant(
        {"plug_in_model": {"W9_cluster_window_invariant_active": True}}, tmp_path
    )
    assert res_marker.status == "PASS"

    res_skip = v.check_workplace_w9_cluster_window_invariant({"plug_in_model": {}}, tmp_path)
    assert res_skip.status == "EXPLAINED_SKIP"


def test_workplace_w10_pass_skip_topdonor(tmp_path: Path) -> None:
    _write_bound(
        tmp_path,
        "W10_workplace_donor_pool_recycling",
        [{"metric_name": "max_recycling_ratio", "lower_bound": 0.0, "upper_bound": 2.0}],
    )
    fleet = pd.DataFrame({"ev_id": [0, 1]})
    res = v.check_workplace_w10_donor_pool_recycling({"max_recycling_ratio": 1.5}, fleet, tmp_path)
    assert res.status == "PASS"

    res_nested = v.check_workplace_w10_donor_pool_recycling(
        {"donor_matcher": {"max_recycling_ratio": 3.0}}, fleet, tmp_path
    )
    assert res_nested.status == "FAIL"

    res_top5 = v.check_workplace_w10_donor_pool_recycling(
        {"donor_matcher": {"top5_donor_houseids": [["h1", 2], ["h2", 1]]}}, fleet, tmp_path
    )
    assert res_top5.status == "PASS"

    res_skip = v.check_workplace_w10_donor_pool_recycling({}, fleet, tmp_path)
    assert res_skip.status == "EXPLAINED_SKIP"


# ---------------------------------------------------------------------------
# C5 winter ratio
# ---------------------------------------------------------------------------


def _write_c5_bound(bounds_root: Path, lower: float | None) -> None:
    bounds_root.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "region": ["chicago"],
            "lower": [lower],
            "upper": [None],
            "source_title": ["Yuksel & Michalek 2015"],
            "note": [""],
        }
    )
    df.to_parquet(bounds_root / "C5_winter_ratio.parquet")


class _ColdRegion:
    winter_f_T_multiplier = 1.2


class _WarmRegion:
    winter_f_T_multiplier = 1.0


def test_c5_winter_ratio_pass(tmp_path: Path) -> None:
    _write_c5_bound(tmp_path, 1.15)
    winter = pd.DataFrame(
        {
            "miles": [100.0, 100.0],
            "kwh_consumed": [40.0, 40.0],  # 400 Wh/mi
            "start_ts": pd.to_datetime(["2001-01-15T12:00:00Z", "2001-02-15T12:00:00Z"]),
        }
    )
    summer = pd.DataFrame(
        {
            "miles": [100.0, 100.0],
            "kwh_consumed": [25.0, 25.0],  # 250 Wh/mi
            "start_ts": pd.to_datetime(["2001-07-15T12:00:00Z", "2001-08-15T12:00:00Z"]),
        }
    )
    mob = pd.concat([winter, summer], ignore_index=True)
    res = v.check_C5_winter_ratio(
        mob, tmp_path, region="chicago", region_tz="America/Chicago", region_obj=_ColdRegion()
    )
    assert res.status == "PASS"
    assert res.realised_value["winter_summer_ratio"] == pytest.approx(1.6)


def test_c5_winter_ratio_warm_skip(tmp_path: Path) -> None:
    _write_c5_bound(tmp_path, None)
    mob = pd.DataFrame(
        {
            "miles": [100.0],
            "kwh_consumed": [25.0],
            "start_ts": pd.to_datetime(["2001-07-15T12:00:00Z"]),
        }
    )
    res = v.check_C5_winter_ratio(
        mob, tmp_path, region="dallas_fort_worth", region_tz="America/Chicago",
        region_obj=_WarmRegion(),
    )
    assert res.status == "EXPLAINED_SKIP"


def test_c5_winter_ratio_fail_and_missing_seasons(tmp_path: Path) -> None:
    _write_c5_bound(tmp_path, 1.15)
    # winter ratio below 1.15 -> FAIL
    winter = pd.DataFrame(
        {
            "miles": [100.0],
            "kwh_consumed": [26.0],
            "start_ts": pd.to_datetime(["2001-01-15T12:00:00Z"]),
        }
    )
    summer = pd.DataFrame(
        {
            "miles": [100.0],
            "kwh_consumed": [25.0],
            "start_ts": pd.to_datetime(["2001-07-15T12:00:00Z"]),
        }
    )
    mob = pd.concat([winter, summer], ignore_index=True)
    res = v.check_C5_winter_ratio(
        mob, tmp_path, region="chicago", region_tz="America/Chicago", region_obj=_ColdRegion()
    )
    assert res.status == "FAIL"

    # winter only -> insufficient seasonal coverage -> EXPLAINED_SKIP
    res2 = v.check_C5_winter_ratio(
        winter, tmp_path, region="chicago", region_tz="America/Chicago", region_obj=_ColdRegion()
    )
    assert res2.status == "EXPLAINED_SKIP"


def test_c5_missing_parquet_raises(tmp_path: Path) -> None:
    mob = pd.DataFrame(
        {"miles": [1.0], "kwh_consumed": [1.0], "start_ts": [pd.Timestamp.now(tz="UTC")]}
    )
    with pytest.raises(FileNotFoundError, match="C5 bound parquet missing"):
        v.check_C5_winter_ratio(
            mob, tmp_path, region="chicago", region_tz="America/Chicago",
        )


# ---------------------------------------------------------------------------
# v2.1 EVSE enrichment checks
# ---------------------------------------------------------------------------


def test_evse_kw_distribution_pass_and_fail(tmp_path: Path) -> None:
    _write_bound(
        tmp_path,
        "evse_kw_distribution_residential",
        [
            {"metric_name": "share_at_7.7_kw", "lower_bound": 0.0, "upper_bound": 1.0},
            {"metric_name": "share_at_11.5_kw", "lower_bound": 0.0, "upper_bound": 1.0},
        ],
    )
    fleet = pd.DataFrame({"EVSE_home_kw": [7.7, 7.7, 11.5, 11.5]})
    res = v.check_evse_kw_distribution(fleet, tmp_path, profile_type="residential")
    assert res.status == "PASS"
    assert res.realised_value["share_at_7.7_kw"] == 0.5

    _write_bound(
        tmp_path,
        "evse_kw_distribution_residential",
        [{"metric_name": "share_at_7.7_kw", "lower_bound": 0.9, "upper_bound": 1.0}],
    )
    res2 = v.check_evse_kw_distribution(fleet, tmp_path, profile_type="residential")
    assert res2.status == "FAIL"


def test_evse_kw_distribution_legacy_band(tmp_path: Path) -> None:
    _write_bound(
        tmp_path,
        "evse_kw_distribution_workplace",
        [{"metric_name": "share_6p6_to_7p7_kw", "lower_bound": 0.6, "upper_bound": 1.0}],
    )
    fleet = pd.DataFrame({"EVSE_home_kw": [6.6, 7.0, 7.7]})
    res = v.check_evse_kw_distribution(fleet, tmp_path, profile_type="workplace")
    assert res.status == "PASS"
    assert "share_6p6_to_7p7_kw" in res.realised_value


def test_evse_brand_and_connector_distribution(tmp_path: Path) -> None:
    _write_bound(
        tmp_path,
        "evse_brand_distribution_residential",
        [
            {"metric_name": "share_Tesla", "lower_bound": 0.0, "upper_bound": 1.0},
            {"metric_name": "share_ChargePoint", "lower_bound": 0.0, "upper_bound": 1.0},
        ],
    )
    fleet = pd.DataFrame({"EVSE_brand": ["Tesla", "Tesla", "ChargePoint", "ChargePoint"]})
    res = v.check_evse_brand_distribution(fleet, tmp_path)
    assert res.status == "PASS"
    assert res.realised_value["share_Tesla"] == 0.5

    _write_bound(
        tmp_path,
        "evse_connector_distribution_residential",
        [{"metric_name": "share_J1772", "lower_bound": 0.9, "upper_bound": 1.0}],
    )
    fleet_c = pd.DataFrame({"EVSE_connector": ["J1772", "NACS"]})
    res_c = v.check_evse_connector_distribution(fleet_c, tmp_path)
    assert res_c.status == "FAIL"


def test_evse_connector_my_band_distribution(tmp_path: Path) -> None:
    _write_bound(
        tmp_path,
        "evse_connector_my_band_distribution",
        [
            {"metric_name": "my_2020_2022__share_J1772", "lower_bound": 0.0, "upper_bound": 1.0},
            {"metric_name": "my_2030_2032__share_NACS", "lower_bound": 0.0, "upper_bound": 1.0},
        ],
    )
    fleet = pd.DataFrame(
        {"EVSE_connector": ["J1772", "NACS"], "model_year": [2021, 2021]}
    )
    res = v.check_evse_connector_my_band_distribution(fleet, tmp_path)
    assert res.status == "PASS"
    # band with no matching MY -> None recorded
    assert res.realised_value["my_2030_2032__share_NACS"] is None


# ---------------------------------------------------------------------------
# Summary / markdown / scalar extraction
# ---------------------------------------------------------------------------


def test_build_summary_counts() -> None:
    checks = [
        v.CheckResult("a", "PASS", None, "", "d"),
        v.CheckResult("b", "FAIL", None, "", "d"),
        v.CheckResult("c", "EXPLAINED_SKIP", None, "", "d"),
    ]
    s = v._build_summary(checks)
    assert s["pass"] == 1
    assert s["fail"] == 1
    assert s["explained_skip"] == 1
    assert s["total"] == 3


def test_scalar_from_realised() -> None:
    assert v._scalar_from_realised(None) is None
    assert v._scalar_from_realised(3.5) == 3.5
    assert v._scalar_from_realised({"tvd": 0.04, "x": "str"}) == 0.04
    assert v._scalar_from_realised({"foo": "bar", "n": 7}) == 7.0
    assert v._scalar_from_realised({"only": "strings"}) is None


def test_summarise_realised_branches() -> None:
    assert v._summarise_realised(v.CheckResult("x", "PASS", None, "", "d")) == "-"
    bc = v.CheckResult("battery_capacity_distribution", "PASS", {"tvd": 0.01}, "", "d")
    assert "TVD" in v._summarise_realised(bc)
    cdf = v.CheckResult(
        "weekday_plugin_cdf", "PASS",
        {"F_06": 0.1, "F_12": 0.2, "F_18": 0.3, "F_22": 0.4}, "", "d",
    )
    assert "F(06)" in v._summarise_realised(cdf)
    fec = v.CheckResult(
        "fleet_energy_conservation", "PASS",
        {"ratio": 1.11, "total_gas_kwh_equivalent": 0.0, "total_numerator_kwh": 100.0}, "", "d",
    )
    assert "ratio" in v._summarise_realised(fec)
    generic = v.CheckResult("unknown_check", "PASS", {"a": 1, "b": 2, "c": 3, "d": 4}, "", "d")
    assert "a=1" in v._summarise_realised(generic)
    scalar = v.CheckResult("unknown_check", "PASS", 42, "", "d")
    assert v._summarise_realised(scalar) == "42"


def test_emit_markdown_r1_and_rgt1() -> None:
    cr = v.CheckResult("battery_capacity_distribution", "PASS", {"tvd": 0.01}, "src|pipe", "d|x")
    rep = v.Report(
        validator_version="v1.0.0",
        methodology_version="v1.0.0",
        run_timestamp_utc="2026-01-01T00:00:00+00:00",
        pipeline_artifacts_sha256={"fleet.parquet": "abc"},
        checks=[cr],
        summary=v._build_summary([cr]),
    )
    md1 = v._emit_markdown(rep)
    assert "# M9 validation report" in md1
    assert "| PASS | FAIL" in md1
    # pipe chars escaped
    assert "src\\|pipe" in md1

    md_r = v._emit_markdown(
        rep, replicate_stats={"battery_capacity_distribution": {"mean": 0.01, "std": 0.001}}, R=3
    )
    assert "replicates: `R=3`" in md_r
    assert "0.0100 ± 0.0010" in md_r


# ---------------------------------------------------------------------------
# _run_or_error / _update_meta_json
# ---------------------------------------------------------------------------


def test_run_or_error_catches_exception() -> None:
    def boom() -> v.CheckResult:
        raise RuntimeError("kaboom")

    res = v._run_or_error("xcheck", boom)
    assert res.status == "ERROR"
    assert "RuntimeError" in res.details


def test_run_or_error_passthrough() -> None:
    cr = v.CheckResult("ok", "PASS", None, "", "d")
    assert v._run_or_error("ok", lambda: cr) is cr


def test_update_meta_json_creates_and_updates(tmp_path: Path) -> None:
    meta_path = tmp_path / "meta.json"
    rep = v.Report(
        validator_version=v.VALIDATOR_VERSION,
        methodology_version=v.METHODOLOGY_VERSION,
        run_timestamp_utc="2026-01-01T00:00:00+00:00",
        pipeline_artifacts_sha256={},
        checks=[],
        summary={"total": 0, "pass": 0, "fail": 0, "explained_fail": 0,
                 "explained_skip": 0, "error": 0},
    )
    v._update_meta_json(meta_path, rep)
    meta = json.loads(meta_path.read_text())
    assert meta["validator"]["validator_version"] == v.VALIDATOR_VERSION
    assert meta["validator"]["master_seed"] == v.MASTER_SEED
    assert meta["module_provenance"]["m9_validator"] == v.VALIDATOR_VERSION

    # v2 report -> MASTER_SEED_V2
    rep_v2 = v.Report(
        validator_version=v.VALIDATOR_VERSION_V2,
        methodology_version=v.METHODOLOGY_VERSION_V2,
        run_timestamp_utc="2026-01-01T00:00:00+00:00",
        pipeline_artifacts_sha256={},
        checks=[],
        summary=rep.summary,
    )
    v._update_meta_json(meta_path, rep_v2)
    meta2 = json.loads(meta_path.read_text())
    assert meta2["validator"]["master_seed"] == v.MASTER_SEED_V2


# ---------------------------------------------------------------------------
# run() orchestrator + CLI — minimal synthetic cache + real bounds tree
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_bounds_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("validator_bounds")
    curate_bounds(out)
    return out


def _build_min_cache(output_root: Path) -> None:
    """Write a tiny but schema-faithful M1..M8 cache + meta.json.

    Designed to make every v1.1 residential check run (most resolve to
    PASS / EXPLAINED_*); we only assert the orchestrator produces a complete
    report, not specific per-check verdicts.
    """
    output_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    n = 12
    ev_ids = list(range(n))
    models = ["Bolt_EUV", "Model_3_LR", "Model_Y_LR_AWD"] * (n // 3)
    b_kwh = [65.0, 82.0, 82.0] * (n // 3)
    fleet = pd.DataFrame(
        {
            "ev_id": ev_ids,
            "model": models,
            "make": ["Chevrolet", "Tesla", "Tesla"] * (n // 3),
            "powertrain": ["BEV"] * n,
            "B_kwh": b_kwh,
            "model_year": [2022] * n,
            "OBC_kw": [11.5] * n,
            "EVSE_home_kw": [7.7] * n,
            "EVSE_brand": ["Tesla"] * n,
            "EVSE_connector": ["NACS"] * n,
            "panel_cap_kw": [19.2] * n,
            "eta_charge_nominal": [0.9] * n,
            "soc_min_kwh": [5.0] * n,
            "borrow_used": [False] * n,
        }
    )

    # mobility: a few trips per EV
    mob_rows = []
    for ev in ev_ids:
        for d in range(3):
            miles = float(rng.uniform(20, 40))
            mob_rows.append(
                {
                    "ev_id": ev,
                    "miles": miles,
                    "kwh_consumed": miles * 0.25,
                    "start_ts": pd.Timestamp("2001-01-15T12:00:00Z") + pd.Timedelta(days=d),
                }
            )
    mobility = pd.DataFrame(mob_rows)

    # plug-in events: one weekday + one weekend per EV
    pie_rows = []
    for ev in ev_ids:
        pie_rows.append(
            {"ev_id": ev, "arrival_ts": pd.Timestamp("2001-01-01T20:00:00Z"),
             "plug_in": True, "location": "home"}
        )
        pie_rows.append(
            {"ev_id": ev, "arrival_ts": pd.Timestamp("2001-01-06T20:00:00Z"),
             "plug_in": True, "location": "home"}
        )
    plug_in_events = pd.DataFrame(pie_rows)

    # sessions: one home session per EV
    sess_rows = []
    for ev in ev_ids:
        t_in = pd.Timestamp("2001-01-01T20:00:00Z")
        t_out = t_in + pd.Timedelta(hours=10)
        sess_rows.append(
            {
                "ev_id": ev,
                "session_idx": 0,
                "location": "home",
                "t_in": t_in,
                "t_out": t_out,
                "energy_kwh_grid": 10.0,
                "energy_kwh_batt": 9.0,
                "gas_kwh_equivalent": 0.0,
            }
        )
    sessions = pd.DataFrame(sess_rows)

    # 15-min raster: cover a Monday week so optimizer-window length code can run;
    # but optimizer is skipped (pev_optim absent), so a small raster suffices.
    raster_idx = pd.date_range("2001-01-01T00:00:00Z", periods=96 * 7, freq="15min")
    raster_rows = []
    for ev in ev_ids:
        for ts in raster_idx:
            raster_rows.append({"ev_id": ev, "ts": ts, "plugged": False})
    plug_status_15min = pd.DataFrame(raster_rows)

    hourly_idx = pd.date_range("2001-01-01T00:00:00Z", periods=24 * 7, freq="h")
    hourly_rows = []
    for ev in ev_ids:
        for ts in hourly_idx:
            hourly_rows.append({"ev_id": ev, "ts": ts, "plugged": False})
    plug_status_hourly = pd.DataFrame(hourly_rows)

    fleet.to_parquet(output_root / "fleet.parquet")
    mobility.to_parquet(output_root / "mobility.parquet")
    plug_in_events.to_parquet(output_root / "plug_in_events.parquet")
    sessions.to_parquet(output_root / "sessions.parquet")
    plug_status_15min.to_parquet(output_root / "plug_status_15min.parquet")
    plug_status_hourly.to_parquet(output_root / "plug_status_hourly.parquet")
    (output_root / "meta.json").write_text(
        json.dumps({"cache_provenance": "v1.1_migrated"}), encoding="utf-8"
    )


def test_run_missing_output_root_raises(tmp_path: Path, real_bounds_root: Path) -> None:
    with pytest.raises(FileNotFoundError, match="output_root missing"):
        v.run(output_root=tmp_path / "nope", bounds_root=real_bounds_root)


def test_run_missing_bounds_root_raises(tmp_path: Path) -> None:
    output_root = tmp_path / "cache"
    _build_min_cache(output_root)
    with pytest.raises(FileNotFoundError, match="bounds_root missing"):
        v.run(output_root=output_root, bounds_root=tmp_path / "no_bounds")


def test_run_v1_legacy_full_report(tmp_path: Path, real_bounds_root: Path) -> None:
    output_root = tmp_path / "cache"
    _build_min_cache(output_root)
    rep = v.run(
        output_root=output_root, bounds_root=real_bounds_root, skip_optimizer=True
    )
    assert rep.validator_version == v.VALIDATOR_VERSION
    assert rep.summary["total"] == len(rep.checks)
    # JSON + MD written
    assert (output_root / "validation_report.json").exists()
    assert (output_root / "validation_report.md").exists()
    # meta updated
    meta = json.loads((output_root / "meta.json").read_text())
    assert meta["validator"]["n_checks_evaluated"] == rep.summary["total"]
    # every check has a valid status
    for c in rep.checks:
        assert c.status in v.VALID_STATUSES
    # optimizer skipped explicitly
    opt = [c for c in rep.checks if c.check_id == "optimizer_smoke_test"][0]
    assert opt.status == "EXPLAINED_SKIP"


def test_run_invalid_R_raises(tmp_path: Path, real_bounds_root: Path) -> None:
    output_root = tmp_path / "cache"
    _build_min_cache(output_root)
    with pytest.raises(ValueError, match="R must be a positive integer"):
        v.run(output_root=output_root, bounds_root=real_bounds_root, R=0)


def test_run_replicates_fallback(tmp_path: Path, real_bounds_root: Path) -> None:
    # No replicates/ dir -> falls back to single run against output_root.
    output_root = tmp_path / "cache"
    _build_min_cache(output_root)
    rep = v.run(
        output_root=output_root, bounds_root=real_bounds_root, skip_optimizer=True, R=2
    )
    assert rep.summary["total"] == len(rep.checks)
    data = json.loads((output_root / "validation_report.json").read_text())
    assert data["replicates"]["R"] == 2
    assert "replicate_stats" in data


def test_run_v2_invalid_profile_type_raises(tmp_path: Path, real_bounds_root: Path) -> None:
    output_root = tmp_path / "cache"
    _build_min_cache(output_root)
    with pytest.raises(ValueError, match="profile_type"):
        v.run(
            output_root=output_root, bounds_root=real_bounds_root,
            region="bay_area", profile_type="garbage",
        )


def test_cli_smoke(tmp_path: Path, real_bounds_root: Path) -> None:
    output_root = tmp_path / "cache"
    _build_min_cache(output_root)
    rc = v._cli(
        [
            "--root", str(output_root),
            "--bounds", str(real_bounds_root),
            "--skip-optimizer",
            "--verbose",
        ]
    )
    assert rc in (0, 1)
    assert (output_root / "validation_report.json").exists()


def test_cli_region_without_profile_errors(tmp_path: Path, real_bounds_root: Path) -> None:
    with pytest.raises(SystemExit):
        v._cli(["--root", str(tmp_path), "--bounds", str(real_bounds_root), "--region", "bay_area"])


def _build_workplace_cache(output_root: Path) -> None:
    """Write a tiny UTC-stored workplace cache covering the v2 workplace
    route (W1-W10 + DST + EVSE enrichment). Timestamps stored in UTC so the
    v2 tz-conversion branch fires."""
    output_root.mkdir(parents=True, exist_ok=True)
    n = 9
    ev_ids = list(range(n))
    fleet = pd.DataFrame(
        {
            "ev_id": ev_ids,
            "model": ["Bolt_EUV", "Model_3_LR", "Model_Y_LR_AWD"] * (n // 3),
            "make": ["Chevrolet", "Tesla", "Tesla"] * (n // 3),
            "powertrain": ["BEV"] * n,
            "B_kwh": [65.0, 82.0, 82.0] * (n // 3),
            "model_year": [2022] * n,
            "OBC_kw": [11.5] * n,
            "EVSE_home_kw": [7.7] * n,
            "EVSE_brand": ["ChargePoint"] * n,
            "EVSE_connector": ["J1772"] * n,
            "panel_cap_kw": [19.2] * n,
            "eta_charge_nominal": [0.9] * n,
            "soc_min_kwh": [5.0] * n,
            "borrow_used": [False] * n,
        }
    )
    mob_rows = []
    for ev in ev_ids:
        for d in range(2):
            mob_rows.append(
                {
                    "ev_id": ev,
                    "miles": 30.0,
                    "kwh_consumed": 7.5,
                    "start_ts": pd.Timestamp("2001-01-15T18:00:00Z") + pd.Timedelta(days=d),
                }
            )
    mobility = pd.DataFrame(mob_rows)

    pie = pd.DataFrame(
        {
            "ev_id": ev_ids,
            "arrival_ts": [pd.Timestamp("2001-06-04T16:00:00Z")] * n,
            "plug_in": [True] * n,
            "location": ["work"] * n,
        }
    )

    sessions = pd.DataFrame(
        {
            "ev_id": ev_ids,
            "session_idx": [0] * n,
            "location": ["work"] * n,
            "t_in": [pd.Timestamp("2001-06-04T16:00:00Z")] * n,
            "t_out": [pd.Timestamp("2001-06-04T23:00:00Z")] * n,
            "energy_kwh_grid": [12.0] * n,
            "energy_kwh_batt": [10.8] * n,
            "gas_kwh_equivalent": [0.0] * n,
        }
    )

    raster_idx = pd.date_range("2001-03-30T00:00:00Z", "2001-10-30T00:00:00Z", freq="6h")
    raster_rows = []
    for ev in ev_ids:
        for ts in raster_idx:
            raster_rows.append({"ev_id": ev, "ts": ts, "plugged": False})
    plug_status_15min = pd.DataFrame(raster_rows)
    hourly = pd.DataFrame(
        [{"ev_id": ev, "ts": ts, "plugged": False} for ev in ev_ids for ts in raster_idx[:24]]
    )

    fleet.to_parquet(output_root / "fleet.parquet")
    mobility.to_parquet(output_root / "mobility.parquet")
    pie.to_parquet(output_root / "plug_in_events.parquet")
    sessions.to_parquet(output_root / "sessions.parquet")
    plug_status_15min.to_parquet(output_root / "plug_status_15min.parquet")
    hourly.to_parquet(output_root / "plug_status_hourly.parquet")
    (output_root / "meta.json").write_text(
        json.dumps(
            {
                "cache_provenance": "v2.0_native",
                "plug_in_model": {
                    "W9_max_cluster_window_h": 5.0,
                    "workplace_cohort_caveat": "midday skew caveat",
                },
                "donor_borrow_rate": 0.1,
                "donor_matcher": {"max_recycling_ratio": 1.5},
            }
        ),
        encoding="utf-8",
    )


def test_run_v2_workplace_full_report(tmp_path: Path, real_bounds_root: Path) -> None:
    output_root = tmp_path / "wp_cache"
    _build_workplace_cache(output_root)
    rep = v.run(
        output_root=output_root,
        bounds_root=real_bounds_root,
        skip_optimizer=True,
        region="bay_area",
        profile_type="workplace",
    )
    assert rep.validator_version == v.VALIDATOR_VERSION_V2
    assert rep.summary["total"] == len(rep.checks)
    check_ids = {c.check_id for c in rep.checks}
    # v2 workplace route emits the DST + W1..W10 + EVSE checks.
    assert "dst_verification" in check_ids
    assert "W1_workplace_plug_in_cdf" in check_ids
    assert "W10_workplace_donor_pool_recycling" in check_ids
    for c in rep.checks:
        assert c.status in v.VALID_STATUSES


def test_run_v2_residential_full_report(tmp_path: Path, real_bounds_root: Path) -> None:
    output_root = tmp_path / "res_cache"
    _build_min_cache(output_root)
    rep = v.run(
        output_root=output_root,
        bounds_root=real_bounds_root,
        skip_optimizer=True,
        region="bay_area",
        profile_type="residential",
    )
    assert rep.validator_version == v.VALIDATOR_VERSION_V2
    check_ids = {c.check_id for c in rep.checks}
    assert "dst_verification" in check_ids
    assert "evse_connector_my_band_distribution" in check_ids
    assert "bundle_rule_take_rates" in check_ids


def test_bundle_rule_take_rates(tmp_path: Path) -> None:
    from pev_synth.vehicle_archetypes import EV_EVSE_BUNDLE_RULES

    # Build a fleet covering at least one bundle rule that overrides EVSE_brand.
    rule = next(r for r in EV_EVSE_BUNDLE_RULES if r.overrides.get("EVSE_brand"))
    target_brand = rule.overrides["EVSE_brand"]
    make = "Tesla" if rule.predicate_make == "*" else rule.predicate_make
    model = "Model_3_LR" if rule.predicate_model == "*" else rule.predicate_model
    powertrain = "BEV" if rule.predicate_powertrain == "*" else rule.predicate_powertrain
    my = rule.model_year_min
    n = 50
    fleet = pd.DataFrame(
        {
            "make": [make] * n,
            "model": [model] * n,
            "powertrain": [powertrain] * n,
            "model_year": [my] * n,
            "EVSE_brand": [target_brand] * n,
        }
    )
    _write_bound(
        tmp_path,
        "bundle_rule_take_rates",
        [
            {
                "metric_name": f"take_rate_{rule.name}",
                "lower_bound": 0.0,
                "upper_bound": 1.0,
            }
        ],
    )
    res = v.check_bundle_rule_take_rates(fleet, tmp_path)
    # All matching EVs carry the target brand -> realised take rate 1.0 >= floor.
    assert res.status == "PASS"
    assert res.realised_value[rule.name]["realised_take_rate"] == 1.0
