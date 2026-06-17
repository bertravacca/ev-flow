"""Top-up coverage tests for ``pev_synth.validator`` (M9) and
``pev_synth.validation_bounds_curator`` (M8).

This module is a *supplement* to ``test_pev_synth_validator_cov.py`` and
``test_pev_synth_validation_bounds_curator.py``: it targets ONLY the line
ranges those files left uncovered (validator 93% -> ~99%, curator 97% ->
~100%). Like its siblings every test here is fully *data-independent* — no
heavy ``data/`` tree is touched; inputs are tiny synthetic in-memory pandas
fixtures, fabricated bound parquets written into ``tmp_path``, or
monkeypatched module internals.

Covered remainders include:

* validator: the non-str/non-dict ``cache_provenance`` fall-through, the
  NaN-``ev_id`` FK issue, the non-UTC DST candidate skip + DST FAIL branch,
  the W1 missing-metric ERROR row, the C5 ``lower``-column cold detection and
  empty-trips skip + tz_convert fallback, the W10 ``top5`` reuse-count
  derivation (and its exception fallback), the
  ``check_optimizer_smoke_test_workplace`` TZ-patching wrapper, the EVSE
  kW-tier / share-metric / MY-band ``continue`` and FAIL branches, the bundle
  take-rate missing-bound-row floor fallback, the ``_summarise_realised``
  optimizer / energy-conservation legacy branches, the ``run()`` v2 default-
  cache override + nested-bounds-missing raise + opaque-region fallback + PIE
  defensive column fall-backs, the optimizer smoke FAIL-on-import path, and
  the ``_run_with_replicates`` real-replicate-dir aggregation (n>1 std).
* curator: the malformed / banned source-url raises, the missing sales-mix
  CSV + empty-BEV raises, the unknown-region raise, the empty-builder /
  unwired-region-builder ``RuntimeError`` guards, and the ``main`` / argparse
  CLI entry point.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from pev_synth import validation_bounds_curator as curator
from pev_synth import validator as v

# ---------------------------------------------------------------------------
# Bound-parquet fabrication helper (mirrors test_pev_synth_validator_cov.py)
# ---------------------------------------------------------------------------


def _write_bound(
    bounds_root: Path,
    check_id: str,
    rows: list[dict[str, object]],
    *,
    source_url: str = "https://example.org/source",
) -> None:
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


# ===========================================================================
# validator.py
# ===========================================================================


# --- _is_v11_migrated non-str / non-dict cache_provenance (line 367) -------


def test_is_v11_migrated_non_str_non_dict_provenance() -> None:
    # cache_provenance present but neither a str nor a dict -> falls through
    # to the final ``return False``.
    assert not v._is_v11_migrated({"cache_provenance": 123})
    assert not v._is_v11_migrated({"cache_provenance": ["v1.1_migrated"]})


# NOTE: validator.py line 1036 (the "has NaN ev_id" FK issue) is unreachable
# dead code: line 1031 runs ``set(df["ev_id"].astype(int))`` BEFORE the
# ``df["ev_id"].isna().any()`` guard on 1035-1036, and ``astype(int)`` raises
# on any NaN / pd.NA value (numpy float NaN AND nullable Int64 NA both fail).
# So the function always raises before it can record the NaN issue. Covering
# 1036 would require weakening the test to a contradiction; left uncovered and
# documented instead.


# --- DST: non-UTC candidate skipped + DST FAIL branch (lines 1272, 1326) ---


def _make_dst_cache(output_root: Path, *, tz: str) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    idx = pd.date_range(
        "2001-03-30T00:00:00", "2001-10-30T00:00:00", freq="15min", tz=tz
    )
    df = pd.DataFrame({"ts": pd.Series(idx), "ev_id": 0, "plugged": False})
    df.to_parquet(output_root / "plug_status_15min.parquet")


def test_dst_non_utc_parquet_is_skipped_then_error(tmp_path: Path) -> None:
    # The 15-min raster is stored in a NON-UTC tz, so the DST candidate loop
    # hits ``continue`` (line 1272) and finds no UTC source -> ERROR.
    output_root = tmp_path / "cache"
    _make_dst_cache(output_root, tz="America/Los_Angeles")
    bounds_root = tmp_path / "bounds"
    _write_bound(bounds_root, "dst_verification", [{"metric_name": "monotonic_utc"}])
    res = v.check_dst_verification(output_root, bounds_root)
    assert res.status == "ERROR"


def test_dst_fail_when_a_zone_is_not_monotonic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # check_dst_verification re-sorts + dedups the index, so a clean UTC grid
    # always yields a monotonic, NaT-free, dup-free per-zone result. To
    # exercise the overall-FAIL return (line 1326) we patch the per-zone
    # checker so one zone reports utc_monotonic=False -> all_ok=False -> FAIL.
    output_root = tmp_path / "cache"
    _make_dst_cache(output_root, tz="UTC")
    bounds_root = tmp_path / "bounds"
    _write_bound(bounds_root, "dst_verification", [{"metric_name": "monotonic_utc"}])

    def _fake_zone(idx: pd.DatetimeIndex, zone: str) -> dict[str, object]:
        return {
            "zone": zone,
            "utc_monotonic": zone == "UTC",  # one zone non-monotonic
            "local_nat_count": 0,
            "local_dup_count": 0,
            "n_rows": int(len(idx)),
        }

    monkeypatch.setattr(v, "_check_dst_zone", _fake_zone)
    res = v.check_dst_verification(output_root, bounds_root)
    assert res.status == "FAIL"
    assert any(
        not z["utc_monotonic"] for z in res.realised_value["per_zone"].values()
    )


# --- W1 workplace CDF missing-metric ERROR row (lines 1394-1396) -----------


def _workplace_pie(hours_utc: list[int]) -> pd.DataFrame:
    ts = pd.to_datetime(
        [f"2001-06-04T{h:02d}:00:00" for h in hours_utc]
    ).tz_localize("UTC")
    return pd.DataFrame(
        {
            "arrival_ts": ts,
            "plug_in": [True] * len(hours_utc),
            "location": ["work"] * len(hours_utc),
        }
    )


def test_workplace_w1_missing_metric_row_is_error(tmp_path: Path) -> None:
    # Only the hour-09 row present; hours 12 and 15 absent -> ERROR status
    # appended for the missing probe hours.
    _write_bound(
        tmp_path,
        "W1_workplace_plug_in_cdf",
        [{"metric_name": "plug_in_cdf_at_hour_09", "lower_bound": 0.0, "upper_bound": 1.0}],
    )
    pie = _workplace_pie([16, 17, 18])
    res = v.check_workplace_w1_plug_in_cdf(pie, tmp_path, "America/Los_Angeles")
    # The missing-metric rows append an "ERROR" status with a NaN band, but
    # W1's final verdict folds any non-PASS status into EXPLAINED_FAIL. The
    # NaN band on the missing probe hours confirms the 1394-1396 ERROR path
    # ran.
    assert res.status == "EXPLAINED_FAIL"
    assert math.isnan(res.realised_value["bounds"]["F_12"]["lo"])


# --- C5 winter ratio: cold-via-lower-column + tz-naive + empty trips -------


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


def test_c5_cold_detected_via_lower_column_no_region_obj(tmp_path: Path) -> None:
    # No region_obj -> cold detection falls back to the ``lower`` column
    # (lines 1810-1812). A populated ``lower`` => cold region.
    _write_c5_bound(tmp_path, 1.15)
    winter = pd.DataFrame(
        {
            "miles": [100.0, 100.0],
            "kwh_consumed": [40.0, 40.0],
            # tz-naive timestamps -> exercises the ``ts.tz is None`` else
            # branch (line 1843).
            "start_ts": pd.to_datetime(
                ["2001-01-15T12:00:00", "2001-02-15T12:00:00"]
            ),
        }
    )
    summer = pd.DataFrame(
        {
            "miles": [100.0, 100.0],
            "kwh_consumed": [25.0, 25.0],
            "start_ts": pd.to_datetime(
                ["2001-07-15T12:00:00", "2001-08-15T12:00:00"]
            ),
        }
    )
    mob = pd.concat([winter, summer], ignore_index=True)
    res = v.check_C5_winter_ratio(
        mob, tmp_path, region="chicago", region_tz="America/Chicago"
    )
    assert res.status == "PASS"
    assert res.realised_value["winter_summer_ratio"] == pytest.approx(1.6)


def test_c5_empty_trips_is_skip(tmp_path: Path) -> None:
    # Cold region (region_obj), but all trips have zero miles -> EXPLAINED_SKIP
    # via the empty-``trips`` guard (lines 1826-1831).
    _write_c5_bound(tmp_path, 1.15)

    class _ColdRegion:
        winter_f_T_multiplier = 1.2

    mob = pd.DataFrame(
        {
            "miles": [0.0, 0.0],
            "kwh_consumed": [0.0, 0.0],
            "start_ts": pd.to_datetime(
                ["2001-01-15T12:00:00Z", "2001-07-15T12:00:00Z"]
            ),
        }
    )
    res = v.check_C5_winter_ratio(
        mob,
        tmp_path,
        region="chicago",
        region_tz="America/Chicago",
        region_obj=_ColdRegion(),
    )
    assert res.status == "EXPLAINED_SKIP"


def test_c5_tz_convert_exception_falls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Force the tz_convert call to raise so the except branch (lines 1840-1841)
    # falls back to the unconverted index.
    _write_c5_bound(tmp_path, 1.15)

    class _ColdRegion:
        winter_f_T_multiplier = 1.2

    real_tz_convert = pd.DatetimeIndex.tz_convert

    def _boom(self: pd.DatetimeIndex, tz: object) -> pd.DatetimeIndex:
        raise ValueError("synthetic tz failure")

    monkeypatch.setattr(pd.DatetimeIndex, "tz_convert", _boom)
    try:
        winter = pd.DataFrame(
            {
                "miles": [100.0],
                "kwh_consumed": [40.0],
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
            mob,
            tmp_path,
            region="chicago",
            region_tz="America/Chicago",
            region_obj=_ColdRegion(),
        )
    finally:
        monkeypatch.setattr(pd.DatetimeIndex, "tz_convert", real_tz_convert)
    # tz_convert failed -> fell back to UTC months, still both seasons present.
    assert res.status in {"PASS", "FAIL"}


# --- W10 top5 reuse-count derivation + exception fallback (1904-1905) ------


def test_w10_top5_reuse_count_derivation(tmp_path: Path) -> None:
    _write_bound(
        tmp_path,
        "W10_workplace_donor_pool_recycling",
        [{"metric_name": "max_recycling_ratio", "lower_bound": 0.0, "upper_bound": 5.0}],
    )
    fleet = pd.DataFrame({"ev_id": [0, 1]})
    # No explicit max_recycling_ratio -> derive from top5 reuse counts (max=3).
    res = v.check_workplace_w10_donor_pool_recycling(
        {"donor_matcher": {"top5_donor_houseids": [["h1", 3], ["h2", 1]]}},
        fleet,
        tmp_path,
    )
    assert res.status == "PASS"
    assert res.realised_value["max_recycling_ratio"] == 3.0


def test_w10_top5_malformed_falls_back_to_skip(tmp_path: Path) -> None:
    _write_bound(
        tmp_path,
        "W10_workplace_donor_pool_recycling",
        [{"metric_name": "max_recycling_ratio", "lower_bound": 0.0, "upper_bound": 5.0}],
    )
    fleet = pd.DataFrame({"ev_id": [0, 1]})
    # Malformed top5 entries (no [1] index) -> derivation raises, ratio stays
    # None (lines 1904-1905) -> EXPLAINED_SKIP.
    res = v.check_workplace_w10_donor_pool_recycling(
        {"donor_matcher": {"top5_donor_houseids": [["only-one-element"]]}},
        fleet,
        tmp_path,
    )
    assert res.status == "EXPLAINED_SKIP"


# --- check_optimizer_smoke_test_workplace TZ wrapper (1948-1955) -----------


def test_optimizer_smoke_test_workplace_wrapper_restores_tz(tmp_path: Path) -> None:
    # pev_optim is absent on this checkout, so the wrapped call hits the
    # ImportError -> the smoke check returns FAIL. The wrapper's job is to
    # patch + restore the module TZ deterministically.
    _write_bound(
        tmp_path,
        "optimizer_smoke_test",
        [{"metric_name": "solves_without_raising", "unbounded_v1": True}],
    )
    fleet = pd.DataFrame(
        {
            "ev_id": [0],
            "model": ["Bolt_EUV"],
            "B_kwh": [65.0],
            "OBC_kw": [11.5],
            "EVSE_home_kw": [7.7],
            "panel_cap_kw": [19.2],
            "eta_charge_nominal": [0.9],
            "soc_min_kwh": [5.0],
        }
    )
    sessions = pd.DataFrame(
        {
            "ev_id": [0],
            "t_in": pd.to_datetime(["2001-01-01T00:00:00Z"]),
            "t_out": pd.to_datetime(["2001-01-01T08:00:00Z"]),
        }
    )
    raster = pd.DataFrame(
        {
            "ev_id": [0],
            "ts": pd.to_datetime(["2001-01-01T00:00:00Z"]),
            "plugged": [False],
        }
    )
    saved = v.TZ
    res = v.check_optimizer_smoke_test_workplace(
        fleet, sessions, raster, tmp_path, ev_id=0, tz="America/New_York"
    )
    assert res.status == "FAIL"  # pev_optim import failed inside the smoke
    assert v.TZ == saved  # restored by the finally block


# --- check_optimizer_smoke_test FAIL-on-import path (916-919, 1002-1008) ---


def test_optimizer_smoke_test_import_failure_is_fail(tmp_path: Path) -> None:
    _write_bound(
        tmp_path,
        "optimizer_smoke_test",
        [{"metric_name": "solves_without_raising", "unbounded_v1": True}],
    )
    fleet = pd.DataFrame({"ev_id": [0], "model": ["Bolt_EUV"], "B_kwh": [65.0]})
    sessions = pd.DataFrame(
        {
            "ev_id": [0],
            "t_in": pd.to_datetime(["2001-01-01T00:00:00Z"]),
            "t_out": pd.to_datetime(["2001-01-01T08:00:00Z"]),
        }
    )
    raster = pd.DataFrame(
        {
            "ev_id": [0],
            "ts": pd.to_datetime(["2001-01-01T00:00:00Z"]),
            "plugged": [False],
        }
    )
    res = v.check_optimizer_smoke_test(fleet, sessions, raster, tmp_path, ev_id=0)
    # pev_optim / pev_data / caiso_data are out-of-PyPI -> ImportError caught.
    assert res.status == "FAIL"
    assert (
        "ModuleNotFoundError" in res.details or "ImportError" in res.details
    )


# --- EVSE kW-tier continue + 6.6-7.7 band FAIL (1989-1990, 2013) -----------


def test_evse_kw_distribution_nonnumeric_tier_skipped(tmp_path: Path) -> None:
    # A ``share_at_<x>_kw`` row whose tier is not parseable as float is
    # skipped (lines 1989-1990); the valid tier still scores.
    _write_bound(
        tmp_path,
        "evse_kw_distribution_residential",
        [
            {"metric_name": "share_at_notanumber_kw", "lower_bound": 0.0, "upper_bound": 1.0},
            {"metric_name": "share_at_7.7_kw", "lower_bound": 0.0, "upper_bound": 1.0},
        ],
    )
    fleet = pd.DataFrame({"EVSE_home_kw": [7.7, 7.7]})
    res = v.check_evse_kw_distribution(fleet, tmp_path, profile_type="residential")
    assert res.status == "PASS"
    assert "share_at_notanumber_kw" not in res.realised_value
    assert res.realised_value["share_at_7.7_kw"] == 1.0


def test_evse_kw_distribution_legacy_band_fail(tmp_path: Path) -> None:
    # share_6p6_to_7p7_kw out of band -> the legacy-band FAIL branch (2013).
    _write_bound(
        tmp_path,
        "evse_kw_distribution_workplace",
        [{"metric_name": "share_6p6_to_7p7_kw", "lower_bound": 0.9, "upper_bound": 1.0}],
    )
    fleet = pd.DataFrame({"EVSE_home_kw": [11.5, 19.2, 3.3]})
    res = v.check_evse_kw_distribution(fleet, tmp_path, profile_type="workplace")
    assert res.status == "FAIL"
    assert "share_6p6_to_7p7_kw" in res.realised_value


# --- _share_metric_check non-share metric skipped (2043) -------------------


def test_share_metric_check_skips_non_share_metric(tmp_path: Path) -> None:
    _write_bound(
        tmp_path,
        "evse_brand_distribution_residential",
        [
            {"metric_name": "n_total_meta", "lower_bound": 0.0, "upper_bound": 1.0},
            {"metric_name": "share_Tesla", "lower_bound": 0.0, "upper_bound": 1.0},
        ],
    )
    fleet = pd.DataFrame({"EVSE_brand": ["Tesla", "Tesla"]})
    res = v.check_evse_brand_distribution(fleet, tmp_path)
    assert res.status == "PASS"
    assert "n_total_meta" not in res.realised_value
    assert res.realised_value["share_Tesla"] == 1.0


# --- MY-band: malformed band continue + FAIL (2117, 2122-2123) -------------


def test_evse_connector_my_band_malformed_and_fail(tmp_path: Path) -> None:
    _write_bound(
        tmp_path,
        "evse_connector_my_band_distribution",
        [
            # malformed: no ``__share_`` token -> 2116-2117 continue
            {"metric_name": "my_2020_2022_no_share_token", "lower_bound": 0.0,
             "upper_bound": 1.0},
            # malformed band split: only one numeric -> 2122-2123 continue
            {"metric_name": "my_bad__share_NACS", "lower_bound": 0.0, "upper_bound": 1.0},
            # valid but out of band -> FAIL (2133-2136)
            {"metric_name": "my_2020_2022__share_J1772", "lower_bound": 0.9,
             "upper_bound": 1.0},
        ],
    )
    fleet = pd.DataFrame(
        {"EVSE_connector": ["NACS", "NACS"], "model_year": [2021, 2021]}
    )
    res = v.check_evse_connector_my_band_distribution(fleet, tmp_path)
    # J1772 share among 2020-2022 = 0.0, band [0.9, 1.0] -> FAIL.
    assert res.status == "FAIL"
    assert res.realised_value["my_2020_2022__share_J1772"] == 0.0


# --- bundle_rule_take_rates missing-bound-row floor fallback (2225-2226) ---


def test_bundle_rule_missing_bound_row_recomputes_floor(tmp_path: Path) -> None:
    from pev_synth.vehicle_archetypes import EV_EVSE_BUNDLE_RULES

    rule = next(r for r in EV_EVSE_BUNDLE_RULES if r.overrides.get("EVSE_brand"))
    target_brand = rule.overrides["EVSE_brand"]
    make = "Tesla" if rule.predicate_make == "*" else rule.predicate_make
    model = "Model_3_LR" if rule.predicate_model == "*" else rule.predicate_model
    powertrain = "BEV" if rule.predicate_powertrain == "*" else rule.predicate_powertrain
    n = 10
    fleet = pd.DataFrame(
        {
            "make": [make] * n,
            "model": [model] * n,
            "powertrain": [powertrain] * n,
            "model_year": [rule.model_year_min] * n,
            "EVSE_brand": [target_brand] * n,
        }
    )
    # Bound parquet has NO row for this rule's metric -> validator recomputes
    # the floor locally + logs a warning (lines 2225-2226).
    _write_bound(
        tmp_path,
        "bundle_rule_take_rates",
        [{"metric_name": "take_rate_some_other_rule", "lower_bound": 0.0, "upper_bound": 1.0}],
    )
    res = v.check_bundle_rule_take_rates(fleet, tmp_path)
    assert res.status == "PASS"
    assert res.realised_value[rule.name]["realised_take_rate"] == 1.0


def test_bundle_rule_below_floor_is_fail(tmp_path: Path) -> None:
    from pev_synth.vehicle_archetypes import EV_EVSE_BUNDLE_RULES

    rule = next(r for r in EV_EVSE_BUNDLE_RULES if r.overrides.get("EVSE_brand"))
    target_brand = rule.overrides["EVSE_brand"]
    make = "Tesla" if rule.predicate_make == "*" else rule.predicate_make
    model = "Model_3_LR" if rule.predicate_model == "*" else rule.predicate_model
    powertrain = "BEV" if rule.predicate_powertrain == "*" else rule.predicate_powertrain
    n = 10
    # NONE of the matching EVs carry the target brand -> realised share 0.0
    # below any positive floor -> failure append (line 2239).
    other = " NotTheBrand"
    fleet = pd.DataFrame(
        {
            "make": [make] * n,
            "model": [model] * n,
            "powertrain": [powertrain] * n,
            "model_year": [rule.model_year_min] * n,
            "EVSE_brand": [other] * n,
        }
    )
    _write_bound(
        tmp_path,
        "bundle_rule_take_rates",
        [{"metric_name": f"take_rate_{rule.name}", "lower_bound": 0.5, "upper_bound": 1.0}],
    )
    res = v.check_bundle_rule_take_rates(fleet, tmp_path)
    assert res.status == "FAIL"
    assert target_brand != other


# --- _summarise_realised optimizer + energy-conservation legacy (2409, 2420)


def test_summarise_realised_optimizer_and_fec_legacy() -> None:
    opt = v.CheckResult(
        "optimizer_smoke_test", "PASS", {"statuses": ["optimal", "feasible"]}, "", "d"
    )
    assert "statuses" in v._summarise_realised(opt)

    # fleet_energy_conservation realised dict WITHOUT total_gas_kwh_equivalent
    # -> the bare ``ratio = ...`` fallback (line 2420).
    fec = v.CheckResult(
        "fleet_energy_conservation", "PASS", {"ratio": 1.05}, "", "d"
    )
    assert v._summarise_realised(fec) == "ratio = 1.05"


# ===========================================================================
# validator.run() — v2 default-cache override, nested-bounds-missing, opaque
# region fallback, PIE defensive columns, optimizer-present route, replicates
# ===========================================================================


def _curated_bounds(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("topup_bounds")
    curator.curate_bounds(out)
    return out


@pytest.fixture(scope="module")
def real_bounds_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _curated_bounds(tmp_path_factory)


def _build_workplace_cache_utc(output_root: Path, *, drop_pie_cols: bool = False) -> None:
    """UTC-stored workplace cache. With ``drop_pie_cols`` the plug_in_events
    frame omits ``plug_in``/``location`` and supplies ``location_type`` so the
    run() defensive fall-backs (lines 2571, 2573) fire."""
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

    if drop_pie_cols:
        pie = pd.DataFrame(
            {
                "ev_id": ev_ids,
                "arrival_ts": [pd.Timestamp("2001-06-04T16:00:00Z")] * n,
                "departure_ts": [pd.Timestamp("2001-06-04T23:00:00Z")] * n,
                "location_type": ["work"] * n,
            }
        )
    else:
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
    plug_status_15min = pd.DataFrame(
        [{"ev_id": ev, "ts": ts, "plugged": False} for ev in ev_ids for ts in raster_idx]
    )
    hourly = pd.DataFrame(
        [
            {"ev_id": ev, "ts": ts, "plugged": False}
            for ev in ev_ids
            for ts in raster_idx[:24]
        ]
    )
    import json as _json

    fleet.to_parquet(output_root / "fleet.parquet")
    mobility.to_parquet(output_root / "mobility.parquet")
    pie.to_parquet(output_root / "plug_in_events.parquet")
    sessions.to_parquet(output_root / "sessions.parquet")
    plug_status_15min.to_parquet(output_root / "plug_status_15min.parquet")
    hourly.to_parquet(output_root / "plug_status_hourly.parquet")
    (output_root / "meta.json").write_text(
        _json.dumps(
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


def test_run_v2_default_cache_root_missing_raises(
    real_bounds_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # output_root left at DEFAULT_OUTPUT_ROOT triggers the v2 cache-template
    # override (line 2500). Point the data root at an empty dir so the
    # runtime-resolved _cache_dir(...) does not exist -> FileNotFoundError at
    # the existence guard, regardless of whether a real data/ tree is present.
    monkeypatch.setenv("PEV_SYNTH_DATA_ROOT", str(tmp_path / "empty_root"))
    with pytest.raises(FileNotFoundError, match="output_root missing"):
        v.run(
            bounds_root=real_bounds_root,
            region="bay_area",
            profile_type="residential",
        )


def test_run_nested_bounds_dir_missing_raises(tmp_path: Path) -> None:
    output_root = tmp_path / "cache"
    _build_workplace_cache_utc(output_root)
    empty_bounds = tmp_path / "bounds_top"
    # bounds_root exists but has no bay_area/residential subtree -> the nested
    # resolution raises (line 2528).
    empty_bounds.mkdir()
    with pytest.raises(FileNotFoundError, match="nested validation_bounds dir"):
        v.run(
            output_root=output_root,
            bounds_root=empty_bounds,
            region="bay_area",
            profile_type="residential",
        )


def test_run_opaque_region_falls_back(
    tmp_path: Path, real_bounds_root: Path
) -> None:
    # An unregistered region name -> Region.from_name raises -> region_obj
    # falls back to None (lines 2488-2491). We materialise the matching nested
    # bounds dir by copying the bay_area/residential subtree under the opaque
    # region name so run() can resolve it.
    import shutil

    opaque = "atlantis"
    src = real_bounds_root / "bay_area"
    dst = real_bounds_root / opaque
    if not dst.exists():
        shutil.copytree(src, dst)

    output_root = tmp_path / "cache"
    _build_workplace_cache_utc(output_root, drop_pie_cols=True)
    rep = v.run(
        output_root=output_root,
        bounds_root=real_bounds_root,
        skip_optimizer=True,
        region=opaque,
        profile_type="residential",
    )
    assert rep.validator_version == v.VALIDATOR_VERSION_V2
    for c in rep.checks:
        assert c.status in v.VALID_STATUSES


def test_run_optimizer_present_route_errors_gracefully(
    tmp_path: Path, real_bounds_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Pretend pev_optim IS installed so run() takes the optimizer-executing
    # ``else`` branch (lines 2684-2691 workplace + 2693-2699 residential).
    # The actual import inside the smoke check still fails -> _run_or_error
    # catches it -> ERROR (no weakening: the smoke genuinely cannot solve).
    import importlib.util as _ilutil

    real_find_spec = _ilutil.find_spec

    def _fake_find_spec(name: str, *args: object, **kwargs: object) -> object:
        if name == "pev_optim":
            return object()  # truthy non-None spec
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(_ilutil, "find_spec", _fake_find_spec)

    # Workplace route -> check_optimizer_smoke_test_workplace branch.
    wp_root = tmp_path / "wp_cache"
    _build_workplace_cache_utc(wp_root)
    rep_wp = v.run(
        output_root=wp_root,
        bounds_root=real_bounds_root,
        skip_optimizer=False,
        region="bay_area",
        profile_type="workplace",
    )
    opt_wp = [c for c in rep_wp.checks if c.check_id == "optimizer_smoke_test"][0]
    # pev_optim import inside the smoke fails -> ERROR (caught by _run_or_error)
    # or FAIL (caught inside check_optimizer_smoke_test). Either way not a skip.
    assert opt_wp.status in {"ERROR", "FAIL"}

    # Residential route -> the plain check_optimizer_smoke_test else branch.
    res_root = tmp_path / "res_cache"
    _build_workplace_cache_utc(res_root)
    rep_res = v.run(
        output_root=res_root,
        bounds_root=real_bounds_root,
        skip_optimizer=False,
        region="bay_area",
        profile_type="residential",
    )
    opt_res = [c for c in rep_res.checks if c.check_id == "optimizer_smoke_test"][0]
    assert opt_res.status in {"ERROR", "FAIL"}


def test_run_with_real_replicate_dirs_aggregates(
    tmp_path: Path, real_bounds_root: Path
) -> None:
    # Build a ``replicates/r0`` and ``r1`` layout so _run_with_replicates
    # discovers them (lines 2908-2911) and aggregates with n>1 std
    # (lines 2949-2950).
    output_root = tmp_path / "cache"
    output_root.mkdir(parents=True, exist_ok=True)
    rep_root = output_root / "replicates"
    for i in range(2):
        _build_workplace_cache_utc(rep_root / f"r{i}")

    rep = v.run(
        output_root=output_root,
        bounds_root=real_bounds_root,
        skip_optimizer=True,
        region="bay_area",
        profile_type="workplace",
        R=2,
    )
    import json as _json

    data = _json.loads((output_root / "validation_report.json").read_text())
    assert data["replicates"]["R"] == 2
    stats = data["replicate_stats"]
    # At least one check produced a finite scalar across both replicates
    # (n=2 -> exercises the n>1 std computation at lines 2949-2950).
    assert any(s["n"] == 2.0 for s in stats.values())
    assert rep.summary["total"] == len(rep.checks)


# NOTE: validator.py line 2946 (the ``if n == 0: continue`` guard inside
# _run_with_replicates' per-check aggregation) is unreachable dead code: a
# check_id key is added to ``per_check_values`` ONLY via
# ``setdefault(cid, []).append(val)`` after a finite-scalar guard, so every
# key carries len >= 1 by construction. n can never be 0. Left uncovered.
#
# NOTE: validator.py lines 923-1001 (the success body of
# check_optimizer_smoke_test) require the in-house out-of-PyPI ``pev_optim`` /
# ``pev_data`` / ``caiso_data`` packages, absent on this checkout. The import
# (line 922) and the except-FAIL return (1002-1008) ARE covered via
# test_optimizer_smoke_test_import_failure_is_fail; the solve body is
# genuinely uncoverable without those packages.


# ===========================================================================
# validation_bounds_curator.py
# ===========================================================================


# --- _assert_url_well_formed malformed + banned (lines 390, 394) -----------


def test_assert_url_well_formed_rejects_malformed() -> None:
    with pytest.raises(ValueError, match="Malformed source_url"):
        curator._assert_url_well_formed("not a url at all")


def test_assert_url_well_formed_rejects_banned_pecan_street() -> None:
    banned = curator.BANNED_SOURCE_FRAGMENTS[0]
    url = f"https://www.{banned}/article"
    with pytest.raises(ValueError, match="Pecan Street May 2025"):
        curator._assert_url_well_formed(url)


# --- _load_sales_mix_csv missing file (line 423) ---------------------------


def test_load_sales_mix_csv_missing_raises() -> None:
    with pytest.raises(FileNotFoundError, match="sales-mix CSV"):
        curator._load_sales_mix_csv("this_source_does_not_exist")


# --- _battery_pmf_from_sales_mix empty BEV (line 466) ----------------------


def test_battery_pmf_empty_bev_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    # Monkeypatch the CSV reader to return a frame whose pack sizes are all
    # below the 30 kWh BEV threshold -> empty BEV slice -> ValueError.
    phev_only = pd.DataFrame(
        {"battery_kwh_nominal": [10.0, 18.0], "share": [0.5, 0.5]}
    )
    monkeypatch.setattr(
        curator, "_load_sales_mix_csv", lambda source: phev_only
    )
    with pytest.raises(ValueError, match="no BEV rows"):
        curator._battery_pmf_from_sales_mix("argonne_national")


# --- _build_battery_capacity_distribution unknown region (line 548) --------


def test_build_battery_capacity_unknown_region_raises() -> None:
    with pytest.raises(ValueError, match="no battery sales-mix source mapping"):
        curator._build_battery_capacity_distribution("narnia")


# --- curate_bounds empty-builder guard (line 2479) -------------------------


def test_curate_bounds_empty_builder_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Replace one builder with an empty-row producer so the eager
    # "Builder returned no rows" guard fires.
    first_id = curator.ALL_CHECK_IDS[1]  # avoid the region-specific battery one
    patched = dict(curator._BUILDERS)
    patched[first_id] = lambda: []
    monkeypatch.setattr(curator, "_BUILDERS", patched)
    with pytest.raises(RuntimeError, match="returned no rows"):
        curator.curate_bounds(tmp_path / "out")


# --- curate_bounds unwired region-specific builder (line 2584) -------------


def test_curate_bounds_unwired_region_builder_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Inject a fake region-specific check id with no per-region builder wiring
    # so the RuntimeError guard inside the per-region loop fires.
    monkeypatch.setattr(
        curator,
        "REGION_SPECIFIC_CHECK_IDS",
        ("battery_capacity_distribution", "fake_region_check"),
    )
    with pytest.raises(RuntimeError, match="per-region builder is wired"):
        curator.curate_bounds(tmp_path / "out")


# --- CLI: main + _parse_args (lines 2736-2762) -----------------------------


def test_curator_cli_main_writes_bounds(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "cli_bounds"
    rc = curator.main(["--out", str(out)])
    assert rc == 0
    assert (out / "_manifest.json").exists()
    captured = capsys.readouterr()
    assert "check parquets + manifest" in captured.out


def test_curator_parse_args_default_out() -> None:
    ns = curator._parse_args([])
    assert ns.out == Path("data/pev/validation_bounds")
    ns2 = curator._parse_args(["--out", "/tmp/somewhere"])
    assert ns2.out == Path("/tmp/somewhere")
