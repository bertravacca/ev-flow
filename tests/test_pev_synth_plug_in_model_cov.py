"""Data-independent coverage tests for M5 — `pev_synth.plug_in_model` (v2.0/v3.0).

These tests are deliberately self-contained: they build tiny synthetic
fleets, mobility frames, workplace cohorts, and SPEECh-shaped JSON fixtures
in-memory or under ``tmp_path`` so the whole file runs and passes with the
heavy ``data/`` tree ABSENT. The bundled SPEECh K=16 JSONs shipped inside
``src/pev_synth/data/speech/`` are the only on-disk inputs relied upon, and
those ship in the wheel by construction.

Scope (complements the skip-gated suite in
``tests/test_pev_synth_plug_in_model.py``):

- helper accessors (`_beta_0_table`, `_beta_table`, `_pi_base_table`,
  `_k_sweep`, `_default_k`, `_location_label_for`, `_beta_0_for_cluster`
  incl. the K=16 SPEECh fallback and out-of-table fallback);
- `_admissible_K` RFC-036 small-cohort guard;
- cohort loaders' missing-file fallbacks + the residential EVWatts filter;
- `run_bic_sweep` over synthetic cohorts: success, degenerate, W9 rejection,
  ambiguous default-K, RFC-037 centroid-drift override, NaN filter;
- `_pi_for_segment` residential modulation + workplace passthrough + padding;
- `assign_clusters` per-EV determinism;
- `compute_p_plug` / `_sigmoid` / `_in_cluster_window` math;
- `iterate_bernoulli` (residential + workplace) over synthetic mobility;
- `aggregate_plug_in_rate`, schema casting, empty-events frame;
- SPEECh loaders (`load_speech_k16_params`, `_speech_group_arrays`,
  `_speech_centres_per_group`, `_resolve_speech_json_path`,
  `_bundled_speech_json_path`, `_build_speech_cluster_fit`,
  `_speech_weekend_centres_from_fit`) using bundled + fixture JSON;
- `fit_and_assign_plug_ins` (both cluster_sources) + error branches;
- persistence (`write_plug_in_events`, `write_fleet`,
  `write_bic_sweep_results`, `_json_default`), `update_meta_with_m5`,
  and the `run` / `main` CLI driver with synthetic on-disk caches.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pev_synth import plug_in_model as pim

SEED = pim.MASTER_SEED_DEFAULT


# --------------------------------------------------------------------------- #
# Synthetic fixtures (no data/ tree needed).
# --------------------------------------------------------------------------- #


def _make_fleet(n: int = 6) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ev_id": np.arange(1, n + 1, dtype=np.int64),
            "B_kwh": np.full(n, 70.0, dtype=np.float64),
            "income_tier": (["t1", "t2", "t4"] * n)[:n],
            "garage_type": (["dedicated", "shared", "curb"] * n)[:n],
        }
    )


def _make_mobility(profile_type: str, n_ev: int = 6) -> pd.DataFrame:
    """A two-trip-per-EV mobility frame: an out trip then a return arrival.

    The second trip's destination is the plug-in target location so each EV
    has exactly one eligible arrival; timestamps are UTC and arrivals land in
    the evening LT for residential / morning LT for workplace.
    """
    target = "work" if profile_type == "workplace" else "home"
    rows: list[dict[str, object]] = []
    for ev in range(1, n_ev + 1):
        # First trip: leave home in the morning UTC.
        rows.append(
            {
                "ev_id": ev,
                "start_ts": pd.Timestamp("2026-05-20 14:00", tz="UTC"),
                "end_ts": pd.Timestamp("2026-05-20 15:00", tz="UTC"),
                "dest_type": "other",
                "kwh_consumed": 5.0,
                "is_weekend": False,
            }
        )
        # Arrival at target location.
        arr_hour_utc = 16 if profile_type == "workplace" else 2  # next-day LT
        rows.append(
            {
                "ev_id": ev,
                "start_ts": pd.Timestamp("2026-05-20 22:00", tz="UTC"),
                "end_ts": pd.Timestamp(f"2026-05-20 {arr_hour_utc:02d}:00", tz="UTC")
                + pd.Timedelta(days=1),
                "dest_type": target,
                "kwh_consumed": 8.0,
                "is_weekend": False,
            }
        )
    return pd.DataFrame(rows)


def _make_workplace_cohort(n: int = 200, *, drift: bool = False) -> pd.DataFrame:
    """Synthetic workplace cohort with tight per-vehicle start-time clusters.

    Two well-separated commuter modes inside the [7, 10.5] band (so W9
    passes and centroids stay in-band) unless ``drift=True``, which pushes
    centres to ~12:00 to trip the RFC-037 commuter-band override.
    """
    rng = np.random.default_rng(123)
    half = n // 2
    if drift:
        c1, c2 = 12.0, 13.0
    else:
        c1, c2 = 8.0, 9.5
    start = np.concatenate(
        [rng.normal(c1, 0.3, half), rng.normal(c2, 0.3, n - half)]
    )
    return pd.DataFrame(
        {
            "vehicle_id": np.arange(n, dtype=np.int64),
            "plug_in_hr_med": start,
            "dur_h_med": rng.normal(8.0, 0.5, n),
            "avg_kw_med": rng.normal(6.5, 0.5, n),
        }
    )


# --------------------------------------------------------------------------- #
# Helper accessors + admissibility.
# --------------------------------------------------------------------------- #


def test_admissible_K_bands() -> None:
    assert pim._admissible_K(50) == {2, 3}
    assert pim._admissible_K(89) == {2, 3}
    assert pim._admissible_K(90) == {2, 3, 4}
    assert pim._admissible_K(149) == {2, 3, 4}
    assert pim._admissible_K(150) == {2, 3, 4, 5}
    assert pim._admissible_K(1000) == {2, 3, 4, 5}


def test_helper_accessors_route_on_profile_type() -> None:
    assert pim._beta_0_table("workplace") is pim.BETA_0_BY_K_WORKPLACE
    assert pim._beta_0_table("residential") is pim.BETA_0_BY_K_RESIDENTIAL
    assert pim._beta_table("workplace") is pim.BETA_WORKPLACE
    assert pim._beta_table("residential") is pim.BETA_RESIDENTIAL
    assert pim._pi_base_table("workplace") is pim.PI_BASE_WORKPLACE
    assert pim._pi_base_table("residential") is pim.PI_BASE_RESIDENTIAL
    assert pim._k_sweep("workplace") == pim.K_SWEEP_WORKPLACE
    assert pim._k_sweep("residential") == pim.K_SWEEP_RESIDENTIAL
    assert pim._default_k("workplace") == pim.DEFAULT_K_WORKPLACE
    assert pim._default_k("residential") == pim.DEFAULT_K_RESIDENTIAL
    assert pim._location_label_for("workplace") == "work"
    assert pim._location_label_for("residential") == "home"


def test_beta_0_for_cluster_canonical_and_fallbacks() -> None:
    # Canonical table entries.
    assert pim._beta_0_for_cluster("residential", 3, 1) == -1.0
    assert pim._beta_0_for_cluster("workplace", 3, 2) == -0.2
    # z out of range for the table tuple → fallback default.
    assert pim._beta_0_for_cluster("residential", 3, 99) == -0.8
    assert pim._beta_0_for_cluster("workplace", 3, 99) == -0.2
    # K not in canonical table (SPEECh K=16) → midday-anchored fallback.
    assert pim._beta_0_for_cluster("residential", 16, 5) == -0.8
    assert pim._beta_0_for_cluster("workplace", 16, 5) == -0.2


# --------------------------------------------------------------------------- #
# Cohort loaders.
# --------------------------------------------------------------------------- #


def test_load_workplace_cohort_missing_file(tmp_path: Path) -> None:
    df = pim._load_workplace_cohort(tmp_path / "missing.parquet")
    assert df.empty
    assert "plug_in_hr_med" in df.columns


def test_load_workplace_cohort_roundtrip(tmp_path: Path) -> None:
    cohort = _make_workplace_cohort(20)
    cohort["ownership"] = "x"  # extra column dropped by the loader keep-list.
    p = tmp_path / "wp.parquet"
    cohort.to_parquet(p)
    out = pim._load_workplace_cohort(p)
    assert list(out.columns) == [
        "vehicle_id", "plug_in_hr_med", "dur_h_med", "avg_kw_med"
    ]
    assert len(out) == 20


def test_load_residential_evwatts_missing_files(tmp_path: Path) -> None:
    out = pim._load_residential_evwatts(
        tmp_path / "s.csv", tmp_path / "v.csv"
    )
    assert out.empty
    assert list(out.columns) == [
        "vehicle_id", "start_hour", "duration_h", "energy_kwh"
    ]


def test_load_residential_evwatts_filters_and_builds(tmp_path: Path) -> None:
    # Build a small vehicles + sessions pair that survives the candidate
    # filter (BEV passenger car), the >=130-sessions fallback (head(84)),
    # and the avg-kw band. We keep two vehicles with 5 sessions each so the
    # head(84) fallback retains both, and avg_kw ~ 4 kW (in [1.5, 11.5]).
    veh = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "vehicle_type": ["PASSENGER CAR", "PASSENGER CAR", "TRUCK"],
            "electrification_level": ["BEV300", "BEV200", "BEV100"],
        }
    )
    n_each = 6
    rows = []
    base = pd.Timestamp("2026-01-01 08:00")
    for vid in (1, 2):
        for k in range(n_each):
            rows.append(
                {
                    "vehicle_id": vid,
                    "start_datetime": (base + pd.Timedelta(hours=k)).isoformat(),
                    "duration": 2.0,
                    "energy_kwh": 8.0,  # avg_kw = 4.0
                }
            )
    sess = pd.DataFrame(rows)
    vpath = tmp_path / "veh.csv"
    spath = tmp_path / "sess.csv"
    veh.to_csv(vpath, index=False)
    sess.to_csv(spath, index=False)
    out = pim._load_residential_evwatts(spath, vpath)
    assert not out.empty
    assert set(out["vehicle_id"]) == {1, 2}
    assert (out["duration_h"] == 2.0).all()
    # start_hour reflects the LT hour-of-day (8..13).
    assert out["start_hour"].min() >= 8.0


@pytest.mark.filterwarnings("ignore:Could not infer format")
def test_load_residential_evwatts_all_bad_datetimes(tmp_path: Path) -> None:
    veh = pd.DataFrame(
        {
            "id": [1],
            "vehicle_type": ["PASSENGER CAR"],
            "electrification_level": ["BEV300"],
        }
    )
    sess = pd.DataFrame(
        {
            "vehicle_id": [1, 1, 1],
            "start_datetime": ["not-a-date", "nope", "bad"],
            "duration": [2.0, 2.0, 2.0],
            "energy_kwh": [8.0, 8.0, 8.0],
        }
    )
    vpath = tmp_path / "veh.csv"
    spath = tmp_path / "sess.csv"
    veh.to_csv(vpath, index=False)
    sess.to_csv(spath, index=False)
    out = pim._load_residential_evwatts(spath, vpath)
    assert out.empty


def test_load_residential_evwatts_out_of_kw_band(tmp_path: Path) -> None:
    # avg_kw = energy/duration far above the [1.5, 11.5] band → empty result.
    veh = pd.DataFrame(
        {
            "id": [1],
            "vehicle_type": ["PASSENGER CAR"],
            "electrification_level": ["BEV300"],
        }
    )
    base = pd.Timestamp("2026-01-01 08:00")
    rows = [
        {
            "vehicle_id": 1,
            "start_datetime": (base + pd.Timedelta(hours=k)).isoformat(),
            "duration": 1.0,
            "energy_kwh": 100.0,  # avg_kw = 100 >> 11.5
        }
        for k in range(5)
    ]
    sess = pd.DataFrame(rows)
    vpath = tmp_path / "veh.csv"
    spath = tmp_path / "sess.csv"
    veh.to_csv(vpath, index=False)
    sess.to_csv(spath, index=False)
    out = pim._load_residential_evwatts(spath, vpath)
    assert out.empty


def test_load_residential_evwatts_no_candidates(tmp_path: Path) -> None:
    veh = pd.DataFrame(
        {
            "id": [1],
            "vehicle_type": ["TRUCK"],
            "electrification_level": ["ICE"],
        }
    )
    sess = pd.DataFrame(
        {
            "vehicle_id": [1],
            "start_datetime": ["2026-01-01 08:00"],
            "duration": [2.0],
            "energy_kwh": [8.0],
        }
    )
    vpath = tmp_path / "veh.csv"
    spath = tmp_path / "sess.csv"
    veh.to_csv(vpath, index=False)
    sess.to_csv(spath, index=False)
    out = pim._load_residential_evwatts(spath, vpath)
    assert out.empty


# --------------------------------------------------------------------------- #
# BIC sweep + W9 + RFC-037.
# --------------------------------------------------------------------------- #


def test_run_bic_sweep_workplace_success_in_band() -> None:
    cohort = _make_workplace_cohort(200, drift=False)
    fit = pim.run_bic_sweep(
        cohort,
        profile_type="workplace",
        feature_cols=("plug_in_hr_med", "dur_h_med", "avg_kw_med"),
        K_sweep=pim.K_SWEEP_WORKPLACE,
        seed=SEED,
    )
    assert not fit.degenerate_flag
    assert fit.K_chosen in pim.K_SWEEP_WORKPLACE
    assert len(fit.cluster_centres_h) == fit.K_chosen
    assert len(fit.cluster_weights) == fit.K_chosen
    # Centres time-sorted ascending.
    assert list(fit.cluster_centres_h) == sorted(fit.cluster_centres_h)
    for w in fit.cluster_windows_h:
        assert w <= pim.WORKPLACE_CLUSTER_WINDOW_MAX_H + 1e-6


def test_run_bic_sweep_rfc037_centroid_drift_override() -> None:
    cohort = _make_workplace_cohort(200, drift=True)
    fit = pim.run_bic_sweep(
        cohort,
        profile_type="workplace",
        feature_cols=("plug_in_hr_med", "dur_h_med", "avg_kw_med"),
        K_sweep=pim.K_SWEEP_WORKPLACE,
        seed=SEED,
    )
    assert fit.bic_override_reason == pim.BIC_OVERRIDE_REASON_CENTROID_DRIFT
    assert fit.K_chosen == pim.EVWATTS_CENTROID_DRIFT_OVERRIDE_K
    assert fit.cluster_centres_h == pim.WINDOW_CENTRES_WORKPLACE_PRIOR[3]


def test_run_bic_sweep_residential_success() -> None:
    rng = np.random.default_rng(7)
    n = 120
    start = np.concatenate(
        [rng.normal(12.0, 1.0, 40), rng.normal(18.0, 1.0, 40),
         rng.normal(23.0, 1.0, 40)]
    )
    df = pd.DataFrame(
        {
            "start_hour": start % 24,
            "duration_h": rng.normal(10.0, 1.0, n),
            "energy_kwh": rng.normal(30.0, 3.0, n),
        }
    )
    fit = pim.run_bic_sweep(
        df,
        profile_type="residential",
        feature_cols=("start_hour", "duration_h", "energy_kwh"),
        K_sweep=pim.K_SWEEP_RESIDENTIAL,
        seed=SEED,
        cluster_window_max_h=float("inf"),
    )
    assert not fit.degenerate_flag
    assert fit.K_chosen in pim.K_SWEEP_RESIDENTIAL
    assert "GMM fit on residential cohort" in fit.note


def test_run_bic_sweep_too_small_degenerate() -> None:
    df = pd.DataFrame(
        {
            "plug_in_hr_med": [8.0, 9.0, 10.0],
            "dur_h_med": [8.0, 8.0, 8.0],
            "avg_kw_med": [6.0, 6.0, 6.0],
        }
    )
    fit = pim.run_bic_sweep(
        df,
        profile_type="workplace",
        feature_cols=("plug_in_hr_med", "dur_h_med", "avg_kw_med"),
        seed=SEED,
    )
    assert fit.degenerate_flag
    assert "degenerate" in fit.note
    assert all(np.isinf(v) for v in fit.bic_per_K.values())


def test_run_bic_sweep_missing_columns_degenerate() -> None:
    df = pd.DataFrame({"plug_in_hr_med": np.arange(20.0)})
    fit = pim.run_bic_sweep(
        df,
        profile_type="workplace",
        feature_cols=("plug_in_hr_med", "dur_h_med", "avg_kw_med"),
        seed=SEED,
    )
    assert fit.degenerate_flag


def test_run_bic_sweep_all_nan_after_filter_degenerate() -> None:
    n = 20
    df = pd.DataFrame(
        {
            "plug_in_hr_med": [np.nan] * n,
            "dur_h_med": np.full(n, 8.0),
            "avg_kw_med": np.full(n, 6.0),
        }
    )
    fit = pim.run_bic_sweep(
        df,
        profile_type="workplace",
        feature_cols=("plug_in_hr_med", "dur_h_med", "avg_kw_med"),
        seed=SEED,
    )
    assert fit.degenerate_flag


def test_run_bic_sweep_w9_rejection() -> None:
    # Very wide start-time spread inside a single mode → 95% window > 6h for
    # every K, so all K are rejected and the fit goes degenerate.
    rng = np.random.default_rng(3)
    n = 200
    df = pd.DataFrame(
        {
            "plug_in_hr_med": rng.uniform(0.0, 24.0, n),
            "dur_h_med": rng.normal(8.0, 0.2, n),
            "avg_kw_med": rng.normal(6.0, 0.2, n),
        }
    )
    fit = pim.run_bic_sweep(
        df,
        profile_type="workplace",
        feature_cols=("plug_in_hr_med", "dur_h_med", "avg_kw_med"),
        K_sweep=pim.K_SWEEP_WORKPLACE,
        seed=SEED,
        cluster_window_max_h=1.0,  # impossible to satisfy with uniform spread.
    )
    assert fit.degenerate_flag
    assert any("W9_violation" in r for r in fit.rejection_reason_per_K.values())


def test_degenerate_cluster_fit_residential_direct() -> None:
    fit = pim._degenerate_cluster_fit(
        profile_type="residential",
        K_sweep=pim.K_SWEEP_RESIDENTIAL,
        reason="unit",
    )
    assert fit.degenerate_flag
    assert fit.ambiguous_flag
    assert fit.K_chosen == pim.DEFAULT_K_RESIDENTIAL
    assert np.isnan(fit.bic_chosen)
    assert len(fit.cluster_centres_h) == fit.K_chosen


def test_cluster_window_h_edges() -> None:
    assert pim._cluster_window_h(np.array([])) == 0.0
    assert pim._cluster_window_h(np.array([8.0])) == 0.0
    assert pim._cluster_window_h(np.array([8.0, 9.0, 10.0])) == 2.0


# --------------------------------------------------------------------------- #
# _pi_for_segment.
# --------------------------------------------------------------------------- #


def test_pi_for_segment_residential_modulation() -> None:
    # t4 income bumps overnight, shared garage bumps midday & dampens overnight.
    pi = pim._pi_for_segment(3, "t4", "shared", "residential")
    assert abs(pi.sum() - 1.0) < 1e-9
    assert (pi >= 0).all()
    # K=4 and K=6 overnight indices.
    pi4 = pim._pi_for_segment(4, "t4", "curb", "residential")
    assert abs(pi4.sum() - 1.0) < 1e-9
    pi6 = pim._pi_for_segment(6, "t1", "dedicated", "residential")
    assert abs(pi6.sum() - 1.0) < 1e-9


def test_pi_for_segment_workplace_passthrough() -> None:
    pi = pim._pi_for_segment(3, "t4", "shared", "workplace")
    assert np.allclose(pi, np.array(pim.PI_BASE_WORKPLACE[3]))


def test_pi_for_segment_speech_k16_no_positional_modulation() -> None:
    base = tuple(np.full(16, 1.0 / 16))
    pi = pim._pi_for_segment(16, "t4", "shared", "residential", base_pi=base)
    assert np.allclose(pi, np.array(base))


def test_pi_for_segment_pads_and_truncates() -> None:
    short = pim._pi_for_segment(4, "t2", "dedicated", "workplace", base_pi=(0.5, 0.5))
    assert short.shape[0] == 4
    long = pim._pi_for_segment(2, "t2", "dedicated", "workplace",
                               base_pi=(0.25, 0.25, 0.25, 0.25))
    assert long.shape[0] == 2


def test_pi_for_segment_unsupported_K_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported K"):
        pim._pi_for_segment(7, "t2", "dedicated", "residential")


def test_pi_for_segment_zero_total_uniform() -> None:
    # All-zero residential weights after clip → uniform fallback.
    pi = pim._pi_for_segment(3, "t2", "dedicated", "residential",
                             base_pi=(0.0, 0.0, 0.0))
    assert np.allclose(pi, np.full(3, 1.0 / 3))


# --------------------------------------------------------------------------- #
# assign_clusters.
# --------------------------------------------------------------------------- #


def test_assign_clusters_range_and_determinism() -> None:
    fleet = _make_fleet(8)
    z1 = pim.assign_clusters(fleet, K_chosen=3, profile_type="residential",
                             master_seed=SEED)
    z2 = pim.assign_clusters(fleet, K_chosen=3, profile_type="residential",
                             master_seed=SEED)
    assert z1.dtype == np.int8
    assert np.array_equal(z1, z2)
    assert int(z1.min()) >= 1 and int(z1.max()) <= 3


def test_assign_clusters_missing_columns_defaults() -> None:
    fleet = pd.DataFrame({"ev_id": np.arange(1, 5, dtype=np.int64)})
    z = pim.assign_clusters(fleet, K_chosen=2, profile_type="workplace",
                            master_seed=SEED)
    assert len(z) == 4
    assert int(z.min()) >= 1 and int(z.max()) <= 2


# --------------------------------------------------------------------------- #
# compute_p_plug / _sigmoid / _in_cluster_window.
# --------------------------------------------------------------------------- #


def test_sigmoid_both_branches() -> None:
    assert abs(pim._sigmoid(0.0) - 0.5) < 1e-12
    assert pim._sigmoid(50.0) > 0.999
    assert pim._sigmoid(-50.0) < 0.001
    # monotone
    assert pim._sigmoid(1.0) > pim._sigmoid(-1.0)


def test_compute_p_plug_wrong_location_zero() -> None:
    p = pim.compute_p_plug(
        soc_at_arrival_frac=0.5,
        is_correct_location=False,
        in_preferred_window=True,
        dwell_h=8.0,
        is_weekend=False,
        K=3, z=2, profile_type="residential",
    )
    assert p == 0.0


def test_compute_p_plug_matches_closed_form_residential() -> None:
    p = pim.compute_p_plug(
        soc_at_arrival_frac=0.6,
        is_correct_location=True,
        in_preferred_window=True,
        dwell_h=14.0,
        is_weekend=False,
        K=3, z=2, profile_type="residential",
    )
    x = -0.5 + 1.8 * 0.4 + 2.5 + 1.5 + 0.6 * float(np.log(14.0))
    assert abs(p - 1.0 / (1.0 + float(np.exp(-x)))) < 1e-9


def test_compute_p_plug_dwell_floor() -> None:
    # Negative/zero dwell clamps to the log floor; no NaN.
    p = pim.compute_p_plug(
        soc_at_arrival_frac=0.5,
        is_correct_location=True,
        in_preferred_window=False,
        dwell_h=0.0,
        is_weekend=False,
        K=3, z=1, profile_type="workplace",
    )
    assert 0.0 < p < 1.0


def test_in_cluster_window_basic_and_irregular() -> None:
    centres = (8.0, 18.0, 23.0)
    assert pim._in_cluster_window(8.5, z=1, centres=centres, K=3,
                                  profile_type="workplace") is True
    assert pim._in_cluster_window(15.0, z=1, centres=centres, K=3,
                                  profile_type="workplace") is False
    # z out of range → False.
    assert pim._in_cluster_window(8.0, z=9, centres=centres, K=3,
                                  profile_type="workplace") is False
    # Residential irregular cluster (K=4 z=4) is always in-window.
    for h in (0.0, 11.0, 23.5):
        assert pim._in_cluster_window(
            h, z=4, centres=pim.WINDOW_CENTRES_RESIDENTIAL[4], K=4,
            profile_type="residential") is True


def test_in_cluster_window_midnight_wrap() -> None:
    # centre 23.5, half-window 2h → 0.5 LT is within window via mod-24 wrap.
    assert pim._in_cluster_window(0.5, z=1, centres=(23.5,), K=1,
                                  profile_type="workplace") is True


# --------------------------------------------------------------------------- #
# iterate_bernoulli + aggregate_plug_in_rate + schema.
# --------------------------------------------------------------------------- #


def test_iterate_bernoulli_empty_mobility() -> None:
    out = pim.iterate_bernoulli(
        pd.DataFrame({"ev_id": [], "start_ts": [], "end_ts": [], "dest_type": []}),
        _make_fleet(2),
        np.array([1, 1], dtype=np.int8),
        K_chosen=3,
        profile_type="residential",
        cluster_centres_h=(12.5, 18.5, 23.5),
        region_tz="America/Los_Angeles",
        master_seed=SEED,
    )
    assert out.empty
    assert "event_id" in out.columns
    assert "plug_in" in out.columns


def test_iterate_bernoulli_residential_produces_events() -> None:
    fleet = _make_fleet(6)
    mob = _make_mobility("residential", 6)
    z = pim.assign_clusters(fleet, K_chosen=3, profile_type="residential",
                            master_seed=SEED)
    events = pim.iterate_bernoulli(
        mob, fleet, z,
        K_chosen=3, profile_type="residential",
        cluster_centres_h=(12.5, 18.5, 23.5),
        region_tz="America/Los_Angeles",
        master_seed=SEED,
    )
    assert len(events) > 0
    assert (events["location_type"].astype(str) == "home").all()
    assert str(events["arrival_ts"].dtype) == "datetime64[ns, UTC]"
    assert (events["soc_target_pct"] > 0).all()
    rate = pim.aggregate_plug_in_rate(mob, events, "residential")
    assert 0.0 < rate <= 1.0


def test_iterate_bernoulli_workplace_truncated_beta_target() -> None:
    fleet = _make_fleet(6)
    mob = _make_mobility("workplace", 6)
    z = pim.assign_clusters(fleet, K_chosen=3, profile_type="workplace",
                            master_seed=SEED)
    # Centre arrivals in-window: workplace arrival LT is morning ~09:00.
    events = pim.iterate_bernoulli(
        mob, fleet, z,
        K_chosen=3, profile_type="workplace",
        cluster_centres_h=(7.5, 8.75, 10.0),
        region_tz="America/Los_Angeles",
        master_seed=SEED,
    )
    if len(events):
        assert (events["location_type"].astype(str) == "work").all()
        # Workplace SoC target lives in [0.50, 0.95].
        assert (events["soc_target_pct"] >= 0.50 - 1e-6).all()
        assert (events["soc_target_pct"] <= 0.95 + 1e-6).all()


def test_iterate_bernoulli_nonpositive_battery_uses_fallback_B() -> None:
    # B_kwh <= 0 / non-finite → fallback B=70 path exercised; also a mobility
    # EV not present in cluster_z lookup falls back to z=1.
    fleet = _make_fleet(6)
    fleet["B_kwh"] = [0.0, -5.0, np.nan, 70.0, 70.0, 70.0]
    mob = _make_mobility("residential", 6)
    z = np.array([1, 1, 1, 1, 1, 1], dtype=np.int8)
    events = pim.iterate_bernoulli(
        mob, fleet, z,
        K_chosen=3, profile_type="residential",
        cluster_centres_h=(12.5, 18.5, 23.5),
        region_tz="America/Los_Angeles",
        master_seed=SEED,
    )
    assert isinstance(events, pd.DataFrame)


def test_iterate_bernoulli_ev_missing_from_fleet_index() -> None:
    # Mobility EV 99 is absent from the fleet → b_kwh.loc raises KeyError and
    # the B=70 fallback fires; cluster_z lookup also defaults to z=1.
    fleet = _make_fleet(2)  # ev_id 1, 2 only.
    mob = _make_mobility("residential", 2)
    extra = _make_mobility("residential", 1).copy()
    extra["ev_id"] = 99
    mob = pd.concat([mob, extra], ignore_index=True)
    z = np.array([1, 1], dtype=np.int8)
    events = pim.iterate_bernoulli(
        mob, fleet, z,
        K_chosen=3, profile_type="residential",
        cluster_centres_h=(12.5, 18.5, 23.5),
        region_tz="America/Los_Angeles",
        master_seed=SEED,
    )
    assert isinstance(events, pd.DataFrame)


def test_iterate_bernoulli_negative_dwell_clamped() -> None:
    # An arrival whose next trip starts *before* it (clock inconsistency)
    # exercises the dwell_h < 0 → 0.0 clamp.
    fleet = _make_fleet(1)
    # Sorted by start_ts: the home arrival (start 01:00) is followed by a trip
    # that starts at 01:30 but ends at 00:30 — the recorded "departure"
    # (next trip start 01:30) minus arrival end (02:00) is negative.
    mob = pd.DataFrame(
        {
            "ev_id": [1, 1],
            "start_ts": [
                pd.Timestamp("2026-05-20 01:00", tz="UTC"),
                pd.Timestamp("2026-05-20 01:30", tz="UTC"),
            ],
            "end_ts": [
                pd.Timestamp("2026-05-20 02:00", tz="UTC"),  # arrival end.
                pd.Timestamp("2026-05-20 03:00", tz="UTC"),
            ],
            "dest_type": ["home", "other"],
            "kwh_consumed": [3.0, 1.0],
            "is_weekend": [False, False],
        }
    )
    events = pim.iterate_bernoulli(
        mob, fleet, np.array([1], dtype=np.int8),
        K_chosen=3, profile_type="residential",
        cluster_centres_h=(12.5, 18.5, 23.5),
        region_tz="America/Los_Angeles",
        master_seed=SEED,
    )
    assert isinstance(events, pd.DataFrame)


def test_iterate_bernoulli_naive_timestamps_localised() -> None:
    # tz-naive timestamps should be localised to UTC without error.
    fleet = _make_fleet(3)
    mob = _make_mobility("residential", 3)
    mob["start_ts"] = mob["start_ts"].dt.tz_localize(None)
    mob["end_ts"] = mob["end_ts"].dt.tz_localize(None)
    z = np.array([2, 2, 2], dtype=np.int8)
    events = pim.iterate_bernoulli(
        mob, fleet, z,
        K_chosen=3, profile_type="residential",
        cluster_centres_h=(12.5, 18.5, 23.5),
        region_tz="America/Los_Angeles",
        master_seed=SEED,
    )
    assert isinstance(events, pd.DataFrame)


def test_iterate_bernoulli_weekend_centres_predicate() -> None:
    # Supplying weekend_cluster_centres_h + a weekend arrival routes the
    # predicate through the weekend centre table (SPEECh K=16 path).
    fleet = _make_fleet(4)
    mob = _make_mobility("residential", 4)
    mob["is_weekend"] = True
    z = pim.assign_clusters(fleet, K_chosen=3, profile_type="residential",
                            master_seed=SEED)
    events = pim.iterate_bernoulli(
        mob, fleet, z,
        K_chosen=3, profile_type="residential",
        cluster_centres_h=(12.5, 18.5, 23.5),
        weekend_cluster_centres_h=(2.0, 18.5, 23.5),
        region_tz="America/Los_Angeles",
        master_seed=SEED,
    )
    assert isinstance(events, pd.DataFrame)


def test_iterate_bernoulli_all_draws_reject_returns_empty(monkeypatch) -> None:
    # Force p_plug=0 for every eligible arrival → every Bernoulli draw fails
    # (the `not plugged: continue` branch) and the loop yields no rows, so the
    # function returns the empty-events frame.
    monkeypatch.setattr(pim, "compute_p_plug", lambda **kw: 0.0)
    fleet = _make_fleet(4)
    mob = _make_mobility("residential", 4)
    z = pim.assign_clusters(fleet, K_chosen=3, profile_type="residential",
                            master_seed=SEED)
    events = pim.iterate_bernoulli(
        mob, fleet, z,
        K_chosen=3, profile_type="residential",
        cluster_centres_h=(12.5, 18.5, 23.5),
        region_tz="America/Los_Angeles",
        master_seed=SEED,
    )
    assert events.empty


def test_aggregate_plug_in_rate_zero_arrivals() -> None:
    mob = pd.DataFrame({"dest_type": ["other", "other"]})
    events = pim._empty_events_df()
    assert pim.aggregate_plug_in_rate(mob, events, "residential") == 0.0


def test_empty_events_df_schema() -> None:
    df = pim._empty_events_df()
    assert df.empty
    for col in ("event_id", "ev_id", "arrival_ts", "departure_ts", "cluster_z",
                "location_type", "soc_target_pct", "profile_type_event",
                "p_plug", "dwell_h", "plug_in", "location"):
        assert col in df.columns


def test_cast_events_to_schema_with_existing_plugin_and_location() -> None:
    df = pd.DataFrame(
        {
            "event_id": [0],
            "ev_id": [1],
            "arrival_ts": [pd.Timestamp("2026-05-20 02:00", tz="UTC")],
            "departure_ts": [pd.Timestamp("2026-05-20 14:00", tz="UTC")],
            "cluster_z": [2],
            "location_type": ["home"],
            "soc_target_pct": [0.85],
            "profile_type_event": ["residential"],
            "p_plug": [0.9],
            "dwell_h": [12.0],
            "plug_in": [True],
            "location": ["home"],
        }
    )
    out = pim._cast_events_to_schema(df)
    assert out["plug_in"].dtype == bool
    assert str(out["arrival_ts"].dtype) == "datetime64[ns, UTC]"
    assert out["ev_id"].dtype == np.int32


def test_cast_events_to_schema_naive_timestamps() -> None:
    df = pd.DataFrame(
        {
            "event_id": [0],
            "ev_id": [1],
            "arrival_ts": [pd.Timestamp("2026-05-20 02:00")],
            "departure_ts": [pd.Timestamp("2026-05-20 14:00")],
            "cluster_z": [1],
            "location_type": ["work"],
            "soc_target_pct": [0.7],
            "profile_type_event": ["workplace"],
            "p_plug": [0.8],
            "dwell_h": [12.0],
        }
    )
    out = pim._cast_events_to_schema(df)
    assert str(out["arrival_ts"].dtype) == "datetime64[ns, UTC]"
    assert out["plug_in"].all()


# --------------------------------------------------------------------------- #
# SPEECh loaders (bundled JSON present in the wheel).
# --------------------------------------------------------------------------- #


def test_bundled_speech_json_path_points_into_package() -> None:
    p = pim._bundled_speech_json_path("residential")
    assert p.name == "speech_k16_residential.json"
    assert p.is_file()


def test_resolve_speech_json_path_falls_back_to_bundled(tmp_path: Path) -> None:
    # data_root has no local override → bundled path returned.
    p = pim._resolve_speech_json_path("workplace", tmp_path)
    assert p == pim._bundled_speech_json_path("workplace")


def test_resolve_speech_json_path_invalid_profile_raises() -> None:
    with pytest.raises(ValueError, match="profile_type must be"):
        pim._resolve_speech_json_path("garage", None)


def test_resolve_speech_json_path_local_override(tmp_path: Path) -> None:
    local = tmp_path / pim.SPEECH_K16_RESIDENTIAL_REL
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_text("{}", encoding="utf-8")
    p = pim._resolve_speech_json_path("residential", tmp_path)
    assert p == local


def test_load_speech_k16_params_bundled_residential() -> None:
    sp = pim.load_speech_k16_params("residential")
    assert sp.profile_type == "residential"
    assert len(sp.P_G) == pim.SPEECH_K_TOTAL
    assert abs(sum(sp.P_G) - 1.0) < 1e-9
    assert len(sp.weekday_means_h) == len(sp.groups_present_weekday)
    assert sp.source_license  # populated from bundled JSON source block.


def test_load_speech_k16_params_bundled_workplace() -> None:
    sp = pim.load_speech_k16_params("workplace")
    assert sp.profile_type == "workplace"
    assert len(sp.groups_present_weekday) >= 1


def test_load_speech_k16_params_missing_raises(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        pim, "_resolve_speech_json_path",
        lambda profile_type, data_root: tmp_path / "nope.json",
    )
    with pytest.raises(FileNotFoundError, match="SPEECh K=16 parameters not found"):
        pim.load_speech_k16_params("residential")


def _write_speech_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _minimal_speech_payload(profile_type: str = "residential") -> dict:
    # Two present groups on each day; group_id 0 and 1.
    def group(gid: int, mean_h: float) -> dict:
        m = mean_h * 3600.0
        cov = (1.0 * 3600.0) ** 2
        return {
            "group_id": gid,
            "means": [[m, 0.0, 0.0]],
            "covariances": [[[cov, 0, 0], [0, 1, 0], [0, 0, 1]]],
            "weights": [1.0],
        }

    pg = [0.0] * 16
    pg[0] = 0.4
    pg[1] = 0.6
    return {
        "profile_type": profile_type,
        "P_G": pg,
        "weekday": {
            "groups_present": [0, 1],
            "groups": [group(0, 8.0), group(1, 18.0)],
        },
        "weekend": {
            "groups_present": [1, 2],
            "groups": [group(1, 18.0), group(2, 10.0)],
        },
        "source": {"doi": "x", "url": "y", "license": "CC-BY-4.0"},
    }


def test_load_speech_k16_params_profile_mismatch(tmp_path: Path, monkeypatch) -> None:
    payload = _minimal_speech_payload("workplace")
    p = tmp_path / "s.json"
    _write_speech_json(p, payload)
    monkeypatch.setattr(pim, "_resolve_speech_json_path",
                        lambda profile_type, data_root: p)
    with pytest.raises(ValueError, match="profile_type mismatch"):
        pim.load_speech_k16_params("residential")


def test_load_speech_k16_params_bad_pg_length(tmp_path: Path, monkeypatch) -> None:
    payload = _minimal_speech_payload("residential")
    payload["P_G"] = [0.5, 0.5]
    p = tmp_path / "s.json"
    _write_speech_json(p, payload)
    monkeypatch.setattr(pim, "_resolve_speech_json_path",
                        lambda profile_type, data_root: p)
    with pytest.raises(ValueError, match="P_G length"):
        pim.load_speech_k16_params("residential")


def test_load_speech_k16_params_no_groups_present(tmp_path: Path, monkeypatch) -> None:
    # P_G mass only on group 5 but no day declares group 5 present → renorm fails.
    payload = _minimal_speech_payload("residential")
    pg = [0.0] * 16
    pg[5] = 1.0
    payload["P_G"] = pg
    p = tmp_path / "s.json"
    _write_speech_json(p, payload)
    monkeypatch.setattr(pim, "_resolve_speech_json_path",
                        lambda profile_type, data_root: p)
    with pytest.raises(ValueError, match="cannot renormalise P_G"):
        pim.load_speech_k16_params("residential")


def test_load_speech_k16_params_minimal_roundtrip(tmp_path: Path, monkeypatch) -> None:
    payload = _minimal_speech_payload("residential")
    p = tmp_path / "s.json"
    _write_speech_json(p, payload)
    monkeypatch.setattr(pim, "_resolve_speech_json_path",
                        lambda profile_type, data_root: p)
    sp = pim.load_speech_k16_params("residential")
    assert sp.groups_present_weekday == (0, 1)
    assert sp.groups_present_weekend == (1, 2)
    # weekday mean of group 0 converts back to ~8.0 hours.
    assert abs(sp.weekday_means_h[0][0] - 8.0) < 1e-6
    # P_G renormalised over union {0,1,2} = {0.4, 0.6, 0.0} / 1.0.
    assert abs(sp.P_G[0] - 0.4) < 1e-9
    assert abs(sp.P_G[1] - 0.6) < 1e-9


def test_speech_group_arrays_inconsistent_raises() -> None:
    block = {
        "groups_present": [0, 7],
        "groups": [{"group_id": 0, "means": [[0, 0, 0]],
                    "covariances": [[[1, 0, 0], [0, 1, 0], [0, 0, 1]]],
                    "weights": [1.0]}],
    }
    with pytest.raises(ValueError, match="missing from groups"):
        pim._speech_group_arrays(block)


def test_speech_centres_per_group_circular_and_fallbacks() -> None:
    # Empty group → 12.0 fallback.
    centres = pim._speech_centres_per_group(((),), ((),))
    assert centres == (12.0,)
    # Zero-weight group → 12.0 fallback.
    centres2 = pim._speech_centres_per_group(((8.0,),), ((0.0,),))
    assert centres2 == (12.0,)
    # Midnight-straddling components collapse near 24/0, not 12.
    centres3 = pim._speech_centres_per_group(((23.5, 0.5),), ((0.5, 0.5),))
    h = centres3[0]
    assert h < 1.0 or h > 23.0


# --------------------------------------------------------------------------- #
# _build_speech_cluster_fit + _speech_weekend_centres_from_fit.
# --------------------------------------------------------------------------- #


def _minimal_speech_params() -> pim.SpeechParams:
    pg = [0.0] * 16
    pg[0], pg[1], pg[2] = 0.4, 0.4, 0.2
    return pim.SpeechParams(
        profile_type="residential",
        P_G=tuple(pg),
        weekday_means_h=((8.0,), (18.0,)),
        weekday_covs_h=((1.0,), (1.0,)),
        weekday_weights=((1.0,), (1.0,)),
        weekend_means_h=((18.0,), (10.0,)),
        weekend_covs_h=((1.0,), (1.0,)),
        weekend_weights=((1.0,), (1.0,)),
        groups_present_weekday=(0, 1),
        groups_present_weekend=(1, 2),
    )


def test_build_speech_cluster_fit_union_and_nan_centre() -> None:
    sp = _minimal_speech_params()
    fit = pim._build_speech_cluster_fit("residential", sp)
    # Union of {0,1} and {1,2} = {0,1,2} → K=3.
    assert fit.K_chosen == 3
    assert fit.speech_params is sp
    assert fit.axes == ("start_h_LT",)
    assert len(fit.cluster_centres_h) == 3
    assert abs(sum(fit.cluster_weights) - 1.0) < 1e-9
    # Group 2 is weekend-only → weekday centre falls back to weekend centre.
    assert not np.isnan(fit.cluster_centres_h[2])


def test_speech_weekend_centres_from_fit() -> None:
    sp = _minimal_speech_params()
    fit = pim._build_speech_cluster_fit("residential", sp)
    we = pim._speech_weekend_centres_from_fit(fit)
    assert we is not None
    assert len(we) == fit.K_chosen
    # Group 0 absent on weekend → NaN centre.
    assert np.isnan(we[0])


def test_speech_weekend_centres_none_for_bic_fit() -> None:
    fit = pim._degenerate_cluster_fit(
        profile_type="residential", K_sweep=(3,), reason="x"
    )
    assert pim._speech_weekend_centres_from_fit(fit) is None


def test_build_speech_cluster_fit_zero_weights_uniform() -> None:
    pg = [0.0] * 16  # no mass on present groups → uniform fallback.
    sp = pim.SpeechParams(
        profile_type="residential",
        P_G=tuple(pg),
        weekday_means_h=((8.0,),),
        weekday_covs_h=((1.0,),),
        weekday_weights=((1.0,),),
        weekend_means_h=((8.0,),),
        weekend_covs_h=((1.0,),),
        weekend_weights=((1.0,),),
        groups_present_weekday=(0,),
        groups_present_weekend=(0,),
    )
    fit = pim._build_speech_cluster_fit("residential", sp)
    assert abs(sum(fit.cluster_weights) - 1.0) < 1e-9


# --------------------------------------------------------------------------- #
# fit_and_assign_plug_ins (both cluster_sources) + error branches.
# --------------------------------------------------------------------------- #


def test_fit_and_assign_speech_default_path() -> None:
    fleet = _make_fleet(6)
    mob = _make_mobility("residential", 6)
    aug, events, fit = pim.fit_and_assign_plug_ins(
        fleet=fleet, mobility=mob, region="bay_area",
        profile_type="residential", seed=SEED,
    )
    assert fit.speech_params is not None
    assert "cluster_z" in aug.columns
    assert "cluster_k" in aug.columns
    assert (aug["cluster_z"] >= 1).all()
    assert (aug["cluster_z"] <= fit.K_chosen).all()
    assert isinstance(events, pd.DataFrame)


def test_fit_and_assign_bic_sweep_residential_no_data(tmp_path: Path) -> None:
    # Residential BIC path with missing EVWatts CSVs → degenerate fallback,
    # base_pi=None branch, cluster_centres padding.
    fleet = _make_fleet(6)
    mob = _make_mobility("residential", 6)
    aug, events, fit = pim.fit_and_assign_plug_ins(
        fleet=fleet, mobility=mob, region="bay_area",
        profile_type="residential", cluster_source="bic_sweep",
        sessions_csv=tmp_path / "no_s.csv", vehicles_csv=tmp_path / "no_v.csv",
        seed=SEED,
    )
    assert fit.speech_params is None
    assert fit.degenerate_flag
    assert fit.K_chosen in pim.K_SWEEP_RESIDENTIAL


def test_fit_and_assign_bic_sweep_workplace_with_cohort(tmp_path: Path) -> None:
    fleet = _make_fleet(6)
    mob = _make_mobility("workplace", 6)
    cohort = _make_workplace_cohort(200, drift=False)
    cpath = tmp_path / "wp.parquet"
    cohort.to_parquet(cpath)
    aug, events, fit = pim.fit_and_assign_plug_ins(
        fleet=fleet, mobility=mob, region="bay_area",
        profile_type="workplace", cluster_source="bic_sweep",
        cohort_parquet=cpath, seed=SEED,
    )
    assert fit.speech_params is None
    assert fit.K_chosen in pim.K_SWEEP_WORKPLACE
    assert "bic_chosen" in aug.columns


def test_run_bic_sweep_gmm_fit_error_branch(monkeypatch) -> None:
    # Force every GMM fit to raise → all K rejected with fit_error reasons →
    # degenerate ClusterFit (exercises the per-K exception handler).
    def _boom(X, K, seed):
        raise RuntimeError("synthetic GMM failure")

    monkeypatch.setattr(pim, "_fit_gmm_for_K", _boom)
    cohort = _make_workplace_cohort(200, drift=False)
    fit = pim.run_bic_sweep(
        cohort, profile_type="workplace",
        feature_cols=("plug_in_hr_med", "dur_h_med", "avg_kw_med"),
        K_sweep=pim.K_SWEEP_WORKPLACE, seed=SEED,
    )
    assert fit.degenerate_flag
    assert any("fit_error" in r for r in fit.rejection_reason_per_K.values())


def test_run_bic_sweep_ambiguous_defaults_to_profile_k(monkeypatch) -> None:
    # Return near-identical BIC for every K so the relative gap < 1% and the
    # sweep falls back to the profile default K (residential default = 3).
    n = 60
    rng = np.random.default_rng(11)

    def _flat_fit(X, K, seed):
        bic = 100.0 + 1e-4 * K  # rel-gap << 1% between any pair.
        centres = np.column_stack(
            [
                np.linspace(12.0, 13.0, K),
                np.full(K, 10.0),
                np.full(K, 30.0),
            ]
        )
        weights = np.full(K, 1.0 / K)
        labels = rng.integers(0, K, size=len(X))
        return bic, centres, weights, labels

    monkeypatch.setattr(pim, "_fit_gmm_for_K", _flat_fit)
    df = pd.DataFrame(
        {
            "start_hour": rng.normal(12.5, 0.2, n),
            "duration_h": rng.normal(10.0, 0.2, n),
            "energy_kwh": rng.normal(30.0, 0.2, n),
        }
    )
    fit = pim.run_bic_sweep(
        df, profile_type="residential",
        feature_cols=("start_hour", "duration_h", "energy_kwh"),
        K_sweep=pim.K_SWEEP_RESIDENTIAL, seed=SEED,
        cluster_window_max_h=float("inf"),
    )
    assert fit.ambiguous_flag
    assert fit.K_chosen == pim.DEFAULT_K_RESIDENTIAL


def test_fit_and_assign_bic_sweep_workplace_default_cohort_missing() -> None:
    # cohort_parquet=None → default repo-relative path (absent) → degenerate.
    fleet = _make_fleet(4)
    mob = _make_mobility("workplace", 4)
    _, _, fit = pim.fit_and_assign_plug_ins(
        fleet=fleet, mobility=mob, region="bay_area",
        profile_type="workplace", cluster_source="bic_sweep", seed=SEED,
    )
    assert fit.degenerate_flag
    assert fit.K_chosen == pim.DEFAULT_K_WORKPLACE


def test_fit_and_assign_bic_sweep_residential_default_csvs() -> None:
    # sessions_csv/vehicles_csv=None → default repo-relative CSV paths
    # (absent) → degenerate residential fallback.
    fleet = _make_fleet(4)
    mob = _make_mobility("residential", 4)
    _, _, fit = pim.fit_and_assign_plug_ins(
        fleet=fleet, mobility=mob, region="bay_area",
        profile_type="residential", cluster_source="bic_sweep", seed=SEED,
    )
    assert fit.speech_params is None
    assert fit.K_chosen in pim.K_SWEEP_RESIDENTIAL


def test_fit_and_assign_accepts_region_object() -> None:
    from pev_synth.regions import Region
    fleet = _make_fleet(4)
    mob = _make_mobility("residential", 4)
    region = Region.from_name("bay_area")
    _, _, fit = pim.fit_and_assign_plug_ins(
        fleet=fleet, mobility=mob, region=region,
        profile_type="residential", seed=SEED,
    )
    assert fit.K_chosen >= 1


def test_fit_and_assign_bad_region_type() -> None:
    with pytest.raises(TypeError, match="region must be Region or str"):
        pim.fit_and_assign_plug_ins(
            fleet=_make_fleet(2), mobility=_make_mobility("residential", 2),
            region=123, profile_type="residential",
        )


def test_fit_and_assign_bad_profile_type() -> None:
    with pytest.raises(ValueError, match="profile_type must be"):
        pim.fit_and_assign_plug_ins(
            fleet=_make_fleet(2), mobility=_make_mobility("residential", 2),
            region="bay_area", profile_type="garage",
        )


def test_fit_and_assign_bad_cluster_source() -> None:
    with pytest.raises(ValueError, match="cluster_source must be"):
        pim.fit_and_assign_plug_ins(
            fleet=_make_fleet(2), mobility=_make_mobility("residential", 2),
            region="bay_area", profile_type="residential",
            cluster_source="magic",
        )


# --------------------------------------------------------------------------- #
# Persistence + meta + CLI.
# --------------------------------------------------------------------------- #


def test_write_plug_in_events_and_fleet(tmp_path: Path) -> None:
    fleet = _make_fleet(6)
    mob = _make_mobility("residential", 6)
    aug, events, _ = pim.fit_and_assign_plug_ins(
        fleet=fleet, mobility=mob, region="bay_area",
        profile_type="residential", seed=SEED,
    )
    ev_path = pim.write_plug_in_events(events, tmp_path / "ev.parquet")
    fl_path = pim.write_fleet(aug, tmp_path / "fleet.parquet")
    assert ev_path.is_file()
    assert fl_path.is_file()
    reloaded = pd.read_parquet(ev_path)
    assert list(reloaded["ev_id"]) == sorted(reloaded["ev_id"])


def test_write_bic_sweep_results_workplace(tmp_path: Path) -> None:
    cohort = _make_workplace_cohort(200, drift=False)
    fit = pim.run_bic_sweep(
        cohort, profile_type="workplace",
        feature_cols=("plug_in_hr_med", "dur_h_med", "avg_kw_med"),
        K_sweep=pim.K_SWEEP_WORKPLACE, seed=SEED,
    )
    out = tmp_path / "bic_sweep_workplace.json"
    p = pim.write_bic_sweep_results(fit, out)
    assert p.is_file()
    payload = json.loads(p.read_text())
    assert payload["profile_type"] == "workplace"
    assert payload["K_chosen"] == fit.K_chosen


def test_write_bic_sweep_results_renames_residential_flavoured_path(
    tmp_path: Path,
) -> None:
    cohort = _make_workplace_cohort(200, drift=False)
    fit = pim.run_bic_sweep(
        cohort, profile_type="workplace",
        feature_cols=("plug_in_hr_med", "dur_h_med", "avg_kw_med"),
        K_sweep=pim.K_SWEEP_WORKPLACE, seed=SEED,
    )
    out = tmp_path / "bic_sweep_residential.json"
    p = pim.write_bic_sweep_results(fit, out)
    assert p.name == "bic_sweep_workplace.json"
    assert p.is_file()


def test_write_bic_sweep_results_residential_noop(tmp_path: Path) -> None:
    fit = pim._degenerate_cluster_fit(
        profile_type="residential", K_sweep=(3,), reason="x"
    )
    out = tmp_path / "bic_sweep_workplace.json"
    p = pim.write_bic_sweep_results(fit, out)
    assert not p.is_file()  # no-op for residential.


def test_json_default_helper() -> None:
    assert pim._json_default(np.int64(3)) == 3
    assert pim._json_default(np.float64(1.5)) == 1.5
    assert pim._json_default(np.float64("inf")) is None
    assert pim._json_default(np.array([1, 2])) == [1, 2]
    with pytest.raises(TypeError, match="unserialisable"):
        pim._json_default(object())


def test_update_meta_with_m5_residential_speech(tmp_path: Path) -> None:
    sp = pim.load_speech_k16_params("residential")
    fit = pim._build_speech_cluster_fit("residential", sp)
    from pev_synth.regions import Region
    meta_path = tmp_path / "meta.json"
    p = pim.update_meta_with_m5(
        meta_path, profile_type="residential",
        region=Region.from_name("bay_area"),
        cluster_fit=fit,
        cluster_size_distribution={1: 3, 2: 3},
        aggregate_plug_in_rate_value=0.8,
        cohort_size=0,
    )
    meta = json.loads(p.read_text())
    block = meta["plug_in_model"]
    assert block["cluster_source"] == "speech_k16"
    assert "speech_k16_provenance" in block
    assert block["profile_type"] == "residential"


def test_update_meta_with_m5_workplace_bic(tmp_path: Path) -> None:
    cohort = _make_workplace_cohort(200, drift=False)
    fit = pim.run_bic_sweep(
        cohort, profile_type="workplace",
        feature_cols=("plug_in_hr_med", "dur_h_med", "avg_kw_med"),
        K_sweep=pim.K_SWEEP_WORKPLACE, seed=SEED,
    )
    from pev_synth.regions import Region
    # Pre-seed meta.json so the is_file() branch is exercised.
    meta_path = tmp_path / "meta.json"
    meta_path.write_text(json.dumps({"existing": True}), encoding="utf-8")
    p = pim.update_meta_with_m5(
        meta_path, profile_type="workplace",
        region=Region.from_name("bay_area"),
        cluster_fit=fit,
        cluster_size_distribution={1: 2, 2: 2, 3: 2},
        aggregate_plug_in_rate_value=0.85,
        cohort_size=200,
    )
    meta = json.loads(p.read_text())
    assert meta["existing"] is True
    block = meta["plug_in_model"]
    assert block["cluster_source"] == "bic_sweep"
    assert "workplace_cohort_caveat" in block
    assert "W9_invariant" in block
    assert "workplace" in meta
    assert meta["workplace"]["selected"] == fit.K_chosen


def test_default_paths_layout(tmp_path: Path) -> None:
    paths = pim._default_paths(tmp_path, "bay_area", "residential")
    assert paths.fleet_in.name == "fleet.parquet"
    assert paths.bic_sweep_out.name == "bic_sweep_workplace.json"
    assert "residential_ev_synth" in str(paths.fleet_out)


def test_run_missing_fleet_raises(tmp_path: Path) -> None:
    paths = pim._default_paths(tmp_path, "bay_area", "residential")
    with pytest.raises(FileNotFoundError, match="fleet.parquet not found"):
        pim.run(paths, region="bay_area", profile_type="residential")


def test_run_missing_mobility_raises(tmp_path: Path) -> None:
    paths = pim._default_paths(tmp_path, "bay_area", "residential")
    paths.fleet_in.parent.mkdir(parents=True, exist_ok=True)
    _make_fleet(6).to_parquet(paths.fleet_in)
    with pytest.raises(FileNotFoundError, match="mobility.parquet not found"):
        pim.run(paths, region="bay_area", profile_type="residential")


def test_run_end_to_end_residential(tmp_path: Path) -> None:
    paths = pim._default_paths(tmp_path, "bay_area", "residential")
    paths.fleet_in.parent.mkdir(parents=True, exist_ok=True)
    _make_fleet(6).to_parquet(paths.fleet_in)
    _make_mobility("residential", 6).to_parquet(paths.mobility_in)
    summary = pim.run(paths, region="bay_area", profile_type="residential",
                      master_seed=SEED)
    assert summary["region"] == "bay_area"
    assert summary["profile_type"] == "residential"
    assert summary["K_chosen"] >= 1
    assert paths.fleet_out.is_file()
    assert paths.plug_in_events_out.is_file()
    assert paths.meta_out.is_file()
    # Residential MUST NOT emit a bic_sweep artefact.
    assert not paths.bic_sweep_out.is_file()


def test_run_end_to_end_workplace_writes_bic_sweep(tmp_path: Path) -> None:
    paths = pim._default_paths(tmp_path, "bay_area", "workplace")
    paths.fleet_in.parent.mkdir(parents=True, exist_ok=True)
    _make_fleet(6).to_parquet(paths.fleet_in)
    _make_mobility("workplace", 6).to_parquet(paths.mobility_in)
    # Lay down the workplace cohort so `run` reads cohort_size from it.
    paths.workplace_cohort_parquet.parent.mkdir(parents=True, exist_ok=True)
    _make_workplace_cohort(50).to_parquet(paths.workplace_cohort_parquet)
    # Workplace SPEECh path (default cluster_source).
    summary = pim.run(paths, region="bay_area", profile_type="workplace",
                      master_seed=SEED)
    assert summary["profile_type"] == "workplace"
    assert paths.bic_sweep_out.is_file()


def test_main_cli_smoke(tmp_path: Path, monkeypatch) -> None:
    # Point repo_root at tmp_path and lay down the synthetic cache so the
    # CLI driver runs end-to-end.
    paths = pim._default_paths(tmp_path, "bay_area", "residential")
    paths.fleet_in.parent.mkdir(parents=True, exist_ok=True)
    _make_fleet(6).to_parquet(paths.fleet_in)
    _make_mobility("residential", 6).to_parquet(paths.mobility_in)
    rc = pim.main([
        "--region", "bay_area",
        "--profile-type", "residential",
        "--repo-root", str(tmp_path),
        "--seed", str(SEED),
    ])
    assert rc == 0
    assert paths.plug_in_events_out.is_file()
