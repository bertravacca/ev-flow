"""Coverage-maximising tests for ``pev_synth.cache_regen``.

The cache_regen module orchestrates the M2..M7 pipeline end-to-end. The
heavy NHTS micro-data tree is intentionally absent in CI, so every test
here is fully data-independent: the per-stage entry points
(``sample_fleet`` / ``match_donors`` / ``build_mobility`` /
``fit_and_assign_plug_ins`` / ``compute_sessions`` / ``build_plug_status``
and the M5 writers) are monkeypatched onto the ``cache_regen`` namespace
with lightweight stubs that emit tiny synthetic parquet/JSON artifacts
into a ``tmp_path`` data root.

Covered seams:

* ``regenerate_cache`` happy path (writes 6 parquets + meta.json), and its
  unknown-region / unknown-profile_type / exception-capture branches.
* ``regenerate_batch`` replicate expansion, time-budget DEFERRAL, and
  progress-log writing.
* ``_out_dir`` flat vs. ``replicates/r{id}/`` layout, default vs. custom
  repo_root.
* ``run_cross_region_checks`` C1-C5 over synthetic caches + markdown render.
* CLI ``main`` for ``one`` / ``batch`` / ``audit`` and the helper functions.

Seeds are fixed (``MASTER_SEED``) wherever an RNG would otherwise be
involved; the stubs are deterministic so all assertions are on concrete
values / shapes / exceptions.
"""

from __future__ import annotations

import json
import types
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from pev_synth import cache_regen
from pev_synth.regions import Region

SEED = cache_regen.MASTER_SEED


# ---------------------------------------------------------------------------
# Synthetic stub artifacts
# ---------------------------------------------------------------------------

def _fake_fleet(n: int) -> pd.DataFrame:
    """A tiny fleet with the columns cache_regen reads."""
    rng = np.random.default_rng(SEED)
    powertrain = np.where(rng.random(n) < 0.7, "BEV", "PHEV")
    archetype = np.where(rng.random(n) < 0.2, "pickup", "sedan")
    return pd.DataFrame(
        {
            "ev_id": [f"EV{i:05d}" for i in range(n)],
            "powertrain": powertrain,
            "archetype": archetype,
        }
    )


def _fake_mobility(n: int, profile_type: str) -> pd.DataFrame:
    """Mobility with winter + June rows so the f_T / Wh-per-mi math runs."""
    target = "work" if profile_type == "workplace" else "home"
    ts = pd.to_datetime(
        ["2020-01-15", "2020-06-15", "2020-12-20", "2020-06-20"], utc=True
    )
    return pd.DataFrame(
        {
            "ev_id": [f"EV{i:05d}" for i in range(4)],
            "start_ts": ts,
            "dest_type": [target, target, target, "other"],
            "miles": [10.0, 12.0, 8.0, 5.0],
            "kwh_consumed": [4.0, 3.0, 3.5, 1.5],
            "f_T_applied": [1.3, 1.0, 1.25, 1.0],
        }
    )


def _fake_events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ev_id": ["EV00000", "EV00001"],
            "arrival_ts": pd.to_datetime(
                ["2020-01-15 18:00", "2020-06-15 19:00"], utc=True
            ),
        }
    )


def _fake_sessions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ev_id": ["EV00001", "EV00000"],
            "session_idx": [0, 0],
            "arrival_ts": pd.to_datetime(
                ["2020-06-15 19:00", "2020-01-15 18:00"], utc=True
            ),
        }
    )


def _fake_cluster_fit() -> types.SimpleNamespace:
    """Only the three attrs cache_regen reads off the fit object."""
    return types.SimpleNamespace(
        K_chosen=3,
        bic_chosen=123.5,
        cluster_centres_h=(8.0, 12.0, 18.0),
    )


@pytest.fixture
def stubbed_pipeline(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Patch every M2..M7 entry point on the cache_regen namespace.

    The stubs write the exact files cache_regen later reads back
    (fleet.parquet, plug_status_*.parquet, meta.json).
    """
    calls: dict[str, Any] = {"profile_type": None}

    def sample_fleet(*, n, seed, region, profile_type):  # noqa: ANN001
        calls["sample_fleet_seed"] = seed
        calls["sample_fleet_n"] = n
        calls["profile_type"] = profile_type
        return _fake_fleet(int(n))

    def write_fleet(df, out_path):  # noqa: ANN001
        df.to_parquet(out_path, engine="pyarrow", index=False)
        return Path(out_path)

    def write_meta(df, out_path, **kw):  # noqa: ANN001, ARG001
        Path(out_path).write_text(json.dumps({"bootstrap": True}), "utf-8")
        return Path(out_path)

    def match_donors(fleet, *, region, profile_type, seed, nhts_data_dir):  # noqa: ANN001, ARG001
        out = fleet.copy()
        out.attrs["donor_match"] = {"borrow_share": 0.25}
        return out

    def build_mobility(fleet, *, region, profile_type, seed, nhts_data_dir):  # noqa: ANN001, ARG001
        return _fake_mobility(len(fleet), profile_type)

    def fit_and_assign_plug_ins(fleet, mobility, *, region, profile_type, seed):  # noqa: ANN001, ARG001
        aug = fleet.copy()
        aug["cluster_z"] = (np.arange(len(aug)) % 3) + 1
        return aug, _fake_events(), _fake_cluster_fit()

    def write_plug_in_events(df, out_path):  # noqa: ANN001
        df.to_parquet(out_path, engine="pyarrow", index=False)
        return Path(out_path)

    def write_bic_sweep_results(cluster_fit, out_path):  # noqa: ANN001, ARG001
        Path(out_path).write_text("{}", "utf-8")
        return Path(out_path)

    def update_meta_with_m5(meta_path, **kw):  # noqa: ANN001, ARG001
        return Path(meta_path)

    def compute_sessions(fleet_aug, mobility, events, *, region, profile_type, seed):  # noqa: ANN001, ARG001
        return _fake_sessions()

    def build_plug_status(  # noqa: ANN001, ARG001
        *, sessions_path, region, profile_type, seed, n_vehicles,
        repo_root, out_dir,
    ):
        p15 = Path(out_dir) / "plug_status_15min.parquet"
        phr = Path(out_dir) / "plug_status_hourly.parquet"
        pd.DataFrame({"plugged": [True, False, True, True]}).to_parquet(
            p15, engine="pyarrow", index=False
        )
        pd.DataFrame({"plugged": [True, False]}).to_parquet(
            phr, engine="pyarrow", index=False
        )
        return {"plug_status_15min": p15, "plug_status_hourly": phr}

    monkeypatch.setattr(cache_regen, "sample_fleet", sample_fleet)
    monkeypatch.setattr(cache_regen, "write_fleet", write_fleet)
    monkeypatch.setattr(cache_regen, "write_meta", write_meta)
    monkeypatch.setattr(cache_regen, "match_donors", match_donors)
    monkeypatch.setattr(cache_regen, "build_mobility", build_mobility)
    monkeypatch.setattr(
        cache_regen, "fit_and_assign_plug_ins", fit_and_assign_plug_ins
    )
    monkeypatch.setattr(cache_regen, "write_plug_in_events", write_plug_in_events)
    monkeypatch.setattr(
        cache_regen, "write_bic_sweep_results", write_bic_sweep_results
    )
    monkeypatch.setattr(cache_regen, "update_meta_with_m5", update_meta_with_m5)
    monkeypatch.setattr(cache_regen, "compute_sessions", compute_sessions)
    monkeypatch.setattr(cache_regen, "build_plug_status", build_plug_status)
    return calls


# ---------------------------------------------------------------------------
# regenerate_cache — happy path + validation branches
# ---------------------------------------------------------------------------

def test_regenerate_cache_happy_path(
    stubbed_pipeline: dict[str, Any], tmp_path: Path
) -> None:
    res = cache_regen.regenerate_cache(
        "boston", "residential", n=12, seed=SEED, repo_root=tmp_path,
    )
    assert res["status"] == "PASS"
    assert res["region"] == "boston"
    assert res["profile_type"] == "residential"
    assert res["n"] == 12
    assert stubbed_pipeline["sample_fleet_seed"] == SEED

    m = res["metrics"]
    assert m["n_fleet"] == 12
    assert 0.0 <= m["bev_share"] <= 1.0
    assert m["donor_borrow_rate"] == 0.25
    assert m["K_chosen"] == 3
    assert m["bic_chosen"] == 123.5
    assert m["n_plug_in_events"] == 2
    assert m["n_sessions"] == 2
    # 15-min raster: 3 of 4 plugged.
    assert m["plug_rate_15min"] == pytest.approx(0.75)
    assert m["n_plug_status_15min"] == 4
    assert m["plug_rate_hourly"] == pytest.approx(0.5)

    out_dir = Path(res["out_dir"])
    for fname in (
        "fleet.parquet", "mobility.parquet", "plug_in_events.parquet",
        "sessions.parquet", "plug_status_15min.parquet",
        "plug_status_hourly.parquet", "meta.json",
    ):
        assert (out_dir / fname).is_file(), f"missing artifact {fname}"
        assert m["file_sizes_bytes"][fname] > 0, f"empty artifact {fname}"
    assert m["file_sizes_total_bytes"] > 0

    # Sessions written sorted by (ev_id, session_idx).
    sess = pd.read_parquet(out_dir / "sessions.parquet")
    assert list(sess["ev_id"]) == ["EV00000", "EV00001"]

    # _finalise_meta block landed in meta.json.
    meta = json.loads((out_dir / "meta.json").read_text("utf-8"))
    assert meta["methodology_version"] == cache_regen.METHODOLOGY_VERSION
    assert meta["cache_provenance"] == "v2.0_native"
    assert meta["master_seed"] == SEED
    assert meta["region"]["name"] == "boston"
    assert meta["cache_regen"]["module"] == "pev_synth.cache_regen"


def test_regenerate_cache_us_national_default_n(
    stubbed_pipeline: dict[str, Any], tmp_path: Path
) -> None:
    # n=None -> _n_for_region; us_national is 500.
    res = cache_regen.regenerate_cache(
        "us_national", "residential", seed=SEED, repo_root=tmp_path,
    )
    assert res["status"] == "PASS"
    assert res["n"] == 500
    assert stubbed_pipeline["sample_fleet_n"] == 500


def test_regenerate_cache_workplace_aggregate_rate(
    stubbed_pipeline: dict[str, Any], tmp_path: Path
) -> None:
    res = cache_regen.regenerate_cache(
        "bay_area", "workplace", n=8, seed=SEED, repo_root=tmp_path,
    )
    assert res["status"] == "PASS"
    # 3 of 4 mobility rows have dest_type == "work"; 2 events -> 2/3.
    assert res["metrics"]["aggregate_plug_in_rate"] == pytest.approx(2.0 / 3.0)


def test_regenerate_cache_empty_mobility(
    monkeypatch: pytest.MonkeyPatch, stubbed_pipeline: dict[str, Any],
    tmp_path: Path,
) -> None:
    # Empty mobility exercises the len(mobility)==0 metric branch + the
    # n_arrivals==0 aggregate-rate guard.
    def empty_mobility(fleet, **kw):  # noqa: ANN001, ARG001
        return _fake_mobility(0, "residential").iloc[0:0]

    monkeypatch.setattr(cache_regen, "build_mobility", empty_mobility)
    res = cache_regen.regenerate_cache(
        "boston", "residential", n=4, seed=SEED, repo_root=tmp_path,
    )
    assert res["status"] == "PASS"
    assert res["metrics"]["f_T_winter_mean"] == 1.0
    assert res["metrics"]["aggregate_plug_in_rate"] == 0.0
    assert np.isnan(res["metrics"]["mean_kwh_per_mi_winter"])


def test_regenerate_cache_workplace_cohort_file(
    monkeypatch: pytest.MonkeyPatch, stubbed_pipeline: dict[str, Any],
    tmp_path: Path,
) -> None:
    # Materialise the workplace cohort parquet so the cohort_size branch
    # reads it (cohort_size = len(parquet)).
    monkeypatch.setattr(cache_regen, "_repo_root", lambda: tmp_path)
    cohort_dir = (
        tmp_path / "data" / "pev" / "raw" / "evwatts_cohorts"
    )
    cohort_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": [1, 2, 3]}).to_parquet(
        cohort_dir / "workplace_subset.parquet", engine="pyarrow", index=False
    )
    res = cache_regen.regenerate_cache(
        "bay_area", "workplace", n=6, seed=SEED, repo_root=tmp_path,
    )
    assert res["status"] == "PASS"


def test_regenerate_cache_unknown_region(tmp_path: Path) -> None:
    res = cache_regen.regenerate_cache(
        "atlantis", "residential", repo_root=tmp_path,
    )
    assert res["status"] == "FAIL"
    assert "unknown region" in res["error"]


def test_regenerate_cache_unknown_profile_type(tmp_path: Path) -> None:
    res = cache_regen.regenerate_cache(
        "boston", "spaceport", repo_root=tmp_path,
    )
    assert res["status"] == "FAIL"
    assert "unknown profile_type" in res["error"]


def test_regenerate_cache_captures_exception(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def boom(**kw):  # noqa: ANN001, ARG001
        raise RuntimeError("fleet sampler exploded")

    monkeypatch.setattr(cache_regen, "sample_fleet", boom)
    res = cache_regen.regenerate_cache(
        "boston", "residential", n=4, repo_root=tmp_path,
    )
    assert res["status"] == "FAIL"
    assert "fleet sampler exploded" in res["error"]
    assert "Traceback" in res["traceback"]
    assert "elapsed_s_until_fail" in res


# ---------------------------------------------------------------------------
# _out_dir resolution
# ---------------------------------------------------------------------------

def test_out_dir_flat_layout_custom_root(tmp_path: Path) -> None:
    d = cache_regen._out_dir(tmp_path, "boston", "residential")
    assert d == tmp_path / "data" / "pev" / "processed" / "boston" / (
        "residential_ev_synth"
    )


def test_out_dir_replicate_layout_custom_root(tmp_path: Path) -> None:
    d = cache_regen._out_dir(
        tmp_path, "boston", "residential", replicate_id=2, r_total=4,
    )
    assert d.name == "r2"
    assert d.parent.name == "replicates"


def test_out_dir_default_repo_root_delegates() -> None:
    from pev_synth._paths import replicate_dir, repo_root

    d = cache_regen._out_dir(repo_root(), "boston", "workplace")
    assert d == replicate_dir("boston", "workplace", 0, 1)


def test_regenerate_cache_replicate_layout(
    stubbed_pipeline: dict[str, Any], tmp_path: Path
) -> None:
    res = cache_regen.regenerate_cache(
        "boston", "residential", n=6, seed=SEED, repo_root=tmp_path,
        replicate_id=1, r_total=3,
    )
    assert res["status"] == "PASS"
    out_dir = Path(res["out_dir"])
    assert out_dir.name == "r1"
    assert out_dir.parent.name == "replicates"


# ---------------------------------------------------------------------------
# regenerate_batch
# ---------------------------------------------------------------------------

def test_regenerate_batch_replicate_expansion(
    stubbed_pipeline: dict[str, Any], tmp_path: Path
) -> None:
    log_path = tmp_path / "progress.json"
    rep = cache_regen.regenerate_batch(
        [("boston", "residential")],
        seed=SEED, repo_root=tmp_path, r_total=2, log_path=log_path,
    )
    assert rep["completed"] == 2
    assert len(rep["results"]) == 2
    rids = sorted(r["replicate_id"] for r in rep["results"])
    assert rids == [0, 1]
    for r in rep["results"]:
        assert r["status"] == "PASS"
        assert r["r_total"] == 2
    # Progress log written.
    assert log_path.is_file()
    blob = json.loads(log_path.read_text("utf-8"))
    assert blob["completed_so_far"] == 2


def test_regenerate_batch_log_path_oserror_swallowed(
    stubbed_pipeline: dict[str, Any], tmp_path: Path
) -> None:
    # log_path is an existing directory -> open("w") raises OSError, which
    # regenerate_batch swallows; the batch still completes.
    log_dir = tmp_path / "log_is_a_dir"
    log_dir.mkdir()
    rep = cache_regen.regenerate_batch(
        [("boston", "residential")],
        seed=SEED, repo_root=tmp_path, r_total=1, log_path=log_dir,
    )
    assert rep["completed"] == 1
    assert rep["results"][0]["status"] == "PASS"


def test_regenerate_batch_time_budget_defers(
    stubbed_pipeline: dict[str, Any], tmp_path: Path
) -> None:
    # Negative budget -> every pair deferred immediately.
    rep = cache_regen.regenerate_batch(
        [("boston", "residential"), ("chicago", "workplace")],
        seed=SEED, repo_root=tmp_path, time_budget_s=-1.0,
    )
    assert rep["completed"] == 2
    assert all(r["status"] == "DEFERRED" for r in rep["results"])
    assert all("time budget" in r["reason"] for r in rep["results"])


def test_regenerate_batch_seed_offset_by_replicate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, stubbed_pipeline: dict[str, Any]
) -> None:
    seen: list[int] = []
    real = cache_regen.regenerate_cache

    def spy(region, profile_type, **kw):  # noqa: ANN001
        seen.append(kw["seed"])
        return real(region, profile_type, **kw)

    monkeypatch.setattr(cache_regen, "regenerate_cache", spy)
    cache_regen.regenerate_batch(
        [("boston", "residential")],
        seed=100, repo_root=tmp_path, r_total=3,
    )
    assert seen == [100, 101, 102]


# ---------------------------------------------------------------------------
# Cross-region helpers + run_cross_region_checks
# ---------------------------------------------------------------------------

def _write_cache(
    repo_root: Path, region: str, profile_type: str, *,
    n: int, pickup_share: float, winter_factor: float,
) -> Path:
    """Materialise a minimal synthetic cache for the C1-C5 audit."""
    d = (
        repo_root / "data" / "pev" / "processed" / region
        / f"{profile_type}_ev_synth"
    )
    d.mkdir(parents=True, exist_ok=True)
    n_pickup = int(round(n * pickup_share))
    archetype = ["pickup"] * n_pickup + ["sedan"] * (n - n_pickup)
    fleet = pd.DataFrame(
        {
            "ev_id": [f"EV{i:05d}" for i in range(n)],
            "archetype": archetype,
            "powertrain": ["BEV"] * n,
        }
    )
    fleet.to_parquet(d / "fleet.parquet", engine="pyarrow", index=False)

    # Mobility: a June row and winter rows with elevated Wh/mi.
    mob = pd.DataFrame(
        {
            "ev_id": ["EV00000", "EV00001", "EV00002"],
            "start_ts": pd.to_datetime(
                ["2020-06-15", "2020-01-15", "2020-12-20"], utc=True
            ),
            "miles": [10.0, 10.0, 10.0],
            "kwh_consumed": [
                3.0, 3.0 * winter_factor, 3.0 * winter_factor,
            ],
        }
    )
    mob.to_parquet(d / "mobility.parquet", engine="pyarrow", index=False)
    return d


def test_processed_root_and_cache_dir(tmp_path: Path) -> None:
    assert cache_regen._processed_root(tmp_path) == (
        tmp_path / "data" / "pev" / "processed"
    )
    assert cache_regen._cache_dir(tmp_path, "boston", "workplace") == (
        tmp_path / "data" / "pev" / "processed" / "boston"
        / "workplace_ev_synth"
    )


def test_existing_caches_empty_when_no_processed(tmp_path: Path) -> None:
    assert cache_regen._existing_caches(tmp_path) == []


def test_existing_caches_detects_written_cache(tmp_path: Path) -> None:
    _write_cache(
        tmp_path, "boston", "residential",
        n=4, pickup_share=0.25, winter_factor=1.3,
    )
    found = cache_regen._existing_caches(tmp_path)
    assert ("boston", "residential") in found


def test_run_cross_region_checks_full(tmp_path: Path) -> None:
    # Fleet sizes must match _n_for_region so C1 passes (1000, except
    # us_national which is 500). DFW high pickup, bay_area low pickup,
    # us_national between -> C4 PASS.
    _write_cache(
        tmp_path, "dallas_fort_worth", "residential",
        n=1000, pickup_share=0.6, winter_factor=1.0,
    )
    _write_cache(
        tmp_path, "bay_area", "residential",
        n=1000, pickup_share=0.1, winter_factor=1.0,
    )
    _write_cache(
        tmp_path, "us_national", "residential",
        n=500, pickup_share=0.3, winter_factor=1.0,
    )
    # Cold region with strong winter f_T -> C5 ratio >= 1.15.
    _write_cache(
        tmp_path, "boston", "residential",
        n=1000, pickup_share=0.2, winter_factor=1.3,
    )

    rep = cache_regen.run_cross_region_checks(repo_root=tmp_path)

    assert rep["n_caches"] == 4
    assert rep["C1_fleet_size"]["verdict"] == "PASS"
    assert rep["C2_utc_timestamps"]["verdict"] == "PASS"
    assert rep["C4_sales_mix"]["verdict"] == "PASS"
    assert rep["C5_winter_f_T"]["verdict"] == "PASS"
    assert "overall_status" in rep
    assert rep["overall_status"] in {"PASS", "MIXED", "FAIL"}

    out_dir = tmp_path / cache_regen.CROSS_REGION_OUT_DIR_REL
    assert (out_dir / "cross_region_report.json").is_file()
    md = (out_dir / "cross_region_report.md").read_text("utf-8")
    assert "Cross-region consistency report" in md
    assert "C1" in md and "C5" in md


def test_run_cross_region_checks_fleet_size_fail(tmp_path: Path) -> None:
    # boston expects N=1000; we wrote 4 -> C1 FAIL.
    _write_cache(
        tmp_path, "boston", "residential",
        n=4, pickup_share=0.2, winter_factor=1.3,
    )
    rep = cache_regen.run_cross_region_checks(repo_root=tmp_path)
    assert rep["C1_fleet_size"]["verdict"] == "FAIL"
    row = rep["C1_fleet_size"]["rows"][0]
    assert row["n_rows"] == 4
    assert row["expected"] == 1000
    assert row["pass"] is False


def test_run_cross_region_checks_c3_schema_mismatch(tmp_path: Path) -> None:
    # Two workplace caches with divergent fleet schemas -> C3 FAIL.
    d1 = _write_cache(
        tmp_path, "boston", "workplace",
        n=1000, pickup_share=0.2, winter_factor=1.0,
    )
    d2 = _write_cache(
        tmp_path, "chicago", "workplace",
        n=1000, pickup_share=0.2, winter_factor=1.0,
    )
    # Append an extra column to the chicago fleet so schemas differ.
    f2 = pd.read_parquet(d2 / "fleet.parquet")
    f2["extra_col"] = 1
    f2.to_parquet(d2 / "fleet.parquet", engine="pyarrow", index=False)
    assert (d1 / "fleet.parquet").is_file()

    rep = cache_regen.run_cross_region_checks(repo_root=tmp_path)
    assert rep["C3_schema_consistency"]["verdict"] == "FAIL"
    by_profile = rep["C3_schema_consistency"]["by_profile"]
    wp = next(b for b in by_profile if b["profile_type"] == "workplace")
    assert wp["pass"] is False
    assert wp["diffs"]


def test_run_cross_region_checks_corrupt_parquet(tmp_path: Path) -> None:
    # A non-parquet file at fleet.parquet exercises the C1 read-error path.
    d = _write_cache(
        tmp_path, "boston", "residential",
        n=1000, pickup_share=0.2, winter_factor=1.3,
    )
    (d / "fleet.parquet").write_text("not a parquet", "utf-8")
    rep = cache_regen.run_cross_region_checks(repo_root=tmp_path)
    # Corrupt fleet is still discovered (file exists) but read raises ->
    # C1 row carries an error and verdict is FAIL.
    assert rep["C1_fleet_size"]["verdict"] == "FAIL"
    assert any(r.get("error") for r in rep["C1_fleet_size"]["rows"])


def test_run_cross_region_checks_c4_c5_read_errors(tmp_path: Path) -> None:
    # DFW fleet corrupt -> C4 read-error branch; bay_area mobility corrupt
    # -> C5 baseline read-error branch.
    d_dfw = _write_cache(
        tmp_path, "dallas_fort_worth", "residential",
        n=1000, pickup_share=0.6, winter_factor=1.0,
    )
    (d_dfw / "fleet.parquet").write_text("corrupt", "utf-8")
    d_bay = _write_cache(
        tmp_path, "bay_area", "residential",
        n=1000, pickup_share=0.1, winter_factor=1.0,
    )
    (d_bay / "mobility.parquet").write_text("corrupt", "utf-8")
    rep = cache_regen.run_cross_region_checks(repo_root=tmp_path)
    c4_rows = rep["C4_sales_mix"]["rows"]
    assert any(
        r.get("region") == "dallas_fort_worth" and r.get("error")
        for r in c4_rows
    )


def test_run_cross_region_checks_no_caches_skips(tmp_path: Path) -> None:
    rep = cache_regen.run_cross_region_checks(repo_root=tmp_path)
    assert rep["n_caches"] == 0
    assert rep["C4_sales_mix"]["verdict"] == "SKIP"
    assert rep["C5_winter_f_T"]["verdict"] == "SKIP"


# ---------------------------------------------------------------------------
# CLI: main()
# ---------------------------------------------------------------------------

def test_main_one_pass(
    stubbed_pipeline: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    # Route the data root through tmp_path via PEV_SYNTH_DATA_ROOT, and pin
    # repo_root for the default-root branch of _out_dir.
    monkeypatch.setattr(cache_regen, "_repo_root", lambda: tmp_path)
    rc = cache_regen.main(
        ["one", "--region", "boston", "--profile-type", "residential",
         "--n", "6", "--seed", str(SEED)]
    )
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "PASS"
    assert out["region"] == "boston"


def test_main_one_fail_returns_1(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cache_regen, "_repo_root", lambda: tmp_path)

    def boom(**kw):  # noqa: ANN001, ARG001
        raise RuntimeError("nope")

    monkeypatch.setattr(cache_regen, "sample_fleet", boom)
    rc = cache_regen.main(
        ["one", "--region", "boston", "--profile-type", "residential", "--n", "4"]
    )
    assert rc == 1
    assert json.loads(capsys.readouterr().out)["status"] == "FAIL"


def test_main_batch_with_limit(
    stubbed_pipeline: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cache_regen, "_repo_root", lambda: tmp_path)
    rc = cache_regen.main(
        ["batch", "--limit", "1", "--seed", str(SEED)]
    )
    assert rc == 0
    rep = json.loads(capsys.readouterr().out)
    assert rep["completed"] == 1
    assert rep["results"][0]["status"] == "PASS"


def test_main_batch_time_budget_zero_defers(
    stubbed_pipeline: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cache_regen, "_repo_root", lambda: tmp_path)
    # 0-minute budget -> defers (elapsed > 0 after first iteration check).
    rc = cache_regen.main(
        ["batch", "--limit", "2", "--time-budget-min", "0"]
    )
    rep = json.loads(capsys.readouterr().out)
    # Deferred runs are not FAIL, so rc stays 0. With a 0s budget the first
    # pair may still run (elapsed ~0 is not > 0 on the first iteration) but
    # at least one later pair is deferred.
    assert rc == 0
    statuses = [r["status"] for r in rep["results"]]
    assert "DEFERRED" in statuses


def test_main_audit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_cache(
        tmp_path, "boston", "residential",
        n=4, pickup_share=0.2, winter_factor=1.3,
    )
    rc = cache_regen.main(["audit", "--repo-root", str(tmp_path)])
    rep = json.loads(capsys.readouterr().out)
    # C1 fails (n=4 != 1000) so overall is not PASS -> rc==1.
    assert rc == 1
    assert rep["overall_status"] in {"MIXED", "FAIL"}


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------

def test_n_for_region_branches() -> None:
    assert cache_regen._n_for_region("us_national") == cache_regen.N_US_NATIONAL
    assert cache_regen._n_for_region("seattle") == cache_regen.N_DEFAULT


def test_repo_root_returns_path() -> None:
    assert isinstance(cache_regen._repo_root(), Path)
    assert cache_regen._repo_root().is_absolute()


def test_finalise_meta_creates_when_missing(tmp_path: Path) -> None:
    region = Region.from_name("boston")
    meta_path = tmp_path / "meta.json"
    cache_regen._finalise_meta(
        meta_path, region=region, profile_type="residential",
        metrics={"x": np.int64(3)}, timings={"a": 0.1}, seed=SEED, n=7,
    )
    meta = json.loads(meta_path.read_text("utf-8"))
    assert meta["n_vehicles"] == 7
    assert meta["region"]["tz"] == region.tz
    assert meta["cache_regen"]["metrics"]["x"] == 3


def test_coerce_for_json_float_and_ndarray() -> None:
    out = cache_regen._coerce_for_json(
        {
            "arr": np.array([1.0, 2.0]),
            "pyfloat_inf": float("inf"),
            "pyfloat_ok": 1.5,
        }
    )
    assert out["arr"] == [1.0, 2.0]
    assert out["pyfloat_inf"] is None
    assert out["pyfloat_ok"] == 1.5


def test_coerce_for_json_numpy_floating_and_path() -> None:
    out = cache_regen._coerce_for_json(
        {
            "npf_ok": np.float64(2.5),
            "npf_nan": np.float64("nan"),
            "p": Path("/tmp/x"),
        }
    )
    assert out["npf_ok"] == 2.5
    assert out["npf_nan"] is None
    assert out["p"] == str(Path("/tmp/x"))


def test_run_cross_region_checks_corrupt_mobility(tmp_path: Path) -> None:
    # Valid fleet, corrupt mobility -> C2/C5 read-error handlers fire.
    d = _write_cache(
        tmp_path, "boston", "residential",
        n=1000, pickup_share=0.2, winter_factor=1.3,
    )
    (d / "mobility.parquet").write_text("not a parquet", "utf-8")
    rep = cache_regen.run_cross_region_checks(repo_root=tmp_path)
    assert rep["C2_utc_timestamps"]["verdict"] == "FAIL"
    assert rep["C5_winter_f_T"]["verdict"] in {"FAIL", "SKIP"}


def test_existing_caches_skips_dir_without_fleet(tmp_path: Path) -> None:
    # A cache dir that lacks fleet.parquet is skipped.
    d = (
        tmp_path / "data" / "pev" / "processed" / "boston"
        / "residential_ev_synth"
    )
    d.mkdir(parents=True, exist_ok=True)
    (d / "mobility.parquet").write_text("x", "utf-8")
    assert cache_regen._existing_caches(tmp_path) == []


def test_is_utc_naive_returns_false() -> None:
    assert cache_regen._is_utc(pd.Series([1, 2, 3])) is False


def test_run_cross_region_checks_c2_non_utc_fail(tmp_path: Path) -> None:
    # Mobility with a naive timestamp column -> C2 FAIL.
    d = (
        tmp_path / "data" / "pev" / "processed" / "boston"
        / "residential_ev_synth"
    )
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"ev_id": ["EV00000"] * 1000, "archetype": ["sedan"] * 1000}
    ).to_parquet(d / "fleet.parquet", engine="pyarrow", index=False)
    pd.DataFrame(
        {
            "ev_id": ["EV00000"],
            "start_ts": pd.to_datetime(["2020-06-15"]),  # tz-naive
            "miles": [10.0],
            "kwh_consumed": [3.0],
        }
    ).to_parquet(d / "mobility.parquet", engine="pyarrow", index=False)
    rep = cache_regen.run_cross_region_checks(repo_root=tmp_path)
    assert rep["C2_utc_timestamps"]["verdict"] == "FAIL"
    row = next(
        r for r in rep["C2_utc_timestamps"]["rows"]
        if r["file"] == "mobility.parquet"
    )
    assert row["non_utc_cols"] == ["start_ts"]
    assert row["pass"] is False


def test_run_cross_region_checks_c4_no_us_national(tmp_path: Path) -> None:
    # DFW + bay_area present, us_national absent -> C4 decided by c4_a only.
    _write_cache(
        tmp_path, "dallas_fort_worth", "residential",
        n=1000, pickup_share=0.6, winter_factor=1.0,
    )
    _write_cache(
        tmp_path, "bay_area", "residential",
        n=1000, pickup_share=0.1, winter_factor=1.0,
    )
    rep = cache_regen.run_cross_region_checks(repo_root=tmp_path)
    assert rep["C4_sales_mix"]["verdict"] == "PASS"


def test_render_cross_region_md_minimal() -> None:
    md = cache_regen._render_cross_region_md(
        {
            "methodology_version": "vX",
            "run_timestamp_utc": "2020-01-01T00:00:00+00:00",
            "n_caches": 0,
            "overall_status": "SKIP",
            "caches_inspected": [],
        }
    )
    assert md.startswith("# Cross-region consistency report")
    assert "vX" in md
