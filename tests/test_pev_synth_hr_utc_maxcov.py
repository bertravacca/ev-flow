"""Max-coverage tests for `pev_synth.hourly_resampler` and `pev_synth._utc_migration`.

This file targets the edge cases, validation/error paths, and the two CLI
``main`` drivers that the acceptance-criterion suites
(``test_pev_synth_hourly_resampler.py`` / ``test_pev_synth__utc_migration.py``)
leave uncovered. It is fully data-independent: every fixture is fabricated in
``tmp_path`` from a fixed seed and no on-disk pipeline cache is read.

Coverage focus
--------------
hourly_resampler.py:
    - empty-index early return + per-session clip / skip branches in
      ``rasterise_one_ev`` (zero-length, year clip, fully-clipped, range
      clamps)
    - ``aggregate_hourly`` divisibility guard
    - ``_build_plug_status_table`` shape-mismatch guard
    - ``_default_paths`` / ``resolve_v2_paths`` / ``build_plug_status``
      profile-type validation + repo_root resolution branch
    - ``rasterise_all_evs`` missing-column + empty-EV + log_every branches
    - ``_validate_sessions`` error paths
    - ``run`` FileNotFoundError
    - ``_json_default`` serialiser
    - the ``main`` CLI (legacy flat layout, --region routing, --out-dir)

_utc_migration.py:
    - ``_detect_ts_columns`` / ``_convert_frame_to_utc`` error paths
    - ``migrate_parquet_tz`` FileNotFoundError
    - ``migrate_bay_area_v1_to_v2`` repo_root relative-join + src-not-dir +
      skipped-file + missing-meta branches
    - ``_resolve_migration_timestamp`` corrupt-meta fallback
    - the ``main`` CLI
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

_CODE_DIR = Path(__file__).resolve().parents[1]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from pev_synth import _utc_migration as um  # noqa: E402
from pev_synth import hourly_resampler as hr  # noqa: E402

LOCAL_TZ = "America/Los_Angeles"
STORAGE_TZ = "UTC"


# ===========================================================================
# hourly_resampler — small synthetic sessions builders
# ===========================================================================

def _tiny_sessions(n_evs: int = 3, seed: int = 42, year: int = 2001) -> pd.DataFrame:
    """A minimal residential sessions frame (a few overnight + 1 DCFC per EV)."""
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for ev_id in range(n_evs):
        days = sorted(rng.choice(np.arange(360), size=4, replace=False).tolist())
        for j, d in enumerate(days):
            t_in = (
                pd.Timestamp(f"{year}-01-01", tz=STORAGE_TZ)
                + pd.Timedelta(days=int(d), hours=20)
            )
            t_out = t_in + pd.Timedelta(hours=8)
            rows.append(
                {
                    "ev_id": int(ev_id),
                    "session_idx": int(j),
                    "t_in": t_in,
                    "t_out": t_out,
                    "energy_kwh": 10.0,
                    "location": "home",
                    "max_kw": 7.2,
                }
            )
        # one DCFC session
        t_in_dcfc = (
            pd.Timestamp(f"{year}-01-01", tz=STORAGE_TZ)
            + pd.Timedelta(days=120, hours=13)
        )
        rows.append(
            {
                "ev_id": int(ev_id),
                "session_idx": 99,
                "t_in": t_in_dcfc,
                "t_out": t_in_dcfc + pd.Timedelta(minutes=25),
                "energy_kwh": 10.0,
                "location": "dcfc_inject",
                "max_kw": 50.0,
            }
        )
    return pd.DataFrame(rows).sort_values(["ev_id", "t_in"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# rasterise_one_ev — edge branches
# ---------------------------------------------------------------------------

def test_rasterise_one_ev_empty_index() -> None:
    """Empty index → empty arrays + zeroed stats (line 272)."""
    empty_idx = pd.DatetimeIndex([], tz=STORAGE_TZ)
    sessions = pd.DataFrame(
        {
            "t_in": [pd.Timestamp("2001-01-01 00:00", tz=STORAGE_TZ)],
            "t_out": [pd.Timestamp("2001-01-01 01:00", tz=STORAGE_TZ)],
            "location": ["home"],
        }
    )
    plugged, locs, stats = hr.rasterise_one_ev(sessions, empty_idx)
    assert plugged.shape == (0,)
    assert locs.shape == (0,)
    assert stats == hr.RasterStats(0, 0, 0, 0, 0)


def test_rasterise_one_ev_zero_length_session_skipped() -> None:
    """t_out <= t_in is silently ignored (line 289)."""
    idx = hr.build_year_index_15min()
    t = pd.Timestamp("2001-03-01 10:00", tz=STORAGE_TZ)
    sessions = pd.DataFrame(
        {"t_in": [t], "t_out": [t], "location": ["home"]}  # zero length
    )
    plugged, _, stats = hr.rasterise_one_ev(sessions, idx)
    assert not plugged.any()
    assert stats.n_slots_plugged == 0


def test_rasterise_one_ev_session_before_year_is_clipped() -> None:
    """A session starting before the year start is clipped (lines 294-295)."""
    idx = hr.build_year_index_15min()
    sessions = pd.DataFrame(
        {
            "t_in": [pd.Timestamp("2000-12-31 22:00", tz=STORAGE_TZ)],
            "t_out": [pd.Timestamp("2001-01-01 02:00", tz=STORAGE_TZ)],
            "location": ["home"],
        }
    )
    plugged, _, stats = hr.rasterise_one_ev(sessions, idx)
    assert stats.n_sessions_clipped == 1
    # First slot (00:00 UTC) is plugged; nothing wraps before year start.
    assert plugged[0]


def test_rasterise_one_ev_fully_before_year_clipped_to_empty() -> None:
    """A session entirely before year start clips to empty (lines 300-302).

    t_in and t_out both precede the year window; after the clip t_in is
    bumped to year_start but t_out stays < year_start, so t_in >= t_out and
    the clipped session contributes nothing but increments the clip count.
    """
    idx = hr.build_year_index_15min()
    sessions = pd.DataFrame(
        {
            "t_in": [pd.Timestamp("2000-12-30 10:00", tz=STORAGE_TZ)],
            "t_out": [pd.Timestamp("2000-12-30 12:00", tz=STORAGE_TZ)],
            "location": ["home"],
        }
    )
    plugged, _, stats = hr.rasterise_one_ev(sessions, idx)
    assert not plugged.any()
    assert stats.n_sessions_clipped == 1


def test_rasterise_one_ev_after_year_clamps_last_slot() -> None:
    """A session ending past year end clamps last_slot to n-1 (line 320)."""
    idx = hr.build_year_index_15min()
    sessions = pd.DataFrame(
        {
            "t_in": [pd.Timestamp("2001-12-31 23:00", tz=STORAGE_TZ)],
            "t_out": [pd.Timestamp("2002-01-02 00:00", tz=STORAGE_TZ)],
            "location": ["home"],
        }
    )
    plugged, _, stats = hr.rasterise_one_ev(sessions, idx)
    assert plugged[-1]
    assert stats.n_sessions_clipped == 1


def test_rasterise_one_ev_unknown_location_defaults_home() -> None:
    """An out-of-domain location string falls back to home idx 0 (line 307)."""
    idx = hr.build_year_index_15min()
    sessions = pd.DataFrame(
        {
            "t_in": [pd.Timestamp("2001-05-01 10:00", tz=STORAGE_TZ)],
            "t_out": [pd.Timestamp("2001-05-01 11:00", tz=STORAGE_TZ)],
            "location": ["garage_charger"],  # unknown → home
        }
    )
    plugged, locs, _ = hr.rasterise_one_ev(sessions, idx)
    assert (locs[plugged] == hr.LOC_HOME_IDX).all()


def test_rasterise_one_ev_none_location_defaults_home() -> None:
    """A None location maps to home (line 306 None-branch)."""
    idx = hr.build_year_index_15min()
    sessions = pd.DataFrame(
        {
            "t_in": [pd.Timestamp("2001-05-01 10:00", tz=STORAGE_TZ)],
            "t_out": [pd.Timestamp("2001-05-01 11:00", tz=STORAGE_TZ)],
            "location": [None],
        }
    )
    plugged, locs, _ = hr.rasterise_one_ev(sessions, idx)
    assert (locs[plugged] == hr.LOC_HOME_IDX).all()


def test_rasterise_one_ev_subslot_session_below_threshold() -> None:
    """A 5-min session inside a single slot covers no slot at >= 7.5 min.

    This exercises the slot loop where last_slot == first_slot and the
    overlap is below threshold (the `if last_slot < first_slot` guard at
    line 321-322 is not tripped here, but the loop body runs once and
    rejects the slot).
    """
    idx = hr.build_year_index_15min()
    sessions = pd.DataFrame(
        {
            "t_in": [pd.Timestamp("2001-04-01 10:02", tz=STORAGE_TZ)],
            "t_out": [pd.Timestamp("2001-04-01 10:07", tz=STORAGE_TZ)],
            "location": ["home"],
        }
    )
    plugged, _, _ = hr.rasterise_one_ev(sessions, idx)
    assert not plugged.any()


# ---------------------------------------------------------------------------
# aggregate_hourly — error path
# ---------------------------------------------------------------------------

def test_aggregate_hourly_not_divisible_by_4_raises() -> None:
    """Length not divisible by 4 raises ValueError (line 373)."""
    with pytest.raises(ValueError, match="not divisible by 4"):
        hr.aggregate_hourly(np.array([True, False, True], dtype=bool))


# ---------------------------------------------------------------------------
# _build_plug_status_table — shape guard
# ---------------------------------------------------------------------------

def test_build_plug_status_table_shape_mismatch_raises() -> None:
    """Mismatched plug-array shape raises ValueError (line 437)."""
    ev_ids = np.array([0, 1], dtype=np.int32)
    ts_index = hr.build_year_index_hourly()  # T = 8760
    bad_plugged = np.zeros((2, 10), dtype=bool)  # wrong T
    bad_loc = np.full((2, 10), -1, dtype=np.int8)
    with pytest.raises(ValueError, match="shape mismatch"):
        hr._build_plug_status_table(ev_ids, ts_index, bad_plugged, bad_loc)


# ---------------------------------------------------------------------------
# path resolution
# ---------------------------------------------------------------------------

def test_default_paths_layout(tmp_path: Path) -> None:
    """_default_paths resolves the legacy flat layout (lines 524-525)."""
    paths = hr._default_paths(tmp_path)
    base = tmp_path / "data" / "pev" / "processed" / "residential_ev_synth"
    assert paths.sessions_in == base / "sessions.parquet"
    assert paths.plug_status_15min_out == base / "plug_status_15min.parquet"
    assert paths.plug_status_hourly_out == base / "plug_status_hourly.parquet"
    assert paths.meta_out == base / "meta.json"


def test_resolve_v2_paths_bad_profile_type_raises(tmp_path: Path) -> None:
    """resolve_v2_paths rejects an unknown profile_type (line 549)."""
    from pev_synth.regions import Region

    region = Region.from_name("bay_area")
    with pytest.raises(ValueError, match="profile_type must be one of"):
        hr.resolve_v2_paths(tmp_path, region, "fleet_yard")


def test_build_plug_status_bad_profile_type_raises(tmp_path: Path) -> None:
    """build_plug_status rejects an unknown profile_type (line 621)."""
    from pev_synth.regions import Region

    region = Region.from_name("bay_area")
    sp = tmp_path / "sessions.parquet"
    with pytest.raises(ValueError, match="profile_type must be one of"):
        hr.build_plug_status(sp, region, "fleet_yard")


def test_build_plug_status_out_dir_branch(tmp_path: Path) -> None:
    """build_plug_status with out_dir writes rasters under out_dir (628-629)."""
    from pev_synth.regions import Region

    region = Region.from_name("bay_area")
    sess_dir = tmp_path / "in"
    sess_dir.mkdir(parents=True, exist_ok=True)
    sp = sess_dir / "sessions.parquet"
    _tiny_sessions(n_evs=2, seed=8).to_parquet(sp, engine="pyarrow", index=False)
    out_dir = tmp_path / "out"
    out = hr.build_plug_status(
        sessions_path=sp,
        region=region,
        profile_type="residential",
        n_vehicles=2,
        out_dir=out_dir,
    )
    assert out["plug_status_15min"] == out_dir / "plug_status_15min.parquet"
    assert out["plug_status_15min"].is_file()


def test_build_plug_status_bare_branch_uses_sessions_parent(tmp_path: Path) -> None:
    """With neither out_dir nor repo_root, rasters land beside sessions (642-643)."""
    from pev_synth.regions import Region

    region = Region.from_name("bay_area")
    sp = tmp_path / "sessions.parquet"
    _tiny_sessions(n_evs=2, seed=9).to_parquet(sp, engine="pyarrow", index=False)
    out = hr.build_plug_status(
        sessions_path=sp,
        region=region,
        profile_type="residential",
        n_vehicles=2,
    )
    assert out["plug_status_15min"] == tmp_path / "plug_status_15min.parquet"
    assert out["plug_status_15min"].is_file()


def test_build_plug_status_repo_root_branch(tmp_path: Path) -> None:
    """build_plug_status with repo_root (no out_dir) routes via resolve_v2_paths.

    Covers lines 636-640: the function resolves the canonical region layout
    from ``repo_root`` and then swaps in the explicit ``sessions_path``.
    """
    from pev_synth.regions import Region

    region = Region.from_name("bay_area")
    # The sessions parquet lives in the canonical region tree so run() can
    # read it, but the rasters are routed via repo_root resolution.
    base = (
        tmp_path / "data" / "pev" / "processed" / region.name / "residential_ev_synth"
    )
    base.mkdir(parents=True, exist_ok=True)
    sessions = _tiny_sessions(n_evs=2, seed=7)
    sp = base / "sessions.parquet"
    sessions.to_parquet(sp, engine="pyarrow", index=False)

    out = hr.build_plug_status(
        sessions_path=sp,
        region=region,
        profile_type="residential",
        n_vehicles=2,
        repo_root=tmp_path,
    )
    assert out["plug_status_15min"] == base / "plug_status_15min.parquet"
    assert out["plug_status_15min"].is_file()
    t = pq.read_table(out["plug_status_15min"])
    assert t.num_rows == 2 * 35_040


# ---------------------------------------------------------------------------
# rasterise_all_evs — guards + logging
# ---------------------------------------------------------------------------

def test_rasterise_all_evs_missing_ev_id_column_raises() -> None:
    """rasterise_all_evs requires an ev_id column (line 684)."""
    idx = hr.build_year_index_15min()
    bad = pd.DataFrame(
        {
            "t_in": [pd.Timestamp("2001-01-01 10:00", tz=STORAGE_TZ)],
            "t_out": [pd.Timestamp("2001-01-01 11:00", tz=STORAGE_TZ)],
            "location": ["home"],
        }
    )
    with pytest.raises(ValueError, match="missing 'ev_id' column"):
        hr.rasterise_all_evs(bad, [0], idx)


def test_rasterise_all_evs_empty_ev_yields_zero_stats() -> None:
    """An EV with no sessions yields a zeroed RasterStats (lines 691-692)."""
    idx = hr.build_year_index_15min()
    sessions = _tiny_sessions(n_evs=1, seed=3)  # only ev_id 0
    ev_ids = [0, 1]  # ev_id 1 absent
    _, plugged, _, stats = hr.rasterise_all_evs(sessions, ev_ids, idx)
    assert stats[1] == hr.RasterStats(0, 0, 0, 0, 0)
    assert not plugged[1].any()


def test_rasterise_all_evs_log_every_branch(caplog: pytest.LogCaptureFixture) -> None:
    """log_every emits a progress log line (line 700)."""
    idx = hr.build_year_index_15min()
    sessions = _tiny_sessions(n_evs=3, seed=5)
    with caplog.at_level(logging.INFO, logger="pev_synth.hourly_resampler"):
        hr.rasterise_all_evs(sessions, [0, 1, 2], idx, log_every=1)
    assert any("rasterised" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# _validate_sessions — error paths
# ---------------------------------------------------------------------------

def test_validate_sessions_missing_column() -> None:
    """Missing required column raises ValueError (line 710)."""
    df = pd.DataFrame({"ev_id": [0], "t_in": [pd.Timestamp("2001-01-01", tz="UTC")]})
    with pytest.raises(ValueError, match="missing required columns"):
        hr._validate_sessions(df)


def test_validate_sessions_non_datetime_column() -> None:
    """A non-datetime t_in raises TypeError (line 715)."""
    df = pd.DataFrame(
        {
            "ev_id": [0],
            "t_in": ["2001-01-01"],  # string, not datetime
            "t_out": [pd.Timestamp("2001-01-01 01:00", tz="UTC")],
            "location": ["home"],
        }
    )
    with pytest.raises(TypeError, match="must be datetime64"):
        hr._validate_sessions(df)


def test_validate_sessions_tz_naive_column() -> None:
    """A tz-naive datetime column raises TypeError (line 717)."""
    df = pd.DataFrame(
        {
            "ev_id": [0],
            "t_in": [pd.Timestamp("2001-01-01 00:00")],  # naive
            "t_out": [pd.Timestamp("2001-01-01 01:00")],
            "location": ["home"],
        }
    )
    with pytest.raises(TypeError, match="must be tz-aware"):
        hr._validate_sessions(df)


# ---------------------------------------------------------------------------
# run — FileNotFoundError + LA-tagged coercion path
# ---------------------------------------------------------------------------

def test_run_missing_sessions_raises(tmp_path: Path) -> None:
    """run raises FileNotFoundError when sessions parquet is absent (line 730)."""
    paths = hr.M7Paths(
        sessions_in=tmp_path / "missing.parquet",
        plug_status_15min_out=tmp_path / "a.parquet",
        plug_status_hourly_out=tmp_path / "b.parquet",
        meta_out=tmp_path / "meta.json",
    )
    with pytest.raises(FileNotFoundError, match="sessions.parquet not found"):
        hr.run(paths)


def test_run_coerces_la_sessions_to_utc(tmp_path: Path) -> None:
    """run accepts LA-tagged sessions and coerces them to UTC (line 744)."""
    # Build sessions in LA wall-clock so the tz != PEV_TZ branch fires.
    rows = []
    for ev_id in range(2):
        t_in = pd.Timestamp("2001-02-01 20:00", tz=LOCAL_TZ) + pd.Timedelta(days=ev_id)
        rows.append(
            {
                "ev_id": ev_id,
                "session_idx": 0,
                "t_in": t_in,
                "t_out": t_in + pd.Timedelta(hours=8),
                "energy_kwh": 10.0,
                "location": "home",
                "max_kw": 7.2,
            }
        )
    sessions = pd.DataFrame(rows)
    sp = tmp_path / "sessions.parquet"
    sessions.to_parquet(sp, engine="pyarrow", index=False)
    paths = hr.M7Paths(
        sessions_in=sp,
        plug_status_15min_out=tmp_path / "p15.parquet",
        plug_status_hourly_out=tmp_path / "ph.parquet",
        meta_out=tmp_path / "meta.json",
    )
    summary = hr.run(paths, n_vehicles=2)
    # On-disk tz is UTC regardless of the LA input tag.
    t = pq.read_table(paths.plug_status_15min_out)
    assert t.schema.field("ts").type.tz == "UTC"
    assert summary["n_vehicles"] == 2


# ---------------------------------------------------------------------------
# _json_default — serialiser branches
# ---------------------------------------------------------------------------

def test_json_default_numpy_integer() -> None:
    """np.integer → int (line 879-880)."""
    assert hr._json_default(np.int64(7)) == 7
    assert isinstance(hr._json_default(np.int64(7)), int)


def test_json_default_numpy_float_finite_and_nonfinite() -> None:
    """np.floating → float; non-finite → None (lines 881-885)."""
    assert hr._json_default(np.float64(1.5)) == 1.5
    assert hr._json_default(np.float64(np.inf)) is None
    assert hr._json_default(np.float64(np.nan)) is None


def test_json_default_numpy_array() -> None:
    """np.ndarray → list (lines 886-887)."""
    assert hr._json_default(np.array([1, 2, 3])) == [1, 2, 3]


def test_json_default_unserialisable_raises() -> None:
    """An unknown type raises TypeError (line 888)."""
    with pytest.raises(TypeError, match="unserialisable"):
        hr._json_default(object())


# ---------------------------------------------------------------------------
# main CLI
# ---------------------------------------------------------------------------

def _write_tiny_sessions(path: Path, n_evs: int = 3, seed: int = 11) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _tiny_sessions(n_evs=n_evs, seed=seed).to_parquet(path, engine="pyarrow", index=False)


def test_main_cli_flat_layout(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """CLI with --sessions + --out-dir (legacy flat layout, no --region)."""
    sp = tmp_path / "in" / "sessions.parquet"
    _write_tiny_sessions(sp, n_evs=2, seed=21)
    out_dir = tmp_path / "out"
    rc = hr.main(
        [
            "--repo-root",
            str(tmp_path),
            "--sessions",
            str(sp),
            "--out-dir",
            str(out_dir),
            "--n-vehicles",
            "2",
            "--log-level",
            "WARNING",
        ]
    )
    assert rc == 0
    assert (out_dir / "plug_status_15min.parquet").is_file()
    assert (out_dir / "plug_status_hourly.parquet").is_file()
    captured = capsys.readouterr()
    assert "M7 hourly_resampler" in captured.out
    assert "n_vehicles=2" in captured.out


def test_main_cli_region_routing(tmp_path: Path) -> None:
    """CLI with --region resolves the v2.0 region-routed layout."""
    region_base = (
        tmp_path / "data" / "pev" / "processed" / "bay_area" / "residential_ev_synth"
    )
    sp = region_base / "sessions.parquet"
    _write_tiny_sessions(sp, n_evs=2, seed=31)
    rc = hr.main(
        [
            "--repo-root",
            str(tmp_path),
            "--region",
            "bay_area",
            "--profile-type",
            "residential",
            "--n-vehicles",
            "2",
            "--log-level",
            "ERROR",
        ]
    )
    assert rc == 0
    assert (region_base / "plug_status_15min.parquet").is_file()
    assert (region_base / "plug_status_hourly.parquet").is_file()


# ===========================================================================
# _utc_migration
# ===========================================================================

def _make_min_v11_cache(
    root: Path, n_evs: int = 2, n_days: int = 3, write_meta: bool = False
) -> dict[str, int]:
    """A miniature v1.1 cache (LA-tagged) — only the files we need to drive
    the branch coverage. We deliberately omit ``plug_in_events.parquet`` so the
    skipped-file branch fires. ``write_meta`` controls whether a v1.1
    meta.json is emitted (to exercise the present-meta read branch)."""
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260520)

    fleet = pd.DataFrame(
        {"ev_id": np.arange(n_evs, dtype=np.int32), "B_kwh": np.full(n_evs, 70.0)}
    )
    fleet.to_parquet(root / "fleet.parquet", engine="pyarrow", index=False)

    def _ts_rows(cols: tuple[str, str]) -> pd.DataFrame:
        rows = []
        for ev in range(n_evs):
            for d in range(n_days):
                base = pd.Timestamp("2001-01-01", tz=LOCAL_TZ) + pd.Timedelta(days=d)
                a = base + pd.Timedelta(hours=18, minutes=int(rng.integers(0, 30)))
                b = a + pd.Timedelta(hours=10)
                rows.append({"ev_id": ev, cols[0]: a, cols[1]: b, "location": "home"})
        return pd.DataFrame(rows)

    _ts_rows(("start_ts", "end_ts")).to_parquet(
        root / "mobility.parquet", engine="pyarrow", index=False
    )
    _ts_rows(("t_in", "t_out")).to_parquet(
        root / "sessions.parquet", engine="pyarrow", index=False
    )

    idx15 = pd.date_range(
        pd.Timestamp("2001-01-01", tz=LOCAL_TZ),
        periods=96 * n_days,
        freq="15min",
    )
    pd.DataFrame(
        {
            "ev_id": np.zeros(len(idx15), dtype=np.int32),
            "ts": idx15,
            "plugged": rng.random(len(idx15)) < 0.5,
        }
    ).to_parquet(root / "plug_status_15min.parquet", engine="pyarrow", index=False)

    idxh = pd.date_range(
        pd.Timestamp("2001-01-01", tz=LOCAL_TZ), periods=24 * n_days, freq="1h"
    )
    pd.DataFrame(
        {
            "ev_id": np.zeros(len(idxh), dtype=np.int32),
            "ts": idxh,
            "plugged": rng.random(len(idxh)) < 0.5,
        }
    ).to_parquet(root / "plug_status_hourly.parquet", engine="pyarrow", index=False)

    if write_meta:
        meta = {
            "methodology_version": "v1.0.0",
            "master_seed": 20260518,
            "tz_primary": LOCAL_TZ,
            "n_vehicles": n_evs,
            "package_versions": {"numpy": np.__version__, "pandas": pd.__version__},
        }
        with (root / "meta.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, sort_keys=True)

    return {
        "fleet.parquet": n_evs,
        "mobility.parquet": n_evs * n_days,
        "sessions.parquet": n_evs * n_days,
        "plug_status_15min.parquet": 96 * n_days,
        "plug_status_hourly.parquet": 24 * n_days,
    }


# ---------------------------------------------------------------------------
# helper-level coverage
# ---------------------------------------------------------------------------

def test_detect_ts_columns() -> None:
    """_detect_ts_columns returns only tz-aware datetime columns (143-148)."""
    df = pd.DataFrame(
        {
            "a": pd.to_datetime(["2001-01-01"]).tz_localize("UTC"),
            "b": [1],
            "c": pd.to_datetime(["2001-01-01"]),  # tz-naive → excluded
        }
    )
    assert um._detect_ts_columns(df) == ["a"]


def test_convert_frame_to_utc_non_datetime_raises() -> None:
    """_convert_frame_to_utc rejects a non-datetime column (line 162)."""
    df = pd.DataFrame({"x": [1, 2, 3]})
    with pytest.raises(TypeError, match="not datetime64"):
        um._convert_frame_to_utc(df, ["x"])


def test_convert_frame_to_utc_naive_raises() -> None:
    """_convert_frame_to_utc rejects a tz-naive column (line 164)."""
    df = pd.DataFrame({"x": pd.to_datetime(["2001-01-01", "2001-01-02"])})
    with pytest.raises(TypeError, match="tz-naive"):
        um._convert_frame_to_utc(df, ["x"])


def test_convert_frame_to_utc_already_utc_noop() -> None:
    """A UTC column passes through unchanged (the no-op conversion)."""
    df = pd.DataFrame({"x": pd.to_datetime(["2001-01-01"]).tz_localize("UTC")})
    out = um._convert_frame_to_utc(df, ["x"])
    assert str(out["x"].dt.tz) == "UTC"


def test_migrate_parquet_tz_missing_source_raises(tmp_path: Path) -> None:
    """migrate_parquet_tz raises FileNotFoundError on a missing src (line 224)."""
    with pytest.raises(FileNotFoundError, match="source parquet not found"):
        um.migrate_parquet_tz(tmp_path / "nope.parquet", tmp_path / "out.parquet", ["t_in"])


# ---------------------------------------------------------------------------
# migrate_bay_area_v1_to_v2 — branches
# ---------------------------------------------------------------------------

def test_migrate_repo_root_relative_join(tmp_path: Path) -> None:
    """Relative src/dst roots are joined to repo_root (lines 317-320)."""
    repo = tmp_path / "repo"
    src_abs = repo / "v11"
    _make_min_v11_cache(src_abs)
    result = um.migrate_bay_area_v1_to_v2(
        src_root=Path("v11"),
        dst_root=Path("v20"),
        repo_root=repo,
    )
    assert (repo / "v20" / "fleet.parquet").is_file()
    # plug_in_events.parquet was omitted from the fixture → skipped.
    assert "plug_in_events.parquet" in result.skipped_files


def test_migrate_src_not_a_directory_raises(tmp_path: Path) -> None:
    """A non-existent src_root raises FileNotFoundError (line 323)."""
    with pytest.raises(FileNotFoundError, match="src_root not found"):
        um.migrate_bay_area_v1_to_v2(src_root=tmp_path / "ghost", dst_root=tmp_path / "out")


def test_migrate_skips_missing_files_and_meta(tmp_path: Path) -> None:
    """Missing parquet + missing meta.json are recorded as skipped (339-341, 360-362)."""
    src = tmp_path / "v11"
    _make_min_v11_cache(src)  # omits plug_in_events.parquet, includes meta? no.
    # _make_min_v11_cache does NOT write meta.json, so the missing-meta branch
    # (360-362) fires too.
    dst = tmp_path / "v20"
    result = um.migrate_bay_area_v1_to_v2(src_root=src, dst_root=dst)
    assert "plug_in_events.parquet" in result.skipped_files
    assert "meta.json" in result.skipped_files
    # meta.json is still written to dst (built from an empty base).
    assert (dst / "meta.json").is_file()
    with (dst / "meta.json").open(encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["region"] == "bay_area"
    assert meta["migrated_from"] == "v1.1"


def test_migrate_reads_present_src_meta(tmp_path: Path) -> None:
    """When src meta.json exists it is read and overlaid (lines 357-358)."""
    src = tmp_path / "v11"
    _make_min_v11_cache(src, write_meta=True)
    dst = tmp_path / "v20"
    result = um.migrate_bay_area_v1_to_v2(src_root=src, dst_root=dst)
    # meta.json present in src → NOT in skipped_files.
    assert "meta.json" not in result.skipped_files
    with (dst / "meta.json").open(encoding="utf-8") as f:
        meta = json.load(f)
    # v1.1 field preserved from the source meta...
    assert meta["n_vehicles"] == 2
    # ...and v2.0 provenance overlaid on top.
    assert meta["methodology_version"] == "v3.0"
    assert meta["migrated_from"] == "v1.1"


def test_resolve_migration_timestamp_corrupt_meta_fallback(tmp_path: Path) -> None:
    """A corrupt prior meta.json falls back to a fresh timestamp (415-416)."""
    bad_meta = tmp_path / "meta.json"
    bad_meta.write_text("{ this is not valid json", encoding="utf-8")
    ts = um._resolve_migration_timestamp(bad_meta, {"fleet.parquet": "abc"})
    # A fresh ISO-8601 UTC timestamp is returned (parses + tz-aware).
    parsed = pd.Timestamp(ts)
    assert parsed.tz is not None


def test_resolve_migration_timestamp_reuses_prior_on_match(tmp_path: Path) -> None:
    """Matching prior SHA-256s reuse the stored timestamp (idempotent path)."""
    meta = tmp_path / "meta.json"
    prior_ts = "2001-01-01T00:00:00+00:00"
    meta.write_text(
        json.dumps(
            {
                "migration_provenance": {
                    "v20_sha256_per_file": {"fleet.parquet": "deadbeef"},
                    "migration_utc_timestamp": prior_ts,
                }
            }
        ),
        encoding="utf-8",
    )
    ts = um._resolve_migration_timestamp(meta, {"fleet.parquet": "deadbeef"})
    assert ts == prior_ts


# ---------------------------------------------------------------------------
# _utc_migration main CLI
# ---------------------------------------------------------------------------

def test_um_main_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """CLI drives migrate_bay_area_v1_to_v2 with explicit --src/--dst (426-476)."""
    src = tmp_path / "v11"
    _make_min_v11_cache(src)
    dst = tmp_path / "v20"
    rc = um.main(
        [
            "--src",
            str(src),
            "--dst",
            str(dst),
            "--log-level",
            "WARNING",
        ]
    )
    assert rc == 0
    assert (dst / "fleet.parquet").is_file()
    assert (dst / "meta.json").is_file()
    captured = capsys.readouterr()
    assert "pev_synth._utc_migration" in captured.out
    assert "n_files_migrated" in captured.out


def test_um_main_cli_with_repo_root(tmp_path: Path) -> None:
    """CLI --repo-root resolves relative --src/--dst (465 repo_root branch)."""
    repo = tmp_path / "repo"
    _make_min_v11_cache(repo / "v11")
    rc = um.main(
        [
            "--src",
            "v11",
            "--dst",
            "v20",
            "--repo-root",
            str(repo),
            "--log-level",
            "ERROR",
        ]
    )
    assert rc == 0
    assert (repo / "v20" / "fleet.parquet").is_file()
