"""Coverage-raising tests for ``pev_synth.sales_mix_data`` (M3).

These tests are fully **data-independent**: they never touch the repo's real
``data/`` tree. Every disk read/write is isolated to ``tmp_path`` by either
passing explicit paths or monkeypatching the module-level default-path
constants. Synthetic EVWatts inputs are built in memory with fixed structure
so the workplace-cohort builder, the sales-mix CSV builder/loader, and the
argparse CLI subcommands are all exercised end-to-end with concrete assertions.

Covered surface:
    - ``_build_sales_mix_df`` for all three sources + unknown-source error.
    - ``_row_iter`` validation branches (bad powertrain split, zero weights).
    - ``_header_block`` for all sources + unknown-source error.
    - ``write_sales_mix_csvs`` round-trips through ``load_sales_mix``.
    - ``load_sales_mix`` missing-file branch (monkeypatched dir).
    - ``build_workplace_cohort`` happy path, missing-column errors,
      empty-aggregate error, out-of-band hard fail, and meta-JSON sidecar.
    - ``_aggregate_vehicle_sessions`` best-365-day-window logic.
    - ``main`` CLI: build-sales-mix / build-workplace-cohort (strict +
      calibrated) / build-all, plus the no-subcommand error.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from pev_synth import sales_mix_data as smd

# Calibrated thresholds (the CLI's public-EVWatts band). Mirrors
# ``smd._PUBLIC_EVWATTS_CALIBRATED`` so we can build a synthetic cohort that
# lands inside the [100, 300] hard band.
_CALIBRATED: dict[str, object] = {
    "min_sessions_yr": 50,
    "plug_in_hour_range": (5, 14),
    "plug_out_hour_range": (10, 22),
    "duration_h_range": (1.0, 16.0),
    "avg_kw_range": (2.0, 15.0),
}


# --------------------------------------------------------------------------- #
# Synthetic EVWatts builders.
# --------------------------------------------------------------------------- #


def _make_vehicles(n_qualifying: int = 150, n_extra: int = 20) -> pd.DataFrame:
    """Build a synthetic ``evwatts.public.vehicles.csv`` frame.

    The first ``n_qualifying`` rows are BEV passenger cars (cohort-eligible).
    The ``n_extra`` rows are non-eligible (PHEV / non-passenger / wrong
    electrification label) so the vehicle-filter branch is exercised.
    """
    rows: list[dict[str, object]] = []
    for vid in range(n_qualifying):
        rows.append(
            {
                "id": vid,
                "vehicle_type": "PASSENGER CAR",
                "electrification_level": "BEV (Battery Electric Vehicle)",
                "ownership": "Personal",
                "state": "CA",
            }
        )
    for k in range(n_extra):
        vid = n_qualifying + k
        rows.append(
            {
                "id": vid,
                "vehicle_type": "TRUCK" if k % 2 == 0 else "PASSENGER CAR",
                "electrification_level": "PHEV (Plug-in Hybrid)",
                "ownership": "Fleet",
                "state": "TX",
            }
        )
    return pd.DataFrame(rows)


def _make_sessions(
    n_qualifying: int = 150,
    sessions_per_vehicle: int = 60,
) -> pd.DataFrame:
    """Build a synthetic ``evwatts.public.vehiclesessions.csv`` frame.

    Each qualifying vehicle gets ``sessions_per_vehicle`` clean sessions whose
    per-vehicle medians land inside the calibrated band (plug-in ~08:00,
    duration ~8 h => plug-out ~16:00, avg_kw ~7). A handful of dirty sessions
    (fatal flag, midnight date-only, zero energy, absurd duration) are added to
    exercise the cleaning branches.
    """
    rows: list[dict[str, object]] = []
    base = datetime(2022, 1, 3, 8, 0, 0)  # a Monday, 08:00
    for vid in range(n_qualifying):
        for s in range(sessions_per_vehicle):
            start = base + timedelta(days=s)  # one session per day
            rows.append(
                {
                    "vehicle_id": vid,
                    "start_datetime": start.isoformat(),
                    "duration": 8.0,  # hours -> plug-out 16:00
                    "energy_kwh": 56.0,  # 56 / 8 = 7 kW avg
                    "flag_id": 0,
                }
            )
    # Dirty sessions attached to vehicle 0 — all must be dropped by cleaning.
    dirty = [
        # fatal flag bit 1 (unrealistic kW)
        {"vehicle_id": 0, "start_datetime": "2022-06-01T09:00:00",
         "duration": 8.0, "energy_kwh": 56.0, "flag_id": 1},
        # midnight date-only encoding
        {"vehicle_id": 0, "start_datetime": "2022-06-02T00:00:00",
         "duration": 8.0, "energy_kwh": 56.0, "flag_id": 0},
        # zero energy
        {"vehicle_id": 0, "start_datetime": "2022-06-03T09:00:00",
         "duration": 8.0, "energy_kwh": 0.0, "flag_id": 0},
        # absurd duration (> 48 h)
        {"vehicle_id": 0, "start_datetime": "2022-06-04T09:00:00",
         "duration": 72.0, "energy_kwh": 56.0, "flag_id": 0},
        # unparseable timestamp -> coerced to NaT, dropped
        {"vehicle_id": 0, "start_datetime": "not-a-date",
         "duration": 8.0, "energy_kwh": 56.0, "flag_id": 0},
    ]
    rows.extend(dirty)
    return pd.DataFrame(rows)


def _write_evwatts(tmp_path: Path) -> tuple[Path, Path]:
    """Write synthetic vehicles + sessions CSVs and return their paths."""
    vehicles_path = tmp_path / "vehicles.csv"
    sessions_path = tmp_path / "sessions.csv"
    _make_vehicles().to_csv(vehicles_path, index=False)
    _make_sessions().to_csv(sessions_path, index=False)
    return vehicles_path, sessions_path


# --------------------------------------------------------------------------- #
# Sales-mix DataFrame builder + header block.
# --------------------------------------------------------------------------- #


def test_build_sales_mix_df_all_sources_sum_to_one() -> None:
    """Each source frame sums to 1.0 and carries the canonical columns."""
    for name in smd.SALES_MIX_SOURCES:
        df = smd._build_sales_mix_df(name)
        assert list(df.columns) == list(smd._CSV_COLUMNS)
        total = float(df["share"].sum())
        assert 0.999 <= total <= 1.001, f"{name}: shares sum to {total}"
        assert (df["share"] > 0).all(), f"{name}: non-positive share present"


def test_build_sales_mix_df_bev_phev_split() -> None:
    """BEV rows of cvrp_ca sum to the documented 0.87 BEV share."""
    df = smd._build_sales_mix_df("cvrp_ca")
    # PHEV models all have battery < 30 kWh in the attr table; BEV >= 60.
    bev_share = float(df[df["battery_kwh_nominal"] >= 30]["share"].sum())
    assert abs(bev_share - smd._CVRP_BEV_SHARE) < 1e-6


def test_build_sales_mix_df_unknown_source_raises() -> None:
    with pytest.raises(ValueError, match="unknown sales-mix source"):
        smd._build_sales_mix_df("not_a_source")


def test_header_block_all_sources_cite_url() -> None:
    for name in smd.SALES_MIX_SOURCES:
        lines = smd._header_block(name)
        joined = "\n".join(lines)
        assert f"methodology_version: {smd.METHODOLOGY_VERSION}" in joined
        assert f"master_seed: {smd.MASTER_SEED}" in joined
        assert "https://" in joined, f"{name}: header missing https URL"


def test_header_block_unknown_source_raises() -> None:
    with pytest.raises(ValueError, match="unknown sales-mix source"):
        smd._header_block("not_a_source")


def test_row_iter_bad_powertrain_split_raises() -> None:
    with pytest.raises(ValueError, match="must equal 1.0"):
        smd._row_iter(
            smd._CVRP_BEV, smd._CVRP_PHEV, 0.5, 0.4, "s", "u"
        )


def test_row_iter_zero_weights_raises() -> None:
    with pytest.raises(ValueError, match="must be > 0"):
        smd._row_iter({}, {}, 0.5, 0.5, "s", "u")


# --------------------------------------------------------------------------- #
# write_sales_mix_csvs + load_sales_mix round trip.
# --------------------------------------------------------------------------- #


def test_write_sales_mix_csvs_round_trip(tmp_path: Path, monkeypatch) -> None:
    """Write the three CSVs and re-read each via load_sales_mix."""
    out_dir = tmp_path / "sales_mix"
    written = smd.write_sales_mix_csvs(out_dir)
    assert set(written.keys()) == set(smd.SALES_MIX_SOURCES)
    for name, path in written.items():
        assert path.exists()
        text = path.read_text(encoding="utf-8")
        assert text.startswith("# sales-mix CSV"), f"{name}: missing header block"

    # Point load_sales_mix at the tmp dir and confirm a faithful round trip.
    monkeypatch.setattr(smd, "_DEFAULT_SALES_MIX_DIR", out_dir)
    for name in smd.SALES_MIX_SOURCES:
        df = smd.load_sales_mix(name)
        assert len(df) > 0
        assert set(smd._CSV_COLUMNS).issubset(df.columns)
        assert 0.999 <= float(df["share"].sum()) <= 1.001


def test_load_sales_mix_missing_file_raises(tmp_path: Path, monkeypatch) -> None:
    """A canonical source name with no CSV on disk raises with a hint."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    monkeypatch.setattr(smd, "_DEFAULT_SALES_MIX_DIR", empty_dir)
    with pytest.raises(ValueError, match="build-sales-mix"):
        smd.load_sales_mix("argonne_national")


def test_load_sales_mix_unknown_source_raises() -> None:
    with pytest.raises(ValueError, match="unknown sales-mix source"):
        smd.load_sales_mix("bogus")


# --------------------------------------------------------------------------- #
# Workplace cohort builder.
# --------------------------------------------------------------------------- #


def test_build_workplace_cohort_happy_path(tmp_path: Path) -> None:
    """End-to-end cohort build lands in band and writes parquet + meta."""
    vehicles_path, sessions_path = _write_evwatts(tmp_path)
    out_path = tmp_path / "cohort" / "workplace_subset.parquet"
    res = smd.build_workplace_cohort(
        vehicles_path, sessions_path, out_path, **_CALIBRATED
    )
    assert res.cohort_filter_passed
    assert res.n_vehicles == 150
    assert out_path.exists()

    cohort = pd.read_parquet(out_path)
    assert len(cohort) == 150
    assert cohort["vehicle_id"].is_unique
    # All medians inside calibrated band.
    assert (cohort["plug_in_hr_med"] == 8).all()
    assert (cohort["plug_out_hr_med"] == 16).all()
    assert (cohort["dur_h_med"] == 8.0).all()
    assert (cohort["avg_kw_med"].sub(7.0).abs() < 1e-3).all()
    # Dirty sessions on vehicle 0 were dropped: still exactly 60 clean.
    assert int(cohort[cohort["vehicle_id"] == 0]["n_sessions"].iloc[0]) == 60


def test_build_workplace_cohort_meta_sidecar(tmp_path: Path) -> None:
    vehicles_path, sessions_path = _write_evwatts(tmp_path)
    out_path = tmp_path / "workplace_subset.parquet"
    smd.build_workplace_cohort(
        vehicles_path, sessions_path, out_path, **_CALIBRATED
    )
    meta_path = out_path.with_name(out_path.stem + "_meta.json")
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["methodology_version"] == smd.METHODOLOGY_VERSION
    assert meta["master_seed"] == smd.MASTER_SEED
    assert meta["filter"]["vehicle_type"] == "PASSENGER CAR"
    assert meta["filter"]["electrification_level"] == "BEV"
    assert meta["filter"]["min_sessions_yr"] == 50
    assert meta["n_vehicles"] == 150
    assert meta["cohort_size_band_pass"] is True
    assert meta["input_files"]["vehicles"] == str(vehicles_path)


def test_build_workplace_cohort_missing_vehicle_columns(tmp_path: Path) -> None:
    bad_vehicles = tmp_path / "v.csv"
    pd.DataFrame({"id": [1], "vehicle_type": ["PASSENGER CAR"]}).to_csv(
        bad_vehicles, index=False
    )
    sessions = tmp_path / "s.csv"
    _make_sessions(n_qualifying=1, sessions_per_vehicle=1).to_csv(
        sessions, index=False
    )
    with pytest.raises(ValueError, match="vehicles CSV missing required columns"):
        smd.build_workplace_cohort(bad_vehicles, sessions, tmp_path / "o.parquet")


def test_build_workplace_cohort_missing_session_columns(tmp_path: Path) -> None:
    vehicles = tmp_path / "v.csv"
    _make_vehicles(n_qualifying=1, n_extra=0).to_csv(vehicles, index=False)
    bad_sessions = tmp_path / "s.csv"
    pd.DataFrame({"vehicle_id": [0], "start_datetime": ["2022-01-03T08:00:00"]}).to_csv(
        bad_sessions, index=False
    )
    with pytest.raises(ValueError, match="sessions CSV missing required columns"):
        smd.build_workplace_cohort(vehicles, bad_sessions, tmp_path / "o.parquet")


def test_build_workplace_cohort_empty_after_cleaning(tmp_path: Path) -> None:
    """All sessions fatally flagged -> empty aggregate -> ValueError."""
    vehicles = tmp_path / "v.csv"
    _make_vehicles(n_qualifying=2, n_extra=0).to_csv(vehicles, index=False)
    sessions = tmp_path / "s.csv"
    pd.DataFrame(
        {
            "vehicle_id": [0, 1],
            "start_datetime": ["2022-01-03T08:00:00", "2022-01-04T08:00:00"],
            "duration": [8.0, 8.0],
            "energy_kwh": [56.0, 56.0],
            "flag_id": [1, 1],  # fatal bit -> dropped
        }
    ).to_csv(sessions, index=False)
    with pytest.raises(ValueError, match="no per-vehicle aggregates"):
        smd.build_workplace_cohort(vehicles, sessions, tmp_path / "o.parquet")


def test_build_workplace_cohort_out_of_band_hard_fail(tmp_path: Path) -> None:
    """A cohort below the size band raises (PM gap #8 hard-fail)."""
    vehicles_path, sessions_path = _write_evwatts(tmp_path)
    out_path = tmp_path / "o.parquet"
    # 150 qualifying vehicles, but require a band of [300, 400] -> too small.
    with pytest.raises(ValueError, match=r"workplace cohort size \d+ is outside"):
        smd.build_workplace_cohort(
            vehicles_path,
            sessions_path,
            out_path,
            cohort_size_band=(300, 400),
            **_CALIBRATED,
        )


def test_build_workplace_cohort_byte_identical_rerun(tmp_path: Path) -> None:
    import hashlib

    vehicles_path, sessions_path = _write_evwatts(tmp_path)
    out_a = tmp_path / "a.parquet"
    out_b = tmp_path / "b.parquet"
    smd.build_workplace_cohort(vehicles_path, sessions_path, out_a, **_CALIBRATED)
    smd.build_workplace_cohort(vehicles_path, sessions_path, out_b, **_CALIBRATED)
    assert (
        hashlib.sha256(out_a.read_bytes()).hexdigest()
        == hashlib.sha256(out_b.read_bytes()).hexdigest()
    )


def test_aggregate_vehicle_sessions_best_window() -> None:
    """The best-365-day window keeps the densest run of sessions."""
    # Vehicle 0: 3 sessions clustered in Jan 2022 + 1 outlier far away.
    starts = pd.to_datetime(
        [
            "2022-01-01T08:00:00",
            "2022-01-02T08:00:00",
            "2022-01-03T08:00:00",
            "2024-01-01T08:00:00",
        ]
    )
    df = pd.DataFrame(
        {
            "vehicle_id": [0, 0, 0, 0],
            "start_datetime": starts,
            "_stop_dt": starts + pd.to_timedelta(8, unit="h"),
            "duration": [8.0, 8.0, 8.0, 8.0],
            "_avg_kw": [7.0, 7.0, 7.0, 7.0],
        }
    )
    agg = smd._aggregate_vehicle_sessions(df)
    assert len(agg) == 1
    # Densest 365-day window holds the 3 January sessions, not the 2024 outlier.
    assert int(agg["n_sessions"].iloc[0]) == 3
    assert float(agg["plug_in_hr_med"].iloc[0]) == 8.0
    assert float(agg["plug_out_hr_med"].iloc[0]) == 16.0


# --------------------------------------------------------------------------- #
# CLI (main).
# --------------------------------------------------------------------------- #


def test_main_build_sales_mix(tmp_path: Path, monkeypatch, capsys) -> None:
    out_dir = tmp_path / "sm"
    monkeypatch.setattr(smd, "_DEFAULT_SALES_MIX_DIR", out_dir)
    rc = smd.main(["build-sales-mix"])
    assert rc == 0
    for name in smd.SALES_MIX_SOURCES:
        assert (out_dir / f"{name}.csv").exists()
    err = capsys.readouterr().err
    assert "wrote" in err


def test_main_build_workplace_cohort_calibrated(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    vehicles_path, sessions_path = _write_evwatts(tmp_path)
    out_path = tmp_path / "wp.parquet"
    rc = smd.main(
        [
            "build-workplace-cohort",
            "--vehicles",
            str(vehicles_path),
            "--sessions",
            str(sessions_path),
            "--out",
            str(out_path),
        ]
    )
    assert rc == 0
    assert out_path.exists()
    err = capsys.readouterr().err
    assert "workplace cohort N=150" in err


def test_main_build_workplace_cohort_strict_hard_fails(
    tmp_path: Path,
) -> None:
    """``--strict`` applies §4.4.2 thresholds; our synthetic plug-out (16:00)
    falls in the strict 15-19 band but min_sessions_yr=130 with 60 sessions
    excludes every vehicle -> empty cohort -> hard fail below band."""
    vehicles_path, sessions_path = _write_evwatts(tmp_path)
    out_path = tmp_path / "wp_strict.parquet"
    with pytest.raises(ValueError, match=r"workplace cohort size \d+ is outside"):
        smd.main(
            [
                "build-workplace-cohort",
                "--strict",
                "--vehicles",
                str(vehicles_path),
                "--sessions",
                str(sessions_path),
                "--out",
                str(out_path),
            ]
        )


def test_main_build_all(tmp_path: Path, monkeypatch, capsys) -> None:
    vehicles_path, sessions_path = _write_evwatts(tmp_path)
    sm_dir = tmp_path / "sm"
    cohort_dir = tmp_path / "cohorts"
    monkeypatch.setattr(smd, "_DEFAULT_SALES_MIX_DIR", sm_dir)
    monkeypatch.setattr(smd, "_DEFAULT_COHORT_DIR", cohort_dir)
    monkeypatch.setattr(smd, "_DEFAULT_EVWATTS_VEHICLES", vehicles_path)
    monkeypatch.setattr(smd, "_DEFAULT_EVWATTS_SESSIONS", sessions_path)
    rc = smd.main(["build-all"])
    assert rc == 0
    for name in smd.SALES_MIX_SOURCES:
        assert (sm_dir / f"{name}.csv").exists()
    assert (cohort_dir / "workplace_subset.parquet").exists()
    err = capsys.readouterr().err
    assert "wrote" in err
    assert "workplace cohort N=150" in err


def test_main_no_subcommand_errors() -> None:
    with pytest.raises(SystemExit) as exc:
        smd.main([])
    assert exc.value.code != 0
