"""Unit tests for M3 — pev_synth.sales_mix_data.

Validates the PM-brief acceptance criteria for Module 3:

    1. All three sales-mix CSVs exist at the canonical location.
    2. Each CSV's shares sum to 1.000 ± 0.001.
    3. ``load_sales_mix(<canonical_source>)`` returns a DataFrame.
    4. ``load_sales_mix('unknown')`` raises ValueError.
    5. ``build_workplace_cohort`` runs end-to-end and writes the parquet.
    6. The workplace cohort size N satisfies 100 <= N <= 300.
    7. The filter applies the configured thresholds correctly.
    8. Re-running ``build_workplace_cohort`` is byte-identical (no RNG).
    9. Each CSV header block cites a ``Source URL:``.
   10. The ``argonne_national`` CSV has at least 8 archetype rows covering
       sedan + crossover + suv + pickup × at least 2 models each.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd
import pytest

# Make the `code/` directory importable when run from anywhere.
_CODE_DIR = Path(__file__).resolve().parents[1]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from pev_synth import sales_mix_data as smd  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SALES_MIX_DIR = _REPO_ROOT / "data" / "pev" / "raw" / "literature" / "sales_mix"
_COHORT_PARQUET = (
    _REPO_ROOT / "data" / "pev" / "raw" / "evwatts_cohorts" / "workplace_subset.parquet"
)
_VEHICLES_CSV = _REPO_ROOT / "data" / "pev" / "raw" / "evwatts.public.vehicles.csv"
_SESSIONS_CSV = _REPO_ROOT / "data" / "pev" / "raw" / "evwatts.public.vehiclesessions.csv"

# Skip decorators for tests that need the raw CSVs (NOT shipped in the repo;
# see .gitignore). These guard the disk reads in `load_sales_mix` and
# `build_workplace_cohort` so the suite is green on a clean checkout.
_needs_sales_mix_csvs = pytest.mark.skipif(
    not _SALES_MIX_DIR.exists()
    or not all(
        (_SALES_MIX_DIR / f"{name}.csv").exists()
        for name in ("cvrp_ca", "argonne_national", "nyserda_drive_clean")
    ),
    reason=(
        f"sales-mix CSVs not present under {_SALES_MIX_DIR}; "
        "run `python -m pev_synth.sales_mix_data build-sales-mix` first"
    ),
)

_needs_evwatts_csvs = pytest.mark.skipif(
    not _VEHICLES_CSV.exists() or not _SESSIONS_CSV.exists(),
    reason=(
        f"EVWatts raw CSVs not present at {_VEHICLES_CSV} / {_SESSIONS_CSV}; "
        "run `python -m pev_synth.sales_mix_data build-sales-mix` first"
    ),
)

# Calibrated parameters used by the CLI for the public-EVWatts release.
# These are documented in detail in ``sales_mix_data._PUBLIC_EVWATTS_CALIBRATED``.
_CALIBRATED: dict[str, object] = {
    "min_sessions_yr": 50,
    "plug_in_hour_range": (5, 14),
    "plug_out_hour_range": (10, 22),
    "duration_h_range": (1.0, 16.0),
    "avg_kw_range": (2.0, 15.0),
}


# --------------------------------------------------------------------------- #
# Sales-mix CSV tests.
# --------------------------------------------------------------------------- #


@_needs_sales_mix_csvs
def test_three_csvs_exist() -> None:
    """AC #1: all three CSVs are present at the canonical location."""
    for name in smd.SALES_MIX_SOURCES:
        path = _SALES_MIX_DIR / f"{name}.csv"
        assert path.exists(), f"missing sales-mix CSV {path}"


@_needs_sales_mix_csvs
def test_csv_shares_sum_to_one() -> None:
    """AC #3: shares sum to 1.000 ± 0.001 for each source."""
    for name in smd.SALES_MIX_SOURCES:
        df = smd.load_sales_mix(name)
        total = float(df["share"].sum())
        assert 0.999 <= total <= 1.001, (
            f"{name}: shares sum to {total}, outside [0.999, 1.001]"
        )


@_needs_sales_mix_csvs
def test_csv_header_has_source_url() -> None:
    """AC #2: each CSV's header block cites a ``Source URL:`` line."""
    for name in smd.SALES_MIX_SOURCES:
        path = _SALES_MIX_DIR / f"{name}.csv"
        header_text = path.read_text(encoding="utf-8")
        # All `#`-prefix lines together must contain the source URL.
        header_lines = [
            line for line in header_text.splitlines() if line.startswith("#")
        ]
        joined = "\n".join(header_lines)
        assert "Source URL:" in joined, f"{name}: header missing 'Source URL:' line"
        # And the URL itself must include https://
        assert "https://" in joined, f"{name}: header missing https URL"


@_needs_sales_mix_csvs
def test_csv_has_required_columns() -> None:
    """Schema check: each CSV has the columns mandated by the PM brief."""
    required = {
        "archetype", "make", "model", "model_year_min", "model_year_max",
        "share", "battery_kwh_nominal", "obc_max_kw", "has_heat_pump",
        "source", "source_url",
    }
    for name in smd.SALES_MIX_SOURCES:
        df = smd.load_sales_mix(name)
        missing = required - set(df.columns)
        assert not missing, f"{name}: missing columns {missing}"


@_needs_sales_mix_csvs
def test_csv_archetypes_subset_of_canonical() -> None:
    """Every CSV's `archetype` values are a subset of the canonical 4-tuple."""
    canonical = set(smd.ARCHETYPES)
    for name in smd.SALES_MIX_SOURCES:
        df = smd.load_sales_mix(name)
        used = set(df["archetype"].unique())
        extras = used - canonical
        assert not extras, f"{name}: unexpected archetypes {extras}"


@_needs_sales_mix_csvs
def test_load_sales_mix_returns_dataframe() -> None:
    """AC #4: load_sales_mix returns a non-empty DataFrame for each source."""
    for name in smd.SALES_MIX_SOURCES:
        df = smd.load_sales_mix(name)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0, f"{name}: empty DataFrame"


def test_load_sales_mix_unknown_raises() -> None:
    """AC #5: load_sales_mix raises ValueError on an unknown source name."""
    with pytest.raises(ValueError, match="unknown sales-mix source"):
        smd.load_sales_mix("unknown")


@_needs_sales_mix_csvs
def test_argonne_has_8_rows_with_2_models_per_archetype() -> None:
    """AC #10: argonne_national has >= 8 rows with >= 2 models per archetype."""
    df = smd.load_sales_mix("argonne_national")
    assert len(df) >= 8, f"argonne_national has only {len(df)} rows; need >= 8"
    by_arch = df.groupby("archetype")["model"].nunique()
    for arch in ("sedan", "crossover", "suv", "pickup"):
        assert arch in by_arch.index, f"argonne_national missing archetype {arch}"
        assert by_arch[arch] >= 2, (
            f"argonne_national has only {by_arch[arch]} models for {arch}; need >= 2"
        )


@_needs_sales_mix_csvs
def test_argonne_pickup_share_positive() -> None:
    """Behavioural: argonne_national pickup share is the DFW overlay base."""
    df = smd.load_sales_mix("argonne_national")
    pickup = float(df[df["archetype"] == "pickup"]["share"].sum())
    # Public Argonne LD EV Monthly Sales Update 2024-2025: pickups ~13-17%
    # of US BEV+PHEV sales. The exact value depends on the rollup window.
    assert 0.10 <= pickup <= 0.25, (
        f"argonne_national pickup share {pickup} outside [0.10, 0.25]; "
        "DFW overlay (+0.10) would produce an unrealistic post-renormalisation share"
    )


# --------------------------------------------------------------------------- #
# Workplace cohort tests.
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def cohort_df() -> pd.DataFrame:
    """Load the produced workplace cohort parquet once per test session."""
    if not _COHORT_PARQUET.exists():
        pytest.skip(f"cohort parquet not built at {_COHORT_PARQUET}")
    return pd.read_parquet(_COHORT_PARQUET)


@_needs_evwatts_csvs
def test_workplace_cohort_runs_end_to_end(tmp_path: Path) -> None:
    """AC #6: build_workplace_cohort writes the parquet end-to-end."""
    out = tmp_path / "workplace_subset.parquet"
    res = smd.build_workplace_cohort(
        _VEHICLES_CSV, _SESSIONS_CSV, out, **_CALIBRATED
    )
    assert res.cohort_filter_passed
    assert out.exists()
    assert res.n_vehicles == len(pd.read_parquet(out))


def test_workplace_cohort_size_in_band(cohort_df: pd.DataFrame) -> None:
    """AC #7: cohort size N is in [100, 300]."""
    assert 100 <= len(cohort_df) <= 300, (
        f"cohort size {len(cohort_df)} outside [100, 300]"
    )


def test_workplace_cohort_filter_uses_correct_thresholds(
    cohort_df: pd.DataFrame,
) -> None:
    """Every row satisfies every configured filter inequality."""
    n_lo = _CALIBRATED["min_sessions_yr"]
    pi_lo, pi_hi = _CALIBRATED["plug_in_hour_range"]  # type: ignore[misc]
    po_lo, po_hi = _CALIBRATED["plug_out_hour_range"]  # type: ignore[misc]
    dur_lo, dur_hi = _CALIBRATED["duration_h_range"]  # type: ignore[misc]
    kw_lo, kw_hi = _CALIBRATED["avg_kw_range"]  # type: ignore[misc]

    assert (cohort_df["n_sessions"] >= n_lo).all()
    assert ((cohort_df["plug_in_hr_med"] >= pi_lo)
            & (cohort_df["plug_in_hr_med"] <= pi_hi)).all()
    assert ((cohort_df["plug_out_hr_med"] >= po_lo)
            & (cohort_df["plug_out_hr_med"] <= po_hi)).all()
    assert ((cohort_df["dur_h_med"] >= dur_lo)
            & (cohort_df["dur_h_med"] <= dur_hi)).all()
    assert ((cohort_df["avg_kw_med"] >= kw_lo)
            & (cohort_df["avg_kw_med"] <= kw_hi)).all()


@_needs_evwatts_csvs
def test_workplace_cohort_strict_band_hard_fail() -> None:
    """AC #8: Out-of-band cohort raises ValueError. We force a size of 0 by
    passing the strict §4.4.2 thresholds, which yield 0 vehicles on the
    public EVWatts release (documented in module header).
    """
    # Use a temp out path so we don't perturb the deliverable parquet.
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "wp.parquet"
        # The function defaults match plan §4.4.2 strict thresholds; passing
        # no overrides should hard-fail on the public release.
        with pytest.raises(ValueError, match=r"workplace cohort size \d+ is outside"):
            smd.build_workplace_cohort(_VEHICLES_CSV, _SESSIONS_CSV, out)


@_needs_evwatts_csvs
def test_workplace_cohort_byte_identical_rerun(tmp_path: Path) -> None:
    """AC #9: re-running the cohort filter produces byte-identical output."""
    out_a = tmp_path / "a.parquet"
    out_b = tmp_path / "b.parquet"
    smd.build_workplace_cohort(_VEHICLES_CSV, _SESSIONS_CSV, out_a, **_CALIBRATED)
    smd.build_workplace_cohort(_VEHICLES_CSV, _SESSIONS_CSV, out_b, **_CALIBRATED)
    sha_a = hashlib.sha256(out_a.read_bytes()).hexdigest()
    sha_b = hashlib.sha256(out_b.read_bytes()).hexdigest()
    assert sha_a == sha_b, "cohort parquet byte-equality failed across reruns"


def test_workplace_cohort_schema(cohort_df: pd.DataFrame) -> None:
    """Cohort parquet has the documented columns and primary-key uniqueness."""
    expected_cols = {
        "vehicle_id", "n_sessions", "plug_in_hr_med", "plug_out_hr_med",
        "dur_h_med", "avg_kw_med", "ownership", "state",
        "best_window_start", "best_window_end",
    }
    assert expected_cols.issubset(set(cohort_df.columns)), (
        f"cohort parquet missing columns: {expected_cols - set(cohort_df.columns)}"
    )
    assert cohort_df["vehicle_id"].is_unique, "vehicle_id must be unique"


@_needs_evwatts_csvs
def test_workplace_cohort_meta_json_emitted(tmp_path: Path) -> None:
    """Sidecar meta JSON exists and records the filter parameters."""
    import json
    out = tmp_path / "workplace_subset.parquet"
    smd.build_workplace_cohort(_VEHICLES_CSV, _SESSIONS_CSV, out, **_CALIBRATED)
    meta_path = out.with_name(out.stem + "_meta.json")
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text())
    assert meta["methodology_version"] == "v3.0"  # RFC-016; v3.0 bump
    assert meta["master_seed"] == 20260520
    assert meta["filter"]["vehicle_type"] == "PASSENGER CAR"
    assert meta["filter"]["electrification_level"] == "BEV"
    assert 100 <= meta["n_vehicles"] <= 300
