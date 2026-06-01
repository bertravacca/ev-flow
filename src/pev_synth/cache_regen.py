"""pev_synth.cache_regen — v2.0 cache regeneration orchestrator.

End-to-end driver that takes a (region, profile_type) pair through the
6-module pipeline (M2 → M3 → M4 → M5 → M6 → M7) and writes the six
parquet artifacts plus the unified ``meta.json`` to

    data/pev/processed/{region.name}/{profile_type}_ev_synth/

Wave 3 Dev I owns this script.  It does NOT modify any Wave 1 / Wave 2
module — it only consumes their public APIs.  Per RFC-012 the M5 module
now emits the brief schema (``location_type`` + ``plug_in:bool`` + the
M6-facing ``location`` synonym) directly, so the previous
``_bridge_events_to_m6_schema`` shim is gone.

Math notation: ASCII.  Lint clean.  No emoji.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa

# Module APIs (do not modify these — consume only).
from ._seeds import (
    MASTER_SEED as _MASTER_SEED,
)
from ._seeds import (
    METHODOLOGY_VERSION as _METHODOLOGY_VERSION,
)
from .donor_matcher import match_donors
from .hourly_resampler import build_plug_status
from .plug_in_model import (
    fit_and_assign_plug_ins,
    update_meta_with_m5,
    write_bic_sweep_results,
    write_plug_in_events,
)
from .regions import REGIONS, Region, list_profile_types, list_regions
from .soc_trajectory import compute_sessions
from .travel_week_builder import build_mobility
from .vehicle_archetypes import sample_fleet, write_fleet, write_meta

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants & helpers
# ---------------------------------------------------------------------------

MASTER_SEED: int = _MASTER_SEED
METHODOLOGY_VERSION: str = _METHODOLOGY_VERSION

#: us_national uses a smaller fleet for the comparative reference per the
#: Dev I brief.  All other regions default to N=1000.
N_DEFAULT: int = 1000
N_US_NATIONAL: int = 500


def _repo_root() -> Path:
    """Return the absolute repo root (the study folder).

    Thin wrapper over :func:`pev_synth._paths.repo_root` for backward
    compatibility with the existing public API.
    """
    from ._paths import repo_root

    return repo_root()


def _out_dir(
    repo_root: Path,
    region_name: str,
    profile_type: str,
    *,
    replicate_id: int = 0,
    r_total: int = 1,
) -> Path:
    """Resolve the per-replicate output directory.

    RFC-022.r: ``R == 1`` keeps the flat ``<region>/<profile_type>_ev_synth/``
    layout (alias ``r=0``). ``R > 1`` nests outputs under
    ``replicates/r{replicate_id}/``. Delegates to
    :func:`pev_synth._paths.replicate_dir` so the path semantics live in
    a single place and stay in lock-step with the validator-side reader
    (WP-A / RFC-033).
    """
    from ._paths import replicate_dir as _replicate_dir

    # _replicate_dir is anchored at the canonical repo_root via repo_root().
    # When the caller passes a non-default repo_root (tests / sandboxes),
    # we honour it by reconstructing the leaf manually.
    from ._paths import repo_root as _default_repo_root

    if Path(repo_root).resolve() == _default_repo_root().resolve():
        return _replicate_dir(region_name, profile_type, replicate_id, r_total)

    base = (
        Path(repo_root) / "data" / "pev" / "processed"
        / region_name / f"{profile_type}_ev_synth"
    )
    if r_total <= 1:
        return base
    return base / "replicates" / f"r{int(replicate_id)}"


def _n_for_region(region_name: str) -> int:
    return N_US_NATIONAL if region_name == "us_national" else N_DEFAULT


# ---------------------------------------------------------------------------
# RFC-012: the M5→M6 schema-bridging shim (`_bridge_events_to_m6_schema`)
# was removed once M5 was updated to emit the brief schema directly
# (``location_type`` + ``plug_in:bool`` + ``location`` synonym).
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Per-cache driver
# ---------------------------------------------------------------------------

def regenerate_cache(
    region_name: str,
    profile_type: str,
    *,
    n: int | None = None,
    seed: int = MASTER_SEED,
    repo_root: Path | None = None,
    overwrite: bool = True,
    replicate_id: int = 0,
    r_total: int = 1,
) -> dict[str, Any]:
    """Generate one (region, profile_type) cache end-to-end.

    Returns a dict with status + per-step metrics.  On failure the dict
    has ``status == "FAIL"`` and a captured ``traceback``; we never raise
    so the caller can continue with other caches.

    RFC-022.r: when ``r_total > 1`` the output bundle is written to
    ``<cache_dir>/replicates/r{replicate_id}/`` rather than the flat
    cache root. ``r_total == 1`` keeps the legacy flat layout (the
    ``replicate_id`` is the implicit alias ``r=0``).
    """
    repo_root = repo_root or _repo_root()
    if region_name not in REGIONS:
        return {
            "region": region_name,
            "profile_type": profile_type,
            "status": "FAIL",
            "error": f"unknown region {region_name!r}",
        }
    if profile_type not in list_profile_types():
        return {
            "region": region_name,
            "profile_type": profile_type,
            "status": "FAIL",
            "error": f"unknown profile_type {profile_type!r}",
        }

    region = Region.from_name(region_name)
    n_eff = int(n) if n is not None else _n_for_region(region_name)

    out_dir = _out_dir(
        repo_root,
        region.name,
        profile_type,
        replicate_id=replicate_id,
        r_total=r_total,
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    timings: dict[str, float] = {}
    metrics: dict[str, Any] = {}

    try:
        # --- Step 1: M2 fleet -------------------------------------------
        t0 = time.time()
        fleet = sample_fleet(
            n=n_eff, seed=seed, region=region, profile_type=profile_type,
        )
        bev_share = float((fleet["powertrain"].astype(str) == "BEV").mean())
        timings["m2_fleet"] = time.time() - t0
        metrics["n_fleet"] = int(len(fleet))
        metrics["bev_share"] = bev_share
        metrics["pickup_share"] = float(
            (fleet["archetype"].astype(str) == "pickup").mean()
        )

        # Write fleet.parquet + bootstrap meta.json (M2 provenance).
        fleet_path = out_dir / "fleet.parquet"
        write_fleet(fleet, fleet_path)
        write_meta(
            fleet,
            out_dir / "meta.json",
            master_seed=seed,
            bev_share=bev_share,
            region=region,
            profile_type=profile_type,
        )

        # --- Step 2: M3 donor matching ----------------------------------
        # RFC-017: pass nhts_data_dir explicitly so M3 and M4 agree on
        # which NHTS source they consume. Until RFC-014 lands and M1
        # emits per-region parquets, the raw NHTS dir is the
        # authoritative source for both M3 and M4 (CSV fall-back inside
        # the loaders). The argument shape future-proofs the swap.
        from ._paths import repo_root as _repo_root_fn

        _nhts_dir = _repo_root_fn() / "data" / "pev" / "raw" / "nhts2017"

        t0 = time.time()
        fleet = match_donors(
            fleet,
            region=region,
            profile_type=profile_type,
            seed=seed,
            nhts_data_dir=_nhts_dir,
        )
        timings["m3_donor"] = time.time() - t0
        donor_attrs = dict(fleet.attrs.get("donor_match", {}))
        metrics["donor_match"] = donor_attrs
        metrics["donor_borrow_rate"] = float(donor_attrs.get("borrow_share", 0.0))

        # Persist enriched fleet (donor columns appended).
        write_fleet(fleet, fleet_path)

        # --- Step 3: M4 mobility ----------------------------------------
        # RFC-017: same nhts_data_dir as M3 — guarantees byte-identical
        # NHTS frames feed both stages.
        t0 = time.time()
        mobility = build_mobility(
            fleet, region=region, profile_type=profile_type, seed=seed,
            nhts_data_dir=_nhts_dir,
        )
        timings["m4_mobility"] = time.time() - t0
        metrics["n_mobility"] = int(len(mobility))
        if len(mobility):
            winter_mask = mobility["start_ts"].dt.month.isin([12, 1, 2, 3])
            metrics["f_T_winter_mean"] = (
                float(mobility.loc[winter_mask, "f_T_applied"].mean())
                if int(winter_mask.sum()) > 0 else 1.0
            )
            metrics["mean_kwh_per_mi_winter"] = (
                float(
                    mobility.loc[winter_mask, "kwh_consumed"].sum()
                     / max(mobility.loc[winter_mask, "miles"].sum(), 1e-9)
                ) if int(winter_mask.sum()) > 0 else float("nan")
            )
            metrics["mean_kwh_per_mi_total"] = (
                float(
                    mobility["kwh_consumed"].sum()
                    / max(mobility["miles"].sum(), 1e-9)
                )
            )
        else:
            metrics["f_T_winter_mean"] = 1.0
            metrics["mean_kwh_per_mi_winter"] = float("nan")
            metrics["mean_kwh_per_mi_total"] = float("nan")

        mob_path = out_dir / "mobility.parquet"
        mobility.to_parquet(mob_path, engine="pyarrow", index=False,
                            compression="snappy")

        # --- Step 4: M5 plug-in model -----------------------------------
        t0 = time.time()
        fleet_aug, events, cluster_fit = fit_and_assign_plug_ins(
            fleet, mobility, region=region, profile_type=profile_type,
            seed=seed,
        )
        timings["m5_plug_in"] = time.time() - t0
        metrics["n_plug_in_events"] = int(len(events))
        metrics["K_chosen"] = int(cluster_fit.K_chosen)
        metrics["bic_chosen"] = (
            float(cluster_fit.bic_chosen)
            if np.isfinite(cluster_fit.bic_chosen) else None
        )
        metrics["cluster_centres_h"] = list(cluster_fit.cluster_centres_h)

        # Persist augmented fleet + events + BIC.
        write_fleet(fleet_aug, fleet_path)
        events_path = out_dir / "plug_in_events.parquet"
        write_plug_in_events(events, events_path)
        # RFC-011: brief layout puts ``bic_sweep_{profile_type}.json`` at
        # the cache root (no ``intermediate/`` subdir). Residential bundles
        # also receive a sweep file for repro completeness.
        write_bic_sweep_results(
            cluster_fit, out_dir / f"bic_sweep_{profile_type}.json",
        )

        # Aggregate plug-in rate vs eligible arrivals (matching M5 helper).
        target_loc = "work" if profile_type == "workplace" else "home"
        n_arrivals = int((mobility["dest_type"].astype(str) == target_loc).sum())
        aggregate_rate = (
            float(len(events) / n_arrivals) if n_arrivals > 0 else 0.0
        )
        metrics["aggregate_plug_in_rate"] = aggregate_rate

        cluster_size_distribution = {
            int(k): int(v)
            for k, v in pd.Series(fleet_aug["cluster_z"])
            .value_counts().sort_index().items()
        }
        # cohort_size from the workplace cohort file if applicable.
        cohort_size = 0
        if profile_type == "workplace":
            cohort_path = (
                _repo_root() / "data" / "pev" / "raw" / "evwatts_cohorts"
                / "workplace_subset.parquet"
            )
            if cohort_path.is_file():
                cohort_size = int(len(pd.read_parquet(cohort_path)))

        update_meta_with_m5(
            out_dir / "meta.json",
            profile_type=profile_type,
            region=region,
            cluster_fit=cluster_fit,
            cluster_size_distribution=cluster_size_distribution,
            aggregate_plug_in_rate_value=aggregate_rate,
            cohort_size=cohort_size,
        )

        # --- Step 5: M6 sessions ----------------------------------------
        # RFC-012: M5 now emits the brief schema (location_type +
        # plug_in:bool) and the M6-facing ``location`` synonym; no
        # bridging shim required.
        t0 = time.time()
        sessions = compute_sessions(
            fleet_aug, mobility, events,
            region=region, profile_type=profile_type, seed=seed,
        )
        timings["m6_sessions"] = time.time() - t0
        metrics["n_sessions"] = int(len(sessions))

        sess_path = out_dir / "sessions.parquet"
        # Sort for deterministic byte-output.
        sessions = (
            sessions.sort_values(["ev_id", "session_idx"], kind="mergesort")
            .reset_index(drop=True)
        )
        sessions.to_parquet(
            sess_path, engine="pyarrow", index=False, compression="snappy",
        )

        # --- Step 6: M7 plug-status rasteriser --------------------------
        t0 = time.time()
        raster_paths = build_plug_status(
            sessions_path=sess_path, region=region,
            profile_type=profile_type, seed=seed, n_vehicles=n_eff,
            repo_root=repo_root,
        )
        timings["m7_raster"] = time.time() - t0

        # Plug rate from the rasterised 15-min table.
        plug_15 = pd.read_parquet(
            raster_paths["plug_status_15min"], columns=["plugged"],
        )
        plug_rate_15 = float(plug_15["plugged"].mean()) if len(plug_15) else 0.0
        metrics["plug_rate_15min"] = plug_rate_15
        metrics["n_plug_status_15min"] = int(len(plug_15))

        plug_hr = pd.read_parquet(
            raster_paths["plug_status_hourly"], columns=["plugged"],
        )
        plug_rate_hr = float(plug_hr["plugged"].mean()) if len(plug_hr) else 0.0
        metrics["plug_rate_hourly"] = plug_rate_hr
        metrics["n_plug_status_hourly"] = int(len(plug_hr))

        # --- Step 7: unified meta.json ----------------------------------
        _finalise_meta(
            out_dir / "meta.json", region=region, profile_type=profile_type,
            metrics=metrics, timings=timings, seed=seed, n=n_eff,
        )

        # --- File sizes -------------------------------------------------
        file_sizes = {}
        for fname in (
            "fleet.parquet", "mobility.parquet", "plug_in_events.parquet",
            "sessions.parquet", "plug_status_15min.parquet",
            "plug_status_hourly.parquet", "meta.json",
        ):
            p = out_dir / fname
            file_sizes[fname] = int(p.stat().st_size) if p.is_file() else 0
        metrics["file_sizes_bytes"] = file_sizes
        metrics["file_sizes_total_bytes"] = int(sum(file_sizes.values()))

        wall = time.time() - t_start
        timings["total_wall_clock"] = wall

        return {
            "region": region.name,
            "profile_type": profile_type,
            "status": "PASS",
            "n": n_eff,
            "metrics": metrics,
            "timings": timings,
            "out_dir": str(out_dir),
        }

    except Exception as exc:  # noqa: BLE001 — we want to capture & continue
        tb = traceback.format_exc()
        logger.error("regenerate_cache(%s, %s) FAILED: %s",
                     region_name, profile_type, exc)
        return {
            "region": region_name,
            "profile_type": profile_type,
            "status": "FAIL",
            "error": str(exc),
            "traceback": tb,
            "timings": timings,
            "metrics": metrics,
            "elapsed_s_until_fail": time.time() - t_start,
        }


def _finalise_meta(
    meta_path: Path,
    *,
    region: Region,
    profile_type: str,
    metrics: dict[str, Any],
    timings: dict[str, float],
    seed: int,
    n: int,
) -> None:
    """Merge the cache_regen summary block into the unified meta.json."""
    if meta_path.is_file():
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
    else:
        meta = {}

    meta["methodology_version"] = METHODOLOGY_VERSION
    meta["cache_provenance"] = "v2.0_native"
    meta["tz_storage"] = "UTC"
    meta["tz_local_anchor"] = region.tz
    meta["profile_type"] = profile_type
    meta["master_seed"] = int(seed)
    meta["n_vehicles"] = int(n)
    meta["run_timestamp_utc"] = datetime.now(timezone.utc).isoformat()

    meta.setdefault("region", {})
    meta["region"].update(
        {
            "name": region.name,
            "display_name": region.display_name,
            "states": list(region.states),
            "cbsas": list(region.cbsas),
            "tz": region.tz,
            "iso_market": region.iso_market,
            "sales_mix_source": region.sales_mix_source,
            "sales_mix_overlay": dict(region.sales_mix_overlay),
            "winter_f_T_multiplier": float(region.winter_f_T_multiplier),
            "workplace_donor_borrow_states": list(
                region.workplace_donor_borrow_states
            ),
            "income_tier_scheme": region.income_tier_scheme,
        }
    )

    meta["cache_regen"] = {
        "module": "pev_synth.cache_regen",
        "methodology_version": METHODOLOGY_VERSION,
        "n_vehicles": int(n),
        "metrics": _coerce_for_json(metrics),
        "timings_s": _coerce_for_json(timings),
        "platform": {
            "python": platform.python_version(),
            "system": platform.system(),
            "release": platform.release(),
        },
    }

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True, default=str)


def _coerce_for_json(obj: Any) -> Any:
    """Recursively coerce numpy scalars / Path / nan to JSON-safe types."""
    if isinstance(obj, dict):
        return {str(k): _coerce_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_coerce_for_json(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if not np.isfinite(v) else v
    if isinstance(obj, np.ndarray):
        return [_coerce_for_json(v) for v in obj.tolist()]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float):
        return None if not np.isfinite(obj) else obj
    return obj


# ---------------------------------------------------------------------------
# Batch orchestration
# ---------------------------------------------------------------------------

PRIORITY_ORDER: tuple[tuple[str, str], ...] = (
    ("bay_area", "workplace"),
    ("us_national", "residential"),
    ("us_national", "workplace"),
    ("la_basin", "residential"),
    ("la_basin", "workplace"),
    ("new_york_metro", "residential"),
    ("new_york_metro", "workplace"),
    ("boston", "residential"),
    ("boston", "workplace"),
    ("chicago", "residential"),
    ("chicago", "workplace"),
    ("dallas_fort_worth", "residential"),
    ("dallas_fort_worth", "workplace"),
    ("seattle", "residential"),
    ("seattle", "workplace"),
)
# 15 caches; bay_area residential is the pre-existing v1.1_migrated cache
# and is intentionally excluded from the priority list.


def regenerate_batch(
    pairs: list[tuple[str, str]],
    *,
    seed: int = MASTER_SEED,
    repo_root: Path | None = None,
    time_budget_s: float | None = None,
    log_path: Path | None = None,
    r_total: int = 1,
) -> dict[str, Any]:
    """Run :func:`regenerate_cache` over an ordered list of pairs.

    Stops early (with status ``DEFERRED`` on remaining pairs) once
    ``time_budget_s`` is exhausted.  Always returns a full report.

    RFC-022.r: ``r_total > 1`` expands every pair into ``r_total``
    replicate runs under ``replicates/r{0..r_total-1}/``. Each
    replicate seed is offset by its index so the runs are deterministic
    but independent.
    """
    repo_root = repo_root or _repo_root()
    results: list[dict[str, Any]] = []
    t0 = time.time()

    expanded: list[tuple[str, str, int]] = [
        (region, profile, r)
        for (region, profile) in pairs
        for r in range(max(1, int(r_total)))
    ]

    for i, (region_name, profile_type, r) in enumerate(expanded):
        elapsed = time.time() - t0
        if time_budget_s is not None and elapsed > time_budget_s:
            results.append({
                "region": region_name,
                "profile_type": profile_type,
                "replicate_id": r,
                "status": "DEFERRED",
                "reason": f"time budget {time_budget_s:.0f}s exhausted "
                          f"(elapsed {elapsed:.0f}s)",
            })
            continue

        logger.info(
            "[%d/%d] regenerating %s × %s (r=%d/R=%d, elapsed=%.0fs)",
            i + 1, len(expanded), region_name, profile_type,
            r, r_total, elapsed,
        )
        res = regenerate_cache(
            region_name, profile_type,
            seed=int(seed) + int(r),
            repo_root=repo_root,
            replicate_id=r,
            r_total=r_total,
        )
        res["replicate_id"] = r
        res["r_total"] = int(r_total)
        results.append(res)
        if log_path is not None:
            try:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with log_path.open("w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "completed_so_far": len(results),
                            "results": _coerce_for_json(results),
                        },
                        f, indent=2,
                    )
            except OSError:
                pass

    return {
        "completed": len(results),
        "results": results,
        "total_wall_clock_s": time.time() - t0,
    }


# ---------------------------------------------------------------------------
# Cross-region consistency checks (C1–C5)
# ---------------------------------------------------------------------------

CROSS_REGION_OUT_DIR_REL: Path = (
    Path("data") / "pev" / "validation_bounds" / "_cross_region"
)
CROSS_REGION_C5_MIN_RATIO: float = 1.15
CROSS_REGION_C4_PICKUP_DELTA_PP: float = 0.05  # DFW vs bay_area
COLD_REGIONS: tuple[str, ...] = ("boston", "chicago", "new_york_metro")


def _processed_root(repo_root: Path) -> Path:
    return repo_root / "data" / "pev" / "processed"


def _cache_dir(repo_root: Path, region_name: str, profile_type: str) -> Path:
    return _processed_root(repo_root) / region_name / f"{profile_type}_ev_synth"


def _existing_caches(repo_root: Path) -> list[tuple[str, str]]:
    """Return list of (region, profile_type) that have a complete cache."""
    out: list[tuple[str, str]] = []
    proc = _processed_root(repo_root)
    if not proc.is_dir():
        return out
    for region_name in list_regions():
        for ptype in list_profile_types():
            d = _cache_dir(repo_root, region_name, ptype)
            if not d.is_dir():
                continue
            if not (d / "fleet.parquet").is_file():
                continue
            out.append((region_name, ptype))
    return out


def _is_utc(series: pd.Series) -> bool:
    """True iff Series has tz-aware datetime dtype with UTC tz."""
    dtype = series.dtype
    tz = getattr(dtype, "tz", None)
    if tz is None:
        return False
    return str(tz) == "UTC"


def _ts_columns(df: pd.DataFrame) -> list[str]:
    """Names of all datetime columns (tz-aware or naive) in df."""
    out: list[str] = []
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_datetime64_any_dtype(s):
            out.append(col)
    return out


def run_cross_region_checks(repo_root: Path | None = None) -> dict[str, Any]:
    """Run C1-C5 across every existing v2.0_native cache.

    Returns a report dict and writes ``cross_region_report.{json,md}``
    under ``data/pev/validation_bounds/_cross_region/``.
    """
    repo_root = repo_root or _repo_root()
    caches = _existing_caches(repo_root)
    report: dict[str, Any] = {
        "methodology_version": METHODOLOGY_VERSION,
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "caches_inspected": [
            {"region": r, "profile_type": p} for r, p in caches
        ],
        "n_caches": len(caches),
    }

    # ---- C1: fleet size ------------------------------------------------
    c1_rows: list[dict[str, Any]] = []
    c1_all_pass = True
    for region_name, profile_type in caches:
        d = _cache_dir(repo_root, region_name, profile_type)
        try:
            df = pd.read_parquet(d / "fleet.parquet", columns=["ev_id"])
            n_rows = int(len(df))
        except (OSError, pa.ArrowInvalid, ValueError) as exc:
            c1_rows.append({
                "region": region_name, "profile_type": profile_type,
                "n_rows": None, "expected": None, "pass": False,
                "error": str(exc),
            })
            c1_all_pass = False
            continue
        expected = _n_for_region(region_name)
        passed = n_rows == expected
        c1_rows.append({
            "region": region_name, "profile_type": profile_type,
            "n_rows": n_rows, "expected": expected, "pass": bool(passed),
        })
        if not passed:
            c1_all_pass = False
    report["C1_fleet_size"] = {
        "verdict": "PASS" if c1_all_pass else "FAIL",
        "rows": c1_rows,
    }

    # ---- C2: UTC timestamps --------------------------------------------
    c2_rows: list[dict[str, Any]] = []
    c2_all_pass = True
    parquets_with_ts = (
        "mobility.parquet", "sessions.parquet", "plug_in_events.parquet",
        "plug_status_15min.parquet", "plug_status_hourly.parquet",
    )
    for region_name, profile_type in caches:
        d = _cache_dir(repo_root, region_name, profile_type)
        for fname in parquets_with_ts:
            p = d / fname
            if not p.is_file():
                continue
            try:
                df = pd.read_parquet(p)
            except (OSError, pa.ArrowInvalid, ValueError) as exc:
                c2_rows.append({
                    "region": region_name, "profile_type": profile_type,
                    "file": fname, "ts_cols": [], "pass": False,
                    "error": str(exc),
                })
                c2_all_pass = False
                continue
            ts_cols = _ts_columns(df)
            non_utc = [c for c in ts_cols if not _is_utc(df[c])]
            passed = len(non_utc) == 0 and len(ts_cols) > 0
            c2_rows.append({
                "region": region_name, "profile_type": profile_type,
                "file": fname, "ts_cols": ts_cols,
                "non_utc_cols": non_utc, "pass": bool(passed),
            })
            if not passed:
                c2_all_pass = False
    report["C2_utc_timestamps"] = {
        "verdict": "PASS" if c2_all_pass else "FAIL",
        "rows": c2_rows,
    }

    # ---- C3: schema consistency (fleet.parquet) ------------------------
    c3_pass = True
    fleet_schemas: dict[str, dict[str, str]] = {}
    for region_name, profile_type in caches:
        d = _cache_dir(repo_root, region_name, profile_type)
        try:
            df = pd.read_parquet(d / "fleet.parquet")
        except (OSError, pa.ArrowInvalid, ValueError):
            continue
        key = f"{region_name}/{profile_type}"
        fleet_schemas[key] = {c: str(df[c].dtype) for c in df.columns}

    # Group by profile_type — residential vs workplace allowed to differ.
    by_profile: dict[str, list[tuple[str, dict[str, str]]]] = {}
    for key, schema in fleet_schemas.items():
        ptype = key.split("/")[-1]
        by_profile.setdefault(ptype, []).append((key, schema))

    c3_details: list[dict[str, Any]] = []
    for ptype, lst in by_profile.items():
        if len(lst) < 2:
            c3_details.append({
                "profile_type": ptype, "n_caches": len(lst), "pass": True,
                "note": "single cache, trivially consistent",
            })
            continue
        ref_key, ref_schema = lst[0]
        ref_cols = set(ref_schema.keys())
        all_ok = True
        diffs: list[dict[str, Any]] = []
        for key, schema in lst[1:]:
            cols = set(schema.keys())
            missing = sorted(ref_cols - cols)
            extra = sorted(cols - ref_cols)
            dtype_mismatch = [
                {"col": c, "ref_dtype": ref_schema[c], "this_dtype": schema[c]}
                for c in (ref_cols & cols)
                if ref_schema[c] != schema[c]
            ]
            ok = (not missing) and (not extra) and (not dtype_mismatch)
            if not ok:
                all_ok = False
                diffs.append({
                    "ref": ref_key, "this": key, "missing_cols": missing,
                    "extra_cols": extra, "dtype_mismatch": dtype_mismatch,
                })
        c3_details.append({
            "profile_type": ptype, "n_caches": len(lst),
            "ref_key": ref_key, "pass": bool(all_ok), "diffs": diffs,
        })
        if not all_ok:
            c3_pass = False
    report["C3_schema_consistency"] = {
        "verdict": "PASS" if c3_pass else "FAIL",
        "by_profile": c3_details,
    }

    # ---- C4: sales mix (DFW pickup share vs bay_area) ------------------
    c4_rows: list[dict[str, Any]] = []
    pickup_share_by_region: dict[str, float] = {}
    for region_name in ("dallas_fort_worth", "bay_area", "us_national"):
        d = _cache_dir(repo_root, region_name, "residential")
        if not (d / "fleet.parquet").is_file():
            continue
        try:
            df = pd.read_parquet(d / "fleet.parquet", columns=["archetype"])
            pickup = float((df["archetype"].astype(str) == "pickup").mean())
            pickup_share_by_region[region_name] = pickup
            c4_rows.append({
                "region": region_name, "pickup_share": pickup,
            })
        except (OSError, pa.ArrowInvalid, ValueError) as exc:
            c4_rows.append({
                "region": region_name, "pickup_share": None,
                "error": str(exc),
            })

    c4_verdict = "SKIP"
    c4_notes: list[str] = []
    if "dallas_fort_worth" in pickup_share_by_region \
            and "bay_area" in pickup_share_by_region:
        dfw = pickup_share_by_region["dallas_fort_worth"]
        bay = pickup_share_by_region["bay_area"]
        c4_a = dfw >= bay + CROSS_REGION_C4_PICKUP_DELTA_PP
        c4_notes.append(
            f"DFW pickup={dfw:.3f} >= bay_area pickup={bay:.3f} + "
            f"{CROSS_REGION_C4_PICKUP_DELTA_PP:.2f}? {c4_a}"
        )
        if "us_national" in pickup_share_by_region:
            us = pickup_share_by_region["us_national"]
            c4_b = bay <= us <= dfw or bay >= us >= dfw
            c4_notes.append(
                f"us_national pickup={us:.3f} between bay & DFW? {c4_b}"
            )
            c4_verdict = "PASS" if (c4_a and c4_b) else "FAIL"
        else:
            c4_verdict = "PASS" if c4_a else "FAIL"
    else:
        c4_notes.append("missing dfw and/or bay_area residential cache")
    report["C4_sales_mix"] = {
        "verdict": c4_verdict,
        "rows": c4_rows,
        "notes": c4_notes,
    }

    # ---- C5: winter f_T applied for cold regions -----------------------
    c5_rows: list[dict[str, Any]] = []
    c5_pass = True

    # Bay-area residential is the baseline (warm region).
    bay_res = _cache_dir(repo_root, "bay_area", "residential")
    baseline_wh_per_mi: float | None = None
    if (bay_res / "mobility.parquet").is_file():
        try:
            base_mob = pd.read_parquet(
                bay_res / "mobility.parquet",
                columns=["start_ts", "miles", "kwh_consumed"],
            )
            june_mask = base_mob["start_ts"].dt.month == 6
            if int(june_mask.sum()) > 0:
                baseline_wh_per_mi = float(
                    1000.0
                    * base_mob.loc[june_mask, "kwh_consumed"].sum()
                    / max(base_mob.loc[june_mask, "miles"].sum(), 1e-9)
                )
        except (OSError, pa.ArrowInvalid, ValueError) as exc:
            c5_rows.append({"baseline_error": str(exc)})

    for region_name in COLD_REGIONS:
        d = _cache_dir(repo_root, region_name, "residential")
        if not (d / "mobility.parquet").is_file():
            c5_rows.append({
                "region": region_name, "pass": None,
                "reason": "no residential cache for this cold region",
            })
            continue
        try:
            mob = pd.read_parquet(
                d / "mobility.parquet",
                columns=["start_ts", "miles", "kwh_consumed"],
            )
            winter_mask = mob["start_ts"].dt.month.isin([12, 1, 2, 3])
            june_mask = mob["start_ts"].dt.month == 6
            winter_wh_per_mi = (
                float(
                    1000.0 * mob.loc[winter_mask, "kwh_consumed"].sum()
                    / max(mob.loc[winter_mask, "miles"].sum(), 1e-9)
                ) if int(winter_mask.sum()) > 0 else float("nan")
            )
            june_wh_per_mi = (
                float(
                    1000.0 * mob.loc[june_mask, "kwh_consumed"].sum()
                    / max(mob.loc[june_mask, "miles"].sum(), 1e-9)
                ) if int(june_mask.sum()) > 0 else float("nan")
            )
            # Ratio winter / june within the same region.
            ratio_same_region = (
                winter_wh_per_mi / june_wh_per_mi
                if np.isfinite(june_wh_per_mi) and june_wh_per_mi > 0
                else float("nan")
            )
            # Ratio vs bay_area baseline (the literal C5 wording in plan §9.2).
            ratio_vs_baseline = (
                winter_wh_per_mi / baseline_wh_per_mi
                if (baseline_wh_per_mi is not None
                    and baseline_wh_per_mi > 0)
                else float("nan")
            )
            passed = (
                np.isfinite(ratio_same_region)
                and ratio_same_region >= CROSS_REGION_C5_MIN_RATIO
            )
            c5_rows.append({
                "region": region_name,
                "winter_wh_per_mi": winter_wh_per_mi,
                "june_wh_per_mi": june_wh_per_mi,
                "baseline_bay_area_june_wh_per_mi": baseline_wh_per_mi,
                "ratio_winter_over_june": ratio_same_region,
                "ratio_winter_over_baseline": ratio_vs_baseline,
                "threshold": CROSS_REGION_C5_MIN_RATIO,
                "pass": bool(passed),
            })
            if not passed:
                c5_pass = False
        except (OSError, pa.ArrowInvalid, ValueError) as exc:
            c5_rows.append({
                "region": region_name, "pass": False, "error": str(exc),
            })
            c5_pass = False

    # Pass C5 only if every cold region present passed, AND at least one
    # cold cache was inspected.
    inspected_cold = [r for r in c5_rows if r.get("pass") is True
                      or r.get("pass") is False]
    if not inspected_cold:
        c5_verdict = "SKIP"
    else:
        c5_verdict = "PASS" if c5_pass else "FAIL"
    report["C5_winter_f_T"] = {
        "verdict": c5_verdict, "rows": c5_rows,
        "min_ratio_threshold": CROSS_REGION_C5_MIN_RATIO,
    }

    # ---- Aggregate verdict --------------------------------------------
    verdicts = {
        k: v["verdict"] for k, v in report.items()
        if isinstance(v, dict) and "verdict" in v
    }
    report["aggregate_verdicts"] = verdicts
    report["overall_status"] = (
        "PASS" if all(v == "PASS" for v in verdicts.values())
        else ("MIXED" if any(v == "PASS" for v in verdicts.values())
              else "FAIL")
    )

    # ---- Write JSON + markdown ----------------------------------------
    out_dir = repo_root / CROSS_REGION_OUT_DIR_REL
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "cross_region_report.json"
    md_path = out_dir / "cross_region_report.md"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(_coerce_for_json(report), f, indent=2)
    md_path.write_text(_render_cross_region_md(report), encoding="utf-8")

    return report


def _render_cross_region_md(report: dict[str, Any]) -> str:
    """Render the cross-region report as Markdown.  ASCII math only."""
    lines: list[str] = []
    lines.append("# Cross-region consistency report (C1-C5)")
    lines.append("")
    lines.append(f"- methodology_version: {report.get('methodology_version')}")
    lines.append(f"- run_timestamp_utc: {report.get('run_timestamp_utc')}")
    lines.append(f"- n_caches_inspected: {report.get('n_caches')}")
    lines.append(f"- overall_status: **{report.get('overall_status')}**")
    lines.append("")
    lines.append("## Caches inspected")
    lines.append("")
    for c in report.get("caches_inspected", []):
        lines.append(f"- {c['region']} / {c['profile_type']}")
    lines.append("")

    # C1
    c1 = report.get("C1_fleet_size", {})
    lines.append(f"## C1 — Fleet size: {c1.get('verdict', '?')}")
    lines.append("")
    lines.append("| region | profile | n_rows | expected | pass |")
    lines.append("|---|---|---|---|---|")
    for r in c1.get("rows", []):
        lines.append(
            f"| {r.get('region')} | {r.get('profile_type')} | "
            f"{r.get('n_rows')} | {r.get('expected')} | {r.get('pass')} |"
        )
    lines.append("")

    # C2
    c2 = report.get("C2_utc_timestamps", {})
    lines.append(f"## C2 — UTC timestamps: {c2.get('verdict', '?')}")
    lines.append("")
    lines.append(
        "| region | profile | file | ts_cols | non_utc_cols | pass |"
    )
    lines.append("|---|---|---|---|---|---|")
    for r in c2.get("rows", []):
        lines.append(
            f"| {r.get('region')} | {r.get('profile_type')} | "
            f"{r.get('file')} | {','.join(r.get('ts_cols', []))} | "
            f"{','.join(r.get('non_utc_cols', []))} | {r.get('pass')} |"
        )
    lines.append("")

    # C3
    c3 = report.get("C3_schema_consistency", {})
    lines.append(f"## C3 — Schema consistency: {c3.get('verdict', '?')}")
    lines.append("")
    for blk in c3.get("by_profile", []):
        lines.append(
            f"- profile_type=**{blk.get('profile_type')}**: "
            f"n_caches={blk.get('n_caches')}, "
            f"ref_key={blk.get('ref_key')}, pass={blk.get('pass')}"
        )
        if blk.get("diffs"):
            for d in blk["diffs"]:
                lines.append(
                    f"    - mismatch: ref={d['ref']} vs this={d['this']}; "
                    f"missing={d.get('missing_cols')}; "
                    f"extra={d.get('extra_cols')}; "
                    f"dtype_mismatch={d.get('dtype_mismatch')}"
                )
        if blk.get("note"):
            lines.append(f"    - note: {blk['note']}")
    lines.append("")

    # C4
    c4 = report.get("C4_sales_mix", {})
    lines.append(f"## C4 — Sales mix: {c4.get('verdict', '?')}")
    lines.append("")
    lines.append("| region | pickup_share |")
    lines.append("|---|---|")
    for r in c4.get("rows", []):
        ps = r.get("pickup_share")
        ps_str = f"{ps:.3f}" if isinstance(ps, (int, float)) else str(ps)
        lines.append(f"| {r.get('region')} | {ps_str} |")
    lines.append("")
    for note in c4.get("notes", []):
        lines.append(f"- {note}")
    lines.append("")

    # C5
    c5 = report.get("C5_winter_f_T", {})
    lines.append(f"## C5 — Winter f_T applied: {c5.get('verdict', '?')}")
    lines.append("")
    lines.append(
        f"Required winter / June Wh/mi ratio ≥ "
        f"{c5.get('min_ratio_threshold')}"
    )
    lines.append("")
    lines.append(
        "| region | winter Wh/mi | June Wh/mi | ratio (winter/June) | "
        "ratio vs bay_area June | pass |"
    )
    lines.append("|---|---|---|---|---|---|")
    for r in c5.get("rows", []):
        w = r.get("winter_wh_per_mi")
        j = r.get("june_wh_per_mi")
        rj = r.get("ratio_winter_over_june")
        rb = r.get("ratio_winter_over_baseline")

        def _fmt(v: Any) -> str:
            if isinstance(v, (int, float)) and v is not None \
                    and (isinstance(v, int) or np.isfinite(v)):
                return f"{v:.2f}"
            return str(v)

        lines.append(
            f"| {r.get('region')} | {_fmt(w)} | {_fmt(j)} | "
            f"{_fmt(rj)} | {_fmt(rb)} | {r.get('pass')} |"
        )

    lines.append("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pev_synth.cache_regen",
        description=(
            "v2.0 cache regeneration orchestrator + cross-region C1-C5 audit."
        ),
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    one = sub.add_parser(
        "one", help="Regenerate a single (region, profile_type) cache.",
    )
    one.add_argument("--region", required=True, choices=list_regions())
    one.add_argument(
        "--profile-type", required=True, choices=list_profile_types(),
    )
    one.add_argument("--n", type=int, default=None)
    one.add_argument("--seed", type=int, default=MASTER_SEED)
    one.add_argument(
        "--replicate-id", type=int, default=0,
        help="RFC-022.r replicate index when --r-total > 1 (default 0).",
    )
    one.add_argument(
        "--r-total", type=int, default=1,
        help="RFC-022.r total replicate count (1 keeps the flat layout).",
    )

    batch = sub.add_parser(
        "batch", help="Regenerate the 15-cache priority list.",
    )
    batch.add_argument("--seed", type=int, default=MASTER_SEED)
    batch.add_argument(
        "--limit", type=int, default=None,
        help="Only process the first N pairs of the priority list.",
    )
    batch.add_argument(
        "--time-budget-min", type=float, default=None,
        help="Stop launching new caches after this many minutes.",
    )
    batch.add_argument(
        "--r-total", type=int, default=1,
        help="RFC-022.r total replicate count per pair (1 keeps flat layout).",
    )

    audit = sub.add_parser(
        "audit", help="Run cross-region C1-C5 checks across existing caches.",
    )
    audit.add_argument(
        "--repo-root", default=None,
        help="Override repo root (defaults to file-relative discovery).",
    )

    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.cmd == "one":
        res = regenerate_cache(
            args.region, args.profile_type, n=args.n, seed=args.seed,
            replicate_id=args.replicate_id, r_total=args.r_total,
        )
        print(json.dumps(_coerce_for_json(res), indent=2))
        return 0 if res.get("status") == "PASS" else 1

    if args.cmd == "batch":
        pairs = list(PRIORITY_ORDER)
        if args.limit is not None:
            pairs = pairs[: args.limit]
        budget = (
            args.time_budget_min * 60.0
            if args.time_budget_min is not None else None
        )
        rep = regenerate_batch(
            pairs, seed=args.seed, time_budget_s=budget,
            r_total=args.r_total,
            log_path=_repo_root() / "data" / "pev" / "processed"
            / "_cross_region" / "regen_progress.json",
        )
        print(json.dumps(_coerce_for_json(rep), indent=2))
        any_fail = any(r.get("status") == "FAIL" for r in rep["results"])
        return 1 if any_fail else 0

    if args.cmd == "audit":
        repo_root = Path(args.repo_root) if args.repo_root else None
        rep = run_cross_region_checks(repo_root=repo_root)
        print(json.dumps(_coerce_for_json(rep), indent=2))
        return 0 if rep.get("overall_status") == "PASS" else 1

    return 2


if __name__ == "__main__":
    sys.exit(main())
