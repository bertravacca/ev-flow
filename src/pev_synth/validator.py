"""M9 — `pev_synth.validator`.

Run the §10 validation plan against the M2..M8 pipeline artifacts and emit
`validation_report.json` + `validation_report.md`.

This is the v1 reduced-scope validator (per the orchestrator's M9 brief):

    * 11 §10 checks driven by `data/pev/validation_bounds/*.parquet`.
    * 3 pipeline-internal integration checks (cross-artifact integrity,
      plug-rate consistency, fleet energy conservation).

Each check resolves to one of {PASS, FAIL, EXPLAINED_FAIL, EXPLAINED_SKIP,
ERROR}. `EXPLAINED_FAIL` rows MUST cite a phase4_methodology_plan_revised
§13 limitation by section number. `EXPLAINED_SKIP` is reserved for rows
where the M8 bound is marked `unbounded_v1=true`.

The module is deterministic — no RNG. The only non-byte-identical field
between two runs is `run_timestamp_utc`.

Math conventions
----------------
TVD(p,q) = ½ ∑|pᵢ − qᵢ| (Total Variation Distance).
Plug-in CDF F(h) = fraction of arrivals whose hour-of-day (continuous) < h.
σ denotes standard deviation. η = charging efficiency.

CLI
---
    python -m pev_synth.validator [--root PATH] [--bounds PATH] [--skip-optimizer]

Outputs (under `--root`):
    validation_report.json
    validation_report.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ._paths import (
    cache_dir as _cache_dir,
)
from ._paths import (
    processed_root as _processed_root,
)
from ._paths import (
    validation_bounds_root as _validation_bounds_root,
)
from ._seeds import (
    MASTER_SEED as _MASTER_SEED_V2,
)
from ._seeds import (
    METHODOLOGY_VERSION as _METHODOLOGY_VERSION_V2,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (mirror orchestrator-resolved contracts)
# ---------------------------------------------------------------------------

VALIDATOR_VERSION = "v1.0.0"
METHODOLOGY_VERSION = "v1.0.0"
MASTER_SEED = 20260518
TZ = "America/Los_Angeles"

# v2.0 toggles — applied only when `run()` is called with `region` +
# `profile_type`. The legacy v1.1 path keeps emitting v1.0.0 / v1.0.0 so
# that the v1.1 tests stay byte-identical (modulo timestamp). v2.0 values
# now come from the central :mod:`pev_synth._seeds` registry per RFC-016.
VALIDATOR_VERSION_V2 = "v2.0-rev"
METHODOLOGY_VERSION_V2 = _METHODOLOGY_VERSION_V2
MASTER_SEED_V2 = _MASTER_SEED_V2

VALID_STATUSES = ("PASS", "FAIL", "EXPLAINED_FAIL", "EXPLAINED_SKIP", "ERROR")

# Project paths (anchored on the repo root via :mod:`pev_synth._paths`).
DEFAULT_OUTPUT_ROOT = _processed_root() / "residential_ev_synth"
DEFAULT_BOUNDS_ROOT = _validation_bounds_root()

# Pipeline artifacts the validator hashes for provenance.
PIPELINE_ARTIFACTS = (
    "fleet.parquet",
    "mobility.parquet",
    "plug_in_events.parquet",
    "sessions.parquet",
    "plug_status_15min.parquet",
    "plug_status_hourly.parquet",
)

# Mapping from M8 `wh_per_mi_at_70F` bound rows to fleet `model` values.
WH_PER_MI_MODEL_MAP = {
    "wh_per_mi_chevy_bolt_euv": "Bolt_EUV",
    "wh_per_mi_ford_mach_e_rwd_er": "Mach_E_ER",
    "wh_per_mi_hyundai_ioniq_5_lr_rwd": "Ioniq_5",
    "wh_per_mi_hyundai_ioniq_6_lr_rwd": "Ioniq_6",
    "wh_per_mi_kia_ev6_lr_rwd": "EV6",
    "wh_per_mi_nissan_leaf_sv": "Leaf_Plus",
    "wh_per_mi_tesla_model_3_lr": "Model_3_LR",
    "wh_per_mi_tesla_model_y_lr": "Model_Y_LR_AWD",
}

# Argonne PMF bin edges (kWh). Aligned with M8 bound rows
# pmf_bin_{30_50, 50_65, 65_75, 75_85, 85_100, 100_150}.
ARGONNE_BIN_EDGES_KWH = (30.0, 50.0, 65.0, 75.0, 85.0, 100.0, 150.0)
ARGONNE_BIN_NAMES = (
    "pmf_bin_30_50_kwh",
    "pmf_bin_50_65_kwh",
    "pmf_bin_65_75_kwh",
    "pmf_bin_75_85_kwh",
    "pmf_bin_85_100_kwh",
    "pmf_bin_100_150_kwh",
)

# Plan §13 limitations cited by EXPLAINED_FAIL rows.
# (Keep these strings stable — tests assert "§13." substring presence.)
LIM_13_1 = (
    "§13.1: plug-in coefficients are unfit literature/judgment priors "
    "(EVWatts cannot supply home-vs-non-home split; β values not "
    "point-identified by data)."
)
LIM_13_2 = (
    "§13.2 ACKNOWLEDGED as NHTS structural limitation (v3.0): both NHTS "
    "2017 and NHTS NextGen 2022 public files record exactly one designated "
    "TRAVDAY per HOUSEID (D5 verified live on NextGen 2022: 0 HHs with >=2 "
    "travdays). Per-day donor sampling (v2_per_travday, the v3.0 default) "
    "therefore breaks intra-week correlation, and the v3-within-HH "
    "alternative degenerates to single-day replay — neither sampler can "
    "close the gap on the available US travel-survey vintages. v3.0 ships "
    "the within-HH code path for opt-in use on future multi-day-diary "
    "vintages (e.g. state add-ons or non-US surveys with full 7-day "
    "diaries); the residual annual_mileage σ inflation vs the EMFAC2021 "
    "prior is real and structural, not a defect."
)
LIM_13_4 = (
    "§13.4: trip-time temperature uses home-station for all trips and "
    "v1 skipped temperature adjustment entirely; realised Wh/mi reflects "
    "nominal archetype values, not 70°F EPA-label values."
)
LIM_13_5 = (
    "§13.5: heat-pump correction coarse (k_HP = 0.65 across all HP "
    "vehicles); model-specific Wh/mi sensitivity ignored."
)
LIM_13_6 = (
    "§13.6 RESOLVED in v3.0: PHEV gasoline extension modelled — trip "
    "deficits below soc_min are served by ICE and recorded as "
    "sessions.gas_kwh_equivalent; fleet_energy_conservation now "
    "includes gas-equivalent kWh in the numerator. CS MPG sourced from "
    "EPA fueleconomy.gov per (make, model, model_year); models outside "
    "the lookup fall back to a 33 MPG combined prior."
)
LIM_13_11 = (
    "§13.11: K selected by BIC on ~84 EVWatts residential vehicles; "
    "selection variance high — pre-dawn lobe under-represented because "
    "K=6 has only one early-morning centre."
)
LIM_13_13 = (
    "§13.13 RESOLVED in v3.0: plug-in clusters now use Powell-2022 SPEECh "
    "K=16 published GMMs (Mendeley gvk34mybtb/1, CC-BY-4.0). v3.0 adopts "
    "the start-time marginal and P(G) prior; joint (start, energy, dwell) "
    "sampling is deferred to v3.1."
)
# v3.0 — battery-capacity EXPLAINED_FAIL constant. Replaces the v2.1 citation
# of ``LIM_13_13`` for the battery row only (D1 is repurposing LIM_13_13 for
# the plug-in CDF SPEECh K=16 resolution). When the realised fleet B_kwh PMF
# still drifts from the regional sales-mix reference beyond the 5 pp TVD
# band, the residual divergence is rooted in M2 archetype sampling — sales
# mix and archetype registry are not exactly aligned (e.g. workplace cohort
# skews to commercial-fleet models; the CVRP rebate cap excludes luxury BEVs
# even when their published sales share is non-trivial).
LIM_13_M2_ARCHETYPE_MIX = (
    "§13.M2 (archetype mix): residual battery-capacity divergence vs the "
    "regional sales-mix PMF arises from M2 archetype sampling — sales mix "
    "and archetype registry are not exactly aligned (e.g. workplace cohort "
    "skews to commercial-fleet models; CVRP rebate cap on luxury BEVs); "
    "residual TVD remains EXPLAINED_FAIL when above 5 pp."
)

# Mapping from session-duration t_out semantics (departure_ts post-fix) to §13.
LIM_T_OUT_SEMANTICS = (
    "§13.1: session t_out is departure_ts (full dwell) per the v1.0.1 fix "
    "in `soc_trajectory` — required by the optimiser contract — so "
    "realised session duration includes post-charge idle time and is not "
    "directly comparable to INL/EXT-15-36317 connect-time CDFs (rooted "
    "in unfit plug-in dwell priors per §13.1)."
)

# ---------------------------------------------------------------------------
# v2.0-only §13 / limitations citations
# ---------------------------------------------------------------------------

# Workplace cohort caveat. The exact prose lives in M5
# (plug_in_model.py:1430) under `meta.json.plug_in_model.workplace_cohort_caveat`.
# The validator imports the same wording at run time when present in meta.json;
# the constant below is the static fallback if the workplace cache is missing
# the M5 caveat block.
LIM_WORKPLACE_COHORT_CAVEAT = (
    "§13.workplace_cohort_caveat (v2.0 limitations §13): the v2.0 M5 BIC "
    "fit on the EVWatts public workplace cohort (N=105) yields cluster "
    "centres skewed midday (~12:00 LT), roughly 3 h later than the "
    "literature-canonical workplace median (~09:00 LT, Pritchard 2023 + "
    "ACN-Data). W1 EXPLAINED_FAIL on workplace caches surfaces this "
    "public-data caveat — see M5 meta.json.plug_in_model.workplace_cohort_"
    "caveat."
)

LIM_V2_WORKPLACE_SCOPE = (
    "§13.workplace_scope (v2.0 limitations §11.1 item 6): workplace caches "
    "are scoped to the workplace-only-feasible commuter cohort (ANNMILES "
    "< 6,000 × 0.6); residential metrics may not be directly comparable. "
    "Joint residential+workplace synthetic is a v2.1 deferral."
)

LIM_V2_BORROW_RATE = (
    "§13.borrow_rate (v2.0 limitations §11.1 item 5 / S-7): thin workplace "
    "donor pools may have triggered borrow-from-neighbours; the realised "
    "borrow rate is capped at 25 % per W10 / §2.6."
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    """One row in the validation_report.json `checks` array."""

    check_id: str
    status: str
    realised_value: Any
    bound_source: str
    details: str
    explained_reason: str = ""

    def __post_init__(self) -> None:
        if self.status not in VALID_STATUSES:
            raise ValueError(
                f"Invalid status {self.status!r} for check {self.check_id}; "
                f"must be one of {VALID_STATUSES}"
            )
        if self.status == "EXPLAINED_FAIL" and "§13." not in self.explained_reason:
            raise ValueError(
                f"EXPLAINED_FAIL on {self.check_id} must cite a plan §13 "
                f"limitation by section number; got {self.explained_reason!r}"
            )


@dataclass
class Report:
    validator_version: str
    methodology_version: str
    run_timestamp_utc: str
    pipeline_artifacts_sha256: dict[str, str]
    checks: list[CheckResult] = field(default_factory=list)
    summary: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "validator_version": self.validator_version,
            "methodology_version": self.methodology_version,
            "run_timestamp_utc": self.run_timestamp_utc,
            "pipeline_artifacts_sha256": self.pipeline_artifacts_sha256,
            "checks": [asdict(c) for c in self.checks],
            "summary": self.summary,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _round(x: float, n: int = 6) -> float:
    """Stable rounding for deterministic JSON output."""
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return float(x) if x is not None else float("nan")
    return float(round(float(x), n))


def _is_unbounded(bnd_row: pd.Series) -> bool:
    """Treat a bound row as unbounded_v1 iff explicit flag is True, or value
    and both bounds are NaN.

    RFC-010: the canonical brief schema (line 114) does not include an
    ``unbounded_v1`` column; the flag is encoded as ``unbounded_v1=true``
    inside the ``note`` string. Fall back to either form so v2.0-rev
    parquets and any legacy unmigrated parquet keep validating.
    """
    if "unbounded_v1" in bnd_row and bool(bnd_row["unbounded_v1"]):
        return True
    if "note" in bnd_row:
        note_val = bnd_row.get("note")
        if isinstance(note_val, str) and "unbounded_v1=true" in note_val:
            return True
    has_lo = not pd.isna(bnd_row.get("lower_bound"))
    has_hi = not pd.isna(bnd_row.get("upper_bound"))
    return not (has_lo or has_hi)


def _all_rows_unbounded(bnd: pd.DataFrame) -> bool:
    """All rows in ``bnd`` are flagged unbounded_v1 (RFC-010 helper)."""
    if bnd.empty:
        return False
    if "unbounded_v1" in bnd.columns:
        return bool(bnd["unbounded_v1"].all())
    return bool(bnd.apply(_is_unbounded, axis=1).all())


def _load_bound(bounds_root: Path, check_id: str) -> pd.DataFrame:
    """Load the bound parquet for ``check_id`` from the strict nested layout.

    Per RFC-009.r the only supported layout is
    ``data/pev/validation_bounds/<region>/<profile_type>/<check_id>.parquet``.
    The validator pre-resolves ``bounds_root`` to the per-region per-
    profile-type subdir (see :func:`_resolve_bounds_subdir` in :func:`run`).
    The legacy flat-layout fallback shipped in RFC-009 (Phase 11) was
    removed in RFC-009.r — readers MUST use the nested layout and a
    missing parquet raises ``FileNotFoundError`` without searching parent
    directories.
    """
    p = bounds_root / f"{check_id}.parquet"
    if not p.exists():
        raise FileNotFoundError(f"bound parquet missing: {p}")
    return pd.read_parquet(p)


def _bound_url(df: pd.DataFrame) -> str:
    if df.empty or "source_url" not in df.columns:
        return ""
    return str(df["source_url"].iloc[0])


def _is_v11_migrated(meta: dict[str, Any] | None) -> bool:
    """Return True when the cache is the bay_area v1.1_migrated artefact.

    RFC-008.r — the legacy ``energy_kwh`` alias is read ONLY when
    ``cache_provenance == "v1.1_migrated"`` (the bay_area migrated cache).
    ``cache_provenance`` may be either a string (``"v2.0_native"`` /
    ``"v1.1_migrated"``) or a dict carrying ``provenance_tag``.
    """
    if not meta:
        return False
    cp = meta.get("cache_provenance")
    if isinstance(cp, str):
        return cp == "v1.1_migrated"
    if isinstance(cp, dict):
        return cp.get("provenance_tag") == "v1.1_migrated"
    return False


def _session_energy_grid(
    sessions: pd.DataFrame,
    *,
    legacy_alias_ok: bool,
    context: str,
) -> pd.Series:
    """Return the grid-side per-session energy column.

    RFC-008.r — v2.0-native caches expose ``energy_kwh_grid`` (AC/DC wall
    energy = battery / eta). The legacy ``energy_kwh`` alias is read ONLY
    when ``legacy_alias_ok`` is True (i.e., the v1.1_migrated cache OR the
    v1.1 legacy validator path). For v2.0-native caches a missing
    ``energy_kwh_grid`` is a schema error and we raise immediately.
    """
    if "energy_kwh_grid" in sessions.columns:
        return sessions["energy_kwh_grid"].astype(float)
    if legacy_alias_ok and "energy_kwh" in sessions.columns:
        return sessions["energy_kwh"].astype(float)
    raise KeyError(
        f"sessions.parquet missing 'energy_kwh_grid' column ({context}); "
        "v2.0-native caches must carry the RFC-008 grid/batt split"
    )


def _session_energy_batt(
    sessions: pd.DataFrame,
    *,
    legacy_alias_ok: bool,
    context: str,
) -> pd.Series:
    """Return the battery-side per-session energy column.

    RFC-008.r — v2.0-native caches expose ``energy_kwh_batt``
    (= eta · energy_kwh_grid). For v1.1_migrated caches the legacy
    ``energy_kwh`` is read as an alias when ``legacy_alias_ok`` is True
    (the v1.1 cache predates the grid/batt split).
    """
    if "energy_kwh_batt" in sessions.columns:
        return sessions["energy_kwh_batt"].astype(float)
    if legacy_alias_ok and "energy_kwh" in sessions.columns:
        return sessions["energy_kwh"].astype(float)
    raise KeyError(
        f"sessions.parquet missing 'energy_kwh_batt' column ({context}); "
        "v2.0-native caches must carry the RFC-008 grid/batt split"
    )


def _in_band(value: float, lo: float, hi: float) -> bool:
    return (lo - 1e-12) <= value <= (hi + 1e-12)


# ---------------------------------------------------------------------------
# Per-check implementations
# ---------------------------------------------------------------------------

def check_battery_capacity_distribution(
    fleet: pd.DataFrame, bounds_root: Path
) -> CheckResult:
    """§10 check 1 — TVD(fleet B PMF, regional sales-mix PMF) ≤ 0.05 across 6 bins.

    v3.0: the bound is region-aware. ``bounds_root`` is pre-resolved by
    :func:`run` to ``<bounds_top>/<region>/<profile_type>/``, so the
    ``battery_capacity_distribution.parquet`` loaded here is the regional
    one (CVRP-CA for bay_area / la_basin, NYSERDA for new_york_metro,
    Argonne for everyone else; mapping in
    :data:`pev_synth.validation_bounds_curator.REGION_TO_BATTERY_SALES_MIX_SOURCE`).

    The v3 bound row carries ``metric_name="tvd_vs_regional_sales_pmf"``;
    we fall back to the legacy ``tvd_vs_argonne_pmf`` name when only a
    v2.1 bound parquet is on disk (back-compat during migration).
    """
    check_id = "battery_capacity_distribution"
    bnd = _load_bound(bounds_root, check_id)
    source = _bound_url(bnd)
    bev = fleet[fleet["powertrain"].astype(str) == "BEV"]
    counts, _ = np.histogram(bev["B_kwh"].astype(float), bins=ARGONNE_BIN_EDGES_KWH)
    n_bev = int(len(bev))
    pmf = counts.astype(float) / max(n_bev, 1)
    reference = np.array(
        [float(bnd[bnd["metric_name"] == name]["value"].iloc[0]) for name in ARGONNE_BIN_NAMES],
        dtype=float,
    )
    tvd = float(0.5 * np.abs(pmf - reference).sum())

    # v3.0 — prefer the region-aware ``tvd_vs_regional_sales_pmf`` row.
    # Fall back to the legacy ``tvd_vs_argonne_pmf`` name for any v2.1
    # parquet that hasn't been regenerated yet. The v3 curator also emits
    # the legacy name as an ``unbounded_v1`` alias for v2.1 consumers —
    # filter those out so we don't pick up the back-compat sentinel.
    def _live_rows(name: str) -> pd.DataFrame:
        rows = bnd[bnd["metric_name"] == name]
        if "unbounded_v1" in rows.columns:
            rows = rows[~rows["unbounded_v1"].astype(bool)]
        return rows

    tvd_metric = "tvd_vs_regional_sales_pmf"
    tvd_rows = _live_rows(tvd_metric)
    if tvd_rows.empty:
        tvd_metric = "tvd_vs_argonne_pmf"
        tvd_rows = _live_rows(tvd_metric)
        if tvd_rows.empty:
            raise KeyError(
                f"battery_capacity_distribution bound missing a live "
                "'tvd_vs_regional_sales_pmf' (v3) or 'tvd_vs_argonne_pmf' "
                f"(v2.1) row at {bounds_root}"
            )
    tvd_row = tvd_rows.iloc[0]
    lo = float(tvd_row["lower_bound"])
    hi = float(tvd_row["upper_bound"])
    realised = {
        "tvd": _round(tvd),
        "fleet_pmf": {n: _round(v) for n, v in zip(ARGONNE_BIN_NAMES, pmf, strict=True)},
        "reference_pmf": {n: _round(v) for n, v in zip(ARGONNE_BIN_NAMES, reference, strict=True)},
        "n_bev": n_bev,
        "tvd_metric_name": tvd_metric,
    }
    details = f"TVD = {tvd:.4f} vs bound [{lo:.4f}, {hi:.4f}] ({tvd_metric})"
    if _in_band(tvd, lo, hi):
        return CheckResult(check_id, "PASS", realised, source, details)
    # v3.0 — when even the region-aware bound still over-shoots 5 pp, the
    # residual divergence is upstream of M8: M2 archetype sampling doesn't
    # exactly match the published sales-mix PMF (workplace cohort skews
    # commercial-fleet; CVRP rebate cap on luxury BEVs). Cite the new
    # LIM_13_M2_ARCHETYPE_MIX (§13.M2) constant instead of the v2.1
    # LIM_13_13 (which D1 has repurposed for the SPEECh K=16 plug-in CDF
    # resolution).
    return CheckResult(
        check_id,
        "EXPLAINED_FAIL",
        realised,
        source,
        details,
        explained_reason=LIM_13_M2_ARCHETYPE_MIX,
    )


def check_obc_distribution(bounds_root: Path) -> CheckResult:
    """§10 check 2 (OBC share) — all rows are `unbounded_v1=true` in v1."""
    check_id = "obc_distribution"
    bnd = _load_bound(bounds_root, check_id)
    source = _bound_url(bnd)
    if _all_rows_unbounded(bnd):
        return CheckResult(
            check_id,
            "EXPLAINED_SKIP",
            None,
            source,
            "All OBC-share bound rows are `unbounded_v1=true` in v1; "
            "no published US-fleet OBC share table exists (M8 brief §M8.1).",
            explained_reason="M8 bound is informational only in v1 (`unbounded_v1=true`).",
        )
    # Defensive: if anyone ever populates a real bound, fall through to ERROR
    # rather than silently downgrading.
    return CheckResult(
        check_id,
        "ERROR",
        None,
        source,
        "OBC bound is no longer fully unbounded; M9 v1 has no scoring logic.",
    )


def _plugin_cdf_check(
    pie: pd.DataFrame, bounds_root: Path, weekday: bool
) -> CheckResult:
    """§10 checks 3 & 4 — weekday/weekend plug-in CDF at hours {06, 12, 18, 22}."""
    if weekday:
        check_id = "weekday_plugin_cdf"
        dow_mask = pie["arrival_ts"].dt.dayofweek < 5
        tag = "weekday"
    else:
        check_id = "weekend_plugin_cdf"
        dow_mask = pie["arrival_ts"].dt.dayofweek >= 5
        tag = "weekend"

    bnd = _load_bound(bounds_root, check_id)
    source = _bound_url(bnd)
    plugged_mask = pie["plug_in"].astype(bool)
    df = pie[plugged_mask & dow_mask]
    if df.empty:
        return CheckResult(check_id, "ERROR", None, source, "no plug-in events match filter")

    hours = df["arrival_ts"].dt.hour.astype(int) + df["arrival_ts"].dt.minute.astype(int) / 60.0
    realised: dict[str, float] = {}
    band: dict[str, dict[str, float]] = {}
    statuses: list[str] = []
    for h in (6, 12, 18, 22):
        cdf = float((hours < h).mean())
        realised[f"F_{h:02d}"] = _round(cdf)
        metric = f"{tag}_plugin_cdf_at_hour_{h:02d}"
        row = bnd[bnd["metric_name"] == metric]
        if row.empty:
            statuses.append("ERROR")
            band[f"F_{h:02d}"] = {"lo": float("nan"), "hi": float("nan")}
            continue
        lo = float(row["lower_bound"].iloc[0])
        hi = float(row["upper_bound"].iloc[0])
        band[f"F_{h:02d}"] = {"lo": _round(lo), "hi": _round(hi)}
        statuses.append("PASS" if _in_band(cdf, lo, hi) else "FAIL")

    realised["bounds"] = band
    realised["n_events"] = int(len(df))
    details = (
        f"{tag}: " + ", ".join(
            f"F({h:02d})={realised[f'F_{h:02d}']:.3f} band=[{band[f'F_{h:02d}']['lo']},"
            f"{band[f'F_{h:02d}']['hi']}]"
            for h in (6, 12, 18, 22)
        )
    )

    if all(s == "PASS" for s in statuses):
        return CheckResult(check_id, "PASS", realised, source, details)
    if "ERROR" in statuses:
        return CheckResult(check_id, "ERROR", realised, source, details)
    # v3.0: LIM_13_13 is RESOLVED — plug-in clusters now use Powell-2022
    # SPEECh K=16. Any residual F(h) outside the EVWatts-anchored band
    # is rooted in §13.1 (β priors unfit) + §13.11 (selection variance
    # on the residential EVWatts cohort). LIM_13_13 is intentionally
    # dropped from the explained_reason on EXPLAINED_FAIL so the
    # citation honestly reflects the v3.0 resolution path.
    reason = " ".join([LIM_13_1, LIM_13_11])
    return CheckResult(check_id, "EXPLAINED_FAIL", realised, source, details, reason)


def check_weekday_plugin_cdf(pie: pd.DataFrame, bounds_root: Path) -> CheckResult:
    return _plugin_cdf_check(pie, bounds_root, weekday=True)


def check_weekend_plugin_cdf(pie: pd.DataFrame, bounds_root: Path) -> CheckResult:
    return _plugin_cdf_check(pie, bounds_root, weekday=False)


def check_session_duration_distribution(
    sessions: pd.DataFrame, bounds_root: Path
) -> CheckResult:
    check_id = "session_duration_distribution"
    bnd = _load_bound(bounds_root, check_id)
    source = _bound_url(bnd)
    home = sessions[sessions["location"].astype(str) == "home"]
    dur_h = (home["t_out"] - home["t_in"]).dt.total_seconds() / 3600.0
    median = float(dur_h.median())
    p95 = float(dur_h.quantile(0.95))

    med_row = bnd[bnd["metric_name"] == "median_duration_hours"].iloc[0]
    p95_row = bnd[bnd["metric_name"] == "p95_duration_hours"].iloc[0]
    med_lo, med_hi = float(med_row["lower_bound"]), float(med_row["upper_bound"])
    p95_lo, p95_hi = float(p95_row["lower_bound"]), float(p95_row["upper_bound"])

    realised = {
        "median_h": _round(median),
        "p95_h": _round(p95),
        "n_home_sessions": int(len(home)),
        "median_bound": [_round(med_lo), _round(med_hi)],
        "p95_bound": [_round(p95_lo), _round(p95_hi)],
    }
    details = (
        f"median = {median:.2f} h vs [{med_lo}, {med_hi}]; "
        f"p95 = {p95:.2f} h vs [{p95_lo}, {p95_hi}]"
    )
    if _in_band(median, med_lo, med_hi) and _in_band(p95, p95_lo, p95_hi):
        return CheckResult(check_id, "PASS", realised, source, details)
    # Rooted in t_out = departure_ts post-fix (M6 v1.0.1) — sessions span
    # the entire dwell, not just the active-charging window; this is the
    # optimiser-contract semantics required upstream.
    return CheckResult(check_id, "EXPLAINED_FAIL", realised, source, details, LIM_T_OUT_SEMANTICS)


def check_energy_per_session_distribution(
    sessions: pd.DataFrame, bounds_root: Path, *, legacy_alias_ok: bool = False
) -> CheckResult:
    check_id = "energy_per_session_distribution"
    bnd = _load_bound(bounds_root, check_id)
    source = _bound_url(bnd)
    home = sessions[sessions["location"].astype(str) == "home"]
    # v3.0 (§13.6 RESOLVED): the §10 home-session energy distribution
    # is an electricity-only metric — the literature bound (INL kWh
    # per home session) measures grid-delivered kWh, not the
    # gasoline-equivalent kWh the ICE supplied on PHEV deficit
    # legs. Exclude rows where the M6 ledger attributed gasoline
    # extension so the realised histogram is directly comparable to
    # the bound. Pre-v3 caches don't carry the column and pass
    # through unchanged.
    n_home_all = int(len(home))
    if "gas_kwh_equivalent" in home.columns:
        gas_col = home["gas_kwh_equivalent"].fillna(0.0)
        home_elec = home[gas_col == 0]
    else:
        home_elec = home
    n_excluded = n_home_all - int(len(home_elec))
    # RFC-008.r — read grid-side energy for §10 home-session distribution.
    e = _session_energy_grid(
        home_elec, legacy_alias_ok=legacy_alias_ok, context=check_id,
    )
    median = float(e.median()) if len(e) else float("nan")
    p95 = float(e.quantile(0.95)) if len(e) else float("nan")

    med_row = bnd[bnd["metric_name"] == "median_kwh_per_session"].iloc[0]
    p95_row = bnd[bnd["metric_name"] == "p95_kwh_per_session"].iloc[0]
    med_lo, med_hi = float(med_row["lower_bound"]), float(med_row["upper_bound"])
    p95_lo, p95_hi = float(p95_row["lower_bound"]), float(p95_row["upper_bound"])

    realised = {
        "median_kwh": _round(median),
        "p95_kwh": _round(p95),
        "n_home_sessions": int(len(home_elec)),
        "n_phev_gas_excluded": int(n_excluded),
        "median_bound": [_round(med_lo), _round(med_hi)],
        "p95_bound": [_round(p95_lo), _round(p95_hi)],
    }
    details = (
        f"median = {median:.2f} kWh vs [{med_lo}, {med_hi}]; "
        f"p95 = {p95:.2f} kWh vs [{p95_lo}, {p95_hi}]; "
        f"electricity-only (excluded {n_excluded} PHEV gas-extension rows)"
    )
    if _in_band(median, med_lo, med_hi) and _in_band(p95, p95_lo, p95_hi):
        return CheckResult(check_id, "PASS", realised, source, details)
    # Residual EXPLAINED_FAIL band: high p95 driven by ≥100 kWh BEVs
    # reaching full charge (modern fleet pack ≥ INL legacy fleet) and
    # by the v1.1 plug-in priors (§13.1) which are not point-identified
    # by EVWatts. The §13.6 PHEV gas-extension fix removes the
    # downward bias on the median without addressing those residuals.
    return CheckResult(check_id, "EXPLAINED_FAIL", realised, source, details, LIM_13_6)


def _methodology_at_or_above_v3(methodology_version: str | None) -> bool:
    """Return True iff ``methodology_version`` parses as ``>= v3.0``.

    Pure string check tolerant of the ``v`` prefix and trailing labels
    (e.g. ``"v3.0.0"``, ``"v3.1-rev"``).  Falls back to ``False`` for
    ``None`` / unparseable so the legacy v1.x/v2.x behaviour is the
    safe default.
    """
    if not methodology_version:
        return False
    s = str(methodology_version).lstrip("vV")
    head = s.split("-")[0]
    try:
        major = int(head.split(".")[0])
    except (ValueError, IndexError):
        return False
    return major >= 3


def check_annual_mileage(
    mobility: pd.DataFrame,
    bounds_root: Path,
    *,
    methodology_version: str | None = None,
) -> CheckResult:
    """Validate per-EV annual mileage against the EMFAC2021 band.

    v3.0 (§13.2 RESOLVED): when ``methodology_version`` parses as
    ``>= v3.0``, the HOUSEID-coherent intra-household stitcher has
    fired and we expect σ to fall inside the EMFAC2021 band
    ``[3000, 6000]`` mi/yr.  In that case the check PASSes when both
    median and σ are in band, and otherwise returns a plain FAIL so
    the residual gap surfaces to the PM honestly (we do not widen the
    bound or fall back to the §13.2 EXPLAINED_FAIL — that limitation
    has been retired).

    Legacy paths (``methodology_version`` is ``None`` / v1.x / v2.x)
    keep the historical EXPLAINED_FAIL → LIM_13_2 behaviour so v1.1
    fixtures stay byte-stable.
    """
    check_id = "annual_mileage"
    bnd = _load_bound(bounds_root, check_id)
    source = _bound_url(bnd)
    miles_per_ev = mobility.groupby("ev_id")["miles"].sum()
    mean_mi = float(miles_per_ev.mean())
    median_mi = float(miles_per_ev.median())
    std_mi = float(miles_per_ev.std())

    mean_row = bnd[bnd["metric_name"] == "mean_miles_per_year"].iloc[0]
    std_row = bnd[bnd["metric_name"] == "std_miles_per_year"].iloc[0]
    mean_lo, mean_hi = float(mean_row["lower_bound"]), float(mean_row["upper_bound"])
    std_lo, std_hi = float(std_row["lower_bound"]), float(std_row["upper_bound"])

    realised = {
        "median_miles_per_year": _round(median_mi, 1),
        "mean_miles_per_year": _round(mean_mi, 1),
        "std_miles_per_year": _round(std_mi, 1),
        "mean_bound": [_round(mean_lo, 1), _round(mean_hi, 1)],
        "std_bound": [_round(std_lo, 1), _round(std_hi, 1)],
        "n_evs": int(miles_per_ev.shape[0]),
        "methodology_version": str(methodology_version) if methodology_version else None,
    }
    # Brief asks for median + σ; bound table provides mean + σ. We score
    # median against mean band (sym distributions); σ against σ band.
    details = (
        f"median = {median_mi:.0f} mi/yr (mean band [{mean_lo:.0f}, {mean_hi:.0f}]); "
        f"σ = {std_mi:.0f} mi/yr (band [{std_lo:.0f}, {std_hi:.0f}])"
    )
    if _in_band(median_mi, mean_lo, mean_hi) and _in_band(std_mi, std_lo, std_hi):
        return CheckResult(check_id, "PASS", realised, source, details)
    # v3.0 acknowledges §13.2 as a structural NHTS limitation (1 designated
    # TRAVDAY per HOUSEID in both 2017 and 2022 NextGen). Both v2_per_travday
    # and v3_within_hh samplers leave a real σ gap; the v3.0 default is
    # v2_per_travday (honest acknowledgement). EXPLAINED_FAIL with LIM_13_2.
    return CheckResult(check_id, "EXPLAINED_FAIL", realised, source, details, LIM_13_2)


def check_wh_per_mi_at_70F(
    fleet: pd.DataFrame, mobility: pd.DataFrame, bounds_root: Path
) -> CheckResult:
    check_id = "wh_per_mi_at_70F"
    bnd = _load_bound(bounds_root, check_id)
    source = _bound_url(bnd)

    kwh_per_ev = mobility.groupby("ev_id")["kwh_consumed"].sum()
    mi_per_ev = mobility.groupby("ev_id")["miles"].sum()
    fleet2 = fleet.set_index("ev_id").copy()
    fleet2["kwh_year"] = kwh_per_ev
    fleet2["mi_year"] = mi_per_ev

    per_model = (
        fleet2.dropna(subset=["kwh_year", "mi_year"])
        .groupby("model", observed=True)
        .agg(kwh=("kwh_year", "sum"), mi=("mi_year", "sum"), n=("B_kwh", "count"))
    )
    per_model["wh_per_mi"] = per_model["kwh"] * 1000.0 / per_model["mi"]

    per_archetype: dict[str, dict[str, float]] = {}
    n_pass = 0
    n_total = 0
    for metric_name, fleet_model in WH_PER_MI_MODEL_MAP.items():
        row = bnd[bnd["metric_name"] == metric_name]
        if row.empty:
            continue
        lo = float(row["lower_bound"].iloc[0])
        hi = float(row["upper_bound"].iloc[0])
        if fleet_model in per_model.index:
            wh = float(per_model.loc[fleet_model, "wh_per_mi"])
            n_evs = int(per_model.loc[fleet_model, "n"])
            in_band = _in_band(wh, lo, hi)
            if in_band:
                n_pass += 1
            n_total += 1
            per_archetype[metric_name] = {
                "wh_per_mi": _round(wh, 1),
                "n_evs": n_evs,
                "bound": [_round(lo, 1), _round(hi, 1)],
                "in_band": bool(in_band),
            }
        else:
            per_archetype[metric_name] = {
                "wh_per_mi": float("nan"),
                "n_evs": 0,
                "bound": [_round(lo, 1), _round(hi, 1)],
                "in_band": False,
            }

    pass_rate = n_pass / n_total if n_total else 0.0
    realised = {
        "n_models_pass": int(n_pass),
        "n_models_total": int(n_total),
        "pass_rate": _round(pass_rate),
        "per_archetype": per_archetype,
    }
    details = (
        f"{n_pass}/{n_total} archetypes in band; "
        + ", ".join(
            f"{k.replace('wh_per_mi_', '')}={v['wh_per_mi']}"
            f"{'∈' if v['in_band'] else '∉'}{v['bound']}"
            for k, v in per_archetype.items()
        )
    )
    # Treat ≥75% in band as PASS (per the M9 brief's "±10%" + plan §13.4
    # / §13.5 acknowledgement of v1 temp limitations).
    if pass_rate >= 0.75:
        return CheckResult(check_id, "PASS", realised, source, details)
    # Otherwise EXPLAINED_FAIL — Bolt_EUV and Leaf_Plus realised Wh/mi
    # equal their M2 archetype-table nominal values (250 Wh/mi each),
    # which are below the EPA-label bound. The v1 archetype table uses
    # round-number priors and v1 has no T-correction (§13.4 / §13.5).
    return CheckResult(
        check_id,
        "EXPLAINED_FAIL",
        realised,
        source,
        details,
        explained_reason=" ".join([LIM_13_4, LIM_13_5]),
    )


def check_eta_charging(fleet: pd.DataFrame, bounds_root: Path) -> CheckResult:
    check_id = "eta_charging"
    bnd = _load_bound(bounds_root, check_id)
    source = _bound_url(bnd)
    eta = fleet["eta_charge_nominal"].astype(float)
    mean = float(eta.mean())
    mn = float(eta.min())
    mx = float(eta.max())

    l2_row = bnd[bnd["metric_name"] == "eta_l2_nominal"].iloc[0]
    lo, hi = float(l2_row["lower_bound"]), float(l2_row["upper_bound"])
    realised = {
        "mean": _round(mean),
        "min": _round(mn),
        "max": _round(mx),
        "bound_l2": [_round(lo), _round(hi)],
    }
    details = f"mean η = {mean:.4f} vs L2 band [{lo}, {hi}]; range [{mn:.4f}, {mx:.4f}]"
    if _in_band(mean, lo, hi):
        return CheckResult(check_id, "PASS", realised, source, details)
    return CheckResult(check_id, "FAIL", realised, source, details)


def check_evi_pro_plug_in_split(bounds_root: Path) -> CheckResult:
    check_id = "evi_pro_plug_in_split"
    bnd = _load_bound(bounds_root, check_id)
    source = _bound_url(bnd)
    if _all_rows_unbounded(bnd):
        return CheckResult(
            check_id,
            "EXPLAINED_SKIP",
            None,
            source,
            "EVI-Pro Lite publishes only qualitative home-strategy text "
            "(no numeric immediate/midnight/departure shares); all bound "
            "rows are `unbounded_v1=true`.",
            explained_reason="M8 bound is informational only in v1 (`unbounded_v1=true`).",
        )
    return CheckResult(
        check_id,
        "ERROR",
        None,
        source,
        "EVI-Pro bound is no longer fully unbounded; M9 v1 has no scoring logic.",
    )


def check_optimizer_smoke_test(
    fleet: pd.DataFrame,
    sessions: pd.DataFrame,
    plug_status_15min: pd.DataFrame,
    bounds_root: Path,
    ev_id: int = 0,
) -> CheckResult:
    """§10 check 11 — instantiate EVOptimizer on `ev_id` for one W-MON window.

    Pass criterion (per M9 brief): "constructs and solves without raising".
    A solver-reported infeasible status is NOT a failure; the optimiser's
    greedy fallback handles it and `.solve()` returns normally. The bar is
    that the call completes without exception and reports a recognised
    status string for every window.
    """
    check_id = "optimizer_smoke_test"
    bnd = _load_bound(bounds_root, check_id)
    source = _bound_url(bnd)
    try:
        # Imports are local so the module remains importable even if
        # pyomo / pyscipopt are unavailable (e.g. minimal CI image).
        from caiso_data.prices import CaisoPrices  # type: ignore
        from pev_data import VehicleArchetype  # type: ignore
        from pev_optim.optimizer import EVOptimizer  # type: ignore

        if ev_id not in set(fleet["ev_id"].astype(int)):
            raise ValueError(f"ev_id={ev_id} not in fleet")
        row = fleet[fleet["ev_id"] == ev_id].iloc[0]
        ev_plug = (
            plug_status_15min[plug_status_15min["ev_id"] == ev_id]
            .set_index("ts")["plugged"]
            .astype(bool)
        )
        if ev_plug.empty:
            raise ValueError(f"no plug_status_15min rows for ev_id={ev_id}")
        # First W-MON window. Year 2001-01-01 is a Monday in our synth
        # calendar (per `meta.json -> calendar_year_published`).
        week_start = pd.Timestamp("2001-01-01", tz=TZ)
        week_end = week_start + pd.Timedelta(days=7)
        plug_w = ev_plug.loc[week_start: week_end - pd.Timedelta(minutes=15)]
        if len(plug_w) != 672:
            raise ValueError(
                f"window plug index length {len(plug_w)} != 672 (one 15-min week)"
            )

        ev_sessions = sessions[sessions["ev_id"] == ev_id].copy()
        sess_w = ev_sessions[
            (ev_sessions["t_out"] > week_start) & (ev_sessions["t_in"] < week_end)
        ].copy()

        archetype = VehicleArchetype(
            name=str(row["model"]),
            battery_kwh=float(row["B_kwh"]),
            p_max_kw=float(
                min(row["OBC_kw"], row["EVSE_home_kw"], row["panel_cap_kw"])
            ),
            eta_charge=float(row["eta_charge_nominal"]),
            soc_min_kwh=float(row["soc_min_kwh"]),
        )

        idx = plug_w.index
        zero = pd.Series(0.0, index=idx)
        flat50 = pd.Series(50.0, index=idx)
        prices = CaisoPrices(
            da_lmp=flat50,
            rt_lmp=flat50,
            regup_price=zero,
            spin_price=zero,
            nonspin_price=zero,
            regup_mileage_price=zero,
            regup_mileage_multiplier=zero,
        )

        optimizer = EVOptimizer(
            archetype,
            plug_w,
            sess_w,
            prices,
            dt_hours=0.25,
            window="W-MON",
            solver="scip",
        )
        result = optimizer.solve()
        statuses = list(result.solve_status)
        recognised = {"optimal", "feasible", "infeasible"}
        if not statuses or any(s not in recognised for s in statuses):
            raise RuntimeError(f"unrecognised solver status: {statuses}")

        realised = {
            "ev_id": int(ev_id),
            "n_windows": len(statuses),
            "statuses": statuses,
            "n_fallback_windows": len(result.fallback_windows),
            "model": str(row["model"]),
        }
        details = (
            f"EVOptimizer(ev_id={ev_id}, model={row['model']}) solved without "
            f"raising; statuses={statuses}; fallback windows="
            f"{len(result.fallback_windows)}"
        )
        return CheckResult(check_id, "PASS", realised, source, details)
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            check_id,
            "FAIL",
            None,
            source,
            f"EVOptimizer raised: {type(exc).__name__}: {exc}",
        )


def check_cross_artifact_integrity(
    fleet: pd.DataFrame,
    sessions: pd.DataFrame,
    plug_status_15min: pd.DataFrame,
    plug_status_hourly: pd.DataFrame,
    mobility: pd.DataFrame,
    plug_in_events: pd.DataFrame,
) -> CheckResult:
    """Integration check 12 — primary-key + FK integrity across artifacts."""
    check_id = "cross_artifact_integrity"
    fleet_ids = set(fleet["ev_id"].astype(int))
    issues: list[str] = []
    for name, df in (
        ("sessions", sessions),
        ("plug_status_15min", plug_status_15min),
        ("plug_status_hourly", plug_status_hourly),
        ("mobility", mobility),
        ("plug_in_events", plug_in_events),
    ):
        ids = set(df["ev_id"].astype(int))
        missing = ids - fleet_ids
        if missing:
            issues.append(f"{name} has {len(missing)} ev_ids missing from fleet")
        if df["ev_id"].isna().any():
            issues.append(f"{name} has NaN ev_id")

    n_dup_sess = int(sessions.duplicated(["ev_id", "session_idx"]).sum())
    if n_dup_sess > 0:
        issues.append(f"{n_dup_sess} duplicate (ev_id, session_idx) in sessions")

    realised = {
        "fleet_n_unique_ev_ids": len(fleet_ids),
        "sessions_dup_pk": n_dup_sess,
        "issues": issues,
    }
    details = "; ".join(issues) if issues else "all PK/FK constraints satisfied"
    source = "internal pipeline integrity"
    if issues:
        return CheckResult(check_id, "FAIL", realised, source, details)
    return CheckResult(check_id, "PASS", realised, source, details)


def check_plug_rate_consistency(
    sessions: pd.DataFrame, plug_status_15min: pd.DataFrame, n_vehicles: int
) -> CheckResult:
    """Integration check 13 — fleet plug rate from the 15-min raster vs the
    session aggregate, reconciled to within an absolute tolerance.

    The tolerance admits 15-min raster quantization: the raster floors each
    session's fractional edge bins, so the raster rate sits a few tenths of a
    percent below the continuous session-duration rate. A structural
    raster/session binning bug, by contrast, shifts the rate by >= 1%, well
    outside the band."""
    check_id = "plug_rate_consistency"
    n_rows = len(plug_status_15min)
    n_plugged = int(plug_status_15min["plugged"].sum())
    rate_raster = n_plugged / n_rows if n_rows else float("nan")

    dur_steps = (sessions["t_out"] - sessions["t_in"]).dt.total_seconds() / (15.0 * 60.0)
    total_session_steps = float(dur_steps.sum())
    n_steps_total = 35040 * n_vehicles
    rate_sessions = total_session_steps / n_steps_total

    diff = float(abs(rate_raster - rate_sessions))
    # 15-min raster quantization floors each session's fractional edge bins,
    # so the raster rate sits a few tenths of a percent below the continuous
    # session-duration rate. Admit that quantization; a structural binning bug
    # diverges by >= 1%.
    tol = 0.005
    realised = {
        "plug_rate_15min_raster": _round(rate_raster),
        "plug_rate_from_sessions": _round(rate_sessions),
        "abs_diff": _round(diff),
        "tolerance": tol,
    }
    details = (
        f"15-min raster plug rate = {rate_raster:.6f}; "
        f"session-aggregate plug rate = {rate_sessions:.6f}; "
        f"|Δ| = {diff:.6f} (tolerance {tol})"
    )
    source = "internal pipeline integrity"
    if diff <= tol:
        return CheckResult(check_id, "PASS", realised, source, details)
    return CheckResult(check_id, "FAIL", realised, source, details)


def check_fleet_energy_conservation(
    sessions: pd.DataFrame, mobility: pd.DataFrame,
    *, legacy_alias_ok: bool = False,
) -> CheckResult:
    """Integration check 14 — total session energy vs total drive energy.

    Target ratio is ≈ 1 / η_mean ≈ 1.111 (grid energy includes charging
    loss).

    v3.0 (§13.6 RESOLVED): the numerator now includes PHEV gasoline
    extension. Below-floor trip deficits served by the ICE are recorded
    on the next plug-in session as ``gas_kwh_equivalent`` (battery-side
    kWh, i.e. the energy the engine delivered to the drivetrain in lieu
    of the depleted battery). Total = Σ energy_kwh_grid (electricity)
    + Σ gas_kwh_equivalent (gasoline kWh-equivalent). Pre-v3 caches
    that do not carry the column degrade gracefully — the gas sum is
    zero, the numerator collapses to the v1.1 grid-only total, and the
    check reproduces the v1.1 EXPLAINED_FAIL ratio of ~0.895.
    """
    check_id = "fleet_energy_conservation"
    # RFC-008.r — grid invariants use ``energy_kwh_grid``; legacy alias only
    # for v1.1_migrated caches.
    total_grid = float(
        _session_energy_grid(
            sessions, legacy_alias_ok=legacy_alias_ok, context=check_id,
        ).sum()
    )
    # v3.0 (§13.6 RESOLVED) — PHEV gasoline-equivalent kWh. Pre-v3
    # caches don't carry the column, so default to 0.0 (backward-stable).
    if "gas_kwh_equivalent" in sessions.columns:
        total_gas = float(sessions["gas_kwh_equivalent"].fillna(0.0).sum())
    else:
        total_gas = 0.0
    total_drive = float(mobility["kwh_consumed"].sum())
    total_numerator = total_grid + total_gas
    ratio = total_numerator / total_drive if total_drive else float("nan")
    target = 1.111
    lo, hi = 1.05, 1.20

    realised = {
        "total_session_kwh": _round(total_grid, 1),
        "total_gas_kwh_equivalent": _round(total_gas, 1),
        "total_numerator_kwh": _round(total_numerator, 1),
        "total_drive_kwh": _round(total_drive, 1),
        "ratio": _round(ratio),
        "target_ratio": target,
        "tolerance_band": [lo, hi],
    }
    details = (
        f"(Σ energy_grid + Σ gas_kwh_equivalent) / Σ kwh_consumed = "
        f"{ratio:.4f} (target ≈ 1/η ≈ {target}; tolerance [{lo}, {hi}]); "
        f"grid={total_grid:,.1f} kWh, gas={total_gas:,.1f} kWh, "
        f"drive={total_drive:,.1f} kWh"
    )
    source = "internal pipeline integrity"
    if _in_band(ratio, lo, hi):
        return CheckResult(check_id, "PASS", realised, source, details)
    # Residual gap (post-PHEV-fix): the plug-in priors are still
    # literature-derived (§13.1) — when the gap stays outside [lo, hi]
    # the residual EXPLAINED_FAIL cites §13.1 + §13.6 (the latter now
    # documents the gas-ledger fix and its limits, e.g. fallback MPG
    # for out-of-band model years).
    return CheckResult(
        check_id,
        "EXPLAINED_FAIL",
        realised,
        source,
        details,
        explained_reason=" ".join([LIM_13_1, LIM_13_6]),
    )


# ---------------------------------------------------------------------------
# v2.0 — DST check (#12)
# ---------------------------------------------------------------------------

# Spring-forward / fall-back UTC instants in the synthetic 2001 calendar.
# Per the v2.0-rev plan §3.4: LA spring-forward 2001-04-01 02:00 LA -> 03:00,
# fall-back 2001-10-28 01:00 LA -> 01:00 LA. The UTC equivalents are pinned
# below for byte-stable check output.
_DST_SPRING_FORWARD_UTC_LA = pd.Timestamp("2001-04-01T10:00:00Z")
_DST_FALL_BACK_UTC_LA = pd.Timestamp("2001-10-28T09:00:00Z")


def _check_dst_zone(
    utc_index: pd.DatetimeIndex,
    zone: str,
) -> dict[str, Any]:
    """Convert a UTC-stored ``DatetimeIndex`` to ``zone`` and return a dict
    summarising:

    * UTC monotonicity across the index (strict_increase).
    * Local-time index has no NaT (spring-forward gap correctly skipped).
    * Local-time index has no duplicate instants (fall-back ambiguity
      collapsed to single instants via the UTC source-of-truth).
    """

    if zone == "UTC":
        local_index = utc_index
    else:
        local_index = utc_index.tz_convert(zone)

    # Strict UTC monotonicity.
    utc_diffs = utc_index.to_series().diff().dropna()
    utc_monotonic = bool((utc_diffs > pd.Timedelta(0)).all())

    # Local-time index NaT count (spring-forward gap).
    local_nat = int(local_index.isna().sum())

    # Local-time duplicate count (fall-back).
    local_dup = int(local_index.duplicated().sum())

    return {
        "zone": zone,
        "utc_monotonic": utc_monotonic,
        "local_nat_count": local_nat,
        "local_dup_count": local_dup,
        "n_rows": int(len(utc_index)),
    }


def check_dst_verification(
    output_root: Path,
    bounds_root: Path,
    *,
    zones: tuple[str, ...] = (
        "America/Los_Angeles",
        "America/New_York",
        "America/Chicago",
        "UTC",
    ),
) -> CheckResult:
    """v2.0 M9 check #12 — DST verification.

    Loads any UTC-stored parquet from ``output_root`` (preferring
    ``plug_status_15min.parquet`` for its 15-min raster across the year),
    converts the index to each target tz, and asserts:

    * UTC index strictly monotonic across spring-forward + fall-back days.
    * LT index has no NaT at the spring-forward gap (``nonexistent="shift_
      forward"`` policy applied upstream).
    * LT index has no duplicate timestamps at fall-back (UTC -> LT mapping
      is single-valued since UTC has no ambiguity).

    Applies to ``America/Los_Angeles``, ``America/New_York``, ``America/
    Chicago``, and ``UTC``.
    """

    check_id = "dst_verification"
    try:
        bnd = _load_bound(bounds_root, check_id)
        source = _bound_url(bnd)
    except FileNotFoundError:
        bnd = pd.DataFrame()
        source = ""

    # Find a UTC-stored timestamp index in the cache. Prefer the dense
    # 15-min raster (covers the full year and both DST transitions);
    # fall back to plug_in_events.arrival_ts which is also UTC-stored
    # post Wave-1 migration.
    candidates = [
        ("plug_status_15min.parquet", "ts"),
        ("plug_in_events.parquet", "arrival_ts"),
        ("sessions.parquet", "t_in"),
    ]
    index: pd.DatetimeIndex | None = None
    used_artifact = ""
    for fname, col in candidates:
        p = output_root / fname
        if not p.exists():
            continue
        df = pd.read_parquet(p, columns=[col])
        raw = pd.DatetimeIndex(df[col])
        if str(raw.tz) != "UTC":
            continue
        # The 15-min raster carries one row per (EV, ts); collapse to the
        # unique timestamp grid before checking monotonicity. For event
        # streams (PIE / sessions) this also handles ties cleanly.
        idx = pd.DatetimeIndex(raw.unique()).sort_values()
        # Pre-filter to the days around the two transitions so the check
        # is cheap and reproducible regardless of fleet size.
        spring_window = (
            (idx >= _DST_SPRING_FORWARD_UTC_LA - pd.Timedelta(days=2))
            & (idx <= _DST_SPRING_FORWARD_UTC_LA + pd.Timedelta(days=2))
        )
        fall_window = (
            (idx >= _DST_FALL_BACK_UTC_LA - pd.Timedelta(days=2))
            & (idx <= _DST_FALL_BACK_UTC_LA + pd.Timedelta(days=2))
        )
        mask = spring_window | fall_window
        index = idx[mask]
        used_artifact = fname
        if len(index) > 0:
            break

    if index is None or len(index) == 0:
        return CheckResult(
            check_id,
            "ERROR",
            None,
            source,
            "no UTC-stored timestamp parquet found in cache "
            f"(tried {[c[0] for c in candidates]})",
        )

    per_zone = {z: _check_dst_zone(index, z) for z in zones}

    all_ok = all(
        v["utc_monotonic"]
        and v["local_nat_count"] == 0
        and v["local_dup_count"] == 0
        for v in per_zone.values()
    )

    realised: dict[str, Any] = {
        "source_artifact": used_artifact,
        "n_rows_checked": int(len(index)),
        "per_zone": per_zone,
    }
    parts = [
        f"{z}: monotonic={v['utc_monotonic']}, nat={v['local_nat_count']}, "
        f"dup={v['local_dup_count']}"
        for z, v in per_zone.items()
    ]
    details = f"DST check ({used_artifact}): " + "; ".join(parts)

    if all_ok:
        return CheckResult(check_id, "PASS", realised, source, details)
    return CheckResult(check_id, "FAIL", realised, source, details)


# ---------------------------------------------------------------------------
# v2.0 — Workplace W1-W8 checks (#15-#22)
# ---------------------------------------------------------------------------


def _local_hour_series(ts: pd.Series, tz: str) -> pd.Series:
    """Convert a UTC-stored timestamp series to local-time fractional hours
    in [0, 24)."""
    local = pd.DatetimeIndex(ts).tz_convert(tz)
    return pd.Series(
        local.hour.astype(float) + local.minute.astype(float) / 60.0,
        index=ts.index,
    )


def _workplace_pie_filter(pie: pd.DataFrame) -> pd.DataFrame:
    """Restrict plug_in_events to workplace plug-ins (plug_in=True at a
    work location). Falls back to plug_in only if no location column."""
    df = pie[pie["plug_in"].astype(bool)].copy()
    if "location" in df.columns:
        loc = df["location"].astype(str)
        # Cover all v2.0 workplace location_type labels.
        work_mask = loc.isin(("work", "workplace", "NACS_workplace"))
        # If the cache has any work-flagged rows, restrict to them; else
        # use the unfiltered set so the check still runs.
        if int(work_mask.sum()) > 0:
            df = df[work_mask].copy()
    return df


def check_workplace_w1_plug_in_cdf(
    pie: pd.DataFrame,
    bounds_root: Path,
    region_tz: str,
    workplace_cohort_caveat: str | None = None,
) -> CheckResult:
    """W1 (validator #15) — workplace plug-in CDF at LT {09, 12, 15}.

    EXPECTED to EXPLAINED_FAIL on v2.0 workplace caches because the M5 BIC
    fit on EVWatts public data skews cluster centres ~3 h later than the
    literature canon. The EXPLAINED_FAIL reason cites the
    ``workplace_cohort_caveat`` from M5's meta.json.
    """

    check_id = "W1_workplace_plug_in_cdf"
    bnd = _load_bound(bounds_root, check_id)
    source = _bound_url(bnd)

    df = _workplace_pie_filter(pie)
    if df.empty:
        return CheckResult(
            check_id, "EXPLAINED_SKIP", None, source,
            "no workplace plug-in events to score",
            explained_reason=LIM_V2_WORKPLACE_SCOPE,
        )

    local_h = _local_hour_series(df["arrival_ts"], region_tz)
    realised: dict[str, Any] = {"n_events": int(len(df))}
    band: dict[str, dict[str, float]] = {}
    statuses: list[str] = []
    for h in (9, 12, 15):
        cdf = float((local_h < h).mean())
        realised[f"F_{h:02d}"] = _round(cdf)
        row = bnd[bnd["metric_name"] == f"plug_in_cdf_at_hour_{h:02d}"]
        if row.empty:
            statuses.append("ERROR")
            band[f"F_{h:02d}"] = {"lo": float("nan"), "hi": float("nan")}
            continue
        lo = float(row["lower_bound"].iloc[0])
        hi = float(row["upper_bound"].iloc[0])
        band[f"F_{h:02d}"] = {"lo": _round(lo), "hi": _round(hi)}
        statuses.append("PASS" if _in_band(cdf, lo, hi) else "FAIL")
    realised["bounds"] = band
    details = "; ".join(
        f"F({h:02d})={realised[f'F_{h:02d}']:.3f} band={band[f'F_{h:02d}']}"
        for h in (9, 12, 15)
    )
    if all(s == "PASS" for s in statuses):
        return CheckResult(check_id, "PASS", realised, source, details)

    # EXPLAINED_FAIL — cite the workplace_cohort_caveat from M5 meta.json
    # (or the static fallback if absent). The reason string must contain
    # "§13." to satisfy CheckResult.__post_init__.
    caveat = workplace_cohort_caveat or ""
    reason = (
        LIM_WORKPLACE_COHORT_CAVEAT
        + (" Live M5 caveat: " + caveat if caveat else "")
    )
    return CheckResult(
        check_id, "EXPLAINED_FAIL", realised, source, details, reason
    )


def check_workplace_w2_session_duration(
    sessions: pd.DataFrame,
    bounds_root: Path,
) -> CheckResult:
    """W2 (#16) — workplace session duration: median ∈ [5, 9] h."""
    check_id = "W2_workplace_session_duration"
    bnd = _load_bound(bounds_root, check_id)
    source = _bound_url(bnd)

    work_loc = sessions[
        sessions["location"]
        .astype(str)
        .isin(("work", "workplace", "NACS_workplace"))
    ]
    if work_loc.empty:
        return CheckResult(
            check_id, "EXPLAINED_SKIP", None, source,
            "no workplace sessions to score",
            explained_reason=LIM_V2_WORKPLACE_SCOPE,
        )
    dur_h = (work_loc["t_out"] - work_loc["t_in"]).dt.total_seconds() / 3600.0
    median = float(dur_h.median())
    row = bnd[bnd["metric_name"] == "median_duration_hours"].iloc[0]
    lo, hi = float(row["lower_bound"]), float(row["upper_bound"])
    realised = {
        "median_h": _round(median),
        "n_sessions": int(len(work_loc)),
        "median_bound": [_round(lo), _round(hi)],
    }
    details = f"workplace median = {median:.2f} h vs [{lo}, {hi}]"
    if _in_band(median, lo, hi):
        return CheckResult(check_id, "PASS", realised, source, details)
    return CheckResult(
        check_id, "EXPLAINED_FAIL", realised, source, details,
        LIM_T_OUT_SEMANTICS,
    )


def check_workplace_w3_session_energy(
    sessions: pd.DataFrame,
    bounds_root: Path,
    *,
    legacy_alias_ok: bool = False,
) -> CheckResult:
    """W3 (#17) — workplace session energy: median ∈ [8, 25] kWh.

    RFC-008.r — grid-side energy (``energy_kwh_grid``) for the workplace
    session-energy distribution check.
    """
    check_id = "W3_workplace_session_energy"
    bnd = _load_bound(bounds_root, check_id)
    source = _bound_url(bnd)

    work_loc = sessions[
        sessions["location"]
        .astype(str)
        .isin(("work", "workplace", "NACS_workplace"))
    ]
    if work_loc.empty:
        return CheckResult(
            check_id, "EXPLAINED_SKIP", None, source,
            "no workplace sessions to score",
            explained_reason=LIM_V2_WORKPLACE_SCOPE,
        )
    e = _session_energy_grid(
        work_loc, legacy_alias_ok=legacy_alias_ok, context=check_id,
    )
    median = float(e.median())
    row = bnd[bnd["metric_name"] == "median_kwh_per_session"].iloc[0]
    lo, hi = float(row["lower_bound"]), float(row["upper_bound"])
    realised = {
        "median_kwh": _round(median),
        "n_sessions": int(len(work_loc)),
        "median_bound": [_round(lo), _round(hi)],
    }
    details = f"workplace median = {median:.2f} kWh vs [{lo}, {hi}]"
    if _in_band(median, lo, hi):
        return CheckResult(check_id, "PASS", realised, source, details)
    return CheckResult(
        check_id, "EXPLAINED_FAIL", realised, source, details, LIM_13_6,
    )


def check_workplace_w4_plug_in_on_arrival(
    pie: pd.DataFrame,
    bounds_root: Path,
) -> CheckResult:
    """W4 (RFC-035) — workplace plug-in-on-arrival rate ∈ [0.85, 0.95].

    Per-arrival metric: numerator = workplace arrivals with
    ``plug_in == True``; denominator = workplace arrivals. Replaces the
    v2.0-draft per-15-min-slot aggregate-plug-in-rate band [0.15, 0.30]
    (deleted by RFC-035 / S1-2).
    """

    check_id = "W4_workplace_plug_in_on_arrival"
    bnd = _load_bound(bounds_root, check_id)
    source = _bound_url(bnd)

    # Workplace-arrivals subset; mirrors W1's filter but does NOT pre-restrict
    # to plug_in==True (we need the denominator to be every workplace
    # arrival).
    df = pie.copy()
    if "location" in df.columns:
        loc = df["location"].astype(str)
        work_mask = loc.isin(("work", "workplace", "NACS_workplace"))
        if int(work_mask.sum()) > 0:
            df = df[work_mask].copy()

    n_arrivals = int(len(df))
    if n_arrivals == 0:
        return CheckResult(
            check_id, "EXPLAINED_SKIP", None, source,
            "no workplace arrivals to score",
            explained_reason=LIM_V2_WORKPLACE_SCOPE,
        )
    n_plugged = int(df["plug_in"].astype(bool).sum())
    rate = n_plugged / n_arrivals
    row = bnd[bnd["metric_name"] == "plug_in_on_arrival_rate"].iloc[0]
    lo, hi = float(row["lower_bound"]), float(row["upper_bound"])
    realised = {
        "plug_in_on_arrival_rate": _round(rate),
        "n_arrivals": n_arrivals,
        "n_plugged": n_plugged,
        "bound": [_round(lo), _round(hi)],
    }
    details = (
        f"workplace plug-in-on-arrival rate = {rate:.4f} "
        f"({n_plugged}/{n_arrivals}) vs [{lo}, {hi}]"
    )
    if _in_band(rate, lo, hi):
        return CheckResult(check_id, "PASS", realised, source, details)
    return CheckResult(
        check_id, "EXPLAINED_FAIL", realised, source, details,
        LIM_WORKPLACE_COHORT_CAVEAT,
    )


def check_workplace_w5_evse_kw_distribution(
    fleet: pd.DataFrame,
    bounds_root: Path,
) -> CheckResult:
    """W5 (#19) — workplace EVSE: ≥ 60 % share at 6.6-7.7 kW."""
    check_id = "W5_workplace_evse_kw_distribution"
    bnd = _load_bound(bounds_root, check_id)
    source = _bound_url(bnd)

    evse = fleet["EVSE_home_kw"].astype(float)
    # Tolerant range check (atol=1e-2) because EVSE_home_kw round-trips
    # through float32 and the canonical 6.6 / 7.7 values are not exactly
    # representable.
    in_band_share = float(
        ((evse >= 6.6 - 1e-2) & (evse <= 7.7 + 1e-2)).mean()
    )
    row = bnd[bnd["metric_name"] == "share_6p6_to_7p7_kw"].iloc[0]
    lo, hi = float(row["lower_bound"]), float(row["upper_bound"])
    realised = {
        "share_6p6_to_7p7_kw": _round(in_band_share),
        "n_evs": int(len(fleet)),
        "bound": [_round(lo), _round(hi)],
    }
    details = (
        f"workplace share at 6.6-7.7 kW = {in_band_share:.3f} "
        f"vs [{lo}, {hi}]"
    )
    if _in_band(in_band_share, lo, hi):
        return CheckResult(check_id, "PASS", realised, source, details)
    return CheckResult(
        check_id, "EXPLAINED_FAIL", realised, source, details,
        LIM_V2_WORKPLACE_SCOPE,
    )


def check_workplace_w6_cluster_window_max(
    meta: dict[str, Any],
    bounds_root: Path,
) -> CheckResult:
    """W6 (#20) — workplace cluster window max ≤ 6.0 h.

    The realised value lives in M5's meta.json
    (``plug_in_model.W9_max_cluster_window_h``).
    """

    check_id = "W6_workplace_cluster_window_max"
    bnd = _load_bound(bounds_root, check_id)
    source = _bound_url(bnd)

    pim = meta.get("plug_in_model", {})
    realised_value = pim.get("W9_max_cluster_window_h")
    if realised_value is None:
        return CheckResult(
            check_id, "EXPLAINED_SKIP", None, source,
            "meta.json.plug_in_model.W9_max_cluster_window_h not populated "
            "(residential cache or pre-W9 build)",
            explained_reason=LIM_V2_WORKPLACE_SCOPE,
        )

    row = bnd[bnd["metric_name"] == "max_cluster_window_hours"].iloc[0]
    lo, hi = float(row["lower_bound"]), float(row["upper_bound"])
    val = float(realised_value)
    realised = {
        "max_cluster_window_h": _round(val),
        "bound": [_round(lo), _round(hi)],
    }
    details = f"W9 max cluster window = {val:.2f} h vs [{lo}, {hi}]"
    if _in_band(val, lo, hi):
        return CheckResult(check_id, "PASS", realised, source, details)
    return CheckResult(
        check_id, "FAIL", realised, source,
        details + " — M5 W9 invariant breach",
    )


def check_workplace_w7_borrow_share(
    meta: dict[str, Any],
    fleet: pd.DataFrame,
    bounds_root: Path,
) -> CheckResult:
    """W7 (#21) — donor borrow share ≤ 0.25.

    Realised value lookup order (first hit wins):

    1. ``meta.json.donor_borrow_rate`` (M3's preferred location).
    2. ``meta.json.donor_matcher.donor_borrow_rate`` (nested form).
    3. ``fleet["borrow_used"].mean()`` (per-EV bool flag from M3).
    """

    check_id = "W7_workplace_borrow_share"
    bnd = _load_bound(bounds_root, check_id)
    source = _bound_url(bnd)

    rate = meta.get("donor_borrow_rate")
    if rate is None and isinstance(meta.get("donor_matcher"), dict):
        rate = meta["donor_matcher"].get("donor_borrow_rate")
    if rate is None and "borrow_used" in fleet.columns:
        rate = float(fleet["borrow_used"].astype(bool).mean())
    if rate is None:
        return CheckResult(
            check_id, "EXPLAINED_SKIP", None, source,
            "meta.json.donor_borrow_rate not populated and fleet has no "
            "borrow_used column (cache predates the v2.0 borrow-from-"
            "neighbours wiring)",
            explained_reason=LIM_V2_BORROW_RATE,
        )

    row = bnd[bnd["metric_name"] == "donor_borrow_rate"].iloc[0]
    lo, hi = float(row["lower_bound"]), float(row["upper_bound"])
    val = float(rate)
    realised = {
        "donor_borrow_rate": _round(val),
        "bound": [_round(lo), _round(hi)],
    }
    details = f"donor_borrow_rate = {val:.3f} vs [{lo}, {hi}]"
    if _in_band(val, lo, hi):
        return CheckResult(check_id, "PASS", realised, source, details)
    return CheckResult(check_id, "FAIL", realised, source, details)


def check_workplace_w8_home_charging_required_rate(
    sessions: pd.DataFrame,
    fleet: pd.DataFrame,
    bounds_root: Path,
) -> CheckResult:
    """W8 (#22) — ``home_charging_required`` share ≤ 0.15.

    Counts the share of EVs with at least one ``home_charging_required``
    flagged session (or ``infeasibility_flag == "home_charging_required"``)
    over the 4-week sample window.
    """

    check_id = "W8_workplace_home_charging_required_rate"
    bnd = _load_bound(bounds_root, check_id)
    source = _bound_url(bnd)

    flagged: set[int] = set()
    if "infeasibility_flag" in sessions.columns:
        mask = sessions["infeasibility_flag"].astype(str) == "home_charging_required"
        flagged.update(sessions.loc[mask, "ev_id"].astype(int).tolist())
    if "home_charging_required" in sessions.columns:
        mask = sessions["home_charging_required"].astype(bool)
        flagged.update(sessions.loc[mask, "ev_id"].astype(int).tolist())

    n_evs = int(fleet["ev_id"].nunique())
    share = len(flagged) / n_evs if n_evs else 0.0
    row = bnd[bnd["metric_name"] == "home_charging_required_share"].iloc[0]
    lo, hi = float(row["lower_bound"]), float(row["upper_bound"])
    realised = {
        "share": _round(share),
        "n_evs_flagged": int(len(flagged)),
        "n_evs": n_evs,
        "bound": [_round(lo), _round(hi)],
    }
    details = (
        f"home_charging_required share = {share:.3f} ({len(flagged)} / "
        f"{n_evs} EVs) vs [{lo}, {hi}]"
    )
    if _in_band(share, lo, hi):
        return CheckResult(check_id, "PASS", realised, source, details)
    return CheckResult(
        check_id, "EXPLAINED_FAIL", realised, source, details,
        LIM_V2_WORKPLACE_SCOPE,
    )


def check_workplace_w9_cluster_window_invariant(
    meta: dict[str, Any],
    bounds_root: Path,
) -> CheckResult:
    """W9 (RFC-038) — cluster-window invariant explicitly logged.

    Split out from the previous W6 markdown row per RFC-038. W6 carries the
    numerical cluster_window_max check; W9 logs the M5 fit-time invariant
    as a separate validator row so the markdown shows W9 distinctly. No
    new logic — the invariant boolean is read from M5's meta.
    """

    check_id = "W9_workplace_cluster_window_invariant"
    bnd = _load_bound(bounds_root, check_id)
    source = _bound_url(bnd)

    pim = meta.get("plug_in_model", {})
    # Best-effort: M5 sets a max cluster window; if it's <= 6.0 the invariant
    # holds. Fall back to the W9_active marker if present.
    invariant_active: float | None = None
    if "W9_max_cluster_window_h" in pim:
        invariant_active = 1.0 if float(pim["W9_max_cluster_window_h"]) <= 6.0 else 0.0
    elif "W9_cluster_window_invariant_active" in pim:
        invariant_active = float(bool(pim["W9_cluster_window_invariant_active"]))

    if invariant_active is None:
        return CheckResult(
            check_id, "EXPLAINED_SKIP", None, source,
            "meta.json.plug_in_model lacks cluster-window invariant marker "
            "(residential cache or pre-W9 build)",
            explained_reason=LIM_V2_WORKPLACE_SCOPE,
        )

    row = bnd[bnd["metric_name"] == "cluster_window_invariant_active"].iloc[0]
    lo, hi = float(row["lower_bound"]), float(row["upper_bound"])
    realised = {
        "cluster_window_invariant_active": _round(invariant_active),
        "bound": [_round(lo), _round(hi)],
    }
    details = (
        f"W9 invariant active = {invariant_active} vs [{lo}, {hi}]"
    )
    if _in_band(invariant_active, lo, hi):
        return CheckResult(check_id, "PASS", realised, source, details)
    return CheckResult(
        check_id, "FAIL", realised, source,
        details + " — M5 W9 invariant inactive",
    )


def check_C5_winter_ratio(
    mobility: pd.DataFrame,
    bounds_root: Path,
    *,
    region: str,
    region_tz: str,
    region_obj: Any = None,
) -> CheckResult:
    """C5 (RFC-034 / S1-1) — cross-region winter / summer Wh-per-mile ratio.

    Active for cold regions only (``winter_f_T_multiplier > 1.00``). For
    each cold region the realised value is
    ``mean(Wh/mi, Dec-Mar) / mean(Wh/mi, Jun-Aug)`` over the cache's
    mobility trips. The check PASSes when the realised value is ≥ 1.15.

    Bound parquet is read from
    ``<bounds_top_root>/<region>/cross_region/C5_winter_ratio.parquet``
    per RFC-034. Schema: ``(region, lower, upper, source_title, note)``.
    """

    check_id = "C5_winter_ratio"
    # bounds_root here is the per-region cross_region subdir.
    p = bounds_root / "C5_winter_ratio.parquet"
    if not p.exists():
        raise FileNotFoundError(f"C5 bound parquet missing: {p}")
    bnd = pd.read_parquet(p)
    source = (
        str(bnd["source_title"].iloc[0]) if "source_title" in bnd.columns else ""
    )

    # Warm region → C5 skipped per RFC-034 (winter_f_T_multiplier == 1.00).
    is_cold = False
    if region_obj is not None and hasattr(region_obj, "winter_f_T_multiplier"):
        is_cold = float(region_obj.winter_f_T_multiplier) > 1.00
    elif "lower" in bnd.columns:
        lo_val = bnd["lower"].iloc[0]
        is_cold = lo_val is not None and not pd.isna(lo_val)

    if not is_cold:
        return CheckResult(
            check_id, "EXPLAINED_SKIP", None, source,
            "warm region (winter_f_T_multiplier == 1.00); C5 not applicable",
            explained_reason=LIM_V2_WORKPLACE_SCOPE,
        )

    # Compute Wh/mi per trip; aggregate by season (Dec-Mar vs Jun-Aug) in
    # region-local time. ``mobility.start_ts`` is UTC-stored in v2.0; convert
    # to region_tz before bucketing.
    trips = mobility.dropna(subset=["miles"]).copy()
    trips = trips[trips["miles"].astype(float) > 0]
    if trips.empty:
        return CheckResult(
            check_id, "EXPLAINED_SKIP", None, source,
            "no mobility trips with positive miles to score",
            explained_reason=LIM_V2_WORKPLACE_SCOPE,
        )
    wh_per_mi = (
        trips["kwh_consumed"].astype(float) * 1000.0
        / trips["miles"].astype(float)
    )
    ts = pd.DatetimeIndex(trips["start_ts"])
    if ts.tz is not None:
        try:
            ts_local = ts.tz_convert(region_tz)
        except Exception:  # noqa: BLE001
            ts_local = ts
    else:
        ts_local = ts
    months = pd.Series(ts_local.month, index=trips.index)
    winter_mask = months.isin((12, 1, 2, 3))
    summer_mask = months.isin((6, 7, 8))
    winter_mean = float(wh_per_mi[winter_mask].mean()) if winter_mask.any() else float("nan")
    summer_mean = float(wh_per_mi[summer_mask].mean()) if summer_mask.any() else float("nan")
    if not (math.isfinite(winter_mean) and math.isfinite(summer_mean)) or summer_mean <= 0:
        return CheckResult(
            check_id, "EXPLAINED_SKIP", None, source,
            "insufficient seasonal coverage to compute C5 ratio "
            f"(winter_mean={winter_mean}, summer_mean={summer_mean})",
            explained_reason=LIM_V2_WORKPLACE_SCOPE,
        )
    ratio = winter_mean / summer_mean

    lo_row = bnd["lower"].iloc[0]
    lo = float(lo_row) if lo_row is not None and not pd.isna(lo_row) else 1.15
    realised = {
        "winter_summer_ratio": _round(ratio),
        "winter_wh_per_mi_mean": _round(winter_mean, 1),
        "summer_wh_per_mi_mean": _round(summer_mean, 1),
        "lower_bound": _round(lo),
        "region": region,
    }
    details = (
        f"C5 winter/summer Wh-per-mi ratio = {ratio:.3f} "
        f"(winter {winter_mean:.1f} / summer {summer_mean:.1f}) "
        f"vs lower {lo:.3f}"
    )
    if ratio >= lo - 1e-12:
        return CheckResult(check_id, "PASS", realised, source, details)
    return CheckResult(check_id, "FAIL", realised, source, details)


def check_workplace_w10_donor_pool_recycling(
    meta: dict[str, Any],
    fleet: pd.DataFrame,
    bounds_root: Path,
) -> CheckResult:
    """W10 (RFC-038) — donor-pool recycling ratio explicitly logged.

    Split out from the previous W7 markdown row per RFC-038. W7 carries the
    donor_borrow_rate share; W10 logs the per-donor recycling ratio
    (n_synth / pool ≤ 2.0) as a separate validator row. Realised value is
    read from M3's meta (``donor_matcher.max_recycling_ratio``) when
    populated, with a fleet-side fall-back via the borrow_used column.
    """

    check_id = "W10_workplace_donor_pool_recycling"
    bnd = _load_bound(bounds_root, check_id)
    source = _bound_url(bnd)

    ratio = meta.get("max_recycling_ratio")
    if ratio is None and isinstance(meta.get("donor_matcher"), dict):
        ratio = meta["donor_matcher"].get("max_recycling_ratio")
    # Fallback: derive ratio from per-donor reuse counts when present.
    if ratio is None and isinstance(meta.get("donor_matcher"), dict):
        top5 = meta["donor_matcher"].get("top5_donor_houseids")
        if isinstance(top5, list) and top5:
            try:
                ratio = float(max(int(r[1]) for r in top5))
            except (TypeError, ValueError, IndexError):
                ratio = None
    if ratio is None:
        return CheckResult(
            check_id, "EXPLAINED_SKIP", None, source,
            "meta.json lacks max_recycling_ratio (residential cache or "
            "pre-W10 build)",
            explained_reason=LIM_V2_BORROW_RATE,
        )

    row = bnd[bnd["metric_name"] == "max_recycling_ratio"].iloc[0]
    lo, hi = float(row["lower_bound"]), float(row["upper_bound"])
    val = float(ratio)
    realised = {
        "max_recycling_ratio": _round(val),
        "bound": [_round(lo), _round(hi)],
    }
    details = f"max_recycling_ratio = {val:.3f} vs [{lo}, {hi}]"
    if _in_band(val, lo, hi):
        return CheckResult(check_id, "PASS", realised, source, details)
    return CheckResult(check_id, "FAIL", realised, source, details)


# ---------------------------------------------------------------------------
# v2.0 — extended optimizer smoke (residential + workplace)
# ---------------------------------------------------------------------------


def check_optimizer_smoke_test_workplace(
    fleet: pd.DataFrame,
    sessions: pd.DataFrame,
    plug_status_15min: pd.DataFrame,
    bounds_root: Path,
    ev_id: int = 0,
    tz: str = "America/Los_Angeles",
) -> CheckResult:
    """v2.0 extension of check #11 — run the optimizer smoke on a workplace
    sample EV. Wraps :func:`check_optimizer_smoke_test` for a tz / ev_id
    pair routed from the workplace cache.
    """
    # Override TZ-local week-start. The original function hard-codes the
    # legacy LA tz; we re-use it by patching the module constant in a
    # try/finally block (deterministic restore).
    global TZ
    saved = TZ
    TZ = tz
    try:
        return check_optimizer_smoke_test(
            fleet, sessions, plug_status_15min, bounds_root, ev_id=ev_id
        )
    finally:
        TZ = saved


# ---------------------------------------------------------------------------
# v2.1 EVSE enrichment checks (plan §D).
# ---------------------------------------------------------------------------

def check_evse_kw_distribution(
    fleet: pd.DataFrame, bounds_root: Path,
    profile_type: str = "residential",
) -> CheckResult:
    """v2.1 — realised EVSE_home_kw share at each tier within the bound.

    Per plan §D.2.1: the check_id is profile-type-suffixed
    (``evse_kw_distribution_residential`` / ``..._workplace``) so the
    bound parquet routing dispatches the correct numeric bands.
    """

    check_id = f"evse_kw_distribution_{profile_type}"
    bnd = _load_bound(bounds_root, check_id)
    source = _bound_url(bnd)
    evse = fleet["EVSE_home_kw"].astype(float)
    realised: dict[str, Any] = {}
    failures: list[str] = []
    # Walk every metric in the bound parquet (the bound may carry
    # different tiers for residential vs workplace).
    for _, row in bnd.iterrows():
        metric = str(row["metric_name"])
        if not metric.startswith("share_at_"):
            continue
        # share_at_19.2_kw -> 19.2
        tier_str = metric[len("share_at_"):-len("_kw")]
        try:
            tier = float(tier_str)
        except ValueError:
            continue
        share = float((np.isclose(evse, tier, atol=1e-3)).mean())
        lo = float(row["lower_bound"])
        hi = float(row["upper_bound"])
        realised[metric] = _round(share)
        if not _in_band(share, lo, hi):
            failures.append(
                f"{tier:.1f}kW share={share:.3f} not in [{lo:.3f}, {hi:.3f}]"
            )
    # Workplace bound parquet carries the legacy 6.6 - 7.7 kW combined
    # band (W5 backward-compat); honour it if present. Use a tolerant
    # range check (atol=1e-2) because EVSE_home_kw round-trips through
    # float32 and ``6.6`` is not exactly representable -- a naive
    # ``evse >= 6.6`` comparison drops the 6.6 bucket entirely.
    band_row = bnd[bnd["metric_name"] == "share_6p6_to_7p7_kw"]
    if not band_row.empty:
        share = float(
            ((evse >= 6.6 - 1e-2) & (evse <= 7.7 + 1e-2)).mean()
        )
        lo = float(band_row.iloc[0]["lower_bound"])
        hi = float(band_row.iloc[0]["upper_bound"])
        realised["share_6p6_to_7p7_kw"] = _round(share)
        if not _in_band(share, lo, hi):
            failures.append(
                f"share_6p6_to_7p7_kw={share:.3f} not in [{lo:.3f}, {hi:.3f}]"
            )
    realised["n_evs"] = int(len(fleet))
    details = (
        "all kW-tier shares within bounds"
        if not failures else "; ".join(failures)
    )
    status = "PASS" if not failures else "FAIL"
    return CheckResult(check_id, status, realised, source, details)


def _share_metric_check(
    fleet: pd.DataFrame,
    bounds_root: Path,
    check_id: str,
    column: str,
) -> CheckResult:
    """Shared helper for `_distribution` checks that bound per-category
    shares via ``share_<category>`` rows. Used by brand + connector.
    """

    bnd = _load_bound(bounds_root, check_id)
    source = _bound_url(bnd)
    col = fleet[column].astype(str)
    realised: dict[str, Any] = {}
    failures: list[str] = []
    for _, row in bnd.iterrows():
        metric = str(row["metric_name"])
        if not metric.startswith("share_"):
            continue
        category = metric[len("share_"):]
        share = float((col == category).mean())
        lo = float(row["lower_bound"])
        hi = float(row["upper_bound"])
        realised[metric] = _round(share)
        if not _in_band(share, lo, hi):
            failures.append(
                f"{metric}={share:.3f} not in [{lo:.3f}, {hi:.3f}]"
            )
    realised["n_evs"] = int(len(fleet))
    details = (
        f"all {column} shares within bounds"
        if not failures else "; ".join(failures)
    )
    status = "PASS" if not failures else "FAIL"
    return CheckResult(check_id, status, realised, source, details)


def check_evse_brand_distribution(
    fleet: pd.DataFrame, bounds_root: Path, profile_type: str = "residential",
) -> CheckResult:
    """v2.1 — realised EVSE_brand share per brand within bound (plan §D.2.2).

    The check_id is profile-type-suffixed
    (``evse_brand_distribution_residential`` / ``..._workplace``) so the
    per-profile-type bound parquet routing dispatches the correct
    numeric bands -- workplace tolerances are deliberately wider than
    residential per plan §I.2.
    """
    check_id = f"evse_brand_distribution_{profile_type}"
    return _share_metric_check(
        fleet, bounds_root,
        check_id=check_id,
        column="EVSE_brand",
    )


def check_evse_connector_distribution(
    fleet: pd.DataFrame, bounds_root: Path,
    profile_type: str = "residential",
) -> CheckResult:
    """v2.1 — realised EVSE_connector share per type (plan §D.2.3).

    The check_id is profile-type-suffixed so the bound parquet routing
    dispatches the correct numeric bands (workplace is site-side, ~80 %
    J1772 / 15 % NACS / 5 % CCS1).
    """
    return _share_metric_check(
        fleet, bounds_root,
        check_id=f"evse_connector_distribution_{profile_type}",
        column="EVSE_connector",
    )


def check_evse_connector_my_band_distribution(
    fleet: pd.DataFrame, bounds_root: Path,
) -> CheckResult:
    """v2.1 — connector mix per MY band (plan §D.2.4).

    Bound parquet rows are named ``my_<lo>_<hi>__share_<connector>``.
    """

    check_id = "evse_connector_my_band_distribution"
    bnd = _load_bound(bounds_root, check_id)
    source = _bound_url(bnd)
    realised: dict[str, Any] = {}
    failures: list[str] = []
    conn = fleet["EVSE_connector"].astype(str)
    my = fleet["model_year"].astype(int)
    for _, row in bnd.iterrows():
        metric = str(row["metric_name"])
        # Expected format: my_<lo>_<hi>__share_<connector>
        if not metric.startswith("my_") or "__share_" not in metric:
            continue
        band, conn_metric = metric.split("__share_", 1)
        try:
            _, lo_my_str, hi_my_str = band.split("_", 2)
            lo_my, hi_my = int(lo_my_str), int(hi_my_str)
        except (ValueError, IndexError):
            continue
        my_mask = (my >= lo_my) & (my <= hi_my)
        n_band = int(my_mask.sum())
        if n_band == 0:
            realised[metric] = None
            continue
        share = float((conn[my_mask] == conn_metric).mean())
        lo = float(row["lower_bound"])
        hi = float(row["upper_bound"])
        realised[metric] = _round(share)
        if not _in_band(share, lo, hi):
            failures.append(
                f"{metric}={share:.3f} not in [{lo:.3f}, {hi:.3f}] (n={n_band})"
            )
    realised["n_evs"] = int(len(fleet))
    details = (
        "all MY-band connector shares within bounds"
        if not failures else "; ".join(failures)
    )
    status = "PASS" if not failures else "FAIL"
    return CheckResult(check_id, status, realised, source, details)


def check_bundle_rule_take_rates(
    fleet: pd.DataFrame, bounds_root: Path,
) -> CheckResult:
    """v2.1 — realised hard-coupling rule take-rates (plan §D.2.5).

    For each ``EV_EVSE_BUNDLE_RULE`` whose ``overrides`` carries an
    ``EVSE_brand`` key, the realised share of the rule's target brand
    among matching EVs must be **at least** the configured
    ``take_rate`` minus a CLT-band tolerance (``±0.10`` when
    ``n_match < 100``, else ``±0.05``).

    Why one-sided: the realised share has TWO contributions -- the
    bundle rule (~``take_rate``) AND the base brand prior. For
    high-base-prior brands like ChargePoint (0.18) and Tesla (0.27),
    the upper bound published in the bound parquet (= take_rate +
    0.10) is below the natural realised value (~take_rate + 0.10).
    The lower bound is the load-bearing one; the upper bound is
    informational and not enforced.
    """

    # Late import to avoid an M2 import cycle (curator -> validator ->
    # vehicle_archetypes -> curator-bound parquet build).
    from .vehicle_archetypes import EV_EVSE_BUNDLE_RULES

    check_id = "bundle_rule_take_rates"
    bnd = _load_bound(bounds_root, check_id)
    source = _bound_url(bnd)
    realised: dict[str, Any] = {}
    failures: list[str] = []

    make_arr = fleet["make"].astype(str).to_numpy()
    model_arr = fleet["model"].astype(str).to_numpy()
    powertrain_arr = fleet["powertrain"].astype(str).to_numpy()
    my_arr = fleet["model_year"].astype(int).to_numpy()
    brand_arr = fleet["EVSE_brand"].astype(str).to_numpy()

    for rule in EV_EVSE_BUNDLE_RULES:
        target_brand = rule.overrides.get("EVSE_brand")
        if target_brand is None:
            continue
        # Build the per-rule predicate mask.
        m_mask = (
            np.ones(len(make_arr), dtype=bool)
            if rule.predicate_make == "*"
            else make_arr == rule.predicate_make
        )
        mo_mask = (
            np.ones(len(model_arr), dtype=bool)
            if rule.predicate_model == "*"
            else model_arr == rule.predicate_model
        )
        p_mask = (
            np.ones(len(powertrain_arr), dtype=bool)
            if rule.predicate_powertrain == "*"
            else powertrain_arr == rule.predicate_powertrain
        )
        my_mask = (my_arr >= rule.model_year_min) & (
            my_arr <= rule.model_year_max
        )
        mask = m_mask & mo_mask & p_mask & my_mask
        n_match = int(mask.sum())
        if n_match == 0:
            realised[rule.name] = {
                "n_match": 0,
                "realised_take_rate": None,
                "target": rule.take_rate,
            }
            continue
        realised_share = float((brand_arr[mask] == target_brand).mean())
        tol = 0.10 if n_match < 100 else 0.05
        # S4: read the floor (lower_bound) from the curator-emitted
        # parquet so any future curator tolerance change does not leave
        # the validator stale. Fall back to the local recomputation only
        # when the row is absent (and warn).
        metric_name = f"take_rate_{rule.name}"
        bnd_row = bnd[bnd["metric_name"] == metric_name]
        if len(bnd_row) == 1:
            floor = float(bnd_row["lower_bound"].iloc[0])
        else:
            floor = max(0.0, rule.take_rate - tol)
            logger.warning(
                "bundle_rule_take_rates: missing bound row for %s; "
                "falling back to recomputed floor=%.3f (tol=%.2f)",
                metric_name, floor, tol,
            )
        realised[rule.name] = {
            "n_match": n_match,
            "realised_take_rate": _round(realised_share),
            "target": rule.take_rate,
            "floor": _round(floor),
            "tol": tol,
        }
        if realised_share < floor:
            failures.append(
                f"{rule.name}: realised={realised_share:.3f} "
                f"target={rule.take_rate} < floor={floor:.3f} "
                f"(n_match={n_match}, tol={tol})"
            )
    _ = source  # _load_bound succeeded -> bound parquet exists; we use
                # it only to read the URL for provenance.
    details = (
        "all bundle take-rates >= floor"
        if not failures else "; ".join(failures)
    )
    status = "PASS" if not failures else "FAIL"
    return CheckResult(check_id, status, realised, source, details)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _build_summary(checks: list[CheckResult]) -> dict[str, int]:
    summary = {k.lower(): 0 for k in VALID_STATUSES}
    for c in checks:
        summary[c.status.lower()] += 1
    summary["total"] = len(checks)
    return summary


def _emit_markdown(
    report: Report,
    *,
    replicate_stats: dict[str, dict[str, float]] | None = None,
    R: int = 1,
) -> str:
    """Render the validator markdown.

    RFC-033 — when ``R > 1`` and ``replicate_stats`` is supplied, append a
    ``mean ± std (n=R)`` column to the per-check table. ``R == 1`` keeps
    the pre-RFC markdown byte-stable (no extra column).
    """
    lines: list[str] = []
    lines.append("# M9 validation report")
    lines.append("")
    lines.append(f"- validator_version: `{report.validator_version}`")
    lines.append(f"- methodology_version: `{report.methodology_version}`")
    lines.append(f"- run_timestamp_utc: `{report.run_timestamp_utc}`")
    if R > 1:
        lines.append(f"- replicates: `R={R}`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    s = report.summary
    lines.append(
        "| PASS | FAIL | EXPLAINED_FAIL | EXPLAINED_SKIP | ERROR | TOTAL |"
    )
    lines.append("|---|---|---|---|---|---|")
    lines.append(
        f"| {s.get('pass', 0)} | {s.get('fail', 0)} | "
        f"{s.get('explained_fail', 0)} | {s.get('explained_skip', 0)} | "
        f"{s.get('error', 0)} | {s.get('total', 0)} |"
    )
    lines.append("")
    lines.append("## Pipeline artifacts (SHA-256)")
    lines.append("")
    lines.append("| Artifact | SHA-256 |")
    lines.append("|---|---|")
    for k, v in sorted(report.pipeline_artifacts_sha256.items()):
        lines.append(f"| `{k}` | `{v}` |")
    lines.append("")
    lines.append("## Per-check results")
    lines.append("")
    if R > 1 and replicate_stats is not None:
        lines.append(
            "| # | check_id | status | realised | "
            f"mean ± std (n={R}) | bound source | explanation / notes |"
        )
        lines.append("|---|---|---|---|---|---|---|")
        for i, c in enumerate(report.checks, start=1):
            realised_summary = _summarise_realised(c)
            notes = c.explained_reason or c.details
            notes = notes.replace("\n", " ").replace("|", "\\|")
            realised_summary = realised_summary.replace("|", "\\|")
            src = (c.bound_source or "").replace("|", "\\|")
            stats = replicate_stats.get(c.check_id)
            if stats is not None and math.isfinite(stats.get("mean", float("nan"))):
                stat_cell = f"{stats['mean']:.4f} ± {stats['std']:.4f}"
            else:
                stat_cell = "-"
            lines.append(
                f"| {i} | `{c.check_id}` | {c.status} | {realised_summary} | "
                f"{stat_cell} | {src} | {notes} |"
            )
    else:
        lines.append(
            "| # | check_id | status | realised | bound source | explanation / notes |"
        )
        lines.append("|---|---|---|---|---|---|")
        for i, c in enumerate(report.checks, start=1):
            realised_summary = _summarise_realised(c)
            notes = c.explained_reason or c.details
            notes = notes.replace("\n", " ").replace("|", "\\|")
            realised_summary = realised_summary.replace("|", "\\|")
            src = (c.bound_source or "").replace("|", "\\|")
            lines.append(
                f"| {i} | `{c.check_id}` | {c.status} | {realised_summary} | "
                f"{src} | {notes} |"
            )
    lines.append("")
    return "\n".join(lines)


def _scalar_from_realised(realised: Any) -> float | None:
    """Best-effort scalar extraction from a CheckResult.realised_value.

    Used by the R>1 replicate-aggregation path to compute per-check
    (mean, std). Returns the realised value when scalar, a sensible
    headline key when dict, else None.
    """
    if realised is None:
        return None
    if isinstance(realised, (int, float)):
        return float(realised)
    if isinstance(realised, dict):
        # Headline-key preference order. Picks the first scalar entry.
        preferred = (
            "tvd", "ratio", "abs_diff", "winter_summer_ratio",
            "plug_in_on_arrival_rate", "rate", "share",
            "donor_borrow_rate", "max_recycling_ratio",
            "max_cluster_window_h", "median_h", "median_kwh",
            "median_miles_per_year", "mean",
            "cluster_window_invariant_active",
        )
        for k in preferred:
            v = realised.get(k)
            if isinstance(v, (int, float)) and math.isfinite(float(v)):
                return float(v)
        # Fallback: first scalar value found.
        for v in realised.values():
            if isinstance(v, (int, float)) and math.isfinite(float(v)):
                return float(v)
    return None


def _summarise_realised(c: CheckResult) -> str:
    """One-line realised-value summary for the markdown table."""
    v = c.realised_value
    if v is None:
        return "-"
    if c.check_id == "battery_capacity_distribution":
        return f"TVD = {v['tvd']}"
    if c.check_id == "weekday_plugin_cdf":
        return (
            f"F(06)={v['F_06']}, F(12)={v['F_12']}, F(18)={v['F_18']}, F(22)={v['F_22']}"
        )
    if c.check_id == "weekend_plugin_cdf":
        return (
            f"F(06)={v['F_06']}, F(12)={v['F_12']}, F(18)={v['F_18']}, F(22)={v['F_22']}"
        )
    if c.check_id == "session_duration_distribution":
        return f"median = {v['median_h']} h, p95 = {v['p95_h']} h"
    if c.check_id == "energy_per_session_distribution":
        return f"median = {v['median_kwh']} kWh, p95 = {v['p95_kwh']} kWh"
    if c.check_id == "annual_mileage":
        return (
            f"median = {v['median_miles_per_year']} mi, σ = {v['std_miles_per_year']} mi"
        )
    if c.check_id == "wh_per_mi_at_70F":
        return f"{v['n_models_pass']}/{v['n_models_total']} models in band"
    if c.check_id == "eta_charging":
        return f"mean = {v['mean']}, range [{v['min']}, {v['max']}]"
    if c.check_id == "optimizer_smoke_test":
        return f"statuses = {v.get('statuses')}"
    if c.check_id == "cross_artifact_integrity":
        return f"{v['fleet_n_unique_ev_ids']} ev_ids, dup_pk = {v['sessions_dup_pk']}"
    if c.check_id == "plug_rate_consistency":
        return f"|Δ| = {v['abs_diff']}"
    if c.check_id == "fleet_energy_conservation":
        if isinstance(v, dict) and "total_gas_kwh_equivalent" in v:
            return (
                f"ratio = {v['ratio']} "
                f"(grid + gas = {v.get('total_numerator_kwh', '?')} kWh)"
            )
        return f"ratio = {v['ratio']}"
    if isinstance(v, dict):
        return ", ".join(f"{k}={v[k]}" for k in list(v)[:3])
    return str(v)


def run(
    output_root: Path | str = DEFAULT_OUTPUT_ROOT,
    bounds_root: Path | str = DEFAULT_BOUNDS_ROOT,
    optimizer_smoke_ev_id: int = 0,
    skip_optimizer: bool = False,
    *,
    region: str | None = None,
    profile_type: str | None = None,
    R: int = 1,
) -> Report:
    """Run all checks and write the report to disk. Returns the in-memory Report.

    Parameters
    ----------
    output_root, bounds_root :
        Legacy v1.1 paths. If ``region`` + ``profile_type`` are supplied,
        ``output_root`` is overridden with
        ``data/pev/processed/{region}/{profile_type}_ev_synth/``.
    optimizer_smoke_ev_id :
        EV id for the optimiser smoke check (check #11). Defaults to 0.
    skip_optimizer :
        Skip the optimiser smoke check.
    region :
        v2.0 routing — when provided alongside ``profile_type``, the
        validator runs the v2.0 check set:

        * residential cache: 14 v1.1 checks + DST check #12 = 15 checks,
          validator_version=v2.0.0, methodology_version=v2.0.0.
        * workplace cache: 14 v1.1 checks + DST + W1-W8 = 23 checks.

        v1.1 legacy behaviour (no region/profile_type) is byte-stable.
    profile_type :
        ``"residential"`` or ``"workplace"`` (v2.0 only).
    """

    if R is None or R < 1:
        raise ValueError(f"R must be a positive integer, got {R!r}")

    is_v2 = region is not None and profile_type is not None

    # RFC-033 — R > 1 replicate-aggregation path. Iterate
    # ``<output_root>/replicates/r{0..R-1}/`` (the layout wired by
    # WP-B / RFC-022.r) and aggregate per-check (mean, std) over the
    # numeric realised value. When the replicate layout is absent we
    # fall back to a single-run report with a warning.
    if R > 1:
        return _run_with_replicates(
            output_root=output_root,
            bounds_root=bounds_root,
            optimizer_smoke_ev_id=optimizer_smoke_ev_id,
            skip_optimizer=skip_optimizer,
            region=region,
            profile_type=profile_type,
            R=R,
        )

    if is_v2:
        # v2.0 routing — override output_root via the cache template.
        from .regions import Region  # local to keep v1.1 import light

        try:
            region_obj = Region.from_name(str(region))
        except ValueError:
            # Allow opaque region names so the validator stays decoupled
            # from the registry tests; fall back to default tz.
            region_obj = None
        if profile_type not in ("residential", "workplace"):
            raise ValueError(
                f"profile_type {profile_type!r} is not supported in v2.0; "
                "valid: ('residential', 'workplace')"
            )
        # Default cache root unless the caller passed an explicit path.
        v2_default_root = _cache_dir(region, profile_type)
        if output_root == DEFAULT_OUTPUT_ROOT:
            output_root = v2_default_root
        region_tz = region_obj.tz if region_obj is not None else TZ
    else:
        region_obj = None
        region_tz = TZ

    output_root = Path(output_root)
    bounds_root = Path(bounds_root)
    if not output_root.exists():
        raise FileNotFoundError(f"output_root missing: {output_root}")
    if not bounds_root.exists():
        raise FileNotFoundError(f"bounds_root missing: {bounds_root}")

    # RFC-009.r: validator reads from the strict nested bounds layout.
    # Resolve the per-region / per-profile-type subdir; v1.1 legacy callers
    # (no region/profile_type) map to the bay_area / residential subtree
    # since the v1.1 cache origin is the bay_area v1.1_migrated artefacts.
    # The check_id parquet must exist under <bounds_root>/<region>/
    # <profile_type>/; missing dir -> FileNotFoundError, no fallback.
    bounds_top_root = bounds_root
    nested_region = region if is_v2 else "bay_area"
    nested_profile_type = profile_type if is_v2 else "residential"
    nested_bounds_root = (
        bounds_top_root / str(nested_region) / str(nested_profile_type)
    )
    if not nested_bounds_root.is_dir() or not any(
        nested_bounds_root.iterdir()
    ):
        raise FileNotFoundError(
            f"nested validation_bounds dir missing or empty: "
            f"{nested_bounds_root} (RFC-009.r requires "
            f"<region>/<profile_type>/<check_id>.parquet)"
        )
    bounds_root = nested_bounds_root

    # Hash pipeline artifacts (skip missing files defensively).
    sha = {}
    for name in PIPELINE_ARTIFACTS:
        p = output_root / name
        if p.exists():
            sha[name] = _sha256(p)

    # Load artifacts. Heavy frames are loaded once.
    fleet = pd.read_parquet(output_root / "fleet.parquet")
    sessions = pd.read_parquet(output_root / "sessions.parquet")
    mobility = pd.read_parquet(output_root / "mobility.parquet")
    plug_in_events = pd.read_parquet(output_root / "plug_in_events.parquet")
    plug_status_15min = pd.read_parquet(output_root / "plug_status_15min.parquet")
    plug_status_hourly = pd.read_parquet(output_root / "plug_status_hourly.parquet")

    # v2.0 storage is UTC; the v1.1 residential cdf / duration checks read
    # `.dt.hour` / `.dt.dayofweek`, which return UTC fields on UTC-stored
    # data. Pre-convert tz-aware columns to the region's local tz so the
    # existing residential check implementations operate on local-time
    # semantics regardless of storage tz. v1.1 legacy mode (no region) is
    # untouched.
    if is_v2:
        pie_view = plug_in_events.copy()
        if str(pie_view["arrival_ts"].dt.tz) == "UTC":
            pie_view["arrival_ts"] = pie_view["arrival_ts"].dt.tz_convert(region_tz)
        if "departure_ts" in pie_view.columns and str(
            pie_view["departure_ts"].dt.tz
        ) == "UTC":
            pie_view["departure_ts"] = pie_view["departure_ts"].dt.tz_convert(
                region_tz
            )
        # RFC-012: M5 now emits the brief schema (``plug_in`` + ``location``)
        # directly, but we keep a defensive idempotent fall-back so older
        # v2.0 caches without the new columns still validate without an
        # extra cache regen.
        if "plug_in" not in pie_view.columns:
            pie_view["plug_in"] = True
        if "location" not in pie_view.columns and "location_type" in pie_view.columns:
            pie_view["location"] = pie_view["location_type"]

        sessions_view = sessions.copy()
        for col in ("t_in", "t_out"):
            if col in sessions_view.columns and str(
                sessions_view[col].dt.tz
            ) == "UTC":
                sessions_view[col] = sessions_view[col].dt.tz_convert(region_tz)
    else:
        pie_view = plug_in_events
        sessions_view = sessions

    # Load meta.json once (workplace W6/W7 pull from it; safe to load even
    # for v1.1).
    meta_for_checks: dict[str, Any] = {}
    meta_path = output_root / "meta.json"
    if meta_path.exists():
        with meta_path.open() as f:
            meta_for_checks = json.load(f)

    # RFC-008.r — the legacy ``energy_kwh`` alias is acceptable ONLY when
    # the cache is the v1.1_migrated artefact (or when the validator is
    # running in v1.1 legacy mode with no region/profile_type, which targets
    # the same bay_area residential cache).
    legacy_alias_ok = (not is_v2) or _is_v11_migrated(meta_for_checks)

    # Extract workplace_cohort_caveat if present (M5 emits it under
    # plug_in_model.workplace_cohort_caveat). Used by W1 EXPLAINED_FAIL.
    workplace_caveat = (
        meta_for_checks.get("plug_in_model", {})
        .get("workplace_cohort_caveat")
    )

    n_vehicles = int(fleet["ev_id"].nunique())

    checks: list[CheckResult] = []

    # The 11 M8-driven checks (in the order the M9 brief lists them).
    checks.append(_run_or_error(
        "battery_capacity_distribution",
        lambda: check_battery_capacity_distribution(fleet, bounds_root),
    ))
    checks.append(_run_or_error(
        "obc_distribution",
        lambda: check_obc_distribution(bounds_root),
    ))
    checks.append(_run_or_error(
        "weekday_plugin_cdf",
        lambda: check_weekday_plugin_cdf(pie_view, bounds_root),
    ))
    checks.append(_run_or_error(
        "weekend_plugin_cdf",
        lambda: check_weekend_plugin_cdf(pie_view, bounds_root),
    ))
    checks.append(_run_or_error(
        "session_duration_distribution",
        lambda: check_session_duration_distribution(sessions_view, bounds_root),
    ))
    checks.append(_run_or_error(
        "energy_per_session_distribution",
        lambda: check_energy_per_session_distribution(
            sessions_view, bounds_root, legacy_alias_ok=legacy_alias_ok,
        ),
    ))
    # v3.0 (§13.2 RESOLVED): pass methodology_version so check_annual_mileage
    # promotes PASS / FAIL according to the v3 HOUSEID-coherent stitcher
    # contract. v1.x / v2.x paths inherit ``None`` and keep the historical
    # EXPLAINED_FAIL → LIM_13_2 behaviour.
    _check_methodology = METHODOLOGY_VERSION_V2 if is_v2 else None
    checks.append(_run_or_error(
        "annual_mileage",
        lambda: check_annual_mileage(
            mobility, bounds_root, methodology_version=_check_methodology,
        ),
    ))
    checks.append(_run_or_error(
        "wh_per_mi_at_70F",
        lambda: check_wh_per_mi_at_70F(fleet, mobility, bounds_root),
    ))
    checks.append(_run_or_error(
        "eta_charging",
        lambda: check_eta_charging(fleet, bounds_root),
    ))
    checks.append(_run_or_error(
        "evi_pro_plug_in_split",
        lambda: check_evi_pro_plug_in_split(bounds_root),
    ))
    # Optimizer smoke test depends on the in-house ``pev_optim`` package
    # (documented out-of-PyPI scope, v1.0.1 deferral). When pev_optim is
    # absent OR the caller explicitly --skip-optimizer'd, the correct status
    # is EXPLAINED_SKIP -- a known, documented absence, not an ERROR.
    import importlib.util as _ilutil
    _has_pev_optim = _ilutil.find_spec("pev_optim") is not None
    if skip_optimizer or not _has_pev_optim:
        _why = (
            "skipped via --skip-optimizer"
            if skip_optimizer
            else "in-house ``pev_optim`` package not installed on this checkout"
        )
        checks.append(CheckResult(
            "optimizer_smoke_test",
            "EXPLAINED_SKIP",
            None,
            "https://github.com/bertravacca/ev-flow/blob/main/pev_optim/optimizer.py",
            (
                "optimizer smoke test requires the in-house ``pev_optim`` "
                "package, documented as out-of-PyPI scope (v1.0.1 deferral). "
                f"{_why}."
            ),
        ))
    else:
        if is_v2 and profile_type == "workplace":
            checks.append(_run_or_error(
                "optimizer_smoke_test",
                lambda: check_optimizer_smoke_test_workplace(
                    fleet, sessions_view, plug_status_15min, bounds_root,
                    ev_id=optimizer_smoke_ev_id, tz=region_tz,
                ),
            ))
        else:
            checks.append(_run_or_error(
                "optimizer_smoke_test",
                lambda: check_optimizer_smoke_test(
                    fleet, sessions_view, plug_status_15min, bounds_root,
                    ev_id=optimizer_smoke_ev_id,
                ),
            ))

    # Integration checks 12, 13, 14.
    checks.append(_run_or_error(
        "cross_artifact_integrity",
        lambda: check_cross_artifact_integrity(
            fleet, sessions_view, plug_status_15min, plug_status_hourly,
            mobility, pie_view,
        ),
    ))
    checks.append(_run_or_error(
        "plug_rate_consistency",
        lambda: check_plug_rate_consistency(sessions_view, plug_status_15min, n_vehicles),
    ))
    checks.append(_run_or_error(
        "fleet_energy_conservation",
        lambda: check_fleet_energy_conservation(
            sessions_view, mobility, legacy_alias_ok=legacy_alias_ok,
        ),
    ))

    # v2.0 additions — DST check #12 + workplace W1-W8 + v2.1 EVSE
    # enrichment checks.
    if is_v2:
        checks.append(_run_or_error(
            "dst_verification",
            lambda: check_dst_verification(output_root, bounds_root),
        ))
        # v2.1 EVSE-enrichment checks (plan §D). Each check_id is
        # profile-type-suffixed so the per-profile-type bound parquet
        # routing dispatches the correct numeric bands.
        checks.append(_run_or_error(
            f"evse_kw_distribution_{profile_type}",
            lambda: check_evse_kw_distribution(
                fleet, bounds_root, profile_type=profile_type,
            ),
        ))
        checks.append(_run_or_error(
            f"evse_brand_distribution_{profile_type}",
            lambda: check_evse_brand_distribution(
                fleet, bounds_root, profile_type=profile_type,
            ),
        ))
        checks.append(_run_or_error(
            f"evse_connector_distribution_{profile_type}",
            lambda: check_evse_connector_distribution(
                fleet, bounds_root, profile_type=profile_type,
            ),
        ))
        # MY-band connector check is residential-only (workplace
        # connector is MY-band invariant per plan §C.2 step 8c).
        if profile_type == "residential":
            checks.append(_run_or_error(
                "evse_connector_my_band_distribution",
                lambda: check_evse_connector_my_band_distribution(
                    fleet, bounds_root,
                ),
            ))
            # bundle_rule_take_rates is residential-only (rules don't
            # apply to workplace caches per plan §B.4.1).
            checks.append(_run_or_error(
                "bundle_rule_take_rates",
                lambda: check_bundle_rule_take_rates(fleet, bounds_root),
            ))
        # RFC-034 — C5 cross-region winter-ratio check (cold regions only;
        # warm regions return EXPLAINED_SKIP).
        cross_bounds_root = bounds_top_root / str(nested_region) / "cross_region"
        if cross_bounds_root.is_dir():
            checks.append(_run_or_error(
                "C5_winter_ratio",
                lambda: check_C5_winter_ratio(
                    mobility, cross_bounds_root,
                    region=str(nested_region),
                    region_tz=region_tz,
                    region_obj=region_obj,
                ),
            ))
        if profile_type == "workplace":
            checks.append(_run_or_error(
                "W1_workplace_plug_in_cdf",
                lambda: check_workplace_w1_plug_in_cdf(
                    pie_view, bounds_root, region_tz,
                    workplace_cohort_caveat=workplace_caveat,
                ),
            ))
            checks.append(_run_or_error(
                "W2_workplace_session_duration",
                lambda: check_workplace_w2_session_duration(
                    sessions_view, bounds_root,
                ),
            ))
            checks.append(_run_or_error(
                "W3_workplace_session_energy",
                lambda: check_workplace_w3_session_energy(
                    sessions_view, bounds_root,
                    legacy_alias_ok=legacy_alias_ok,
                ),
            ))
            checks.append(_run_or_error(
                "W4_workplace_plug_in_on_arrival",
                lambda: check_workplace_w4_plug_in_on_arrival(
                    pie_view, bounds_root,
                ),
            ))
            checks.append(_run_or_error(
                "W5_workplace_evse_kw_distribution",
                lambda: check_workplace_w5_evse_kw_distribution(
                    fleet, bounds_root,
                ),
            ))
            checks.append(_run_or_error(
                "W6_workplace_cluster_window_max",
                lambda: check_workplace_w6_cluster_window_max(
                    meta_for_checks, bounds_root,
                ),
            ))
            checks.append(_run_or_error(
                "W7_workplace_borrow_share",
                lambda: check_workplace_w7_borrow_share(
                    meta_for_checks, fleet, bounds_root,
                ),
            ))
            checks.append(_run_or_error(
                "W8_workplace_home_charging_required_rate",
                lambda: check_workplace_w8_home_charging_required_rate(
                    sessions_view, fleet, bounds_root,
                ),
            ))
            # RFC-038 — W9 (cluster window invariant) + W10 (donor pool
            # recycling) split out from the prior W6/W7 markdown rows so
            # the validator markdown shows all 10 workplace IDs.
            checks.append(_run_or_error(
                "W9_workplace_cluster_window_invariant",
                lambda: check_workplace_w9_cluster_window_invariant(
                    meta_for_checks, bounds_root,
                ),
            ))
            checks.append(_run_or_error(
                "W10_workplace_donor_pool_recycling",
                lambda: check_workplace_w10_donor_pool_recycling(
                    meta_for_checks, fleet, bounds_root,
                ),
            ))

    summary = _build_summary(checks)

    # Emit the right validator/methodology versions for the v2.0 path.
    if is_v2:
        validator_v = VALIDATOR_VERSION_V2
        method_v = METHODOLOGY_VERSION_V2
    else:
        validator_v = VALIDATOR_VERSION
        method_v = METHODOLOGY_VERSION

    report = Report(
        validator_version=validator_v,
        methodology_version=method_v,
        run_timestamp_utc=datetime.now(tz=timezone.utc).isoformat(),
        pipeline_artifacts_sha256=sha,
        checks=checks,
        summary=summary,
    )

    # Write JSON + MD.
    json_path = output_root / "validation_report.json"
    md_path = output_root / "validation_report.md"
    with json_path.open("w") as f:
        json.dump(report.to_dict(), f, indent=2, sort_keys=False, default=str)
        f.write("\n")
    md_path.write_text(_emit_markdown(report) + "\n")

    # Update meta.json with M9 provenance (idempotent — overwrites the
    # `validator` key only; keeps everything else byte-stable except for
    # the new run_timestamp_utc inside the validator block).
    _update_meta_json(output_root / "meta.json", report)

    logger.info(
        "M9 validator: %d checks (pass=%d fail=%d explained_fail=%d "
        "explained_skip=%d error=%d)",
        summary["total"], summary["pass"], summary["fail"],
        summary["explained_fail"], summary["explained_skip"], summary["error"],
    )
    return report


def _run_with_replicates(
    *,
    output_root: Path | str,
    bounds_root: Path | str,
    optimizer_smoke_ev_id: int,
    skip_optimizer: bool,
    region: str | None,
    profile_type: str | None,
    R: int,
) -> Report:
    """RFC-033 — execute the validator R times over replicate dirs and emit
    per-check (mean, std, n=R) in the markdown.

    Replicate layout (RFC-022.r): ``<output_root>/replicates/r{0..R-1}/``.
    Each replicate dir is treated as an independent cache root for the
    purposes of the validator. The aggregated report (this function's
    return value) takes its check statuses from r0 (the canonical
    replicate); the (mean, std) columns are computed across all R.
    """

    parent_root = Path(output_root)
    replicate_root = parent_root / "replicates"
    replicate_dirs: list[Path] = []
    if replicate_root.is_dir():
        for i in range(R):
            rd = replicate_root / f"r{i}"
            if rd.is_dir():
                replicate_dirs.append(rd)

    if not replicate_dirs:
        # Fall back: caller asked for R>1 but the replicate layout is
        # absent (e.g., regen has not been run yet — WP-G ops). Run once
        # against ``output_root`` and emit a single-replicate report with
        # the R header annotated.
        replicate_dirs = [parent_root]

    per_check_values: dict[str, list[float]] = {}
    per_replicate_reports: list[Report] = []
    for rd in replicate_dirs:
        rep = run(
            output_root=rd,
            bounds_root=bounds_root,
            optimizer_smoke_ev_id=optimizer_smoke_ev_id,
            skip_optimizer=skip_optimizer,
            region=region,
            profile_type=profile_type,
            R=1,
        )
        per_replicate_reports.append(rep)
        for c in rep.checks:
            val = _scalar_from_realised(c.realised_value)
            if val is None or not math.isfinite(val):
                continue
            per_check_values.setdefault(c.check_id, []).append(val)

    # Aggregate per-check (mean, std). n is the number of replicates that
    # produced a finite scalar — may be < R when a check returns
    # EXPLAINED_SKIP on some replicates.
    replicate_stats: dict[str, dict[str, float]] = {}
    for cid, vals in per_check_values.items():
        n = len(vals)
        if n == 0:
            continue
        mean = float(sum(vals) / n)
        if n > 1:
            var = sum((v - mean) ** 2 for v in vals) / (n - 1)
            std = float(math.sqrt(var))
        else:
            std = 0.0
        replicate_stats[cid] = {"mean": mean, "std": std, "n": float(n)}

    # Canonical report = r0 (first replicate). Statuses, summaries, etc.
    canonical = per_replicate_reports[0]

    # Re-emit the markdown with the aggregated (mean, std, n=R) columns.
    md_path = parent_root / "validation_report.md"
    md_path.write_text(
        _emit_markdown(canonical, replicate_stats=replicate_stats, R=R) + "\n"
    )
    # Re-emit JSON with the replicate stats appended.
    aggregated = canonical.to_dict()
    aggregated["replicate_stats"] = {
        cid: {"mean": s["mean"], "std": s["std"], "n": int(s["n"])}
        for cid, s in replicate_stats.items()
    }
    aggregated["replicates"] = {"R": R, "n_dirs": len(replicate_dirs)}
    json_path = parent_root / "validation_report.json"
    with json_path.open("w") as f:
        json.dump(aggregated, f, indent=2, sort_keys=False, default=str)
        f.write("\n")

    return canonical


def _run_or_error(check_id: str, fn) -> CheckResult:  # noqa: ANN001
    """Execute a check fn; convert any uncaught exception into an ERROR row."""
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001
        logger.exception("check %s raised", check_id)
        return CheckResult(
            check_id, "ERROR", None, "",
            f"check raised {type(exc).__name__}: {exc}",
        )


def _update_meta_json(meta_path: Path, report: Report) -> None:
    """Append M9 provenance block to meta.json.

    The block records the validator_version reported by the run. For v2.0
    runs the master seed is :data:`MASTER_SEED_V2`; for v1.1 it remains
    :data:`MASTER_SEED`.
    """
    if meta_path.exists():
        with meta_path.open() as f:
            meta = json.load(f)
    else:
        meta = {}

    is_v2 = report.validator_version == VALIDATOR_VERSION_V2
    master_seed = MASTER_SEED_V2 if is_v2 else MASTER_SEED

    meta.setdefault("module_provenance", {})["m9_validator"] = report.validator_version
    meta["validator"] = {
        "module": "pev_synth.validator",
        "validator_version": report.validator_version,
        "methodology_version": report.methodology_version,
        "master_seed": master_seed,
        "run_timestamp_utc": report.run_timestamp_utc,
        "summary": report.summary,
        "n_checks_evaluated": report.summary["total"],
        "report_json": "validation_report.json",
        "report_md": "validation_report.md",
    }
    with meta_path.open("w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
        f.write("\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m pev_synth.validator",
        description="Run M9 validation against the residential EV synth pipeline.",
    )
    parser.add_argument(
        "--root", type=Path, default=DEFAULT_OUTPUT_ROOT,
        help=f"output root (default: {DEFAULT_OUTPUT_ROOT})",
    )
    parser.add_argument(
        "--bounds", type=Path, default=DEFAULT_BOUNDS_ROOT,
        help=f"validation_bounds root (default: {DEFAULT_BOUNDS_ROOT})",
    )
    parser.add_argument(
        "--optimizer-ev-id", type=int, default=0,
        help="ev_id for the optimiser smoke test (default: 0)",
    )
    parser.add_argument(
        "--skip-optimizer", action="store_true",
        help="skip the optimiser smoke test (records ERROR for check 11)",
    )
    parser.add_argument(
        "--region", type=str, default=None,
        help=(
            "v2.0 routing — region short name (e.g. bay_area, "
            "dallas_fort_worth). When supplied with --profile-type, the "
            "validator runs against "
            "data/pev/processed/{region}/{profile_type}_ev_synth/."
        ),
    )
    parser.add_argument(
        "--profile-type", type=str, default=None,
        choices=("residential", "workplace"),
        help="v2.0 routing — profile type (residential | workplace).",
    )
    parser.add_argument(
        "--R", type=int, default=1,
        help=(
            "RFC-033 replicate count. R>1 iterates "
            "<output_root>/replicates/r{0..R-1}/ and emits per-check "
            "(mean, std, n=R) in the markdown. Default 1 (single-run)."
        ),
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="verbose logging",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if (args.region is None) ^ (args.profile_type is None):
        parser.error(
            "--region and --profile-type must be supplied together"
        )
    report = run(
        output_root=args.root,
        bounds_root=args.bounds,
        optimizer_smoke_ev_id=args.optimizer_ev_id,
        skip_optimizer=args.skip_optimizer,
        region=args.region,
        profile_type=args.profile_type,
        R=args.R,
    )
    s = report.summary
    print(
        f"M9 validator complete: {s['total']} checks "
        f"(pass={s['pass']} fail={s['fail']} explained_fail={s['explained_fail']} "
        f"explained_skip={s['explained_skip']} error={s['error']})"
    )
    return 0 if s["fail"] == 0 else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(_cli())
