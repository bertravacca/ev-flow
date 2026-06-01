"""M5 — `pev_synth.plug_in_model` (v2.0).

Three-layer Bernoulli plug-in behaviour model for residential AND
workplace synthetic fleets (Phase 10.3 revised plan §4.4 + §5.5).

v2.0 changes relative to v1.1:

- Accepts a `Region` object and a `profile_type ∈ {residential,
  workplace}`. Cluster centres, BIC sweep, β priors and Bernoulli
  iteration all route on `profile_type`.
- Storage timezone is UTC. Cluster windows are evaluated in
  `region.tz` local time (the modal hour-of-day is a local-clock
  concept; UTC conversion is purely for storage).
- **Workplace path (NEW).** Loads
  `data/pev/raw/evwatts_cohorts/workplace_subset.parquet` (Dev C),
  fits GMMs over (plug_in_hr_med, plug_out_hr_med, dur_h_med) per
  vehicle for K ∈ {2, 3, 4, 5}, applies the W9 invariant (every
  selected cluster's 95% start-time window ≤ 6 h), and picks the
  lowest-BIC non-rejected K. If BIC is ambiguous (rel-gap < 1%) the
  default is K=3.
- **β priors per profile_type:**
  - residential keeps v1.1: β₂=2.5, β₃=1.5, β₄=0.6, β₅=−0.2, β₁=1.8;
  - workplace per revised plan §4.4.4: β₂=2.8 (HIGHER than
    residential 2.5 — S-2 sign flip), β₃=1.5 (LOWERED from draft
    1.8), β₄=0.3, β₅=−3.0 (strong weekend dampening), β₁=1.2.
- For workplace, plug-in fires on trips with `dest_type == "work"`
  (derived from WHYTRP1S==10 upstream).

Honest lineage (NOT SPEECh / NOT Powell):
- Munkhammar et al. 2015, *Applied Energy* 142:135-143 — Bernoulli
  per-arrival plug-in at the home location.
- Quirós-Tortós, Navarro-Espinosa, Ochoa & Butler 2018 *PSCC*
  (IEEE doc 8442988) — empirical residential weekday/weekend
  start-time PDFs.
- Lee, Li & Low 2019 *ACM e-Energy* (ACN-Data) — GMM clustering over
  (start_time, duration, energy) per user. Canonical workplace
  methodology reference.
- Pritchard, Borlaug, Yang & Gonder 2023 *EVS-36* / NREL-CP-5400-85902
  — workplace L2 venue statistics (Q2 anchor).
- Kreft, Brudermueller, Fleisch & Staake 2024 *Applied Energy*
  370:123544 — workplace β₁ qualitative anchor (NOT "Köhler 2024").

Workplace cohort caveat (documented):
The workplace EVWatts cohort (N=105) is a daytime-charging-fleet-mix
calibrated by Dev C; its plug-in median (~12:00 LT) is approximately
3 hours later than the literature-canonical workplace median of
~09:00. v2.0 cluster centres are fit from this cohort and will skew
midday. W1-W4 validator checks flag this divergence as
EXPLAINED_FAIL — not a bug in M5.

Methodology version: v2.0.0.  Master seed: 20260520.  Module offset: +5.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Final, Literal

import numpy as np
import pandas as pd
import sklearn
from sklearn.mixture import GaussianMixture

# ---------------------------------------------------------------------------
# Orchestrator-resolved contracts (Phase 10.4 PM brief).
# ---------------------------------------------------------------------------
from ._seeds import (
    MASTER_SEED as _MASTER_SEED,
)
from ._seeds import (
    METHODOLOGY_VERSION as _METHODOLOGY_VERSION,
)
from ._seeds import (
    module_offset as _module_offset,
)
from .regions import REGIONS, Region

METHODOLOGY_VERSION: Final[str] = _METHODOLOGY_VERSION
MASTER_SEED_DEFAULT: Final[int] = _MASTER_SEED
MODULE_OFFSET: Final[int] = _module_offset("plug_in_model")  # +5

# Sub-seed namespaces, offset to avoid collision with other modules' streams.
# RFC-001/-016: sourced from the central registry; M5 offsets must remain
# pairwise ≥ N_EV_MAX apart from M6 :data:`SOC0_SUBSEED_OFFSET`.
from ._seeds import (  # noqa: E402
    ARRIVAL_JITTER_SUBSEED_OFFSET as _ARRIVAL_JITTER_SUBSEED_OFFSET,
)
from ._seeds import (  # noqa: E402
    BERNOULLI_SUBSEED_OFFSET as _BERNOULLI_SUBSEED_OFFSET,
)
from ._seeds import (  # noqa: E402
    CLUSTER_ASSIGN_SUBSEED_OFFSET as _CLUSTER_ASSIGN_SUBSEED_OFFSET,
)

# Storage tz (canonical v2.0).
STORAGE_TZ: Final[str] = "UTC"

# BIC sweep ranges per profile_type.
K_SWEEP_RESIDENTIAL: Final[tuple[int, ...]] = (3, 4, 6)
K_SWEEP_WORKPLACE: Final[tuple[int, ...]] = (2, 3, 4, 5)

# Default K when BIC is ambiguous (rel-gap < BIC_AMBIGUOUS_REL_GAP).
DEFAULT_K_RESIDENTIAL: Final[int] = 3
DEFAULT_K_WORKPLACE: Final[int] = 3

# Ambiguity thresholds.
BIC_AMBIGUOUS_REL_GAP: Final[float] = 0.01  # 1 %

# W9 invariant: every selected GMM cluster has 95% start-time window ≤ 6 h.
WORKPLACE_CLUSTER_WINDOW_MAX_H: Final[float] = 6.0

# Workplace cohort size threshold (plan §4.4.3): below this, cap K ≤ 4.
WORKPLACE_MIN_COHORT_FOR_K5: Final[int] = 150

# RFC-036: tighter K-cap when cohort is very small (n < 90 → K ≤ 3).
WORKPLACE_MIN_COHORT_FOR_K4: Final[int] = 90

# RFC-037: workplace commuter-window band for fit-centroid hours. If
# ≥ 2 fit centroids fall outside this band, BIC's K is overridden with
# the commuter-prior K=3 anchor.
EVWATTS_CENTROID_BAND_LO_H: Final[float] = 7.0
EVWATTS_CENTROID_BAND_HI_H: Final[float] = 10.5
EVWATTS_CENTROID_OUT_OF_BAND_THRESHOLD: Final[int] = 2
EVWATTS_CENTROID_DRIFT_OVERRIDE_K: Final[int] = 3
BIC_OVERRIDE_REASON_CENTROID_DRIFT: Final[str] = "evwatts_centroid_drift"


def _admissible_K(n_cohort: int) -> set[int]:
    """Return the admissible BIC sweep K-set given the cohort size.

    RFC-036 small-cohort guard:

    * ``n < 90``        → ``{2, 3}``       (K ≤ 3)
    * ``90 ≤ n < 150``  → ``{2, 3, 4}``    (K ≤ 4)
    * ``n ≥ 150``       → ``{2, 3, 4, 5}`` (full workplace sweep)

    The cap is intended for workplace cohorts (EVWatts subset). Callers
    on residential cohorts may compute admissibility separately; the
    function takes only the cohort size and is profile-agnostic.
    """
    n = int(n_cohort)
    if n < WORKPLACE_MIN_COHORT_FOR_K4:
        return {2, 3}
    if n < WORKPLACE_MIN_COHORT_FOR_K5:
        return {2, 3, 4}
    return {2, 3, 4, 5}


# ---------------------------------------------------------------------------
# β coefficients per profile_type.
# ---------------------------------------------------------------------------

# β₀(z) per cluster.  Clusters are numbered 1..K (1 = earliest centre after
# time-sort).  All entries `[prior, unfit]`.
BETA_0_BY_K_RESIDENTIAL: Final[dict[int, tuple[float, ...]]] = {
    3: (-1.0, -0.5, -2.5),   # midday, evening, overnight (v1.1)
    4: (-1.0, -0.5, -2.5, -2.5),
    6: (-0.8, -0.4, -0.6, -1.4, -2.5, -2.8),
}

# Workplace β₀(z) per K — plan §4.4.4 anchors the K=3 vector to
# (-0.9, -0.2, -1.1).  For K=2 / K=4 / K=5 we extrapolate so that the
# σ() anchor (in-cluster, dwell=8h, SoC=0.55) lands in [0.85, 0.99]
# per acceptance criterion #4.  Values pinned to make σ() ∈ [0.85, 0.99]
# at every cluster z (see anchor check in tests).
BETA_0_BY_K_WORKPLACE: Final[dict[int, tuple[float, ...]]] = {
    2: (-0.5, -0.9),
    3: (-0.9, -0.2, -1.1),   # plan §4.4.4 canonical
    4: (-0.9, -0.2, -1.1, -1.5),
    5: (-0.9, -0.2, -1.1, -1.3, -1.8),
}

# Residential β coefficients (v1.1 carry-forward).
BETA_RESIDENTIAL: Final[dict[str, float]] = {
    "beta_1": 1.8,   # low-SoC nudge
    "beta_2": 2.5,   # home preference (Munkhammar 2015 Table 1)
    "beta_3": 1.5,   # in-preferred-window effect
    "beta_4": 0.6,   # β₄ · log(max(dwell_h, 0.1))
    "beta_5": -0.2,  # weekend dampening
}

# Workplace β coefficients (revised plan §4.4.4).
BETA_WORKPLACE: Final[dict[str, float]] = {
    "beta_1": 1.2,   # Kreft 2024 anchor — habit dominates SoC
    "beta_2": 2.8,   # FLIPPED higher than residential per S-2
    "beta_3": 1.5,   # lowered from 1.8 draft so total stays in [0.85, 0.99]
    "beta_4": 0.3,   # short dwell sensitivity (most workplace = 6-9 h)
    "beta_5": -3.0,  # very strong weekend dampening
}

# Minimum dwell (hours) for the log term.
DWELL_LOG_FLOOR_H: Final[float] = 0.1


# ---------------------------------------------------------------------------
# Cluster mixing weights π by profile_type and K (priors).
# ---------------------------------------------------------------------------

# Residential mixing weights (v1.1 carry-forward).
PI_BASE_RESIDENTIAL: Final[dict[int, tuple[float, ...]]] = {
    3: (0.18, 0.62, 0.20),
    4: (0.16, 0.55, 0.18, 0.11),
    6: (0.10, 0.30, 0.25, 0.15, 0.13, 0.07),
}

# Workplace mixing weights (plan §4.4.3).  Time-sorted: early / standard /
# late commuter (K=3 default).  K=2/4/5 follow the same anchor distribution.
PI_BASE_WORKPLACE: Final[dict[int, tuple[float, ...]]] = {
    2: (0.40, 0.60),
    3: (0.25, 0.55, 0.20),
    4: (0.20, 0.50, 0.20, 0.10),
    5: (0.20, 0.40, 0.20, 0.12, 0.08),
}

# Income / garage modulation deltas (residential only).
INCOME_T4_OVERNIGHT_DELTA: Final[float] = +0.10
GARAGE_SHARED_OR_CURB_MIDDAY_DELTA: Final[float] = +0.10
GARAGE_SHARED_OR_CURB_OVERNIGHT_DELTA: Final[float] = -0.10


# ---------------------------------------------------------------------------
# Cluster-window centres (hours of day) — fixed PRIORS used when the BIC
# fit is degenerate.  When the BIC fit succeeds we OVERRIDE these with the
# fit centroids' first dimension.
# ---------------------------------------------------------------------------

# Residential (Quirós-Tortós 2018 weekday modes).
WINDOW_CENTRES_RESIDENTIAL: Final[dict[int, tuple[float, ...]]] = {
    3: (12.5, 18.5, 23.5),
    4: (12.5, 18.5, 23.5, 24.0),
    6: (10.5, 17.5, 19.5, 21.5, 23.5, 3.5),
}

# Workplace priors (plan §4.4.3 default, K=3).  Note: these are
# literature-anchored centres (07:30 / 08:45 / 10:00).  At fit time we
# typically override with EVWatts-cohort GMM centroids, which skew
# later (~12:00).  The override is honestly documented in
# `meta.json.workplace.cluster_centres_source`.
WINDOW_CENTRES_WORKPLACE_PRIOR: Final[dict[int, tuple[float, ...]]] = {
    2: (8.0, 10.0),
    3: (7.5, 8.75, 10.0),
    4: (7.0, 8.5, 10.0, 12.0),
    5: (7.0, 8.0, 9.0, 10.5, 12.5),
}

CLUSTER_WINDOW_HALF_H: Final[float] = 2.0

# "Irregular / 24h" clusters whose window covers every hour (residential
# K=4 z=4 and K=6 z=6, per v1.1).  Workplace has no irregular cluster.
IRREGULAR_CLUSTER_IDS_BY_K: Final[dict[int, frozenset[int]]] = {
    3: frozenset(),
    4: frozenset({4}),
    6: frozenset({6}),
}


# ---------------------------------------------------------------------------
# Output schema for plug_in_events.parquet (v2.0).
# ---------------------------------------------------------------------------

LOCATION_CATEGORIES: Final[tuple[str, ...]] = (
    "home",
    "work",
    "public_L2",
    "DCFC_emergency",
    "NACS_workplace",
)

PROFILE_TYPE_CATEGORIES: Final[tuple[str, ...]] = ("residential", "workplace")


# ---------------------------------------------------------------------------
# PROVENANCE TABLE — every numeric default cited.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProvenanceRow:
    parameter: str
    value: str
    source: str
    fit_status: str  # "[prior, unfit]" or "[fit, BIC]"


def _build_provenance() -> tuple[ProvenanceRow, ...]:
    rows: list[ProvenanceRow] = []
    # ---- residential β (v1.1 carry-forward) ----
    rows.append(ProvenanceRow(
        "beta_2_residential", "2.5",
        "Munkhammar 2015 Table 1 (residential p_home ≈ 0.75)",
        "[prior, unfit]"
    ))
    rows.append(ProvenanceRow(
        "beta_3_residential", "1.5",
        "Revised plan §5.1; Pecan Street routine-dominates",
        "[prior, unfit]"
    ))
    rows.append(ProvenanceRow(
        "beta_4_residential", "0.6",
        "Revised plan §5.1 log-dwell; calibrated to σ(5.9)≈0.997 anchor",
        "[prior, unfit]"
    ))
    rows.append(ProvenanceRow(
        "beta_5_residential", "-0.2",
        "Quirós-Tortós 2018 weekday-vs-weekend PDF shift",
        "[prior, unfit]"
    ))
    rows.append(ProvenanceRow(
        "beta_1_residential", "1.8",
        "Revised plan §5.1; Kreft 2024 anchor (low-SoC weaker than habit)",
        "[prior, unfit]"
    ))
    # ---- workplace β (v2.0-rev §4.4.4) ----
    rows.append(ProvenanceRow(
        "beta_2_workplace", "2.8",
        "Revised plan §4.4.4 / S-2 / Q8: FLIPPED HIGHER than residential 2.5; "
        "Pritchard 2023 §3 (85-95% workplace utilisation); Pareschi 2025",
        "[prior, unfit]"
    ))
    rows.append(ProvenanceRow(
        "beta_3_workplace", "1.5",
        "Revised plan §4.4.4 / S-2: LOWERED from v2.0-draft 1.8; lift now "
        "comes from β₂ not β₃",
        "[prior, unfit]"
    ))
    rows.append(ProvenanceRow(
        "beta_4_workplace", "0.3",
        "Revised plan §4.4.4: lower than residential 0.6; most workplace "
        "dwells 6-9h regardless",
        "[prior, unfit]"
    ))
    rows.append(ProvenanceRow(
        "beta_5_workplace", "-3.0",
        "Revised plan §4.4.4: much stronger negative than residential -0.2",
        "[prior, unfit]"
    ))
    rows.append(ProvenanceRow(
        "beta_1_workplace", "1.2",
        "Revised plan §4.4.4: Kreft, Brudermueller, Fleisch & Staake 2024 "
        "Applied Energy 370:123544 (NOT 'Köhler 2024' — citation fixed per RC-2)",
        "[prior, unfit]"
    ))
    rows.append(ProvenanceRow(
        "beta_0_workplace_K3", "(-0.9, -0.2, -1.1)",
        "Revised plan §4.4.4 anchor σ(5.26)≈0.995 in-cluster",
        "[prior, unfit]"
    ))
    # ---- W9 invariant ----
    rows.append(ProvenanceRow(
        "workplace_cluster_window_max_h", "6.0",
        "Revised plan §4.4.3 RC-3 / S-3 / Q2; degenerate-cluster rejection",
        "[invariant]"
    ))
    return tuple(rows)


PROVENANCE: Final[tuple[ProvenanceRow, ...]] = _build_provenance()


# ---------------------------------------------------------------------------
# Helper accessors.
# ---------------------------------------------------------------------------

def _beta_0_table(profile_type: str) -> dict[int, tuple[float, ...]]:
    return (
        BETA_0_BY_K_WORKPLACE
        if profile_type == "workplace"
        else BETA_0_BY_K_RESIDENTIAL
    )


def _beta_0_for_cluster(profile_type: str, K: int, z: int) -> float:
    """Return ``β₀(z)`` for cluster ``z`` (1-based) under K clusters.

    For the v2 BIC-sweep K-values (residential {3,4,6}; workplace
    {2,3,4,5}) this returns the canonical entry. For SPEECh K=16 (or
    any other K not in the canonical table) we use a midday-anchored
    default — the same value as the K=6 residential midday cluster
    (``-0.8``) for residential and the K=3 workplace mid cluster
    (``-0.2``) for workplace. The SPEECh path's lift comes primarily
    from the published P(G) prior, the cluster-window match, and the
    β₂ / β₄ terms; β₀(z) drift across clusters has limited bite at
    K=16 because all 16 clusters share a roughly common scale anchor.
    """
    table = _beta_0_table(profile_type)
    if K in table:
        try:
            return float(table[K][z - 1])
        except (IndexError, TypeError):
            pass
    # Fallback for K=16 / SPEECh path (any K not in the canonical table).
    if profile_type == "workplace":
        return -0.2  # K=3 mid cluster anchor (Pritchard 2023 / Lee 2019).
    return -0.8  # K=6 midday-anchored default (Quirós-Tortós 2018).


def _beta_table(profile_type: str) -> dict[str, float]:
    return BETA_WORKPLACE if profile_type == "workplace" else BETA_RESIDENTIAL


def _pi_base_table(profile_type: str) -> dict[int, tuple[float, ...]]:
    return (
        PI_BASE_WORKPLACE
        if profile_type == "workplace"
        else PI_BASE_RESIDENTIAL
    )


def _k_sweep(profile_type: str) -> tuple[int, ...]:
    return K_SWEEP_WORKPLACE if profile_type == "workplace" else K_SWEEP_RESIDENTIAL


def _default_k(profile_type: str) -> int:
    return DEFAULT_K_WORKPLACE if profile_type == "workplace" else DEFAULT_K_RESIDENTIAL


def _location_label_for(profile_type: str) -> str:
    return "work" if profile_type == "workplace" else "home"


# ---------------------------------------------------------------------------
# Cohort loaders.
# ---------------------------------------------------------------------------

DEFAULT_WORKPLACE_COHORT_REL: Final[Path] = Path(
    "data/pev/raw/evwatts_cohorts/workplace_subset.parquet"
)
DEFAULT_RESIDENTIAL_SESSIONS_REL: Final[Path] = Path(
    "data/pev/raw/evwatts.public.vehiclesessions.csv"
)
DEFAULT_RESIDENTIAL_VEHICLES_REL: Final[Path] = Path(
    "data/pev/raw/evwatts.public.vehicles.csv"
)


def _load_workplace_cohort(parquet_path: Path) -> pd.DataFrame:
    """Load the v2.0 workplace EVWatts cohort (Dev C output).

    Expected columns: vehicle_id, n_sessions, plug_in_hr_med,
    plug_out_hr_med, dur_h_med, avg_kw_med, ownership, state,
    best_window_start, best_window_end.
    """
    if not parquet_path.is_file():
        return pd.DataFrame(
            columns=[
                "vehicle_id", "plug_in_hr_med", "plug_out_hr_med",
                "dur_h_med", "avg_kw_med",
            ]
        )
    df = pd.read_parquet(parquet_path)
    keep = ["vehicle_id", "plug_in_hr_med", "plug_out_hr_med", "dur_h_med", "avg_kw_med"]
    return df[[c for c in keep if c in df.columns]].copy()


EVWATTS_VEHICLE_TYPE: Final[str] = "PASSENGER CAR"
EVWATTS_BEV_PREFIX: Final[str] = "BEV"
EVWATTS_MIN_SESSIONS_IN_WINDOW: Final[int] = 130
EVWATTS_MEDIAN_AVG_KW_LO: Final[float] = 1.5
EVWATTS_MEDIAN_AVG_KW_HI: Final[float] = 11.5


def _load_residential_evwatts(
    sessions_csv: Path,
    vehicles_csv: Path,
) -> pd.DataFrame:
    """Load EVWatts session triples for the residential candidate subset
    (v1.1 logic carry-forward)."""
    if not (sessions_csv.is_file() and vehicles_csv.is_file()):
        return pd.DataFrame(
            columns=["vehicle_id", "start_hour", "duration_h", "energy_kwh"]
        )

    veh = pd.read_csv(vehicles_csv)
    sess = pd.read_csv(sessions_csv)

    veh_cand = veh[
        (veh["vehicle_type"] == EVWATTS_VEHICLE_TYPE)
        & veh["electrification_level"].astype(str).str.startswith(EVWATTS_BEV_PREFIX)
    ].copy()
    cand_ids = set(veh_cand["id"].astype(int).tolist())
    if not cand_ids:
        return pd.DataFrame(
            columns=["vehicle_id", "start_hour", "duration_h", "energy_kwh"]
        )

    s = sess[sess["vehicle_id"].isin(cand_ids)].copy()
    s = s.dropna(subset=["start_datetime", "duration", "energy_kwh"])
    s["start_dt"] = pd.to_datetime(s["start_datetime"], errors="coerce")
    s = s.dropna(subset=["start_dt"])
    if s.empty:
        return pd.DataFrame(
            columns=["vehicle_id", "start_hour", "duration_h", "energy_kwh"]
        )

    counts = s.groupby("vehicle_id").size()
    vehicles_kept = counts[counts >= EVWATTS_MIN_SESSIONS_IN_WINDOW].index
    if len(vehicles_kept) == 0:
        vehicles_kept = counts.sort_values(ascending=False).head(84).index
    s = s[s["vehicle_id"].isin(vehicles_kept)].copy()

    s["avg_kw"] = s["energy_kwh"] / s["duration"].clip(lower=1e-6)
    median_kw = s.groupby("vehicle_id")["avg_kw"].median()
    vehicles_in_band = median_kw[
        (median_kw >= EVWATTS_MEDIAN_AVG_KW_LO)
        & (median_kw <= EVWATTS_MEDIAN_AVG_KW_HI)
    ].index
    s = s[s["vehicle_id"].isin(vehicles_in_band)].copy()

    if s.empty:
        return pd.DataFrame(
            columns=["vehicle_id", "start_hour", "duration_h", "energy_kwh"]
        )

    s["start_hour"] = s["start_dt"].dt.hour.astype(np.float64) + (
        s["start_dt"].dt.minute.astype(np.float64) / 60.0
    )

    out = pd.DataFrame(
        {
            "vehicle_id": s["vehicle_id"].astype(np.int64).to_numpy(),
            "start_hour": s["start_hour"].to_numpy(),
            "duration_h": s["duration"].astype(np.float64).to_numpy(),
            "energy_kwh": s["energy_kwh"].astype(np.float64).to_numpy(),
        }
    )
    out = out[(out["duration_h"] > 0.0) & (out["duration_h"] < 48.0)]
    out = out[(out["energy_kwh"] >= 0.0) & (out["energy_kwh"] < 200.0)]
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# v3.0 SPEECh K=16 published-GMM loader.
# ---------------------------------------------------------------------------

#: Repo-relative canonical path for the SPEECh JSON pair (residential +
#: workplace). The data tree under ``data/pev/raw/speech/`` is gitignored;
#: the JSONs themselves are produced by the one-shot conversion script
#: alongside the Mendeley pickles (see ``docs/v3/modeling/k16_speech.md``).
SPEECH_DATA_REL: Final[Path] = Path("data/pev/raw/speech")
SPEECH_K16_RESIDENTIAL_REL: Final[Path] = SPEECH_DATA_REL / "speech_k16_residential.json"
SPEECH_K16_WORKPLACE_REL: Final[Path] = SPEECH_DATA_REL / "speech_k16_workplace.json"

#: Public start-time axis name in the SPEECh JSON.
SPEECH_START_TIME_AXIS: Final[str] = "start_time_s_LT"

#: Total number of SPEECh driver groups (Powell-2022 K=16).
SPEECH_K_TOTAL: Final[int] = 16

#: Default value for ``ClusterFit.axes`` under the v2 BIC-sweep path
#: (backwards-compatible: ``("start_h_LT",)``).
CLUSTER_AXES_V2_DEFAULT: Final[tuple[str, ...]] = ("start_h_LT",)


@dataclass(frozen=True)
class SpeechParams:
    """v3.0 published-SPEECh K=16 mixture parameters (start-time marginal + P(G) prior).

    Powell, Cezar, Rajagopal 2022 Applied Energy 309 — Mendeley
    ``gvk34mybtb/1`` (CC-BY-4.0). v3.0 adopts the **start-time marginal**
    across the 3-D joint ``(start_time, energy, duration)``; joint
    sampling is deferred to v3.1.

    Field semantics:

    * ``P_G`` — length-16 driver-group prior, reweighted over
      ``groups_present`` per ``day_type`` so a sampler can draw a group
      from ``Categorical(P_G_*)`` without rejection. The original prior
      is preserved on disk in the SPEECh JSON for reproducibility; this
      ``SpeechParams.P_G`` carries the per-day reweighted vector unioned
      over weekday + weekend (so all present groups across either day
      get a non-zero probability mass). For per-day sampling, callers
      should use ``weekday_means_h`` / ``weekend_means_h`` and the
      corresponding ``groups_present_*`` tuple — the ``P_G`` field is
      the cluster-prior surfaced to ``ClusterFit`` for byte-stable
      mixing-weight bookkeeping.

    * ``weekday_means_h``, ``weekday_covs_h``, ``weekday_weights`` are
      tuples-of-tuples indexed by the order in ``groups_present_weekday``
      (outer: groups; inner: GMM components). ``means`` are start-time
      means already converted to **hours-LT** (seconds-of-LT divided by
      3600). ``covs`` are the start-time-axis-only variances (the (0, 0)
      entry of each 3×3 SPEECh covariance), kept in **hours² LT**
      (variance in seconds² scaled by 1/3600²).

    * ``weekend_*`` mirror the weekday tuples for the weekend GMM.
    """

    profile_type: str           # "residential" | "workplace"
    P_G: tuple[float, ...]      # length 16, renormalised over union of present groups
    weekday_means_h: tuple[tuple[float, ...], ...]   # outer: present groups; inner: components
    weekday_covs_h: tuple[tuple[float, ...], ...]    # start-time variances (h²)
    weekday_weights: tuple[tuple[float, ...], ...]
    weekend_means_h: tuple[tuple[float, ...], ...]
    weekend_covs_h: tuple[tuple[float, ...], ...]
    weekend_weights: tuple[tuple[float, ...], ...]
    groups_present_weekday: tuple[int, ...]
    groups_present_weekend: tuple[int, ...]
    source_doi: str = ""
    source_url: str = ""
    source_license: str = ""


def _resolve_speech_json_path(profile_type: str, data_root: Path | None) -> Path:
    """Resolve ``speech_k16_<profile_type>.json`` under ``data/pev/raw/speech``.

    If ``data_root`` is None, attempt ``repo_root()`` resolution. The
    function does not check existence; callers must.
    """
    if profile_type == "residential":
        rel = SPEECH_K16_RESIDENTIAL_REL
    elif profile_type == "workplace":
        rel = SPEECH_K16_WORKPLACE_REL
    else:
        raise ValueError(
            f"profile_type must be 'residential' or 'workplace'; got {profile_type!r}"
        )
    if data_root is not None:
        return Path(data_root) / rel
    # Resolve via the repo helper.
    from ._paths import repo_root as _repo_root_fn  # noqa: PLC0415
    return _repo_root_fn() / rel


def _speech_group_arrays(
    day_block: dict, axis_index: int = 0
) -> tuple[
    tuple[tuple[float, ...], ...],
    tuple[tuple[float, ...], ...],
    tuple[tuple[float, ...], ...],
    tuple[int, ...],
]:
    """Project the SPEECh ``weekday``/``weekend`` block onto a single axis marginal.

    Returns ``(means_h, covs_h, weights, group_ids)`` with the outer tuple
    ordered by the JSON-declared ``groups_present`` list, so a sampler can
    iterate over present groups in a deterministic order.

    Axis 0 (default) is the start-time axis; means are converted from
    seconds-of-LT to hours-LT (÷3600) and covariances from seconds² to
    hours² (÷3600²).
    """
    groups_present = tuple(int(g) for g in day_block["groups_present"])
    groups = {int(g["group_id"]): g for g in day_block["groups"]}
    means_h: list[tuple[float, ...]] = []
    covs_h: list[tuple[float, ...]] = []
    weights: list[tuple[float, ...]] = []
    sec2_to_h2 = 1.0 / (3600.0 * 3600.0)
    sec_to_h = 1.0 / 3600.0
    for g in groups_present:
        if g not in groups:
            # Defensive: JSON declares a present group but no GMM in groups[].
            raise ValueError(
                f"SPEECh JSON inconsistency: group_id {g} listed in "
                f"groups_present but missing from groups[]"
            )
        gmm = groups[g]
        gmm_means = gmm["means"]
        gmm_covs = gmm["covariances"]
        gmm_weights = gmm["weights"]
        means_h.append(
            tuple(float(comp[axis_index]) * sec_to_h for comp in gmm_means)
        )
        covs_h.append(
            tuple(
                float(comp_cov[axis_index][axis_index]) * sec2_to_h2
                for comp_cov in gmm_covs
            )
        )
        weights.append(tuple(float(w) for w in gmm_weights))
    return (tuple(means_h), tuple(covs_h), tuple(weights), groups_present)


def load_speech_k16_params(
    profile_type: str,
    data_root: Path | None = None,
) -> SpeechParams:
    """Load SPEECh K=16 parameters from ``data/pev/raw/speech/``.

    Reads ``speech_k16_<profile_type>.json``, projects the 3-D joint
    onto the start-time axis marginal (means → hours-LT, covariances →
    hours² LT), and reweights ``P_G`` over the **union** of weekday and
    weekend present groups (so callers can draw a group categorically
    once and then pick the per-day GMM components). For per-day sampling
    a downstream consumer should reweight again over
    ``groups_present_weekday`` / ``groups_present_weekend`` separately.

    Parameters
    ----------
    profile_type
        ``"residential"`` (loads ``home`` segment) or ``"workplace"``
        (loads ``work`` segment).
    data_root
        Optional override for the repository root (defaults to
        ``_paths.repo_root()``).

    Raises
    ------
    FileNotFoundError
        If the JSON is not on disk. Run the S1 conversion script first
        (see ``docs/v3/modeling/k16_speech.md``).
    """
    json_path = _resolve_speech_json_path(profile_type, data_root)
    if not json_path.is_file():
        raise FileNotFoundError(
            f"SPEECh K=16 parameters not found at {json_path!s}. "
            f"Convert the Mendeley pickles first via "
            f"``data/pev/raw/speech/_convert_pickles.py`` (see "
            f"docs/v3/modeling/k16_speech.md)."
        )
    with json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if payload.get("profile_type") != profile_type:
        raise ValueError(
            f"SPEECh JSON profile_type mismatch: file declares "
            f"{payload.get('profile_type')!r}, expected {profile_type!r}"
        )
    p_g_raw = [float(x) for x in payload["P_G"]]
    if len(p_g_raw) != SPEECH_K_TOTAL:
        raise ValueError(
            f"SPEECh JSON P_G length must be {SPEECH_K_TOTAL}; got "
            f"{len(p_g_raw)}"
        )

    weekday_means_h, weekday_covs_h, weekday_weights, gp_weekday = (
        _speech_group_arrays(payload["weekday"], axis_index=0)
    )
    weekend_means_h, weekend_covs_h, weekend_weights, gp_weekend = (
        _speech_group_arrays(payload["weekend"], axis_index=0)
    )

    # Reweight P_G over the union of present groups (weekday ∪ weekend).
    # This keeps the ``ClusterFit.cluster_weights`` field a length-16
    # vector indexed by group_id with mass on every group that fires
    # for at least one day_type; per-day weights live on the daily
    # ``*_weights`` tuples. Float-drift renormalisation: divide by sum.
    union_present = set(gp_weekday) | set(gp_weekend)
    p_g_masked = [
        (p if g in union_present else 0.0) for g, p in enumerate(p_g_raw)
    ]
    total = float(sum(p_g_masked))
    if total <= 0.0:
        raise ValueError(
            "SPEECh JSON: no groups present across weekday∪weekend; "
            "cannot renormalise P_G"
        )
    p_g_renorm = tuple(float(p) / total for p in p_g_masked)

    source_block = payload.get("source", {}) if isinstance(payload, dict) else {}
    return SpeechParams(
        profile_type=str(profile_type),
        P_G=p_g_renorm,
        weekday_means_h=weekday_means_h,
        weekday_covs_h=weekday_covs_h,
        weekday_weights=weekday_weights,
        weekend_means_h=weekend_means_h,
        weekend_covs_h=weekend_covs_h,
        weekend_weights=weekend_weights,
        groups_present_weekday=gp_weekday,
        groups_present_weekend=gp_weekend,
        source_doi=str(source_block.get("doi", "")),
        source_url=str(source_block.get("url", "")),
        source_license=str(source_block.get("license", "")),
    )


def _speech_centres_per_group(
    means_per_group: tuple[tuple[float, ...], ...],
    weights_per_group: tuple[tuple[float, ...], ...],
) -> tuple[float, ...]:
    """Collapse a (group, component) start-hour table to a per-group centre.

    For each present group, the centre is the **weighted circular mean**
    of its GMM components on the 24-hour clock. We treat hours-LT as
    angles on the unit circle (``θ = h · 2π/24``) so that midnight-
    straddling components (e.g. arrivals at 23.5 and 0.5) collapse to a
    sensible 24:00 centre rather than the linear arithmetic mean of 12.0.

    The returned tuple is indexed positionally over the present-group
    list (so ``cluster_centres_h[i]`` matches
    ``groups_present_*[i]``). Empty groups (zero weights) are filled
    with 12.0 as a defensive fallback so the consumer's
    ``_in_cluster_window`` predicate never sees NaN.
    """
    out: list[float] = []
    for means, weights in zip(means_per_group, weights_per_group, strict=True):
        if not means or not weights:
            out.append(12.0)
            continue
        w = np.asarray(weights, dtype=np.float64)
        m = np.asarray(means, dtype=np.float64)
        wsum = float(w.sum())
        if wsum <= 0.0 or not np.isfinite(wsum):
            out.append(12.0)
            continue
        theta = m * (2.0 * np.pi / 24.0)
        sin_w = float(np.sum(w * np.sin(theta))) / wsum
        cos_w = float(np.sum(w * np.cos(theta))) / wsum
        ang = float(np.arctan2(sin_w, cos_w))
        h = (ang * 24.0 / (2.0 * np.pi)) % 24.0
        out.append(float(h))
    return tuple(out)


# ---------------------------------------------------------------------------
# BIC sweep + W9 invariant.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClusterFit:
    """Output of one cluster-fit call (BIC sweep OR SPEECh K=16 load).

    v3.0 extends this dataclass with two optional fields:

    * ``axes`` — names of the axes the cluster table spans. For the
      v2 BIC sweep this is always ``("start_h_LT",)``; for SPEECh K=16
      v3.0 still consumes only the start-time marginal, so axes are the
      same (joint dwell/energy projection is v3.1 scope).
    * ``speech_params`` — when ``cluster_source="speech_k16"``, carries
      the full SPEECh parameter block (per-day GMMs + P(G) prior) for
      downstream Bernoulli iteration and weekday/weekend selection.
      ``None`` under the legacy BIC-sweep path.

    For weekday/weekend selection at the per-arrival predicate, the
    ``cluster_centres_h`` and ``cluster_weights`` fields are populated
    from the **weekday** SPEECh GMMs (deterministic byte-stable
    bookkeeping); the predicate threads the weekend variant directly
    out of ``speech_params`` when ``is_weekend=True``.
    """

    profile_type: str
    K_sweep: tuple[int, ...]
    K_chosen: int
    bic_chosen: float
    bic_per_K: dict[int, float]
    cluster_centres_h: tuple[float, ...]  # hour-of-day centre per cluster
    cluster_windows_h: tuple[float, ...]  # 95% start-time window per cluster
    cluster_weights: tuple[float, ...]
    ambiguous_flag: bool
    degenerate_flag: bool
    rejected_K: tuple[int, ...]
    rejection_reason_per_K: dict[int, str]
    note: str
    #: RFC-037: when BIC's K is overridden post-sweep (e.g. EVWatts
    #: cohort centroid drift outside the commuter band), this carries
    #: the human-readable reason. ``None`` when no override applied.
    bic_override_reason: str | None = None
    #: v3.0: axis labels for ``cluster_centres_h``. Defaults to
    #: ``("start_h_LT",)`` for byte-stability with v2 BIC-sweep caches.
    axes: tuple[str, ...] = CLUSTER_AXES_V2_DEFAULT
    #: v3.0: SPEECh K=16 parameter block when ``cluster_source=
    #: "speech_k16"``. ``None`` under the legacy BIC-sweep path so the
    #: v2.1 byte-stable cache continues to serialise unchanged.
    speech_params: SpeechParams | None = None


def _fit_gmm_for_K(
    X: np.ndarray,
    K: int,
    seed: int,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Fit a K-component GMM on X. Returns (bic, centres, weights, labels)."""
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    Xs = (X - mu) / sd
    gmm = GaussianMixture(
        n_components=int(K),
        covariance_type="full",
        random_state=int(seed) + MODULE_OFFSET + int(K),
        max_iter=200,
        n_init=3,
        reg_covar=1e-4,
    )
    gmm.fit(Xs)
    bic = float(gmm.bic(Xs))
    centres = gmm.means_ * sd + mu
    labels = gmm.predict(Xs).astype(int)
    weights = gmm.weights_.astype(float)
    return bic, centres, weights, labels


def _cluster_window_h(values: np.ndarray) -> float:
    """Return the 95% start-time window (h) for a single cluster's members.

    Uses (p97.5 - p2.5) which is the standard 95% inter-percentile
    range. With < 4 members we fall back to (max - min).
    """
    if len(values) == 0:
        return 0.0
    if len(values) < 4:
        return float(values.max() - values.min())
    lo = float(np.percentile(values, 2.5))
    hi = float(np.percentile(values, 97.5))
    return hi - lo


def run_bic_sweep(
    df: pd.DataFrame,
    *,
    profile_type: str,
    feature_cols: tuple[str, str, str],
    K_sweep: tuple[int, ...] | None = None,
    seed: int = MASTER_SEED_DEFAULT,
    cluster_window_max_h: float = WORKPLACE_CLUSTER_WINDOW_MAX_H,
    min_cohort_for_K5: int = WORKPLACE_MIN_COHORT_FOR_K5,
) -> ClusterFit:
    """Fit GMM(K) for K ∈ K_sweep on the cohort feature matrix.

    Parameters
    ----------
    df : pd.DataFrame
        One row per vehicle (workplace) or per session (residential).
    profile_type : str
        "residential" or "workplace".
    feature_cols : (start_hr, duration_h, energy_or_avg_kw)
        Column names in df. For workplace cohort that's
        (plug_in_hr_med, dur_h_med, avg_kw_med). For residential it's
        (start_hour, duration_h, energy_kwh).
    cluster_window_max_h : float
        W9 invariant threshold. Workplace caches apply; residential
        ignores (set to +inf by callers if needed).
    """
    if K_sweep is None:
        K_sweep = _k_sweep(profile_type)
    K_sweep = tuple(int(k) for k in K_sweep)
    K_sweep_full = K_sweep  # remember for reporting
    # If cohort is small, cap K via RFC-036 admissibility rules.
    # n < 90 → K ≤ 3 ; 90 ≤ n < 150 → K ≤ 4 ; n ≥ 150 → full sweep.
    n_rows = int(len(df))
    if profile_type == "workplace":
        admissible = _admissible_K(n_rows)
        K_sweep = tuple(k for k in K_sweep if k in admissible)

    if n_rows < 8 or any(c not in df.columns for c in feature_cols):
        return _degenerate_cluster_fit(
            profile_type=profile_type,
            K_sweep=K_sweep_full,
            reason="cohort_too_small_or_missing_cols",
        )

    X = df[list(feature_cols)].to_numpy(dtype=np.float64)
    mask = np.isfinite(X).all(axis=1)
    X = X[mask]
    if len(X) < 8:
        return _degenerate_cluster_fit(
            profile_type=profile_type,
            K_sweep=K_sweep_full,
            reason="cohort_too_small_after_nan_filter",
        )

    bic_per_K: dict[int, float] = {}
    fit_centres: dict[int, np.ndarray] = {}
    fit_weights: dict[int, np.ndarray] = {}
    fit_labels: dict[int, np.ndarray] = {}
    rejected_K: list[int] = []
    rejection_reason_per_K: dict[int, str] = {}

    for K in K_sweep:
        try:
            bic, centres, weights, labels = _fit_gmm_for_K(X, K, seed=seed)
        except Exception as exc:  # numerical failure
            bic_per_K[K] = float("inf")
            rejected_K.append(K)
            rejection_reason_per_K[K] = f"fit_error: {exc}"
            continue
        bic_per_K[K] = bic
        fit_centres[K] = centres
        fit_weights[K] = weights
        fit_labels[K] = labels

        # W9 invariant for workplace: every cluster's 95% start-time window ≤ 6h.
        if profile_type == "workplace":
            max_window = 0.0
            for z in range(K):
                members = X[labels == z, 0]  # column 0 = start hour
                w = _cluster_window_h(members)
                if w > max_window:
                    max_window = w
            if max_window > cluster_window_max_h:
                rejected_K.append(K)
                rejection_reason_per_K[K] = (
                    f"W9_violation: max_cluster_window={max_window:.2f}h "
                    f"> {cluster_window_max_h:.1f}h"
                )

    # All BIC infinite or all K rejected → degenerate.
    valid_K = [K for K in K_sweep if K in bic_per_K and K not in rejected_K
               and np.isfinite(bic_per_K[K])]
    if not valid_K:
        return _degenerate_cluster_fit(
            profile_type=profile_type,
            K_sweep=K_sweep_full,
            reason="no_K_survived_W9_or_all_fits_failed",
            rejected_K=tuple(sorted(set(rejected_K))),
            rejection_reason_per_K=rejection_reason_per_K,
        )

    # Choose by min BIC; if ambiguous (rel-gap < 1%), default to profile-K.
    sorted_valid = sorted(valid_K, key=lambda k: bic_per_K[k])
    K_chosen = sorted_valid[0]
    best = bic_per_K[K_chosen]
    runner_up = bic_per_K[sorted_valid[1]] if len(sorted_valid) >= 2 else best
    rel_gap = abs(runner_up - best) / abs(best) if abs(best) > 0 else float("inf")
    ambiguous = rel_gap < BIC_AMBIGUOUS_REL_GAP
    if ambiguous:
        default_K = _default_k(profile_type)
        if default_K in valid_K:
            K_chosen = default_K

    centres = fit_centres[K_chosen]
    labels = fit_labels[K_chosen]
    weights = fit_weights[K_chosen]

    # Compute the start-time window per cluster (for reporting).
    windows: list[float] = []
    for z in range(K_chosen):
        members = X[labels == z, 0]
        windows.append(_cluster_window_h(members))

    # Time-sort clusters by their start-hour centre and re-permute weights/windows.
    order = np.argsort(centres[:, 0])
    centres = centres[order]
    weights = weights[order]
    windows = [windows[i] for i in order]

    centres_h_tuple = tuple(float(c) for c in centres[:, 0])
    windows_tuple = tuple(float(w) for w in windows)
    weights_tuple = tuple(float(w) for w in weights)

    # RFC-037: post-BIC commuter-window sanity check. For workplace
    # cohorts (EVWatts), if ≥ 2 fit-centroid hours fall outside the
    # canonical commuter band [7.0, 10.5], override BIC's K with the
    # commuter-prior K=3 anchor. The literature anchor (Lee/Li/Low
    # 2019; Pritchard 2023) places workplace plug-in centres at
    # ~07:30 / 08:45 / 10:00 LT; EVWatts midday drift is documented
    # in `workplace_cohort_caveat`.
    bic_override_reason: str | None = None
    if profile_type == "workplace":
        out_of_band = sum(
            1
            for h in centres_h_tuple
            if not (EVWATTS_CENTROID_BAND_LO_H <= h <= EVWATTS_CENTROID_BAND_HI_H)
        )
        if out_of_band >= EVWATTS_CENTROID_OUT_OF_BAND_THRESHOLD:
            K_chosen = int(EVWATTS_CENTROID_DRIFT_OVERRIDE_K)
            prior_centres = WINDOW_CENTRES_WORKPLACE_PRIOR[K_chosen]
            prior_weights = PI_BASE_WORKPLACE[K_chosen]
            centres_h_tuple = tuple(float(c) for c in prior_centres)
            weights_tuple = tuple(float(w) for w in prior_weights)
            windows_tuple = tuple(2.0 * CLUSTER_WINDOW_HALF_H for _ in prior_centres)
            bic_override_reason = BIC_OVERRIDE_REASON_CENTROID_DRIFT

    return ClusterFit(
        profile_type=profile_type,
        K_sweep=K_sweep_full,
        K_chosen=int(K_chosen),
        bic_chosen=float(best),
        bic_per_K={int(k): float(v) for k, v in bic_per_K.items()},
        cluster_centres_h=centres_h_tuple,
        cluster_windows_h=windows_tuple,
        cluster_weights=weights_tuple,
        ambiguous_flag=bool(ambiguous),
        degenerate_flag=False,
        rejected_K=tuple(sorted(set(rejected_K))),
        rejection_reason_per_K={int(k): str(v) for k, v in rejection_reason_per_K.items()},
        note=(
            f"GMM fit on {profile_type} cohort (n={len(X)}); features "
            f"={feature_cols}; W9_threshold={cluster_window_max_h:.1f}h; "
            f"K_default_on_ambiguity={_default_k(profile_type)}"
        ),
        bic_override_reason=bic_override_reason,
    )


def _degenerate_cluster_fit(
    *,
    profile_type: str,
    K_sweep: tuple[int, ...],
    reason: str,
    rejected_K: tuple[int, ...] = (),
    rejection_reason_per_K: dict[int, str] | None = None,
) -> ClusterFit:
    """Return a fall-back ClusterFit using prior centres / weights."""
    K_chosen = _default_k(profile_type)
    if profile_type == "workplace":
        centres = WINDOW_CENTRES_WORKPLACE_PRIOR.get(K_chosen, (8.0, 9.0, 10.0))
    else:
        centres = WINDOW_CENTRES_RESIDENTIAL.get(K_chosen, (12.5, 18.5, 23.5))
    weights = _pi_base_table(profile_type)[K_chosen]
    return ClusterFit(
        profile_type=profile_type,
        K_sweep=K_sweep,
        K_chosen=int(K_chosen),
        bic_chosen=float("nan"),
        bic_per_K={int(k): float("inf") for k in K_sweep},
        cluster_centres_h=tuple(float(c) for c in centres),
        cluster_windows_h=tuple(2.0 * CLUSTER_WINDOW_HALF_H for _ in centres),
        cluster_weights=tuple(float(w) for w in weights),
        ambiguous_flag=True,
        degenerate_flag=True,
        rejected_K=tuple(rejected_K),
        rejection_reason_per_K=dict(rejection_reason_per_K or {}),
        note=(
            f"BIC sweep degenerate ({reason}); falling back to "
            f"K={K_chosen} with literature priors per "
            f"revised plan §4.4.3 / §5.1"
        ),
    )


# ---------------------------------------------------------------------------
# π construction by (income_tier, garage_type) and per-EV cluster assignment.
# ---------------------------------------------------------------------------

def _pi_for_segment(
    K: int,
    income_tier: str,
    garage_type: str,
    profile_type: str,
    base_pi: tuple[float, ...] | None = None,
) -> np.ndarray:
    """Return a length-K mixing-weight vector for the (income_tier,
    garage_type) cell.

    For residential, applies the income/garage modulation deltas (plan
    §5.1).  For workplace, returns the base π unchanged (workplace is
    not segmented on residential garage-type).
    """
    table = _pi_base_table(profile_type)
    if K not in table and base_pi is None:
        raise ValueError(f"Unsupported K={K} for profile_type={profile_type}")
    pi = np.array(base_pi if base_pi is not None else table[K], dtype=np.float64).copy()
    if pi.shape[0] != K:
        # Pad / truncate defensively.
        if pi.shape[0] < K:
            extra = np.full(K - pi.shape[0], 1.0 / K)
            pi = np.concatenate([pi, extra])
        else:
            pi = pi[:K]
        pi = pi / pi.sum() if pi.sum() > 0 else np.full(K, 1.0 / K)

    if profile_type == "workplace":
        # Workplace cohort is not segmented on residential garage_type;
        # return base π unchanged.
        return pi

    # Residential: apply income / garage deltas only on the v2 BIC-sweep
    # K ∈ {3, 4, 6} clusterings where the (midday, overnight) cluster
    # indices are anchored. For SPEECh K=16 (or any larger K_union) the
    # income/garage modulation is not well-defined positionally — return
    # the base π unchanged (the P(G) prior already encodes mass on the
    # right driver groups).
    if K not in (3, 4, 6):
        return pi

    idx_midday = 0
    if K == 3:
        idx_overnight = 2
    elif K == 4:
        idx_overnight = 2
    else:  # K = 6
        idx_overnight = 4

    if str(income_tier) == "t4":
        pi[idx_overnight] += INCOME_T4_OVERNIGHT_DELTA
    if str(garage_type) in {"shared", "curb"}:
        pi[idx_midday] += GARAGE_SHARED_OR_CURB_MIDDAY_DELTA
        pi[idx_overnight] += GARAGE_SHARED_OR_CURB_OVERNIGHT_DELTA

    pi = np.clip(pi, a_min=0.0, a_max=None)
    total = pi.sum()
    if total <= 0.0:
        return np.full(K, 1.0 / K, dtype=np.float64)
    return pi / total


def assign_clusters(
    fleet: pd.DataFrame,
    K_chosen: int,
    profile_type: str = "residential",
    *,
    master_seed: int = MASTER_SEED_DEFAULT,
    base_pi: tuple[float, ...] | None = None,
) -> np.ndarray:
    """Sample latent cluster z ∈ {1..K_chosen} per ev_id (deterministic
    per-EV)."""
    n = len(fleet)
    z = np.empty(n, dtype=np.int8)
    income_arr = (
        fleet["income_tier"].astype(str).to_numpy()
        if "income_tier" in fleet.columns
        else np.array(["t2"] * n)
    )
    garage_arr = (
        fleet["garage_type"].astype(str).to_numpy()
        if "garage_type" in fleet.columns
        else np.array(["dedicated"] * n)
    )
    ev_id_arr = fleet["ev_id"].astype(np.int64).to_numpy()

    for i in range(n):
        sub_seed = (
            int(master_seed) + _CLUSTER_ASSIGN_SUBSEED_OFFSET + int(ev_id_arr[i])
        )
        rng = np.random.default_rng(sub_seed)
        pi = _pi_for_segment(
            K_chosen, income_arr[i], garage_arr[i], profile_type, base_pi=base_pi,
        )
        z[i] = int(rng.choice(K_chosen, p=pi)) + 1  # 1-based labels
    return z


# ---------------------------------------------------------------------------
# Bernoulli iteration.
# ---------------------------------------------------------------------------

def _in_cluster_window(
    t_hour: float,
    z: int,
    centres: tuple[float, ...],
    K: int,
    profile_type: str,
    half_h: float = CLUSTER_WINDOW_HALF_H,
) -> bool:
    """True iff `t_hour` ∈ [centre(z) - half_h, centre(z) + half_h] mod 24.

    Residential irregular clusters (K=4 z=4, K=6 z=6) always return True
    under the legacy BIC-sweep path. v3.0 SPEECh K=16 clusters have no
    "irregular catch-all" — callers pass the appropriate weekday or
    weekend centres explicitly (and ``IRREGULAR_CLUSTER_IDS_BY_K`` only
    has entries for K ∈ {3, 4, 6} so the short-circuit is a no-op when
    K=16).
    """
    if profile_type == "residential" and z in IRREGULAR_CLUSTER_IDS_BY_K.get(K, frozenset()):
        return True
    if not 1 <= z <= len(centres):
        return False
    centre = centres[z - 1]
    d = abs((t_hour - centre + 12.0) % 24.0 - 12.0)
    return d <= half_h


def _sigmoid(x: float) -> float:
    """Numerically stable logistic."""
    if x >= 0:
        return 1.0 / (1.0 + float(np.exp(-x)))
    e = float(np.exp(x))
    return e / (1.0 + e)


def compute_p_plug(
    *,
    soc_at_arrival_frac: float,
    is_correct_location: bool,
    in_preferred_window: bool,
    dwell_h: float,
    is_weekend: bool,
    K: int,
    z: int,
    profile_type: str,
) -> float:
    """σ(β₀(z) + β₁·(1−SoC) + β₂·𝟙[loc] + β₃·𝟙[window] + β₄·log(dwell_h) +
    β₅·𝟙[weekend]).

    `is_correct_location` is True iff this is a `home` arrival for
    residential or a `work` arrival for workplace. Non-matching
    locations return p_plug = 0.
    """
    if not is_correct_location:
        return 0.0
    beta_0 = _beta_0_for_cluster(profile_type, K, z)
    beta = _beta_table(profile_type)
    dwell_safe = max(float(dwell_h), DWELL_LOG_FLOOR_H)
    x = (
        beta_0
        + beta["beta_1"] * (1.0 - float(soc_at_arrival_frac))
        + beta["beta_2"]
        + beta["beta_3"] * (1.0 if in_preferred_window else 0.0)
        + beta["beta_4"] * float(np.log(dwell_safe))
        + beta["beta_5"] * (1.0 if is_weekend else 0.0)
    )
    return _sigmoid(x)


@dataclass(frozen=True)
class BernoulliConfig:
    """Configurable knobs for the per-arrival Bernoulli loop."""

    initial_soc_frac: float = 0.70
    soc_recovery_on_plugin: float = 0.85
    soc_min_frac: float = 0.10
    soc_max_frac: float = 1.00
    last_arrival_pseudo_dwell_h: float = 12.0
    workplace_soc_target_lo: float = 0.50
    workplace_soc_target_hi: float = 0.95
    workplace_soc_target_alpha: float = 2.0
    workplace_soc_target_beta: float = 2.0


#: Shared default Bernoulli config. ``BernoulliConfig`` is frozen, so a single
#: module-level instance is safe to share across all calls (and matches the
#: prior def-time-evaluated default exactly).
_DEFAULT_BERNOULLI_CONFIG = BernoulliConfig()


def iterate_bernoulli(
    mobility: pd.DataFrame,
    fleet: pd.DataFrame,
    cluster_z: np.ndarray,
    *,
    K_chosen: int,
    profile_type: str,
    cluster_centres_h: tuple[float, ...],
    region_tz: str,
    master_seed: int = MASTER_SEED_DEFAULT,
    bcfg: BernoulliConfig = _DEFAULT_BERNOULLI_CONFIG,
    weekend_cluster_centres_h: tuple[float, ...] | None = None,
) -> pd.DataFrame:
    """Iterate the Bernoulli plug-in model over every relevant arrival.

    For residential: fires on `dest_type == "home"`.
    For workplace: fires on `dest_type == "work"`.

    Returns a DataFrame with the v2.0 plug_in_events schema. Only
    plug-in events with `plug_in == True` are kept (per the revised
    plan §4.4 — "Sample Bernoulli(p_plug); retain plug-in events
    that sample True"). Tests can introspect p_plug via
    `compute_p_plug` directly.
    """
    if mobility.empty:
        return _empty_events_df()

    fleet_ix = fleet.set_index("ev_id")
    b_kwh = fleet_ix["B_kwh"].astype(np.float32)

    # We need to read the local-time hour-of-day. mobility timestamps
    # are UTC (v2.0). Convert to region_tz for the cluster-window check.
    mob = mobility.copy()
    for col in ("start_ts", "end_ts"):
        s = pd.to_datetime(mob[col])
        if getattr(s.dt, "tz", None) is None:
            s = s.dt.tz_localize("UTC")
        mob[col] = s
    mob_local_end = mob["end_ts"].dt.tz_convert(region_tz)
    mob = mob.sort_values(["ev_id", "start_ts"]).reset_index(drop=True)
    mob_local_end = mob_local_end.reindex(mob.index)
    mob["_local_end"] = mob_local_end
    mob["_next_start_ts"] = mob.groupby("ev_id")["start_ts"].shift(-1)

    target_location = _location_label_for(profile_type)
    rows: list[dict[str, object]] = []

    # Build an ev_id → cluster_z lookup that handles permutations.
    fleet_ev_ids = fleet["ev_id"].astype(np.int64).to_numpy()
    ev_to_z: dict[int, int] = {
        int(fleet_ev_ids[i]): int(cluster_z[i]) for i in range(len(cluster_z))
    }

    for ev_id, sub in mob.groupby("ev_id", sort=True):
        ev_id_int = int(ev_id)
        z = int(ev_to_z.get(ev_id_int, 1))
        soc_running = float(bcfg.initial_soc_frac)
        try:
            B = float(b_kwh.loc[ev_id_int])
        except KeyError:
            B = 70.0
        if not np.isfinite(B) or B <= 0:
            B = 70.0

        rng_bern = np.random.default_rng(
            int(master_seed) + _BERNOULLI_SUBSEED_OFFSET + ev_id_int
        )
        rng_target = np.random.default_rng(
            int(master_seed) + _ARRIVAL_JITTER_SUBSEED_OFFSET + ev_id_int
        )

        sub_records = sub.to_dict("records")
        for rec in sub_records:
            kwh = float(rec.get("kwh_consumed", 0.0))
            soc_running = max(bcfg.soc_min_frac, soc_running - kwh / B)
            soc_running = min(bcfg.soc_max_frac, soc_running)

            dest_type = str(rec.get("dest_type", "other"))
            if dest_type != target_location:
                continue

            arrival_ts = rec["end_ts"]
            arrival_local = rec["_local_end"]
            next_start_ts = rec["_next_start_ts"]
            if pd.isna(next_start_ts):
                departure_ts = arrival_ts + pd.Timedelta(
                    hours=bcfg.last_arrival_pseudo_dwell_h
                )
            else:
                departure_ts = next_start_ts
            dwell_h = (departure_ts - arrival_ts).total_seconds() / 3600.0
            if dwell_h < 0:
                dwell_h = 0.0

            is_weekend = bool(
                rec.get("is_weekend", arrival_local.dayofweek in (5, 6))
            )

            t_hour = float(arrival_local.hour) + float(arrival_local.minute) / 60.0
            # v3.0: when a separate weekend centre table is supplied
            # (SPEECh K=16 path), pick it for weekend arrivals; the
            # weekday centres remain authoritative for weekday arrivals.
            # Under the v2 BIC-sweep path ``weekend_cluster_centres_h``
            # is ``None`` and we fall back to the single-table predicate
            # for byte-stable behaviour.
            if weekend_cluster_centres_h is not None and is_weekend:
                _centres_for_predicate = weekend_cluster_centres_h
                _K_for_predicate = len(weekend_cluster_centres_h)
            else:
                _centres_for_predicate = cluster_centres_h
                _K_for_predicate = K_chosen
            in_window = _in_cluster_window(
                t_hour, z=z, centres=_centres_for_predicate,
                K=_K_for_predicate, profile_type=profile_type,
            )

            p_plug = compute_p_plug(
                soc_at_arrival_frac=soc_running,
                is_correct_location=True,
                in_preferred_window=in_window,
                dwell_h=dwell_h,
                is_weekend=is_weekend,
                K=K_chosen,
                z=z,
                profile_type=profile_type,
            )
            u = float(rng_bern.random())
            plugged = bool(u < p_plug)
            if not plugged:
                continue

            # SoC target.
            if profile_type == "workplace":
                lo = max(soc_running, bcfg.workplace_soc_target_lo)
                hi = bcfg.workplace_soc_target_hi
                # TruncatedBeta(α=2, β=2, [lo, hi]).  rng_target → Beta then scale.
                u_beta = float(rng_target.beta(
                    bcfg.workplace_soc_target_alpha,
                    bcfg.workplace_soc_target_beta,
                ))
                soc_target = lo + u_beta * (hi - lo)
            else:
                soc_target = bcfg.soc_recovery_on_plugin

            soc_running = max(soc_running, soc_target)

            location_tag = target_location
            # Future: detect NACS_workplace via fleet["EVSE_kw"] tagging — we
            # do not have a unified EVSE-type column in v2.0-rev so we tag
            # location_type to the profile_type default.

            rows.append(
                {
                    "ev_id": ev_id_int,
                    "arrival_ts": arrival_ts,
                    "departure_ts": departure_ts,
                    "cluster_z": int(z),
                    "location_type": location_tag,
                    "soc_target_pct": float(soc_target),
                    "profile_type_event": profile_type,
                    "p_plug": float(p_plug),
                    "dwell_h": float(dwell_h),
                }
            )

    if not rows:
        return _empty_events_df()
    out = pd.DataFrame(rows)
    out = out.sort_values(["ev_id", "arrival_ts"]).reset_index(drop=True)
    out.insert(0, "event_id", np.arange(len(out), dtype=np.int64))
    return _cast_events_to_schema(out)


def _empty_events_df() -> pd.DataFrame:
    cols = [
        "event_id", "ev_id", "arrival_ts", "departure_ts", "cluster_z",
        "location_type", "soc_target_pct", "profile_type_event",
        "p_plug", "dwell_h",
    ]
    df = pd.DataFrame({c: [] for c in cols})
    return _cast_events_to_schema(df)


def _cast_events_to_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Cast plug_in_events to the canonical v2.0 dtypes.

    RFC-012: M5 now writes the brief schema (line 107) directly —
    ``plug_in:bool`` alongside ``location_type``. M6 consumes this schema
    via the legacy ``location`` synonym which we also emit (as a string
    column) so the shim previously living in ``cache_regen`` and
    ``validator`` is no longer required.
    """
    out = df.copy()
    n = len(out)
    out["event_id"] = (
        out["event_id"].astype(np.int64) if n else np.arange(n, dtype=np.int64)
    )
    out["ev_id"] = out["ev_id"].astype(np.int32) if n else out["ev_id"].astype(np.int32)
    for col in ("arrival_ts", "departure_ts"):
        s = pd.to_datetime(out[col], errors="coerce", utc=False)
        if hasattr(s, "dt") and getattr(s.dt, "tz", None) is None:
            s = s.dt.tz_localize("UTC")
        else:
            s = s.dt.tz_convert("UTC")
        out[col] = s
    out["cluster_z"] = out["cluster_z"].astype(np.int8) if n else out["cluster_z"].astype(np.int8)
    out["location_type"] = pd.Categorical(
        out["location_type"].astype(str) if n else [],
        categories=list(LOCATION_CATEGORIES),
    )
    out["soc_target_pct"] = (
        out["soc_target_pct"].astype(np.float32) if n
        else out["soc_target_pct"].astype(np.float32)
    )
    out["profile_type_event"] = pd.Categorical(
        out["profile_type_event"].astype(str) if n else [],
        categories=list(PROFILE_TYPE_CATEGORIES),
    )
    out["p_plug"] = (
        out["p_plug"].astype(np.float32) if n else out["p_plug"].astype(np.float32)
    )
    out["dwell_h"] = (
        out["dwell_h"].astype(np.float32) if n else out["dwell_h"].astype(np.float32)
    )
    # RFC-012: emit ``plug_in`` (bool) explicitly. Every row that survives
    # the M5 Bernoulli draw is a plug-in event, so the column is True for
    # all rows present in the materialised events frame.
    if "plug_in" in out.columns:
        out["plug_in"] = out["plug_in"].astype(bool)
    else:
        out["plug_in"] = np.ones(n, dtype=bool) if n else np.array([], dtype=bool)
    # ``location`` is the M6-facing synonym for ``location_type`` (string).
    if "location" not in out.columns:
        if n:
            out["location"] = out["location_type"].astype(str)
        else:
            out["location"] = pd.Series([], dtype="object")
    return out[[
        "event_id", "ev_id", "arrival_ts", "departure_ts", "cluster_z",
        "location_type", "soc_target_pct", "profile_type_event",
        "p_plug", "dwell_h", "plug_in", "location",
    ]]


# ---------------------------------------------------------------------------
# Aggregate plug-in rate calculator (acceptance criterion #11).
# ---------------------------------------------------------------------------

def aggregate_plug_in_rate(
    mobility: pd.DataFrame,
    events: pd.DataFrame,
    profile_type: str,
) -> float:
    """Return n_plug_in / n_eligible_arrivals."""
    target_location = _location_label_for(profile_type)
    n_arrivals = int((mobility["dest_type"] == target_location).sum())
    if n_arrivals == 0:
        return 0.0
    return float(len(events)) / float(n_arrivals)


# ---------------------------------------------------------------------------
# Top-level public API: fit + assign in one call.
# ---------------------------------------------------------------------------

def _build_speech_cluster_fit(
    profile_type: str,
    speech: SpeechParams,
) -> ClusterFit:
    """Map a ``SpeechParams`` block into a ``ClusterFit`` suitable for the
    Bernoulli iteration.

    The K=16 SPEECh clusters are indexed by the **union** of weekday and
    weekend present groups (length ``K_chosen``). For each position in
    the union we record the **weekday** centre in ``cluster_centres_h``
    (so legacy single-table consumers see a sane value) and reweight
    ``P_G`` across the union as the cluster-mixing prior. The weekend
    centres travel on the ``SpeechParams`` block for the per-arrival
    predicate.

    Missing-on-weekday groups get centre = NaN, which makes the
    distance-modulo-24 predicate return ``False`` (so those clusters
    never fire on weekday arrivals — exactly the right behaviour per
    SPEECh's ``groups_missing`` semantics).
    """
    weekday_set = set(speech.groups_present_weekday)
    weekend_set = set(speech.groups_present_weekend)
    union = sorted(weekday_set | weekend_set)

    # Per-group centres for weekday and weekend (positional in
    # ``groups_present_*`` order).
    weekday_centres_by_pos = _speech_centres_per_group(
        speech.weekday_means_h, speech.weekday_weights
    )
    weekend_centres_by_pos = _speech_centres_per_group(
        speech.weekend_means_h, speech.weekend_weights
    )
    weekday_centre_by_group: dict[int, float] = {
        g: weekday_centres_by_pos[i]
        for i, g in enumerate(speech.groups_present_weekday)
    }
    weekend_centre_by_group: dict[int, float] = {
        g: weekend_centres_by_pos[i]
        for i, g in enumerate(speech.groups_present_weekend)
    }

    # Project per-day centres + weights onto the union ordering. Missing
    # on a day → NaN centre (predicate returns False) and zero weight
    # (cluster does not fire). For ``cluster_centres_h`` (the legacy
    # single-table field) we pick weekday; if a group is missing on
    # weekday but present on weekend, we use the weekend centre so the
    # single-table fall-back is at least sensible (consumers that do
    # not thread weekend separately will still see the right hour for
    # weekend-only groups).
    cluster_centres_h_list: list[float] = []
    for g in union:
        if g in weekday_centre_by_group:
            cluster_centres_h_list.append(weekday_centre_by_group[g])
        elif g in weekend_centre_by_group:
            cluster_centres_h_list.append(weekend_centre_by_group[g])
        else:
            cluster_centres_h_list.append(float("nan"))

    # ``cluster_weights`` is the cluster prior under the v3 SPEECh path
    # = P(G) reweighted over the union. ``SpeechParams.P_G`` already
    # carries that vector but indexed by raw group_id 0..15; project
    # onto the union order. Renormalise defensively.
    p_g_full = speech.P_G  # length 16, mass only on union groups
    cluster_weights_list = [float(p_g_full[g]) for g in union]
    total = sum(cluster_weights_list)
    if total > 0.0:
        cluster_weights_list = [w / total for w in cluster_weights_list]
    else:
        cluster_weights_list = [1.0 / max(len(union), 1)] * len(union)

    K_chosen = len(union)
    cluster_windows_list = tuple(
        2.0 * CLUSTER_WINDOW_HALF_H for _ in range(K_chosen)
    )

    return ClusterFit(
        profile_type=profile_type,
        K_sweep=(SPEECH_K_TOTAL,),
        K_chosen=int(K_chosen),
        bic_chosen=float("nan"),
        bic_per_K={int(SPEECH_K_TOTAL): float("nan")},
        cluster_centres_h=tuple(cluster_centres_h_list),
        cluster_windows_h=cluster_windows_list,
        cluster_weights=tuple(cluster_weights_list),
        ambiguous_flag=False,
        degenerate_flag=False,
        rejected_K=(),
        rejection_reason_per_K={},
        note=(
            "SPEECh K=16 published GMM (Powell, Cezar, Rajagopal 2022 "
            "Applied Energy 309; Mendeley gvk34mybtb/1; CC-BY-4.0). "
            "v3.0 consumes the start-time marginal + P(G) prior; "
            f"K_chosen={K_chosen} reflects the union of weekday + "
            "weekend present-group sets per segment."
        ),
        bic_override_reason=None,
        axes=("start_h_LT",),
        speech_params=speech,
    )


def _speech_weekend_centres_from_fit(cluster_fit: ClusterFit) -> tuple[float, ...] | None:
    """Return the weekend per-cluster centres aligned to the union ordering.

    Mirrors the projection in :func:`_build_speech_cluster_fit`. Returns
    ``None`` when ``cluster_fit.speech_params`` is ``None`` (v2 BIC-
    sweep path).
    """
    speech = cluster_fit.speech_params
    if speech is None:
        return None
    weekday_set = set(speech.groups_present_weekday)
    weekend_set = set(speech.groups_present_weekend)
    union = sorted(weekday_set | weekend_set)
    weekend_centres_by_pos = _speech_centres_per_group(
        speech.weekend_means_h, speech.weekend_weights
    )
    weekend_centre_by_group: dict[int, float] = {
        g: weekend_centres_by_pos[i]
        for i, g in enumerate(speech.groups_present_weekend)
    }
    centres = []
    for g in union:
        if g in weekend_centre_by_group:
            centres.append(weekend_centre_by_group[g])
        else:
            centres.append(float("nan"))
    return tuple(centres)


def fit_and_assign_plug_ins(
    fleet: pd.DataFrame,
    mobility: pd.DataFrame,
    region: Region | str,
    profile_type: str,
    *,
    cluster_source: Literal["speech_k16", "bic_sweep"] = "speech_k16",
    cohort_parquet: Path | None = None,
    sessions_csv: Path | None = None,
    vehicles_csv: Path | None = None,
    seed: int = MASTER_SEED_DEFAULT,
) -> tuple[pd.DataFrame, pd.DataFrame, ClusterFit]:
    """Run the full M5 pipeline on one (region, profile_type) cache.

    Parameters
    ----------
    cluster_source
        v3.0 default ``"speech_k16"`` loads the published Powell-2022
        SPEECh K=16 mixture parameters from
        ``data/pev/raw/speech/speech_k16_<profile_type>.json`` and uses
        them in place of the in-house BIC sweep. Pass
        ``"bic_sweep"`` for back-compat with v2.1 (BIC over EVWatts).
        ``legacy_v2_1_compat=True`` in :mod:`cache_regen` flips this to
        ``"bic_sweep"`` for one CI canary cache.

    Returns
    -------
    (augmented_fleet, plug_in_events, cluster_fit)

    ``augmented_fleet`` has ``cluster_z``, ``cluster_k``, ``bic_chosen``
    columns appended. ``plug_in_events`` is the UTC-stored event ledger.
    ``cluster_fit`` is the BIC sweep summary (v2 path) or the SPEECh
    K=16 projection (v3 path) for ``meta.json`` provenance.
    """
    if isinstance(region, str):
        region_obj = Region.from_name(region)
    elif isinstance(region, Region):
        region_obj = region
    else:
        raise TypeError(f"region must be Region or str; got {type(region).__name__}")

    if profile_type not in ("residential", "workplace"):
        raise ValueError(
            f"profile_type must be 'residential' or 'workplace'; got {profile_type!r}"
        )
    if cluster_source not in ("speech_k16", "bic_sweep"):
        raise ValueError(
            f"cluster_source must be 'speech_k16' or 'bic_sweep'; got "
            f"{cluster_source!r}"
        )

    # ---- 1. Cluster fit (SPEECh K=16 by default; BIC sweep on opt-in) ----
    if cluster_source == "speech_k16":
        speech = load_speech_k16_params(profile_type)
        cluster_fit = _build_speech_cluster_fit(profile_type, speech)
    else:
        # Legacy v2.1 path: BIC sweep on the appropriate cohort.
        if profile_type == "workplace":
            if cohort_parquet is None:
                cohort_parquet = DEFAULT_WORKPLACE_COHORT_REL
            cohort = _load_workplace_cohort(Path(cohort_parquet))
            cluster_fit = run_bic_sweep(
                cohort,
                profile_type="workplace",
                feature_cols=("plug_in_hr_med", "dur_h_med", "avg_kw_med"),
                K_sweep=K_SWEEP_WORKPLACE,
                seed=seed,
            )
        else:
            if sessions_csv is None:
                sessions_csv = DEFAULT_RESIDENTIAL_SESSIONS_REL
            if vehicles_csv is None:
                vehicles_csv = DEFAULT_RESIDENTIAL_VEHICLES_REL
            sessions = _load_residential_evwatts(Path(sessions_csv), Path(vehicles_csv))
            cluster_fit = run_bic_sweep(
                sessions,
                profile_type="residential",
                feature_cols=("start_hour", "duration_h", "energy_kwh"),
                K_sweep=K_SWEEP_RESIDENTIAL,
                seed=seed,
                cluster_window_max_h=float("inf"),  # W9 not applied to residential
            )

    K_chosen = cluster_fit.K_chosen

    # Override mixing weights with the fit weights if not degenerate.
    base_pi = (
        cluster_fit.cluster_weights
        if not cluster_fit.degenerate_flag
        else None
    )

    # ---- 2. Per-EV cluster assignment ----
    cluster_z = assign_clusters(
        fleet, K_chosen=K_chosen, profile_type=profile_type,
        master_seed=seed, base_pi=base_pi,
    )

    # ---- 3. Bernoulli iteration ----
    cluster_centres_h: tuple[float, ...] = cluster_fit.cluster_centres_h
    if len(cluster_centres_h) != K_chosen:
        cluster_centres_h = tuple(
            (cluster_centres_h + (12.0,) * K_chosen)[:K_chosen]
        )
    weekend_centres = _speech_weekend_centres_from_fit(cluster_fit)
    events = iterate_bernoulli(
        mobility=mobility,
        fleet=fleet,
        cluster_z=cluster_z,
        K_chosen=K_chosen,
        profile_type=profile_type,
        cluster_centres_h=cluster_centres_h,
        region_tz=region_obj.tz if region_obj.tz != "UTC" else "America/Los_Angeles",
        master_seed=seed,
        weekend_cluster_centres_h=weekend_centres,
    )

    # ---- 4. Augment fleet ----
    augmented = fleet.copy()
    # Re-align cluster_z to the original fleet row order. K_chosen can be
    # up to 16 under the SPEECh path; int8 still holds it (max 127).
    augmented["cluster_z"] = cluster_z.astype(np.int8)
    augmented["cluster_k"] = np.int8(K_chosen)
    augmented["bic_chosen"] = np.float32(
        cluster_fit.bic_chosen if np.isfinite(cluster_fit.bic_chosen) else float("nan")
    )

    return augmented, events, cluster_fit


# ---------------------------------------------------------------------------
# Persistence.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class M5Paths:
    """Resolved input/output paths."""

    fleet_in: Path
    mobility_in: Path
    workplace_cohort_parquet: Path
    evwatts_sessions_csv: Path
    evwatts_vehicles_csv: Path
    fleet_out: Path
    plug_in_events_out: Path
    bic_sweep_out: Path
    meta_out: Path


def _default_paths(repo_root: Path, region_name: str, profile_type: str) -> M5Paths:
    base = repo_root / "data" / "pev"
    syn = base / "processed" / region_name / f"{profile_type}_ev_synth"
    return M5Paths(
        fleet_in=syn / "fleet.parquet",
        mobility_in=syn / "mobility.parquet",
        workplace_cohort_parquet=base / "raw" / "evwatts_cohorts" / "workplace_subset.parquet",
        evwatts_sessions_csv=base / "raw" / "evwatts.public.vehiclesessions.csv",
        evwatts_vehicles_csv=base / "raw" / "evwatts.public.vehicles.csv",
        fleet_out=syn / "fleet.parquet",
        plug_in_events_out=syn / "plug_in_events.parquet",
        # RFC-011.r: BIC sweep file lives at the cache root (no
        # ``intermediate/`` subdir) and uses the literal filename
        # ``bic_sweep_workplace.json``. Residential caches MUST NOT
        # emit a ``bic_sweep_*.json`` artefact; the path is still
        # populated here for record-keeping but writes are gated in
        # ``run()`` by ``profile_type == "workplace"``.
        bic_sweep_out=syn / "bic_sweep_workplace.json",
        meta_out=syn / "meta.json",
    )


def write_plug_in_events(df: pd.DataFrame, out_path: str | os.PathLike[str]) -> Path:
    """Write `plug_in_events.parquet` (sorted by ev_id, arrival_ts)."""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    cast = _cast_events_to_schema(df)
    cast = cast.sort_values(["ev_id", "arrival_ts"]).reset_index(drop=True)
    cast.to_parquet(p, engine="pyarrow", index=False, compression="snappy")
    return p


def write_fleet(df: pd.DataFrame, out_path: str | os.PathLike[str]) -> Path:
    """Write extended fleet parquet (sorted by ev_id)."""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    out = df.sort_values("ev_id").reset_index(drop=True)
    out.to_parquet(p, engine="pyarrow", index=False, compression="snappy")
    return p


def write_bic_sweep_results(
    cluster_fit: ClusterFit,
    out_path: str | os.PathLike[str],
) -> Path:
    """Write ``bic_sweep_workplace.json``.

    RFC-011.r: residential ``ClusterFit`` instances MUST NOT produce a
    ``bic_sweep_*.json`` artefact. When called with a non-workplace
    fit, this function is a no-op and returns ``Path(out_path)`` without
    creating any file. The guard lives here (not at the call sites)
    so any caller (``run``, ``cache_regen.regenerate_cache``, …) is
    protected uniformly.
    """
    p = Path(out_path)
    if cluster_fit.profile_type != "workplace":
        return p
    # Force the literal filename when callers pass a residential-
    # flavoured path (e.g. ``bic_sweep_residential.json``).
    if p.name != "bic_sweep_workplace.json":
        p = p.with_name("bic_sweep_workplace.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "profile_type": cluster_fit.profile_type,
        "K_sweep": list(cluster_fit.K_sweep),
        "K_chosen": int(cluster_fit.K_chosen),
        "bic_chosen": (
            float(cluster_fit.bic_chosen)
            if np.isfinite(cluster_fit.bic_chosen)
            else None
        ),
        "bic_per_K": {
            f"K={k}": (None if not np.isfinite(v) else float(v))
            for k, v in cluster_fit.bic_per_K.items()
        },
        "cluster_centres_h": list(cluster_fit.cluster_centres_h),
        "cluster_windows_h": list(cluster_fit.cluster_windows_h),
        "cluster_weights": list(cluster_fit.cluster_weights),
        "ambiguous_flag": bool(cluster_fit.ambiguous_flag),
        "degenerate_flag": bool(cluster_fit.degenerate_flag),
        "rejected_K": list(cluster_fit.rejected_K),
        "rejection_reason_per_K": {
            f"K={k}": v for k, v in cluster_fit.rejection_reason_per_K.items()
        },
        "note": cluster_fit.note,
        "bic_override_reason": cluster_fit.bic_override_reason,
    }
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=_json_default)
    return p


def _json_default(o: object) -> object:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        v = float(o)
        if not np.isfinite(v):
            return None
        return v
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"unserialisable: {type(o)}")


def update_meta_with_m5(
    meta_path: str | os.PathLike[str],
    *,
    profile_type: str,
    region: Region,
    cluster_fit: ClusterFit,
    cluster_size_distribution: dict[int, int],
    aggregate_plug_in_rate_value: float,
    cohort_size: int,
) -> Path:
    """Append the M5 v2.0 provenance block to meta.json."""
    p = Path(meta_path)
    if p.is_file():
        with p.open("r", encoding="utf-8") as f:
            meta = json.load(f)
    else:
        meta = {}

    beta_block = _beta_table(profile_type)
    beta_0 = [
        _beta_0_for_cluster(profile_type, cluster_fit.K_chosen, z)
        for z in range(1, cluster_fit.K_chosen + 1)
    ]

    block: dict[str, object] = {
        "methodology_version": METHODOLOGY_VERSION,
        "module": "pev_synth.plug_in_model",
        "region_name": region.name,
        "profile_type": profile_type,
        "K_chosen_by_BIC": int(cluster_fit.K_chosen),
        "bic_chosen": (
            float(cluster_fit.bic_chosen)
            if np.isfinite(cluster_fit.bic_chosen)
            else None
        ),
        "bic_sweep_K_values": list(cluster_fit.K_sweep),
        "bic_sweep_ambiguous_flag": bool(cluster_fit.ambiguous_flag),
        "bic_sweep_degenerate_flag": bool(cluster_fit.degenerate_flag),
        # RFC-037: when the EVWatts cohort's fit centroids drift outside
        # the commuter window [7.0, 10.5], BIC's K is overridden with
        # the commuter-prior K=3 anchor and this key surfaces the reason.
        "bic_override_reason": cluster_fit.bic_override_reason,
        "cluster_centres_h": list(cluster_fit.cluster_centres_h),
        "cluster_windows_h": list(cluster_fit.cluster_windows_h),
        "cluster_weights": list(cluster_fit.cluster_weights),
        "cluster_size_distribution": {
            str(k): int(v) for k, v in sorted(cluster_size_distribution.items())
        },
        "cohort_size": int(cohort_size),
        "aggregate_plug_in_rate": float(aggregate_plug_in_rate_value),
        "beta_priors": {
            "beta_0": beta_0,
            "beta_1": beta_block["beta_1"],
            "beta_2": beta_block["beta_2"],
            "beta_3": beta_block["beta_3"],
            "beta_4": beta_block["beta_4"],
            "beta_5": beta_block["beta_5"],
        },
        "provenance_table": [
            {
                "parameter": r.parameter,
                "value": r.value,
                "source": r.source,
                "fit_status": r.fit_status,
            }
            for r in PROVENANCE
        ],
        "honest_citations": {
            "munkhammar_2015": "Applied Energy 142:135-143 (Bernoulli per-arrival plug-in)",
            "quiros_tortos_2018": "IEEE PSCC doc 8442988 (residential weekday PDFs)",
            "lee_li_low_2019": "ACM e-Energy (GMM clustering on ACN-Data; workplace canonical)",
            "pritchard_2023": "NREL/CP-5400-85902 (workplace L2 venue statistics)",
            "kreft_2024": (
                "Kreft, Brudermueller, Fleisch & Staake 2024 "
                "Applied Energy 370:123544 (workplace β₁ anchor) — "
                "NOT 'Köhler 2024' (citation fixed per RC-2)"
            ),
        },
        "explicit_non_provenance": (
            [
                "Not Pecan Street numeric mixing weights — Pecan publishes "
                "only a qualitative 3-cluster split.",
            ]
            if cluster_fit.speech_params is not None
            else [
                "Not Powell-Cezar-Rajagopal 2022 SPEECh (RC-1).",
                "Not Pecan Street numeric mixing weights — Pecan publishes "
                "only a qualitative 3-cluster split.",
            ]
        ),
        "cluster_source": (
            "speech_k16"
            if cluster_fit.speech_params is not None
            else "bic_sweep"
        ),
        "cluster_axes": list(cluster_fit.axes),
    }

    if cluster_fit.speech_params is not None:
        sp = cluster_fit.speech_params
        block["speech_k16_provenance"] = {
            "citation": (
                "Powell, S.; Cezar, G.V.; Rajagopal, R. (2022) Scalable "
                "Probabilistic Estimates of Electric Vehicle Charging Given "
                "Observed Driver Behavior. Applied Energy 309, "
                "doi:10.1016/j.apenergy.2021.118382"
            ),
            "url": sp.source_url,
            "doi": sp.source_doi,
            "license": sp.source_license,
            "segment_source": (
                "home" if profile_type == "residential" else "work"
            ),
            "groups_present_weekday": list(sp.groups_present_weekday),
            "groups_present_weekend": list(sp.groups_present_weekend),
            "axes_consumed_v3_0": ["start_h_LT"],
            "joint_dwell_energy_sampling": "deferred to v3.1",
        }

    if profile_type == "workplace":
        block["workplace_cohort_caveat"] = (
            "Workplace EVWatts cohort (N=105) is a daytime-charging-fleet-mix; "
            "its plug-in median (~12:00 LT) is approximately 3h later than "
            "the literature-canonical workplace median of ~09:00. v2.0 "
            "cluster centres fit from this cohort skew midday. W1-W4 "
            "validator checks may flag this divergence as EXPLAINED_FAIL."
        )
        block["W9_invariant"] = (
            f"Every selected cluster has 95% start-time window ≤ "
            f"{WORKPLACE_CLUSTER_WINDOW_MAX_H:.1f} h"
        )
        block["W9_max_cluster_window_h"] = float(
            max(cluster_fit.cluster_windows_h) if cluster_fit.cluster_windows_h else 0.0
        )
        meta_workplace = meta.setdefault("workplace", {})
        meta_workplace.update({
            "bic_sweep": {
                f"K={k}": (None if not np.isfinite(v) else float(v))
                for k, v in cluster_fit.bic_per_K.items()
            },
            "cohort_size": int(cohort_size),
            "plug_in_rate_observed": float(aggregate_plug_in_rate_value),
            "beta_priors": block["beta_priors"],
            "selected": int(cluster_fit.K_chosen),
            "rejected_K": list(cluster_fit.rejected_K),
        })

    meta["plug_in_model"] = block
    meta.setdefault("module_provenance", {})["m5_plug_in_model"] = METHODOLOGY_VERSION

    pkg_versions = meta.get("package_versions", {})
    pkg_versions.setdefault("numpy", np.__version__)
    pkg_versions.setdefault("pandas", pd.__version__)
    pkg_versions.setdefault("python", platform.python_version())
    try:
        import pyarrow
        pkg_versions["pyarrow"] = pyarrow.__version__
    except ImportError:  # pragma: no cover
        pkg_versions.setdefault("pyarrow", "unknown")
    pkg_versions["scikit-learn"] = sklearn.__version__
    meta["package_versions"] = pkg_versions
    meta["run_timestamp_utc"] = datetime.now(timezone.utc).isoformat()

    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True, default=_json_default)
    return p


# ---------------------------------------------------------------------------
# End-to-end CLI driver.
# ---------------------------------------------------------------------------

def run(
    paths: M5Paths,
    region: Region | str,
    profile_type: str,
    master_seed: int = MASTER_SEED_DEFAULT,
) -> dict[str, object]:
    """End-to-end M5 v2.0 driver."""
    region_obj = Region.from_name(region) if isinstance(region, str) else region

    if not paths.fleet_in.is_file():
        raise FileNotFoundError(f"fleet.parquet not found at {paths.fleet_in}")
    if not paths.mobility_in.is_file():
        raise FileNotFoundError(f"mobility.parquet not found at {paths.mobility_in}")

    fleet = pd.read_parquet(paths.fleet_in)
    mobility = pd.read_parquet(paths.mobility_in)

    cohort_size = 0
    if profile_type == "workplace" and paths.workplace_cohort_parquet.is_file():
        cohort_size = int(len(pd.read_parquet(paths.workplace_cohort_parquet)))

    augmented_fleet, events, cluster_fit = fit_and_assign_plug_ins(
        fleet=fleet,
        mobility=mobility,
        region=region_obj,
        profile_type=profile_type,
        cohort_parquet=paths.workplace_cohort_parquet,
        sessions_csv=paths.evwatts_sessions_csv,
        vehicles_csv=paths.evwatts_vehicles_csv,
        seed=master_seed,
    )

    # RFC-011.r: only workplace caches emit ``bic_sweep_workplace.json``;
    # residential MUST NOT produce a ``bic_sweep_*.json`` artefact.
    if profile_type == "workplace":
        write_bic_sweep_results(cluster_fit, paths.bic_sweep_out)
    write_fleet(augmented_fleet, paths.fleet_out)
    write_plug_in_events(events, paths.plug_in_events_out)

    rate = aggregate_plug_in_rate(mobility, events, profile_type)
    cluster_size_distribution = {
        int(k): int(v)
        for k, v in pd.Series(augmented_fleet["cluster_z"]).value_counts().sort_index().items()
    }

    update_meta_with_m5(
        paths.meta_out,
        profile_type=profile_type,
        region=region_obj,
        cluster_fit=cluster_fit,
        cluster_size_distribution=cluster_size_distribution,
        aggregate_plug_in_rate_value=rate,
        cohort_size=cohort_size,
    )

    return {
        "region": region_obj.name,
        "profile_type": profile_type,
        "K_chosen": int(cluster_fit.K_chosen),
        "bic_chosen": (
            float(cluster_fit.bic_chosen)
            if np.isfinite(cluster_fit.bic_chosen)
            else None
        ),
        "n_arrivals_eligible": int(
            (mobility["dest_type"] == _location_label_for(profile_type)).sum()
        ),
        "n_plug_in_events": int(len(events)),
        "aggregate_plug_in_rate": float(rate),
        "cluster_size_distribution": cluster_size_distribution,
        "cluster_centres_h": list(cluster_fit.cluster_centres_h),
        "cluster_windows_h": list(cluster_fit.cluster_windows_h),
        "rejected_K": list(cluster_fit.rejected_K),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pev_synth.plug_in_model",
        description="M5 v2.0: Bernoulli plug-in behaviour model (residential + workplace).",
    )
    parser.add_argument("--seed", type=int, default=MASTER_SEED_DEFAULT)
    parser.add_argument(
        "--region", type=str, default="bay_area",
        choices=sorted(REGIONS.keys()),
    )
    parser.add_argument(
        "--profile-type", type=str, default="residential",
        choices=("residential", "workplace"),
    )
    parser.add_argument(
        "--repo-root", type=str, default=None,
    )
    args = parser.parse_args(argv)

    from ._paths import repo_root as _repo_root_fn  # noqa: PLC0415

    repo_root = (
        Path(args.repo_root).resolve() if args.repo_root else _repo_root_fn()
    )
    paths = _default_paths(repo_root, args.region, args.profile_type)
    summary = run(
        paths,
        region=args.region,
        profile_type=args.profile_type,
        master_seed=int(args.seed),
    )

    print(
        "M5 v2.0: region={r} profile={p} K_chosen={K} "
        "n_arrivals={n} n_plugged={pi} rate={rt:.3f}".format(
            r=summary["region"],
            p=summary["profile_type"],
            K=summary["K_chosen"],
            n=summary["n_arrivals_eligible"],
            pi=summary["n_plug_in_events"],
            rt=summary["aggregate_plug_in_rate"],
        ),
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "BETA_0_BY_K_RESIDENTIAL",
    "BETA_0_BY_K_WORKPLACE",
    "BETA_RESIDENTIAL",
    "BETA_WORKPLACE",
    "BIC_AMBIGUOUS_REL_GAP",
    "CLUSTER_AXES_V2_DEFAULT",
    "CLUSTER_WINDOW_HALF_H",
    "DEFAULT_K_RESIDENTIAL",
    "DEFAULT_K_WORKPLACE",
    "DWELL_LOG_FLOOR_H",
    "IRREGULAR_CLUSTER_IDS_BY_K",
    "K_SWEEP_RESIDENTIAL",
    "K_SWEEP_WORKPLACE",
    "MASTER_SEED_DEFAULT",
    "METHODOLOGY_VERSION",
    "MODULE_OFFSET",
    "PI_BASE_RESIDENTIAL",
    "PI_BASE_WORKPLACE",
    "PROFILE_TYPE_CATEGORIES",
    "PROVENANCE",
    "SPEECH_K16_RESIDENTIAL_REL",
    "SPEECH_K16_WORKPLACE_REL",
    "SPEECH_K_TOTAL",
    "STORAGE_TZ",
    "WINDOW_CENTRES_RESIDENTIAL",
    "WINDOW_CENTRES_WORKPLACE_PRIOR",
    "WORKPLACE_CLUSTER_WINDOW_MAX_H",
    "BernoulliConfig",
    "ClusterFit",
    "LOCATION_CATEGORIES",
    "M5Paths",
    "SpeechParams",
    "aggregate_plug_in_rate",
    "assign_clusters",
    "compute_p_plug",
    "fit_and_assign_plug_ins",
    "iterate_bernoulli",
    "load_speech_k16_params",
    "main",
    "run",
    "run_bic_sweep",
    "write_bic_sweep_results",
    "write_fleet",
    "write_plug_in_events",
]
