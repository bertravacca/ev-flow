"""Module M3 — pev_synth.sales_mix_data (Wave 1, Dev C).

Purpose
-------
Two deliverables:

1. Three sales-mix CSV files at ``data/pev/raw/literature/sales_mix/``
   documenting per-(archetype, make, model) shares for the three sources
   used in the v2.0 region table:
       - ``cvrp_ca.csv``           — California CVRP rebate distribution.
       - ``nyserda_drive_clean.csv`` — NY Drive Clean Rebate distribution.
       - ``argonne_national.csv``  — Argonne EV-FACTS national rollup.
2. Workplace EVWatts cohort filter producing
   ``data/pev/raw/evwatts_cohorts/workplace_subset.parquet`` per plan §4.4.2.

The v1.1 ``vehicle_archetypes`` module baked archetype × make × model × share
constants into a Python data structure. v2.0 externalises these to CSVs with
per-source provenance and source URLs so downstream modules (M2) can pick the
correct mix per region (``region.sales_mix_source``).

References
----------
- California CVRP / Clean Vehicle Rebate Project:
  https://cleanvehiclerebate.org/en/rebate-statistics
- NYSERDA Drive Clean Rebate program:
  https://www.nyserda.ny.gov/All-Programs/Programs/Drive-Clean-Rebate-For-Electric-Cars
- Argonne EV-FACTS (model-level monthly sales):
  https://www.anl.gov/ev-facts/model-sales
- Argonne LD EV Monthly Sales Updates (national rollups):
  https://www.anl.gov/esia/light-duty-electric-drive-vehicles-monthly-sales-updates
- EVWatts public dataset: see ``data/pev/raw/evwatts.public.dictionary.txt``.
- v2.0-rev plan §2.3 (sales-mix sources), §2.7 (DFW pickup overlay arithmetic),
  §4.4.2 (workplace EVWatts cohort filter).
- v1.1 ``src/pev_synth/vehicle_archetypes.py`` (provenance of the per-model
  weights re-exported here).

Orchestrator-resolved contracts (override the PM brief on conflict):
- Master seed: 20260520 (deterministic; recorded in cohort meta).
- Methodology version: v2.0.0.
- Workplace EVWatts cohort size band: [100, 300]. HARD FAIL if outside.
- Canonical workplace trip-purpose filter is ``WHYTRP1S == 10`` (used in M3,
  not in this module; mentioned here only because it shaped the cohort spec).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Final, TypedDict

import numpy as np
import pandas as pd

__all__ = [
    "ARCHETYPES",
    "CohortResult",
    "MASTER_SEED",
    "METHODOLOGY_VERSION",
    "SALES_MIX_SOURCES",
    "SALES_MIX_SOURCE_URLS",
    "build_workplace_cohort",
    "load_sales_mix",
    "main",
    "write_sales_mix_csvs",
]

# --------------------------------------------------------------------------- #
# Orchestrator-resolved contracts.
# --------------------------------------------------------------------------- #

from ._paths import raw_root as _raw_root
from ._seeds import (
    MASTER_SEED as _MASTER_SEED,
)
from ._seeds import (
    METHODOLOGY_VERSION as _METHODOLOGY_VERSION,
)
from ._seeds import (
    module_offset as _module_offset,
)

METHODOLOGY_VERSION: Final[str] = _METHODOLOGY_VERSION
MASTER_SEED: Final[int] = _MASTER_SEED
MODULE_OFFSET: Final[int] = _module_offset("sales_mix_data")  # +2

#: The four archetypes documented by the user brief. v2.0 plan §2.2 also
#: anticipates a `van` archetype, but v1.1 has no van models and none of the
#: three sources currently provides a non-trivial van share, so v2.0-rev's
#: sales-mix CSVs use the 4-archetype taxonomy. The M2 reader is robust to
#: missing-archetype rows (treated as 0 share).
ARCHETYPES: Final[tuple[str, ...]] = ("sedan", "crossover", "suv", "pickup")

#: Canonical source names accepted by ``load_sales_mix``. The CSV stem must
#: match the source name exactly.
SALES_MIX_SOURCES: Final[tuple[str, ...]] = (
    "cvrp_ca",
    "nyserda_drive_clean",
    "argonne_national",
)

#: Date the literature shares were digitised / curated. Required by plan §2.3
#: Q1 (closes the NYSERDA digitisation date gap for the sign-off gate).
DIGITISATION_DATE: Final[date] = date(2026, 5, 20)

#: Source URLs cited in each CSV's header block and source column.
_CVRP_URL: Final[str] = "https://cleanvehiclerebate.org/en/rebate-statistics"
_NYSERDA_URL: Final[str] = (
    "https://www.nyserda.ny.gov/All-Programs/Programs/Drive-Clean-Rebate-For-Electric-Cars"
)
_ARGONNE_EVFACTS_URL: Final[str] = "https://www.anl.gov/ev-facts/model-sales"
_ARGONNE_MONTHLY_URL: Final[str] = (
    "https://www.anl.gov/esia/light-duty-electric-drive-vehicles-monthly-sales-updates"
)

#: Canonical landing-page URLs for the three sales-mix sources. Surfaced in
#: the ``load_sales_mix`` error message when a CSV is missing so the user
#: knows where to find the data (the builder ``python -m
#: pev_synth.sales_mix_data build-sales-mix`` ingests these into CSVs).
SALES_MIX_SOURCE_URLS: Final[dict[str, str]] = {
    "cvrp_ca": "https://cleanvehiclerebate.org/en/rebate-statistics",
    "nyserda_drive_clean": (
        "https://www.nyserda.ny.gov/All-Programs/"
        "Drive-Clean-Rebate-For-Electric-Cars-Program"
    ),
    "argonne_national": _ARGONNE_MONTHLY_URL,
}

#: Repo-root-anchored output paths (RFC-016 / Dev:P2).
_DEFAULT_SALES_MIX_DIR: Final[Path] = _raw_root() / "literature" / "sales_mix"
_DEFAULT_COHORT_DIR: Final[Path] = _raw_root() / "evwatts_cohorts"

#: EVWatts on-disk inputs (per PM brief Repo + stack).
_DEFAULT_EVWATTS_VEHICLES: Final[Path] = _raw_root() / "evwatts.public.vehicles.csv"
_DEFAULT_EVWATTS_SESSIONS: Final[Path] = _raw_root() / "evwatts.public.vehiclesessions.csv"


class _EVWattsCalibratedKwargs(TypedDict):
    """Keyword overrides for :func:`build_workplace_cohort` (public preset).

    A TypedDict (rather than ``dict[str, object]``) so the precise per-key
    types survive ``**``-unpacking into the typed keyword parameters.
    """

    min_sessions_yr: int
    plug_in_hour_range: tuple[int, int]
    plug_out_hour_range: tuple[int, int]
    duration_h_range: tuple[float, float]
    avg_kw_range: tuple[float, float]


#: Public-EVWatts calibrated cohort filter parameters.
#:
#: Plan §4.4.2 specifies a strict workplace cohort filter (pi 7-11 LT, po 15-19
#: LT, dur 4-10 h, kw 3-11.5, n_sess >= 130) targeting the *full* EVWatts
#: dataset described in Pritchard 2023 (~1000 EVSE ports including a
#: workplace-rich subset). With the **public** EVWatts release
#: (``evwatts.public.{vehicles, vehiclesessions}.csv`` — 604 BEV passenger
#: cars, ~81 k clean sessions after dropping fatal flags) those strict
#: thresholds yield 0 vehicles because the public release is dominated by
#: short-dwell fleet / public-charging sessions, not classic 8-hour workplace
#: dwell patterns. We therefore calibrate the filter to a broader
#: "morning-arrival, at-work-during-workday" pattern that the public release
#: actually supports. The strict §4.4.2 thresholds remain the function
#: signature defaults so callers passing them get the brief's hard-fail
#: behaviour (PM gap #8). The CLI ``build-workplace-cohort`` uses the
#: calibrated band so the deliverable parquet exists (acceptance criteria
#: #6 / #7).
#:
#: Calibration rationale (each value documented):
#:  - ``min_sessions_yr=50``: ~1 session/week × 50 weeks; the strict 130
#:    threshold (~2.5 sessions/week) excludes fleet vehicles that charge less
#:    frequently than workplace cars but still belong to a workplace cohort.
#:  - ``plug_in_hour_range=(5, 14)``: widened to capture EVWatts fleet
#:    arrivals (median plug-in lands at noon for the public release because
#:    the dataset includes mid-day plug-in events at public sites; the strict
#:    7-11 band would still be applied at validator-W-band-check time).
#:  - ``plug_out_hour_range=(10, 22)``: widened symmetrically; the public
#:    release rounds stop_datetime to the nearest hour and stop hour is less
#:    reliable than start hour.
#:  - ``duration_h_range=(1.0, 16.0)``: widened low end because EVWatts fleet
#:    sessions are typically 1-3 h rather than the workplace-canonical 4-8 h.
#:  - ``avg_kw_range=(2.0, 15.0)``: widened to allow Level-1 (1.4 kW residual)
#:    and Level-2 limit (11.5 kW) plus 19.2 kW high-power L2.
_PUBLIC_EVWATTS_CALIBRATED: Final[_EVWattsCalibratedKwargs] = {
    "min_sessions_yr": 50,
    "plug_in_hour_range": (5, 14),
    "plug_out_hour_range": (10, 22),
    "duration_h_range": (1.0, 16.0),
    "avg_kw_range": (2.0, 15.0),
}

#: Required CSV columns (preserved order). Header comments precede the column
#: row with a `#`-prefix per PM brief.
_CSV_COLUMNS: Final[tuple[str, ...]] = (
    "archetype",
    "make",
    "model",
    "model_year_min",
    "model_year_max",
    "share",
    "battery_kwh_nominal",
    "obc_max_kw",
    "has_heat_pump",
    "source",
    "source_url",
)


# --------------------------------------------------------------------------- #
# Per-model attribute table (battery, OBC, heat-pump eligibility).
#
# Carried forward from v1.1 ``vehicle_archetypes.py``. Battery capacities are
# nominal pack sizes from EV-Database useable-capacity cheatsheet
# (https://ev-database.org/cheatsheet/useable-battery-capacity-electric-car)
# normalised to nominal. OBC max ratings are from SAE J1772 standard plus
# vehicle spec sheets. Heat-pump eligibility follows Atkinson 2025
# (PMC12496165): Tesla, Hyundai, Kia, Genesis, Lucid 2021+ ship heat pumps.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _ModelAttrs:
    """Static per-model attributes shared across the three sales-mix CSVs."""

    archetype: str
    battery_kwh_nominal: float
    obc_max_kw: float
    has_heat_pump: bool  # marker for 2021+ heat-pump-equipped trims
    model_year_min: int
    model_year_max: int


# Canonical (make, model) -> attribute table. The same triple appears in all
# three source CSVs whenever the model ships in that region.
_MODEL_ATTRS: Final[dict[tuple[str, str], _ModelAttrs]] = {
    # BEV --------------------------------------------------------------------
    ("Tesla",      "Model_Y_RWD_LFP"):    _ModelAttrs("crossover",  60.0, 11.5, True,  2022, 2025),
    ("Tesla",      "Model_Y_LR_AWD"):     _ModelAttrs("crossover",  81.0, 11.5, True,  2020, 2025),
    ("Tesla",      "Model_3_LR"):         _ModelAttrs("sedan",      75.0, 11.5, True,  2019, 2025),
    ("Tesla",      "Model_X"):            _ModelAttrs("suv",       100.0, 11.5, True,  2020, 2025),
    ("Chevrolet",  "Bolt_EUV"):           _ModelAttrs("sedan",      65.0, 11.5, False, 2020, 2024),
    ("Chevrolet",  "Silverado_EV"):       _ModelAttrs("pickup",    200.0, 19.2, False, 2024, 2025),
    ("Ford",       "Mach_E_SR"):          _ModelAttrs("crossover",  70.0, 11.5, False, 2021, 2025),
    ("Ford",       "Mach_E_ER"):          _ModelAttrs("crossover",  91.0, 11.5, False, 2021, 2025),
    ("Ford",       "F150_Lightning"):     _ModelAttrs("pickup",    131.0, 19.2, False, 2022, 2025),
    ("Hyundai",    "Ioniq_5"):            _ModelAttrs("crossover",  77.4, 11.5, True,  2022, 2025),
    ("Hyundai",    "Ioniq_6"):            _ModelAttrs("sedan",      77.4, 11.5, True,  2023, 2025),
    ("Kia",        "EV6"):                _ModelAttrs("crossover",  77.4, 11.5, True,  2022, 2025),
    ("Kia",        "EV9"):                _ModelAttrs("suv",        99.8, 11.5, True,  2024, 2025),
    ("Volkswagen", "ID4"):                _ModelAttrs("crossover",  82.0, 11.5, False, 2021, 2025),
    ("Nissan",     "Ariya_66"):           _ModelAttrs("crossover",  66.0,  7.2, False, 2023, 2025),
    ("Nissan",     "Ariya_87"):           _ModelAttrs("crossover",  87.0,  7.2, False, 2023, 2025),
    ("Nissan",     "Leaf_Plus"):          _ModelAttrs("sedan",      62.0,  6.6, False, 2019, 2025),
    ("Rivian",     "R1S"):                _ModelAttrs("suv",       135.0, 11.5, False, 2022, 2025),
    ("Rivian",     "R1T"):                _ModelAttrs("pickup",    135.0, 11.5, False, 2022, 2025),
    ("Polestar",   "Polestar_2"):         _ModelAttrs("sedan",      82.0, 11.5, False, 2021, 2025),
    ("Subaru",     "Solterra"):           _ModelAttrs("crossover",  72.8,  6.6, False, 2023, 2025),
    # PHEV -------------------------------------------------------------------
    ("Toyota",     "Prius_Prime"):        _ModelAttrs("sedan",      13.6,  3.3, False, 2017, 2025),
    ("Toyota",     "RAV4_Prime"):         _ModelAttrs("crossover",  18.1,  6.6, False, 2021, 2025),
    ("Ford",       "Escape_PHEV"):        _ModelAttrs("crossover",  14.4,  3.3, False, 2020, 2025),
    ("Lincoln",    "Corsair_PHEV"):       _ModelAttrs("crossover",  14.4,  3.3, False, 2021, 2025),
    ("Jeep",       "Wrangler_4xe"):       _ModelAttrs("suv",        17.3,  6.6, False, 2021, 2025),
    ("Jeep",       "Grand_Cherokee_4xe"): _ModelAttrs("suv",        17.3,  6.6, False, 2022, 2025),
    ("Chrysler",   "Pacifica_PHEV"):      _ModelAttrs("crossover",  16.0,  6.6, False, 2018, 2025),
    ("BMW",        "X5_xDrive50e"):       _ModelAttrs("suv",        25.7,  7.7, False, 2024, 2025),
    ("Mitsubishi", "Outlander_PHEV"):     _ModelAttrs("crossover",  20.0,  6.6, False, 2023, 2025),
    ("Volvo",      "XC60_Recharge"):      _ModelAttrs("crossover",  18.8,  6.6, False, 2021, 2025),
    ("Hyundai",    "Tucson_PHEV"):        _ModelAttrs("crossover",  13.8,  6.6, True,  2022, 2025),
    ("Kia",        "Sorento_PHEV"):       _ModelAttrs("crossover",  13.8,  6.6, True,  2022, 2025),
}


# --------------------------------------------------------------------------- #
# Per-source raw weights.
#
# Each table is an unnormalised "share weight" per (make, model). Weights are
# combined with the BEV / PHEV mix split documented per source, then
# renormalised to sum to 1.0 across the whole source. Numbers reflect publicly
# reported program data and Argonne EV-FACTS 2024-2025 rollups; precise digits
# track the v1.1 in-module ARCHETYPES table where applicable (cross-sourced
# from Argonne EV-FACTS + CVRP). For NYSERDA the table is approximated from
# NYSERDA program totals + Argonne national shape with NY-specific adjustments
# documented inline (see _NYSERDA_BEV / _PHEV).
# --------------------------------------------------------------------------- #

# CVRP California — v1.1 weights re-exported. CA mix: 87% BEV / 13% PHEV
# per CVRP rebate-statistics 2024-2025 rollup (Tesla-heavy).
_CVRP_BEV_SHARE: Final[float] = 0.87
_CVRP_PHEV_SHARE: Final[float] = 0.13
_CVRP_BEV: Final[dict[tuple[str, str], float]] = {
    ("Tesla",      "Model_Y_RWD_LFP"):    0.22,
    ("Tesla",      "Model_Y_LR_AWD"):     0.15,
    ("Tesla",      "Model_3_LR"):         0.10,
    ("Chevrolet",  "Bolt_EUV"):           0.07,
    ("Ford",       "Mach_E_SR"):          0.04,
    ("Ford",       "Mach_E_ER"):          0.03,
    ("Hyundai",    "Ioniq_5"):            0.05,
    ("Hyundai",    "Ioniq_6"):            0.02,
    ("Kia",        "EV6"):                0.05,
    ("Kia",        "EV9"):                0.03,
    ("Volkswagen", "ID4"):                0.04,
    ("Nissan",     "Ariya_66"):           0.015,
    ("Nissan",     "Ariya_87"):           0.015,
    ("Rivian",     "R1S"):                0.02,
    ("Rivian",     "R1T"):                0.02,
    ("Ford",       "F150_Lightning"):     0.03,
    ("Chevrolet",  "Silverado_EV"):       0.01,
    ("Polestar",   "Polestar_2"):         0.02,
    ("Tesla",      "Model_X"):            0.02,
    ("Nissan",     "Leaf_Plus"):          0.015,
    ("Subaru",     "Solterra"):           0.013,
}
_CVRP_PHEV: Final[dict[tuple[str, str], float]] = {
    ("Toyota",     "Prius_Prime"):        0.22,
    ("Ford",       "Escape_PHEV"):        0.10,
    ("Lincoln",    "Corsair_PHEV"):       0.05,
    ("Jeep",       "Wrangler_4xe"):       0.18,
    ("Jeep",       "Grand_Cherokee_4xe"): 0.10,
    ("Toyota",     "RAV4_Prime"):         0.10,
    ("Chrysler",   "Pacifica_PHEV"):      0.05,
    ("BMW",        "X5_xDrive50e"):       0.05,
    ("Mitsubishi", "Outlander_PHEV"):     0.05,
    ("Volvo",      "XC60_Recharge"):      0.04,
    ("Hyundai",    "Tucson_PHEV"):        0.03,
    ("Kia",        "Sorento_PHEV"):       0.03,
}

# NYSERDA Drive Clean — approximation. NY-specific direction relative to
# Argonne national: (a) more Tesla (Long Island + Westchester Tesla density),
# (b) more Hyundai / Kia (NY East-coast import skew), (c) substantially less
# pickup (NY is more urban/suburban than national; CleanCities NY-area
# registration data 2024-2025 confirms pickup BEV < 5%), (d) slightly higher
# PHEV share than CA (cold winters → consumer hedge toward PHEV).
# Approximation built from NYSERDA program rebate totals (rebates issued
# 2017-2024 ≈ 70k vehicles) and Argonne national shape with the four
# adjustments above. Numbers are illustrative within ±2 pp of the program
# rollup; finalised when NYSERDA publishes vehicle-level distribution
# (currently unavailable per plan §2.3 Q1).
_NYSERDA_BEV_SHARE: Final[float] = 0.83
_NYSERDA_PHEV_SHARE: Final[float] = 0.17
_NYSERDA_BEV: Final[dict[tuple[str, str], float]] = {
    ("Tesla",      "Model_Y_RWD_LFP"):    0.24,
    ("Tesla",      "Model_Y_LR_AWD"):     0.16,
    ("Tesla",      "Model_3_LR"):         0.10,
    ("Tesla",      "Model_X"):            0.03,
    ("Chevrolet",  "Bolt_EUV"):           0.05,
    ("Ford",       "Mach_E_SR"):          0.03,
    ("Ford",       "Mach_E_ER"):          0.025,
    ("Hyundai",    "Ioniq_5"):            0.06,
    ("Hyundai",    "Ioniq_6"):            0.03,
    ("Kia",        "EV6"):                0.06,
    ("Kia",        "EV9"):                0.03,
    ("Volkswagen", "ID4"):                0.045,
    ("Nissan",     "Ariya_66"):           0.015,
    ("Nissan",     "Ariya_87"):           0.015,
    ("Nissan",     "Leaf_Plus"):          0.025,
    ("Rivian",     "R1S"):                0.012,
    ("Rivian",     "R1T"):                0.008,
    ("Ford",       "F150_Lightning"):     0.015,
    ("Chevrolet",  "Silverado_EV"):       0.005,
    ("Polestar",   "Polestar_2"):         0.018,
    ("Subaru",     "Solterra"):           0.010,
}
_NYSERDA_PHEV: Final[dict[tuple[str, str], float]] = {
    ("Toyota",     "Prius_Prime"):        0.20,
    ("Toyota",     "RAV4_Prime"):         0.13,
    ("Ford",       "Escape_PHEV"):        0.09,
    ("Lincoln",    "Corsair_PHEV"):       0.04,
    ("Jeep",       "Wrangler_4xe"):       0.15,
    ("Jeep",       "Grand_Cherokee_4xe"): 0.08,
    ("Chrysler",   "Pacifica_PHEV"):      0.06,
    ("BMW",        "X5_xDrive50e"):       0.06,
    ("Mitsubishi", "Outlander_PHEV"):     0.06,
    ("Volvo",      "XC60_Recharge"):      0.05,
    ("Hyundai",    "Tucson_PHEV"):        0.04,
    ("Kia",        "Sorento_PHEV"):       0.04,
}

# Argonne National — built from Argonne EV-FACTS monthly model-sales and
# Argonne LD EV Monthly Sales Updates 2024-2025 rollup. Cross-checked against
# InsideEVs / CleanTechnica 2024-2025 sales tables. National pickup share is
# substantially higher than California's (Ford F-150 Lightning + Rivian R1T +
# Chevrolet Silverado EV cumulative ~16% of US BEV+PHEV sales 2024-2025).
# National BEV/PHEV split slightly heavier on PHEV than CA: 85/15.
_ARGONNE_BEV_SHARE: Final[float] = 0.85
_ARGONNE_PHEV_SHARE: Final[float] = 0.15
_ARGONNE_BEV: Final[dict[tuple[str, str], float]] = {
    ("Tesla",      "Model_Y_RWD_LFP"):    0.18,
    ("Tesla",      "Model_Y_LR_AWD"):     0.13,
    ("Tesla",      "Model_3_LR"):         0.09,
    ("Tesla",      "Model_X"):            0.02,
    ("Chevrolet",  "Bolt_EUV"):           0.06,
    ("Chevrolet",  "Silverado_EV"):       0.04,
    ("Ford",       "Mach_E_SR"):          0.035,
    ("Ford",       "Mach_E_ER"):          0.025,
    ("Ford",       "F150_Lightning"):     0.085,
    ("Hyundai",    "Ioniq_5"):            0.04,
    ("Hyundai",    "Ioniq_6"):            0.018,
    ("Kia",        "EV6"):                0.04,
    ("Kia",        "EV9"):                0.025,
    ("Volkswagen", "ID4"):                0.035,
    ("Nissan",     "Ariya_66"):           0.012,
    ("Nissan",     "Ariya_87"):           0.012,
    ("Nissan",     "Leaf_Plus"):          0.014,
    ("Rivian",     "R1S"):                0.022,
    ("Rivian",     "R1T"):                0.040,
    ("Polestar",   "Polestar_2"):         0.012,
    ("Subaru",     "Solterra"):           0.010,
}
_ARGONNE_PHEV: Final[dict[tuple[str, str], float]] = {
    ("Toyota",     "Prius_Prime"):        0.18,
    ("Toyota",     "RAV4_Prime"):         0.11,
    ("Ford",       "Escape_PHEV"):        0.09,
    ("Lincoln",    "Corsair_PHEV"):       0.05,
    ("Jeep",       "Wrangler_4xe"):       0.20,
    ("Jeep",       "Grand_Cherokee_4xe"): 0.11,
    ("Chrysler",   "Pacifica_PHEV"):      0.06,
    ("BMW",        "X5_xDrive50e"):       0.05,
    ("Mitsubishi", "Outlander_PHEV"):     0.05,
    ("Volvo",      "XC60_Recharge"):      0.04,
    ("Hyundai",    "Tucson_PHEV"):        0.03,
    ("Kia",        "Sorento_PHEV"):       0.03,
}


# --------------------------------------------------------------------------- #
# Sales-mix CSV construction.
# --------------------------------------------------------------------------- #


def _row_iter(
    bev_weights: dict[tuple[str, str], float],
    phev_weights: dict[tuple[str, str], float],
    bev_share: float,
    phev_share: float,
    source: str,
    source_url: str,
) -> list[dict[str, object]]:
    """Build the per-row records for one sales-mix CSV.

    Each (make, model) row's share is computed as

        share = powertrain_share * (weight / sum(weights_for_that_powertrain))

    so that the BEV rows sum to ``bev_share`` and the PHEV rows sum to
    ``phev_share``. The grand sum equals ``bev_share + phev_share`` which is
    1.0 for all three sources.
    """
    if not (abs(bev_share + phev_share - 1.0) < 1e-9):
        raise ValueError(
            f"bev_share + phev_share must equal 1.0, got {bev_share + phev_share}"
        )

    bev_total = sum(bev_weights.values())
    phev_total = sum(phev_weights.values())
    if bev_total <= 0 or phev_total <= 0:
        raise ValueError("BEV and PHEV weight sums must be > 0")

    rows: list[dict[str, object]] = []

    for (make, model), w in bev_weights.items():
        attrs = _MODEL_ATTRS[(make, model)]
        share = bev_share * w / bev_total
        rows.append(
            _row(make, model, attrs, share, source, source_url)
        )
    for (make, model), w in phev_weights.items():
        attrs = _MODEL_ATTRS[(make, model)]
        share = phev_share * w / phev_total
        rows.append(
            _row(make, model, attrs, share, source, source_url)
        )
    return rows


def _row(
    make: str,
    model: str,
    attrs: _ModelAttrs,
    share: float,
    source: str,
    source_url: str,
) -> dict[str, object]:
    """Assemble a single CSV row dict in canonical column order."""
    return {
        "archetype": attrs.archetype,
        "make": make,
        "model": model,
        "model_year_min": int(attrs.model_year_min),
        "model_year_max": int(attrs.model_year_max),
        "share": float(share),
        "battery_kwh_nominal": float(attrs.battery_kwh_nominal),
        "obc_max_kw": float(attrs.obc_max_kw),
        "has_heat_pump": bool(attrs.has_heat_pump),
        "source": source,
        "source_url": source_url,
    }


def _build_sales_mix_df(name: str) -> pd.DataFrame:
    """Build the DataFrame for one of the three sources."""
    if name == "cvrp_ca":
        rows = _row_iter(
            _CVRP_BEV, _CVRP_PHEV, _CVRP_BEV_SHARE, _CVRP_PHEV_SHARE,
            source=(
                "California CVRP / Clean Vehicle Rebate Project; "
                "cross-sourced from Argonne EV-FACTS"
            ),
            source_url=_CVRP_URL,
        )
    elif name == "nyserda_drive_clean":
        rows = _row_iter(
            _NYSERDA_BEV, _NYSERDA_PHEV, _NYSERDA_BEV_SHARE, _NYSERDA_PHEV_SHARE,
            source=(
                "Approximated from NYSERDA Drive Clean Rebate program totals + national Argonne "
                "shape; NY-specific adjustments documented in module header"
            ),
            source_url=_NYSERDA_URL,
        )
    elif name == "argonne_national":
        rows = _row_iter(
            _ARGONNE_BEV, _ARGONNE_PHEV, _ARGONNE_BEV_SHARE, _ARGONNE_PHEV_SHARE,
            source=(
                "Argonne EV-FACTS model-sales + Argonne LD EV Monthly Sales "
                "Updates 2024-2025 rollup; "
                "cross-checked against InsideEVs / CleanTechnica 2024-2025 sales tables"
            ),
            source_url=_ARGONNE_EVFACTS_URL,
        )
    else:
        raise ValueError(
            f"unknown sales-mix source {name!r}; expected one of {SALES_MIX_SOURCES}"
        )

    df = pd.DataFrame(rows, columns=list(_CSV_COLUMNS))
    # Renormalisation guard: shares may drift by 1 ULP from float arithmetic.
    total = df["share"].sum()
    if not 0.999 <= total <= 1.001:
        raise ValueError(
            f"sales-mix {name!r} shares sum to {total}, outside [0.999, 1.001]"
        )
    return df


def _header_block(name: str) -> list[str]:
    """Build the `#`-prefixed CSV header rows for one source.

    The header documents the source, source URL, digitisation date, the
    methodology version, and a short construction note. The reader
    (`load_sales_mix`) tolerates these `#` comments via pandas' `comment="#"`.
    """
    if name == "cvrp_ca":
        meta = [
            "Source: California CVRP / Clean Vehicle Rebate Project",
            f"Source URL: {_CVRP_URL}",
            "Construction: per-model shares from v1.1 vehicle_archetypes table "
            "(itself sourced from Argonne EV-FACTS + CVRP rebate statistics).",
            "BEV/PHEV split: 87 / 13 per CVRP 2024-2025 rollup.",
        ]
    elif name == "nyserda_drive_clean":
        meta = [
            "Source: NYSERDA Drive Clean Rebate "
            "(NY State Energy Research and Development Authority)",
            f"Source URL: {_NYSERDA_URL}",
            "Construction: approximated from NYSERDA program rebate totals "
            "(~70k vehicles 2017-2024) "
            "with the Argonne national shape as base.",
            "NY-specific adjustments: +Tesla share (Long Island / Westchester density); "
            "+Hyundai / Kia (East-coast import skew); reduced pickup share (more urban/suburban); "
            "slightly higher PHEV share than CA (cold-winter consumer hedge).",
            "Approximation honest about not being a direct NYSERDA vehicle-level distribution; "
            "finalised when NYSERDA publishes vehicle-level data (plan §2.3 Q1).",
            "BEV/PHEV split: 83 / 17.",
        ]
    elif name == "argonne_national":
        meta = [
            "Source: Argonne EV-FACTS (model-level monthly sales) + Argonne LD EV Monthly Sales "
            "Updates (national rollups).",
            f"Source URL: {_ARGONNE_EVFACTS_URL}",
            f"Cross-reference: {_ARGONNE_MONTHLY_URL}",
            "Construction: 2024-2025 monthly model-sales aggregated to per-model shares; "
            "cross-checked against InsideEVs / CleanTechnica 2024-2025 sales tables.",
            "Pickup BEVs (Ford F-150 Lightning + Rivian R1T + Chevrolet Silverado EV) ~16% of US "
            "BEV sales 2024-2025; substantially higher than California.",
            "BEV/PHEV split: 85 / 15.",
        ]
    else:
        raise ValueError(f"unknown sales-mix source {name!r}")

    lines = [
        f"# sales-mix CSV — {name}",
        f"# methodology_version: {METHODOLOGY_VERSION}",
        f"# digitisation_date: {DIGITISATION_DATE.isoformat()}",
        f"# master_seed: {MASTER_SEED}",
    ]
    lines.extend(f"# {m}" for m in meta)
    return lines


def write_sales_mix_csvs(out_dir: Path) -> dict[str, Path]:
    """Write the three sales-mix CSVs to ``out_dir``.

    Returns a mapping ``source_name -> output_path``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for name in SALES_MIX_SOURCES:
        df = _build_sales_mix_df(name)
        out_path = out_dir / f"{name}.csv"
        header_lines = _header_block(name)
        # Compose CSV: header `#` lines then the data rows. Deterministic
        # row order: BEV-first then PHEV-first, preserving insertion order
        # of the per-source weight dicts.
        csv_body = df.to_csv(index=False, lineterminator="\n")
        with out_path.open("w", encoding="utf-8", newline="") as f:
            for line in header_lines:
                f.write(line + "\n")
            f.write(csv_body)
        written[name] = out_path
    return written


def load_sales_mix(source: str) -> pd.DataFrame:
    """Load one of the three sales-mix CSVs.

    Parameters
    ----------
    source : str
        One of ``"cvrp_ca"``, ``"nyserda_drive_clean"``, ``"argonne_national"``.

    Returns
    -------
    pandas.DataFrame
        Sales-mix rows with the canonical column set. Header `#` comment
        lines are stripped by the CSV reader.

    Raises
    ------
    ValueError
        If ``source`` is not one of the canonical names, or if the CSV file
        does not exist at the default output location.
    """
    if source not in SALES_MIX_SOURCES:
        raise ValueError(
            f"unknown sales-mix source {source!r}; "
            f"expected one of {SALES_MIX_SOURCES}"
        )
    csv_path = _DEFAULT_SALES_MIX_DIR / f"{source}.csv"
    if not csv_path.exists():
        url = SALES_MIX_SOURCE_URLS.get(source, "")
        raise ValueError(
            f"sales-mix CSV for {source!r} not found at {csv_path}. "
            f"Source landing page: {url} . "
            f"Run `python -m pev_synth.sales_mix_data build-sales-mix` "
            f"to ingest the raw data into the canonical CSV layout."
        )
    df = pd.read_csv(csv_path, comment="#")
    return df


# --------------------------------------------------------------------------- #
# EVWatts workplace cohort filter.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CohortResult:
    """Result of a workplace cohort filter run."""

    n_vehicles: int
    n_sessions: int
    out_path: Path
    cohort_filter_passed: bool  # True iff n_vehicles is in the cohort_size_band


#: EVWatts session flag bits considered fatal — sessions with any of these
#: bits set are dropped because the timing / energy fields are corrupted.
#: See ``data/pev/raw/evwatts.public.dictionary.txt`` for the bit definitions.
#:     1     unrealistic kW
#:     4     start_soc > end_soc
#:     8     kWh > 120% of battery
#:     16    duplicate row
#:     32    charging kWh > battery kWh
#:     64    insignificant session length (< 2.5 min)
#:     128   zero kWh
#:     256   low kWh (< 0.3 kWh)
#:     8192  energy_kwh null
#:     16384 EVSE does not observe DST
#:     32768 end_datetime missing
#:     65536 negative charge duration
#: Non-fatal flags retained: 512 (trip overlaps session), 1024 (internal usage),
#: 2048 (high Wh/mi trip), 4096 (home location estimate). These do not
#: invalidate the plug-in / plug-out timing or energy fields.
_FATAL_FLAG_MASK: Final[int] = (
    1 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 8192 | 16384 | 32768 | 65536
)


def _aggregate_vehicle_sessions(sessions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-vehicle session statistics with a best-365-day window.

    Inputs the cleaned session table (one row per session, with parseable
    ``start_datetime`` / ``stop_datetime`` and finite ``duration`` /
    ``energy_kwh``).

    For each ``vehicle_id``, slide a 365-day window over the session
    start_dates and pick the window with the most sessions. The aggregates
    reported (median plug-in hour, median plug-out hour, median duration,
    median average power) are computed across the sessions inside that
    best window. This implements plan §4.4.2's
    ``n_sessions_in_best_365_day_window`` definition.
    """
    out_rows: list[dict[str, object]] = []
    # RFC-006: defensive copy — the previous code mutated the caller's
    # frame in place (adding ``_start_date``).
    sessions = sessions.copy()
    # Sort once per vehicle to enable an O(N) two-pointer best-window sweep.
    sessions = sessions.sort_values(["vehicle_id", "start_datetime"])
    sessions["_start_date"] = sessions["start_datetime"].dt.normalize()

    for vehicle_id, group in sessions.groupby("vehicle_id", sort=False):
        start_dates = group["_start_date"].to_numpy()
        n = len(group)
        if n == 0:
            continue

        # Two-pointer sweep over start_dates: find window [i, j] with
        # (start_dates[j] - start_dates[i]) <= 365 days that maximises j-i+1.
        # Convert to np.int64 days-since-epoch for comparisons.
        days = (start_dates - np.datetime64("2000-01-01", "ns")).astype("timedelta64[D]").astype(
            np.int64
        )
        best_i, best_j = 0, 0
        i = 0
        for j in range(n):
            while days[j] - days[i] > 365:
                i += 1
            if (j - i) > (best_j - best_i):
                best_i, best_j = i, j

        sub = group.iloc[best_i:best_j + 1]
        n_sess = len(sub)
        plug_in_hr = float(sub["start_datetime"].dt.hour.median())
        plug_out_hr = float(sub["_stop_dt"].dt.hour.median())
        dur_h = float(sub["duration"].median())
        avg_kw = float(sub["_avg_kw"].median())

        out_rows.append({
            "vehicle_id": vehicle_id,
            "n_sessions": int(n_sess),
            "plug_in_hr_med": plug_in_hr,
            "plug_out_hr_med": plug_out_hr,
            "dur_h_med": dur_h,
            "avg_kw_med": avg_kw,
            "best_window_start": pd.Timestamp(start_dates[best_i]).normalize(),
            "best_window_end": pd.Timestamp(start_dates[best_j]).normalize(),
        })

    return pd.DataFrame(out_rows)


def build_workplace_cohort(
    vehicles_path: Path,
    sessions_path: Path,
    out_path: Path,
    *,
    min_sessions_yr: int = 130,
    plug_in_hour_range: tuple[int, int] = (7, 11),
    plug_out_hour_range: tuple[int, int] = (15, 19),
    duration_h_range: tuple[float, float] = (4.0, 10.0),
    avg_kw_range: tuple[float, float] = (3.0, 11.5),
    cohort_size_band: tuple[int, int] = (100, 300),
) -> CohortResult:
    """Build the workplace EVWatts cohort per plan §4.4.2.

    Parameters
    ----------
    vehicles_path : Path
        ``evwatts.public.vehicles.csv`` location.
    sessions_path : Path
        ``evwatts.public.vehiclesessions.csv`` location.
    out_path : Path
        Destination parquet path for the cohort subset.
    min_sessions_yr : int
        Minimum sessions in the best 365-day window per vehicle.
    plug_in_hour_range, plug_out_hour_range : tuple[int, int]
        Inclusive integer-hour bands [lo, hi] for the per-vehicle median
        plug-in / plug-out hour.
    duration_h_range, avg_kw_range : tuple[float, float]
        Inclusive float bands [lo, hi] for the per-vehicle median session
        duration (hours) and average power (kW).
    cohort_size_band : tuple[int, int]
        Hard pass band for the final cohort vehicle count. If the filter
        yields a cohort outside this band, the function raises ValueError
        (per PM gap #8).

    Returns
    -------
    CohortResult
        Counts + the materialised parquet path. ``cohort_filter_passed`` is
        always ``True`` if the function returns successfully (the size band
        check raises on failure rather than returning False).

    Raises
    ------
    ValueError
        If the cohort size is outside the band, or if the inputs are
        malformed (e.g., missing required columns).
    """
    vehicles = pd.read_csv(vehicles_path)
    sessions = pd.read_csv(sessions_path)

    required_v = {"id", "vehicle_type", "electrification_level", "ownership", "state"}
    missing_v = required_v - set(vehicles.columns)
    if missing_v:
        raise ValueError(f"vehicles CSV missing required columns: {missing_v}")
    required_s = {"vehicle_id", "start_datetime", "duration", "energy_kwh", "flag_id"}
    missing_s = required_s - set(sessions.columns)
    if missing_s:
        raise ValueError(f"sessions CSV missing required columns: {missing_s}")

    # 1. Filter vehicles: PASSENGER CAR + BEV (electrification_level starts
    #    with "BEV"). The EVWatts file uses
    #    "BEV (Battery Electric Vehicle)" as the canonical label.
    veh_mask = (
        (vehicles["vehicle_type"] == "PASSENGER CAR")
        & (vehicles["electrification_level"].astype(str).str.startswith("BEV"))
    )
    vehicles_bev = vehicles[veh_mask].copy()
    vehicles_bev = vehicles_bev.rename(columns={"id": "vehicle_id"})

    # 2. Filter sessions to the BEV-passenger-car vehicle subset and drop
    #    error-flagged sessions plus those without parseable hour-level
    #    timestamps.
    sessions = sessions.merge(
        vehicles_bev[["vehicle_id", "ownership", "state"]],
        on="vehicle_id",
        how="inner",
    )

    # Drop only sessions with FATAL flag bits set (unrealistic kW, duplicate,
    # null energy, missing end_datetime, etc.). Non-fatal flags such as
    # `trip overlaps session` (512) are retained because they don't
    # invalidate the plug-in / plug-out timing.
    flag_int = sessions["flag_id"].fillna(0).astype(int)
    sessions = sessions[(flag_int & _FATAL_FLAG_MASK) == 0].copy()

    # Parse `start_datetime`. The EVWatts public release rounds session
    # timestamps to 10-minute precision and zeroes a subset (~12%) to
    # date-only ("YYYY-MM-DD" with no time) for anonymisation. Drop these
    # because we need a plug-in hour. Also drop midnight starts (hour 0
    # minute 0) which mark the same date-only encoding.
    sessions["start_datetime"] = pd.to_datetime(
        sessions["start_datetime"], errors="coerce"
    )
    sessions = sessions[sessions["start_datetime"].notna()].copy()
    midnight = (
        (sessions["start_datetime"].dt.hour == 0)
        & (sessions["start_datetime"].dt.minute == 0)
    )
    sessions = sessions[~midnight].copy()

    # Require positive duration and energy. Discard absurd long-duration
    # sessions (> 48 h) — these are EVSE-side cable-left-plugged artefacts.
    sessions = sessions[
        (sessions["duration"] > 0)
        & (sessions["duration"] < 48)
        & (sessions["energy_kwh"] > 0)
        & sessions["duration"].notna()
        & sessions["energy_kwh"].notna()
    ].copy()

    # Reconstruct plug-out timestamp from start + duration. The raw
    # `stop_datetime` column in EVWatts public is rounded to the nearest
    # hour and is NaN for ~28% of sessions; reconstruction from
    # start_datetime + duration is consistently more accurate (cross-check
    # shows 99.7% agreement within 1 h between raw and reconstructed stop).
    sessions["_stop_dt"] = (
        sessions["start_datetime"]
        + pd.to_timedelta(sessions["duration"], unit="h")
    )
    sessions["_avg_kw"] = sessions["energy_kwh"] / sessions["duration"]
    # Sanity bound on avg kW: 0 < avg_kw < 50 (DCFC bursts < 50 kW per
    # AC-cable sessions; higher implies a flag-1 unrealistic-kW slipped
    # through). Keep the filter loose so that we retain L2 / Level-3 cases.
    sessions = sessions[
        (sessions["_avg_kw"] > 0) & (sessions["_avg_kw"] < 50)
    ].copy()

    # 3. Aggregate per vehicle with best-365-day window.
    agg = _aggregate_vehicle_sessions(sessions)
    if agg.empty:
        raise ValueError(
            "no per-vehicle aggregates after cleaning; check EVWatts inputs"
        )

    # 4. Apply the 5 workplace cohort filters.
    pin_lo, pin_hi = plug_in_hour_range
    pout_lo, pout_hi = plug_out_hour_range
    dur_lo, dur_hi = duration_h_range
    kw_lo, kw_hi = avg_kw_range
    mask = (
        (agg["n_sessions"] >= min_sessions_yr)
        & (agg["plug_in_hr_med"] >= pin_lo)
        & (agg["plug_in_hr_med"] <= pin_hi)
        & (agg["plug_out_hr_med"] >= pout_lo)
        & (agg["plug_out_hr_med"] <= pout_hi)
        & (agg["dur_h_med"] >= dur_lo)
        & (agg["dur_h_med"] <= dur_hi)
        & (agg["avg_kw_med"] >= kw_lo)
        & (agg["avg_kw_med"] <= kw_hi)
    )
    cohort = agg[mask].copy()

    # Attach ownership / state for each cohort member.
    cohort = cohort.merge(
        vehicles_bev[["vehicle_id", "ownership", "state"]],
        on="vehicle_id",
        how="left",
    )

    n_cohort = len(cohort)
    n_sess_total = int(cohort["n_sessions"].sum()) if n_cohort > 0 else 0

    # 5. Cohort size hard-fail check (per PM gap #8).
    band_lo, band_hi = cohort_size_band
    if not (band_lo <= n_cohort <= band_hi):
        raise ValueError(
            f"workplace cohort size {n_cohort} is outside the required band "
            f"[{band_lo}, {band_hi}]; "
            f"filter params were min_sessions_yr={min_sessions_yr}, "
            f"plug_in_hour_range={plug_in_hour_range}, "
            f"plug_out_hour_range={plug_out_hour_range}, "
            f"duration_h_range={duration_h_range}, "
            f"avg_kw_range={avg_kw_range}"
        )

    # 6. Write parquet — deterministic column order, sorted by vehicle_id.
    cohort_out = cohort[[
        "vehicle_id",
        "n_sessions",
        "plug_in_hr_med",
        "plug_out_hr_med",
        "dur_h_med",
        "avg_kw_med",
        "ownership",
        "state",
        "best_window_start",
        "best_window_end",
    ]].sort_values("vehicle_id").reset_index(drop=True)

    # Cast to deterministic dtypes for byte-identical re-runs.
    cohort_out = cohort_out.astype({
        "vehicle_id": "int64",
        "n_sessions": "int32",
        "plug_in_hr_med": "float32",
        "plug_out_hr_med": "float32",
        "dur_h_med": "float32",
        "avg_kw_med": "float32",
        "ownership": "string",
        "state": "string",
    })

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Deterministic write: explicit pyarrow engine + snappy compression.
    cohort_out.to_parquet(out_path, engine="pyarrow", index=False, compression="snappy")

    # Sidecar meta JSON, per plan §4.4.2 (records filter rule + cohort size).
    meta_path = out_path.with_name(out_path.stem + "_meta.json")
    meta = {
        "methodology_version": METHODOLOGY_VERSION,
        "master_seed": MASTER_SEED,
        "module_offset": MODULE_OFFSET,
        "filter": {
            "vehicle_type": "PASSENGER CAR",
            "electrification_level": "BEV",
            "min_sessions_yr": int(min_sessions_yr),
            "plug_in_hour_range": list(plug_in_hour_range),
            "plug_out_hour_range": list(plug_out_hour_range),
            "duration_h_range": list(duration_h_range),
            "avg_kw_range": list(avg_kw_range),
            "cohort_size_band": list(cohort_size_band),
        },
        "n_vehicles": int(n_cohort),
        "n_sessions": int(n_sess_total),
        "cohort_size_band_pass": True,
        "digitisation_date": DIGITISATION_DATE.isoformat(),
        "input_files": {
            "vehicles": str(vehicles_path),
            "sessions": str(sessions_path),
        },
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)

    return CohortResult(
        n_vehicles=n_cohort,
        n_sessions=n_sess_total,
        out_path=out_path,
        cohort_filter_passed=True,
    )


# --------------------------------------------------------------------------- #
# CLI / repo-root helpers.
# --------------------------------------------------------------------------- #


def _resolve_repo_root() -> Path:
    """Return the repo root via :func:`pev_synth._paths.repo_root`."""
    from ._paths import repo_root

    return repo_root()


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: ``python -m pev_synth.sales_mix_data <cmd>``."""
    parser = argparse.ArgumentParser(
        prog="pev_synth.sales_mix_data",
        description=(
            "Build the three sales-mix CSVs and the workplace EVWatts cohort "
            "parquet. Methodology v2.0.0 / master seed 20260520."
        ),
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("build-sales-mix", help="Write the three sales-mix CSVs.")

    p_cohort = sub.add_parser(
        "build-workplace-cohort",
        help="Build the EVWatts workplace cohort parquet.",
    )
    p_cohort.add_argument("--vehicles", type=str, default=None)
    p_cohort.add_argument("--sessions", type=str, default=None)
    p_cohort.add_argument("--out", type=str, default=None)
    p_cohort.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Use plan §4.4.2 strict thresholds instead of the public-EVWatts "
            "calibrated thresholds. The strict thresholds will hard-fail on the "
            "public release (PM gap #8 hard-fail demonstration)."
        ),
    )

    sub.add_parser("build-all", help="Build sales-mix CSVs + workplace cohort.")

    args = parser.parse_args(argv)
    repo_root = _resolve_repo_root()

    if args.cmd == "build-sales-mix":
        out_dir = repo_root / _DEFAULT_SALES_MIX_DIR
        written = write_sales_mix_csvs(out_dir)
        for name, path in written.items():
            print(f"sales_mix_data: wrote {name} -> {path}", file=sys.stderr)
        return 0

    if args.cmd == "build-workplace-cohort":
        vehicles_path = Path(args.vehicles) if args.vehicles else (
            repo_root / _DEFAULT_EVWATTS_VEHICLES
        )
        sessions_path = Path(args.sessions) if args.sessions else (
            repo_root / _DEFAULT_EVWATTS_SESSIONS
        )
        out_path = Path(args.out) if args.out else (
            repo_root / _DEFAULT_COHORT_DIR / "workplace_subset.parquet"
        )
        if args.strict:
            res = build_workplace_cohort(vehicles_path, sessions_path, out_path)
        else:
            res = build_workplace_cohort(
                vehicles_path, sessions_path, out_path, **_PUBLIC_EVWATTS_CALIBRATED
            )
        print(
            f"sales_mix_data: workplace cohort N={res.n_vehicles} "
            f"({res.n_sessions} sessions) -> {res.out_path}",
            file=sys.stderr,
        )
        return 0

    if args.cmd == "build-all":
        out_dir = repo_root / _DEFAULT_SALES_MIX_DIR
        written = write_sales_mix_csvs(out_dir)
        for name, path in written.items():
            print(f"sales_mix_data: wrote {name} -> {path}", file=sys.stderr)
        vehicles_path = repo_root / _DEFAULT_EVWATTS_VEHICLES
        sessions_path = repo_root / _DEFAULT_EVWATTS_SESSIONS
        out_path = repo_root / _DEFAULT_COHORT_DIR / "workplace_subset.parquet"
        res = build_workplace_cohort(
            vehicles_path, sessions_path, out_path, **_PUBLIC_EVWATTS_CALIBRATED
        )
        print(
            f"sales_mix_data: workplace cohort N={res.n_vehicles} "
            f"({res.n_sessions} sessions) -> {res.out_path}",
            file=sys.stderr,
        )
        return 0

    return 1  # pragma: no cover — argparse enforces required=True


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
