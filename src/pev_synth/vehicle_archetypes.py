"""M2 — pev_synth.vehicle_archetypes (v2.0 refactor).

Sample N synthetic-vehicle attribute vectors per (region, profile_type) and
write a parquet fleet table.  v2.0 lifts the v1.1 single-CA assumption: the
sales mix is read at runtime from the per-source CSVs produced by
``pev_synth.sales_mix_data`` (one of ``cvrp_ca``, ``nyserda_drive_clean``,
``argonne_national``), the optional per-region overlay
(``region.sales_mix_overlay``) is applied additively-per-archetype and the
mix re-normalised, and the EVSE distribution is dispatched on
``profile_type`` so workplace fleets get the J1772 6.6 - 7.7 kW dominant
prior described in v2.0-rev §4.5.

The module accepts ``region: Region | str`` and ``profile_type: str``; the
defaults reproduce v1.1 (Bay Area, residential) so callers that ignore the
new arguments continue to work.

Schema changes vs v1.1 (per PM brief v2.0 §4.5):
* New columns ``profile_type`` (category) and ``region_name`` (category).
* ``EVSE_home_kw`` is retained as the canonical EVSE column name so the
  downstream ``api.Fleet`` consumer keeps working without code churn; the
  PM brief permits "keep one canonical name -- your call; document it"
  (see §4.5 / Module 4 §6 step 4).  Workplace fleets populate this column
  using the workplace prior; the column name does not mutate.

References
----------
* Argonne LD EV Monthly Sales Updates, Argonne EV-FACTS, CVRP rebate-stats
  dashboard, NYSERDA Drive Clean program, EV-Database pack-capacity
  cheatsheet, SAE J1772, Borlaug 2020 Joule 4(7):1470-1485, Ge 2021
  NREL/TP-5400-81065, Sears 2014 TRR 2454:86-92, Atkinson 2025 PMC12496165.
* v2.0 methodology: region table (§2), DFW pickup overlay arithmetic
  (§2.7), workplace EVSE prior (§4.5). Module 4 (vehicle_archetypes).

Methodology version: v2.0.0.  Master seed: 20260520.  Module offset: +1.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
from scipy.stats import truncnorm

from ._phev_fuel_economy import PHEV_CS_MPG_FALLBACK, cs_mpg
from ._seeds import (
    MASTER_SEED as _MASTER_SEED,
)
from ._seeds import (
    METHODOLOGY_VERSION as _METHODOLOGY_VERSION,
)
from ._seeds import (
    MODULE_OFFSETS as _MODULE_OFFSETS,
)
from ._seeds import (
    module_offset as _module_offset,
)
from .regions import PROFILE_TYPES, REGIONS, Region
from .sales_mix_data import SALES_MIX_SOURCES, load_sales_mix

# ---------------------------------------------------------------------------
# Constants and orchestrator-resolved contracts.
# ---------------------------------------------------------------------------

METHODOLOGY_VERSION: Final[str] = _METHODOLOGY_VERSION
MASTER_SEED_DEFAULT: Final[int] = _MASTER_SEED
MODULE_OFFSET: Final[int] = _module_offset("vehicle_archetypes")  # +1
N_VEHICLES_DEFAULT: Final[int] = 1000

# v1.1 legacy default BEV share (CA CVRP 2024).  Kept as a fallback for
# callers that hand us a sales-mix source we cannot classify by powertrain.
BEV_SHARE_LEGACY_DEFAULT: Final[float] = 0.87
PHEV_SHARE_LEGACY_DEFAULT: Final[float] = 0.13

# eta_charge_nominal prior (Sears 2014; plan §2.4, §3.2 step 7).
ETA_MU: Final[float] = 0.90
ETA_SIGMA: Final[float] = 0.02
ETA_LO: Final[float] = 0.85
ETA_HI: Final[float] = 0.94

# SoC_min fraction (plan §6.1; PM brief).
SOC_MIN_FRAC: Final[float] = 0.10

# Panel-cap prior (Ge 2021 NREL/TP-5400-81065; plan §2.4).
PANEL_CAP_PROB_RESIDENTIAL: Final[float] = 0.15
PANEL_CAP_PROB_WORKPLACE: Final[float] = 0.0  # no panel cap on workplace lots
PANEL_CAP_KW: Final[float] = 7.7  # 32 A x 240 V.

# Garage prior (Ge 2021 NREL/TP-5400-81065 Tables 2-3; plan §2.4).
GARAGE_DEDICATED_PROB: Final[float] = 0.78

# Income-tier priors (legacy v1.1 + acs_national from PM brief §6 step 9).
# Bay Area / LA Basin keep the CA-specific 4-tier prior; the rest use the
# 2023 ACS 5-yr national 4-tier distribution.
INCOME_TIERS: Final[tuple[str, ...]] = ("t1", "t2", "t3", "t4")
INCOME_TIER_WEIGHTS_CVRP_CA: Final[tuple[float, ...]] = (0.18, 0.32, 0.32, 0.18)
INCOME_TIER_WEIGHTS_ACS_NATIONAL: Final[tuple[float, ...]] = (0.22, 0.30, 0.28, 0.20)

# Home station defaults if region.temperature_stations is empty (Bay Area
# stations preserved for v1.1 backwards behaviour).
HOME_STATIONS_BAY_AREA_FALLBACK: Final[tuple[str, ...]] = (
    "KSFO", "KOAK", "KSJC", "KSTS", "KAPC", "KCCR",
)

# OBC categorical (plan §2.4; PM brief §6 step 7).
OBC_MIX_BEV: Final[dict[float, float]] = {3.3: 0.05, 6.6: 0.20, 7.7: 0.10, 11.5: 0.55, 19.2: 0.10}
OBC_MIX_PHEV: Final[dict[float, float]] = {3.3: 0.40, 6.6: 0.55, 7.7: 0.05}

# Home-EVSE nameplate categorical -- residential (Borlaug 2020; plan §2.4)
# and workplace (plan §4.5 / PM brief §6 step 4).  The workplace prior
# concentrates mass on J1772 6.6 - 7.7 kW per Pritchard 2023 + DOE
# workplace charging guidance.
#
# v2.1 recalibration (evse_enrichment_plan §B.2):
#   * Residential adds a 9.6 kW bucket (NEMA 14-50 40 A ceiling) [lit §5.54]
#     and downweights the unconditional 19.2 kW mass from 0.10 to 0.03
#     because 19.2 kW residential only makes physical sense paired with a
#     small "80 A-eligible" OBC list (Lucid Air, GMC Hummer EV, Silverado
#     EV, MY22-23 F-150 Lightning ER, select Tesla S/X) — see
#     [lit §5.58, §5.59, §5.60].  The OEM-attach 19.2 kW share is supplied
#     by the hard-coupling layer in §B.4 (~3-5 pp on top of the 3 %
#     unconditional mass).
#   * Workplace modestly downweights the 19.2 kW pedestal (rare outside
#     fleet depots [lit §2.20, §5.55]) and adds a 9.6 kW bucket for
#     NEMA 14-50 fleet-vehicle pedestals.
EVSE_MIX_RESIDENTIAL: Final[dict[float, float]] = {
    3.3:  0.08,   # legacy L1 / older portable [lit §1.3]
    6.6:  0.27,   # J1772 32 A on a NEMA 14-50 [lit §1.6], [lit §5.54]
    7.7:  0.22,   # 32 A hardwired (HCS-40 family) [lit §3.34], [lit §5.54]
    9.6:  0.14,   # NEW BUCKET — 40 A NEMA 14-50 ceiling [lit §5.54]
    11.5: 0.26,   # 48 A hardwired (Tesla Wall Connector / Emporia Pro)
                  # [lit §3.25], [lit §3.33], [lit §5.54]
    19.2: 0.03,   # 80 A; reserves for the §B.4 hard-coupling layer to
                  # consume [lit §5.58], [lit §5.60]
}
EVSE_MIX_WORKPLACE: Final[dict[float, float]] = {
    3.3:  0.04,
    6.6:  0.38,   # J1772 32 A dual-port shared-50A-circuit [lit §2.20]
    7.7:  0.34,   # J1772 32 A single hardwired (workplace canonical)
                  # [lit §2.17], [lit §2.18]
    9.6:  0.08,   # NEMA 14-50 fleet-vehicle pedestals
    11.5: 0.14,   # 48 A workplace upgrade tier
    19.2: 0.02,   # depot / charge-management special-case [lit §5.55]
}

# ---------------------------------------------------------------------------
# v2.1 — EVSE enrichment constants (plan §B).
# ---------------------------------------------------------------------------

# v2.1 EVSE-enrichment category universes (plan §A.2). Both must be
# package-wide stable -- same across all 8 regions and both profile types
# -- so the parquet categorical code indices are reproducible.
EVSE_BRAND_CATEGORIES: Final[tuple[str, ...]] = (
    "Tesla",                # [lit §3.21], [lit §3.25-26]
    "ChargePoint",          # [lit §3.27-28], [lit §3.39], [lit §9.91]
    "Wallbox",              # [lit §3.29-30], [lit §3.36]
    "Emporia",              # [lit §3.33], [lit §3.21]
    "ClipperCreek",         # [lit §3.34] (Enphase brand)
    "Grizzl-E",             # [lit §3.35]
    "JuiceBox",             # [lit §6.69] (legacy installed base; net-new = 0)
    "Blink",                # [lit §3.31-32], [lit §3.40]
    "FLO",                  # [lit §3.45]
    "EVgo",                 # [lit §3.41]
    "Electrify_America",    # [lit §3.42]
    "Schneider",            # [lit §3.37]
    "Lectron",              # [lit §3.38]
    "Rivian",               # [lit §9.88]
    "Lucid",                # [lit §9.89]
    "GM_Energy",            # [lit §9.90], [lit §7.75]
    "Ford_CSP",             # [lit §7.71], [lit §9.87]
    "Other",                # tail / unknown
)

EVSE_CONNECTOR_CATEGORIES: Final[tuple[str, ...]] = (
    "J1772",                # SAE J1772 [lit §5.53]
    "NACS",                 # SAE J3400 [lit §4.46], [lit §4.50]
    "Tesla_NACS_native",    # pre-J3400 Tesla wallboxes [lit §3.25]
    "Universal_NACS_J1772", # Tesla Universal Wall Connector [lit §3.26]
    "CCS1",                 # commercial DCFC [lit §4.47], [lit §4.51]
    "CHAdeMO",              # legacy DCFC (residual; commercial only)
    "NEMA_14_50",           # portable / plug-in residential L2 [lit §4.52]
    "NEMA_6_50",            # plug-in residential L2 (no-neutral garage) [lit §4.52]
    "NEMA_14_30",           # dryer-retrofit plug-in L2 [lit §4.52]
)

# B.1.1 — Residential EVSE brand prior. Triangulated from:
#   - J.D. Power EVX 2024/2025 brand-pool ranking [lit §3.21, §3.22];
#   - Consumer Reports owner survey "Tesla + ChargePoint ~50 %" [lit §3.23];
#   - electrician-side EnergySage convenience sample [lit §3.24];
#   - ChargePoint 10-K residential vertical [lit §3.27];
#   - Wallbox 20-F NA segment [lit §3.29-30];
#   - Blink 10-K residential HQ200/IQ200 footprint [lit §3.31];
#   - JuiceBox legacy install base [lit §6.69];
#   - ClipperCreek HCS-40 dumb-tier reference [lit §3.34];
#   - Grizzl-E EVX #3 [lit §3.35];
#   - Emporia EVX #2 [lit §3.33];
#   - "OEM-bundled accounts for 46.92 % of market share in 2024" sanity
#     check from Mordor Intelligence 2025 [lit §10.111].
# Sum: 1.000 (guarded by _assert_evse_brand_priors_ok).
EVSE_BRAND_MIX_RESIDENTIAL: Final[dict[str, float]] = {
    "Tesla":         0.27,   # [lit §3.21], [lit §3.23]
    "ChargePoint":   0.18,   # [lit §3.23], [lit §3.27], [lit §9.91]
    "Emporia":       0.10,   # [lit §3.33]
    "ClipperCreek":  0.08,   # [lit §3.34]
    "Grizzl-E":      0.07,   # [lit §3.35]
    "Wallbox":       0.06,   # [lit §3.29-30]
    "JuiceBox":      0.05,   # [lit §6.69] (installed-base figure)
    "Blink":         0.03,   # [lit §3.31]
    "Schneider":     0.02,   # [lit §3.37]
    "Lectron":       0.04,   # [lit §3.38]
    "Rivian":        0.02,   # [lit §9.88]
    "Lucid":         0.005,  # [lit §9.89]
    "GM_Energy":     0.01,   # [lit §9.90]
    "Ford_CSP":      0.02,   # [lit §7.71], [lit §9.87]
    # Tail brand (NRG eVgo Home, ADC, Hwisel, DIY portable). Calibrated
    # to 0.045 so the prior sums to exactly 1.000 -- the plan §B.1.1
    # table's 0.055 was a rounding-induced 0.01 overshoot that
    # _assert_evse_brand_priors_ok() catches at import time.
    "Other":         0.045,
}

# B.1.2 — Workplace EVSE brand prior. Thinner evidence base; the lit
# review explicitly notes "no published port-by-brand workplace
# tabulation". ChargePoint dominance is attested ([lit §2.18, §3.27])
# but the long tail is industry-disclosure-only.
EVSE_BRAND_MIX_WORKPLACE: Final[dict[str, float]] = {
    "ChargePoint":        0.55,  # [lit §3.27-28], [lit §2.18]
    "Blink":              0.15,  # [lit §3.31-32] post-SemaConnect rebrand
    "FLO":                0.08,  # [lit §3.45]
    "Tesla":              0.06,  # commercial Wall Connector pedestals
    "Wallbox":            0.04,  # [lit §3.29] eM4/eMC commercial AC
    "EVgo":               0.03,  # [lit §3.41]
    "Electrify_America":  0.02,  # [lit §3.42]
    "ClipperCreek":       0.02,  # [lit §3.34] educational/gov bulk
    "Schneider":          0.02,  # [lit §3.37]
    "Other":              0.03,  # ABB, Tritium, Delta, BTC tail
}

# B.1.3 — OBC-conditioned brand multiplier. Soft prior tie-breaker for
# the residential brand draw: boosts the high-OBC bundle SKUs
# (Ford_CSP, GM_Energy, Lucid) inside the 19.2 kW bucket and
# downweights price-aggressive low-amp brands at the 3.3 / 6.6 kW
# tail. Multiplicative reweighting on top of the base mix; "no entry"
# is treated as 1.0 (identity). All renormalisation happens in §C.
OBC_BUCKET_BRAND_MULTIPLIERS: Final[dict[float, dict[str, float]]] = {
    19.2: {  # 80 A-eligible OBC bucket (Lucid, GM, Lightning ER MY22-23)
        "Tesla": 0.6, "ChargePoint": 0.8, "Emporia": 0.4,
        "Grizzl-E": 0.3, "Lectron": 0.05,
        "Ford_CSP": 4.0, "GM_Energy": 4.0, "Lucid": 6.0,
        "Wallbox": 1.5, "Other": 1.8,
    },
    11.5: {},  # identity (base prior)
    7.7:  {"Lectron": 1.6, "Grizzl-E": 1.4, "JuiceBox": 1.2, "Ford_CSP": 0.05},
    6.6:  {"Lectron": 2.0, "ClipperCreek": 1.4, "Ford_CSP": 0.05, "GM_Energy": 0.1},
    3.3:  {"Lectron": 3.0, "ClipperCreek": 1.4, "Ford_CSP": 0.0, "GM_Energy": 0.0},
}


def _assert_evse_brand_priors_ok() -> None:
    """Module-import-time invariant for the EVSE brand priors (plan §B.1.4).

    Verifies that:
      * residential + workplace dicts each sum to 1.0;
      * every key is registered in EVSE_BRAND_CATEGORIES;
      * the OBC-bucket multipliers reference only registered brand keys.
    """

    for name, prior in (
        ("EVSE_BRAND_MIX_RESIDENTIAL", EVSE_BRAND_MIX_RESIDENTIAL),
        ("EVSE_BRAND_MIX_WORKPLACE",   EVSE_BRAND_MIX_WORKPLACE),
    ):
        s = sum(prior.values())
        if abs(s - 1.0) > 1e-9:
            raise AssertionError(
                f"{name} sums to {s!r}, expected 1.0 (gap {abs(s - 1.0):.3e})"
            )
        for k in prior:
            if k not in EVSE_BRAND_CATEGORIES:
                raise AssertionError(
                    f"{name} key {k!r} not in EVSE_BRAND_CATEGORIES"
                )
    for bucket, mults in OBC_BUCKET_BRAND_MULTIPLIERS.items():
        for k in mults:
            if k not in EVSE_BRAND_CATEGORIES:
                raise AssertionError(
                    f"OBC_BUCKET_BRAND_MULTIPLIERS[{bucket!r}] key {k!r} "
                    "not in EVSE_BRAND_CATEGORIES"
                )


_assert_evse_brand_priors_ok()


# B.3 — Model-year-conditional OBC overrides (Ford Lightning ER step-down
# documented at [lit §5.60]; Lucid, GM 19.2 kW SKUs at [lit §5.58]). Each
# tuple is ``(make, model, my_min, my_max, OBC_kw)``; a ``model == "*"``
# entry matches all models for that make. Applied AFTER the categorical
# OBC draw in :func:`sample_fleet` so it overrides the random sample but
# stays consistent with the per-model OBC physical ceiling listed in
# ``sales_mix_data._MODEL_ATTRS``.
OBC_MODEL_YEAR_OVERRIDES: Final[tuple[tuple[str, str, int, int, float], ...]] = (
    # (make,        model,             my_min, my_max, OBC_kw)
    ("Ford",        "F150_Lightning",  2022,   2023,   19.2),  # [lit §5.60], [lit §9.87]
    ("Ford",        "F150_Lightning",  2024,   2099,   11.5),  # [lit §5.60] step-down
    ("Lucid",       "*",               2021,   2099,   19.2),  # [lit §5.58], [lit §9.89]
    ("Chevrolet",   "Silverado_EV",    2024,   2099,   19.2),  # [lit §5.58], [lit §9.90]
    ("GMC",         "Hummer_EV",       2022,   2099,   19.2),  # [lit §5.58]
    ("Rivian",      "R1T",             2022,   2099,   11.5),  # [lit §9.88]
    ("Rivian",      "R1S",             2022,   2099,   11.5),  # [lit §9.88]
)


# B.4 — Hard EV<->EVSE bundle rules. Applied AFTER the base brand/kW draw
# but BEFORE the panel-cap integrity check. Wildcard semantics:
#   * ``predicate_model == "*"``     matches any model for that make.
#   * ``predicate_powertrain == "*"`` matches BEV or PHEV.
# Exclusivity: first rule in tuple order wins; order rules from
# most-specific to most-general. ``overrides`` is a dict that may carry:
#   * ``"EVSE_brand"`` -> replace the base-sampled brand;
#   * ``"EVSE_home_kw"`` -> replace the base-sampled kW;
#   * ``"EVSE_connector"`` -> hard-pin the connector (passes through to
#     :func:`_sample_connector` as a pending override);
#   * ``"EVSE_home_kw_max"`` -> CLIP the base-sampled kW to ``min(base,
#     max)`` instead of replacing. Used by the MY24+ Lightning step-down,
#     which forces a power ceiling without forcing a brand.
@dataclass(frozen=True)
class EvEvseBundleRule:
    """One row of :data:`EV_EVSE_BUNDLE_RULES`.

    Hard-coupling rule applying an EVSE-attribute override to EVs matching
    a (make, model, powertrain, model_year band) predicate, with a
    Bernoulli take-rate gating per-EV application.
    """

    name: str
    predicate_make: str          # exact match; "*" wildcard
    predicate_model: str         # exact match; "*" wildcard
    predicate_powertrain: str    # "BEV" / "PHEV" / "*"
    model_year_min: int
    model_year_max: int
    overrides: dict             # {"EVSE_brand": ..., "EVSE_home_kw": ..., ...}
    take_rate: float
    citation: str                # "[lit §N.M]" reference for the validator


EV_EVSE_BUNDLE_RULES: Final[tuple[EvEvseBundleRule, ...]] = (
    # ---- Ford Lightning Extended Range Charge Station Pro bundle ----
    # User-locked take-rate 0.70 (plan §I.3 rationale: bundle is technically
    # 1.0 attach gratis, but resale + aftermarket switching shrinks the
    # installed-base attach to ~0.70).
    EvEvseBundleRule(
        name="ford_lightning_er_my22_23_csp",
        predicate_make="Ford", predicate_model="F150_Lightning",
        predicate_powertrain="BEV",
        model_year_min=2022, model_year_max=2023,
        overrides={
            "EVSE_brand": "Ford_CSP",
            "EVSE_home_kw": 19.2,
            "EVSE_connector": "J1772",
        },
        take_rate=0.70,
        citation="[lit §7.71], [lit §7.73], [lit §9.87]",
    ),
    # MY24+ step-down: no CSP bundle; EVSE_home_kw forced <= 11.5 because
    # OBC = 11.3 kW. Brand reverts to the base prior.
    EvEvseBundleRule(
        name="ford_lightning_er_my24p_step_down",
        predicate_make="Ford", predicate_model="F150_Lightning",
        predicate_powertrain="BEV",
        model_year_min=2024, model_year_max=2099,
        overrides={"EVSE_home_kw_max": 11.5},
        take_rate=1.0,
        citation="[lit §5.60]",
    ),

    # ---- Hyundai / Kia ChargePoint Home Flex bundle (MY24-26) ----
    # [lit §9.91] does not disclose charger-vs-credit split; the ~0.55
    # take-rate is a back-of-envelope assumption (see plan §I.1). Wide
    # validator band gates it.
    EvEvseBundleRule(
        name="hyundai_ioniq_chargepoint_home_flex",
        predicate_make="Hyundai", predicate_model="*",
        predicate_powertrain="BEV",
        model_year_min=2024, model_year_max=2026,
        overrides={
            "EVSE_brand": "ChargePoint",
            "EVSE_home_kw": 11.5,
            "EVSE_connector": "J1772",
        },
        take_rate=0.55,
        citation="[lit §9.91]",
    ),
    EvEvseBundleRule(
        name="kia_ioniq_chargepoint_home_flex",
        predicate_make="Kia", predicate_model="*",
        predicate_powertrain="BEV",
        model_year_min=2024, model_year_max=2026,
        overrides={
            "EVSE_brand": "ChargePoint",
            "EVSE_home_kw": 11.5,
            "EVSE_connector": "J1772",
        },
        take_rate=0.55,
        citation="[lit §9.91]",
    ),

    # ---- Tesla self-selection (soft, not OEM-bundled) ----
    # Modelled as a high take-rate self-selection rather than a true
    # bundle. The [lit §3.23] Consumer Reports "Tesla + ChargePoint ~50 %"
    # data point combined with the J.D. Power EVX #1 Tesla 790/1000
    # ranking [lit §3.21] back out ~0.62 Tesla owners installing a Tesla
    # Wall Connector.
    EvEvseBundleRule(
        name="tesla_wall_connector_self_selection",
        predicate_make="Tesla", predicate_model="*",
        predicate_powertrain="BEV",
        model_year_min=2018, model_year_max=2099,
        overrides={
            # S2: do NOT pin EVSE_connector here -- _sample_connector
            # already forces Tesla_NACS_native pre-2024 and draws from
            # the Tesla band post-2024 (plan §B.5,
            # EVSE_CONNECTOR_MIX_BY_MAKE_MY["Tesla"] = 95% native / 5% NACS).
            # S3: Tesla Wall Connector is physically capped at 11.5 kW
            # (lit §3.25), so clip the EVSE_home_kw when this self-
            # selection rule fires.
            "EVSE_brand": "Tesla",
            "EVSE_home_kw_max": 11.5,
        },
        take_rate=0.62,
        citation="[lit §3.21], [lit §3.23], [lit §9.94]",
    ),

    # ---- Rivian R1T / R1S Wall Charger bundle (closed Dec 31, 2023) ----
    # [lit §9.88] documents a $2,000 Qmerit credit OR the Rivian Wall
    # Charger; take-rate not disclosed -- ~0.60 is a triangulated guess
    # (see plan §I.1).
    EvEvseBundleRule(
        name="rivian_r1tr1s_wall_charger_bundle",
        predicate_make="Rivian", predicate_model="*",
        predicate_powertrain="BEV",
        model_year_min=2022, model_year_max=2023,
        overrides={
            "EVSE_brand": "Rivian",
            "EVSE_home_kw": 11.5,
            "EVSE_connector": "J1772",
        },
        take_rate=0.60,
        citation="[lit §9.88]",
    ),

    # ---- Lucid Air 19.2 kW optional wallbox (OBC-physics-driven) ----
    # No published OEM bundle; the 19.2 kW OBC + Lucid Connected Home
    # Charging Station coupling is OBC-physics + a Lucid-store optional
    # SKU. ~0.45 take the Lucid SKU; remainder split among aftermarket
    # 19.2 kW SKUs (Wallbox Quasar 2, dcbel Ara, NRG/Hwisel). PLACEHOLDER
    # rule -- Lucid not currently in sales_mix_data._MODEL_ATTRS, so it
    # is a no-op until that table is expanded (plan §I.6).
    EvEvseBundleRule(
        name="lucid_air_19_2kw_optional_wallbox",
        predicate_make="Lucid", predicate_model="*",
        predicate_powertrain="BEV",
        model_year_min=2021, model_year_max=2099,
        overrides={
            "EVSE_brand": "Lucid",
            "EVSE_home_kw": 19.2,
            "EVSE_connector": "J1772",
        },
        take_rate=0.45,
        citation="[lit §5.58], [lit §9.89], [lit §9.97]",
    ),

    # ---- GM 19.2 kW models (Silverado EV / Hummer EV) ----
    # PowerShift is a paid optional SKU rather than a true bundle, so the
    # ~0.40 take-rate captures the "willing to pay" subset of buyers.
    EvEvseBundleRule(
        name="gm_19_2kw_powershift_charger",
        predicate_make="Chevrolet", predicate_model="Silverado_EV",
        predicate_powertrain="BEV",
        model_year_min=2024, model_year_max=2099,
        overrides={
            "EVSE_brand": "GM_Energy",
            "EVSE_home_kw": 19.2,
            "EVSE_connector": "J1772",
        },
        take_rate=0.40,
        citation="[lit §5.58], [lit §7.75], [lit §9.90]",
    ),
    # Hummer EV — PLACEHOLDER rule (plan §I.6). Not currently in
    # sales_mix_data._MODEL_ATTRS so this is a no-op until the model is
    # added in a future release. Left in to document intent.
    EvEvseBundleRule(
        name="gmc_hummer_ev_powershift_charger",
        predicate_make="GMC", predicate_model="Hummer_EV",
        predicate_powertrain="BEV",
        model_year_min=2022, model_year_max=2099,
        overrides={
            "EVSE_brand": "GM_Energy",
            "EVSE_home_kw": 19.2,
            "EVSE_connector": "J1772",
        },
        take_rate=0.40,
        citation="[lit §5.58], [lit §7.75], [lit §9.90]",
    ),
)


# B.5 — Connector-mix-by-(make, model_year) tables. The SAE J3400 / NACS
# transition timeline per [lit §4.46-50] and OEM-specific press releases.
# Each value is a tuple of ``(my_min, my_max, prior_dict)`` ordered so the
# first matching range wins. The "*" key is the default for any OEM not
# listed below. Brand is sampled separately; connector is dispatched on
# (make, model_year).
EVSE_CONNECTOR_MIX_BY_MAKE_MY: Final[dict[str, tuple[tuple[int, int, dict[str, float]], ...]]] = {
    "Tesla": (
        (2018, 2099, {"Tesla_NACS_native": 0.95, "NACS": 0.05}),
    ),
    "Ford": (
        (2018, 2023, {"J1772": 0.98, "NACS": 0.02}),
        (2024, 2024, {"J1772": 0.85, "NACS": 0.15}),  # [lit §4.49]
        (2025, 2099, {"J1772": 0.50, "NACS": 0.50}),
    ),
    "Chevrolet": (
        (2018, 2024, {"J1772": 0.95, "NACS": 0.05}),
        (2025, 2099, {"J1772": 0.45, "NACS": 0.55}),  # [lit §4.49]
    ),
    "GMC": (
        (2022, 2024, {"J1772": 0.95, "NACS": 0.05}),
        (2025, 2099, {"J1772": 0.45, "NACS": 0.55}),
    ),
    "Rivian": (
        (2022, 2024, {"J1772": 0.95, "NACS": 0.05}),
        (2025, 2099, {"J1772": 0.40, "NACS": 0.60}),  # [lit §9.88]
    ),
    "Hyundai": (
        (2022, 2024, {"J1772": 0.95, "NACS": 0.05}),
        (2025, 2099, {"J1772": 0.50, "NACS": 0.50}),
    ),
    "Kia": (
        (2022, 2024, {"J1772": 0.95, "NACS": 0.05}),
        (2025, 2099, {"J1772": 0.50, "NACS": 0.50}),
    ),
    # Default: J1772 with a slow NACS ramp from MY 2025.
    "*": (
        (2018, 2024, {"J1772": 0.97, "NACS": 0.03}),
        (2025, 2099, {"J1772": 0.75, "NACS": 0.25}),
    ),
}


# B.5 — NEMA outlet prior used when ``EVSE_brand == "Lectron"`` (portable-
# only per [lit §3.38]) or when the EV ships only with a mobile connector.
# Represents the electrical outlet feeding a portable EVSE -- still an
# ``EVSE_connector`` value because it is what the car ultimately plugs into.
NEMA_OUTLET_MIX: Final[dict[str, float]] = {
    "NEMA_14_50": 0.78,   # [lit §4.52]
    "NEMA_6_50":  0.12,   # detached-garage 3-wire [lit §4.52]
    "NEMA_14_30": 0.10,   # dryer-retrofit [lit §4.52]
}


# Workplace site-side connector prior (MY-band invariant). Workplace EVSE
# is owned by the site, not the vehicle, so the EV-side make/model is
# irrelevant. Defaults to 80 % J1772 / 15 % NACS / 5 % CCS1.
EVSE_CONNECTOR_MIX_WORKPLACE: Final[dict[str, float]] = {
    "J1772": 0.80,
    "NACS":  0.15,
    "CCS1":  0.05,
}

# Wh/mi nominal per archetype (EPA labels; plan §2.5).
WH_PER_MI_BY_ARCHETYPE: Final[dict[str, float]] = {
    "sedan": 250.0,
    "crossover": 295.0,
    "suv": 450.0,
    "pickup": 510.0,
}
WH_PER_MI_PHEV_OVERRIDE: Final[float] = 320.0  # PHEV-electric (plan §2.5).

# Makes whose 2021+ vehicles ship with heat-pump cabin HVAC
# (Atkinson 2025 PMC12496165; plan §2.5).
HEATPUMP_MAKES: Final[frozenset[str]] = frozenset(
    {"Tesla", "Hyundai", "Kia", "Genesis", "Lucid"}
)
HEATPUMP_MIN_MODEL_YEAR: Final[int] = 2021

#: RFC-032: per-EV sub-seed namespace offset for the stratified
#: heat-pump resampling RNG. The RNG is keyed by
#: ``master_seed + MODULE_OFFSETS["vehicle_archetypes"] + 9_000_000``
#: so it is disjoint from every other M2 stream.
HEATPUMP_RESAMPLE_SUBSEED_OFFSET: Final[int] = 9_000_000

#: v2.1 — per-EV sub-seed namespace offset for the EVSE brand draw.
#: Keyed by ``master_seed + MODULE_OFFSETS["vehicle_archetypes"] +
#: EVSE_BRAND_SAMPLE_SUBSEED_OFFSET``. Registered in
#: :data:`pev_synth._seeds.SUBSEED_OFFSETS` so the pairwise-gap invariant
#: covers it (gap from soc0=8M -> heat_pump=9M -> brand=10M is exactly
#: N_EV_MAX=1M).
EVSE_BRAND_SAMPLE_SUBSEED_OFFSET: Final[int] = 10_000_000

#: v2.1 — per-EV sub-seed namespace offset for the EVSE connector draw.
#: Keyed by ``master_seed + MODULE_OFFSETS["vehicle_archetypes"] +
#: EVSE_CONNECTOR_SAMPLE_SUBSEED_OFFSET``. Gap of 1M from
#: :data:`EVSE_BRAND_SAMPLE_SUBSEED_OFFSET` (10M -> 11M) hits the
#: minimum-gap invariant.
EVSE_CONNECTOR_SAMPLE_SUBSEED_OFFSET: Final[int] = 11_000_000

# CVRP model-year histogram for 2018..2025 (plan §2.6 RI-5).
# v1.1 carry-forward; the v2.0 sales-mix CSVs carry per-model
# (model_year_min, model_year_max), but the per-row model year is still
# sampled from the empirical CVRP rebate-by-model-year histogram per
# powertrain (the brief's §6 step 3 lookup table).  Out-of-band years are
# clipped to the per-model [model_year_min, model_year_max].
CVRP_MODEL_YEAR_HIST: Final[dict[str, dict[int, int]]] = {
    "BEV": {
        2018: 350, 2019: 600, 2020: 800, 2021: 1500,
        2022: 4000, 2023: 5500, 2024: 6500, 2025: 3000,
    },
    "PHEV": {
        2018: 600, 2019: 700, 2020: 750, 2021: 1200,
        2022: 1800, 2023: 2200, 2024: 1900, 2025: 900,
    },
}

# Models classified as PHEV by the canonical (make, model) -> attrs table
# in sales_mix_data.  We re-derive the set from the loaded CSV at runtime,
# but maintain a tested allow-list here so a malformed CSV still produces
# a sensible BEV/PHEV split.  This list mirrors the PHEV rows in
# sales_mix_data._MODEL_ATTRS.
PHEV_MODEL_REGISTRY: Final[frozenset[tuple[str, str]]] = frozenset(
    {
        ("Toyota", "Prius_Prime"),
        ("Toyota", "RAV4_Prime"),
        ("Ford", "Escape_PHEV"),
        ("Lincoln", "Corsair_PHEV"),
        ("Jeep", "Wrangler_4xe"),
        ("Jeep", "Grand_Cherokee_4xe"),
        ("Chrysler", "Pacifica_PHEV"),
        ("BMW", "X5_xDrive50e"),
        ("Mitsubishi", "Outlander_PHEV"),
        ("Volvo", "XC60_Recharge"),
        ("Hyundai", "Tucson_PHEV"),
        ("Kia", "Sorento_PHEV"),
    }
)


# ---------------------------------------------------------------------------
# Schema and dtype contract.
# ---------------------------------------------------------------------------

POWERTRAIN_CATEGORIES: Final[tuple[str, ...]] = ("BEV", "PHEV")
ARCHETYPE_CATEGORIES: Final[tuple[str, ...]] = ("sedan", "crossover", "suv", "pickup")
GARAGE_CATEGORIES: Final[tuple[str, ...]] = ("dedicated", "shared", "curb", "workplace_lot")
# EVSE_BRAND_CATEGORIES and EVSE_CONNECTOR_CATEGORIES are defined earlier
# (just before EVSE_BRAND_MIX_RESIDENTIAL) because the import-time
# brand-prior invariant references them. They remain part of the
# canonical category-set used by ``_cast_to_schema`` below.

# (column, pandas-dtype, nullable).  Order is the canonical column order
# written to fleet.parquet.  ``profile_type`` and ``region_name`` are the
# two new v2.0 columns; the rest mirror v1.1.
SCHEMA: Final[tuple[tuple[str, str, bool], ...]] = (
    ("ev_id", "int32", False),
    ("profile_type", "category", False),
    ("region_name", "category", False),
    ("powertrain", "category", False),
    ("make", "category", False),
    ("model", "category", False),
    ("model_year", "int16", False),
    ("archetype", "category", False),
    ("B_kwh", "float32", False),
    ("OBC_kw", "float32", False),
    ("EVSE_home_kw", "float32", False),
    ("panel_cap_kw", "float32", False),
    # v2.1 — EVSE enrichment (plan §A.1). Both non-nullable categorical;
    # the category universes are EVSE_BRAND_CATEGORIES and
    # EVSE_CONNECTOR_CATEGORIES respectively. Descriptive — they do NOT
    # alter the charging-power bottleneck semantics of EVSE_home_kw.
    ("EVSE_brand", "category", False),
    ("EVSE_connector", "category", False),
    ("eta_charge_nominal", "float32", False),
    ("has_heat_pump", "bool", False),
    ("Wh_per_mi_nominal", "float32", False),
    ("garage_type", "category", False),
    ("income_tier", "category", False),
    ("home_station", "category", False),
    ("seed", "int64", False),
    ("soc_min_kwh", "float32", False),
    # v3.0 (§13.6 RESOLVED) — PHEV charge-sustaining combined MPG from
    # EPA fueleconomy.gov per (make, model, model_year); used by the M6
    # SoC ledger to translate below-floor trip deficits into
    # ``gas_kwh_equivalent``. Nullable: BEV rows carry NaN since the
    # lookup is powertrain-specific. PHEV rows fall back to
    # :data:`pev_synth._phev_fuel_economy.PHEV_CS_MPG_FALLBACK`
    # (33 MPG combined, EPA per-vehicle median across the v3.0 PHEV
    # cohort) when the (make, model, model_year) triple is not listed
    # in :data:`pev_synth._phev_fuel_economy.PHEV_CS_MPG`.
    ("phev_fuel_economy_mpg", "float32", True),
)


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------

def _resolve_region(region: Region | str) -> Region:
    """Return a ``Region`` instance from a name string or pass-through."""
    if isinstance(region, Region):
        return region
    if isinstance(region, str):
        return Region.from_name(region)
    raise TypeError(
        f"region must be a Region instance or a registered region name, "
        f"got {type(region).__name__}"
    )


def _validate_profile_type(profile_type: str) -> None:
    if profile_type not in PROFILE_TYPES:
        raise ValueError(
            f"profile_type {profile_type!r} not supported; "
            f"valid: {sorted(PROFILE_TYPES)}"
        )


def _classify_powertrain(mix: pd.DataFrame) -> pd.Series:
    """Add a 'powertrain' label column to a sales-mix DataFrame."""
    # The canonical PHEV (make, model) set lives in PHEV_MODEL_REGISTRY.
    keys = list(zip(mix["make"].astype(str), mix["model"].astype(str), strict=True))
    return pd.Series(
        ["PHEV" if k in PHEV_MODEL_REGISTRY else "BEV" for k in keys],
        index=mix.index,
        dtype="string",
    )


def _apply_sales_mix_overlay(
    mix: pd.DataFrame,
    overlay: dict[str, float] | tuple[tuple[str, float], ...] | object,
) -> pd.DataFrame:
    """Apply ``region.sales_mix_overlay`` arithmetic per plan §2.7.

    The overlay is a ``{archetype_key: delta}`` mapping (e.g. DFW's
    ``{"pickup_overlay": 0.10}`` or the synonym ``{"pickup": 0.10}`` from
    the PM brief).  For each (archetype, delta) entry, we:

        1. Scale every row of that archetype so the archetype's share
           increases by ``delta`` (additively, in absolute share units).
        2. Renormalise the entire mix so shares still sum to 1.0.

    If the overlay key is not recognised as an archetype, it is treated
    as a no-op (we never invent new archetype rows).  Negative deltas are
    permitted; the resulting per-row share is floored at 0.0 before
    renormalisation.
    """
    # Coerce overlay to a plain dict for downstream use; accepts the v2.0
    # canonical tuple-of-pairs (RFC-007) or a legacy dict.
    if not isinstance(overlay, dict):
        overlay = dict(overlay)  # type: ignore[arg-type]
    if not overlay:
        return mix

    out = mix.copy()
    # Translate the brief's "pickup" key to "pickup_overlay" and v.v. so
    # both spellings are accepted (PM brief uses "pickup"; regions.py uses
    # "pickup_overlay").  Any other key with the "_overlay" suffix is also
    # stripped for archetype matching.
    def _archetype_of(k: str) -> str:
        if k.endswith("_overlay"):
            return k[: -len("_overlay")]
        return k

    archetype_totals = out.groupby("archetype")["share"].sum().to_dict()
    deltas_by_archetype: dict[str, float] = {}
    for raw_key, delta in overlay.items():
        deltas_by_archetype[_archetype_of(raw_key)] = float(delta)

    new_shares = out["share"].to_numpy(dtype=np.float64).copy()
    for arch, delta in deltas_by_archetype.items():
        mask = (out["archetype"].astype(str) == arch).to_numpy()
        cur_total = float(archetype_totals.get(arch, 0.0))
        target_total = cur_total + float(delta)
        if cur_total > 0 and target_total > 0:
            scale = target_total / cur_total
            new_shares[mask] = new_shares[mask] * scale
        elif target_total <= 0:
            new_shares[mask] = 0.0
        else:
            # archetype absent from the source mix and a positive delta:
            # we cannot synthesise rows, leave as-is (no-op).
            pass

    new_shares = np.maximum(new_shares, 0.0)
    total = new_shares.sum()
    if total <= 0:
        raise ValueError(
            f"sales-mix overlay {overlay} collapsed the mix to a zero "
            "total; cannot renormalise"
        )
    out["share"] = new_shares / total
    return out


def _powertrain_subset(mix: pd.DataFrame, powertrain: str) -> pd.DataFrame:
    """Return the rows for one powertrain with normalised weights."""
    if "powertrain" not in mix.columns:
        mix = mix.copy()
        mix["powertrain"] = _classify_powertrain(mix)
    sub = mix[mix["powertrain"].astype(str) == powertrain].reset_index(drop=True)
    if sub.empty:
        raise ValueError(
            f"sales mix has no rows for powertrain {powertrain!r}; cannot sample"
        )
    s = sub["share"].to_numpy(dtype=np.float64)
    sub = sub.copy()
    sub["weight_norm"] = s / s.sum()
    return sub


def _infer_bev_share(mix: pd.DataFrame) -> float:
    """Compute the BEV share implied by the (powertrain-labelled) mix."""
    if "powertrain" not in mix.columns:
        mix = mix.assign(powertrain=_classify_powertrain(mix))
    grouped = mix.groupby("powertrain")["share"].sum()
    bev = float(grouped.get("BEV", 0.0))
    phev = float(grouped.get("PHEV", 0.0))
    if bev + phev <= 0:
        return BEV_SHARE_LEGACY_DEFAULT
    return bev / (bev + phev)


def _sample_model_year(
    rng: np.random.Generator,
    powertrain: str,
    n: int,
    mix_pt: pd.DataFrame,
    chosen_idx: np.ndarray,
) -> np.ndarray:
    """Sample model_year per EV.

    Draws from the CVRP empirical histogram per powertrain (§2.6 RI-5)
    then clips each draw to the chosen model's
    ``[model_year_min, model_year_max]``.  If the row lacks year bounds
    (older CSVs), the unclipped draw is used.
    """
    hist = CVRP_MODEL_YEAR_HIST[powertrain]
    years = np.array(sorted(hist.keys()), dtype=np.int16)
    counts = np.array([hist[int(y)] for y in years], dtype=np.float64)
    probs = counts / counts.sum()
    raw = rng.choice(years, size=n, replace=True, p=probs).astype(np.int16)
    if "model_year_min" in mix_pt.columns and "model_year_max" in mix_pt.columns:
        my_min = mix_pt["model_year_min"].to_numpy(dtype=np.int16)[chosen_idx]
        my_max = mix_pt["model_year_max"].to_numpy(dtype=np.int16)[chosen_idx]
        raw = np.clip(raw, my_min, my_max)
    return raw.astype(np.int16)


def _sample_categorical(
    rng: np.random.Generator, mix: dict[float, float], n: int
) -> np.ndarray:
    """Draw ``n`` samples from a discrete-categorical {value: weight} map."""
    values = np.array(list(mix.keys()), dtype=np.float64)
    weights = np.array(list(mix.values()), dtype=np.float64)
    weights = weights / weights.sum()
    return rng.choice(values, size=n, replace=True, p=weights).astype(np.float64)


def _draw_eta(rng: np.random.Generator, n: int) -> np.ndarray:
    """Draw eta_nominal from truncated-N(mu, sigma^2) on [ETA_LO, ETA_HI]."""
    a = (ETA_LO - ETA_MU) / ETA_SIGMA
    b = (ETA_HI - ETA_MU) / ETA_SIGMA
    return truncnorm.rvs(a, b, loc=ETA_MU, scale=ETA_SIGMA, size=n, random_state=rng)


def _draw_garage(
    rng: np.random.Generator, n: int, profile_type: str
) -> np.ndarray:
    """Draw garage type.

    For ``profile_type == 'workplace'`` every row gets ``workplace_lot``.
    For residential: 0.78 dedicated, remainder split 50/50 shared/curb.
    """
    if profile_type == "workplace":
        return np.full(n, "workplace_lot", dtype=object)
    u = rng.random(size=n)
    out = np.empty(n, dtype=object)
    is_dedicated = u < GARAGE_DEDICATED_PROB
    out[is_dedicated] = "dedicated"
    rest = ~is_dedicated
    u2 = rng.random(size=rest.sum())
    rest_idx = np.where(rest)[0]
    out[rest_idx] = np.where(u2 < 0.5, "shared", "curb")
    return out


def _sample_archetype_rows(
    rng: np.random.Generator,
    mix_pt: pd.DataFrame,
    n: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Sample ``n`` rows from ``mix_pt``; return (idx, picked-frame)."""
    idx = rng.choice(
        len(mix_pt), size=n, replace=True, p=mix_pt["weight_norm"].to_numpy()
    )
    picked = mix_pt.iloc[idx][
        ["make", "model", "archetype", "battery_kwh_nominal", "obc_max_kw"]
    ].reset_index(drop=True)
    return idx, picked


def _resolve_home_stations(region: Region) -> tuple[str, ...]:
    """Return the temperature stations for ``region`` with a fallback."""
    if region.temperature_stations:
        return region.temperature_stations
    if region.name == "bay_area":
        return HOME_STATIONS_BAY_AREA_FALLBACK
    # us_national and any region with no stations yet: use the bay-area
    # fallback set so we always emit a non-empty categorical (v1.1
    # behaviour).  Downstream weather modules will replace this anyway.
    return HOME_STATIONS_BAY_AREA_FALLBACK


def _income_weights(region: Region) -> tuple[float, ...]:
    """Return the income-tier prior for ``region.income_tier_scheme``."""
    if region.income_tier_scheme == "acs_national":
        return INCOME_TIER_WEIGHTS_ACS_NATIONAL
    # cvrp_ca, or any unknown scheme: fall back to the v1.1 CA prior.
    return INCOME_TIER_WEIGHTS_CVRP_CA


def _evse_mix_for_profile(profile_type: str) -> dict[float, float]:
    if profile_type == "workplace":
        return EVSE_MIX_WORKPLACE
    return EVSE_MIX_RESIDENTIAL


def _panel_cap_prob_for_profile(profile_type: str) -> float:
    if profile_type == "workplace":
        return PANEL_CAP_PROB_WORKPLACE
    return PANEL_CAP_PROB_RESIDENTIAL


def _stratified_resample_heat_pump(
    has_heat_pump: np.ndarray,
    archetype_arr: np.ndarray,
    target_rate: float,
    seed: int,
) -> np.ndarray:
    """RFC-032: deterministically resample ``has_heat_pump`` to ``target_rate``.

    Strata are vehicle archetypes (``sedan``, ``crossover``, ``suv``,
    ``pickup``). Within each stratum we flip the minimum number of rows
    needed to hit the per-stratum target share (= ``target_rate``).
    Flips happen in deterministic row order (via a stable per-stratum
    ``np.random.default_rng(seed + stratum_idx)``), so two runs with the
    same ``seed`` produce identical fleets.
    """
    if not 0.0 <= float(target_rate) <= 1.0:
        raise ValueError(
            f"heat_pump_share_override must be in [0, 1]; got {target_rate}"
        )
    out = has_heat_pump.astype(bool).copy()
    n = out.shape[0]
    if n == 0:
        return out
    # Global target count anchored on the requested share. We allocate
    # per-stratum counts proportional to stratum size with stable
    # tie-breaking so the total matches the rounded global count.
    target_total = int(round(float(target_rate) * n))
    target_total = max(0, min(n, target_total))

    strata = list(ARCHETYPE_CATEGORIES)
    # Per-stratum target with proportional allocation; correct any
    # rounding drift on the largest stratum so totals match exactly.
    sizes = np.array(
        [int((archetype_arr == s).sum()) for s in strata], dtype=np.int64
    )
    raw = sizes * float(target_rate)
    per_stratum = np.floor(raw).astype(np.int64)
    drift = int(target_total - per_stratum.sum())
    if drift != 0:
        order = np.argsort(-(raw - per_stratum), kind="stable")
        for k in range(abs(drift)):
            j = int(order[k % len(order)])
            per_stratum[j] = max(
                0, min(int(sizes[j]), int(per_stratum[j]) + (1 if drift > 0 else -1))
            )

    for stratum_idx, stratum_name in enumerate(strata):
        idx = np.where(archetype_arr == stratum_name)[0]
        if idx.size == 0:
            continue
        rng = np.random.default_rng(int(seed) + int(stratum_idx))
        current = int(out[idx].sum())
        want = int(per_stratum[stratum_idx])
        if want == current:
            continue
        if want > current:
            # Need to flip more False rows to True.
            cand = idx[~out[idx]]
            n_flip = min(want - current, cand.size)
            if n_flip > 0:
                pick = rng.choice(cand, size=n_flip, replace=False)
                out[pick] = True
        else:
            # Need to flip True rows to False.
            cand = idx[out[idx]]
            n_flip = min(current - want, cand.size)
            if n_flip > 0:
                pick = rng.choice(cand, size=n_flip, replace=False)
                out[pick] = False
    return out


# ---------------------------------------------------------------------------
# v2.1 EVSE-enrichment helpers (plan §C).
# ---------------------------------------------------------------------------

def _apply_obc_overrides(
    obc_kw: np.ndarray,
    make_arr: np.ndarray,
    model_arr: np.ndarray,
    model_year: np.ndarray,
) -> np.ndarray:
    """Apply model-year-conditional OBC overrides (plan §B.3 option 2).

    Walks :data:`OBC_MODEL_YEAR_OVERRIDES` in order; for each rule, rows
    matching ``(make, model)`` with ``model_year in [my_min, my_max]`` are
    hard-pinned to the rule's ``OBC_kw``. ``model == "*"`` matches any
    model for the rule's make. The first matching rule wins per row.

    Citation: [lit §5.60] (Ford Lightning step-down), [lit §5.58]
    (Lucid / GM 19.2 kW SKUs), [lit §9.88] (Rivian).
    """

    out = obc_kw.astype(np.float32, copy=True)
    n = out.shape[0]
    applied = np.zeros(n, dtype=bool)
    for make, model, my_min, my_max, target_kw in OBC_MODEL_YEAR_OVERRIDES:
        if applied.all():
            break
        if model == "*":
            mask = (make_arr == make)
        else:
            mask = (make_arr == make) & (model_arr == model)
        mask &= (model_year >= my_min) & (model_year <= my_max) & (~applied)
        if not mask.any():
            continue
        out[mask] = np.float32(target_kw)
        applied[mask] = True
    return out


def _sample_brand(
    rng: np.random.Generator,
    base_mix: dict[str, float],
    bucket_kw: np.ndarray,
    obc_multipliers: dict[float, dict[str, float]],
    n: int,
) -> np.ndarray:
    """Per-EV draw from the OBC-conditioned brand prior (plan §C.2 step 8a).

    For each unique ``bucket_kw`` value (the OBC_kw bucket per plan §B.1.3),
    multiply ``base_mix`` by the per-bucket multiplier dict (identity if no
    entry), renormalise, then draw. Returns a length-``n`` object array of
    brand strings. Brand categories must be a subset of
    :data:`EVSE_BRAND_CATEGORIES`.
    """

    out = np.empty(n, dtype=object)
    base_keys = tuple(base_mix.keys())
    base_vals = np.array([base_mix[k] for k in base_keys], dtype=np.float64)
    # Pre-build the multiplier vector for each unique bucket so we draw
    # once per bucket rather than once per EV.
    unique_buckets = np.unique(bucket_kw)
    for bucket in unique_buckets:
        mults = obc_multipliers.get(float(bucket), {})
        if mults:
            mult_vec = np.array(
                [float(mults.get(k, 1.0)) for k in base_keys], dtype=np.float64
            )
        else:
            mult_vec = np.ones_like(base_vals)
        weights = base_vals * mult_vec
        total = weights.sum()
        if total <= 0.0:
            # All keys masked out — fall back to identity prior.
            weights = base_vals.copy()
            total = weights.sum()
        probs = weights / total
        bucket_mask = bucket_kw == bucket
        n_bucket = int(bucket_mask.sum())
        if n_bucket == 0:
            continue
        draws = rng.choice(np.array(base_keys, dtype=object),
                           size=n_bucket, replace=True, p=probs)
        out[bucket_mask] = draws
    return out


def _match_bundle_predicate(
    rule: EvEvseBundleRule,
    make_arr: np.ndarray,
    model_arr: np.ndarray,
    model_year: np.ndarray,
    powertrain_arr: np.ndarray,
) -> np.ndarray:
    """Return a boolean mask of rows matching ``rule``'s predicate."""

    if rule.predicate_make == "*":
        m_mask = np.ones(len(make_arr), dtype=bool)
    else:
        m_mask = make_arr == rule.predicate_make
    if rule.predicate_model == "*":
        mo_mask = np.ones(len(model_arr), dtype=bool)
    else:
        mo_mask = model_arr == rule.predicate_model
    if rule.predicate_powertrain == "*":
        p_mask = np.ones(len(powertrain_arr), dtype=bool)
    else:
        p_mask = powertrain_arr == rule.predicate_powertrain
    my_mask = (model_year >= rule.model_year_min) & (
        model_year <= rule.model_year_max
    )
    return m_mask & mo_mask & p_mask & my_mask


def _apply_bundle_rules(
    rng: np.random.Generator,
    make_arr: np.ndarray,
    model_arr: np.ndarray,
    model_year: np.ndarray,
    powertrain_arr: np.ndarray,
    evse_home_kw: np.ndarray,
    evse_brand_arr: np.ndarray,
    rules: tuple[EvEvseBundleRule, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, dict]]:
    """Apply hard EV<->EVSE coupling rules (plan §C.2 step 8b).

    Walks ``rules`` in order; for each rule, builds the matched mask,
    draws Bernoulli(take_rate), and applies the ``overrides`` dict to the
    marked rows. ``EVSE_home_kw_max`` is treated specially: it CLIPS the
    base-sampled kW to ``min(base, max)`` instead of replacing.

    Returns
    -------
    (evse_home_kw, evse_brand_arr, connector_pending, realised)
        ``connector_pending`` is a length-n object array carrying any
        connector forced by a bundle rule (or ``None`` when no rule fired
        for that row). ``realised`` is a per-rule dict
        ``{rule.name: {"n_match": int, "n_fired": int}}`` used by
        :func:`write_meta` to log realised take-rates.

    Notes
    -----
    First-match-wins semantics: once a row is touched by a rule that
    forces the brand, subsequent rules whose predicate also matches that
    row will skip it. (``EVSE_home_kw_max``-only rules do NOT mark a row
    as "touched" because they don't claim brand ownership.)
    """

    n = evse_home_kw.shape[0]
    evse_home_kw = evse_home_kw.astype(np.float32, copy=True)
    evse_brand_arr = evse_brand_arr.copy()
    connector_pending = np.full(n, None, dtype=object)
    realised: dict[str, dict] = {}
    claimed = np.zeros(n, dtype=bool)

    for rule in rules:
        pred_mask = _match_bundle_predicate(
            rule, make_arr, model_arr, model_year, powertrain_arr,
        )
        n_match = int(pred_mask.sum())
        eligible_mask = pred_mask & (~claimed)
        n_eligible = int(eligible_mask.sum())
        if n_eligible == 0:
            realised[rule.name] = {
                "n_match": n_match,
                "n_eligible": 0,
                "n_fired": 0,
                "target_take_rate": float(rule.take_rate),
            }
            continue
        u = rng.random(size=n_eligible)
        fire = u < float(rule.take_rate)
        fire_idx = np.where(eligible_mask)[0][fire]
        n_fired = int(fire_idx.size)

        overrides = rule.overrides
        # Apply EVSE_home_kw_max as a clip, not a replace.
        if "EVSE_home_kw_max" in overrides and n_fired:
            kw_max = np.float32(overrides["EVSE_home_kw_max"])
            evse_home_kw[fire_idx] = np.minimum(
                evse_home_kw[fire_idx], kw_max
            )
            # Note: clip-only rules do NOT claim brand ownership; the
            # base brand draw stays intact and downstream rules may
            # still fire on these rows.
        if "EVSE_home_kw" in overrides and n_fired:
            evse_home_kw[fire_idx] = np.float32(overrides["EVSE_home_kw"])
        if "EVSE_brand" in overrides and n_fired:
            evse_brand_arr[fire_idx] = overrides["EVSE_brand"]
            claimed[fire_idx] = True
        if "EVSE_connector" in overrides and n_fired:
            connector_pending[fire_idx] = overrides["EVSE_connector"]

        realised[rule.name] = {
            "n_match": n_match,
            "n_eligible": n_eligible,
            "n_fired": n_fired,
            "target_take_rate": float(rule.take_rate),
        }
    return evse_home_kw, evse_brand_arr, connector_pending, realised


def _connector_prior_for_make_my(make: str, model_year: int) -> dict[str, float]:
    """Look up the make/MY-band connector prior (plan §B.5)."""

    bands = EVSE_CONNECTOR_MIX_BY_MAKE_MY.get(make)
    if bands is None:
        bands = EVSE_CONNECTOR_MIX_BY_MAKE_MY["*"]
    for my_min, my_max, prior in bands:
        if my_min <= model_year <= my_max:
            return prior
    # No matching MY band -- fall back to the default wildcard's first
    # bucket. Should not happen for the MY ranges the sampler emits.
    return EVSE_CONNECTOR_MIX_BY_MAKE_MY["*"][0][2]


def _sample_connector(
    rng: np.random.Generator,
    make_arr: np.ndarray,
    model_year: np.ndarray,
    evse_brand_arr: np.ndarray,
    connector_pending: np.ndarray,
    profile_type: str,
) -> np.ndarray:
    """Per-EV connector draw (plan §C.2 step 8c).

    Dispatch order, per the plan:
      1. ``connector_pending[i]`` honoured if not None (bundle override).
      2. ``EVSE_brand == "Tesla"`` and pre-2024 -> ``Tesla_NACS_native``.
      3. ``EVSE_brand == "Lectron"`` (portable-only) -> draw from
         :data:`NEMA_OUTLET_MIX`.
      4. Workplace short-circuit: draw from
         :data:`EVSE_CONNECTOR_MIX_WORKPLACE` (site-side, MY-invariant).
      5. Make/MY-band lookup in
         :data:`EVSE_CONNECTOR_MIX_BY_MAKE_MY` with "*" fallback.
    """

    n = make_arr.shape[0]
    out = np.empty(n, dtype=object)
    # Pre-build categorical universe for workplace draws once.
    if profile_type == "workplace":
        wp_keys = np.array(list(EVSE_CONNECTOR_MIX_WORKPLACE.keys()), dtype=object)
        wp_probs = np.array(
            list(EVSE_CONNECTOR_MIX_WORKPLACE.values()), dtype=np.float64
        )
        wp_probs = wp_probs / wp_probs.sum()
    nema_keys = np.array(list(NEMA_OUTLET_MIX.keys()), dtype=object)
    nema_probs = np.array(list(NEMA_OUTLET_MIX.values()), dtype=np.float64)
    nema_probs = nema_probs / nema_probs.sum()

    for i in range(n):
        # 1. Bundle override pending.
        if connector_pending[i] is not None:
            out[i] = connector_pending[i]
            continue
        # 4. Workplace short-circuit (site-side).
        if profile_type == "workplace":
            out[i] = rng.choice(wp_keys, p=wp_probs)
            continue
        brand_i = evse_brand_arr[i]
        # 2. Tesla brand: pre-2024 -> Tesla_NACS_native; 2024+ -> draw
        # from Tesla's standard NACS/native mix.
        if brand_i == "Tesla":
            if model_year[i] < 2024:
                out[i] = "Tesla_NACS_native"
                continue
            tesla_bands = EVSE_CONNECTOR_MIX_BY_MAKE_MY["Tesla"]
            prior = tesla_bands[0][2]
            keys = np.array(list(prior.keys()), dtype=object)
            probs = np.array(list(prior.values()), dtype=np.float64)
            probs = probs / probs.sum()
            out[i] = rng.choice(keys, p=probs)
            continue
        # 3. Portable-only brand: outlet draw.
        if brand_i == "Lectron":
            out[i] = rng.choice(nema_keys, p=nema_probs)
            continue
        # 5. Make/MY-band lookup.
        prior = _connector_prior_for_make_my(str(make_arr[i]), int(model_year[i]))
        keys = np.array(list(prior.keys()), dtype=object)
        probs = np.array(list(prior.values()), dtype=np.float64)
        probs = probs / probs.sum()
        out[i] = rng.choice(keys, p=probs)
    return out


# ---------------------------------------------------------------------------
# Main sampler.
# ---------------------------------------------------------------------------

def sample_fleet(
    n: int = N_VEHICLES_DEFAULT,
    seed: int = MASTER_SEED_DEFAULT,
    region: Region | str = "bay_area",
    profile_type: str = "residential",
    bev_share: float | None = None,
) -> pd.DataFrame:
    """Sample a fleet of ``n`` synthetic vehicles for (region, profile_type).

    Parameters
    ----------
    n : int
        Number of vehicles to sample (default 1000; PM brief / plan §3.1).
    seed : int
        Master seed.  This module uses ``seed + MODULE_OFFSET`` (= seed+1)
        for its RNG so other modules cannot accidentally desync.
    region : Region | str
        ``Region`` instance or a registered region short-name (e.g.
        ``"bay_area"``, ``"dallas_fort_worth"``).  String lookups raise
        ``ValueError`` for unknown names via ``Region.from_name``.
    profile_type : str
        ``"residential"`` (default; v1.1 behaviour) or ``"workplace"``.
        Workplace fleets get the J1772 6.6 - 7.7 kW EVSE prior and a
        ``garage_type == "workplace_lot"`` for every row.
    bev_share : float | None
        Optional BEV share override in [0, 1].  If ``None`` the share is
        inferred from the loaded sales-mix CSV (cvrp_ca ~0.87,
        argonne_national ~0.85, nyserda_drive_clean ~0.83).

    Returns
    -------
    pd.DataFrame
        Fleet table conforming to ``SCHEMA``.
    """
    if not 1 <= n <= 50_000:
        raise ValueError(f"n must be in [1, 50_000]; got {n}")
    _validate_profile_type(profile_type)
    region_obj = _resolve_region(region)

    if region_obj.sales_mix_source not in SALES_MIX_SOURCES:
        raise ValueError(
            f"region {region_obj.name!r} references unknown sales_mix_source "
            f"{region_obj.sales_mix_source!r}; expected one of {SALES_MIX_SOURCES}"
        )

    # 1. Load the regional sales mix and apply the overlay (plan §2.7).
    mix_raw = load_sales_mix(region_obj.sales_mix_source)
    mix = _apply_sales_mix_overlay(mix_raw, region_obj.sales_mix_overlay)
    mix["powertrain"] = _classify_powertrain(mix)

    # 2. Resolve effective BEV share.
    eff_bev_share = (
        float(bev_share) if bev_share is not None else _infer_bev_share(mix)
    )
    if not 0.0 <= eff_bev_share <= 1.0:
        raise ValueError(f"bev_share must be in [0, 1]; got {eff_bev_share}")

    # 3. RNG bookkeeping.
    rng = np.random.default_rng(seed + MODULE_OFFSET)
    ev_id = np.arange(n, dtype=np.int32)
    per_vehicle_seed = (
        seed + 1_000_000 + np.arange(n, dtype=np.int64)
    ).astype(np.int64)

    # 4. Powertrain (Bernoulli on the inferred share).
    u_pt = rng.random(size=n)
    powertrain_arr = np.where(u_pt < eff_bev_share, "BEV", "PHEV")
    bev_mask = powertrain_arr == "BEV"
    bev_idx = np.where(bev_mask)[0]
    phev_idx = np.where(~bev_mask)[0]
    n_bev = int(bev_mask.sum())
    n_phev = n - n_bev

    mix_bev = _powertrain_subset(mix, "BEV") if n_bev > 0 else None
    mix_phev = _powertrain_subset(mix, "PHEV") if n_phev > 0 else None

    # 5. Archetype + (make, model, B_kwh, obc_max_kw).
    make_arr = np.empty(n, dtype=object)
    model_arr = np.empty(n, dtype=object)
    archetype_arr = np.empty(n, dtype=object)
    b_kwh = np.empty(n, dtype=np.float32)
    obc_typical = np.empty(n, dtype=np.float32)
    model_year = np.empty(n, dtype=np.int16)

    if n_bev > 0:
        assert mix_bev is not None  # for type-checkers
        bev_picked_idx, bev_rows = _sample_archetype_rows(rng, mix_bev, n_bev)
        make_arr[bev_idx] = bev_rows["make"].to_numpy()
        model_arr[bev_idx] = bev_rows["model"].to_numpy()
        archetype_arr[bev_idx] = bev_rows["archetype"].to_numpy()
        b_kwh[bev_idx] = bev_rows["battery_kwh_nominal"].to_numpy(dtype=np.float32)
        obc_typical[bev_idx] = bev_rows["obc_max_kw"].to_numpy(dtype=np.float32)
        model_year[bev_idx] = _sample_model_year(
            rng, "BEV", n_bev, mix_bev, bev_picked_idx
        )

    if n_phev > 0:
        assert mix_phev is not None
        phev_picked_idx, phev_rows = _sample_archetype_rows(rng, mix_phev, n_phev)
        make_arr[phev_idx] = phev_rows["make"].to_numpy()
        model_arr[phev_idx] = phev_rows["model"].to_numpy()
        archetype_arr[phev_idx] = phev_rows["archetype"].to_numpy()
        b_kwh[phev_idx] = phev_rows["battery_kwh_nominal"].to_numpy(dtype=np.float32)
        obc_typical[phev_idx] = phev_rows["obc_max_kw"].to_numpy(dtype=np.float32)
        model_year[phev_idx] = _sample_model_year(
            rng, "PHEV", n_phev, mix_phev, phev_picked_idx
        )

    # 6. Wh/mi nominal.
    wh_per_mi = np.empty(n, dtype=np.float32)
    for arch_name, val in WH_PER_MI_BY_ARCHETYPE.items():
        mask = (archetype_arr == arch_name) & bev_mask
        wh_per_mi[mask] = val
    wh_per_mi[phev_idx] = WH_PER_MI_PHEV_OVERRIDE

    # 7. OBC_kw (powertrain-conditioned categorical, then model-year
    #    conditional overrides for the Lightning ER step-down and the
    #    19.2 kW SKU list — plan §B.3 option 2).
    obc_kw = np.empty(n, dtype=np.float32)
    if n_bev > 0:
        obc_kw[bev_idx] = _sample_categorical(rng, OBC_MIX_BEV, n_bev)
    if n_phev > 0:
        obc_kw[phev_idx] = _sample_categorical(rng, OBC_MIX_PHEV, n_phev)
    obc_kw = _apply_obc_overrides(obc_kw, make_arr, model_arr, model_year)

    # 8. EVSE base draw (profile-type-dispatched). Note: panel cap is
    # computed AFTER the brand/rule layer in step 8d below so that hard
    # rules which mutate EVSE_home_kw are seen by the panel-cap clamp.
    evse_mix = _evse_mix_for_profile(profile_type)
    panel_cap_prob = _panel_cap_prob_for_profile(profile_type)
    evse_home_kw = _sample_categorical(rng, evse_mix, n).astype(np.float32)

    # 8a. EVSE brand draw (plan §C.2 step 8a). Per-EV OBC-conditioned
    # multinomial draw from EVSE_BRAND_MIX_{RESIDENTIAL,WORKPLACE}.
    brand_rng = np.random.default_rng(
        int(seed)
        + int(_MODULE_OFFSETS["vehicle_archetypes"])
        + int(EVSE_BRAND_SAMPLE_SUBSEED_OFFSET)
    )
    base_brand_mix = (
        EVSE_BRAND_MIX_WORKPLACE if profile_type == "workplace"
        else EVSE_BRAND_MIX_RESIDENTIAL
    )
    # Pass post-override OBC_kw (plan §B.1.3): the multiplier conditions
    # on the EV's true on-board charger, not on the EVSE-side hardware.
    evse_brand_arr = _sample_brand(
        brand_rng, base_brand_mix, obc_kw,
        OBC_BUCKET_BRAND_MULTIPLIERS, n,
    )

    # 8b. Hard EV<->EVSE coupling rules (residential only -- workplace
    # EVSE is site-owned, not vehicle-owned). The +500_000 offset within
    # the brand sub-seed namespace keeps the take-rate-Bernoulli stream
    # disjoint from the brand-sampling stream.
    bundle_realised: dict[str, dict] = {}
    if profile_type == "residential":
        rule_rng = np.random.default_rng(
            int(seed)
            + int(_MODULE_OFFSETS["vehicle_archetypes"])
            + int(EVSE_BRAND_SAMPLE_SUBSEED_OFFSET)
            + 500_000
        )
        evse_home_kw, evse_brand_arr, connector_pending, bundle_realised = (
            _apply_bundle_rules(
                rule_rng,
                make_arr, model_arr, model_year, powertrain_arr,
                evse_home_kw, evse_brand_arr,
                EV_EVSE_BUNDLE_RULES,
            )
        )
    else:
        connector_pending = np.full(n, None, dtype=object)

    # 8c. EVSE connector draw (plan §C.2 step 8c).
    conn_rng = np.random.default_rng(
        int(seed)
        + int(_MODULE_OFFSETS["vehicle_archetypes"])
        + int(EVSE_CONNECTOR_SAMPLE_SUBSEED_OFFSET)
    )
    evse_connector_arr = _sample_connector(
        conn_rng, make_arr, model_year, evse_brand_arr,
        connector_pending, profile_type,
    )

    # 8d. Panel cap (AFTER rules -- semantics: panel cap is a household
    # trait, applied AFTER the EVSE is fully resolved).
    if panel_cap_prob > 0:
        panel_hit = rng.random(size=n) < panel_cap_prob
        panel_kw = np.where(
            panel_hit, np.minimum(evse_home_kw, PANEL_CAP_KW), evse_home_kw
        )
        panel_cap_kw = np.minimum(evse_home_kw, panel_kw).astype(np.float32)
    else:
        # Workplace: panel_cap_kw == EVSE_home_kw (no panel constraint).
        panel_cap_kw = evse_home_kw.astype(np.float32)

    # 9. eta_charge_nominal.
    eta = _draw_eta(rng, n).astype(np.float32)

    # 10. Heat pump flag.
    make_in_hp = np.array([m in HEATPUMP_MAKES for m in make_arr], dtype=bool)
    has_heat_pump = make_in_hp & (model_year >= HEATPUMP_MIN_MODEL_YEAR)

    # RFC-032: optional stratified resampling to a region-specific share.
    if region_obj.heat_pump_share_override is not None:
        resample_seed = (
            int(seed)
            + int(_MODULE_OFFSETS["vehicle_archetypes"])
            + int(HEATPUMP_RESAMPLE_SUBSEED_OFFSET)
        )
        has_heat_pump = _stratified_resample_heat_pump(
            has_heat_pump,
            archetype_arr,
            target_rate=float(region_obj.heat_pump_share_override),
            seed=resample_seed,
        )

    # 11. Garage.
    garage_type = _draw_garage(rng, n, profile_type)

    # 12. Income tier (region scheme-dispatched).
    income_weights = _income_weights(region_obj)
    income_tier = rng.choice(
        list(INCOME_TIERS),
        size=n,
        replace=True,
        p=list(income_weights),
    )

    # 13. Home station.
    home_stations = _resolve_home_stations(region_obj)
    home_station = rng.choice(list(home_stations), size=n, replace=True)

    # 14. soc_min_kwh.
    soc_min_kwh = (SOC_MIN_FRAC * b_kwh).astype(np.float32)

    # 14b. v3.0 (§13.6 RESOLVED) — PHEV charge-sustaining MPG (EPA
    # fueleconomy.gov ``comb08``) per (make, model, model_year). BEV
    # rows carry NaN — the column is powertrain-specific and the M6
    # SoC ledger only reads it on PHEV rows (via the
    # ``is_phev`` branch). For PHEV (make, model, model_year)
    # triples missing from the registry (e.g. out-of-band model
    # years like a 2017 Wrangler 4xe), :func:`cs_mpg` falls back to
    # :data:`PHEV_CS_MPG_FALLBACK` (33.0 — the EPA per-vehicle
    # median across the v3.0 PHEV cohort).
    phev_mpg = np.full(n, np.nan, dtype=np.float32)
    if n_phev > 0:
        phev_makes = make_arr[phev_idx]
        phev_models = model_arr[phev_idx]
        phev_years = model_year[phev_idx]
        phev_mpg[phev_idx] = np.array(
            [
                cs_mpg(str(mk), str(md), int(yr))
                for mk, md, yr in zip(
                    phev_makes, phev_models, phev_years, strict=True,
                )
            ],
            dtype=np.float32,
        )
    _ = PHEV_CS_MPG_FALLBACK  # imported for fleet-meta surface (below)

    # 15. Assemble.
    # Categorical universes that span all 8 regions / 2 profile types so
    # serialised parquet files compare identically across runs / regions.
    profile_categories = list(PROFILE_TYPES)
    region_categories = sorted(REGIONS.keys())
    home_categories = sorted(set(home_stations))

    df = pd.DataFrame(
        {
            "ev_id": ev_id,
            "profile_type": pd.Categorical(
                np.full(n, profile_type, dtype=object),
                categories=profile_categories,
            ),
            "region_name": pd.Categorical(
                np.full(n, region_obj.name, dtype=object),
                categories=region_categories,
            ),
            "powertrain": pd.Categorical(
                powertrain_arr, categories=list(POWERTRAIN_CATEGORIES)
            ),
            "make": pd.Categorical(make_arr.astype(str)),
            "model": pd.Categorical(model_arr.astype(str)),
            "model_year": model_year,
            "archetype": pd.Categorical(
                archetype_arr.astype(str), categories=list(ARCHETYPE_CATEGORIES)
            ),
            "B_kwh": b_kwh,
            "OBC_kw": obc_kw,
            "EVSE_home_kw": evse_home_kw,
            "panel_cap_kw": panel_cap_kw,
            # v2.1 EVSE-enrichment columns. Categorical universes are
            # package-wide stable (plan §A.2) so parquet category codes
            # are reproducible across regions and runs.
            "EVSE_brand": pd.Categorical(
                evse_brand_arr.astype(str),
                categories=list(EVSE_BRAND_CATEGORIES),
            ),
            "EVSE_connector": pd.Categorical(
                evse_connector_arr.astype(str),
                categories=list(EVSE_CONNECTOR_CATEGORIES),
            ),
            "eta_charge_nominal": eta,
            "has_heat_pump": has_heat_pump,
            "Wh_per_mi_nominal": wh_per_mi,
            "garage_type": pd.Categorical(
                garage_type.astype(str), categories=list(GARAGE_CATEGORIES)
            ),
            "income_tier": pd.Categorical(
                income_tier.astype(str), categories=list(INCOME_TIERS)
            ),
            "home_station": pd.Categorical(
                home_station.astype(str), categories=home_categories
            ),
            "seed": per_vehicle_seed,
            "soc_min_kwh": soc_min_kwh,
            # v3.0 (§13.6 RESOLVED) — PHEV CS MPG per (make, model,
            # year). NaN on BEV rows by construction.
            "phev_fuel_economy_mpg": phev_mpg,
        }
    )

    _ = obc_typical  # informational only; effective OBC is the sampled draw

    # v2.1 — stash per-rule realised take-rate counts on df.attrs so
    # write_meta() can surface them in meta.json without re-running the
    # sampler (plan §C.4). Attrs ride along on DataFrame operations and
    # are dropped when the frame is serialised to parquet.
    df.attrs["bundle_rule_realised_take_rates"] = bundle_realised

    _validate_fleet(df, profile_type=profile_type)
    return df


def _validate_fleet(df: pd.DataFrame, profile_type: str) -> None:
    """Run cheap assertions matching the PM-brief acceptance criteria."""
    expected_cols = [c for c, _, _ in SCHEMA]
    assert list(df.columns) == expected_cols, "fleet schema column order drift"
    assert df["ev_id"].is_unique, "ev_id must be unique"
    assert df["ev_id"].min() == 0 and df["ev_id"].max() == len(df) - 1, (
        "ev_id must be contiguous 0..N-1"
    )
    # v3.0 nullability: ``phev_fuel_economy_mpg`` is NaN on BEV rows by
    # design (PHEV-only EPA lookup), so the global no-null assertion now
    # restricts to non-nullable columns. NaN on PHEV rows would still
    # leak a real bug (a PHEV without an MPG falls back to the cohort
    # median inside :func:`cs_mpg`, so it can never be NaN if the row
    # came through ``sample_fleet``) — checked separately below.
    nonnull_cols = [c for c, _, nullable in SCHEMA if not nullable]
    assert df[nonnull_cols].notna().all().all(), (
        "no nulls allowed in non-nullable M2-owned columns"
    )
    phev_mask = df["powertrain"].astype(str) == "PHEV"
    if phev_mask.any():
        assert df.loc[phev_mask, "phev_fuel_economy_mpg"].notna().all(), (
            "phev_fuel_economy_mpg must be finite on every PHEV row"
        )
    bev_mask = df["powertrain"].astype(str) == "BEV"
    if bev_mask.any():
        assert df.loc[bev_mask, "phev_fuel_economy_mpg"].isna().all(), (
            "phev_fuel_economy_mpg must be NaN on every BEV row"
        )

    assert (
        (df["eta_charge_nominal"] >= ETA_LO)
        & (df["eta_charge_nominal"] <= ETA_HI)
    ).all(), "eta_charge_nominal out of [0.85, 0.94]"
    assert (df["panel_cap_kw"] <= df["EVSE_home_kw"] + 1e-6).all(), (
        "panel_cap_kw > EVSE_home_kw"
    )
    assert np.allclose(
        df["soc_min_kwh"].to_numpy(),
        SOC_MIN_FRAC * df["B_kwh"].to_numpy(),
        rtol=0,
        atol=1e-6,
    ), "soc_min_kwh != 0.10 * B_kwh"

    # v2.1 EVSE-enrichment column invariants (plan §C.3).
    assert isinstance(df["EVSE_brand"].dtype, pd.CategoricalDtype), (
        "EVSE_brand must be categorical"
    )
    assert isinstance(df["EVSE_connector"].dtype, pd.CategoricalDtype), (
        "EVSE_connector must be categorical"
    )
    assert tuple(df["EVSE_brand"].cat.categories) == EVSE_BRAND_CATEGORIES, (
        "EVSE_brand category order drift"
    )
    assert tuple(df["EVSE_connector"].cat.categories) == EVSE_CONNECTOR_CATEGORIES, (
        "EVSE_connector category order drift"
    )
    # No empty / whitespace-only strings.
    assert (df["EVSE_brand"].astype(str).str.len() > 0).all(), (
        "EVSE_brand contains empty strings"
    )
    assert (df["EVSE_connector"].astype(str).str.len() > 0).all(), (
        "EVSE_connector contains empty strings"
    )

    if profile_type == "workplace":
        # PM brief §4.5 acceptance #3: workplace lots have no panel cap.
        assert (
            df["panel_cap_kw"].to_numpy() == df["EVSE_home_kw"].to_numpy()
        ).all(), "workplace panel_cap_kw must equal EVSE_home_kw"
        assert (df["garage_type"].astype(str) == "workplace_lot").all(), (
            "workplace garage_type must be 'workplace_lot' for every row"
        )


# ---------------------------------------------------------------------------
# Persistence.
# ---------------------------------------------------------------------------

def _cast_to_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Cast columns to the canonical dtypes from ``SCHEMA``."""
    out = df.copy()
    for col, dtype, _nullable in SCHEMA:
        if dtype == "category":
            if not isinstance(out[col].dtype, pd.CategoricalDtype):
                out[col] = out[col].astype("category")
        else:
            out[col] = out[col].astype(dtype)
    return out


def write_fleet(df: pd.DataFrame, out_path: str | os.PathLike[str]) -> Path:
    """Write fleet DataFrame to parquet, cast to the canonical schema dtypes."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cast = _cast_to_schema(df)
    cast = cast.sort_values("ev_id").reset_index(drop=True)
    cast.to_parquet(out, engine="pyarrow", index=False, compression="snappy")
    return out


def _default_out_dir(repo_root: Path, region_name: str, profile_type: str) -> Path:
    return repo_root / "data" / "pev" / "processed" / region_name / f"{profile_type}_ev_synth"


def write_meta(
    df: pd.DataFrame,
    out_path: str | os.PathLike[str],
    master_seed: int,
    bev_share: float,
    region: Region,
    profile_type: str,
) -> Path:
    """Write ``meta.json`` with the M2 v2.0 fields populated."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    pkg_versions = {
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "python": platform.python_version(),
    }
    try:
        import pyarrow  # local import - only needed for meta
        pkg_versions["pyarrow"] = pyarrow.__version__
    except ImportError:  # pragma: no cover - pyarrow is a hard dep
        pkg_versions["pyarrow"] = "unknown"
    try:
        import scipy
        pkg_versions["scipy"] = scipy.__version__
    except ImportError:  # pragma: no cover
        pkg_versions["scipy"] = "unknown"

    meta = {
        "methodology_version": METHODOLOGY_VERSION,
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "master_seed": int(master_seed),
        "n_vehicles": int(len(df)),
        "tz_storage": "UTC",
        "tz_local_anchor": region.tz,
        "cache_provenance": "v2.0_native",
        "region": {
            "name": region.name,
            "display_name": region.display_name,
            "states": list(region.states),
            "cbsas": list(region.cbsas),
            "tz": region.tz,
            "iso_market": region.iso_market,
            "sales_mix_source": region.sales_mix_source,
            "winter_f_T_multiplier": float(region.winter_f_T_multiplier),
            "winter_t_c": (
                float(region.winter_t_c)
                if region.winter_t_c is not None else None
            ),
            "workplace_donor_borrow_states": list(
                region.workplace_donor_borrow_states
            ),
            "heat_pump_share_override": (
                float(region.heat_pump_share_override)
                if region.heat_pump_share_override is not None else None
            ),
        },
        "profile_type": profile_type,
        "sales_mix_overlay": dict(region.sales_mix_overlay),
        "raw_input_sha256": {
            "sales_mix_csv": _hash_sales_mix(region.sales_mix_source),
            "cvrp_model_year_hist": hashlib.sha256(
                json.dumps(CVRP_MODEL_YEAR_HIST, sort_keys=True).encode("utf-8")
            ).hexdigest(),
            # v2.1 — fingerprint the new EVSE-enrichment priors so any
            # change to the brand/kW priors invalidates the cache.
            "evse_priors": hashlib.sha256(
                json.dumps(
                    {
                        "EVSE_BRAND_MIX_RESIDENTIAL": EVSE_BRAND_MIX_RESIDENTIAL,
                        "EVSE_BRAND_MIX_WORKPLACE":   EVSE_BRAND_MIX_WORKPLACE,
                        "EVSE_MIX_RESIDENTIAL":       EVSE_MIX_RESIDENTIAL,
                        "EVSE_MIX_WORKPLACE":         EVSE_MIX_WORKPLACE,
                        "NEMA_OUTLET_MIX":            NEMA_OUTLET_MIX,
                        "EVSE_CONNECTOR_MIX_WORKPLACE": EVSE_CONNECTOR_MIX_WORKPLACE,
                    },
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest(),
        },
        "package_versions": pkg_versions,
        "git_commit": _resolve_git_commit(),
        "vehicle_archetypes": {
            "bev_share": float(bev_share),
            "phev_share": float(1.0 - bev_share),
            "pickup_share": float(
                (df["archetype"].astype(str) == "pickup").mean()
            ),
            "evse_mix_profile": "workplace" if profile_type == "workplace" else "residential",
            # v2.1 — realised EVSE-enrichment distributions (plan §C.4).
            "evse_brand_realised_share": {
                str(k): float(v) for k, v in
                df["EVSE_brand"].value_counts(normalize=True).items()
            },
            "evse_connector_realised_share": {
                str(k): float(v) for k, v in
                df["EVSE_connector"].value_counts(normalize=True).items()
            },
            "evse_kw_realised_share": {
                f"{float(k):.1f}": float(v) for k, v in
                df["EVSE_home_kw"].value_counts(normalize=True).items()
            },
            "bundle_rule_realised_take_rates": (
                df.attrs.get("bundle_rule_realised_take_rates", {})
            ),
        },
    }

    # RFC-032: persist achieved heat-pump share in consumption_model.
    achieved_hp_share = float(df["has_heat_pump"].astype(bool).mean()) if len(df) else 0.0
    cm = meta.setdefault("consumption_model", {})
    if isinstance(cm, dict):
        cm["heat_pump_share"] = achieved_hp_share
        cm["heat_pump_share_override"] = (
            float(region.heat_pump_share_override)
            if region.heat_pump_share_override is not None
            else None
        )

    with out.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
    return out


def _hash_sales_mix(source: str) -> str:
    """Return SHA-256 of the loaded sales-mix CSV (after pd.read_csv)."""
    try:
        df = load_sales_mix(source)
    except Exception:  # pragma: no cover - defensive
        return "unavailable"
    return hashlib.sha256(df.to_csv(index=False).encode("utf-8")).hexdigest()


def _resolve_git_commit() -> str:
    """Return the current git commit SHA, or 'unknown' if unavailable."""
    try:
        import subprocess

        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:  # pragma: no cover - best-effort
        pass
    return "unknown"


# ---------------------------------------------------------------------------
# CLI entry-point.
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    """CLI: sample fleet for (region, profile_type) and write parquet + meta."""
    parser = argparse.ArgumentParser(
        prog="pev_synth.vehicle_archetypes",
        description=(
            "M2 v2.0: sample N synthetic EV attribute vectors for "
            "(region, profile_type) and write fleet.parquet + meta.json."
        ),
    )
    parser.add_argument("--n", type=int, default=N_VEHICLES_DEFAULT)
    parser.add_argument("--seed", type=int, default=MASTER_SEED_DEFAULT)
    parser.add_argument(
        "--region", type=str, default="bay_area",
        help="Region short name; e.g. bay_area, dallas_fort_worth, boston.",
    )
    parser.add_argument(
        "--profile-type", type=str, default="residential",
        choices=list(PROFILE_TYPES),
    )
    parser.add_argument(
        "--bev-share", type=float, default=None,
        help="Override BEV share (default: inferred from region's sales mix).",
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help=(
            "Output parquet path; defaults to "
            "data/pev/processed/<region>/<profile_type>_ev_synth/fleet.parquet."
        ),
    )
    parser.add_argument(
        "--meta-out", type=str, default=None,
        help="meta.json output path (defaults next to --out).",
    )
    parser.add_argument(
        "--repo-root", type=str, default=None,
        help="Repo root (defaults to two parents up from this file).",
    )
    args = parser.parse_args(argv)

    from ._paths import repo_root as _repo_root_fn  # noqa: PLC0415

    repo_root = (
        Path(args.repo_root).resolve() if args.repo_root else _repo_root_fn()
    )
    region = Region.from_name(args.region)
    out_dir = _default_out_dir(repo_root, region.name, args.profile_type)
    out_path = Path(args.out) if args.out else (out_dir / "fleet.parquet")
    if not out_path.is_absolute():
        out_path = repo_root / out_path
    meta_path = Path(args.meta_out) if args.meta_out else (out_dir / "meta.json")
    if not meta_path.is_absolute():
        meta_path = repo_root / meta_path

    df = sample_fleet(
        n=args.n, seed=args.seed,
        region=region, profile_type=args.profile_type,
        bev_share=args.bev_share,
    )
    eff_bev = (
        float(args.bev_share)
        if args.bev_share is not None
        else float((df["powertrain"].astype(str) == "BEV").mean())
    )
    written = write_fleet(df, out_path)
    meta_written = write_meta(
        df, meta_path,
        master_seed=args.seed, bev_share=eff_bev,
        region=region, profile_type=args.profile_type,
    )

    print(
        f"M2 vehicle_archetypes v2.0: wrote {len(df)} rows -> {written}",
        file=sys.stderr,
    )
    print(f"M2 vehicle_archetypes v2.0: wrote meta -> {meta_written}", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
