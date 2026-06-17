---
title: 'ev-flow: Reproducible, NHTS-grounded synthetic plug-in electric vehicle charging behavior for U.S. regions'
tags:
  - Python
  - electric vehicles
  - synthetic data
  - charging behavior
  - energy systems
  - power systems
  - demand modeling
  - mobility
authors:
  - name: Bertrand Travacca
    orcid: 0000-0001-7167-7455
    corresponding: true
    affiliation: 1
affiliations:
  - name: Independent Researcher, United States
    index: 1
date: 17 June 2026
bibliography: paper.bib
---

# Summary

`ev-flow` (Python import name `pev_synth`) is an open-source, MIT-licensed
Python package that generates synthetic plug-in electric-vehicle (PEV) charging
behavior for eight U.S. regions. Studies of grid integration, electricity-rate
design, distribution-system hosting capacity, and managed charging need large
populations of behaviorally realistic *individual* charging profiles, yet
measured charging telemetry is scarce, proprietary, and privacy-restricted.
`ev-flow` produces such populations entirely from public inputs: it grounds
every synthetic vehicle in 2017 National Household Travel Survey (NHTS)
microdata [@nhts2017] and a per-region vehicle sales-mix model, then carries it
through a deterministic nine-stage pipeline (M1–M9) to a time-stamped charging
profile. It ships residential and workplace profile types, descriptive
charging-equipment (EVSE brand and connector) enrichment, and both 15-minute and
hourly time grids. Every output is stored in UTC, is timezone-aware, and is
bit-for-bit reproducible from a single master random seed. The library exposes a
high-level `generate_profiles` / `Fleet` / `Profile` API for downstream studies,
together with `ev-flow bootstrap` and `ev-flow doctor` command-line tools that
build and diagnose the underlying data on a fresh machine.

# Statement of need

Quantifying the electricity demand and temporal flexibility of EV fleets is
central to power-systems planning, transportation electrification, and energy
economics. Because real per-vehicle charging records are rarely shareable,
researchers turn to *bottom-up simulation*: deriving plausible charging load
from travel behavior plus techno-economic assumptions. Existing open generators,
however, are calibrated to non-U.S. mobility surveys or collapse the regional,
seasonal, and equipment heterogeneity that drives aggregate demand. `ev-flow`
targets that gap with a U.S.-focused, NHTS-grounded generator whose every
modeling choice is traceable to a public source, and whose outputs are
deterministic and auditable so that downstream results can be reproduced exactly.

## State of the field

Several open tools generate or simulate EV charging, but they occupy adjacent
niches. `emobpy` [@gaetemorales2021emobpy], `venco.py` [@miorelli2025vencopy],
and RAMP-mobility [@mangipinto2022ramp] build EV demand and flexibility profiles
from *European* national travel surveys; their behavioral calibration and
charging-availability assumptions do not transfer cleanly to U.S. fleets.
`datafev` [@gumrukcu2023datafev] and ACN-Sim [@lee2021acnsim] are
charging-infrastructure and control simulators that *consume* charging sessions
to test management algorithms, rather than *generating* survey-grounded
behavioral populations. `ev-flow` is complementary: it produces U.S.-regional,
NHTS-grounded behavioral profiles that can serve directly as boundary inputs to
those optimization, market, and control models.

# Implementation

The nine pipeline stages stitch NHTS person-days into donor-matched 365-day
travel calendars with a temperature-dependent winter energy uplift
[@yuksel2015regional]; sample behavioral plug-in start times from the published
SPEECh K=16 Gaussian-mixture parameterization [@powell2022speech;
@powell2022speechdata]; evaluate a three-layer Bernoulli plug-in decision model
[@munkhammar2015probability]; propagate a continuous-time state-of-charge ledger
with an explicit PHEV gasoline range-extension term; and rasterize plug status to
the output grids. Regional differentiation comes from the sales-mix model, so
battery-capacity and powertrain distributions differ across the bay_area,
la_basin, new_york_metro, boston, chicago, dallas_fort_worth, seattle, and
us_national regions. The package is pure Python (requires Python ≥ 3.10), depends
only on the scientific-Python stack, and ships a typed (`py.typed`) public API.

# Reproducibility and validation

Determinism is a first-class contract: a single seed reproduces every profile
byte-for-byte, and a continuous-integration job re-runs the pipeline to confirm
identical cached output. A built-in validation runner compares the generated
distributions against bounds drawn from the literature and classifies each
divergence as PASS, an *explained* failure (a documented, sourced modeling
limitation), or an explained skip. The runner defines 32 distributional checks,
of which 21 apply to a residential run. For the reference `bay_area` residential
profile it reports 11 PASS, 0 *unexplained* failures, 6 explained failures, and
4 explained skips — every non-PASS carrying an explicit literature rationale
rather than being hidden. The
package is covered by an extensive `pytest` suite (500+ tests) run on a
multi-OS, multi-version CI matrix with strict type-checking on the public API,
and each release is archived to Zenodo for a citable DOI.

# Acknowledgements

This work builds directly on public data and methods, in particular the NHTS
program of the U.S. Federal Highway Administration and the SPEECh charging model
of Powell, Cezar, and Rajagopal. We thank the maintainers of the open
scientific-Python ecosystem on which `ev-flow` depends.

# References
