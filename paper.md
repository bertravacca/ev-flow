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
through a deterministic nine-stage pipeline to a time-stamped charging profile.
It ships residential and workplace profile types, descriptive charging-equipment
(EVSE brand and connector) enrichment, and both 15-minute and hourly time grids.
Every output is stored in UTC, is timezone-aware, and is bit-for-bit reproducible
from a single master random seed. The library exposes a high-level
`generate_profiles` / `Fleet` / `Profile` interface for downstream studies,
together with `ev-flow bootstrap` and `ev-flow doctor` command-line tools that
build and diagnose the underlying data on a fresh machine.

# Statement of need

Quantifying the electricity demand and temporal flexibility of EV fleets is
central to power-systems planning, transportation electrification, and energy
economics. Because real per-vehicle charging records are rarely shareable,
researchers turn to *bottom-up simulation*: deriving plausible charging load
from travel behavior plus techno-economic assumptions. The problem `ev-flow`
solves is the absence of an open, U.S.-grounded generator that preserves the
regional, seasonal, and equipment heterogeneity driving aggregate demand while
remaining fully deterministic and auditable. Its target users are energy-systems
modelers, distribution-planning engineers, rate-design analysts, and researchers
who need reproducible charging populations they can regenerate exactly rather
than a fixed, opaque dataset.

# State of the field

Several open tools generate or simulate EV charging, but they occupy adjacent
niches. `emobpy` [@gaetemorales2021emobpy], `venco.py` [@miorelli2025vencopy],
and RAMP-mobility [@mangipinto2022ramp] build EV demand and flexibility profiles
from *European* national travel surveys; their behavioral calibration and
charging-availability assumptions do not transfer cleanly to U.S. fleets, whose
vehicle mix, trip structure, and home/work charging access differ materially.
`datafev` [@gumrukcu2023datafev] and ACN-Sim [@lee2021acnsim] are
charging-infrastructure and control simulators that *consume* charging sessions
to test management algorithms, rather than *generating* survey-grounded
behavioral populations. Rather than re-skinning a European generator or
hard-coding U.S. assumptions onto an infrastructure simulator, `ev-flow`
contributes a distinct artifact: it ties every behavioral choice to a U.S.
public source (NHTS travel diaries; the SPEECh charging model
[@powell2022speech]), differentiates eight regions through an explicit sales-mix
model, and emits profiles usable directly as boundary inputs to the optimization,
market, and control models the tools above target. No existing open tool fills
that U.S.-focused, NHTS-grounded, fully-reproducible niche, which is why building
`ev-flow` was warranted rather than contributing to an existing project.

# Software design

`ev-flow` is organized as a nine-stage pipeline (internally M1–M9) behind a small
public API, a structure chosen to make every transformation independently
testable and cacheable. The central design trade-off is **determinism over raw
throughput**: the entire pipeline is seeded from a single master seed so that a
given `(region, profile_type, seed)` reproduces every profile byte-for-byte, and
a continuous-integration job re-runs the pipeline to confirm identical cached
output. This makes downstream scientific results exactly reproducible at the cost
of some vectorization opportunities. Key modeling choices each reflect a
grounding decision rather than convenience: NHTS single-day person records are
stitched into donor-matched 365-day travel calendars with a temperature-dependent
winter energy uplift [@yuksel2015regional]; behavioral plug-in start times are
sampled from the published SPEECh K=16 Gaussian-mixture parameterization
[@powell2022speech; @powell2022speechdata]; a three-layer Bernoulli plug-in model
[@munkhammar2015probability] decides whether each opportunity is taken; and a
continuous-time state-of-charge ledger with an explicit PHEV gasoline
range-extension term propagates energy before plug status is rasterized to the
15-minute and hourly grids. Regional heterogeneity is isolated in the sales-mix
model, so battery-capacity and powertrain distributions vary across regions
without touching the pipeline. A deliberate design principle is **auditable
validation**: rather than hide divergences, a validation runner compares
generated distributions against literature bounds and classifies each as PASS, an
*explained* failure (a documented, sourced limitation), or an explained skip. The
runner defines 32 distributional checks, 21 of which apply to a residential run;
for the reference `bay_area` residential profile it reports 11 PASS, 0
*unexplained* failures, 6 explained failures, and 4 explained skips — every
non-PASS carrying an explicit rationale. The package is pure Python (requires
Python ≥ 3.10), depends only on the scientific-Python stack, and ships a typed
(`py.typed`) public API covered by an extensive `pytest` suite (500+ tests) on a
multi-OS, multi-version CI matrix.

# Research impact statement

`ev-flow` was extracted from an aggregated-EV grid-study research codebase and
continues to supply that study's synthetic charging populations, so its outputs
are already in active research use rather than being a prospective tool. A
companion methodology preprint describing the same software has been posted to
arXiv (2026), and each tagged release is archived to Zenodo with a citable DOI,
giving downstream work a stable, versioned reference. The package is built for
adoption by other groups: `ev-flow bootstrap` reconstructs the full data tree on
a clean machine from public sources, `ev-flow doctor` diagnoses a broken
install, three worked tutorials and a browser-based quickstart notebook lower the
entry cost, and the deterministic, bit-reproducible output plus the published
validation report are concrete community-readiness signals a reviewer or user can
verify directly. Because its profiles are designed as boundary inputs, `ev-flow`
is positioned to feed energy-system optimization, electricity-market, and
charging-control models — the same downstream consumers served by the related
tools above — with U.S.-regional behavior they currently lack.

# AI usage disclosure

Generative-AI assistants — specifically Anthropic's Claude (used via the Claude
Code command-line tool) and the Cursor AI code editor — assisted with portions of
the software development, documentation, and the drafting of this paper. All
AI-assisted output was
reviewed and verified by the author and is not accepted on trust: the code is
covered by an automated `pytest` suite and continuous integration, every
quantitative claim in this paper was checked against the cited sources and
re-derived from the package's own validation runner, and the manuscript was put
through two independent adversarial review passes before submission.

# Acknowledgements

This work received no specific grant or external financial support. It builds
directly on public data and methods, in particular the NHTS program of the U.S.
Federal Highway Administration and the SPEECh charging model of Powell, Cezar,
and Rajagopal. We thank the maintainers of the open scientific-Python ecosystem
on which `ev-flow` depends.

# References
