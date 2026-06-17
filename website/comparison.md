# Feature comparison

This page places ev-flow alongside the open-source EV charging generators and
simulators it relates to most directly. It is a companion to
[When to use ev-flow](when-to-use.md): use that page to choose a tool, and this
one to check capabilities cell by cell.

!!! note "`ev-flow` on PyPI, `pev_synth` in Python"
    You `pip install ev-flow` and `import pev_synth` — one project, two names.

!!! info "How to read the table"
    **Yes** = first-class, documented, callable without re-implementing it.
    **Partial** = supported with user wiring, or only along part of the
    dimension. **No** = not supported. **—** = not a stated capability of that
    tool in its published descriptor; treat as "not claimed here, see that
    tool's own docs" rather than as a negative we assert. Cells are grounded in
    the ev-flow paper's related-work survey and the differentiation notes; we do
    not invent capabilities of peer tools.

## Tools compared

| Tool | Primary role | Reference |
|---|---|---|
| **ev-flow** (`pev_synth`) | Generator — NHTS-grounded, per-vehicle | This package; Travacca 2026 |
| **emobpy** | Generator — mobility → charging demand | Gaete-Morales et al. 2021 |
| **VencoPy** | Flexibility-envelope tool from travel survey | Wulff et al. 2021 |
| **RAMP-mobility** | Bottom-up stochastic EV load profiles | Mangipinto et al. 2022 |
| **datafev** | Scenario generator + scheduler benchmark | Gümrükcü et al. 2023 |
| **ACN-Sim** | Per-EV simulator on real workplace data | Lee et al. 2021 |

SimBEV, SpiceEV, EV-SDG, and SPEECh are discussed in the notes below; they are
named in the paper's related work but omitted from the table to keep it
legible (SpiceEV and ACN-Sim are simulators rather than generators; EV-SDG is
session-only; SPEECh is an aggregate-demand model whose K=16 parameters ev-flow
itself consumes).

## Capability matrix

| Capability | ev-flow | emobpy | VencoPy | RAMP-mobility | datafev | ACN-Sim |
|---|---|---|---|---|---|---|
| Regional grounding | 8 U.S. regions (NHTS) | European (German) | European (MiD) | European countries | Parametric | Caltech/JPL facilities |
| Survey-microdata basis | Yes (NHTS 2017) | Partial (empirical stats) | Yes (MiD survey) | Partial (stochastic, survey-informed) | No (parametric specs) | No (real ACN telemetry) |
| Output: per-vehicle profiles | Yes | Yes | No (flexibility envelopes) | No (aggregate load) | Partial (sessions/scenarios) | Yes |
| Output: aggregate / flexibility | Yes (load curve) | Partial | Yes (envelopes) | Yes (load) | Partial | Partial |
| Residential charging | Yes | Yes | Yes | Yes | Partial | No |
| Workplace charging | Yes (cohort caveat) | Yes | Partial | — | Partial | Yes |
| Public / DC-fast charging | No | Yes | — | — | Partial | No |
| Continuous-time SoC trajectory | Yes | Yes | Partial | — | No | Yes |
| Charging-session records | Yes | Partial | Partial | — | Yes | Yes |
| 15-min / hourly raster | Yes | Yes (configurable) | Partial | Yes | Partial | Yes |
| Timezone-aware (UTC-canonical) | Yes | Partial | — | — | — | Partial |
| Charging-control / strategy modeling | No (by design) | No | No | No | Yes (scheduler lib) | Yes (algorithms) |
| Vehicle-to-grid / bidirectional | No | No | No | No | — | Yes |
| Validation runner vs published bounds | Yes (classified) | No | Partial (paper-only) | — | No | — |
| Reproducible from a single seed | Yes | Partial | — | Partial | — | — |
| Distributed on PyPI | Yes (`ev-flow`) | Yes | Yes | Yes | Yes | Yes (`acnportal`) |
| License | MIT | Open (see project) | Open (see project) | Open (see project) | Open (see project) | Open (see project) |

!!! warning "Honest caveat"
    This matrix reflects the ev-flow author's reading of each peer tool's
    **published descriptor and documentation**, not exhaustive hands-on testing
    of every cell. Peer projects evolve; a **—** means the capability is not a
    claimed feature in the sources we consulted, not that it is impossible.
    Where you depend on a specific capability of another tool, confirm it
    against that tool's current docs. ev-flow is **not** universally superior:
    on several axes a peer tool is the more mature choice (see below).

## Where peers lead ev-flow

Being candid about non-overlap, the open ecosystem is ahead of ev-flow on
several axes:

- **Non-U.S. coverage.** emobpy, VencoPy, RAMP-mobility, and SimBEV are
  purpose-built for European mobility surveys; ev-flow is U.S.-only and
  NHTS-2017-bound.
- **Charging-strategy libraries.** datafev and SpiceEV ship non-trivial
  scheduling algorithms on top of the data; ev-flow ships none and delegates
  control entirely.
- **Charging physics & V2G.** ACN-Sim models the charger/battery interaction at
  higher fidelity (taper, three-phase) and supports bidirectional flow; ev-flow
  uses a deliberately simple, one-way, constant-current SoC ledger.
- **Real-telemetry replay.** ACN-Sim is built around a real workplace dataset;
  ev-flow is synthetic by design.

## What ev-flow uniquely combines

No single peer tool offers all of the following in one open, MIT-licensed,
`pip install`-able package:

1. **NHTS-microdata donor-stitched** travel weeks (not parametric sampling over
   aggregate statistics);
2. **per-vehicle `Profile` objects** unifying trips, plug status, sessions, and
   SoC, indexable by `ev_id`, with a `Fleet` aggregation API;
3. **eight pre-baked U.S. regions** spanning CAISO / NYISO / ISO-NE / PJM /
   ERCOT / WECC, each with its own CBSAs, time zone, sales-mix source, and
   winter uplift;
4. **residential and workplace** profile types;
5. **UTC-canonical storage with tz-aware query**, DST-verified; and
6. an **honestly-classified validation runner** — the reference `bay_area`
   residential profile rolls up to 11 PASS / 0 FAIL / 6 EXPLAINED_FAIL /
   4 EXPLAINED_SKIP / 0 ERROR, with every non-PASS row tied to a sourced reason.

The contribution is the *union*, not dominance on any one column. See
[When to use ev-flow](when-to-use.md) for the use-case-by-use-case guidance.
