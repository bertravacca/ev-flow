# When to use ev-flow

ev-flow fills a specific niche: **open, U.S.-focused, NHTS-grounded, per-vehicle
generation of residential and workplace plug-in EV charging profiles.** It is
deliberately *not* a universal tool. This page helps you decide whether ev-flow
is the right fit for your problem — and points you at a better-suited tool when
it is not. For a side-by-side capability table, see the
[feature comparison](comparison.md).

!!! note "`ev-flow` on PyPI, `pev_synth` in Python"
    You `pip install ev-flow` and then `import pev_synth` — the same project
    under two names, mirroring the `scikit-learn` / `sklearn` convention.

## The short version

ev-flow is a strong fit when **all** of the following hold:

- your study region is in the **United States**;
- you need **per-vehicle** profiles (trips, plug status, charging sessions, and
  a state-of-charge trajectory), not just an aggregate load shape or a
  flexibility envelope;
- you want **synthetic** data grounded in a national travel survey, because real
  telemetry at population scale is unavailable to you;
- **reproducibility** matters — you want the same fleet, bit-for-bit, from a
  seed; and
- you are modeling **uncontrolled** residential and/or workplace charging, not
  vehicle-to-grid or a managed-charging control strategy.

If you find yourself answering "no" to several of these, one of the peer tools
below is probably the better starting point.

## Decision tree

```text
Start: I need EV charging data for a study.
│
├─ Do I have real telemetry at population scale I can use directly?
│   └─ YES → use your data. ev-flow is for when you do NOT.
│            (For replaying a real workplace dataset, see ACN-Sim.)
│
├─ Is my study region inside the United States?
│   ├─ NO  → ev-flow does not model non-US geographies. Use emobpy,
│   │        VencoPy, RAMP-mobility, or SimBEV (Europe-grounded).
│   └─ YES → continue.
│
├─ Do I need per-vehicle profiles, or an aggregate / flexibility result?
│   ├─ Aggregate national/regional demand → EVI-Pro / TEMPO style models,
│   │        or SPEECh for CA-telemetry-fit aggregate load by segment.
│   ├─ Flexibility envelopes (SoC bounds, max charge/discharge) → VencoPy.
│   └─ Per-vehicle trips + sessions + SoC → continue.
│
├─ Do I need a charging-control / scheduling strategy modeled?
│   ├─ YES, and I want a benchmark workload → datafev (scenario gen +
│   │        scheduler library), or feed an ev-flow fleet into SpiceEV.
│   ├─ YES, vehicle-to-grid / bidirectional → ev-flow does NOT model V2G.
│   │        Use ACN-Sim or SpiceEV.
│   └─ NO, I want uncontrolled charging behavior → continue.
│
└─ Residential and/or workplace, US, per-vehicle, reproducible, uncontrolled?
    └─ ev-flow is a good fit.
        (Workplace: read the EVWatts cohort caveat below first.)
```

## By use case

| Use case | ev-flow fit | Notes |
|---|---|---|
| **Grid integration / hosting capacity** | Strong | Per-vehicle, region-resolved populations with aggregate load curves; UTC-canonical, tz-aware. This is the design target. |
| **Load forecasting** | Strong | Behaviorally diverse fleets across eight US regions with seasonal (winter uplift) energy. Bottom-up, not a single representative shape. |
| **Managed- / smart-charging research** | Partial | ev-flow generates the *uncontrolled* population such algorithms consume; it does **not** ship a scheduler. Feed its fleets into your optimizer, datafev, or SpiceEV. |
| **Vehicle-to-grid (V2G) studies** | Not a fit | No bidirectional power flow is modeled. Use ACN-Sim or SpiceEV. |
| **Teaching / reproducible coursework** | Strong | `pip install`-able, seed-reproducible, MIT-licensed, with a candid validator that makes limitations a teaching point. |
| **Workplace charging analysis** | Use with care | A shipped profile type, but fit from a small (105-vehicle) EVWatts cohort that plugs in late — see the caveat below. Treat output as exploratory. |
| **Public / DC-fast charging** | Not a fit | ev-flow models home and workplace dwell-time charging only. |

## By constraint

### U.S. vs non-U.S.

ev-flow is grounded entirely in the **2017 NHTS** (with an opt-in NHTS NextGen
2022 national supplementary stratum) and U.S. regional sales-mix sources. It
ships eight U.S. regions and has no non-US geography. For European work, the
survey-grounded generators **emobpy** (German/EU), **VencoPy** (German MiD),
**RAMP-mobility**, and **SimBEV** are purpose-built.

### Residential vs workplace

Both ship end-to-end. **Residential** is the mature, reference-validated path.
**Workplace** is real but carries a documented data caveat:

!!! warning "Workplace EVWatts cohort caveat (W1–W4)"
    The workplace clusters are fit from a **105-vehicle public EVWatts cohort**
    whose plug-in median (~12:00 local time) is roughly **three hours later**
    than the literature-canonical workplace median (~09:00 LT) seen in ACN-Data.
    The validator flags **W1–W4 as `EXPLAINED_FAIL`** — a documented, sourced
    divergence, not a bug — and constructing any `workplace` fleet emits a
    `RuntimeWarning` at `Fleet.__init__`. Treat workplace output as exploratory.

### V2G needed?

No. Energy flows one way only in this release; neither V2G nor vehicle-to-home
is modeled, and the networked/non-networked and hardwired/plug-in EVSE
distinctions are deferred. If bidirectional flow is central to your study,
ev-flow is the wrong tool — use **ACN-Sim** or **SpiceEV**.

### Real telemetry vs synthetic

ev-flow produces **synthetic** profiles for when real, population-scale
telemetry is unavailable or privacy-restricted. If you already have suitable
real data, use it. If you want to *replay or simulate* a real workplace dataset,
**ACN-Sim** (Caltech/JPL ACN data) is built for that.

### Charging control vs behavior generation

ev-flow generates the *behavioral population* — when vehicles arrive, dwell, and
plug in under uncontrolled charging. It deliberately delegates control strategy:
it ships **no** smart-charging or scheduling algorithm. If you need a scheduler
benchmark, use **datafev**; to simulate a strategy on a fixed schedule, feed an
ev-flow fleet into **SpiceEV**.

## When another tool is the better fit

- **You are in Europe** → emobpy, VencoPy, RAMP-mobility, or SimBEV.
- **You need aggregate national/regional demand**, not per-vehicle profiles →
  EVI-Pro / TEMPO-style models; SPEECh for CA-telemetry-fit aggregate load.
- **You need a flexibility envelope** (SoC bounds, max charge/discharge) →
  VencoPy.
- **You are benchmarking a charging-management algorithm** → datafev.
- **You need vehicle-to-grid or high-fidelity charging physics** (taper,
  three-phase) → ACN-Sim or SpiceEV.
- **You want to replay a real workplace charging dataset** → ACN-Sim.

ev-flow's distinctive contribution is the *combination* — NHTS-grounded donor
matching, eight U.S. regions, residential **and** workplace, descriptive EVSE
detail, and a reproducibility-first, honestly-validated design — delivered in one
open, MIT-licensed package. Where it overlaps another tool on a single axis, that
tool is often more mature on that axis; ev-flow's value is the union, not any one
column. The [feature comparison](comparison.md) lays this out cell by cell.
