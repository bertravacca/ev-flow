# API Reference

This page is generated directly from the docstrings in the `pev_synth`
source via [mkdocstrings](https://mkdocstrings.github.io/). It is the
authoritative reference for the public surface — the names exported in
`pev_synth.__all__`.

!!! note "`ev-flow` on PyPI, `pev_synth` in Python"
    Install with `pip install ev-flow`; import with `import pev_synth`. Every
    symbol below is reachable as `pev_synth.<name>` (for example
    `pev_synth.generate_profiles` or `pev_synth.Fleet`).

## Package overview

::: pev_synth
    options:
      members: false
      show_root_heading: false
      show_root_toc_entry: false

## Generating fleets

The primary entry point is `generate_profiles`, which draws a reproducible
subset from the local cache and returns a [`Fleet`](#fleet).

::: pev_synth.generate_profiles

::: pev_synth.regenerate_fleet

## Fleet

A `Fleet` is a lazy collection of [`Profile`](#profile) objects backed by
parquet files. Only the static `fleet.parquet` is read at construction time;
larger artifacts are read on demand. `Fleet` is iterable, indexable
(`fleet[i]` / `fleet['ev_0042']`), and filterable, and exposes wide,
per-EV-column versions of the time-window queries plus a fleet-level
`aggregate_load`.

::: pev_synth.Fleet

## Profile

A `Profile` is a single synthetic EV. Its static attributes (battery,
charge power, archetype, EVSE details) come from one row of the fleet table;
its time-window methods (`plug_status`, `charging_sessions`,
`soc_trajectory`, `trips`) read the dynamic artifacts lazily.

::: pev_synth.Profile

## Regions

ev-flow models eight US regions. Each `Region` carries the climate, market and
timezone context that drives the pipeline; the region's `tz` is the default
local zone applied to timezone-naive `t_start` / `t_stop` inputs on every
time-window method.

::: pev_synth.Region

::: pev_synth.list_regions

::: pev_synth.list_profile_types

### The region registry

`pev_synth.REGIONS` is the canonical mapping of region short-name to `Region`
instance. The eight registered regions are `bay_area`, `boston`, `chicago`,
`dallas_fort_worth`, `la_basin`, `new_york_metro`, `seattle` and
`us_national`. Look one up by name with
[`Region.from_name`][pev_synth.Region.from_name] or by indexing `REGIONS`
directly:

```python
import pev_synth as ps

ps.REGIONS['bay_area'].tz          # 'America/Los_Angeles'
ps.Region.from_name('seattle').display_name   # 'Seattle + Portland'
```

## Type aliases

::: pev_synth.ProfileType
    options:
      show_root_heading: true
      show_source: false
