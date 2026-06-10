# Save and reload a fleet

Persist a fleet — or a filtered subset of one — to a self-contained bundle on
disk, then reload it later without re-querying the full cache.

!!! note "`ev-flow` on PyPI, `pev_synth` in Python"
    You **`pip install ev-flow`** and then **`import pev_synth`** — same
    project, two names (the `scikit-learn` / `sklearn` convention). The
    command-line tool is `ev-flow`; the importable Python package is
    `pev_synth`.

!!! warning "This tutorial needs a built cache"
    The wheel ships only the Python code, **not** the cached fleet data. Build
    the local cache once with `ev-flow bootstrap` before running anything
    below — see the Quickstart's
    [Bootstrap the data cache](../quickstart.md#2-bootstrap-the-data-cache)
    step. `Fleet.save(...)` writes a *new* bundle from data the cache already
    holds, so the cache must exist first.

## 1. Generate a fleet

Start from a residential Bay Area fleet, exactly as in the other tutorials:

```python
import pev_synth as ps

fleet = ps.generate_profiles('residential', n=500, region='bay_area', seed=42)
fleet
# Fleet(profile_type='residential', region='bay_area', n=500, data_root=...)
```

## 2. (Optional) take a subset with `filter`

You rarely need to persist a whole fleet. [`Fleet.filter`][pev_synth.Fleet.filter]
returns a **new** `Fleet` containing only the EVs matching its predicates
(predicates are AND-combined), and the result saves and reloads just like a full
fleet. Here we keep the battery-electric vehicles with a reasonably fast home
connection:

```python
bevs = fleet.filter(powertrain='BEV', max_charge_kw_gte=7.0)
len(bevs), len(fleet)
# (e.g. 318, 500) -> the subset is smaller

bevs.profile_ids()[:5]   # the ev_id ints carried into the subset
```

Supported `filter` keys include `archetype`, `powertrain`, `cluster_z`,
`evse_brand`, `evse_connector`, and numeric thresholds such as
`battery_kwh_gte` / `battery_kwh_lt` and `max_charge_kw_gte` /
`max_charge_kw_lt`. An unsupported key raises `KeyError` listing the valid ones.

## 3. Save to disk

[`Fleet.save`][pev_synth.Fleet.save] writes a self-contained bundle under a
`<profile_type>_fleet/` subdirectory of the path you give it, and returns the
`Path` to that inner bundle directory. The bundle is a subset of the cache's
parquet artifacts (the static `fleet.parquet`, the plug-status matrices, the
sessions and mobility tables, plug-in events when present) plus a `meta.json`
recording the profile type, region, timezone, and the exact `ev_ids`.

```python
from pathlib import Path

out = bevs.save('exports/bay_area_run')
out
# PosixPath('exports/bay_area_run/residential_fleet')

# What got written:
sorted(p.name for p in Path(out).iterdir())
# ['fleet.parquet', 'meta.json', 'mobility.parquet',
#  'plug_in_events.parquet', 'plug_status_15min.parquet',
#  'plug_status_hourly.parquet', 'sessions.parquet']
```

!!! note "One bundle per profile type per directory"
    The bundle directory is named after the fleet's `profile_type`
    (`residential_fleet` or `workplace_fleet`). Saving two fleets of the same
    profile type to the *same* path overwrites the earlier bundle — give each
    its own path (or save a `residential` and a `workplace` fleet side by side
    in one directory, since their bundle names differ).

## 4. Reload it

[`Fleet.load`][pev_synth.Fleet.load] is the inverse of `save`. It accepts either
the inner bundle directory (the `Path` that `save` returned) **or** its parent
directory (a convenience — it finds the single `*_fleet` bundle inside):

```python
# Reload from the inner bundle directory returned by save:
reloaded = ps.Fleet.load(out)

# ...or from the parent path you passed to save (equivalent here):
reloaded = ps.Fleet.load('exports/bay_area_run')

reloaded
# Fleet(profile_type='residential', region='bay_area', n=318, data_root=...)
```

The reloaded fleet is backed by the saved bundle, not by the original cache, so
it is fully portable: copy the `residential_fleet/` directory to another machine
(no `PEV_SYNTH_DATA_ROOT`, no re-bootstrap) and `Fleet.load` it there.

## 5. Confirm the round-trip

The reloaded fleet exposes the same API as the original and carries the same
EVs. Check that the membership and a downstream query match:

```python
# Same EVs, same order:
reloaded.profile_ids() == bevs.profile_ids()
# True

# A downstream query returns identical results from the saved bundle:
window = ('2001-06-04', '2001-06-11')
orig_kw = bevs.aggregate_load(*window, freq='1h')
new_kw = reloaded.aggregate_load(*window, freq='1h')
new_kw.equals(orig_kw)
# True

# The bundle's metadata is available via the .meta property:
reloaded.meta['profile_type'], reloaded.meta['region'], reloaded.meta['n']
# ('residential', 'bay_area', 318)
```

## What you should see

- `save` creates `exports/bay_area_run/residential_fleet/` containing the
  parquet artifacts and a `meta.json`, and returns that inner directory's path.
- `Fleet.load` reconstructs a `Fleet` whose `profile_type`, `region`, and member
  `ev_ids` match what you saved — `len(reloaded)` equals the subset size (318 in
  the example above, not the original 500).
- Re-running a query such as `aggregate_load` against the reloaded fleet returns
  results identical to the original, because the bundle contains the same
  underlying session and plug-status rows for those EVs.

!!! tip "Why save a subset?"
    A saved bundle is the natural unit to hand to a collaborator, attach to a
    paper's reproducibility artifact, or load in a downstream notebook — it is
    small, self-describing (via `meta.json`), and needs no NHTS download or
    cache rebuild to reopen. Subsetting with `filter` first keeps the bundle to
    just the EVs your analysis actually uses.

## Next steps

- [Residential aggregate load](residential-load.md) — build load curves from a
  fleet (or a reloaded bundle).
- [Comparing regions](regional-comparison.md) — save one bundle per region and
  reload them for a side-by-side comparison.
- The [API Reference](../api.md) documents `Fleet.save`, `Fleet.load`,
  `Fleet.filter`, and the full `Fleet` / `Profile` surface.
