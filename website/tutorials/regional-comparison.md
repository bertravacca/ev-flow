# Comparing regions

Generate residential fleets for two regions, overlay their aggregate load
shapes, and read off where — and why — they differ.

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
    step. You need a cache for **each** region you compare; bootstrap can build
    them, or use `python -m pev_synth.cache_regen one --region <name>
    --profile-type residential` per region in a dev checkout. Until a region's
    cache exists, `generate_profiles(...)` raises `FileNotFoundError` for it.

The plotting helper also needs matplotlib (`pip install "ev-flow[plotting]"`).

## 1. Generate two regional fleets

The eight registered regions are always listed by
[`list_regions`][pev_synth.list_regions], even before any cache is built:

```python
import pev_synth as ps

ps.list_regions()
# ['bay_area', 'boston', 'chicago', 'dallas_fort_worth',
#  'la_basin', 'new_york_metro', 'seattle', 'us_national']
```

Draw a residential fleet for each of two contrasting regions. We use the same
`n` and `seed` for both so the comparison isolates the *region*, not the fleet
size or the random draw.

```python
bay = ps.generate_profiles('residential', n=500, region='bay_area', seed=7)
nyc = ps.generate_profiles('residential', n=500, region='new_york_metro', seed=7)

bay.region.display_name, nyc.region.display_name
# ('Bay Area, CA', 'New York Metro')
bay.region.tz, nyc.region.tz
# ('America/Los_Angeles', 'America/New_York')
```

!!! tip "Compare like with like"
    Both fleets share `profile_type='residential'`, the same `n`, and the same
    `seed`. Keeping those fixed means any difference you see in the curves comes
    from the regional model (travel patterns, sales mix, climate), not from
    sampling noise or scale.

## 2. Overlay the plugged-in curves

Pull each fleet's wide plug-status matrix over the same representative week.
Read both on **each region's own local wall-clock** (`tz=...region.tz`) so the
daily shapes line up by local time of day rather than by UTC — New York is three
hours ahead of the Bay Area, and overlaying raw UTC would shift the curves apart
for that reason alone.

```python
window = ('2001-06-04', '2001-06-11')

bay_plug = bay.plug_status(*window, freq='15min', tz=bay.region.tz)
nyc_plug = nyc.plug_status(*window, freq='15min', tz=nyc.region.tz)
```

The `pev_synth.plotting.plot_aggregate_load` helper sums each wide matrix to a
plugged-in count and draws it. Pass a shared `ax` to overlay both on one figure,
and give each call a distinct `label` and `color`:

```python
import matplotlib.pyplot as plt
from pev_synth import plotting

fig, ax = plt.subplots(figsize=(10, 4))
plotting.plot_aggregate_load(bay_plug, ax=ax, label='Bay Area', color='#0072B2')
plotting.plot_aggregate_load(nyc_plug, ax=ax, label='New York Metro', color='#D55E00')
ax.set_title('Residential plugged-in count — Bay Area vs New York Metro (local time)')
plt.savefig('region_comparison_plugged_in.png', dpi=150, bbox_inches='tight')
```

To compare the *shape* rather than the *level* (both fleets are 500 EVs here,
but in general they need not be), normalise each curve to a plugged-in fraction
before plotting:

```python
bay_frac = bay_plug.mean(axis=1).to_frame('frac')   # mean over EVs == fraction plugged in
nyc_frac = nyc_plug.mean(axis=1).to_frame('frac')

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(bay_frac.index, bay_frac['frac'], color='#0072B2', label='Bay Area')
ax.plot(nyc_frac.index, nyc_frac['frac'], color='#D55E00', label='New York Metro')
ax.set_ylabel('fraction of fleet plugged in')
ax.set_xlabel('time (each region, local)')
ax.legend(loc='best')
```

## 3. Overlay the charging-load curves (kW)

The plugged-in *count* shows connection; the kW **load** shows power drawn.
[`Fleet.aggregate_load`][pev_synth.Fleet.aggregate_load] returns the charge-as-
soon-as-plugged-in kW baseline as a `pandas.Series`:

```python
bay_kw = bay.aggregate_load(*window, freq='1h', tz=bay.region.tz)
nyc_kw = nyc.aggregate_load(*window, freq='1h', tz=nyc.region.tz)

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(bay_kw.index, bay_kw.to_numpy(), color='#0072B2', label='Bay Area')
ax.plot(nyc_kw.index, nyc_kw.to_numpy(), color='#D55E00', label='New York Metro')
ax.set_ylabel('aggregate charging power (kW)')
ax.set_xlabel('time (each region, local)')
ax.set_title('Residential charging load — Bay Area vs New York Metro')
ax.legend(loc='best')
plt.savefig('region_comparison_load_kw.png', dpi=150, bbox_inches='tight')
```

## What you should see, and why

Both regions share the residential signature — an evening-led plug-in build-up,
an overnight plateau, and a morning unplug. The differences between them trace
back to how the regions are defined in ev-flow's region registry, so keep the
interpretation grounded in what the model actually varies:

- **Sales mix differs.** Bay Area uses a California CVRP-derived sales mix;
  New York Metro uses the NYSERDA Drive Clean mix. Different battery-size and
  powertrain mixes shift per-session energy and the achievable `max_charge_kw`,
  which moves the **height** of the kW curve more than its timing.
- **Climate differs.** New York Metro carries a winter energy-uplift multiplier
  (cold-weather consumption is higher per mile), whereas the Bay Area's is
  pinned at 1.0. In a **summer** week like the one above this barely shows; the
  gap widens if you query a **winter** week (e.g. `('2001-01-08', '2001-01-15')`)
  because higher per-mile energy means more kWh to deliver per session.
- **Housing differs (interpret with care).** New York Metro is flagged in the
  region notes as multi-unit-dwelling (MUD) dominant, while the Bay Area is
  more single-family. This is the regional context most relevant to *residential
  access to home charging* in the real world.

!!! warning "Don't over-read the housing difference"
    ev-flow's v3.x residential pipeline grounds **travel and plug-in behaviour**
    in NHTS micro-data and the regional sales mix; it does **not** model
    curbside, garage-orphan, or shared-MUD charging *infrastructure* as a
    distinct mechanism, and it does not model public or workplace charging as a
    substitute for absent home charging. The MUD note is regional context for
    interpreting the result, not a claim that ev-flow simulates MUD charging
    access. The dominant driver of the *shape* you see is still the at-home
    overnight travel/plug-in pattern shared by both regions. Read region-to-
    region differences as directional, and confirm against the
    [Methodology](../api.md) and region notes before drawing strong conclusions.

The honest summary: expect the two regions to look broadly similar in **shape**
(residential overnight charging dominates everywhere) and to differ mostly in
**level** and in **seasonal sensitivity**, driven by the sales mix and the
winter multiplier. If you need a starker contrast, compare a mild-climate region
against a cold one (e.g. `seattle` vs `chicago`) on a winter week.

## Next steps

- [Residential aggregate load](residential-load.md) — the single-region version
  of these curves, with more on what each one represents.
- [Save and reload a fleet](save-reload.md) — persist each regional fleet so a
  later comparison reloads instantly.
- The [API Reference](../api.md) and region registry document each region's
  states, timezone, sales-mix source and climate multiplier.
