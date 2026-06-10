# Residential aggregate load

Generate a residential Bay Area fleet, build its aggregate plugged-in and
charging-load curves over a representative week, and plot them.

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
    step. Until the cache exists, `generate_profiles(...)` raises
    `FileNotFoundError`.

The plotting helpers also need matplotlib, which is an optional extra:

```bash
pip install "ev-flow[plotting]"
```

## 1. Generate a residential fleet

[`generate_profiles`][pev_synth.generate_profiles] returns a
[`Fleet`][pev_synth.Fleet] — a lazy collection of synthetic EVs backed by the
local parquet cache. Draw 500 residential EVs for the Bay Area; `seed` makes the
random subset selection reproducible.

```python
import pev_synth as ps

fleet = ps.generate_profiles('residential', n=500, region='bay_area', seed=42)
fleet
# Fleet(profile_type='residential', region='bay_area', n=500, data_root=...)

len(fleet)
# 500
```

## 2. Build the aggregate plugged-in curve

The cache lives on a **2001 synthetic calendar**, so query it with 2001 dates.
Time windows are half-open over `[t_start, t_stop)`; here we take one
representative week.

[`Fleet.plug_status`][pev_synth.Fleet.plug_status] (an alias of
[`generate_presence_absence`][pev_synth.Fleet.generate_presence_absence])
returns a **wide boolean matrix**: the index is the timestamp, and there is one
column per `ev_id`, with `True` where that EV is plugged in. Summed across the
columns, that matrix is the number of EVs plugged in at each timestep.

```python
# Wide bool matrix: index = 15-min timestamps, columns = ev_id.
plug = fleet.plug_status('2001-06-04', '2001-06-11', freq='15min')

plug.shape
# (672, 500)  -> 7 days x 96 fifteen-minute steps, 500 EVs

# Plugged-in EV count per timestep is just the row-sum of the bool matrix:
plug.sum(axis=1).head()
```

By default the index is in UTC (the storage timezone). Pass `tz=fleet.region.tz`
(here `America/Los_Angeles`) to read the same data on a local wall-clock index
instead, which is usually what you want for an interpretable daily shape:

```python
plug_local = fleet.plug_status(
    '2001-06-04', '2001-06-11', freq='15min', tz=fleet.region.tz
)
```

## 3. Plot the plugged-in curve

The `pev_synth.plotting.plot_aggregate_load` helper consumes that wide
plug-status matrix directly and draws the **plugged-in EV count** over time. It
sums across the per-EV columns for you, so pass the matrix as-is. The helper
returns the Matplotlib `Axes` it drew on and never calls `plt.show()`, so
rendering or saving is up to you.

```python
import matplotlib.pyplot as plt
from pev_synth import plotting

ax = plotting.plot_aggregate_load(plug_local, label='plugged-in EVs (500-EV fleet)')
ax.set_title('Bay Area residential fleet — plugged-in count, 1 week')
plt.savefig('bay_area_residential_plugged_in.png', dpi=150, bbox_inches='tight')
```

If you start from a 15-minute matrix but want a smoother hourly line, pass a
pandas offset alias via `freq` and the helper resamples the per-step count to
that grid using the mean:

```python
ax = plotting.plot_aggregate_load(plug_local, freq='1h')
```

## 4. The charging-load curve (kW)

The plugged-in *count* above tells you how many cars are **connected**; it is
not the same as the power they **draw**. For the latter, ask the fleet directly
for its aggregate charging power with
[`Fleet.aggregate_load`][pev_synth.Fleet.aggregate_load], which returns a
`pandas.Series` of kW under a charge-as-soon-as-plugged-in baseline (each
session draws constant `max_kw` from `t_in` until it has delivered its energy,
then idles).

```python
# Fleet-aggregate charging power in kW, hourly, over the same week.
load_kw = fleet.aggregate_load(
    '2001-06-04', '2001-06-11', freq='1h', tz=fleet.region.tz
)

# Series.plot is plain pandas/matplotlib — no pev_synth helper needed here.
ax = load_kw.plot(
    ylabel='aggregate charging power (kW)',
    xlabel='time (America/Los_Angeles)',
    title='Bay Area residential fleet — charging load, 1 week',
)
plt.savefig('bay_area_residential_load_kw.png', dpi=150, bbox_inches='tight')
```

## What you should see

Two related but distinct weekly curves on a local-time axis:

- **Plugged-in count** (step 3): a daily rhythm that rises through the evening
  as vehicles return home and plug in, plateaus high overnight while most of the
  fleet sits connected, and falls through the morning as cars unplug to drive.
  Residential connection is dominated by the at-home overnight window.
- **Charging load in kW** (step 4): sharper than the connection curve, because
  power is drawn only while a session is actively delivering energy, not for the
  whole time a car is plugged in. Under the charge-as-soon-as-plugged-in
  baseline, the kW curve peaks when many vehicles begin charging together
  (typically the evening arrival window) and decays as each session finishes and
  idles, well before the car unplugs.

The exact heights depend on `n`, the region, the week you pick, and your random
`seed`. The shapes — an evening-led plug-in build-up with an overnight plateau,
and a sharper evening-led kW peak — are the residential signal to expect.

!!! note "What this curve is, and is not"
    `aggregate_load` is an *uncontrolled* charge-as-soon-as-plugged-in baseline.
    ev-flow does **not** model smart-charging, managed/throttled charging,
    time-of-use optimisation, or vehicle-to-grid (V2G). Treat the kW curve as
    the unmanaged reference load, not a controlled-charging schedule.

## Next steps

- [Comparing regions](regional-comparison.md) — overlay this Bay Area shape
  against another region.
- [Save and reload a fleet](save-reload.md) — persist this fleet (or a subset)
  so you can reload it without re-querying the full cache.
- The [API Reference](../api.md) documents the timezone and year-remap rules and
  every `Fleet` / `Profile` method in full.
