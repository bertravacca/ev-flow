# Quickstart

This page takes you from a clean environment to a charging **load curve** in a
few steps. The only slow part is a one-time data bootstrap; after that, the
library API is fast and local.

!!! note "`ev-flow` on PyPI, `pev_synth` in Python"
    You **`pip install ev-flow`** and then **`import pev_synth`** — same
    project, two names (the `scikit-learn` / `sklearn` convention). The
    command-line tool is `ev-flow`; the importable Python package is
    `pev_synth`.

## 1. Install

```bash
pip install ev-flow
```

ev-flow supports Python 3.10–3.13. The wheel installs the `pev_synth` package
and its runtime dependencies (numpy, pandas, pyarrow, scikit-learn, scipy).

## 2. Bootstrap the data cache

The wheel ships the Python code plus the small SPEECh K=16 parameters. It does
**not** bundle the cached fleet data — that is built locally from NHTS 2017
microdata the first time you run ev-flow. Bootstrap it once:

```bash
ev-flow bootstrap
```

!!! warning "First run downloads ~80 MB and takes a few minutes"
    `ev-flow bootstrap` downloads the NHTS 2017 public-use file (~80 MB from
    ORNL) and then builds the local fleet cache. Budget a few minutes and a
    working internet connection. It only needs to run once per machine /
    data root; subsequent `generate_profiles(...)` calls read the local cache
    and need no network.

By default the cache is written under a per-user data directory. To control
where it lives, set `PEV_SYNTH_DATA_ROOT` before bootstrapping:

```bash
export PEV_SYNTH_DATA_ROOT=/path/to/your/ev-flow-data
ev-flow bootstrap
```

??? info "Building the cache without the `ev-flow` CLI"
    The `ev-flow bootstrap` / `ev-flow doctor` commands are the recommended
    first-run path. The same result can be produced with the underlying
    module entry points, which is useful in a `pip install -e .` dev checkout:

    ```bash
    # (a) one-time: download (~80 MB ORNL zip) + process NHTS 2017.
    python -m pev_synth.nhts_loader

    # (b) build a cache for the (region, profile_type) you want.
    python -m pev_synth.cache_regen one \
        --region bay_area --profile-type residential
    ```

    If you keep a prebuilt data tree elsewhere, point `PEV_SYNTH_DATA_ROOT`
    at it instead of bootstrapping. The expected layout is
    `<root>/pev/processed/<region>/<profile_type>_ev_synth/`.

## 3. Verify the install

```bash
ev-flow doctor
```

`ev-flow doctor` checks that the package imports, that `PEV_SYNTH_DATA_ROOT`
resolves, and that at least one fleet cache is present and readable — so you
catch a half-finished bootstrap before you write analysis code.

## 4. Generate profiles

Once the cache exists, the library API works entirely offline. The factory is
[`generate_profiles`][pev_synth.generate_profiles], which returns a
[`Fleet`][pev_synth.Fleet] of synthetic EVs:

```python
import pev_synth as ps

# The eight regions and two profile types are always available, even before
# the cache is built:
ps.list_regions()
# ['bay_area', 'boston', 'chicago', 'dallas_fort_worth',
#  'la_basin', 'new_york_metro', 'seattle', 'us_national']
ps.list_profile_types()
# ['residential', 'workplace']

# Draw a 500-EV residential fleet for the Bay Area. `seed` makes the random
# subset selection reproducible.
fleet = ps.generate_profiles('residential', n=500, region='bay_area', seed=42)
fleet
# Fleet(profile_type='residential', region='bay_area', n=500, data_root=...)

# Index into the fleet to get one synthetic EV (a Profile):
prof = fleet[0]
prof.archetype          # e.g. 'compact_bev'
prof.battery_kwh        # usable battery capacity in kWh
prof.max_charge_kw      # min(on-board charger kW, home EVSE kW)
```

!!! tip "No cache yet?"
    If you call `generate_profiles(...)` before bootstrapping, it raises
    `FileNotFoundError` with a message pointing back at the bootstrap step.
    Run [step 2](#2-bootstrap-the-data-cache) first.

## 5. Pull plug status and sessions

Every time-window method accepts string or `pandas.Timestamp` endpoints over a
half-open `[t_start, t_stop)` interval. The cache lives on a 2001 synthetic
calendar, so use 2001 dates (timestamps in another year are re-mapped to 2001,
preserving month / day / hour-of-day).

```python
# Boolean plug-in status at 15-minute resolution for one EV over a week.
# Returns a pandas Series indexed by timestamp; True == plugged in.
plugged = prof.plug_status('2001-01-01', '2001-01-08', freq='15min')
plugged.mean()          # fraction of the week this EV was plugged in

# `generate_presence_absence` is the same call under its methodological name:
same = prof.generate_presence_absence('2001-01-01', '2001-01-08', freq='15min')

# Charging sessions whose [t_in, t_out) intersects the window, as a DataFrame:
sessions = prof.charging_sessions('2001-06-01', '2001-06-08')
sessions[['t_in', 't_out', 'energy_kwh', 'max_kw']].head()

# Continuous state-of-charge trajectory (kWh) for the same EV:
soc = prof.soc_trajectory('2001-06-01', '2001-06-08', freq='15min')
```

By default every series / frame is indexed in UTC (the storage timezone). Pass
`tz=fleet.region.tz` (or any IANA zone) to get the same data on a local
wall-clock index instead.

## 6. Reach a load curve

To get a fleet-level charging **load curve**, ask the `Fleet` directly for its
aggregate power draw:

```python
# Fleet-aggregate charging power in kW, hourly, over one week.
load_kw = fleet.aggregate_load('2001-06-01', '2001-06-08', freq='1h')

# Plot it (requires matplotlib: `pip install matplotlib`):
ax = load_kw.plot(ylabel='aggregate charging power (kW)',
                  title='Bay Area residential fleet — 1 week')
```

That is the ten-minute path: install, bootstrap once, and you have a
reproducible synthetic load curve for a regional EV fleet.

## Working with the whole fleet

`Fleet` is iterable and filterable, and exposes the same time-window methods as
`Profile` (returning wide, per-EV-column frames):

```python
# Per-EV static summary (one row per EV):
fleet.summary().head()

# Narrow to battery-electric vehicles with a fast home connection:
bevs = fleet.filter(powertrain='BEV', max_charge_kw_gte=7.0)
len(bevs)

# Wide boolean plug-in matrix: index = timestamps, columns = ev_id.
matrix = fleet.plug_status('2001-01-01', '2001-01-02', freq='1h')
```

## Workplace caveat

ev-flow ships a `workplace` profile type alongside `residential`, but it comes
with an important, deliberately-surfaced limitation:

```python
work = ps.generate_profiles('workplace', n=200, region='bay_area')
# RuntimeWarning: Workplace fleets are fit from the 105-vehicle public EVWatts
# cohort; plug-in median ~12:00 LT is ~3h later than the literature-canonical
# workplace median of ~09:00 ...
```

The workplace cluster centres are fit from a small (105-vehicle) public EVWatts
cohort whose plug-in median (~12:00 local time) is roughly three hours later
than the literature-canonical workplace median (~09:00). The validator's W1–W4
checks flag this as `EXPLAINED_FAIL` — a documented, understood divergence
rather than a bug. ev-flow surfaces it as a `RuntimeWarning` at fleet
construction so it can never be silently mis-interpreted. Treat the workplace
output as exploratory and keep this caveat in mind for any downstream analysis.

Vehicle-to-grid (V2G), smart-charging optimisation, fleet-depot scheduling and
public-charging behaviour are **not** modelled by ev-flow.

## Next steps

- The full [API Reference](api.md) documents every `Fleet` / `Profile` method,
  the timezone and year-remap rules, and the region registry.
- The [Migration](migration.md) page covers the `ev-flow` / `pev_synth` naming
  split and how existing callers keep working.
