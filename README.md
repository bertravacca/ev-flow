# ev-flow

[![CI](https://github.com/bertravacca/ev-flow/actions/workflows/ci.yml/badge.svg)](https://github.com/bertravacca/ev-flow/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/bertravacca/ev-flow/branch/main/graph/badge.svg)](https://codecov.io/gh/bertravacca/ev-flow)
[![security](https://github.com/bertravacca/ev-flow/actions/workflows/security.yml/badge.svg)](https://github.com/bertravacca/ev-flow/actions/workflows/security.yml)
[![PyPI version](https://img.shields.io/pypi/v/ev-flow.svg)](https://pypi.org/project/ev-flow/)
[![Python versions](https://img.shields.io/pypi/pyversions/ev-flow.svg)](https://pypi.org/project/ev-flow/)
[![DOI](https://zenodo.org/badge/1245084094.svg)](https://zenodo.org/badge/latestdoi/1245084094)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Synthetic plug-in electric vehicle (PEV) charging dataset pipeline and library API.

`ev-flow` generates realistic, fleet-scale charging behavior for residential and workplace EVs, grounded in the National Household Travel Survey (NHTS) and a regional sales-mix model. It exposes both a low-level pipeline (NHTS loading, donor matching, travel-week building, plug-in modeling, state-of-charge trajectory, hourly rasterisation) and a clean `Fleet` / `Profile` library API for downstream studies.

[![Open quickstart in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bertravacca/ev-flow/blob/main/notebooks/ev_flow_quickstart.ipynb) — try the quickstart in your browser, no local install needed.

## Install

```bash
pip install ev-flow
```

The wheel ships the Python package and the small bundled SPEECh K=16 parameters — but no cached fleet bundles and no NHTS microdata. After installing, run the one-time **First run** below to build them (or point `PEV_SYNTH_DATA_ROOT` at a prebuilt data tree). Until then `generate_profiles(...)` raises `FileNotFoundError`.

## First run

`ev-flow` installs a console command, `ev-flow`, with two subcommands:

- **`ev-flow bootstrap`** — one-time setup: writes the offline sales-mix CSVs, downloads the NHTS 2017 microdata (**~80 MB ORNL zip**), and builds the cached fleet bundle for the region(s) you ask for. It runs the full M1–M9 pipeline, so the first run takes **a few minutes**. Every step is **idempotent** — re-running skips work whose outputs already exist.
- **`ev-flow doctor`** — read-only diagnostics: prints a `Check | Status | Details` table covering your Python version, the runtime dependencies, whether the data root is writable, and which sales-mix CSVs / NHTS parquets / per-region caches are present. It never mutates anything.

```bash
# (1) one-time setup — build the default bay_area / residential cache.
#     Downloads ~80 MB of NHTS data and runs the pipeline (a few minutes).
ev-flow bootstrap

#     ...or pick regions / profile types explicitly:
ev-flow bootstrap --regions bay_area la_basin --profile-types residential workplace

# (2) diagnose your environment any time:
ev-flow doctor            # add --verbose to also list passing rows
                          # add --smoke to try generate_profiles(..., n=1)

# (3) now the library API works:
python -c "import pev_synth as ps; print(ps.generate_profiles('residential', n=10, region='bay_area'))"
```

`ev-flow doctor` exits non-zero if the environment is not usable yet (a missing runtime dependency, an unwritable data root, or a requested cache that has not been built) — handy in a `ev-flow doctor && ...` gate. Pass `--data-root PATH` to either subcommand to target a specific data directory (it sets `PEV_SYNTH_DATA_ROOT` for that run).

> ACS PUMS calibration is **off by default** during bootstrap (`--skip-acs`, the default), because it needs a `CENSUS_API_KEY`. The built-in regions do not require it.

If you installed from PyPI and already have a prebuilt data tree, you do **not** need `bootstrap` — just point `PEV_SYNTH_DATA_ROOT` at it (next section) and run `ev-flow doctor` to confirm.

### Equivalent module invocations (dev checkout)

`ev-flow bootstrap` is a thin, idempotent wrapper over the same module entry points you can also call directly in a `pip install -e .` dev checkout:

```bash
# (a) one-time NHTS 2017 download + processing (~80 MB ORNL zip).
python -m pev_synth.nhts_loader

# (b) build a cache for one (region, profile_type).
python -m pev_synth.cache_regen one --region bay_area --profile-type residential
```

## Data directory

`ev-flow` ships only the Python package; the cached fleet bundles (NHTS-derived parquets etc.) are not bundled in the wheel. Point the package at your local data directory via the `PEV_SYNTH_DATA_ROOT` environment variable:

```bash
export PEV_SYNTH_DATA_ROOT=/path/to/your/ev-flow-data
```

The directory should contain the `pev/processed/<region>/<profile_type>_ev_synth/` layout that `ev-flow bootstrap` (and `python -m pev_synth.cache_regen one ...`) writes. If `PEV_SYNTH_DATA_ROOT` is unset, the package falls back to `<repo_root>/data/` — only useful in a `pip install -e .` dev checkout where the `data/` tree sits next to `src/`.

The NHTS download step runs only once: the loader persists the national `hhpub.csv` / `vehpub.csv`, so non-CA regions (`boston`, `chicago`, `dallas_fort_worth`, `new_york_metro`, `seattle`) are then built without re-downloading.

## Quick start

```python
import pev_synth as ps

ps.list_regions()
# ['bay_area', 'boston', 'chicago', 'dallas_fort_worth',
#  'la_basin', 'new_york_metro', 'seattle', 'us_national']

ps.list_profile_types()
# ['residential', 'workplace']

fleet = ps.generate_profiles('residential', n=1000, region='bay_area', seed=42)
prof  = fleet[0]

pa   = prof.generate_presence_absence('2001-01-01', '2001-01-08', freq='15min')
sess = prof.charging_sessions('2001-06-01', '2001-06-08')
soc  = prof.soc_trajectory('2001-06-01', '2001-06-08', freq='15min')
```

The PyPI distribution name is **`ev-flow`** but the Python import name is **`pev_synth`** (this mirrors the `scikit-learn` / `sklearn` convention).

## Workplace caveat

In v2.0 the `workplace` cluster centres are fit from the 105-vehicle public EVWatts cohort, whose plug-in median is ~12:00 LT — approximately 3 hours later than the literature-canonical workplace median of ~09:00 LT. The W1-W4 validator checks flag this divergence as `EXPLAINED_FAIL` rather than as a bug. `pev_synth` surfaces this caveat as a `RuntimeWarning` at `Fleet.__init__` whenever `profile_type == 'workplace'`. See `src/pev_synth/plug_in_model.py:42-48` for the full discussion.

## Modules

| Module | Purpose |
|---|---|
| `nhts_loader` | National Household Travel Survey 2017 public-use file loader |
| `vehicle_archetypes` | N-EV archetype sampler |
| `donor_matcher` | NHTS donor-vehicle matcher |
| `travel_week_builder` | One-year travel sequence builder |
| `plug_in_model` | Session plug-in / dwell sampler |
| `soc_trajectory` | Continuous-time state-of-charge ledger + session extraction |
| `hourly_resampler` | 15-minute and hourly plug-status rasteriser |
| `validation_bounds_curator` | Bound curation |
| `validator` | Validation runner + report writer (11 §10 + 3 integration + 1 DST + 1 winter + 10 workplace + 1 workplace-optim checks) |
| `regions` | 8-region registry |

Full library API reference and methodology rationale live in the [`documentation/`](documentation/) folder (expanding ahead of the docs-site launch).

## Development

```bash
git clone https://github.com/bertravacca/ev-flow
cd ev-flow
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Versioning

`ev-flow` follows [Semantic Versioning](https://semver.org). See
[`documentation/versioning.md`](documentation/versioning.md) for what counts as
a major/minor/patch change, the deprecation policy, and the distinction between
the package version and a cache's `methodology_version`.

## Data sources & attribution

`ev-flow` is grounded in public data sources. Two upstream notices are required:

- **SPEECh Original Model** (Powell, Cezar & Rajagopal, Mendeley Data, 2021) is
  licensed **CC BY 4.0**. ev-flow consumes a modified (pickle→JSON, reweighted)
  subset of its driver-group mixtures. <https://doi.org/10.17632/gvk34mybtb.1>
- This product uses the **U.S. Census Bureau Data API** but is not endorsed or
  certified by the Census Bureau.

Full per-source licensing, citations, and the modification statement are in
[`ATTRIBUTION.md`](ATTRIBUTION.md) (NHTS, Census ACS PUMS, EPA fueleconomy.gov,
EV WATTS, CVRP, NYSERDA, Argonne EV-FACTS, NOAA, NREL).

## License

MIT. See `LICENSE`. Note that the MIT license covers the `ev-flow` *code*; the
upstream data sources retain their own licenses — see
[`ATTRIBUTION.md`](ATTRIBUTION.md).
