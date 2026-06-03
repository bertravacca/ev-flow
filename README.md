# ev-flow

[![CI](https://github.com/bertravacca/ev-flow/actions/workflows/ci.yml/badge.svg)](https://github.com/bertravacca/ev-flow/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/ev-flow.svg)](https://pypi.org/project/ev-flow/)
[![Python versions](https://img.shields.io/pypi/pyversions/ev-flow.svg)](https://pypi.org/project/ev-flow/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Synthetic plug-in electric vehicle (PEV) charging dataset pipeline and library API.

`ev-flow` generates realistic, fleet-scale charging behavior for residential and workplace EVs, grounded in the National Household Travel Survey (NHTS) and a regional sales-mix model. It exposes both a low-level pipeline (NHTS loading, donor matching, travel-week building, plug-in modeling, state-of-charge trajectory, hourly rasterisation) and a clean `Fleet` / `Profile` library API for downstream studies.

## Install

```bash
pip install ev-flow
```

Then set `PEV_SYNTH_DATA_ROOT` to point at your data tree — see the next section. Without that step, `generate_profiles(...)` will raise `FileNotFoundError` because the wheel does not bundle the cached fleet bundles.

## Data directory

`ev-flow` ships only the Python package; the cached fleet bundles (NHTS-derived parquets etc.) are not bundled in the wheel. Point the package at your local data directory via the `PEV_SYNTH_DATA_ROOT` environment variable:

```bash
export PEV_SYNTH_DATA_ROOT=/path/to/your/ev-flow-data
```

The directory should contain the `pev/processed/<region>/<profile_type>_ev_synth/` layout that `python -m pev_synth.cache_regen one ...` writes. If `PEV_SYNTH_DATA_ROOT` is unset, the package falls back to `<repo_root>/data/` — only useful in a `pip install -e .` dev checkout where the `data/` tree sits next to `src/`.

### First run / bootstrap (dev checkout)

The cached fleet bundles are **not** in the repo and **not** in the wheel — you build them from NHTS 2017 microdata, which is also not bundled. For a fresh `pip install -e .` dev checkout the one-time sequence is:

```bash
# (a) one-time: download (~84 MB ORNL zip) + process NHTS 2017.
#     Writes the California parquets and the national hhpub.csv/vehpub.csv
#     to data/pev/raw/nhts2017/.
python -m pev_synth.nhts_loader

# (b) build a cache for the (region, profile_type) you want.
#     Subcommands are `one`, `batch`, `audit`.
python -m pev_synth.cache_regen one --region bay_area --profile-type residential

# (c) now the library API works:
python -c "import pev_synth as ps; print(ps.generate_profiles('residential', n=10, region='bay_area'))"
```

Step (a) runs once: the loader persists the national `hhpub.csv` / `vehpub.csv`, so non-CA regions (`boston`, `chicago`, `dallas_fort_worth`, `new_york_metro`, `seattle`) are then handled automatically by `cache_regen one` without re-downloading.

Pip-installed (non-dev) users do not run the bootstrap — instead point `PEV_SYNTH_DATA_ROOT` at a prebuilt data tree as described above.

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
