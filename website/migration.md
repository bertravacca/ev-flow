# Migration

This page explains the one piece of ev-flow that surprises newcomers — that
the install name and the import name differ — and how existing callers move
onto the published package without changing a line of import code.

## `ev-flow` (PyPI) vs `pev_synth` (import)

ev-flow is distributed on PyPI under the name **`ev-flow`**, but the Python
package you import is **`pev_synth`**:

```bash
pip install ev-flow
```

```python
import pev_synth as ps
```

This is intentional and follows the most familiar precedent in the scientific
Python ecosystem: you `pip install scikit-learn` and then `import sklearn`.
The distribution name (what you install, what shows on PyPI, what a `vX.Y.Z`
git tag publishes) is decoupled from the import name (the top-level package on
`sys.path`).

### Why the split exists

`pev_synth` is the original Python package name — it was the import path used
throughout the internal *aggregated EVs study* before the library was extracted
into a standalone, publishable project. When that extraction happened, the
public **distribution** was given the more discoverable, product-style name
**`ev-flow`**, while the **import** name stayed `pev_synth` so that:

- every existing `import pev_synth` in the study (and in any notebook, script
  or downstream module) keeps working unchanged, and
- the public package gets a memorable, collision-resistant name on PyPI.

Renaming the import package as well would have been a hard, breaking change for
every caller for no functional benefit. Keeping `pev_synth` as the import name
is the backward-compatible choice.

!!! note "One project, two names"
    Throughout this documentation, "ev-flow" and "`pev_synth`" refer to the
    same project. Use `ev-flow` when you install or cite it; use `pev_synth`
    when you import it in Python.

## Callers from the internal study

Code in the *aggregated EVs study* imports `pev_synth` directly, e.g.:

```python
import pev_synth as ps

fleet = ps.generate_profiles('residential', n=1000, region='bay_area')
```

That code continues to work as long as `ev-flow` is installed in the active
environment. Nothing in the import statements or the call sites needs to
change: installing the `ev-flow` distribution puts the `pev_synth` package on
`sys.path` exactly as before.

So the migration for an existing caller is simply:

```bash
pip install ev-flow
```

…and the existing `import pev_synth` code keeps running.

### Editable install during development

If you are developing the study and ev-flow side by side and want your edits to
ev-flow picked up immediately, install it in editable mode from a local
checkout instead of from PyPI:

```bash
# from the study's environment, pointing at a local ev-flow checkout
pip install -e ../ev-flow
```

An editable install also exposes the package as `pev_synth`, so imports are
identical to a regular install — you just get live source instead of a pinned
wheel. This is the recommended setup when iterating on both repositories at
once.

## What ships, and what does not

It is worth being precise about the packaging boundary, because it affects
first-run behaviour:

- The published wheel contains **only** the `src/pev_synth/` Python package.
- It does **not** bundle the cached fleet data (the NHTS-derived parquet
  bundles). Those are built locally on first run — see the
  [Quickstart](quickstart.md#2-bootstrap-the-data-cache).
- Repository-only files (`README.md`, tests, CI config, the maintainer's
  working notes) are not part of the distribution.

So after `pip install ev-flow`, `import pev_synth` works immediately, but
`generate_profiles(...)` needs the one-time bootstrap before it returns data.

## Versioning & SemVer

ev-flow follows [Semantic Versioning](https://semver.org). Two version numbers
coexist and must not be conflated:

- the **package version** (`pev_synth.__version__`, kept in sync with
  `pyproject.toml`), which a `vX.Y.Z` git tag releases and which governs the
  public Python API and wheel contents; and
- the cache **`methodology_version`** written into each cache's `meta.json`,
  which governs the statistical content of generated data and moves
  independently of the package version.

The public API the policy covers is exactly the names in `pev_synth.__all__`:
`Fleet`, `Profile`, `ProfileType`, `REGIONS`, `Region`, `generate_profiles`,
`list_profile_types`, `list_regions`, `regenerate_fleet`, and `__version__`.
Underscore-prefixed names and the `FleetReader` protocol are internal and
exempt.

For the full breakdown of what counts as a major / minor / patch change, the
deprecation policy, and cache-compatibility rules, see
[`documentation/versioning.md`](https://github.com/bertravacca/ev-flow/blob/main/documentation/versioning.md)
in the repository.
