# Versioning Policy

`ev-flow` follows [Semantic Versioning](https://semver.org). The PyPI
distribution is `ev-flow`; the import package is `pev_synth`. Two version
numbers coexist and must not be conflated:

- **Package version** — `pev_synth.__version__` (currently `3.0.1`) and the
  `version` field in `pyproject.toml`, kept in sync. Governs the public
  Python API and wheel contents. This is the SemVer number a `vX.Y.Z` tag
  releases.
- **`methodology_version`** — written into a cache's `meta.json` by
  `cache_regen` / the pipeline modules (e.g. `v3.0`). Governs the
  *statistical content* of generated data. The two versions move
  independently: a package release MAY ship without bumping the methodology,
  and a methodology bump MAY ship in a minor package release.

The public API surface this policy covers is exactly the names in
`pev_synth.__all__`: `Fleet`, `Profile`, `ProfileType`, `REGIONS`, `Region`,
`generate_profiles`, `list_profile_types`, `list_regions`, `regenerate_fleet`,
and `__version__`. Names not in `__all__` (e.g. the `FleetReader` protocol,
`_utc_migration`, anything underscore-prefixed) are internal and exempt.

## MAJOR (breaking)

- Removing or renaming any name in `pev_synth.__all__`, or changing the
  signature or return type of `Fleet`, `Profile`, `generate_profiles`,
  `list_regions`, `list_profile_types`, or the `Region` / `REGIONS` registry.
- Changing output **schema** — column names, ordering, or dtypes — of the
  frames returned by `Fleet`/`Profile` methods: `charging_sessions`,
  `generate_presence_absence` (and its `plug_status` alias), `soc_trajectory`,
  `summary`, or `aggregate_load`.
- Removing a region or profile type, or changing `ev_id` semantics.
- A `methodology_version` bump that invalidates existing caches such that
  consumers must regenerate via `python -m pev_synth.cache_regen`.

## MINOR (backward-compatible)

- New regions or profile types; new optional keyword arguments (with safe
  defaults); new public helpers, properties, or `Fleet.filter` predicates.
- Promoting a currently-stub entry point to a working implementation —
  `regenerate_fleet` today raises `NotImplementedError` and directs callers
  to `cache_regen`; giving it a real return value is additive.
- Additive `meta.json` keys that default to absent / `None`. The v3.0
  optional provenance fields (`nhts_vintage_mix`, `houseid_stitch_mode`,
  `phev_modeled`, `acs_calibration_applied`, `speech_k16_axis_projection`)
  are the template: a v2.1 `meta.json` still validates unchanged.
- A methodology bump that is forward-compatible: prior caches remain
  readable and queryable without regeneration.

## PATCH

- Bug fixes that preserve API and output schema; dependency-pin tightening
  (e.g. the `pandas<3` cap in 3.0.1); documentation and packaging fixes.

## Deprecation

We do not yet have a formal deprecation window. When one is established, the
intended process is: public API slated for removal is marked with a
`DeprecationWarning` (message naming the replacement and the removal version),
emitted at call time, for at least one minor release before removal in a
subsequent MAJOR. No fixed-day guarantee (e.g. an LTS or N-day window) is
promised at this time. Note that the warnings currently emitted by the API
(the year-remap day-of-week `UserWarning` and the workplace-cohort
`RuntimeWarning`) are informational, not deprecations.

## Cache compatibility

Caches produced by `cache_regen` carry a `meta.json` whose canonical schema
(`pev_synth._meta.MetaSchema`) includes `methodology_version`, `master_seed`,
and `storage_timezone` (`"UTC"`). `Fleet.save` writes a subset bundle whose
`meta.json` records `profile_type`, `region`, the subset `ev_ids`, `n`, and
`storage_timezone`, plus any inherited keys.

`Fleet.load` currently validates only that `profile_type` and `region` are
present and well-formed; it does **not** compare `methodology_version` against
the installed package or auto-reject mismatched caches. Methodology /
schema compatibility is therefore the caller's responsibility today: a cache
written by a newer methodology, or one predating a cache-invalidating MAJOR,
should be regenerated with `python -m pev_synth.cache_regen` before use.
Runtime `meta.json` validation lives in `pev_synth.validator.run`, not in the
loader.
