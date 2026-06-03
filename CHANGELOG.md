# Changelog

All notable changes to **ev-flow** are documented here. This project follows
[Semantic Versioning](https://semver.org).

## [Unreleased]

### Added
- `py.typed` marker so downstream type checkers see ev-flow's type hints (PEP 561).
- `ATTRIBUTION.md` documenting every upstream data source's license and the
  required SPEECh (CC BY 4.0) and Census Data API notices; surfaced in the README.
- `documentation/versioning.md` — the SemVer / deprecation / cache-compatibility policy.
- CI now runs the full `{ubuntu, macos, windows} × {3.10, 3.11, 3.12, 3.13}` matrix; README CI/PyPI/license badges.

### Fixed
- Cross-platform correctness: UTF-8 encoding on JSON/Markdown report and `meta.json`
  I/O (`validator`, `api`); OS-agnostic path assertions in the test suite so Windows
  CI passes.

## [3.0.1] - 2026-06-02

### Fixed
- Pin `pandas>=2.2,<3`. pandas 3.0 changed the default datetime resolution
  from nanoseconds to microseconds, which broke the 15-minute plug-status
  rasteriser, the SoC-trajectory parquet schema, and UTC/DST timestamp
  handling for users who resolved pandas 3.0 on Python 3.11+. The published
  3.0.0 declared an uncapped `pandas>=2.2`; this release caps it.

## [3.0.0] - 2026-06-01

### Added
- Initial public release. Synthetic plug-in electric vehicle charging dataset
  pipeline and library API, grounded in NHTS 2017 micro-data and a regional
  sales-mix model across 8 US regions.
- PyPI distribution name `ev-flow`; Python import name `pev_synth`.

[3.0.1]: https://github.com/bertravacca/ev-flow/releases/tag/v3.0.1
[3.0.0]: https://github.com/bertravacca/ev-flow/releases/tag/v3.0.0
