# Changelog

All notable changes to **ev-flow** are documented here. This project follows
[Semantic Versioning](https://semver.org).

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
