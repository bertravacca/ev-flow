# Changelog

All notable changes to **ev-flow** are documented here. This project follows
[Semantic Versioning](https://semver.org).

## [Unreleased]

### Added
- The SPEECh K=16 parameter JSONs (start-time marginals + group priors) are now
  **bundled in the wheel and sdist** under `pev_synth/data/speech/`
  (CC-BY-4.0 — see ATTRIBUTION.md). As a result `ev-flow bootstrap` builds on a
  fresh `pip install` without obtaining the upstream Mendeley source; previously
  the M5 plug-in step raised `FileNotFoundError`. A maintainer's locally-derived
  copy under `data/pev/raw/speech/` still takes precedence.
- Static type-checking with **mypy** is now a required CI gate (F27). The
  public API surface and lighter helpers/loaders are checked under
  `mypy --strict`; the heavy internal pipeline modules are relaxed for now as
  a deliberate gradual-typing step (tracked for a follow-up pass). Config lives
  under `[tool.mypy]` in `pyproject.toml`; `mypy>=1.10` was added to the `dev`
  extra. This is a repo/CI-only change — it does not affect the published wheel.

## [3.0.2] - 2026-06-09

### Added
- The PEP 561 `py.typed` marker now ships inside the `pev_synth` wheel, so
  downstream type checkers (mypy, pyright) pick up ev-flow's inline type hints
  when the package is installed from PyPI.
- `ATTRIBUTION.md` is now included in the source distribution, documenting every
  upstream data source's license and the required SPEECh (CC BY 4.0) and U.S.
  Census Bureau Data API notices.

### Fixed
- Cross-platform correctness: all `meta.json`, `validation_report.json`, and
  `validation_report.md` reads and writes in `api` and `validator` now pass an
  explicit `encoding="utf-8"`, instead of relying on the platform default
  encoding. This makes bundle metadata and validation reports round-trip
  identically on Windows (where the default was previously `cp1252`).

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

[3.0.2]: https://github.com/bertravacca/ev-flow/releases/tag/v3.0.2
[3.0.1]: https://github.com/bertravacca/ev-flow/releases/tag/v3.0.1
[3.0.0]: https://github.com/bertravacca/ev-flow/releases/tag/v3.0.0
