# Contributing to ev-flow

Thanks for your interest in improving **ev-flow**. This guide covers local
setup, the test workflow, and what we expect from pull requests.

By participating you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

## Project naming: `ev-flow` vs `pev_synth`

The **PyPI distribution name** is `ev-flow` (what you `pip install`), while the
**Python import name** is `pev_synth` (what you `import`). This split is
intentional and follows the `scikit-learn` / `sklearn` precedent:

```bash
pip install ev-flow
```
```python
import pev_synth as ps
```

You will see `pev_synth` throughout the source tree (`src/pev_synth/`), the
tests (`tests/test_pev_synth_*.py`), and module commands
(`python -m pev_synth.…`). That is the import name and is correct.

## Development setup

ev-flow targets **Python 3.10–3.13**. From a fresh checkout:

```bash
# 1. Editable install with the dev toolchain (pytest, ruff, build, bandit, …).
pip install -e ".[dev]"

# 2. Build the literature sales-mix CSVs the validator tests depend on.
#    The data/ directory is gitignored, so these are NOT in the repo and
#    MUST be generated on every fresh checkout before the test suite will pass.
#    This is offline — the CSVs are built from in-code priors, no network.
python -m pev_synth.sales_mix_data build-sales-mix

# 3. Run the linter and the test suite.
ruff check .
pytest
```

If you skip step 2, validator tests fail with a `FileNotFoundError` pointing you
back at the `build-sales-mix` command — that is expected, just run it.

### Optional: full data bootstrap

The sales-mix CSVs above are enough to run the test suite. If you want the
`Fleet` / `generate_profiles` library API to actually produce profiles in your
dev checkout, you additionally need the NHTS-derived cache, which is also not in
the repo. See the **First run / bootstrap** section of the [README](README.md)
for the one-time `nhts_loader` + `cache_regen` sequence.

## Tests

The pytest suite lives in `tests/`, with files named
`tests/test_pev_synth_*.py`. Add or update tests alongside any behavior change.

```bash
# Whole suite.
pytest

# A single file or test.
pytest tests/test_pev_synth_api.py
pytest tests/test_pev_synth_api.py::test_generate_profiles_basic

# With coverage, as CI runs it.
pytest --cov=pev_synth --cov-report=term-missing
```

Coverage is reported to **Codecov** from CI (a single Linux / Python 3.12 cell
uploads, so the matrix produces one clean report).

## Linting

We use [ruff](https://docs.astral.sh/ruff/) for linting and import sorting
(rule sets `E`, `F`, `I`, `B`, `UP`; line length 100 — both configured in
`pyproject.toml`). CI runs `ruff check .` and will fail on any finding.

```bash
ruff check .          # report
ruff check . --fix    # auto-fix what is safely fixable
```

## Pull request flow

1. **Branch** off `main` and make your change.
2. **Run `ruff check .` and `pytest` locally** until both are green.
3. Update the [CHANGELOG](CHANGELOG.md) under an *Unreleased* / next-version
   heading if your change is user-facing.
4. Update docs (`docs/`, `README.md`) if behavior, the API, or setup changed.
5. **Open a pull request** against `main`. The PR template's checklist will
   prompt you for the items above.
6. Wait for CI to go green and address review feedback.

### CI must pass

Every PR runs the **`ci`** workflow across the full matrix:

| | |
|---|---|
| **OS** | `ubuntu-latest`, `macos-latest`, `windows-latest` |
| **Python** | 3.10, 3.11, 3.12, 3.13 |

That is **12 cells**, each running `ruff check .`, the sales-mix build, and
`pytest`, plus a separate `build` job that runs `python -m build` and
`twine check dist/*`. The **`security`** workflow additionally runs
[`bandit`](https://bandit.readthedocs.io/) (static security lint, gated on
medium+ severity) and [`pip-audit`](https://pypi.org/project/pip-audit/)
(dependency vulnerability scan). All of these must pass before a PR can merge.

### Signed commits

The `main` branch is protected: it requires a pull request, requires all CI
checks to pass, and **requires every commit to be signed**. Configure commit
signing (GPG or SSH) before contributing — see GitHub's guide on
[signing commits](https://docs.github.com/authentication/managing-commit-signature-verification/signing-commits).
Unsigned commits will be rejected by branch protection.

## Versioning

ev-flow follows [Semantic Versioning](https://semver.org). `__version__` in
`src/pev_synth/__init__.py` and the `version` field in `pyproject.toml` are kept
in sync; a maintainer bumps them and cuts releases by pushing a `v*` tag (which
triggers the `publish` workflow's PyPI trusted-publishing flow). You do not need
to bump the version in a feature PR — note user-facing changes in the CHANGELOG
and a maintainer will handle the release.

## Reporting bugs and requesting features

Please use the issue forms under **New issue** (bug report / feature request).
For security vulnerabilities, do **not** open a public issue — follow
[SECURITY.md](SECURITY.md) instead.
