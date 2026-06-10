"""Tests for the ``ev-flow`` CLI (``pev_synth.cli``): bootstrap (F6) + doctor (F8).

These tests are **network-free and self-contained**. The only step of
``bootstrap`` that would hit the network (NHTS download) and the only step that
is heavy (``regenerate_cache``) are monkeypatched out; the genuinely offline
step (sales-mix CSV generation) is exercised for real against a ``tmp_path``
data root. ``doctor`` is exercised against an empty temp data root and must
report MISSING artifacts + return nonzero WITHOUT importing-time crashes or
network access.

Mirrors the style of ``tests/test_pev_synth_cache_regen.py`` (direct calls into
the module under test; argparse smoke via the parser builder).
"""

from __future__ import annotations

import json

import pytest

from pev_synth import __version__, cli
from pev_synth.sales_mix_data import write_sales_mix_csvs

# ---------------------------------------------------------------------------
# Env isolation: snapshot/restore PEV_SYNTH_DATA_ROOT around every test so that
# CLIs invoked with --data-root (which sets the env var) cannot leak across
# tests.
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolate_data_root_env(monkeypatch):
    monkeypatch.delenv("PEV_SYNTH_DATA_ROOT", raising=False)
    yield


# ---------------------------------------------------------------------------
# --version / --help / parser construction
# ---------------------------------------------------------------------------

def test_version_flag_prints_version_and_exits(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert __version__ in out
    assert "ev-flow" in out


def test_help_flag_exits_zero(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "bootstrap" in out
    assert "doctor" in out


def test_no_subcommand_errors() -> None:
    # required=True subparsers -> argparse exits nonzero when none is given.
    with pytest.raises(SystemExit) as exc:
        cli.main([])
    assert exc.value.code != 0


def test_bootstrap_subcommand_help_exits_zero(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["bootstrap", "--help"])
    assert exc.value.code == 0
    assert "bootstrap" in capsys.readouterr().out


def test_doctor_subcommand_help_exits_zero(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["doctor", "--help"])
    assert exc.value.code == 0
    assert "diagnostics" in capsys.readouterr().out.lower()


# ---------------------------------------------------------------------------
# argparse defaults / parsing for both subcommands
# ---------------------------------------------------------------------------

def test_bootstrap_arg_defaults() -> None:
    parser = cli._build_parser()
    ns = parser.parse_args(["bootstrap"])
    assert ns.cmd == "bootstrap"
    assert ns.func is cli._bootstrap
    assert ns.regions == ["bay_area"]
    assert ns.profile_types == ["residential"]
    assert ns.n is None
    assert ns.seed is None
    assert ns.data_root is None
    assert ns.skip_acs is True  # default True — ACS needs a CENSUS_API_KEY
    assert ns.validate is False


def test_bootstrap_arg_overrides() -> None:
    parser = cli._build_parser()
    ns = parser.parse_args([
        "bootstrap",
        "--regions", "bay_area", "la_basin",
        "--profile-types", "residential", "workplace",
        "--n", "25",
        "--seed", "7",
        "--data-root", "/tmp/whatever",
        "--no-skip-acs",
        "--validate",
    ])
    assert ns.regions == ["bay_area", "la_basin"]
    assert ns.profile_types == ["residential", "workplace"]
    assert ns.n == 25
    assert ns.seed == 7
    assert ns.data_root == "/tmp/whatever"
    assert ns.skip_acs is False
    assert ns.validate is True


def test_doctor_arg_defaults() -> None:
    parser = cli._build_parser()
    ns = parser.parse_args(["doctor"])
    assert ns.cmd == "doctor"
    assert ns.func is cli._doctor
    assert ns.regions == ["bay_area"]
    assert ns.profile_types == ["residential"]
    assert ns.smoke is False
    assert ns.verbose is False
    assert ns.data_root is None


def test_doctor_arg_overrides() -> None:
    parser = cli._build_parser()
    ns = parser.parse_args([
        "doctor",
        "--regions", "chicago",
        "--profile-types", "workplace",
        "--data-root", "/tmp/x",
        "--smoke",
        "--verbose",
    ])
    assert ns.regions == ["chicago"]
    assert ns.profile_types == ["workplace"]
    assert ns.data_root == "/tmp/x"
    assert ns.smoke is True
    assert ns.verbose is True


# ---------------------------------------------------------------------------
# doctor against a fresh / empty data root (no network, no crash, nonzero exit)
# ---------------------------------------------------------------------------

def test_doctor_empty_data_root_reports_missing_and_returns_nonzero(
    tmp_path, capsys
) -> None:
    rc = cli.main(["doctor", "--data-root", str(tmp_path)])
    out = capsys.readouterr().out
    # The requested cache must be reported MISSING ...
    assert "cache:bay_area/residential" in out
    assert "MISSING" in out
    # ... and the empty environment is not usable -> nonzero exit.
    assert rc == 1


def test_doctor_empty_data_root_via_monkeypatched_env(
    tmp_path, monkeypatch, capsys
) -> None:
    # Same as above but exercising the PEV_SYNTH_DATA_ROOT path (no --data-root).
    monkeypatch.setenv("PEV_SYNTH_DATA_ROOT", str(tmp_path))
    rc = cli.main(["doctor"])
    out = capsys.readouterr().out
    assert "MISSING" in out
    assert rc == 1


def test_doctor_reports_runtime_deps_ok(tmp_path, capsys) -> None:
    # numpy/pandas/etc. are installed in the test env -> reported OK, no FAIL.
    cli.main(["doctor", "--data-root", str(tmp_path), "--verbose"])
    out = capsys.readouterr().out
    assert "dep:numpy" in out
    assert "dep:pandas" in out
    # No hard FAIL rows on a healthy interpreter (only MISSING data artifacts):
    # the summary line reports zero FAILs.
    assert "0 FAIL." in out


def test_doctor_smoke_without_cache_does_not_crash(tmp_path, capsys) -> None:
    # --smoke with no cache present must degrade gracefully (MISSING, not crash).
    rc = cli.main(["doctor", "--data-root", str(tmp_path), "--smoke"])
    out = capsys.readouterr().out
    assert "smoke:generate_profiles" in out
    assert rc == 1  # caches still missing


def test_doctor_reports_missing_dependency_without_crashing(
    tmp_path, monkeypatch, capsys
) -> None:
    """A missing runtime dep must surface as a FAIL row, not an ImportError."""
    real_try_import = cli._try_import

    def fake_try_import(mod_name: str):
        if mod_name == "scipy":
            return False, "import failed: simulated absence"
        return real_try_import(mod_name)

    monkeypatch.setattr(cli, "_try_import", fake_try_import)
    rc = cli.main(["doctor", "--data-root", str(tmp_path), "--verbose"])
    out = capsys.readouterr().out
    assert "dep:scipy" in out
    assert "FAIL" in out
    assert rc == 1


# ---------------------------------------------------------------------------
# bootstrap: offline prerequisite (sales-mix CSVs) is real; network is faked.
# ---------------------------------------------------------------------------

def test_bootstrap_offline_sales_mix_generation(tmp_path) -> None:
    """The genuinely-offline bootstrap step writes the three sales-mix CSVs.

    Calls the same underlying function bootstrap calls, targeted at the cli's
    resolved sales-mix dir under a temp data root. No network.
    """
    import os

    os.environ["PEV_SYNTH_DATA_ROOT"] = str(tmp_path)
    try:
        sales_dir = cli._sales_mix_dir()
        assert not cli._sales_mix_present()
        written = write_sales_mix_csvs(sales_dir)
    finally:
        os.environ.pop("PEV_SYNTH_DATA_ROOT", None)

    assert set(written) == set(cli.SALES_MIX_STEMS)
    for stem in cli.SALES_MIX_STEMS:
        assert (sales_dir / f"{stem}.csv").is_file()


def test_bootstrap_full_run_offline_steps_only(
    tmp_path, monkeypatch, capsys
) -> None:
    """End-to-end ``_bootstrap`` with the NHTS download and the heavy cache
    build monkeypatched, so the run is offline and fast.

    Verifies: sales-mix CSVs really get written; the NHTS + cache steps are
    invoked through our stubs (never the network); the run returns 0.
    """
    monkeypatch.setenv("PEV_SYNTH_DATA_ROOT", str(tmp_path))

    calls: dict[str, object] = {}

    # Stub the NHTS network download: pretend it succeeded.
    def fake_load_nhts(*args, **kwargs):
        calls["nhts"] = kwargs.get("cache_dir", args[0] if args else None)
        return object()

    monkeypatch.setattr(
        "pev_synth.nhts_loader.load_nhts_2017_ca", fake_load_nhts
    )

    # Stub the heavy pipeline: return a PASS result dict like regenerate_cache.
    def fake_regen(region_name, profile_type, **kwargs):
        calls.setdefault("regen", []).append((region_name, profile_type))
        return {
            "region": region_name,
            "profile_type": profile_type,
            "status": "PASS",
            "metrics": {"n_fleet": 10},
            "out_dir": str(tmp_path),
        }

    monkeypatch.setattr("pev_synth.cache_regen.regenerate_cache", fake_regen)

    rc = cli.main(["bootstrap", "--data-root", str(tmp_path), "--n", "10"])
    out = capsys.readouterr().out

    assert rc == 0
    # The offline sales-mix CSVs were really written under the temp root.
    sales_dir = tmp_path / "pev" / "raw" / "literature" / "sales_mix"
    for stem in cli.SALES_MIX_STEMS:
        assert (sales_dir / f"{stem}.csv").is_file()
    # The (faked) NHTS and cache steps were invoked.
    assert "nhts" in calls
    assert calls.get("regen") == [("bay_area", "residential")]
    assert "Bootstrap complete" in out


def test_bootstrap_idempotent_skips_existing_outputs(
    tmp_path, monkeypatch, capsys
) -> None:
    """A second bootstrap run skips the offline sales-mix step it already did,
    skips the (faked) NHTS fetch, and skips a cache dir that is already
    complete — invoking neither the network nor the heavy pipeline."""
    monkeypatch.setenv("PEV_SYNTH_DATA_ROOT", str(tmp_path))

    # Anchor the NHTS dir under tmp_path for this test so we NEVER touch the
    # real repo-rooted data/pev/raw/nhts2017 tree.
    nhts_dir = tmp_path / "nhts2017"
    monkeypatch.setattr(cli, "_nhts_dir", lambda: nhts_dir)

    # Pre-create the sales-mix CSVs so step 2 should SKIP.
    write_sales_mix_csvs(cli._sales_mix_dir())
    assert cli._sales_mix_present()

    # Pre-create a "complete" bay_area/residential cache dir so step 4 SKIPs it.
    from pev_synth._paths import cache_dir as _cache_dir

    cdir = _cache_dir("bay_area", "residential")
    cdir.mkdir(parents=True, exist_ok=True)
    for name in cli.CACHE_ARTIFACTS:
        if name == "meta.json":
            (cdir / name).write_text(json.dumps({"ok": True}), encoding="utf-8")
        else:
            (cdir / name).write_bytes(b"")  # presence is all _cache_complete checks

    # NHTS pretend-present so step 3 SKIPs too (write the expected files).
    nhts_dir.mkdir(parents=True, exist_ok=True)
    for name in cli.NHTS_PARQUET_BASENAMES:
        (nhts_dir / name).write_bytes(b"")
    (nhts_dir / cli.NHTS_MANIFEST_BASENAME).write_text("{}", encoding="utf-8")

    # If either the network or the heavy step were called, fail loudly.
    def boom_nhts(*a, **k):  # pragma: no cover - must not be called
        raise AssertionError("NHTS step should have been skipped")

    def boom_regen(*a, **k):  # pragma: no cover - must not be called
        raise AssertionError("cache regen should have been skipped")

    monkeypatch.setattr("pev_synth.nhts_loader.load_nhts_2017_ca", boom_nhts)
    monkeypatch.setattr("pev_synth.cache_regen.regenerate_cache", boom_regen)

    rc = cli.main(["bootstrap", "--data-root", str(tmp_path)])

    out = capsys.readouterr().out
    assert rc == 0
    assert "SKIP" in out
    # The sales-mix CSVs under tmp_path are still present after the run.
    assert cli._sales_mix_present()
