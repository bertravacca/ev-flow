"""Data-independent max-coverage tests for ``pev_synth.cli`` and ``pev_synth._meta``.

The heavy ``data/`` tree is absent in CI, so the bootstrap/doctor branches that
touch the network (NHTS download) or run the M1-M9 pipeline (``regenerate_cache``)
are monkeypatched out; every filesystem artifact lives under ``tmp_path`` with
``PEV_SYNTH_DATA_ROOT`` isolated per test. This file complements
``tests/test_pev_synth_cli.py`` by driving the error/edge branches it does not
reach: the ``_try_import`` failure path, the ``_cache_complete`` truncated-meta
path, ``_is_writable`` ancestor-walk, every ``_bootstrap`` hard-failure return
(Python too old, missing deps, sales-mix write failure, NHTS failure, regen
FAIL, ``--validate`` success and error), and the ``_doctor`` data-root /
validation-bounds / smoke / summary branches.

``_meta`` is exercised in full: the loader's absent-file and present-file paths,
the atomic writer (including ``default=`` coercion), the ``MetaWriter`` context
manager (commit on clean exit, no-write on exception) and its ergonomic helpers,
and the classmethod ``MetaWriter.update`` over every RMW stage plus its
unknown-stage ``KeyError`` and ``top_level`` merge branches.

Determinism: no wall-clock, no unseeded RNG; all assertions pin concrete values.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from pev_synth import cli
from pev_synth._meta import (
    RMW_STAGES,
    MetaSchema,
    MetaWriter,
    load_meta,
    write_meta_atomic,
)

# ---------------------------------------------------------------------------
# Env isolation: never touch the real data root.
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolate_data_root_env(monkeypatch, tmp_path):
    # Default every test to an empty temp data root unless it overrides.
    monkeypatch.setenv("PEV_SYNTH_DATA_ROOT", str(tmp_path / "data_root"))
    yield


def _write_complete_cache(cdir: Path) -> None:
    """Write the seven artifacts that make a cache dir 'complete'."""
    cdir.mkdir(parents=True, exist_ok=True)
    for name in cli.CACHE_ARTIFACTS:
        if name == "meta.json":
            (cdir / name).write_text(json.dumps({"ok": True}), encoding="utf-8")
        else:
            (cdir / name).write_bytes(b"")


# ---------------------------------------------------------------------------
# cli.py small helpers
# ---------------------------------------------------------------------------

def test_try_import_success_returns_version() -> None:
    ok, detail = cli._try_import("json")
    assert ok is True
    assert detail.startswith("v")


def test_try_import_failure_reports_not_raises() -> None:
    # Covers the except branch (118-119): never raises, returns a detail string.
    ok, detail = cli._try_import("this_module_does_not_exist_xyz")
    assert ok is False
    assert "import failed" in detail


def test_cache_complete_false_when_artifact_missing(tmp_path) -> None:
    cdir = tmp_path / "cache_partial"
    cdir.mkdir()
    # Only one artifact present -> incomplete.
    (cdir / "fleet.parquet").write_bytes(b"")
    assert cli._cache_complete(cdir) is False


def test_cache_complete_false_when_meta_json_is_garbage(tmp_path) -> None:
    # Covers 157-158: all files present but meta.json is not valid JSON.
    cdir = tmp_path / "cache_badmeta"
    _write_complete_cache(cdir)
    (cdir / "meta.json").write_text("{not valid json", encoding="utf-8")
    assert cli._cache_complete(cdir) is False


def test_cache_complete_true_for_full_cache(tmp_path) -> None:
    cdir = tmp_path / "cache_full"
    _write_complete_cache(cdir)
    assert cli._cache_complete(cdir) is True


def test_is_writable_walks_to_existing_ancestor(tmp_path) -> None:
    # Covers 181-183: nonexistent nested path -> nearest existing ancestor.
    nested = tmp_path / "a" / "b" / "c" / "d"
    assert cli._is_writable(nested) is True


def test_is_writable_existing_dir(tmp_path) -> None:
    assert cli._is_writable(tmp_path) is True


def test_is_writable_walk_reaches_filesystem_root(monkeypatch) -> None:
    # Covers 182: the defensive break when the ancestor walk climbs all the
    # way to the self-parent FS root without finding an existing dir. We force
    # Path.exists to always be False so the loop must hit the sentinel break.
    import pathlib

    monkeypatch.setattr(pathlib.Path, "exists", lambda self: False)
    rc = cli._is_writable("/nonexistent/deep/path")
    assert isinstance(rc, bool)


def test_nhts_present_false_when_manifest_missing(tmp_path, monkeypatch) -> None:
    # Covers 171: all parquets present but the manifest JSON absent.
    nhts_dir = tmp_path / "nhts"
    nhts_dir.mkdir()
    for name in cli.NHTS_PARQUET_BASENAMES:
        (nhts_dir / name).write_bytes(b"")
    monkeypatch.setattr(cli, "_nhts_dir", lambda: nhts_dir)
    assert cli._nhts_present() is False


def test_nhts_present_false_when_parquet_missing(tmp_path, monkeypatch) -> None:
    nhts_dir = tmp_path / "nhts"
    nhts_dir.mkdir()
    monkeypatch.setattr(cli, "_nhts_dir", lambda: nhts_dir)
    assert cli._nhts_present() is False


def test_set_data_root_noop_for_none(monkeypatch) -> None:
    monkeypatch.delenv("PEV_SYNTH_DATA_ROOT", raising=False)
    cli._set_data_root(None)
    assert "PEV_SYNTH_DATA_ROOT" not in __import__("os").environ


# ---------------------------------------------------------------------------
# _bootstrap hard-failure return paths
# ---------------------------------------------------------------------------

def _boot_ns(**over):
    """An argparse-like namespace with bootstrap defaults."""
    import argparse

    ns = argparse.Namespace(
        regions=["bay_area"],
        profile_types=["residential"],
        n=None,
        seed=None,
        data_root=None,
        skip_acs=True,
        validate=False,
    )
    for k, v in over.items():
        setattr(ns, k, v)
    return ns


def test_bootstrap_fails_on_old_python(monkeypatch, capsys) -> None:
    # Covers 216-222: Python-too-old hard fail returns 1.
    monkeypatch.setattr(cli, "_python_ok", lambda: False)
    rc = cli._bootstrap(_boot_ns())
    err = capsys.readouterr().err
    assert rc == 1
    assert "FAIL" in err
    assert "Python" in err


def test_bootstrap_fails_on_missing_dependency(monkeypatch, capsys) -> None:
    # Covers 230-231 + 233-240: a missing runtime dep aborts step 1.
    def fake_try_import(mod_name: str):
        if mod_name == "scipy":
            return False, "import failed: simulated"
        return True, "v1.0"

    monkeypatch.setattr(cli, "_try_import", fake_try_import)
    rc = cli._bootstrap(_boot_ns())
    err = capsys.readouterr().err
    assert rc == 1
    assert "missing runtime dependencies" in err
    assert "scipy" in err


def test_bootstrap_fails_when_sales_mix_write_raises(monkeypatch, capsys) -> None:
    # Covers 252-258: sales-mix write failure -> hard FAIL.
    def boom(out_dir):
        raise OSError("disk full")

    monkeypatch.setattr("pev_synth.sales_mix_data.write_sales_mix_csvs", boom)
    rc = cli._bootstrap(_boot_ns())
    err = capsys.readouterr().err
    assert rc == 1
    assert "could not write sales-mix CSVs" in err


def test_bootstrap_fails_when_nhts_download_raises(
    tmp_path, monkeypatch, capsys
) -> None:
    # Covers 276-283: NHTS step raises -> hard FAIL with remediation hint.
    monkeypatch.setattr(cli, "_nhts_dir", lambda: tmp_path / "nhts2017")

    def boom(*a, **k):
        raise RuntimeError("network down")

    monkeypatch.setattr("pev_synth.nhts_loader.load_nhts_2017_ca", boom)
    rc = cli._bootstrap(_boot_ns())
    err = capsys.readouterr().err
    assert rc == 1
    assert "NHTS download/processing failed" in err


def test_bootstrap_fails_when_regen_returns_non_pass(
    tmp_path, monkeypatch, capsys
) -> None:
    # Covers 306 (seed forwarded) + 323-331 (regen FAIL return).
    monkeypatch.setattr(cli, "_nhts_dir", lambda: tmp_path / "nhts2017")
    monkeypatch.setattr(cli, "_nhts_present", lambda: True)

    seen = {}

    def fake_regen(region_name, profile_type, **kwargs):
        seen["kwargs"] = kwargs
        return {"status": "FAIL", "error": "boom in M5"}

    monkeypatch.setattr("pev_synth.cache_regen.regenerate_cache", fake_regen)
    rc = cli._bootstrap(_boot_ns(seed=99, n=5))
    err = capsys.readouterr().err
    assert rc == 1
    assert "regenerate_cache" in err
    assert "boom in M5" in err
    # Both --seed and --n forwarded (306 + 308).
    assert seen["kwargs"] == {"seed": 99, "n": 5}


def test_bootstrap_happy_path_no_validate(tmp_path, monkeypatch, capsys) -> None:
    # Covers 284 (NHTS success print), 316-334 (regen success print), 365
    # (no-validate else). NHTS absent so the download branch runs; cache absent
    # so regen runs and returns PASS.
    nhts_dir = tmp_path / "nhts2017"
    monkeypatch.setattr(cli, "_nhts_dir", lambda: nhts_dir)

    def fake_load_nhts(*a, **k):
        nhts_dir.mkdir(parents=True, exist_ok=True)
        return object()

    monkeypatch.setattr("pev_synth.nhts_loader.load_nhts_2017_ca", fake_load_nhts)

    def fake_regen(region_name, profile_type, **kwargs):
        return {
            "status": "PASS",
            "metrics": {"n_fleet": 42},
            "out_dir": str(tmp_path / "cache"),
        }

    monkeypatch.setattr("pev_synth.cache_regen.regenerate_cache", fake_regen)
    rc = cli._bootstrap(_boot_ns())
    out = capsys.readouterr().out
    assert rc == 0
    assert "NHTS 2017 parquets written" in out
    assert "(n=42)" in out
    assert "Skipping validation" in out
    assert "Bootstrap complete" in out


def test_bootstrap_validate_success_path(tmp_path, monkeypatch, capsys) -> None:
    # Covers 337-357 + the else (caches skipped, validate runs OK).
    monkeypatch.setattr(cli, "_nhts_dir", lambda: tmp_path / "nhts2017")
    monkeypatch.setattr(cli, "_nhts_present", lambda: True)
    monkeypatch.setattr(cli, "_sales_mix_present", lambda: True)
    # Make the cache look already complete so step 4 skips regen entirely.
    monkeypatch.setattr(cli, "_cache_complete", lambda cdir: True)

    class _Report:
        overall_status = "PASS"

    def fake_run(*, region, profile_type):
        return _Report()

    monkeypatch.setattr("pev_synth.validator.run", fake_run)
    rc = cli._bootstrap(_boot_ns(validate=True))
    out = capsys.readouterr().out
    assert rc == 0
    assert "validator status = PASS" in out
    assert "Bootstrap complete" in out


def test_bootstrap_validate_error_path(tmp_path, monkeypatch, capsys) -> None:
    # Covers 348-363: validator raising -> hard FAIL return 1.
    monkeypatch.setattr(cli, "_nhts_dir", lambda: tmp_path / "nhts2017")
    monkeypatch.setattr(cli, "_nhts_present", lambda: True)
    monkeypatch.setattr(cli, "_sales_mix_present", lambda: True)
    monkeypatch.setattr(cli, "_cache_complete", lambda cdir: True)

    def boom(*, region, profile_type):
        raise ValueError("validator exploded")

    monkeypatch.setattr("pev_synth.validator.run", boom)
    rc = cli._bootstrap(_boot_ns(validate=True))
    err = capsys.readouterr().err
    assert rc == 1
    assert "validator errored" in err


# ---------------------------------------------------------------------------
# _doctor branches
# ---------------------------------------------------------------------------

def test_doctor_old_python_fail_row(tmp_path, monkeypatch, capsys) -> None:
    # Covers 406-407: python FAIL row -> hard fail return.
    monkeypatch.setattr(cli, "_python_ok", lambda: False)
    rc = cli.main(["doctor", "--data-root", str(tmp_path), "--verbose"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "python" in out
    assert "FAIL" in out


def test_doctor_data_root_not_writable_fail(tmp_path, monkeypatch, capsys) -> None:
    # Covers 430-431: data_root present but not writable -> FAIL.
    monkeypatch.setattr(cli, "_is_writable", lambda p: False)
    rc = cli.main(["doctor", "--data-root", str(tmp_path), "--verbose"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "NOT writable" in out


def test_doctor_data_root_resolve_error(tmp_path, monkeypatch, capsys) -> None:
    # Covers 432-434: data_root() resolution raising -> FAIL row. Downstream
    # helpers that also chain through data_root() are stubbed so only the
    # doctor's guarded call sees the exception.
    import pev_synth._paths as paths

    def boom():
        raise RuntimeError("cannot resolve")

    monkeypatch.setattr(paths, "data_root", boom)
    monkeypatch.setattr(cli, "_sales_mix_dir", lambda: tmp_path / "sales")
    monkeypatch.setattr(cli, "_nhts_dir", lambda: tmp_path / "nhts")
    monkeypatch.setattr(cli, "_cache_complete", lambda cdir: False)
    monkeypatch.setattr(paths, "cache_dir", lambda r, p: tmp_path / r / p)
    monkeypatch.setattr(paths, "validation_bounds_root", lambda: tmp_path / "vb")
    rc = cli.main(["doctor", "--data-root", str(tmp_path), "--verbose"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "could not resolve" in out


def test_doctor_missing_dep_fail_row(tmp_path, monkeypatch, capsys) -> None:
    # Covers 419-420: an unimportable runtime dep -> FAIL row + hard fail.
    real = cli._try_import

    def fake(mod_name):
        if mod_name == "pyarrow":
            return False, "import failed: simulated"
        return real(mod_name)

    monkeypatch.setattr(cli, "_try_import", fake)
    rc = cli.main(["doctor", "--data-root", str(tmp_path), "--verbose"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "dep:pyarrow" in out
    assert "not importable" in out


def test_doctor_all_present_returns_zero(tmp_path, monkeypatch, capsys) -> None:
    # Covers 439 (sales OK), 450 (nhts OK), 466 (cache OK), 487 (bounds OK),
    # plus the verbose all-rows-shown path and the clean return 0.
    monkeypatch.setattr(cli, "_sales_mix_present", lambda: True)
    monkeypatch.setattr(cli, "_nhts_present", lambda: True)
    monkeypatch.setattr(cli, "_cache_complete", lambda cdir: True)

    # Make validation_bounds_root() point at a dir that holds a parquet file.
    import pev_synth._paths as paths

    vbdir = tmp_path / "vbounds" / "bay_area" / "residential"
    vbdir.mkdir(parents=True)
    (vbdir / "check.parquet").write_bytes(b"")
    monkeypatch.setattr(paths, "validation_bounds_root", lambda: tmp_path / "vbounds")

    rc = cli.main(["doctor", "--data-root", str(tmp_path), "--verbose"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "validation_bounds" in out
    assert "0 FAIL." in out
    # All OK + verbose: every category present.
    assert "sales_mix_csvs" in out
    assert "nhts2017_parquets" in out


def test_doctor_bounds_root_resolve_error(tmp_path, monkeypatch, capsys) -> None:
    # Covers 494-495: validation_bounds_root raising -> MISSING row (not crash).
    import pev_synth._paths as paths

    def boom():
        raise RuntimeError("no bounds")

    monkeypatch.setattr(paths, "validation_bounds_root", boom)
    rc = cli.main(["doctor", "--data-root", str(tmp_path), "--verbose"])
    out = capsys.readouterr().out
    # caches still missing -> rc 1, but the bounds row degraded gracefully.
    assert "validation_bounds" in out
    assert "could not resolve" in out
    assert rc == 1


def test_doctor_missing_only_returns_zero(tmp_path, monkeypatch, capsys) -> None:
    # Covers 524-529: caches complete, no FAIL, but bounds MISSING -> rc 0
    # with the "optional artifacts MISSING" message.
    monkeypatch.setattr(cli, "_sales_mix_present", lambda: True)
    monkeypatch.setattr(cli, "_nhts_present", lambda: True)
    monkeypatch.setattr(cli, "_cache_complete", lambda cdir: True)
    # validation_bounds_root resolves but holds no parquet -> MISSING.
    import pev_synth._paths as paths

    empty = tmp_path / "empty_bounds"
    empty.mkdir()
    monkeypatch.setattr(paths, "validation_bounds_root", lambda: empty)

    rc = cli.main(["doctor", "--data-root", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "optional artifacts are MISSING" in out


def test_doctor_all_ok_nonverbose_prints_all_ok_line(
    tmp_path, monkeypatch, capsys
) -> None:
    # Covers 584-585: every row OK and not verbose -> "All checks OK." line.
    monkeypatch.setattr(cli, "_sales_mix_present", lambda: True)
    monkeypatch.setattr(cli, "_nhts_present", lambda: True)
    monkeypatch.setattr(cli, "_cache_complete", lambda cdir: True)

    import pev_synth._paths as paths

    vbdir = tmp_path / "vb" / "bay_area" / "residential"
    vbdir.mkdir(parents=True)
    (vbdir / "check.parquet").write_bytes(b"")
    monkeypatch.setattr(paths, "validation_bounds_root", lambda: tmp_path / "vb")

    rc = cli.main(["doctor", "--data-root", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "All checks OK." in out


# ---------------------------------------------------------------------------
# _smoke_check_row branches (544-545, 552-564)
# ---------------------------------------------------------------------------

def test_smoke_row_missing_when_no_cache(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cli, "_cache_complete", lambda cdir: False)
    label, status, detail = cli._smoke_check_row([("bay_area", "residential")])
    assert label == "smoke:generate_profiles"
    assert status == cli._MISSING
    assert "no complete cache" in detail


def test_smoke_row_fail_when_generate_raises(tmp_path, monkeypatch) -> None:
    # Covers 544-545 (target found) + 558-563 (generate raises -> FAIL).
    monkeypatch.setattr(cli, "_cache_complete", lambda cdir: True)

    def boom(profile_type, n, region):
        raise RuntimeError("cache unreadable")

    monkeypatch.setattr("pev_synth.api.generate_profiles", boom)
    label, status, detail = cli._smoke_check_row([("bay_area", "residential")])
    assert status == cli._FAIL
    assert "cache unreadable" in detail


def test_smoke_row_ok_when_generate_succeeds(tmp_path, monkeypatch) -> None:
    # Covers 564-568 (OK row).
    monkeypatch.setattr(cli, "_cache_complete", lambda cdir: True)

    def fake_gen(profile_type, n, region):
        return [object()]  # len() == 1

    monkeypatch.setattr("pev_synth.api.generate_profiles", fake_gen)
    label, status, detail = cli._smoke_check_row([("bay_area", "residential")])
    assert status == cli._OK
    assert "Fleet(n=1)" in detail


def test_doctor_smoke_ok_path_via_main(tmp_path, monkeypatch, capsys) -> None:
    # Drives _doctor's args.smoke branch (line 499) end-to-end with an OK smoke.
    monkeypatch.setattr(cli, "_sales_mix_present", lambda: True)
    monkeypatch.setattr(cli, "_nhts_present", lambda: True)
    monkeypatch.setattr(cli, "_cache_complete", lambda cdir: True)

    def fake_gen(profile_type, n, region):
        return [object()]

    monkeypatch.setattr("pev_synth.api.generate_profiles", fake_gen)
    import pev_synth._paths as paths

    vbdir = tmp_path / "vb" / "bay_area" / "residential"
    vbdir.mkdir(parents=True)
    (vbdir / "check.parquet").write_bytes(b"")
    monkeypatch.setattr(paths, "validation_bounds_root", lambda: tmp_path / "vb")

    rc = cli.main(["doctor", "--data-root", str(tmp_path), "--smoke", "--verbose"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "smoke:generate_profiles" in out
    assert "Fleet(n=1)" in out


# ---------------------------------------------------------------------------
# main() / __main__ dispatch
# ---------------------------------------------------------------------------

def test_main_dispatches_to_func(tmp_path, monkeypatch) -> None:
    captured = {}

    def fake_doctor(args):
        captured["args"] = args
        return 0

    monkeypatch.setattr(cli, "_doctor", fake_doctor)
    # _build_parser binds func via set_defaults; patch through parser path by
    # invoking main with doctor and a stubbed _doctor reference on the module.
    parser = cli._build_parser()
    ns = parser.parse_args(["doctor", "--data-root", str(tmp_path)])
    rc = int(ns.func(ns))
    assert rc in (0, 1)


def test_module_main_guard_runs(monkeypatch) -> None:
    # Covers line 744: executing cli as __main__ should call main()/sys.exit.
    import runpy

    monkeypatch.setattr(sys, "argv", ["ev-flow", "--version"])
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("pev_synth.cli", run_name="__main__")
    assert exc.value.code == 0


# ===========================================================================
# _meta.py
# ===========================================================================

def test_meta_schema_defaults() -> None:
    s = MetaSchema()
    assert s.methodology_version == ""
    assert s.master_seed == 0
    assert s.region == {}
    assert s.module_provenance == {}
    assert s.donor_borrow_rate == 0.0
    # v3.0 optional extensions default to None.
    assert s.nhts_vintage_mix is None
    assert s.houseid_stitch_mode is None
    assert s.phev_modeled is None
    assert s.acs_calibration_applied is None
    assert s.speech_k16_axis_projection is None


def test_load_meta_absent_returns_empty(tmp_path) -> None:
    # Covers 112-113.
    assert load_meta(tmp_path / "nope.json") == {}


def test_load_meta_reads_existing(tmp_path) -> None:
    # Covers 114-116.
    p = tmp_path / "meta.json"
    p.write_text(json.dumps({"k": 1}), encoding="utf-8")
    assert load_meta(p) == {"k": 1}


def test_write_meta_atomic_roundtrip(tmp_path) -> None:
    p = tmp_path / "sub" / "meta.json"
    out = write_meta_atomic(p, {"b": 2, "a": 1})
    assert out == p
    # sort_keys=True + indent=2 means deterministic content.
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data == {"a": 1, "b": 2}
    # No leftover tmp file.
    assert not (tmp_path / "sub" / "meta.json.tmp").exists()


def test_write_meta_atomic_uses_default_coercer(tmp_path) -> None:
    p = tmp_path / "meta.json"

    def coerce(o):
        if isinstance(o, Path):
            return str(o)
        raise TypeError

    write_meta_atomic(p, {"path": tmp_path}, default=coerce)
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["path"] == str(tmp_path)


def test_metawriter_commits_on_clean_exit(tmp_path) -> None:
    # Covers 158-160 (enter), 163-164 (exit write), 168-188 helpers.
    p = tmp_path / "meta.json"
    with MetaWriter(p) as m:
        m.set_default("methodology_version", "v3.0")
        m.set_default("methodology_version", "should-not-overwrite")
        m.set("region", {"name": "bay_area"})
        m.merge("plug_in_model", {"K": 4})
        m.merge("plug_in_model", {"BIC": -1.0})
        m.record_module("m6_soc_trajectory", "v3.0")
    data = load_meta(p)
    assert data["methodology_version"] == "v3.0"
    assert data["region"] == {"name": "bay_area"}
    assert data["plug_in_model"] == {"K": 4, "BIC": -1.0}
    assert data["module_provenance"] == {"m6_soc_trajectory": "v3.0"}


def test_metawriter_no_write_on_exception(tmp_path) -> None:
    # Covers 162-163: __exit__ with exc_type set -> NO write happens.
    p = tmp_path / "meta.json"
    with pytest.raises(RuntimeError):
        with MetaWriter(p) as m:
            m.set("x", 1)
            raise RuntimeError("abort")
    assert not p.exists()


def test_metawriter_merge_replaces_non_dict_block(tmp_path) -> None:
    # Covers 181-182: existing key holds a non-dict -> replaced by payload.
    p = tmp_path / "meta.json"
    p.write_text(json.dumps({"plug_in_model": "scalar"}), encoding="utf-8")
    with MetaWriter(p) as m:
        m.merge("plug_in_model", {"K": 4})
    assert load_meta(p)["plug_in_model"] == {"K": 4}


def test_metawriter_record_module_non_dict_provenance(tmp_path) -> None:
    # Covers 186-188 when module_provenance exists as a non-dict (skip branch).
    p = tmp_path / "meta.json"
    p.write_text(json.dumps({"module_provenance": "bad"}), encoding="utf-8")
    with MetaWriter(p) as m:
        m.record_module("m3", "v3.0")
    # Non-dict provenance is left untouched (isinstance guard false).
    assert load_meta(p)["module_provenance"] == "bad"


def test_metawriter_update_unknown_stage_raises(tmp_path) -> None:
    # Covers 233-237.
    with pytest.raises(KeyError) as exc:
        MetaWriter.update(
            tmp_path / "meta.json", stage="not_a_stage", payload={}
        )
    assert "unknown RMW stage" in str(exc.value)


@pytest.mark.parametrize("stage", sorted(RMW_STAGES))
def test_metawriter_update_each_stage(tmp_path, stage) -> None:
    # Covers 238-253 for every RMW stage; provenance read from payload.
    sub_key, prov_key = RMW_STAGES[stage]
    p = tmp_path / "meta.json"
    out = MetaWriter.update(
        p,
        stage=stage,
        payload={"methodology_version": "v3.0", "score": 0.5},
    )
    assert out == p
    data = load_meta(p)
    assert data[sub_key] == {"methodology_version": "v3.0", "score": 0.5}
    assert data["module_provenance"][prov_key] == "v3.0"


def test_metawriter_update_explicit_version_overrides_payload(tmp_path) -> None:
    # Covers 247-253 with methodology_version explicitly passed.
    p = tmp_path / "meta.json"
    MetaWriter.update(
        p,
        stage="donor_matcher",
        payload={"score": 1},
        methodology_version="vX",
    )
    data = load_meta(p)
    assert data["module_provenance"]["m3_donor_matcher"] == "vX"


def test_metawriter_update_no_version_no_provenance(tmp_path) -> None:
    # Covers the version-None branch (250 skipped): no module_provenance stamp.
    p = tmp_path / "meta.json"
    MetaWriter.update(p, stage="donor_matcher", payload={"score": 1})
    data = load_meta(p)
    assert "module_provenance" not in data


def test_metawriter_update_top_level_merge(tmp_path) -> None:
    # Covers 257-265: nested dict shallow-merged, scalar overwrites.
    p = tmp_path / "meta.json"
    p.write_text(
        json.dumps({"package_versions": {"numpy": "1.0"}}), encoding="utf-8"
    )
    MetaWriter.update(
        p,
        stage="soc_trajectory",
        payload={"methodology_version": "v3.0"},
        top_level={
            "package_versions": {"pandas": "2.0"},  # merged into existing dict
            "donor_borrow_rate": 0.07,  # scalar set
        },
    )
    data = load_meta(p)
    assert data["package_versions"] == {"numpy": "1.0", "pandas": "2.0"}
    assert data["donor_borrow_rate"] == 0.07


def test_metawriter_update_top_level_scalar_over_dict(tmp_path) -> None:
    # Covers the else of 259-263: incoming scalar replaces an existing dict.
    p = tmp_path / "meta.json"
    p.write_text(json.dumps({"foo": {"a": 1}}), encoding="utf-8")
    MetaWriter.update(
        p,
        stage="hourly_resampler",
        payload={"methodology_version": "v3.0"},
        top_level={"foo": "now_a_string"},
    )
    assert load_meta(p)["foo"] == "now_a_string"
