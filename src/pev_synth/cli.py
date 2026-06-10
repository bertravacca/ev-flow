"""``ev-flow`` command-line interface: ``bootstrap`` (F6) and ``doctor`` (F8).

This module is the console-script entry point declared in ``pyproject.toml``
as ``ev-flow = "pev_synth.cli:main"``. It deliberately keeps its *module-level*
imports to the standard library plus ``pev_synth.__version__`` only.

Why lazy imports
----------------
``ev-flow doctor`` must be able to run on a half-installed / fresh environment
and *report* a missing third-party dependency (numpy, pandas, pyarrow,
scikit-learn, scipy, requests, …) as a row in its diagnostics table rather
than crashing on import. If this module imported those packages — or the
``pev_synth`` pipeline modules that import them — at the top level, a missing
optional dependency would raise ``ImportError`` before ``main`` ever runs, and
the user would get a stack trace instead of a clean ``MISSING`` line telling
them what to install. Every heavy/optional import therefore happens *inside*
the command / check functions.

Both subcommands are wired through :func:`main`, which returns an ``int`` exit
code (0 = success / no hard failure, 1 = first hard failure). ``ev-flow``,
``python -m pev_synth.cli`` and a direct ``main([...])`` call all behave
identically.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys

from . import __version__

# --------------------------------------------------------------------------- #
# Module-level constants (stdlib-only; no third-party imports up here).
# --------------------------------------------------------------------------- #

#: Minimum supported Python (matches ``requires-python`` in pyproject.toml).
MIN_PYTHON: tuple[int, int] = (3, 10)

#: Runtime third-party dependencies (import name -> pip/distribution name).
#: Mirrors the ``[project].dependencies`` table in pyproject.toml. Used by both
#: ``bootstrap`` (step 1 hard-check) and ``doctor`` (one row each).
RUNTIME_DEPS: tuple[tuple[str, str], ...] = (
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("pyarrow", "pyarrow"),
    ("requests", "requests"),
    ("sklearn", "scikit-learn"),
    ("scipy", "scipy"),
    ("pytz", "pytz"),
)

#: The six parquet artifacts + meta.json that make a cache directory "complete".
#: Matches what ``pev_synth.cache_regen.regenerate_cache`` writes.
CACHE_ARTIFACTS: tuple[str, ...] = (
    "fleet.parquet",
    "mobility.parquet",
    "plug_in_events.parquet",
    "sessions.parquet",
    "plug_status_15min.parquet",
    "plug_status_hourly.parquet",
    "meta.json",
)

#: The three offline sales-mix CSVs (stems) written by
#: ``pev_synth.sales_mix_data.write_sales_mix_csvs``.
SALES_MIX_STEMS: tuple[str, ...] = (
    "cvrp_ca",
    "nyserda_drive_clean",
    "argonne_national",
)

#: NHTS 2017 CA parquet basenames + manifest expected on disk after the
#: one-time loader run. Kept in sync with ``pev_synth.nhts_loader``.
NHTS_PARQUET_BASENAMES: tuple[str, ...] = (
    "nhts2017_ca_hh.parquet",
    "nhts2017_ca_per.parquet",
    "nhts2017_ca_veh.parquet",
    "nhts2017_ca_trip.parquet",
)
NHTS_MANIFEST_BASENAME: str = "nhts2017_download_manifest.json"

# Status tokens for the doctor table.
_OK = "OK"
_MISSING = "MISSING"
_FAIL = "FAIL"
_WARN = "WARN"


# --------------------------------------------------------------------------- #
# Small stdlib-only helpers.
# --------------------------------------------------------------------------- #


def _python_version_str() -> str:
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def _python_ok() -> bool:
    return sys.version_info[:2] >= MIN_PYTHON


def _try_import(mod_name: str) -> tuple[bool, str]:
    """Attempt to import ``mod_name``; return ``(ok, detail)``.

    ``detail`` is the version string on success or the import error text on
    failure. Never raises.
    """
    try:
        mod = importlib.import_module(mod_name)
    except Exception as exc:  # noqa: BLE001 — we want to report, not crash.
        return False, f"import failed: {exc}"
    ver = getattr(mod, "__version__", "?")
    return True, f"v{ver}"


def _sales_mix_dir():  # -> Path
    """Directory holding the three sales-mix CSVs (honours PEV_SYNTH_DATA_ROOT)."""
    from ._paths import raw_root

    return raw_root() / "literature" / "sales_mix"


def _nhts_dir():  # -> Path
    """Directory the NHTS 2017 CA parquets live in.

    Anchored at ``repo_root()/data/pev/raw/nhts2017`` to match exactly where
    ``pev_synth.cache_regen`` reads the NHTS frames from. NOTE: this is the
    repo-root data tree, which is independent of ``PEV_SYNTH_DATA_ROOT`` — the
    same convention the pipeline itself uses.
    """
    from ._paths import repo_root

    return repo_root() / "data" / "pev" / "raw" / "nhts2017"


def _cache_complete(cache_dir) -> bool:
    """True iff ``cache_dir`` holds every artifact in :data:`CACHE_ARTIFACTS`
    and ``meta.json`` parses as JSON."""
    import json

    for name in CACHE_ARTIFACTS:
        p = cache_dir / name
        if not p.is_file():
            return False
    # meta.json must be valid JSON, not a truncated/half-written file.
    try:
        with (cache_dir / "meta.json").open(encoding="utf-8") as f:
            json.load(f)
    except (OSError, ValueError):
        return False
    return True


def _sales_mix_present() -> bool:
    d = _sales_mix_dir()
    return all((d / f"{stem}.csv").is_file() for stem in SALES_MIX_STEMS)


def _nhts_present() -> bool:
    d = _nhts_dir()
    if not all((d / name).is_file() for name in NHTS_PARQUET_BASENAMES):
        return False
    return (d / NHTS_MANIFEST_BASENAME).is_file()


def _is_writable(path) -> bool:
    """Whether ``path`` (an existing dir, or its nearest existing ancestor) is
    writable. Used by doctor's data-root check without mutating anything."""
    from pathlib import Path

    p = Path(path)
    while not p.exists():
        if p.parent == p:  # reached filesystem root
            break
        p = p.parent
    return os.access(p, os.W_OK)


def _set_data_root(data_root: str | None) -> None:
    """Set ``PEV_SYNTH_DATA_ROOT`` so all ``_paths`` helpers resolve there.

    The path registry reads this env var lazily on every call, so setting it
    before invoking any pipeline function reroutes the whole data tree.
    """
    if data_root:
        os.environ["PEV_SYNTH_DATA_ROOT"] = str(data_root)


# --------------------------------------------------------------------------- #
# bootstrap (F6)
# --------------------------------------------------------------------------- #


def _bootstrap(args: argparse.Namespace) -> int:
    """One-time environment bootstrap: deps -> sales-mix -> NHTS -> caches.

    Idempotent: every step skips work whose outputs already exist. Returns 0
    on success, 1 on the first hard failure (with a remediation hint printed).
    """
    _set_data_root(args.data_root)

    print("ev-flow bootstrap")
    print("=================")

    # --- Step 1: Python + runtime imports ------------------------------- #
    print("[1/5] Checking Python and runtime dependencies ...")
    if not _python_ok():
        print(
            f"  FAIL: Python {_python_version_str()} < "
            f"{MIN_PYTHON[0]}.{MIN_PYTHON[1]} required.\n"
            f"        Install Python >= {MIN_PYTHON[0]}.{MIN_PYTHON[1]} and retry.",
            file=sys.stderr,
        )
        return 1
    print(f"  OK: Python {_python_version_str()}")
    missing: list[str] = []
    for mod_name, dist_name in RUNTIME_DEPS:
        ok, detail = _try_import(mod_name)
        if ok:
            print(f"  OK: {dist_name} {detail}")
        else:
            missing.append(dist_name)
            print(f"  MISSING: {dist_name} ({detail})", file=sys.stderr)
    if missing:
        print(
            "  FAIL: missing runtime dependencies: "
            f"{', '.join(missing)}.\n"
            f"        Install them, e.g. `pip install {' '.join(missing)}` "
            "(or `pip install ev-flow`).",
            file=sys.stderr,
        )
        return 1

    # --- Step 2: offline sales-mix CSVs --------------------------------- #
    print("[2/5] Building offline sales-mix CSVs ...")
    sales_dir = _sales_mix_dir()
    if _sales_mix_present():
        print(f"  SKIP: sales-mix CSVs already present in {sales_dir}")
    else:
        try:
            from .sales_mix_data import write_sales_mix_csvs

            written = write_sales_mix_csvs(sales_dir)
        except Exception as exc:  # noqa: BLE001
            print(
                f"  FAIL: could not write sales-mix CSVs: {exc}\n"
                f"        Check that {sales_dir} is writable.",
                file=sys.stderr,
            )
            return 1
        for name, path in written.items():
            print(f"  wrote {name} -> {path}")

    # --- Step 3: NHTS 2017 CA microdata (network, idempotent) ----------- #
    print("[3/5] Fetching NHTS 2017 microdata (only if not already present) ...")
    nhts_dir = _nhts_dir()
    if _nhts_present():
        print(f"  SKIP: NHTS 2017 parquets already present in {nhts_dir}")
    else:
        print(
            f"  Downloading ~84 MB ORNL zip and processing into {nhts_dir} "
            "(this can take a few minutes) ..."
        )
        try:
            from .nhts_loader import load_nhts_2017_ca

            load_nhts_2017_ca(cache_dir=nhts_dir)
        except Exception as exc:  # noqa: BLE001
            print(
                f"  FAIL: NHTS download/processing failed: {exc}\n"
                "        Check your network connection and that "
                f"{nhts_dir} is writable, then re-run `ev-flow bootstrap`.",
                file=sys.stderr,
            )
            return 1
        print(f"  OK: NHTS 2017 parquets written to {nhts_dir}")

    # --- Step 4: per-(region, profile_type) caches ---------------------- #
    pairs = [(r, p) for r in args.regions for p in args.profile_types]
    print(
        f"[4/5] Building {len(pairs)} cache(s): "
        f"{', '.join(f'{r}/{p}' for r, p in pairs)} ..."
    )
    from ._paths import cache_dir as _cache_dir
    from .cache_regen import regenerate_cache

    for region_name, profile_type in pairs:
        cdir = _cache_dir(region_name, profile_type)
        if _cache_complete(cdir):
            print(f"  SKIP: {region_name}/{profile_type} cache already complete "
                  f"({cdir})")
            continue
        print(f"  Generating {region_name}/{profile_type} ...")
        res = regenerate_cache(
            region_name,
            profile_type,
            n=args.n,
            seed=args.seed,
        )
        if res.get("status") != "PASS":
            err = res.get("error", "unknown error")
            print(
                f"  FAIL: regenerate_cache({region_name!r}, {profile_type!r}) "
                f"-> {res.get('status')}: {err}\n"
                "        See the traceback above; re-run `ev-flow bootstrap` "
                "after fixing the cause (the step is idempotent).",
                file=sys.stderr,
            )
            return 1
        n_fleet = res.get("metrics", {}).get("n_fleet", "?")
        print(f"  OK: {region_name}/{profile_type} (n={n_fleet}) -> "
              f"{res.get('out_dir', cdir)}")

    # --- Step 5: optional validation ------------------------------------ #
    if args.validate:
        print("[5/5] Running validator on the generated cache(s) ...")
        from .validator import run as _run_validator

        any_validation_fail = False
        for region_name, profile_type in pairs:
            try:
                report = _run_validator(
                    region=region_name,
                    profile_type=profile_type,
                )
            except Exception as exc:  # noqa: BLE001
                any_validation_fail = True
                print(
                    f"  FAIL: validator errored for {region_name}/{profile_type}: "
                    f"{exc}",
                    file=sys.stderr,
                )
                continue
            status = getattr(report, "overall_status", None)
            print(f"  {region_name}/{profile_type}: validator status = {status}")
        if any_validation_fail:
            print(
                "  FAIL: one or more validation runs errored (see above).",
                file=sys.stderr,
            )
            return 1
    else:
        print("[5/5] Skipping validation (pass --validate to run it).")

    print("")
    print("Bootstrap complete. Try: `ev-flow doctor` or "
          "`python -c \"import pev_synth as ps; "
          "ps.generate_profiles('residential', n=10, region='bay_area')\"`")
    return 0


# --------------------------------------------------------------------------- #
# doctor (F8)
# --------------------------------------------------------------------------- #


def _doctor(args: argparse.Namespace) -> int:
    """Read-only diagnostics. Never mutates state.

    Prints a ``Check | Status | Details`` table. Exit code policy:

    * Any ``FAIL`` row (a genuinely broken environment — Python too old, a
      runtime dependency not importable, or an unwritable data root) -> 1.
    * Any *requested* ``(region, profile_type)`` cache that is incomplete or
      absent -> 1, because without a complete cache the library cannot serve
      ``generate_profiles`` and the environment is not yet usable. (This is the
      common "fresh / empty data root" case: ``ev-flow bootstrap`` is the fix.)
    * Otherwise (all requested caches complete, no hard FAIL) -> 0, even if some
      *optional* infrastructure such as ``validation_bounds`` is still MISSING.

    A missing dependency is always reported as a row instead of crashing on
    import (the whole reason this module's heavy imports are lazy).
    """
    _set_data_root(args.data_root)

    rows: list[tuple[str, str, str]] = []
    hard_fail = False
    caches_incomplete = 0

    # -- Python version -------------------------------------------------- #
    if _python_ok():
        rows.append(("python", _OK, _python_version_str()))
    else:
        hard_fail = True
        rows.append((
            "python",
            _FAIL,
            f"{_python_version_str()} < {MIN_PYTHON[0]}.{MIN_PYTHON[1]} required",
        ))

    # -- Runtime dependencies (a missing one is a hard FAIL) ------------- #
    for mod_name, dist_name in RUNTIME_DEPS:
        ok, detail = _try_import(mod_name)
        if ok:
            rows.append((f"dep:{dist_name}", _OK, detail))
        else:
            hard_fail = True
            rows.append((f"dep:{dist_name}", _FAIL, f"not importable ({detail})"))

    # -- data_root resolves + is writable -------------------------------- #
    try:
        from ._paths import data_root

        droot = data_root()
        if _is_writable(droot):
            rows.append(("data_root", _OK, f"{droot} (writable)"))
        else:
            hard_fail = True
            rows.append(("data_root", _FAIL, f"{droot} (NOT writable)"))
    except Exception as exc:  # noqa: BLE001
        hard_fail = True
        rows.append(("data_root", _FAIL, f"could not resolve: {exc}"))

    # -- sales-mix CSVs -------------------------------------------------- #
    sales_dir = _sales_mix_dir()
    if _sales_mix_present():
        rows.append(("sales_mix_csvs", _OK, str(sales_dir)))
    else:
        rows.append((
            "sales_mix_csvs",
            _MISSING,
            f"absent under {sales_dir}; run `ev-flow bootstrap`",
        ))

    # -- NHTS 2017 parquets ---------------------------------------------- #
    nhts_dir = _nhts_dir()
    if _nhts_present():
        rows.append(("nhts2017_parquets", _OK, str(nhts_dir)))
    else:
        rows.append((
            "nhts2017_parquets",
            _MISSING,
            f"absent under {nhts_dir}; run `ev-flow bootstrap`",
        ))

    # -- per-(region, profile_type) cache completeness ------------------- #
    pairs = [(r, p) for r in args.regions for p in args.profile_types]
    from ._paths import cache_dir as _cache_dir

    for region_name, profile_type in pairs:
        cdir = _cache_dir(region_name, profile_type)
        label = f"cache:{region_name}/{profile_type}"
        if _cache_complete(cdir):
            rows.append((label, _OK, str(cdir)))
        else:
            caches_incomplete += 1
            rows.append((
                label,
                _MISSING,
                f"incomplete/absent under {cdir}; run `ev-flow bootstrap "
                f"--regions {region_name} --profile-types {profile_type}`",
            ))

    # -- validation bounds present --------------------------------------- #
    try:
        from ._paths import validation_bounds_root

        vbroot = validation_bounds_root()
        # A bounds tree is "present" if any per-(region/profile_type) subdir
        # holds at least one .parquet check file (RFC-009 nested layout).
        present = False
        if vbroot.is_dir():
            present = any(vbroot.rglob("*.parquet"))
        if present:
            rows.append(("validation_bounds", _OK, str(vbroot)))
        else:
            rows.append((
                "validation_bounds",
                _MISSING,
                f"no bound parquets under {vbroot}",
            ))
    except Exception as exc:  # noqa: BLE001
        rows.append(("validation_bounds", _MISSING, f"could not resolve: {exc}"))

    # -- optional smoke: generate_profiles(..., n=1) --------------------- #
    if args.smoke:
        rows.append(_smoke_check_row(pairs))

    _print_doctor_table(rows, verbose=args.verbose)

    n_fail = sum(1 for _, status, _ in rows if status == _FAIL)
    n_missing = sum(1 for _, status, _ in rows if status == _MISSING)
    print("")
    print(
        f"Summary: {sum(1 for _, s, _ in rows if s == _OK)} OK, "
        f"{n_missing} MISSING, {n_fail} FAIL."
    )
    if hard_fail or n_fail:
        print(
            "doctor: environment has hard failures (FAIL rows above). "
            "Fix those before using ev-flow.",
            file=sys.stderr,
        )
        return 1
    if caches_incomplete:
        print(
            f"doctor: {caches_incomplete} requested cache(s) are not built yet. "
            "Run `ev-flow bootstrap` to make the environment usable.",
            file=sys.stderr,
        )
        return 1
    if n_missing:
        print(
            "doctor: caches are ready; some optional artifacts are MISSING "
            "(e.g. validation_bounds). That's fine for `generate_profiles`.",
        )
    return 0


def _smoke_check_row(pairs: list[tuple[str, str]]) -> tuple[str, str, str]:
    """Try ``generate_profiles(..., n=1)`` on the first complete cache.

    Read-only: it reads an existing cache, never writes. Reports OK on
    success, MISSING when no complete cache exists to smoke-test, FAIL if a
    present cache cannot be loaded.
    """
    from ._paths import cache_dir as _cache_dir

    target: tuple[str, str] | None = None
    for region_name, profile_type in pairs:
        if _cache_complete(_cache_dir(region_name, profile_type)):
            target = (region_name, profile_type)
            break
    if target is None:
        return (
            "smoke:generate_profiles",
            _MISSING,
            "no complete cache to smoke-test; run `ev-flow bootstrap` first",
        )
    region_name, profile_type = target
    try:
        from .api import generate_profiles

        fleet = generate_profiles(profile_type, n=1, region=region_name)
        n = len(fleet)
    except Exception as exc:  # noqa: BLE001
        return (
            "smoke:generate_profiles",
            _FAIL,
            f"{region_name}/{profile_type}: {exc}",
        )
    return (
        "smoke:generate_profiles",
        _OK,
        f"{region_name}/{profile_type}: built Fleet(n={n})",
    )


def _print_doctor_table(
    rows: list[tuple[str, str, str]], *, verbose: bool
) -> None:
    """Render the ``Check | Status | Details`` table.

    Non-verbose hides ``OK`` rows (so problems stand out); ``--verbose`` shows
    everything.
    """
    shown = [
        r for r in rows if verbose or r[1] != _OK
    ]
    if not shown:
        # Everything passed and not verbose — say so explicitly.
        print("All checks OK. (use --verbose to list passing rows)")
        return

    check_w = max(len("Check"), max(len(r[0]) for r in shown))
    status_w = max(len("Status"), max(len(r[1]) for r in shown))
    header = f"{'Check':<{check_w}}  {'Status':<{status_w}}  Details"
    print(header)
    print(f"{'-' * check_w}  {'-' * status_w}  {'-' * 7}")
    for check, status, details in shown:
        print(f"{check:<{check_w}}  {status:<{status_w}}  {details}")


# --------------------------------------------------------------------------- #
# argument parser + main
# --------------------------------------------------------------------------- #


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ev-flow",
        description=(
            "ev-flow command-line interface. `bootstrap` performs the one-time "
            "environment setup (sales-mix CSVs, NHTS 2017 download, cache "
            "build); `doctor` runs read-only diagnostics."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"ev-flow {__version__}",
        help="Show the installed ev-flow (pev_synth) version and exit.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # ---- bootstrap ---- #
    p_boot = sub.add_parser(
        "bootstrap",
        help="One-time setup: build sales-mix CSVs, fetch NHTS, build cache(s).",
        description=(
            "One-time, idempotent environment setup. Each step skips work whose "
            "outputs already exist. NOTE: the NHTS step downloads ~84 MB and the "
            "cache build runs the M1-M9 pipeline, so a first run takes a few "
            "minutes."
        ),
    )
    p_boot.add_argument(
        "--regions",
        nargs="+",
        default=["bay_area"],
        metavar="REGION",
        help="Region short-name(s) to build caches for (default: bay_area).",
    )
    p_boot.add_argument(
        "--profile-types",
        nargs="+",
        default=["residential"],
        metavar="TYPE",
        help="Profile type(s): residential and/or workplace (default: "
        "residential).",
    )
    p_boot.add_argument(
        "--n",
        type=int,
        default=None,
        help="EVs per cache (default: the pipeline default, 1000; 500 for "
        "us_national).",
    )
    p_boot.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Master seed for cache generation (default: the pipeline master "
        "seed).",
    )
    p_boot.add_argument(
        "--data-root",
        default=None,
        help="Override the data directory (sets PEV_SYNTH_DATA_ROOT for this "
        "run).",
    )
    p_boot.add_argument(
        "--skip-acs",
        dest="skip_acs",
        action="store_true",
        default=True,
        help="Skip ACS PUMS calibration (default: True; ACS needs a "
        "CENSUS_API_KEY).",
    )
    p_boot.add_argument(
        "--no-skip-acs",
        dest="skip_acs",
        action="store_false",
        help="Opt back into ACS calibration (requires CENSUS_API_KEY).",
    )
    p_boot.add_argument(
        "--validate",
        action="store_true",
        help="After building, run the M9 validator on each cache.",
    )
    p_boot.set_defaults(func=_bootstrap)

    # ---- doctor ---- #
    p_doc = sub.add_parser(
        "doctor",
        help="Read-only environment diagnostics (deps, data tree, caches).",
        description=(
            "Read-only diagnostics. Reports Python version, runtime "
            "dependencies, data-root writability, and the presence/completeness "
            "of sales-mix CSVs, NHTS parquets, per-(region, profile_type) "
            "caches, and validation bounds. Never mutates anything. Exits 0 "
            "unless a hard FAIL is found."
        ),
    )
    p_doc.add_argument(
        "--regions",
        nargs="+",
        default=["bay_area"],
        metavar="REGION",
        help="Region short-name(s) whose caches to check (default: bay_area).",
    )
    p_doc.add_argument(
        "--profile-types",
        nargs="+",
        default=["residential"],
        metavar="TYPE",
        help="Profile type(s) whose caches to check (default: residential).",
    )
    p_doc.add_argument(
        "--data-root",
        default=None,
        help="Override the data directory (sets PEV_SYNTH_DATA_ROOT for this "
        "run).",
    )
    p_doc.add_argument(
        "--smoke",
        action="store_true",
        help="Additionally try generate_profiles(..., n=1) on an existing "
        "cache.",
    )
    p_doc.add_argument(
        "--verbose",
        action="store_true",
        help="Show passing (OK) rows too, not just problems.",
    )
    p_doc.set_defaults(func=_doctor)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``ev-flow`` / ``python -m pev_synth.cli``.

    Returns a process exit code: 0 on success / no hard failure, 1 otherwise.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
