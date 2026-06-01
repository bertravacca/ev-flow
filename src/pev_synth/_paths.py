"""Central path registry for ``pev_synth``.

This module is the **single source of truth** for the repo-root path and
every ``data/pev/...`` subtree the package reads or writes. Replaces six
repeated ``Path(__file__).resolve().parents[2]`` computations across
modules (Arch:A2-1, Dev:P2).

All helpers return absolute :class:`pathlib.Path` instances.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "repo_root",
    "data_root",
    "raw_root",
    "processed_root",
    "intermediate_root",
    "nhts_intermediate_root",
    "validation_bounds_root",
    "cache_dir",
    "replicate_dir",
]


def repo_root() -> Path:
    """Return the absolute path to the repository root.

    Anchored on this file's location: ``src/pev_synth/_paths.py`` is two
    levels deep under the repo root.
    """
    return Path(__file__).resolve().parents[2]


def data_root() -> Path:
    """Return the package data root.

    Resolution order:
      1. If env var ``PEV_SYNTH_DATA_ROOT`` is set, use that (resolved to
         an absolute path). This is the supported override for users who
         pip-installed ``ev-flow`` and keep their data elsewhere on disk.
      2. Otherwise fall back to ``<repo_root>/data`` — the dev layout
         where ``data/`` sits next to ``src/``.

    All downstream helpers (``raw_root``, ``processed_root``,
    ``intermediate_root``, ``validation_bounds_root``, ``cache_dir``,
    ``replicate_dir``) chain through this function, so the override
    propagates everywhere.
    """
    env = os.environ.get("PEV_SYNTH_DATA_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return repo_root() / "data"


def raw_root() -> Path:
    """``<repo_root>/data/pev/raw`` — raw inputs (CSVs, NHTS, …)."""
    return data_root() / "pev" / "raw"


def processed_root() -> Path:
    """``<repo_root>/data/pev/processed`` — per-region cached outputs."""
    return data_root() / "pev" / "processed"


def intermediate_root() -> Path:
    """``<repo_root>/data/pev/intermediate``."""
    return data_root() / "pev" / "intermediate"


def nhts_intermediate_root(region: str | None = None) -> Path:
    """Return ``data/pev/intermediate/nhts`` or, optionally, a region subdir.

    The per-region NHTS parquets (``hhpub.parquet`` etc.) live under
    ``intermediate/nhts/<region.name>/`` per the brief (line 103) and
    RFC-014 / RFC-017.
    """
    base = intermediate_root() / "nhts"
    if region is None:
        return base
    return base / region


def validation_bounds_root() -> Path:
    """``<repo_root>/data/pev/validation_bounds``.

    Per RFC-009 the brief layout is per-region:
    ``<root>/<region>/<profile_type>/<check_id>.parquet``.
    """
    return data_root() / "pev" / "validation_bounds"


def cache_dir(region: str, profile_type: str) -> Path:
    """Return the v2.0-native cache root for one ``(region, profile_type)``.

    ``<processed_root>/<region>/<profile_type>_ev_synth``. Matches the brief
    line 11 layout.
    """
    return processed_root() / region / f"{profile_type}_ev_synth"


def replicate_dir(region: str, profile_type: str, r: int, R: int = 1) -> Path:
    """Return the replicate directory for ``(region, profile_type, r)``.

    Per brief line 11 / RFC-022:

    * R == 1: flat layout — returns the cache_dir itself (r=0 alias).
    * R > 1: nested layout — ``<cache_dir>/replicates/r{r}/``.
    """
    base = cache_dir(region, profile_type)
    if R <= 1:
        return base
    return base / "replicates" / f"r{int(r)}"
