"""Regression test for the W3 red-team rename.

H9 rename: ``pev_synth.utc_migration`` (public-looking module) was demoted to
``pev_synth._utc_migration`` (package-internal; leading underscore drops it
from auto-discovery and from the public surface). This test pins the rename
so a future refactor cannot accidentally re-expose the migrator under the
public name without an explicit decision.
"""

from __future__ import annotations

import importlib

import pytest


def test_private_module_importable() -> None:
    """``pev_synth._utc_migration`` is the package-internal module name."""
    mod = importlib.import_module("pev_synth._utc_migration")
    # Spot-check that the migration entry-point is exposed on the renamed module.
    assert hasattr(mod, "migrate_bay_area_v1_to_v2")


def test_public_alias_removed() -> None:
    """``pev_synth.utc_migration`` (no leading underscore) must not import.

    H9 removed the public-looking module from the surface; an import attempt
    should fail with ``ModuleNotFoundError`` (a subclass of ``ImportError``).
    """
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("pev_synth.utc_migration")
