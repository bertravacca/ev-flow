"""Smoke tests for ``pev_synth.cache_regen``.

These tests verify the CLI signature + helper functions but do NOT
materialise full 1000-EV caches (too slow for unit tests).  Full-cache
generation is exercised by the orchestrator at session time.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pev_synth import cache_regen

# ---------------------------------------------------------------------------
# Module-level constants & helpers
# ---------------------------------------------------------------------------

def test_master_seed_matches_brief() -> None:
    assert cache_regen.MASTER_SEED == 20260520


def test_priority_order_lengths_15_caches() -> None:
    # bay_area residential is intentionally absent (pre-existing).
    assert len(cache_regen.PRIORITY_ORDER) == 15
    pairs = set(cache_regen.PRIORITY_ORDER)
    assert ("bay_area", "residential") not in pairs
    assert ("bay_area", "workplace") in pairs
    assert ("us_national", "residential") in pairs
    assert ("us_national", "workplace") in pairs


def test_n_for_region_us_national_is_500() -> None:
    assert cache_regen._n_for_region("us_national") == 500
    assert cache_regen._n_for_region("bay_area") == 1000
    assert cache_regen._n_for_region("boston") == 1000


# ---------------------------------------------------------------------------
# Schema bridging shim (RFC-012 removed): M5 now emits the brief schema
# (``location_type`` + ``plug_in:bool`` + ``location`` synonym) directly,
# so the cache_regen shim is gone. The matching test was retired.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# JSON coercion helper
# ---------------------------------------------------------------------------

def test_coerce_for_json_handles_numpy_and_nan() -> None:
    inp = {
        "i32": np.int32(7),
        "f64": np.float64(3.5),
        "nan": float("nan"),
        "arr": np.array([1, 2, 3]),
        "nested": {"v": np.float32(0.1)},
        "path": Path("/tmp/abc"),
    }
    out = cache_regen._coerce_for_json(inp)
    s = json.dumps(out)
    blob = json.loads(s)
    assert blob["i32"] == 7
    assert blob["nan"] is None
    assert blob["arr"] == [1, 2, 3]
    assert isinstance(blob["nested"]["v"], float)
    # str(Path(...)) round-trip so the assertion is OS-agnostic (Windows uses
    # backslash separators), matching cache_regen._coerce_for_json's str(Path).
    assert blob["path"] == str(Path("/tmp/abc"))


# ---------------------------------------------------------------------------
# Cross-region check helpers
# ---------------------------------------------------------------------------

def test_is_utc_detects_tz_aware_utc() -> None:
    s_utc = pd.Series(
        pd.to_datetime(["2001-01-01", "2001-02-01"], utc=True),
    )
    s_naive = pd.Series(pd.to_datetime(["2001-01-01"]))
    s_la = s_utc.dt.tz_convert("America/Los_Angeles")
    assert cache_regen._is_utc(s_utc)
    assert not cache_regen._is_utc(s_naive)
    assert not cache_regen._is_utc(s_la)


def test_ts_columns_picks_datetime_only() -> None:
    df = pd.DataFrame({
        "a": [1, 2, 3],
        "t": pd.to_datetime(["2001-01-01", "2001-02-01", "2001-03-01"]),
        "s": ["x", "y", "z"],
    })
    cols = cache_regen._ts_columns(df)
    assert cols == ["t"]


# ---------------------------------------------------------------------------
# CLI smoke (argparse construction)
# ---------------------------------------------------------------------------

def test_cli_builds_subcommands() -> None:
    parser = cache_regen._build_arg_parser()
    ns = parser.parse_args(["one", "--region", "bay_area",
                            "--profile-type", "residential"])
    assert ns.cmd == "one"
    assert ns.region == "bay_area"
    assert ns.profile_type == "residential"
    assert ns.seed == cache_regen.MASTER_SEED

    ns = parser.parse_args(["batch", "--limit", "3", "--time-budget-min", "1"])
    assert ns.cmd == "batch"
    assert ns.limit == 3
    assert ns.time_budget_min == 1.0

    ns = parser.parse_args(["audit"])
    assert ns.cmd == "audit"


def test_cli_rejects_unknown_region() -> None:
    parser = cache_regen._build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["one", "--region", "atlantis",
                           "--profile-type", "residential"])
