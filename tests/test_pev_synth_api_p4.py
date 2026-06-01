"""Self-contained tests for the P4 minor fixes in ``pev_synth.api``.

Covers:

* ``Fleet.__getitem__`` rejects ``bool`` (footgun: ``bool`` is an ``int``
  subclass in Python so the prior ``isinstance(key, (int, np.integer))``
  silently accepted ``True``/``False``).
* The new explicit ``Fleet.by_ev_id`` and ``Fleet.by_position`` methods
  give unambiguous access semantics (the old ``fleet[i]`` falls through
  by-id → positional, which can flip semantics after ``filter(...)``).
* ``_aggregate_kw_from_sessions`` (the vectorised kernel underlying
  ``Fleet.aggregate_load``) matches the original slow nested-loop
  reference on synthetic input that exercises the edge cases (sessions
  fully before, fully after, straddling either edge, multi-bucket
  sessions, etc.).

These tests build a minimal in-memory ``Fleet`` directly (a single
``fleet.parquet`` row, no other artifacts) so they run unconditionally
regardless of whether the bay_area residential cache is present.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pev_synth as ps
from pev_synth.api import _aggregate_kw_from_sessions

# Columns required by ``Fleet`` construction (mirrors test_pev_synth_api_p1).
# v2.1 EVSE-enrichment columns (``EVSE_brand`` / ``EVSE_connector``) are
# required by ``_row_for`` so we include plausible values.
_FLEET_COLS_BASE = {
    "powertrain": "BEV",
    "make": "Test",
    "model": "TestModel",
    "model_year": 2024,
    "archetype": "commuter",
    "B_kwh": 60.0,
    "OBC_kw": 7.2,
    "EVSE_home_kw": 7.2,
    "panel_cap_kw": 20.0,
    "EVSE_brand": "ChargePoint",
    "EVSE_connector": "J1772",
    "eta_charge_nominal": 0.9,
    "has_heat_pump": True,
    "Wh_per_mi_nominal": 300.0,
    "garage_type": "attached",
    "income_tier": "mid",
    "home_station": "L2",
    "seed": 1,
    "soc_min_kwh": 6.0,
    "nhts_donor_houseid": "h1",
    "nhts_donor_vehid": "v1",
    "cluster_z": 0,
}


def _write_multi_ev_fleet_parquet(target: Path, ev_ids: list[int]) -> None:
    """Write a minimal ``fleet.parquet`` covering the given ev_ids."""
    rows = [{"ev_id": int(ev), **_FLEET_COLS_BASE} for ev in ev_ids]
    pd.DataFrame(rows).to_parquet(target / "fleet.parquet", index=False)


def _write_meta(target: Path) -> None:
    (target / "meta.json").write_text(
        json.dumps({"profile_type": "residential", "region": "bay_area"})
    )


@pytest.fixture
def tiny_fleet(tmp_path: Path) -> ps.Fleet:
    """Three-EV in-memory Fleet with non-contiguous ev_ids (3, 7, 15).

    These ev_ids are chosen so that the positional/by-id distinction
    matters: ``fleet[0]`` falls through to positional and returns ev_id
    3, while ``fleet[3]`` returns ev_id 3 by-id and ``fleet[15]``
    returns ev_id 15 by-id.
    """
    _write_multi_ev_fleet_parquet(tmp_path, [3, 7, 15])
    _write_meta(tmp_path)
    return ps.Fleet.load(tmp_path)


# ----------------------------------------------------------------------
# __getitem__ bool guard
# ----------------------------------------------------------------------

def test_getitem_rejects_bool_true(tiny_fleet: ps.Fleet) -> None:
    with pytest.raises(TypeError, match="bool"):
        tiny_fleet[True]


def test_getitem_rejects_bool_false(tiny_fleet: ps.Fleet) -> None:
    with pytest.raises(TypeError, match="bool"):
        tiny_fleet[False]


def test_getitem_int_still_works(tiny_fleet: ps.Fleet) -> None:
    # By-id semantics: ev_id 3 is present, so fleet[3] is by-id.
    assert tiny_fleet[3].ev_id == 3
    # Positional fall-through: 0 is not an ev_id; resolves positionally.
    assert tiny_fleet[0].ev_id == 3  # first ev_id in sorted order
    # String access still works.
    assert tiny_fleet["ev_0007"].ev_id == 7


# ----------------------------------------------------------------------
# by_ev_id
# ----------------------------------------------------------------------

def test_by_ev_id_present(tiny_fleet: ps.Fleet) -> None:
    prof = tiny_fleet.by_ev_id(7)
    assert prof.ev_id == 7


def test_by_ev_id_absent(tiny_fleet: ps.Fleet) -> None:
    with pytest.raises(KeyError, match="ev_id 99 not in this Fleet"):
        tiny_fleet.by_ev_id(99)


def test_by_ev_id_rejects_bool(tiny_fleet: ps.Fleet) -> None:
    with pytest.raises(TypeError, match="bool"):
        tiny_fleet.by_ev_id(True)


def test_by_ev_id_rejects_non_int(tiny_fleet: ps.Fleet) -> None:
    with pytest.raises(TypeError, match="ev_id must be int"):
        tiny_fleet.by_ev_id("ev_0003")  # type: ignore[arg-type]


def test_by_ev_id_accepts_numpy_integer(tiny_fleet: ps.Fleet) -> None:
    assert tiny_fleet.by_ev_id(np.int64(15)).ev_id == 15


# ----------------------------------------------------------------------
# by_position
# ----------------------------------------------------------------------

def test_by_position_in_range(tiny_fleet: ps.Fleet) -> None:
    # ev_ids sorted: [3, 7, 15].
    assert tiny_fleet.by_position(0).ev_id == 3
    assert tiny_fleet.by_position(1).ev_id == 7
    assert tiny_fleet.by_position(2).ev_id == 15


def test_by_position_disambiguates_from_getitem(tiny_fleet: ps.Fleet) -> None:
    # fleet[3] is by-id (ev_id 3, position 0). by_position(3) is OOR.
    assert tiny_fleet[3].ev_id == 3
    with pytest.raises(IndexError, match="position out of range"):
        tiny_fleet.by_position(3)


def test_by_position_out_of_range_negative(tiny_fleet: ps.Fleet) -> None:
    with pytest.raises(IndexError, match="position out of range"):
        tiny_fleet.by_position(-1)


def test_by_position_rejects_bool(tiny_fleet: ps.Fleet) -> None:
    with pytest.raises(TypeError, match="bool"):
        tiny_fleet.by_position(False)


def test_by_position_rejects_non_int(tiny_fleet: ps.Fleet) -> None:
    with pytest.raises(TypeError, match="position must be int"):
        tiny_fleet.by_position(1.5)  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# _aggregate_kw_from_sessions: vectorised kernel vs slow reference
# ----------------------------------------------------------------------

def _reference_aggregate_kw(
    a_ns: np.ndarray,
    b_ns: np.ndarray,
    p_kw: np.ndarray,
    ts0_ns: int,
    step_ns: int,
    n_buckets: int,
) -> np.ndarray:
    """Slow nested-loop reference: the implementation we replaced."""
    load_kw = np.zeros(n_buckets, dtype=float)
    for ai, bi, pi in zip(a_ns, b_ns, p_kw, strict=True):
        i0 = int((ai - ts0_ns) // step_ns)
        i1 = int((bi - ts0_ns - 1) // step_ns) + 1
        i0 = max(0, i0)
        i1 = min(n_buckets, i1)
        for j in range(i0, i1):
            bucket_lo = ts0_ns + j * step_ns
            bucket_hi = bucket_lo + step_ns
            overlap = max(0, min(bi, bucket_hi) - max(ai, bucket_lo))
            load_kw[j] += pi * (overlap / step_ns)
    return load_kw


# 15-min step in ns and a 24-bucket (6-hour) window starting at epoch.
_STEP_NS = 15 * 60 * 1_000_000_000
_TS0_NS = 0
_N_BUCKETS = 24
_WINDOW_END_NS = _TS0_NS + _N_BUCKETS * _STEP_NS  # exclusive


def test_aggregate_kernel_empty() -> None:
    a = np.array([], dtype=np.int64)
    b = np.array([], dtype=np.int64)
    p = np.array([], dtype=float)
    got = _aggregate_kw_from_sessions(a, b, p, _TS0_NS, _STEP_NS, _N_BUCKETS)
    assert got.shape == (_N_BUCKETS,)
    assert np.all(got == 0.0)


def test_aggregate_kernel_single_session_one_bucket() -> None:
    # Full bucket 5: start of bucket 5 to start of bucket 6, 7.2 kW.
    a = np.array([_TS0_NS + 5 * _STEP_NS], dtype=np.int64)
    b = np.array([_TS0_NS + 6 * _STEP_NS], dtype=np.int64)
    p = np.array([7.2], dtype=float)
    got = _aggregate_kw_from_sessions(a, b, p, _TS0_NS, _STEP_NS, _N_BUCKETS)
    ref = _reference_aggregate_kw(a, b, p, _TS0_NS, _STEP_NS, _N_BUCKETS)
    np.testing.assert_allclose(got, ref)
    assert got[5] == pytest.approx(7.2)
    # Only bucket 5 is touched.
    assert np.count_nonzero(got) == 1


def test_aggregate_kernel_session_fully_before_window() -> None:
    # Session ends before bucket 0 even starts; should contribute nothing.
    a = np.array([_TS0_NS - 5 * _STEP_NS], dtype=np.int64)
    b = np.array([_TS0_NS - 1 * _STEP_NS], dtype=np.int64)
    p = np.array([7.2], dtype=float)
    got = _aggregate_kw_from_sessions(a, b, p, _TS0_NS, _STEP_NS, _N_BUCKETS)
    ref = _reference_aggregate_kw(a, b, p, _TS0_NS, _STEP_NS, _N_BUCKETS)
    np.testing.assert_allclose(got, ref)
    assert np.all(got == 0.0)


def test_aggregate_kernel_session_fully_after_window() -> None:
    a = np.array([_WINDOW_END_NS + _STEP_NS], dtype=np.int64)
    b = np.array([_WINDOW_END_NS + 5 * _STEP_NS], dtype=np.int64)
    p = np.array([7.2], dtype=float)
    got = _aggregate_kw_from_sessions(a, b, p, _TS0_NS, _STEP_NS, _N_BUCKETS)
    ref = _reference_aggregate_kw(a, b, p, _TS0_NS, _STEP_NS, _N_BUCKETS)
    np.testing.assert_allclose(got, ref)
    assert np.all(got == 0.0)


def test_aggregate_kernel_mixed_edges() -> None:
    """4 sessions: straddle-left, fully-inside, straddle-right, before."""
    sessions = [
        # Straddles the left edge: half outside the window, half in bucket 0.
        (_TS0_NS - _STEP_NS // 2, _TS0_NS + _STEP_NS // 2, 7.2),
        # Fully inside: spans buckets 4..6 (2.5 buckets long).
        (
            _TS0_NS + 4 * _STEP_NS,
            _TS0_NS + 6 * _STEP_NS + _STEP_NS // 2,
            11.0,
        ),
        # Straddles the right edge: half in last bucket, half outside.
        (
            _WINDOW_END_NS - _STEP_NS // 2,
            _WINDOW_END_NS + _STEP_NS,
            3.3,
        ),
        # Fully before: contributes nothing.
        (_TS0_NS - 10 * _STEP_NS, _TS0_NS - 5 * _STEP_NS, 100.0),
    ]
    a = np.array([s[0] for s in sessions], dtype=np.int64)
    b = np.array([s[1] for s in sessions], dtype=np.int64)
    p = np.array([s[2] for s in sessions], dtype=float)
    got = _aggregate_kw_from_sessions(a, b, p, _TS0_NS, _STEP_NS, _N_BUCKETS)
    ref = _reference_aggregate_kw(a, b, p, _TS0_NS, _STEP_NS, _N_BUCKETS)
    np.testing.assert_allclose(got, ref)


def test_aggregate_kernel_random_matches_reference() -> None:
    """Random synthetic sessions: vectorised == nested-loop reference."""
    rng = np.random.default_rng(20260521)
    n_sessions = 200
    # Sessions span ~3 buckets on average, with start positions across
    # (and a bit outside) the window. Power in [1, 20] kW.
    starts = rng.integers(
        _TS0_NS - 2 * _STEP_NS, _WINDOW_END_NS + 2 * _STEP_NS, size=n_sessions
    )
    durations = rng.integers(
        _STEP_NS // 4, 5 * _STEP_NS, size=n_sessions
    )
    a = starts.astype(np.int64)
    b = (starts + durations).astype(np.int64)
    p = rng.uniform(1.0, 20.0, size=n_sessions)

    got = _aggregate_kw_from_sessions(a, b, p, _TS0_NS, _STEP_NS, _N_BUCKETS)
    ref = _reference_aggregate_kw(a, b, p, _TS0_NS, _STEP_NS, _N_BUCKETS)
    np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-12)
