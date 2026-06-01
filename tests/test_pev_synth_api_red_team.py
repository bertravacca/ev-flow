"""Regression tests for the red-team remediation pass on ``pev_synth.api``.

Covers five red-team findings, all scoped to ``src/pev_synth/api.py``:

* **M3** — ``_normalize_window`` re-validates ``ts1 > ts0`` after the
  year-remap, so windows that span the cache-year boundary fail loudly
  instead of silently collapsing.
* **L3** — ``Profile.summary().name`` and the dynamic-query Series
  names derive their zero-pad width from the largest ``ev_id`` in the
  Fleet, with a floor of 4 for backwards-compat aesthetics.
* **C2** — ``Fleet.to_optimizer_inputs`` is demoted to
  ``Fleet._to_optimizer_inputs`` (the bridge to the in-house
  ``pev_optim`` sibling is not part of the PyPI surface).
* **H6** — ``Fleet.__init__`` emits a ``RuntimeWarning`` carrying the
  workplace cohort caveat when ``profile_type == 'workplace'``.
* **H3** — ``_normalize_window`` emits a ``UserWarning`` when the
  year-remap shifts the ISO weekday on either endpoint.

All tests are self-contained and synthesize a tiny on-disk fleet bundle
under ``tmp_path`` so they run unconditionally (no cache required).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd
import pytest

import pev_synth as ps
from pev_synth.api import _normalize_window
from pev_synth.regions import Region

# Columns required by ``Fleet._row_for`` (mirrors test_pev_synth_api_p1).
# The v2.1 EVSE-enrichment columns ``EVSE_brand`` / ``EVSE_connector`` are
# required by ``_row_for``; we hard-code plausible values for the synthetic
# bundle below so the red-team scenarios still construct a valid Fleet.
_FLEET_COLS_TEMPLATE: dict[str, object] = {
    "ev_id": 0,
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


def _write_fleet_bundle(target: Path, ev_ids: list[int]) -> None:
    """Write a minimal ``fleet.parquet`` to ``target`` with the given ev_ids."""
    rows = []
    for ev in ev_ids:
        row = dict(_FLEET_COLS_TEMPLATE)
        row["ev_id"] = int(ev)
        rows.append(row)
    pd.DataFrame(rows).to_parquet(target / "fleet.parquet", index=False)


def _make_fleet(
    tmp_path: Path,
    ev_ids: list[int],
    profile_type: str = "residential",
    region: str = "bay_area",
) -> ps.Fleet:
    _write_fleet_bundle(tmp_path, ev_ids)
    return ps.Fleet(
        data_root=tmp_path,
        ev_ids=ev_ids,
        profile_type=profile_type,
        region=Region.from_name(region),
        meta={},
    )


# ----------------------------------------------------------------------
# M3 — post-remap window inversion raises
# ----------------------------------------------------------------------

def test_normalize_window_raises_when_year_remap_inverts_window() -> None:
    # 2025-12-30 and 2026-01-05 individually remap to 2001-12-30 and
    # 2001-01-05, which inverts the order post-remap. The half-open
    # slice would silently return zero rows without this check.
    with pytest.raises(ValueError, match="year-remap inverts"):
        _normalize_window(
            "2025-12-30", "2026-01-05", "America/Los_Angeles"
        )


# ----------------------------------------------------------------------
# L3 — Profile.summary().name pad-width derives from fleet size
# ----------------------------------------------------------------------

def test_profile_summary_name_uses_min_pad_width_4(tmp_path: Path) -> None:
    fleet = _make_fleet(tmp_path, ev_ids=[0])
    s = fleet[0].summary()
    # Floor is 4 to preserve the v1.1 ``ev_0042`` aesthetic on small fleets.
    assert s.name == "ev_0000"


def test_profile_summary_name_pad_grows_with_max_ev_id(tmp_path: Path) -> None:
    fleet = _make_fleet(tmp_path, ev_ids=[12345])
    s = fleet[12345].summary()
    # Width derives from len(str(max(ev_ids))) when it exceeds 4.
    assert s.name == "ev_12345"


# ----------------------------------------------------------------------
# C2 — to_optimizer_inputs is demoted to _to_optimizer_inputs
# ----------------------------------------------------------------------

def test_to_optimizer_inputs_is_demoted_to_private(tmp_path: Path) -> None:
    fleet = _make_fleet(tmp_path, ev_ids=[0])
    assert hasattr(fleet, "to_optimizer_inputs") is False
    assert hasattr(fleet, "_to_optimizer_inputs") is True


def test_to_optimizer_inputs_importerror_mentions_in_house(tmp_path: Path) -> None:
    """ADDED BY QA: stronger C2 test.

    A bare ``hasattr`` invariant pins the rename but does not exercise
    the ``ImportError`` message that the C2 plan tightened (the message
    must call out that this is the in-house bridge to ``pev_optim`` and
    is not installable from PyPI). Call the renamed method directly so a
    future re-introduction of a misleading message (e.g. suggesting
    ``pip install ev-flow[optim]``) is caught.

    ``pev_optim`` is not on PyPI and is not present in this test env, so
    the import inside ``_to_optimizer_inputs`` will fall through to the
    re-raised ``ImportError``; we pin the substring "in-house" on the
    message body.
    """
    fleet = _make_fleet(tmp_path, ev_ids=[0])
    with pytest.raises(ImportError, match="in-house"):
        fleet._to_optimizer_inputs()


# ----------------------------------------------------------------------
# H6 — workplace fleets emit a RuntimeWarning on construction
# ----------------------------------------------------------------------

def test_workplace_fleet_emits_runtimewarning(tmp_path: Path) -> None:
    _write_fleet_bundle(tmp_path, ev_ids=[0])
    with pytest.warns(RuntimeWarning, match="105-vehicle"):
        ps.Fleet(
            data_root=tmp_path,
            ev_ids=[0],
            profile_type="workplace",
            region=Region.from_name("bay_area"),
            meta={},
        )


def test_residential_fleet_emits_no_runtimewarning(tmp_path: Path) -> None:
    _write_fleet_bundle(tmp_path, ev_ids=[0])
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        # Should NOT raise: no RuntimeWarning on residential construction.
        ps.Fleet(
            data_root=tmp_path,
            ev_ids=[0],
            profile_type="residential",
            region=Region.from_name("bay_area"),
            meta={},
        )


# ----------------------------------------------------------------------
# H3 — year-remap that shifts day-of-week emits a UserWarning
# ----------------------------------------------------------------------

def test_normalize_window_warns_on_day_of_week_shift() -> None:
    # 2025-06-01 is Sunday; 2001-06-01 is Friday. After remap both
    # endpoints sit in 2001 and the ISO weekday differs, so the
    # weekday-conditional plug-in model would receive a different
    # weekday/weekend mix than the original calendar week.
    with pytest.warns(UserWarning, match="year-remap"):
        _normalize_window(
            "2025-06-01", "2025-06-08", "America/Los_Angeles"
        )
