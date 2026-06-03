"""Self-contained tests for the P1 fail-loud fixes in ``pev_synth.api``.

These tests do not depend on any real cached bundle and therefore run
unconditionally (unlike ``test_pev_synth_api.py`` which skips when the
bay_area residential cache is absent).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import pev_synth as ps
from pev_synth.api import _to_region_timestamp

# Columns required by ``Fleet._row_for`` so that ``Fleet.load`` succeeds.
# v2.1 EVSE-enrichment columns (``EVSE_brand`` / ``EVSE_connector``) are
# required by ``_row_for`` so we include plausible values.
_FLEET_COLS = {
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


def _write_fleet_parquet(target: Path) -> None:
    df = pd.DataFrame([_FLEET_COLS])
    df.to_parquet(target / "fleet.parquet", index=False)


# ----------------------------------------------------------------------
# Test 1: Fleet.load requires meta.json
# ----------------------------------------------------------------------

def test_fleet_load_requires_meta_json(tmp_path: Path) -> None:
    _write_fleet_parquet(tmp_path)

    # No meta.json present -> FileNotFoundError naming the path. Match on the
    # bare filename: on Windows the full tmp_path contains backslashes that are
    # invalid regex escapes for pytest.raises' re.search-based `match`.
    with pytest.raises(FileNotFoundError, match="meta.json"):
        ps.Fleet.load(tmp_path)

    # meta.json missing 'profile_type' -> ValueError("profile_type").
    (tmp_path / "meta.json").write_text(json.dumps({"region": "bay_area"}))
    with pytest.raises(ValueError, match="profile_type"):
        ps.Fleet.load(tmp_path)

    # meta.json missing 'region' -> ValueError("region").
    (tmp_path / "meta.json").write_text(
        json.dumps({"profile_type": "residential"})
    )
    with pytest.raises(ValueError, match="region"):
        ps.Fleet.load(tmp_path)

    # Complete meta.json -> load succeeds.
    (tmp_path / "meta.json").write_text(
        json.dumps({"profile_type": "residential", "region": "bay_area"})
    )
    fleet = ps.Fleet.load(tmp_path)
    assert fleet.region.name == "bay_area"
    assert fleet.profile_type == "residential"
    assert len(fleet) == 1


# ----------------------------------------------------------------------
# Test 2: _to_region_timestamp raises typed ValueError on DST edges
# ----------------------------------------------------------------------

def test_to_region_timestamp_ambiguous_dst_fall_back() -> None:
    with pytest.raises(ValueError, match="ambiguous"):
        _to_region_timestamp("2024-11-03 01:30:00", "America/Los_Angeles")


def test_to_region_timestamp_nonexistent_dst_spring_forward() -> None:
    with pytest.raises(ValueError, match="non-existent"):
        _to_region_timestamp("2024-03-10 02:30:00", "America/Los_Angeles")


def test_to_region_timestamp_normal_noon_succeeds() -> None:
    out = _to_region_timestamp("2024-06-15 12:00:00", "America/Los_Angeles")
    assert isinstance(out, pd.Timestamp)
    assert out.tzinfo is not None
    assert str(out.tz) == "America/Los_Angeles"


# ----------------------------------------------------------------------
# Test 3: rebuild-cache error message points at the right CLI
# ----------------------------------------------------------------------

def test_generate_profiles_missing_cache_points_at_cache_regen(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError) as exc_info:
        ps.generate_profiles(
            "residential", n=10, region="bay_area", data_root=tmp_path
        )
    msg = str(exc_info.value)
    assert "pev_synth.cache_regen" in msg
    assert "pev_synth.cli" not in msg
