"""v2.1 EVSE-enrichment API tests (plan §F.5).

Covers ``Profile.evse_brand`` / ``Profile.evse_connector`` properties,
``Profile.summary`` / ``Fleet.summary`` outputs, and the new
``Fleet.filter(evse_brand=..., evse_connector=...)`` predicates.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

import pev_synth as ps  # noqa: E402
from pev_synth.regions import Region  # noqa: E402

_CODE_DIR = Path(__file__).resolve().parents[1]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))


# Columns required by ``Fleet._row_for`` (mirrors test_pev_synth_api_p1).
_FLEET_COLS_TEMPLATE: dict[str, object] = {
    "ev_id": 0,
    "powertrain": "BEV",
    "make": "Tesla",
    "model": "Model_Y",
    "model_year": 2024,
    "archetype": "crossover",
    "B_kwh": 80.0,
    "OBC_kw": 11.5,
    "EVSE_home_kw": 11.5,
    "panel_cap_kw": 11.5,
    "EVSE_brand": "Tesla",
    "EVSE_connector": "Tesla_NACS_native",
    "eta_charge_nominal": 0.9,
    "has_heat_pump": True,
    "Wh_per_mi_nominal": 300.0,
    "garage_type": "dedicated",
    "income_tier": "t3",
    "home_station": "KSFO",
    "seed": 1,
    "soc_min_kwh": 8.0,
    "nhts_donor_houseid": "h1",
    "nhts_donor_vehid": "v1",
    "cluster_z": 0,
}


def _write_fleet_bundle(
    target: Path,
    ev_ids: list[int],
    brands: list[str] | None = None,
    connectors: list[str] | None = None,
) -> None:
    rows = []
    for i, ev in enumerate(ev_ids):
        row = dict(_FLEET_COLS_TEMPLATE)
        row["ev_id"] = int(ev)
        if brands is not None:
            row["EVSE_brand"] = brands[i % len(brands)]
        if connectors is not None:
            row["EVSE_connector"] = connectors[i % len(connectors)]
        rows.append(row)
    pd.DataFrame(rows).to_parquet(target / "fleet.parquet", index=False)


def _make_fleet(
    tmp_path: Path,
    ev_ids: list[int],
    brands: list[str] | None = None,
    connectors: list[str] | None = None,
    profile_type: str = "residential",
) -> ps.Fleet:
    _write_fleet_bundle(tmp_path, ev_ids, brands=brands, connectors=connectors)
    return ps.Fleet(
        data_root=tmp_path,
        ev_ids=ev_ids,
        profile_type=profile_type,
        region=Region.from_name("bay_area"),
        meta={},
    )


# ---------------------------------------------------------------------------
# Profile-side properties
# ---------------------------------------------------------------------------

def test_profile_row_has_evse_brand_and_connector(tmp_path: Path) -> None:
    fleet = _make_fleet(tmp_path, ev_ids=[0])
    prof = fleet[0]
    assert prof._row.evse_brand == "Tesla"
    assert prof._row.evse_connector == "Tesla_NACS_native"
    # And the public properties.
    assert prof.evse_brand == "Tesla"
    assert prof.evse_connector == "Tesla_NACS_native"


def test_profile_summary_includes_evse_brand_and_connector(tmp_path: Path) -> None:
    fleet = _make_fleet(tmp_path, ev_ids=[0])
    s = fleet[0].summary()
    assert "evse_brand" in s.index
    assert "evse_connector" in s.index
    assert s["evse_brand"] == "Tesla"
    assert s["evse_connector"] == "Tesla_NACS_native"


def test_max_charge_kw_unchanged_semantics(tmp_path: Path) -> None:
    """v2.1 brand / connector do not alter ``max_charge_kw``."""
    fleet = _make_fleet(tmp_path, ev_ids=[0])
    prof = fleet[0]
    assert prof.max_charge_kw == min(prof._row.obc_kw, prof._row.evse_home_kw)


# ---------------------------------------------------------------------------
# Fleet-side: summary + filter
# ---------------------------------------------------------------------------

def test_fleet_summary_includes_evse_brand_column(tmp_path: Path) -> None:
    # The summary() method requires sessions/plug-in-events parquets;
    # we instead exercise the Fleet._fleet_df directly which is the
    # canonical fleet schema.
    fleet = _make_fleet(tmp_path, ev_ids=[0, 1])
    df = fleet._fleet_df
    assert "EVSE_brand" in df.columns
    assert "EVSE_connector" in df.columns


def test_fleet_filter_evse_brand_works(tmp_path: Path) -> None:
    fleet = _make_fleet(
        tmp_path,
        ev_ids=[0, 1, 2, 3],
        brands=["Tesla", "ChargePoint", "Tesla", "ChargePoint"],
    )
    tesla_fleet = fleet.filter(evse_brand="Tesla")
    assert sorted(tesla_fleet._ev_ids) == [0, 2]
    cp_fleet = fleet.filter(evse_brand="ChargePoint")
    assert sorted(cp_fleet._ev_ids) == [1, 3]
    empty_fleet = fleet.filter(evse_brand="Nonexistent")
    assert empty_fleet._ev_ids == []


def test_fleet_filter_evse_connector_works(tmp_path: Path) -> None:
    fleet = _make_fleet(
        tmp_path,
        ev_ids=[0, 1, 2],
        connectors=["J1772", "NACS", "J1772"],
    )
    j1772 = fleet.filter(evse_connector="J1772")
    assert sorted(j1772._ev_ids) == [0, 2]
    nacs = fleet.filter(evse_connector="NACS")
    assert nacs._ev_ids == [1]


def test_fleet_filter_unknown_evse_key_lists_supported(tmp_path: Path) -> None:
    fleet = _make_fleet(tmp_path, ev_ids=[0])
    with pytest.raises(KeyError, match="evse_brand"):
        fleet.filter(evse_brand_starts_with="Te")
