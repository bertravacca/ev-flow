"""Unit tests for ``pev_synth.regions`` (Module 1, Wave 1 v2.0)."""

from __future__ import annotations

import dataclasses

import pytest

from pev_synth.regions import (
    PROFILE_TYPES,
    REGIONS,
    Region,
    list_profile_types,
    list_regions,
)

EXPECTED_REGION_NAMES = [
    "bay_area",
    "boston",
    "chicago",
    "dallas_fort_worth",
    "la_basin",
    "new_york_metro",
    "seattle",
    "us_national",
]


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------


def test_list_regions_returns_8_sorted() -> None:
    names = list_regions()
    assert len(names) == 8
    assert names == EXPECTED_REGION_NAMES
    assert names == sorted(names)


def test_list_profile_types_returns_residential_and_workplace_only() -> None:
    types_ = list_profile_types()
    assert types_ == ["residential", "workplace"]
    assert "fleet_depot" not in types_
    assert len(types_) == 2


def test_profile_types_constant_matches_helper() -> None:
    assert list(PROFILE_TYPES) == list_profile_types()


# ---------------------------------------------------------------------------
# ``Region.from_name`` lookup
# ---------------------------------------------------------------------------


def test_from_name_returns_correct_region() -> None:
    bay = Region.from_name("bay_area")
    assert bay.name == "bay_area"
    assert bay.tz == "America/Los_Angeles"
    assert bay.iso_market == "CAISO"
    assert bay.sales_mix_source == "cvrp_ca"
    assert bay is REGIONS["bay_area"]


def test_from_name_raises_on_unknown() -> None:
    with pytest.raises(ValueError) as exc_info:
        Region.from_name("atlantis")
    msg = str(exc_info.value)
    assert "atlantis" in msg
    # Helpful: lists at least one valid name so the caller can self-correct.
    assert "bay_area" in msg


# ---------------------------------------------------------------------------
# Dataclass invariants
# ---------------------------------------------------------------------------


def test_region_is_frozen() -> None:
    region = Region.from_name("bay_area")
    with pytest.raises(dataclasses.FrozenInstanceError):
        region.name = "mutated"  # type: ignore[misc]


def test_region_is_hashable() -> None:
    region = Region.from_name("bay_area")
    # Hashable -> usable as dict key / set member.
    routing: dict[Region, str] = {region: "cache_path"}
    assert routing[region] == "cache_path"
    assert {region, region} == {region}


def test_every_region_has_required_fields_populated() -> None:
    for name, region in REGIONS.items():
        assert region.name == name
        assert region.display_name, f"{name}: display_name empty"
        assert isinstance(region.states, tuple) and region.states, (
            f"{name}: states empty"
        )
        assert isinstance(region.cbsas, tuple)
        assert isinstance(region.cdivmsar_filter, tuple)
        assert region.tz, f"{name}: tz empty"
        assert region.sales_mix_source, f"{name}: sales_mix_source empty"
        # RFC-007: overlay is a canonically-sorted tuple of (key, value) pairs.
        # Use ``sales_mix_overlay_map`` for a read-only Mapping view.
        assert isinstance(region.sales_mix_overlay, tuple)
        assert isinstance(dict(region.sales_mix_overlay), dict)
        assert isinstance(region.workplace_donor_borrow_states, tuple)
        assert region.income_tier_scheme
        assert isinstance(region.winter_f_T_multiplier, float)
        assert region.notes


def test_cbsas_empty_only_for_us_national() -> None:
    # Per the spec, ``cbsas`` is empty only for the national reference region.
    for name, region in REGIONS.items():
        if name == "us_national":
            assert region.cbsas == ()
        else:
            assert len(region.cbsas) >= 1, f"{name}: expected at least 1 CBSA"


# ---------------------------------------------------------------------------
# Region-specific spec checks
# ---------------------------------------------------------------------------


def test_boston_includes_ct_per_rc1() -> None:
    boston = Region.from_name("boston")
    assert boston.states == ("MA", "NH", "RI", "CT")
    assert "CT" in boston.states


def test_dfw_has_pickup_overlay_s5() -> None:
    dfw = Region.from_name("dallas_fort_worth")
    # RFC-007: overlay storage is now a canonical tuple-of-pairs.
    assert dict(dfw.sales_mix_overlay) == {"pickup_overlay": 0.10}
    assert dict(dfw.sales_mix_overlay_map) == {"pickup_overlay": 0.10}
    assert dfw.iso_market == "ERCOT"


def test_us_national_has_51_states() -> None:
    nat = Region.from_name("us_national")
    assert len(nat.states) == 51
    assert "DC" in nat.states
    assert "CA" in nat.states
    assert "WY" in nat.states
    # Excluded non-state territories per the orchestrator contract.
    for excluded in ("PR", "VI", "GU", "AS", "MP"):
        assert excluded not in nat.states
    assert nat.cbsas == ()
    assert nat.iso_market is None
    assert nat.tz == "UTC"


def test_workplace_donor_borrow_states_listed_order() -> None:
    # Listed order is the borrow-attempt order per orchestrator-resolved
    # contract: boston tries NY -> VT -> ME.
    assert Region.from_name("boston").workplace_donor_borrow_states == (
        "NY",
        "VT",
        "ME",
    )
    assert Region.from_name("chicago").workplace_donor_borrow_states == (
        "MI",
        "MN",
    )
    assert Region.from_name("seattle").workplace_donor_borrow_states == (
        "ID",
    )
    assert Region.from_name("new_york_metro").workplace_donor_borrow_states == (
        "PA",
    )


def test_cold_regions_have_winter_uplift() -> None:
    # RFC-031: only Boston / Chicago / NYC are "cold" under the
    # Yuksel-Michalek polynomial calibrated against NOAA Dec-Mar
    # normals. DFW is no longer in the cold set (band [1.10, 1.40]
    # for cold regions; 1.00 elsewhere).
    cold = {name for name, r in REGIONS.items() if r.winter_f_T_multiplier > 1.0}
    assert cold == {"boston", "chicago", "new_york_metro"}
    # Polynomial: f_T(-2) = 1 + 0.018·22 = 1.396 (Boston, NYC).
    assert Region.from_name("boston").winter_f_T_multiplier == 1.396
    assert Region.from_name("new_york_metro").winter_f_T_multiplier == 1.396
    # Chicago: f_T(-4) = 1.432, clamped at band ceiling 1.40.
    assert Region.from_name("chicago").winter_f_T_multiplier == 1.40
    # Non-cold regions (including DFW) sit at the neutral multiplier 1.0.
    for warm in ("bay_area", "la_basin", "seattle", "us_national", "dallas_fort_worth"):
        assert Region.from_name(warm).winter_f_T_multiplier == 1.0


def test_iso_market_distribution() -> None:
    expected = {
        "bay_area": "CAISO",
        "la_basin": "CAISO",
        "new_york_metro": "NYISO",
        "boston": "ISO-NE",
        "chicago": "PJM",
        "dallas_fort_worth": "ERCOT",
        "seattle": None,
        "us_national": None,
    }
    for name, market in expected.items():
        assert Region.from_name(name).iso_market == market


def test_top_level_pev_synth_exports() -> None:
    # The v2.0 public surface must be reachable from the top-level package.
    import pev_synth as ps

    assert ps.Region is Region
    assert ps.list_regions() == list_regions()
    assert ps.list_profile_types() == ["residential", "workplace"]
    assert ps.REGIONS is REGIONS
