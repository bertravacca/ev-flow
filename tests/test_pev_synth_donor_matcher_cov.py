"""Data-independent coverage tests for M3 -- pev_synth.donor_matcher.

These tests exercise the donor-matching pipeline end to end without the heavy
NHTS data tree: every input (CA parquets, national hhpub/vehpub CSVs, NextGen
2022 parquets, the M2 fleet) is fabricated in-memory or written to ``tmp_path``.
They cover the loaders, region narrowing, workplace ANNMILES filter,
borrow-from-neighbours share-cap logic, the full :func:`match_donors` matcher,
the relaxation ladder, persistence helpers, the meta writer, and the CLI
``main`` entry-point.

Determinism: every matcher / RNG path is driven by an explicit seed so the
assertions pin concrete houseids / vehids rather than wall-clock or unseeded
draws.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pev_synth import donor_matcher as dm
from pev_synth.regions import REGIONS, Region

# --------------------------------------------------------------------------- #
# Fixture builders (tiny synthetic NHTS frames).
# --------------------------------------------------------------------------- #


def _ca_hh_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "houseid": pd.array([1, 2, 3, 4], dtype="int64"),
            "hhstate": pd.array(["CA", "CA", "CA", "CA"], dtype="string"),
            "hh_cbsa": pd.array(["41860", "41860", "41860", "99999"], dtype="string"),
            "urbrur": pd.array([1, 1, 2, 1], dtype="Int8"),
            "hhfaminc": pd.array([6, 8, 3, 10], dtype="Int8"),
            "hhsize": pd.array([2, 3, 1, 4], dtype="Int8"),
            "hhvehcnt": pd.array([2, 2, 1, 5], dtype="Int8"),
            "wthhfin": pd.Series([100.0, 200.0, 50.0, 300.0], dtype="float64"),
        }
    )


def _ca_veh_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "houseid": pd.array([1, 1, 2, 3, 4], dtype="int64"),
            "vehid": pd.array([1, 2, 1, 1, 1], dtype="int32"),
            "vehtype": pd.array([1, 1, 1, 1, 1], dtype="Int8"),
            "make": pd.array(
                ["Tesla", "Toyota", "Tesla", "Honda", "Nissan"], dtype="string"
            ),
            "model": pd.array(
                ["Model3", "Prius", "ModelY", "Civic", "Leaf"], dtype="string"
            ),
            "vehyear": pd.array([2020, 2019, 2021, 2018, 2017], dtype="Int16"),
            "annmiles": pd.array(
                [8000.0, 1000.0, 9000.0, 2000.0, 12000.0], dtype="float32"
            ),
            "hfuel": pd.array([3, 2, 3, 2, 3], dtype="Int8"),
        }
    )


def _write_ca_parquets(nhts_dir: Path) -> None:
    nhts_dir.mkdir(parents=True, exist_ok=True)
    _ca_hh_frame().to_parquet(nhts_dir / "nhts2017_ca_hh.parquet")
    _ca_veh_frame().to_parquet(nhts_dir / "nhts2017_ca_veh.parquet")


def _write_national_csvs(
    nhts_dir: Path,
    *,
    home_state: str = "MA",
    n_home: int = 4,
    borrow_state: str = "NH",
    n_borrow: int = 16,
    annmiles: float = 1000.0,
) -> None:
    """Write minimal national hhpub/vehpub CSVs (uppercase NHTS columns)."""
    nhts_dir.mkdir(parents=True, exist_ok=True)
    states = [home_state] * n_home + [borrow_state] * n_borrow
    ids = list(range(1, len(states) + 1))
    hh = pd.DataFrame(
        {
            "HOUSEID": ids,
            "HHSTATE": states,
            "HH_CBSA": ["14460"] * len(ids),
            "URBRUR": [1] * len(ids),
            "HHFAMINC": [6] * len(ids),
            "HHSIZE": [2] * len(ids),
            "HHVEHCNT": [2] * len(ids),
            "WTHHFIN": [100.0] * len(ids),
        }
    )
    veh = pd.DataFrame(
        {
            "HOUSEID": ids,
            "VEHID": [1] * len(ids),
            "VEHTYPE": [1] * len(ids),
            "MAKE": ["Tesla"] * len(ids),
            "MODEL": ["Model3"] * len(ids),
            "VEHYEAR": [2020] * len(ids),
            "ANNMILES": [annmiles] * len(ids),
            "HFUEL": [3] * len(ids),
        }
    )
    hh.to_csv(nhts_dir / "hhpub.csv", index=False)
    veh.to_csv(nhts_dir / "vehpub.csv", index=False)


def _ca_region(**overrides: object) -> Region:
    base = dict(name="tiny_ca", states=("CA",), cbsas=(41860,))
    base.update(overrides)
    return dataclasses.replace(REGIONS["bay_area"], **base)


def _ma_workplace_region(**overrides: object) -> Region:
    base = dict(
        name="tiny_ma",
        states=("MA",),
        cbsas=(),
        workplace_donor_borrow_states=("NH",),
    )
    base.update(overrides)
    return dataclasses.replace(REGIONS["boston"], **base)


def _fleet(n: int, powertrains: list[str] | None = None) -> pd.DataFrame:
    pts = powertrains or (["BEV", "PHEV"] * n)[:n]
    tiers = (["t1", "t2", "t3", "t4"] * n)[:n]
    return pd.DataFrame(
        {
            "ev_id": np.arange(n, dtype=np.int64),
            "powertrain": pts,
            "income_tier": tiers,
        }
    )


# --------------------------------------------------------------------------- #
# Pure helpers: _band_4plus, _map_income_tier, build_match_keys.
# --------------------------------------------------------------------------- #


def test_band_4plus_raises_on_na() -> None:
    s = pd.Series(pd.array([1, pd.NA, 3], dtype="Int8"))
    with pytest.raises(ValueError, match="band_4plus received NA"):
        dm._band_4plus(s)


def test_band_4plus_clips_and_maps_zero() -> None:
    s = pd.Series(pd.array([0, 1, 4, 9], dtype="Int8"))
    out = dm._band_4plus(s)
    assert out.tolist() == ["1", "1", "4+", "4+"]


def test_build_match_keys_columns_and_income_default() -> None:
    hh = pd.DataFrame(
        {
            "hhfaminc": pd.array([1, 6, 9, 11, pd.NA], dtype="Int8"),
            "hhsize": pd.array([1, 2, 3, 4, 2], dtype="Int8"),
            "hhvehcnt": pd.array([1, 2, 3, 6, 1], dtype="Int8"),
            "urbrur": pd.array([1, 2, 1, 2, 1], dtype="Int8"),
        }
    )
    out = dm.build_match_keys(hh)
    assert out["income_tier"].tolist() == ["t1", "t2", "t3", "t4", "t2"]
    assert out["hhsize_band"].tolist() == ["1", "2", "3", "4+", "2"]
    assert out["hhvehcnt_band"].tolist() == ["1", "2", "3", "4+", "1"]
    assert out["urbrur_int"].tolist() == [1, 2, 1, 2, 1]


# --------------------------------------------------------------------------- #
# _ensure_vintage_column (both branches).
# --------------------------------------------------------------------------- #


def test_ensure_vintage_column_stamps_when_absent() -> None:
    df = pd.DataFrame({"x": [1, 2, 3]})
    out = dm._ensure_vintage_column(df, dm.NHTS_VINTAGE_2017)
    assert "nhts_vintage" not in df.columns  # input not mutated
    assert out["nhts_vintage"].tolist() == ["2017", "2017", "2017"]
    assert out["nhts_vintage"].dtype == "string"


def test_ensure_vintage_column_preserves_when_present() -> None:
    df = pd.DataFrame({"x": [1], "nhts_vintage": ["2022_nextgen"]})
    out = dm._ensure_vintage_column(df, dm.NHTS_VINTAGE_2017)
    assert out["nhts_vintage"].tolist() == ["2022_nextgen"]
    assert out["nhts_vintage"].dtype == "string"


# --------------------------------------------------------------------------- #
# Loaders: _read_ca_parquets, _read_national_csvs, _read_nextgen_parquets.
# --------------------------------------------------------------------------- #


def test_read_ca_parquets_stamps_vintage(tmp_path: Path) -> None:
    nd = tmp_path / "nhts2017"
    _write_ca_parquets(nd)
    hh, veh = dm._read_ca_parquets(nd)
    assert set(hh["nhts_vintage"].dropna()) == {"2017"}
    assert set(veh["nhts_vintage"].dropna()) == {"2017"}
    assert len(hh) == 4


def test_read_national_csvs_casts_and_lowercases(tmp_path: Path) -> None:
    nd = tmp_path / "nhts2017"
    _write_national_csvs(nd)
    hh, veh = dm._read_national_csvs(nd)
    assert "hhstate" in hh.columns and "HHSTATE" not in hh.columns
    assert hh["hhsize"].dtype == "Int8"
    assert hh["wthhfin"].dtype == "float64"
    assert veh["houseid"].dtype == "int64"
    assert veh["annmiles"].dtype == "float32"
    assert set(hh["nhts_vintage"].dropna()) == {"2017"}


def test_read_nextgen_parquets_returns_none_when_absent(tmp_path: Path) -> None:
    assert dm._read_nextgen_parquets(tmp_path / "nope") is None


def test_read_nextgen_parquets_loads_when_present(tmp_path: Path) -> None:
    ng = tmp_path / "nhts2022"
    ng.mkdir()
    hh = pd.DataFrame({"houseid": [1], "wthhfin": [1.0]})
    veh = pd.DataFrame({"houseid": [1], "vehid": [1]})
    hh.to_parquet(ng / "nhts2022_national_hh.parquet")
    veh.to_parquet(ng / "nhts2022_national_veh.parquet")
    loaded = dm._read_nextgen_parquets(ng)
    assert loaded is not None
    ng_hh, ng_veh = loaded
    assert set(ng_hh["nhts_vintage"].dropna()) == {"2022_nextgen"}
    assert set(ng_veh["nhts_vintage"].dropna()) == {"2022_nextgen"}


# --------------------------------------------------------------------------- #
# load_region_nhts (CA path, national path, NextGen union, missing-warn).
# --------------------------------------------------------------------------- #


def test_load_region_nhts_ca_path(tmp_path: Path) -> None:
    nd = tmp_path / "nhts2017"
    _write_ca_parquets(nd)
    hh, veh = dm.load_region_nhts(_ca_region(), nd)
    assert (hh["hhstate"] == "CA").all()
    assert veh["houseid"].isin(hh["houseid"]).all()


def test_load_region_nhts_national_path_state_filter(tmp_path: Path) -> None:
    nd = tmp_path / "nhts2017"
    _write_national_csvs(nd)
    region = _ma_workplace_region()
    hh, veh = dm.load_region_nhts(region, nd)
    # State filter narrows to MA only (4 home rows).
    assert (hh["hhstate"] == "MA").all()
    assert len(hh) == 4
    assert veh["houseid"].isin(hh["houseid"]).all()


def test_load_region_nhts_unions_nextgen(tmp_path: Path) -> None:
    nd = tmp_path / "nhts2017"
    _write_ca_parquets(nd)
    ng = tmp_path / "nhts2022"
    ng.mkdir()
    hh_ng = pd.DataFrame(
        {
            "houseid": pd.array([900001, 900002], dtype="int64"),
            "hhstate": pd.array([pd.NA, pd.NA], dtype="string"),
            "hh_cbsa": pd.array([pd.NA, pd.NA], dtype="string"),
            "urbrur": pd.array([1, 2], dtype="Int8"),
            "hhfaminc": pd.array([7, 5], dtype="Int8"),
            "hhsize": pd.array([2, 1], dtype="Int8"),
            "hhvehcnt": pd.array([2, 1], dtype="Int8"),
            "wthhfin": pd.Series([500.0, 600.0], dtype="float64"),
        }
    )
    veh_ng = pd.DataFrame(
        {
            "houseid": pd.array([900001, 900002], dtype="int64"),
            "vehid": pd.array([1, 1], dtype="int32"),
            "make": pd.array(["Tesla", "Honda"], dtype="string"),
            "model": pd.array([pd.NA, pd.NA], dtype="string"),
        }
    )
    hh_ng.to_parquet(ng / "nhts2022_national_hh.parquet")
    veh_ng.to_parquet(ng / "nhts2022_national_veh.parquet")

    hh, veh = dm.load_region_nhts(
        _ca_region(), nd, extra_vintages=("2022_nextgen",)
    )
    assert set(hh["nhts_vintage"].dropna()) == {"2017", "2022_nextgen"}
    assert set(veh["nhts_vintage"].dropna()) == {"2017", "2022_nextgen"}


def test_load_region_nhts_nextgen_missing_warns(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    nd = tmp_path / "nhts2017"
    _write_ca_parquets(nd)
    with caplog.at_level("WARNING"):
        hh, _ = dm.load_region_nhts(
            _ca_region(), nd, extra_vintages=("2022_nextgen",)
        )
    assert any("2022_nextgen" in r.message for r in caplog.records)
    assert set(hh["nhts_vintage"].dropna()) == {"2017"}


# --------------------------------------------------------------------------- #
# narrow_to_region branches.
# --------------------------------------------------------------------------- #


def test_narrow_to_region_missing_cbsa_column_raises() -> None:
    hh = pd.DataFrame({"houseid": [1], "hhstate": ["CA"]})
    with pytest.raises(KeyError, match="hh_cbsa"):
        dm.narrow_to_region(hh, _ca_region())


def test_narrow_to_region_legacy_no_vintage() -> None:
    region = _ca_region()
    hh = pd.DataFrame(
        {
            "houseid": [1, 2, 3],
            "hhstate": pd.array(["CA", "CA", "NV"], dtype="string"),
            "hh_cbsa": pd.array(["41860", "99999", "41860"], dtype="string"),
        }
    )
    out = dm.narrow_to_region(hh, region)
    assert out["houseid"].tolist() == [1]


def test_narrow_to_region_all_legacy_with_vintage_col() -> None:
    """vintage column present but no nextgen rows -> legacy reset branch."""
    region = _ca_region()
    hh = pd.DataFrame(
        {
            "houseid": [1, 2],
            "hhstate": pd.array(["CA", "CA"], dtype="string"),
            "hh_cbsa": pd.array(["41860", "99999"], dtype="string"),
            "nhts_vintage": pd.array(["2017", "2017"], dtype="string"),
        }
    )
    out = dm.narrow_to_region(hh, region)
    assert out["houseid"].tolist() == [1]


# --------------------------------------------------------------------------- #
# _apply_vintage_mix error / degenerate branches.
# --------------------------------------------------------------------------- #


def test_apply_vintage_mix_missing_vintage_col_raises() -> None:
    with pytest.raises(KeyError, match="nhts_vintage"):
        dm._apply_vintage_mix(pd.DataFrame({"wthhfin": [1.0]}), {"2017": 1.0})


def test_apply_vintage_mix_missing_weight_col_raises() -> None:
    hh = pd.DataFrame({"nhts_vintage": pd.array(["2017"], dtype="string")})
    with pytest.raises(KeyError, match="wthhfin"):
        dm._apply_vintage_mix(hh, {"2017": 1.0})


def test_apply_vintage_mix_empty_mix_raises() -> None:
    hh = pd.DataFrame(
        {
            "wthhfin": [1.0],
            "nhts_vintage": pd.array(["2017"], dtype="string"),
        }
    )
    with pytest.raises(ValueError, match="non-empty vintage_mix"):
        dm._apply_vintage_mix(hh, {})


def test_apply_vintage_mix_degenerate_zero_weights_uniform() -> None:
    """A vintage whose weights sum to <= 0 gets uniform target-share split."""
    hh = pd.DataFrame(
        {
            "houseid": [1, 2],
            "wthhfin": [0.0, 0.0],
            "nhts_vintage": pd.array(["2017", "2017"], dtype="string"),
        }
    )
    out = dm._apply_vintage_mix(hh, {"2017": 1.0})
    assert out["wthhfin"].tolist() == pytest.approx([0.5, 0.5])


def test_apply_vintage_mix_skips_vintage_absent_from_frame() -> None:
    """A vintage in the mix but absent from the frame is skipped (continue)."""
    hh = pd.DataFrame(
        {
            "houseid": [1, 2],
            "wthhfin": [10.0, 30.0],
            "nhts_vintage": pd.array(["2017", "2017"], dtype="string"),
        }
    )
    out = dm._apply_vintage_mix(hh, {"2017": 1.0, "2022_nextgen": 0.5})
    assert out["wthhfin"].sum() == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# _model_backoff_draw_from_2017 error / branch coverage.
# --------------------------------------------------------------------------- #


def test_model_backoff_missing_vintage_col_raises() -> None:
    with pytest.raises(KeyError, match="nhts_vintage"):
        dm._model_backoff_draw_from_2017(pd.DataFrame({"model": [pd.NA]}))


def test_model_backoff_missing_model_col_raises() -> None:
    veh = pd.DataFrame({"nhts_vintage": pd.array(["2017"], dtype="string")})
    with pytest.raises(KeyError, match="model"):
        dm._model_backoff_draw_from_2017(veh)


def test_model_backoff_default_rng_fills_deterministically() -> None:
    veh = pd.DataFrame(
        {
            "houseid": [1, 90],
            "vehid": pd.array([1, 1], dtype="int32"),
            "make": pd.array(["Toyota", "Toyota"], dtype="string"),
            "model": pd.array(["Camry", pd.NA], dtype="string"),
            "vehyear": pd.array([2018, 2018], dtype="Int16"),
            "hfuel": pd.array([-1, -1], dtype="Int8"),
            "fueltype": pd.array([1, 1], dtype="Int8"),
            "nhts_vintage": pd.array(["2017", "2022_nextgen"], dtype="string"),
        }
    )
    out = dm._model_backoff_draw_from_2017(veh)  # rng=None -> default_rng(0)
    assert out.loc[out["houseid"] == 90, "model"].iloc[0] == "Camry"


def test_model_backoff_backs_off_to_make_hfuel_when_year_missing() -> None:
    """NextGen row with NA vehyear skips the 2 keys containing vehyear."""
    veh = pd.DataFrame(
        {
            "houseid": [1, 90],
            "vehid": pd.array([1, 1], dtype="int32"),
            "make": pd.array(["Toyota", "Toyota"], dtype="string"),
            "model": pd.array(["Camry", pd.NA], dtype="string"),
            "vehyear": pd.array([2018, pd.NA], dtype="Int16"),
            "hfuel": pd.array([2, 2], dtype="Int8"),
            "fueltype": pd.array([1, 1], dtype="Int8"),
            "nhts_vintage": pd.array(["2017", "2022_nextgen"], dtype="string"),
        }
    )
    out = dm._model_backoff_draw_from_2017(veh, rng=np.random.default_rng(1))
    assert out.loc[out["houseid"] == 90, "model"].iloc[0] == "Camry"


# --------------------------------------------------------------------------- #
# apply_workplace_annmiles_filter.
# --------------------------------------------------------------------------- #


def test_workplace_annmiles_filter_strict() -> None:
    hh = pd.DataFrame({"houseid": [1, 2, 3], "wthhfin": [1.0, 1.0, 1.0]})
    veh = pd.DataFrame(
        {
            "houseid": [1, 1, 2, 3],
            "vehid": pd.array([1, 2, 1, 1], dtype="int32"),
            "annmiles": pd.array([3000.0, 8000.0, 3600.0, -9.0], dtype="float32"),
            "hfuel": pd.array([3, -1, 2, -1], dtype="Int8"),
        }
    )
    hh_keep, veh_keep = dm.apply_workplace_annmiles_filter(hh, veh, annmiles_max=3600.0)
    assert hh_keep["houseid"].tolist() == [1]
    assert set(veh_keep["houseid"]) == {1}


# --------------------------------------------------------------------------- #
# borrow_from_neighbours: residential no-op, share-cap, in-region skip.
# --------------------------------------------------------------------------- #


def test_borrow_noop_residential(tmp_path: Path) -> None:
    nd = tmp_path / "nhts2017"
    _write_national_csvs(nd)
    hh = pd.DataFrame(
        {"houseid": [1, 2], "hhstate": pd.array(["MA", "MA"], dtype="string")}
    )
    veh = pd.DataFrame(
        {
            "houseid": [1, 2],
            "vehid": pd.array([1, 1], dtype="int32"),
            "annmiles": pd.array([100.0, 200.0], dtype="float32"),
            "hfuel": pd.array([3, 3], dtype="Int8"),
        }
    )
    out_hh, _out_veh, n_borrowed = dm.borrow_from_neighbours(
        hh, veh, _ma_workplace_region(), "residential", nd
    )
    assert n_borrowed == 0
    assert (~out_hh["borrow_used"]).all()
    assert (out_hh["donor_origin_state"] == "").all()


def test_borrow_noop_when_pool_above_threshold(tmp_path: Path) -> None:
    nd = tmp_path / "nhts2017"
    _write_national_csvs(nd)
    hh = pd.DataFrame(
        {"houseid": [1, 2], "hhstate": pd.array(["MA", "MA"], dtype="string")}
    )
    veh = pd.DataFrame(
        {
            "houseid": [1, 2],
            "vehid": pd.array([1, 1], dtype="int32"),
            "annmiles": pd.array([100.0, 200.0], dtype="float32"),
            "hfuel": pd.array([3, 3], dtype="Int8"),
        }
    )
    _, _, n_borrowed = dm.borrow_from_neighbours(
        hh, veh, _ma_workplace_region(), "workplace", nd, pool_threshold=1
    )
    assert n_borrowed == 0


def test_borrow_workplace_respects_share_cap(tmp_path: Path) -> None:
    nd = tmp_path / "nhts2017"
    # 4 MA home rows, 16 NH borrow rows, all annmiles<3600.
    _write_national_csvs(nd, n_home=4, n_borrow=16, annmiles=1000.0)
    hh = pd.DataFrame(
        {
            "houseid": [1, 2, 3, 4],
            "hhstate": pd.array(["MA"] * 4, dtype="string"),
            "wthhfin": pd.Series([100.0] * 4, dtype="float64"),
        }
    )
    veh = pd.DataFrame(
        {
            "houseid": [1, 2, 3, 4],
            "vehid": pd.array([1, 1, 1, 1], dtype="int32"),
            "annmiles": pd.array([1000.0] * 4, dtype="float32"),
            "hfuel": pd.array([3, 3, 3, 3], dtype="Int8"),
        }
    )
    out_hh, _out_veh, n_borrowed = dm.borrow_from_neighbours(
        hh, veh, _ma_workplace_region(), "workplace", nd,
        pool_threshold=100, share_cap=0.25,
    )
    # max_borrow_total = int(0.25 * 4 / 0.75) = 1.
    assert n_borrowed == 1
    borrowed = out_hh[out_hh["borrow_used"]]
    assert set(borrowed["donor_origin_state"]) == {"NH"}


def test_borrow_share_cap_one_allows_full_pool(tmp_path: Path) -> None:
    nd = tmp_path / "nhts2017"
    _write_national_csvs(nd, n_home=2, n_borrow=5, annmiles=1000.0)
    hh = pd.DataFrame(
        {
            "houseid": [1, 2],
            "hhstate": pd.array(["MA", "MA"], dtype="string"),
            "wthhfin": pd.Series([100.0, 100.0], dtype="float64"),
        }
    )
    veh = pd.DataFrame(
        {
            "houseid": [1, 2],
            "vehid": pd.array([1, 1], dtype="int32"),
            "annmiles": pd.array([1000.0, 1000.0], dtype="float32"),
            "hfuel": pd.array([3, 3], dtype="Int8"),
        }
    )
    # share_cap >= 1.0 takes the unbounded branch; threshold hit stops borrow.
    _, _, n_borrowed = dm.borrow_from_neighbours(
        hh, veh, _ma_workplace_region(), "workplace", nd,
        pool_threshold=4, share_cap=1.0,
    )
    assert n_borrowed >= 2  # enough to reach threshold of 4


def test_borrow_skips_state_already_in_region(tmp_path: Path) -> None:
    """A borrow state that is also a region state is skipped (continue)."""
    nd = tmp_path / "nhts2017"
    _write_national_csvs(nd, home_state="MA", n_home=2, borrow_state="NH", n_borrow=5)
    region = _ma_workplace_region(
        states=("MA", "NH"), workplace_donor_borrow_states=("NH",)
    )
    hh = pd.DataFrame(
        {
            "houseid": [1, 2],
            "hhstate": pd.array(["MA", "MA"], dtype="string"),
            "wthhfin": pd.Series([100.0, 100.0], dtype="float64"),
        }
    )
    veh = pd.DataFrame(
        {
            "houseid": [1, 2],
            "vehid": pd.array([1, 1], dtype="int32"),
            "annmiles": pd.array([1000.0, 1000.0], dtype="float32"),
            "hfuel": pd.array([3, 3], dtype="Int8"),
        }
    )
    _, _, n_borrowed = dm.borrow_from_neighbours(
        hh, veh, region, "workplace", nd, pool_threshold=100, share_cap=0.5,
    )
    # NH is in region.states so it is skipped; nothing borrowed.
    assert n_borrowed == 0


# --------------------------------------------------------------------------- #
# Matcher internal helpers.
# --------------------------------------------------------------------------- #


def test_vehicle_lookup_groups_by_houseid() -> None:
    veh = pd.DataFrame(
        {
            "houseid": [2, 1, 1],
            "vehid": pd.array([1, 2, 1], dtype="int32"),
            "hfuel": pd.array([3, 2, 3], dtype="Int8"),
        }
    )
    lut = dm._vehicle_lookup(veh)
    assert set(lut) == {1, 2}
    assert lut[1]["vehid"].tolist() == [1, 2]


def test_pick_vehicle_prefers_matching_hfuel() -> None:
    pool = pd.DataFrame(
        {
            "houseid": [1, 1],
            "vehid": pd.array([10, 11], dtype="int32"),
            "hfuel": pd.array([dm.HFUEL_PHEV, dm.HFUEL_BEV], dtype="Int8"),
        }
    )
    rng = np.random.default_rng(0)
    assert dm._pick_vehicle(pool, "BEV", rng) == 11
    assert dm._pick_vehicle(pool, "PHEV", rng) == 10


def test_pick_vehicle_falls_back_when_no_preferred() -> None:
    pool = pd.DataFrame(
        {
            "houseid": [1],
            "vehid": pd.array([10], dtype="int32"),
            "hfuel": pd.array([dm.HFUEL_HEV_NON_PLUGIN], dtype="Int8"),
        }
    )
    assert dm._pick_vehicle(pool, "BEV", np.random.default_rng(0)) == 10


def test_pick_vehicle_empty_raises() -> None:
    pool = pd.DataFrame({"vehid": pd.array([], dtype="int32"), "hfuel": []})
    with pytest.raises(ValueError, match="empty HH vehicle pool"):
        dm._pick_vehicle(pool, "BEV", np.random.default_rng(0))


def test_draw_houseid_weighted_empty_raises() -> None:
    with pytest.raises(ValueError, match="empty pool"):
        dm._draw_houseid_weighted(
            np.random.default_rng(0), pd.DataFrame({"houseid": [], "wthhfin": []})
        )


def test_draw_houseid_weighted_falls_back_on_bad_weights() -> None:
    """Non-finite / non-positive weights -> uniform probs branch."""
    pool = pd.DataFrame(
        {"houseid": [5, 6], "wthhfin": [0.0, 0.0]}
    )
    hid, n = dm._draw_houseid_weighted(np.random.default_rng(0), pool)
    assert hid in {5, 6}
    assert n == 2


def test_candidate_pool_relaxation_empty_raises() -> None:
    hh = pd.DataFrame(
        {
            "houseid": pd.array([], dtype="int64"),
            "income_tier": pd.array([], dtype="string"),
            "hhsize_band": pd.array([], dtype="string"),
            "urbrur_int": pd.array([], dtype="Int8"),
            "hhvehcnt_band": pd.array([], dtype="string"),
        }
    )
    with pytest.raises(RuntimeError, match="donor pool is empty"):
        dm._candidate_pool_with_relaxation(
            hh, income_tier="t2", hhsize_band="2", urbrur=1, hhvehcnt_band="2"
        )


# --------------------------------------------------------------------------- #
# match_donors: full end-to-end (synthetic CA parquets) + error branches.
# --------------------------------------------------------------------------- #


def test_match_donors_residential_full_pipeline(tmp_path: Path) -> None:
    nd = tmp_path / "nhts2017"
    _write_ca_parquets(nd)
    region = _ca_region()
    fleet = _fleet(8)
    out = dm.match_donors(fleet, region, "residential", seed=42, nhts_data_dir=nd)
    assert out["nhts_donor_houseid"].notna().all()
    assert out["nhts_donor_vehid"].notna().all()
    attrs = out.attrs["donor_match"]
    assert attrs["region_name"] == "tiny_ca"
    assert attrs["n_hh_borrowed"] == 0
    # CBSA narrowing keeps houseids 1,2,3 (41860); houseid 4 is 99999.
    assert set(out["nhts_donor_houseid"]).issubset({1, 2, 3})
    assert (~out["borrow_used"]).all()
    assert out["donor_origin_state"].isna().all()


def test_match_donors_seed_reproducible(tmp_path: Path) -> None:
    nd = tmp_path / "nhts2017"
    _write_ca_parquets(nd)
    region = _ca_region()
    fleet = _fleet(8)
    a = dm.match_donors(fleet, region, "residential", seed=7, nhts_data_dir=nd)
    b = dm.match_donors(fleet, region, "residential", seed=7, nhts_data_dir=nd)
    pd.testing.assert_frame_equal(
        a.reset_index(drop=True), b.reset_index(drop=True)
    )


def test_match_donors_workplace_with_borrow(tmp_path: Path) -> None:
    nd = tmp_path / "nhts2017"
    _write_national_csvs(nd, n_home=4, n_borrow=16, annmiles=1000.0)
    region = _ma_workplace_region()
    fleet = _fleet(10, powertrains=["BEV"] * 10)
    out = dm.match_donors(
        fleet, region, "workplace", seed=42, nhts_data_dir=nd,
        borrow_pool_threshold=100, borrow_share_cap=0.25,
    )
    attrs = out.attrs["donor_match"]
    assert attrs["n_hh_borrowed"] == 1
    assert attrs["borrow_share"] == pytest.approx(0.2)
    assert out["borrow_used"].any()
    borrowed = out[out["borrow_used"]]
    assert borrowed["donor_origin_state"].notna().all()
    assert set(borrowed["donor_origin_state"].dropna()) == {"NH"}


def test_match_donors_invalid_profile_type_raises() -> None:
    fleet = _fleet(1)
    with pytest.raises(ValueError, match="profile_type"):
        dm.match_donors(fleet, _ca_region(), "garage")


def test_match_donors_missing_columns_raises() -> None:
    fleet = pd.DataFrame({"ev_id": [0]})
    with pytest.raises(KeyError, match="fleet missing required columns"):
        dm.match_donors(fleet, _ca_region(), "residential")


def test_match_donors_warns_on_thin_pool(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    nd = tmp_path / "nhts2017"
    _write_ca_parquets(nd)
    fleet = _fleet(3)
    with caplog.at_level("WARNING"):
        dm.match_donors(fleet, _ca_region(), "residential", seed=1, nhts_data_dir=nd)
    assert any("donor pool" in r.message for r in caplog.records)


# --------------------------------------------------------------------------- #
# Persistence + path helpers.
# --------------------------------------------------------------------------- #


def test_resolve_default_nhts_dir_is_absolute() -> None:
    p = dm._resolve_default_nhts_dir()
    assert p.is_absolute()
    assert p.name == "nhts2017"


def test_default_repo_root_is_absolute() -> None:
    assert dm._default_repo_root().is_absolute()


def test_write_fleet_inplace_roundtrip(tmp_path: Path) -> None:
    nd = tmp_path / "nhts2017"
    _write_ca_parquets(nd)
    out = dm.match_donors(_fleet(8), _ca_region(), "residential", seed=3, nhts_data_dir=nd)
    out_path = tmp_path / "fleet.parquet"
    dm._write_fleet_inplace(out, out_path)
    assert out_path.is_file()
    back = pd.read_parquet(out_path)
    assert back["nhts_donor_houseid"].dtype == "int64"
    assert back["nhts_donor_vehid"].dtype == "int32"
    assert back["match_relaxation_level"].dtype == "int8"
    assert back["borrow_used"].dtype == "bool"
    assert back["ev_id"].is_monotonic_increasing


def test_update_meta_writes_provenance(tmp_path: Path) -> None:
    nd = tmp_path / "nhts2017"
    _write_ca_parquets(nd)
    enriched = dm.match_donors(
        _fleet(8), _ca_region(), "residential", seed=3, nhts_data_dir=nd
    )
    meta_path = tmp_path / "meta.json"
    dm._update_meta(meta_path, enriched)
    assert meta_path.is_file()
    meta = json.loads(meta_path.read_text())
    assert "donor_borrow_rate" in meta
    assert meta["donor_borrow_rate"] == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# CLI: arg parser + main.
# --------------------------------------------------------------------------- #


def test_build_arg_parser_defaults() -> None:
    ns = dm._build_arg_parser().parse_args([])
    assert ns.seed == dm.MASTER_SEED_DEFAULT
    assert ns.region == "bay_area"
    assert ns.profile_type == "residential"
    assert ns.log_level == "INFO"


def test_main_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    nd = tmp_path / "nhts2017"
    _write_ca_parquets(nd)
    in_fleet = tmp_path / "in_fleet.parquet"
    _fleet(8).to_parquet(in_fleet)
    out_fleet = tmp_path / "out_fleet.parquet"
    meta_path = tmp_path / "meta.json"

    rc = dm.main(
        [
            "--region", "bay_area",
            "--profile-type", "residential",
            "--seed", "5",
            "--in-fleet", str(in_fleet),
            "--out-fleet", str(out_fleet),
            "--meta", str(meta_path),
            "--nhts-data-dir", str(nd),
            "--repo-root", str(tmp_path),
            "--log-level", "WARNING",
        ]
    )
    assert rc == 0
    assert out_fleet.is_file()
    assert meta_path.is_file()
    captured = capsys.readouterr()
    assert "M3 donor_matcher" in captured.err

    written = pd.read_parquet(out_fleet)
    assert "nhts_donor_houseid" in written.columns
    assert written["nhts_donor_houseid"].notna().all()


# --------------------------------------------------------------------------- #
# _assign_strata + apply_acs_calibration_weights error branches.
# --------------------------------------------------------------------------- #


def test_assign_strata_missing_hhsize_col_raises() -> None:
    with pytest.raises(KeyError, match="hhsize"):
        dm._assign_strata(pd.DataFrame({"hhvehcnt": [1]}))


def test_assign_strata_missing_hhvehcnt_col_raises() -> None:
    with pytest.raises(KeyError, match="hhvehcnt"):
        dm._assign_strata(pd.DataFrame({"hhsize": [1]}))


def test_apply_acs_calibration_weights_empty_hh_raises() -> None:
    expected = pd.DataFrame(
        {
            "income_tier": pd.array(["t1"], dtype="string"),
            "hhsize_band": pd.array(["1"], dtype="string"),
            "hhvehcnt_band": pd.array(["0"], dtype="string"),
            "p_expected": [1.0],
        }
    )
    with pytest.raises(ValueError, match="hh is empty"):
        dm.apply_acs_calibration_weights(pd.DataFrame(), expected)


def test_apply_acs_calibration_weights_missing_pmf_cols_raises() -> None:
    hh = pd.DataFrame(
        {
            "income_tier": pd.array(["t1"], dtype="string"),
            "hhsize_band": pd.array(["1"], dtype="string"),
            "hhvehcnt_band": pd.array(["0"], dtype="string"),
            "wthhfin": [100.0],
        }
    )
    bad_pmf = pd.DataFrame({"income_tier": pd.array(["t1"], dtype="string")})
    with pytest.raises(KeyError, match="expected_pmf missing required"):
        dm.apply_acs_calibration_weights(hh, bad_pmf)


def test_apply_acs_calibration_weights_nonpositive_total_raises() -> None:
    hh = pd.DataFrame(
        {
            "income_tier": pd.array(["t1"], dtype="string"),
            "hhsize_band": pd.array(["1"], dtype="string"),
            "hhvehcnt_band": pd.array(["0"], dtype="string"),
            "wthhfin": [0.0],
        }
    )
    expected = pd.DataFrame(
        {
            "income_tier": pd.array(["t1"], dtype="string"),
            "hhsize_band": pd.array(["1"], dtype="string"),
            "hhvehcnt_band": pd.array(["0"], dtype="string"),
            "p_expected": [1.0],
        }
    )
    with pytest.raises(ValueError, match="non-positive"):
        dm.apply_acs_calibration_weights(hh, expected)


def test_apply_acs_calibration_weights_missing_hh_cols_raises() -> None:
    hh = pd.DataFrame({"foo": [1, 2]})
    expected = pd.DataFrame(
        {
            "income_tier": pd.array(["t1"], dtype="string"),
            "hhsize_band": pd.array(["1"], dtype="string"),
            "hhvehcnt_band": pd.array(["0"], dtype="string"),
            "p_expected": [1.0],
        }
    )
    with pytest.raises(KeyError, match="hh missing required columns"):
        dm.apply_acs_calibration_weights(hh, expected)


def test_apply_acs_calibration_weights_invalid_caps_raise() -> None:
    hh = pd.DataFrame(
        {
            "income_tier": pd.array(["t1"], dtype="string"),
            "hhsize_band": pd.array(["1"], dtype="string"),
            "hhvehcnt_band": pd.array(["0"], dtype="string"),
            "wthhfin": [100.0],
        }
    )
    expected = pd.DataFrame(
        {
            "income_tier": pd.array(["t1"], dtype="string"),
            "hhsize_band": pd.array(["1"], dtype="string"),
            "hhvehcnt_band": pd.array(["0"], dtype="string"),
            "p_expected": [1.0],
        }
    )
    with pytest.raises(ValueError, match="cap_min/cap_max invalid"):
        dm.apply_acs_calibration_weights(hh, expected, cap_min=5.0, cap_max=0.2)


def test_apply_acs_calibration_weights_success_matches_expected_pmf() -> None:
    """Realised post-rake PMF matches expected within float tolerance."""
    hh = pd.DataFrame(
        {
            "houseid": pd.array(range(8), dtype="int64"),
            "income_tier": pd.array(
                ["t1", "t1", "t2", "t2", "t2", "t3", "t4", "t4"], dtype="string"
            ),
            "hhsize_band": pd.array(
                ["1", "1", "2", "2", "2", "3", "4", "4"], dtype="string"
            ),
            "hhvehcnt_band": pd.array(
                ["0", "0", "1", "1", "1", "2", "3+", "3+"], dtype="string"
            ),
            "wthhfin": pd.Series([100.0] * 8, dtype="float64"),
        }
    )
    expected = pd.DataFrame(
        {
            "income_tier": pd.array(["t1", "t2", "t3", "t4"], dtype="string"),
            "hhsize_band": pd.array(["1", "2", "3", "4"], dtype="string"),
            "hhvehcnt_band": pd.array(["0", "1", "2", "3+"], dtype="string"),
            "p_expected": [0.10, 0.50, 0.20, 0.20],
        }
    )
    out = dm.apply_acs_calibration_weights(hh, expected)
    assert "acs_cal_factor" in out.columns
    assert "acs_cal_factor" not in hh.columns  # input not mutated
    keys = ["income_tier", "hhsize_band", "hhvehcnt_band"]
    realised = out.groupby(keys, observed=True)["wthhfin"].sum().rename("w").reset_index()
    realised["p"] = realised["w"] / realised["w"].sum()
    merged = expected.merge(realised, on=keys, how="left")
    diff = (merged["p"] - merged["p_expected"]).abs().max()
    assert float(diff) < 1e-9
    attrs = out.attrs["acs_calibration"]
    assert attrs["n_cells_total"] == 4
    assert attrs["n_cells_capped"] == 0
    assert attrs["cap_fraction"] == 0.0


def test_apply_acs_calibration_weights_caps_and_warns() -> None:
    """One pathological cell clips but stays under the 5% fail threshold."""
    rows = [(f"c{i}", "x", "y") for i in range(40)] + [("hot", "x", "y")]
    df = pd.DataFrame(rows, columns=["income_tier", "hhsize_band", "hhvehcnt_band"])
    df["wthhfin"] = pd.Series([100.0] * len(df), dtype="float64")
    for c in ("income_tier", "hhsize_band", "hhvehcnt_band"):
        df[c] = df[c].astype("string")
    each = 1.0 / 41.0
    hot = each * 6.0
    cold = (1.0 - hot) / 40.0
    erows = [(f"c{i}", "x", "y", cold) for i in range(40)] + [("hot", "x", "y", hot)]
    expected = pd.DataFrame(
        erows, columns=["income_tier", "hhsize_band", "hhvehcnt_band", "p_expected"]
    )
    for c in ("income_tier", "hhsize_band", "hhvehcnt_band"):
        expected[c] = expected[c].astype("string")
    out = dm.apply_acs_calibration_weights(df, expected)
    attrs = out.attrs["acs_calibration"]
    assert attrs["n_cells_capped"] == 1
    assert attrs["n_cells_total"] == 41


def test_apply_acs_calibration_weights_too_many_capped_raises() -> None:
    hh = pd.DataFrame(
        {
            "houseid": pd.array(range(8), dtype="int64"),
            "income_tier": pd.array(
                ["t1", "t1", "t2", "t2", "t2", "t3", "t4", "t4"], dtype="string"
            ),
            "hhsize_band": pd.array(
                ["1", "1", "2", "2", "2", "3", "4", "4"], dtype="string"
            ),
            "hhvehcnt_band": pd.array(
                ["0", "0", "1", "1", "1", "2", "3+", "3+"], dtype="string"
            ),
            "wthhfin": pd.Series([100.0] * 8, dtype="float64"),
        }
    )
    expected = pd.DataFrame(
        {
            "income_tier": pd.array(["t1", "t2", "t3", "t4"], dtype="string"),
            "hhsize_band": pd.array(["1", "2", "3", "4"], dtype="string"),
            "hhvehcnt_band": pd.array(["0", "1", "2", "3+"], dtype="string"),
            "p_expected": [0.95, 0.02, 0.02, 0.01],
        }
    )
    with pytest.raises(ValueError, match="required scale-cap clipping"):
        dm.apply_acs_calibration_weights(hh, expected)


def test_assign_strata_bins_and_preserves() -> None:
    hh = pd.DataFrame(
        {
            "hhsize": pd.array([1, 2, 3, 4, 5, 7, pd.NA], dtype="Int8"),
            "hhvehcnt": pd.array([0, 1, 2, 3, 5, 7, pd.NA], dtype="Int8"),
        }
    )
    out = dm._assign_strata(hh)
    assert out["hhsize_band"].tolist()[:6] == ["1", "2", "3", "4", "5+", "5+"]
    assert out["hhvehcnt_band"].tolist()[:6] == ["0", "1", "2", "3+", "3+", "3+"]
    # NA source -> NA band.
    assert pd.isna(out["hhsize_band"].iloc[6])
    assert pd.isna(out["hhvehcnt_band"].iloc[6])


def test_assign_strata_overwrite_flag() -> None:
    hh = pd.DataFrame(
        {
            "hhsize": pd.array([2, 3], dtype="Int8"),
            "hhvehcnt": pd.array([1, 2], dtype="Int8"),
            "hhsize_band": pd.Series(["keep", "keep"], dtype="string"),
            "hhvehcnt_band": pd.Series(["keep", "keep"], dtype="string"),
        }
    )
    preserved = dm._assign_strata(hh)
    assert preserved["hhsize_band"].tolist() == ["keep", "keep"]
    overwritten = dm._assign_strata(hh, overwrite=True)
    assert overwritten["hhsize_band"].tolist() == ["2", "3"]
    assert overwritten["hhvehcnt_band"].tolist() == ["1", "2"]


# --------------------------------------------------------------------------- #
# Extra branch coverage: narrow nextgen concat, model-backoff, borrow edges.
# --------------------------------------------------------------------------- #


def test_narrow_to_region_concats_nextgen_passthrough() -> None:
    region = _ca_region()
    hh = pd.DataFrame(
        {
            "houseid": [1, 2, 90, 91],
            "hhstate": pd.array(["CA", "CA", pd.NA, pd.NA], dtype="string"),
            "hh_cbsa": pd.array(["41860", "99999", pd.NA, pd.NA], dtype="string"),
            "nhts_vintage": pd.array(
                ["2017", "2017", "2022_nextgen", "2022_nextgen"], dtype="string"
            ),
        }
    )
    out = dm.narrow_to_region(hh, region)
    # 2017 row 1 kept (41860); both nextgen rows pass through.
    assert sorted(out["houseid"].tolist()) == [1, 90, 91]


def test_model_backoff_noop_when_nothing_missing() -> None:
    veh = pd.DataFrame(
        {
            "houseid": [1, 2],
            "vehid": pd.array([1, 1], dtype="int32"),
            "make": pd.array(["Toyota", "Honda"], dtype="string"),
            "model": pd.array(["Camry", "Civic"], dtype="string"),
            "vehyear": pd.array([2018, 2019], dtype="Int16"),
            "hfuel": pd.array([-1, -1], dtype="Int8"),
            "fueltype": pd.array([1, 1], dtype="Int8"),
            "nhts_vintage": pd.array(["2017", "2017"], dtype="string"),
        }
    )
    out = dm._model_backoff_draw_from_2017(veh)
    pd.testing.assert_frame_equal(
        out.reset_index(drop=True), veh.reset_index(drop=True)
    )


def test_model_backoff_unknown_when_make_absent() -> None:
    veh = pd.DataFrame(
        {
            "houseid": [1, 90],
            "vehid": pd.array([1, 1], dtype="int32"),
            "make": pd.array(["Toyota", "Rivian"], dtype="string"),
            "model": pd.array(["Camry", pd.NA], dtype="string"),
            "vehyear": pd.array([2018, 2022], dtype="Int16"),
            "hfuel": pd.array([-1, 3], dtype="Int8"),
            "fueltype": pd.array([1, 3], dtype="Int8"),
            "nhts_vintage": pd.array(["2017", "2022_nextgen"], dtype="string"),
        }
    )
    out = dm._model_backoff_draw_from_2017(veh, rng=np.random.default_rng(0))
    assert out.loc[out["houseid"] == 90, "model"].iloc[0] == "UNKNOWN"


def test_model_backoff_empty_2017_pool_falls_to_unknown() -> None:
    """No 2017 rows at all -> every conditional pool is empty -> UNKNOWN."""
    veh = pd.DataFrame(
        {
            "houseid": [90],
            "vehid": pd.array([1], dtype="int32"),
            "make": pd.array(["Tesla"], dtype="string"),
            "model": pd.array([pd.NA], dtype="string"),
            "vehyear": pd.array([2022], dtype="Int16"),
            "hfuel": pd.array([3], dtype="Int8"),
            "fueltype": pd.array([3], dtype="Int8"),
            "nhts_vintage": pd.array(["2022_nextgen"], dtype="string"),
        }
    )
    out = dm._model_backoff_draw_from_2017(veh, rng=np.random.default_rng(0))
    assert out.loc[out["houseid"] == 90, "model"].iloc[0] == "UNKNOWN"


def test_model_backoff_make_only_fallback_single_key_groupby() -> None:
    """Make matches but year+hfuel differ -> falls to the (make,) pool.

    The single-key groupby yields a scalar key that the helper must wrap in
    a 1-tuple before lookup.
    """
    veh = pd.DataFrame(
        {
            "houseid": [1, 90],
            "vehid": pd.array([1, 1], dtype="int32"),
            "make": pd.array(["Toyota", "Toyota"], dtype="string"),
            "model": pd.array(["Camry", pd.NA], dtype="string"),
            # 2017 row year/hfuel differ from the nextgen row so the first
            # three back-off levels miss and only (make,) hits.
            "vehyear": pd.array([2010, 2022], dtype="Int16"),
            "hfuel": pd.array([4, 3], dtype="Int8"),
            "fueltype": pd.array([4, 3], dtype="Int8"),
            "nhts_vintage": pd.array(["2017", "2022_nextgen"], dtype="string"),
        }
    )
    out = dm._model_backoff_draw_from_2017(veh, rng=np.random.default_rng(0))
    assert out.loc[out["houseid"] == 90, "model"].iloc[0] == "Camry"


def test_apply_acs_calibration_weights_empty_pmf_raises() -> None:
    hh = pd.DataFrame(
        {
            "income_tier": pd.array(["t1"], dtype="string"),
            "hhsize_band": pd.array(["1"], dtype="string"),
            "hhvehcnt_band": pd.array(["0"], dtype="string"),
            "wthhfin": [100.0],
        }
    )
    empty = pd.DataFrame(
        columns=["income_tier", "hhsize_band", "hhvehcnt_band", "p_expected"]
    )
    with pytest.raises(ValueError, match="expected_pmf is empty"):
        dm.apply_acs_calibration_weights(hh, empty)


def test_match_donors_empty_pool_after_vehicle_coverage_raises(tmp_path: Path) -> None:
    """Households with no vehicles at all -> empty pool raise."""
    nd = tmp_path / "nhts2017"
    nd.mkdir(parents=True)
    hh = pd.DataFrame(
        {
            "houseid": pd.array([1, 2], dtype="int64"),
            "hhstate": pd.array(["CA", "CA"], dtype="string"),
            "hh_cbsa": pd.array(["41860", "41860"], dtype="string"),
            "urbrur": pd.array([1, 1], dtype="Int8"),
            "hhfaminc": pd.array([6, 6], dtype="Int8"),
            "hhsize": pd.array([2, 2], dtype="Int8"),
            "hhvehcnt": pd.array([2, 2], dtype="Int8"),
            "wthhfin": pd.Series([100.0, 100.0], dtype="float64"),
        }
    )
    # Vehicle frame references houseids not present in hh -> coverage filter
    # empties the pool.
    veh = pd.DataFrame(
        {
            "houseid": pd.array([999], dtype="int64"),
            "vehid": pd.array([1], dtype="int32"),
            "vehtype": pd.array([1], dtype="Int8"),
            "make": pd.array(["Tesla"], dtype="string"),
            "model": pd.array(["Model3"], dtype="string"),
            "vehyear": pd.array([2020], dtype="Int16"),
            "annmiles": pd.array([8000.0], dtype="float32"),
            "hfuel": pd.array([3], dtype="Int8"),
        }
    )
    hh.to_parquet(nd / "nhts2017_ca_hh.parquet")
    veh.to_parquet(nd / "nhts2017_ca_veh.parquet")
    with pytest.raises(RuntimeError, match="empty after vehicle-coverage"):
        dm.match_donors(_fleet(2), _ca_region(), "residential", seed=1, nhts_data_dir=nd)


def test_match_donors_uniform_probs_when_weights_nonpositive(tmp_path: Path) -> None:
    """All wthhfin == 0 forces the uniform-probability fallback in the matcher."""
    nd = tmp_path / "nhts2017"
    nd.mkdir(parents=True)
    hh = _ca_hh_frame()
    hh["wthhfin"] = pd.Series([0.0, 0.0, 0.0, 0.0], dtype="float64")
    hh.to_parquet(nd / "nhts2017_ca_hh.parquet")
    _ca_veh_frame().to_parquet(nd / "nhts2017_ca_veh.parquet")
    out = dm.match_donors(_fleet(6), _ca_region(), "residential", seed=9, nhts_data_dir=nd)
    assert out["nhts_donor_houseid"].notna().all()


def test_borrow_breaks_when_capacity_exhausted(tmp_path: Path) -> None:
    """Two borrow states; the first fills the share-cap so the second loop
    iteration hits ``remaining_capacity <= 0`` and breaks."""
    nd = tmp_path / "nhts2017"
    nd.mkdir(parents=True)
    # 4 MA home rows (capacity = int(0.25*4/0.75) = 1), plus NH + VT donors.
    states = ["MA"] * 4 + ["NH"] * 5 + ["VT"] * 5
    ids = list(range(1, len(states) + 1))
    pd.DataFrame(
        {
            "HOUSEID": ids,
            "HHSTATE": states,
            "HH_CBSA": ["14460"] * len(ids),
            "URBRUR": [1] * len(ids),
            "HHFAMINC": [6] * len(ids),
            "HHSIZE": [2] * len(ids),
            "HHVEHCNT": [2] * len(ids),
            "WTHHFIN": [100.0] * len(ids),
        }
    ).to_csv(nd / "hhpub.csv", index=False)
    pd.DataFrame(
        {
            "HOUSEID": ids,
            "VEHID": [1] * len(ids),
            "VEHTYPE": [1] * len(ids),
            "MAKE": ["Tesla"] * len(ids),
            "MODEL": ["Model3"] * len(ids),
            "VEHYEAR": [2020] * len(ids),
            "ANNMILES": [1000.0] * len(ids),
            "HFUEL": [3] * len(ids),
        }
    ).to_csv(nd / "vehpub.csv", index=False)

    region = _ma_workplace_region(workplace_donor_borrow_states=("NH", "VT"))
    hh = pd.DataFrame(
        {
            "houseid": [1, 2, 3, 4],
            "hhstate": pd.array(["MA"] * 4, dtype="string"),
            "wthhfin": pd.Series([100.0] * 4, dtype="float64"),
        }
    )
    veh = pd.DataFrame(
        {
            "houseid": [1, 2, 3, 4],
            "vehid": pd.array([1, 1, 1, 1], dtype="int32"),
            "annmiles": pd.array([1000.0] * 4, dtype="float32"),
            "hfuel": pd.array([3, 3, 3, 3], dtype="Int8"),
        }
    )
    _, _, n_borrowed = dm.borrow_from_neighbours(
        hh, veh, region, "workplace", nd, pool_threshold=100, share_cap=0.25,
    )
    # Capacity is 1; the first state fills it, the second iteration breaks.
    assert n_borrowed == 1


def test_borrow_skips_empty_state_subpool(tmp_path: Path) -> None:
    """A borrow state with no households after the annmiles filter is skipped."""
    nd = tmp_path / "nhts2017"
    # NH rows all have annmiles >= 3600 -> filtered out for workplace.
    _write_national_csvs(nd, n_home=2, borrow_state="NH", n_borrow=5, annmiles=9000.0)
    hh = pd.DataFrame(
        {
            "houseid": [1, 2],
            "hhstate": pd.array(["MA", "MA"], dtype="string"),
            "wthhfin": pd.Series([100.0, 100.0], dtype="float64"),
        }
    )
    veh = pd.DataFrame(
        {
            "houseid": [1, 2],
            "vehid": pd.array([1, 1], dtype="int32"),
            "annmiles": pd.array([1000.0, 1000.0], dtype="float32"),
            "hfuel": pd.array([3, 3], dtype="Int8"),
        }
    )
    _, _, n_borrowed = dm.borrow_from_neighbours(
        hh, veh, _ma_workplace_region(), "workplace", nd,
        pool_threshold=100, share_cap=0.9,
    )
    assert n_borrowed == 0
