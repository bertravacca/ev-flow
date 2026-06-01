"""Unit tests for ``pev_synth.nhts_loader`` (Module M1, Wave 1, Dev A).

Tests use a synthetic four-CSV bundle stuffed into a local zip and served
via a ``file://`` URL. This avoids hitting the 84 MB ORNL download in CI
while exercising the full pipeline: download → unpack → CA filter →
schema enforcement → parquet write → cached re-read.

References:
- PM dev brief Module 1 §9 (unit-test list).
- Methodology plan §2.1 (NHTS inputs) and §4.6 (household constraints).
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from pev_synth.nhts_loader import (
    DEFAULT_CACHE_REL,
    HFUEL_BEV,
    HFUEL_BIODIESEL,
    HFUEL_HEV_NON_PLUGIN,
    HFUEL_PHEV,
    HH_DTYPES,
    NHTS_CSV_MEMBERS,
    NHTS_ZIP_URL,
    PARQUET_BASENAMES,
    PER_DTYPES,
    TRIP_DTYPES,
    VEH_DTYPES,
    _build_arg_parser,
    _compute_dwell_minutes,
    _default_repo_root,
    _sha256_of_file,
    load_nhts_2017_ca,
)

# --------------------------------------------------------------------------- #
# Synthetic fixture
# --------------------------------------------------------------------------- #


def _make_synthetic_hhpub() -> str:
    """5 CA households + 4 non-CA + 1 with sentinel income."""
    rows = [
        # houseid, hhstate, hhstfips, hh_cbsa, urbrur, hhfaminc, hhsize,
        # hhvehcnt, homeown, wthhfin, cdivmsar
        ("10000001", "CA", 6, "41860", 1, 7, 2, 2, 1, 100.5, 53),
        ("10000002", "CA", 6, "41940", 1, 5, 3, 2, 2, 200.0, 53),
        ("10000003", "CA", 6, "XXXXX", 2, -7, 4, 3, 1, 150.25, 99),
        ("10000004", "CA", 6, "42220", 1, 9, 1, 1, 2, 75.0, 53),
        ("10000005", "CA", 6, "46700", 1, 3, 5, 4, 3, 125.5, 53),
        ("20000001", "TX", 48, "26420", 1, 8, 2, 2, 1, 110.0, 31),
        ("20000002", "NY", 36, "35620", 1, 11, 4, 3, 1, 90.0, 13),
        ("20000003", "WA", 53, "42660", 1, 6, 2, 1, 2, 80.0, 91),
        ("20000004", "OR", 41, "38900", 1, 4, 3, 2, 1, 105.0, 91),
    ]
    header = (
        '"HOUSEID","HHSTATE","HHSTFIPS","HH_CBSA","URBRUR","HHFAMINC",'
        '"HHSIZE","HHVEHCNT","HOMEOWN","WTHHFIN","CDIVMSAR"\n'
    )
    body = "\n".join(
        f'"{r[0]}","{r[1]}",{r[2]},"{r[3]}",{r[4]},{r[5]},{r[6]},{r[7]},{r[8]},{r[9]},{r[10]}'
        for r in rows
    )
    return header + body + "\n"


def _make_synthetic_perpub() -> str:
    rows = [
        # houseid, personid, r_age, r_sex, worker, driver, wtperfin
        ("10000001", 1, 45, 1, 1, 1, 100.5),
        ("10000001", 2, 42, 2, 1, 1, 100.5),
        ("10000002", 1, 30, 1, 1, 1, 200.0),
        ("10000002", 2, 28, 2, 1, 1, 200.0),
        ("10000002", 3, 5, 2, 2, 2, 200.0),
        ("10000003", 1, -7, -7, -7, -7, 150.25),  # all sentinels
        ("10000004", 1, 65, 1, 2, 1, 75.0),
        ("10000005", 1, 50, 2, 1, 1, 125.5),
        ("20000001", 1, 40, 1, 1, 1, 110.0),
        ("20000002", 1, 55, 2, 1, 1, 90.0),
    ]
    header = '"HOUSEID","PERSONID","R_AGE","R_SEX","WORKER","DRIVER","WTPERFIN"\n'
    body = "\n".join(f'"{r[0]}",{r[1]},{r[2]},{r[3]},{r[4]},{r[5]},{r[6]}' for r in rows)
    return header + body + "\n"


def _make_synthetic_vehpub() -> str:
    rows = [
        # houseid, vehid, vehtype, fueltype, make, model, vehyear, annmiles, hfuel
        ("10000001", 1, 1, 1, "Toyota", "Camry", 2018, 12000, -1),
        ("10000001", 2, 3, 3, "Tesla", "Model Y", 2022, 11000, -1),  # BEV
        ("10000002", 1, 2, 1, "Honda", "Odyssey", 2019, 9000, -1),
        ("10000002", 2, 5, 1, "Yamaha", "MT-07", 2020, 3000, -1),  # motorcycle → dropped
        ("10000003", 1, 4, 5, "Ford", "Escape PHEV", 2021, 8500, -1),  # PHEV
        ("10000004", 1, 1, 1, "Honda", "Civic", 2017, 14000, -1),
        ("10000005", 1, 3, 1, "Subaru", "Outback", 2018, 11500, -1),
        ("10000005", 2, 4, 1, "Ford", "F-150", 2020, 8000, -1),
        ("20000001", 1, 1, 1, "Toyota", "Corolla", 2019, 15000, -1),
        ("20000002", 1, 3, 3, "Tesla", "Model 3", 2021, 10000, -1),  # non-CA BEV
    ]
    header = (
        '"HOUSEID","VEHID","VEHTYPE","FUELTYPE","MAKE","MODEL",'
        '"VEHYEAR","ANNMILES","HFUEL"\n'
    )
    body = "\n".join(
        f'"{r[0]}",{r[1]},{r[2]},{r[3]},"{r[4]}","{r[5]}",{r[6]},{r[7]},{r[8]}'
        for r in rows
    )
    return header + body + "\n"


def _make_synthetic_trippub() -> str:
    """Two trips on same personid-day to exercise dwell_min computation.

    HH 10000001, person 1: trip 1 ends 0815 → trip 2 starts 1730 → dwell 555.
    """
    rows = [
        # houseid, personid, vehid, tdtrpnum, strttime, endtime, trpmiles,
        # trvlcmin, whyfrom, whyto, tdwknd, travday, wttrdfin
        ("10000001", 1, 1, 1, 800, 815, 5.2, 15, 1, 3, 2, 3, 100.5),
        ("10000001", 1, 1, 2, 1730, 1800, 6.0, 30, 3, 1, 2, 3, 100.5),
        ("10000002", 1, 1, 1, 700, 730, 8.0, 30, 1, 3, 2, 4, 200.0),
        ("10000002", 1, 1, 2, 1700, 1745, 8.0, 45, 3, 1, 2, 4, 200.0),
        ("10000005", 1, 1, 1, 900, 920, 4.0, 20, 1, 11, 1, 7, 125.5),
        # Non-CA trip — should be excluded.
        ("20000001", 1, 1, 1, 800, 830, 10.0, 30, 1, 3, 2, 3, 110.0),
    ]
    header = (
        '"HOUSEID","PERSONID","VEHID","TDTRPNUM","STRTTIME","ENDTIME",'
        '"TRPMILES","TRVLCMIN","WHYFROM","WHYTO","TDWKND","TRAVDAY","WTTRDFIN"\n'
    )
    body = "\n".join(
        f'"{r[0]}",{r[1]},{r[2]},{r[3]},{r[4]},{r[5]},{r[6]},{r[7]},{r[8]},{r[9]},{r[10]},{r[11]},{r[12]}'
        for r in rows
    )
    return header + body + "\n"


@pytest.fixture(scope="module")
def synthetic_zip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build a synthetic ORNL-style zip on disk and return its path."""
    tmp = tmp_path_factory.mktemp("nhts_zip_src")
    zip_path = tmp / "csv.zip"
    contents = {
        "hhpub.csv": _make_synthetic_hhpub(),
        "perpub.csv": _make_synthetic_perpub(),
        "vehpub.csv": _make_synthetic_vehpub(),
        "trippub.csv": _make_synthetic_trippub(),
    }
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, body in contents.items():
            zf.writestr(name, body)
    return zip_path


def _file_url(p: Path) -> str:
    return p.resolve().as_uri()


# --------------------------------------------------------------------------- #
# Required tests (PM brief §9)
# --------------------------------------------------------------------------- #


def test_default_download_url_is_canonical_ornl_zip() -> None:
    """`test_zip_url_unchanged` analogue (PM brief §9 test 1)."""
    assert NHTS_ZIP_URL == "https://nhts.ornl.gov/assets/2016/download/csv.zip"
    assert NHTS_CSV_MEMBERS == ("hhpub.csv", "perpub.csv", "vehpub.csv", "trippub.csv")


def test_load_nhts_2017_ca_filters_to_california_and_propagates(
    tmp_path: Path, synthetic_zip: Path
) -> None:
    """California filter (HHSTATE == 'CA') propagates to all four tables.

    PM brief §9 test 3 (CA scope analogue) + orchestrator-resolved contract.
    """
    cache = tmp_path / "cache"
    data = load_nhts_2017_ca(cache_dir=cache, download_url=_file_url(synthetic_zip))

    # 5 CA households in the synthetic fixture.
    assert len(data.hh) == 5
    assert set(data.hh["hhstate"].astype("string").tolist()) == {"CA"}

    ca_ids = set(data.hh["houseid"].astype("int64").tolist())
    # Persons / vehicles / trips must all be subsets of CA houseids.
    assert set(data.per["houseid"].astype("int64").tolist()).issubset(ca_ids)
    assert set(data.veh["houseid"].astype("int64").tolist()).issubset(ca_ids)
    assert set(data.trip["houseid"].astype("int64").tolist()).issubset(ca_ids)

    # Concrete counts from the fixture.
    assert len(data.per) == 8  # 7 CA persons + houseid 10000003's lone sentinel-laden person
    # Veh: 1+1 + 1+(0, motorcycle dropped) + 1 + 1 + 1+1 = 7 surviving CA rows
    assert len(data.veh) == 7
    assert len(data.trip) == 5  # 5 CA trips


def test_sentinels_converted_to_na_in_income_and_age(
    tmp_path: Path, synthetic_zip: Path
) -> None:
    """PM brief §9 test 2: −7/−8/−9 in HHFAMINC, R_AGE → pd.NA."""
    cache = tmp_path / "cache"
    data = load_nhts_2017_ca(cache_dir=cache, download_url=_file_url(synthetic_zip))

    # Household 10000003 has hhfaminc = -7 in the fixture → NA in output.
    hh3 = data.hh.loc[data.hh["houseid"] == 10000003].iloc[0]
    assert pd.isna(hh3["hhfaminc"])

    # Person row for 10000003 has all-sentinel age/sex/worker/driver → all NA.
    per_3 = data.per.loc[data.per["houseid"] == 10000003].iloc[0]
    assert pd.isna(per_3["r_age"])
    assert pd.isna(per_3["r_sex"])
    assert pd.isna(per_3["worker"])
    assert pd.isna(per_3["driver"])


def test_vehtype_filter_drops_motorcycles(tmp_path: Path, synthetic_zip: Path) -> None:
    """PM brief §9 test 6: VEHTYPE=5 (motorcycle) rows are dropped."""
    cache = tmp_path / "cache"
    data = load_nhts_2017_ca(cache_dir=cache, download_url=_file_url(synthetic_zip))
    assert set(data.veh["vehtype"].dropna().astype("int64").tolist()).issubset({1, 2, 3, 4})
    # Specifically: HH 10000002 had a vehid=2 motorcycle; veh table must not
    # contain (10000002, 2).
    assert not (
        (data.veh["houseid"] == 10000002) & (data.veh["vehid"] == 2)
    ).any()


def test_dwell_min_computed_correctly_within_personid_day() -> None:
    """PM brief §9 test 5: 2-trip personid (end 0815, start 1730) → 555 min.

    Last trip of the day → NaN.
    """
    df = pd.DataFrame(
        {
            "travday": [3, 3],
            "strttime": [800, 1730],
            "endtime": [815, 1800],
        }
    )
    dwell = _compute_dwell_minutes(df)
    # Trip 1: 17:30 − 08:15 = 9 h 15 m = 555 min.
    assert int(dwell.iloc[0]) == 555
    # Trip 2 is the last trip of the day.
    assert pd.isna(dwell.iloc[1])


def test_schemas_match_plan_section_8(tmp_path: Path, synthetic_zip: Path) -> None:
    """PM brief §8 acceptance #4: dtype contract on all four parquets."""
    cache = tmp_path / "cache"
    load_nhts_2017_ca(cache_dir=cache, download_url=_file_url(synthetic_zip))

    hh = pd.read_parquet(cache / PARQUET_BASENAMES["hh"])
    per = pd.read_parquet(cache / PARQUET_BASENAMES["per"])
    veh = pd.read_parquet(cache / PARQUET_BASENAMES["veh"])
    trip = pd.read_parquet(cache / PARQUET_BASENAMES["trip"])

    assert list(hh.columns) == list(HH_DTYPES.keys())
    assert list(per.columns) == list(PER_DTYPES.keys())
    assert list(veh.columns) == list(VEH_DTYPES.keys())
    assert list(trip.columns) == list(TRIP_DTYPES.keys())

    for col, dt in HH_DTYPES.items():
        assert str(hh[col].dtype) == dt, f"hh.{col}: expected {dt}, got {hh[col].dtype}"
    for col, dt in PER_DTYPES.items():
        assert str(per[col].dtype) == dt, f"per.{col}: expected {dt}, got {per[col].dtype}"
    for col, dt in VEH_DTYPES.items():
        assert str(veh[col].dtype) == dt, f"veh.{col}: expected {dt}, got {veh[col].dtype}"
    for col, dt in TRIP_DTYPES.items():
        assert str(trip[col].dtype) == dt, f"trip.{col}: expected {dt}, got {trip[col].dtype}"


def test_idempotent_second_call_returns_cached(tmp_path: Path, synthetic_zip: Path) -> None:
    """PM brief §9 test 7: rerun is byte-identical (SHA-256 match per file)."""
    cache = tmp_path / "cache"

    load_nhts_2017_ca(cache_dir=cache, download_url=_file_url(synthetic_zip))
    first = {
        name: _sha256_of_file(cache / fname)
        for name, fname in PARQUET_BASENAMES.items()
    }

    # Second call — outputs already on disk; the loader must return the cached
    # frames without overwriting (so SHA-256 is unchanged).
    load_nhts_2017_ca(cache_dir=cache, download_url=_file_url(synthetic_zip))
    second = {
        name: _sha256_of_file(cache / fname)
        for name, fname in PARQUET_BASENAMES.items()
    }
    assert first == second


def test_manifest_contains_required_fields(tmp_path: Path, synthetic_zip: Path) -> None:
    """PM brief §8 acceptance #5: manifest captures SHA-256s and filter rule."""
    cache = tmp_path / "cache"
    data = load_nhts_2017_ca(cache_dir=cache, download_url=_file_url(synthetic_zip))
    m = data.manifest

    # Required top-level keys.
    for key in (
        "url",
        "zip_sha256",
        "csv_sha256",
        "fetch_timestamp_utc",
        "filter_rule",
        "ca_subset_size",
        "total_size",
        "methodology_version",
        "master_seed",
    ):
        assert key in m, f"manifest missing key {key!r}"

    # SHA-256 map covers all four CSV members.
    assert set(m["csv_sha256"].keys()) == set(NHTS_CSV_MEMBERS)
    for h in m["csv_sha256"].values():
        # 64-hex sha256.
        assert isinstance(h, str) and len(h) == 64 and all(c in "0123456789abcdef" for c in h)

    # Confirm zip sha256 matches an independent computation.
    assert m["zip_sha256"] == hashlib.sha256(synthetic_zip.read_bytes()).hexdigest()

    # Orchestrator-resolved contract values.
    assert m["methodology_version"] == "v1.0.0"
    assert m["master_seed"] == 20260518
    assert "HHSTATE == 'CA'" in m["filter_rule"]


def test_manifest_round_trips_through_disk(tmp_path: Path, synthetic_zip: Path) -> None:
    """Manifest JSON on disk parses back to the in-memory manifest dict."""
    cache = tmp_path / "cache"
    data = load_nhts_2017_ca(cache_dir=cache, download_url=_file_url(synthetic_zip))
    with (cache / "nhts2017_download_manifest.json").open() as fh:
        on_disk = json.load(fh)
    assert on_disk == data.manifest


# --------------------------------------------------------------------------- #
# Fix-up tests (Wave 1 expert review §M1.4 + PM review B4)
# --------------------------------------------------------------------------- #


def test_hfuel_constants_match_nhts_2017_codebook_v1_2() -> None:
    """HFUEL constants must match NHTS 2017 Codebook v1.2 p.55 exactly.

    HFUEL semantics (codebook p.55):
        01 = Biodiesel
        02 = Plug-in Hybrid (PHEV)
        03 = Electric (BEV)
        04 = Hybrid (gas/electric, not plug-in)

    Additionally, the cached CA parquet must show a non-trivial PHEV and BEV
    donor cohort (≥ 240 PHEV rows, ≥ 280 BEV rows) — the empirical CA totals
    are 251 PHEV rows / 288 BEV rows; we leave a small margin.
    """
    assert HFUEL_BIODIESEL == 1
    assert HFUEL_PHEV == 2
    assert HFUEL_BEV == 3
    assert HFUEL_HEV_NON_PLUGIN == 4
    assert {HFUEL_BIODIESEL, HFUEL_PHEV, HFUEL_BEV, HFUEL_HEV_NON_PLUGIN} == {1, 2, 3, 4}

    # The cached CA parquet is committed under the study repo root. Skip the
    # distribution check if the artefact is not available (e.g. fresh clone
    # before M1 has been run end-to-end).
    repo_root = _default_repo_root()
    parquet = repo_root / DEFAULT_CACHE_REL / PARQUET_BASENAMES["veh"]
    if not parquet.is_file():
        pytest.skip(f"cached CA vehicle parquet not present at {parquet}")

    veh = pd.read_parquet(parquet)
    phev_rows = int((veh["hfuel"] == HFUEL_PHEV).sum())
    bev_rows = int((veh["hfuel"] == HFUEL_BEV).sum())
    assert phev_rows >= 240, f"expected >= 240 PHEV rows in CA, got {phev_rows}"
    assert bev_rows >= 280, f"expected >= 280 BEV rows in CA, got {bev_rows}"


def test_cli_repo_root_default_resolves_to_repo() -> None:
    """`--repo-root` default must resolve to the study repo root.

    The default is ``Path(__file__).resolve().parents[2]`` (matches M2's
    ``vehicle_archetypes`` pattern), i.e. two parents up from
    ``src/pev_synth/nhts_loader.py``. We assert structural properties of
    the resolved path rather than a hard-coded absolute string so the test
    survives repo relocation.
    """
    parser = _build_arg_parser()
    args = parser.parse_args([])
    # The parser default for --repo-root is None; main() then falls back to
    # _default_repo_root(). The contract under test is that fallback.
    assert args.repo_root is None
    assert args.cache_dir is None

    repo_root = _default_repo_root()
    assert repo_root.is_absolute()
    # The repo root must contain the source package and the data directory.
    assert (repo_root / "src" / "pev_synth" / "nhts_loader.py").is_file(), (
        f"_default_repo_root() returned {repo_root}, which does not contain "
        "src/pev_synth/nhts_loader.py — the --repo-root anchor is wrong."
    )
    # The default cache dir must resolve under repo_root, not under cwd.
    default_cache = repo_root / DEFAULT_CACHE_REL
    assert default_cache.is_absolute()
    assert str(default_cache).endswith("data/pev/raw/nhts2017")
