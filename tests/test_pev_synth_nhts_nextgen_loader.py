"""Unit tests for ``pev_synth.nhts_nextgen_loader`` (Module M1b, Dev D5).

Tests use a synthetic four-CSV bundle stuffed into a local zip and
served via a ``file://`` URL — DO NOT hit the live ORNL endpoint in CI.
Coverage:

* canonical URL + member list constants;
* schema deltas vs 2017 (absent ``hhstate``/``hh_cbsa``/``hhstfips``/``model``);
* renamed columns (``r_sex_imp`` -> ``r_sex``, ``dwell_time`` -> ``dwell_min``);
* sentinel coercion (-1/-7/-8/-9 -> NA where appropriate);
* ``nhts_vintage = "2022_nextgen"`` injected on every row of every table;
* manifest content + parquet round-trip;
* CLI ``--add-on`` stub raises ``NotImplementedError`` (v3.1 deferral).
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import pandas as pd
import pytest

_CODE_DIR = Path(__file__).resolve().parents[1] / "src"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from pev_synth.nhts_nextgen_loader import (  # noqa: E402
    NEXTGEN_PARQUET_BASENAMES,
    NHTS_NEXTGEN_CSV_MEMBERS,
    NHTS_NEXTGEN_MANIFEST_BASENAME,
    NHTS_NEXTGEN_VINTAGE,
    NHTS_NEXTGEN_ZIP_URL,
    _build_arg_parser,
    _sha256_of_file,
    load_nhts_nextgen_2022,
    main,
)

# --------------------------------------------------------------------------- #
# Synthetic fixture
# --------------------------------------------------------------------------- #


def _make_synthetic_hhv2pub() -> str:
    """4 NextGen 2022 households. Note the public-file column set: no
    HHSTATE/HHSTFIPS/HH_CBSA. CDIVMSAR + STRATUMID are kept."""
    rows = [
        # HOUSEID, CDIVMSAR, URBRUR, HHFAMINC, HHSIZE, HHVEHCNT, HOMEOWN,
        # WTHHFIN, STRATUMID
        ("30000001", 53, 1, 7, 2, 2, 1, 1000.5, 1),
        ("30000002", 53, 1, 5, 3, 2, 2, 2000.0, 1),
        ("30000003", 99, 2, -7, 4, 3, 1, 1500.25, 2),  # sentinel income
        ("30000004", 31, 1, 11, 1, 1, 2, 750.0, 3),
    ]
    header = (
        '"HOUSEID","CDIVMSAR","URBRUR","HHFAMINC","HHSIZE","HHVEHCNT",'
        '"HOMEOWN","WTHHFIN","STRATUMID"\n'
    )
    body = "\n".join(
        f'"{r[0]}",{r[1]},{r[2]},{r[3]},{r[4]},{r[5]},{r[6]},{r[7]},{r[8]}'
        for r in rows
    )
    return header + body + "\n"


def _make_synthetic_perv2pub() -> str:
    """R_SEX is published as R_SEX_IMP in NextGen 2022."""
    rows = [
        # HOUSEID, PERSONID, R_AGE, R_SEX_IMP, WORKER, DRIVER, WTPERFIN
        ("30000001", 1, 45, 1, 1, 1, 1000.5),
        ("30000001", 2, 42, 2, 1, 1, 1000.5),
        ("30000002", 1, 30, 1, 1, 1, 2000.0),
        ("30000003", 1, -7, -7, -7, -7, 1500.25),  # all sentinels
        ("30000004", 1, 65, 1, 2, 1, 750.0),
    ]
    header = '"HOUSEID","PERSONID","R_AGE","R_SEX_IMP","WORKER","DRIVER","WTPERFIN"\n'
    body = "\n".join(
        f'"{r[0]}",{r[1]},{r[2]},{r[3]},{r[4]},{r[5]},{r[6]}' for r in rows
    )
    return header + body + "\n"


def _make_synthetic_vehv2pub() -> str:
    """MODEL is intentionally absent (NextGen 2022 privacy suppression)."""
    rows = [
        # HOUSEID, VEHID, VEHTYPE, FUELTYPE, MAKE, VEHYEAR, ANNMILES, HFUEL
        ("30000001", 1, 1, 1, "Toyota", 2018, 12000, -1),
        ("30000001", 2, 3, 3, "Tesla", 2022, 11000, 3),  # BEV
        ("30000002", 1, 2, 1, "Honda", 2019, 9000, -1),
        ("30000002", 2, 5, 1, "Yamaha", 2020, 3000, -1),  # motorcycle -> dropped
        ("30000003", 1, 4, 5, "Ford", 2021, 8500, 2),  # PHEV
        ("30000004", 1, 1, 1, "Honda", 2017, 14000, -1),
    ]
    header = (
        '"HOUSEID","VEHID","VEHTYPE","FUELTYPE","MAKE",'
        '"VEHYEAR","ANNMILES","HFUEL"\n'
    )
    body = "\n".join(
        f'"{r[0]}",{r[1]},{r[2]},{r[3]},"{r[4]}",{r[5]},{r[6]},{r[7]}'
        for r in rows
    )
    return header + body + "\n"


def _make_synthetic_tripv2pub() -> str:
    """DWELL_MIN is named DWELL_TIME in NextGen 2022; native WHYTRP1S + TRIPMODE."""
    rows = [
        # HOUSEID, PERSONID, VEHID, TDTRPNUM, STRTTIME, ENDTIME, TRPMILES,
        # TRVLCMIN, WHYFROM, WHYTO, TDWKND, TRAVDAY, WTTRDFIN, DWELL_TIME,
        # WHYTRP1S, TRIPMODE
        ("30000001", 1, 1, 1, 800, 815, 5.2, 15, 1, 3, 2, 3, 100.5, 555, 10, 3),
        ("30000001", 1, 1, 2, 1730, 1800, 6.0, 30, 3, 1, 2, 3, 100.5, -9, 1, 3),
        ("30000002", 1, 1, 1, 700, 730, 8.0, 30, 1, 3, 2, 4, 200.0, 600, 10, 3),
        ("30000004", 1, 1, 1, 900, 920, 4.0, 20, 1, 11, 1, 7, 125.5, -1, 50, 3),
    ]
    header = (
        '"HOUSEID","PERSONID","VEHID","TDTRPNUM","STRTTIME","ENDTIME",'
        '"TRPMILES","TRVLCMIN","WHYFROM","WHYTO","TDWKND","TRAVDAY",'
        '"WTTRDFIN","DWELL_TIME","WHYTRP1S","TRIPMODE"\n'
    )
    body = "\n".join(
        f'"{r[0]}",{r[1]},{r[2]},{r[3]},{r[4]},{r[5]},{r[6]},{r[7]},{r[8]},'
        f"{r[9]},{r[10]},{r[11]},{r[12]},{r[13]},{r[14]},{r[15]}"
        for r in rows
    )
    return header + body + "\n"


@pytest.fixture(scope="module")
def synthetic_nextgen_zip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp = tmp_path_factory.mktemp("nhts_nextgen_zip_src")
    zip_path = tmp / "csv.zip"
    contents = {
        "hhv2pub.csv": _make_synthetic_hhv2pub(),
        "perv2pub.csv": _make_synthetic_perv2pub(),
        "vehv2pub.csv": _make_synthetic_vehv2pub(),
        "tripv2pub.csv": _make_synthetic_tripv2pub(),
    }
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, body in contents.items():
            zf.writestr(name, body)
    return zip_path


def _file_url(p: Path) -> str:
    return p.resolve().as_uri()


# --------------------------------------------------------------------------- #
# Required tests
# --------------------------------------------------------------------------- #


def test_canonical_url_and_csv_members_unchanged() -> None:
    """Pinned URL + member list are part of the v3.0 contract."""
    assert NHTS_NEXTGEN_ZIP_URL == "https://nhts.ornl.gov/assets/2022/download/csv.zip"
    assert NHTS_NEXTGEN_CSV_MEMBERS == (
        "hhv2pub.csv",
        "perv2pub.csv",
        "vehv2pub.csv",
        "tripv2pub.csv",
    )
    assert NHTS_NEXTGEN_VINTAGE == "2022_nextgen"


def test_loader_writes_four_parquets_and_manifest(
    tmp_path: Path, synthetic_nextgen_zip: Path
) -> None:
    cache = tmp_path / "nextgen_cache"
    data = load_nhts_nextgen_2022(
        cache_dir=cache, download_url=_file_url(synthetic_nextgen_zip)
    )
    for key, basename in NEXTGEN_PARQUET_BASENAMES.items():
        assert (cache / basename).is_file(), f"missing parquet for {key}"
    assert (cache / NHTS_NEXTGEN_MANIFEST_BASENAME).is_file()

    # Counts derived from the synthetic fixture.
    assert len(data.hh) == 4
    # Persons: HH 30000001 contributes 2; rest 1 each -> 5.
    assert len(data.per) == 5
    # Vehicles: 6 raw - 1 motorcycle (vehtype=5) - 0 (PHEV row vehtype=4 OK) = 5.
    assert len(data.veh) == 5
    assert len(data.trip) == 4


def test_absent_columns_inserted_as_na(
    tmp_path: Path, synthetic_nextgen_zip: Path
) -> None:
    """HHSTATE / HHSTFIPS / HH_CBSA (HH) + MODEL (VEH) are NA placeholders."""
    cache = tmp_path / "cache"
    data = load_nhts_nextgen_2022(
        cache_dir=cache, download_url=_file_url(synthetic_nextgen_zip)
    )
    for col in ("hhstate", "hhstfips", "hh_cbsa"):
        assert col in data.hh.columns, f"hh missing placeholder column {col}"
        assert data.hh[col].isna().all(), f"hh.{col} expected all-NA"
    assert "model" in data.veh.columns
    assert data.veh["model"].isna().all(), "veh.model expected all-NA"


def test_renamed_columns_r_sex_and_dwell_min(
    tmp_path: Path, synthetic_nextgen_zip: Path
) -> None:
    cache = tmp_path / "cache"
    data = load_nhts_nextgen_2022(
        cache_dir=cache, download_url=_file_url(synthetic_nextgen_zip)
    )
    # PER: R_SEX_IMP -> r_sex.
    assert "r_sex" in data.per.columns
    assert "r_sex_imp" not in data.per.columns
    # TRIP: DWELL_TIME -> dwell_min.
    assert "dwell_min" in data.trip.columns
    assert "dwell_time" not in data.trip.columns


def test_nhts_vintage_stamped_on_every_row_of_every_table(
    tmp_path: Path, synthetic_nextgen_zip: Path
) -> None:
    cache = tmp_path / "cache"
    data = load_nhts_nextgen_2022(
        cache_dir=cache, download_url=_file_url(synthetic_nextgen_zip)
    )
    for name, df in (
        ("hh", data.hh),
        ("per", data.per),
        ("veh", data.veh),
        ("trip", data.trip),
    ):
        assert "nhts_vintage" in df.columns, f"{name} missing nhts_vintage column"
        unique_vals = set(df["nhts_vintage"].astype("string").dropna().tolist())
        assert unique_vals == {NHTS_NEXTGEN_VINTAGE}, (
            f"{name}.nhts_vintage has unexpected values: {unique_vals}"
        )


def test_sentinels_converted_to_na(tmp_path: Path, synthetic_nextgen_zip: Path) -> None:
    """HHFAMINC -7 -> NA; PER all-sentinel row -> all NA."""
    cache = tmp_path / "cache"
    data = load_nhts_nextgen_2022(
        cache_dir=cache, download_url=_file_url(synthetic_nextgen_zip)
    )
    hh3 = data.hh.loc[data.hh["houseid"].astype("string") == "30000003"].iloc[0]
    assert pd.isna(hh3["hhfaminc"])

    per3 = data.per.loc[data.per["houseid"].astype("string") == "30000003"].iloc[0]
    assert pd.isna(per3["r_age"])
    assert pd.isna(per3["r_sex"])
    assert pd.isna(per3["worker"])
    assert pd.isna(per3["driver"])


def test_new_columns_preserved_whytrp1s_tripmode_stratumid(
    tmp_path: Path, synthetic_nextgen_zip: Path
) -> None:
    """Native NextGen 2022 additions ride through verbatim."""
    cache = tmp_path / "cache"
    data = load_nhts_nextgen_2022(
        cache_dir=cache, download_url=_file_url(synthetic_nextgen_zip)
    )
    assert "whytrp1s" in data.trip.columns
    assert "tripmode" in data.trip.columns
    assert "stratumid" in data.hh.columns


def test_manifest_schema_deltas_and_sha256(
    tmp_path: Path, synthetic_nextgen_zip: Path
) -> None:
    cache = tmp_path / "cache"
    load_nhts_nextgen_2022(
        cache_dir=cache, download_url=_file_url(synthetic_nextgen_zip)
    )
    with (cache / NHTS_NEXTGEN_MANIFEST_BASENAME).open() as fh:
        manifest = json.load(fh)
    for key in (
        "url",
        "zip_sha256",
        "csv_sha256",
        "fetch_timestamp_utc",
        "filter_rule",
        "subset_size",
        "raw_size",
        "schema_deltas_vs_2017",
        "nhts_vintage",
    ):
        assert key in manifest, f"manifest missing key {key!r}"
    assert manifest["nhts_vintage"] == NHTS_NEXTGEN_VINTAGE
    deltas = manifest["schema_deltas_vs_2017"]
    assert "MODEL" in deltas["absent_in_public"]
    assert deltas["renamed"] == {"R_SEX_IMP": "r_sex", "DWELL_TIME": "dwell_min"}

    # SHA-256s present for every CSV member.
    assert set(manifest["csv_sha256"].keys()) == set(NHTS_NEXTGEN_CSV_MEMBERS)
    for h in manifest["csv_sha256"].values():
        assert isinstance(h, str) and len(h) == 64


def test_idempotent_second_call_returns_cached(
    tmp_path: Path, synthetic_nextgen_zip: Path
) -> None:
    cache = tmp_path / "cache"
    load_nhts_nextgen_2022(
        cache_dir=cache, download_url=_file_url(synthetic_nextgen_zip)
    )
    first = {
        key: _sha256_of_file(cache / fname)
        for key, fname in NEXTGEN_PARQUET_BASENAMES.items()
    }
    load_nhts_nextgen_2022(
        cache_dir=cache, download_url=_file_url(synthetic_nextgen_zip)
    )
    second = {
        key: _sha256_of_file(cache / fname)
        for key, fname in NEXTGEN_PARQUET_BASENAMES.items()
    }
    assert first == second


def test_cli_addon_stub_raises_for_v31_deferral(tmp_path: Path) -> None:
    """v3.1 deferral: VDOT/TDOT/OMPO add-on path is not implemented."""
    with pytest.raises(NotImplementedError, match="VDOT/TDOT/OMPO"):
        main(["--add-on", "vdot", "--cache-dir", str(tmp_path)])


def test_cli_arg_parser_surface() -> None:
    """CLI mirrors nhts_loader's: --repo-root, --cache-dir, --force, --url,
    --timeout, --log-level, plus the --add-on stub."""
    parser = _build_arg_parser()
    args = parser.parse_args([])
    assert args.repo_root is None
    assert args.cache_dir is None
    assert args.force is False
    assert args.url == NHTS_NEXTGEN_ZIP_URL
    assert args.log_level == "INFO"
    assert args.add_on is None
