"""Coverage-maximising tests for the three ev-flow data loaders.

Targets the remaining uncovered error/validation branches, small private
helpers, and the module CLIs of:

* :mod:`pev_synth.nhts_loader`        (M1  — NHTS 2017 California)
* :mod:`pev_synth.nhts_nextgen_loader` (M1b — NHTS NextGen 2022 national)
* :mod:`pev_synth.acs_loader`         (M3  — ACS PUMS 5-year household mix)

Every test here is fully data-independent: the heavy gitignored ``data/``
tree is intentionally absent, so we feed synthetic CSV/parquet fixtures via
``tmp_path`` and monkeypatch all network I/O (``requests.get``). The
``file://`` happy paths are already covered by the per-module test files;
this file deliberately exercises the *http(s)* download branch (by stubbing
``requests``), the dtype/contract guards, and the ``main`` CLIs.

Conventions mirror ``tests/test_pev_synth_nhts_loader.py`` and
``tests/test_pev_synth_acs_loader.py``: synthetic fixtures, fixed values,
real assertions, no live network.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Any
from unittest import mock

import pandas as pd
import pytest

from pev_synth import acs_loader as al
from pev_synth import nhts_loader as nl
from pev_synth import nhts_nextgen_loader as ng

# --------------------------------------------------------------------------- #
# Synthetic NHTS 2017 zip (minimal, one CA household).
# --------------------------------------------------------------------------- #


def _make_nhts2017_zip(dest: Path) -> Path:
    """Write a minimal valid NHTS-2017-style ``csv.zip`` to ``dest``."""
    hh = (
        '"HOUSEID","HHSTATE","HHSTFIPS","HH_CBSA","URBRUR","HHFAMINC",'
        '"HHSIZE","HHVEHCNT","HOMEOWN","WTHHFIN","CDIVMSAR"\n'
        '"10000001","CA",6,"41860",1,7,2,2,1,100.5,53\n'
    )
    per = (
        '"HOUSEID","PERSONID","R_AGE","R_SEX","WORKER","DRIVER","WTPERFIN"\n'
        '"10000001",1,45,1,1,1,100.5\n'
    )
    veh = (
        '"HOUSEID","VEHID","VEHTYPE","FUELTYPE","MAKE","MODEL",'
        '"VEHYEAR","ANNMILES","HFUEL"\n'
        '"10000001",1,1,1,"Toyota","Camry",2018,12000,-1\n'
    )
    trip = (
        '"HOUSEID","PERSONID","VEHID","TDTRPNUM","STRTTIME","ENDTIME",'
        '"TRPMILES","TRVLCMIN","WHYFROM","WHYTO","TDWKND","TRAVDAY","WTTRDFIN"\n'
        '"10000001",1,1,1,800,815,5.2,15,1,3,2,3,100.5\n'
    )
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("hhpub.csv", hh)
        zf.writestr("perpub.csv", per)
        zf.writestr("vehpub.csv", veh)
        zf.writestr("trippub.csv", trip)
    return dest


def _make_nextgen2022_zip(dest: Path) -> Path:
    """Write a minimal valid NextGen-2022-style ``csv.zip`` to ``dest``."""
    hh = (
        '"HOUSEID","CDIVMSAR","URBRUR","HHFAMINC","HHSIZE","HHVEHCNT",'
        '"HOMEOWN","WTHHFIN","STRATUMID"\n'
        '"30000001",53,1,7,2,2,1,1000.5,1\n'
    )
    per = (
        '"HOUSEID","PERSONID","R_AGE","R_SEX_IMP","WORKER","DRIVER","WTPERFIN"\n'
        '"30000001",1,45,1,1,1,1000.5\n'
    )
    veh = (
        '"HOUSEID","VEHID","VEHTYPE","FUELTYPE","MAKE",'
        '"VEHYEAR","ANNMILES","HFUEL"\n'
        '"30000001",1,3,3,"Tesla",2022,11000,3\n'
    )
    trip = (
        '"HOUSEID","PERSONID","VEHID","TDTRPNUM","STRTTIME","ENDTIME",'
        '"TRPMILES","TRVLCMIN","WHYFROM","WHYTO","TDWKND","TRAVDAY",'
        '"WTTRDFIN","DWELL_TIME","WHYTRP1S","TRIPMODE"\n'
        '"30000001",1,1,1,800,815,5.2,15,1,3,2,3,100.5,555,10,3\n'
    )
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("hhv2pub.csv", hh)
        zf.writestr("perv2pub.csv", per)
        zf.writestr("vehv2pub.csv", veh)
        zf.writestr("tripv2pub.csv", trip)
    return dest


class _FakeStreamResp:
    """Minimal ``requests.Response``-shaped context-manager stub.

    Mimics the streaming download path used by ``_download_zip``: supports
    ``raise_for_status`` and ``iter_content`` and the ``with ... as r:``
    context-manager protocol.
    """

    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def __enter__(self) -> _FakeStreamResp:
        return self

    def __exit__(self, *_: object) -> bool:
        return False

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, chunk_size: int = 1 << 20) -> Any:
        stream = io.BytesIO(self._payload)
        return iter(lambda: stream.read(chunk_size), b"")


# --------------------------------------------------------------------------- #
# nhts_loader: HTTP download branch (lines 241-246).
# --------------------------------------------------------------------------- #


def test_nhts_loader_http_download_branch(tmp_path: Path) -> None:
    """A non-file:// URL routes through requests.get streaming (lines 241-246)."""
    src_zip = _make_nhts2017_zip(tmp_path / "src.zip")
    payload = src_zip.read_bytes()
    cache = tmp_path / "cache"

    with mock.patch.object(
        nl.requests, "get", return_value=_FakeStreamResp(payload)
    ) as g:
        data = nl.load_nhts_2017_ca(
            cache_dir=cache, download_url="https://example.invalid/csv.zip"
        )

    # requests.get must have been used (i.e. we hit the http branch, not file://).
    assert g.call_count == 1
    assert g.call_args.args[0] == "https://example.invalid/csv.zip"
    assert len(data.hh) == 1
    assert data.hh["hhstate"].iloc[0] == "CA"
    assert (cache / "csv.zip").is_file()


# --------------------------------------------------------------------------- #
# nhts_loader: small private helpers + dtype guards.
# --------------------------------------------------------------------------- #


def test_nhts_coerce_int_non_numeric_path() -> None:
    """_coerce_int on a stringy column hits the to_numeric branch (line 278)."""
    s = pd.Series(["1", "2", "x", "-7"])
    out = nl._coerce_int(s)
    assert str(out.dtype) == "Int64"
    assert out.iloc[0] == 1
    assert pd.isna(out.iloc[2])  # "x" -> NA


def test_nhts_coerce_int_already_numeric_passthrough() -> None:
    """Numeric input returns unchanged (line 277 happy path)."""
    s = pd.Series([1, 2, 3], dtype="int64")
    out = nl._coerce_int(s)
    assert out is s


def test_nhts_enforce_dtypes_fills_missing_column_with_na() -> None:
    """Missing column is filled with NA then cast (line 293)."""
    # hh_cbsa (string) is absent; _enforce_dtypes must synthesise an all-NA col.
    df = pd.DataFrame({"houseid": [1], "hhstate": ["CA"]})
    out = nl._enforce_dtypes(df, nl.HH_DTYPES)
    assert "hh_cbsa" in out.columns
    assert out["hh_cbsa"].isna().all()
    assert str(out["hh_cbsa"].dtype) == "string"
    assert list(out.columns) == list(nl.HH_DTYPES.keys())


def test_nhts_enforce_dtypes_na_in_nonnullable_int_raises() -> None:
    """NA in a non-nullable int contract column raises ValueError (line 298)."""
    # houseid is "int64" (non-nullable); supply an unparseable value.
    df = pd.DataFrame({"houseid": ["not-a-number"], "hhstate": ["CA"]})
    with pytest.raises(ValueError, match="non-nullable"):
        nl._enforce_dtypes(df, nl.HH_DTYPES)


def test_nhts_process_hh_missing_hhstate_raises() -> None:
    """hhpub without HHSTATE cannot apply the CA filter (line 324)."""
    df = pd.DataFrame({"houseid": [1], "hhfaminc": [5]})
    with pytest.raises(KeyError, match="HHSTATE"):
        nl._process_hh(df)


# --------------------------------------------------------------------------- #
# nhts_loader: CLI main() (lines 674-706).
# --------------------------------------------------------------------------- #


def test_nhts_loader_main_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """End-to-end CLI: --url file:// + explicit --cache-dir, prints summary."""
    src_zip = _make_nhts2017_zip(tmp_path / "src.zip")
    cache = tmp_path / "cache"
    rc = nl.main(
        [
            "--url",
            src_zip.resolve().as_uri(),
            "--cache-dir",
            str(cache),
            "--log-level",
            "ERROR",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "NHTS 2017 CA subset written." in out
    assert "hh=1" in out
    assert (cache / nl.PARQUET_BASENAMES["hh"]).is_file()


def test_nhts_loader_main_cli_relative_cache_dir(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A relative --cache-dir is resolved against --repo-root (lines 686-687)."""
    src_zip = _make_nhts2017_zip(tmp_path / "src.zip")
    rc = nl.main(
        [
            "--url",
            src_zip.resolve().as_uri(),
            "--repo-root",
            str(tmp_path),
            "--cache-dir",
            "out/nhts",
            "--log-level",
            "ERROR",
        ]
    )
    assert rc == 0
    assert (tmp_path / "out" / "nhts" / nl.PARQUET_BASENAMES["hh"]).is_file()
    assert "hh=1" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# nhts_nextgen_loader: HTTP download branch (lines 221-226).
# --------------------------------------------------------------------------- #


def test_nhts_loader_main_cli_default_cache_under_repo_root(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """No --cache-dir → default <repo-root>/data/pev/raw/nhts2017 (line 683)."""
    src_zip = _make_nhts2017_zip(tmp_path / "src.zip")
    rc = nl.main(
        [
            "--url",
            src_zip.resolve().as_uri(),
            "--repo-root",
            str(tmp_path),
            "--log-level",
            "ERROR",
        ]
    )
    assert rc == 0
    default_cache = tmp_path / nl.DEFAULT_CACHE_REL
    assert (default_cache / nl.PARQUET_BASENAMES["hh"]).is_file()
    assert "hh=1" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# nhts_nextgen_loader: HTTP download branch (lines 221-226).
# --------------------------------------------------------------------------- #


def test_nextgen_loader_http_download_branch(tmp_path: Path) -> None:
    """Non-file:// URL routes through requests.get streaming (lines 221-226)."""
    src_zip = _make_nextgen2022_zip(tmp_path / "src.zip")
    payload = src_zip.read_bytes()
    cache = tmp_path / "cache"

    with mock.patch.object(
        ng.requests, "get", return_value=_FakeStreamResp(payload)
    ) as g:
        data = ng.load_nhts_nextgen_2022(
            cache_dir=cache, download_url="https://example.invalid/csv.zip"
        )

    assert g.call_count == 1
    assert g.call_args.args[0] == "https://example.invalid/csv.zip"
    assert len(data.hh) == 1
    assert data.hh["nhts_vintage"].iloc[0] == ng.NHTS_NEXTGEN_VINTAGE
    assert (cache / "csv.zip").is_file()


def test_nextgen_coerce_int_non_numeric_path() -> None:
    """_coerce_int on a stringy column hits the to_numeric branch (line 257)."""
    s = pd.Series(["1", "2", "x"])
    out = ng._coerce_int(s)
    assert str(out.dtype) == "Int64"
    assert pd.isna(out.iloc[2])


def test_nextgen_coerce_int_already_numeric_passthrough() -> None:
    s = pd.Series([1, 2, 3], dtype="int64")
    assert ng._coerce_int(s) is s


def test_nextgen_default_repo_root_resolves(tmp_path: Path) -> None:
    """_default_repo_root delegates to _paths.repo_root (lines 576-578)."""
    root = ng._default_repo_root()
    assert root.is_absolute()
    assert (root / "src" / "pev_synth" / "nhts_nextgen_loader.py").is_file()


def test_nextgen_loader_main_cli(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """End-to-end CLI happy path through main() (lines 658-685)."""
    src_zip = _make_nextgen2022_zip(tmp_path / "src.zip")
    cache = tmp_path / "cache"
    rc = ng.main(
        [
            "--url",
            src_zip.resolve().as_uri(),
            "--cache-dir",
            str(cache),
            "--log-level",
            "ERROR",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "NHTS NextGen 2022 national subset written." in out
    assert "hh=1" in out
    assert (cache / ng.NEXTGEN_PARQUET_BASENAMES["hh"]).is_file()


def test_nextgen_loader_main_cli_relative_cache_dir(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Relative --cache-dir resolved against --repo-root (lines 664-666)."""
    src_zip = _make_nextgen2022_zip(tmp_path / "src.zip")
    rc = ng.main(
        [
            "--url",
            src_zip.resolve().as_uri(),
            "--repo-root",
            str(tmp_path),
            "--cache-dir",
            "out/nextgen",
            "--log-level",
            "ERROR",
        ]
    )
    assert rc == 0
    assert (tmp_path / "out" / "nextgen" / ng.NEXTGEN_PARQUET_BASENAMES["hh"]).is_file()
    assert "hh=1" in capsys.readouterr().out


def test_nextgen_loader_main_cli_default_cache_under_repo_root(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """No --cache-dir → default <repo-root>/data/pev/raw/nhts2022 (line 662)."""
    src_zip = _make_nextgen2022_zip(tmp_path / "src.zip")
    rc = ng.main(
        [
            "--url",
            src_zip.resolve().as_uri(),
            "--repo-root",
            str(tmp_path),
            "--log-level",
            "ERROR",
        ]
    )
    assert rc == 0
    default_cache = tmp_path / ng.DEFAULT_CACHE_REL
    assert (default_cache / ng.NEXTGEN_PARQUET_BASENAMES["hh"]).is_file()
    assert "hh=1" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# acs_loader: _fetch_state error branches (lines 338, 344).
# --------------------------------------------------------------------------- #


class _AcsResp:
    def __init__(self, *, ctype: str, text: str, body: Any = None) -> None:
        self.headers = {"Content-Type": ctype}
        self.text = text
        self.url = "https://api.census.gov/mock"
        self._body = body

    def raise_for_status(self) -> None:
        return None

    def json(self) -> Any:
        return self._body


def test_acs_fetch_state_non_json_html_without_missing_key_raises() -> None:
    """HTML response that is NOT the 'missing key' stub → non-JSON error (line 338)."""
    resp = _AcsResp(
        ctype="text/html",
        text="<html><body>Internal Server Error</body></html>",
    )
    with mock.patch.object(al.requests, "get", return_value=resp):
        with pytest.raises(RuntimeError, match="non-JSON"):
            al._fetch_state("https://api.census.gov/x", timeout_s=5)


def test_acs_fetch_state_unexpected_body_shape_raises() -> None:
    """A JSON dict (not list-of-lists) → unexpected body shape error (line 344)."""
    resp = _AcsResp(ctype="application/json", text="{}", body={"oops": 1})
    with mock.patch.object(al.requests, "get", return_value=resp):
        with pytest.raises(RuntimeError, match="unexpected Census API body shape"):
            al._fetch_state("https://api.census.gov/x", timeout_s=5)


def test_acs_fetch_state_empty_list_body_raises() -> None:
    """An empty JSON list is also rejected (line 344, falsy branch)."""
    resp = _AcsResp(ctype="application/json", text="[]", body=[])
    with mock.patch.object(al.requests, "get", return_value=resp):
        with pytest.raises(RuntimeError, match="unexpected Census API body shape"):
            al._fetch_state("https://api.census.gov/x", timeout_s=5)


# --------------------------------------------------------------------------- #
# acs_loader: _default_cache_root (lines 445-447) + _write_parquet guard (499).
# --------------------------------------------------------------------------- #


def test_acs_default_cache_root_resolves() -> None:
    """_default_cache_root chains through _paths.data_root (lines 445-447)."""
    root = al._default_cache_root()
    assert root.is_absolute()
    assert root.as_posix().endswith("pev/raw/acs_pums")


def test_acs_write_parquet_missing_column_raises(tmp_path: Path) -> None:
    """_write_parquet refuses a frame missing a contract column (line 499)."""
    df = pd.DataFrame({"serialno": ["x"]})  # missing nearly all required cols
    with pytest.raises(KeyError, match="missing required column"):
        al._write_parquet(df, tmp_path)


# --------------------------------------------------------------------------- #
# acs_loader: compute_household_mix_pmf guards (lines 641, 661).
# --------------------------------------------------------------------------- #


def test_acs_pmf_unsupported_bin_scheme_raises() -> None:
    """Non-default bin strings raise ValueError (line 641)."""
    df = pd.DataFrame(
        {
            "income_tier": pd.array(["t1"], dtype="string"),
            "hhsize_band": pd.array(["1"], dtype="string"),
            "hhvehcnt_band": pd.array(["0"], dtype="string"),
            "wgtp": pd.array([1.0], dtype="float64"),
        }
    )
    with pytest.raises(ValueError, match="default bin schemes"):
        al.compute_household_mix_pmf(df, hhsize_bins="1,2,3")


def test_acs_pmf_nonpositive_total_wgtp_raises() -> None:
    """All-zero WGTP → non-positive total raises ValueError (line 661)."""
    df = pd.DataFrame(
        {
            "income_tier": pd.array(["t1", "t2"], dtype="string"),
            "hhsize_band": pd.array(["1", "2"], dtype="string"),
            "hhvehcnt_band": pd.array(["0", "1"], dtype="string"),
            "wgtp": pd.array([0.0, 0.0], dtype="float64"),
        }
    )
    with pytest.raises(ValueError, match="non-positive"):
        al.compute_household_mix_pmf(df)


# --------------------------------------------------------------------------- #
# acs_loader: CLI empty --states (lines 733-734).
# --------------------------------------------------------------------------- #


def test_acs_main_blank_states_returns_2(capsys: pytest.CaptureFixture[str]) -> None:
    """--states made of only whitespace/commas → rc 2 with stderr note."""
    rc = al.main(["--states", " , ,", "--log-level", "ERROR"])
    assert rc == 2
    assert "at least one USPS code" in capsys.readouterr().err
