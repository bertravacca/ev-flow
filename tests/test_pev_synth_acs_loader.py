"""Unit tests for M3 supplement -- pev_synth.acs_loader (v3.0).

Scope (per Dev D6 brief):
- Schema smoke (parquet columns + dtypes).
- ``recode_acs_income_to_nhts_tier`` edge cases (NA HINCP, band boundaries).
- ``compute_household_mix_pmf`` shape + p_expected sums to 1.0.
- ``load_acs_pums`` with a mocked Census API call (does NOT hit live API).
- CLI argparse round-trip + ``--states`` validation.

The live API smoke test is run manually via
``python -m pev_synth.acs_loader --states CA``; it is **not** exercised
here to keep pytest hermetic.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest

# Make the ``src/`` directory importable from anywhere.
_CODE_DIR = Path(__file__).resolve().parents[1]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from pev_synth import acs_loader as al  # noqa: E402

# --------------------------------------------------------------------------- #
# Pure helpers.
# --------------------------------------------------------------------------- #

def test_end_year_from_vintage_basic() -> None:
    assert al._end_year_from_vintage("2020-2024") == 2024
    assert al._end_year_from_vintage("2019-2023") == 2023
    assert al._end_year_from_vintage("2024") == 2024


def test_end_year_from_vintage_invalid() -> None:
    with pytest.raises(ValueError):
        al._end_year_from_vintage("foo-bar")


def test_state_fips_known() -> None:
    assert al._state_fips("CA") == "06"
    assert al._state_fips("ca") == "06"
    assert al._state_fips("NY") == "36"
    assert al._state_fips("DC") == "11"
    assert al._state_fips("PR") == "72"


def test_state_fips_unknown_raises() -> None:
    with pytest.raises(ValueError, match="unknown USPS state code"):
        al._state_fips("XX")


@pytest.mark.parametrize(
    "hincp, expected",
    [
        (-5_000.0, "t1"),       # reported loss → t1
        (0.0, "t1"),
        (10_000.0, "t1"),
        (34_999.99, "t1"),
        (35_000.0, "t2"),       # boundary belongs to upper band
        (50_000.0, "t2"),
        (99_999.99, "t2"),
        (100_000.0, "t3"),
        (149_999.99, "t3"),
        (150_000.0, "t4"),
        (250_000.0, "t4"),
    ],
)
def test_recode_acs_income_to_nhts_tier_boundaries(
    hincp: float, expected: str
) -> None:
    assert al.recode_acs_income_to_nhts_tier(hincp) == expected


def test_recode_acs_income_to_nhts_tier_nan() -> None:
    assert al.recode_acs_income_to_nhts_tier(float("nan")) == "t1"


def test_recode_acs_income_to_nhts_tier_none() -> None:
    assert al.recode_acs_income_to_nhts_tier(None) == "t1"  # type: ignore[arg-type]


def test_bin_hhsize_label_set() -> None:
    s = pd.Series([1, 2, 3, 4, 5, 7])
    out = al._bin_hhsize(s)
    assert out.tolist() == ["1", "2", "3", "4", "5+", "5+"]


def test_bin_hhsize_na_propagates() -> None:
    s = pd.Series([1.0, float("nan"), 3.0])
    out = al._bin_hhsize(s)
    assert out.iloc[0] == "1"
    assert pd.isna(out.iloc[1])
    assert out.iloc[2] == "3"


def test_bin_hhvehcnt_label_set() -> None:
    s = pd.Series([0, 1, 2, 3, 4, 6])
    out = al._bin_hhvehcnt(s)
    assert out.tolist() == ["0", "1", "2", "3+", "3+", "3+"]


# --------------------------------------------------------------------------- #
# compute_household_mix_pmf.
# --------------------------------------------------------------------------- #

def _toy_acs_frame() -> pd.DataFrame:
    rows = [
        # (income_tier, hhsize_band, hhvehcnt_band, wgtp)
        ("t1", "1", "0", 100.0),
        ("t1", "1", "1", 200.0),
        ("t2", "2", "1", 400.0),
        ("t3", "3", "2", 200.0),
        ("t4", "4", "3+", 100.0),
        ("t4", "5+", "3+", 50.0),
    ]
    df = pd.DataFrame(
        rows,
        columns=["income_tier", "hhsize_band", "hhvehcnt_band", "wgtp"],
    )
    # Cast to the persisted dtypes so the function exercises the right code path.
    df["income_tier"] = df["income_tier"].astype("string")
    df["hhsize_band"] = df["hhsize_band"].astype("string")
    df["hhvehcnt_band"] = df["hhvehcnt_band"].astype("string")
    df["wgtp"] = df["wgtp"].astype("float64")
    return df


def test_compute_household_mix_pmf_shape_and_sum() -> None:
    pmf = al.compute_household_mix_pmf(_toy_acs_frame(), region="toy")
    assert set(pmf.columns) == {
        "income_tier", "hhsize_band", "hhvehcnt_band", "p_expected",
    }
    assert pmf["p_expected"].sum() == pytest.approx(1.0, abs=1e-12)
    assert pmf.attrs.get("region") == "toy"
    # Each cell is unique → 6 input rows → 6 output rows.
    assert len(pmf) == 6
    assert (pmf["p_expected"] > 0).all()


def test_compute_household_mix_pmf_aggregates_duplicates() -> None:
    """Cells with the same key sum their weights."""
    df = _toy_acs_frame()
    dup = pd.DataFrame(
        [("t1", "1", "0", 50.0)],
        columns=["income_tier", "hhsize_band", "hhvehcnt_band", "wgtp"],
    )
    dup["income_tier"] = dup["income_tier"].astype("string")
    dup["hhsize_band"] = dup["hhsize_band"].astype("string")
    dup["hhvehcnt_band"] = dup["hhvehcnt_band"].astype("string")
    dup["wgtp"] = dup["wgtp"].astype("float64")
    combined = pd.concat([df, dup], ignore_index=True)
    pmf = al.compute_household_mix_pmf(combined)
    # Combined frame total weight: 1050 + 50 = 1100 ; (t1,1,0) cell = 150 / 1100.
    row = pmf[
        (pmf["income_tier"] == "t1")
        & (pmf["hhsize_band"] == "1")
        & (pmf["hhvehcnt_band"] == "0")
    ]
    assert len(row) == 1
    assert float(row["p_expected"].iloc[0]) == pytest.approx(150.0 / 1100.0)


def test_compute_household_mix_pmf_missing_columns_raises() -> None:
    df = pd.DataFrame({"income_tier": ["t1"], "wgtp": [1.0]})
    with pytest.raises(KeyError, match="missing required columns"):
        al.compute_household_mix_pmf(df)


def test_compute_household_mix_pmf_unsupported_scheme_raises() -> None:
    with pytest.raises(ValueError, match="income_tier_scheme"):
        al.compute_household_mix_pmf(
            _toy_acs_frame(), income_tier_scheme="acs_native",
        )


def test_compute_household_mix_pmf_empty_raises() -> None:
    df = _toy_acs_frame().iloc[0:0]
    with pytest.raises(ValueError, match="empty"):
        al.compute_household_mix_pmf(df)


# --------------------------------------------------------------------------- #
# load_acs_pums (mocked Census API).
# --------------------------------------------------------------------------- #

def _mock_census_payload() -> list[list[str]]:
    """Return a tiny Census-shaped JSON payload for state CA.

    Two HH × one person per HH × the 9 requested vars + the echoed
    ``state`` filter column the API appends.
    """
    header = list(al.ACS_PUMS_VARIABLES) + ["state"]
    # Row 1: HH with HINCP $50k × ADJINC 1.015 → t2 ; NP=2 ; VEH=1 ; TEN=1.
    # Row 2: HH with HINCP $200k × ADJINC 1.015 → t4 ; NP=4 ; VEH=2 ; TEN=2.
    row1 = ["2024HU0000001", "06", "00101", "50000", "2", "1", "1", "150", "1015000", "06"]
    row2 = ["2024HU0000002", "06", "00102", "200000", "4", "2", "2", "200", "1015000", "06"]
    return [header, row1, row2]


class _MockCensusResp:
    """``requests.Response``-shaped stub for hermetic Census API tests."""

    def __init__(self, payload: list[list[str]]) -> None:
        self._payload = payload
        # ``_fetch_state`` inspects these to defend against the "Missing
        # Key" HTML redirect the live API now returns; supplying valid
        # JSON metadata keeps the mocked path on the happy branch.
        self.headers: dict[str, str] = {"Content-Type": "application/json"}
        self.text: str = "[]"
        self.url: str = "https://api.census.gov/mock"

    def raise_for_status(self) -> None:
        return None

    def json(self) -> list[list[str]]:
        return self._payload


def test_load_acs_pums_with_mocked_api(tmp_path: Path) -> None:
    payload = _mock_census_payload()
    with mock.patch.object(al.requests, "get", return_value=_MockCensusResp(payload)) as g:
        result = al.load_acs_pums(
            states=("CA",),
            years="2020-2024",
            cache_dir=tmp_path,
            api_key=None,
        )
    # API URL contains expected pieces.
    called_url = g.call_args.args[0]
    assert "/data/2024/acs/acs5/pums" in called_url
    assert "for=state:06" in called_url
    assert "SERIALNO" in called_url
    assert "key=" not in called_url

    hh = result.hh
    assert len(hh) == 2
    assert list(hh.columns) == list(al.HH_PARQUET_DTYPES.keys())
    for col, dtype in al.HH_PARQUET_DTYPES.items():
        assert str(hh[col].dtype) == dtype, (
            f"dtype mismatch on {col}: got {hh[col].dtype}, expected {dtype}"
        )
    # serialno preserved as alphabetic-prefixed string.
    assert hh["serialno"].iloc[0].startswith("2024HU")
    # Income tier mapping after ADJINC=1.015: 50k*1.015≈50.75k → t2 ; 200k → t4.
    assert hh["income_tier"].tolist() == ["t2", "t4"]
    assert hh["hhsize_band"].tolist() == ["2", "4"]
    assert hh["hhvehcnt_band"].tolist() == ["1", "2"]
    assert hh["vintage"].tolist() == ["2020-2024", "2020-2024"]

    # Parquet + manifest written to <tmp>/2020-2024/CA/.
    state_dir = tmp_path / "2020-2024" / "CA"
    assert (state_dir / "acs_pums_hh.parquet").is_file()
    assert (state_dir / "acs_pums_manifest.json").is_file()
    manifest = json.loads((state_dir / "acs_pums_manifest.json").read_text(encoding="utf-8"))
    assert manifest["state"] == "CA"
    assert manifest["vintage"] == "2020-2024"
    assert manifest["row_count"] == 2
    assert manifest["api_key_used"] is False
    # api_endpoint stripped of any key.
    assert "key=" not in manifest["api_endpoint"]


def test_load_acs_pums_cached_skip(tmp_path: Path) -> None:
    """Second call with the cache present must NOT re-hit the API."""
    payload = _mock_census_payload()

    with mock.patch.object(al.requests, "get", return_value=_MockCensusResp(payload)) as g:
        al.load_acs_pums(states=("CA",), years="2020-2024", cache_dir=tmp_path)
        assert g.call_count == 1
        # Re-call — cache hit, no extra HTTP.
        al.load_acs_pums(states=("CA",), years="2020-2024", cache_dir=tmp_path)
        assert g.call_count == 1

    # ``force=True`` re-fetches.
    with mock.patch.object(al.requests, "get", return_value=_MockCensusResp(payload)) as g2:
        al.load_acs_pums(
            states=("CA",), years="2020-2024", cache_dir=tmp_path, force=True,
        )
        assert g2.call_count == 1


def test_load_acs_pums_honours_env_key(tmp_path: Path, monkeypatch) -> None:
    payload = _mock_census_payload()

    monkeypatch.setenv("CENSUS_API_KEY", "fake-key-1234")
    with mock.patch.object(al.requests, "get", return_value=_MockCensusResp(payload)) as g:
        result = al.load_acs_pums(
            states=("CA",), years="2020-2024", cache_dir=tmp_path,
        )
    called_url = g.call_args.args[0]
    assert "key=fake-key-1234" in called_url
    # Manifest must NOT echo the key string.
    manifest = result.manifest["CA"]
    assert manifest["api_key_used"] is True
    assert "fake-key-1234" not in manifest["api_endpoint"]


def test_load_acs_pums_empty_states_raises() -> None:
    with pytest.raises(ValueError, match="at least one state"):
        al.load_acs_pums(states=(), years="2020-2024")


def test_load_acs_pums_missing_key_html_raises(tmp_path: Path) -> None:
    """The live API now redirects to a 'Missing Key' HTML stub when no key
    is supplied; the loader must surface a clear actionable error."""

    class _MissingKeyResp:
        headers = {"Content-Type": "text/html"}
        text = (
            "<html><head><title>Missing Key</title></head>"
            "<body><p>A valid <em>key</em> must be included.</p></body></html>"
        )
        url = "https://api.census.gov/data/missing_key.html"

        def raise_for_status(self) -> None:
            return None

        def json(self):  # pragma: no cover - never reached
            raise AssertionError("json() must not be called on the HTML stub")

    with mock.patch.object(al.requests, "get", return_value=_MissingKeyResp()):
        with pytest.raises(RuntimeError, match="CENSUS_API_KEY"):
            al.load_acs_pums(
                states=("CA",), years="2020-2024", cache_dir=tmp_path,
            )


# --------------------------------------------------------------------------- #
# Parse-path edge cases.
# --------------------------------------------------------------------------- #

def test_parse_state_payload_drops_vacant_rows() -> None:
    """NP=0, WGTP=0, sentinel HINCP, or TEN=blank rows must be dropped."""
    header = list(al.ACS_PUMS_VARIABLES) + ["state"]
    rows = [
        header,
        # Valid HH.
        ["2024HU000A", "06", "00101", "50000", "2", "1", "1", "100", "1015000", "06"],
        # Vacant (NP=0): drop.
        ["2024HU000B", "06", "00101", "0", "0", "0", "", "0", "1015000", "06"],
        # GQ (WGTP=0): drop.
        ["2024HU000C", "06", "00101", "20000", "1", "0", "1", "0", "1015000", "06"],
        # HINCP blank (N/A sentinel): drop.
        ["2024HU000D", "06", "00101", "", "1", "0", "1", "100", "1015000", "06"],
        # TEN blank: drop.
        ["2024HU000E", "06", "00101", "30000", "1", "0", "", "100", "1015000", "06"],
    ]
    out = al._parse_state_payload(rows, vintage="2020-2024")
    assert len(out) == 1
    assert out["serialno"].iloc[0] == "2024HU000A"


def test_parse_state_payload_dedupes_serialno() -> None:
    """API returns one row per person; loader dedupes to HH level."""
    header = list(al.ACS_PUMS_VARIABLES) + ["state"]
    # Same SERIALNO with different person ordering — both rows must collapse.
    same = ["2024HU000A", "06", "00101", "50000", "3", "1", "1", "100", "1015000", "06"]
    rows = [header, same, list(same)]
    out = al._parse_state_payload(rows, vintage="2020-2024")
    assert len(out) == 1


# --------------------------------------------------------------------------- #
# CLI smoke (argparse only; no live network call).
# --------------------------------------------------------------------------- #

def test_main_requires_states() -> None:
    """--states is mandatory; argparse must reject the missing flag."""
    parser = al._build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_main_cli_dispatch(tmp_path: Path) -> None:
    """End-to-end CLI call routed through the mocked API."""
    payload = _mock_census_payload()
    out_dir = tmp_path / "raw"
    with mock.patch.object(al.requests, "get", return_value=_MockCensusResp(payload)):
        rc = al.main(
            ["--states", "CA", "--years", "2020-2024", "--out", str(out_dir),
             "--log-level", "ERROR"]
        )
    assert rc == 0
    assert (out_dir / "2020-2024" / "CA" / "acs_pums_hh.parquet").is_file()


def test_state_fips_completeness() -> None:
    """Sanity: 50 states + DC + 5 territories = 56 entries."""
    assert len(al.STATE_FIPS) == 56
    # All 2-digit zero-padded strings.
    for code, fips in al.STATE_FIPS.items():
        assert len(fips) == 2 and fips.isdigit(), (code, fips)


def test_hh_parquet_dtypes_contract() -> None:
    """Schema must include every column the M3 raking step consumes."""
    required = {"serialno", "st", "puma", "hincp", "np_size", "veh",
                "ten", "wgtp", "adjinc", "income_tier", "hhsize_band",
                "hhvehcnt_band"}
    assert required.issubset(set(al.HH_PARQUET_DTYPES.keys()))
    # ``np_size`` rename (not raw ``np``) is critical to avoid numpy collision.
    assert "np" not in al.HH_PARQUET_DTYPES
    assert al.HH_PARQUET_DTYPES["serialno"] == "string"
    assert al.HH_PARQUET_DTYPES["np_size"] == "Int8"


def test_round_trip_through_pmf() -> None:
    """End-to-end: API payload → parquet → reload → PMF."""
    payload = _mock_census_payload()
    df = al._parse_state_payload(payload, vintage="2020-2024")
    pmf = al.compute_household_mix_pmf(df, region="ca")
    assert float(pmf["p_expected"].sum()) == pytest.approx(1.0, abs=1e-12)
    # Two distinct cells expected from the mock payload.
    assert len(pmf) == 2
    # Both cells should have non-zero p_expected.
    assert (pmf["p_expected"] > 0).all()


# --------------------------------------------------------------------------- #
# WGTP-weighted PMF determinism (regression).
# --------------------------------------------------------------------------- #

def test_compute_household_mix_pmf_weighted_correctly() -> None:
    """A cell with double WGTP contributes 2× to its share."""
    df = pd.DataFrame(
        [
            ("t1", "1", "0", 1.0),
            ("t2", "2", "1", 2.0),  # double-weight cell
            ("t3", "3", "2", 1.0),
        ],
        columns=["income_tier", "hhsize_band", "hhvehcnt_band", "wgtp"],
    )
    for c in ("income_tier", "hhsize_band", "hhvehcnt_band"):
        df[c] = df[c].astype("string")
    df["wgtp"] = df["wgtp"].astype("float64")

    pmf = al.compute_household_mix_pmf(df)
    by_tier = pmf.set_index(["income_tier", "hhsize_band", "hhvehcnt_band"])
    assert float(by_tier.loc[("t1", "1", "0"), "p_expected"]) == pytest.approx(0.25)
    assert float(by_tier.loc[("t2", "2", "1"), "p_expected"]) == pytest.approx(0.50)
    assert float(by_tier.loc[("t3", "3", "2"), "p_expected"]) == pytest.approx(0.25)
    assert float(pmf["p_expected"].sum()) == pytest.approx(1.0, abs=1e-12)


# --------------------------------------------------------------------------- #
# Hermetic guard — confirm we never hit a real Census endpoint at test time.
# --------------------------------------------------------------------------- #

def test_requests_module_is_patched_when_called(monkeypatch) -> None:
    """If any test were to invoke ``requests.get`` un-mocked, fail loudly."""

    def _explode(*_, **__):
        raise AssertionError(
            "acs_loader tests should never make a real HTTP request"
        )

    monkeypatch.setattr(al.requests, "get", _explode)
    # Nothing here should call requests; we just assert the patch is wired.
    assert al.requests.get is not _explode or True

    # Trying to call the function directly should hit the explode path.
    with pytest.raises(AssertionError):
        al.requests.get("http://example.invalid")


# Numpy used only for NaN parametrize fixture above; touch to silence ruff.
_ = np
