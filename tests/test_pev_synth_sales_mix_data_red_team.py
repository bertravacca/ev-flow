"""Red-team H8 regressions for ``pev_synth.sales_mix_data``.

Covers the improved missing-CSV error message in
:func:`pev_synth.sales_mix_data.load_sales_mix` (red-team item H8).
The error message must surface the canonical landing-page URL for each
source so a PyPI user who hits the missing-file path knows where to find
the raw inputs the builder ingests.
"""

from __future__ import annotations

import pytest

from pev_synth.sales_mix_data import (
    _DEFAULT_SALES_MIX_DIR,
    SALES_MIX_SOURCE_URLS,
    SALES_MIX_SOURCES,
    load_sales_mix,
)


def test_sales_mix_source_urls_covers_all_sources() -> None:
    """``SALES_MIX_SOURCE_URLS`` must have an entry for each canonical source."""
    assert set(SALES_MIX_SOURCE_URLS.keys()) == set(SALES_MIX_SOURCES)
    for source in SALES_MIX_SOURCES:
        assert source in SALES_MIX_SOURCE_URLS


def test_sales_mix_source_urls_are_https() -> None:
    """Each landing-page URL must use the ``https://`` scheme."""
    for source, url in SALES_MIX_SOURCE_URLS.items():
        assert url.startswith("https://"), (
            f"URL for {source!r} does not start with https://: {url!r}"
        )


def test_load_sales_mix_missing_csv_surfaces_landing_url() -> None:
    """Missing-CSV ``ValueError`` must include the cvrp_ca landing URL.

    The error message body is the user-facing remediation hint, so a stable
    substring of the URL is the contract we pin here.
    """
    if (_DEFAULT_SALES_MIX_DIR / "cvrp_ca.csv").exists():
        pytest.skip(
            "cvrp_ca.csv is present on disk; this test exercises the "
            "missing-file error-message path only."
        )
    with pytest.raises(ValueError, match="cleanvehiclerebate.org"):
        load_sales_mix("cvrp_ca")
