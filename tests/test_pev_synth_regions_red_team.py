"""Regression tests for the red-team REGIONS cluster (H5, M4).

These tests pin the fixes from the red-team remediation plan for
``src/pev_synth/regions.py``:

* **H5** — Seattle is no longer tagged with the bogus ``iso_market="WECC"``
  (WECC is a NERC region, not an ISO/RTO). Seattle's ISO is ``None`` and its
  ``balancing_authority`` is ``"BPA"``. Chicago's ``balancing_authority`` flags
  the MISO Zone 2 portion (Milwaukee/WeEnergies). The ``ISOMarket`` literal no
  longer accepts ``"WECC"``.
* **M4** — Importing ``pev_synth.regions`` emits a ``UserWarning`` listing
  any state that appears in more than one region (today: CT in both
  ``boston`` and ``new_york_metro``).

M6 is wording-only and is verified by a ``grep`` invariant in the PM plan,
not by a runtime assertion.
"""

from __future__ import annotations

import importlib
import typing
import warnings


def test_h5_seattle_iso_market_is_none_balancing_authority_is_bpa() -> None:
    from pev_synth.regions import Region

    seattle = Region.from_name("seattle")
    assert seattle.iso_market is None
    assert seattle.balancing_authority == "BPA"


def test_h5_chicago_balancing_authority_flags_miso_zone_2() -> None:
    from pev_synth.regions import Region

    chicago = Region.from_name("chicago")
    # ComEd remains in PJM ...
    assert chicago.iso_market == "PJM"
    # ... but the WeEnergies/Milwaukee portion settles in MISO Zone 2.
    assert chicago.balancing_authority == "MISO_Z2"


def test_h5_iso_market_literal_no_longer_includes_wecc() -> None:
    from pev_synth.regions import ISOMarket

    # WECC was removed from the ISOMarket Literal alias because it is a
    # NERC region, not an ISO/RTO. Surface a typed-args invariant so future
    # refactors don't silently re-add it.
    allowed = set(typing.get_args(ISOMarket))
    assert "WECC" not in allowed
    assert {"CAISO", "ERCOT", "NYISO", "PJM", "MISO", "ISO-NE", None} <= allowed


def test_m4_state_overlap_warning_emitted_on_import() -> None:
    import pev_synth.regions as regions_mod

    # `importlib.reload` rebinds `pev_synth.regions` to a fresh module object
    # whose `Region` class is a DISTINCT type from the one already-imported
    # modules (e.g. `travel_week_builder`) captured at their import time. If
    # we leave that reloaded module in `sys.modules`, every later test that
    # passes a `REGIONS[...]` object into code guarded by `isinstance(x,
    # Region)` fails with "region must be Region; got Region" (two Region
    # classes). Snapshot the original module dict and restore it after the
    # reload so this red-team test stays hermetic.
    _saved = dict(regions_mod.__dict__)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            importlib.reload(regions_mod)

        user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
        assert user_warnings, "expected at least one UserWarning on regions reload"
    finally:
        # Restore the original module contents (incl. the original `Region`
        # class object and `REGIONS` instances) so module identity is
        # preserved for the rest of the session.
        regions_mod.__dict__.clear()
        regions_mod.__dict__.update(_saved)

    # The intentional overlap is CT in both boston and new_york_metro.
    ct_msgs = [str(w.message) for w in user_warnings if "'CT'" in str(w.message)]
    assert ct_msgs, (
        "expected a UserWarning mentioning CT, got: "
        f"{[str(w.message) for w in user_warnings]}"
    )
    joined = " | ".join(ct_msgs)
    assert "boston" in joined
    assert "new_york_metro" in joined
