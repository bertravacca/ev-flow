"""pev_synth.regions — Region registry for v2.0 of the aggregated-EV study.

This module is the foundation layer of the v2.0 ``pev_synth`` refactor. It
exposes:

* The frozen ``Region`` dataclass describing one regional / climatic / market
  slice of the U.S. EV synthesis pipeline.
* A registry ``REGIONS`` containing the 8 named regions of the v2.0
  methodology (one CBSA-anchored slice per regional climate / market mix).
* The discovery helpers ``list_regions()`` and ``list_profile_types()``.

Per the v2.0-rev plan §2.6 the legacy ``fleet_depot`` profile type has been
de-scoped; ``list_profile_types()`` therefore returns ``["residential",
"workplace"]`` only, and downstream callers should raise a ``ValueError`` when
``fleet_depot`` is requested.

Methodology version: v2.0.0.  This module is deterministic — no RNG.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

__all__ = [
    "ProfileType",
    "ISOMarket",
    "Region",
    "REGIONS",
    "PROFILE_TYPES",
    "list_regions",
    "list_profile_types",
]


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

ProfileType = Literal["residential", "workplace"]
ISOMarket = Literal["CAISO", "ERCOT", "NYISO", "PJM", "MISO", "ISO-NE", None]


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: The two supported profile types in v2.0.  ``fleet_depot`` was dropped per
#: plan §2.6 / Open Question 6.
PROFILE_TYPES: tuple[str, ...] = ("residential", "workplace")

#: Profile types that were valid in v1.1 but were intentionally removed in
#: v2.0.  Used by the API layer (Dev H, Module 8 brief) to raise a clean
#: ``ValueError`` when a caller still passes them.
_DESCOPED_PROFILE_TYPES: tuple[str, ...] = ("fleet_depot",)

#: All 50 U.S. states + DC (USPS codes, sorted alphabetically).  Used for the
#: ``us_national`` region whose frame is the full NHTS-2017 file.  PR / VI /
#: GU / AS / MP are excluded because they are outside the NHTS frame.
_US_NATIONAL_STATES: tuple[str, ...] = (
    "AK", "AL", "AR", "AZ", "CA", "CO", "CT", "DC", "DE", "FL",
    "GA", "HI", "IA", "ID", "IL", "IN", "KS", "KY", "LA", "MA",
    "MD", "ME", "MI", "MN", "MO", "MS", "MT", "NC", "ND", "NE",
    "NH", "NJ", "NM", "NV", "NY", "OH", "OK", "OR", "PA", "RI",
    "SC", "SD", "TN", "TX", "UT", "VA", "VT", "WA", "WI", "WV",
    "WY",
)


# ---------------------------------------------------------------------------
# Region dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Region:
    """One regional slice of the v2.0 EV synthesis pipeline.

    Frozen and hashable so a ``Region`` can be used as a cache key in
    downstream modules (donor matcher, travel-week builder, SoC ledger).
    All fields are populated explicitly per plan §2.2; ``None`` is only
    used for ``iso_market`` on the national reference region.

    Notes
    -----
    Citation pointers:

    * CBSA codes verified against OMB 2023 ``list1_2023.xlsx``.
    * Winter f_T multipliers are a linear approximation calibrated against
      Yuksel & Michalek (2015) winter uplift, evaluated at the Dec-Mar mean
      ambient temperature from NOAA ISD 1991-2020 normals
      (formula: ``1 + 0.018 * (20 - T_C)``).
    * DFW pickup overlay per Cox Automotive Q4-2024 / Q1-2025 Texas EV
      registration data.
    """

    name: str
    display_name: str
    states: tuple[str, ...]
    cbsas: tuple[int, ...]
    cdivmsar_filter: tuple[int, ...]
    tz: str
    iso_market: ISOMarket
    #: Optional NERC balancing-authority code when no ISO/RTO governs the
    #: region, or the dominant BA when the region spans multiple ISO/BA
    #: boundaries.
    balancing_authority: str | None = None
    sales_mix_source: str = ""
    #: RFC-007: stored as a canonically-sorted tuple-of-pairs so the
    #: dataclass is truly immutable (frozen + slots can't catch in-place
    #: mutation of a dict). Access via :attr:`sales_mix_overlay_map` (a
    #: ``MappingProxyType``) for a read-only dict-like view.
    sales_mix_overlay: tuple[tuple[str, float], ...] = ()
    temperature_stations: tuple[str, ...] = ()
    income_tier_scheme: str = "acs_national"
    winter_f_T_multiplier: float = 1.0
    #: RFC-031: NOAA Dec-Mar normal ambient temperature (°C) used to
    #: evaluate the Yuksel-Michalek linear uplift approximation for
    #: cold-region winter multipliers. ``None`` for regions whose pin
    #: is fixed at 1.0 (no source temperature).
    winter_t_c: float | None = None
    workplace_donor_borrow_states: tuple[str, ...] = ()
    #: RFC-032: optional override for the achieved ``has_heat_pump``
    #: share in the M2 fleet. ``None`` leaves the natural make + model-year
    #: derived share untouched; a float in [0, 1] triggers deterministic
    #: stratified resampling. Immutable + hashable so two Regions
    #: differing only in this knob are != with different hashes.
    heat_pump_share_override: float | None = None
    # v3.0 enrichment knobs — defaults preserve v2.1 byte-stability for every
    # existing region. Stored as hashable types so the frozen dataclass
    # remains hashable (no dict fields).
    #: NHTS vintage mix as canonical tuple-of-pairs (vintage_label, share).
    #: Default ``(("2017", 1.0),)`` reproduces v2.1 single-vintage behaviour.
    nhts_vintage_mix: tuple[tuple[str, float], ...] = (("2017", 1.0),)
    #: When True, M3 applies ACS PUMS raking weights to the household-mix
    #: expected PMF for this region. Default False = v2.1 behaviour.
    acs_calibration_enabled: bool = False
    #: Census ACS PUMS vintage label consumed when
    #: :attr:`acs_calibration_enabled` is True. Default ``"2020-2024"`` is
    #: the 5-year PUMS aligned with the v3.0 release.
    acs_pums_vintage: str = "2020-2024"
    notes: str = ""

    @property
    def nhts_vintage_mix_map(self) -> Mapping[str, float]:
        """Read-only ``Mapping`` view of :attr:`nhts_vintage_mix`.

        Convenience accessor mirroring :attr:`sales_mix_overlay_map`. The
        underlying storage is a canonical tuple-of-pairs so the frozen
        dataclass stays hashable; this property surfaces it as an
        immutable ``MappingProxyType`` for ergonomic consumer access.
        """
        return MappingProxyType(dict(self.nhts_vintage_mix))

    @property
    def sales_mix_overlay_map(self) -> Mapping[str, float]:
        """Read-only ``Mapping`` view of :attr:`sales_mix_overlay`.

        Backed by :class:`types.MappingProxyType` — in-place mutation
        attempts raise ``TypeError``.
        """
        return MappingProxyType(dict(self.sales_mix_overlay))

    def __hash__(self) -> int:
        # RFC-007 + RFC-032: hash on (name, overlay, heat_pump_share_override)
        # so two Regions differing only in those knobs are != with
        # different hashes. The frozen dataclass plus tuple-of-pairs
        # storage guarantees overlay is hashable.
        return hash(
            (self.name, self.sales_mix_overlay, self.heat_pump_share_override)
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Region):
            return NotImplemented
        return (
            self.name == other.name
            and self.sales_mix_overlay == other.sales_mix_overlay
            and self.heat_pump_share_override == other.heat_pump_share_override
        )

    @classmethod
    def from_name(cls, name: str) -> Region:
        """Look up a region by its short name.

        Parameters
        ----------
        name:
            Region short name, e.g. ``"bay_area"``.

        Returns
        -------
        Region
            The matching frozen ``Region`` instance.

        Raises
        ------
        ValueError
            If ``name`` is not a registered region.  The error message lists
            the valid names so callers can self-correct.
        """
        try:
            return REGIONS[name]
        except KeyError as exc:
            valid = sorted(REGIONS.keys())
            raise ValueError(
                f"Unknown region {name!r}; valid names are {valid}"
            ) from exc


# ---------------------------------------------------------------------------
# Region registry — the 8 v2.0 regions
# ---------------------------------------------------------------------------

# 1. bay_area — v1.1 default region, NHTS CA add-on rich.
_BAY_AREA = Region(
    name="bay_area",
    display_name="Bay Area, CA",
    states=("CA",),
    cbsas=(41860, 41940, 42220, 46700, 34900, 42100),
    cdivmsar_filter=(),
    tz="America/Los_Angeles",
    iso_market="CAISO",
    sales_mix_source="cvrp_ca",
    sales_mix_overlay=(),
    temperature_stations=(),
    income_tier_scheme="acs_national",
    winter_f_T_multiplier=1.0,
    workplace_donor_borrow_states=(),
    notes="Bay Area, CA. v1.1 default region. NHTS CA add-on rich.",
)

# 2. la_basin — Los Angeles basin, commute-heavy CAISO region.
_LA_BASIN = Region(
    name="la_basin",
    display_name="Los Angeles Basin, CA",
    states=("CA",),
    cbsas=(31080, 40140, 37100, 12540),  # LA, Riverside, Oxnard, Bakersfield
    cdivmsar_filter=(),
    tz="America/Los_Angeles",
    iso_market="CAISO",
    sales_mix_source="cvrp_ca",
    sales_mix_overlay=(),
    temperature_stations=(),
    income_tier_scheme="acs_national",
    winter_f_T_multiplier=1.0,
    workplace_donor_borrow_states=(),
    notes="LA Basin, CA. CAISO, commute-heavy.",
)

# 3. new_york_metro — NYC + tri-state, multi-unit-dwelling dominant.
# RFC-031: winter_f_T = yuksel_michalek_linear_uplift(-2 °C) = 1 + 0.018·22 = 1.396.
_NEW_YORK_METRO = Region(
    name="new_york_metro",
    display_name="New York Metro",
    states=("NY", "NJ", "CT"),
    cbsas=(35620, 14860, 39100),  # NYC-Newark, Bridgeport, Poughkeepsie
    cdivmsar_filter=(),
    tz="America/New_York",
    iso_market="NYISO",
    sales_mix_source="nyserda_drive_clean",
    sales_mix_overlay=(),
    temperature_stations=(),
    income_tier_scheme="acs_national",
    winter_f_T_multiplier=1.396,
    winter_t_c=-2.0,
    workplace_donor_borrow_states=("PA",),
    notes=(
        "NYC + tri-state. Multi-unit dwelling dominant. "
        "Note: CT shares state coverage with boston (Worcester CBSA spans MA/CT)."
    ),
)

# 4. boston — CT included per RC-1 because Worcester CBSA 49340 spans
# Worcester County MA and Windham County CT.
# RFC-031: winter_f_T = yuksel_michalek_linear_uplift(-2 °C) = 1 + 0.018·22 = 1.396.
_BOSTON = Region(
    name="boston",
    display_name="Greater Boston",
    states=("MA", "NH", "RI", "CT"),
    cbsas=(14460, 39300, 49340),  # Boston, Providence, Worcester
    cdivmsar_filter=(),
    tz="America/New_York",
    iso_market="ISO-NE",
    sales_mix_source="argonne_national",
    sales_mix_overlay=(),
    temperature_stations=(),
    income_tier_scheme="acs_national",
    winter_f_T_multiplier=1.396,
    winter_t_c=-2.0,
    workplace_donor_borrow_states=("NY", "VT", "ME"),
    notes=(
        "Greater Boston. Workplace donor pool thin; borrows from NY/VT/ME. "
        "Note: CT shares state coverage with new_york_metro "
        "(Worcester CBSA 49340 spans MA/CT)."
    ),
)

# 5. chicago — Chicago + Milwaukee on the PJM / MISO border.
# RFC-031: NOAA Dec-Mar normal = -4 °C → yuksel_michalek_linear_uplift(-4) = 1.432.
# Clamped at band ceiling 1.40 for the cold-region band [1.10, 1.40].
_CHICAGO = Region(
    name="chicago",
    display_name="Chicago + Milwaukee",
    states=("IL", "IN", "WI"),
    cbsas=(16980, 33340),  # Chicago, Milwaukee
    cdivmsar_filter=(),
    tz="America/Chicago",
    iso_market="PJM",
    balancing_authority="MISO_Z2",
    sales_mix_source="argonne_national",
    sales_mix_overlay=(),
    temperature_stations=(),
    income_tier_scheme="acs_national",
    winter_f_T_multiplier=1.40,
    winter_t_c=-4.0,
    workplace_donor_borrow_states=("MI", "MN"),
    notes=(
        "Chicago + Milwaukee. ComEd settles in PJM; WeEnergies "
        "(Milwaukee CBSA 33340) is in MISO Zone 2. Cold winters."
    ),
)

# 6. dallas_fort_worth — ERCOT, pickup-heavy fleet (S-5: +10pp pickup share).
_DALLAS_FORT_WORTH = Region(
    name="dallas_fort_worth",
    display_name="Dallas-Fort Worth + Austin + Houston",
    states=("TX",),
    cbsas=(19100, 12420, 26420),  # DFW, Austin, Houston
    cdivmsar_filter=(),
    tz="America/Chicago",
    iso_market="ERCOT",
    sales_mix_source="argonne_national",
    sales_mix_overlay=(("pickup_overlay", 0.10),),  # +10pp pickup share per S-5
    temperature_stations=(),
    income_tier_scheme="acs_national",
    # RFC-031: non-cold region → pin = 1.00; winter_t_c left as None.
    winter_f_T_multiplier=1.00,
    workplace_donor_borrow_states=(),
    notes=(
        "DFW + Austin + Houston. ERCOT. Pickup-heavy fleet "
        "(TX BEV registrations skew pickup)."
    ),
)

# 7. seattle — Seattle + Portland, mild marine climate, BPA hydropower.
# Pacific Northwest has no RTO/ISO; ``balancing_authority="BPA"`` carries
# the settlement context that ``iso_market`` previously misrepresented as
# a NERC region.
_SEATTLE = Region(
    name="seattle",
    display_name="Seattle + Portland",
    states=("WA", "OR"),
    cbsas=(42660, 38900),  # Seattle-Tacoma, Portland
    cdivmsar_filter=(),
    tz="America/Los_Angeles",
    iso_market=None,
    balancing_authority="BPA",
    sales_mix_source="argonne_national",
    sales_mix_overlay=(),
    temperature_stations=(),
    income_tier_scheme="acs_national",
    winter_f_T_multiplier=1.0,  # mild marine climate
    workplace_donor_borrow_states=("ID",),
    notes=(
        "Seattle + Portland. No ISO/RTO in the Pacific Northwest; "
        "BPA hydropower remains the dominant balancing authority. Mild climate."
    ),
)

# 8. us_national — full NHTS-2017 frame, US-weighted Argonne mix.
_US_NATIONAL = Region(
    name="us_national",
    display_name="United States (national)",
    states=_US_NATIONAL_STATES,
    cbsas=(),
    cdivmsar_filter=(),
    tz="UTC",
    iso_market=None,
    sales_mix_source="argonne_national",
    sales_mix_overlay=(),
    temperature_stations=(),
    income_tier_scheme="acs_national",
    winter_f_T_multiplier=1.0,
    workplace_donor_borrow_states=(),
    notes=(
        "Full national NHTS, US-weighted Argonne mix. Comparative reference."
    ),
)


REGIONS: dict[str, Region] = {
    _BAY_AREA.name: _BAY_AREA,
    _LA_BASIN.name: _LA_BASIN,
    _NEW_YORK_METRO.name: _NEW_YORK_METRO,
    _BOSTON.name: _BOSTON,
    _CHICAGO.name: _CHICAGO,
    _DALLAS_FORT_WORTH.name: _DALLAS_FORT_WORTH,
    _SEATTLE.name: _SEATTLE,
    _US_NATIONAL.name: _US_NATIONAL,
}


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------


def list_regions() -> list[str]:
    """Return all registered region short-names, sorted alphabetically."""
    return sorted(REGIONS.keys())


def list_profile_types() -> list[str]:
    """Return the supported profile types in v2.0.

    ``fleet_depot`` was de-scoped in v2.0 and is intentionally absent;
    only ``residential`` and ``workplace`` are supported.
    """
    return list(PROFILE_TYPES)


# ---------------------------------------------------------------------------
# Module-level self-check — RFC M4
# ---------------------------------------------------------------------------


def _warn_on_state_overlap() -> None:
    """Emit UserWarning if any state appears in more than one region.

    The national reference region ``us_national`` is excluded from the check
    because its ``states`` tuple is the full 50-state + DC frame by design;
    including it would flag every state as an overlap and drown out the
    intentional metro overlaps (e.g. CT shared by boston / new_york_metro).
    """
    import warnings
    from collections import defaultdict

    by_state: dict[str, list[str]] = defaultdict(list)
    for region in REGIONS.values():
        if region.name == "us_national":
            continue
        for state in region.states:
            by_state[state].append(region.name)
    overlaps = {s: regs for s, regs in by_state.items() if len(regs) > 1}
    if overlaps:
        for state, regs in sorted(overlaps.items()):
            warnings.warn(
                f"state {state!r} appears in multiple regions: {sorted(regs)}; "
                "donor pools will overlap (see Region.notes).",
                UserWarning,
                stacklevel=2,
            )


_warn_on_state_overlap()
