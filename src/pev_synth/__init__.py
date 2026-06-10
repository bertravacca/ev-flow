"""pev_synth — Synthetic EV charging dataset pipeline + library API.

Library API (v2.0)
------------------
The public surface for downstream code is:

    >>> import pev_synth as ps
    >>> ps.list_regions()
    ['bay_area', 'boston', 'chicago', 'dallas_fort_worth', 'la_basin',
     'new_york_metro', 'seattle', 'us_national']
    >>> ps.list_profile_types()
    ['residential', 'workplace']
    >>> fleet = ps.generate_profiles('residential', n=1000,
    ...                              region='bay_area')
    >>> prof = fleet[0]
    >>> pa = prof.generate_presence_absence('2001-01-01', '2001-01-08')

See ``src/pev_synth/api.py`` and ``docs/pev_synth_api.md`` for details.

Pipeline modules (M1..M9, methodology v2.0.0, master seed 20260520)
-------------------------------------------------------------------
* ``nhts_loader``               — NHTS 2017 public-use file loader.
* ``vehicle_archetypes``        — N-EV archetype sampler (M2).
* ``donor_matcher``             — NHTS donor-vehicle matcher (M3).
* ``travel_week_builder``       — one-year travel sequence builder (M4).
* ``plug_in_model``             — session plug-in / dwell sampler (M5).
* ``soc_trajectory``            — continuous-time SoC ledger + sessions (M6).
* ``hourly_resampler``          — 15-min + hourly plug-status rasteriser (M7).
* ``validation_bounds_curator`` — bound curation (M8).
* ``validator``                 — §10 validation runner + report writer (M9).
* ``regions``                   — 8-region registry (Region dataclass).
* ``_utc_migration``            — package-internal v1.1 → v2.0 UTC cache
                                  migrator (leading underscore = not part of
                                  the public API; used only by in-house
                                  callers with a v1.1 cache to upgrade).

The library API wraps the artifacts these modules produce.
"""

from __future__ import annotations

from .api import (
    Fleet,
    Profile,
    ProfileType,
    generate_profiles,
    regenerate_fleet,
)

# v2.0: ``list_profile_types`` and the Region registry come from
# ``pev_synth.regions`` so the public listing returns ``["residential",
# "workplace"]`` (``fleet_depot`` was de-scoped — plan §2.6).
from .regions import REGIONS, Region, list_profile_types, list_regions

__version__ = "3.0.2"

__all__ = [
    "Fleet",
    "Profile",
    "ProfileType",
    "REGIONS",
    "Region",
    "generate_profiles",
    "list_profile_types",
    "list_regions",
    "regenerate_fleet",
    "__version__",
]
