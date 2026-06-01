"""Central seed, methodology-version, and sub-seed registry.

This module is the **single source of truth** for the v2.0 ``pev_synth``
master seed, methodology-version string, per-module DAG offsets, and per-EV
sub-seed namespace offsets. Every other module imports from here.

Per RFC-016 / RFC-001 (Phase 11 PM RFC):

* ``MASTER_SEED = 20260520`` (Phase 10.4 Wave-2 brief).
* ``METHODOLOGY_VERSION = "v3.0"`` (v3.0 release; bumped from v2.0-rev).
* ``MODULE_OFFSETS`` maps DAG module name → integer offset.
* Sub-seed offsets for per-EV streams must be pairwise ≥ ``N_EV_MAX``
  apart so the streams in different modules cannot collide on the same
  ev_id (RFC-001 finding: M5 jitter at +7M and M6 SoC0 at +7M were only
  6 apart per ev_id; M6 SoC0 is now +8M).

Math notation: subscripts in ASCII (e.g. ``soc0`` not ``soc₀``).
"""

from __future__ import annotations

from typing import Final

__all__ = [
    "MASTER_SEED",
    "METHODOLOGY_VERSION",
    "MODULE_OFFSETS",
    "MODULE_OFFSET_ALIASES",
    "module_offset",
    "N_EV_MAX",
    "CLUSTER_ASSIGN_SUBSEED_OFFSET",
    "BERNOULLI_SUBSEED_OFFSET",
    "ARRIVAL_JITTER_SUBSEED_OFFSET",
    "SOC0_SUBSEED_OFFSET",
    "EVSE_BRAND_SAMPLE_SUBSEED_OFFSET",
    "EVSE_CONNECTOR_SAMPLE_SUBSEED_OFFSET",
    "PHEV_GAS_EVENT_SUBSEED_OFFSET",
    "HOUSEID_WEEK_DRAW_SUBSEED_OFFSET",
    "NHTS_VINTAGE_SELECT_SUBSEED_OFFSET",
    "SUBSEED_OFFSETS",
]

#: Phase 10.4 Wave-2 master seed (Phase 10.4 brief).
MASTER_SEED: Final[int] = 20260520

#: Methodology version string written to ``meta.json`` provenance.
METHODOLOGY_VERSION: Final[str] = "v3.0"

#: Per-DAG-module integer offsets. M0 (regions) has no RNG so no offset.
#:
#: ``rng = default_rng(master_seed + MODULE_OFFSETS[name])``.
MODULE_OFFSETS: Final[dict[str, int]] = {
    "regions": 0,
    "vehicle_archetypes": 1,  # M1
    "utc_migration": 1,        # M2 (legacy alias; same offset as M1 keeps v1 fixtures stable)
    "sales_mix_data": 2,       # M2
    "donor_matcher": 2,        # M3 — offset +2 (historical; do not reshuffle)
    "travel_week_builder": 4,  # M4
    "plug_in_model": 5,        # M5
    "soc_trajectory": 6,       # M6
    "hourly_resampler": 6,     # M7 — uses +6 historically (kept stable)
}

#: Pairs of modules in :data:`MODULE_OFFSETS` that intentionally share an
#: offset to preserve byte-identical v1.1 fixtures. Any other pair sharing
#: an offset is treated as a typo / unintentional collision and trips
#: :func:`_assert_module_offsets_unique_unless_aliased`.
MODULE_OFFSET_ALIASES: Final[frozenset[frozenset[str]]] = frozenset({
    frozenset({"vehicle_archetypes", "utc_migration"}),
    frozenset({"sales_mix_data", "donor_matcher"}),
    frozenset({"soc_trajectory", "hourly_resampler"}),
})


def _assert_module_offsets_unique_unless_aliased() -> None:
    """Every module in :data:`MODULE_OFFSETS` has a unique offset, except
    pairs explicitly registered in :data:`MODULE_OFFSET_ALIASES`.

    Protects against the failure mode where a contributor adds
    ``"new_module": 5`` to the offsets dict, not realising 5 is already in
    use by ``plug_in_model``, and silently shares its RNG stream. The
    import-time call below makes this a load-bearing invariant: if it
    fires, the module fails to import.
    """
    from itertools import combinations
    by_offset: dict[int, list[str]] = {}
    for name, offset in MODULE_OFFSETS.items():
        by_offset.setdefault(offset, []).append(name)
    for offset, names in by_offset.items():
        if len(names) <= 1:
            continue
        for a, b in combinations(sorted(names), 2):
            if frozenset({a, b}) not in MODULE_OFFSET_ALIASES:
                raise AssertionError(
                    f"MODULE_OFFSETS: modules {a!r} and {b!r} both map to "
                    f"offset {offset}, but their pair is not in "
                    f"MODULE_OFFSET_ALIASES. If this collision is "
                    f"intentional, register the pair in "
                    f"MODULE_OFFSET_ALIASES; otherwise change one of the "
                    f"offsets."
                )


def module_offset(name: str) -> int:
    """Return the integer DAG offset for the module ``name``.

    Raises :class:`KeyError` for unknown names so a typo doesn't silently
    fall through to a zero offset.
    """
    return MODULE_OFFSETS[name]


#: Upper bound on the number of EVs in a single cache. Used to assert
#: pairwise sub-seed gaps in :func:`_assert_subseed_gaps_ok`.
N_EV_MAX: Final[int] = 1_000_000

# Per-EV sub-seed namespace offsets. Each is an *additive* offset on top of
# ``master_seed + module_offset(module)``. To keep two modules' per-EV
# streams disjoint at all ev_id ≤ N_EV_MAX, the pairwise gap between
# offsets must be ≥ N_EV_MAX (= 1_000_000).

#: M5 plug-in-model — cluster assignment.
CLUSTER_ASSIGN_SUBSEED_OFFSET: Final[int] = 5_000_000
#: M5 plug-in-model — Bernoulli plug-in draw.
BERNOULLI_SUBSEED_OFFSET: Final[int] = 6_000_000
#: M5 plug-in-model — arrival-time jitter.
ARRIVAL_JITTER_SUBSEED_OFFSET: Final[int] = 7_000_000
#: M6 soc-trajectory — initial-SoC draw. Bumped 7_000_000 → 8_000_000 per
#: RFC-001 to avoid collision with M5 arrival-jitter at +7_000_000 per
#: ev_id (gap was only +1 module offset = 1).
SOC0_SUBSEED_OFFSET: Final[int] = 8_000_000

#: M2 vehicle_archetypes — v2.1 EVSE brand draw (RFC for the EVSE
#: enrichment feature). Pairwise gap from SOC0 (8M) -> heat-pump (9M,
#: registered in vehicle_archetypes.HEATPUMP_RESAMPLE_SUBSEED_OFFSET) ->
#: this offset (10M) is exactly N_EV_MAX so the assert passes.
EVSE_BRAND_SAMPLE_SUBSEED_OFFSET: Final[int] = 10_000_000
#: M2 vehicle_archetypes — v2.1 EVSE connector draw. Pairwise gap from
#: EVSE_BRAND (10M) -> EVSE_CONNECTOR (11M) is N_EV_MAX.
EVSE_CONNECTOR_SAMPLE_SUBSEED_OFFSET: Final[int] = 11_000_000

#: M6 PHEV gas-event reservoir (reserved for stochastic plug-or-gas decision; ledger is
#: deterministic in v3.0 so currently unused — registered to reserve the namespace).
PHEV_GAS_EVENT_SUBSEED_OFFSET: Final[int] = 12_000_000

#: M4 HOUSEID-week within-household sampler. Used when houseid_stitch_mode == "v3_within_hh".
HOUSEID_WEEK_DRAW_SUBSEED_OFFSET: Final[int] = 13_000_000

#: M4 NHTS vintage selection (per-EV Bernoulli when nhts_vintage_mix has >1 vintage).
NHTS_VINTAGE_SELECT_SUBSEED_OFFSET: Final[int] = 14_000_000

SUBSEED_OFFSETS: Final[dict[str, int]] = {
    "cluster_assign": CLUSTER_ASSIGN_SUBSEED_OFFSET,
    "bernoulli": BERNOULLI_SUBSEED_OFFSET,
    "arrival_jitter": ARRIVAL_JITTER_SUBSEED_OFFSET,
    "soc0": SOC0_SUBSEED_OFFSET,
    "evse_brand_sample": EVSE_BRAND_SAMPLE_SUBSEED_OFFSET,
    "evse_connector_sample": EVSE_CONNECTOR_SAMPLE_SUBSEED_OFFSET,
    "phev_gas_event": PHEV_GAS_EVENT_SUBSEED_OFFSET,
    "houseid_week_draw": HOUSEID_WEEK_DRAW_SUBSEED_OFFSET,
    "nhts_vintage_select": NHTS_VINTAGE_SELECT_SUBSEED_OFFSET,
}


def _assert_subseed_gaps_ok(n_ev_max: int = N_EV_MAX) -> None:
    """Verify pairwise sub-seed offsets are ≥ ``n_ev_max`` apart.

    Module offsets in MODULE_OFFSETS are O(10) — negligible vs. n_ev_max.
    The relevant check is that, within any module, the per-EV sub-seed
    offsets sit on disjoint integer ranges of length n_ev_max.
    """
    offsets = sorted(SUBSEED_OFFSETS.values())
    # pairwise sliding window: offsets[1:] is intentionally one shorter, so zip must truncate
    for a, b in zip(offsets, offsets[1:], strict=False):
        if b - a < n_ev_max:
            raise AssertionError(
                f"sub-seed offsets {a:_} and {b:_} are < {n_ev_max:_} apart"
            )


# Self-check at import time (cheap, deterministic).
_assert_module_offsets_unique_unless_aliased()
_assert_subseed_gaps_ok()
