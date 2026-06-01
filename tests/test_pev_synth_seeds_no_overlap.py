"""RFC-001 / RFC-016 — Sub-seed namespace pairwise-gap regression test.

The per-EV RNG sub-seed offsets used by M5 (cluster_assign, bernoulli,
arrival_jitter) and M6 (soc0) must sit on disjoint integer ranges of
length ``N_EV_MAX`` so that ``master_seed + module_offset +
subseed_offset + ev_id`` collides for no pair of (module, ev_id).
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

_CODE_DIR = Path(__file__).resolve().parents[1]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from pev_synth._seeds import (  # noqa: E402
    N_EV_MAX,
    SUBSEED_OFFSETS,
    _assert_subseed_gaps_ok,
)


def test_pairwise_subseed_gaps_at_least_n_ev_max() -> None:
    """Every pair of sub-seed offsets is ≥ N_EV_MAX apart."""
    pairs = list(combinations(sorted(SUBSEED_OFFSETS.values()), 2))
    assert pairs, "expected at least 2 sub-seed offsets registered"
    for a, b in pairs:
        assert abs(b - a) >= N_EV_MAX, (
            f"sub-seed offsets {a:_} and {b:_} are < {N_EV_MAX:_} apart "
            "— per-EV streams will collide on at least one ev_id"
        )


def test_subseed_assert_helper_runs() -> None:
    """The import-time self-check helper is callable and passes."""
    _assert_subseed_gaps_ok()


def test_soc0_offset_is_8m_not_7m() -> None:
    """RFC-001: bumped from 7_000_000 to 8_000_000 to clear M5 collision."""
    assert SUBSEED_OFFSETS["soc0"] == 8_000_000
    assert SUBSEED_OFFSETS["arrival_jitter"] == 7_000_000


def test_module_offset_aliases_match_declared_collisions() -> None:
    """The aliases registry mirrors the deliberate collisions in MODULE_OFFSETS."""
    from collections import defaultdict

    from pev_synth._seeds import MODULE_OFFSET_ALIASES, MODULE_OFFSETS

    by_offset: dict[int, set[str]] = defaultdict(set)
    for name, off in MODULE_OFFSETS.items():
        by_offset[off].add(name)

    declared_pairs = {frozenset(pair) for pair in MODULE_OFFSET_ALIASES}
    actual_pairs: set[frozenset[str]] = set()
    for names in by_offset.values():
        if len(names) >= 2:
            sorted_names = sorted(names)
            for i in range(len(sorted_names)):
                for j in range(i + 1, len(sorted_names)):
                    actual_pairs.add(frozenset({sorted_names[i], sorted_names[j]}))

    assert actual_pairs == declared_pairs, (
        f"MODULE_OFFSET_ALIASES is out of sync with MODULE_OFFSETS. "
        f"Declared but not present: {declared_pairs - actual_pairs}. "
        f"Present but not declared: {actual_pairs - declared_pairs}."
    )


def test_module_offset_assertion_helper_passes() -> None:
    """The import-time self-check helper is callable and currently passes."""
    from pev_synth._seeds import _assert_module_offsets_unique_unless_aliased
    _assert_module_offsets_unique_unless_aliased()


def test_module_offset_assertion_fires_on_unregistered_collision(
    monkeypatch,
) -> None:
    """Adding a new module that shares an offset without registering the
    alias must trip the guard."""
    from pev_synth import _seeds

    bad_offsets = dict(_seeds.MODULE_OFFSETS)
    bad_offsets["typo_for_plug_in"] = 5  # collides with plug_in_model
    monkeypatch.setattr(_seeds, "MODULE_OFFSETS", bad_offsets)

    import pytest
    with pytest.raises(AssertionError, match="typo_for_plug_in"):
        _seeds._assert_module_offsets_unique_unless_aliased()
