"""Regression tests for red-team citation fixes (H2 cluster).

These tests pin the corrected Quirós-Tortós 2018 PSCC author list
(Quirós-Tortós, Navarro-Espinosa, Ochoa & Butler) in both
``plug_in_model.py`` and ``validation_bounds_curator.py``, and guard
against re-introduction of the historical misattribution to "Lees".

Also pins the workplace cohort caveat in ``plug_in_model.py`` so a
future cleanup pass cannot silently strip the explanatory block.
"""

from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PLUG_IN_MODEL = _REPO_ROOT / "src" / "pev_synth" / "plug_in_model.py"
_BOUNDS_CURATOR = _REPO_ROOT / "src" / "pev_synth" / "validation_bounds_curator.py"


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_quiros_tortos_citation_in_plug_in_model() -> None:
    """plug_in_model.py must cite Navarro-Espinosa alongside Quirós-Tortós."""
    text = _read_text(_PLUG_IN_MODEL)
    # Accept either the accented form or an ASCII-folded variant.
    assert (
        "Quirós-Tortós, Navarro-Espinosa, Ochoa" in text
        or "Quiros-Tortos, Navarro-Espinosa, Ochoa" in text
    ), "Quirós-Tortós citation in plug_in_model.py is missing Navarro-Espinosa"
    assert "Butler" in text, "Butler must appear in the corrected author list"
    # Keep the IEEE doc number anchor so we don't lose provenance.
    assert "8442988" in text


def test_quiros_tortos_citation_in_bounds_curator() -> None:
    """validation_bounds_curator.py must cite Navarro-Espinosa alongside Quirós-Tortós."""
    text = _read_text(_BOUNDS_CURATOR)
    assert (
        "Quirós-Tortós, Navarro-Espinosa, Ochoa" in text
        or "Quiros-Tortos, Navarro-Espinosa, Ochoa" in text
    ), "Quirós-Tortós citation in validation_bounds_curator.py is missing Navarro-Espinosa"
    assert "Butler" in text
    assert "8442988" in text


def test_no_lees_misattribution() -> None:
    """No mention of "Lees" in either module — the 2018 PSCC paper has no Lees author."""
    for path in (_PLUG_IN_MODEL, _BOUNDS_CURATOR):
        text = _read_text(path)
        # Use a word-boundary regex so we don't accidentally trip on
        # substrings inside unrelated identifiers (none currently exist,
        # but the boundary keeps the test robust).
        matches = re.findall(r"\bLees\b", text)
        assert matches == [], (
            f"'Lees' still appears in {path.name}: "
            f"{len(matches)} match(es). The 2018 PSCC paper authors are "
            "Quirós-Tortós, Navarro-Espinosa, Ochoa & Butler."
        )


def test_workplace_cohort_caveat_present_in_plug_in_model() -> None:
    """The workplace-cohort caveat block (N=105, ~12:00 LT) must remain in
    the top of plug_in_model.py so callers reading the docstring see it.
    """
    text = _read_text(_PLUG_IN_MODEL)
    head = "\n".join(text.splitlines()[:80])
    assert "105" in head, "workplace cohort size N=105 missing from caveat"
    assert "12:00" in head, "workplace plug-in median ~12:00 LT missing from caveat"
    assert "EXPLAINED_FAIL" in head, (
        "workplace caveat must say W1-W4 flag the divergence as EXPLAINED_FAIL"
    )


# ----------------------------------------------------------------------
# H7 — INL/EXT-15-36317 year-of-publication string is 2015 not 2018.
# ADDED BY QA: H7 is a wording-only citation fix; pin it with a grep
# invariant so a future cleanup pass cannot silently flip the year back.
# ----------------------------------------------------------------------

def test_h7_inl_ext_smart_salisbury_dated_2015_not_2018() -> None:
    """The ``-15-`` prefix on INL/EXT-15-36317 is unambiguous — the
    report is 2015. The historical citation incorrectly said 2018.
    """
    text = _read_text(_BOUNDS_CURATOR)
    assert "(Smart & Salisbury, 2015)" in text, (
        "validation_bounds_curator.py must cite Smart & Salisbury, 2015 "
        "(not 2018) for INL/EXT-15-36317."
    )
    assert "(Smart & Salisbury, 2018)" not in text, (
        "validation_bounds_curator.py must not cite Smart & Salisbury, 2018 "
        "for INL/EXT-15-36317: the -15- prefix is a 2015 report number."
    )
