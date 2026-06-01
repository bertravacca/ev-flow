"""Pin doc / source invariants from the red-team remediation effort.

These are cheap text-level assertions to keep future doc rewrites from
silently re-introducing v1.1-era staleness. Each test maps to one of the
red-team findings noted in ``docs/red_team_r.md``.
"""

from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_API_MD = _REPO_ROOT / "docs" / "pev_synth_api.md"
_DIFF_MD = _REPO_ROOT / "docs" / "pev_synth_differentiation.md"
_README = _REPO_ROOT / "README.md"
_VALIDATOR = _REPO_ROOT / "src" / "pev_synth" / "validator.py"
_HOURLY = _REPO_ROOT / "src" / "pev_synth" / "hourly_resampler.py"


def test_api_md_title_is_versioned() -> None:
    """C3: api.md must declare the current v2.1 release in its first line."""
    first_line = _API_MD.read_text(encoding="utf-8").splitlines()[0]
    assert "v2.1" in first_line, first_line


def test_api_md_no_fleet_depot() -> None:
    """P3 / C3: fleet_depot is descoped from v2.0; no mention in api.md."""
    content = _API_MD.read_text(encoding="utf-8")
    assert "fleet_depot" not in content, (
        "api.md still mentions 'fleet_depot' — it was descoped in v2.0."
    )


def test_differentiation_no_hale_lavin() -> None:
    """H1: replace 'Hale 2022 / Lavin-Hale 2026' attribution in the diff doc."""
    content = _DIFF_MD.read_text(encoding="utf-8")
    assert "Hale 2022" not in content
    assert "Lavin-Hale 2026" not in content


def test_hourly_resampler_no_hale_lavin() -> None:
    """H1: same scrub in the hourly_resampler module-level docstring."""
    content = _HOURLY.read_text(encoding="utf-8")
    assert "Hale 2022" not in content
    assert "Lavin-Hale 2026" not in content


def test_validator_check_count_matches_doc() -> None:
    """H4: validator ships a check-family roll-up, not '11 checks'.

    Loose lower bound — confirm the actual count is >= 20 (the doc
    states 11 + 3 + 1 + 1 + 10 + 1 ≈ 27 callables).
    """
    src = _VALIDATOR.read_text(encoding="utf-8")
    count = len(re.findall(r"(?m)^def check_", src))
    assert count >= 20, f"expected >= 20 check_* callables, found {count}"
