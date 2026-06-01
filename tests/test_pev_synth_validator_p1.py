"""P1 regression: validator default v2 cache root must be absolute and cwd-independent."""

from __future__ import annotations

from pev_synth._paths import cache_dir


def test_v2_default_cache_root_is_absolute(tmp_path, monkeypatch) -> None:
    """Even when cwd changes, cache_dir resolves to an absolute repo-anchored path."""
    monkeypatch.chdir(tmp_path)
    p = cache_dir("bay_area", "residential")
    assert p.is_absolute(), f"{p} is not absolute"
    assert p.name == "residential_ev_synth"
    assert p.parent.name == "bay_area"
    assert p.parent.parent.name == "processed"


def test_v2_default_cache_root_matches_replicate_dir_default() -> None:
    """cache_dir is the r_total=1 alias of replicate_dir; confirm both agree."""
    from pev_synth._paths import replicate_dir
    assert cache_dir("bay_area", "residential") == replicate_dir(
        "bay_area", "residential", 0, 1
    )
