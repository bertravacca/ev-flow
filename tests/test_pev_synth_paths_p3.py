"""P3 regression: PEV_SYNTH_DATA_ROOT env var overrides data_root()."""

from __future__ import annotations

from pev_synth._paths import (
    cache_dir,
    data_root,
    processed_root,
    raw_root,
    repo_root,
)


def test_data_root_default_anchors_under_repo_root(monkeypatch) -> None:
    """No env var → data_root() is <repo_root>/data."""
    monkeypatch.delenv("PEV_SYNTH_DATA_ROOT", raising=False)
    assert data_root() == repo_root() / "data"


def test_data_root_env_var_overrides(monkeypatch, tmp_path) -> None:
    """Env var → data_root() points at the env path (absolute)."""
    monkeypatch.setenv("PEV_SYNTH_DATA_ROOT", str(tmp_path))
    p = data_root()
    assert p.is_absolute()
    assert p == tmp_path.resolve()


def test_downstream_helpers_inherit_env_var(monkeypatch, tmp_path) -> None:
    """raw_root / processed_root / cache_dir all chain through data_root()."""
    monkeypatch.setenv("PEV_SYNTH_DATA_ROOT", str(tmp_path))
    assert raw_root() == tmp_path.resolve() / "pev" / "raw"
    assert processed_root() == tmp_path.resolve() / "pev" / "processed"
    assert cache_dir("bay_area", "residential") == (
        tmp_path.resolve() / "pev" / "processed" / "bay_area" / "residential_ev_synth"
    )


def test_env_var_expands_user_home(monkeypatch) -> None:
    """A '~/' env value is expanded to the home directory."""
    monkeypatch.setenv("PEV_SYNTH_DATA_ROOT", "~/some-evflow-data")
    p = data_root()
    assert "~" not in str(p)
    assert p.is_absolute()
