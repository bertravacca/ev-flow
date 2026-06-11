"""Bundled-SPEECh-data tests (F: ship SPEECh K=16 JSONs in the wheel).

These deliberately live in their OWN module with NO skip guard. The main
``test_pev_synth_plug_in_model.py`` is skipped wholesale in CI (it needs the
NHTS/fleet inputs that aren't in a fresh checkout), which would also hide the
bundled-data fallback. The SPEECh JSONs ship as package data, so these run
everywhere — including CI — and are exactly the regression that proves a fresh
``pip install ev-flow`` can build without the upstream Mendeley source.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pev_synth import plug_in_model as pim


@pytest.mark.parametrize("profile_type", ["residential", "workplace"])
def test_bundled_speech_json_ships_and_loads(profile_type: str, tmp_path: Path) -> None:
    """With no locally-derived copy under ``data_root``, the loader falls back to
    the SPEECh K=16 JSON bundled as package data, and it loads successfully."""
    # tmp_path has no data/pev/raw/speech tree, so resolution must pick the bundle.
    resolved = pim._resolve_speech_json_path(profile_type, tmp_path)
    assert resolved == pim._bundled_speech_json_path(profile_type)
    assert resolved.is_file(), "SPEECh JSON should ship as package data"

    sp = pim.load_speech_k16_params(profile_type, data_root=tmp_path)
    assert sp.profile_type == profile_type
    assert len(sp.P_G) == pim.SPEECH_K_TOTAL
    assert sp.source_license == "CC-BY-4.0"


def test_local_copy_takes_precedence_over_bundled(tmp_path: Path) -> None:
    """A locally-derived JSON under ``data_root`` wins over the bundled copy."""
    local = tmp_path / "data" / "pev" / "raw" / "speech"
    local.mkdir(parents=True)
    target = local / "speech_k16_residential.json"
    target.write_text("{}", encoding="utf-8")  # contents irrelevant; path wins
    assert pim._resolve_speech_json_path("residential", tmp_path) == target


def test_missing_everywhere_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """If neither a local nor the bundled copy exists (a broken install), the
    loader raises a clear FileNotFoundError."""
    monkeypatch.setattr(pim, "_BUNDLED_SPEECH_DIR", tmp_path / "nonexistent")
    with pytest.raises(FileNotFoundError, match="broken install"):
        pim.load_speech_k16_params("residential", data_root=tmp_path)
