"""Coverage sweep for small ``pev_synth`` modules toward 100%.

This file targets the few remaining uncovered lines in five lightweight
modules, exercising the reachable defensive / branch paths with minimal
synthetic inputs and monkeypatch. It is fully data-independent: the gitignored
``data/`` tree is NOT required (no cache reads, no NHTS, no parquet on disk).

Covered remainders by module:

* ``regions.py``    — :attr:`Region.nhts_vintage_mix_map` property and the
  ``__eq__`` ``NotImplemented`` branch (compare against a non-Region).
* ``_paths.py``     — :func:`intermediate_root` and both branches of
  :func:`nhts_intermediate_root` (region=None and a region subdir).
* ``_seeds.py``     — the :func:`_assert_subseed_gaps_ok` ``AssertionError``
  raise, triggered with an oversized ``n_ev_max``.
* ``plotting.py``   — the empty-DataFrame branch of :func:`plot_aggregate_load`
  and the empty-``sub`` ``continue`` branch of :func:`plot_soc_traces`.
* ``api.py``        — :func:`generate_profiles` with the default
  ``data_root=None`` so the canonical-cache-path branch runs (then raises
  ``FileNotFoundError`` because no cache exists in this data-free env).

Genuinely uncoverable without unpublished deps / a real on-disk cache, so
NOT attempted here:

* ``api.py`` 1421-1424 — ``Fleet._to_optimizer_inputs`` body runs only after a
  successful ``import pev_optim`` (the unpublished sibling package); the
  ImportError guard above it is the only data-free-reachable path.
* ``api.py`` 1662 inner success — ``Fleet.cache_path`` is *called* here (we do
  cover the call), but the *successful load past it* needs a real
  ``fleet.parquet`` on disk, which does not exist in the CI / data-free env.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from pev_synth import generate_profiles, plotting  # noqa: E402
from pev_synth._paths import (  # noqa: E402
    intermediate_root,
    nhts_intermediate_root,
)
from pev_synth._seeds import _assert_subseed_gaps_ok  # noqa: E402
from pev_synth.regions import Region  # noqa: E402

MASTER_SEED = 20260520


# --------------------------------------------------------------------------- #
# regions.py
# --------------------------------------------------------------------------- #
def test_nhts_vintage_mix_map_is_readonly_view() -> None:
    """`nhts_vintage_mix_map` surfaces the tuple-of-pairs as an immutable map."""
    bay = Region.from_name("bay_area")
    m = bay.nhts_vintage_mix_map
    assert dict(m) == dict(bay.nhts_vintage_mix)
    # Default vintage mix is the single 2017 NHTS wave at full weight.
    assert m["2017"] == 1.0
    with pytest.raises(TypeError):
        m["2017"] = 0.5  # type: ignore[index]


def test_region_eq_against_non_region_is_not_implemented() -> None:
    """Comparing a Region to a foreign type hits the NotImplemented branch."""
    bay = Region.from_name("bay_area")
    # __eq__ returns NotImplemented -> Python falls back to identity, giving False.
    assert (bay == "bay_area") is False
    assert (bay != object()) is True
    assert bay == Region.from_name("bay_area")


# --------------------------------------------------------------------------- #
# _paths.py
# --------------------------------------------------------------------------- #
def test_intermediate_root_under_data_root(monkeypatch, tmp_path) -> None:
    """intermediate_root() chains through data_root()."""
    monkeypatch.setenv("PEV_SYNTH_DATA_ROOT", str(tmp_path))
    assert intermediate_root() == tmp_path.resolve() / "pev" / "intermediate"


def test_nhts_intermediate_root_both_branches(monkeypatch, tmp_path) -> None:
    """region=None returns the base; a region name returns the subdir."""
    monkeypatch.setenv("PEV_SYNTH_DATA_ROOT", str(tmp_path))
    base = tmp_path.resolve() / "pev" / "intermediate" / "nhts"
    assert nhts_intermediate_root() == base
    assert nhts_intermediate_root(None) == base
    assert nhts_intermediate_root("bay_area") == base / "bay_area"


# --------------------------------------------------------------------------- #
# _seeds.py
# --------------------------------------------------------------------------- #
def test_subseed_gap_assert_passes_at_default() -> None:
    """Default n_ev_max keeps all adjacent offset gaps >= n_ev_max."""
    # Should not raise.
    _assert_subseed_gaps_ok()


def test_subseed_gap_assert_raises_when_n_ev_max_too_large() -> None:
    """An oversized n_ev_max makes the 1M-spaced offsets collide -> raise."""
    with pytest.raises(AssertionError, match="apart"):
        _assert_subseed_gaps_ok(n_ev_max=2_000_000)


# --------------------------------------------------------------------------- #
# plotting.py
# --------------------------------------------------------------------------- #
def test_plot_aggregate_load_empty_columns() -> None:
    """A zero-column plug-status matrix yields an empty count Series, no error."""
    idx = pd.date_range("2026-01-01", periods=4, freq="15min", tz="UTC")
    empty = pd.DataFrame(index=idx)  # zero columns -> shape[1] == 0
    assert empty.shape[1] == 0
    ax = plotting.plot_aggregate_load(empty)
    assert ax is not None
    assert ax.get_ylabel() == "plugged-in EVs (count)"
    plt.close(ax.figure)


def test_plot_soc_traces_skips_missing_ev_id() -> None:
    """An ev_id absent from the sessions table produces an empty sub -> continue."""
    idx = pd.date_range("2026-01-01", periods=2, freq="h", tz="UTC")
    sessions = pd.DataFrame(
        {
            "ev_id": [0, 0],
            "t_in": [idx[0], idx[1]],
            "t_out": [idx[0] + pd.Timedelta(minutes=30), idx[1] + pd.Timedelta(minutes=30)],
            "soc_in_kwh": [10.0, 20.0],
            "soc_target_out_kwh": [40.0, 50.0],
        }
    )
    # ev_id 999 is missing -> the loop body hits `if sub.empty: continue`.
    ax = plotting.plot_soc_traces(sessions, ev_ids=[999, 0])
    assert ax is not None
    plt.close(ax.figure)


# --------------------------------------------------------------------------- #
# api.py
# --------------------------------------------------------------------------- #
def test_generate_profiles_default_data_root_resolves_canonical_cache_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """data_root=None drives the canonical Fleet.cache_path branch.

    Point the data root at an empty dir so the runtime-resolved canonical
    cache path does not exist (covering the branch) and a FileNotFoundError
    is raised, regardless of whether a real data/ tree is present locally.
    """
    monkeypatch.setenv("PEV_SYNTH_DATA_ROOT", str(tmp_path / "empty_root"))
    with pytest.raises(FileNotFoundError, match="No cached fleet"):
        generate_profiles(
            profile_type="residential",
            n=1,
            region="bay_area",
            seed=MASTER_SEED,
            data_root=None,
        )
