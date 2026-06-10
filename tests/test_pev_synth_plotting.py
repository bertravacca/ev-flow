"""Unit tests for the optional ``pev_synth.plotting`` helpers (F26).

These tests are network- and cache-free: the gitignored ``data/`` tree is not
required. Each helper is exercised against tiny SYNTHETIC DataFrames whose
schemas mirror the real public-API outputs:

* :meth:`pev_synth.Fleet.plug_status` -> wide bool matrix (``DatetimeIndex`` ×
  ``ev_id`` columns) for :func:`plot_aggregate_load`.
* :meth:`pev_synth.Fleet.charging_sessions` -> long sessions table
  (``ev_id``, ``t_in``, ``t_out``, ``soc_in_kwh``, ``soc_target_out_kwh``,
  ``energy_kwh``, ``max_kw``, ...) for the remaining helpers.

The matplotlib ``Agg`` backend is forced before pyplot is touched so the suite
runs headless in CI. Helpers must never call ``plt.show()``; we assert each
returns a non-None Axes that actually has artists drawn on it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

# Make the ``src/`` package importable when run from anywhere.
_CODE_DIR = Path(__file__).resolve().parents[1]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from pev_synth import plotting  # noqa: E402

PEV_TZ = "America/Los_Angeles"


# --------------------------------------------------------------------------- #
# Synthetic fixtures (mirror the real API output schemas).
# --------------------------------------------------------------------------- #


def _make_plug_status(n_evs: int = 4, n_steps: int = 48) -> pd.DataFrame:
    """Wide bool plug-status matrix like ``Fleet.plug_status`` returns."""
    idx = pd.date_range(
        "2001-01-01 00:00", periods=n_steps, freq="1h", tz="UTC"
    )
    idx.name = "ts"
    data = {}
    for e in range(n_evs):
        # Deterministic alternating overnight-style plug pattern per EV.
        plugged = [((h + 2 * e) % 24) < 10 for h in range(n_steps)]
        data[e] = plugged
    df = pd.DataFrame(data, index=idx)
    df.columns.name = "ev_id"
    return df


def _make_sessions(n_evs: int = 4, sessions_per_ev: int = 3) -> pd.DataFrame:
    """Long sessions table like ``Fleet.charging_sessions`` returns."""
    rows = []
    base = pd.Timestamp("2001-01-01 18:00", tz=PEV_TZ)
    for e in range(n_evs):
        for s in range(sessions_per_ev):
            t_in = base + pd.Timedelta(days=s) + pd.Timedelta(hours=e)
            t_out = t_in + pd.Timedelta(hours=8)
            soc_in = 10.0 + e
            soc_out = 50.0 + e
            rows.append(
                {
                    "ev_id": e,
                    "session_idx": s,
                    "t_in": t_in,
                    "t_out": t_out,
                    "soc_in_pct": 20.0,
                    "soc_out_pct": 90.0,
                    "soc_in_kwh": soc_in,
                    "soc_target_out_kwh": soc_out,
                    "energy_kwh": soc_out - soc_in,
                    "energy_kwh_grid": soc_out - soc_in,
                    "energy_kwh_batt": soc_out - soc_in,
                    "max_kw": 7.2,
                    "location": "home",
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture(autouse=True)
def _close_figs():
    """Close any figures a test created so the Agg backend stays tidy."""
    yield
    plt.close("all")


# --------------------------------------------------------------------------- #
# plot_aggregate_load
# --------------------------------------------------------------------------- #


def test_aggregate_load_returns_axes_with_line():
    df = _make_plug_status()
    ax = plotting.plot_aggregate_load(df)
    assert ax is not None
    assert ax.has_data()
    assert len(ax.lines) == 1
    # The summed plug-in count never exceeds the EV count.
    (line,) = ax.lines
    assert line.get_ydata().max() <= df.shape[1]


def test_aggregate_load_respects_existing_ax():
    df = _make_plug_status()
    fig, ax = plt.subplots()
    out = plotting.plot_aggregate_load(df, ax=ax)
    assert out is ax
    assert ax.has_data()


def test_aggregate_load_resample_freq():
    df = _make_plug_status(n_steps=48)
    ax = plotting.plot_aggregate_load(df, freq="6h")
    assert ax.has_data()
    (line,) = ax.lines
    # 48 hourly steps resampled to 6h -> 8 points.
    assert len(line.get_ydata()) == 8


# --------------------------------------------------------------------------- #
# plot_plugin_time_distribution
# --------------------------------------------------------------------------- #


def test_plugin_time_distribution_returns_axes_with_patches():
    df = _make_sessions()
    ax = plotting.plot_plugin_time_distribution(df)
    assert ax is not None
    assert ax.has_data()
    assert len(ax.patches) > 0
    assert ax.get_xlim() == (0.0, 24.0)


def test_plugin_time_distribution_respects_existing_ax():
    df = _make_sessions()
    fig, ax = plt.subplots()
    out = plotting.plot_plugin_time_distribution(df, ax=ax, bins=12)
    assert out is ax
    assert len(ax.patches) > 0


# --------------------------------------------------------------------------- #
# plot_session_energy_distribution
# --------------------------------------------------------------------------- #


def test_session_energy_distribution_returns_axes_with_patches():
    df = _make_sessions()
    ax = plotting.plot_session_energy_distribution(df)
    assert ax is not None
    assert ax.has_data()
    assert len(ax.patches) > 0


def test_session_energy_distribution_respects_existing_ax():
    df = _make_sessions()
    fig, ax = plt.subplots()
    out = plotting.plot_session_energy_distribution(df, ax=ax, bins=10)
    assert out is ax
    assert len(ax.patches) > 0


# --------------------------------------------------------------------------- #
# plot_soc_traces
# --------------------------------------------------------------------------- #


def test_soc_traces_returns_axes_with_lines():
    df = _make_sessions(n_evs=4, sessions_per_ev=3)
    ax = plotting.plot_soc_traces(df, max_traces=3)
    assert ax is not None
    assert ax.has_data()
    # One line per selected EV (capped at max_traces).
    assert len(ax.lines) == 3


def test_soc_traces_explicit_ev_ids():
    df = _make_sessions(n_evs=4, sessions_per_ev=2)
    ax = plotting.plot_soc_traces(df, ev_ids=[1])
    assert ax.has_data()
    assert len(ax.lines) == 1


def test_soc_traces_respects_existing_ax():
    df = _make_sessions()
    fig, ax = plt.subplots()
    out = plotting.plot_soc_traces(df, ax=ax)
    assert out is ax
    assert ax.has_data()


# --------------------------------------------------------------------------- #
# Optional-dependency contract: friendly ImportError when matplotlib is absent.
# --------------------------------------------------------------------------- #


def test_missing_matplotlib_raises_friendly_importerror(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "matplotlib.pyplot" or name.startswith("matplotlib.pyplot"):
            raise ImportError("No module named 'matplotlib'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    with pytest.raises(ImportError, match=r"pip install ev-flow\[plotting\]"):
        plotting._import_pyplot()


def test_plain_import_pev_synth_does_not_import_matplotlib():
    """``import pev_synth`` must not pull matplotlib in (optional extra)."""
    code = (
        "import sys; import pev_synth; "
        "assert 'matplotlib' not in sys.modules, "
        "'plain import pev_synth must not import matplotlib'"
    )
    import subprocess

    res = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(_CODE_DIR),
    )
    assert res.returncode == 0, res.stderr
