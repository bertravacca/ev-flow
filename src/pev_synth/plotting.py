"""Optional matplotlib plotting helpers for ``pev_synth``.

This module is **not** imported by ``import pev_synth`` and matplotlib is
**not** a core dependency. Import the helpers explicitly::

    >>> from pev_synth import plotting
    >>> ax = plotting.plot_aggregate_load(fleet.plug_status(t0, t1))

and install the optional extra::

    pip install ev-flow[plotting]

Each helper operates directly on the **DataFrames** returned by the public
:class:`pev_synth.Fleet` / :class:`pev_synth.Profile` API, so the helpers are
unit-testable without a real on-disk cache:

* :func:`plot_aggregate_load` consumes the wide plug-status matrix returned by
  :meth:`Fleet.plug_status` / :meth:`Fleet.generate_presence_absence`
  (``DatetimeIndex`` × ``ev_id`` columns of bool).
* :func:`plot_plugin_time_distribution` and
  :func:`plot_session_energy_distribution` consume the long sessions table
  returned by :meth:`Fleet.charging_sessions` (columns include ``ev_id``,
  ``t_in``, ``t_out``, ``energy_kwh``, ``max_kw``, ...).
* :func:`plot_soc_traces` reconstructs per-EV state-of-charge step traces from
  the sessions table's ``t_in`` / ``t_out`` / ``soc_in_kwh`` /
  ``soc_target_out_kwh`` anchor columns.

Conventions
-----------
* matplotlib (and ``pyplot``) are imported **lazily inside each function** via
  :func:`_import_pyplot`, which raises a friendly :class:`ImportError` when the
  optional extra is missing.
* Every helper accepts an optional pre-existing ``ax`` and returns the
  :class:`matplotlib.axes.Axes` it drew on (creating a fresh figure/axes when
  ``ax`` is ``None``).
* No helper ever calls ``plt.show()`` — rendering / saving is left to the
  caller so the helpers stay headless-friendly (e.g. under ``Agg``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover - typing-only imports
    from matplotlib.axes import Axes

__all__ = [
    "plot_aggregate_load",
    "plot_plugin_time_distribution",
    "plot_session_energy_distribution",
    "plot_soc_traces",
]

_IMPORT_ERROR_MSG = (
    "matplotlib is required for pev_synth.plotting; "
    "install with `pip install ev-flow[plotting]`"
)


def _import_pyplot() -> Any:
    """Return the ``matplotlib.pyplot`` module, or raise a friendly error.

    matplotlib is an optional extra (``ev-flow[plotting]``); importing it
    lazily here keeps plain ``import pev_synth`` free of the dependency.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
        raise ImportError(_IMPORT_ERROR_MSG) from exc
    return plt


def _resolve_ax(ax: Axes | None) -> Axes:
    """Return ``ax`` unchanged, or a freshly created Axes when ``ax`` is None."""
    if ax is not None:
        return ax
    plt = _import_pyplot()
    _, ax = plt.subplots()
    return ax


def plot_aggregate_load(
    plug_status_df: pd.DataFrame,
    ax: Axes | None = None,
    *,
    freq: str | None = None,
    label: str = "plugged-in EVs",
    color: str = "#0072B2",
) -> Axes:
    """Plot the fleet-aggregate plugged-in count over time.

    Parameters
    ----------
    plug_status_df:
        Wide boolean plug-status matrix as returned by
        :meth:`pev_synth.Fleet.plug_status` /
        :meth:`pev_synth.Fleet.generate_presence_absence`: a
        :class:`~pandas.DataFrame` with a ``DatetimeIndex`` and one column per
        ``ev_id`` whose cells are ``True`` when that EV is plugged in.
    ax:
        Pre-existing Axes to draw on. A new figure/axes is created when
        ``None``.
    freq:
        Optional pandas offset alias (e.g. ``"1h"``). When given the per-step
        count is resampled to ``freq`` using the mean, smoothing a 15-min
        matrix onto a coarser grid. When ``None`` (default) the matrix is
        plotted at its native resolution.
    label:
        Legend label for the drawn line.
    color:
        Line color (defaults to a colorblind-safe blue from the Okabe-Ito
        palette).

    Returns
    -------
    matplotlib.axes.Axes
        The Axes the curve was drawn on.
    """
    ax = _resolve_ax(ax)
    # Sum across the per-EV columns -> plugged-in count per timestep.
    if plug_status_df.shape[1] == 0:
        count = pd.Series(dtype=float, index=plug_status_df.index)
    else:
        count = plug_status_df.astype(float).sum(axis=1)
    if freq is not None and not count.empty:
        count = count.resample(freq).mean()
    ax.plot(count.index, count.to_numpy(), color=color, label=label)
    ax.set_xlabel("time")
    ax.set_ylabel("plugged-in EVs (count)")
    ax.set_title("Fleet aggregate plug-in load")
    ax.margins(x=0)
    ax.legend(loc="best")
    return ax


def plot_plugin_time_distribution(
    sessions_df: pd.DataFrame,
    ax: Axes | None = None,
    *,
    bins: int = 24,
    color: str = "#009E73",
    t_in_col: str = "t_in",
) -> Axes:
    """Plot a histogram of charging-session plug-in hour-of-day.

    Parameters
    ----------
    sessions_df:
        Long charging-session table as returned by
        :meth:`pev_synth.Fleet.charging_sessions`. The ``t_in_col`` column must
        be datetime-like (tz-aware is fine; the hour is taken from whatever
        wall-clock the column carries, so convert to a local tz upstream if you
        want local hour-of-day).
    ax:
        Pre-existing Axes to draw on. A new figure/axes is created when
        ``None``.
    bins:
        Number of histogram bins spanning ``[0, 24)`` hours (default 24, i.e.
        one bin per hour).
    color:
        Bar color (defaults to a colorblind-safe green from the Okabe-Ito
        palette).
    t_in_col:
        Name of the plug-in timestamp column (default ``"t_in"``).

    Returns
    -------
    matplotlib.axes.Axes
        The Axes the histogram was drawn on.
    """
    ax = _resolve_ax(ax)
    ts = pd.to_datetime(sessions_df[t_in_col])
    hour = ts.dt.hour.to_numpy(dtype=float) + ts.dt.minute.to_numpy(dtype=float) / 60.0
    ax.hist(hour, bins=bins, range=(0.0, 24.0), color=color, edgecolor="black")
    ax.set_xlabel("plug-in hour of day")
    ax.set_ylabel("session count")
    ax.set_title("Plug-in start-time distribution")
    ax.set_xlim(0.0, 24.0)
    ax.set_xticks(range(0, 25, 6))
    return ax


def plot_session_energy_distribution(
    sessions_df: pd.DataFrame,
    ax: Axes | None = None,
    *,
    bins: int = 30,
    color: str = "#D55E00",
    energy_col: str = "energy_kwh",
) -> Axes:
    """Plot a histogram of per-session delivered energy.

    Parameters
    ----------
    sessions_df:
        Long charging-session table as returned by
        :meth:`pev_synth.Fleet.charging_sessions`. The ``energy_col`` column is
        the grid-side session energy in kWh.
    ax:
        Pre-existing Axes to draw on. A new figure/axes is created when
        ``None``.
    bins:
        Number of histogram bins (default 30).
    color:
        Bar color (defaults to a colorblind-safe orange from the Okabe-Ito
        palette).
    energy_col:
        Name of the session-energy column (default ``"energy_kwh"``).

    Returns
    -------
    matplotlib.axes.Axes
        The Axes the histogram was drawn on.
    """
    ax = _resolve_ax(ax)
    energy = pd.to_numeric(sessions_df[energy_col], errors="coerce").to_numpy(
        dtype=float
    )
    energy = energy[np.isfinite(energy)]
    ax.hist(energy, bins=bins, color=color, edgecolor="black")
    ax.set_xlabel("session energy (kWh)")
    ax.set_ylabel("session count")
    ax.set_title("Session-energy distribution")
    return ax


def plot_soc_traces(
    sessions_df: pd.DataFrame,
    ev_ids: list[int] | None = None,
    ax: Axes | None = None,
    *,
    max_traces: int = 5,
    ev_id_col: str = "ev_id",
    t_in_col: str = "t_in",
    t_out_col: str = "t_out",
    soc_in_col: str = "soc_in_kwh",
    soc_out_col: str = "soc_target_out_kwh",
) -> Axes:
    """Plot per-EV state-of-charge traces reconstructed from sessions.

    The :class:`pev_synth.Profile` API exposes a continuous SoC trajectory only
    as a :class:`~pandas.Series` (:meth:`Profile.soc_trajectory`), not as a
    DataFrame. To stay cache-free and DataFrame-driven, this helper instead
    reconstructs a step trace per EV directly from the sessions table's SoC
    anchor columns: within each session SoC ramps from ``soc_in_col`` to
    ``soc_out_col`` over ``[t_in, t_out)``. This mirrors the in-session ramp of
    :meth:`Profile.soc_trajectory` (the between-session driving discharge is not
    drawn, as it is not encoded in the sessions table alone).

    Parameters
    ----------
    sessions_df:
        Long charging-session table as returned by
        :meth:`pev_synth.Fleet.charging_sessions`.
    ev_ids:
        Specific ``ev_id`` values to plot. When ``None`` (default) the first
        ``max_traces`` distinct EVs present in ``sessions_df`` are used.
    ax:
        Pre-existing Axes to draw on. A new figure/axes is created when
        ``None``.
    max_traces:
        Cap on the number of EV traces drawn (default 5) when ``ev_ids`` is not
        given.
    ev_id_col, t_in_col, t_out_col, soc_in_col, soc_out_col:
        Column-name overrides for the sessions schema.

    Returns
    -------
    matplotlib.axes.Axes
        The Axes the traces were drawn on.
    """
    ax = _resolve_ax(ax)

    if ev_ids is None:
        distinct = list(dict.fromkeys(sessions_df[ev_id_col].tolist()))
        ev_ids = distinct[:max_traces]

    # Colorblind-safe Okabe-Ito cycle + distinct line styles so traces are
    # distinguishable without relying on color alone.
    palette = (
        "#0072B2",
        "#D55E00",
        "#009E73",
        "#CC79A7",
        "#E69F00",
        "#56B4E9",
        "#F0E442",
        "#000000",
    )
    line_styles = ("-", "--", "-.", ":")

    for i, ev in enumerate(ev_ids):
        sub = sessions_df.loc[sessions_df[ev_id_col] == ev].copy()
        if sub.empty:
            continue
        sub = sub.sort_values(t_in_col)
        t_in = pd.to_datetime(sub[t_in_col])
        t_out = pd.to_datetime(sub[t_out_col])
        soc_in = pd.to_numeric(sub[soc_in_col], errors="coerce")
        soc_out = pd.to_numeric(sub[soc_out_col], errors="coerce")
        # Interleave session start/end anchors into a single step trace.
        times = np.empty(2 * len(sub), dtype=object)
        socs = np.empty(2 * len(sub), dtype=float)
        times[0::2] = t_in.to_numpy()
        times[1::2] = t_out.to_numpy()
        socs[0::2] = soc_in.to_numpy(dtype=float)
        socs[1::2] = soc_out.to_numpy(dtype=float)
        ax.plot(
            times,
            socs,
            color=palette[i % len(palette)],
            linestyle=line_styles[i % len(line_styles)],
            marker="o",
            markersize=3,
            label=f"ev_{int(ev):04d}",
        )

    ax.set_xlabel("time")
    ax.set_ylabel("state of charge (kWh)")
    ax.set_title("Per-EV SoC traces")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="best", fontsize="small")
    return ax
