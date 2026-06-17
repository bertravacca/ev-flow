#!/usr/bin/env python3
"""F35 hero figure: aggregate EV-fleet charging load curve.

Regenerates ``website/assets/hero-load-curve.png`` (and ``.svg``) from the
validated ``bay_area`` residential cache via the public ``pev_synth`` API.
No data is fabricated: every point is ``Fleet.aggregate_load`` over one
representative week.

Reproducibility
---------------
* profile_type : residential
* region       : bay_area
* n            : 1000
* seed         : 42
* window       : 2001-07-09 (Mon) .. 2001-07-16 (Mon, exclusive), local time
* freq         : 15min

Run from anywhere::

    python3 scripts/make_hero_figure.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless, deterministic raster/vector output

import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

import pev_synth as ps  # noqa: E402

# --- reproducibility constants (hard-coded, see module docstring) ---------
PROFILE_TYPE = "residential"
REGION = "bay_area"
N = 1000
SEED = 42
WEEK_START = "2001-07-09"  # Monday on the 2001 synthetic calendar
WEEK_STOP = "2001-07-16"  # next Monday (half-open window -> Mon..Sun)
FREQ = "15min"

# --- brand palette (docs/v4/brand/generate_logo.py) -----------------------
INK = "#16271F"  # deep evergreen-slate
ACCENT = "#3FA351"  # leaf green (the load-curve accent)
TEAL = "#00B3A4"  # teal (peak marker)
WEEKEND_SHADE = "#16271F"  # ink, drawn at low alpha

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "website" / "assets"


def main() -> None:
    fleet = ps.generate_profiles(
        PROFILE_TYPE, n=N, region=REGION, seed=SEED
    )
    tz = fleet.region.tz
    load = fleet.aggregate_load(WEEK_START, WEEK_STOP, freq=FREQ, tz=tz)

    peak_t = load.idxmax()
    peak_kw = float(load.max())

    # ~1600 px wide at 200 dpi -> 8.0 in; 3.6 in tall keeps a wide hero ratio.
    fig, ax = plt.subplots(figsize=(8.0, 3.6), dpi=200)

    # Shade weekends (Sat/Sun) so the weekday/weekend pattern reads at a glance.
    for day in load.index.normalize().unique():
        if day.weekday() >= 5:  # 5=Sat, 6=Sun
            ax.axvspan(
                day,
                day + (load.index[1] - load.index[0]) * 96,
                color=WEEKEND_SHADE,
                alpha=0.05,
                zorder=0,
            )

    ax.plot(
        load.index,
        load.to_numpy(),
        color=ACCENT,
        linewidth=1.8,
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=3,
    )
    ax.fill_between(
        load.index,
        load.to_numpy(),
        color=ACCENT,
        alpha=0.12,
        zorder=2,
    )

    # Annotate the nightly evening peak.
    ax.scatter(
        [peak_t], [peak_kw], color=TEAL, s=28, zorder=4, edgecolor="white",
        linewidth=0.8,
    )
    ax.annotate(
        f"evening peak\n{peak_kw:,.0f} kW",
        xy=(peak_t, peak_kw),
        xytext=(10, -4),
        textcoords="offset points",
        fontsize=8.5,
        color=INK,
        va="top",
        ha="left",
    )

    # Axes cosmetics.
    ax.set_ylabel("aggregate charging power [kW]", fontsize=10, color=INK)
    ax.set_xlabel("local time (America/Los_Angeles)", fontsize=10, color=INK)
    ax.set_title(
        "Synthetic EV fleet charging load - 1,000 residential EVs, Bay Area",
        fontsize=11.5,
        color=INK,
        pad=8,
    )

    ax.set_xlim(load.index[0], load.index[-1])
    ax.set_ylim(0, peak_kw * 1.18)

    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a", tz=load.index.tz))
    ax.tick_params(axis="both", labelsize=9, colors=INK)

    ax.grid(True, which="major", axis="y", color=INK, alpha=0.08, linewidth=0.7)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(INK)
        ax.spines[spine].set_alpha(0.3)

    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUT_DIR / "hero-load-curve.png"
    svg_path = OUT_DIR / "hero-load-curve.svg"
    fig.savefig(png_path, dpi=200, facecolor="white")
    fig.savefig(svg_path, facecolor="white")
    plt.close(fig)

    print(
        "# provenance: "
        f"{PROFILE_TYPE} | region={REGION} | n={N} | seed={SEED} | "
        f"window={WEEK_START}..{WEEK_STOP} ({FREQ}, {tz}) | "
        f"peak={peak_kw:,.0f} kW @ {peak_t} | "
        f"pev_synth {ps.__version__}"
    )
    print(f"wrote {png_path}")
    print(f"wrote {svg_path}")


if __name__ == "__main__":
    main()
