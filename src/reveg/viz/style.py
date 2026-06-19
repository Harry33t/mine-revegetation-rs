"""Shared, restrained figure style so all matplotlib outputs look consistent
and publication-clean (light background, thin spines, tabular feel)."""

from __future__ import annotations

import matplotlib as mpl


def apply() -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 150,
        "figure.dpi": 120,
        "axes.titlesize": 12.5,
        "axes.titleweight": "bold",
        "axes.titlecolor": "#1c1c1a",
        "axes.labelcolor": "#444444",
        "axes.labelsize": 11,
        "axes.edgecolor": "#cfcdc2",
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#ececE4",
        "grid.linewidth": 0.6,
        "text.color": "#1c1c1a",
        "xtick.color": "#666666",
        "ytick.color": "#666666",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.frameon": False,
        "legend.fontsize": 10,
    })
