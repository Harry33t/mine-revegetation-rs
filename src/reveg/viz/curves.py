"""Restoration-trajectory curve plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def scatter_xy(x, y, path: str | Path, *, xlabel="", ylabel="", title=None, r=None) -> Path:
    """Scatter of two aligned arrays (e.g. canopy-cover fraction vs S2 NDVI)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    import numpy as np

    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    ok = np.isfinite(x) & np.isfinite(y)
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(x[ok], y[ok], s=4, alpha=0.25, edgecolors="none", color="#1b7837")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    t = title or ""
    if r is not None:
        t = f"{t}  (Pearson r = {r:.2f}, n = {int(ok.sum())})"
    ax.set_title(t)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def density_xy(x, y, path: str | Path, *, xlabel="", ylabel="", title=None,
               vline=None, hline=None) -> Path:
    """2D density (hexbin) of two aligned arrays, with optional guide lines.
    Used to show structure(canopy cover) vs greenness(NDVI) decoupling."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    import numpy as np

    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    ok = np.isfinite(x) & np.isfinite(y)
    fig, ax = plt.subplots(figsize=(7, 6))
    hb = ax.hexbin(x[ok], y[ok], gridsize=45, cmap="magma_r", bins="log", mincnt=1)
    fig.colorbar(hb, ax=ax, label="cells (log)")
    if vline is not None:
        ax.axvline(vline, color="#1b7837", ls="--", lw=1)
    if hline is not None:
        ax.axhline(hline, color="#1b7837", ls="--", lw=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_trajectory(df, path: str | Path, *, title: str | None = None) -> Path:
    """Plot mean NDVI + vegetation cover fraction (and recovery fraction if present)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    t = df.index

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(t, df["mean_ndvi"], "o-", color="#1b7837", label="mean NDVI")
    ax1.set_ylabel("mean NDVI", color="#1b7837")
    ax1.tick_params(axis="y", labelcolor="#1b7837")
    ax1.set_xlabel("date")

    ax2 = ax1.twinx()
    ax2.plot(t, df["veg_cover_frac"], "s--", color="#7570b3", label="veg cover fraction")
    if "recovery_fraction" in df:
        ax2.plot(t, df["recovery_fraction"], "^-", color="#d95f02", label="recovery fraction")
    ax2.set_ylabel("fraction", color="#555555")
    ax2.set_ylim(0, 1.05)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=9)
    if title:
        ax1.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
