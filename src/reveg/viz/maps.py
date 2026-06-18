"""Quicklook rendering: Sentinel-2 RGB composites and colorized class maps."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from ..labels import classes as C


def s2_rgb(composite, *, bands=("B04", "B03", "B02"), pclip=(2, 98)) -> np.ndarray:
    """Percentile-stretched true-colour uint8 RGB from an S2 composite (no time dim)."""
    chans = []
    for b in bands:
        a = np.asarray(composite[b], dtype="float32")
        finite = a[np.isfinite(a)]
        if finite.size:
            lo, hi = np.percentile(finite, pclip)
        else:
            lo, hi = 0.0, 1.0
        a = np.clip((a - lo) / (hi - lo + 1e-9), 0, 1)
        chans.append(np.nan_to_num(a))
    return (np.stack(chans, axis=-1) * 255).astype("uint8")


def colorize(label: np.ndarray) -> np.ndarray:
    """Map a class-id array to an RGB uint8 image via CLASS_COLORS."""
    label = np.asarray(label)
    rgb = np.zeros((*label.shape, 3), dtype="uint8")
    for cid, color in C.CLASS_COLORS.items():
        rgb[label == cid] = color
    return rgb


def _legend_handles(present_ids):
    return [
        mpatches.Patch(color=np.array(C.CLASS_COLORS[c]) / 255.0, label=C.CLASS_NAMES[c])
        for c in present_ids
        if c != C.NODATA
    ]


def save_label_map(label: np.ndarray, path: str | Path, *, title: str | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    present = sorted(int(v) for v in np.unique(label))
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(colorize(label))
    ax.set_axis_off()
    if title:
        ax.set_title(title)
    ax.legend(handles=_legend_handles(present), loc="lower right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def save_mask_overlay(
    rgb: np.ndarray, mask, path: str | Path, *, title: str | None = None, color=(255, 60, 60)
) -> Path:
    """Dim the RGB and highlight `mask` pixels in `color` (e.g. the rehab footprint)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    m = np.asarray(mask).astype(bool)
    over = (rgb.astype("float32") * 0.55).astype("uint8")
    over[m] = np.array(color, dtype="uint8")
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(over)
    ax.set_axis_off()
    if title:
        ax.set_title(title)
    ax.legend(
        handles=[mpatches.Patch(color=np.array(color) / 255.0, label="baseline cleared / rehab")],
        loc="lower right", fontsize=8, framealpha=0.9,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def save_field(
    arr: np.ndarray, path: str | Path, *, title: str | None = None,
    cmap: str = "viridis", vmin=None, vmax=None, label: str | None = None,
) -> Path:
    """Render a continuous raster (CHM, canopy fraction, NDVI) with a colorbar."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(np.asarray(arr, dtype="float32"), cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_axis_off()
    if title:
        ax.set_title(title)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if label:
        cb.set_label(label)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def save_rgb(rgb: np.ndarray, path: str | Path, *, title: str | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(rgb)
    ax.set_axis_off()
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path
