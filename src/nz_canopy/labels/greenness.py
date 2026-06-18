"""Aerial greenness gate to suppress rooftop false positives.

LiDAR height + building footprints alone leave rooftop residual where a building
has no (or an undersized) footprint: a roof is tall (CHM >= 2 m) but not a tree.
Trees are green, roofs are not -- so a chromatic Excess-Green index computed from
the aerial RGB, resampled onto the CHM grid, gates them out: a tall, non-green
region is a roof, not canopy. This catches roofs that the footprint mask misses.

Chromatic (normalized) ExG = 2g - r - b on r,g,b = R,G,B / (R+G+B) is fairly
illumination-robust. Grass is also green but is < 2 m, so already excluded by the
height threshold -- height + greenness together are clean.
"""

from __future__ import annotations

import numpy as np
from rasterio.transform import from_bounds
from rasterio.warp import Resampling, reproject
from scipy.ndimage import label as _label


def normalized_exg(img_rgb: np.ndarray) -> np.ndarray:
    """Chromatic Excess-Green (2g - r - b) on normalized RGB; HxW float32 in ~[-1,2]."""
    rgb = img_rgb[..., :3].astype(np.float32)
    s = rgb.sum(axis=2)
    s[s == 0] = 1.0
    r, g, b = rgb[..., 0] / s, rgb[..., 1] / s, rgb[..., 2] / s
    return (2.0 * g - r - b).astype(np.float32)


def greenness_on_chm(aerial_img: np.ndarray, aerial_extent, chm, *, exg_threshold: float = 0.0):
    """Resample aerial ExG to the CHM grid. Returns (exg_chm float32, green bool)."""
    left, right, bottom, top = aerial_extent  # imagery extent order
    h, w = aerial_img.shape[:2]
    src_transform = from_bounds(left, bottom, right, top, w, h)
    exg_src = normalized_exg(aerial_img)
    exg_chm = np.full(chm.array.shape, np.nan, dtype=np.float32)
    reproject(
        source=exg_src, destination=exg_chm,
        src_transform=src_transform, src_crs=chm.crs,
        dst_transform=chm.transform, dst_crs=chm.crs,
        resampling=Resampling.average,
    )
    green = np.nan_to_num(exg_chm, nan=-1.0) >= exg_threshold
    return exg_chm, green


def suppress_low_green_components(mask, exg_chm, *, exg_comp_threshold: float = 0.04, min_area_px: int = 40):
    """Drop labeled blobs whose *median* ExG is roof-low.

    The per-pixel ExG>=0 gate still leaks on large flat/dark/metal commercial roofs
    that read faintly green (ExG ~0.01-0.04) and have no LINZ footprint. Real tree
    crowns sit much higher (ExG ~0.07-0.25). So a connected component that is both
    large and faintly-green is an un-footprinted roof, not canopy -> remove it whole.
    Small components are spared (a shadowed real tree can read low) so this only
    targets the big roof residuals. Returns a new mask.
    """
    if exg_chm is None:
        return mask
    e = np.nan_to_num(exg_chm, nan=-1.0)
    lab, n = _label(mask)
    out = mask.copy()
    for i in range(1, n + 1):
        comp = lab == i
        if comp.sum() >= min_area_px and float(np.median(e[comp])) < exg_comp_threshold:
            out[comp] = False
    return out
