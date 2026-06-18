"""Terrain slope from the DEM (bare earth), for the M2.1 slope audit.

Slope is computed with numpy.gradient (no richdem/gdal dependency):
    slope_deg = degrees(arctan(hypot(dz/dx, dz/dy)))
where gradients are scaled by the pixel size in metres.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# (label, lo_inclusive, hi_exclusive_deg)
SLOPE_BINS = [("flat", 0, 5), ("gentle", 5, 10), ("moderate", 10, 20), ("steep", 20, 1e9)]


def classify_slope(deg: float) -> str:
    if deg is None or not np.isfinite(deg):
        return "unknown"
    for label, lo, hi in SLOPE_BINS:
        if lo <= deg < hi:
            return label
    return "steep"


def compute_slope_deg(dem: np.ndarray, res: tuple, nodata: float | None = None) -> np.ndarray:
    """Per-pixel slope in degrees. NaN where DEM is invalid."""
    dem = np.asarray(dem, dtype=np.float64)
    z = dem.copy()
    if nodata is not None:
        z[z == nodata] = np.nan
    xres, yres = abs(res[0]), abs(res[1])
    # np.gradient returns (d/drow, d/dcol); scale by metres per pixel
    dzdy, dzdx = np.gradient(z, yres, xres)
    slope = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))
    return slope.astype(np.float32)


def _crop(array, bounds, res, bbox):
    left, bottom, right, top = bounds
    xres, yres = res
    bx0, by0, bx1, by1 = bbox
    h, w = array.shape[:2]
    c0 = max(0, int(np.floor((bx0 - left) / xres)))
    c1 = min(w, int(np.ceil((bx1 - left) / xres)))
    r0 = max(0, int(np.floor((top - by1) / yres)))
    r1 = min(h, int(np.ceil((top - by0) / yres)))
    return array[r0:r1, c0:c1]


def tile_slope_stats(slope: np.ndarray, dem_layer, grid) -> pd.DataFrame:
    """Per-tile slope stats over the M1 tile grid. dem_layer supplies bounds/res."""
    rows = []
    for _, tile in grid.iterrows():
        bbox = tuple(tile.geometry.bounds)
        sub = _crop(slope, dem_layer.bounds, dem_layer.res, bbox)
        vals = sub[np.isfinite(sub)]
        if vals.size:
            mean_s = float(np.mean(vals))
            rows.append(
                {
                    "tile_id": tile.tile_id,
                    "mean_slope_deg": round(mean_s, 2),
                    "p95_slope_deg": round(float(np.percentile(vals, 95)), 2),
                    "max_slope_deg": round(float(np.max(vals)), 2),
                    "slope_bin": classify_slope(mean_s),
                }
            )
        else:
            rows.append({"tile_id": tile.tile_id, "mean_slope_deg": np.nan,
                         "p95_slope_deg": np.nan, "max_slope_deg": np.nan, "slope_bin": "unknown"})
    return pd.DataFrame(rows)
