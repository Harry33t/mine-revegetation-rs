"""Rasterize (buffered) building footprints onto the CHM grid.

Day-0 confirmed buildings exceed CHM > 2 m, so without removing them the model
would learn 'roof = canopy'. Buildings are buffered by `building_buffer_m` to
catch eaves / slight misregistration before being removed from the canopy mask.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
from rasterio.features import rasterize


def rasterize_buildings(
    buildings: gpd.GeoDataFrame, transform, shape: tuple, buffer_m: float = 1.0
) -> np.ndarray:
    """Return a bool array (True where a buffered building covers the pixel)."""
    if buildings is None or len(buildings) == 0:
        return np.zeros(shape, dtype=bool)
    geoms = buildings.geometry
    if buffer_m and buffer_m > 0:
        geoms = geoms.buffer(buffer_m)
    shapes = [(g, 1) for g in geoms if g is not None and not g.is_empty]
    if not shapes:
        return np.zeros(shape, dtype=bool)
    burned = rasterize(
        shapes, out_shape=shape, transform=transform, fill=0, dtype="uint8", all_touched=True
    )
    return burned.astype(bool)
