"""Clip vector layers to a bbox in the target CRS (mirrors the Day-0 pattern)."""

from __future__ import annotations

import geopandas as gpd
from shapely.geometry import box

from ..utils.crs import TARGET_CRS, to_epsg


def clip_to_bbox(gdf: gpd.GeoDataFrame, bbox: tuple, target_crs: str = TARGET_CRS) -> gpd.GeoDataFrame:
    """Return rows of `gdf` intersecting bbox=(left, bottom, right, top), reprojected to target_crs."""
    if gdf.crs is None:
        raise ValueError("Input GeoDataFrame has no CRS")
    g = gdf if to_epsg(gdf.crs) == to_epsg(target_crs) else gdf.to_crs(target_crs)
    region = box(*bbox)
    return g[g.geometry.intersects(region)].copy()
