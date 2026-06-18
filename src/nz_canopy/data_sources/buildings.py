"""Building-outline reader. Reprojects to the target CRS (EPSG:2193)."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd

from ..utils import paths
from ..utils.crs import TARGET_CRS


def load_buildings(city: str | None = None, *, folder: str | Path | None = None) -> gpd.GeoDataFrame:
    f = Path(folder) if folder is not None else paths.layer_dir(city, "buildings")
    shps = sorted(Path(f).glob("*.shp"))
    if not shps:
        raise FileNotFoundError(f"No building shapefile (*.shp) found in {f}")
    gdf = gpd.read_file(shps[0])
    if gdf.crs is None:
        raise ValueError(f"Building layer {shps[0]} has no CRS")
    return gdf.to_crs(TARGET_CRS)
