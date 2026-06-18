"""CHM for a city: load aligned DSM/DEM (M1) and compute CHM = DSM - DEM."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio

from ..data_sources.elevation import load_dsm_dem
from ..processing.raster_align import compute_chm


@dataclass
class CHM:
    array: np.ndarray  # 2D float32, NaN where invalid
    transform: object
    crs: object
    bounds: tuple  # (left, bottom, right, top)
    res: tuple


def chm_for_city(city: str | None = None, *, dsm_folder=None, dem_folder=None) -> CHM:
    pair = load_dsm_dem(city, dsm_folder=dsm_folder, dem_folder=dem_folder)
    arr = compute_chm(pair.dsm.array, pair.dem.array, pair.dsm.nodata, pair.dem.nodata)
    return CHM(arr, pair.dsm.transform, pair.dsm.crs, pair.dsm.bounds, pair.dsm.res)


def save_chm_tif(chm: CHM, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = chm.array.shape
    with rasterio.open(
        path, "w", driver="GTiff", height=h, width=w, count=1, dtype="float32",
        crs=chm.crs, transform=chm.transform, nodata=np.nan,
    ) as dst:
        dst.write(chm.array, 1)
    return path
