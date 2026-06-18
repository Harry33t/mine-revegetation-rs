"""DSM / DEM readers.

A city's LiDAR may be one tile (Christchurch BX24.tif) or many (8-11 tiles
elsewhere). `rasterio.merge.merge` handles both uniformly — merging one tile
returns that tile — so there is a single code path and no per-city special case.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.merge import merge

from ..utils import paths
from ..utils.crs import assert_target_crs


@dataclass
class RasterLayer:
    array: np.ndarray  # 2D float32
    transform: object
    crs: object
    nodata: float | None
    bounds: tuple  # (left, bottom, right, top)
    res: tuple  # (xres, yres), both positive

    @property
    def shape(self) -> tuple:
        return self.array.shape


def _merge_dir(folder: Path, *, band: int = 1) -> RasterLayer:
    files = sorted(Path(folder).glob("*.tif"))
    if not files:
        raise FileNotFoundError(f"No .tif files found in {folder}")
    srcs = [rasterio.open(f) for f in files]
    try:
        arr, transform = merge(srcs)
        crs = srcs[0].crs
        nodata = srcs[0].nodata
    finally:
        for s in srcs:
            s.close()
    a = arr[band - 1].astype(np.float32)
    h, w = a.shape
    left, top = transform.c, transform.f
    bottom = top + h * transform.e  # transform.e is negative for north-up
    right = left + w * transform.a
    res = (abs(transform.a), abs(transform.e))
    return RasterLayer(a, transform, crs, nodata, (left, bottom, right, top), res)


def load_dsm(city: str | None = None, *, folder: str | Path | None = None) -> RasterLayer:
    f = Path(folder) if folder is not None else paths.layer_dir(city, "dsm")
    layer = _merge_dir(f)
    assert_target_crs(layer.crs, where=f"{city or f} DSM")
    return layer


def load_dem(city: str | None = None, *, folder: str | Path | None = None) -> RasterLayer:
    f = Path(folder) if folder is not None else paths.layer_dir(city, "dem")
    layer = _merge_dir(f)
    assert_target_crs(layer.crs, where=f"{city or f} DEM")
    return layer


@dataclass
class DsmDemPair:
    dsm: RasterLayer
    dem: RasterLayer


def load_dsm_dem(city: str | None = None, *, dsm_folder=None, dem_folder=None) -> DsmDemPair:
    dsm = load_dsm(city, folder=dsm_folder)
    dem = load_dem(city, folder=dem_folder)
    if dsm.array.shape != dem.array.shape:
        raise ValueError(f"DSM/DEM shape mismatch: {dsm.array.shape} vs {dem.array.shape}")
    return DsmDemPair(dsm, dem)
