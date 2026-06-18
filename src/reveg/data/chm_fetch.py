"""Fetch the Meta/WRI 1 m global canopy-height (Tolan 2024) over an AOI.

This is the ON-MINE fine-scale structural layer for the scale bridge: it covers
Australia (and all our mine AOIs) at ~1 m, CC-BY, on the AWS Open Data bucket
`dataforgood-fb-data`. Honest caveat: it is a MODEL estimate (~2-3 m MAE), not a
drone/LiDAR ground truth — but it is co-located with the actual mine, which is the
point. Tiles are Bing quadkeys (zoom 9) in EPSG:3857; read a window via /vsicurl/.
"""

from __future__ import annotations

import math

import numpy as np
import rasterio
from affine import Affine
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds as window_from_bounds

BASE = "https://dataforgood-fb-data.s3.amazonaws.com/forests/v1/alsgedi_global_v6_float/chm/"
_ENV = dict(GDAL_HTTP_MAX_RETRY="8", GDAL_HTTP_RETRY_DELAY="1", GDAL_HTTP_TIMEOUT="60")


def lonlat_to_quadkey(lon: float, lat: float, z: int = 9) -> str:
    """Bing Maps quadkey for a lon/lat at zoom `z` (Meta CHM tiles are zoom 9)."""
    sinlat = math.sin(lat * math.pi / 180.0)
    x = (lon + 180.0) / 360.0
    y = 0.5 - math.log((1 + sinlat) / (1 - sinlat)) / (4 * math.pi)
    n = 2 ** z
    tx, ty = int(x * n), int(y * n)
    qk = ""
    for i in range(z, 0, -1):
        d, m = 0, 1 << (i - 1)
        if tx & m:
            d += 1
        if ty & m:
            d += 2
        qk += str(d)
    return qk


def fetch_chm(bbox_lonlat, *, z: int = 9, max_height_m: float = 120.0, out_res_m: float | None = None):
    """Read Meta CHM over an AOI. Returns (chm float32 [NaN nodata], transform, crs).

    bbox_lonlat = (west, south, east, north) in EPSG:4326. The AOI must fall within a
    single zoom-`z` quadkey tile (true for our mine windows). `out_res_m` decimates the
    read to ~that ground resolution (still much finer than 10 m S2) to bound memory on
    larger windows; None reads native (~1.2 m).
    """
    w, s, e, n = bbox_lonlat
    qk = lonlat_to_quadkey((w + e) / 2, (s + n) / 2, z)
    url = f"/vsicurl/{BASE}{qk}.tif"
    with rasterio.Env(**_ENV):
        with rasterio.open(url) as src:
            b = transform_bounds("EPSG:4326", src.crs, w, s, e, n)
            win = window_from_bounds(*b, transform=src.transform)
            transform = src.window_transform(win)
            if out_res_m and out_res_m > abs(src.res[0]):
                scale = out_res_m / abs(src.res[0])
                out_h = max(1, int(round(win.height / scale)))
                out_w = max(1, int(round(win.width / scale)))
                arr = src.read(1, window=win, out_shape=(out_h, out_w),
                               resampling=Resampling.average).astype("float32")
                transform = transform * Affine.scale(win.width / out_w, win.height / out_h)
            else:
                arr = src.read(1, window=win).astype("float32")
            nod = src.nodata
    if nod is not None:
        arr[arr == nod] = np.nan
    arr[(arr < 0) | (arr > max_height_m)] = np.nan
    return arr, transform, "EPSG:3857"
