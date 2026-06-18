"""Fetch co-located LINZ aerial + 1 m DSM/DEM tiles from the open AWS buckets.

NZ LINZ open data (CC-BY) lives in public, no-auth S3 buckets with STAC catalogs:
  nz-imagery  — RGB / RGBNIR ortho (here: Bay of Plenty 2018-2019, 0.1 m, RGB)
  nz-elevation — 1 m DSM and DEM (here: Bay of Plenty 2018-2019)
Both EPSG:2193. A real CHM = DSM - DEM, the high-quality "teacher" for the scale
bridge. Read via GDAL /vsicurl/ (windowed/decimated COG reads — only needed bytes).
"""

from __future__ import annotations

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_bounds as transform_from_bounds
from rasterio.warp import transform_bounds
from rasterio.windows import bounds as window_bounds
from rasterio.windows import from_bounds as window_from_bounds

IMG_BUCKET = "https://nz-imagery.s3.ap-southeast-2.amazonaws.com/"
ELV_BUCKET = "https://nz-elevation.s3.ap-southeast-2.amazonaws.com/"

# A fully-specified co-located triple (Bay of Plenty 2018-2019, same vintage).
DEFAULT_RGB = IMG_BUCKET + "bay-of-plenty/bay-of-plenty_2018-2019_0.1m/rgb/2193/BC36_1000_2314.tiff"
DEFAULT_DSM = ELV_BUCKET + "bay-of-plenty/bay-of-plenty_2018-2019/dsm_1m/2193/BC36_10000_0302.tiff"
DEFAULT_DEM = ELV_BUCKET + "bay-of-plenty/bay-of-plenty_2018-2019/dem_1m/2193/BC36_10000_0302.tiff"

_ENV = dict(GDAL_HTTP_MAX_RETRY="10", GDAL_HTTP_RETRY_DELAY="1", GDAL_HTTP_TIMEOUT="60")


def _vsicurl(url: str) -> str:
    return url if url.startswith("/vsicurl/") else "/vsicurl/" + url


def read_full_decimated(url: str, *, out_res: float = 1.0, indexes=None, resampling="average"):
    """Read a whole COG decimated to ~out_res metres. Returns (arr, transform, crs, bounds)."""
    with rasterio.Env(**_ENV):
        with rasterio.open(_vsicurl(url)) as src:
            left, bottom, right, top = src.bounds
            w = max(1, int(round((right - left) / out_res)))
            h = max(1, int(round((top - bottom) / out_res)))
            count = len(indexes) if indexes else src.count
            arr = src.read(
                indexes=indexes,
                out_shape=(count, h, w),
                resampling=Resampling[resampling],
            )
            transform = transform_from_bounds(left, bottom, right, top, w, h)
            return arr, transform, src.crs, src.bounds


def read_window(url: str, bounds_in_crs, *, indexes=None, resampling="bilinear"):
    """Read a COG windowed to `bounds_in_crs` (in the COG's own CRS) at native res.
    Returns (arr, transform, crs)."""
    with rasterio.Env(**_ENV):
        with rasterio.open(_vsicurl(url)) as src:
            win = window_from_bounds(*bounds_in_crs, transform=src.transform)
            arr = src.read(indexes=indexes, window=win)
            transform = src.window_transform(win)
            return arr, transform, src.crs


def bounds_to_lonlat(bounds, src_crs) -> tuple[float, float, float, float]:
    """Reproject (left,bottom,right,top) to EPSG:4326 (for the Sentinel-2 fetch)."""
    return tuple(transform_bounds(src_crs, "EPSG:4326", *bounds))
