"""Raster alignment + CHM (preview only).

`compute_chm` mirrors the Day-0 scripts exactly: mask nodata in DSM and DEM,
then `dsm - dem`, NaN elsewhere. No thresholding to labels, no building removal,
no morphology — those are M2.
"""

from __future__ import annotations

import numpy as np
from rasterio.warp import Resampling, calculate_default_transform, reproject

from ..utils.crs import to_epsg


def compute_chm(dsm, dem, dsm_nodata=None, dem_nodata=None) -> np.ndarray:
    dsm = np.asarray(dsm, dtype=np.float32)
    dem = np.asarray(dem, dtype=np.float32)
    if dsm.shape != dem.shape:
        raise ValueError(f"DSM/DEM shape mismatch: {dsm.shape} vs {dem.shape}")
    mask = np.ones(dsm.shape, dtype=bool)
    if dsm_nodata is not None:
        mask &= dsm != dsm_nodata
    if dem_nodata is not None:
        mask &= dem != dem_nodata
    return np.where(mask, dsm - dem, np.nan).astype(np.float32)


def check_alignment(dsm, dem, *, tol: float = 1e-3) -> list[str]:
    """Return a list of human-readable warnings about DSM/DEM grid mismatch."""
    warnings: list[str] = []
    if to_epsg(dsm.crs) != to_epsg(dem.crs):
        warnings.append(f"CRS mismatch: DSM {dsm.crs} vs DEM {dem.crs}")
    if tuple(dsm.shape) != tuple(dem.shape):
        warnings.append(f"Shape mismatch: DSM {dsm.shape} vs DEM {dem.shape}")
    if any(abs(a - b) > tol for a, b in zip(dsm.res, dem.res)):
        warnings.append(f"Resolution mismatch: DSM {dsm.res} vs DEM {dem.res}")
    if any(abs(a - b) > tol for a, b in zip(dsm.bounds, dem.bounds)):
        warnings.append(f"Bounds mismatch: DSM {dsm.bounds} vs DEM {dem.bounds}")
    return warnings


def resample_to(array, src_transform, src_crs, dst_res, *, resampling="bilinear", nodata=None):
    """Resample a single-band array to `dst_res` metres within the same CRS.

    Returns (dst_array, dst_transform).
    """
    array = np.asarray(array, dtype=np.float32)
    h, w = array.shape
    left = src_transform.c
    top = src_transform.f
    right = left + w * src_transform.a
    bottom = top + h * src_transform.e
    dst_transform, dst_w, dst_h = calculate_default_transform(
        src_crs, src_crs, w, h, left, bottom, right, top, resolution=dst_res
    )
    dst = np.empty((dst_h, dst_w), dtype=np.float32)
    reproject(
        source=array,
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=src_crs,
        src_nodata=nodata,
        dst_nodata=nodata,
        resampling=Resampling[resampling],
    )
    return dst, dst_transform
