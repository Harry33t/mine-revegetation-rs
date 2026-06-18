"""CRS helpers centred on EPSG:2193 (NZTM2000), the locked target CRS."""

from __future__ import annotations

from typing import Protocol

from rasterio.crs import CRS
from rasterio.warp import transform_bounds

TARGET_CRS = "EPSG:2193"
TARGET_EPSG = 2193


def to_epsg(crs) -> int | None:
    if crs is None:
        return None
    return CRS.from_user_input(crs).to_epsg()


def assert_target_crs(crs, *, where: str = "") -> None:
    """Raise if `crs` is missing or not EPSG:2193."""
    suffix = f" for {where}" if where else ""
    if crs is None:
        raise ValueError(f"Missing CRS{suffix}")
    epsg = to_epsg(crs)
    if epsg != TARGET_EPSG:
        raise ValueError(f"Expected {TARGET_CRS}{suffix}, got {CRS.from_user_input(crs).to_string()}")


def reproject_bounds(bounds, src_crs, dst_crs):
    """Reproject (left, bottom, right, top) from src_crs to dst_crs."""
    left, bottom, right, top = bounds
    return transform_bounds(src_crs, dst_crs, left, bottom, right, top)


class _HasGrid(Protocol):
    bounds: tuple
    shape: tuple
    res: tuple
    crs: object


def same_grid(a: _HasGrid, b: _HasGrid, *, tol: float = 1e-3) -> bool:
    """True when two raster layers share CRS, pixel grid shape, resolution, and bounds."""
    if to_epsg(a.crs) != to_epsg(b.crs):
        return False
    if tuple(a.shape) != tuple(b.shape):
        return False
    if any(abs(x - y) > tol for x, y in zip(a.res, b.res)):
        return False
    return all(abs(x - y) <= tol for x, y in zip(a.bounds, b.bounds))
