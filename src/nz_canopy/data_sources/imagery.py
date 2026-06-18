"""Aerial imagery reader.

Aerial tiles are merged into one mosaic. The year defaults to the city's
`primary_aerial_year` (matched to the LiDAR survey). `decimate` downsamples a
mosaic for display only — never for analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.merge import merge

from .. import config
from ..utils import paths


@dataclass
class AerialMosaic:
    image: np.ndarray  # (H, W, C) uint8
    extent: tuple  # (left, right, bottom, top) — matplotlib imshow order
    crs: object
    year: str


def _resolve_year(city: str, year: str | None) -> str:
    if year is not None:
        return year
    primary = config.get_city(city).primary_aerial_year
    if primary is None:
        raise ValueError(f"No primary_aerial_year configured for {city}; pass year explicitly")
    return primary


def load_aerial(city: str, year: str | None = None, *, folder: str | Path | None = None) -> AerialMosaic:
    yr = _resolve_year(city, year) if folder is None else (year or "unknown")
    f = Path(folder) if folder is not None else (paths.layer_dir(city, "aerial") / yr)
    files = sorted(Path(f).glob("*.jpg"))
    if not files:
        raise FileNotFoundError(f"No .jpg aerial tiles found in {f}")
    srcs = [rasterio.open(p) for p in files]
    try:
        mosaic, transform = merge(srcs)
        crs = srcs[0].crs
    finally:
        for s in srcs:
            s.close()
    img = mosaic.transpose(1, 2, 0)
    h, w = img.shape[:2]
    left, top = transform.c, transform.f
    right = left + w * transform.a
    bottom = top + h * transform.e
    return AerialMosaic(img, (left, right, bottom, top), crs, yr)


def decimate(image: np.ndarray, max_px: int = 2500) -> np.ndarray:
    """Stride-subsample a (H, W, C) image so its longest side <= max_px (display only)."""
    h, w = image.shape[:2]
    step = max(1, int(max(h, w) / max_px))
    return image[::step, ::step] if step > 1 else image
