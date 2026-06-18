"""Spectral indices for weak labels and trajectories.

Pure-numpy, band-array in / index-array out. Inputs may be reflectance (0..1 or
0..10000) — indices are ratios so scale-invariant. NaN-safe: divide-by-zero -> NaN.
"""

from __future__ import annotations

import numpy as np


def _ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    num = a - b
    den = a + b
    out = np.full(num.shape, np.nan, np.float32)
    np.divide(num, den, out=out, where=den != 0)
    return out


def ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """(NIR - Red) / (NIR + Red). Vegetation greenness/cover proxy."""
    return _ratio(nir, red)


def ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """McFeeters NDWI = (Green - NIR) / (Green + NIR). Open-water proxy."""
    return _ratio(green, nir)


def ndre(nir: np.ndarray, red_edge: np.ndarray) -> np.ndarray:
    """(NIR - RedEdge) / (NIR + RedEdge). Chlorophyll / stress sensitivity."""
    return _ratio(nir, red_edge)


def exg(red: np.ndarray, green: np.ndarray, blue: np.ndarray) -> np.ndarray:
    """Chromatic Excess-Green 2g - r - b on normalized RGB (RGB-only greenness)."""
    r = np.asarray(red, np.float32)
    g = np.asarray(green, np.float32)
    b = np.asarray(blue, np.float32)
    s = r + g + b
    s[s == 0] = 1.0
    return (2.0 * g / s - r / s - b / s).astype(np.float32)
