"""Morphological cleaning + small-object removal (memo §2 RQ1 'Cleaning')."""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def clean_mask(mask: np.ndarray, min_area_px: int, opening_iters: int = 1) -> np.ndarray:
    """Despeckle via binary opening, then drop connected components smaller than
    `min_area_px`. At 1 m CHM resolution, min_area_px == min_canopy_area_m2."""
    m = np.asarray(mask, dtype=bool)
    if opening_iters and opening_iters > 0:
        m = ndimage.binary_opening(m, iterations=opening_iters)
    if min_area_px and min_area_px > 1:
        labels, n = ndimage.label(m)
        if n > 0:
            sizes = ndimage.sum_labels(np.ones_like(labels), labels, index=range(1, n + 1))
            too_small = {i + 1 for i, s in enumerate(sizes) if s < min_area_px}
            if too_small:
                m = m & ~np.isin(labels, list(too_small))
    return m
