"""CHM thresholding into a candidate canopy mask."""

from __future__ import annotations

import numpy as np


def threshold_mask(chm: np.ndarray, threshold_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (mask, valid).

    valid = finite CHM pixels (the denominator for canopy %).
    mask  = canopy candidate = valid & (chm >= threshold_m).
    NaN pixels are invalid: never canopy, never counted.
    """
    chm = np.asarray(chm, dtype=np.float32)
    valid = np.isfinite(chm)
    mask = valid & (chm >= threshold_m)
    return mask, valid
