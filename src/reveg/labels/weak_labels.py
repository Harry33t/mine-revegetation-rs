"""Multi-class structural/functional weak labels.

Two labellers share one threshold spec ([[classes]] LabelSpec):

  multiclass_height_spectral  — fine-scale aerial + CHM (NZ bridge track). Uses
      height tiers to resolve low-veg / shrub / canopy. This is the high-quality
      "teacher" whose labels get resampled to 10 m for the scale bridge.

  multiclass_spectral_only    — Sentinel-2 (10 m, no CHM). Honest collapse: a
      single VEGETATION class instead of structural tiers.

Both return uint8 arrays with ids from classes.py (0 = nodata). Assignment is by
priority, not morphology; per-class cleaning is left to the caller (see
nz_canopy.morphology.clean_mask for the binary recipe).
"""

from __future__ import annotations

import numpy as np

from . import classes as C
from .classes import LabelSpec


def _gates(ndvi, ndwi, spec):
    ndvi = np.asarray(ndvi, np.float32)
    valid = np.isfinite(ndvi)
    vigorous = valid & (ndvi >= spec.veg_ndvi)
    senescent = valid & (ndvi >= spec.senescent_ndvi) & (ndvi < spec.veg_ndvi)
    water = valid & ~vigorous & np.isfinite(np.asarray(ndwi, np.float32)) & (
        np.asarray(ndwi, np.float32) >= spec.water_ndwi
    )
    return valid, vigorous, senescent, water


def multiclass_height_spectral(
    chm: np.ndarray,
    ndvi: np.ndarray,
    ndwi: np.ndarray,
    *,
    spec: LabelSpec = LabelSpec(),
) -> np.ndarray:
    """Fine-scale label: structure from CHM height, function from indices."""
    chm = np.asarray(chm, np.float32)
    valid, vigorous, senescent, water = _gates(ndvi, ndwi, spec)

    out = np.full(ndvi.shape, C.NODATA, np.uint8)
    # bare first (lowest priority among valid), then overwrite upward.
    out[valid] = C.BARE
    out[senescent] = C.SENESCENT
    # vigorous vegetation split by height; missing height -> default low veg.
    tall = vigorous & np.isfinite(chm) & (chm >= spec.canopy_min_m)
    mid = vigorous & np.isfinite(chm) & (chm >= spec.shrub_min_m) & (chm < spec.canopy_min_m)
    low = vigorous & ~tall & ~mid
    out[low] = C.LOW_VEG
    out[mid] = C.SHRUB
    out[tall] = C.CANOPY
    # water wins over bare/senescent but not over clearly vigorous veg (handled by ~vigorous in gate).
    out[water] = C.WATER
    return out


def multiclass_height_greenness(
    chm: np.ndarray,
    exg: np.ndarray,
    *,
    spec: LabelSpec = LabelSpec(),
    exg_threshold: float = 0.0,
) -> np.ndarray:
    """Fine-scale label for RGB+CHM (no NIR): structure from CHM, greenness from ExG.

    Used for the NZ LINZ bridge track where aerial is RGB-only (matching the
    nz_canopy pipeline's ExG gate). No water/senescent split without NIR.
    """
    chm = np.asarray(chm, np.float32)
    exg = np.asarray(exg, np.float32)
    valid = np.isfinite(exg)
    veg = valid & (exg >= exg_threshold)

    out = np.full(exg.shape, C.NODATA, np.uint8)
    out[valid] = C.BARE
    tall = veg & np.isfinite(chm) & (chm >= spec.canopy_min_m)
    mid = veg & np.isfinite(chm) & (chm >= spec.shrub_min_m) & (chm < spec.canopy_min_m)
    low = veg & ~tall & ~mid
    out[low] = C.LOW_VEG
    out[mid] = C.SHRUB
    out[tall] = C.CANOPY
    return out


def multiclass_spectral_only(
    ndvi: np.ndarray,
    ndwi: np.ndarray,
    *,
    spec: LabelSpec = LabelSpec(),
) -> np.ndarray:
    """Sentinel-2 label: no height, so woody/low collapse into one VEGETATION class."""
    valid, vigorous, senescent, water = _gates(ndvi, ndwi, spec)
    out = np.full(np.asarray(ndvi).shape, C.NODATA, np.uint8)
    out[valid] = C.BARE
    out[senescent] = C.SENESCENT
    out[vigorous] = C.VEGETATION
    out[water] = C.WATER
    return out


def class_fractions(label: np.ndarray) -> dict[str, float]:
    """Fraction of each class among valid (non-nodata) pixels — sanity check."""
    label = np.asarray(label)
    valid = label != C.NODATA
    denom = int(valid.sum()) or 1
    out: dict[str, float] = {}
    for cid, name in C.CLASS_NAMES.items():
        if cid == C.NODATA:
            continue
        out[name] = float((label == cid).sum()) / denom
    return out
