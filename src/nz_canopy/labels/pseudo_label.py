"""Pseudo-label orchestrator: CHM -> threshold -> remove buildings -> clean.

The final mask is a WEAK LABEL (pseudo-label), not ground truth; it must be
audited against manual labels in M3.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio

from .. import config
from ..data_sources.buildings import load_buildings
from .building_mask import rasterize_buildings
from .chm import CHM, chm_for_city
from .morphology import clean_mask
from .threshold import threshold_mask


@dataclass
class PseudoLabelResult:
    chm: CHM
    threshold_m: float
    valid: np.ndarray
    raw_mask: np.ndarray  # chm >= threshold
    building_mask: np.ndarray  # buffered building footprints
    building_removed_mask: np.ndarray  # raw & ~building
    final_mask: np.ndarray  # after morphology cleaning
    green_mask: np.ndarray | None = None  # aerial greenness gate (None if no aerial supplied)


def generate(
    city: str,
    threshold_m: float = 2.0,
    *,
    proc=None,
    chm: CHM | None = None,
    buildings=None,
    aerial=None,
    exg_threshold: float | None = None,
) -> PseudoLabelResult:
    proc = proc or config.load_processing()
    chm = chm if chm is not None else chm_for_city(city)
    if buildings is None:
        try:
            buildings = load_buildings(city)
        except FileNotFoundError:
            buildings = None

    raw, valid = threshold_mask(chm.array, threshold_m)
    bld = rasterize_buildings(buildings, chm.transform, chm.array.shape, proc.chm["building_buffer_m"])
    removed = raw & ~bld

    # Aerial greenness gate (default ON): drop tall, non-green regions = roofs the
    # footprint mask missed. aerial=None auto-loads it; pass aerial=False to disable.
    green = None
    exg_chm = None
    if aerial is not False:
        if aerial is None:
            try:
                from ..data_sources.imagery import load_aerial
                aerial = load_aerial(city)
            except Exception:
                aerial = None
        if aerial is not None:
            from .greenness import greenness_on_chm
            thr = exg_threshold if exg_threshold is not None else proc.chm.get("exg_threshold", 0.0)
            exg_chm, green = greenness_on_chm(aerial.image, aerial.extent, chm, exg_threshold=thr)
            removed = removed & green

    # 1 px == 1 m2 at the 1 m CHM grid; map min_canopy_area_m2 -> pixel count.
    px_area = abs(chm.res[0] * chm.res[1]) or 1.0
    min_area_px = int(round(proc.chm["min_canopy_area_m2"] / px_area))
    final = clean_mask(removed, min_area_px)

    # Commercial-roof suppression: the per-pixel ExG gate still leaks on large
    # flat/dark/metal roofs (faintly green, no footprint). Drop whole low-median-ExG
    # blobs. Verified to clear ~54% of label on the 10 confirmed roof-leak tiles
    # while removing only ~4% on clean tree tiles. Default ON; disabled if no aerial.
    if exg_chm is not None:
        from .greenness import suppress_low_green_components
        roof_thr = proc.chm.get("roof_exg_threshold", 0.04)
        roof_min_px = int(round(proc.chm.get("roof_min_area_m2", 40) / px_area))
        final = suppress_low_green_components(
            final, exg_chm, exg_comp_threshold=roof_thr, min_area_px=roof_min_px
        )

    return PseudoLabelResult(chm, threshold_m, valid, raw, bld, removed, final, green)


def save_mask_tif(mask: np.ndarray, chm: CHM, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = mask.shape
    with rasterio.open(
        path, "w", driver="GTiff", height=h, width=w, count=1, dtype="uint8",
        crs=chm.crs, transform=chm.transform, nodata=0, compress="deflate",
    ) as dst:
        dst.write(mask.astype("uint8"), 1)
    return path
