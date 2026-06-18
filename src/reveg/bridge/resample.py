"""Scale-bridge core: aggregate fine-scale (1 m) categorical labels onto the
Sentinel-2 10 m grid by AREA FRACTION.

A categorical label must not be bilinearly resampled. Instead, for each class we
reproject its binary plane with area-weighted averaging (Resampling.average) onto
the exact destination grid; that yields, per 10 m pixel, the fraction of its area
covered by each fine class. The majority class is the bridge label; the winning
fraction (relative to total valid coverage) is a per-pixel PURITY = confidence the
S2 model can be supervised/validated against.

This is the only validatable form of a drone/aerial -> satellite "scale bridge":
fine and satellite must observe the SAME ground (here: a NZ AOI with LINZ aerial +
1 m LiDAR AND Sentinel-2). Across disjoint sites it would be unvalidatable domain
transfer — see the project memo.
"""

from __future__ import annotations

import numpy as np
from rasterio.warp import Resampling, reproject

from ..labels import classes as C


def grid_from_dataset(ds):
    """(transform, crs, (height, width)) for an odc-stac / rioxarray Sentinel-2 cube."""
    try:  # odc-stac path
        gb = ds.odc.geobox
        return gb.transform, gb.crs, (gb.height, gb.width)
    except Exception:
        crs = ds.rio.crs
        tr = ds.rio.transform()
        return tr, crs, (int(ds.sizes["y"]), int(ds.sizes["x"]))


def class_fraction_stack(
    label: np.ndarray,
    src_transform,
    src_crs,
    dst_transform,
    dst_crs,
    dst_shape: tuple[int, int],
    *,
    class_ids,
) -> dict[int, np.ndarray]:
    """Per-class area fraction on the destination grid. Returns {class_id: frac (dst_shape)}."""
    label = np.asarray(label)
    fracs: dict[int, np.ndarray] = {}
    for cid in class_ids:
        src_plane = (label == cid).astype("float32")
        dst = np.zeros(dst_shape, dtype="float32")
        reproject(
            source=src_plane,
            destination=dst,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.average,
        )
        fracs[cid] = dst
    return fracs


def aggregate_binary_fraction(
    mask: np.ndarray,
    src_transform,
    src_crs,
    dst_transform,
    dst_crs,
    dst_shape: tuple[int, int],
) -> np.ndarray:
    """Area fraction of a fine binary mask (e.g. canopy = CHM>=h) per coarse cell.

    Reprojects the 0/1 mask with area-weighted averaging onto the destination grid:
    each 10 m cell gets the fraction of its area that was True at fine scale — i.e.
    sub-pixel canopy cover the satellite cannot resolve. Returns float32 in [0, 1].
    """
    src_plane = np.asarray(mask, dtype="float32")
    dst = np.full(dst_shape, np.nan, dtype="float32")
    reproject(
        source=src_plane,
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.average,
    )
    return dst


def aggregate_labels_to_grid(
    label: np.ndarray,
    src_transform,
    src_crs,
    dst_transform,
    dst_crs,
    dst_shape: tuple[int, int],
    *,
    class_ids=None,
    min_coverage: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate a fine categorical label onto the destination grid.

    Returns (coarse_label uint8, purity float32). A destination pixel whose total
    valid (non-nodata) coverage is below `min_coverage` is set to NODATA.
    """
    if class_ids is None:
        class_ids = [c for c in C.CLASS_NAMES if c != C.NODATA]
    fracs = class_fraction_stack(
        label, src_transform, src_crs, dst_transform, dst_crs, dst_shape, class_ids=class_ids
    )
    ids = list(class_ids)
    stack = np.stack([fracs[c] for c in ids], axis=0)  # (n_classes, H, W)
    total = stack.sum(axis=0)  # valid coverage per pixel
    arg = stack.argmax(axis=0)
    coarse = np.array(ids, dtype="uint8")[arg]
    with np.errstate(invalid="ignore", divide="ignore"):
        purity = np.where(total > 0, stack.max(axis=0) / total, 0.0).astype("float32")
    coarse = np.where(total >= min_coverage, coarse, C.NODATA).astype("uint8")
    purity = np.where(total >= min_coverage, purity, 0.0).astype("float32")
    return coarse, purity
