"""Validate the scale bridge: compare fine-derived labels aggregated to 10 m
against the Sentinel-2 spectral-only labels on the SAME grid.

Because the fine track resolves structure (low_veg/shrub/canopy) that S2 cannot,
both sides are first collapsed to the common S2 scheme before scoring. Outputs:
overall agreement, per-class IoU, and a confusion matrix.
"""

from __future__ import annotations

import numpy as np

from ..labels import classes as C

# fine structural classes -> common (S2) scheme
_COLLAPSE = {
    C.LOW_VEG: C.VEGETATION,
    C.SHRUB: C.VEGETATION,
    C.CANOPY: C.VEGETATION,
}
# the comparable common scheme
COMMON_IDS = (C.BARE, C.WATER, C.SENESCENT, C.VEGETATION)


def collapse_to_spectral_scheme(label: np.ndarray) -> np.ndarray:
    """Map structural vegetation classes to the single VEGETATION class."""
    out = np.asarray(label).copy()
    for src, dst in _COLLAPSE.items():
        out[out == src] = dst
    return out


def confusion_matrix(a: np.ndarray, b: np.ndarray, *, class_ids=COMMON_IDS) -> np.ndarray:
    """Confusion matrix over pixels valid in BOTH (rows = a, cols = b)."""
    a = np.asarray(a)
    b = np.asarray(b)
    valid = (a != C.NODATA) & (b != C.NODATA)
    ids = list(class_ids)
    idx = {c: i for i, c in enumerate(ids)}
    cm = np.zeros((len(ids), len(ids)), dtype="int64")
    for av, bv in zip(a[valid].ravel(), b[valid].ravel()):
        if av in idx and bv in idx:
            cm[idx[av], idx[bv]] += 1
    return cm


def agreement(a: np.ndarray, b: np.ndarray, *, class_ids=COMMON_IDS) -> dict:
    """Overall agreement + per-class IoU between two label arrays on the same grid."""
    a = collapse_to_spectral_scheme(a)
    b = collapse_to_spectral_scheme(b)
    cm = confusion_matrix(a, b, class_ids=class_ids)
    total = int(cm.sum())
    overall = float(np.trace(cm)) / total if total else float("nan")
    ious: dict[str, float] = {}
    for i, c in enumerate(class_ids):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        denom = tp + fp + fn
        ious[C.CLASS_NAMES[c]] = (float(tp) / denom) if denom else float("nan")
    valid_ious = [v for v in ious.values() if v == v]
    return {
        "overall_agreement": overall,
        "per_class_iou": ious,
        "mean_iou": float(np.mean(valid_ious)) if valid_ious else float("nan"),
        "n_pixels": total,
    }
