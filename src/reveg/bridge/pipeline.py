"""End-to-end scale-bridge orchestration over a co-located AOI.

Given a fine-scale (1 m) multi-class label with its geotransform/CRS, and a
Sentinel-2 composite cube over the SAME ground, produce:
  - the fine label aggregated to the S2 10 m grid (+ per-pixel purity),
  - the S2 spectral-only label on that grid,
  - an agreement report between them (the validation of the bridge).

The fine label is produced upstream by reveg.labels.weak_labels.multiclass_height_
spectral from a real CHM (nz_canopy.labels.chm) + aerial indices; this function
takes it as input so it stays testable without LINZ downloads.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from ..labels import weak_labels as wl
from ..labels.classes import LabelSpec
from ..labels.indices import ndvi as _ndvi, ndwi as _ndwi
from . import resample as RS
from . import validate as V


def _s2_spectral_label(ds: xr.Dataset, spec: LabelSpec, nir: str, red: str, green: str) -> np.ndarray:
    """Collapse an S2 cube to one composite, then spectral-only multiclass label."""
    composite = ds.median(dim="time", skipna=True) if "time" in ds.dims else ds
    nir_a = np.asarray(composite[nir], dtype="float32")
    red_a = np.asarray(composite[red], dtype="float32")
    green_a = np.asarray(composite[green], dtype="float32")
    nd = _ndvi(nir_a, red_a)
    ndw = _ndwi(green_a, nir_a)
    return wl.multiclass_spectral_only(nd, ndw, spec=spec)


def bridge_report(
    fine_label: np.ndarray,
    src_transform,
    src_crs,
    s2_ds: xr.Dataset,
    *,
    spec: LabelSpec = LabelSpec(),
    min_coverage: float = 0.5,
    nir: str = "B08",
    red: str = "B04",
    green: str = "B03",
) -> dict:
    """Aggregate fine label to the S2 grid and validate against the S2 label."""
    dst_transform, dst_crs, dst_shape = RS.grid_from_dataset(s2_ds)
    coarse, purity = RS.aggregate_labels_to_grid(
        fine_label, src_transform, src_crs, dst_transform, dst_crs, dst_shape,
        min_coverage=min_coverage,
    )
    s2_label = _s2_spectral_label(s2_ds, spec, nir, red, green)
    report = V.agreement(coarse, s2_label)
    return {
        "coarse_label": coarse,
        "purity": purity,
        "s2_label": s2_label,
        "agreement": report,
    }
