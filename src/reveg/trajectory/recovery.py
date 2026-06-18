"""Restoration trajectory: vegetation-cover / greenness curves vs a standardised
baseline, from a Sentinel-2 period-composite cube (output of reveg.data.s2_fetch).

Two curves per AOI:
  - mean NDVI per period (greenness proxy)
  - vegetation cover fraction per period (% pixels with NDVI >= veg threshold)

Then a STANDARDISED recovery fraction so AOIs/sites are comparable:

    recovery_t = (NDVI_t - NDVI_bare) / (NDVI_ref - NDVI_bare)

where NDVI_bare = post-disturbance / year-0 baseline (recovery == 0) and
NDVI_ref = an intact reference or rehab target (recovery == 1). Honest caveat:
10 m pixels mix cover types and S2 "structure" is a greenness proxy, not height.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from ..labels.classes import LabelSpec


def ndvi_cube(ds: xr.Dataset, *, nir: str = "B08", red: str = "B04") -> xr.DataArray:
    """Per-pixel NDVI for every time step; keeps laziness if ds is dask-backed."""
    a = ds[nir].astype("float32")
    b = ds[red].astype("float32")
    den = a + b
    return ((a - b) / den.where(den != 0)).rename("ndvi")


def rehab_mask_from_baseline(
    ds: xr.Dataset,
    *,
    year_index: int = 0,
    ndvi_max: float = 0.40,
    nir: str = "B08",
    red: str = "B04",
) -> xr.DataArray:
    """Boolean (y, x) mask of pixels cleared/bare in the BASELINE period.

    Selection uses only the start state (NDVI below ndvi_max at year_index), never
    the later outcome — so a subsequent NDVI rise over this mask is a genuine
    recovery signal, not selection-on-outcome. Roads/active pits that stay bare
    remain in the mask and honestly damp the mean."""
    nd0 = ndvi_cube(ds, nir=nir, red=red).isel(time=year_index)
    return (nd0 < ndvi_max) & nd0.notnull()


def cover_series(
    ds: xr.Dataset,
    *,
    nir: str = "B08",
    red: str = "B04",
    spec: LabelSpec = LabelSpec(),
    mask: xr.DataArray | None = None,
) -> pd.DataFrame:
    """Per-period summary: mean NDVI and vegetation cover fraction.

    If `mask` (a y,x boolean array) is given, the aggregation is restricted to those
    pixels (e.g. the baseline rehab footprint). Returns a DataFrame indexed by time
    with columns mean_ndvi, veg_cover_frac.
    """
    nd = ndvi_cube(ds, nir=nir, red=red)
    if mask is not None:
        nd = nd.where(mask)
    mean_ndvi = nd.mean(dim=("y", "x"), skipna=True)
    is_veg = nd >= spec.veg_ndvi
    valid = nd.notnull()
    veg_frac = is_veg.sum(dim=("y", "x")) / valid.sum(dim=("y", "x")).where(
        valid.sum(dim=("y", "x")) != 0
    )
    df = pd.DataFrame(
        {
            "mean_ndvi": mean_ndvi.compute().to_series(),
            "veg_cover_frac": veg_frac.compute().to_series(),
        }
    )
    df.index.name = "time"
    return df


def add_recovery_fraction(
    df: pd.DataFrame,
    *,
    ndvi_bare: float,
    ndvi_ref: float,
    column: str = "mean_ndvi",
    clip: bool = True,
) -> pd.DataFrame:
    """Append standardised recovery fraction vs (bare, reference) anchors."""
    if ndvi_ref == ndvi_bare:
        raise ValueError("ndvi_ref must differ from ndvi_bare")
    rec = (df[column] - ndvi_bare) / (ndvi_ref - ndvi_bare)
    if clip:
        rec = rec.clip(lower=0.0, upper=1.0)
    out = df.copy()
    out["recovery_fraction"] = rec
    return out


def anchors_from_series(df: pd.DataFrame, *, column: str = "mean_ndvi") -> tuple[float, float]:
    """Heuristic anchors when none are supplied: bare = min, reference = 95th pct.
    Use real pre-disturbance / reference-site values when available — this is a fallback."""
    s = df[column].dropna()
    if s.empty:
        raise ValueError("empty series")
    return float(s.min()), float(s.quantile(0.95))


def fit_recovery_time(
    df: pd.DataFrame,
    *,
    target: float = 0.8,
    column: str = "recovery_fraction",
) -> dict:
    """Fit a saturating (logistic) recovery curve and estimate time-to-target.

    Returns {k, t0, L, years_to_target}. Falls back to NaN params if the fit fails
    (too few points / no convergence). Time axis = decimal years from series start.
    """
    s = df[column].dropna()
    if len(s) < 4:
        return {"k": np.nan, "t0": np.nan, "L": np.nan, "years_to_target": np.nan}
    t = (s.index.year - s.index.year.min()).astype("float64").to_numpy()
    y = s.to_numpy(dtype="float64")

    def logistic(x, L, k, t0):
        return L / (1.0 + np.exp(-k * (x - t0)))

    try:
        from scipy.optimize import curve_fit

        p0 = [max(y.max(), 1e-3), 0.5, float(np.median(t))]
        popt, _ = curve_fit(logistic, t, y, p0=p0, maxfev=10000)
        L, k, t0 = (float(v) for v in popt)
        # invert logistic for target (within achievable range)
        yt = None
        if 0 < target < L:
            yt = t0 - np.log(L / target - 1.0) / k
        return {"k": k, "t0": t0, "L": L, "years_to_target": (float(yt) if yt is not None else np.nan)}
    except Exception:
        return {"k": np.nan, "t0": np.nan, "L": np.nan, "years_to_target": np.nan}
