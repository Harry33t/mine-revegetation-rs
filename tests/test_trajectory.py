import numpy as np
import pandas as pd
import xarray as xr

from reveg.trajectory import recovery as R


def _rising_cube():
    years = pd.date_range("2019-01-01", periods=6, freq="YS")
    target = np.array([0.10, 0.25, 0.40, 0.55, 0.66, 0.72])
    red = 0.10
    nir = red * (1 + target) / (1 - target)
    B04 = np.broadcast_to(np.float32(red), (6, 5, 5)).astype("float32")
    B08 = (np.ones((6, 5, 5), "float32") * nir[:, None, None]).astype("float32")
    return xr.Dataset(
        {"B04": (("time", "y", "x"), B04), "B08": (("time", "y", "x"), B08)},
        coords={"time": years, "y": np.arange(5), "x": np.arange(5)},
    )


def test_cover_series_monotonic():
    df = R.cover_series(_rising_cube())
    assert df["mean_ndvi"].is_monotonic_increasing
    assert df["veg_cover_frac"].iloc[-1] == 1.0


def test_recovery_fraction_spans_unit_interval():
    df = R.cover_series(_rising_cube())
    bare, ref = R.anchors_from_series(df)
    df = R.add_recovery_fraction(df, ndvi_bare=bare, ndvi_ref=ref)
    assert abs(df["recovery_fraction"].iloc[0]) < 1e-6
    assert abs(df["recovery_fraction"].iloc[-1] - 1.0) < 1e-6


def test_logistic_fit_returns_positive_time():
    df = R.cover_series(_rising_cube())
    bare, ref = R.anchors_from_series(df)
    df = R.add_recovery_fraction(df, ndvi_bare=bare, ndvi_ref=ref)
    fit = R.fit_recovery_time(df, target=0.8)
    assert fit["years_to_target"] > 0
