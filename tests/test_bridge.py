import numpy as np
from affine import Affine

from reveg.bridge import resample as RS, validate as V
from reveg.labels import classes as C

CRS = "EPSG:32760"  # a metric UTM zone (NZ)
X0, Y0 = 500000.0, 6000000.0
SRC_T = Affine(1, 0, X0, 0, -1, Y0)   # 1 m
DST_T = Affine(10, 0, X0, 0, -10, Y0)  # 10 m


def _fine():
    fine = np.full((30, 30), C.BARE, np.uint8)  # 3x3 grid of 10 m cells
    fine[0:10, 0:10] = C.CANOPY
    fine[0:10, 10:20] = C.WATER
    fine[10:20, 10:20] = C.BARE
    fine[10:16, 10:20] = C.LOW_VEG  # 60% of the cell
    return fine


def test_aggregate_majority_and_purity():
    coarse, purity = RS.aggregate_labels_to_grid(_fine(), SRC_T, CRS, DST_T, CRS, (3, 3))
    assert C.CLASS_NAMES[coarse[0, 0]] == "canopy"
    assert C.CLASS_NAMES[coarse[0, 1]] == "water"
    assert C.CLASS_NAMES[coarse[1, 1]] == "low_veg"
    assert abs(purity[0, 0] - 1.0) < 1e-3
    assert abs(purity[1, 1] - 0.6) < 0.05


def test_low_coverage_becomes_nodata():
    # a fine label that is mostly nodata -> coarse pixel below min_coverage drops out
    fine = np.full((30, 30), C.NODATA, np.uint8)
    fine[0:2, 0:2] = C.CANOPY  # only 4% of cell (0,0)
    coarse, _ = RS.aggregate_labels_to_grid(fine, SRC_T, CRS, DST_T, CRS, (3, 3), min_coverage=0.5)
    assert coarse[0, 0] == C.NODATA


def test_agreement_perfect_when_matched():
    coarse, _ = RS.aggregate_labels_to_grid(_fine(), SRC_T, CRS, DST_T, CRS, (3, 3))
    s2 = np.full((3, 3), C.BARE, np.uint8)
    s2[0, 0] = C.VEGETATION
    s2[0, 1] = C.WATER
    s2[1, 1] = C.VEGETATION
    rep = V.agreement(coarse, s2)
    assert abs(rep["overall_agreement"] - 1.0) < 1e-9
    assert abs(rep["mean_iou"] - 1.0) < 1e-9
    assert rep["n_pixels"] == 9


def test_collapse_maps_structure_to_vegetation():
    lab = np.array([[C.CANOPY, C.SHRUB, C.LOW_VEG, C.BARE]], np.uint8)
    out = V.collapse_to_spectral_scheme(lab)
    assert list(out.ravel()) == [C.VEGETATION, C.VEGETATION, C.VEGETATION, C.BARE]


def test_bridge_report_orchestrator():
    import pandas as pd
    import xarray as xr
    import rioxarray  # noqa: F401  (registers .rio accessor)

    from reveg.bridge import pipeline as P

    fine = np.full((30, 30), C.BARE, np.uint8)
    fine[0:10, 0:10] = C.CANOPY
    fine[0:10, 10:20] = C.WATER

    xs = np.array([X0 + 5, X0 + 15, X0 + 25])
    ys = np.array([Y0 - 5, Y0 - 15, Y0 - 25])

    def band(vals):
        return ("time", "y", "x"), np.array(vals, "float32")[None]

    nir = [[0.40, 0.05, 0.20], [0.20, 0.20, 0.20], [0.20, 0.20, 0.20]]
    red = [[0.05, 0.06, 0.18], [0.18, 0.18, 0.18], [0.18, 0.18, 0.18]]
    grn = [[0.06, 0.10, 0.16], [0.16, 0.16, 0.16], [0.16, 0.16, 0.16]]
    ds = xr.Dataset(
        {"B08": band(nir), "B04": band(red), "B03": band(grn)},
        coords={"time": pd.to_datetime(["2023-01-01"]), "y": ys, "x": xs},
    ).rio.write_crs(CRS)

    out = P.bridge_report(fine, SRC_T, CRS, ds)
    assert C.CLASS_NAMES[out["coarse_label"][0, 0]] == "canopy"
    assert C.CLASS_NAMES[out["s2_label"][0, 0]] == "vegetation"
    assert abs(out["agreement"]["overall_agreement"] - 1.0) < 1e-9
