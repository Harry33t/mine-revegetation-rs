import numpy as np

from reveg.labels import classes as C
from reveg.labels import indices, weak_labels as wl


def _scene():
    # tall tree | grass | bare | water
    nir = np.array([[0.40, 0.38, 0.20, 0.05]], np.float32)
    red = np.array([[0.05, 0.06, 0.18, 0.06]], np.float32)
    green = np.array([[0.06, 0.07, 0.16, 0.10]], np.float32)
    chm = np.array([[8.0, 0.2, 0.0, 0.0]], np.float32)
    return chm, indices.ndvi(nir, red), indices.ndwi(green, nir)


def test_height_spectral_classes():
    chm, nd, ndw = _scene()
    lab = wl.multiclass_height_spectral(chm, nd, ndw)
    assert [C.CLASS_NAMES[v] for v in lab.ravel()] == ["canopy", "low_veg", "bare", "water"]


def test_spectral_only_collapses_structure():
    chm, nd, ndw = _scene()
    lab = wl.multiclass_spectral_only(nd, ndw)
    assert [C.CLASS_NAMES[v] for v in lab.ravel()] == ["vegetation", "vegetation", "bare", "water"]


def test_senescent_band():
    nd = np.array([[0.30]], np.float32)
    ndw = np.array([[0.0]], np.float32)
    assert wl.multiclass_spectral_only(nd, ndw).ravel()[0] == C.SENESCENT


def test_nodata_preserved():
    nd = np.array([[np.nan]], np.float32)
    ndw = np.array([[np.nan]], np.float32)
    assert wl.multiclass_spectral_only(nd, ndw).ravel()[0] == C.NODATA


def test_class_fractions_sum_to_one():
    chm, nd, ndw = _scene()
    lab = wl.multiclass_height_spectral(chm, nd, ndw)
    fr = wl.class_fractions(lab)
    assert abs(sum(fr.values()) - 1.0) < 1e-6
