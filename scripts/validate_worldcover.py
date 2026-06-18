"""Honest external validation: agreement of our weak labels with ESA WorldCover.

ESA WorldCover (10 m, 2021) is an INDEPENDENT global land-cover product — it does
not use our CHM/index weak-labeling rule, so agreement with it is a real, defensible
check (Cohen's kappa + overall agreement), not self-consistency. Both are crosswalked
to a common 3-class scheme {vegetation, bare, water} for a fair comparison.

    python scripts/validate_worldcover.py --aois alcoa_huntly ranger mt_owen --year 2021
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import click
import numpy as np

from reveg.config import get_aoi
from reveg.data import s2_fetch
from reveg.labels import indices, weak_labels as wl
from reveg.labels.classes import BARE, NODATA, SENESCENT, VEGETATION, WATER

# common scheme
C_VEG, C_BARE, C_WATER = 1, 2, 3

# our weak-label ids -> common
OURS = {VEGETATION: C_VEG, SENESCENT: C_VEG, BARE: C_BARE, WATER: C_WATER}
# ESA WorldCover codes -> common
WC = {10: C_VEG, 20: C_VEG, 30: C_VEG, 40: C_VEG, 90: C_VEG, 95: C_VEG, 100: C_VEG,
      50: C_BARE, 60: C_BARE, 70: C_BARE, 80: C_WATER}


def _require(m):
    import importlib
    return importlib.import_module(m)


def fetch_worldcover(bbox, like_ds):
    """Load ESA WorldCover over bbox, on the SAME grid as like_ds (the S2 cube)."""
    pc = _require("planetary_computer")
    pystac_client = _require("pystac_client")
    odc_stac = _require("odc.stac")
    client = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
    items = list(client.search(collections=["esa-worldcover"], bbox=list(bbox)).items())
    if not items:
        raise RuntimeError("no WorldCover items for bbox")
    gb = like_ds.odc.geobox
    wc = odc_stac.load(items, bands=["map"], geobox=gb, chunks={}).isel(time=0)
    return np.asarray(wc["map"].values)


def kappa(a, b, k=3):
    """Cohen's kappa over labels 1..k (0 = ignored)."""
    cm = np.zeros((k, k), dtype="int64")
    m = (a >= 1) & (b >= 1)
    for av, bv in zip(a[m].ravel(), b[m].ravel()):
        cm[av - 1, bv - 1] += 1
    n = cm.sum()
    if n == 0:
        return float("nan"), float("nan"), int(0)
    po = np.trace(cm) / n
    pe = ((cm.sum(0) / n) * (cm.sum(1) / n)).sum()
    kap = (po - pe) / (1 - pe) if (1 - pe) else float("nan")
    return float(kap), float(po), int(n)


def remap(arr, table):
    out = np.zeros(arr.shape, dtype="uint8")
    for src, dst in table.items():
        out[arr == src] = dst
    return out


@click.command()
@click.option("--aois", multiple=True, default=("alcoa_huntly", "ranger", "mt_owen"))
@click.option("--year", default="2021")
def main(aois, year):
    bands = ["B03", "B04", "B08", "SCL"]
    per_site = {}
    pooled_a, pooled_b = [], []
    for name in aois:
        aoi = get_aoi(name)
        click.echo(f"[{name}] fetching S2 + WorldCover ...")
        ds = s2_fetch.fetch_aoi(aoi, f"{year}-01-01", f"{year}-12-31", period="annual",
                                max_cloud=25, bands=bands).compute()
        last = ds.isel(time=-1)
        nd = indices.ndvi(np.asarray(last["B08"]), np.asarray(last["B04"]))
        ndw = indices.ndwi(np.asarray(last["B03"]), np.asarray(last["B08"]))
        ours = remap(wl.multiclass_spectral_only(nd, ndw), OURS)  # NODATA->0
        wc_raw = fetch_worldcover(aoi.bbox, ds)
        h = min(ours.shape[0], wc_raw.shape[0]); w = min(ours.shape[1], wc_raw.shape[1])
        a = ours[:h, :w]
        b = remap(wc_raw[:h, :w], WC)
        kap, po, n = kappa(a, b)
        per_site[name] = {"kappa": round(kap, 3), "overall_agreement": round(po, 3), "n_pixels": n}
        click.echo(f"  kappa={kap:.3f}  agreement={po:.3f}  n={n}")
        pooled_a.append(a.ravel()); pooled_b.append(b.ravel())
    A = np.concatenate(pooled_a); B = np.concatenate(pooled_b)
    kap, po, n = kappa(A, B)
    out = {"per_site": per_site, "pooled": {"kappa": round(kap, 3),
           "overall_agreement": round(po, 3), "n_pixels": n},
           "reference": "ESA WorldCover 2021 (10 m)", "scheme": "vegetation/bare/water", "year": year}
    click.echo(f"\nPOOLED: Cohen's kappa = {kap:.3f}  overall agreement = {po:.3f}  (n={n:,})")
    p = ROOT / "outputs" / "validation_worldcover.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    click.echo(f"saved -> {p}")


if __name__ == "__main__":
    main()
