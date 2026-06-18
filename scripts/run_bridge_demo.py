"""NZ scale-bridge demo on REAL co-located data (CPU).

Co-located triple (LINZ, Bay of Plenty 2018-2019): RGB aerial + 1 m DSM + 1 m DEM.
  real CHM = DSM - DEM  ->  fine multi-class label (height tiers + ExG greenness)
  aggregate to the Sentinel-2 10 m grid (area-fraction majority + purity)
  fetch co-located Sentinel-2  ->  spectral label  ->  agreement (validates the bridge)

    python scripts/run_bridge_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import click
import numpy as np
from rasterio.warp import Resampling, reproject

from reveg.bridge import pipeline as BP
from reveg.config import AOI
from reveg.data import linz_fetch as LF
from reveg.data import s2_fetch
from reveg.labels import indices, weak_labels as wl
from reveg.labels.classes import LabelSpec


@click.command()
@click.option("--rgb", default=LF.DEFAULT_RGB)
@click.option("--dsm", default=LF.DEFAULT_DSM)
@click.option("--dem", default=LF.DEFAULT_DEM)
@click.option("--year", default="2019")
@click.option("--figs", default=str(ROOT / "figs"))
def main(rgb, dsm, dem, year, figs):
    figs = Path(figs)
    spec = LabelSpec()
    click.echo("NZ scale-bridge demo (LINZ Bay of Plenty 2018-2019)")

    # 1) aerial RGB decimated to ~1 m -> ExG greenness
    click.echo("  reading aerial RGB ...")
    rgb_arr, t_img, crs_img, b_img = LF.read_full_decimated(rgb, out_res=1.0, indexes=[1, 2, 3])
    exg_img = indices.exg(rgb_arr[0], rgb_arr[1], rgb_arr[2])

    # 2) DSM/DEM windowed to the aerial extent -> real CHM (master grid)
    click.echo("  reading DSM/DEM, building real CHM ...")
    dsm_arr, t_chm, crs_chm = LF.read_window(dsm, b_img)
    dem_arr, _, _ = LF.read_window(dem, b_img)
    dsm2 = dsm_arr[0].astype("float32")
    dem2 = dem_arr[0].astype("float32")
    dsm2[dsm2 < -1000] = np.nan
    dem2[dem2 < -1000] = np.nan
    chm = (dsm2 - dem2).astype("float32")

    # 3) ExG -> CHM grid, then fine multi-class label
    exg_on_chm = np.full(chm.shape, np.nan, "float32")
    reproject(exg_img, exg_on_chm, src_transform=t_img, src_crs=crs_img,
              dst_transform=t_chm, dst_crs=crs_chm, resampling=Resampling.average)
    fine = wl.multiclass_height_greenness(chm, exg_on_chm, spec=spec)
    fr = wl.class_fractions(fine)
    click.echo("  fine label fractions: " + ", ".join(f"{k}={v:.2f}" for k, v in fr.items() if v > 0.005))

    # 4) co-located Sentinel-2 over the same ground
    lon = LF.bounds_to_lonlat(b_img, crs_chm)
    click.echo(f"  fetching co-located Sentinel-2 ({year}) over {tuple(round(v,4) for v in lon)} ...")
    aoi = AOI(name="linz_bridge", track="bridge", label="LINZ bridge", bbox=lon, crs="EPSG:4326")
    s2 = s2_fetch.fetch_aoi(aoi, f"{year}-01-01", f"{year}-12-31", period="annual",
                            max_cloud=30, bands=["B02", "B03", "B04", "B08", "SCL"]).compute()

    # 5) bridge: aggregate fine -> S2 grid, validate
    out = BP.bridge_report(fine, t_chm, crs_chm, s2, spec=spec)
    rep = out["agreement"]
    click.echo(f"  S2 grid: {out['s2_label'].shape}  | aggregated fine purity mean="
               f"{float(np.nanmean(out['purity'][out['purity']>0])):.2f}")
    click.echo(f"  AGREEMENT overall={rep['overall_agreement']:.2f} mean_iou={rep['mean_iou']:.2f} "
               f"n={rep['n_pixels']}")
    click.echo("  per-class IoU: " + ", ".join(
        f"{k}={v:.2f}" for k, v in rep["per_class_iou"].items() if v == v))

    # 6) figures
    from reveg.viz import maps
    rgb_disp = np.moveaxis(rgb_arr, 0, -1)
    maps.save_rgb(rgb_disp, figs / "bridge_aerial_rgb.png", title="LINZ aerial RGB (~1 m)")
    maps.save_label_map(fine, figs / "bridge_fine_label_1m.png", title="fine label (CHM+ExG, 1 m)")
    maps.save_label_map(out["coarse_label"], figs / "bridge_fine_to_10m.png",
                        title="fine label aggregated to S2 10 m")
    maps.save_label_map(out["s2_label"], figs / "bridge_s2_label_10m.png",
                        title="Sentinel-2 spectral label (10 m)")
    click.echo(f"  [figs] {figs}")
    click.echo("  DONE")


if __name__ == "__main__":
    main()
