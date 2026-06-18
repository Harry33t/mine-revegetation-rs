"""ON-MINE scale bridge (CPU): Meta 1 m canopy height vs Sentinel-2 over Alcoa.

Demonstrates the drone->satellite scale-bridge value WITHOUT leaving the actual
mine: a 1 m canopy-height layer (Meta/WRI, co-located on the mine) is aggregated to
the Sentinel-2 10 m grid as sub-pixel CANOPY-COVER FRACTION, then checked for
consistency against the satellite's own vegetation signal (NDVI). High agreement =
the fine layer can supervise/validate the coarse satellite over the same ground.

Honest caveats: the 1 m CHM is a model estimate (~2-3 m MAE), not drone LiDAR; and
height misses grass (low-but-green), so canopy fraction tracks woody cover, not all
vegetation. Both are stated, not hidden.

    python scripts/run_bridge_mine.py --aoi alcoa_huntly --year 2021
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import click
import numpy as np

from reveg.bridge import resample as RS
from reveg.config import AOI, get_aoi
from reveg.data import chm_fetch, s2_fetch
from reveg.trajectory import recovery as R
from reveg.viz import curves, maps


def _subwindow(bbox, km=3.0):
    """Centred ~km box (deg) inside the AOI — keeps the 1 m read tractable."""
    w, s, e, n = bbox
    cx, cy = (w + e) / 2, (s + n) / 2
    dlon = (km / 2) / 88.0
    dlat = (km / 2) / 111.0
    return (cx - dlon, cy - dlat, cx + dlon, cy + dlat)


@click.command()
@click.option("--aoi", "aoi_name", default="alcoa_huntly")
@click.option("--year", default="2021")
@click.option("--km", default=3.0, help="Sub-window size (km).")
@click.option("--canopy-min-m", default=3.0, help="Fine pixel is canopy if CHM >= this.")
@click.option("--chm-res", default=None, type=float, help="Decimate CHM read to ~this m (bigger windows).")
@click.option("--figs", default=str(ROOT / "figs"))
def main(aoi_name, year, km, canopy_min_m, chm_res, figs):
    figs = Path(figs)
    aoi = get_aoi(aoi_name)
    bbox = _subwindow(aoi.bbox, km=km)
    click.echo(f"[{aoi.name}] on-mine bridge over {tuple(round(v,4) for v in bbox)} ({km} km)")

    # 1) fine 1 m canopy height (Meta), co-located on the mine
    click.echo("  reading Meta 1 m canopy height ...")
    chm, t_chm, crs_chm = chm_fetch.fetch_chm(bbox, out_res_m=chm_res)
    canopy = np.isfinite(chm) & (chm >= canopy_min_m)
    click.echo(f"  CHM {chm.shape}  canopy(>= {canopy_min_m} m) cover = {canopy.mean():.1%}")

    # 2) co-located Sentinel-2 (annual composite, baseline-harmonized)
    click.echo(f"  fetching Sentinel-2 {year} ...")
    sub = AOI(name=f"{aoi.name}_sub", track="bridge", label=aoi.label, bbox=bbox, crs="EPSG:4326")
    s2 = s2_fetch.fetch_aoi(sub, f"{year}-01-01", f"{year}-12-31", period="annual",
                            max_cloud=20, bands=["B02", "B03", "B04", "B08", "SCL"]).compute()
    s2_last = s2.isel(time=-1)
    ndvi = R.ndvi_cube(s2).isel(time=-1).values

    # 3) aggregate 1 m canopy -> S2 10 m grid as cover fraction
    dst_t, dst_crs, dst_shape = RS.grid_from_dataset(s2)
    canopy_frac = RS.aggregate_binary_fraction(canopy, t_chm, crs_chm, dst_t, dst_crs, dst_shape)

    # 4) structure vs greenness: the satellite's NDVI does NOT encode canopy structure
    cf = canopy_frac.ravel()
    nd = np.asarray(ndvi, "float32").ravel()
    ok = np.isfinite(cf) & np.isfinite(nd)
    cfo, ndo = cf[ok], nd[ok]
    r = float(np.corrcoef(cfo, ndo)[0, 1]) if ok.sum() > 2 else float("nan")
    veg_thr, struct_thr = 0.60, 0.20
    green = ndo >= veg_thr                       # S2 says "vegetated"
    green_no_structure = green & (cfo < struct_thr)  # ...but no woody canopy
    pct = 100.0 * green_no_structure.sum() / max(int(green.sum()), 1)
    click.echo(f"  10 m cells: {int(ok.sum())}  | Pearson r(canopy_frac, NDVI) = {r:.2f} (weak by design)")
    click.echo(f"  KEY: of cells S2 calls vegetated (NDVI>={veg_thr}), {pct:.0f}% have NO woody "
               f"structure (canopy cover <{struct_thr}) -> greenness != structure")

    # save arrays for offline re-plotting / the React front-end
    out = ROOT / "data" / "processed" / "bridge" / aoi.name
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "canopy_frac_10m.npy", canopy_frac)
    np.save(out / "ndvi_10m.npy", np.asarray(ndvi, "float32"))

    # 5) figures
    maps.save_field(chm, figs / f"{aoi.name}_bridge_chm_1m.png", title="Meta ~1 m canopy height (m)",
                    cmap="YlGn", vmin=0, vmax=25, label="height (m)")
    maps.save_field(canopy_frac, figs / f"{aoi.name}_bridge_canopyfrac_10m.png",
                    title="canopy-cover fraction aggregated to 10 m (fine layer)", cmap="YlGn",
                    vmin=0, vmax=1, label="canopy fraction")
    maps.save_field(ndvi, figs / f"{aoi.name}_bridge_s2_ndvi_10m.png",
                    title=f"Sentinel-2 NDVI {year} (10 m)", cmap="RdYlGn", vmin=0, vmax=0.9, label="NDVI")
    curves.density_xy(cfo, ndo, figs / f"{aoi.name}_bridge_structure_vs_greenness.png",
                      xlabel="1 m canopy-cover fraction (structure)", ylabel="Sentinel-2 NDVI (greenness)",
                      title="structure vs greenness — NDVI saturates, can't see canopy",
                      vline=struct_thr, hline=veg_thr)
    click.echo(f"  [figs] {figs}")
    click.echo("  DONE")


if __name__ == "__main__":
    main()
