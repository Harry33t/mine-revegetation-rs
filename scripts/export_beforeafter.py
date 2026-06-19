"""Before/after imagery for the web swipe slider.

Finds the window within an AOI where vegetation recovered most (largest NDVI gain
2019->2024), then renders aligned true-colour and NDVI images for both years. The
slider then lets a viewer wipe between 2019 and 2024 over real ground.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import click
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from reveg.config import get_aoi
from reveg.data import s2_fetch
from reveg.labels import indices

ASSETS = ROOT / "web" / "public" / "assets"


def _rgb(comp, p=(2, 98)):
    chans = []
    for b in ("B04", "B03", "B02"):
        a = np.asarray(comp[b], "float32")
        fin = a[np.isfinite(a)]
        lo, hi = (np.percentile(fin, p) if fin.size else (0, 1))
        chans.append(np.clip(np.nan_to_num((a - lo) / (hi - lo + 1e-9)), 0, 1))
    return (np.stack(chans, -1) * 255).astype("uint8")


@click.command()
@click.option("--aoi", "aoi_name", default="alcoa_huntly")
@click.option("--start-year", default=2019)
@click.option("--end-year", default=2024)
@click.option("--win-km", default=3.0)
def main(aoi_name, start_year, end_year, win_km):
    aoi = get_aoi(aoi_name)
    ASSETS.mkdir(parents=True, exist_ok=True)
    bands = ["B02", "B03", "B04", "B08", "SCL"]
    # fetch only the two endpoint years (fast), not the whole series
    d0 = s2_fetch.fetch_aoi(aoi, f"{start_year}-01-01", f"{start_year}-12-31", period="annual",
                            max_cloud=20, bands=bands).compute()
    d1 = s2_fetch.fetch_aoi(aoi, f"{end_year}-01-01", f"{end_year}-12-31", period="annual",
                            max_cloud=20, bands=bands).compute()
    years = [start_year, end_year]
    first, last = d0.isel(time=0), d1.isel(time=0)
    nd0 = indices.ndvi(np.asarray(first["B08"]), np.asarray(first["B04"]))
    nd1 = indices.ndvi(np.asarray(last["B08"]), np.asarray(last["B04"]))
    gain = np.nan_to_num(nd1 - nd0, nan=-9)

    # find the win_km window with the largest mean NDVI gain (most recovery)
    h, w = gain.shape
    win = min(int(win_km * 100), h, w)  # 10 m px
    stride = max(1, win // 4)
    best, byx = -9.0, (0, 0)
    for y in range(0, h - win + 1, stride):
        for x in range(0, w - win + 1, stride):
            m = float(gain[y:y + win, x:x + win].mean())
            if m > best:
                best, byx = m, (y, x)
    y0, x0 = byx
    sl = (slice(y0, y0 + win), slice(x0, x0 + win))
    click.echo(f"[{aoi.name}] recovery window mean NDVI gain={best:.3f} "
               f"({years[0]}->{years[-1]})")

    def save_rgb(comp, name):
        plt.imsave(ASSETS / name, _rgb(comp.isel(y=sl[0], x=sl[1])))

    def save_ndvi(nd, name):
        plt.imsave(ASSETS / name, nd[sl], cmap="RdYlGn", vmin=0.0, vmax=0.8)

    save_rgb(first, "ba_rgb_2019.png"); save_rgb(last, "ba_rgb_2024.png")
    save_ndvi(nd0, "ba_ndvi_2019.png"); save_ndvi(nd1, "ba_ndvi_2024.png")
    click.echo(f"  window NDVI: {years[0]}={float(nd0[sl].mean()):.3f}  "
               f"{years[-1]}={float(nd1[sl].mean()):.3f}")
    click.echo(f"  saved 4 images -> {ASSETS}")


if __name__ == "__main__":
    main()
