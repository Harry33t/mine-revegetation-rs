"""Export (Sentinel-2 image, weak-label mask) tile pairs for segmentation training.

Fetches an annual S2 composite over an AOI, builds the multi-class spectral weak
label, and tiles both into fixed patches -> data/processed/tiles/<aoi>/{images,masks}
+ manifest.csv. The GPU SegFormer/U-Net trainer (reveg.models) consumes these.

Image channels = [B04, B03, B02, B08] (R,G,B,NIR), reflectance float32.

    python scripts/export_training_tiles.py --aoi alcoa_huntly --year 2021 --tile 128
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import click
import numpy as np
import pandas as pd

from reveg.config import get_aoi
from reveg.data import s2_fetch
from reveg.labels import indices, weak_labels as wl
from reveg.labels.classes import LabelSpec, NODATA

IMG_BANDS = ["B04", "B03", "B02", "B08"]  # R,G,B,NIR


@click.command()
@click.option("--aoi", "aoi_name", default="alcoa_huntly")
@click.option("--year", default="2021")
@click.option("--tile", "tile_px", default=128)
@click.option("--max-cloud", default=20.0)
@click.option("--min-valid", default=0.6, help="Skip tiles with < this fraction labelled.")
@click.option("--out", default=str(ROOT / "data" / "processed" / "tiles"))
def main(aoi_name, year, tile_px, max_cloud, min_valid, out):
    aoi = get_aoi(aoi_name)
    spec = LabelSpec()
    click.echo(f"[{aoi.name}] exporting {tile_px}px tiles for {year}")

    bands = IMG_BANDS + ["SCL"]
    ds = s2_fetch.fetch_aoi(aoi, f"{year}-01-01", f"{year}-12-31", period="annual",
                            max_cloud=max_cloud, bands=bands).isel(time=-1).compute()
    img = np.stack([np.asarray(ds[b], "float32") for b in IMG_BANDS], axis=0)  # C,H,W
    nd = indices.ndvi(img[3], img[0])
    ndw = indices.ndwi(img[1], img[3])
    mask = wl.multiclass_spectral_only(nd, ndw, spec=spec)  # H,W uint8

    out_dir = Path(out) / aoi.name
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)

    _, H, W = img.shape
    rows = []
    n = 0
    for yi in range(0, H - tile_px + 1, tile_px):
        for xi in range(0, W - tile_px + 1, tile_px):
            m = mask[yi:yi + tile_px, xi:xi + tile_px]
            if (m != NODATA).mean() < min_valid:
                continue
            im = img[:, yi:yi + tile_px, xi:xi + tile_px]
            if not np.isfinite(im).all():
                im = np.nan_to_num(im, nan=0.0)
            tid = f"{aoi.name}_{year}_{yi:04d}_{xi:04d}"
            np.save(out_dir / "images" / f"{tid}.npy", im.astype("float32"))
            np.save(out_dir / "masks" / f"{tid}.npy", m.astype("uint8"))
            rows.append({"id": tid, "aoi": aoi.name, "year": year, "row": yi, "col": xi})
            n += 1
    pd.DataFrame(rows).to_csv(out_dir / "manifest.csv", index=False)
    fr = wl.class_fractions(mask)
    click.echo(f"  wrote {n} tiles -> {out_dir}")
    click.echo("  full-AOI class fractions: " + ", ".join(f"{k}={v:.2f}" for k, v in fr.items() if v > 0.005))


if __name__ == "__main__":
    main()
