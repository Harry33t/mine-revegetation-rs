"""End-to-end satellite-track demo on REAL Sentinel-2 data (CPU, no GPU needed).

Fetch annual S2 composites over an Australian rehab-mine AOI, then:
  1) true-colour RGB quicklook of the latest year,
  2) multi-class spectral weak-label map (bare/water/vegetation/senescent),
  3) restoration trajectory (mean NDVI + veg cover + standardised recovery).

Figures -> figs/, composite + trajectory CSV -> data/processed/ (gitignored).

    python scripts/run_satellite_demo.py --aoi alcoa_huntly --start 2019-01-01 --end 2024-12-31
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import click
import numpy as np

from reveg.config import get_aoi
from reveg.data import s2_fetch
from reveg.labels import indices, weak_labels as wl
from reveg.labels.classes import LabelSpec
from reveg.trajectory import recovery as R
from reveg.viz import curves, maps

BANDS = ["B02", "B03", "B04", "B08", "SCL"]  # blue, green, red, NIR, scene-class


@click.command()
@click.option("--aoi", "aoi_name", default="alcoa_huntly")
@click.option("--start", default="2019-01-01")
@click.option("--end", default="2024-12-31")
@click.option("--max-cloud", default=20.0)
@click.option("--figs", default=str(ROOT / "figs"))
@click.option("--out", default=str(ROOT / "data" / "processed" / "s2"))
def main(aoi_name, start, end, max_cloud, figs, out):
    aoi = get_aoi(aoi_name)
    spec = LabelSpec()
    figs = Path(figs)
    click.echo(f"[{aoi.name}] {aoi.label}\n  {start}..{end}  bbox={aoi.bbox}")

    click.echo("  fetching annual Sentinel-2 composites ...")
    ds = s2_fetch.fetch_aoi(aoi, start, end, period="annual", max_cloud=max_cloud, bands=BANDS)
    ds = ds.compute()
    click.echo(f"  cube: {dict(ds.sizes)}  years={[int(t.dt.year) for t in ds.time]}")

    # 1) RGB quicklook of the latest year
    latest = ds.isel(time=-1)
    rgb = maps.s2_rgb(latest)
    p_rgb = maps.save_rgb(rgb, figs / f"{aoi.name}_s2_rgb.png",
                          title=f"{aoi.label} — Sentinel-2 RGB ({int(latest.time.dt.year)})")
    click.echo(f"  [fig] {p_rgb}")

    # 2) multi-class spectral weak label (latest year)
    nd = indices.ndvi(np.asarray(latest["B08"]), np.asarray(latest["B04"]))
    ndw = indices.ndwi(np.asarray(latest["B03"]), np.asarray(latest["B08"]))
    label = wl.multiclass_spectral_only(nd, ndw, spec=spec)
    fr = wl.class_fractions(label)
    click.echo("  class fractions: " + ", ".join(f"{k}={v:.2f}" for k, v in fr.items() if v > 0.005))
    p_lab = maps.save_label_map(label, figs / f"{aoi.name}_weak_label.png",
                                title=f"{aoi.label} — weak label ({int(latest.time.dt.year)})")
    click.echo(f"  [fig] {p_lab}")

    # 3) restoration trajectory
    df = R.cover_series(ds, spec=spec)
    bare, ref = R.anchors_from_series(df)
    df = R.add_recovery_fraction(df, ndvi_bare=bare, ndvi_ref=ref)
    fit = R.fit_recovery_time(df, target=0.8)
    out = Path(out) / aoi.name
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{aoi.name}_trajectory.csv")
    p_curve = curves.plot_trajectory(df, figs / f"{aoi.name}_trajectory.png",
                                     title=f"{aoi.label} — restoration trajectory")
    click.echo(f"  [fig] {p_curve}")
    click.echo(f"  anchors: bare NDVI={bare:.3f} ref NDVI={ref:.3f}")
    yt = fit.get("years_to_target")
    click.echo(f"  logistic fit: years_to_80%_recovery="
               + (f"{yt:.1f}" if yt == yt else "n/a"))
    click.echo("  DONE")


if __name__ == "__main__":
    main()
