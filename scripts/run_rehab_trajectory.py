"""Data-driven restoration trajectory over the BASELINE-cleared footprint.

Instead of guessing a rehab-pit bounding box, this selects pixels that were
cleared/bare in the first year (NDVI below a threshold) — using only the start
state, not the later outcome — and tracks their NDVI recovery toward the
surrounding mature-forest baseline. Honest recovery signal, not selection-on-outcome.

    python scripts/run_rehab_trajectory.py --aoi alcoa_huntly --start 2019-01-01 --end 2024-12-31
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
from reveg.trajectory import recovery as R
from reveg.viz import curves, maps

BANDS = ["B02", "B03", "B04", "B08", "SCL"]


@click.command()
@click.option("--aoi", "aoi_name", default="alcoa_huntly")
@click.option("--start", default="2019-01-01")
@click.option("--end", default="2024-12-31")
@click.option("--max-cloud", default=15.0)
@click.option("--baseline-ndvi-max", default=0.40, help="Pixel is 'cleared' if baseline NDVI < this.")
@click.option("--figs", default=str(ROOT / "figs"))
@click.option("--out", default=str(ROOT / "data" / "processed" / "s2"))
def main(aoi_name, start, end, max_cloud, baseline_ndvi_max, figs, out):
    aoi = get_aoi(aoi_name)
    figs = Path(figs)
    click.echo(f"[{aoi.name}] {aoi.label}\n  {start}..{end}")

    click.echo("  fetching annual Sentinel-2 composites ...")
    ds = s2_fetch.fetch_aoi(aoi, start, end, period="annual", max_cloud=max_cloud, bands=BANDS).compute()
    years = [int(t.dt.year) for t in ds.time]
    click.echo(f"  cube: {dict(ds.sizes)}  years={years}")

    # baseline-cleared footprint (start state only)
    mask = R.rehab_mask_from_baseline(ds, year_index=0, ndvi_max=baseline_ndvi_max)
    n_pix = int(mask.sum())
    frac = n_pix / float(mask.size)
    click.echo(f"  baseline-cleared pixels: {n_pix} ({frac:.1%} of AOI; NDVI<{baseline_ndvi_max} in {years[0]})")

    # mature-forest reference = 90th pct of last-year NDVI over the whole AOI
    nd_last = R.ndvi_cube(ds).isel(time=-1)
    ndvi_ref = float(nd_last.quantile(0.90).compute())

    # trajectory over the cleared footprint, recovery toward the forest reference
    df = R.cover_series(ds, mask=mask)
    ndvi_bare = float(df["mean_ndvi"].iloc[0])
    df = R.add_recovery_fraction(df, ndvi_bare=ndvi_bare, ndvi_ref=ndvi_ref)
    fit = R.fit_recovery_time(df, target=0.8)
    click.echo("  per-year mean NDVI over cleared footprint: "
               + ", ".join(f"{y}:{v:.3f}" for y, v in zip(years, df["mean_ndvi"])))
    click.echo(f"  anchors: bare(={years[0]})={ndvi_bare:.3f}  forest-ref(90pct)={ndvi_ref:.3f}")
    yt = fit.get("years_to_target")
    click.echo("  logistic years_to_80%_recovery=" + (f"{yt:.1f}" if yt == yt else "n/a"))

    # figures
    rgb = maps.s2_rgb(ds.isel(time=-1))
    p_over = maps.save_mask_overlay(rgb, mask.values, figs / f"{aoi.name}_rehab_footprint.png",
                                    title=f"{aoi.label} — baseline-cleared footprint ({years[0]})")
    p_curve = curves.plot_trajectory(df, figs / f"{aoi.name}_rehab_trajectory.png",
                                     title=f"{aoi.label} — recovery over cleared footprint")
    click.echo(f"  [fig] {p_over}\n  [fig] {p_curve}")

    out = Path(out) / aoi.name
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{aoi.name}_rehab_trajectory.csv")
    click.echo("  DONE")


if __name__ == "__main__":
    main()
