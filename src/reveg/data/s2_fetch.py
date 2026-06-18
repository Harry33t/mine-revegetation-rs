"""Sentinel-2 L2A time-series ingestion via the Microsoft Planetary Computer STAC.

No account / institutional email required: the Planetary Computer STAC is public and
assets are signed anonymously with ``planetary_computer.sign_inplace``. This is the
fallback chosen over the DEA Sandbox (which needs a .edu.au/.gov.au email).

Pipeline: search -> load cube (odc-stac) -> SCL cloud/shadow mask -> period median
composite. Heavy deps (odc-stac, pystac-client, planetary-computer) are imported
lazily so the rest of the package imports on a bare dev box.

CLI:
    python -m reveg.data.s2_fetch alcoa_huntly --start 2019-01-01 --end 2024-12-31 \
        --period annual --max-cloud 20 --out data/processed/s2/alcoa_huntly
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoid hard import at module load
    import xarray as xr

from ..config import AOI, get_aoi

# Surface-reflectance bands we use (10 m + 20 m red-edge/SWIR, resampled to 10 m).
# B08 = NIR, B04 = red, B03 = green, B02 = blue, B05/B06/B07 = red-edge, B11 = SWIR.
DEFAULT_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B11", "SCL"]

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "sentinel-2-l2a"

# Scene Classification Layer codes to KEEP (drop clouds/shadow/snow/saturated/nodata).
# 2 dark-area, 4 vegetation, 5 bare-soil, 6 water, 7 unclassified.
SCL_KEEP = (2, 4, 5, 6, 7)


def _require(modname: str):
    import importlib

    try:
        return importlib.import_module(modname)
    except ImportError as e:  # pragma: no cover - environment guard
        raise ImportError(
            f"'{modname}' is required for Sentinel-2 ingestion. Install with: "
            "pip install odc-stac pystac-client planetary-computer"
        ) from e


def search_items(
    bbox: tuple[float, float, float, float],
    start: str,
    end: str,
    *,
    max_cloud: float = 20.0,
):
    """Return STAC items for the AOI/date range below a cloud-cover threshold."""
    planetary_computer = _require("planetary_computer")
    pystac_client = _require("pystac_client")
    client = pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)
    search = client.search(
        collections=[COLLECTION],
        bbox=list(bbox),
        datetime=f"{start}/{end}",
        query={"eo:cloud_cover": {"lt": max_cloud}},
    )
    return list(search.items())


def load_cube(
    items,
    bbox: tuple[float, float, float, float],
    *,
    bands=DEFAULT_BANDS,
    resolution: int = 10,
    crs: str | None = None,
) -> "xr.Dataset":
    """Lazily load matched items into an xarray cube on a common 10 m grid."""
    odc_stac = _require("odc.stac")
    # Robust remote-COG reads: GDAL HTTP retries + cloud-friendly defaults. Without
    # this, a single flaky tile read raises "Chunk and warp failed" and kills the load.
    try:
        odc_stac.configure_rio(
            cloud_defaults=True,
            GDAL_HTTP_MAX_RETRY=10,
            GDAL_HTTP_RETRY_DELAY=1,
        )
    except Exception:
        pass
    ds = odc_stac.load(
        items,
        bands=bands,
        bbox=list(bbox),
        resolution=resolution,
        crs=crs or "utm",  # odc picks the AOI's UTM zone
        chunks={"x": 2048, "y": 2048},
        groupby="solar_day",
    )
    return ds


def mask_clouds(ds: "xr.Dataset", *, scl_keep=SCL_KEEP) -> "xr.Dataset":
    """Mask every band where SCL is not a kept (clear) class."""
    xr = _require("xarray")  # noqa: F841  (ensures xarray present)
    if "SCL" not in ds:
        return ds
    keep = ds["SCL"].isin(list(scl_keep))
    data_vars = [v for v in ds.data_vars if v != "SCL"]
    return ds[data_vars].where(keep)


HARMONIZE_CUTOFF = "2022-01-25"  # Sentinel-2 processing baseline 04.00 switch
HARMONIZE_OFFSET = 1000          # +1000 DN BOA offset added from that date


def harmonize_baseline(
    ds: "xr.Dataset", *, cutoff: str = HARMONIZE_CUTOFF, offset: int = HARMONIZE_OFFSET
) -> "xr.Dataset":
    """Undo the Sentinel-2 baseline-04.00 radiometric offset so a multi-year series
    is comparable. From 2022-01-25, L2A reflectance carries a +`offset` DN shift the
    Planetary Computer does not auto-correct; left in, it adds to the NDVI denominator
    and fakes a step drop at the 2021/2022 boundary. Subtract it from optical bands
    (not SCL) for scenes on/after the cutoff, clipping at 0."""
    import numpy as np

    if "time" not in ds.dims:
        return ds
    post = ds["time"] >= np.datetime64(cutoff)
    optical = [v for v in ds.data_vars if v != "SCL"]
    for v in optical:
        shifted = (ds[v] - offset).clip(min=0)
        ds[v] = ds[v].where(~post, shifted)
    return ds


def period_median(ds: "xr.Dataset", *, period: str = "annual") -> "xr.Dataset":
    """Cloud-free composite per period. period in {'annual','seasonal','monthly'}."""
    freq = {"annual": "YS", "seasonal": "QS", "monthly": "MS"}.get(period)
    if freq is None:
        raise ValueError(f"period must be annual|seasonal|monthly, got {period!r}")
    return ds.resample(time=freq).median(skipna=True)


def fetch_aoi(
    aoi: AOI | str,
    start: str,
    end: str,
    *,
    period: str = "annual",
    max_cloud: float = 20.0,
    bands=DEFAULT_BANDS,
    resolution: int = 10,
) -> "xr.Dataset":
    """End-to-end: search -> load -> cloud-mask -> period composite for one AOI."""
    aoi_obj = aoi if isinstance(aoi, AOI) else get_aoi(aoi)
    items = search_items(aoi_obj.bbox, start, end, max_cloud=max_cloud)
    if not items:
        raise RuntimeError(
            f"No Sentinel-2 items for {aoi_obj.name} {start}..{end} (cloud<{max_cloud}%). "
            "Widen the date range or raise --max-cloud."
        )
    cube = load_cube(items, aoi_obj.bbox, bands=bands, resolution=resolution)
    cube = harmonize_baseline(cube)
    cube = mask_clouds(cube)
    return period_median(cube, period=period)


def save_netcdf(ds: "xr.Dataset", out_dir: str | Path, name: str) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.nc"
    ds.to_netcdf(path)
    return path


def _main() -> None:
    import click

    @click.command()
    @click.argument("aoi_name")
    @click.option("--start", required=True, help="ISO start date, e.g. 2019-01-01")
    @click.option("--end", required=True, help="ISO end date, e.g. 2024-12-31")
    @click.option("--period", default="annual", type=click.Choice(["annual", "seasonal", "monthly"]))
    @click.option("--max-cloud", default=20.0, help="Max scene cloud cover %.")
    @click.option("--out", default="data/processed/s2", help="Output directory.")
    def cli(aoi_name, start, end, period, max_cloud, out):
        aoi = get_aoi(aoi_name)
        click.echo(f"[{aoi.name}] {aoi.label}\n  bbox={aoi.bbox} {start}..{end} period={period}")
        ds = fetch_aoi(aoi, start, end, period=period, max_cloud=max_cloud)
        n_t = ds.sizes.get("time", 0)
        click.echo(f"  composited cube: {n_t} time steps, vars={list(ds.data_vars)}")
        path = save_netcdf(ds, Path(out) / aoi.name, f"{aoi.name}_{period}_{start}_{end}")
        click.echo(f"  saved -> {path}")

    cli()


if __name__ == "__main__":
    _main()
