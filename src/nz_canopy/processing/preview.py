"""Per-tile composite PNG + preview report (CHM preview only).

Renders, for each sampled tile, a 1x5 panel: aerial / DSM / DEM / CHM /
aerial+canopy(>=2m)+building overlay. Mirrors the Day-0 rendering. The 2 m
overlay is a visual sanity check, not a final label.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

from .vector_clip import clip_to_bbox  # noqa: E402

_CANOPY_CMAP = ListedColormap(["#2ca02c"])
_CANOPY_THRESHOLD_M = 2.0


def _crop(array: np.ndarray, bounds: tuple, res: tuple, bbox: tuple):
    """Crop a north-up array to bbox. bounds=(l,b,r,t); returns (sub, imshow_extent)."""
    left, bottom, right, top = bounds
    xres, yres = res
    bx0, by0, bx1, by1 = bbox
    h, w = array.shape[:2]
    col0 = max(0, int(np.floor((bx0 - left) / xres)))
    col1 = min(w, int(np.ceil((bx1 - left) / xres)))
    row0 = max(0, int(np.floor((top - by1) / yres)))
    row1 = min(h, int(np.ceil((top - by0) / yres)))
    sub = array[row0:row1, col0:col1]
    ext = (left + col0 * xres, left + col1 * xres, top - row1 * yres, top - row0 * yres)
    return sub, ext


@dataclass
class TilePreviewResult:
    tile_id: str
    png_path: Path
    canopy_pct: float


def render_tile_composite(
    tile_id: str,
    bbox: tuple,
    aerial_img: np.ndarray,
    aerial_extent: tuple,
    dsm,
    dem,
    chm: np.ndarray,
    buildings,
    out_png: Path,
) -> TilePreviewResult:
    """bbox=(left,bottom,right,top). aerial_extent=(l,r,b,t). dsm/dem are RasterLayer."""
    al, ar, ab, at = aerial_extent
    aerial_bounds = (al, ab, ar, at)
    aerial_res = ((ar - al) / aerial_img.shape[1], (at - ab) / aerial_img.shape[0])

    aer_sub, aer_ext = _crop(aerial_img, aerial_bounds, aerial_res, bbox)
    dsm_sub, dsm_ext = _crop(dsm.array, dsm.bounds, dsm.res, bbox)
    dem_sub, _ = _crop(dem.array, dem.bounds, dem.res, bbox)
    chm_sub, chm_ext = _crop(chm, dsm.bounds, dsm.res, bbox)

    valid = np.isfinite(chm_sub)
    canopy = np.where(valid, (chm_sub >= _CANOPY_THRESHOLD_M).astype(float), np.nan)
    canopy_pct = float(np.nansum(canopy) / valid.sum() * 100) if valid.sum() else 0.0

    fig, axes = plt.subplots(1, 5, figsize=(26, 6))
    fig.suptitle(f"Tile {tile_id}  (canopy>=2m: {canopy_pct:.1f}%)", fontsize=12)

    axes[0].imshow(aer_sub, extent=aer_ext)
    axes[0].set_title("Aerial")

    dsm_disp = np.where(dsm_sub == dsm.nodata, np.nan, dsm_sub) if dsm.nodata is not None else dsm_sub
    im1 = axes[1].imshow(dsm_disp, extent=dsm_ext, cmap="terrain")
    axes[1].set_title("DSM (m)")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    dem_disp = np.where(dem_sub == dem.nodata, np.nan, dem_sub) if dem.nodata is not None else dem_sub
    im2 = axes[2].imshow(dem_disp, extent=dsm_ext, cmap="terrain")
    axes[2].set_title("DEM (m)")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    im3 = axes[3].imshow(np.clip(chm_sub, 0, 30), extent=chm_ext, cmap="Greens", vmin=0, vmax=25)
    axes[3].set_title("CHM = DSM - DEM (m)")
    plt.colorbar(im3, ax=axes[3], fraction=0.046)

    axes[4].imshow(aer_sub, extent=aer_ext, alpha=0.85)
    axes[4].imshow(np.where(canopy == 1, 1, np.nan), extent=chm_ext, cmap=_CANOPY_CMAP, alpha=0.45)
    try:
        b_clip = clip_to_bbox(buildings, bbox)
        if len(b_clip) > 0:
            b_clip.boundary.plot(ax=axes[4], color="red", linewidth=0.6)
    except Exception:
        pass
    axes[4].set_title("Aerial + canopy(>=2m) + buildings")

    for ax in axes:
        ax.set_xlim(bbox[0], bbox[2])
        ax.set_ylim(bbox[1], bbox[3])
        ax.set_aspect("equal")

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return TilePreviewResult(tile_id, out_png, canopy_pct)


def write_preview_report(
    city: str,
    out_md: Path,
    *,
    aerial_year: str,
    aerial_crs,
    aerial_res: tuple,
    lidar_crs,
    lidar_res: tuple,
    working_pixel_size_m: float,
    lidar_survey_id: str | None,
    n_tiles_total: int,
    tiles: list[TilePreviewResult],
    alignment_warnings: list[str],
    gpkg_path: Path,
) -> Path:
    out_md = Path(out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Preview report — {city}",
        "",
        "## Layers",
        f"- Aerial CRS: {aerial_crs}  | resolution: {aerial_res[0]:.3g} m  | year: {aerial_year}",
        f"- LiDAR CRS: {lidar_crs}  | resolution: {lidar_res[0]:.3g} m  | survey: {lidar_survey_id}",
        f"- Working pixel size (target): {working_pixel_size_m} m",
        "",
        "## Tile grid",
        f"- Total tiles over extent: {n_tiles_total}",
        f"- Sampled tiles: {len(tiles)}",
        f"- Tile grid GeoPackage: `{gpkg_path.name}`",
        "",
        "## Sampled tiles",
        "| tile_id | canopy>=2m % | png |",
        "|---|---|---|",
    ]
    for t in tiles:
        lines.append(f"| {t.tile_id} | {t.canopy_pct:.1f} | `{t.png_path.name}` |")
    lines += ["", "## Alignment warnings"]
    lines += [f"- {w}" for w in alignment_warnings] if alignment_warnings else ["- none"]
    lines.append("")
    out_md.write_text("\n".join(lines), encoding="utf-8")
    return out_md
