"""Label-quality audit: per-tile stats, CSV, report, and quicklook panels.

These quantify the pseudo-labels (canopy %, how much was removed as buildings)
and produce auditable artifacts for M3. They do NOT validate against ground
truth — that is the M3 manual pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

from .pseudo_label import PseudoLabelResult  # noqa: E402

_MASK_CMAP = ListedColormap(["#2ca02c"])


def _crop(array, bounds, res, bbox):
    left, bottom, right, top = bounds
    xres, yres = res
    bx0, by0, bx1, by1 = bbox
    h, w = array.shape[:2]
    c0 = max(0, int(np.floor((bx0 - left) / xres)))
    c1 = min(w, int(np.ceil((bx1 - left) / xres)))
    r0 = max(0, int(np.floor((top - by1) / yres)))
    r1 = min(h, int(np.ceil((top - by0) / yres)))
    sub = array[r0:r1, c0:c1]
    ext = (left + c0 * xres, left + c1 * xres, top - r1 * yres, top - r0 * yres)
    return sub, ext


def tile_stats(result: PseudoLabelResult, grid) -> pd.DataFrame:
    """Per-tile stats over the M1 tile grid."""
    chm = result.chm
    px_area = abs(chm.res[0] * chm.res[1]) or 1.0
    rows = []
    for _, tile in grid.iterrows():
        bbox = tuple(tile.geometry.bounds)
        valid, _ = _crop(result.valid, chm.bounds, chm.res, bbox)
        final, _ = _crop(result.final_mask, chm.bounds, chm.res, bbox)
        bld, _ = _crop(result.building_mask, chm.bounds, chm.res, bbox)
        raw, _ = _crop(result.raw_mask, chm.bounds, chm.res, bbox)
        valid_px = int(valid.sum())
        canopy_px = int((final & valid).sum())
        raw_px = int((raw & valid).sum())
        removed_px = int((raw & bld).sum())
        rows.append(
            {
                "tile_id": tile.tile_id,
                "threshold_m": result.threshold_m,
                "valid_px": valid_px,
                "canopy_px": canopy_px,
                "canopy_pct": round(canopy_px / valid_px * 100, 2) if valid_px else 0.0,
                "raw_canopy_pct": round(raw_px / valid_px * 100, 2) if valid_px else 0.0,
                "building_px": int(bld.sum()),
                "removed_area_m2": round(removed_px * px_area, 1),
            }
        )
    return pd.DataFrame(rows)


def slope_bin_summary(stats: pd.DataFrame) -> pd.DataFrame:
    """Per slope_bin: tile count, mean raw vs final canopy %, total building-removed area.
    Requires `slope_bin` (from M2.1) and `raw_canopy_pct` columns."""
    order = ["flat", "gentle", "moderate", "steep", "unknown"]
    g = stats.groupby("slope_bin")
    out = g.agg(
        tile_count=("tile_id", "size"),
        mean_raw_canopy_pct=("raw_canopy_pct", "mean"),
        mean_final_canopy_pct=("canopy_pct", "mean"),
        total_removed_area_m2=("removed_area_m2", "sum"),
    ).round(2)
    out = out.reindex([b for b in order if b in out.index])
    return out.reset_index()


def write_slope_audit_report(city: str, stats: pd.DataFrame, summary: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Slope audit (M2.1) — {city}",
        "",
        "> Audit only. Purpose: check whether bare/steep slopes inflate CHM>=2m into false "
        "canopy (memo §15). The pseudo-label algorithm is NOT changed here; slope-aware "
        "refinement (M2.2) is warranted only if steep bins show clearly inflated final canopy.",
        "",
        "## Canopy by slope bin",
        "| slope_bin | tiles | mean raw canopy % | mean final canopy % | removed area m2 |",
        "|---|---|---|---|---|",
    ]
    for _, r in summary.iterrows():
        lines.append(
            f"| {r['slope_bin']} | {int(r['tile_count'])} | {r['mean_raw_canopy_pct']:.1f} | "
            f"{r['mean_final_canopy_pct']:.1f} | {r['total_removed_area_m2']:.0f} |"
        )
    steep = summary[summary["slope_bin"] == "steep"]
    lines += ["", "## Read"]
    if len(steep):
        s = steep.iloc[0]
        flag = s["mean_final_canopy_pct"] > 60
        lines.append(
            f"- Steep bin (>20 deg): {int(s['tile_count'])} tiles, mean final canopy "
            f"{s['mean_final_canopy_pct']:.1f}%. "
            + ("**Elevated — inspect quicklooks for bare-slope false positives; M2.2 may be needed.**"
               if flag else "Not obviously inflated; slope-aware refinement likely unnecessary for this AOI.")
        )
    else:
        lines.append("- No steep (>20 deg) tiles in this AOI.")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_label_stats_csv(stats: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stats.to_csv(path, index=False, encoding="utf-8")
    return path


def write_audit_report(city: str, result: PseudoLabelResult, stats: pd.DataFrame, proc, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    valid_px = int(result.valid.sum())
    px_area = abs(result.chm.res[0] * result.chm.res[1]) or 1.0
    canopy_area = int(result.final_mask.sum()) * px_area
    removed_area = int((result.raw_mask & result.building_mask).sum()) * px_area
    lines = [
        f"# Pseudo-label audit — {city}",
        "",
        "> **These are pseudo-labels (weak labels), not ground truth.** They require "
        "manual audit in M3 (calibration + double-pass + Meta/WRI triangulation, memo §9).",
        "",
        "## Settings",
        f"- CRS: {result.chm.crs}",
        f"- Threshold: {result.threshold_m} m  (candidates {proc.chm['thresholds_m']})",
        f"- Building buffer: {proc.chm['building_buffer_m']} m",
        f"- Min canopy area: {proc.chm['min_canopy_area_m2']} m^2",
        "",
        "## City summary",
        f"- Valid area: {valid_px * px_area / 1e6:.3f} km^2",
        f"- Final canopy: {canopy_area / 1e6:.3f} km^2 ({canopy_area / (valid_px * px_area) * 100:.1f}%)",
        f"- Removed as buildings (within raw mask): {removed_area / 1e6:.4f} km^2",
        "",
        "## Per-tile distribution",
        f"- Tiles: {len(stats)}",
        f"- Canopy %% — mean {stats['canopy_pct'].mean():.1f}, median {stats['canopy_pct'].median():.1f}, "
        f"min {stats['canopy_pct'].min():.1f}, max {stats['canopy_pct'].max():.1f}",
        "",
        "## Next step (M3)",
        "- Sample candidate tiles stratified by canopy density for the 500+ tile manual audit.",
        "- Flag tiles where pseudo-label and Meta/WRI 1 m canopy disagree for priority review.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


@dataclass
class _Img:
    image: np.ndarray
    extent: tuple


def quicklook(tile_id: str, bbox: tuple, aerial: _Img, result: PseudoLabelResult, out_png: Path) -> Path:
    chm = result.chm
    al, ar, ab, at = aerial.extent
    aer_bounds = (al, ab, ar, at)
    aer_res = ((ar - al) / aerial.image.shape[1], (at - ab) / aerial.image.shape[0])
    aer_sub, aer_ext = _crop(aerial.image, aer_bounds, aer_res, bbox)
    chm_sub, cext = _crop(chm.array, chm.bounds, chm.res, bbox)
    raw_sub, _ = _crop(result.raw_mask, chm.bounds, chm.res, bbox)
    rem_sub, _ = _crop(result.building_removed_mask, chm.bounds, chm.res, bbox)
    fin_sub, _ = _crop(result.final_mask, chm.bounds, chm.res, bbox)

    fig, axes = plt.subplots(1, 5, figsize=(26, 6))
    fig.suptitle(f"Pseudo-label {tile_id} (threshold {result.threshold_m} m)", fontsize=12)
    axes[0].imshow(aer_sub, extent=aer_ext)
    axes[0].set_title("Aerial")
    im = axes[1].imshow(np.clip(chm_sub, 0, 30), extent=cext, cmap="Greens", vmin=0, vmax=25)
    axes[1].set_title("CHM (m)")
    plt.colorbar(im, ax=axes[1], fraction=0.046)
    for ax, m, title in [
        (axes[2], raw_sub, "Raw mask (>=thr)"),
        (axes[3], rem_sub, "Building-removed"),
        (axes[4], fin_sub, "Final (cleaned)"),
    ]:
        ax.imshow(aer_sub, extent=aer_ext, alpha=0.7)
        ax.imshow(np.where(m, 1, np.nan), extent=cext, cmap=_MASK_CMAP, alpha=0.55)
        ax.set_title(title)
    for ax in axes:
        ax.set_xlim(bbox[0], bbox[2])
        ax.set_ylim(bbox[1], bbox[3])
        ax.set_aspect("equal")
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out_png
