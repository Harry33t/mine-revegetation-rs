"""Export data for the 3D front-end widgets into web/public/data:

  heightmap.json  a downsampled Meta canopy-height grid over Alcoa -> a 3D,
                  auto-rotating terrain surface (the "structure the satellite
                  can't see", now in 3D).
  sites.json      AOI centroids (lat/lon) -> markers on an auto-rotating globe.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from reveg.config import load_aois
from reveg.data import chm_fetch

DATA = ROOT / "web" / "public" / "data"


def _subwindow(bbox, km=4.0):
    w, s, e, n = bbox
    cx, cy = (w + e) / 2, (s + n) / 2
    dlon, dlat = (km / 2) / 88.0, (km / 2) / 111.0
    return (cx - dlon, cy - dlat, cx + dlon, cy + dlat)


def _most_disturbed_window(chm, win_px, canopy_min_m=3.0):
    """Find the win_px-square sub-window with the most canopy/clearing CONTRAST
    (highest std of canopy presence) — i.e. the mine/forest boundary, not pure
    forest. Returns (y0, x0). Falls back to centre if the AOI is uniform."""
    canopy = (np.nan_to_num(chm, nan=0.0) >= canopy_min_m).astype("float32")
    h, w = canopy.shape
    win = min(win_px, h, w)
    best, best_yx = -1.0, (max(0, (h - win) // 2), max(0, (w - win) // 2))
    stride = max(1, win // 4)
    for y0 in range(0, h - win + 1, stride):
        for x0 in range(0, w - win + 1, stride):
            block = canopy[y0:y0 + win, x0:x0 + win]
            s = float(block.std())  # 0 = all-forest or all-bare; max ~0.5 = half/half
            if s > best:
                best, best_yx = s, (y0, x0)
    return best_yx, win


def export_heightmap(aoi_name="alcoa_huntly", grid=96):
    aois = load_aois()
    # read the WHOLE AOI coarse, then zoom into the most disturbed (mine) window
    chm_full, _, _ = chm_fetch.fetch_chm(aois[aoi_name].bbox, out_res_m=20.0)
    win_px = grid * 2  # ~3.8 km window at 20 m
    (y0, x0), win = _most_disturbed_window(chm_full, win_px)
    region = np.nan_to_num(chm_full[y0:y0 + win, x0:x0 + win], nan=0.0).astype("float32")
    # block-reduce to grid x grid
    yi = np.linspace(0, win - 1, grid).astype(int)
    xi = np.linspace(0, win - 1, grid).astype(int)
    small = region[np.ix_(yi, xi)]
    hi = np.percentile(small[small > 0], 99) if (small > 0).any() else 1.0
    small = np.clip(small, 0, hi)
    canopy_frac = float((small >= 3.0).mean())
    out = {
        "rows": grid, "cols": grid,
        "zmin": float(small.min()), "zmax": float(small.max()),
        "data": [[round(float(v), 2) for v in row] for row in small],
        "label": aois[aoi_name].label,
        "canopy_fraction": round(canopy_frac, 2),
    }
    (DATA / "heightmap.json").write_text(json.dumps(out))
    print(f"  heightmap {grid}x{grid} (mine/forest boundary)  canopy={canopy_frac:.0%} "
          f"zmax={out['zmax']:.1f}m -> heightmap.json")


def export_sites():
    aois = load_aois()
    sites = []
    for name, a in aois.items():
        w, s, e, n = a.bbox
        sites.append({
            "name": name, "label": a.label, "track": a.track,
            "lon": round((w + e) / 2, 4), "lat": round((s + n) / 2, 4),
            "mineral": a.raw.get("mineral", ""),
        })
    (DATA / "sites.json").write_text(json.dumps(sites, indent=2))
    print(f"  {len(sites)} sites -> sites.json")


if __name__ == "__main__":
    DATA.mkdir(parents=True, exist_ok=True)
    export_sites()
    export_heightmap()
    print("3D web data done")
