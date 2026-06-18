"""Tile grid + spatial-block cross-validation split.

512 px x 0.3 m = 153.6 m square tiles over a city extent in EPSG:2193, grouped
into 1 km blocks. The CV split is by block (not by tile) to prevent spatial
autocorrelation leakage, with a >= 1-tile buffer between train and test blocks
(memo §2 RQ2).
"""

from __future__ import annotations

import math

import geopandas as gpd
import numpy as np
from shapely.geometry import box

# TODO M2: stratified sampling by canopy density / tile type (high-density urban /
#          suburban / park / industrial), per memo §14.4.
# TODO M4: finalize the buffered block-CV split used for the headline results.


def make_tile_grid(extent: tuple, crs, tile_size_m: float = 153.6) -> gpd.GeoDataFrame:
    """Build a regular tile grid over extent=(left, bottom, right, top)."""
    left, bottom, right, top = extent
    ncols = max(1, math.ceil((right - left) / tile_size_m))
    nrows = max(1, math.ceil((top - bottom) / tile_size_m))
    records = []
    for r in range(nrows):
        y1 = top - r * tile_size_m
        y0 = y1 - tile_size_m
        for c in range(ncols):
            x0 = left + c * tile_size_m
            x1 = x0 + tile_size_m
            records.append(
                {"tile_id": f"r{r:03d}c{c:03d}", "row": r, "col": c, "geometry": box(x0, y0, x1, y1)}
            )
    return gpd.GeoDataFrame(records, crs=crs)


def assign_blocks(grid: gpd.GeoDataFrame, block_size_m: float = 1000.0, tile_size_m: float = 153.6) -> gpd.GeoDataFrame:
    grid = grid.copy()
    grid["block_row"] = (grid["row"] * tile_size_m // block_size_m).astype(int)
    grid["block_col"] = (grid["col"] * tile_size_m // block_size_m).astype(int)
    grid["block_id"] = grid["block_row"].astype(str) + "_" + grid["block_col"].astype(str)
    return grid


def assign_cv_split(
    grid: gpd.GeoDataFrame,
    ratios: tuple[float, float, float] = (0.6, 0.2, 0.2),
    buffer_tiles: int = 1,
    seed: int = 42,
    tile_size_m: float = 153.6,
    block_size_m: float = 1000.0,
) -> gpd.GeoDataFrame:
    """Assign train/val/test by block, then demote train tiles within `buffer_tiles`
    (Chebyshev distance) of any test tile to 'buffer' so no train tile touches a test tile."""
    if "block_id" not in grid.columns:
        grid = assign_blocks(grid, block_size_m=block_size_m, tile_size_m=tile_size_m)
    else:
        grid = grid.copy()

    blocks = sorted(grid["block_id"].unique())
    rng = np.random.default_rng(seed)
    rng.shuffle(blocks)
    n = len(blocks)
    n_train = int(round(n * ratios[0]))
    n_val = int(round(n * ratios[1]))
    split_of = {}
    for i, b in enumerate(blocks):
        split_of[b] = "train" if i < n_train else ("val" if i < n_train + n_val else "test")
    grid["split"] = grid["block_id"].map(split_of)

    test_rc = set(zip(grid.loc[grid.split == "test", "row"], grid.loc[grid.split == "test", "col"]))
    if test_rc and buffer_tiles > 0:
        offsets = [
            (dr, dc)
            for dr in range(-buffer_tiles, buffer_tiles + 1)
            for dc in range(-buffer_tiles, buffer_tiles + 1)
            if not (dr == 0 and dc == 0)
        ]

        def near_test(row: int, col: int) -> bool:
            return any((row + dr, col + dc) in test_rc for dr, dc in offsets)

        demote = grid.apply(lambda t: t.split == "train" and near_test(int(t.row), int(t.col)), axis=1)
        grid.loc[demote, "split"] = "buffer"
    return grid


def to_geopackage(grid: gpd.GeoDataFrame, path, layer: str = "tiles") -> None:
    grid.to_file(path, layer=layer, driver="GPKG")
