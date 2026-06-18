"""Class scheme + weak-label thresholds for revegetation segmentation.

Design note (honesty): we deliberately do NOT classify native-vs-exotic species.
That is a biogeographic-provenance property, not a spectral one, and is not
separable from RGB+NIR or Sentinel-2 broadband reflectance without hyperspectral
or species-level ground truth. Classes here are STRUCTURAL + FUNCTIONAL: what a
height model and broadband indices can actually defend.

  - Structure (canopy / shrub / low veg) comes from CHM height tiers -> only
    available on the fine-scale aerial+LiDAR (NZ bridge) track.
  - On Sentinel-2 (10 m, no CHM) only the functional split is defensible, so the
    spectral-only labeller collapses woody/low structure into one VEGETATION class.
"""

from __future__ import annotations

from dataclasses import dataclass

# Canonical class ids (uint8). 0 is reserved for invalid/nodata.
NODATA = 0
BARE = 1
WATER = 2
LOW_VEG = 3      # grass / herbaceous (CHM < shrub_min_m)
SHRUB = 4        # mid structure (shrub_min_m .. canopy_min_m)
CANOPY = 5       # tree canopy (>= canopy_min_m)
SENESCENT = 6    # senescent / stressed vegetation (some green, low vigour)
VEGETATION = 7   # generic vegetation, used when no height is available (S2 track)

CLASS_NAMES = {
    NODATA: "nodata",
    BARE: "bare",
    WATER: "water",
    LOW_VEG: "low_veg",
    SHRUB: "shrub",
    CANOPY: "canopy",
    SENESCENT: "senescent",
    VEGETATION: "vegetation",
}

# Display colours (RGB 0-255) for quicklooks/overlays.
CLASS_COLORS = {
    NODATA: (0, 0, 0),
    BARE: (170, 120, 70),
    WATER: (40, 90, 200),
    LOW_VEG: (160, 220, 110),
    SHRUB: (70, 160, 70),
    CANOPY: (20, 90, 30),
    SENESCENT: (210, 180, 60),
    VEGETATION: (90, 180, 90),
}


@dataclass(frozen=True)
class LabelSpec:
    """Thresholds for weak-label assignment. Defaults are sensible starting points;
    calibrate per AOI against a small hand-labelled set (garbage thresholds ->
    garbage labels propagate through self-training)."""

    # Spectral gates
    veg_ndvi: float = 0.45        # >= this NDVI -> vigorous vegetation
    senescent_ndvi: float = 0.20  # [senescent_ndvi, veg_ndvi) -> senescent/stressed
    water_ndwi: float = 0.10      # >= this NDWI (and not vigorous veg) -> water

    # Structural height tiers (metres, from CHM) — fine-scale track only
    shrub_min_m: float = 0.5      # >= this -> at least shrub
    canopy_min_m: float = 3.0     # >= this -> tree canopy

    # Cleaning
    min_area_m2: float = 4.0      # drop connected components smaller than this
