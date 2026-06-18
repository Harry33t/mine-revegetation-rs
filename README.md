# mine-revegetation-rs

**Weak-Supervised Revegetation Monitoring for Mine Rehabilitation**

Label-efficient (weak / self-supervised) remote-sensing monitoring of mine-site
revegetation: semantic segmentation of vegetation classes (native / exotic / bare / canopy)
from drone and satellite imagery, with a **drone→satellite scale bridge**, **multi-temporal
restoration trajectories**, and **per-pixel uncertainty** to flag where field checks are needed.
Targets the recurring rehabilitation-monitoring gaps: manual field plots that don't scale,
cm-drone↔satellite scale mismatch, and missing uncertainty.

## Highlights
- **Weak-label semantic segmentation** (CHM + spectral pseudo-labels → self-training).
- **Label-efficiency curve**: accuracy at 1 / 5 / 10 / 100% labels.
- **Drone → satellite scale bridge** with spatial cross-validation across sites.
- **Multi-temporal restoration trajectory** (annual cover / structure vs a standardised baseline).
- **Uncertainty-targeted active learning** map ("which pixels most need fieldwork").

## Data (open)
- Drone / aerial RGB+NIR · Sentinel-2 / Landsat time series via Digital Earth Australia (DEA) · optional PlanetScope.

## Stack
PyTorch · SegFormer / U-Net · Google Earth Engine / Digital Earth Australia · rasterio.

## Status
Work in progress. See project board for the build schedule.

## License
TBD.
