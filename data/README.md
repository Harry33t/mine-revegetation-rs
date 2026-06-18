# Data menu — mine-revegetation-rs

All open. Imagery into `data/raw/`, derived labels/tiles under `data/processed/`.

| Dataset | Use | Source |
|---|---|---|
| Existing multi-city drone / aerial RGB+NIR | Reuse weak-label pipeline; cm-scale | (local) |
| Sentinel-2 / Landsat time series over an Australian rehab-mine AOI | Restoration trajectory + satellite scale | Digital Earth Australia (DEA) / Google Earth Engine |
| PlanetScope / Dove 3 m (optional) | Mid-scale bridge | as needed |
| Public mine-rehab drone imagery (optional) | Fine-scale supplement (scarce) | search |

> Weak labels: CHM (DSM − DEM → remove buildings → greenness gate) + spectral indices
> (NDVI / ExG) → pseudo-labels with self-training. There is no public *labelled* mine-rehab
> imagery — the scarcity is the motivation for the weak / self-supervised approach.

Scope discipline: MVP = segmentation + label-efficiency + scale bridge + trajectory.
Hyperspectral and species-level classification are stretch goals.
