# Weak-Supervised Revegetation Monitoring for Mine Rehabilitation

Label-efficient remote-sensing monitoring of mine-site revegetation: semantic
segmentation of vegetation classes from Sentinel-2 over rehabilitating Australian
mine sites, with multi-temporal restoration trajectories, a fine-scale
structure comparison, cross-site transfer, and per-pixel uncertainty.

**Guanxiong Huang** · Northwest A&F University · harry.huang@nwafu.edu.cn

**Live demo:** https://harry33t.github.io/mine-revegetation-rs/

## Motivation

Mine-site rehabilitation has to be monitored for years to decades, but the standard
evidence base — manually surveyed field plots — does not scale, is expensive, and is
site-specific. There is also no public *labelled* remote-sensing dataset for
revegetation, so conventional supervised learning cannot be applied directly. This
project treats that label scarcity as the starting point: vegetation classes
(bare, water, senescent, vegetation) are learned from **weak, automatically generated
labels** (canopy-height thresholds and spectral indices) rather than hand annotations,
and the labels are then improved by **self-training** on unlabelled imagery.

Around this core it addresses four recurring gaps in the rehabilitation-monitoring
literature: (i) labels are scarce → a label-efficiency analysis; (ii) models do not
transfer between sites → a leave-one-site-out transfer matrix; (iii) greenness is not
structure → a 1 m canopy-height vs. satellite-NDVI comparison; and (iv) predictions
lack uncertainty → MC-dropout maps that flag where field checks would be most useful.
Because no ground-truth labels exist, the weak labels are checked against an
independent product (ESA WorldCover).

## Method

- **Weak labels.** Canopy-height model (DSM − DEM) thresholds + spectral indices
  (NDVI, NDWI, ExG) → per-pixel pseudo-labels, no manual annotation.
- **Segmentation.** SegFormer-B0 with a 4-channel (RGB + NIR) input; U-Net baseline.
- **Self-training.** Confidence-thresholded pseudo-labels on unlabelled tiles are
  added to the labelled set and the model retrained.
- **Uncertainty.** MC-dropout predictive entropy at inference.
- **Cross-site transfer.** Leave-one-site-out training and evaluation.
- **Structure bridge.** Meta 1 m global canopy height aggregated to the 10 m grid,
  compared with Sentinel-2 NDVI.
- **Trajectory.** Annual Sentinel-2 composites over each site's baseline-cleared
  footprint, with a standardised recovery fraction.

## Results (real data)

| Result | Value |
|---|---|
| Label efficiency (3 sites, mIoU) | 0.43 (5%) → 0.48 (10%) → 0.59 (25%) → **0.78** (100%) |
| Self-training gain at 5% / 10% / 25% labels | **+0.036** / +0.024 / −0.005 (saturates as labels grow) |
| Cross-site transfer (within-site mIoU, diagonal) | 0.54 / 0.76 / 0.64; cross-site lower |
| External validation vs ESA WorldCover (Cohen's κ) | **0.745** (96.7% agreement, 4.0 M pixels, 3 sites) |
| Restoration trajectory, Alcoa cleared footprint | mean NDVI 0.182 → 0.309 (2019–2024) |

Honesty notes: the κ validates the *weak labels* against an independent product, not
model predictions against human ground truth; cross-site transfer is reported precisely
because it is *imperfect*; and self-training helps only at low label budgets, as
expected for semi-supervised learning.

## Study sites

| Site | Commodity | State |
|---|---|---|
| Alcoa Huntly | bauxite (jarrah forest) | WA |
| Ranger | uranium | NT |
| Mt Owen | coal | NSW |
| Iluka Eneabba | mineral sands | WA |

## Data (all open)

- **Sentinel-2** surface reflectance — Microsoft Planetary Computer (no account needed).
- **Meta** global 1 m canopy height (Tolan et al.) — AWS Open Data.
- **ESA WorldCover** 10 m land cover — validation reference.
- **LINZ** open aerial + 1 m LiDAR (New Zealand) — used by the reused weak-label pipeline.

## Reproduce

Web demo:

```bash
cd web && npm install && npm run dev      # or open the live link above
```

Python pipeline (CPU for data/labels/trajectory; GPU for training):

```bash
pip install -e ".[ml]"
python scripts/run_satellite_demo.py   --aoi alcoa_huntly --start 2019-01-01 --end 2024-12-31
python scripts/run_rehab_trajectory.py --aoi alcoa_huntly
python scripts/run_bridge_mine.py      --aoi alcoa_huntly
python scripts/validate_worldcover.py  --aois alcoa_huntly --aois ranger --aois mt_owen
python scripts/export_training_tiles.py --aoi alcoa_huntly                 # then, on a GPU:
python -m reveg.models.train --mode label_efficiency --sites data/processed/tiles/alcoa_huntly
```

## Stack

PyTorch · SegFormer (transformers) · segmentation-models-pytorch · rasterio / rioxarray /
odc-stac · React + TypeScript (web demo).

## Author

**Guanxiong Huang** — Northwest A&F University — harry.huang@nwafu.edu.cn

## Status

Work in progress. Reported accuracy is against weak labels and an independent land-cover
product; a manually labelled validation set is the natural next step.
