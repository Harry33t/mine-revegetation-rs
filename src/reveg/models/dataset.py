"""Torch Dataset over exported (S2 image, weak-label mask) tile pairs.

Reads the npy tiles written by scripts/export_training_tiles.py. Maps the spectral
weak-label ids to contiguous training indices (NODATA -> ignore_index 255).

Image = [R,G,B,NIR] reflectance float32; normalized to ~[0,1] (reflectance/10000).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from ..labels.classes import BARE, NODATA, SENESCENT, VEGETATION, WATER

# spectral_only weak-label ids -> contiguous training classes
LABEL_REMAP = {BARE: 0, WATER: 1, SENESCENT: 2, VEGETATION: 3}
TRAIN_CLASS_NAMES = ["bare", "water", "senescent", "vegetation"]
NUM_CLASSES = len(LABEL_REMAP)
IGNORE_INDEX = 255
REFLECTANCE_SCALE = 10000.0


def remap_mask(mask: np.ndarray) -> np.ndarray:
    """Weak-label ids -> training indices; NODATA and anything unmapped -> IGNORE_INDEX."""
    out = np.full(mask.shape, IGNORE_INDEX, dtype="uint8")
    for src, dst in LABEL_REMAP.items():
        out[mask == src] = dst
    return out


class TileDataset(Dataset):
    """Loads tiles listed in one or more manifests. `ids` restricts to a subset
    (e.g. a label-efficiency fraction or a LOCO train/test split)."""

    def __init__(self, tiles_dirs, *, ids: list[str] | None = None, augment: bool = False):
        if isinstance(tiles_dirs, (str, Path)):
            tiles_dirs = [tiles_dirs]
        self.records: list[tuple[Path, str]] = []
        for d in tiles_dirs:
            d = Path(d)
            man = pd.read_csv(d / "manifest.csv")
            for tid in man["id"]:
                if ids is None or tid in ids:
                    self.records.append((d, tid))
        self.augment = augment

    def __len__(self) -> int:
        return len(self.records)

    def all_ids(self) -> list[str]:
        return [tid for _, tid in self.records]

    def __getitem__(self, i):
        import torch

        d, tid = self.records[i]
        img = np.load(d / "images" / f"{tid}.npy").astype("float32") / REFLECTANCE_SCALE
        img = np.clip(img, 0.0, 1.0)
        mask = remap_mask(np.load(d / "masks" / f"{tid}.npy")).astype("int64")
        if self.augment:
            if np.random.rand() < 0.5:
                img = img[:, :, ::-1].copy(); mask = mask[:, ::-1].copy()
            if np.random.rand() < 0.5:
                img = img[:, ::-1, :].copy(); mask = mask[::-1, :].copy()
        return torch.from_numpy(img), torch.from_numpy(mask)
