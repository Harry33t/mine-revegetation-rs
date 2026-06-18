"""Stratified candidate-tile sampling for the M3 manual-validation pilot.

Strata = city x canopy-density bin x building-removed bin x slope bin (+ edge flag).
Sampling is deterministic (seeded), balanced across strata via round-robin, and
front-loads the priority tile types that carry the most label-quality risk.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# (label, lo_inclusive, hi_exclusive) on canopy_pct
CANOPY_BINS = [("low", 0, 5), ("sparse", 5, 20), ("medium", 20, 40), ("high", 40, 60), ("park", 60, 1e9)]


def _canopy_bin(pct: float) -> str:
    for label, lo, hi in CANOPY_BINS:
        if lo <= pct < hi:
            return label
    return "park"


def _building_bin(ratio: float) -> str:
    if ratio <= 0:
        return "none"
    return "low" if ratio < 0.05 else "high"


def assign_strata(stats: pd.DataFrame) -> pd.DataFrame:
    df = stats.copy()
    # 1 px == 1 m2 at the CHM grid, so removed ratio = removed_area_m2 / valid_px.
    df["building_removed_ratio"] = np.where(
        df["valid_px"] > 0, df.get("removed_area_m2", 0) / df["valid_px"], 0.0
    )
    df["canopy_bin"] = df["canopy_pct"].apply(_canopy_bin)
    df["building_bin"] = df["building_removed_ratio"].apply(_building_bin)
    if "slope_bin" not in df.columns:
        df["slope_bin"] = "unknown"
    full = df["valid_px"].max() if len(df) else 0
    df["edge_flag"] = df["valid_px"] < 0.9 * full if full else False
    if "city" not in df.columns:
        df["city"] = "unknown"
    df["stratum"] = (
        df["city"].astype(str) + "|" + df["canopy_bin"] + "|" + df["building_bin"] + "|" + df["slope_bin"]
    )
    return df


def _priority(row) -> int:
    city = str(row.get("city", "")).lower()
    cb, bb, sb = row["canopy_bin"], row["building_bin"], row["slope_bin"]
    if city == "wellington" and sb == "steep" and cb in ("medium", "high", "park"):
        return 0  # Wellington steep + canopy = the headline risk
    if city == "auckland" and bb == "high":
        return 1
    if city == "christchurch" and cb == "park":
        return 2
    if city == "tauranga" and cb in ("high", "park"):
        return 3
    if cb == "low":
        return 4  # low-canopy hard negatives, all cities
    return 5


def sample(stats: pd.DataFrame, n_samples: int, seed: int = 42) -> pd.DataFrame:
    df = assign_strata(stats)
    df = df.copy()
    df["priority"] = df.apply(_priority, axis=1)

    rng = np.random.default_rng(seed)
    # per-stratum seeded ordering
    buckets: dict[str, list[int]] = {}
    for stratum, idx in df.groupby("stratum").groups.items():
        order = list(idx)
        rng.shuffle(order)
        buckets[stratum] = order
    # strata ordered by (min priority within, name) for deterministic, risk-first round-robin
    strata_order = sorted(
        buckets, key=lambda s: (int(df.loc[buckets[s], "priority"].min()), s)
    )

    picked: list[int] = []
    while len(picked) < n_samples and any(buckets[s] for s in strata_order):
        for s in strata_order:
            if buckets[s]:
                picked.append(buckets[s].pop(0))
                if len(picked) >= n_samples:
                    break

    out = df.loc[picked].copy()
    out.insert(0, "sample_id", [f"m3_{i:04d}" for i in range(len(out))])
    return out.reset_index(drop=True)
