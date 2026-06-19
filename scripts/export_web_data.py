"""Bundle pipeline artifacts into web/public for the React+TS front-end.

Copies signature figures -> web/public/assets, converts trajectory CSVs and the
label-efficiency JSON -> web/public/data, and writes a manifest the app reads.
Re-run any time after producing new artifacts.
"""

from __future__ import annotations

import glob
import json
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web" / "public"
ASSETS = WEB / "assets"
DATA = WEB / "data"


def copy_assets() -> list[str]:
    ASSETS.mkdir(parents=True, exist_ok=True)
    wanted = [
        "figs/alcoa_huntly_s2_rgb.png",
        "figs/alcoa_huntly_weak_label.png",
        "figs/alcoa_huntly_rehab_footprint.png",
        "figs/alcoa_huntly_bridge_structure_vs_greenness.png",
        "figs/alcoa_huntly_bridge_chm_1m.png",
        "figs/alcoa_huntly_bridge_canopyfrac_10m.png",
        "figs/alcoa_huntly_bridge_s2_ndvi_10m.png",
        "outputs/seg_3site_sample.png",
        "outputs/le_3seed/label_efficiency.png",
        "outputs/loco/transfer_matrix.png",
    ]
    copied = []
    for rel in wanted:
        src = ROOT / rel
        if src.exists():
            shutil.copy(src, ASSETS / src.name)
            copied.append(src.name)
    return copied


def trajectories() -> dict:
    DATA.mkdir(parents=True, exist_ok=True)
    out = {}
    for csv in glob.glob(str(ROOT / "data" / "processed" / "s2" / "*" / "*trajectory.csv")):
        df = pd.read_csv(csv)
        tcol = df.columns[0]
        df[tcol] = pd.to_datetime(df[tcol]).dt.year
        key = Path(csv).stem
        out[key] = df.rename(columns={tcol: "year"}).to_dict(orient="records")
    return out


def transfer_matrix() -> dict | None:
    for cand in ["outputs/loco3/transfer_matrix.json", "outputs/loco/transfer_matrix.json"]:
        p = ROOT / cand
        if p.exists():
            return json.loads(p.read_text())
    return None


def label_efficiency() -> list | None:
    for cand in ["outputs/le_3seed/label_efficiency.json", "outputs/seg_3site/label_efficiency.json"]:
        p = ROOT / cand
        if p.exists():
            return json.loads(p.read_text())
    return None


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    assets = copy_assets()
    traj = trajectories()
    (DATA / "trajectories.json").write_text(json.dumps(traj, indent=2))
    le = label_efficiency()
    if le is not None:
        (DATA / "label_efficiency.json").write_text(json.dumps(le, indent=2))
    tm = transfer_matrix()
    if tm is not None:
        (DATA / "transfer_matrix.json").write_text(json.dumps(tm, indent=2))
    val = ROOT / "outputs" / "validation_worldcover.json"
    if val.exists():
        (DATA / "validation.json").write_text(val.read_text())
    st = ROOT / "outputs" / "selftrain" / "self_training.json"
    if st.exists():
        (DATA / "self_training.json").write_text(st.read_text())
    manifest = {
        "assets": assets,
        "trajectories": sorted(traj.keys()),
        "has_label_efficiency": le is not None,
        "has_transfer_matrix": tm is not None,
    }
    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"web data -> {WEB}")
    print(f"  assets: {len(assets)}  trajectories: {len(traj)}  label_eff: {le is not None}")


if __name__ == "__main__":
    main()
