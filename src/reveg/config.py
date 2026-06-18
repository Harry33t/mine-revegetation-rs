"""AOI config loader (configs/aois.yaml). UTF-8 explicit (configs contain non-ASCII)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


def repo_root() -> Path:
    # src/reveg/config.py -> repo root is three parents up.
    return Path(__file__).resolve().parents[2]


def config_dir() -> Path:
    return repo_root() / "configs"


@dataclass(frozen=True)
class AOI:
    name: str
    track: str  # "satellite" | "bridge"
    label: str
    bbox: tuple[float, float, float, float]  # west, south, east, north (EPSG:4326)
    crs: str
    city: str | None = None  # nz_canopy city key, for bridge AOIs
    raw: dict = field(repr=False, default_factory=dict)

    @classmethod
    def from_dict(cls, name: str, d: dict) -> "AOI":
        b = d["bbox"]
        return cls(
            name=name,
            track=d.get("track", "satellite"),
            label=d.get("label", name),
            bbox=(float(b[0]), float(b[1]), float(b[2]), float(b[3])),
            crs=d.get("crs", "EPSG:4326"),
            city=d.get("city"),
            raw=d,
        )


def load_aois(config_dir_override: str | Path | None = None) -> dict[str, AOI]:
    cdir = Path(config_dir_override) if config_dir_override is not None else config_dir()
    with open(cdir / "aois.yaml", encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    return {name: AOI.from_dict(name, d) for name, d in doc["aois"].items()}


def get_aoi(name: str, config_dir_override: str | Path | None = None) -> AOI:
    aois = load_aois(config_dir_override)
    if name not in aois:
        raise KeyError(f"Unknown AOI '{name}'. Known: {sorted(aois)}")
    return aois[name]
