"""Project path resolution.

All directory locations derive from the repo layout and the `paths:` block of
configs/processing.yaml. This module reads only that small block directly (no
dependency on config.py) so it can be imported anywhere without cycles.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml


def project_root() -> Path:
    """The `nz-urban-canopy/` directory (this file is src/nz_canopy/utils/paths.py)."""
    return Path(__file__).resolve().parents[3]


def config_dir() -> Path:
    return project_root() / "configs"


@lru_cache(maxsize=1)
def _path_cfg() -> dict:
    with open(config_dir() / "processing.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)["paths"]


def data_root() -> Path:
    """Resolved absolute path to the data root (processing.yaml `data_root`, e.g. ../data)."""
    return (project_root() / _path_cfg()["data_root"]).resolve()


def raw_root() -> Path:
    return data_root() / _path_cfg()["raw_dir"]


def outputs_root() -> Path:
    return project_root() / _path_cfg().get("outputs_dir", "outputs")


def city_raw_dir(city: str) -> Path:
    return raw_root() / city


def layer_dir(city: str, layer: str) -> Path:
    """Directory for a layer under a city, e.g. layer_dir('christchurch', 'dsm').

    For aerial, pass 'aerial/<year>' or use the imagery reader which appends the year.
    """
    return city_raw_dir(city) / layer
