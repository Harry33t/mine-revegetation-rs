"""Typed loaders for the three YAML configs.

All reads are UTF-8 explicit: the configs contain non-ASCII (e.g. km²) and the
Windows default encoding (GBK) cannot decode them. `config_dir` is overridable
so tests can point at synthetic config files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .utils import paths


def _load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_dir(config_dir: str | Path | None) -> Path:
    return Path(config_dir) if config_dir is not None else paths.config_dir()


@dataclass(frozen=True)
class SpatialCV:
    block_size_m: float
    buffer_tiles: int


@dataclass(frozen=True)
class ProcessingConfig:
    target_crs: str
    working_pixel_size_m: float
    tile_size_px: int
    tile_size_m: float
    spatial_cv: SpatialCV
    resampling: dict[str, str]
    chm: dict[str, Any]
    normalization: dict[str, Any]
    random_seeds: list[int]
    deterministic: bool
    raw: dict = field(repr=False, default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "ProcessingConfig":
        cv = d["spatial_cv"]
        repro = d["reproducibility"]
        return cls(
            target_crs=d["target_crs"],
            working_pixel_size_m=float(d["working_pixel_size_m"]),
            tile_size_px=int(d["tile_size_px"]),
            tile_size_m=float(d["tile_size_m"]),
            spatial_cv=SpatialCV(float(cv["block_size_m"]), int(cv["buffer_tiles"])),
            resampling=dict(d["resampling"]),
            chm=dict(d["chm"]),
            normalization=dict(d["normalization"]),
            random_seeds=list(repro["random_seeds"]),
            deterministic=bool(repro["deterministic"]),
            raw=d,
        )


@dataclass(frozen=True)
class CityConfig:
    name: str
    city_name: str
    region: str
    target_crs: str
    status: str
    raw_dir: str
    primary_aerial_year: str | None
    available_aerial_years: list[str]
    lidar_survey_id: str | None
    test_tile_ids: list[str]
    sample_strategy: str
    hard_negative_categories: list[str]
    output_dir: str
    raw: dict = field(repr=False, default_factory=dict)

    @classmethod
    def from_dict(cls, name: str, d: dict) -> "CityConfig":
        return cls(
            name=name,
            city_name=d.get("city_name", name),
            region=d.get("region", ""),
            target_crs=d.get("target_crs", "EPSG:2193"),
            status=d.get("status", "unknown"),
            raw_dir=d.get("raw_dir", name),
            primary_aerial_year=d.get("primary_aerial_year"),
            available_aerial_years=list(d.get("available_aerial_years") or []),
            lidar_survey_id=d.get("lidar_survey_id"),
            test_tile_ids=list(d.get("test_tile_ids") or []),
            sample_strategy=d.get("sample_strategy", ""),
            hard_negative_categories=list(d.get("hard_negative_categories") or []),
            output_dir=d.get("output_dir", name),
            raw=d,
        )


@dataclass(frozen=True)
class DataSourcesConfig:
    linz_resource_zip_prefix: str
    resource_subdirs: dict
    data_layout: dict
    expected_aerial_years: dict
    raw: dict = field(repr=False, default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "DataSourcesConfig":
        return cls(
            linz_resource_zip_prefix=d.get("linz_resource_zip_prefix", ""),
            resource_subdirs=dict(d.get("resource_subdirs", {})),
            data_layout=dict(d.get("data_layout", {})),
            expected_aerial_years=dict(d.get("expected_aerial_years", {})),
            raw=d,
        )


def load_processing(config_dir: str | Path | None = None) -> ProcessingConfig:
    return ProcessingConfig.from_dict(_load_yaml(_resolve_dir(config_dir) / "processing.yaml"))


def load_cities(config_dir: str | Path | None = None) -> dict[str, CityConfig]:
    raw = _load_yaml(_resolve_dir(config_dir) / "cities.yaml")
    return {name: CityConfig.from_dict(name, d) for name, d in raw.items()}


def get_city(name: str, config_dir: str | Path | None = None) -> CityConfig:
    cities = load_cities(config_dir)
    key = name.lower()
    if key not in cities:
        raise KeyError(f"Unknown city '{name}'. Known: {sorted(cities)}")
    return cities[key]


def load_data_sources(config_dir: str | Path | None = None) -> DataSourcesConfig:
    return DataSourcesConfig.from_dict(_load_yaml(_resolve_dir(config_dir) / "data_sources.yaml"))
