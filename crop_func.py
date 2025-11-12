"""Utilities for cropping Sentinel Hub downloads into GeoTIFFs."""
from __future__ import annotations

from pathlib import Path

import rasterio
from rasterio.windows import from_bounds as window_from_bounds

from download_func import DOWNLOAD_DIR, download_scene

CROP_DIR = DOWNLOAD_DIR / "crops"


def _resolve_crop_path(output_name: str | Path) -> Path:
    target = CROP_DIR / Path(output_name)
    if target.suffix.lower() not in {".tif", ".tiff"}:
        target = target.with_suffix(".tif")
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def crop_scene_to_tif(lat: float, lon: float, lat_degree: float, lon_degree: float, output_name: str) -> Path:
    """
    Ensure the specified AOI is downloaded and return a GeoTIFF crop suitable for rasterio.
    """
    source_path = Path(download_scene(lat, lon, lat_degree, lon_degree, output_name))
    crop_path = _resolve_crop_path(output_name)

    west, south = lon - lon_degree, lat - lat_degree
    east, north = lon + lon_degree, lat + lat_degree

    with rasterio.open(source_path) as src:
        window = window_from_bounds(west, south, east, north, src.transform)
        window = window.round_offsets().round_lengths()
        data = src.read(window=window)
        mask = src.dataset_mask(window=window)
        profile = src.profile.copy()
        profile.update(
            height=int(window.height),
            width=int(window.width),
            transform=src.window_transform(window),
        )
        tags = src.tags()

    with rasterio.open(crop_path, "w", **profile) as dst:
        dst.write(data)
        dst.write_mask(mask)
        dst.update_tags(
            **tags,
            lat=str(lat),
            lon=str(lon),
            lat_degree=str(lat_degree),
            lon_degree=str(lon_degree),
        )

    return crop_path


__all__ = ["crop_scene_to_tif"]
