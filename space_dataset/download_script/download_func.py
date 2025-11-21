"""Reusable helper for Sentinel Hub downloads."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
import math
import hashlib
import dotenv
dotenv.load_dotenv()

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from sentinelhub import (
    SHConfig,
    BBox,
    CRS,
    DataCollection,
    MimeType,
    SentinelHubRequest,
    bbox_to_dimensions,
    MosaickingOrder,
)

TimeInterval = Tuple[str, str]

DOWNLOAD_DIR = Path("downloads")
CACHE_DIR = DOWNLOAD_DIR / "sentinel_cache"
DEFAULT_RESOLUTION = 10  # meters per pixel
MAX_TILE_SIZE = 2000
DEFAULT_L2A_WINDOW: TimeInterval = ("2024-04-01", "2024-07-31")

EVALSCRIPT_RGB = """
//VERSION=3
function setup() {
  return {
    input: ["B04", "B03", "B02", "SCL", "dataMask"],
    output: { bands: 5, sampleType: "FLOAT32" }
  };
}
function evaluatePixel(s) {
  return [s.B04, s.B03, s.B02, s.SCL, s.dataMask];
}
"""

@dataclass
class DownloadResult:
    """Lightweight container for download data and metadata."""

    data: np.ndarray
    mask: np.ndarray
    collection: str


def _build_config() -> SHConfig:
    cfg = SHConfig()
    cfg.sh_client_id = os.getenv("SH_CLIENT_ID", cfg.sh_client_id)
    cfg.sh_client_secret = os.getenv("SH_CLIENT_SECRET", cfg.sh_client_secret)
    instance_id = os.getenv("SH_INSTANCE_ID")
    if instance_id:
        cfg.instance_id = instance_id
    if not cfg.sh_client_id or not cfg.sh_client_secret:
        raise RuntimeError("Sentinel Hub credentials are missing. Set SH_CLIENT_ID/SH_CLIENT_SECRET.")
    return cfg


def _ensure_dirs() -> None:
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _request_data(
    *,
    cfg: SHConfig,
    bbox: BBox,
    size: Tuple[int, int],
    evalscript: str,
    time_interval: TimeInterval,
    collection: DataCollection,
) -> np.ndarray:
    request = SentinelHubRequest(
        evalscript=evalscript,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=collection,
                time_interval=time_interval,
                mosaicking_order=MosaickingOrder.LEAST_CC,
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=bbox,
        size=size,
        data_folder=str(CACHE_DIR),
        config=cfg,
    )
    return request.get_data()[0]


def _tile_cache_path(bbox: BBox, size: Tuple[int, int]) -> Path:
    key_parts = (
        f"{bbox.lower_left[0]:.7f}",
        f"{bbox.lower_left[1]:.7f}",
        f"{bbox.upper_right[0]:.7f}",
        f"{bbox.upper_right[1]:.7f}",
        str(int(size[0])),
        str(int(size[1])),
        DEFAULT_L2A_WINDOW[0],
        DEFAULT_L2A_WINDOW[1],
        str(DEFAULT_RESOLUTION),
    )
    digest = hashlib.sha1("_".join(key_parts).encode("utf-8")).hexdigest()
    return CACHE_DIR / f"tile_{digest}.npz"


def _load_cached_tile(cache_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with np.load(cache_path) as payload:
        data = payload["data"]
        mask = payload["mask"].astype(bool)
    return data, mask


def _store_cached_tile(cache_path: Path, data: np.ndarray, mask: np.ndarray) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, data=data, mask=mask.astype(np.uint8))


def _request_l2a_chunk(cfg: SHConfig, bbox: BBox, size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    data = _request_data(
        cfg=cfg,
        bbox=bbox,
        size=size,
        evalscript=EVALSCRIPT_RGB,
        time_interval=DEFAULT_L2A_WINDOW,
        collection=DataCollection.SENTINEL2_L2A,
    )
    mask = data[..., -1] > 0.5
    return data, mask


def _load_or_request_tile(cfg: SHConfig, bbox: BBox, size: Tuple[int, int], cache: bool) -> Tuple[np.ndarray, np.ndarray]:
    if cache:
        cache_path = _tile_cache_path(bbox, size)
        if cache_path.exists():
            return _load_cached_tile(cache_path)
    data, mask = _request_l2a_chunk(cfg, bbox, size)
    if cache:
        _store_cached_tile(cache_path, data, mask)
    return data, mask


def _fetch_l2a(cfg: SHConfig, bbox: BBox, size: Tuple[int, int]) -> DownloadResult:
    width, height = map(int, size)
    if width <= MAX_TILE_SIZE and height <= MAX_TILE_SIZE:
        data, mask = _load_or_request_tile(cfg, bbox, size, False)
        if not mask.any():
            raise RuntimeError("No valid pixels returned for L2A within the requested window.")
        return DownloadResult(data=data, mask=mask, collection="SENTINEL2_L2A")

    return _fetch_l2a_tiled(cfg, bbox, (width, height))


def _fetch_l2a_tiled(cfg: SHConfig, bbox: BBox, size: Tuple[int, int]) -> DownloadResult:
    total_width, total_height = size
    cols = max(1, math.ceil(total_width / MAX_TILE_SIZE))
    rows = max(1, math.ceil(total_height / MAX_TILE_SIZE))

    lon_min, lat_min = bbox.lower_left
    lon_max, lat_max = bbox.upper_right
    lon_edges = np.linspace(lon_min, lon_max, cols + 1)
    lat_edges = np.linspace(lat_max, lat_min, rows + 1)  # north to south

    mosaic_data: np.ndarray | None = None
    mosaic_mask: np.ndarray | None = None
    row_offset = 0

    for r in range(rows):
        tile_lat_top = float(lat_edges[r])
        tile_lat_bottom = float(lat_edges[r + 1])
        col_offset = 0
        row_height = None

        for c in range(cols):
            tile_lon_left = float(lon_edges[c])
            tile_lon_right = float(lon_edges[c + 1])
            tile_bbox = BBox(((tile_lon_left, tile_lat_bottom), (tile_lon_right, tile_lat_top)), crs=CRS.WGS84)
            tile_width, tile_height = map(int, bbox_to_dimensions(tile_bbox, resolution=DEFAULT_RESOLUTION))
            data, mask = _load_or_request_tile(cfg, tile_bbox, (tile_width, tile_height), True)

            remaining_width = total_width - col_offset
            remaining_height = total_height - row_offset
            tile_width = min(tile_width, remaining_width)
            tile_height = min(tile_height, remaining_height)
            data = data[:tile_height, :tile_width, :]
            mask = mask[:tile_height, :tile_width]

            if mosaic_data is None:
                bands = data.shape[-1]
                mosaic_data = np.zeros((total_height, total_width, bands), dtype=data.dtype)
                mosaic_mask = np.zeros((total_height, total_width), dtype=bool)

            mosaic_data[row_offset : row_offset + tile_height, col_offset : col_offset + tile_width, :] = data
            mosaic_mask[row_offset : row_offset + tile_height, col_offset : col_offset + tile_width] = mask

            if row_height is None:
                row_height = tile_height
            col_offset += tile_width

        row_offset += row_height or 0

    if mosaic_data is None or mosaic_mask is None or not mosaic_mask.any():
        raise RuntimeError("No valid pixels returned across tiled L2A requests.")

    return DownloadResult(data=mosaic_data, mask=mosaic_mask, collection="SENTINEL2_L2A")


def _resolve_output_path(output_name: str | Path) -> Path:
    candidate = DOWNLOAD_DIR / Path(output_name)
    if candidate.suffix.lower() not in {".tif", ".tiff"}:
        candidate = candidate.with_suffix(".tif")
    return candidate


def _save_product(
    *,
    result: DownloadResult,
    bbox: BBox,
    lat: float,
    lon: float,
    lat_degree: float,
    lon_degree: float,
    output_name: str | Path,
) -> Path:
    output_path = _resolve_output_path(output_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rgb_data = result.data[..., :3]  # Sentinel-2 bands B04/B03/B02 (R,G,B)
    width = int(rgb_data.shape[1])
    height = int(rgb_data.shape[0])
    transform = from_bounds(
        bbox.lower_left[0],
        bbox.lower_left[1],
        bbox.upper_right[0],
        bbox.upper_right[1],
        width,
        height,
    )
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": rgb_data.shape[-1],
        "dtype": rgb_data.dtype,
        "crs": "EPSG:4326",
        "transform": transform,
        "compress": "deflate",
    }
    bands_first = np.moveaxis(rgb_data, -1, 0)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(bands_first)
        dst.write_mask(result.mask.astype(np.uint8) * 255)
        dst.update_tags(
            collection=result.collection,
            lat=str(lat),
            lon=str(lon),
            lat_degree=str(lat_degree),
            lon_degree=str(lon_degree),
        )
    return output_path


def download_scene(lat: float, lon: float, lat_degree: float, lon_degree: float, output_name: str) -> Path:
    """Fetch Sentinel-2 data for the requested AOI and store it under downloads/."""
    if lat_degree <= 0 or lon_degree <= 0:
        raise ValueError("lat_degree and lon_degree must be greater than zero.")
    _ensure_dirs()
    existing_path = _resolve_output_path(output_name)
    if existing_path.exists():
        return existing_path
    cfg = _build_config()
    bbox = BBox(((lon - lon_degree, lat - lat_degree), (lon + lon_degree, lat + lat_degree)), crs=CRS.WGS84)
    size = bbox_to_dimensions(bbox, resolution=DEFAULT_RESOLUTION)
    result = _fetch_l2a(cfg, bbox, size)
    return _save_product(
        result=result,
        bbox=bbox,
        lat=lat,
        lon=lon,
        lat_degree=lat_degree,
        lon_degree=lon_degree,
        output_name=output_name,
    )


__all__ = ["download_scene", "DOWNLOAD_DIR"]
