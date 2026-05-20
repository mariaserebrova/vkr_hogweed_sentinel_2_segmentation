from pathlib import Path
from typing import Iterable

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window
from rasterio.warp import reproject


def discover_final_rasters(prepared_root: Path, run_only_dates: Iterable[str] | None = None):
    date_dirs = []
    for path in sorted(prepared_root.iterdir()):
        if not path.is_dir():
            continue
        if path.name.startswith(".") or path.name.startswith("_"):
            continue
        date_dirs.append(path)

    rasters = []
    for date_dir in date_dirs:
        date_name = date_dir.name
        if run_only_dates is not None and date_name not in run_only_dates:
            continue

        final_dir = date_dir / "final"
        if not final_dir.exists():
            continue

        tif_files = sorted(final_dir.glob("*_final_clip_12ch.tif"))
        if tif_files:
            rasters.append((date_name, tif_files[0]))

    if run_only_dates is not None:
        order = {date: i for i, date in enumerate(run_only_dates)}
        rasters = sorted(rasters, key=lambda item: order[item[0]])

    return rasters


def build_mask_to_reference(full_mask_path: Path, ref_raster_path: Path, out_mask_path: Path) -> Path:
    out_mask_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(ref_raster_path) as ref:
        ref_profile = ref.profile.copy()
        ref_transform = ref.transform
        ref_crs = ref.crs
        ref_height = ref.height
        ref_width = ref.width

    with rasterio.open(full_mask_path) as src:
        src_data = src.read(1)
        src_transform = src.transform
        src_crs = src.crs

    dst = np.zeros((ref_height, ref_width), dtype=np.uint8)

    reproject(
        source=(src_data > 0).astype(np.uint8),
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=ref_transform,
        dst_crs=ref_crs,
        resampling=Resampling.nearest,
    )

    ref_profile.update(dtype=rasterio.uint8, count=1, nodata=0)

    with rasterio.open(out_mask_path, "w", **ref_profile) as dst_file:
        dst_file.write(dst, 1)

    return out_mask_path


def raster_geom_signature(path: Path) -> dict:
    with rasterio.open(path) as src:
        return {
            "height": src.height,
            "width": src.width,
            "crs": str(src.crs),
            "transform": tuple(src.transform),
        }


def verify_geometry(reference_raster_path: Path, mask_path: Path) -> None:
    ref_sig = raster_geom_signature(reference_raster_path)
    mask_sig = raster_geom_signature(mask_path)

    if ref_sig["height"] != mask_sig["height"]:
        raise ValueError("Raster and mask heights differ.")
    if ref_sig["width"] != mask_sig["width"]:
        raise ValueError("Raster and mask widths differ.")
    if ref_sig["crs"] != mask_sig["crs"]:
        raise ValueError("Raster and mask CRS differ.")
    if ref_sig["transform"] != mask_sig["transform"]:
        raise ValueError("Raster and mask transforms differ.")


def read_binary_mask(mask_path: Path) -> np.ndarray:
    with rasterio.open(mask_path) as src:
        return (src.read(1) > 0).astype(np.uint8)


def load_valid_mask(image_path: Path) -> np.ndarray:
    with rasterio.open(image_path) as src:
        arr = src.read([1, 2, 3, 4]).astype(np.float32)
    return np.any(arr != 0, axis=0).astype(np.uint8)


def read_model_tile(
    image_path: Path,
    window: Window,
    keep_channel_idxs_1based: list[int],
    clip_min: float,
    clip_max: float,
    scale_div: float,
    nodata_value: float | None = None,
) -> np.ndarray:
    with rasterio.open(image_path) as src:
        tile = src.read(indexes=keep_channel_idxs_1based, window=window).astype(np.float32)

    if nodata_value is not None:
        tile[tile == nodata_value] = 0.0

    tile = np.nan_to_num(tile, nan=0.0, posinf=0.0, neginf=0.0)
    tile = np.clip(tile, clip_min, clip_max) / scale_div
    return tile.astype(np.float32)


def read_window_mask(mask_path: Path, window: Window) -> np.ndarray:
    with rasterio.open(mask_path) as src:
        mask = src.read(1, window=window).astype(np.float32)
    return (mask > 0).astype(np.float32)[None, ...]
