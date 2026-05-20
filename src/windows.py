from collections import Counter
from typing import Iterable

import numpy as np
import pandas as pd
from rasterio.windows import Window


def make_candidate_windows(
    *,
    height: int,
    width: int,
    tile_size: int,
    stride: int,
    min_valid_fraction: float,
    valid_mask: np.ndarray,
    full_mask: np.ndarray,
    grid_type: str,
):
    records = []

    for row in range(0, height - tile_size + 1, stride):
        for col in range(0, width - tile_size + 1, stride):
            valid_fraction = float(valid_mask[row:row + tile_size, col:col + tile_size].mean())
            if valid_fraction < min_valid_fraction:
                continue

            pos_px = int(full_mask[row:row + tile_size, col:col + tile_size].sum())
            records.append({
                "window": Window(col, row, tile_size, tile_size),
                "row": row,
                "col": col,
                "grid_type": grid_type,
                "valid_fraction": valid_fraction,
                "positive_pixel_count": pos_px,
                "is_positive": pos_px > 0,
                "x_center": col + tile_size / 2.0,
            })

    return records


def density_bin_name(pos_px: int, bins: list[int], names: list[str]) -> str:
    for idx in range(len(names)):
        if bins[idx] <= pos_px < bins[idx + 1]:
            return names[idx]
    return names[-1]


def build_x_bands(records: list[dict], n_bands: int):
    unique_x = sorted({record["x_center"] for record in records})
    if len(unique_x) < n_bands:
        raise ValueError(f"Unique x-centers={len(unique_x)} are fewer than n_bands={n_bands}.")

    split_x = np.array_split(np.array(unique_x, dtype=float), n_bands)
    x_to_band = {}

    for band_idx, arr in enumerate(split_x):
        for x in arr.tolist():
            x_to_band[float(x)] = band_idx

    return x_to_band, split_x


def band_stats(records: list[dict], x_to_band: dict[float, int], density_bins: list[int], density_names: list[str]) -> pd.DataFrame:
    rows = []
    n_bands = max(x_to_band.values()) + 1

    for band_idx in range(n_bands):
        band_records = [record for record in records if x_to_band[record["x_center"]] == band_idx]
        row = {
            "band_idx": band_idx,
            "total_windows": len(band_records),
        }

        for name in density_names:
            row[f"{name}_windows"] = sum(
                density_bin_name(record["positive_pixel_count"], density_bins, density_names) == name
                for record in band_records
            )

        row["medium_plus_windows"] = row["medium_windows"] + row["large_windows"] + row["very_large_windows"]
        row["large_plus_windows"] = row["large_windows"] + row["very_large_windows"]
        rows.append(row)

    return pd.DataFrame(rows)


def normalized_density_hist(records: list[dict], density_bins: list[int], density_names: list[str]) -> np.ndarray:
    counts = Counter(
        density_bin_name(record["positive_pixel_count"], density_bins, density_names)
        for record in records
    )
    values = np.array([counts.get(name, 0) for name in density_names], dtype=float)
    return values / max(values.sum(), 1.0)


def l1_hist_distance(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.abs(first - second).sum())


def choose_stratified_val_test_bands(
    *,
    stats_df: pd.DataFrame,
    records: list[dict],
    x_to_band: dict[float, int],
    density_bins: list[int],
    density_names: list[str],
    buffer_bands: int,
    min_medium_windows_per_split: int,
    min_large_windows_per_split: int,
):
    bands = stats_df["band_idx"].tolist()
    overall_hist = normalized_density_hist(records, density_bins, density_names)
    best = None

    for val_band in bands:
        for test_band in bands:
            if test_band == val_band:
                continue

            blocked = {val_band, test_band}
            for band_idx in range(val_band - buffer_bands, val_band + buffer_bands + 1):
                if band_idx in bands:
                    blocked.add(band_idx)
            for band_idx in range(test_band - buffer_bands, test_band + buffer_bands + 1):
                if band_idx in bands:
                    blocked.add(band_idx)

            train_bands = [band_idx for band_idx in bands if band_idx not in blocked]
            if not train_bands:
                continue

            val_records = [record for record in records if x_to_band[record["x_center"]] == val_band]
            test_records = [record for record in records if x_to_band[record["x_center"]] == test_band]
            train_records = [record for record in records if x_to_band[record["x_center"]] in train_bands]

            val_hist = normalized_density_hist(val_records, density_bins, density_names)
            test_hist = normalized_density_hist(test_records, density_bins, density_names)
            train_hist = normalized_density_hist(train_records, density_bins, density_names)

            val_row = stats_df.loc[stats_df["band_idx"] == val_band].iloc[0]
            test_row = stats_df.loc[stats_df["band_idx"] == test_band].iloc[0]

            if int(val_row["medium_plus_windows"]) < min_medium_windows_per_split:
                continue
            if int(test_row["medium_plus_windows"]) < min_medium_windows_per_split:
                continue
            if int(val_row["large_plus_windows"]) < min_large_windows_per_split:
                continue
            if int(test_row["large_plus_windows"]) < min_large_windows_per_split:
                continue

            train_large_plus = int(stats_df.loc[stats_df["band_idx"].isin(train_bands), "large_plus_windows"].sum())
            train_medium_plus = int(stats_df.loc[stats_df["band_idx"].isin(train_bands), "medium_plus_windows"].sum())

            if train_medium_plus < max(2 * min_medium_windows_per_split, 8):
                continue
            if train_large_plus < max(2 * min_large_windows_per_split, 8):
                continue

            hist_penalty = (
                l1_hist_distance(val_hist, overall_hist)
                + l1_hist_distance(test_hist, overall_hist)
                + 0.5 * l1_hist_distance(train_hist, overall_hist)
            )

            score = (
                -hist_penalty,
                min(int(val_row["large_plus_windows"]), int(test_row["large_plus_windows"])),
                min(int(val_row["medium_plus_windows"]), int(test_row["medium_plus_windows"])),
                -abs(val_band - test_band),
            )

            candidate = {
                "val_band": val_band,
                "test_band": test_band,
                "train_bands": train_bands,
                "blocked_bands": sorted(blocked),
                "score": score,
                "hist_penalty": hist_penalty,
            }

            if best is None or candidate["score"] > best["score"]:
                best = candidate

    return best


def band_role(band_idx: int, split_cfg: dict, buffer_bands: int) -> str:
    if band_idx == split_cfg["val_band"]:
        return "val"
    if band_idx == split_cfg["test_band"]:
        return "test"
    if abs(band_idx - split_cfg["val_band"]) <= buffer_bands:
        return "buffer"
    if abs(band_idx - split_cfg["test_band"]) <= buffer_bands:
        return "buffer"
    return "train"


def assign_split_by_band(
    records: list[dict],
    x_to_band: dict[float, int],
    split_cfg: dict,
    buffer_bands: int,
    density_bins: list[int],
    density_names: list[str],
):
    output = []

    for record in records:
        item = dict(record)
        band_idx = x_to_band[record["x_center"]]
        item["band_idx"] = int(band_idx)
        item["density_bin"] = density_bin_name(record["positive_pixel_count"], density_bins, density_names)
        item["split"] = band_role(int(band_idx), split_cfg, buffer_bands)
        output.append(item)

    return output


def keep_for_main_eval(record: dict, min_pos_pixels_eval: int) -> bool:
    pos_px = record["positive_pixel_count"]
    return pos_px == 0 or pos_px >= min_pos_pixels_eval


def select_spatial_windows(
    *,
    train_candidates: list[dict],
    eval_candidates: list[dict],
    keep_empty_train_fraction: float,
    min_pos_pixels_eval: int,
    seed: int,
):
    rng = np.random.default_rng(seed)

    train_windows = []
    for record in train_candidates:
        if record["split"] != "train":
            continue
        if record["is_positive"]:
            train_windows.append(record)
        elif rng.random() < keep_empty_train_fraction:
            train_windows.append(record)

    val_windows = [
        record for record in eval_candidates
        if record["split"] == "val" and keep_for_main_eval(record, min_pos_pixels_eval)
    ]
    test_windows = [
        record for record in eval_candidates
        if record["split"] == "test" and keep_for_main_eval(record, min_pos_pixels_eval)
    ]

    val_reference = [record for record in eval_candidates if record["split"] == "val"]
    test_reference = [record for record in eval_candidates if record["split"] == "test"]

    return train_windows, val_windows, test_windows, val_reference, test_reference


def expand_windows_to_samples(
    spatial_windows: list[dict],
    date_rasters: list[tuple[str, str]],
    mask_path,
    split_name: str,
    density_bins: list[int],
    density_names: list[str],
):
    date_name_to_idx = {date_name: idx for idx, (date_name, _) in enumerate(date_rasters)}
    samples = []

    for record in spatial_windows:
        for date_name, image_path in date_rasters:
            samples.append({
                "date_name": date_name,
                "date_idx": date_name_to_idx[date_name],
                "image_path": image_path,
                "mask_path": mask_path,
                "window": record["window"],
                "row": record["row"],
                "col": record["col"],
                "band_idx": record["band_idx"],
                "grid_type": record["grid_type"],
                "split": split_name,
                "valid_fraction": record["valid_fraction"],
                "positive_pixel_count": record["positive_pixel_count"],
                "is_positive": record["is_positive"],
                "density_bin": record.get(
                    "density_bin",
                    density_bin_name(record["positive_pixel_count"], density_bins, density_names),
                ),
            })

    return samples


def sample_stats(samples: list[dict], min_pos_pixels_eval: int) -> dict:
    pos_pixels = np.array([sample["positive_pixel_count"] for sample in samples], dtype=np.int64)
    positive = pos_pixels[pos_pixels > 0]

    return {
        "total": len(samples),
        "empty": int((pos_pixels == 0).sum()),
        "tiny_positive": int(((pos_pixels > 0) & (pos_pixels < min_pos_pixels_eval)).sum()),
        "strong_positive": int((pos_pixels >= min_pos_pixels_eval).sum()),
        "positive_pixel_quantiles": np.quantile(positive, [0, 0.25, 0.5, 0.75, 0.9, 0.99]).tolist() if positive.size else [],
        "by_date": dict(Counter(sample["date_name"] for sample in samples)),
        "by_density": dict(Counter(sample["density_bin"] for sample in samples)),
        "by_band": dict(Counter(sample["band_idx"] for sample in samples)),
    }
