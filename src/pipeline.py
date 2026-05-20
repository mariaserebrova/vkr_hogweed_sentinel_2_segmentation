from collections import Counter
from pathlib import Path

import pandas as pd
import rasterio

from .dataset import build_dataloaders, sampler_weight_frame
from .raster_io import (
    build_mask_to_reference,
    discover_final_rasters,
    load_valid_mask,
    read_binary_mask,
    verify_geometry,
)
from .windows import (
    assign_split_by_band,
    band_stats,
    build_x_bands,
    choose_stratified_val_test_bands,
    expand_windows_to_samples,
    make_candidate_windows,
    sample_stats,
    select_spatial_windows,
)


def prepare_data(config):
    date_rasters = discover_final_rasters(config.PREPARED_ROOT, config.RUN_ONLY_DATES)
    if not date_rasters:
        raise ValueError("No final rasters found for the selected dates.")

    if config.BUILD_MASK_FROM_FULL or not config.MASK_FINAL_PATH.exists():
        build_mask_to_reference(config.FULL_MASK_PATH, date_rasters[0][1], config.MASK_FINAL_PATH)

    if not config.MASK_FINAL_PATH.exists():
        raise FileNotFoundError(config.MASK_FINAL_PATH)

    verify_geometry(date_rasters[0][1], config.MASK_FINAL_PATH)

    with rasterio.open(config.MASK_FINAL_PATH) as src:
        height, width = src.height, src.width

    full_mask = read_binary_mask(config.MASK_FINAL_PATH)
    valid_mask = load_valid_mask(date_rasters[0][1])

    train_candidates = make_candidate_windows(
        height=height,
        width=width,
        tile_size=config.TILE_SIZE,
        stride=config.TRAIN_STRIDE,
        min_valid_fraction=config.MIN_VALID_FRACTION_TRAIN,
        valid_mask=valid_mask,
        full_mask=full_mask,
        grid_type="train",
    )
    eval_candidates = make_candidate_windows(
        height=height,
        width=width,
        tile_size=config.TILE_SIZE,
        stride=config.EVAL_STRIDE,
        min_valid_fraction=config.MIN_VALID_FRACTION_EVAL,
        valid_mask=valid_mask,
        full_mask=full_mask,
        grid_type="eval",
    )

    if not train_candidates or not eval_candidates:
        raise ValueError("No candidate windows remained after valid-fraction filtering.")

    x_to_band, split_x = build_x_bands(train_candidates + eval_candidates, config.N_SPATIAL_BANDS)
    band_df = band_stats(eval_candidates, x_to_band, config.DENSITY_BINS, config.DENSITY_BIN_NAMES)

    best_split = choose_stratified_val_test_bands(
        stats_df=band_df,
        records=eval_candidates,
        x_to_band=x_to_band,
        density_bins=config.DENSITY_BINS,
        density_names=config.DENSITY_BIN_NAMES,
        buffer_bands=config.BUFFER_BANDS,
        min_medium_windows_per_split=config.MIN_MEDIUM_WINDOWS_PER_SPLIT,
        min_large_windows_per_split=config.MIN_LARGE_WINDOWS_PER_SPLIT,
    )

    if best_split is None:
        raise ValueError("No stratified spatial split satisfies the requested constraints.")

    train_candidates = assign_split_by_band(
        train_candidates,
        x_to_band,
        best_split,
        config.BUFFER_BANDS,
        config.DENSITY_BINS,
        config.DENSITY_BIN_NAMES,
    )
    eval_candidates = assign_split_by_band(
        eval_candidates,
        x_to_band,
        best_split,
        config.BUFFER_BANDS,
        config.DENSITY_BINS,
        config.DENSITY_BIN_NAMES,
    )

    train_windows, val_windows, test_windows, val_reference, test_reference = select_spatial_windows(
        train_candidates=train_candidates,
        eval_candidates=eval_candidates,
        keep_empty_train_fraction=config.KEEP_EMPTY_TRAIN_FRACTION,
        min_pos_pixels_eval=config.MIN_POS_PIXELS_EVAL,
        seed=config.SEED,
    )

    if not val_windows or not test_windows:
        raise ValueError("Validation or test windows are empty after filtering.")

    train_samples = expand_windows_to_samples(
        train_windows,
        date_rasters,
        config.MASK_FINAL_PATH,
        "train",
        config.DENSITY_BINS,
        config.DENSITY_BIN_NAMES,
    )
    val_samples = expand_windows_to_samples(
        val_windows,
        date_rasters,
        config.MASK_FINAL_PATH,
        "val",
        config.DENSITY_BINS,
        config.DENSITY_BIN_NAMES,
    )
    test_samples = expand_windows_to_samples(
        test_windows,
        date_rasters,
        config.MASK_FINAL_PATH,
        "test",
        config.DENSITY_BINS,
        config.DENSITY_BIN_NAMES,
    )
    val_reference_samples = expand_windows_to_samples(
        val_reference,
        date_rasters,
        config.MASK_FINAL_PATH,
        "val_reference",
        config.DENSITY_BINS,
        config.DENSITY_BIN_NAMES,
    )
    test_reference_samples = expand_windows_to_samples(
        test_reference,
        date_rasters,
        config.MASK_FINAL_PATH,
        "test_reference",
        config.DENSITY_BINS,
        config.DENSITY_BIN_NAMES,
    )

    loaders, datasets, sampler_weights = build_dataloaders(
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=test_samples,
        val_reference_samples=val_reference_samples,
        test_reference_samples=test_reference_samples,
        keep_channel_idxs_1based=config.KEEP_CHANNEL_IDXS_1BASED,
        clip_min=config.CLIP_MIN,
        clip_max=config.CLIP_MAX,
        scale_div=config.SCALE_DIV,
        nodata_value=config.NODATA_VALUE,
        date_weights=config.DATE_WEIGHTS,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    sample_summary = {
        "train": sample_stats(train_samples, config.MIN_POS_PIXELS_EVAL),
        "val": sample_stats(val_samples, config.MIN_POS_PIXELS_EVAL),
        "test": sample_stats(test_samples, config.MIN_POS_PIXELS_EVAL),
        "val_reference": sample_stats(val_reference_samples, config.MIN_POS_PIXELS_EVAL),
        "test_reference": sample_stats(test_reference_samples, config.MIN_POS_PIXELS_EVAL),
    }

    split_counts = {
        "train_grid": dict(Counter(record["split"] for record in train_candidates)),
        "eval_grid": dict(Counter(record["split"] for record in eval_candidates)),
    }

    return {
        "date_rasters": date_rasters,
        "full_mask": full_mask,
        "valid_mask": valid_mask,
        "height": height,
        "width": width,
        "train_candidates": train_candidates,
        "eval_candidates": eval_candidates,
        "x_to_band": x_to_band,
        "split_x": split_x,
        "band_df": band_df,
        "best_split": best_split,
        "train_windows": train_windows,
        "val_windows": val_windows,
        "test_windows": test_windows,
        "val_reference_windows": val_reference,
        "test_reference_windows": test_reference,
        "train_samples": train_samples,
        "val_samples": val_samples,
        "test_samples": test_samples,
        "val_reference_samples": val_reference_samples,
        "test_reference_samples": test_reference_samples,
        "loaders": loaders,
        "datasets": datasets,
        "sampler_weights": sampler_weights,
        "sampler_weight_frame": sampler_weight_frame(train_samples, config.DATE_WEIGHTS),
        "sample_summary": sample_summary,
        "split_counts": split_counts,
    }
