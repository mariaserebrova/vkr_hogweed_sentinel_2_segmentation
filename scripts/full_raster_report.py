import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
from src.inference import predict_full_raster_with_overlap
from src.metrics import (
    buffered_f2,
    eval_full_fast,
    full_raster_threshold_sweep,
    object_metrics,
    remove_small_components,
)
from src.model import build_model
from src.pipeline import prepare_data
from src.raster_io import read_binary_mask
from src.reproducibility import seed_everything
from src.visualization import get_component_bboxes, plot_component_crop, plot_full_raster_result


def main():
    seed_everything(config.SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config.TABLE_DIR.mkdir(parents=True, exist_ok=True)
    config.FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    prepared = prepare_data(config)
    date_to_path = {str(date_name): image_path for date_name, image_path in prepared["date_rasters"]}

    if config.FULL_RASTER_DATE not in date_to_path:
        raise ValueError(f"Date {config.FULL_RASTER_DATE} is absent from date_rasters.")

    model = build_model(
        encoder_name=config.ENCODER_NAME,
        encoder_weights=config.ENCODER_WEIGHTS,
        in_channels=config.IN_CHANNELS,
        num_classes=config.NUM_CLASSES,
        device=device,
    )
    checkpoint = torch.load(config.BEST_MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    image_path = date_to_path[config.FULL_RASTER_DATE]
    gt = read_binary_mask(config.MASK_FINAL_PATH)

    prob, pred_raw = predict_full_raster_with_overlap(
        model=model,
        image_path=image_path,
        device=device,
        keep_channel_idxs_1based=config.KEEP_CHANNEL_IDXS_1BASED,
        clip_min=config.CLIP_MIN,
        clip_max=config.CLIP_MAX,
        scale_div=config.SCALE_DIV,
        nodata_value=config.NODATA_VALUE,
        threshold=config.FINAL_FULL_RASTER_THRESHOLD,
        tile_size=config.TILE_SIZE,
        stride=config.FULL_RASTER_STRIDE,
        batch_size=config.FULL_RASTER_BATCH_SIZE,
        progress_desc=f"Full overlap {config.FULL_RASTER_DATE}",
    )

    sweep_df, candidates_df, best_row = full_raster_threshold_sweep(
        prob=prob,
        gt=gt,
        thresholds=config.FULL_SWEEP_THRESHOLDS,
        min_component_areas=config.FULL_SWEEP_MIN_COMPONENT_AREAS,
        alpha=config.TVERSKY_ALPHA,
        beta=config.TVERSKY_BETA,
    )

    sweep_path = config.TABLE_DIR / f"{config.RUN_NAME}_full_raster_sweep.csv"
    candidates_path = config.TABLE_DIR / f"{config.RUN_NAME}_full_raster_candidates.csv"
    sweep_df.to_csv(sweep_path, index=False)
    candidates_df.to_csv(candidates_path, index=False)

    final_pred = remove_small_components(
        (prob >= config.FINAL_FULL_RASTER_THRESHOLD).astype(np.uint8),
        min_area=config.FINAL_FULL_RASTER_MIN_COMPONENT_AREA,
    )
    final_metrics = eval_full_fast(final_pred, gt, config.TVERSKY_ALPHA, config.TVERSKY_BETA)

    full_figure_path = config.FIGURE_DIR / f"{config.RUN_NAME}_full_raster_result.png"
    fig = plot_full_raster_result(
        prob=prob,
        pred=final_pred,
        gt=gt,
        threshold=config.FINAL_FULL_RASTER_THRESHOLD,
        min_component_area=config.FINAL_FULL_RASTER_MIN_COMPONENT_AREA,
        out_path=full_figure_path,
    )
    fig.clf()

    object_rows = []
    for area in [25, 50, 100, 200, 300, 500, 1000, 2000]:
        metrics = object_metrics(final_pred, gt, min_gt_area=area)
        object_rows.append({
            "min_component_area_px": area,
            "approx_area_ha_at_10m": area * 100 / 10000,
            "object_recall": metrics[f"object_recall_area{area}"],
            "object_precision": metrics[f"object_precision_area{area}"],
            "object_f2": metrics[f"object_f2_area{area}"],
            "n_gt": metrics[f"n_gt_area{area}"],
            "n_pred": metrics[f"n_pred_area{area}"],
            "n_found_gt": metrics[f"n_found_gt_area{area}"],
            "n_matched_pred": metrics[f"n_matched_pred_area{area}"],
        })
    object_df = pd.DataFrame(object_rows)

    buffer_rows = []
    for radius in [1, 2, 3, 5, 7]:
        metrics = buffered_f2(final_pred, gt, radius=radius)
        buffer_rows.append({
            "buffer_radius_px": radius,
            "buffer_radius_m_at_10m": radius * 10,
            "buffered_precision": metrics[f"buffered_precision_r{radius}"],
            "buffered_recall": metrics[f"buffered_recall_r{radius}"],
            "buffered_f2": metrics[f"buffered_f2_r{radius}"],
        })
    buffer_df = pd.DataFrame(buffer_rows)

    object_path = config.TABLE_DIR / f"{config.RUN_NAME}_full_raster_object_metrics.csv"
    buffer_path = config.TABLE_DIR / f"{config.RUN_NAME}_full_raster_buffered_metrics.csv"
    object_df.to_csv(object_path, index=False)
    buffer_df.to_csv(buffer_path, index=False)

    crop_dir = config.FIGURE_DIR / "gt_component_crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    for item in get_component_bboxes(gt, min_area=300, max_components=8, pad=100):
        y1, y2, x1, x2 = item["bbox"]
        fig = plot_component_crop(
            prob_crop=prob[y1:y2, x1:x2],
            pred_crop=final_pred[y1:y2, x1:x2],
            gt_crop=gt[y1:y2, x1:x2],
            title=f"GT component {item['component_id']}, area={item['area_px']} px",
            out_path=crop_dir / f"gt_component_{item['component_id']:03d}.png",
        )
        fig.clf()

    fp_mask = ((final_pred == 1) & (gt == 0)).astype(np.uint8)
    fp_crop_dir = config.FIGURE_DIR / "false_positive_crops"
    fp_crop_dir.mkdir(parents=True, exist_ok=True)
    for item in get_component_bboxes(fp_mask, min_area=300, max_components=6, pad=100):
        y1, y2, x1, x2 = item["bbox"]
        fig = plot_component_crop(
            prob_crop=prob[y1:y2, x1:x2],
            pred_crop=final_pred[y1:y2, x1:x2],
            gt_crop=gt[y1:y2, x1:x2],
            title=f"FP component {item['component_id']}, area={item['area_px']} px",
            out_path=fp_crop_dir / f"fp_component_{item['component_id']:03d}.png",
        )
        fig.clf()

    summary = {
        "date": config.FULL_RASTER_DATE,
        "threshold": config.FINAL_FULL_RASTER_THRESHOLD,
        "min_component_area": config.FINAL_FULL_RASTER_MIN_COMPONENT_AREA,
        "full_metrics": final_metrics,
        "sweep_best_row": best_row.to_dict(),
        "artifacts": {
            "sweep": str(sweep_path),
            "sweep_candidates": str(candidates_path),
            "object_metrics": str(object_path),
            "buffered_metrics": str(buffer_path),
            "full_raster_figure": str(full_figure_path),
        },
    }
    summary_path = config.TABLE_DIR / f"{config.RUN_NAME}_full_raster_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8")

    np.save(config.TABLE_DIR / f"{config.RUN_NAME}_full_raster_probability.npy", prob)
    np.save(config.TABLE_DIR / f"{config.RUN_NAME}_full_raster_prediction.npy", final_pred)

    print(f"Sweep: {sweep_path}")
    print(f"Object metrics: {object_path}")
    print(f"Buffered metrics: {buffer_path}")
    print(f"Summary: {summary_path}")
    print(f"Figure: {full_figure_path}")


if __name__ == "__main__":
    main()
