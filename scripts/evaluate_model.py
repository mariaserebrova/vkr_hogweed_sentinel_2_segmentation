import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
from src.metrics import evaluate_loader, evaluate_threshold_grid
from src.model import build_losses, build_model
from src.pipeline import prepare_data
from src.reproducibility import seed_everything


def serialize_stats(stats: dict) -> dict:
    output = {}
    for key, value in stats.items():
        if key == "by_date":
            output[key] = value.to_dict(orient="records")
        else:
            output[key] = float(value) if isinstance(value, (np.floating, float)) else int(value) if isinstance(value, (np.integer,)) else value
    return output


def main():
    seed_everything(config.SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config.TABLE_DIR.mkdir(parents=True, exist_ok=True)

    prepared = prepare_data(config)
    loaders = prepared["loaders"]

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

    bce_loss, tversky_loss = build_losses(
        pos_weight=config.POS_WEIGHT,
        alpha=config.TVERSKY_ALPHA,
        beta=config.TVERSKY_BETA,
        device=device,
    )

    threshold_df = evaluate_threshold_grid(
        model=model,
        loader=loaders["val"],
        device=device,
        thresholds=np.arange(0.10, 0.96, 0.05),
        bce_loss=bce_loss,
        tversky_loss=tversky_loss,
        bce_loss_weight=config.BCE_LOSS_WEIGHT,
        tversky_loss_weight=config.TVERSKY_LOSS_WEIGHT,
        date_rasters=prepared["date_rasters"],
        tversky_alpha=config.TVERSKY_ALPHA,
        tversky_beta=config.TVERSKY_BETA,
    )
    threshold_df = threshold_df.sort_values(
        ["tversky_positive", "f2_positive", "recall_positive", "dice_positive"],
        ascending=False,
    ).reset_index(drop=True)

    best_threshold = float(threshold_df.iloc[0]["threshold"])

    val_final = evaluate_loader(
        model=model,
        loader=loaders["val"],
        device=device,
        threshold=best_threshold,
        bce_loss=bce_loss,
        tversky_loss=tversky_loss,
        bce_loss_weight=config.BCE_LOSS_WEIGHT,
        tversky_loss_weight=config.TVERSKY_LOSS_WEIGHT,
        date_rasters=prepared["date_rasters"],
        tversky_alpha=config.TVERSKY_ALPHA,
        tversky_beta=config.TVERSKY_BETA,
    )
    test_final = evaluate_loader(
        model=model,
        loader=loaders["test"],
        device=device,
        threshold=best_threshold,
        bce_loss=bce_loss,
        tversky_loss=tversky_loss,
        bce_loss_weight=config.BCE_LOSS_WEIGHT,
        tversky_loss_weight=config.TVERSKY_LOSS_WEIGHT,
        date_rasters=prepared["date_rasters"],
        tversky_alpha=config.TVERSKY_ALPHA,
        tversky_beta=config.TVERSKY_BETA,
    )
    val_reference = evaluate_loader(
        model=model,
        loader=loaders["val_reference"],
        device=device,
        threshold=best_threshold,
        bce_loss=bce_loss,
        tversky_loss=tversky_loss,
        bce_loss_weight=config.BCE_LOSS_WEIGHT,
        tversky_loss_weight=config.TVERSKY_LOSS_WEIGHT,
        date_rasters=prepared["date_rasters"],
        tversky_alpha=config.TVERSKY_ALPHA,
        tversky_beta=config.TVERSKY_BETA,
    )
    test_reference = evaluate_loader(
        model=model,
        loader=loaders["test_reference"],
        device=device,
        threshold=best_threshold,
        bce_loss=bce_loss,
        tversky_loss=tversky_loss,
        bce_loss_weight=config.BCE_LOSS_WEIGHT,
        tversky_loss_weight=config.TVERSKY_LOSS_WEIGHT,
        date_rasters=prepared["date_rasters"],
        tversky_alpha=config.TVERSKY_ALPHA,
        tversky_beta=config.TVERSKY_BETA,
    )

    threshold_path = config.TABLE_DIR / f"{config.RUN_NAME}_threshold_search.csv"
    threshold_df.to_csv(threshold_path, index=False)

    result = {
        "best_threshold": best_threshold,
        "val_main": serialize_stats(val_final),
        "test_main": serialize_stats(test_final),
        "val_reference": serialize_stats(val_reference),
        "test_reference": serialize_stats(test_reference),
    }
    result_path = config.TABLE_DIR / f"{config.RUN_NAME}_evaluation.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Best threshold: {best_threshold}")
    print(f"Threshold table: {threshold_path}")
    print(f"Evaluation: {result_path}")


if __name__ == "__main__":
    main()
