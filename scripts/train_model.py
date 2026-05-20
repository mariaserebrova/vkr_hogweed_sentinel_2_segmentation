import json
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
from src.metrics import evaluate_loader
from src.model import build_losses, build_model
from src.pipeline import prepare_data
from src.reproducibility import seed_everything
from src.training import fit_model


def main():
    seed_everything(config.SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config.BEST_MODEL_DIR.mkdir(parents=True, exist_ok=True)
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
    bce_loss, tversky_loss = build_losses(
        pos_weight=config.POS_WEIGHT,
        alpha=config.TVERSKY_ALPHA,
        beta=config.TVERSKY_BETA,
        device=device,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.NUM_EPOCHS,
        eta_min=config.LR * 0.05,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    checkpoint_config = {
        "run_name": config.RUN_NAME,
        "dates": config.RUN_ONLY_DATES,
        "keep_channels": config.KEEP_CHANNEL_NAMES,
        "min_pos_pixels_eval": config.MIN_POS_PIXELS_EVAL,
        "split_mode": config.SPLIT_MODE,
        "loss": {
            "bce_weight": config.BCE_LOSS_WEIGHT,
            "tversky_weight": config.TVERSKY_LOSS_WEIGHT,
            "tversky_alpha": config.TVERSKY_ALPHA,
            "tversky_beta": config.TVERSKY_BETA,
            "pos_weight": config.POS_WEIGHT,
        },
    }

    history_df, best_epoch, best_score = fit_model(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
        bce_loss=bce_loss,
        tversky_loss=tversky_loss,
        bce_loss_weight=config.BCE_LOSS_WEIGHT,
        tversky_loss_weight=config.TVERSKY_LOSS_WEIGHT,
        evaluate_loader_fn=evaluate_loader,
        date_rasters=prepared["date_rasters"],
        tversky_alpha=config.TVERSKY_ALPHA,
        tversky_beta=config.TVERSKY_BETA,
        num_epochs=config.NUM_EPOCHS,
        early_stopping_patience=config.EARLY_STOPPING_PATIENCE,
        checkpoint_path=config.BEST_MODEL_PATH,
        checkpoint_config=checkpoint_config,
    )

    history_path = config.TABLE_DIR / f"{config.RUN_NAME}_history.csv"
    history_df.to_csv(history_path, index=False)

    summary = {
        "best_epoch": best_epoch,
        "best_val_tversky_positive": best_score,
        "checkpoint_path": str(config.BEST_MODEL_PATH),
        "history_path": str(history_path),
        "split_counts": prepared["split_counts"],
        "sample_summary": prepared["sample_summary"],
    }
    summary_path = config.TABLE_DIR / f"{config.RUN_NAME}_training_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"History: {history_path}")
    print(f"Summary: {summary_path}")
    print(f"Checkpoint: {config.BEST_MODEL_PATH}")


if __name__ == "__main__":
    main()
