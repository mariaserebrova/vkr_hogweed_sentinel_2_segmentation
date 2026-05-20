import time
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm


def train_one_epoch(
    *,
    model,
    loader,
    optimizer,
    scheduler,
    scaler,
    device: str,
    bce_loss,
    tversky_loss,
    bce_loss_weight: float,
    tversky_loss_weight: float,
):
    model.train()
    total_loss = 0.0

    for images, masks, _ in tqdm(loader, desc="Train", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(device == "cuda")):
            logits = model(images)
            loss = bce_loss_weight * bce_loss(logits, masks) + tversky_loss_weight * tversky_loss(logits, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    scheduler.step()
    return total_loss / max(len(loader), 1)


def fit_model(
    *,
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    scaler,
    device: str,
    bce_loss,
    tversky_loss,
    bce_loss_weight: float,
    tversky_loss_weight: float,
    evaluate_loader_fn,
    date_rasters,
    tversky_alpha: float,
    tversky_beta: float,
    num_epochs: int,
    early_stopping_patience: int,
    checkpoint_path: Path,
    checkpoint_config: dict,
):
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    history = []
    best_val_tversky_positive = -1.0
    best_epoch = -1
    epochs_without_improve = 0

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            bce_loss=bce_loss,
            tversky_loss=tversky_loss,
            bce_loss_weight=bce_loss_weight,
            tversky_loss_weight=tversky_loss_weight,
        )

        val_stats = evaluate_loader_fn(
            model=model,
            loader=val_loader,
            device=device,
            threshold=0.50,
            bce_loss=bce_loss,
            tversky_loss=tversky_loss,
            bce_loss_weight=bce_loss_weight,
            tversky_loss_weight=tversky_loss_weight,
            date_rasters=date_rasters,
            tversky_alpha=tversky_alpha,
            tversky_beta=tversky_beta,
        )

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss,
            "val_loss": val_stats["loss"],
            "val_tversky_pos@0.50": val_stats["tversky_positive"],
            "val_f2_pos@0.50": val_stats["f2_positive"],
            "val_dice_pos@0.50": val_stats["dice_positive"],
            "val_iou_pos@0.50": val_stats["micro_iou_positive"],
            "val_precision_pos@0.50": val_stats["precision_positive"],
            "val_recall_pos@0.50": val_stats["recall_positive"],
            "epoch_sec": time.time() - start_time,
        }
        history.append(row)

        print(pd.DataFrame([row]).round(4).to_string(index=False))

        if val_stats["tversky_positive"] > best_val_tversky_positive:
            best_val_tversky_positive = val_stats["tversky_positive"]
            best_epoch = epoch
            epochs_without_improve = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_epoch": best_epoch,
                    "best_val_tversky_positive": best_val_tversky_positive,
                    "config": checkpoint_config,
                },
                checkpoint_path,
            )
            print(f"Saved checkpoint: {checkpoint_path}")
        else:
            epochs_without_improve += 1

        if epochs_without_improve >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
            break

    return pd.DataFrame(history), best_epoch, best_val_tversky_positive
