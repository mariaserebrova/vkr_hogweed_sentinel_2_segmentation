from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from scipy import ndimage as ndi
from tqdm.auto import tqdm


def positive_tversky_from_counts(tp, fp, fn, alpha: float, beta: float, eps: float = 1e-6):
    return tp / (tp + alpha * fp + beta * fn + eps)


def positive_fbeta_from_counts(tp, fp, fn, beta: float = 2.0, eps: float = 1e-6):
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    beta_sq = beta ** 2
    return (1 + beta_sq) * precision * recall / (beta_sq * precision + recall + eps)


@torch.no_grad()
def evaluate_loader(
    *,
    model,
    loader,
    device: str,
    threshold: float,
    bce_loss,
    tversky_loss,
    bce_loss_weight: float,
    tversky_loss_weight: float,
    date_rasters: list[tuple[str, str]],
    tversky_alpha: float,
    tversky_beta: float,
    eps: float = 1e-6,
):
    model.eval()

    total_loss = 0.0
    macro_iou_all_sum = 0.0
    macro_iou_all_n = 0

    tp_all_total = 0.0
    fp_all_total = 0.0
    fn_all_total = 0.0

    tp_pos_total = 0.0
    fp_pos_total = 0.0
    fn_pos_total = 0.0

    macro_iou_pos_sum = 0.0
    macro_iou_pos_n = 0

    total_positive_tiles = 0
    total_empty_tiles = 0

    per_date = defaultdict(lambda: {
        "positive_tiles": 0,
        "tp": 0.0,
        "fp": 0.0,
        "fn": 0.0,
    })

    for images, masks, date_idx in tqdm(loader, desc="Eval", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        date_idx = date_idx.cpu().numpy()

        logits = model(images)
        loss = bce_loss_weight * bce_loss(logits, masks) + tversky_loss_weight * tversky_loss(logits, masks)
        total_loss += loss.item()

        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).float()

        tp = (preds * masks).sum(dim=(1, 2, 3))
        fp = (preds * (1 - masks)).sum(dim=(1, 2, 3))
        fn = ((1 - preds) * masks).sum(dim=(1, 2, 3))
        union = tp + fp + fn

        iou_all = torch.where(union > 0, tp / (union + eps), torch.ones_like(union))
        macro_iou_all_sum += iou_all.sum().item()
        macro_iou_all_n += iou_all.numel()

        tp_all_total += tp.sum().item()
        fp_all_total += fp.sum().item()
        fn_all_total += fn.sum().item()

        has_positive = masks.sum(dim=(1, 2, 3)) > 0
        total_positive_tiles += has_positive.sum().item()
        total_empty_tiles += (~has_positive).sum().item()

        if has_positive.any():
            tp_pos = tp[has_positive]
            fp_pos = fp[has_positive]
            fn_pos = fn[has_positive]
            union_pos = tp_pos + fp_pos + fn_pos
            iou_pos = tp_pos / (union_pos + eps)

            macro_iou_pos_sum += iou_pos.sum().item()
            macro_iou_pos_n += iou_pos.numel()

            tp_pos_total += tp_pos.sum().item()
            fp_pos_total += fp_pos.sum().item()
            fn_pos_total += fn_pos.sum().item()

        for idx in range(len(date_idx)):
            curr_date = int(date_idx[idx])
            if bool(has_positive[idx].item()):
                per_date[curr_date]["positive_tiles"] += 1
                per_date[curr_date]["tp"] += float(tp[idx].item())
                per_date[curr_date]["fp"] += float(fp[idx].item())
                per_date[curr_date]["fn"] += float(fn[idx].item())

    macro_iou_all = macro_iou_all_sum / max(macro_iou_all_n, 1)
    macro_iou_positive = macro_iou_pos_sum / max(macro_iou_pos_n, 1)

    micro_iou_positive = tp_pos_total / (tp_pos_total + fp_pos_total + fn_pos_total + eps)
    dice_positive = (2 * tp_pos_total) / (2 * tp_pos_total + fp_pos_total + fn_pos_total + eps)
    precision_positive = tp_pos_total / (tp_pos_total + fp_pos_total + eps)
    recall_positive = tp_pos_total / (tp_pos_total + fn_pos_total + eps)
    tversky_positive = positive_tversky_from_counts(
        tp_pos_total, fp_pos_total, fn_pos_total, tversky_alpha, tversky_beta, eps
    )
    f2_positive = positive_fbeta_from_counts(tp_pos_total, fp_pos_total, fn_pos_total, beta=2.0, eps=eps)

    iou_all_positive_class = tp_all_total / (tp_all_total + fp_all_total + fn_all_total + eps)
    dice_all_positive_class = (2 * tp_all_total) / (2 * tp_all_total + fp_all_total + fn_all_total + eps)
    precision_all_positive_class = tp_all_total / (tp_all_total + fp_all_total + eps)
    recall_all_positive_class = tp_all_total / (tp_all_total + fn_all_total + eps)
    tversky_all_positive_class = positive_tversky_from_counts(
        tp_all_total, fp_all_total, fn_all_total, tversky_alpha, tversky_beta, eps
    )
    f2_all_positive_class = positive_fbeta_from_counts(tp_all_total, fp_all_total, fn_all_total, beta=2.0, eps=eps)

    idx_to_name = {idx: date for idx, (date, _) in enumerate(date_rasters)}
    rows = []

    for date_idx in sorted(per_date.keys()):
        tp = per_date[date_idx]["tp"]
        fp = per_date[date_idx]["fp"]
        fn = per_date[date_idx]["fn"]

        rows.append({
            "date_idx": date_idx,
            "date_name": idx_to_name[date_idx],
            "positive_tiles": per_date[date_idx]["positive_tiles"],
            "micro_iou_positive": tp / (tp + fp + fn + eps),
            "dice_positive": (2 * tp) / (2 * tp + fp + fn + eps),
            "precision_positive": tp / (tp + fp + eps),
            "recall_positive": tp / (tp + fn + eps),
            "tversky_positive": positive_tversky_from_counts(tp, fp, fn, tversky_alpha, tversky_beta, eps),
            "f2_positive": positive_fbeta_from_counts(tp, fp, fn, beta=2.0, eps=eps),
        })

    return {
        "loss": total_loss / max(len(loader), 1),
        "macro_iou_all": macro_iou_all,
        "macro_iou_positive": macro_iou_positive,
        "micro_iou_positive": micro_iou_positive,
        "dice_positive": dice_positive,
        "precision_positive": precision_positive,
        "recall_positive": recall_positive,
        "tversky_positive": tversky_positive,
        "f2_positive": f2_positive,
        "iou_all_positive_class": iou_all_positive_class,
        "dice_all_positive_class": dice_all_positive_class,
        "precision_all_positive_class": precision_all_positive_class,
        "recall_all_positive_class": recall_all_positive_class,
        "tversky_all_positive_class": tversky_all_positive_class,
        "f2_all_positive_class": f2_all_positive_class,
        "positive_tiles": total_positive_tiles,
        "empty_tiles": total_empty_tiles,
        "by_date": pd.DataFrame(rows),
    }


@torch.no_grad()
def evaluate_threshold_grid(
    *,
    model,
    loader,
    device: str,
    thresholds,
    bce_loss,
    tversky_loss,
    bce_loss_weight: float,
    tversky_loss_weight: float,
    date_rasters: list[tuple[str, str]],
    tversky_alpha: float,
    tversky_beta: float,
):
    rows = []

    for threshold in thresholds:
        stats = evaluate_loader(
            model=model,
            loader=loader,
            device=device,
            threshold=float(threshold),
            bce_loss=bce_loss,
            tversky_loss=tversky_loss,
            bce_loss_weight=bce_loss_weight,
            tversky_loss_weight=tversky_loss_weight,
            date_rasters=date_rasters,
            tversky_alpha=tversky_alpha,
            tversky_beta=tversky_beta,
        )
        rows.append({
            "threshold": float(threshold),
            "tversky_positive": stats["tversky_positive"],
            "f2_positive": stats["f2_positive"],
            "micro_iou_positive": stats["micro_iou_positive"],
            "dice_positive": stats["dice_positive"],
            "precision_positive": stats["precision_positive"],
            "recall_positive": stats["recall_positive"],
            "tversky_all_positive_class": stats["tversky_all_positive_class"],
            "f2_all_positive_class": stats["f2_all_positive_class"],
            "positive_tiles": stats["positive_tiles"],
            "empty_tiles": stats["empty_tiles"],
        })

    return pd.DataFrame(rows)


def remove_small_components(mask: np.ndarray, min_area: int = 0) -> np.ndarray:
    mask = (mask > 0).astype(np.uint8)
    if min_area is None or min_area <= 0:
        return mask

    labels, n_labels = ndi.label(mask)
    if n_labels == 0:
        return mask

    sizes = np.bincount(labels.ravel())
    keep = sizes >= min_area
    keep[0] = False
    return keep[labels].astype(np.uint8)


def basic_metrics(pred: np.ndarray, gt: np.ndarray, alpha: float, beta: float) -> dict:
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()

    eps = 1e-7

    precision = tp / max(tp + fp, eps)
    recall = tp / max(tp + fn, eps)
    f2 = 5 * precision * recall / max(4 * precision + recall, eps)
    dice = 2 * tp / max(2 * tp + fp + fn, eps)
    iou = tp / max(tp + fp + fn, eps)
    tversky = tp / max(tp + alpha * fp + beta * fn, eps)

    return {
        "precision": precision,
        "recall": recall,
        "f2": f2,
        "dice": dice,
        "iou": iou,
        "tversky": tversky,
        "positive_fraction": float(pred.mean()),
        "area_over_gt": pred.sum() / max(gt.sum(), 1),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
    }


def object_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    min_gt_area: int = 300,
    match_gt_frac: float = 0.10,
    match_pred_frac: float = 0.10,
) -> dict:
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)

    gt_labels, gt_n = ndi.label(gt)
    pred_labels, pred_n = ndi.label(pred)

    gt_sizes = np.bincount(gt_labels.ravel()) if gt_n > 0 else np.array([0])
    pred_sizes = np.bincount(pred_labels.ravel()) if pred_n > 0 else np.array([0])

    gt_ids = [idx for idx in range(1, gt_n + 1) if gt_sizes[idx] >= min_gt_area]
    pred_ids = [idx for idx in range(1, pred_n + 1) if pred_sizes[idx] >= min_gt_area]

    found_gt = 0
    for comp_id in gt_ids:
        comp = gt_labels == comp_id
        overlap = pred[comp].sum()
        if overlap / max(comp.sum(), 1) >= match_gt_frac:
            found_gt += 1

    matched_pred = 0
    for comp_id in pred_ids:
        comp = pred_labels == comp_id
        overlap = gt[comp].sum()
        if overlap / max(comp.sum(), 1) >= match_pred_frac:
            matched_pred += 1

    obj_recall = found_gt / max(len(gt_ids), 1)
    obj_precision = matched_pred / max(len(pred_ids), 1)
    obj_f2 = 5 * obj_precision * obj_recall / max(4 * obj_precision + obj_recall, 1e-7)

    return {
        f"object_recall_area{min_gt_area}": obj_recall,
        f"object_precision_area{min_gt_area}": obj_precision,
        f"object_f2_area{min_gt_area}": obj_f2,
        f"n_gt_area{min_gt_area}": len(gt_ids),
        f"n_pred_area{min_gt_area}": len(pred_ids),
        f"n_found_gt_area{min_gt_area}": found_gt,
        f"n_matched_pred_area{min_gt_area}": matched_pred,
    }


def buffered_f2(pred: np.ndarray, gt: np.ndarray, radius: int = 2) -> dict:
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    structure = ndi.generate_binary_structure(2, 1)
    pred_dil = ndi.binary_dilation(pred, structure=structure, iterations=radius)
    gt_dil = ndi.binary_dilation(gt, structure=structure, iterations=radius)

    precision = np.logical_and(pred, gt_dil).sum() / max(pred.sum(), 1)
    recall = np.logical_and(gt, pred_dil).sum() / max(gt.sum(), 1)
    f2 = 5 * precision * recall / max(4 * precision + recall, 1e-7)

    return {
        f"buffered_precision_r{radius}": precision,
        f"buffered_recall_r{radius}": recall,
        f"buffered_f2_r{radius}": f2,
    }


def eval_full_fast(pred: np.ndarray, gt: np.ndarray, alpha: float, beta: float) -> dict:
    metrics = {}
    metrics.update(basic_metrics(pred, gt, alpha, beta))
    metrics.update(object_metrics(pred, gt, min_gt_area=300))
    metrics.update(buffered_f2(pred, gt, radius=2))
    return metrics


def full_raster_threshold_sweep(
    *,
    prob: np.ndarray,
    gt: np.ndarray,
    thresholds: list[float],
    min_component_areas: list[int],
    alpha: float,
    beta: float,
):
    rows = []

    for threshold in thresholds:
        raw = (prob >= threshold).astype(np.uint8)

        for min_area in min_component_areas:
            pred = remove_small_components(raw, min_area=min_area)
            metrics = eval_full_fast(pred, gt, alpha, beta)
            metrics["threshold"] = float(threshold)
            metrics["min_component_area"] = int(min_area)

            area_penalty = 0.0
            if metrics["area_over_gt"] > 4.0:
                area_penalty += 0.08 * (metrics["area_over_gt"] - 4.0)
            if metrics["positive_fraction"] > 0.03:
                area_penalty += 2.0 * (metrics["positive_fraction"] - 0.03)

            metrics["score_clean"] = (
                1.40 * metrics["tversky"]
                + 0.90 * metrics["f2"]
                + 0.70 * metrics["object_f2_area300"]
                + 0.45 * metrics["buffered_f2_r2"]
                + 0.30 * metrics["precision"]
                - area_penalty
            )

            rows.append(metrics)

    sweep_df = pd.DataFrame(rows)
    candidates = sweep_df[
        (sweep_df["positive_fraction"] <= 0.04)
        & (sweep_df["area_over_gt"] <= 8.0)
        & (sweep_df["recall"] >= 0.05)
    ].copy()

    if candidates.empty:
        candidates = sweep_df.copy()

    best_row = candidates.sort_values(
        ["score_clean", "tversky", "object_f2_area300"],
        ascending=False,
    ).iloc[0]

    return sweep_df, candidates, best_row
