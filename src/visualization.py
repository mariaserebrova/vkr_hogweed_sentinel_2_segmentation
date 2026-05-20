from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from rasterio.enums import Resampling
from scipy import ndimage as ndi

from .dataset import MultiDateWindowDataset


def normalize_rgb(rgb: np.ndarray) -> np.ndarray:
    rgb = rgb.astype(np.float32)
    p2 = np.percentile(rgb, 2)
    p98 = np.percentile(rgb, 98)
    return np.clip((rgb - p2) / (p98 - p2 + 1e-6), 0, 1)


def read_rgb_downsampled(path, rgb_idxs_1based=(4, 3, 2), out_max_size: int = 900, nodata_value: int | None = 65535):
    with rasterio.open(path) as src:
        scale = max(src.height / out_max_size, src.width / out_max_size, 1)
        out_h = int(src.height / scale)
        out_w = int(src.width / scale)
        arr = src.read(
            indexes=list(rgb_idxs_1based),
            out_shape=(3, out_h, out_w),
            resampling=Resampling.bilinear,
        ).astype(np.float32)

    if nodata_value is not None:
        arr[arr == nodata_value] = np.nan

    return np.transpose(arr, (1, 2, 0))


def percentile_stretch_rgb(rgb: np.ndarray, p_low: float = 2, p_high: float = 98) -> np.ndarray:
    out = np.zeros_like(rgb, dtype=np.float32)

    for channel_idx in range(3):
        band = rgb[..., channel_idx]
        valid = np.isfinite(band)

        if valid.sum() == 0:
            continue

        low, high = np.nanpercentile(band[valid], [p_low, p_high])
        if high > low:
            out[..., channel_idx] = (band - low) / (high - low)

    out = np.clip(out, 0, 1)
    out[~np.isfinite(out)] = 0
    return out


def plot_selected_dates_grid(
    *,
    date_rasters: list[tuple[str, str]],
    dates_to_show: list[str],
    date_labels: dict[str, str],
    out_path: Path | None = None,
    out_max_size: int = 900,
    n_cols: int = 3,
):
    date_to_path = {str(date_name): path for date_name, path in date_rasters}
    selected = [(date, date_to_path[date]) for date in dates_to_show]

    n_items = len(selected)
    ncols = min(n_cols, n_items)
    nrows = int(np.ceil(n_items / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 5.6 * nrows))

    axes = np.array([axes]) if n_items == 1 else np.array(axes).ravel()

    for ax, (date_name, image_path) in zip(axes, selected):
        rgb = read_rgb_downsampled(image_path, out_max_size=out_max_size)
        rgb_vis = percentile_stretch_rgb(rgb)
        ax.imshow(rgb_vis)
        ax.set_title(date_labels.get(date_name, f"Дата: {date_name}"), fontsize=15)
        ax.axis("off")

    for ax in axes[n_items:]:
        ax.axis("off")

    plt.tight_layout()

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=220, bbox_inches="tight")

    return fig


@torch.no_grad()
def show_prediction_examples(
    *,
    samples: list[dict],
    model,
    device: str,
    threshold: float,
    n_per_date: int,
    date_rasters: list[tuple[str, str]],
    keep_channel_idxs_1based: list[int],
    clip_min: float,
    clip_max: float,
    scale_div: float,
    nodata_value: float | None,
):
    idx_by_date = defaultdict(list)
    for sample_idx, sample in enumerate(samples):
        if sample["is_positive"]:
            idx_by_date[sample["date_idx"]].append(sample_idx)

    dataset = MultiDateWindowDataset(
        samples,
        keep_channel_idxs_1based=keep_channel_idxs_1based,
        clip_min=clip_min,
        clip_max=clip_max,
        scale_div=scale_div,
        nodata_value=nodata_value,
        augment=False,
    )

    figs = []
    for date_idx in sorted(idx_by_date.keys()):
        chosen = idx_by_date[date_idx][:n_per_date]

        for sample_idx in chosen:
            sample = samples[sample_idx]
            image, mask, _ = dataset[sample_idx]
            image_in = image.unsqueeze(0).to(device)
            mask_np = mask.squeeze(0).numpy()

            probs = torch.sigmoid(model(image_in))[0, 0].cpu().numpy()
            pred = (probs >= threshold).astype(np.uint8)

            with rasterio.open(sample["image_path"]) as src:
                raw = src.read(window=sample["window"]).astype(np.float32)

            rgb = np.stack([raw[3], raw[2], raw[1]], axis=-1)
            rgb = normalize_rgb(rgb)

            fig, axes = plt.subplots(1, 5, figsize=(24, 5))
            axes[0].imshow(rgb)
            axes[0].set_title("RGB")
            axes[0].axis("off")

            axes[1].imshow(mask_np, cmap="gray")
            axes[1].set_title("True mask")
            axes[1].axis("off")

            im = axes[2].imshow(probs, cmap="viridis", vmin=0, vmax=1)
            axes[2].set_title("Prob")
            axes[2].axis("off")
            plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

            axes[3].imshow(pred, cmap="gray")
            axes[3].set_title("Pred mask")
            axes[3].axis("off")

            overlay = rgb.copy()
            overlay[..., 1] = np.clip(overlay[..., 1] + 0.5 * pred, 0, 1)
            axes[4].imshow(overlay)
            axes[4].set_title("Overlay")
            axes[4].axis("off")

            plt.suptitle(
                f"{sample['date_name']} row={sample['row']} col={sample['col']} "
                f"pos_px={sample['positive_pixel_count']}"
            )
            plt.tight_layout()
            figs.append(fig)

    return figs


def overlap_overlay(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    overlay = np.zeros((*gt.shape, 3), dtype=np.float32)
    overlay[..., 1] = (pred == 1) & (gt == 1)
    overlay[..., 0] = (pred == 1) & (gt == 0)
    overlay[..., 2] = (pred == 0) & (gt == 1)
    return overlay


def plot_full_raster_result(
    *,
    prob: np.ndarray,
    pred: np.ndarray,
    gt: np.ndarray,
    threshold: float,
    min_component_area: int,
    out_path: Path | None = None,
):
    overlay = overlap_overlay(pred, gt)
    fig, axes = plt.subplots(1, 4, figsize=(26, 7))

    axes[0].imshow(prob, cmap="viridis", vmin=0.0, vmax=1.0)
    axes[0].set_title("Score map / probability", fontsize=14)
    axes[0].axis("off")

    axes[1].imshow(pred, cmap="gray")
    axes[1].set_title(f"Prediction\nthreshold={threshold}, min_area={min_component_area}", fontsize=14)
    axes[1].axis("off")

    axes[2].imshow(gt, cmap="gray")
    axes[2].set_title("Ground truth", fontsize=14)
    axes[2].axis("off")

    axes[3].imshow(overlay)
    axes[3].set_title("Overlay: TP green / FP red / FN blue", fontsize=14)
    axes[3].axis("off")

    plt.tight_layout()

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=220, bbox_inches="tight")

    return fig


def get_component_bboxes(mask: np.ndarray, min_area: int = 300, max_components: int = 8, pad: int = 100):
    mask = (mask > 0).astype(np.uint8)
    labels, n_labels = ndi.label(mask)

    if n_labels == 0:
        return []

    sizes = np.bincount(labels.ravel())
    component_ids = [idx for idx in range(1, n_labels + 1) if sizes[idx] >= min_area]
    component_ids = sorted(component_ids, key=lambda idx: sizes[idx], reverse=True)[:max_components]

    height, width = mask.shape
    bboxes = []

    for component_id in component_ids:
        ys, xs = np.where(labels == component_id)
        if len(ys) == 0:
            continue

        y1, y2 = ys.min(), ys.max()
        x1, x2 = xs.min(), xs.max()

        y1 = max(0, y1 - pad)
        y2 = min(height, y2 + pad)
        x1 = max(0, x1 - pad)
        x2 = min(width, x2 + pad)

        bboxes.append({
            "component_id": int(component_id),
            "area_px": int(sizes[component_id]),
            "area_ha": float(sizes[component_id] * 100 / 10000),
            "bbox": (int(y1), int(y2), int(x1), int(x2)),
        })

    return bboxes


def plot_component_crop(
    *,
    prob_crop: np.ndarray,
    pred_crop: np.ndarray,
    gt_crop: np.ndarray,
    title: str,
    out_path: Path | None = None,
):
    overlay = overlap_overlay(pred_crop, gt_crop)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(prob_crop, cmap="viridis", vmin=0.0, vmax=1.0)
    axes[0].set_title("Score")
    axes[0].axis("off")

    axes[1].imshow(pred_crop, cmap="gray")
    axes[1].set_title("Prediction")
    axes[1].axis("off")

    axes[2].imshow(gt_crop, cmap="gray")
    axes[2].set_title("Ground truth")
    axes[2].axis("off")

    axes[3].imshow(overlay)
    axes[3].set_title("TP green / FP red / FN blue")
    axes[3].axis("off")

    fig.suptitle(title)
    plt.tight_layout()

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=220, bbox_inches="tight")

    return fig
