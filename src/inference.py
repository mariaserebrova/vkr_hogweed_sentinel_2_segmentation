import numpy as np
import rasterio
import torch
from rasterio.windows import Window
from tqdm.auto import tqdm

from .raster_io import read_model_tile


@torch.no_grad()
def predict_full_raster_with_overlap(
    *,
    model,
    image_path,
    device: str,
    keep_channel_idxs_1based: list[int],
    clip_min: float,
    clip_max: float,
    scale_div: float,
    nodata_value: float | None,
    threshold: float,
    tile_size: int,
    stride: int,
    batch_size: int,
    progress_desc: str = "Full raster inference",
):
    model.eval()

    with rasterio.open(image_path) as src:
        height, width = src.height, src.width

    acc_prob = np.zeros((height, width), dtype=np.float32)
    acc_weight = np.zeros((height, width), dtype=np.float32)

    wy = np.hanning(tile_size).astype(np.float32)
    wx = np.hanning(tile_size).astype(np.float32)
    wy = np.maximum(wy, 0.05)
    wx = np.maximum(wx, 0.05)
    weight_2d = np.outer(wy, wx).astype(np.float32)

    rows = list(range(0, max(height - tile_size + 1, 1), stride))
    cols = list(range(0, max(width - tile_size + 1, 1), stride))

    if rows[-1] != height - tile_size:
        rows.append(height - tile_size)
    if cols[-1] != width - tile_size:
        cols.append(width - tile_size)

    rows = [max(0, row) for row in rows]
    cols = [max(0, col) for col in cols]
    windows = [(row, col) for row in rows for col in cols]

    batch_tensors = []
    batch_rc = []

    for row, col in tqdm(windows, desc=progress_desc):
        window = Window(col, row, tile_size, tile_size)
        tile = read_model_tile(
            image_path,
            window,
            keep_channel_idxs_1based,
            clip_min,
            clip_max,
            scale_div,
            nodata_value,
        )

        batch_tensors.append(torch.from_numpy(tile).float().unsqueeze(0))
        batch_rc.append((row, col))

        if len(batch_tensors) == batch_size:
            batch = torch.cat(batch_tensors, dim=0).to(device)
            logits = model(batch)
            probs = torch.sigmoid(logits).detach().cpu().numpy()[:, 0]

            for idx, (row_item, col_item) in enumerate(batch_rc):
                acc_prob[row_item:row_item + tile_size, col_item:col_item + tile_size] += probs[idx] * weight_2d
                acc_weight[row_item:row_item + tile_size, col_item:col_item + tile_size] += weight_2d

            batch_tensors = []
            batch_rc = []

    if batch_tensors:
        batch = torch.cat(batch_tensors, dim=0).to(device)
        logits = model(batch)
        probs = torch.sigmoid(logits).detach().cpu().numpy()[:, 0]

        for idx, (row_item, col_item) in enumerate(batch_rc):
            acc_prob[row_item:row_item + tile_size, col_item:col_item + tile_size] += probs[idx] * weight_2d
            acc_weight[row_item:row_item + tile_size, col_item:col_item + tile_size] += weight_2d

    prob = acc_prob / np.maximum(acc_weight, 1e-12)
    pred = (prob >= threshold).astype(np.uint8)
    return prob, pred
