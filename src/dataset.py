import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .raster_io import read_model_tile, read_window_mask


class MultiDateWindowDataset(Dataset):
    def __init__(
        self,
        samples: list[dict],
        *,
        keep_channel_idxs_1based: list[int],
        clip_min: float,
        clip_max: float,
        scale_div: float,
        nodata_value: float | None = None,
        augment: bool = False,
    ):
        self.samples = samples
        self.keep_channel_idxs_1based = keep_channel_idxs_1based
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.scale_div = scale_div
        self.nodata_value = nodata_value
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def _augment(self, image: np.ndarray, mask: np.ndarray):
        if random.random() < 0.5:
            image = np.flip(image, axis=2).copy()
            mask = np.flip(mask, axis=2).copy()

        if random.random() < 0.5:
            image = np.flip(image, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()

        if random.random() < 0.5:
            k = random.randint(1, 3)
            image = np.rot90(image, k=k, axes=(1, 2)).copy()
            mask = np.rot90(mask, k=k, axes=(1, 2)).copy()

        return image, mask

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image = read_model_tile(
            sample["image_path"],
            sample["window"],
            self.keep_channel_idxs_1based,
            self.clip_min,
            self.clip_max,
            self.scale_div,
            self.nodata_value,
        )
        mask = read_window_mask(sample["mask_path"], sample["window"])

        if self.augment:
            image, mask = self._augment(image, mask)

        return (
            torch.from_numpy(image),
            torch.from_numpy(mask),
            torch.tensor(sample["date_idx"], dtype=torch.long),
        )


def large_focus_weight(sample: dict) -> float:
    pos_px = sample["positive_pixel_count"]
    if pos_px == 0:
        return 0.15
    if pos_px < 25:
        return 0.20
    if pos_px < 100:
        return 0.60
    if pos_px < 200:
        return 1.00
    if pos_px < 500:
        return 2.00
    if pos_px < 2000:
        return 3.00
    return 3.50


def sampler_weight_frame(samples: list[dict], date_weights: dict[str, float]) -> pd.DataFrame:
    rows = []
    for sample in samples:
        weight = date_weights.get(sample["date_name"], 1.0) * large_focus_weight(sample)
        rows.append({
            "density_bin": sample["density_bin"],
            "date_name": sample["date_name"],
            "weight": float(weight),
        })
    return pd.DataFrame(rows)


def build_dataloaders(
    *,
    train_samples: list[dict],
    val_samples: list[dict],
    test_samples: list[dict],
    val_reference_samples: list[dict],
    test_reference_samples: list[dict],
    keep_channel_idxs_1based: list[int],
    clip_min: float,
    clip_max: float,
    scale_div: float,
    nodata_value: float | None,
    date_weights: dict[str, float],
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
):
    train_ds = MultiDateWindowDataset(
        train_samples,
        keep_channel_idxs_1based=keep_channel_idxs_1based,
        clip_min=clip_min,
        clip_max=clip_max,
        scale_div=scale_div,
        nodata_value=nodata_value,
        augment=True,
    )
    val_ds = MultiDateWindowDataset(
        val_samples,
        keep_channel_idxs_1based=keep_channel_idxs_1based,
        clip_min=clip_min,
        clip_max=clip_max,
        scale_div=scale_div,
        nodata_value=nodata_value,
        augment=False,
    )
    test_ds = MultiDateWindowDataset(
        test_samples,
        keep_channel_idxs_1based=keep_channel_idxs_1based,
        clip_min=clip_min,
        clip_max=clip_max,
        scale_div=scale_div,
        nodata_value=nodata_value,
        augment=False,
    )
    val_reference_ds = MultiDateWindowDataset(
        val_reference_samples,
        keep_channel_idxs_1based=keep_channel_idxs_1based,
        clip_min=clip_min,
        clip_max=clip_max,
        scale_div=scale_div,
        nodata_value=nodata_value,
        augment=False,
    )
    test_reference_ds = MultiDateWindowDataset(
        test_reference_samples,
        keep_channel_idxs_1based=keep_channel_idxs_1based,
        clip_min=clip_min,
        clip_max=clip_max,
        scale_div=scale_div,
        nodata_value=nodata_value,
        augment=False,
    )

    weights = torch.as_tensor(
        [date_weights.get(sample["date_name"], 1.0) * large_focus_weight(sample) for sample in train_samples],
        dtype=torch.double,
    )

    train_sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
    )

    loaders = {
        "train": DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "val": DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "test": DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "val_reference": DataLoader(
            val_reference_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "test_reference": DataLoader(
            test_reference_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
    }

    datasets = {
        "train": train_ds,
        "val": val_ds,
        "test": test_ds,
        "val_reference": val_reference_ds,
        "test_reference": test_reference_ds,
    }

    return loaders, datasets, weights
