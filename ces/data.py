"""DUTS loading, 256x256 resize rules, and train/val split."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


IMG_SIZE = 256
MASK_THRESH = 128


def resize_image(bgr_or_rgb: np.ndarray, is_bgr: bool = True) -> np.ndarray:
    """Resize photo to 256x256 with INTER_AREA; returns RGB uint8."""
    if is_bgr:
        rgb = cv2.cvtColor(bgr_or_rgb, cv2.COLOR_BGR2RGB)
    else:
        rgb = bgr_or_rgb
    return cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)


def resize_mask(mask: np.ndarray) -> np.ndarray:
    """Resize mask with INTER_NEAREST; return binary {0,1} uint8."""
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    m = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
    return (m >= MASK_THRESH).astype(np.uint8)


def list_pairs(image_dir: Path, mask_dir: Path) -> list[Tuple[Path, Path]]:
    images = sorted(
        [
            p
            for p in image_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        ]
    )
    pairs = []
    for ip in images:
        for ext in (".png", ".jpg", ".jpeg"):
            mp = mask_dir / f"{ip.stem}{ext}"
            if mp.exists():
                pairs.append((ip, mp))
                break
        else:
            raise FileNotFoundError(f"No mask for {ip.name} in {mask_dir}")
    return pairs


def train_val_split(
    pairs: list[Tuple[Path, Path]], val_ratio: float = 0.1, seed: int = 42
) -> Tuple[list, list]:
    rng = np.random.RandomState(seed)
    idx = np.arange(len(pairs))
    rng.shuffle(idx)
    n_val = int(round(len(pairs) * val_ratio))
    val_idx = set(idx[:n_val].tolist())
    train, val = [], []
    for i, p in enumerate(pairs):
        (val if i in val_idx else train).append(p)
    return train, val


class SegPairDataset(Dataset):
    def __init__(
        self,
        pairs: list[Tuple[Path, Path]],
        augment: bool = False,
        seed: int = 0,
    ):
        self.pairs = pairs
        self.augment = augment
        self.rng = np.random.RandomState(seed)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int):
        ip, mp = self.pairs[i]
        bgr = cv2.imread(str(ip), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Failed to read {ip}")
        mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Failed to read {mp}")
        rgb = resize_image(bgr, is_bgr=True)
        m = resize_mask(mask)

        if self.augment:
            if self.rng.rand() < 0.5:
                rgb = np.ascontiguousarray(rgb[:, ::-1])
                m = np.ascontiguousarray(m[:, ::-1])
            # mild brightness/contrast
            if self.rng.rand() < 0.5:
                alpha = 0.85 + 0.3 * self.rng.rand()
                beta = self.rng.randint(-15, 16)
                rgb = np.clip(alpha * rgb.astype(np.float32) + beta, 0, 255).astype(
                    np.uint8
                )

        x = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        y = torch.from_numpy(m.astype(np.int64))
        return x, y, ip.stem


def load_rgb256_from_path(path: str | Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return resize_image(bgr, is_bgr=True)
