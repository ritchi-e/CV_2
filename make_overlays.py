#!/usr/bin/env python3
"""Create true vs predicted semi-transparent overlays for every captured photo."""

from __future__ import annotations

import os
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

cv2.setNumThreads(1)
torch.set_num_threads(1)

from ces.pipeline import load_checkpoint, predict_mask


def overlay(rgb: np.ndarray, mask01: np.ndarray, color=(0, 220, 0)) -> np.ndarray:
    out = rgb.astype(np.float32).copy()
    sel = mask01.astype(bool)
    c = np.array(color, dtype=np.float32)
    out[sel] = 0.45 * out[sel] + 0.55 * c
    return np.clip(out, 0, 255).astype(np.uint8)


def main():
    model = load_checkpoint("checkpoints/model.pt", map_location="cpu")
    img_dir = Path("captured/images_256")
    msk_dir = Path("captured/masks_256")
    out_dir = Path("figures/captured_overlays")
    out_dir.mkdir(parents=True, exist_ok=True)

    names = sorted(p.stem for p in img_dir.glob("*.png"))
    n = len(names)
    fig, axes = plt.subplots(n, 2, figsize=(6, 2.2 * n))
    if n == 1:
        axes = np.array([axes])

    for i, name in enumerate(names):
        bgr = cv2.imread(str(img_dir / f"{name}.png"))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        gt = (cv2.imread(str(msk_dir / f"{name}.png"), cv2.IMREAD_GRAYSCALE) >= 128).astype(
            np.uint8
        )
        pred = predict_mask(model, rgb)
        true_ov = overlay(rgb, gt, color=(0, 200, 0))
        pred_ov = overlay(rgb, pred, color=(220, 40, 40))
        axes[i, 0].imshow(true_ov)
        axes[i, 0].set_title(f"{name} — true")
        axes[i, 0].axis("off")
        axes[i, 1].imshow(pred_ov)
        axes[i, 1].set_title(f"{name} — predicted")
        axes[i, 1].axis("off")
        cv2.imwrite(
            str(out_dir / f"{name}_true.png"),
            cv2.cvtColor(true_ov, cv2.COLOR_RGB2BGR),
        )
        cv2.imwrite(
            str(out_dir / f"{name}_pred.png"),
            cv2.cvtColor(pred_ov, cv2.COLOR_RGB2BGR),
        )

    fig.tight_layout()
    fig.savefig("figures/captured_true_vs_pred.png", dpi=140)
    print("Wrote figures/captured_true_vs_pred.png and", out_dir)


if __name__ == "__main__":
    main()
