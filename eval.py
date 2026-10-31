#!/usr/bin/env python3
"""Evaluate on validation / DUTS-TE / captured sets; measure T_CPU on DUTS-TE."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

# Single-thread before NumPy / OpenCV / Torch (required by assignment for timed path).
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import cv2
import numpy as np
import torch

cv2.setNumThreads(1)
torch.set_num_threads(1)

from ces.data import list_pairs, resize_image, resize_mask, train_val_split
from ces.metrics import (
    update_confusion,
    miou_from_confusion,
    boundary_f1,
    quality,
    ces_score,
)
from ces.pipeline import load_checkpoint, predict_mask, DEFAULT_CKPT


def eval_pairs(model, pairs, desc="set"):
    cm = np.zeros((2, 2), dtype=np.int64)
    bf1s = []
    for ip, mp in pairs:
        bgr = cv2.imread(str(ip), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        rgb = resize_image(bgr, is_bgr=True)
        gt = resize_mask(mask)
        pred = predict_mask(model, rgb)
        update_confusion(cm, pred, gt)
        bf1s.append(boundary_f1(pred, gt))
    miou = miou_from_confusion(cm)
    bf1 = float(np.mean(bf1s)) if bf1s else 0.0
    q = quality(miou, bf1)
    print(f"{desc}: mIoU={miou:.4f} BF1={bf1:.4f} Q={q:.4f} (n={len(pairs)})")
    return {"mIoU": miou, "BF1": bf1, "Q": q, "n": len(pairs)}


def time_duts_te(model, pairs):
    # Warmup
    ip0, _ = pairs[0]
    bgr = cv2.imread(str(ip0), cv2.IMREAD_COLOR)
    rgb0 = resize_image(bgr, is_bgr=True)
    _ = predict_mask(model, rgb0)

    total = 0.0
    for ip, _ in pairs:
        bgr = cv2.imread(str(ip), cv2.IMREAD_COLOR)
        rgb = resize_image(bgr, is_bgr=True)
        t0 = time.process_time()
        _ = predict_mask(model, rgb)
        total += time.process_time() - t0
    t_ms = 1000.0 * total / len(pairs)
    print(f"T_CPU ms (DUTS-TE avg): {t_ms:.4f}")
    return t_ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--skip-timing", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("results/metrics.json"))
    args = ap.parse_args()

    model = load_checkpoint(args.ckpt, map_location="cpu")
    model.eval()

    tr_pairs = list_pairs(
        args.data_root / "DUTS-TR" / "DUTS-TR-Image",
        args.data_root / "DUTS-TR" / "DUTS-TR-Mask",
    )
    _, val_pairs = train_val_split(tr_pairs, val_ratio=args.val_ratio, seed=42)
    te_pairs = list_pairs(
        args.data_root / "DUTS-TE" / "DUTS-TE-Image",
        args.data_root / "DUTS-TE" / "DUTS-TE-Mask",
    )

    results = {}
    results["val"] = eval_pairs(model, val_pairs, "validation")
    results["duts_te"] = eval_pairs(model, te_pairs, "DUTS-TE")

    cap_img = Path("captured/images_256")
    cap_msk = Path("captured/masks_256")
    if cap_img.exists() and cap_msk.exists():
        cap_pairs = list_pairs(cap_img, cap_msk)
        results["captured"] = eval_pairs(model, cap_pairs, "captured")

    s_disk = os.path.getsize(args.ckpt)
    results["S_disk_bytes"] = s_disk
    print(f"S_disk bytes: {s_disk}")

    if args.skip_timing:
        t_ms = None
    else:
        t_ms = time_duts_te(model, te_pairs)
    results["T_CPU_ms"] = t_ms

    if t_ms is not None:
        ces = ces_score(results["duts_te"]["Q"], t_ms, s_disk)
        results["CES"] = ces
        print(f"CES: {ces:.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print("Wrote", args.out)


if __name__ == "__main__":
    main()
