#!/usr/bin/env python3
"""
Usage:
  python predict.py --image photo.jpg --out mask.png

Loads parameters, resizes with INTER_AREA, segments, saves 0/255 PNG,
then times 20 more runs and prints exactly two lines:
  T CPU ms: ...
  S disk bytes: ...
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import cv2
import numpy as np
import torch

cv2.setNumThreads(1)
torch.set_num_threads(1)

from ces.data import resize_image
from ces.pipeline import load_checkpoint, predict_mask, DEFAULT_CKPT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    args = ap.parse_args()

    model = load_checkpoint(args.ckpt, map_location="cpu")
    model.eval()

    bgr = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if bgr is None:
        raise SystemExit(f"Could not read image: {args.image}")
    rgb = resize_image(bgr, is_bgr=True)

    # First run (library warmup / graph setup) — not timed for the average below,
    # but we still use its mask as the saved output.
    mask = predict_mask(model, rgb)
    out = (mask * 255).astype(np.uint8)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), out)

    times = []
    for _ in range(20):
        t0 = time.process_time()
        _ = predict_mask(model, rgb)
        times.append(time.process_time() - t0)
    t_ms = 1000.0 * (sum(times) / len(times))
    s_disk = os.path.getsize(args.ckpt)
    print(f"T CPU ms: {t_ms:.2f}")
    print(f"S disk bytes: {s_disk}")


if __name__ == "__main__":
    main()
