#!/usr/bin/env python3
"""Train TinyUNet on DUTS-TR (with a held-out validation split)."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from ces.data import list_pairs, train_val_split, SegPairDataset
from ces.metrics import update_confusion, miou_from_confusion, boundary_f1, quality
from ces.pipeline import build_model, save_checkpoint


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # logits: N,2,H,W ; target: N,H,W
    probs = F.softmax(logits, dim=1)[:, 1]
    t = target.float()
    inter = (probs * t).sum(dim=(1, 2))
    denom = probs.sum(dim=(1, 2)) + t.sum(dim=(1, 2))
    dice = (2 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


@torch.inference_mode()
def evaluate(model, loader, device, max_bf1_images: int = 200):
    model.eval()
    cm = __import__("numpy").zeros((2, 2), dtype=__import__("numpy").int64)
    bf1s = []
    for x, y, _ in loader:
        x = x.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1).cpu().numpy()
        gt = y.numpy()
        for i in range(pred.shape[0]):
            update_confusion(cm, pred[i], gt[i])
            if len(bf1s) < max_bf1_images:
                bf1s.append(boundary_f1(pred[i], gt[i]))
    miou = miou_from_confusion(cm)
    bf1 = float(sum(bf1s) / max(len(bf1s), 1))
    return {"mIoU": miou, "BF1": bf1, "Q": quality(miou, bf1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--base", type=int, default=16)
    ap.add_argument("--internal-size", type=int, default=128)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--out", type=Path, default=Path("checkpoints/model.pt"))
    ap.add_argument("--split-file", type=Path, default=Path("checkpoints/split.json"))
    args = ap.parse_args()

    tr_img = args.data_root / "DUTS-TR" / "DUTS-TR-Image"
    tr_msk = args.data_root / "DUTS-TR" / "DUTS-TR-Mask"
    pairs = list_pairs(tr_img, tr_msk)
    train_pairs, val_pairs = train_val_split(pairs, val_ratio=args.val_ratio, seed=42)
    args.split_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.split_file, "w") as f:
        json.dump(
            {
                "train_size": len(train_pairs),
                "val_size": len(val_pairs),
                "val_stems": [p[0].stem for p in val_pairs],
            },
            f,
        )
    print(f"Train {len(train_pairs)} | Val {len(val_pairs)}")

    train_ds = SegPairDataset(train_pairs, augment=True, seed=0)
    val_ds = SegPairDataset(val_pairs, augment=False)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("Device:", device)

    model = build_model(base=args.base, internal_size=args.internal_size).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_q = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        losses = []
        for x, y, _ in tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=False):
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            # class imbalance: background often dominates
            ce = F.cross_entropy(logits, y, weight=torch.tensor([1.0, 1.5], device=device))
            dice = soft_dice_loss(logits, y)
            loss = 0.6 * ce + 0.4 * dice
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        sched.step()
        metrics = evaluate(model, val_loader, device)
        metrics["loss"] = float(sum(losses) / max(len(losses), 1))
        metrics["epoch"] = epoch
        metrics["sec"] = time.time() - t0
        history.append(metrics)
        print(
            f"epoch {epoch}: loss={metrics['loss']:.4f} "
            f"mIoU={metrics['mIoU']:.4f} BF1={metrics['BF1']:.4f} "
            f"Q={metrics['Q']:.4f} ({metrics['sec']:.1f}s)"
        )
        if metrics["Q"] > best_q:
            best_q = metrics["Q"]
            save_checkpoint(model.cpu(), args.out, args.base, args.internal_size)
            model.to(device)
            print(f"  saved best -> {args.out} (Q={best_q:.4f})")

    with open(args.out.with_suffix(".history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print("Done. Best val Q:", best_q)


if __name__ == "__main__":
    # Training may use MPS / multiple threads; timing script enforces single-thread later.
    main()
