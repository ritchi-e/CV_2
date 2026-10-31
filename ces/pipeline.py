"""Inference helpers: load float16 weights and produce a binary mask."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ces.model import TinyUNet

DEFAULT_CKPT = Path(__file__).resolve().parent.parent / "checkpoints" / "model.pt"


def build_model(base: int = 16, internal_size: int = 128) -> TinyUNet:
    return TinyUNet(base=base, internal_size=internal_size)


def save_checkpoint(model: TinyUNet, path: str | Path, base: int, internal_size: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Store weights in float16 to keep S_disk small (no extra compression).
    state = {k: v.detach().cpu().half() for k, v in model.state_dict().items()}
    torch.save(
        {
            "state_dict": state,
            "base": base,
            "internal_size": internal_size,
        },
        path,
    )


def load_checkpoint(path: str | Path = DEFAULT_CKPT, map_location="cpu") -> TinyUNet:
    ckpt = torch.load(str(path), map_location=map_location, weights_only=False)
    model = build_model(base=int(ckpt["base"]), internal_size=int(ckpt["internal_size"]))
    # Cast back to float32 for CPU inference stability.
    sd = {k: v.float() if torch.is_floating_point(v) else v for k, v in ckpt["state_dict"].items()}
    model.load_state_dict(sd)
    model.eval()
    return model


@torch.inference_mode()
def predict_mask(model: TinyUNet, rgb_u8: np.ndarray) -> np.ndarray:
    """
    Timed core: 256x256 RGB uint8 in memory -> 256x256 {0,1} mask.
    Includes colour/tensor conversion and network forward.
    """
    x = torch.from_numpy(rgb_u8).permute(2, 0, 1).float().div_(255.0).unsqueeze(0)
    logits = model(x)
    pred = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    return pred
