"""mIoU and Boundary F1 computed from scratch (no skimage / scipy morphology)."""

from __future__ import annotations

import numpy as np


def update_confusion(cm: np.ndarray, pred: np.ndarray, gt: np.ndarray) -> None:
    """Accumulate a 2x2 confusion matrix. pred/gt are binary {0,1} arrays."""
    p = pred.astype(np.uint8).ravel()
    g = gt.astype(np.uint8).ravel()
    # index = gt * 2 + pred
    idx = g.astype(np.int64) * 2 + p.astype(np.int64)
    binc = np.bincount(idx, minlength=4)
    cm += binc.reshape(2, 2)


def miou_from_confusion(cm: np.ndarray) -> float:
    ious = []
    for c in range(2):
        inter = cm[c, c]
        union = cm[c, :].sum() + cm[:, c].sum() - inter
        ious.append(float(inter) / float(union) if union > 0 else 0.0)
    return 0.5 * (ious[0] + ious[1])


def boundary_map(mask: np.ndarray) -> np.ndarray:
    """Object pixel with a background 4-neighbour (outside image = background)."""
    m = mask.astype(bool)
    up = np.ones_like(m, dtype=bool)
    down = np.ones_like(m, dtype=bool)
    left = np.ones_like(m, dtype=bool)
    right = np.ones_like(m, dtype=bool)
    up[1:] = m[:-1]
    down[:-1] = m[1:]
    left[:, 1:] = m[:, :-1]
    right[:, :-1] = m[:, 1:]
    # neighbour is background when ~neighbour; outside already treated as bg via True pads? 
    # Outside = background means "has bg neighbour" if on border next to outside.
    # For border pixels, missing neighbour outside the image counts as background,
    # so up[0], down[-1], left[:,0], right[:,-1] should be False (bg).
    up[0] = False
    down[-1] = False
    left[:, 0] = False
    right[:, -1] = False
    bg_nb = (~up) | (~down) | (~left) | (~right)
    return m & bg_nb


def _dilate5(binary: np.ndarray) -> np.ndarray:
    """5x5 square dilation via two 1-D max-filters (allowed: NumPy only)."""
    m = binary.astype(np.uint8)
    # pad 2
    p = np.pad(m, 2, mode="constant", constant_values=0)
    # horizontal max over 5
    h = np.maximum.reduce([p[:, i : i + m.shape[1]] for i in range(5)])
    # vertical max over 5
    v = np.maximum.reduce([h[i : i + m.shape[0], :] for i in range(5)])
    return v.astype(bool)


def boundary_f1(pred: np.ndarray, gt: np.ndarray) -> float:
    pb = boundary_map(pred)
    gb = boundary_map(gt)
    n_p = int(pb.sum())
    n_g = int(gb.sum())
    if n_p == 0 and n_g == 0:
        return 1.0
    if n_p == 0 or n_g == 0:
        return 0.0
    g_dil = _dilate5(gb)
    p_dil = _dilate5(pb)
    tp_p = int((pb & g_dil).sum())
    tp_g = int((gb & p_dil).sum())
    precision = tp_p / n_p
    recall = tp_g / n_g
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def quality(miou: float, bf1: float) -> float:
    return 0.70 * miou + 0.30 * bf1


def ces_score(q: float, t_cpu_ms: float, s_disk: int) -> float:
    if t_cpu_ms > 50.0:
        return 0.0
    return float(
        q
        * (
            1.0
            - 0.20 * (t_cpu_ms / 50.0)
            - 0.10 * min(1.0, s_disk / 4_194_304.0)
        )
    )
