"""Tiny U-Net FCN for cost-efficient salient-object segmentation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    def __init__(self, cin: int, cout: int, k: int = 3):
        super().__init__()
        p = k // 2
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, k, padding=p, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DWSeparable(nn.Module):
    """Depthwise-separable conv for a cheap receptive field."""

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cin, 3, padding=1, groups=cin, bias=False),
            nn.BatchNorm2d(cin),
            nn.ReLU(inplace=True),
            nn.Conv2d(cin, cout, 1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TinyUNet(nn.Module):
    """
    Compact encoder–decoder with U-Net skip concatenations.

    Internally runs at `internal_size` (default 128) then bilinear-upsamples
    logits to 256 so CPU inference stays well under the 50 ms budget.
    """

    def __init__(self, base: int = 16, internal_size: int = 128):
        super().__init__()
        self.internal_size = internal_size
        b = base
        self.enc1 = nn.Sequential(ConvBNReLU(3, b), DWSeparable(b, b))
        self.enc2 = nn.Sequential(ConvBNReLU(b, b * 2), DWSeparable(b * 2, b * 2))
        self.enc3 = nn.Sequential(ConvBNReLU(b * 2, b * 4), DWSeparable(b * 4, b * 4))
        self.bottleneck = nn.Sequential(
            ConvBNReLU(b * 4, b * 8),
            DWSeparable(b * 8, b * 8),
        )
        self.up3 = nn.ConvTranspose2d(b * 8, b * 4, 2, stride=2)
        self.dec3 = nn.Sequential(ConvBNReLU(b * 8, b * 4), DWSeparable(b * 4, b * 4))
        self.up2 = nn.ConvTranspose2d(b * 4, b * 2, 2, stride=2)
        self.dec2 = nn.Sequential(ConvBNReLU(b * 4, b * 2), DWSeparable(b * 2, b * 2))
        self.up1 = nn.ConvTranspose2d(b * 2, b, 2, stride=2)
        self.dec1 = nn.Sequential(ConvBNReLU(b * 2, b), DWSeparable(b, b))
        self.head = nn.Conv2d(b, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, 3, 256, 256) float in [0, 1]
        if x.shape[-1] != self.internal_size or x.shape[-2] != self.internal_size:
            x_in = F.interpolate(
                x,
                size=(self.internal_size, self.internal_size),
                mode="bilinear",
                align_corners=False,
            )
        else:
            x_in = x

        e1 = self.enc1(x_in)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        b = self.bottleneck(F.max_pool2d(e3, 2))

        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        logits = self.head(d1)

        if logits.shape[-1] != 256 or logits.shape[-2] != 256:
            logits = F.interpolate(
                logits, size=(256, 256), mode="bilinear", align_corners=False
            )
        return logits
