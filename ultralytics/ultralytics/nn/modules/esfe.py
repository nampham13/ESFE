# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""ESFENet backbone modules."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ("ESFENet",)


class _CBS(nn.Module):
    """Conv + BatchNorm + SiLU."""

    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, p: int | None = None, g: int = 1):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class _DWConv(nn.Module):
    """Depthwise-separable downsampling block."""

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 2):
        super().__init__()
        p = k // 2
        self.dw = nn.Conv2d(c1, c1, k, s, p, groups=c1, bias=False)
        self.dw_bn = nn.BatchNorm2d(c1)
        self.pw = nn.Conv2d(c1, c2, 1, 1, 0, bias=False)
        self.pw_bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.dw_bn(self.dw(x)))
        x = self.act(self.pw_bn(self.pw(x)))
        return x


class _LightConv(nn.Module):
    """1x1 pointwise conv + depthwise conv."""

    def __init__(self, c1: int, c2: int, k: int = 5):
        super().__init__()
        p = k // 2
        self.pw = _CBS(c1, c2, k=1)
        self.dw = nn.Conv2d(c2, c2, k, 1, p, groups=c2, bias=False)
        self.dw_bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pw(x)
        return self.act(self.dw_bn(self.dw(x)))


class _GlobalResponseNorm(nn.Module):
    """Global Response Normalization (GRN)."""

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps)
        return self.gamma * (x * nx) + self.beta + x


class _HGStem(nn.Module):
    """Shallow stem with parallel pooled and convolved paths."""

    def __init__(self, c1: int = 3, cm: int = 16, c2: int = 32):
        super().__init__()
        self.stem1 = _CBS(c1, cm, k=3, s=2)
        self.stem2a = _CBS(cm, cm, k=2, s=1, p=0)
        self.stem2b = _CBS(cm, cm, k=2, s=1, p=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0)
        self.stem3 = _CBS(cm * 2, cm * 2, k=3, s=2)
        self.stem4 = _CBS(cm * 2, c2, k=1, s=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem1(x)
        x2 = F.pad(x, (0, 1, 0, 1))
        xa = self.stem2a(x2)
        xa = F.pad(xa, (0, 1, 0, 1))
        xa = self.stem2b(xa)
        xb = self.pool(F.pad(x, (0, 1, 0, 1)))
        x = torch.cat([xa, xb], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class _HGRNBlock(nn.Module):
    """Hierarchical GRN block."""

    def __init__(self, c1: int, c2: int, n: int = 6, light: bool = False, shortcut: bool = False, k: int = 5):
        super().__init__()
        self.shortcut = shortcut and c1 == c2
        cm = c2 // n if c2 >= n else c2
        layer = _LightConv if light else lambda ic, oc, _k=k: _CBS(ic, oc, k=_k)

        self.convs = nn.ModuleList()
        ch = c1
        for _ in range(n):
            self.convs.append(layer(ch, cm, k))
            ch = cm

        total_ch = c1 + n * cm
        self.grn = _GlobalResponseNorm(total_ch)
        self.act = nn.GELU()
        self.proj = _CBS(total_ch, c2, k=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [x]
        y = x
        for conv in self.convs:
            y = conv(y)
            feats.append(y)

        out = torch.cat(feats, dim=1)
        out = self.grn(out)
        out = self.act(out)
        out = self.proj(out)
        return out + x if self.shortcut else out


class _SPPF(nn.Module):
    """Fast SPP block."""

    def __init__(self, c1: int, c2: int, k: int = 5):
        super().__init__()
        cm = c1 // 2
        self.cv1 = _CBS(c1, cm, k=1)
        self.cv2 = _CBS(cm * 4, c2, k=1)
        self.pool = nn.MaxPool2d(k, stride=1, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], dim=1))


class ESFENet(nn.Module):
    """Efficient and Stable Feature Extraction Network returning P3, P4, P5 features."""

    def __init__(self, c1: int = 3):
        super().__init__()

        self.stem = _HGStem(c1=c1, cm=16, c2=32)

        self.stage1_block = _HGRNBlock(32, 32, n=4, light=False, shortcut=False, k=3)
        self.stage1_down = _DWConv(32, 32, k=3, s=2)

        self.stage2_block = _HGRNBlock(32, 128, n=4, light=False, shortcut=False, k=3)
        self.stage2_down = _DWConv(128, 128, k=3, s=2)

        self.stage3_block1 = _HGRNBlock(128, 256, n=6, light=False, shortcut=False, k=5)
        self.stage3_block2 = _HGRNBlock(256, 256, n=6, light=True, shortcut=False, k=5)
        self.stage3_block3 = _HGRNBlock(256, 256, n=6, light=True, shortcut=True, k=5)
        self.stage3_down = _DWConv(256, 256, k=3, s=2)

        self.stage4_block = _HGRNBlock(256, 256, n=6, light=True, shortcut=False, k=5)
        self.stage4_sppf = _SPPF(256, 256, k=5)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(x)

        x = self.stage1_block(x)
        x = self.stage1_down(x)

        x = self.stage2_block(x)
        p3 = x
        x = self.stage2_down(x)

        x = self.stage3_block1(x)
        x = self.stage3_block2(x)
        x = self.stage3_block3(x)
        p4 = x
        x = self.stage3_down(x)

        x = self.stage4_block(x)
        p5 = self.stage4_sppf(x)

        return [p3, p4, p5]
