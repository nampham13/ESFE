"""
ESFENet - Efficient and Stable Feature Extraction Network
Backbone for UCM-Net (Universal Underground Coal Mine Object Detection Network)

Based on: "Enhancing Object Detection in Underground Mines: UCM-Net and
Self-Supervised Pre-Training" (Sensors 2025, 25, 2103)

Architecture:
  - HGStem: initial feature extraction (5 conv layers + max pool)
  - HGRNBlock: hierarchical feature extraction with GRN + GELU
  - DWConv: depthwise separable convolution for downsampling
  - SPPF: spatial pyramid pooling fast (Stage 4)

4-stage design:
  Stage 0: HGStem          640x640x3  -> 160x160x16 -> 160x160x32
  Stage 1: HGRNBlock       160x160x32 -> 160x160x32
           DWConv           160x160x32 ->  80x80x32
  Stage 2: HGRNBlock        80x80x32  ->  80x80x128
           DWConv            80x80x128 ->  40x40x128
  Stage 3: HGRNBlock x3    40x40x128  ->  40x40x256
           DWConv            40x40x256 ->  20x20x256
  Stage 4: HGRNBlock        20x20x256 ->  20x20x256
           SPPF              20x20x256 ->  20x20x256
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building Blocks
# ---------------------------------------------------------------------------

class CBS(nn.Module):
    """Conv + BatchNorm + SiLU (standard conv block)."""

    def __init__(self, in_ch, out_ch, k=1, s=1, p=None, g=1):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, groups=g, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DWConv(nn.Module):
    """
    Depthwise Separable Convolution used for downsampling.
    Decomposes conv into: depthwise (per-channel) + pointwise (1x1).
    Stride=2 halves spatial dims.
    """

    def __init__(self, in_ch, out_ch, k=3, s=2):
        super().__init__()
        p = k // 2
        # Depthwise: each channel convolved independently
        self.dw = nn.Conv2d(in_ch, in_ch, k, s, p, groups=in_ch, bias=False)
        self.dw_bn  = nn.BatchNorm2d(in_ch)
        # Pointwise: 1x1 to mix channels
        self.pw = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)
        self.pw_bn  = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.act(self.dw_bn(self.dw(x)))
        x = self.act(self.pw_bn(self.pw(x)))
        return x


class LightConv(nn.Module):
    """
    Lightweight convolution = 1x1 conv + depthwise conv (no stride).
    Used inside HGRNBlock when light=True.
    """

    def __init__(self, in_ch, out_ch, k=5):
        super().__init__()
        p = k // 2
        self.pw = CBS(in_ch, out_ch, k=1)
        self.dw = nn.Conv2d(out_ch, out_ch, k, 1, p, groups=out_ch, bias=False)
        self.dw_bn = nn.BatchNorm2d(out_ch)
        self.act   = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.pw(x)
        return self.act(self.dw_bn(self.dw(x)))


# ---------------------------------------------------------------------------
# Global Response Normalization (GRN)
# ---------------------------------------------------------------------------

class GlobalResponseNorm(nn.Module):
    """
    GRN normalises each channel by its global L2 norm relative to the mean
    norm across channels, then applies learnable scale (gamma) and shift (beta).

    For a feature map x of shape (B, C, H, W):
      Gx[b,c] = ||x[b,c,:,:]||_2            (per-channel L2 norm)
      Nx[b,c] = Gx[b,c] / (mean_c(Gx) + eps)
      y = gamma * (x * Nx) + beta + x        (residual scaling)
    """

    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.eps   = eps
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.beta  = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        # Global L2 norm per channel: (B, C, 1, 1)
        gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
        # Normalise by mean across channel dim
        nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps)
        return self.gamma * (x * nx) + self.beta + x


# ---------------------------------------------------------------------------
# HGStem
# ---------------------------------------------------------------------------

class HGStem(nn.Module):
    """
    HGStem: efficient shallow feature extractor.
    Two parallel paths from stem1 output:
      Path A: stem2a -> stem2b  (local features via 2x2 conv)
      Path B: MaxPool2d         (global features)
    Concatenated -> stem3 (3x3) -> stem4 (1x1) -> Fs

    Input:  (B, 3,   640, 640)
    Output: (B, out_ch, 160, 160)  [two stride-2 ops inside]
    """

    def __init__(self, in_ch=3, mid_ch=16, out_ch=32):
        super().__init__()
        # stem1: 3x3, stride 2  =>  320x320
        self.stem1  = CBS(in_ch,   mid_ch, k=3, s=2)
        # stem2a/b: 2x2 (no bias), stride 1 then stride 2  =>  160x160
        self.stem2a = CBS(mid_ch,  mid_ch, k=2, s=1, p=0)
        self.stem2b = CBS(mid_ch,  mid_ch, k=2, s=1, p=0)
        self.pool   = nn.MaxPool2d(kernel_size=2, stride=1, padding=0)
        # After cat: mid_ch * 2
        self.stem3  = CBS(mid_ch * 2, mid_ch * 2, k=3, s=2)
        self.stem4  = CBS(mid_ch * 2, out_ch, k=1, s=1)

    def forward(self, x):
        x = self.stem1(x)           # (B, mid,  320, 320)

        # Align spatial size: both paths produce 159x159 from 320x320
        # stem2 path needs padding to match pool path output size
        x2 = F.pad(x, (0, 1, 0, 1))   # -> 321x321
        xa = self.stem2a(x2)           # 320x320
        xa = F.pad(xa, (0, 1, 0, 1))
        xa = self.stem2b(xa)           # 320x320

        xb = self.pool(F.pad(x, (0, 1, 0, 1)))   # 320x320

        x = torch.cat([xa, xb], dim=1)   # (B, mid*2, 320, 320)
        x = self.stem3(x)                # (B, mid*2, 160, 160)  stride-2
        x = self.stem4(x)                # (B, out_ch, 160, 160)
        return x


# ---------------------------------------------------------------------------
# HGRNBlock
# ---------------------------------------------------------------------------

class HGRNBlock(nn.Module):
    """
    Hierarchical Global Response Normalization Block.
    Redesigns HGBlock (from HGNetv2) by replacing standard conv with
    GRN + GELU for better stability in mine environments.

    Internal layers 5,6,7,9 use LightConv when light=True.
    Optional residual shortcut when in_ch == out_ch and shortcut=True.

    Forward:
      1. N convolution branches (CBS or LightConv), each taking previous output
      2. Concatenate all intermediate feature maps
      3. GRN normalisation
      4. GELU activation
      5. Squeeze-excitation-style 1x1 CBS to project to out_ch
      6. Optional shortcut add
    """

    def __init__(self, in_ch, out_ch, n=6, light=False, shortcut=False, k=5):
        super().__init__()
        self.shortcut = shortcut and (in_ch == out_ch)
        mid_ch = out_ch // n if out_ch >= n else out_ch

        ConvLayer = LightConv if light else lambda ic, oc, k=k: CBS(ic, oc, k=k)

        # Build n conv layers; each takes the previous layer's output
        self.convs = nn.ModuleList()
        ch = in_ch
        for i in range(n):
            self.convs.append(ConvLayer(ch, mid_ch, k))
            ch = mid_ch

        # After concat: in_ch + n * mid_ch channels
        total_ch = in_ch + n * mid_ch
        self.grn  = GlobalResponseNorm(total_ch)
        self.act  = nn.GELU()
        self.proj = CBS(total_ch, out_ch, k=1)

    def forward(self, x):
        feats = [x]
        y = x
        for conv in self.convs:
            y = conv(y)
            feats.append(y)

        out = torch.cat(feats, dim=1)   # concat all intermediate maps
        out = self.grn(out)
        out = self.act(out)
        out = self.proj(out)

        if self.shortcut:
            out = out + x
        return out


# ---------------------------------------------------------------------------
# SPPF (Spatial Pyramid Pooling - Fast)
# ---------------------------------------------------------------------------

class SPPF(nn.Module):
    """
    Fast SPP: apply max-pool k=5 three times sequentially, then concat
    all four tensors (original + 3 pooled), followed by a 1x1 conv.
    Equivalent to SPP with pools of size 5, 9, 13 but faster.
    """

    def __init__(self, in_ch, out_ch, k=5):
        super().__init__()
        mid = in_ch // 2
        self.cv1  = CBS(in_ch, mid, k=1)
        self.cv2  = CBS(mid * 4, out_ch, k=1)
        self.pool = nn.MaxPool2d(k, stride=1, padding=k // 2)

    def forward(self, x):
        x  = self.cv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], dim=1))


# ---------------------------------------------------------------------------
# ESFENet Backbone
# ---------------------------------------------------------------------------

class ESFENet(nn.Module):
    """
    Efficient and Stable Feature Extraction Network (ESFENet).

    4-Stage architecture:
      Stage 0 (HGStem):
        640x640x3  -> 160x160x32

      Stage 1:
        HGRNBlock(32->32,  light=False, shortcut=False, k=3)  160x160x32
        DWConv(32->32, s=2)                                   -> 80x80x32

      Stage 2:
        HGRNBlock(32->128, light=False, shortcut=False, k=3)   80x80x128
        DWConv(128->128, s=2)                                 -> 40x40x128

      Stage 3:
        HGRNBlock(128->256, light=False, shortcut=False, k=5)  40x40x256
        HGRNBlock(256->256, light=True,  shortcut=False, k=5)  40x40x256
        HGRNBlock(256->256, light=True,  shortcut=True,  k=5)  40x40x256
        DWConv(256->256, s=2)                                 -> 20x20x256

      Stage 4:
        HGRNBlock(256->256, light=True, shortcut=False, k=5)   20x20x256
        SPPF(256->256, k=5)                                    20x20x256

    Output feature maps (matching YOLO neck inputs at 3 scales):
      P3: 80x80x128   (after Stage 2 HGRNBlock, before DWConv)
      P4: 40x40x256   (after Stage 3 HGRNBlocks, before DWConv)
      P5: 20x20x256   (after Stage 4 SPPF)
    """

    def __init__(self, in_channels=3):
        super().__init__()

        # ---- Stage 0: HGStem ----
        self.stem = HGStem(in_ch=in_channels, mid_ch=16, out_ch=32)
        # out: (B, 32, 160, 160)

        # ---- Stage 1 ----
        self.stage1_block = HGRNBlock(32, 32, n=4, light=False, shortcut=False, k=3)
        self.stage1_down  = DWConv(32, 32, k=3, s=2)
        # out: (B, 32, 80, 80)

        # ---- Stage 2 ----
        self.stage2_block = HGRNBlock(32, 128, n=4, light=False, shortcut=False, k=3)
        self.stage2_down  = DWConv(128, 128, k=3, s=2)
        # out: (B, 128, 40, 40)

        # ---- Stage 3 ----
        self.stage3_block1 = HGRNBlock(128, 256, n=6, light=False, shortcut=False, k=5)
        self.stage3_block2 = HGRNBlock(256, 256, n=6, light=True,  shortcut=False, k=5)
        self.stage3_block3 = HGRNBlock(256, 256, n=6, light=True,  shortcut=True,  k=5)
        self.stage3_down   = DWConv(256, 256, k=3, s=2)
        # out: (B, 256, 20, 20)

        # ---- Stage 4 ----
        self.stage4_block = HGRNBlock(256, 256, n=6, light=True, shortcut=False, k=5)
        self.stage4_sppf  = SPPF(256, 256, k=5)
        # out: (B, 256, 20, 20)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Returns:
            p3: (B, 128, H/8,  W/8)   -- small object scale
            p4: (B, 256, H/16, W/16)  -- medium object scale
            p5: (B, 256, H/32, W/32)  -- large object scale
        """
        # Stage 0
        x = self.stem(x)                  # 160x160x32

        # Stage 1
        x = self.stage1_block(x)          # 160x160x32
        x = self.stage1_down(x)           # 80x80x32

        # Stage 2
        x  = self.stage2_block(x)         # 80x80x128
        p3 = x                            # <- P3 feature map
        x  = self.stage2_down(x)          # 40x40x128

        # Stage 3
        x  = self.stage3_block1(x)        # 40x40x256
        x  = self.stage3_block2(x)        # 40x40x256
        x  = self.stage3_block3(x)        # 40x40x256
        p4 = x                            # <- P4 feature map
        x  = self.stage3_down(x)          # 20x20x256

        # Stage 4
        x  = self.stage4_block(x)         # 20x20x256
        p5 = self.stage4_sppf(x)          # 20x20x256  <- P5 feature map

        return p3, p4, p5


# ---------------------------------------------------------------------------
# Parameter & FLOPs summary helper
# ---------------------------------------------------------------------------

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model = ESFENet(in_channels=3)
    model.eval()

    total, trainable = count_parameters(model)
    print(f"Total parameters    : {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Parameter size (MB) : {total * 4 / 1024**2:.2f}")

    # Forward pass with dummy input
    dummy = torch.zeros(1, 3, 640, 640)
    with torch.no_grad():
        p3, p4, p5 = model(dummy)

    print("\nOutput feature maps:")
    print(f"  P3 (80x80)  : {p3.shape}")
    print(f"  P4 (40x40)  : {p4.shape}")
    print(f"  P5 (20x20)  : {p5.shape}")

    # Optional: FLOPs estimate via thop
    try:
        from thop import profile
        prof = profile(model, inputs=(dummy,), verbose=False)
        flops = prof[0]
        params = prof[1]
        print(f"\nFLOPs : {flops / 1e9:.2f} G")
        print(f"Params: {params / 1e6:.2f} M")
    except ImportError:
        print("\n(Install `thop` for FLOPs counting: pip install thop)")