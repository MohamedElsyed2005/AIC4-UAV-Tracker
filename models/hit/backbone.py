"""
backbone.py  -  AlexNet Backbone for HiT Tracker
==================================================
Replaces MobileViT with a lightweight AlexNet-based feature extractor.

Why AlexNet?
  - Much lighter than MobileViT (~1.9M params vs 5M+ for MobileViT)
  - Fast inference (pure conv, no attention overhead)
  - Pretrained ImageNet weights available via torchvision
  - No timm dependency (simpler deployment)

Architecture:
  Input: (B, 3, H, W)
  AlexNet features[:8]           -> non-power-of-2 spatial size (15x15 / 7x7)
  AdaptiveAvgPool2d(target_hw)   -> exact target size (16x16 / 8x8)  [FIX]
  extra_conv (3x3, stride=1)     -> channel reduction, same spatial size
  adapter    (1x1)               -> 96 channels
  Output: (B, 96, target_hw, target_hw)

WHY AdaptiveAvgPool2d IS NEEDED -- AlexNet MaxPool arithmetic:
  AlexNet uses MaxPool2d(kernel=3, stride=2, padding=0).
  With floor division, 256x256 input gives:
    Conv(11,s=4,p=2): 63  ->  MaxPool(3,s=2): 31  ->  MaxPool(3,s=2): 15
  And 128x128 gives:
    Conv(11,s=4,p=2): 31  ->  MaxPool(3,s=2): 15  ->  MaxPool(3,s=2): 7
  These are 15x15 and 7x7, NOT 16x16 and 8x8.
  The assertion s_feat.shape == (B,96,16,16) therefore fires.

  Fix: insert F.adaptive_avg_pool2d(x, target_hw) immediately after
  features[:8] to normalize the spatial size to the configured target.
  This is a standard, stable PyTorch operation with no learned parameters.

Output:
  - Channels: 96
  - Spatial: template_hw x template_hw  for template inputs (default 8)
             search_hw   x search_hw    for search inputs   (default 16)
  - Compatible with CrossAttentionTransformer (in_channels=96)
  - Ns = search_hw^2 = 256  (perfect square, required by TrackingHead)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional


# -------------------------------------------------------------------------
# AlexNet Backbone
# -------------------------------------------------------------------------

class AlexNetBackbone(nn.Module):
    """
    Lightweight AlexNet-based feature extractor with adaptive spatial pooling.

    The backbone uses AlexNet features[:8] (stride ~16, 384 channels) followed
    by AdaptiveAvgPool2d to normalize the spatial size to the configured target,
    a 3x3 channel-reduction conv, and a 1x1 projection to 96 channels.

    KEY FIX:
      AlexNet MaxPool layers use floor division so 256x256->15x15 and
      128x128->7x7 (not the expected 16x16 and 8x8).  A previous attempt to
      fix this by setting extra_conv stride=1 was correct in principle but
      still produced 15/7 instead of 16/8.  This version inserts
      F.adaptive_avg_pool2d(x, target_hw) after features[:8] to force the
      correct output size regardless of the exact MaxPool arithmetic.

    Target size selection:
      The backbone infers which target size to use from the input resolution:
        input_hw <= 160  ->  template branch  ->  target = template_hw (8)
        input_hw  > 160  ->  search  branch   ->  target = search_hw   (16)
      This covers the standard 128/256 split.  Change template_hw/search_hw
      in the constructor if you change input resolutions.

    Total params: ~1.9M
    Nominal stride: 16
    Output channels: 96

    Args:
        pretrained:   load ImageNet weights for AlexNet layers (default True)
        template_hw:  output spatial size for template inputs (default 8)
        search_hw:    output spatial size for search inputs   (default 16)
    """

    def __init__(self,
                 pretrained:  bool = True,
                 template_hw: int  = 8,
                 search_hw:   int  = 16):
        super().__init__()

        self._template_hw = template_hw
        self._search_hw   = search_hw

        # Load AlexNet pretrained backbone
        alexnet = models.alexnet(
            weights=models.AlexNet_Weights.IMAGENET1K_V1 if pretrained else None
        )

        # AlexNet features[:8] (indices 0-7):
        #   [0] Conv2d(3,64, 11,stride=4,padding=2)   stride~4   64ch
        #   [1] ReLU
        #   [2] MaxPool2d(3, stride=2)                stride~8   64ch
        #   [3] Conv2d(64,192, 5,padding=2)            stride~8  192ch
        #   [4] ReLU
        #   [5] MaxPool2d(3, stride=2)                stride~16 192ch
        #   [6] Conv2d(192,384, 3,padding=1)           stride~16 384ch
        #   [7] ReLU                                   OUTPUT 384ch
        #
        # Actual spatial output (floor-division MaxPool, no padding):
        #   256x256 -> 15x15   (NOT 16x16 -- fixed by adaptive_avg_pool2d below)
        #   128x128 -> 7x7     (NOT 8x8   -- fixed by adaptive_avg_pool2d below)
        self.features = alexnet.features[:8]   # (B, 384, ~H/16, ~W/16)

        # Channel reduction conv, stride=1: spatial size stays at target_hw
        # (adaptive pool above already set the exact size).
        self.extra_conv = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Project to 96 channels (required by transformer)
        self.adapter = nn.Sequential(
            nn.Conv2d(256, 96, kernel_size=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
        )

        self._init_weights()

    def _init_weights(self):
        for m in [self.extra_conv, self.adapter]:
            for layer in m.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        layer.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(layer, nn.BatchNorm2d):
                    nn.init.ones_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) input image, ImageNet-normalized
               H <= 160 -> template branch -> output (B, 96, template_hw, template_hw)
               H  > 160 -> search  branch  -> output (B, 96, search_hw,   search_hw)

        Returns:
            feat: (B, 96, target_hw, target_hw)
                  template 128x128 -> (B, 96, 8,  8)
                  search   256x256 -> (B, 96, 16, 16)
        """
        in_hw     = x.shape[2]
        target_hw = self._template_hw if in_hw <= 160 else self._search_hw

        x = self.features(x)           # (B, 384, ~in_hw/16, ~in_hw/16)

        # Normalize to exact target size.
        # AlexNet MaxPool floor-division: 256->15 and 128->7 instead of 16, 8.
        # adaptive_avg_pool2d corrects this without extra parameters.
        if x.shape[2] != target_hw or x.shape[3] != target_hw:
            x = F.adaptive_avg_pool2d(x, target_hw)

        x = self.extra_conv(x)         # (B, 256, target_hw, target_hw)
        x = self.adapter(x)            # (B, 96,  target_hw, target_hw)
        return x

    def freeze(self):
        """Freeze AlexNet base features (keep extra_conv and adapter trainable)."""
        for param in self.features.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True

    def freeze_all(self):
        """Freeze everything including adapter."""
        for param in self.parameters():
            param.requires_grad = False

    @classmethod
    def pretrained(cls, source: Optional[str] = "imagenet",
                   ckpt_path: Optional[str] = None) -> "AlexNetBackbone":
        """
        Factory method to build backbone with optional pretrained weights.

        Args:
            source:    'imagenet' | 'local' | 'none'
            ckpt_path: path to local .pth checkpoint (required if source='local')
        """
        if source is None or source == "none":
            return cls(pretrained=False)
        if source in ("imagenet", "timm"):
            return cls(pretrained=True)
        if source == "local":
            if ckpt_path is None:
                raise ValueError("ckpt_path must be provided for source='local'")
            model = cls(pretrained=False)
            ckpt  = torch.load(ckpt_path, map_location="cpu")
            state = ckpt.get("model_state", ckpt.get("state_dict", ckpt))
            model.load_state_dict(state, strict=False)
            return model
        raise ValueError(f"Unknown pretrained source: {source!r}")

    @property
    def out_channels(self) -> int:
        return 96

    @property
    def stride(self) -> int:
        """Nominal stride (used by HiTConfig to compute feature map sizes)."""
        return 16


# -------------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------------

# Keep backward-compatible alias
MobileViTBackbone = AlexNetBackbone


def count_parameters(model: nn.Module) -> int:
    """Count total (not just trainable) parameters."""
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model: nn.Module) -> int:
    """Count only trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# -------------------------------------------------------------------------
# Sanity check
# -------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    bb = AlexNetBackbone(pretrained=False)
    bb.eval()

    template = torch.randn(2, 3, 128, 128)
    search   = torch.randn(2, 3, 256, 256)

    with torch.no_grad():
        t0     = time.time()
        t_feat = bb(template)
        s_feat = bb(search)
        elapsed = (time.time() - t0) * 1000

    params = count_parameters(bb)

    print("=" * 60)
    print("AlexNet Backbone -- Sanity Check (AdaptiveAvgPool2d fix)")
    print("=" * 60)
    print(f"Template: {tuple(template.shape)} -> {tuple(t_feat.shape)}")
    print(f"Search:   {tuple(search.shape)} -> {tuple(s_feat.shape)}")
    print(f"Params:   {params:,}  ({params/1e6:.2f}M)")
    print(f"Time:     {elapsed:.1f} ms (CPU)")
    print(f"Budget:   {'OK' if params < 10e6 else 'OVER'}  (<10M)")
    print(f"Channels: {bb.out_channels}  (expected 96)")

    assert t_feat.shape == (2, 96, 8, 8),   f"FAIL template feat {t_feat.shape}"
    assert s_feat.shape == (2, 96, 16, 16), f"FAIL search feat   {s_feat.shape}"
    print("\nShape assertions: PASSED")

    # Test freeze/unfreeze
    bb.freeze()
    frozen = count_trainable_parameters(bb)
    bb.unfreeze()
    all_p  = count_parameters(bb)
    assert frozen < all_p, "Freeze did not reduce trainable count"
    print(f"Freeze test: {frozen:,} trainable when frozen (full={all_p:,}): PASSED")
    print("=" * 60)