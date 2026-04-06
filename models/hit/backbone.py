"""
backbone.py – Lightweight MobileViT-XS Backbone
================================================
Performs feature extraction from template and search crops.

Why MobileViT-XS?
- Parameters: ~2.3M (fits within the ~10M budget)
- FLOPs: very lightweight, suitable for real-time applications
- Combines CNN (local features) with Transformer (global context)
- Pretrained on ImageNet → faster convergence

Output:
  - template features: (B, C, H/16, W/16)
  - search features:   (B, C, H/16, W/16)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Basic building blocks
# ─────────────────────────────────────────────────────────────────────────────

class ConvBNAct(nn.Module):
    """Conv → BN → Activation  (standard mobile block)"""
    def __init__(self, in_ch, out_ch, kernel=3, stride=1,
                 padding=1, groups=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride,
                              padding, groups=groups, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class InvertedResidual(nn.Module):
    """
    MobileNetV2-style Inverted Residual block.
    expand → depthwise → project
    """
    def __init__(self, in_ch, out_ch, stride=1, expand_ratio=4):
        super().__init__()
        mid_ch = int(in_ch * expand_ratio)
        self.use_res = (stride == 1 and in_ch == out_ch)

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNAct(in_ch, mid_ch, kernel=1, padding=0))
        layers += [
            ConvBNAct(mid_ch, mid_ch, stride=stride, groups=mid_ch),  # DW
            ConvBNAct(mid_ch, out_ch, kernel=1, padding=0, act=False), # PW
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)
        if self.use_res:
            out = out + x
        return out


# ─────────────────────────────────────────────────────────────────────────────
# MobileViT block  (Local + Global processing)
# ─────────────────────────────────────────────────────────────────────────────

class MultiHeadSelfAttention(nn.Module):
    """Efficient MHSA for MobileViT patches."""
    def __init__(self, dim, heads=1, dropout=0.0):
        super().__init__()
        self.heads   = heads
        self.scale   = (dim // heads) ** -0.5
        self.qkv     = nn.Linear(dim, dim * 3, bias=False)
        self.proj    = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        h = self.heads
        qkv = self.qkv(x).reshape(B, N, 3, h, C // h).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.dropout(attn.softmax(dim=-1))

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block: MHSA + FFN."""
    def __init__(self, dim, heads=1, mlp_ratio=2, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = MultiHeadSelfAttention(dim, heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn   = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class MobileViTBlock(nn.Module):
    """
    MobileViT block:
    1. Local conv processing
    2. Unfold into patches
    3. Global transformer across patches
    4. Fold back
    5. Fusion conv
    """
    def __init__(self, in_ch, dim, patch_size=2, depth=2, heads=1):
        super().__init__()
        self.patch_size = patch_size
        self.ph = patch_size
        self.pw = patch_size

        # Local conv
        self.local_rep = nn.Sequential(
            ConvBNAct(in_ch, in_ch),
            ConvBNAct(in_ch, dim, kernel=1, padding=0, act=False),
        )

        # Global transformer
        self.transformer = nn.Sequential(
            *[TransformerBlock(dim, heads) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)

        # Projection back
        self.proj = ConvBNAct(dim, in_ch, kernel=1, padding=0, act=False)

        # Fusion
        self.fusion = ConvBNAct(2 * in_ch, in_ch, kernel=1, padding=0)

    def forward(self, x):
        B, C, H, W = x.shape
        ph, pw = self.ph, self.pw

        # Pad if needed
        pad_h = (ph - H % ph) % ph
        pad_w = (pw - W % pw) % pw
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        _, _, Hp, Wp = x.shape
        nph = Hp // ph
        npw = Wp // pw
        num_patches = nph * npw

        # Local representation
        y = self.local_rep(x)          # (B, dim, Hp, Wp)
        dim = y.shape[1]

        # Unfold: (B, dim, nph, ph, npw, pw) → (B*num_patches, ph*pw, dim)
        y = y.reshape(B, dim, nph, ph, npw, pw)
        y = y.permute(0, 2, 4, 3, 5, 1)           # (B, nph, npw, ph, pw, dim)
        y = y.reshape(B * num_patches, ph * pw, dim)

        # Global attention
        y = self.transformer(y)
        y = self.norm(y)

        # Fold back: (B, dim, Hp, Wp)
        y = y.reshape(B, nph, npw, ph, pw, dim)
        y = y.permute(0, 5, 1, 3, 2, 4)           # (B, dim, nph, ph, npw, pw)
        y = y.reshape(B, dim, Hp, Wp)

        # Project back to in_ch
        y = self.proj(y)

        # Remove padding
        if pad_h > 0 or pad_w > 0:
            y = y[:, :, :H, :W]
            x = x[:, :, :H, :W]

        # Fusion
        out = self.fusion(torch.cat([x, y], dim=1))
        return out


# ─────────────────────────────────────────────────────────────────────────────
# MobileViT-XS Backbone
# ─────────────────────────────────────────────────────────────────────────────

class MobileViTBackbone(nn.Module):
    """
    MobileViT-XS backbone adapted for UAV tracking.

    Architecture:
      Stem → MV2×2 → MV2×3 → MViT → MV2×4 → MViT → MV2×3 → MViT
      Total params: ~2.3M

    Input:  (B, 3, H, W)   — normalised RGB
    Output: (B, 96, H/16, W/16)  — rich semantic features

    Config (XS):
      channels = [16, 32, 48, 64, 80, 96]
      dims     = [96, 120, 144]
    """

    def __init__(self, in_ch=3, dropout=0.0):
        super().__init__()

        # ── Stem ──────────────────────────────────────────────────────────────
        self.stem = ConvBNAct(in_ch, 16, stride=2)   # /2

        # ── Stage 1 — MV2 blocks ──────────────────────────────────────────────
        self.stage1 = nn.Sequential(
            InvertedResidual(16, 32, stride=1, expand_ratio=4),
        )   # /2 total

        # ── Stage 2 — MV2 blocks ──────────────────────────────────────────────
        self.stage2 = nn.Sequential(
            InvertedResidual(32, 48, stride=2, expand_ratio=4),
            InvertedResidual(48, 48, stride=1, expand_ratio=4),
            InvertedResidual(48, 48, stride=1, expand_ratio=4),
        )   # /4 total

        # ── Stage 3 — MobileViT ───────────────────────────────────────────────
        self.stage3 = nn.Sequential(
            InvertedResidual(48, 64, stride=2, expand_ratio=4),
            MobileViTBlock(64, dim=96, patch_size=2, depth=2, heads=1),
        )   # /8 total

        # ── Stage 4 — MV2 + MobileViT ─────────────────────────────────────────
        self.stage4 = nn.Sequential(
            InvertedResidual(64, 80, stride=2, expand_ratio=4),
            MobileViTBlock(80, dim=120, patch_size=2, depth=4, heads=2),
        )   # /16 total

        # ── Stage 5 — MV2 + MobileViT ─────────────────────────────────────────
        self.stage5 = nn.Sequential(
            InvertedResidual(80, 96, stride=2, expand_ratio=4),
            MobileViTBlock(96, dim=144, patch_size=2, depth=3, heads=2),
        )   # /32 total

        # ── Feature neck — reduce stride/32 to stride/16 output ───────────────
        # For tracking we want richer spatial resolution → upsample stage5
        self.neck = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBNAct(96, 96, kernel=1, padding=0),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W)
        Returns:
            feat: (B, 96, H//16, W//16)
        """
        x = self.stem(x)     # /2
        x = self.stage1(x)   # /2
        x = self.stage2(x)   # /4
        x = self.stage3(x)   # /8
        x = self.stage4(x)   # /16
        x = self.stage5(x)   # /32
        x = self.neck(x)     # /16  ← final output
        return x

    @property
    def out_channels(self):
        return 96

    @property
    def stride(self):
        return 16


# ─────────────────────────────────────────────────────────────────────────────
# Sanity check
# ─────────────────────────────────────────────────────────────────────────────

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    import time

    backbone = MobileViTBackbone()
    backbone.eval()

    # Template: 128×128,  Search: 256×256
    template = torch.randn(1, 3, 128, 128)
    search   = torch.randn(1, 3, 256, 256)

    with torch.no_grad():
        t0 = time.time()
        t_feat = backbone(template)
        s_feat = backbone(search)
        elapsed = (time.time() - t0) * 1000

    params = count_parameters(backbone)

    print("=" * 50)
    print("MobileViT-XS Backbone — Sanity Check")
    print("=" * 50)
    print(f"Template input:  {tuple(template.shape)}")
    print(f"Template output: {tuple(t_feat.shape)}")
    print(f"Search input:    {tuple(search.shape)}")
    print(f"Search output:   {tuple(s_feat.shape)}")
    print(f"Parameters:      {params:,}  ({params/1e6:.2f}M)")
    print(f"Inference time:  {elapsed:.1f} ms  (CPU)")
    print("=" * 50)

    # Budget checks
    budget_params = 10_000_000
    print(f"\nBudget checks:")
    print(f"  Params: {params/1e6:.2f}M / 10M  {'✓' if params < budget_params else '✗'}")
    print(f"  Output stride: {backbone.stride}  (expected 16)")
    print(f"  Output channels: {backbone.out_channels}  (expected 96)")