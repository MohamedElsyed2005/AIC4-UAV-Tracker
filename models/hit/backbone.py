"""
backbone.py  –  MobileViT-XS Backbone  (v2)
============================================
Changes vs v1
─────────────
1. pretrained(source, ckpt_path) class method:
   Builds the backbone and optionally loads pretrained weights.
   source = 'timm'     → load from timm mobilevit_xs
   source = 'local'    → load from a local .pth checkpoint
   source = None/'none' → random init (original behaviour)

2. freeze_backbone() / unfreeze_backbone():
   Call freeze_backbone() at the start of training when using pretrained
   weights.  Unfreeze after a few warmup epochs so the backbone adapts.
   This is handled automatically by HiTTracker if --pretrained_backbone is set.

3. No architectural changes — the MobileViT-XS structure is identical
   to v1, ensuring full compatibility with the transformer head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Basic building blocks
# ─────────────────────────────────────────────────────────────────────────────

class ConvBNAct(nn.Module):
    """Conv → BN → Activation (standard mobile block)."""
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
    """MobileNetV2-style Inverted Residual: expand → depthwise → project."""
    def __init__(self, in_ch, out_ch, stride=1, expand_ratio=4):
        super().__init__()
        mid_ch = int(in_ch * expand_ratio)
        self.use_res = (stride == 1 and in_ch == out_ch)
        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNAct(in_ch, mid_ch, kernel=1, padding=0))
        layers += [
            ConvBNAct(mid_ch, mid_ch, stride=stride, groups=mid_ch),
            ConvBNAct(mid_ch, out_ch, kernel=1, padding=0, act=False),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)
        return out + x if self.use_res else out


# ─────────────────────────────────────────────────────────────────────────────
# MobileViT block
# ─────────────────────────────────────────────────────────────────────────────

class MultiHeadSelfAttention(nn.Module):
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
        attn = self.dropout((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        return self.proj((attn @ v).transpose(1, 2).reshape(B, N, C))


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=1, mlp_ratio=2, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = MultiHeadSelfAttention(dim, heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn   = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio), nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim), nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class MobileViTBlock(nn.Module):
    """Local CNN + global transformer on patches."""
    def __init__(self, in_ch, dim, patch_size=2, depth=2, heads=1):
        super().__init__()
        self.ph = self.pw = patch_size
        self.local_rep = nn.Sequential(
            ConvBNAct(in_ch, in_ch),
            ConvBNAct(in_ch, dim, kernel=1, padding=0, act=False),
        )
        self.transformer = nn.Sequential(
            *[TransformerBlock(dim, heads) for _ in range(depth)]
        )
        self.norm  = nn.LayerNorm(dim)
        self.proj  = ConvBNAct(dim, in_ch, kernel=1, padding=0, act=False)
        self.fusion = ConvBNAct(2 * in_ch, in_ch, kernel=1, padding=0)

    def forward(self, x):
        B, C, H, W = x.shape
        ph, pw = self.ph, self.pw
        pad_h = (ph - H % ph) % ph
        pad_w = (pw - W % pw) % pw
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        _, _, Hp, Wp = x.shape
        nph, npw = Hp // ph, Wp // pw
        num_patches = nph * npw

        y = self.local_rep(x)
        dim = y.shape[1]
        y = y.reshape(B, dim, nph, ph, npw, pw)
        y = y.permute(0, 2, 4, 3, 5, 1).reshape(B * num_patches, ph * pw, dim)
        y = self.norm(self.transformer(y))
        y = y.reshape(B, nph, npw, ph, pw, dim)
        y = y.permute(0, 5, 1, 3, 2, 4).reshape(B, dim, Hp, Wp)
        y = self.proj(y)
        if pad_h or pad_w:
            y = y[:, :, :H, :W]
            x = x[:, :, :H, :W]
        return self.fusion(torch.cat([x, y], dim=1))


# ─────────────────────────────────────────────────────────────────────────────
# MobileViT-XS Backbone (v2: + pretrained loading)
# ─────────────────────────────────────────────────────────────────────────────

class MobileViTBackbone(nn.Module):
    """
    MobileViT-XS backbone for UAV tracking.

    Architecture:  Stem → Stage1-5 → neck upsample
    Output stride: 16
    Output channels: 96
    Parameters: ~2.3M
    """

    def __init__(self, in_ch=3, dropout=0.0):
        super().__init__()

        self.stem   = ConvBNAct(in_ch, 16, stride=2)

        self.stage1 = nn.Sequential(
            InvertedResidual(16, 32, stride=1, expand_ratio=4),
        )
        self.stage2 = nn.Sequential(
            InvertedResidual(32, 48, stride=2, expand_ratio=4),
            InvertedResidual(48, 48, stride=1, expand_ratio=4),
            InvertedResidual(48, 48, stride=1, expand_ratio=4),
        )
        self.stage3 = nn.Sequential(
            InvertedResidual(48, 64, stride=2, expand_ratio=4),
            MobileViTBlock(64, dim=96,  patch_size=2, depth=2, heads=1),
        )
        self.stage4 = nn.Sequential(
            InvertedResidual(64, 80, stride=2, expand_ratio=4),
            MobileViTBlock(80, dim=120, patch_size=2, depth=4, heads=2),
        )
        self.stage5 = nn.Sequential(
            InvertedResidual(80, 96, stride=2, expand_ratio=4),
            MobileViTBlock(96, dim=144, patch_size=2, depth=3, heads=2),
        )
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
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.neck(x)
        return x

    @property
    def out_channels(self):
        return 96

    @property
    def stride(self):
        return 16

    # ── v2: pretrained loading ────────────────────────────────────────────

    def load_from_timm(self) -> bool:
        """
        Load matching weights from timm's mobilevit_xs (ImageNet pretrained).
        Returns True if at least 50% of layers matched.

        HOW IT WORKS:
          timm mobilevit_xs and our MobileViTBackbone share the same Conv/BN
          naming convention in their stem, stage1-stage5 blocks.  We match by
          layer name AND shape — any mismatch is silently skipped.  The neck
          (our custom upsample+conv) never matches and starts from scratch.
        """
        try:
            import timm
            ref = timm.create_model("mobilevit_xs", pretrained=True, num_classes=0)
        except ImportError:
            return False
        except Exception:
            return False

        ref_state = ref.state_dict()
        our_state = self.state_dict()
        matched   = {}

        for k, v in ref_state.items():
            if k in our_state and our_state[k].shape == v.shape:
                matched[k] = v

        if len(matched) < len(our_state) * 0.5:
            # Less than 50% matched — likely a different architecture version
            del ref
            return False

        self.load_state_dict(matched, strict=False)
        del ref
        return True

    def load_from_checkpoint(self, path: str) -> bool:
        """Load backbone weights from a local checkpoint."""
        if not __import__("os").path.exists(path):
            return False
        ckpt  = torch.load(path, map_location="cpu")
        state = ckpt.get("model_state", ckpt.get("state_dict", ckpt))
        # Strip common prefixes
        cleaned = {}
        for k, v in state.items():
            for prefix in ("backbone.", "model.backbone.", ""):
                if k.startswith(prefix):
                    cleaned[k[len(prefix):]] = v
                    break
        missing, unexpected = self.load_state_dict(cleaned, strict=False)
        return len(missing) < len(self.state_dict()) * 0.5

    def freeze(self):
        """Freeze all backbone parameters (use during early warmup)."""
        for p in self.parameters():
            p.requires_grad_(False)

    def unfreeze(self):
        """Unfreeze all backbone parameters."""
        for p in self.parameters():
            p.requires_grad_(True)

    @classmethod
    def pretrained(
        cls,
        source: Optional[str] = "timm",
        ckpt_path: Optional[str] = None,
        **kwargs,
    ) -> "MobileViTBackbone":
        """
        Build backbone and optionally load pretrained weights.

        Args:
            source:    'timm' | 'local' | None
            ckpt_path: path to local .pth when source='local'
        """
        bb = cls(**kwargs)
        if source == "timm":
            ok = bb.load_from_timm()
            if not ok:
                import warnings
                warnings.warn(
                    "MobileViTBackbone: timm load failed — using random init. "
                    "Install timm: pip install timm --break-system-packages"
                )
        elif source == "local" and ckpt_path:
            bb.load_from_checkpoint(ckpt_path)
        return bb


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# Sanity check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time

    bb = MobileViTBackbone()
    bb.eval()

    template = torch.randn(1, 3, 128, 128)
    search   = torch.randn(1, 3, 256, 256)

    with torch.no_grad():
        t0 = time.time()
        t_feat = bb(template)
        s_feat = bb(search)
        elapsed = (time.time() - t0) * 1000

    params = count_parameters(bb)
    print("=" * 50)
    print("MobileViT-XS Backbone v2 — Sanity Check")
    print("=" * 50)
    print(f"Template: {tuple(template.shape)} → {tuple(t_feat.shape)}")
    print(f"Search:   {tuple(search.shape)} → {tuple(s_feat.shape)}")
    print(f"Params:   {params:,}  ({params/1e6:.2f}M)")
    print(f"Time:     {elapsed:.1f} ms (CPU)")
    print(f"Budget:   {'✓' if params < 10e6 else '✗'}  (<10M)")
    print(f"Stride:   {bb.stride}  (expected 16)")
    print(f"Channels: {bb.out_channels}  (expected 96)")

    # Test pretrained factory (prints warning if timm not installed)
    bb2 = MobileViTBackbone.pretrained(source="timm")
    print(f"\npretrained() factory OK — params: {count_parameters(bb2):,}")
    print("=" * 50)