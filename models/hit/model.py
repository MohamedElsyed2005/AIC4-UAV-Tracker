"""
model.py  –  HiT Tracker  (v4 — shape-consistency fix)
========================================================

CHANGES vs v3:
  1. HiTConfig documentation corrected:
     - backbone_stride is now explicitly documented as the EFFECTIVE stride
       after ALL layers (features[:8] + extra_conv with stride=1 = 16)
     - search_hw and template_hw property comments updated

  2. Runtime shape assertion added in forward():
     - After backbone, checks that search feature map is search_hw × search_hw
     - Catches any future backbone/config drift immediately with a clear message
     - Assertion only runs in training (removed via torch.jit / eval if needed)

  3. compute_loss() passes output dict directly — no changes to loss API
     (TrackingLoss.forward now reads score_map shape dynamically)

  4. param_count() unchanged (already fixed in v3 to count ALL params)
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys
import os
from typing import Optional, List, Dict

_ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(_ROOT))

from backbone    import AlexNetBackbone, count_parameters
from transformer import CrossAttentionTransformer
from head        import TrackingHead, TrackingLoss


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

class HiTConfig:
    """
    All hyperparameters in one place.

    Spatial resolution chain (backbone stride = 16):
      template 128×128  →  backbone  →  8×8    (Nt = 64   tokens)
      search   256×256  →  backbone  →  16×16  (Ns = 256  tokens)

    The backbone effective stride is:
      AlexNet features[:8]  →  stride 16  (two MaxPool layers at idx 2 and 5)
      extra_conv stride=1   →  stride 16  (no additional downsampling)
      adapter 1×1           →  stride 16  (no downsampling)
      TOTAL effective stride = 16
    """

    # Input sizes
    template_size: int = 128
    search_size:   int = 256

    # Backbone
    # effective stride = 16  (features[:8] = stride 16, extra_conv stride=1)
    backbone_out_channels: int = 96
    backbone_stride:       int = 16   # MUST match AlexNetBackbone.stride

    @property
    def template_hw(self) -> int:
        """Template feature map side length.  128 // 16 = 8."""
        return self.template_size // self.backbone_stride   # 8

    @property
    def search_hw(self) -> int:
        """Search feature map side length.  256 // 16 = 16."""
        return self.search_size // self.backbone_stride     # 16

    # Transformer
    transformer_dim:     int   = 128
    transformer_heads:   int   = 4
    transformer_layers:  int   = 6  #Increase transformer capacity, was 4 — still within 10M budget
    transformer_dropout: float = 0.1

    # Head
    head_hidden: int = 256

    # Loss weights (rebalanced in v3, unchanged here)
    loss_w_cls:  float = 2.0
    loss_w_giou: float = 2.0
    loss_w_l1:   float = 1.0


# ─────────────────────────────────────────────────────────────────────────────
# HiT Tracker Model  (v4)
# ─────────────────────────────────────────────────────────────────────────────

class HiTTracker(nn.Module):
    """
    Full HiT Tracker — AlexNet backbone + transformer + head.

    Training:
        output = model(template, search)
        losses = model.compute_loss(output, gt_boxes)

    Inference:
        model.initialize(template_crop)
        output = model.track_crop(search_crop)
    """

    def __init__(self, cfg: HiTConfig = None):
        super().__init__()
        self.cfg = cfg or HiTConfig()
        cfg = self.cfg

        # ── Build backbone ────────────────────────────────────────────────
        self.backbone = AlexNetBackbone(pretrained=True)

        # Sanity-check backbone stride matches config at build time
        assert self.backbone.stride == cfg.backbone_stride, (
            f"Backbone stride {self.backbone.stride} != "
            f"config backbone_stride {cfg.backbone_stride}. "
            f"Update HiTConfig.backbone_stride or fix AlexNetBackbone."
        )

        self.transformer = CrossAttentionTransformer(
            in_channels = cfg.backbone_out_channels,
            dim         = cfg.transformer_dim,
            num_heads   = cfg.transformer_heads,
            num_layers  = cfg.transformer_layers,
            dropout     = cfg.transformer_dropout,
        )

        self.head = TrackingHead(
            dim       = cfg.transformer_dim,
            search_hw = cfg.search_hw,   # 16  (reference only — head uses dynamic Ns)
            hidden    = cfg.head_hidden,
        )

        self.criterion = TrackingLoss(
            w_cls     = cfg.loss_w_cls,
            w_iou     = cfg.loss_w_giou,
            w_l1      = cfg.loss_w_l1,
            search_hw = cfg.search_hw,   # 16  (reference only — loss uses dynamic shape)
        )

        # Inference state
        self._template_feat = None

    # ── Backbone freeze controls ──────────────────────────────────────────

    def freeze_backbone(self):
        """
        Freeze AlexNet base features.
        extra_conv and adapter remain trainable for domain adaptation.
        This is the recommended strategy during warmup epochs.
        """
        if hasattr(self.backbone, 'freeze'):
            self.backbone.freeze()
        else:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze all backbone parameters."""
        if hasattr(self.backbone, 'unfreeze'):
            self.backbone.unfreeze()
        else:
            for param in self.backbone.parameters():
                param.requires_grad = True

    # ── Optimiser helper ──────────────────────────────────────────────────

    def param_groups(self, backbone_lr_scale: float = 0.1) -> List[Dict]:
        """
        Return two parameter groups for AdamW:
          - backbone at (lr × backbone_lr_scale)
          - transformer + head at full lr
        """
        backbone_ids        = {id(p) for p in self.backbone.parameters()}
        backbone_params     = list(self.backbone.parameters())
        non_backbone_params = [
            p for p in self.parameters() if id(p) not in backbone_ids
        ]
        return [
            {"params": backbone_params,     "lr_scale": backbone_lr_scale,
             "name": "backbone"},
            {"params": non_backbone_params, "lr_scale": 1.0,
             "name": "head_transformer"},
        ]

    # ── Training forward ──────────────────────────────────────────────────

    def forward(self, template: torch.Tensor,
                search:   torch.Tensor) -> dict:
        """
        Full forward pass.

        Args:
            template: (B, 3, 128, 128)
            search:   (B, 3, 256, 256)

        Returns:
            dict with pred_boxes, score_map, score_map_sigmoid
        """
        t_feat = self.backbone(template)   # (B, 96,  8,  8)
        s_feat = self.backbone(search)     # (B, 96, 16, 16)

        # Runtime assertion — catches backbone/config drift immediately.
        expected_s_hw = self.cfg.search_hw   # 16
        expected_t_hw = self.cfg.template_hw  # 8
        if self.training:
            assert s_feat.shape[2] == expected_s_hw and s_feat.shape[3] == expected_s_hw, (
                f"Search feature map is {s_feat.shape[2]}×{s_feat.shape[3]}, "
                f"expected {expected_s_hw}×{expected_s_hw}. "
                f"Check backbone stride ({self.backbone.stride}) vs "
                f"HiTConfig.backbone_stride ({self.cfg.backbone_stride}) "
                f"and search_size ({self.cfg.search_size})."
            )
            assert t_feat.shape[2] == expected_t_hw and t_feat.shape[3] == expected_t_hw, (
                f"Template feature map is {t_feat.shape[2]}×{t_feat.shape[3]}, "
                f"expected {expected_t_hw}×{expected_t_hw}. "
                f"Check backbone stride vs template_size ({self.cfg.template_size})."
            )

        tokens = self.transformer(t_feat, s_feat)   # (B, 256, 128)
        return self.head(tokens)

    def compute_loss(self, output: dict,
                     gt_boxes: torch.Tensor) -> dict:
        """
        Compute combined tracking loss.

        TrackingLoss.forward() now reads score_map shape dynamically,
        so target heatmap always matches score_map exactly.
        """
        return self.criterion(output, gt_boxes)

    # ── Inference ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def initialize(self, template_crop: torch.Tensor):
        self.eval()
        self._template_feat = self.backbone(template_crop)

    @torch.no_grad()
    def track_crop(self, search_crop: torch.Tensor) -> dict:
        self.eval()
        assert self._template_feat is not None, \
            "Call initialize() before track_crop()"
        s_feat = self.backbone(search_crop)
        tokens = self.transformer(self._template_feat, s_feat)
        return self.head(tokens)

    # ── Utilities ──────────────────────────────────────────────────────────

    def param_count(self) -> dict:
        """Count ALL parameters (not just trainable)."""
        p_bb  = count_parameters(self.backbone)
        p_tr  = count_parameters(self.transformer)
        p_hd  = count_parameters(self.head)
        total = p_bb + p_tr + p_hd
        return {
            "backbone":    p_bb,
            "transformer": p_tr,
            "head":        p_hd,
            "total":       total,
            "total_M":     round(total / 1e6, 3),
        }

    def save(self, path: str):
        torch.save({
            "model_state": self.state_dict(),
            "cfg":         self.cfg.__dict__,
            "params":      self.param_count(),
        }, path)
        print(f"[HiTTracker] Saved → {path}")

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "HiTTracker":
        ckpt  = torch.load(path, map_location=device)
        model = cls()
        model.load_state_dict(ckpt["model_state"], strict=False)
        model.eval()
        params = model.param_count()
        print(f"[HiTTracker] Loaded ← {path}  "
              f"({params['total_M']}M params)")
        return model


# ─────────────────────────────────────────────────────────────────────────────
# Builder
# ─────────────────────────────────────────────────────────────────────────────

def build_hit_tracker(cfg: HiTConfig = None) -> HiTTracker:
    return HiTTracker(cfg or HiTConfig())


# ─────────────────────────────────────────────────────────────────────────────
# Sanity check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time, tempfile

    print("=" * 60)
    print("HiTTracker v4 (AlexNet stride-16) — Full Pipeline Sanity Check")
    print("=" * 60)

    model = build_hit_tracker()

    cfg = model.cfg
    print(f"Config:  template_hw={cfg.template_hw}  search_hw={cfg.search_hw}  "
          f"backbone_stride={cfg.backbone_stride}")

    # Training forward
    model.train()
    template = torch.randn(2, 3, 128, 128)
    search   = torch.randn(2, 3, 256, 256)
    gt_boxes = torch.tensor([[0.5, 0.5, 0.2, 0.3],
                              [0.3, 0.4, 0.1, 0.2]])

    t0 = time.time()
    output = model(template, search)
    losses = model.compute_loss(output, gt_boxes)
    losses["loss"].backward()
    train_ms = (time.time() - t0) * 1000

    print(f"\n[Train]  score_map shape: {tuple(output['score_map'].shape)}  "
          f"(expected [2,1,16,16])")
    print(f"[Train]  loss={losses['loss'].item():.4f}  ({train_ms:.1f} ms)")

    assert output['score_map'].shape == (2, 1, 16, 16), \
        f"FAIL score_map {output['score_map'].shape}"
    print("[Train]  Shape assertion: ✓")

    # Inference
    model.eval()
    model.initialize(torch.randn(1, 3, 128, 128))
    out = model.track_crop(torch.randn(1, 3, 256, 256))
    print(f"\n[Infer]  boxes={[round(v,4) for v in out['pred_boxes'][0].tolist()]}")

    # Freeze/unfreeze
    model.freeze_backbone()
    frozen = sum(1 for p in model.backbone.features.parameters()
                 if not p.requires_grad)
    model.unfreeze_backbone()
    print(f"\n[Freeze] {frozen} base-feature params frozen (adapter stays trainable)")

    # Param count
    params = model.param_count()
    print(f"\nParameter Summary:")
    print(f"  Backbone:    {params['backbone']:>10,}  ({params['backbone']/1e6:.2f}M)")
    print(f"  Transformer: {params['transformer']:>10,}  ({params['transformer']/1e6:.2f}M)")
    print(f"  Head:        {params['head']:>10,}  ({params['head']/1e6:.2f}M)")
    print(f"  TOTAL:       {params['total']:>10,}  ({params['total_M']}M)")
    print(f"\n  Budget: {'✓' if params['total'] < 10e6 else '✗'}  (<10M params)")
    print("=" * 60)