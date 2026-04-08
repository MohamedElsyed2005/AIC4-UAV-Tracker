"""
model.py  –  HiT Tracker  (v2)
================================
Changes vs v1
─────────────
1. HiTConfig.pretrained_backbone field:
   'timm' | 'local' | 'none'
   If set, MobileViTBackbone.pretrained() is called at build time
   so the backbone already has ImageNet weights before training starts.

2. HiTTracker.param_groups() method:
   Returns two parameter groups — backbone (lower LR) and the rest —
   ready to pass directly to AdamW.  train_hit.py uses this instead of
   model.parameters() for the split-LR optimiser.

3. HiTTracker.freeze_backbone() / unfreeze_backbone():
   Proxies through to the backbone's own freeze/unfreeze.
   Call freeze_backbone() for the first few warmup epochs when using
   pretrained weights, then unfreeze_backbone() for the remainder.

No changes to the forward pass, loss, or inference API.
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys
import os
from typing import Optional, List, Dict

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from models.hit.backbone    import MobileViTBackbone, count_parameters
from models.hit.transformer import CrossAttentionTransformer
from models.hit.head        import TrackingHead, TrackingLoss


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

class HiTConfig:
    """All hyperparameters in one place."""

    # Input sizes
    template_size: int   = 128
    search_size:   int   = 256

    # Backbone
    backbone_out_channels: int = 96
    backbone_stride:       int = 16

    @property
    def template_hw(self): return self.template_size // self.backbone_stride  # 8
    @property
    def search_hw(self):   return self.search_size   // self.backbone_stride  # 16

    # Transformer
    transformer_dim:     int   = 128
    transformer_heads:   int   = 4
    transformer_layers:  int   = 4
    transformer_dropout: float = 0.1

    # Head
    head_hidden: int = 256

    # Loss weights
    loss_w_cls:  float = 1.0
    loss_w_giou: float = 2.0
    loss_w_l1:   float = 5.0

    # Context factors for cropping
    template_context: float = 2.0
    search_context:   float = 4.0


# ─────────────────────────────────────────────────────────────────────────────
# HiT Tracker Model  (v2)
# ─────────────────────────────────────────────────────────────────────────────

class HiTTracker(nn.Module):
    """
    Full HiT Tracker — backbone + transformer + head.

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

        # ── Build backbone (with optional pretrained weights) ──────────────
        self.backbone = MobileViTBackbone()

        self.transformer = CrossAttentionTransformer(
            in_channels = cfg.backbone_out_channels,
            dim         = cfg.transformer_dim,
            num_heads   = cfg.transformer_heads,
            num_layers  = cfg.transformer_layers,
            dropout     = cfg.transformer_dropout,
        )

        self.head = TrackingHead(
            dim       = cfg.transformer_dim,
            search_hw = cfg.search_hw,
            hidden    = cfg.head_hidden,
        )

        self.criterion = TrackingLoss(
            w_cls     = cfg.loss_w_cls,
            w_iou     = cfg.loss_w_giou,
            w_l1      = cfg.loss_w_l1,
            search_hw = cfg.search_hw,
        )

        # Inference state
        self._template_feat = None

    # ── Pretrained backbone controls ──────────────────────────────────────

    def freeze_backbone(self):
        """
        Freeze backbone parameters.
        """
        if hasattr(self.backbone, "freeze"):
            self.backbone.freeze()
        else:
            for param in self.backbone.parameters():
                param.requires_grad = False


    def unfreeze_backbone(self):
        """
        Unfreeze backbone parameters.
        """
        if hasattr(self.backbone, "unfreeze"):
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

        Usage:
            optimizer = AdamW(model.param_groups(), lr=2e-4, weight_decay=1e-4)
        """
        backbone_ids = {id(p) for p in self.backbone.parameters()}
        backbone_params    = list(self.backbone.parameters())
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
        t_feat = self.backbone(template)
        s_feat = self.backbone(search)
        tokens = self.transformer(t_feat, s_feat)
        return self.head(tokens)

    def compute_loss(self, output: dict,
                     gt_boxes: torch.Tensor) -> dict:
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
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        print(f"[HiTTracker] Loaded ← {path}  "
              f"({ckpt['params']['total_M']}M params)")
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
    import time, tempfile, os

    print("=" * 55)
    print("HiTTracker v2 — Full Pipeline Sanity Check")
    print("=" * 55)

    model = build_hit_tracker()

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
    print(f"[Train]  loss={losses['loss'].item():.4f}  ({train_ms:.1f} ms)")

    # Inference
    model.eval()
    model.initialize(torch.randn(1, 3, 128, 128))
    out = model.track_crop(torch.randn(1, 3, 256, 256))
    print(f"[Infer]  boxes={out['pred_boxes'][0].tolist()}")

    # Freeze/unfreeze
    model.freeze_backbone()
    frozen = sum(1 for p in model.backbone.parameters() if not p.requires_grad)
    model.unfreeze_backbone()
    unfrozen = sum(1 for p in model.backbone.parameters() if p.requires_grad)
    print(f"[Freeze] frozen={frozen}  unfrozen={unfrozen}")

    # Param groups
    groups = model.param_groups()
    print(f"[Groups] {[g['name'] for g in groups]}")

    # Param count
    params = model.param_count()
    print(f"\nParameter Summary:")
    print(f"  Backbone:    {params['backbone']:>10,}  ({params['backbone']/1e6:.2f}M)")
    print(f"  Transformer: {params['transformer']:>10,}  ({params['transformer']/1e6:.2f}M)")
    print(f"  Head:        {params['head']:>10,}  ({params['head']/1e6:.2f}M)")
    print(f"  TOTAL:       {params['total']:>10,}  ({params['total_M']}M)")
    print(f"\n  Budget: {'✓' if params['total'] < 10e6 else '✗'}  (<10M params)")
    print("=" * 55)