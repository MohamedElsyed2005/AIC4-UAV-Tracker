"""
model.py  –  HiT Tracker  (Hybrid inference Tracker)
=====================================================
Wraps backbone + transformer + head into a single nn.Module.

Usage:
    model = build_hit_tracker()

    # Training
    output = model(template, search)
    losses = model.compute_loss(output, gt_boxes)

    # Inference  (tracker.py calls these)
    model.initialize(first_frame, init_box)
    pred_box = model.track(search_frame)
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys
import os

# Make sure project root is in path
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from models.hit.backbone    import MobileViTBackbone, count_parameters
from models.hit.transformer import CrossAttentionTransformer
from models.hit.head        import TrackingHead, TrackingLoss


# ─────────────────────────────────────────────────────────────────────────────
# Config dataclass
# ─────────────────────────────────────────────────────────────────────────────

class HiTConfig:
    """All hyperparameters in one place — easy to tune."""

    # Input sizes
    template_size : int   = 128     # template crop size (pixels)
    search_size   : int   = 256     # search crop size   (pixels)

    # Backbone
    backbone_out_channels : int = 96
    backbone_stride       : int = 16

    # Computed feature map sizes
    @property
    def template_hw(self):  return self.template_size // self.backbone_stride  # 8
    @property
    def search_hw(self):    return self.search_size   // self.backbone_stride  # 16

    # Transformer
    transformer_dim     : int   = 128
    transformer_heads   : int   = 4
    transformer_layers  : int   = 4
    transformer_dropout : float = 0.1

    # Head
    head_hidden : int = 256

    # Loss weights
    loss_w_cls  : float = 1.0
    loss_w_giou : float = 2.0
    loss_w_l1   : float = 5.0

    # Context factors for cropping (used by tracker.py)
    template_context : float = 2.0
    search_context   : float = 4.0


# ─────────────────────────────────────────────────────────────────────────────
# HiT Tracker Model
# ─────────────────────────────────────────────────────────────────────────────

class HiTTracker(nn.Module):
    """
    Full HiT Tracker.

    Forward pass (training):
        template: (B, 3, 128, 128)
        search:   (B, 3, 256, 256)
        → dict with pred_boxes, score_map, score_map_sigmoid

    Stateful inference (evaluation / submission):
        model.initialize(frame, box)  — call once on frame 0
        model.track(frame)            — call on every subsequent frame
                                        returns [x, y, w, h] in pixel coords
    """

    def __init__(self, cfg: HiTConfig = None):
        super().__init__()
        self.cfg = cfg or HiTConfig()
        cfg = self.cfg

        # ── Modules ──────────────────────────────────────────────────────────
        self.backbone = MobileViTBackbone(in_ch=3)

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

        # ── Inference state (set by initialize()) ─────────────────────────
        self._template_feat = None   # cached template features
        self._last_box      = None   # last predicted box [x,y,w,h] pixels
        self._search_size   = None   # (frame_h, frame_w) of current sequence

    # ── Training forward ─────────────────────────────────────────────────────

    def forward(self, template: torch.Tensor,
                search: torch.Tensor) -> dict:
        """
        Training forward pass.

        Args:
            template: (B, 3, H_t, W_t)
            search:   (B, 3, H_s, W_s)
        Returns:
            dict: pred_boxes, score_map, score_map_sigmoid
        """
        t_feat = self.backbone(template)          # (B, C, Ht, Wt)
        s_feat = self.backbone(search)            # (B, C, Hs, Ws)
        tokens = self.transformer(t_feat, s_feat) # (B, Ns, dim)
        output = self.head(tokens)                # dict
        return output

    def compute_loss(self, output: dict,
                     gt_boxes: torch.Tensor) -> dict:
        """
        Compute training loss.

        Args:
            output:   dict from forward()
            gt_boxes: (B, 4) [cx, cy, w, h] normalised to [0,1]
        Returns:
            dict: loss, loss_cls, loss_giou, loss_l1
        """
        return self.criterion(output, gt_boxes)

    # ── Inference API (called by tracker.py) ─────────────────────────────────

    @torch.no_grad()
    def initialize(self, template_crop: torch.Tensor):
        """
        Cache template features from the first frame crop.
        Call once at the start of each sequence.

        Args:
            template_crop: (1, 3, 128, 128) preprocessed tensor
        """
        self.eval()
        self._template_feat = self.backbone(template_crop)  # (1, C, 8, 8)

    @torch.no_grad()
    def track_crop(self, search_crop: torch.Tensor) -> dict:
        """
        Run tracker on a preprocessed search crop.

        Args:
            search_crop: (1, 3, 256, 256) preprocessed tensor
        Returns:
            dict: pred_boxes (normalised), score_map
        """
        self.eval()
        assert self._template_feat is not None, \
            "Call initialize() before track_crop()"

        s_feat = self.backbone(search_crop)
        tokens = self.transformer(self._template_feat, s_feat)
        output = self.head(tokens)
        return output

    # ── Utilities ─────────────────────────────────────────────────────────────

    def param_count(self) -> dict:
        """Return parameter counts per module and total."""
        p_bb  = count_parameters(self.backbone)
        p_tr  = count_parameters(self.transformer)
        p_hd  = count_parameters(self.head)
        total = p_bb + p_tr + p_hd
        return {
            'backbone':    p_bb,
            'transformer': p_tr,
            'head':        p_hd,
            'total':       total,
            'total_M':     round(total / 1e6, 3),
        }

    def save(self, path: str):
        """Save model weights."""
        torch.save({
            'model_state': self.state_dict(),
            'cfg':         self.cfg.__dict__,
            'params':      self.param_count(),
        }, path)
        print(f"[HiTTracker] Saved → {path}")

    @classmethod
    def load(cls, path: str, device: str = 'cpu') -> 'HiTTracker':
        """Load model from checkpoint."""
        ckpt  = torch.load(path, map_location=device)
        model = cls()
        model.load_state_dict(ckpt['model_state'])
        model.eval()
        print(f"[HiTTracker] Loaded ← {path}  "
              f"({ckpt['params']['total_M']}M params)")
        return model


# ─────────────────────────────────────────────────────────────────────────────
# Builder function
# ─────────────────────────────────────────────────────────────────────────────

def build_hit_tracker(cfg: HiTConfig = None) -> HiTTracker:
    """Build and return HiTTracker with given config."""
    return HiTTracker(cfg or HiTConfig())


# ─────────────────────────────────────────────────────────────────────────────
# Sanity Check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time

    print("=" * 55)
    print("HiTTracker — Full Pipeline Sanity Check")
    print("=" * 55)

    model = build_hit_tracker()

    # ── Training forward ──────────────────────────────────────────────────
    model.train()
    template = torch.randn(2, 3, 128, 128)
    search   = torch.randn(2, 3, 256, 256)
    gt_boxes = torch.tensor([[0.5, 0.5, 0.2, 0.3],
                              [0.3, 0.4, 0.1, 0.2]])

    t0 = time.time()
    output = model(template, search)
    losses = model.compute_loss(output, gt_boxes)
    losses['loss'].backward()
    train_ms = (time.time() - t0) * 1000

    print(f"[Training]")
    print(f"  pred_boxes:   {tuple(output['pred_boxes'].shape)}")
    print(f"  score_map:    {tuple(output['score_map'].shape)}")
    print(f"  loss total:   {losses['loss'].item():.4f}")
    print(f"  loss_cls:     {losses['loss_cls'].item():.4f}")
    print(f"  loss_giou:    {losses['loss_giou'].item():.4f}")
    print(f"  loss_l1:      {losses['loss_l1'].item():.4f}")
    print(f"  Forward+backward: {train_ms:.1f} ms (CPU)")
    print()

    # ── Inference (stateful) ──────────────────────────────────────────────
    model.eval()
    template_crop = torch.randn(1, 3, 128, 128)
    search_crop   = torch.randn(1, 3, 256, 256)

    t0 = time.time()
    model.initialize(template_crop)
    out1 = model.track_crop(search_crop)
    out2 = model.track_crop(search_crop)   # second frame
    infer_ms = (time.time() - t0) * 1000

    print(f"[Inference]")
    print(f"  initialize() → template feat cached")
    print(f"  track_crop() → boxes: {out1['pred_boxes'][0].tolist()}")
    print(f"  Inference time (init+2 frames): {infer_ms:.1f} ms (CPU)")
    print()

    # ── Save / Load ───────────────────────────────────────────────────────
    import tempfile, os
    tmp = tempfile.mktemp(suffix='.pth')
    model.save(tmp)
    model2 = HiTTracker.load(tmp)
    os.remove(tmp)
    print()

    # ── Parameter summary ─────────────────────────────────────────────────
    params = model.param_count()
    print(f"Parameter Summary:")
    print(f"  Backbone:      {params['backbone']:>10,}  ({params['backbone']/1e6:.2f}M)")
    print(f"  Transformer:   {params['transformer']:>10,}  ({params['transformer']/1e6:.2f}M)")
    print(f"  Head:          {params['head']:>10,}  ({params['head']/1e6:.2f}M)")
    print(f"  {'─'*38}")
    print(f"  TOTAL:         {params['total']:>10,}  ({params['total_M']}M)")
    print()
    print(f"Budget checks:")
    print(f"  Params {params['total_M']}M / 10M   {'✓' if params['total'] < 10e6 else '✗'}")
    print(f"  Backward pass works            ✓")
    print(f"  Save / Load works              ✓")
    print("=" * 55)