"""
head.py  –  Tracking Head for HiT Tracker  (v4 — shape-consistency fix)
=========================================================================

FIXES vs v3:
  1. make_score_target() now derives H, W from score_map.shape  [CRITICAL FIX]
     - Previously used self.search_hw (hardcoded at init time)
     - If backbone stride != expected value, self.search_hw was wrong
     - This caused the crash:
         ValueError: Target size ([B,1,16,16]) != input size ([B,1,8,8])
     - Fix: read H, W directly from score_map at runtime — always consistent

  2. focal_loss() shape robustness:
     - Added explicit .view_as() to align target with pred before BCE
     - Prevents silent broadcasting bugs if shapes ever drift

  3. TrackingLoss.forward() receives score_map explicitly:
     - H, W extracted from score_map.shape and passed to make_score_target
     - make_score_target signature updated: now accepts score_map tensor
       to derive its spatial size, not a stored integer

  4. search_hw kept in TrackingLoss.__init__ for backward compatibility
     (used only if score_map is not provided — never in normal training)

Input:  (B, Ns, dim)  — transformer output, Ns = Hs*Ws
Output: dict with:
    'pred_boxes':        (B, 4)         — [cx, cy, w, h] normalised [0,1]
    'score_map':         (B, 1, Hs, Ws) — confidence at each location (logits)
    'score_map_sigmoid': (B, 1, Hs, Ws) — confidence (probabilities)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ─────────────────────────────────────────────────────────────────────────────
# MLP helper
# ─────────────────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    """Simple multi-layer perceptron with GELU activations."""
    def __init__(self, in_dim: int, hidden_dim: int,
                 out_dim: int, num_layers: int = 3):
        super().__init__()
        layers = []
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# Tracking Head
# ─────────────────────────────────────────────────────────────────────────────

class TrackingHead(nn.Module):
    """
    Predicts bounding box from transformer search tokens.

    Three parallel prediction branches:
      1. Score branch  → (B, 1, Hs, Ws)   confidence heatmap
      2. Offset branch → (B, 2, Hs, Ws)   sub-token cx,cy offset [0,1]
      3. Size branch   → (B, 2, Hs, Ws)   target w,h  [0,1]

    Uses soft-argmax (expectation) instead of hard argmax for decoding.
    This gives a differentiable, sub-token-precision prediction.

    H and W are derived DYNAMICALLY from sqrt(Ns) — no hardcoded grid size.

    Args:
        dim:       transformer output dim (128)
        search_hw: search feature map size (16 for 256×256 input / stride 16)
                   stored for reference but NOT used to reshape tokens — Ns is
        hidden:    MLP hidden dim (256)
    """

    def __init__(self, dim: int = 128, search_hw: int = 16, hidden: int = 256):
        super().__init__()
        self.search_hw = search_hw  # kept for reference / external queries

        # Score map — is the target centered here?
        self.score_head = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

        # Offset — fine sub-token center offset (cx, cy) in [-0.5, 0.5]
        self.offset_head = MLP(dim, hidden, 2, num_layers=3)

        # Size — target (w, h) normalised to search region [0,1]
        self.size_head = MLP(dim, hidden, 2, num_layers=3)

        self._init_weights()

    def _init_weights(self):
        """Stable initialisation: score head bias for ~1% prior probability."""
        nn.init.constant_(self.score_head[-1].bias, -4.6)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None and m is not self.score_head[-1]:
                    nn.init.zeros_(m.bias)

    def forward(self, tokens: torch.Tensor) -> dict:
        """
        Args:
            tokens: (B, Ns, dim)  where Ns = Hs * Ws

        Returns:
            dict with pred_boxes, score_map, score_map_sigmoid
        """
        B, Ns, C = tokens.shape

        # Dynamic resolution — derived from actual Ns, not from self.search_hw.
        # This is immune to any backbone-stride misconfiguration.
        H = W = int(math.sqrt(Ns))
        assert H * W == Ns, (
            f"Non-square token count Ns={Ns}. "
            f"Expected a perfect square (e.g. 256 for 16×16 or 64 for 8×8)."
        )

        score  = self.score_head(tokens)     # (B, Ns, 1)
        offset = self.offset_head(tokens)    # (B, Ns, 2)
        size   = self.size_head(tokens)      # (B, Ns, 2)

        score_map = score.reshape(B, 1, H, W)

        pred_boxes = self._soft_decode_bbox(
            score_map,
            offset.reshape(B, Ns, 2),
            size.reshape(B, Ns, 2),
        )

        return {
            'pred_boxes':        pred_boxes,
            'score_map':         score_map,
            'score_map_sigmoid': score_map.sigmoid(),
        }

    def _soft_decode_bbox(self,
                          score_map: torch.Tensor,
                          offset_map: torch.Tensor,
                          size_map: torch.Tensor) -> torch.Tensor:
        """
        Soft-argmax decoding: compute expected position over score distribution.
        Fully dynamic — H and W are read from score_map.shape, not stored state.
        """
        B   = score_map.shape[0]
        H   = score_map.shape[2]
        W   = score_map.shape[3]
        Ns  = H * W

        device = score_map.device
        attn   = score_map.reshape(B, Ns).softmax(dim=1)  # (B, Ns)

        ys = (torch.arange(H, device=device).float() + 0.5) / H
        xs = (torch.arange(W, device=device).float() + 0.5) / W

        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        grid_x = grid_x.reshape(-1)   # (Ns,)
        grid_y = grid_y.reshape(-1)   # (Ns,)

        cx_token = (attn * grid_x.unsqueeze(0)).sum(dim=1)  # (B,)
        cy_token = (attn * grid_y.unsqueeze(0)).sum(dim=1)  # (B,)

        offset = (attn.unsqueeze(-1) * offset_map).sum(dim=1)  # (B, 2)
        size   = (attn.unsqueeze(-1) * size_map).sum(dim=1)    # (B, 2)

        dx = offset[:, 0].tanh() * 0.5 / W
        dy = offset[:, 1].tanh() * 0.5 / H

        cx = (cx_token + dx).clamp(0.0, 1.0)
        cy = (cy_token + dy).clamp(0.0, 1.0)

        w = size[:, 0].sigmoid().clamp(0.01, 1.0)
        h = size[:, 1].sigmoid().clamp(0.01, 1.0)

        return torch.stack([cx, cy, w, h], dim=1)  # (B, 4)

    def decode_inference(self, tokens: torch.Tensor,
                         score_threshold: float = 0.0) -> dict:

        out = self.forward(tokens)
        score_map_sig = out['score_map_sigmoid']

        B = tokens.shape[0]
        best_score = score_map_sig.reshape(B, -1).max(dim=1).values

        out['best_score'] = best_score
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Loss Functions  (v4 — dynamic target resolution)
# ─────────────────────────────────────────────────────────────────────────────

class TrackingLoss(nn.Module):
    """
    Combined loss for training the tracker:

      L = w_cls * L_focal + w_iou * L_giou + w_l1 * L_l1

    CRITICAL FIX (v4):
      make_score_target() now derives H, W from the actual score_map tensor
      passed in forward(), NOT from the stored self.search_hw integer.
      This ensures target heatmap size always matches score_map size,
      regardless of backbone stride or any other config mismatch.
    """

    def __init__(self, w_cls: float = 1.0, w_iou: float = 3.0,
                 w_l1: float = 2.0, search_hw: int = 16):
        super().__init__()
        self.w_cls     = w_cls
        self.w_iou     = w_iou
        self.w_l1      = w_l1
        self.search_hw = search_hw  # kept for backward compat, NOT used in forward

    def focal_loss(self, pred_logits: torch.Tensor,
                   targets: torch.Tensor,
                   alpha: float = 0.25,
                   gamma: float = 2.0) -> torch.Tensor:
        """
        Focal loss.  targets must be the same shape as pred_logits.
        FIX: explicit .view_as() guards against silent broadcast bugs.
        """
        targets = targets.view_as(pred_logits)   # FIX: enforce shape match

        pred  = pred_logits.sigmoid()
        bce   = F.binary_cross_entropy_with_logits(
            pred_logits, targets, reduction='none')
        p_t   = pred * targets + (1 - pred) * (1 - targets)
        loss  = bce * ((1 - p_t) ** gamma)

        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss    = alpha_t * loss

        return loss.mean()

    def giou_loss(self, pred_boxes: torch.Tensor,
                  gt_boxes: torch.Tensor) -> torch.Tensor:
        def xyxy(b):
            return torch.stack([
                b[:, 0] - b[:, 2] / 2,
                b[:, 1] - b[:, 3] / 2,
                b[:, 0] + b[:, 2] / 2,
                b[:, 1] + b[:, 3] / 2,
            ], dim=1)

        p = xyxy(pred_boxes)
        g = xyxy(gt_boxes)

        inter = (
            (torch.min(p[:, 2], g[:, 2]) - torch.max(p[:, 0], g[:, 0])).clamp(0)
            * (torch.min(p[:, 3], g[:, 3]) - torch.max(p[:, 1], g[:, 1])).clamp(0)
        )

        area_p = (pred_boxes[:, 2] * pred_boxes[:, 3]).clamp(min=1e-6)
        area_g = (gt_boxes[:, 2]   * gt_boxes[:, 3]).clamp(min=1e-6)

        union = area_p + area_g - inter + 1e-6
        iou   = inter / union

        enc = (
            (torch.max(p[:, 2], g[:, 2]) - torch.min(p[:, 0], g[:, 0])).clamp(0)
            * (torch.max(p[:, 3], g[:, 3]) - torch.min(p[:, 1], g[:, 1])).clamp(0)
            + 1e-6
        )

        giou = iou - (enc - union) / enc
        return (1 - giou).mean()

    def make_score_target(self, gt_boxes: torch.Tensor,
                          score_map: torch.Tensor) -> torch.Tensor:
        """
        Build Gaussian heatmap target with the SAME spatial size as score_map.

        CRITICAL FIX:
          H and W are now read from score_map.shape[2:], not from
          self.search_hw.  This guarantees target shape == score_map shape
          regardless of backbone stride or any config inconsistency.

        Args:
            gt_boxes:   (B, 4)           — [cx, cy, w, h] normalised [0,1]
            score_map:  (B, 1, H, W)     — model output (used for shape only)

        Returns:
            heatmap:    (B, 1, H, W)     — Gaussian target in [0,1]
        """
        B      = gt_boxes.shape[0]
        H, W   = score_map.shape[2], score_map.shape[3]   # FIX: dynamic
        device = gt_boxes.device

        ys = (torch.arange(H, device=device).float() + 0.5) / H
        xs = (torch.arange(W, device=device).float() + 0.5) / W

        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')

        heatmaps = []
        for b in range(B):
            cx, cy, w, h = gt_boxes[b]
            sigma = ((w + h) / 8).clamp(min=1.5 / H, max=0.5)

            dist    = (grid_x - cx) ** 2 + (grid_y - cy) ** 2
            heatmap = torch.exp(-dist / (2 * sigma ** 2))
            heatmaps.append(heatmap / (heatmap.max() + 1e-8))

        return torch.stack(heatmaps).unsqueeze(1)   # (B, 1, H, W)

    def forward(self, pred: dict, gt_boxes: torch.Tensor) -> dict:
        """
        Args:
            pred:     dict from TrackingHead.forward()
            gt_boxes: (B, 4) — [cx, cy, w, h] normalised [0,1]

        Returns:
            dict with 'loss', 'loss_cls', 'loss_giou', 'loss_l1'
        """
        pred_boxes = pred['pred_boxes']    # (B, 4)
        score_map  = pred['score_map']     # (B, 1, H, W)

        # FIX: pass score_map so make_score_target reads H,W from it
        target_heatmap = self.make_score_target(gt_boxes, score_map)

        # target_heatmap.shape == score_map.shape  (guaranteed by fix)
        l_cls = self.focal_loss(score_map, target_heatmap)
        l_iou = self.giou_loss(pred_boxes, gt_boxes)
        l_l1  = F.l1_loss(pred_boxes, gt_boxes)

        total = self.w_cls * l_cls + self.w_iou * l_iou + self.w_l1 * l_l1

        return {
            'loss':      total,
            'loss_cls':  l_cls,
            'loss_giou': l_iou,
            'loss_l1':   l_l1,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Sanity Check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from backbone    import AlexNetBackbone, count_parameters
    from transformer import CrossAttentionTransformer

    print("=" * 60)
    print("TrackingHead v4 — Sanity Check (shape-consistency fix)")
    print("=" * 60)

    backbone    = AlexNetBackbone(pretrained=False).eval()
    transformer = CrossAttentionTransformer(
        in_channels=96, dim=128, num_heads=4, num_layers=4).eval()
    head        = TrackingHead(dim=128, search_hw=16, hidden=256).eval()
    criterion   = TrackingLoss()

    template = torch.randn(2, 3, 128, 128)
    search   = torch.randn(2, 3, 256, 256)
    gt_boxes = torch.tensor([[0.5, 0.5, 0.2, 0.3],
                              [0.3, 0.4, 0.1, 0.15]])

    with torch.no_grad():
        t0 = time.time()
        t_feat = backbone(template)
        s_feat = backbone(search)
        tokens = transformer(t_feat, s_feat)
        pred   = head(tokens)
        elapsed = (time.time() - t0) * 1000

    print(f"t_feat shape:  {tuple(t_feat.shape)}   (expected [2,96,8,8])")
    print(f"s_feat shape:  {tuple(s_feat.shape)}   (expected [2,96,16,16])")
    print(f"tokens shape:  {tuple(tokens.shape)}  (expected [2,256,128])")
    print(f"pred_boxes:    {tuple(pred['pred_boxes'].shape)}")
    print(f"score_map:     {tuple(pred['score_map'].shape)}  (expected [2,1,16,16])")

    assert t_feat.shape  == (2, 96, 8, 8),   f"FAIL t_feat {t_feat.shape}"
    assert s_feat.shape  == (2, 96, 16, 16), f"FAIL s_feat {s_feat.shape}"
    assert tokens.shape  == (2, 256, 128),   f"FAIL tokens {tokens.shape}"
    assert pred['score_map'].shape == (2, 1, 16, 16), \
        f"FAIL score_map {pred['score_map'].shape}"

    backbone.train(); transformer.train(); head.train()
    t_feat = backbone(template)
    s_feat = backbone(search)
    tokens = transformer(t_feat, s_feat)
    pred   = head(tokens)
    losses = criterion(pred, gt_boxes)   # must NOT crash

    p_bb  = count_parameters(backbone)
    p_tr  = count_parameters(transformer)
    p_hd  = count_parameters(head)
    total = p_bb + p_tr + p_hd

    print(f"\nBackbone:      {p_bb:>10,}  ({p_bb/1e6:.2f}M)")
    print(f"Transformer:   {p_tr:>10,}  ({p_tr/1e6:.2f}M)")
    print(f"Head:          {p_hd:>10,}  ({p_hd/1e6:.2f}M)")
    print(f"TOTAL:         {total:>10,}  ({total/1e6:.2f}M)")
    print(f"Infer time:    {elapsed:.1f} ms (CPU)")
    print()
    print(f"Losses (must not crash):")
    print(f"  total:  {losses['loss'].item():.4f}")
    print(f"  cls:    {losses['loss_cls'].item():.4f}")
    print(f"  giou:   {losses['loss_giou'].item():.4f}")
    print(f"  l1:     {losses['loss_l1'].item():.4f}")
    print()
    print(f"Budget: {'✓' if total < 10e6 else '✗'}  (<10M)")
    print(f"Boxes in [0,1]: {'✓' if pred['pred_boxes'].min()>=0 and pred['pred_boxes'].max()<=1 else '✗'}")
    print(f"Shape assertions: ✓")
    print("=" * 60)