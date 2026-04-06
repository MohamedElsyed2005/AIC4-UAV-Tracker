"""
head.py  –  Tracking Head for HiT Tracker
==========================================
Takes the enhanced search tokens from the transformer and predicts
the target bounding box.

Strategy: Corner-based prediction (better than direct cx,cy,w,h)
  - Predict a score map  → which token contains the target center
  - Predict offset maps  → fine-grained x,y offset within the token
  - Predict size maps    → target width and height at each location

Final bbox = argmax(score) token position + offset + size

Why corner/score-based instead of direct regression?
  - More stable training (no mode collapse)
  - Score map gives explicit spatial supervision
  - Used in state-of-the-art trackers (OSTrack, MixFormer, etc.)

Input:  (B, Ns, dim)  — transformer output, Ns = Hs*Ws = 256
Output: dict with:
    'pred_boxes':  (B, 4)  — [cx, cy, w, h] normalised to [0,1]
    'score_map':   (B, 1, Hs, Ws)  — confidence at each location
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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

    Final bbox decoded from argmax of score map + offset + size.

    Args:
        dim:       transformer output dim (128)
        search_hw: search feature map size (16 for 256×256 input / stride 16)
        hidden:    MLP hidden dim (256)
    """

    def __init__(self, dim: int = 128, search_hw: int = 16, hidden: int = 256):
        super().__init__()
        self.search_hw = search_hw

        # Score map — is the target centered here?
        self.score_head = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

        # Offset — fine sub-token center offset (cx, cy) in [0,1]
        self.offset_head = MLP(dim, hidden, 2, num_layers=3)

        # Size — target (w, h) normalised to search region [0,1]
        self.size_head = MLP(dim, hidden, 2, num_layers=3)

        self._init_weights()

    def _init_weights(self):
        # Initialize score head bias for stable early training
        # (most locations are background → prior ~0.01 probability)
        nn.init.constant_(self.score_head[-1].bias, -4.0)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None and m not in [self.score_head[-1]]:
                    nn.init.zeros_(m.bias)

    def forward(self, tokens: torch.Tensor) -> dict:
        """
        Args:
            tokens: (B, Ns, dim)  — Ns = search_hw * search_hw
        Returns:
            dict:
                'pred_boxes':  (B, 4)         [cx, cy, w, h] in [0,1]
                'score_map':   (B, 1, H, W)   raw logits
                'score_map_sigmoid': (B, 1, H, W)  probabilities
        """
        B, Ns, C = tokens.shape
        H = W = self.search_hw
        assert Ns == H * W, f"Expected {H*W} tokens, got {Ns}"

        # ── Branch predictions ─────────────────────────────────────────────
        score  = self.score_head(tokens)      # (B, Ns, 1)
        offset = self.offset_head(tokens)     # (B, Ns, 2)
        size   = self.size_head(tokens)       # (B, Ns, 2)

        # Reshape to 2D maps
        score_map  = score.reshape(B, 1, H, W)           # (B, 1, H, W)
        offset_map = offset.reshape(B, H, W, 2)          # (B, H, W, 2)
        size_map   = size.reshape(B, H, W, 2)            # (B, H, W, 2)

        # ── Decode bounding box ────────────────────────────────────────────
        pred_boxes = self._decode_bbox(score_map, offset_map, size_map)

        return {
            'pred_boxes':        pred_boxes,       # (B, 4)
            'score_map':         score_map,         # (B, 1, H, W) raw logits
            'score_map_sigmoid': score_map.sigmoid(), # (B, 1, H, W) probs
        }

    def _decode_bbox(self,
                     score_map: torch.Tensor,
                     offset_map: torch.Tensor,
                     size_map: torch.Tensor) -> torch.Tensor:
        """
        Decode final bbox from score argmax + offset + size.

        All outputs are normalised to [0, 1] relative to the search crop.

        Args:
            score_map:  (B, 1, H, W)
            offset_map: (B, H, W, 2)  — (dx, dy) sub-token offsets
            size_map:   (B, H, W, 2)  — (w, h) predictions
        Returns:
            boxes: (B, 4)  [cx, cy, w, h] in [0, 1]
        """
        B, _, H, W = score_map.shape

        # Flatten and find argmax token (most likely target location)
        score_flat = score_map.reshape(B, -1)          # (B, H*W)
        best_idx   = score_flat.argmax(dim=1)           # (B,)

        # Convert flat index → (row, col)
        best_row = best_idx // W   # (B,)
        best_col = best_idx % W    # (B,)

        # Gather offset and size at the best location
        # (B, H, W, 2) → gather at (best_row, best_col) for each batch
        offset = offset_map[torch.arange(B), best_row, best_col]  # (B, 2)
        size   = size_map  [torch.arange(B), best_row, best_col]  # (B, 2)

        # Token center in normalised coords [0, 1]
        # token (r, c) center = ((c + 0.5) / W,  (r + 0.5) / H)
        cx_token = (best_col.float() + 0.5) / W   # (B,)
        cy_token = (best_row.float() + 0.5) / H   # (B,)

        # Add sub-token offset (predicted in [-0.5, 0.5] range via sigmoid)
        cx = cx_token + (offset[:, 0].sigmoid() - 0.5) / W
        cy = cy_token + (offset[:, 1].sigmoid() - 0.5) / H

        # Size predictions — sigmoid to keep in (0, 1)
        w = size[:, 0].sigmoid()
        h = size[:, 1].sigmoid()

        # Clamp to valid range
        cx = cx.clamp(0.0, 1.0)
        cy = cy.clamp(0.0, 1.0)
        w  = w.clamp(1e-3, 1.0)
        h  = h.clamp(1e-3, 1.0)

        boxes = torch.stack([cx, cy, w, h], dim=1)  # (B, 4)
        return boxes


# ─────────────────────────────────────────────────────────────────────────────
# Loss Functions
# ─────────────────────────────────────────────────────────────────────────────

class TrackingLoss(nn.Module):
    """
    Combined loss for training the tracker:

      L = w_cls * L_focal + w_iou * L_giou + w_l1 * L_l1

    - Focal loss  → score map supervision (handles class imbalance)
    - GIoU loss   → bounding box quality
    - L1 loss     → bounding box coordinate regression

    Args:
        w_cls:  weight for focal loss   (default 1.0)
        w_iou:  weight for GIoU loss    (default 2.0)
        w_l1:   weight for L1 loss      (default 5.0)
        search_hw: search feature map size (16)
    """

    def __init__(self, w_cls: float = 1.0, w_iou: float = 2.0,
                 w_l1: float = 5.0, search_hw: int = 16):
        super().__init__()
        self.w_cls     = w_cls
        self.w_iou     = w_iou
        self.w_l1      = w_l1
        self.search_hw = search_hw

    def focal_loss(self, pred_logits: torch.Tensor,
                   targets: torch.Tensor,
                   alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
        """
        Focal loss for score map.
        pred_logits: (B, 1, H, W) raw logits
        targets:     (B, 1, H, W) gaussian heatmap [0,1]
        """
        pred = pred_logits.sigmoid()
        bce  = F.binary_cross_entropy_with_logits(
            pred_logits, targets, reduction='none')
        p_t  = pred * targets + (1 - pred) * (1 - targets)
        loss = bce * ((1 - p_t) ** gamma)
        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * loss
        return loss.mean()

    def giou_loss(self, pred_boxes: torch.Tensor,
                  gt_boxes: torch.Tensor) -> torch.Tensor:
        """
        GIoU loss between predicted and gt boxes.
        Both in [cx, cy, w, h] normalised format.
        """
        # Convert to x1y1x2y2
        def to_xyxy(b):
            return torch.stack([
                b[:, 0] - b[:, 2] / 2,
                b[:, 1] - b[:, 3] / 2,
                b[:, 0] + b[:, 2] / 2,
                b[:, 1] + b[:, 3] / 2,
            ], dim=1)

        p = to_xyxy(pred_boxes)
        g = to_xyxy(gt_boxes)

        # Intersection
        inter_x1 = torch.max(p[:, 0], g[:, 0])
        inter_y1 = torch.max(p[:, 1], g[:, 1])
        inter_x2 = torch.min(p[:, 2], g[:, 2])
        inter_y2 = torch.min(p[:, 3], g[:, 3])
        inter_w  = (inter_x2 - inter_x1).clamp(0)
        inter_h  = (inter_y2 - inter_y1).clamp(0)
        inter    = inter_w * inter_h

        # Union
        area_p = pred_boxes[:, 2] * pred_boxes[:, 3]
        area_g = gt_boxes[:, 2]   * gt_boxes[:, 3]
        union  = area_p + area_g - inter + 1e-6

        iou = inter / union

        # Enclosing box
        enc_x1 = torch.min(p[:, 0], g[:, 0])
        enc_y1 = torch.min(p[:, 1], g[:, 1])
        enc_x2 = torch.max(p[:, 2], g[:, 2])
        enc_y2 = torch.max(p[:, 3], g[:, 3])
        enc    = (enc_x2 - enc_x1).clamp(0) * (enc_y2 - enc_y1).clamp(0) + 1e-6

        giou = iou - (enc - union) / enc
        return (1 - giou).mean()

    def make_score_target(self, gt_boxes: torch.Tensor) -> torch.Tensor:
        """
        Build a Gaussian heatmap target for the score map.
        Target peak is at the gt box center, spread with sigma.

        Args:
            gt_boxes: (B, 4)  [cx, cy, w, h] in [0, 1]
        Returns:
            heatmap: (B, 1, H, W)
        """
        B = gt_boxes.shape[0]
        H = W = self.search_hw
        device = gt_boxes.device

        # Grid of token centers [0, 1]
        ys = (torch.arange(H, device=device).float() + 0.5) / H  # (H,)
        xs = (torch.arange(W, device=device).float() + 0.5) / W  # (W,)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')   # (H, W)

        heatmaps = []
        for b in range(B):
            cx, cy, w, h = gt_boxes[b]
            # Sigma proportional to target size
            sigma = (w + h) / 8.0
            sigma = sigma.clamp(min=1.0 / H)

            dist = (grid_x - cx) ** 2 + (grid_y - cy) ** 2
            heatmap = torch.exp(-dist / (2 * sigma ** 2))
            heatmaps.append(heatmap)

        heatmaps = torch.stack(heatmaps).unsqueeze(1)  # (B, 1, H, W)
        return heatmaps

    def forward(self, pred: dict, gt_boxes: torch.Tensor) -> dict:
        """
        Args:
            pred:     output dict from TrackingHead.forward()
            gt_boxes: (B, 4)  [cx, cy, w, h] normalised ground truth
        Returns:
            dict with individual losses and total loss
        """
        pred_boxes = pred['pred_boxes']       # (B, 4)
        score_map  = pred['score_map']        # (B, 1, H, W)

        # Score target: gaussian heatmap
        score_target = self.make_score_target(gt_boxes)  # (B, 1, H, W)

        # Losses
        l_cls  = self.focal_loss(score_map, score_target)
        l_iou  = self.giou_loss(pred_boxes, gt_boxes)
        l_l1   = F.l1_loss(pred_boxes, gt_boxes)

        total = self.w_cls * l_cls + self.w_iou * l_iou + self.w_l1 * l_l1

        return {
            'loss':       total,
            'loss_cls':   l_cls,
            'loss_giou':  l_iou,
            'loss_l1':    l_l1,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Sanity Check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

    from models.hit.backbone    import MobileViTBackbone, count_parameters
    from models.hit.transformer import CrossAttentionTransformer

    print("=" * 55)
    print("TrackingHead — Sanity Check")
    print("=" * 55)

    # Build all three modules
    backbone    = MobileViTBackbone().eval()
    transformer = CrossAttentionTransformer(in_channels=96, dim=128,
                                            num_heads=4, num_layers=4).eval()
    head        = TrackingHead(dim=128, search_hw=16, hidden=256).eval()
    criterion   = TrackingLoss()

    # Dummy inputs
    template = torch.randn(2, 3, 128, 128)   # batch of 2
    search   = torch.randn(2, 3, 256, 256)
    gt_boxes = torch.tensor([[0.5, 0.5, 0.2, 0.3],   # gt for sample 1
                              [0.3, 0.4, 0.1, 0.15]]) # gt for sample 2

    with torch.no_grad():
        t0 = time.time()

        t_feat = backbone(template)                   # (2, 96, 8, 8)
        s_feat = backbone(search)                     # (2, 96, 16, 16)
        tokens = transformer(t_feat, s_feat)          # (2, 256, 128)
        pred   = head(tokens)                         # dict

        elapsed = (time.time() - t0) * 1000

    # Loss (needs grad, so run separately)
    backbone.train(); transformer.train(); head.train()
    t_feat = backbone(template)
    s_feat = backbone(search)
    tokens = transformer(t_feat, s_feat)
    pred   = head(tokens)
    losses = criterion(pred, gt_boxes)

    # Parameter counts
    p_bb  = count_parameters(backbone)
    p_tr  = count_parameters(transformer)
    p_hd  = count_parameters(head)
    total = p_bb + p_tr + p_hd

    print(f"Input template:  {tuple(template.shape)}")
    print(f"Input search:    {tuple(search.shape)}")
    print()
    print(f"pred_boxes:      {tuple(pred['pred_boxes'].shape)}  — values: {pred['pred_boxes'][0].tolist()}")
    print(f"score_map:       {tuple(pred['score_map'].shape)}")
    print()
    print(f"Backbone params:     {p_bb:>10,}  ({p_bb/1e6:.2f}M)")
    print(f"Transformer params:  {p_tr:>10,}  ({p_tr/1e6:.2f}M)")
    print(f"Head params:         {p_hd:>10,}  ({p_hd/1e6:.2f}M)")
    print(f"─" * 42)
    print(f"TOTAL params:        {total:>10,}  ({total/1e6:.2f}M)")
    print(f"Inference time:      {elapsed:.1f} ms  (CPU, full pipeline)")
    print()
    print(f"Losses:")
    print(f"  total:  {losses['loss'].item():.4f}")
    print(f"  cls:    {losses['loss_cls'].item():.4f}")
    print(f"  giou:   {losses['loss_giou'].item():.4f}")
    print(f"  l1:     {losses['loss_l1'].item():.4f}")
    print()
    print(f"Budget checks:")
    print(f"  Total params: {total/1e6:.2f}M / 10M  {'✓' if total < 10e6 else '✗'}")
    print(f"  pred_boxes range [0,1]:  {'✓' if pred['pred_boxes'].min() >= 0 and pred['pred_boxes'].max() <= 1 else '✗'}")
    print("=" * 55)