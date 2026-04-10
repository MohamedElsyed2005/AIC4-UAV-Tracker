"""
transformer.py  –  Cross-Attention Transformer for HiT Tracker
===============================================================
Takes template features and search features from the backbone,
then uses cross-attention to find where the target is located
in the search region.

How it works:
  - Template features → Keys & Values  (what we're looking for)
  - Search features   → Queries        (where we're looking)
  - Cross-attention output → enhanced search features with
    target location awareness

Architecture:
  CrossAttentionLayer × N_layers
  Each layer:
    1. Self-attention on search features
    2. Cross-attention: search queries → template keys/values
    3. FFN

Output: enhanced search features (B, H*W, C) ready for the head

Feature map sizes (backbone stride=16):
  template 128×128 → backbone → (B, 96, 8,  8)   Nt=64  tokens
  search   256×256 → backbone → (B, 96, 16, 16)  Ns=256 tokens

FIX (v4): PositionEncoding2D max sizes tightened to match actual usage:
  - pos_enc_template: max_h=max_w=8   (template feature is 8×8)
  - pos_enc_search:   max_h=max_w=16  (search feature is 16×16)
  Previously max_h=max_w=32 for search, which wasted embedding table space.
  Tightening catches out-of-range indices instead of silently accepting them.
  If you change input resolution, update these values accordingly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Position Encoding
# ─────────────────────────────────────────────────────────────────────────────

class PositionEncoding2D(nn.Module):
    """
    Learnable 2D position encoding.
    Adds spatial awareness to the flat token sequences.

    Args:
        channels: transformer hidden dim (must be even)
        max_h:    maximum feature map height  (template=8, search=16)
        max_w:    maximum feature map width   (template=8, search=16)
    """
    def __init__(self, channels: int, max_h: int = 16, max_w: int = 16):
        super().__init__()
        self.row_embed = nn.Embedding(max_h, channels // 2)
        self.col_embed = nn.Embedding(max_w, channels // 2)
        self._init_weights()

    def _init_weights(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) feature map
        Returns:
            pos: (1, H*W, channels) position encoding
        """
        B, C, H, W = x.shape
        device = x.device

        rows = torch.arange(H, device=device)
        cols = torch.arange(W, device=device)

        row_enc = self.row_embed(rows)  # (H, channels//2)
        col_enc = self.col_embed(cols)  # (W, channels//2)

        # Broadcast and concatenate
        row_enc = row_enc.unsqueeze(1).expand(H, W, -1)  # (H, W, channels//2)
        col_enc = col_enc.unsqueeze(0).expand(H, W, -1)  # (H, W, channels//2)

        pos = torch.cat([row_enc, col_enc], dim=-1)  # (H, W, channels)
        pos = pos.reshape(1, H * W, -1)              # (1, H*W, channels)
        return pos


# ─────────────────────────────────────────────────────────────────────────────
# Attention Modules
# ─────────────────────────────────────────────────────────────────────────────

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention — works for both self and cross attention.
    query, key, value can come from different sources (cross-attention)
    or the same source (self-attention).
    """
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0, \
            f"dim {dim} must be divisible by num_heads {num_heads}"

        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5

        self.q_proj   = nn.Linear(dim, dim)
        self.k_proj   = nn.Linear(dim, dim)
        self.v_proj   = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout  = nn.Dropout(dropout)

    def forward(self,
                query: torch.Tensor,
                key:   torch.Tensor,
                value: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query: (B, Nq, C)
            key:   (B, Nk, C)
            value: (B, Nk, C)
        Returns:
            out:   (B, Nq, C)
        """
        B, Nq, C = query.shape
        Nk = key.shape[1]
        h  = self.num_heads
        d  = self.head_dim

        Q = self.q_proj(query).reshape(B, Nq, h, d).transpose(1, 2)  # (B, h, Nq, d)
        K = self.k_proj(key).reshape(B, Nk, h, d).transpose(1, 2)    # (B, h, Nk, d)
        V = self.v_proj(value).reshape(B, Nk, h, d).transpose(1, 2)  # (B, h, Nk, d)

        attn = (Q @ K.transpose(-2, -1)) * self.scale                 # (B, h, Nq, Nk)
        attn = self.dropout(attn.softmax(dim=-1))

        out = (attn @ V).transpose(1, 2).reshape(B, Nq, C)            # (B, Nq, C)
        return self.out_proj(out)


# ─────────────────────────────────────────────────────────────────────────────
# Cross-Attention Layer
# ─────────────────────────────────────────────────────────────────────────────

class CrossAttentionLayer(nn.Module):
    """
    One full transformer layer with:
      1. Self-attention on search tokens
      2. Cross-attention: search attends to template
      3. Feed-forward network

    All with pre-norm (more stable training).
    """
    def __init__(self, dim: int, num_heads: int = 8,
                 mlp_ratio: int = 4, dropout: float = 0.0):
        super().__init__()

        # Self-attention
        self.norm_sa   = nn.LayerNorm(dim)
        self.self_attn = MultiHeadAttention(dim, num_heads, dropout)

        # Cross-attention
        self.norm_ca    = nn.LayerNorm(dim)
        self.norm_ca_kv = nn.LayerNorm(dim)
        self.cross_attn = MultiHeadAttention(dim, num_heads, dropout)

        # FFN
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout),
        )

    def forward(self,
                search:   torch.Tensor,
                template: torch.Tensor) -> torch.Tensor:
        """
        Args:
            search:   (B, Ns, C)  — search region tokens
            template: (B, Nt, C)  — template tokens (keys/values)
        Returns:
            search:   (B, Ns, C)  — enhanced search tokens
        """
        # 1. Self-attention on search
        s = self.norm_sa(search)
        search = search + self.self_attn(s, s, s)

        # 2. Cross-attention: search queries → template keys/values
        s  = self.norm_ca(search)
        t  = self.norm_ca_kv(template)
        search = search + self.cross_attn(s, t, t)

        # 3. FFN
        search = search + self.ffn(self.norm_ffn(search))

        return search


# ─────────────────────────────────────────────────────────────────────────────
# Full Cross-Attention Transformer
# ─────────────────────────────────────────────────────────────────────────────

class CrossAttentionTransformer(nn.Module):
    """
    Full transformer that takes backbone features and produces
    enhanced search features ready for the tracking head.

    Pipeline:
      1. Flatten spatial dims → token sequences
      2. Add position encodings
      3. N × CrossAttentionLayer
      4. Output: (B, Ns, C)

    Args:
        in_channels: backbone output channels (96)
        dim:         transformer hidden dim (128 — projected from backbone)
        num_heads:   attention heads (4)
        num_layers:  number of cross-attention layers (4)
        mlp_ratio:   FFN expansion ratio (4)
        dropout:     attention dropout (0.1)

    FIX (v4): PositionEncoding2D max sizes set to match actual feature maps:
        pos_enc_template: max_h=max_w=8   (template 128/stride16 = 8)
        pos_enc_search:   max_h=max_w=16  (search   256/stride16 = 16)
        Previously these were max_h=16/max_w=16 and max_h=32/max_w=32.
        If you change template_size or search_size, update accordingly.
    """

    def __init__(
        self,
        in_channels: int = 96,
        dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim

        # Project backbone channels → transformer dim
        self.input_proj = nn.Linear(in_channels, dim)

        # Position encodings — sized to match actual feature maps
        # template: 128×128 input / stride 16 = 8×8 feature map
        # search:   256×256 input / stride 16 = 16×16 feature map
        self.pos_enc_template = PositionEncoding2D(dim, max_h=8,  max_w=8)   # FIX: was 16
        self.pos_enc_search   = PositionEncoding2D(dim, max_h=16, max_w=16)  # FIX: was 32

        # Stack of cross-attention layers
        self.layers = nn.ModuleList([
            CrossAttentionLayer(dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])

        self.norm_out = nn.LayerNorm(dim)

    def forward(self,
                template_feat: torch.Tensor,
                search_feat:   torch.Tensor) -> torch.Tensor:
        """
        Args:
            template_feat: (B, C, Ht, Wt)  e.g. (B, 96, 8,  8)
            search_feat:   (B, C, Hs, Ws)  e.g. (B, 96, 16, 16)
        Returns:
            out: (B, Hs*Ws, dim)  enhanced search tokens
                 e.g. (B, 256, 128)
        """
        B, C, Ht, Wt = template_feat.shape
        _,  _, Hs, Ws = search_feat.shape

        # ── Flatten spatial dims → tokens ──────────────────────────────────
        # (B, C, H, W) → (B, H*W, C)
        t_tokens = template_feat.flatten(2).transpose(1, 2)  # (B, Nt, C)
        s_tokens = search_feat.flatten(2).transpose(1, 2)    # (B, Ns, C)

        # ── Project to transformer dim ──────────────────────────────────────
        t_tokens = self.input_proj(t_tokens)  # (B, Nt, dim)
        s_tokens = self.input_proj(s_tokens)  # (B, Ns, dim)

        # ── Add position encodings ──────────────────────────────────────────
        t_tokens = t_tokens + self.pos_enc_template(template_feat)
        s_tokens = s_tokens + self.pos_enc_search(search_feat)

        # ── Cross-attention layers ──────────────────────────────────────────
        for layer in self.layers:
            s_tokens = layer(s_tokens, t_tokens)

        # ── Final norm ──────────────────────────────────────────────────────
        out = self.norm_out(s_tokens)  # (B, Ns, dim)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Sanity Check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from backbone import AlexNetBackbone, count_parameters

    print("=" * 60)
    print("CrossAttentionTransformer v4 — Sanity Check (fixed pos enc)")
    print("=" * 60)

    backbone    = AlexNetBackbone(pretrained=False).eval()
    transformer = CrossAttentionTransformer(
        in_channels=96,
        dim=128,
        num_heads=4,
        num_layers=4,
    ).eval()

    template = torch.randn(1, 3, 128, 128)
    search   = torch.randn(1, 3, 256, 256)

    with torch.no_grad():
        t0 = time.time()

        t_feat = backbone(template)   # (1, 96, 8, 8)
        s_feat = backbone(search)     # (1, 96, 16, 16)
        out    = transformer(t_feat, s_feat)   # (1, 256, 128)

        elapsed = (time.time() - t0) * 1000

    backbone_params    = count_parameters(backbone)
    transformer_params = count_parameters(transformer)
    total_params       = backbone_params + transformer_params

    print(f"Template feat:   {tuple(t_feat.shape)}   (expected [1,96,8,8])")
    print(f"Search feat:     {tuple(s_feat.shape)}   (expected [1,96,16,16])")
    print(f"Transformer out: {tuple(out.shape)}  (expected [1,256,128])")
    print()

    assert t_feat.shape == (1, 96, 8, 8),    f"FAIL t_feat {t_feat.shape}"
    assert s_feat.shape == (1, 96, 16, 16),  f"FAIL s_feat {s_feat.shape}"
    assert out.shape    == (1, 256, 128),    f"FAIL out {out.shape}"
    print("Shape assertions: ✓")

    print()
    print(f"Backbone params:     {backbone_params:>10,}  ({backbone_params/1e6:.2f}M)")
    print(f"Transformer params:  {transformer_params:>10,}  ({transformer_params/1e6:.2f}M)")
    print(f"Total so far:        {total_params:>10,}  ({total_params/1e6:.2f}M)")
    print(f"Inference time:      {elapsed:.1f} ms  (CPU)")
    print()
    print(f"Budget: {total_params/1e6:.2f}M / 10M  {'✓' if total_params < 10e6 else '✗'}")
    print("=" * 60)