"""
models/hift_full.py
===================
Full HiFT model reconstructed to exactly match the pretrained checkpoint
(first.pth) key names and layer shapes.

Checkpoint structure (confirmed from paper repo + error messages):
  backbone.*   -> AlexNet with named layers layer1..layer5
                  layer1: Conv(3,96,11,s=2) + BN + MaxPool + ReLU
                  layer2: Conv(96,256,5)    + BN + MaxPool  + ReLU
                  layer3: Conv(256,384,3)   + BN + ReLU
                  layer4: Conv(384,384,3)   + BN + ReLU
                  layer5: Conv(384,256,3)   + BN
  grader.*     -> HiFT head
                  conv1:    Conv(384,192,3,s=2) + BN + ReLU  [layer3 xcorr]
                  conv3:    Conv(384,192,3,s=2) + BN + ReLU  [layer4 xcorr]
                  conv2:    Conv(256,192,3,s=2) + BN + ReLU  [layer5 xcorr]
                  transformer.*
                  convloc:  4x Conv(192,192,3) + GN + ReLU -> Conv(192,4,3)
                  convcls:  3x Conv(192,192,3) + GN + ReLU
                  cls1:     Conv(192,2,3)
                  cls2:     Conv(192,1,3)
                  row_embed, col_embed: positional embeddings

NOTE: The original uses GroupNorm with 32 groups (cfg.TRAIN.groupchannel=32).
      This is hardcoded here since we no longer need the cfg object.
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention, ModuleList


# ─────────────────────────────────────────────────────────────────────────────
# Backbone  (key prefix: backbone.*)
# ─────────────────────────────────────────────────────────────────────────────

class AlexNet(nn.Module):
    """
    Matches the original repo's AlexNet exactly:
      layer1: Conv(3,96,11,s=2) + BN(96) + MaxPool(3,s=2) + ReLU
      layer2: Conv(96,256,5)    + BN(256) + MaxPool(3,s=2) + ReLU
      layer3: Conv(256,384,3)   + BN(384) + ReLU
      layer4: Conv(384,384,3)   + BN(384) + ReLU
      layer5: Conv(384,256,3)   + BN(256)
    Returns a list [f1, f2, f3, f4, f5] so the grader can pick any layer.
    """
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=2),
            nn.BatchNorm2d(96),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
        )

    def forward(self, x):
        """Returns list [f3, f4, f5] — the three levels used by the grader."""
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        f5 = self.layer5(f4)
        return [f3, f4, f5]


# ─────────────────────────────────────────────────────────────────────────────
# Channel attention  (used inside transformer encoder)
# ─────────────────────────────────────────────────────────────────────────────

class Cattention(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(in_dim * 2, in_dim, kernel_size=1, stride=1),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.linear1  = nn.Conv2d(in_dim, in_dim // 6, 1, bias=False)
        self.linear2  = nn.Conv2d(in_dim // 6, in_dim, 1, bias=False)
        self.gamma     = nn.Parameter(torch.zeros(1))
        self.activation = nn.ReLU(inplace=True)
        self.dropout    = nn.Dropout()

    def forward(self, x, y):
        ww     = self.linear2(self.dropout(self.activation(self.linear1(self.avg_pool(y)))))
        weight = self.conv1(torch.cat((x, y), 1)) * ww
        return x + self.gamma * weight * x


# ─────────────────────────────────────────────────────────────────────────────
# Transformer  (encoder + decoder, matching original key names exactly)
# ─────────────────────────────────────────────────────────────────────────────

def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for _ in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":  return F.relu
    if activation == "gelu":  return F.gelu
    raise RuntimeError(f"Unknown activation: {activation}")


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu"):
        super().__init__()
        channel = 192
        self.self_attn  = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cross_attn = Cattention(channel)
        self.eles = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(32, channel),
            nn.ReLU(inplace=True),
        )
        self.linear1   = nn.Linear(d_model, dim_feedforward)
        self.dropout   = nn.Dropout(dropout)
        self.linear2   = nn.Linear(dim_feedforward, d_model)
        self.norm0     = nn.LayerNorm(d_model)
        self.norm1     = nn.LayerNorm(d_model)
        self.norm2     = nn.LayerNorm(d_model)
        self.dropout1  = nn.Dropout(dropout)
        self.dropout2  = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)

    def forward(self, src, srcc, src_mask=None, src_key_padding_mask=None):
        b, c, s = src.permute(1, 2, 0).size()
        src2 = self.self_attn(
            self.norm0(src + srcc), self.norm0(src + srcc), src,
            attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        side = int(s ** 0.5)
        src = self.cross_attn(
            src.view(b, c, side, side),
            srcc.contiguous().view(b, c, side, side)
        ).view(b, c, -1).permute(2, 0, 1)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src  = src + self.dropout2(src2)
        src  = self.norm2(src)
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers     = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm       = norm

    def forward(self, src, srcc, mask=None, src_key_padding_mask=None):
        output = src
        for mod in self.layers:
            output = mod(output, srcc, src_mask=mask,
                         src_key_padding_mask=src_key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)
        return output


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu"):
        super().__init__()
        self.self_attn      = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1   = nn.Linear(d_model, dim_feedforward)
        self.dropout   = nn.Dropout(dropout)
        self.linear2   = nn.Linear(dim_feedforward, d_model)
        self.norm1     = nn.LayerNorm(d_model)
        self.norm2     = nn.LayerNorm(d_model)
        self.norm3     = nn.LayerNorm(d_model)
        self.dropout1  = nn.Dropout(dropout)
        self.dropout2  = nn.Dropout(dropout)
        self.dropout3  = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt  = self.norm1(tgt + self.dropout1(tgt2))
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt  = self.norm2(tgt + self.dropout2(tgt2))
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt  = self.norm3(tgt + self.dropout3(tgt2))
        return tgt


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers     = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm       = norm

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        output = tgt
        for mod in self.layers:
            output = mod(output, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)
        return output


class Transformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu"):
        super().__init__()
        enc_layer  = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                             dropout, activation)
        self.encoder = TransformerEncoder(enc_layer, num_encoder_layers,
                                          nn.LayerNorm(d_model))
        dec_layer  = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                             dropout, activation)
        self.decoder = TransformerDecoder(dec_layer, num_decoder_layers,
                                          nn.LayerNorm(d_model))
        self.d_model = d_model
        self.nhead   = nhead
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, srcc, tgt):
        memory = self.encoder(src, srcc)
        output = self.decoder(tgt, memory)
        return output


# ─────────────────────────────────────────────────────────────────────────────
# HiFT grader head  (key prefix: grader.*)
# ─────────────────────────────────────────────────────────────────────────────

class HiFTGrader(nn.Module):
    """
    Matches the original HiFT class from the paper repo exactly.
    groupchannel=32 (was cfg.TRAIN.groupchannel).
    """
    def __init__(self, groupchannel: int = 32):
        super().__init__()
        channel = 192

        # Feature compression convs (with stride=2, matching ckpt shapes)
        self.conv1 = nn.Sequential(
            nn.Conv2d(384, channel, kernel_size=3, bias=False, stride=2, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(384, channel, kernel_size=3, bias=False, stride=2, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(256, channel, kernel_size=3, bias=False, stride=2, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
        )

        # Localisation head
        self.convloc = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(groupchannel, channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(groupchannel, channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(groupchannel, channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, 4,       kernel_size=3, stride=1, padding=1),
        )

        # Classification head
        self.convcls = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(groupchannel, channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(groupchannel, channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(groupchannel, channel),
            nn.ReLU(inplace=True),
        )

        # Positional embeddings
        self.row_embed = nn.Embedding(50, channel // 2)
        self.col_embed = nn.Embedding(50, channel // 2)
        self._reset_pos()

        # Transformer: d_model=192, nhead=6, 1 encoder layer, 2 decoder layers
        self.transformer = Transformer(channel, 6, 1, 2)

        # Final cls heads
        self.cls1 = nn.Conv2d(channel, 2, kernel_size=3, stride=1, padding=1)
        self.cls2 = nn.Conv2d(channel, 1, kernel_size=3, stride=1, padding=1)

        # Weight init for conv1
        for l in self.conv1.modules():
            if isinstance(l, nn.Conv2d):
                nn.init.normal_(l.weight, std=0.01)

    def _reset_pos(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    @staticmethod
    def xcorr_depthwise(x, kernel):
        batch   = kernel.size(0)
        channel = kernel.size(1)
        x      = x.view(1, batch * channel, x.size(2), x.size(3))
        kernel = kernel.view(batch * channel, 1, kernel.size(2), kernel.size(3))
        out    = F.conv2d(x, kernel, groups=batch * channel)
        return out.view(batch, channel, out.size(2), out.size(3))

    def forward(self, x, z):
        """
        x: list [x3, x4, x5] — search features at layers 3, 4, 5
        z: list [z3, z4, z5] — template features at layers 3, 4, 5
        """
        res1 = self.conv1(self.xcorr_depthwise(x[0], z[0]))  # layer3 xcorr
        res2 = self.conv3(self.xcorr_depthwise(x[1], z[1]))  # layer4 xcorr
        res3 = self.conv2(self.xcorr_depthwise(x[2], z[2]))  # layer5 xcorr

        h, w = res3.shape[-2:]
        device = res3.device
        i = torch.arange(w, device=device)
        j = torch.arange(h, device=device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(res3.shape[0], 1, 1, 1)

        b, c, ww, hh = res3.size()
        res = self.transformer(
            (pos + res1).view(b, c, -1).permute(2, 0, 1),
            (pos + res2).view(b, c, -1).permute(2, 0, 1),
            res3.view(b, c, -1).permute(2, 0, 1),
        )
        res = res.permute(1, 2, 0).view(b, c, ww, hh)

        loc  = self.convloc(res)
        acls = self.convcls(res)
        cls1 = self.cls1(acls)
        cls2 = self.cls2(acls)

        return loc, cls1, cls2


# ─────────────────────────────────────────────────────────────────────────────
# Top-level model  (matches checkpoint top-level keys: backbone.* / grader.*)
# ─────────────────────────────────────────────────────────────────────────────

class HiFT(nn.Module):
    """
    Full HiFT model. No cfg needed — all hyperparameters are hardcoded to
    match the pretrained checkpoint (first.pth).

    Usage:
        model = HiFT()
        model.load_pretrained("checkpoints/first.pth")
        loc, cls1, cls2 = model(template, search)
    """

    def __init__(self):
        super().__init__()
        self.backbone = AlexNet()
        self.grader   = HiFTGrader(groupchannel=32)

    def forward(self, template: torch.Tensor, search: torch.Tensor):
        zf = self.backbone(template)   # [z3, z4, z5]
        xf = self.backbone(search)     # [x3, x4, x5]
        return self.grader(xf, zf)     # loc, cls1, cls2

    def load_pretrained(self, path: str, device: str = "cpu") -> None:
        print("\n====== PRETRAINED LOADING REPORT ======\n")
        ckpt = torch.load(path, map_location=device, weights_only=False)
        sd   = ckpt.get("state_dict", ckpt.get("net", ckpt.get("model", ckpt)))

        model_sd = self.state_dict()
        loaded, skipped = [], []
        new_sd = {}

        for k, v in sd.items():
            k2 = k.replace("module.", "")
            if k2 in model_sd and model_sd[k2].shape == v.shape:
                new_sd[k2] = v
                loaded.append(k2)
            else:
                skipped.append(f"{k}  ckpt={tuple(v.shape)}"
                               + (f"  model={tuple(model_sd[k2].shape)}"
                                  if k2 in model_sd else "  NOT-IN-MODEL"))

        model_sd.update(new_sd)
        self.load_state_dict(model_sd, strict=False)

        missing = [k for k in model_sd if k not in loaded]
        print(f"Total model keys : {len(model_sd)}")
        print(f"Loaded           : {len(loaded)}")
        print(f"Missing (random init) : {len(missing)}")
        print(f"Skipped (shape/name mismatch) : {len(skipped)}")
        if skipped:
            print("\nSkipped keys (first 10):")
            for s in skipped[:10]:
                print(" ⚠", s)
        print("\n=======================================\n")