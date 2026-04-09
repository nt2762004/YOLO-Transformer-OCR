from __future__ import annotations

import math

import torch
import torch.nn as nn
import torchvision.models as tv_models

from .utils import generate_causal_mask


class CropEncoder(nn.Module):
    def __init__(self, d_model: int = 256, freeze_backbone: bool = False, dropout: float = 0.1):
        super().__init__()
        resnet = tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False
        self.project = nn.Conv2d(512, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        features = self.project(features)
        features = self.dropout(features)
        batch_size, channels, height, width = features.shape
        return features.flatten(2).transpose(1, 2).contiguous().view(batch_size, height * width, channels)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 512, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.activation = nn.GELU()

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: torch.Tensor | None = None) -> torch.Tensor:
        attn_out, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)
        tgt = self.norm1(tgt + self.dropout(attn_out))

        cross_out, _ = self.cross_attn(tgt, memory, memory)
        tgt = self.norm2(tgt + self.dropout(cross_out))

        ff = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        return self.norm3(tgt + self.dropout(ff))


class OCRTransformerDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_len: int = 100,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, dropout=dropout, max_len=max_len)
        self.layers = nn.ModuleList(
            [TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.generator = nn.Linear(d_model, vocab_size)

    def forward(self, tgt_ids: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        tgt = self.token_emb(tgt_ids) * math.sqrt(self.d_model)
        tgt = self.pos_enc(tgt)
        tgt_mask = generate_causal_mask(tgt.size(1), tgt.device)
        for layer in self.layers:
            tgt = layer(tgt, memory, tgt_mask=tgt_mask)
        tgt = self.norm(tgt)
        return self.generator(tgt)


class OCRModel(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 256, nhead: int = 8, num_layers: int = 6, dim_feedforward: int = 512, dropout: float = 0.1, max_len: int = 100, freeze_backbone: bool = False):
        super().__init__()
        self.encoder = CropEncoder(d_model=d_model, freeze_backbone=freeze_backbone, dropout=dropout)
        self.decoder = OCRTransformerDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_len=max_len,
        )

    def forward(self, images: torch.Tensor, tgt_ids: torch.Tensor) -> torch.Tensor:
        memory = self.encoder(images)
        return self.decoder(tgt_ids, memory)
