"""Reusable Transformer and cross-attention blocks for modality fusion."""

from typing import Optional
import torch
import torch.nn as nn

TRANSFORMER = "transformer"
CROSS_ATTENTION = "cross_attention"


def normalize_fusion_mode(mode: Optional[str]) -> str:
    """Normalize supported multimodal fusion modes."""
    if mode is None:
        return TRANSFORMER

    normalized = str(mode).strip().lower().replace("-", "_")
    if normalized == "transformer":
        return TRANSFORMER
    if normalized in {"cross_attention", "crossattention", "cross_attn", "crossattn"}:
        return CROSS_ATTENTION
    raise ValueError(
        f"Unsupported fusion mode: {mode}. Choose one of: '{TRANSFORMER}', '{CROSS_ATTENTION}'."
    )


class Transformer(nn.Module):
    def __init__(
        self,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=None,
        dropout=0.1,
        batch_first=False,
    ):
        super(Transformer, self).__init__()
        if dim_feedforward is None:
            dim_feedforward = d_model * 4

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=batch_first,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, src):
        output = self.transformer_encoder(src)
        return output


def feature_tensor(x: dict, key: str, device: str) -> torch.Tensor:
    """Read a semantic feature tensor and move/coerce to long dtype."""
    value = x[key]
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=torch.long)
    return torch.tensor([value], dtype=torch.long).to(device)


class CrossAttentionFusion(nn.Module):
    """Cross-attention fusion between image and tabular embeddings."""

    def __init__(self, embed_dim: int, nhead: int = 2, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim * 2, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, image_embedding: torch.Tensor, tabular_embedding: torch.Tensor) -> torch.Tensor:
        image_tokens = image_embedding.unsqueeze(1)
        tabular_tokens = tabular_embedding.unsqueeze(1) if tabular_embedding.ndim == 2 else tabular_embedding

        image_to_tabular, _ = self.attn(query=image_tokens, key=tabular_tokens, value=tabular_tokens)
        tabular_to_image, _ = self.attn(query=tabular_tokens, key=image_tokens, value=image_tokens)

        fused = torch.cat(
            [image_to_tabular.squeeze(1), tabular_to_image.mean(dim=1)],
            dim=-1,
        )
        fused = self.proj(fused)
        fused = self.dropout(fused)
        return self.norm(fused)
