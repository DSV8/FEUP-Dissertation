"""Tabular feature encoders shared by CF and multimodal models."""

import torch
import torch.nn as nn
from .fusion import Transformer


class TabularEncoder(nn.Module):
    """Encode a raw tabular vector with an MLP or feature-token Transformer.

    ``forward`` intentionally returns a pooled ``(B, out_dim)`` tensor for
    compatibility with the NCF towers. ``forward_tokens`` exposes the
    feature-token sequence for multimodal cross-attention.
    """

    def __init__(
        self,
        in_features: int = 5,
        tabular_dim: int = 32,
        out_dim: int = 32,
        nhead: int = 1,
        num_layers: int = 1,
        encoder_type: str = "transformer",
    ):
        super().__init__()
        normalized_type = str(encoder_type).strip().lower().replace("-", "_")
        if normalized_type not in {"transformer", "mlp"}:
            raise ValueError("Unsupported tabular encoder type. Choose 'transformer' or 'mlp'.")
        self.encoder_type = normalized_type
        self.feature_projection = nn.Linear(1, tabular_dim)
        self.feature_embedding = nn.Parameter(torch.randn(1, in_features, tabular_dim))
        self.projection = nn.Linear(in_features, tabular_dim)
        self.projection_ln = nn.LayerNorm(tabular_dim)
        if self.encoder_type == "transformer":
            self.transformer = Transformer(
                d_model=tabular_dim,
                nhead=nhead,
                num_layers=num_layers,
                dim_feedforward=tabular_dim * 4,
                batch_first=True,
            )
        else:
            self.mlp = nn.Sequential(
                self.projection,
                self.projection_ln,
                nn.GELU(),
                nn.Linear(tabular_dim, tabular_dim),
                nn.LayerNorm(tabular_dim),
                nn.GELU(),
            )
        self.out_projection = nn.Linear(tabular_dim, out_dim)
        self.out_ln = nn.LayerNorm(out_dim)

    def forward_tokens(self, tabular: torch.Tensor) -> torch.Tensor:
        """Return one token per raw feature for attention-based fusion."""
        if self.encoder_type != "transformer":
            raise ValueError("Feature tokens are only available for the transformer tabular encoder.")
        x = self.feature_projection(tabular.unsqueeze(-1))
        x = x + self.feature_embedding
        return self.transformer(x)

    def forward(self, tabular: torch.Tensor) -> torch.Tensor:
        if self.encoder_type == "transformer":
            x = self.forward_tokens(tabular).mean(dim=1)
        else:
            x = self.mlp(tabular)
        x = self.out_projection(x)
        x = self.out_ln(x)
        return x
