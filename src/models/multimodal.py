"""Image/tabular embedding model used by MM and the joint pipeline."""

from typing import Optional
import torch
import torch.nn as nn
from .fusion import CROSS_ATTENTION, CrossAttentionFusion, Transformer, normalize_fusion_mode
from .tabular_encoders import TabularEncoder


class MultimodalEncoder(nn.Module):
    """
    Multimodal encoder combining:
    - GCViT for image encoding
    - Transformer for tabular encoding
    - Transformer or cross-attention for fusion of image and tabular embeddings
    """

    def __init__(
        self,
        image_encoder: nn.Module,
        tabular_in_features: int = 5,
        tabular_embedding_dim: int = 32,
        fusion_embedding_dim: int = 64,
        tabular_nhead: int = 1,
        tabular_num_layers: int = 1,
        tabular_encoder_type: str = "transformer",
        fusion_nhead: int = 2,
        fusion_num_layers: int = 2,
        image_encoder_output_dim: Optional[int] = None,
        fusion_mode: str = CROSS_ATTENTION,
    ):
        """
        Args:
            image_encoder: Pre-initialized image encoder
            tabular_in_features: Number of tabular input features (default: 5)
            tabular_embedding_dim: Internal dimension used by the tabular transformer
            fusion_embedding_dim: Shared dimension used by the image projection and fusion transformer
            tabular_nhead: Number of attention heads in the tabular transformer
            tabular_num_layers: Number of layers in the tabular transformer
            fusion_nhead: Number of attention heads in the fusion transformer
            fusion_num_layers: Number of layers in the fusion transformer
            image_encoder_output_dim: Optional explicit image feature size (If not provided, tries to infer automatically)
        """
        super(MultimodalEncoder, self).__init__()
        self.image_encoder = image_encoder
        self.tabular_embedding_dim = tabular_embedding_dim
        self.fusion_embedding_dim = fusion_embedding_dim

        # Optional image projection allows a compact fusion dimension when backbone output is larger than desired fusion size.
        if image_encoder_output_dim is None:
            if hasattr(image_encoder, "model") and hasattr(image_encoder.model, "num_features"):
                image_encoder_output_dim = int(image_encoder.model.num_features)
            elif hasattr(image_encoder, "num_features"):
                image_encoder_output_dim = int(image_encoder.num_features)

        self.image_encoder_output_dim = image_encoder_output_dim
        if image_encoder_output_dim is not None and image_encoder_output_dim != fusion_embedding_dim:
            self.image_projection = nn.Linear(image_encoder_output_dim, fusion_embedding_dim)
            self.image_projection_ln = nn.LayerNorm(fusion_embedding_dim)
        else:
            self.image_projection = nn.Identity()
            self.image_projection_ln = nn.Identity()

        # Reuse the dedicated tabular module
        self.tabular_encoder = TabularEncoder(
            in_features=tabular_in_features,
            tabular_dim=tabular_embedding_dim,
            out_dim=fusion_embedding_dim,
            nhead=tabular_nhead,
            num_layers=tabular_num_layers,
            encoder_type=tabular_encoder_type,
        )
        self.fusion_mode = normalize_fusion_mode(fusion_mode)
        if self.fusion_mode == CROSS_ATTENTION:
            self.fusion_module = CrossAttentionFusion(
                embed_dim=fusion_embedding_dim,
                nhead=fusion_nhead,
                dropout=0.1,
            )
        else:
            self.fusion_module = Transformer(
                d_model=fusion_embedding_dim,
                nhead=fusion_nhead,
                num_layers=fusion_num_layers,
                dim_feedforward=fusion_embedding_dim * 4,
                batch_first=True,
            )

    def forward(self, image: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for multimodal encoder.

        Args:
            image: Image tensor of shape (batch_size, channels, height, width)
            tabular: Tabular tensor of shape (batch_size, num_features)

        Returns:
            Fused embedding of shape (batch_size, fusion_embedding_dim)
        """
        # Encode image.
        image_embedding = self.image_encoder(image)                             # (batch_size, image_encoder_output_dim)
        if image_embedding.shape[-1] != self.fusion_embedding_dim:
            if isinstance(self.image_projection, nn.Identity):
                raise ValueError(
                    "Image encoder output dim mismatch: "
                    f"got {image_embedding.shape[-1]}, expected {self.fusion_embedding_dim}. "
                    "Provide image_encoder_output_dim to enable projection, "
                    "or set fusion_embedding_dim to match the image encoder output."
                )
            image_embedding = self.image_projection(image_embedding)
            image_embedding = self.image_projection_ln(image_embedding)

        # Encode tabular with the reusable tabular module.
        tabular_embedding = self.tabular_encoder(tabular)                       # (batch_size, fusion_embedding_dim)

        if self.fusion_mode == CROSS_ATTENTION:
            if self.tabular_encoder.encoder_type == "transformer":
                tabular_tokens = self.tabular_encoder.forward_tokens(tabular)
                tabular_tokens = self.tabular_encoder.out_projection(tabular_tokens)
                tabular_tokens = self.tabular_encoder.out_ln(tabular_tokens)
            else:
                tabular_tokens = tabular_embedding.unsqueeze(1)
            fused_embedding = self.fusion_module(image_embedding, tabular_tokens)
        else:
            image_token = image_embedding.unsqueeze(1)
            if self.tabular_encoder.encoder_type == "transformer":
                tabular_tokens = self.tabular_encoder.forward_tokens(tabular)
                tabular_tokens = self.tabular_encoder.out_projection(tabular_tokens)
                tabular_tokens = self.tabular_encoder.out_ln(tabular_tokens)
                fused_input = torch.cat([image_token, tabular_tokens], dim=1)
            else:
                fused_input = torch.cat([image_token, tabular_embedding.unsqueeze(1)], dim=1)
            fused_output = self.fusion_module(fused_input)

            # Mean pooling across the image and tabular tokens.
            fused_embedding = fused_output.mean(dim=1)

        fused_embedding = torch.nn.functional.normalize(fused_embedding, p=2, dim=1)
        return fused_embedding
