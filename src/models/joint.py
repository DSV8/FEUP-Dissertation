"""Shared model and loss-weighting module for joint CF/MM optimization."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from models.collaborative import FM_model, NCF_model, NN_model
from models.multimodal import MultimodalEncoder


class JointCFMMModel(nn.Module):
    """MM encoder + configurable CF scorer (FM/NN/NCF) with joint multi-loss training."""

    def __init__(
        self,
        image_embedding_dim: int,
        tabular_dim: int = 5,
        tabular_embedding_dim: int = 64,
        fusion_embedding_dim: int = 64,
        tabular_nhead: int = 1,
        tabular_num_layers: int = 1,
        tabular_encoder_type: str = "transformer",
        fusion_nhead: int = 2,
        fusion_num_layers: int = 2,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        image_encoder: Optional[nn.Module] = None,
        device: str = "cpu",
        cf_model_variant: str = "NCF",
        cf_use_fused_input: bool = True,
        cf_projection_dim: int = 64,
        cf_fm_k: int = 8,
        cf_lay_1: int = 256,
        cf_lay_2: int = 128,
        cf_lay_3: int = 64,
        cf_lay_4: int = 32,
        cf_ncf_fusion_heads: int = 2,
        cf_ncf_fusion_layers: int = 2,
        cf_ncf_fusion_dim: int = 64,
        cf_ncf_dropout: Optional[float] = None,
        fusion_mode: str = "cross_attention",
        freeze_image: bool = False,
        freeze_tabular: bool = False,
    ):
        super().__init__()
        _ = hidden_dim  # kept for backward compatibility
        self.device = device
        self.cf_model_variant = str(cf_model_variant).strip().upper()
        self.cf_use_fused_input = bool(cf_use_fused_input)
        if self.cf_model_variant not in {"FM", "NN", "NCF"}:
            raise ValueError("Invalid cf_model_variant. Use one of: 'FM', 'NN', 'NCF'.")

        # FM/NN/NCF tabular towers support variable tabular width via config.
        # This keeps Mixed catalogue (6 features) compatible when selected.
        self.tabular_dim = int(tabular_dim)
        self.fusion_embedding_dim = fusion_embedding_dim
        self.image_encoder = image_encoder

        # MM branch: always use the dedicated MultimodalEncoder.
        # In precomputed mode, image tensors are already embeddings, so Identity is used.
        mm_image_encoder = image_encoder if image_encoder is not None else nn.Identity()
        self.mm_encoder = MultimodalEncoder(
            image_encoder=mm_image_encoder,
            tabular_in_features=tabular_dim,
            tabular_embedding_dim=tabular_embedding_dim,
            fusion_embedding_dim=fusion_embedding_dim,
            tabular_nhead=tabular_nhead,
            tabular_num_layers=tabular_num_layers,
            tabular_encoder_type=tabular_encoder_type,
            fusion_nhead=fusion_nhead,
            fusion_num_layers=fusion_num_layers,
            image_encoder_output_dim=image_embedding_dim,
            fusion_mode=fusion_mode,
        )

        cfg = {
            "device": device,
            "dropout": dropout,
            "image_embedding_dim": fusion_embedding_dim,
            "projection_dim": cf_projection_dim,
            "k": cf_fm_k,
            "lay_1": cf_lay_1,
            "lay_2": cf_lay_2,
            "lay_3": cf_lay_3,
            "lay_4": cf_lay_4,
            "tabular_in_features": tabular_dim,
            "tabular_encoder_dim": tabular_embedding_dim,
            "tabular_nhead": tabular_nhead,
            "tabular_num_layers": tabular_num_layers,
            "tabular_encoder_type": tabular_encoder_type,
            "ncf_fusion_dim": cf_ncf_fusion_dim,
            "ncf_fusion_heads": cf_ncf_fusion_heads,
            "ncf_fusion_layers": cf_ncf_fusion_layers,
            "ncf_dropout": cf_ncf_dropout if cf_ncf_dropout is not None else dropout,
            "fused_input": self.cf_use_fused_input,
        }
        if self.cf_model_variant == "FM":
            self.cf_model = FM_model(cfg)
        elif self.cf_model_variant == "NN":
            self.cf_model = NN_model(cfg)
        else:
            fm_model = FM_model(cfg)
            nn_model = NN_model(cfg)
            self.cf_model = NCF_model(
                cfg,
                nn_model=nn_model,
                fm_model=fm_model,
                fusion_mode=fusion_mode,
            )

        self.freeze_backbones(freeze_image=freeze_image, freeze_tabular=freeze_tabular)

    def freeze_backbones(
        self,
        freeze_image: bool = False,
        freeze_tabular: bool = False,
    ) -> None:
        """Freeze modality encoders and their projections for ablation runs."""
        self.freeze_image = bool(freeze_image)
        self.freeze_tabular = bool(freeze_tabular)

        if hasattr(self.mm_encoder, "image_encoder"):
            for param in self.mm_encoder.image_encoder.parameters():
                param.requires_grad = not self.freeze_image
        if hasattr(self.mm_encoder, "image_projection"):
            for param in self.mm_encoder.image_projection.parameters():
                param.requires_grad = not self.freeze_image
        if hasattr(self.mm_encoder, "image_projection_ln"):
            for param in self.mm_encoder.image_projection_ln.parameters():
                param.requires_grad = not self.freeze_image

        if hasattr(self.mm_encoder, "tabular_encoder"):
            for param in self.mm_encoder.tabular_encoder.parameters():
                param.requires_grad = not self.freeze_tabular

        # With both modalities fixed, keep the multimodal representation fixed too.
        fusion_requires_grad = not (self.freeze_image and self.freeze_tabular)
        if hasattr(self.mm_encoder, "fusion_module"):
            for param in self.mm_encoder.fusion_module.parameters():
                param.requires_grad = fusion_requires_grad

    def enforce_frozen_backbone_eval_mode(self) -> None:
        """Keep frozen representation modules deterministic while the scorer trains."""
        if self.freeze_image and hasattr(self.mm_encoder, "image_encoder"):
            self.mm_encoder.image_encoder.eval()
        if self.freeze_image and hasattr(self.mm_encoder, "image_projection"):
            self.mm_encoder.image_projection.eval()
        if self.freeze_tabular and hasattr(self.mm_encoder, "tabular_encoder"):
            self.mm_encoder.tabular_encoder.eval()
        if self.freeze_image and self.freeze_tabular:
            if hasattr(self.mm_encoder, "fusion_module"):
                self.mm_encoder.fusion_module.eval()

    def to_image_embedding_if_needed(self, image_tensor: torch.Tensor) -> torch.Tensor:
        if image_tensor.ndim <= 2:
            return image_tensor
        if self.image_encoder is None:
            raise ValueError("Received raw image tensors, but model has no image encoder configured.")
        return self.image_encoder(image_tensor)

    def encode(self, image_emb: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        return self.mm_encoder(image_emb, tabular)

    def score(
        self,
        query_img: torch.Tensor,
        item_img: torch.Tensor,
        query_tab: torch.Tensor,
        item_tab: torch.Tensor,
    ) -> torch.Tensor:
        if self.cf_use_fused_input:
            # The MM encoder creates a shared representation; the CF head then
            # learns a scalar relevance score for each query-item pair.
            query_emb = self.encode(query_img, query_tab)
            item_emb = self.encode(item_img, item_tab)
            return self.cf_model((query_emb, item_emb, query_tab, item_tab)).view(-1)
        return self.cf_model((query_img, item_img, query_tab, item_tab)).view(-1)

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        query_img, item_img, query_tab, item_tab = x
        return self.score(query_img, item_img, query_tab, item_tab)


class UncertaintyWeightingLoss(nn.Module):
    """Uncertainty weighting for the CF and MM objectives."""

    def __init__(self):
        super().__init__()
        self.log_var_cf = nn.Parameter(torch.zeros(1))
        self.log_var_mm = nn.Parameter(torch.zeros(1))

    def forward(self, cf_loss: torch.Tensor, mm_loss: torch.Tensor) -> torch.Tensor:
        weighted_cf = torch.exp(-self.log_var_cf) * cf_loss + self.log_var_cf
        weighted_mm = torch.exp(-self.log_var_mm) * mm_loss + self.log_var_mm
        return weighted_cf + weighted_mm

    def stats(self) -> Dict[str, float]:
        return {
            "log_var_cf": float(self.log_var_cf.detach().cpu().item()),
            "log_var_mm": float(self.log_var_mm.detach().cpu().item()),
            "w_cf": float(torch.exp(-self.log_var_cf).detach().cpu().item()),
            "w_mm": float(torch.exp(-self.log_var_mm).detach().cpu().item()),
        }