"""Image-only CF ablation, reusing CF interactions and training utilities."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from cf import train_cf_model
from utils.image_embedding import (
    IMAGE_EMBEDDING_MODE_ENCODED_AT_RUNTIME,
    IMAGE_EMBEDDING_MODE_PRECOMPUTED,
    normalize_image_embedding_mode,
)
from utils.cf.cf_data import (
    CFDatasetBundle,
    CFEncodedAtRuntimeScoringWrapper,
    get_cf_dataloaders,
    prepare_cf_datasets,
)
from utils.image_preprocessing import transform_fn
from models.collaborative import FM_model, NCF_model, NN_model


def build_image_only_cf_model(
    model_variant: str,
    image_embedding_dim: int,
    hidden_dims: Optional[List[int]] = None,
    dropout: float = 0.4,
    k: int = 8,
    fusion_dim: Optional[int] = None,
    fusion_heads: int = 2,
    fusion_layers: int = 2,
    fusion_dropout: Optional[float] = None,
    device: str = "cpu",
) -> nn.Module:
    """Build the same FM/NN/NCF architecture as cf.py but without tabular inputs.

    The tabular branches are intentionally ignored by setting fused_input=True,
    which keeps the regular CF model structure while making the tabular side a
    no-op for the ablation pipeline.
    """
    normalized_variant = model_variant.strip().upper()
    model_config: Dict[str, Any] = {
        "device": device,
        "image_embedding_dim": image_embedding_dim,
        "projection_dim": 64,
        "k": k,
        "lay_1": 256,
        "lay_2": 128,
        "lay_3": 64,
        "lay_4": 32,
        "dropout": dropout,
        "use_tabular": False,
        "ncf_fusion_dim": fusion_dim or 64,
        "ncf_fusion_heads": fusion_heads,
        "ncf_fusion_layers": fusion_layers,
        "ncf_dropout": fusion_dropout if fusion_dropout is not None else dropout,
    }

    if normalized_variant == "FM":
        return FM_model(model_config)
    if normalized_variant == "NN":
        if hidden_dims is None:
            hidden_dims = [128, 64, 32, 16]
        model_config["lay_1"] = hidden_dims[0]
        model_config["lay_2"] = hidden_dims[1]
        model_config["lay_3"] = hidden_dims[2]
        model_config["lay_4"] = hidden_dims[3]
        return NN_model(model_config)
    if normalized_variant == "NCF":
        if hidden_dims is None:
            hidden_dims = [128, 64, 32, 16]
        model_config["lay_1"] = hidden_dims[0]
        model_config["lay_2"] = hidden_dims[1]
        model_config["lay_3"] = hidden_dims[2]
        model_config["lay_4"] = hidden_dims[3]
        fm_model = FM_model(model_config)
        nn_model = NN_model(model_config)
        return NCF_model(
            model_config,
            nn_model=nn_model,
            fm_model=fm_model,
            fusion_mode="cross_attention",
        )

    raise ValueError("Invalid model_variant. Use one of: 'FM', 'NN', 'NCF'.")


class ImageOnlyScoringWrapper(nn.Module):
    """Compatibility wrapper that uses the same CF scoring path with zero tabular inputs."""

    def __init__(self, base_model: nn.Module, image_encoder: nn.Module):
        super().__init__()
        self.base_model = base_model
        self.image_encoder = image_encoder

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        query_img, item_img, query_tab, item_tab = x
        if query_img.ndim > 2:
            query_img = self.image_encoder(query_img)
        if item_img.ndim > 2:
            item_img = self.image_encoder(item_img)
        zero_query_tab = torch.zeros_like(query_tab)
        zero_item_tab = torch.zeros_like(item_tab)
        return self.base_model((query_img, item_img, zero_query_tab, zero_item_tab))


def prepare_cf_io_datasets(
    data_dir: str,
    image_encoder: nn.Module,
    split_ratio: float = 0.8,
    seed: Optional[int] = 10,
    catalogue_type: Optional[str] = None,
    device: str = "cpu",
    image_transform: Callable[[str], torch.Tensor] = transform_fn,
    cache_dir: Optional[str] = None,
    use_cache: bool = True,
    model_variant: str = "NCF",
    image_embedding_mode: str = IMAGE_EMBEDDING_MODE_PRECOMPUTED,
    normalize_tabular: bool = False,
) -> CFDatasetBundle:
    """Prepare CF dataset bundle while making the tabular branch inactive."""
    return prepare_cf_datasets(
        data_dir=data_dir,
        image_encoder=image_encoder,
        split_ratio=split_ratio,
        seed=seed,
        catalogue_type=catalogue_type,
        device=device,
        image_transform=image_transform,
        cache_dir=cache_dir,
        use_cache=use_cache,
        normalize_tabular=normalize_tabular,
        model_variant=model_variant,
        image_embedding_mode=normalize_image_embedding_mode(image_embedding_mode),
    )


get_cf_io_dataloaders = get_cf_dataloaders
train_cf_io_model = train_cf_model


__all__ = [
    "CFDatasetBundle",
    "CFEncodedAtRuntimeScoringWrapper",
    "ImageOnlyScoringWrapper",
    "build_image_only_cf_model",
    "prepare_cf_io_datasets",
    "get_cf_io_dataloaders",
    "train_cf_io_model",
    "IMAGE_EMBEDDING_MODE_PRECOMPUTED",
    "IMAGE_EMBEDDING_MODE_ENCODED_AT_RUNTIME",
    "normalize_image_embedding_mode",
]
