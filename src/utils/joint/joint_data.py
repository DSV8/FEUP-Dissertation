"""Build aligned CF and MM datasets for shared-parameter joint training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Callable
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from utils.image_embedding import (
    IMAGE_EMBEDDING_MODE_PRECOMPUTED,
    IMAGE_EMBEDDING_MODE_ENCODED_AT_RUNTIME,
    normalize_image_embedding_mode,
)
from utils.image_preprocessing import transform_fn
from utils.catalogue_utils import (
    Catalogue,
    build_catalogues,
    collect_catalogue_image_paths,
    normalize_catalogue_type,
    split_catalogues_canonical,
)
from utils.cf.cf_data import (
    CFInteraction,
    CFEncodedAtRuntimeInteraction,
    EncodedAtRuntimeInteractionCFDataset,
    create_cf_interactions as create_cf_precomputed_interactions,
    create_cf_encoded_at_runtime_interactions,
    infer_image_embedding_dim,
)
from utils.mm.mm_data import (
    normalize_tabular_if_needed,
    tabular_vec_from_img_path,
    prepare_multimodal_datasets_from_catalogues,
)

def resolve_image_embedding_mode_plan(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve image embedding mode once and expose reusable flags/resources."""
    mode = normalize_image_embedding_mode(
        str(metadata.get("image_embedding_mode", IMAGE_EMBEDDING_MODE_ENCODED_AT_RUNTIME))
    )
    use_precomputed = mode == IMAGE_EMBEDDING_MODE_PRECOMPUTED
    return {
        "mode": mode,
        "use_precomputed": use_precomputed,
        "image_embedding_map": metadata.get("image_embedding_map") if use_precomputed else None,
        "user_image_embedding_map": metadata.get("user_image_embedding_map") if use_precomputed else None,
    }

@dataclass
class MMTripletInteraction:
    query_img: np.ndarray
    query_tab: np.ndarray
    pos_img: np.ndarray
    pos_tab: np.ndarray
    neg_img: np.ndarray
    neg_tab: np.ndarray
    rank_gap: float

@dataclass
class JointCFMMBundle:
    cf_train_dataset: Dataset
    cf_test_dataset: Dataset
    mm_train_dataset: Dataset
    mm_test_dataset: Dataset
    train_catalogues: List[Catalogue]
    test_catalogues: List[Catalogue]
    image_embedding_dim: int
    tabular_stats: Dict[str, np.ndarray]
    metadata: Dict[str, Any]
    model_cardinalities: Optional[Dict[str, int]] = None

class EmbeddingTripletDataset(Dataset):
    """Triplet dataset that reads precomputed image embeddings + tabular vectors."""

    def __init__(self, interactions: List[MMTripletInteraction]):
        self.interactions = interactions

    def __len__(self) -> int:
        return len(self.interactions)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        x = self.interactions[index]
        return {
            "query_img": torch.tensor(x.query_img, dtype=torch.float32),
            "query_tab": torch.tensor(x.query_tab, dtype=torch.float32),
            "pos_img": torch.tensor(x.pos_img, dtype=torch.float32),
            "pos_tab": torch.tensor(x.pos_tab, dtype=torch.float32),
            "neg_img": torch.tensor(x.neg_img, dtype=torch.float32),
            "neg_tab": torch.tensor(x.neg_tab, dtype=torch.float32),
            "rank_gap": torch.tensor(x.rank_gap, dtype=torch.float32),
        }

class EncodedAtRuntimeMMAdapterDataset(Dataset):
    """Adapter to expose MM runtime-encoded dataset keys expected by the joint trainer."""

    def __init__(self, base_dataset: Dataset):
        self.base_dataset = base_dataset

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        x = self.base_dataset[index]
        return {
            "query_img": x["query_image"],
            "query_tab": x["query_tabular"],
            "pos_img": x["pos_image"],
            "pos_tab": x["pos_tabular"],
            "neg_img": x["neg_image"],
            "neg_tab": x["neg_tabular"],
            "rank_gap": x["rank_gap"],
        }

def create_mm_triplet_interactions(
    catalogues: List[Catalogue],
    image_embedding_map: Dict[str, np.ndarray],
    image_embedding_dim: int,
    tabular_stats: Optional[Dict[str, np.ndarray]] = None,
    include_catalogue_type_feature: bool = False,
) -> List[MMTripletInteraction]:
    """Build embedded MM triplets using the shared catalogue triplet builder."""
    zero_emb = np.zeros(image_embedding_dim, dtype=np.float32)
    interactions: List[MMTripletInteraction] = []
    triplets = create_multimodal_triplets(
        catalogues,
        include_catalogue_type_feature=include_catalogue_type_feature,
    )
    for triplet in triplets:
        interactions.append(
            MMTripletInteraction(
                query_img=image_embedding_map.get(triplet.query.image_path, zero_emb),
                query_tab=normalize_tabular_if_needed(triplet.query.tabular, tabular_stats),
                pos_img=image_embedding_map.get(triplet.pos.image_path, zero_emb),
                pos_tab=normalize_tabular_if_needed(triplet.pos.tabular, tabular_stats),
                neg_img=image_embedding_map.get(triplet.neg.image_path, zero_emb),
                neg_tab=normalize_tabular_if_needed(triplet.neg.tabular, tabular_stats),
                rank_gap=float(triplet.rank_gap),
            )
        )
    return interactions

def prepare_joint_cf_mm_datasets(
    data_dir: str,
    image_encoder: nn.Module,
    split_ratio: float = 0.8,
    seed: Optional[int] = 10,
    catalogue_type: Optional[str] = None,
    device: str = "cpu",
    image_transform: Callable[[str], torch.Tensor] = transform_fn,
    cache_dir: Optional[str] = None,
    use_cache: bool = True,
    normalize_tabular: bool = True,
    image_embedding_mode: str = IMAGE_EMBEDDING_MODE_PRECOMPUTED,
    tabular_dim: int = 5,
    tabular_embedding_dim: int = 32,
    fusion_embedding_dim: int = 64,
    tabular_nhead: int = 1,
    tabular_num_layers: int = 1,
    fusion_nhead: int = 2,
    fusion_num_layers: int = 2,
    fusion_mode: str = "cross_attention",
) -> JointCFMMBundle:
    """Prepare CF and MM datasets from one shared catalogue split and one embedding map."""
    image_embedding_mode = normalize_image_embedding_mode(image_embedding_mode)
    # One split and (in precomputed mode) one path-to-embedding map keep both
    # task datasets aligned to identical catalogue examples.
    catalogues = build_catalogues(data_dir)
    selected_type = normalize_catalogue_type(catalogue_type)
    train_catalogues, test_catalogues = split_catalogues_canonical(
        catalogues,
        split_ratio=split_ratio,
        seed=seed,
        catalogue_type=catalogue_type,
    )
    print(f"Loaded {len(catalogues)} catalogues — train: {len(train_catalogues)}, test: {len(test_catalogues)}")

    include_catalogue_type_feature = selected_type == "Mixed"

    tabular_stats: Dict[str, np.ndarray] = {}
    if normalize_tabular:
        train_paths = list({p for c in train_catalogues for p in [c.query] + list(c.neighbors)})
        train_tabs = [
            t
            for t in (
                tabular_vec_from_img_path(p, include_catalogue_type_feature=include_catalogue_type_feature)
                for p in train_paths
            )
            if t is not None
        ]
        if train_tabs:
            mat = np.stack(train_tabs, axis=0)
            tabular_stats = {"min": mat.min(axis=0), "max": mat.max(axis=0)}

    if image_embedding_mode == IMAGE_EMBEDDING_MODE_ENCODED_AT_RUNTIME:
        cf_train_dataset = EncodedAtRuntimeInteractionCFDataset(
            create_cf_encoded_at_runtime_interactions(
                train_catalogues,
                tabular_stats=tabular_stats,
                include_catalogue_type_feature=include_catalogue_type_feature,
            ),
            image_transform=image_transform,
            cache_dir=cache_dir,
            use_cache=use_cache,
        )
        cf_test_dataset = EncodedAtRuntimeInteractionCFDataset(
            create_cf_encoded_at_runtime_interactions(
                test_catalogues,
                tabular_stats=tabular_stats,
                include_catalogue_type_feature=include_catalogue_type_feature,
            ),
            image_transform=image_transform,
            cache_dir=cache_dir,
            use_cache=use_cache,
        )

        mm_train_base, mm_test_base, _ = prepare_multimodal_datasets_from_catalogues(
            train_catalogues,
            test_catalogues,
            normalize_tabular=normalize_tabular,
            image_transform=image_transform,
            cache_dir=cache_dir,
            use_cache=use_cache,
            include_catalogue_type_feature=include_catalogue_type_feature,
        )
        mm_train_dataset: Dataset = EncodedAtRuntimeMMAdapterDataset(mm_train_base)
        mm_test_dataset: Dataset = EncodedAtRuntimeMMAdapterDataset(mm_test_base)

        sample_path = catalogues[0].query if len(catalogues) > 0 else None
        image_embedding_dim = infer_image_embedding_dim(
            image_encoder=image_encoder,
            image_transform=image_transform,
            sample_path=sample_path,
            device=device,
            cache_dir=cache_dir,
            use_cache=use_cache,
        )

        metadata = {
            "image_embedding_mode": IMAGE_EMBEDDING_MODE_ENCODED_AT_RUNTIME,
            "image_transform": image_transform,
            "cache_dir": cache_dir,
            "use_cache": use_cache,
            "include_catalogue_type_feature": include_catalogue_type_feature,
        }

        return JointCFMMBundle(
            cf_train_dataset=cf_train_dataset,
            cf_test_dataset=cf_test_dataset,
            mm_train_dataset=mm_train_dataset,
            mm_test_dataset=mm_test_dataset,
            train_catalogues=train_catalogues,
            test_catalogues=test_catalogues,
            image_embedding_dim=image_embedding_dim,
            tabular_stats=tabular_stats,
            metadata=metadata,
            model_cardinalities={},
        )

    all_paths = collect_catalogue_image_paths(catalogues)
    print(f"Pre-computing image embeddings for {len(all_paths)} paths...")
    all_image_embedding_map = compute_img_embeddings(
        all_paths,
        image_encoder=image_encoder,
        device=device,
        image_transform=image_transform,
        cache_dir=cache_dir,
        use_cache=use_cache,
    )

    image_embedding_dim = int(next(iter(all_image_embedding_map.values())).shape[-1]) if all_image_embedding_map else 384
    user_image_embedding_map = {c.query: all_image_embedding_map[c.query] for c in catalogues if c.query in all_image_embedding_map}

    cf_train_dataset: Dataset = InteractionCFDataset(create_cf_precomputed_interactions(
        train_catalogues,
        all_image_embedding_map,
        image_embedding_dim,
        tabular_stats=tabular_stats,
        include_catalogue_type_feature=include_catalogue_type_feature,
    ))
    cf_test_dataset: Dataset = InteractionCFDataset(create_cf_precomputed_interactions(
        test_catalogues,
        all_image_embedding_map,
        image_embedding_dim,
        tabular_stats=tabular_stats,
        include_catalogue_type_feature=include_catalogue_type_feature,
    ))

    mm_train_dataset = EmbeddingTripletDataset(create_mm_triplet_interactions(
        train_catalogues,
        all_image_embedding_map,
        image_embedding_dim,
        tabular_stats=tabular_stats,
        include_catalogue_type_feature=include_catalogue_type_feature,
    ))
    mm_test_dataset = EmbeddingTripletDataset(create_mm_triplet_interactions(
        test_catalogues,
        all_image_embedding_map,
        image_embedding_dim,
        tabular_stats=tabular_stats,
        include_catalogue_type_feature=include_catalogue_type_feature,
    ))

    metadata = {
        "image_embedding_mode": IMAGE_EMBEDDING_MODE_PRECOMPUTED,
        "image_embedding_map": all_image_embedding_map,
        "user_image_embedding_map": user_image_embedding_map,
        "include_catalogue_type_feature": include_catalogue_type_feature,
    }

    return JointCFMMBundle(
        cf_train_dataset=cf_train_dataset,
        cf_test_dataset=cf_test_dataset,
        mm_train_dataset=mm_train_dataset,
        mm_test_dataset=mm_test_dataset,
        train_catalogues=train_catalogues,
        test_catalogues=test_catalogues,
        image_embedding_dim=image_embedding_dim,
        tabular_stats=tabular_stats,
        metadata=metadata,
        model_cardinalities={},
    )

def get_mm_dataloaders(
    train_dataset: Dataset,
    test_dataset: Dataset,
    batch_size: int = 256,
    num_workers: int = 8,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    return train_loader, test_loader
