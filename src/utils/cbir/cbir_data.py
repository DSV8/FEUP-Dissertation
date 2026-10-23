"""CBIR datasets and utilities for runtime or precomputed image embeddings."""

import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.cache_preprocessing import get_or_create_cached_image
from utils.image_preprocessing import transform_fn
from utils.image_embedding import (
    IMAGE_EMBEDDING_MODE_ENCODED_AT_RUNTIME,
    IMAGE_EMBEDDING_MODE_PRECOMPUTED,
    normalize_image_embedding_mode,
)
from utils.catalogue_utils import (
    Catalogue,
    build_catalogues,
    collect_catalogue_image_paths,
    create_triplets,
    infer_catalogue_type_id_from_image_path,
    parse_filename,
    split_catalogues_canonical,
)


class CatalogueConcatEncoder(torch.nn.Module):
    """Wrap an image encoder and concatenate catalogue-type label to embeddings."""

    def __init__(self, base_encoder: torch.nn.Module):
        super().__init__()
        self.base_encoder = base_encoder

    def forward(self, image: torch.Tensor, catalogue_type_id: Optional[torch.Tensor] = None) -> torch.Tensor:
        base_embedding = self.base_encoder(image)
        if catalogue_type_id is None:
            return base_embedding

        type_feature = catalogue_type_id.float().view(-1, 1)
        return torch.cat([base_embedding, type_feature], dim=1)


class PrecomputedEmbeddingEncoder(torch.nn.Module):
    """Lightweight trainable encoder for precomputed embedding inputs."""

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.projection = torch.nn.Linear(embedding_dim, embedding_dim)
        with torch.no_grad():
            self.projection.weight.copy_(torch.eye(embedding_dim))
            self.projection.bias.zero_()

    def forward(self, embedding: torch.Tensor, catalogue_type_id: Optional[torch.Tensor] = None) -> torch.Tensor:
        _ = catalogue_type_id
        return self.projection(embedding)


def compute_rank_gap_from_paths(pos_path: str, neg_path: str) -> int:
    """Compute rank gap (neg_rank - pos_rank) from neighbor filenames."""
    pos_metadata = parse_filename(os.path.basename(pos_path))
    neg_metadata = parse_filename(os.path.basename(neg_path))
    if pos_metadata is None or neg_metadata is None:
        return 1

    pos_rank = pos_metadata.get("rank")
    neg_rank = neg_metadata.get("rank")
    if pos_rank is None or neg_rank is None:
        return 1

    # A gap of one is the minimum valid preference and receives the base loss weight.
    return max(1, int(neg_rank) - int(pos_rank))


def compute_rank_weight(rank_gap: torch.Tensor, rank_weight_alpha: float) -> torch.Tensor:
    """Map rank gap to a multiplicative loss weight."""
    return 1.0 + rank_weight_alpha * torch.clamp(rank_gap - 1.0, min=0.0)


class TripletImageDataset(Dataset):
    """Triplet training dataset that loads cached or transformed images on demand."""

    def __init__(
        self,
        triplets: List[Tuple[Any, ...]],
        transform: Optional[Callable[[str], torch.Tensor]] = None,
        cache_dir: Optional[str] = None,
    ):
        self.triplets = triplets
        self.transform = transform
        self.cache_dir = cache_dir

    def __len__(self) -> int:
        return len(self.triplets)

    def load_cached_image(self, image_path: str) -> torch.Tensor:
        if self.cache_dir is not None:
            cached_tensor = get_or_create_cached_image(image_path, self.cache_dir)
            if cached_tensor is not None:
                return cached_tensor

        if self.transform is not None:
            return self.transform(image_path)
        raise ValueError(f"No cache available and no transform provided for {image_path}")

    def __getitem__(self, index: int) -> Dict[str, Any]:
        triplet = self.triplets[index]
        if len(triplet) == 4:
            query_path, pos_path, neg_path, catalogue_type_id = triplet
        else:
            query_path, pos_path, neg_path = triplet
            catalogue_type_id = infer_catalogue_type_id_from_image_path(query_path)

        rank_gap = float(compute_rank_gap_from_paths(pos_path, neg_path))
        return {
            "query": self.load_cached_image(query_path),
            "pos": self.load_cached_image(pos_path),
            "neg": self.load_cached_image(neg_path),
            "rank_gap": torch.tensor(rank_gap, dtype=torch.float32),
            "catalogue_type_id": torch.tensor(int(catalogue_type_id), dtype=torch.long),
        }


class EmbeddingTripletDataset(Dataset):
    """Triplet dataset backed by precomputed image embeddings."""

    def __init__(
        self,
        triplets: List[Tuple[Any, ...]],
        image_embedding_map: Dict[str, np.ndarray],
        image_embedding_dim: int,
    ):
        self.triplets = triplets
        self.image_embedding_map = image_embedding_map
        self.image_embedding_dim = image_embedding_dim
        self.zero_emb = np.zeros(image_embedding_dim, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        triplet = self.triplets[index]
        if len(triplet) == 4:
            query_path, pos_path, neg_path, catalogue_type_id = triplet
        else:
            query_path, pos_path, neg_path = triplet
            catalogue_type_id = infer_catalogue_type_id_from_image_path(query_path)

        rank_gap = float(compute_rank_gap_from_paths(pos_path, neg_path))
        return {
            "query": torch.tensor(self.image_embedding_map.get(query_path, self.zero_emb), dtype=torch.float32),
            "pos": torch.tensor(self.image_embedding_map.get(pos_path, self.zero_emb), dtype=torch.float32),
            "neg": torch.tensor(self.image_embedding_map.get(neg_path, self.zero_emb), dtype=torch.float32),
            "rank_gap": torch.tensor(rank_gap, dtype=torch.float32),
            "catalogue_type_id": torch.tensor(int(catalogue_type_id), dtype=torch.long),
        }


def compute_img_embeddings_cbir(
    image_paths: List[str],
    image_encoder: torch.nn.Module,
    device: str,
    transform: Callable[[str], torch.Tensor],
    cache_dir: Optional[str] = None,
    use_cache: bool = True,
    include_catalogue_type_id: bool = False,
    batch_size: int = 64,
) -> Dict[str, np.ndarray]:
    """Compute image embeddings for CBIR precomputed mode."""
    image_encoder.eval()
    image_encoder.to(device)
    embedding_map: Dict[str, np.ndarray] = {}

    with torch.no_grad():
        for start_idx in range(0, len(image_paths), batch_size):
            chunk = image_paths[start_idx : start_idx + batch_size]
            batch_tensors: List[torch.Tensor] = []
            valid_paths: List[str] = []

            for path in chunk:
                tensor: Optional[torch.Tensor] = None
                if use_cache and cache_dir is not None:
                    tensor = get_or_create_cached_image(path, cache_dir)
                if tensor is None:
                    tensor = transform(path)
                batch_tensors.append(tensor)
                valid_paths.append(path)

            if not batch_tensors:
                continue

            embs = image_encoder(torch.stack(batch_tensors).to(device))
            embs_np = embs.detach().cpu().numpy().astype(np.float32)
            for path, emb in zip(valid_paths, embs_np):
                if include_catalogue_type_id:
                    type_id = float(infer_catalogue_type_id_from_image_path(path))
                    emb = np.concatenate([emb, np.array([type_id], dtype=np.float32)], axis=0)
                embedding_map[path] = emb

    return embedding_map


def build_precomputed_dataset_from_catalogues(
    catalogues: List[Catalogue],
    image_embedding_map: Dict[str, np.ndarray],
    image_embedding_dim: int,
    include_catalogue_type_id: bool = False,
) -> EmbeddingTripletDataset:
    triplets = create_triplets(catalogues, include_catalogue_type_id=include_catalogue_type_id)
    return EmbeddingTripletDataset(triplets, image_embedding_map, image_embedding_dim)


def build_dataset_from_catalogues(
    catalogues: List[Catalogue],
    transform: Optional[Callable[[str], torch.Tensor]],
    cache_dir: Optional[str],
    include_catalogue_type_id: bool = False,
) -> TripletImageDataset:
    """Create a triplet dataset from a list of catalogues."""
    triplets = create_triplets(catalogues, include_catalogue_type_id=include_catalogue_type_id)
    return TripletImageDataset(triplets, transform=transform, cache_dir=cache_dir)


def build_ranking_transform(cache_dir: str, use_cache: bool = True) -> Callable[[str], torch.Tensor]:
    """Create a ranking-evaluation transform that prefers cached tensors."""
    def ranking_transform(image_path: str) -> torch.Tensor:
        cached_tensor = get_or_create_cached_image(image_path, cache_dir) if use_cache else None
        if cached_tensor is not None:
            return cached_tensor
        return transform_fn(image_path)

    return ranking_transform


def prepare_datasets(
    data_dir: str,
    split_ratio: float = 0.8,
    seed: Optional[int] = 10,
    cache_dir: Optional[str] = None,
    use_cache: bool = True,
    catalogue_type: Optional[str] = None,
    include_catalogue_type_id: bool = False,
) -> Tuple[TripletImageDataset, TripletImageDataset]:
    """Build triplet datasets from a catalogue directory and canonical split."""
    catalogues = build_catalogues(data_dir)
    print(f"Loaded {len(catalogues)} catalogues from {data_dir}")
    train_catalogues, test_catalogues = split_catalogues_canonical(
        catalogues, split_ratio=split_ratio, seed=seed, catalogue_type=catalogue_type
    )
    print(f"Split: {len(train_catalogues)} train, {len(test_catalogues)} test")
    active_cache_dir = cache_dir if use_cache else None
    train_dataset = build_dataset_from_catalogues(
        train_catalogues, transform_fn, active_cache_dir, include_catalogue_type_id
    )
    test_dataset = build_dataset_from_catalogues(
        test_catalogues, transform_fn, active_cache_dir, include_catalogue_type_id
    )
    print(f"Generated {len(train_dataset)} train triplets, {len(test_dataset)} test triplets")
    return train_dataset, test_dataset


def prepare_datasets_from_catalogues(
    train_catalogues: List[Catalogue],
    test_catalogues: List[Catalogue],
    cache_dir: Optional[str] = None,
    use_cache: bool = True,
    include_catalogue_type_id: bool = False,
) -> Tuple[TripletImageDataset, TripletImageDataset]:
    """Build train/test datasets from already-split catalogue lists."""
    active_cache_dir = cache_dir if use_cache else None
    return (
        build_dataset_from_catalogues(train_catalogues, transform_fn, active_cache_dir, include_catalogue_type_id),
        build_dataset_from_catalogues(test_catalogues, transform_fn, active_cache_dir, include_catalogue_type_id),
    )


def prepare_precomputed_datasets_from_catalogues(
    train_catalogues: List[Catalogue],
    test_catalogues: List[Catalogue],
    image_encoder: torch.nn.Module,
    device: str,
    cache_dir: Optional[str] = None,
    use_cache: bool = True,
    include_catalogue_type_id: bool = False,
) -> Tuple[EmbeddingTripletDataset, EmbeddingTripletDataset, Dict[str, Any]]:
    """Build precomputed-embedding triplet datasets from already split catalogues."""
    all_catalogues = list(train_catalogues) + list(test_catalogues)
    all_paths = collect_catalogue_image_paths(all_catalogues)
    print(f"Pre-computing CBIR embeddings for {len(all_paths)} paths...")
    image_embedding_map = compute_img_embeddings_cbir(
        all_paths,
        image_encoder=image_encoder,
        device=device,
        transform=transform_fn,
        cache_dir=cache_dir,
        use_cache=use_cache,
        include_catalogue_type_id=include_catalogue_type_id,
    )
    image_embedding_dim = int(next(iter(image_embedding_map.values())).shape[-1]) if image_embedding_map else 384
    train_dataset = build_precomputed_dataset_from_catalogues(
        train_catalogues, image_embedding_map, image_embedding_dim, include_catalogue_type_id
    )
    test_dataset = build_precomputed_dataset_from_catalogues(
        test_catalogues, image_embedding_map, image_embedding_dim, include_catalogue_type_id
    )
    metadata: Dict[str, Any] = {
        "image_embedding_mode": IMAGE_EMBEDDING_MODE_PRECOMPUTED,
        "image_embedding_map": image_embedding_map,
        "image_embedding_dim": image_embedding_dim,
        "include_catalogue_type_id": include_catalogue_type_id,
    }
    return train_dataset, test_dataset, metadata