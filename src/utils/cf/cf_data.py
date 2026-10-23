"""CF interaction records, image loading, rank targets, and dataset assembly."""

from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from utils.image_embedding import (
	IMAGE_EMBEDDING_MODE_PRECOMPUTED,
	IMAGE_EMBEDDING_MODE_ENCODED_AT_RUNTIME,
	normalize_image_embedding_mode,
)
from utils.cache_preprocessing import get_or_create_cached_image
from utils.image_preprocessing import (
    transform_fn,
)
from utils.catalogue_utils import (
	Catalogue,
	build_catalogues,
	normalize_catalogue_type,
	parse_filename,
	split_catalogues_canonical,
)
from utils.mm.mm_data import normalize_tabular_if_needed, tabular_vec_from_img_path

@dataclass
class CFDatasetBundle:
	train_dataset: Dataset
	test_dataset: Dataset
	train_catalogues: List[Catalogue]
	test_catalogues: List[Catalogue]
	image_embedding_dim: int
	tabular_stats: Dict[str, np.ndarray]
	metadata: Dict[str, Any]   # image_embedding_map, user_image_embedding_map
	model_cardinalities: Optional[Dict[str, int]] = None

@dataclass
class CFInteraction:
	query_img: np.ndarray
	query_tab: np.ndarray
	item_img: np.ndarray
	item_tab: np.ndarray
	target: float

@dataclass
class CFEncodedAtRuntimeInteraction:
	query_path: str
	query_tab: np.ndarray
	item_path: str
	item_tab: np.ndarray
	target: float

class InteractionCFDataset(Dataset):
	"""Pointwise interaction dataset for true user-item supervision."""

	def __init__(self, interactions: List[CFInteraction]):
		self.interactions = interactions

	def __len__(self) -> int:
		return len(self.interactions)

	def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
		x = self.interactions[index]
		return {
			"query_img": torch.tensor(x.query_img, dtype=torch.float32),
			"query_tab": torch.tensor(x.query_tab, dtype=torch.float32),
			"item_img": torch.tensor(x.item_img, dtype=torch.float32),
			"item_tab": torch.tensor(x.item_tab, dtype=torch.float32),
			"target": torch.tensor(x.target, dtype=torch.float32),
		}

class EncodedAtRuntimeInteractionCFDataset(Dataset):
	"""Pointwise interaction dataset backed by image tensors encoded at runtime + tabular vectors."""

	def __init__(
		self,
		interactions: List[CFEncodedAtRuntimeInteraction],
		image_transform: Callable[[str], torch.Tensor] = transform_fn,
		cache_dir: Optional[str] = None,
		use_cache: bool = True,
	):
		self.interactions = interactions
		self.image_transform = image_transform
		self.cache_dir = cache_dir
		self.use_cache = use_cache

	def __len__(self) -> int:
		return len(self.interactions)

	def _load_image(self, image_path: str) -> torch.Tensor:
		if self.use_cache and self.cache_dir is not None:
			cached_tensor = get_or_create_cached_image(image_path, self.cache_dir)
			if cached_tensor is not None:
				return cached_tensor
		return self.image_transform(image_path)

	def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
		x = self.interactions[index]
		return {
			"query_img": self._load_image(x.query_path),
			"query_tab": torch.tensor(x.query_tab, dtype=torch.float32),
			"item_img": self._load_image(x.item_path),
			"item_tab": torch.tensor(x.item_tab, dtype=torch.float32),
			"target": torch.tensor(x.target, dtype=torch.float32),
		}

class CFEncodedAtRuntimeScoringWrapper(nn.Module):
	"""Wrap a CF scorer so it can consume raw image tensors in runtime-encoding mode."""

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
		return self.base_model((query_img, item_img, query_tab, item_tab))

def rank_to_target(rank: int, max_rank: int) -> float:
	"""Map rank to [0,1], where 1 is best rank and 0 is worst rank."""
	# Scores are trained with MSE and ranked descending: rank 1 maps to 1.
	if max_rank <= 1:
		return 1.0
	r = max(1, min(rank, max_rank))
	return float((max_rank - r) / (max_rank - 1))

def create_cf_interactions(
	catalogues: List[Catalogue],
	image_embedding_map: Dict[str, np.ndarray],
	image_embedding_dim: int,
	tabular_stats: Optional[Dict[str, np.ndarray]] = None,
	include_catalogue_type_feature: bool = False,
) -> List[CFInteraction]:
	"""Build explicit user-item interactions for pointwise CF supervision."""
	zero_emb = np.zeros(image_embedding_dim, dtype=np.float32)
	interactions: List[CFInteraction] = []

	for catalogue in catalogues:
		query_meta = parse_filename(os.path.basename(catalogue.query))
		q_tab_raw = tabular_vec_from_img_path(
			catalogue.query,
			include_catalogue_type_feature=include_catalogue_type_feature,
		)
		if query_meta is None or q_tab_raw is None:
			continue
		query_img = image_embedding_map.get(catalogue.query, zero_emb)
		query_tab = normalize_tabular_if_needed(q_tab_raw, tabular_stats)
		max_rank = max(1, len(catalogue.neighbors))

		for idx, neighbor_path in enumerate(catalogue.neighbors):
			item_meta = parse_filename(os.path.basename(neighbor_path))
			item_tab_raw = tabular_vec_from_img_path(
				neighbor_path,
				include_catalogue_type_feature=include_catalogue_type_feature,
			)
			if item_meta is None or item_tab_raw is None:
				continue

			rank = int(item_meta["rank"]) if item_meta.get("rank") is not None else (idx + 1)
			target = rank_to_target(rank, max_rank)

			interactions.append(CFInteraction(
				query_img=query_img,
				query_tab=query_tab,
				item_img=image_embedding_map.get(neighbor_path, zero_emb),
				item_tab=normalize_tabular_if_needed(item_tab_raw, tabular_stats),
				target=target,
			))

	return interactions

def create_cf_encoded_at_runtime_interactions(
	catalogues: List[Catalogue],
	tabular_stats: Optional[Dict[str, np.ndarray]] = None,
	include_catalogue_type_feature: bool = False,
) -> List[CFEncodedAtRuntimeInteraction]:
	"""Build pointwise interactions using image paths encoded at runtime."""
	interactions: List[CFEncodedAtRuntimeInteraction] = []

	for catalogue in catalogues:
		query_meta = parse_filename(os.path.basename(catalogue.query))
		q_tab_raw = tabular_vec_from_img_path(
			catalogue.query,
			include_catalogue_type_feature=include_catalogue_type_feature,
		)
		if query_meta is None or q_tab_raw is None:
			continue

		query_tab = normalize_tabular_if_needed(q_tab_raw, tabular_stats)
		max_rank = max(1, len(catalogue.neighbors))

		for idx, neighbor_path in enumerate(catalogue.neighbors):
			item_meta = parse_filename(os.path.basename(neighbor_path))
			item_tab_raw = tabular_vec_from_img_path(
				neighbor_path,
				include_catalogue_type_feature=include_catalogue_type_feature,
			)
			if item_meta is None or item_tab_raw is None:
				continue

			rank = int(item_meta["rank"]) if item_meta.get("rank") is not None else (idx + 1)
			target = rank_to_target(rank, max_rank)

			interactions.append(CFEncodedAtRuntimeInteraction(
				query_path=catalogue.query,
				query_tab=query_tab,
				item_path=neighbor_path,
				item_tab=normalize_tabular_if_needed(item_tab_raw, tabular_stats),
				target=target,
			))

	return interactions

def _make_pointwise_scoring_inputs(batch: Dict[str, torch.Tensor], device: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
	return (
		batch["query_img"].to(device),
		batch["item_img"].to(device),
		batch["query_tab"].to(device),
		batch["item_tab"].to(device),
	)

def compute_img_embeddings(
	image_paths: List[str],
	image_encoder: nn.Module,
	device: str,
	image_transform: Callable[[str], torch.Tensor],
	cache_dir: Optional[str] = None,
	use_cache: bool = True,
	batch_size: int = 64,
) -> Dict[str, np.ndarray]:
	"""Encode images with the backbone and return L2-normalised embedding vectors."""
	image_encoder.eval()
	image_encoder.to(device)
	embedding_map: Dict[str, np.ndarray] = {}
	failed_paths: List[str] = []

	with torch.no_grad():
		for start_idx in range(0, len(image_paths), batch_size):
			chunk = image_paths[start_idx : start_idx + batch_size]
			batch_tensors: List[torch.Tensor] = []
			valid_paths: List[str] = []
			for path in chunk:
				tensor: Optional[torch.Tensor] = None
				try:
					if use_cache and cache_dir is not None:
						tensor = get_or_create_cached_image(path, cache_dir)
					if tensor is None:
						tensor = image_transform(path)
				except Exception as exc:
					print(f"Skipping image embedding for {path}: {exc}")
					failed_paths.append(path)
					continue

				batch_tensors.append(tensor)
				valid_paths.append(path)

			if not batch_tensors:
				continue

			batch = torch.stack(batch_tensors).to(device)
			embs = image_encoder(batch)
			embs = F.normalize(embs, p=2, dim=1, eps=1e-12)
			for path, emb in zip(valid_paths, embs.detach().cpu().numpy().astype(np.float32)):
				embedding_map[path] = emb

	if failed_paths:
		print(f"Warning: failed to encode {len(failed_paths)} image(s). They may fall back to zero embeddings.")

	return embedding_map

def load_img_tensor(
	image_path: str,
	image_transform: Callable[[str], torch.Tensor],
	cache_dir: Optional[str],
	use_cache: bool,
) -> torch.Tensor:
	if use_cache and cache_dir is not None:
		cached_tensor = get_or_create_cached_image(image_path, cache_dir)
		if cached_tensor is not None:
			return cached_tensor
	return image_transform(image_path)

def infer_image_embedding_dim(
	image_encoder: nn.Module,
	image_transform: Callable[[str], torch.Tensor],
	sample_path: Optional[str],
	device: str,
	cache_dir: Optional[str] = None,
	use_cache: bool = True,
) -> int:
	if hasattr(image_encoder, "model") and hasattr(image_encoder.model, "num_features"):
		return int(image_encoder.model.num_features)
	if hasattr(image_encoder, "num_features"):
		return int(image_encoder.num_features)
	if sample_path is None:
		return 384

	img_tensor = load_img_tensor(sample_path, image_transform, cache_dir, use_cache)
	was_training = image_encoder.training
	image_encoder.eval()
	image_encoder.to(device)
	with torch.no_grad():
		out = image_encoder(img_tensor.unsqueeze(0).to(device))
	if was_training:
		image_encoder.train()
	return int(out.shape[-1]) if out.ndim >= 2 else 384

def prepare_cf_datasets(
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
	model_variant: str = "NCF",
	image_embedding_mode: str = IMAGE_EMBEDDING_MODE_PRECOMPUTED,
) -> CFDatasetBundle:
	"""
	Prepare train/test TripletCFDatasets from catalogue data, mirroring prepare_multimodal_datasets.

	Images are pre-computed once with the backbone encoder so the training loop
	never needs to run the backbone again.
	"""
	image_embedding_mode = normalize_image_embedding_mode(image_embedding_mode)
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

	# Fit scaling on training paths only to prevent test-set statistics leaking
	# into model inputs.
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

	normalized_variant = model_variant.strip().upper()
	if normalized_variant not in {"FM", "NN", "NCF"}:
		raise ValueError("Invalid model_variant for CF pipeline. Use one of: 'FM', 'NN', 'NCF'.")

	if image_embedding_mode == IMAGE_EMBEDDING_MODE_PRECOMPUTED:
		# Collect all unique paths that need embedding
		all_paths = sorted({path for c in catalogues for path in [c.query] + list(c.neighbors)})
		all_queries = sorted({c.query for c in catalogues})

		print(f"Pre-computing image embeddings for {len(all_paths)} paths...")
		all_image_embedding_map = compute_img_embeddings(
			all_paths, image_encoder, device, image_transform, cache_dir, use_cache,
		)
		user_image_embedding_map = {p: all_image_embedding_map[p] for p in all_queries if p in all_image_embedding_map}

		image_embedding_dim = int(next(iter(all_image_embedding_map.values())).shape[-1]) if all_image_embedding_map else 384

		train_dataset: Dataset = InteractionCFDataset(create_cf_interactions(
			train_catalogues,
			all_image_embedding_map,
			image_embedding_dim,
			tabular_stats,
			include_catalogue_type_feature=include_catalogue_type_feature,
		))
		test_dataset: Dataset = InteractionCFDataset(create_cf_interactions(
			test_catalogues,
			all_image_embedding_map,
			image_embedding_dim,
			tabular_stats,
			include_catalogue_type_feature=include_catalogue_type_feature,
		))
		print(f"Samples — train: {len(train_dataset)}, test: {len(test_dataset)}")

		metadata = {
			"image_embedding_mode": IMAGE_EMBEDDING_MODE_PRECOMPUTED,
			"image_embedding_map": all_image_embedding_map,
			"user_image_embedding_map": user_image_embedding_map,
			"include_catalogue_type_feature": include_catalogue_type_feature,
		}
	else:
		train_dataset = EncodedAtRuntimeInteractionCFDataset(
			create_cf_encoded_at_runtime_interactions(
				train_catalogues,
				tabular_stats=tabular_stats,
				include_catalogue_type_feature=include_catalogue_type_feature,
			),
			image_transform=image_transform,
			cache_dir=cache_dir,
			use_cache=use_cache,
		)
		test_dataset = EncodedAtRuntimeInteractionCFDataset(
			create_cf_encoded_at_runtime_interactions(
				test_catalogues,
				tabular_stats=tabular_stats,
				include_catalogue_type_feature=include_catalogue_type_feature,
			),
			image_transform=image_transform,
			cache_dir=cache_dir,
			use_cache=use_cache,
		)
		image_embedding_dim = infer_image_embedding_dim(
			image_encoder=image_encoder,
			image_transform=image_transform,
			sample_path=(catalogues[0].query if len(catalogues) > 0 else None),
			device=device,
			cache_dir=cache_dir,
			use_cache=use_cache,
		)
		print(f"Samples — train: {len(train_dataset)}, test: {len(test_dataset)}")
		metadata = {
			"image_embedding_mode": IMAGE_EMBEDDING_MODE_ENCODED_AT_RUNTIME,
			"image_transform": image_transform,
			"cache_dir": cache_dir,
			"use_cache": use_cache,
			"include_catalogue_type_feature": include_catalogue_type_feature,
		}

	return CFDatasetBundle(
		train_dataset=train_dataset,
		test_dataset=test_dataset,
		train_catalogues=train_catalogues,
		test_catalogues=test_catalogues,
		image_embedding_dim=image_embedding_dim,
		tabular_stats=tabular_stats,
		model_cardinalities={},
		metadata=metadata,
	)

def get_cf_dataloaders(
	train_dataset: Dataset,
	test_dataset: Dataset,
	batch_size: int = 256,
	num_workers: int = 0,
	pin_memory: bool = False,
	persistent_workers: bool = False,
	prefetch_factor: int = 2,
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
	test_loader  = DataLoader(
		test_dataset,
		batch_size=batch_size,
		shuffle=False,
		num_workers=num_workers,
		pin_memory=pin_memory,
		persistent_workers=persistent_workers,
		prefetch_factor=prefetch_factor,
	)
	return train_loader, test_loader
