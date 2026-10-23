"""Multimodal triplet data, filename-derived features, and normalization."""

import os
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from utils.image_embedding import (
    IMAGE_EMBEDDING_MODE_PRECOMPUTED,
    normalize_image_embedding_mode,
)
from utils.cache_preprocessing import get_or_create_cached_image
from utils.image_preprocessing import (
    transform_fn,
)
from utils.catalogue_utils import (
	Catalogue,
	build_catalogues,
	collect_catalogue_image_paths,
	normalize_catalogue_type,
	parse_catalogue_folder,
	parse_filename,
	split_catalogues_canonical,
)

def resolve_image_embedding_mode_plan(
	image_embedding_mode: str,
	image_embedding_map: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, Any]:
	"""Resolve mode once and expose small helpers for downstream conditionals."""
	mode = normalize_image_embedding_mode(image_embedding_mode)
	use_precomputed = mode == IMAGE_EMBEDDING_MODE_PRECOMPUTED
	return {
		"mode": mode,
		"use_precomputed": use_precomputed,
		"rank_eval_embedding_map": image_embedding_map if use_precomputed else None,
	}

@dataclass
class MultimodalSample:
	image_path: str
	tabular: np.ndarray

@dataclass
class MultimodalTriplet:
	query: MultimodalSample
	pos: MultimodalSample
	neg: MultimodalSample
	rank_gap: int = 1

@dataclass
class EmbeddingMultimodalTriplet:
	query_img: np.ndarray
	pos_img: np.ndarray
	neg_img: np.ndarray
	query_tab: np.ndarray
	pos_tab: np.ndarray
	neg_tab: np.ndarray
	rank_gap: int

def compute_rank_weight(
	rank_gap: torch.Tensor,
	rank_weight_alpha: float,
) -> torch.Tensor:
	"""Map rank gap to multiplicative triplet loss weights."""
	# gap=1 -> weight=1.0; larger gaps receive proportionally larger weights.
	weights = 1.0 + rank_weight_alpha * torch.clamp(rank_gap - 1.0, min=0.0)
	return weights

def convert_brasize_to_underbust(bra_band: str) -> Optional[float]:
	"""
	Convert bra band size (string) to midpoint under-bust in cm.
	Example: "75" -> midpoint of [75, 80).
	"""
	bra_band_dict: Dict[str, float] = {}
	for value in range(55, 155, 5):
		bra_band_dict[str(value)] = float(value + 2.5)

	try:
		return bra_band_dict[str(bra_band)]
	except Exception:
		return None

def convert_bracup_to_oubustdiff(bra_cup: str) -> Optional[float]:
	"""
	Convert bra cup label to midpoint over-under bust difference in cm.
	"""
	cups = ["AA"] + list("ABCDEFGHIJKLMNOPRST")
	cups_dict: Dict[str, float] = {}
	for cup, base in zip(cups, range(10, 50, 2)):
		cups_dict[cup] = float(base + 1.0)

	try:
		return cups_dict[str(bra_cup).upper()]
	except Exception:
		return None

def catalogue_type_feature_from_image_path(image_path: str) -> Optional[float]:
	"""Return catalogue type feature from image path: Excl -> 0.0, Fair -> 1.0."""
	catalogue_folder = os.path.basename(os.path.dirname(image_path))
	catalogue_type, _ = parse_catalogue_folder(catalogue_folder)
	if catalogue_type == "Excl":
		return 0.0
	if catalogue_type == "Fair":
		return 1.0
	return None

def tabular_vec_from_img_path(image_path: str, include_catalogue_type_feature: bool = False) -> Optional[np.ndarray]:
	"""
	Build tabular feature vector from encoded filename metadata.

	Output order: [age, height, weight, bra_cup_diff, bra_size_underbust]
	If include_catalogue_type_feature=True, append catalogue-type feature:
	Excl -> 0.0, Fair -> 1.0
	"""
	metadata = parse_filename(image_path.split("/")[-1])
	if metadata is None:
		return None

	age = float(metadata["age"])
	height = float(metadata["height"])
	weight = float(metadata["weight"])
	bra_cup_diff = convert_bracup_to_oubustdiff(metadata["bra_cup"])
	bra_size_underbust = convert_brasize_to_underbust(metadata["bra_size"])

	# These continuous values replace categorical cup/band labels with numeric
	# midpoints; malformed or missing fields cause the sample to be skipped.
	features = [age, height, weight, bra_cup_diff, bra_size_underbust]
	if include_catalogue_type_feature:
		catalogue_type_feature = catalogue_type_feature_from_image_path(image_path)
		features.append(catalogue_type_feature)

	if any(value is None for value in features):
		return None

	return np.asarray(features, dtype=np.float32)

def create_multimodal_triplets(
	catalogues: List[Catalogue],
	include_catalogue_type_feature: bool = False,
) -> List[MultimodalTriplet]:
	"""
	Create multimodal triplets by keeping the same ranking logic.

	For each catalogue and each pair of neighbors (i, j), keep triplet only when
	rank(i) < rank(j), i being positive and j being negative.
	"""
	triplets: List[MultimodalTriplet] = []

	for catalogue in catalogues:
		query_path = catalogue.query
		q_tab = tabular_vec_from_img_path(query_path, include_catalogue_type_feature=include_catalogue_type_feature)
		if q_tab is None:
			continue

		for i, j in combinations(range(len(catalogue.neighbors)), 2):
			pos_path = catalogue.neighbors[i]
			neg_path = catalogue.neighbors[j]

			pos_meta = parse_filename(pos_path.split("/")[-1])
			neg_meta = parse_filename(neg_path.split("/")[-1])

			if pos_meta is None or neg_meta is None:
				continue
			pos_rank = int(pos_meta["rank"]) if pos_meta.get("rank") is not None else i + 1
			neg_rank = int(neg_meta["rank"]) if neg_meta.get("rank") is not None else j + 1
			if pos_rank >= neg_rank:
				continue

			pos_tab = tabular_vec_from_img_path(pos_path, include_catalogue_type_feature=include_catalogue_type_feature)
			neg_tab = tabular_vec_from_img_path(neg_path, include_catalogue_type_feature=include_catalogue_type_feature)
			
			if pos_tab is None or neg_tab is None:
				continue

			triplets.append(
				MultimodalTriplet(
					query=MultimodalSample(image_path=query_path, tabular=q_tab),
					pos=MultimodalSample(image_path=pos_path, tabular=pos_tab),
					neg=MultimodalSample(image_path=neg_path, tabular=neg_tab),
					rank_gap=max(1, neg_rank - pos_rank),
				)
			)

	return triplets

def build_dataset_from_catalogues(
	catalogues: List[Catalogue],
	image_transform: Callable[[str], torch.Tensor],
	cache_dir: Optional[str],
	include_catalogue_type_feature: bool,
	tabular_stats: Optional[Dict[str, np.ndarray]],
) -> "TripletMultimodalDataset":
	"""Build one multimodal triplet dataset from a catalogue list."""
	triplets = create_multimodal_triplets(
		catalogues,
		include_catalogue_type_feature=include_catalogue_type_feature,
	)
	if tabular_stats is not None:
		triplets = apply_tabular_minmax(
			triplets,
			tabular_stats["min"],
			tabular_stats["max"],
		)

	return TripletMultimodalDataset(
		triplets,
		image_transform=image_transform,
		cache_dir=cache_dir,
	)

def fit_tabular_minmax(triplets: List[MultimodalTriplet]) -> Tuple[np.ndarray, np.ndarray]:
	"""
	Fit per-feature min/max statistics over all tabular vectors in triplets.
	"""
	vectors: List[np.ndarray] = []
	for t in triplets:
		vectors.extend([t.query.tabular, t.pos.tabular, t.neg.tabular])

	mat = np.stack(vectors, axis=0)
	feature_min = mat.min(axis=0)
	feature_max = mat.max(axis=0)
	return feature_min, feature_max

def apply_tabular_minmax(
	triplets: List[MultimodalTriplet],
	feature_min: np.ndarray,
	feature_max: np.ndarray,
) -> List[MultimodalTriplet]:
	"""
	Apply min-max normalization with precomputed stats.
	"""
	denom = np.where(feature_max > feature_min, feature_max - feature_min, 1.0)
	norm_triplets: List[MultimodalTriplet] = []

	for t in triplets:
		q_tab = ((t.query.tabular - feature_min) / denom).astype(np.float32)
		p_tab = ((t.pos.tabular - feature_min) / denom).astype(np.float32)
		n_tab = ((t.neg.tabular - feature_min) / denom).astype(np.float32)

		norm_triplets.append(
			MultimodalTriplet(
				query=MultimodalSample(image_path=t.query.image_path, tabular=q_tab),
				pos=MultimodalSample(image_path=t.pos.image_path, tabular=p_tab),
				neg=MultimodalSample(image_path=t.neg.image_path, tabular=n_tab),
				rank_gap=t.rank_gap,
			)
		)

	return norm_triplets

class TripletMultimodalDataset(Dataset):
	"""
	Dataset returning image tensors and tabular tensors for triplet training.
	"""

	def __init__(
		self,
		triplets: List[MultimodalTriplet],
		image_transform: Callable[[str], torch.Tensor] = transform_fn,
		cache_dir: Optional[str] = None,
	):
		self.triplets = triplets
		self.image_transform = image_transform
		self.cache_dir = cache_dir

	def __len__(self) -> int:
		return len(self.triplets)

	def load_cached_image(self, image_path: str) -> torch.Tensor:
		"""
		Load image tensor from cache when available, otherwise use transform.
		"""
		if self.cache_dir is not None:
			cached_tensor = get_or_create_cached_image(image_path, self.cache_dir)
			if cached_tensor is not None:
				return cached_tensor

		return self.image_transform(image_path)

	def __getitem__(self, index: int) -> Dict[str, Any]:
		triplet = self.triplets[index]

		return {
			"query_image": self.load_cached_image(triplet.query.image_path),
			"pos_image": self.load_cached_image(triplet.pos.image_path),
			"neg_image": self.load_cached_image(triplet.neg.image_path),
			"query_tabular": torch.tensor(triplet.query.tabular, dtype=torch.float32),
			"pos_tabular": torch.tensor(triplet.pos.tabular, dtype=torch.float32),
			"neg_tabular": torch.tensor(triplet.neg.tabular, dtype=torch.float32),
			"rank_gap": torch.tensor(float(triplet.rank_gap), dtype=torch.float32),
		}

class EmbeddingTripletMultimodalDataset(Dataset):
	"""Triplet dataset backed by precomputed image embeddings + tabular vectors."""

	def __init__(self, triplets: List[EmbeddingMultimodalTriplet]):
		self.triplets = triplets

	def __len__(self) -> int:
		return len(self.triplets)

	def __getitem__(self, index: int) -> Dict[str, Any]:
		t = self.triplets[index]
		return {
			"query_image": torch.tensor(t.query_img, dtype=torch.float32),
			"pos_image": torch.tensor(t.pos_img, dtype=torch.float32),
			"neg_image": torch.tensor(t.neg_img, dtype=torch.float32),
			"query_tabular": torch.tensor(t.query_tab, dtype=torch.float32),
			"pos_tabular": torch.tensor(t.pos_tab, dtype=torch.float32),
			"neg_tabular": torch.tensor(t.neg_tab, dtype=torch.float32),
			"rank_gap": torch.tensor(float(t.rank_gap), dtype=torch.float32),
		}

def compute_img_embeddings_mm(
	image_paths: List[str],
	image_encoder: nn.Module,
	device: str,
	image_transform: Callable[[str], torch.Tensor],
	cache_dir: Optional[str] = None,
	use_cache: bool = True,
	batch_size: int = 64,
) -> Dict[str, np.ndarray]:
	"""Encode images with a backbone and return L2-normalized vectors."""
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
					tensor = image_transform(path)
				batch_tensors.append(tensor)
				valid_paths.append(path)

			if not batch_tensors:
				continue

			batch = torch.stack(batch_tensors).to(device)
			embs = image_encoder(batch)
			embs = F.normalize(embs, p=2, dim=1, eps=1e-12)
			for path, emb in zip(valid_paths, embs.detach().cpu().numpy().astype(np.float32)):
				embedding_map[path] = emb

	return embedding_map

def build_embedding_triplet_dataset_from_path_dataset(
	path_dataset: TripletMultimodalDataset,
	image_embedding_map: Dict[str, np.ndarray],
	image_embedding_dim: int,
) -> EmbeddingTripletMultimodalDataset:
	zero_emb = np.zeros(image_embedding_dim, dtype=np.float32)
	emb_triplets: List[EmbeddingMultimodalTriplet] = []

	for t in path_dataset.triplets:
		emb_triplets.append(
			EmbeddingMultimodalTriplet(
				query_img=image_embedding_map.get(t.query.image_path, zero_emb),
				pos_img=image_embedding_map.get(t.pos.image_path, zero_emb),
				neg_img=image_embedding_map.get(t.neg.image_path, zero_emb),
				query_tab=t.query.tabular,
				pos_tab=t.pos.tabular,
				neg_tab=t.neg.tabular,
				rank_gap=t.rank_gap,
			)
		)

	return EmbeddingTripletMultimodalDataset(emb_triplets)

def prepare_precomputed_multimodal_datasets_from_catalogues(
	train_catalogues: List[Catalogue],
	test_catalogues: List[Catalogue],
	image_encoder: nn.Module,
	device: str,
	normalize_tabular: bool = True,
	image_transform: Callable[[str], torch.Tensor] = transform_fn,
	cache_dir: Optional[str] = None,
	use_cache: bool = True,
	include_catalogue_type_feature: bool = False,
) -> Tuple[EmbeddingTripletMultimodalDataset, EmbeddingTripletMultimodalDataset, Dict[str, np.ndarray], Dict[str, Any]]:
	"""Prepare MM datasets with precomputed image embeddings from fixed catalogue splits."""
	train_path_dataset, test_path_dataset, stats = prepare_multimodal_datasets_from_catalogues(
		train_catalogues,
		test_catalogues,
		normalize_tabular=normalize_tabular,
		image_transform=image_transform,
		cache_dir=cache_dir,
		use_cache=use_cache,
		include_catalogue_type_feature=include_catalogue_type_feature,
	)

	all_catalogues = list(train_catalogues) + list(test_catalogues)
	all_paths = collect_catalogue_image_paths(all_catalogues)
	print(f"Pre-computing image embeddings for MM on {len(all_paths)} paths...")
	image_embedding_map = compute_img_embeddings_mm(
		all_paths,
		image_encoder=image_encoder,
		device=device,
		image_transform=image_transform,
		cache_dir=cache_dir,
		use_cache=use_cache,
	)

	image_embedding_dim = int(next(iter(image_embedding_map.values())).shape[-1]) if image_embedding_map else 384
	train_dataset = build_embedding_triplet_dataset_from_path_dataset(train_path_dataset, image_embedding_map, image_embedding_dim)
	test_dataset = build_embedding_triplet_dataset_from_path_dataset(test_path_dataset, image_embedding_map, image_embedding_dim)

	metadata: Dict[str, Any] = {
		"image_embedding_mode": IMAGE_EMBEDDING_MODE_PRECOMPUTED,
		"image_embedding_map": image_embedding_map,
		"image_embedding_dim": image_embedding_dim,
	}
	return train_dataset, test_dataset, stats, metadata

def prepare_multimodal_datasets(
	root_dir: str,
	split_ratio: float = 0.8,
	seed: Optional[int] = 10,
	normalize_tabular: bool = True,
	image_transform: Callable[[str], torch.Tensor] = transform_fn,
	cache_dir: Optional[str] = None,
	use_cache: bool = True,
	catalogue_type: Optional[str] = None,
	include_catalogue_type_feature: bool = False,
) -> Tuple[TripletMultimodalDataset, TripletMultimodalDataset, Dict[str, np.ndarray]]:
	"""
	Build train/test multimodal triplet datasets from catalogue folders.

	Returns:
		train_dataset, test_dataset, stats
	where stats has "min" and "max" arrays for tabular features.
	"""
	catalogues = build_catalogues(root_dir)
	print(f"Loaded {len(catalogues)} catalogues from {root_dir}")

	selected_type = normalize_catalogue_type(catalogue_type)
	train_catalogues, test_catalogues = split_catalogues_canonical(
		catalogues,
		split_ratio=split_ratio,
		seed=seed,
		catalogue_type=catalogue_type,
	)
	print(f"Split: {len(train_catalogues)} train, {len(test_catalogues)} test")

	train_triplets = create_multimodal_triplets(
		train_catalogues,
		include_catalogue_type_feature=include_catalogue_type_feature,
	)
	test_triplets = create_multimodal_triplets(
		test_catalogues,
		include_catalogue_type_feature=include_catalogue_type_feature,
	)
	print(f"Generated {len(train_triplets)} train triplets, {len(test_triplets)} test triplets")

	stats: Dict[str, np.ndarray] = {}
	if normalize_tabular and len(train_triplets) > 0:
		feature_min, feature_max = fit_tabular_minmax(train_triplets)
		train_triplets = apply_tabular_minmax(train_triplets, feature_min, feature_max)
		test_triplets = apply_tabular_minmax(test_triplets, feature_min, feature_max)
		stats = {"min": feature_min, "max": feature_max}

	active_cache_dir = cache_dir if use_cache else None

	train_dataset = TripletMultimodalDataset(
		train_triplets,
		image_transform=image_transform,
		cache_dir=active_cache_dir,
	)
	test_dataset = TripletMultimodalDataset(
		test_triplets,
		image_transform=image_transform,
		cache_dir=active_cache_dir,
	)

	return train_dataset, test_dataset, stats

def prepare_multimodal_datasets_from_catalogues(
	train_catalogues: List[Catalogue],
	test_catalogues: List[Catalogue],
	normalize_tabular: bool = True,
	image_transform: Callable[[str], torch.Tensor] = transform_fn,
	cache_dir: Optional[str] = None,
	use_cache: bool = True,
	include_catalogue_type_feature: bool = False,
) -> Tuple[TripletMultimodalDataset, TripletMultimodalDataset, Dict[str, np.ndarray]]:
	"""Build multimodal datasets from already-split catalogue lists."""
	train_triplets = create_multimodal_triplets(
		train_catalogues,
		include_catalogue_type_feature=include_catalogue_type_feature,
	)
	test_triplets = create_multimodal_triplets(
		test_catalogues,
		include_catalogue_type_feature=include_catalogue_type_feature,
	)

	stats: Dict[str, np.ndarray] = {}
	# Derive feature ranges from train triplets, then apply those same ranges to
	# test data to avoid test statistics influencing the learned representation.
	if normalize_tabular and len(train_triplets) > 0:
		feature_min, feature_max = fit_tabular_minmax(train_triplets)
		train_triplets = apply_tabular_minmax(train_triplets, feature_min, feature_max)
		test_triplets = apply_tabular_minmax(test_triplets, feature_min, feature_max)
		stats = {"min": feature_min, "max": feature_max}

	active_cache_dir = cache_dir if use_cache else None

	train_dataset = TripletMultimodalDataset(
		train_triplets,
		image_transform=image_transform,
		cache_dir=active_cache_dir,
	)
	test_dataset = TripletMultimodalDataset(
		test_triplets,
		image_transform=image_transform,
		cache_dir=active_cache_dir,
	)

	return train_dataset, test_dataset, stats

def get_dataloaders_multimodal(
	train_dataset: TripletMultimodalDataset,
	test_dataset: TripletMultimodalDataset,
	batch_size: int = 16,
	num_workers: int = 0,
	pin_memory: bool = True,
	persistent_workers: bool = False,
	prefetch_factor: int = 2,
) -> Tuple[DataLoader, DataLoader]:
	"""
	Create dataloaders for multimodal triplet datasets
	"""
	use_persistent_workers = persistent_workers and num_workers > 0
	loader_kwargs: Dict[str, Any] = {
		"batch_size": batch_size,
		"num_workers": num_workers,
		"pin_memory": pin_memory,
		"persistent_workers": use_persistent_workers,
	}
	if num_workers > 0:
		loader_kwargs["prefetch_factor"] = prefetch_factor

	train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
	test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

	return train_loader, test_loader

def build_ranking_transform_multimodal(cache_dir: str, use_cache: bool = True) -> Callable[[str], torch.Tensor]:
	"""Create a ranking-evaluation transform that prefers cached tensors for multimodal data."""
	def ranking_transform(image_path: str) -> torch.Tensor:
		if cache_dir is not None and use_cache:
			cached_tensor = get_or_create_cached_image(image_path, cache_dir)
			if cached_tensor is not None:
				return cached_tensor
		return transform_fn(image_path)
	return ranking_transform

def normalize_tabular_if_needed(
	tabular_vec: np.ndarray,
	tabular_stats: Optional[Dict[str, np.ndarray]],
) -> np.ndarray:
	"""Normalize one tabular vector using provided min/max stats."""
	if tabular_stats is None:
		return tabular_vec.astype(np.float32)

	feature_min = tabular_stats.get("min")
	feature_max = tabular_stats.get("max")
	if feature_min is None or feature_max is None:
		return tabular_vec.astype(np.float32)

	denom = np.where(feature_max > feature_min, feature_max - feature_min, 1.0)
	return ((tabular_vec - feature_min) / denom).astype(np.float32)
