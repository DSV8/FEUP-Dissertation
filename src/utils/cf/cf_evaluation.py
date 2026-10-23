"""CF pairwise accuracy and ranking metrics for both image modes."""

from __future__ import annotations
import os
from itertools import combinations
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
)
import numpy as np
import torch
import torch.nn as nn
from utils.image_embedding import (
    IMAGE_EMBEDDING_MODE_PRECOMPUTED,
	IMAGE_EMBEDDING_MODE_ENCODED_AT_RUNTIME,
	normalize_image_embedding_mode,
)
from utils.image_preprocessing import (
    transform_fn,
)
from utils.catalogue_utils import (
    Catalogue,
    parse_filename,
)
from utils.mm.mm_data import normalize_tabular_if_needed, tabular_vec_from_img_path

def evaluate_cf_accuracy(
	catalogues: List[Catalogue],
	model: nn.Module,
	metadata: Dict[str, Any],
	device: str,
	tabular_stats: Optional[Dict[str, np.ndarray]] = None,
	model_variant: str = "NCF",
) -> float:
	"""Pairwise ordering accuracy: fraction of correctly ordered neighbour pairs."""
	image_embedding_mode = normalize_image_embedding_mode(
		str(metadata.get("image_embedding_mode", IMAGE_EMBEDDING_MODE_PRECOMPUTED))
	)
	include_catalogue_type_feature = bool(metadata.get("include_catalogue_type_feature", False))
	if image_embedding_mode == IMAGE_EMBEDDING_MODE_ENCODED_AT_RUNTIME:
		return evaluate_cf_accuracy_encoded_at_runtime(
			catalogues,
			model,
			device=device,
			tabular_stats=tabular_stats,
			image_transform=metadata.get("image_transform", transform_fn),
			cache_dir=metadata.get("cache_dir"),
			use_cache=bool(metadata.get("use_cache", True)),
			include_catalogue_type_feature=include_catalogue_type_feature,
			model_variant=model_variant,
		)

	model.eval()
	image_embedding_map = metadata["image_embedding_map"]
	user_image_embedding_map = metadata["user_image_embedding_map"]
	include_catalogue_type_feature = bool(metadata.get("include_catalogue_type_feature", False))

	image_emb_dim = int(next(iter(image_embedding_map.values())).shape[-1]) if image_embedding_map else 384
	zero_emb = np.zeros(image_emb_dim, dtype=np.float32)

	normalized_variant = model_variant.strip().upper()
	if normalized_variant not in {"FM", "NN", "NCF"}:
		raise ValueError("Invalid model_variant for CF pairwise accuracy. Use one of: 'FM', 'NN', 'NCF'.")

	correct_pairs = 0
	total_pairs = 0

	with torch.no_grad():
		for catalogue in catalogues:
			if len(catalogue.neighbors) < 2:
				continue

			q_tab_raw = tabular_vec_from_img_path(
				catalogue.query,
				include_catalogue_type_feature=include_catalogue_type_feature,
			)
			if q_tab_raw is None:
				continue
			q_tab = normalize_tabular_if_needed(q_tab_raw, tabular_stats)
			q_img_np = user_image_embedding_map.get(catalogue.query, zero_emb)

			n_imgs: List[np.ndarray] = []
			n_tabs: List[np.ndarray] = []
			ranks: List[int] = []
			valid_count = 0

			for idx, path in enumerate(catalogue.neighbors):
				meta = parse_filename(os.path.basename(path))
				tab = tabular_vec_from_img_path(
					path,
					include_catalogue_type_feature=include_catalogue_type_feature,
				)
				if meta is None or tab is None:
					continue

				rank = int(meta["rank"]) if meta.get("rank") is not None else (idx + 1)
				n_imgs.append(image_embedding_map.get(path, zero_emb))
				n_tabs.append(normalize_tabular_if_needed(tab, tabular_stats))
				ranks.append(rank)
				valid_count += 1

			if valid_count < 2:
				continue

			user_img_batch = torch.tensor(np.repeat(q_img_np[None, :], valid_count, axis=0), dtype=torch.float32, device=device)
			item_img_batch = torch.tensor(np.stack(n_imgs), dtype=torch.float32, device=device)
			user_tab_batch = torch.tensor(np.repeat(q_tab[None, :], valid_count, axis=0), dtype=torch.float32, device=device)
			item_tab_batch = torch.tensor(np.stack(n_tabs), dtype=torch.float32, device=device)

			raw_scores = model((user_img_batch, item_img_batch, user_tab_batch, item_tab_batch)).detach().cpu().numpy()

			for i, j in combinations(range(valid_count), 2):
				rank_i, rank_j = ranks[i], ranks[j]
				if rank_i == rank_j:
					continue

				score_i, score_j = float(raw_scores[i]), float(raw_scores[j])
				if rank_i < rank_j:
					correct_pairs += int(score_i > score_j)
				else:
					correct_pairs += int(score_j > score_i)
				total_pairs += 1

	return (correct_pairs / total_pairs) if total_pairs > 0 else 0.0

def evaluate_cf_accuracy_encoded_at_runtime(
	catalogues: List[Catalogue],
	model: nn.Module,
	device: str,
	tabular_stats: Optional[Dict[str, np.ndarray]] = None,
	image_transform: Callable[[str], torch.Tensor] = transform_fn,
	cache_dir: Optional[str] = None,
	use_cache: bool = True,
	include_catalogue_type_feature: bool = False,
	model_variant: str = "NCF",
) -> float:
	"""Pairwise ordering accuracy with runtime image encoding path."""
	model.eval()

	normalized_variant = model_variant.strip().upper()
	if normalized_variant not in {"FM", "NN", "NCF"}:
		raise ValueError("Invalid model_variant for CF pairwise accuracy. Use one of: 'FM', 'NN', 'NCF'.")

	correct_pairs = 0
	total_pairs = 0

	with torch.no_grad():
		for catalogue in catalogues:
			if len(catalogue.neighbors) < 2:
				continue

			q_tab_raw = tabular_vec_from_img_path(
				catalogue.query,
				include_catalogue_type_feature=include_catalogue_type_feature,
			)
			if q_tab_raw is None:
				continue
			q_tab = normalize_tabular_if_needed(q_tab_raw, tabular_stats)
			q_img_tensor = load_img_tensor(catalogue.query, image_transform, cache_dir, use_cache)

			n_imgs: List[torch.Tensor] = []
			n_tabs: List[np.ndarray] = []
			ranks: List[int] = []

			for idx, path in enumerate(catalogue.neighbors):
				meta = parse_filename(os.path.basename(path))
				tab = tabular_vec_from_img_path(
					path,
					include_catalogue_type_feature=include_catalogue_type_feature,
				)
				if meta is None or tab is None:
					continue
				rank = int(meta["rank"]) if meta.get("rank") is not None else (idx + 1)
				n_imgs.append(load_img_tensor(path, image_transform, cache_dir, use_cache))
				n_tabs.append(normalize_tabular_if_needed(tab, tabular_stats))
				ranks.append(rank)

			valid_count = len(n_imgs)
			if valid_count < 2:
				continue

			user_img_batch = torch.stack([q_img_tensor] * valid_count).to(device)
			item_img_batch = torch.stack(n_imgs).to(device)
			user_tab_batch = torch.tensor(np.repeat(q_tab[None, :], valid_count, axis=0), dtype=torch.float32, device=device)
			item_tab_batch = torch.tensor(np.stack(n_tabs), dtype=torch.float32, device=device)
			raw_scores = model((user_img_batch, item_img_batch, user_tab_batch, item_tab_batch)).detach().cpu().numpy()

			for i, j in combinations(range(valid_count), 2):
				rank_i, rank_j = ranks[i], ranks[j]
				if rank_i == rank_j:
					continue

				score_i, score_j = float(raw_scores[i]), float(raw_scores[j])
				if rank_i < rank_j:
					correct_pairs += int(score_i > score_j)
				else:
					correct_pairs += int(score_j > score_i)
				total_pairs += 1

	return (correct_pairs / total_pairs) if total_pairs > 0 else 0.0

def evaluate_catalogue_metrics_cf(
	catalogues: List[Catalogue],
	model: nn.Module,
	metadata: Dict[str, Any],
	device: str,
	tabular_stats: Optional[Dict[str, np.ndarray]] = None,
	gain: str = "linear",
	model_variant: str = "NCF",
) -> Dict[str, float]:
	"""Ranking metrics over full catalogues using score-based FM/NN/NCF predictions."""
	image_embedding_mode = normalize_image_embedding_mode(
		str(metadata.get("image_embedding_mode", IMAGE_EMBEDDING_MODE_PRECOMPUTED))
	)
	include_catalogue_type_feature = bool(metadata.get("include_catalogue_type_feature", False))
	if image_embedding_mode == IMAGE_EMBEDDING_MODE_ENCODED_AT_RUNTIME:
		return evaluate_catalogue_metrics_cf_encoded_at_runtime(
			catalogues,
			model,
			device=device,
			tabular_stats=tabular_stats,
			gain=gain,
			image_transform=metadata.get("image_transform", transform_fn),
			cache_dir=metadata.get("cache_dir"),
			use_cache=bool(metadata.get("use_cache", True)),
			include_catalogue_type_feature=include_catalogue_type_feature,
			model_variant=model_variant,
		)

	model.eval()
	image_embedding_map      = metadata["image_embedding_map"]
	user_image_embedding_map = metadata["user_image_embedding_map"]

	image_emb_dim = int(next(iter(image_embedding_map.values())).shape[-1]) if image_embedding_map else 384
	zero_emb = np.zeros(image_emb_dim, dtype=np.float32)

	ndcg_values:   List[float] = []
	all_pred_scores: List[float] = []
	all_gt_targets:  List[float] = []

	normalized_variant = model_variant.strip().upper()
	if normalized_variant not in {"FM", "NN", "NCF"}:
		raise ValueError("Invalid model_variant for CF ranking metrics. Use one of: 'FM', 'NN', 'NCF'.")

	with torch.no_grad():
		for catalogue in catalogues:
			if len(catalogue.neighbors) < 2:
				continue

			query_meta = parse_filename(os.path.basename(catalogue.query))
			q_tab_raw = tabular_vec_from_img_path(
				catalogue.query,
				include_catalogue_type_feature=include_catalogue_type_feature,
			)
			if query_meta is None or q_tab_raw is None:
				continue
			q_tab = normalize_tabular_if_needed(q_tab_raw, tabular_stats)
			q_img_np = user_image_embedding_map.get(catalogue.query, zero_emb)

			n_imgs, n_tabs, valid_paths = [], [], []
			for path in catalogue.neighbors:
				meta = parse_filename(os.path.basename(path))
				tab = tabular_vec_from_img_path(
					path,
					include_catalogue_type_feature=include_catalogue_type_feature,
				)
				if meta is None or tab is None:
					continue
				n_imgs.append(image_embedding_map.get(path, zero_emb))
				n_tabs.append(normalize_tabular_if_needed(tab, tabular_stats))
				valid_paths.append(path)

			if len(valid_paths) < 2:
				continue

			num_neighbors = len(valid_paths)
			user_img_batch = torch.tensor(np.repeat(q_img_np[None, :], num_neighbors, axis=0), dtype=torch.float32, device=device)
			item_img_batch = torch.tensor(np.stack(n_imgs), dtype=torch.float32, device=device)
			user_tab_batch = torch.tensor(np.repeat(q_tab[None, :], num_neighbors, axis=0), dtype=torch.float32, device=device)
			item_tab_batch = torch.tensor(np.stack(n_tabs), dtype=torch.float32, device=device)
			raw_scores = model((user_img_batch, item_img_batch, user_tab_batch, item_tab_batch)).detach().cpu().numpy()
			pred_indices = np.argsort(-raw_scores)

			# Pointwise RMSE
			# GT targets: rank 1 -> 1.0, rank N -> 0.0
			gt_targets = np.arange(num_neighbors - 1, -1, -1, dtype=np.float64)
			if num_neighbors > 1:
				gt_targets = gt_targets / (num_neighbors - 1)
			# Predicted scores: normalize model scores to [0, 1]
			d_min, d_max = raw_scores.min(), raw_scores.max()
			if d_max > d_min:
				pred_scores = (raw_scores - d_min) / (d_max - d_min)
			else:
				pred_scores = np.ones(num_neighbors, dtype=np.float64)
			all_pred_scores.extend(pred_scores.tolist())
			all_gt_targets.extend(gt_targets.tolist())

			# nDCG
			gt_relevance = np.arange(num_neighbors, 0, -1, dtype=np.float64)
			if gain == "graded":
				pred_gain  = 2.0 ** gt_relevance[pred_indices] - 1.0
				ideal_gain = 2.0 ** gt_relevance - 1.0
			else:  # linear
				pred_gain  = gt_relevance[pred_indices]
				ideal_gain = gt_relevance
			discounts = np.log2(np.arange(2, num_neighbors + 2))
			dcg  = float(np.sum(pred_gain  / discounts))
			idcg = float(np.sum(ideal_gain / discounts))
			ndcg_values.append(dcg / idcg if idcg > 0 else 0.0)

	if not ndcg_values:
		return {"ndcg": 0.0, "rmse": 0.0}

	rmse = float(np.sqrt(np.mean((np.array(all_pred_scores) - np.array(all_gt_targets)) ** 2)))
	return {
		"ndcg": float(np.mean(ndcg_values)),
		"rmse": rmse,
	}

def evaluate_catalogue_metrics_cf_encoded_at_runtime(
	catalogues: List[Catalogue],
	model: nn.Module,
	device: str,
	tabular_stats: Optional[Dict[str, np.ndarray]] = None,
	gain: str = "linear",
	image_transform: Callable[[str], torch.Tensor] = transform_fn,
	cache_dir: Optional[str] = None,
	use_cache: bool = True,
	include_catalogue_type_feature: bool = False,
	model_variant: str = "NCF",
) -> Dict[str, float]:
	"""Ranking metrics with runtime image encoding path."""
	model.eval()

	normalized_variant = model_variant.strip().upper()
	if normalized_variant not in {"FM", "NN", "NCF"}:
		raise ValueError("Invalid model_variant for CF ranking metrics. Use one of: 'FM', 'NN', 'NCF'.")

	ndcg_values: List[float] = []
	all_pred_scores: List[float] = []
	all_gt_targets: List[float] = []

	with torch.no_grad():
		for catalogue in catalogues:
			if len(catalogue.neighbors) < 2:
				continue

			q_tab_raw = tabular_vec_from_img_path(
				catalogue.query,
				include_catalogue_type_feature=include_catalogue_type_feature,
			)
			if q_tab_raw is None:
				continue
			q_tab = normalize_tabular_if_needed(q_tab_raw, tabular_stats)
			q_img = load_img_tensor(catalogue.query, image_transform, cache_dir, use_cache)

			n_imgs: List[torch.Tensor] = []
			n_tabs: List[np.ndarray] = []
			valid_paths: List[str] = []

			for path in catalogue.neighbors:
				tab = tabular_vec_from_img_path(
					path,
					include_catalogue_type_feature=include_catalogue_type_feature,
				)
				if tab is None:
					continue
				n_imgs.append(load_img_tensor(path, image_transform, cache_dir, use_cache))
				n_tabs.append(normalize_tabular_if_needed(tab, tabular_stats))
				valid_paths.append(path)

			if len(valid_paths) < 2:
				continue

			num_neighbors = len(valid_paths)
			user_img_batch = torch.stack([q_img] * num_neighbors).to(device)
			item_img_batch = torch.stack(n_imgs).to(device)
			user_tab_batch = torch.tensor(np.repeat(q_tab[None, :], num_neighbors, axis=0), dtype=torch.float32, device=device)
			item_tab_batch = torch.tensor(np.stack(n_tabs), dtype=torch.float32, device=device)
			raw_scores = model((user_img_batch, item_img_batch, user_tab_batch, item_tab_batch)).detach().cpu().numpy()
			pred_indices = np.argsort(-raw_scores)

			gt_targets = np.arange(num_neighbors - 1, -1, -1, dtype=np.float64)
			if num_neighbors > 1:
				gt_targets = gt_targets / (num_neighbors - 1)
			d_min, d_max = raw_scores.min(), raw_scores.max()
			if d_max > d_min:
				pred_scores = (raw_scores - d_min) / (d_max - d_min)
			else:
				pred_scores = np.ones(num_neighbors, dtype=np.float64)
			all_pred_scores.extend(pred_scores.tolist())
			all_gt_targets.extend(gt_targets.tolist())

			gt_relevance = np.arange(num_neighbors, 0, -1, dtype=np.float64)
			if gain == "graded":
				pred_gain = 2.0 ** gt_relevance[pred_indices] - 1.0
				ideal_gain = 2.0 ** gt_relevance - 1.0
			else:
				pred_gain = gt_relevance[pred_indices]
				ideal_gain = gt_relevance
			discounts = np.log2(np.arange(2, num_neighbors + 2))
			dcg = float(np.sum(pred_gain / discounts))
			idcg = float(np.sum(ideal_gain / discounts))
			ndcg_values.append(dcg / idcg if idcg > 0 else 0.0)

	if not ndcg_values:
		return {"ndcg": 0.0, "rmse": 0.0}

	rmse = float(np.sqrt(np.mean((np.array(all_pred_scores) - np.array(all_gt_targets)) ** 2)))
	return {
		"ndcg": float(np.mean(ndcg_values)),
		"rmse": rmse,
	}
