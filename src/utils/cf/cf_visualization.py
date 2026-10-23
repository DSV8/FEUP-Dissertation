"""Visualize CF relevance scores as ranked catalogue neighbours."""

from __future__ import annotations
import os
from typing import Any, Callable, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from torchvision import transforms
from utils.image_embedding import (
    IMAGE_EMBEDDING_MODE_PRECOMPUTED,
	IMAGE_EMBEDDING_MODE_ENCODED_AT_RUNTIME,
	normalize_image_embedding_mode,
)
from utils.image_preprocessing import pad_img, transform_fn
from utils.catalogue_utils import (
    Catalogue,
    parse_filename,
)
from utils.mm.mm_data import normalize_tabular_if_needed, tabular_vec_from_img_path

def predict_catalogue_ranking_cf(
	model: nn.Module,
	catalogue: Catalogue,
	metadata: Dict[str, Any],
	device: str = "cpu",
	tabular_stats: Optional[Dict[str, np.ndarray]] = None,
	top_k: Optional[int] = None,
	model_variant: str = "NCF",
) -> List[Tuple[str, float]]:
	"""Return neighbour paths sorted by predicted score (descending)."""
	image_embedding_mode = normalize_image_embedding_mode(
		str(metadata.get("image_embedding_mode", IMAGE_EMBEDDING_MODE_PRECOMPUTED))
	)
	include_catalogue_type_feature = bool(metadata.get("include_catalogue_type_feature", False))
	if image_embedding_mode == IMAGE_EMBEDDING_MODE_ENCODED_AT_RUNTIME:
		return predict_catalogue_ranking_cf_encoded_at_runtime(
			model=model,
			catalogue=catalogue,
			device=device,
			tabular_stats=tabular_stats,
			top_k=top_k,
			model_variant=model_variant,
			image_transform=metadata.get("image_transform", transform_fn),
			cache_dir=metadata.get("cache_dir"),
			use_cache=bool(metadata.get("use_cache", True)),
			include_catalogue_type_feature=include_catalogue_type_feature,
		)

	image_embedding_map      = metadata["image_embedding_map"]
	user_image_embedding_map = metadata["user_image_embedding_map"]
	image_emb_dim = int(next(iter(image_embedding_map.values())).shape[-1]) if image_embedding_map else 384
	zero_emb = np.zeros(image_emb_dim, dtype=np.float32)

	neighbor_paths = catalogue.neighbors[:top_k] if top_k is not None else list(catalogue.neighbors)
	if not neighbor_paths:
		return []

	q_tab_raw = tabular_vec_from_img_path(
		catalogue.query,
		include_catalogue_type_feature=include_catalogue_type_feature,
	)
	if q_tab_raw is None:
		return []
	q_tab = normalize_tabular_if_needed(q_tab_raw, tabular_stats)
	q_img_np = user_image_embedding_map.get(catalogue.query, zero_emb)

	n_imgs, n_tabs, valid_paths = [], [], []
	for path in neighbor_paths:
		tab = tabular_vec_from_img_path(
			path,
			include_catalogue_type_feature=include_catalogue_type_feature,
		)
		if tab is None:
			continue
		n_imgs.append(image_embedding_map.get(path, zero_emb))
		n_tabs.append(normalize_tabular_if_needed(tab, tabular_stats))
		valid_paths.append(path)

	if not valid_paths:
		return []

	normalized_variant = model_variant.strip().upper()
	if normalized_variant not in {"FM", "NN", "NCF"}:
		raise ValueError("Invalid model_variant for CF ranking. Use one of: 'FM', 'NN', 'NCF'.")

	model.eval()
	with torch.no_grad():
		num_neighbors = len(valid_paths)
		user_img_batch = torch.tensor(np.repeat(q_img_np[None, :], num_neighbors, axis=0), dtype=torch.float32, device=device)
		item_img_batch = torch.tensor(np.stack(n_imgs), dtype=torch.float32, device=device)
		user_tab_batch = torch.tensor(np.repeat(q_tab[None, :], num_neighbors, axis=0), dtype=torch.float32, device=device)
		item_tab_batch = torch.tensor(np.stack(n_tabs), dtype=torch.float32, device=device)
		raw_scores = model((user_img_batch, item_img_batch, user_tab_batch, item_tab_batch)).detach().cpu().numpy()
		ranked = sorted(zip(valid_paths, raw_scores.tolist()), key=lambda x: x[1], reverse=True)
	return ranked

def predict_catalogue_ranking_cf_encoded_at_runtime(
	model: nn.Module,
	catalogue: Catalogue,
	device: str = "cpu",
	tabular_stats: Optional[Dict[str, np.ndarray]] = None,
	top_k: Optional[int] = None,
	model_variant: str = "NCF",
	image_transform: Callable[[str], torch.Tensor] = transform_fn,
	cache_dir: Optional[str] = None,
	use_cache: bool = True,
) -> List[Tuple[str, float]]:
	"""Return neighbour paths sorted by predicted score (descending) in runtime-encoding mode."""
	neighbor_paths = catalogue.neighbors[:top_k] if top_k is not None else list(catalogue.neighbors)
	if not neighbor_paths:
		return []

	q_tab_raw = tabular_vec_from_img_path(catalogue.query)
	if q_tab_raw is None:
		return []
	q_tab = normalize_tabular_if_needed(q_tab_raw, tabular_stats)
	q_img = load_img_tensor(catalogue.query, image_transform, cache_dir, use_cache)

	n_imgs: List[torch.Tensor] = []
	n_tabs: List[np.ndarray] = []
	valid_paths: List[str] = []
	for path in neighbor_paths:
		tab = tabular_vec_from_img_path(path)
		if tab is None:
			continue
		n_imgs.append(load_img_tensor(path, image_transform, cache_dir, use_cache))
		n_tabs.append(normalize_tabular_if_needed(tab, tabular_stats))
		valid_paths.append(path)

	if not valid_paths:
		return []

	normalized_variant = model_variant.strip().upper()
	if normalized_variant not in {"FM", "NN", "NCF"}:
		raise ValueError("Invalid model_variant for CF ranking. Use one of: 'FM', 'NN', 'NCF'.")

	model.eval()
	with torch.no_grad():
		num_neighbors = len(valid_paths)
		user_img_batch = torch.stack([q_img] * num_neighbors).to(device)
		item_img_batch = torch.stack(n_imgs).to(device)
		user_tab_batch = torch.tensor(np.repeat(q_tab[None, :], num_neighbors, axis=0), dtype=torch.float32, device=device)
		item_tab_batch = torch.tensor(np.stack(n_tabs), dtype=torch.float32, device=device)
		raw_scores = model((user_img_batch, item_img_batch, user_tab_batch, item_tab_batch)).detach().cpu().numpy()
		ranked = sorted(zip(valid_paths, raw_scores.tolist()), key=lambda x: x[1], reverse=True)
	return ranked

def visualize_predicted_ranking_cf(
	model: nn.Module,
	catalogue: Catalogue,
	metadata: Dict[str, Any],
	device: str = "cpu",
	tabular_stats: Optional[Dict[str, np.ndarray]] = None,
	top_k: Optional[int] = None,
	model_variant: str = "NCF",
) -> None:
	"""Visualize CF predicted ranking against the ground-truth neighbour order."""
	user_path = catalogue.query
	neighbor_paths = catalogue.neighbors
	if not neighbor_paths:
		raise ValueError("No neighbor images were provided for ranking visualization.")

	display_paths = neighbor_paths[:top_k] if top_k is not None else list(neighbor_paths)
	gt_rank_by_path = {path: idx + 1 for idx, path in enumerate(neighbor_paths)}

	ranked_neighbors = predict_catalogue_ranking_cf(
		model=model,
		catalogue=Catalogue(
			catalogue_type=catalogue.catalogue_type,
			catalogue_id=catalogue.catalogue_id,
			query=catalogue.query,
			neighbors=display_paths,
		),
		metadata=metadata,
		device=device,
		tabular_stats=tabular_stats,
		top_k=top_k,
		model_variant=model_variant,
	)
	if not ranked_neighbors:
		raise ValueError("No valid neighbours were found for ranking visualization.")

	num_neighbors = len(ranked_neighbors)
	neighbor_rows = 2
	neighbor_cols = max(1, (num_neighbors + 1) // neighbor_rows)

	fig = plt.figure(figsize=(2.5 + 2 * neighbor_cols, 4.5))
	gs_main = GridSpec(1, 2, figure=fig, width_ratios=[1.5, neighbor_cols], wspace=0.02)
	gs_left = GridSpecFromSubplotSpec(3, 1, subplot_spec=gs_main[0, 0], height_ratios=[0.1, 1, 0.15], hspace=0.02)

	title_ax = fig.add_subplot(gs_left[0, 0])
	title_ax.text(0.5, 0.5, f"Catalogue\n{catalogue.catalogue_type}_{catalogue.catalogue_id}",
		ha="center", va="center", fontsize=10, fontweight="bold")
	title_ax.axis("off")

	query_ax = fig.add_subplot(gs_left[1, 0])
	query_ax.imshow(transforms.Resize((224, 224))(pad_img(user_path)))
	query_ax.set_title("Query", fontsize=10, pad=3)
	query_ax.axis("off")
	user_meta = parse_filename(os.path.basename(user_path))
	if user_meta:
		query_ax.text(0.05, 0.95,
			"\n".join([str(user_meta[k]) for k in ("age", "height", "weight", "bra_cup", "bra_size")]),
			transform=query_ax.transAxes, fontsize=7, verticalalignment="top",
			bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

	subtitle_ax = fig.add_subplot(gs_left[2, 0])
	subtitle_ax.text(0.5, 0.5, "Legend:\nage / height / weight\nbra cup / bra size",
		ha="center", va="center", fontsize=8, style="italic")
	subtitle_ax.axis("off")

	gs_right = GridSpecFromSubplotSpec(neighbor_rows, neighbor_cols, subplot_spec=gs_main[0, 1], wspace=0.0, hspace=0.0)
	for idx, (path, _dist) in enumerate(ranked_neighbors):
		row = idx // neighbor_cols
		col = idx % neighbor_cols
		if row >= neighbor_rows:
			break
		pred_rank = idx + 1
		gt_rank   = gt_rank_by_path.get(path)
		color     = "green" if gt_rank == pred_rank else "red"
		ax = fig.add_subplot(gs_right[row, col])
		ax.imshow(transforms.Resize((224, 224))(pad_img(path)))
		ax.set_title(f"Pred {pred_rank} | GT {gt_rank}", fontsize=8, pad=3, color=color)
		ax.axis("off")
		meta = parse_filename(os.path.basename(path))
		if meta:
			ax.text(0.05, 0.95,
				"\n".join([str(meta[k]) for k in ("age", "height", "weight", "bra_cup", "bra_size")]),
				transform=ax.transAxes, fontsize=6, verticalalignment="top",
				bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

	plt.show()
