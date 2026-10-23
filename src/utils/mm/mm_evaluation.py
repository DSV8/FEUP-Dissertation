"""Triplet and catalogue ranking metrics for image-plus-tabular retrieval."""

from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
)
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import (
    DataLoader,
)
from utils.image_preprocessing import (
    transform_fn,
)
from utils.catalogue_utils import (
    Catalogue,
)
from models.multimodal import MultimodalEncoder

def evaluate_multimodal(
	model: MultimodalEncoder,
	dataloader: DataLoader,
	device: torch.device,
	distance_metric: str = "cosine",
) -> float:
	"""
	Evaluate multimodal model on a dataloader using triplet accuracy.

	Args:
		model: Multimodal encoder
		dataloader: DataLoader to evaluate on
		device: Device to evaluate on

	Returns:
		Accuracy as a float in [0, 1]
	"""
	model.eval()
	correct_predictions = 0
	total_triplets = 0

	with torch.no_grad():
		for batch in dataloader:
			q_img = batch["query_image"].to(device)
			pos_img = batch["pos_image"].to(device)
			neg_img = batch["neg_image"].to(device)

			q_tab = batch["query_tabular"].to(device)
			pos_tab = batch["pos_tabular"].to(device)
			neg_tab = batch["neg_tabular"].to(device)

			# Get fused embeddings
			query_emb = model(q_img, q_tab)
			pos_emb = model(pos_img, pos_tab)
			neg_emb = model(neg_img, neg_tab)

			# Compute pairwise distances
			pos_dist = comp_pairwise_distance(query_emb, pos_emb, distance_metric=distance_metric)
			neg_dist = comp_pairwise_distance(query_emb, neg_emb, distance_metric=distance_metric)

			# Count correct triplets (positive closer than negative)
			correct_predictions += (pos_dist < neg_dist).sum().item()
			total_triplets += q_img.size(0)

	accuracy = correct_predictions / total_triplets if total_triplets > 0 else 0.0
	return accuracy

def evaluate_catalogue_metrics_multimodal(
	catalogues: List[Catalogue],
	model: MultimodalEncoder,
	device: str,
	tabular_stats: Optional[Dict[str, np.ndarray]] = None,
	image_transform: Optional[Callable[[str], torch.Tensor]] = None,
	image_embedding_map: Optional[Dict[str, np.ndarray]] = None,
	include_catalogue_type_feature: bool = False,
	ks: Tuple[int, ...] = (1, 5, 10),
	relevant_top_n: int = 5,
	map_k: int = 10,
	distance_metric: str = "cosine",
	gain: str = "linear",
) -> Dict[str, float]:
	"""Evaluate multimodal ranking metrics over full catalogues."""
	if len(catalogues) == 0:
		raise ValueError("No catalogues provided for ranking metric evaluation.")

	_ = map_k  # Kept for compatibility with CBIR API.
	transform = image_transform if image_transform is not None else transform_fn
	image_emb_dim = int(next(iter(image_embedding_map.values())).shape[-1]) if image_embedding_map else 384
	zero_emb = np.zeros(image_emb_dim, dtype=np.float32)

	model.eval()

	k_list = sorted(set(int(k) for k in ks if int(k) > 0))
	if len(k_list) == 0:
		k_list = [10]

	sum_precision = {k: 0.0 for k in k_list}
	sum_recall = {k: 0.0 for k in k_list}
	sum_f1 = {k: 0.0 for k in k_list}
	sum_ap = {k: 0.0 for k in k_list}

	ndcg_values: List[float] = []

	with torch.no_grad():
		for catalogue in catalogues:
			neighbors = catalogue.neighbors

			q_tab = tabular_vec_from_img_path(
				catalogue.query,
				include_catalogue_type_feature=include_catalogue_type_feature,
			)
			if q_tab is None:
				continue
			q_tab = normalize_tabular_if_needed(q_tab, tabular_stats)

			n_tabs: List[np.ndarray] = []
			valid_neighbors: List[str] = []
			for neighbor_path in neighbors:
				tab = tabular_vec_from_img_path(
					neighbor_path,
					include_catalogue_type_feature=include_catalogue_type_feature,
				)
				if tab is None:
					continue
				n_tabs.append(normalize_tabular_if_needed(tab, tabular_stats))
				valid_neighbors.append(neighbor_path)

			num_neighbors = len(valid_neighbors)
			if num_neighbors < 2:
				continue

			if image_embedding_map is None:
				q_img_tensor = transform(catalogue.query).unsqueeze(0).to(device)
			else:
				q_img_tensor = torch.tensor(
					image_embedding_map.get(catalogue.query, zero_emb),
					dtype=torch.float32,
				).unsqueeze(0).to(device)
			q_tab_tensor = torch.tensor(q_tab, dtype=torch.float32).unsqueeze(0).to(device)
			q_emb = model(q_img_tensor, q_tab_tensor)

			if image_embedding_map is None:
				n_img_tensors = torch.stack([transform(path) for path in valid_neighbors]).to(device)
			else:
				n_img_tensors = torch.tensor(
					np.stack([image_embedding_map.get(path, zero_emb) for path in valid_neighbors], axis=0),
					dtype=torch.float32,
				).to(device)
			n_tab_tensors = torch.tensor(np.stack(n_tabs, axis=0), dtype=torch.float32).to(device)
			n_embs = model(n_img_tensors, n_tab_tensors)
			q_emb_expanded = q_emb.expand_as(n_embs)
			distances = comp_pairwise_distance(q_emb_expanded, n_embs, distance_metric=distance_metric).detach().cpu().numpy()

			pred_indices = np.argsort(distances)

			gt_relevance = np.arange(num_neighbors, 0, -1, dtype=np.float64)

			if gain == "graded":
				pred_gain = (2.0 ** gt_relevance[pred_indices] - 1.0)
				ideal_gain = (2.0 ** gt_relevance - 1.0)
			elif gain == "linear":
				pred_gain = gt_relevance[pred_indices]
				ideal_gain = gt_relevance
			else:
				raise ValueError("Unsupported gain: choose 'graded' or 'linear'.")

			dcg = float(np.sum(pred_gain / np.log2(np.arange(2, num_neighbors + 2))))
			idcg = float(np.sum(ideal_gain / np.log2(np.arange(2, num_neighbors + 2))))
			ndcg = dcg / idcg if idcg > 0 else 0.0

			ndcg_values.append(ndcg)

			relevant_count = max(1, min(relevant_top_n, num_neighbors))
			relevant_set = set(range(relevant_count))

			for k in k_list:
				k_eff = min(k, num_neighbors)
				topk_pred = pred_indices[:k_eff]
				hits = sum(1 for idx in topk_pred if idx in relevant_set)

				precision_k = hits / k_eff if k_eff > 0 else 0.0
				recall_k = hits / relevant_count if relevant_count > 0 else 0.0
				f1_k = (2 * precision_k * recall_k / (precision_k + recall_k)) if (precision_k + recall_k) > 0 else 0.0

				sum_precision[k] += precision_k
				sum_recall[k] += recall_k
				sum_f1[k] += f1_k

				hit_count = 0
				ap_accum = 0.0
				for rank_idx, item_idx in enumerate(topk_pred, start=1):
					if item_idx in relevant_set:
						hit_count += 1
						ap_accum += hit_count / rank_idx

				ap_denom = min(relevant_count, k_eff)
				ap_k = (ap_accum / ap_denom) if ap_denom > 0 else 0.0
				sum_ap[k] += ap_k

	n_eval = len(ndcg_values)
	if n_eval == 0:
		raise ValueError("No valid catalogues with at least two neighbors were found for metric evaluation.")

	metrics: Dict[str, float] = {
		"ndcg": float(np.mean(ndcg_values)),
	}

	for k in k_list:
		metrics[f"precision_at_{k}"] = sum_precision[k] / n_eval
		metrics[f"recall_at_{k}"] = sum_recall[k] / n_eval
		metrics[f"f1_at_{k}"] = sum_f1[k] / n_eval
		metrics[f"map_at_{k}"] = sum_ap[k] / n_eval

	return metrics

def comp_pairwise_distance(a: torch.Tensor, b: torch.Tensor, distance_metric: str = "cosine") -> torch.Tensor:
	"""Compute pairwise distance (lower => more similar)."""
	metric = distance_metric.lower()
	if metric == "euclidean":
		return torch.norm(a - b, p=2, dim=1)
	if metric == "cosine":
		return 1.0 - F.cosine_similarity(a, b, dim=1)
	raise ValueError(f"Unsupported distance metric: {distance_metric}. Use 'euclidean' or 'cosine'.")
