"""Triplet accuracy and catalogue-level retrieval metrics for CBIR."""

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from utils.catalogue_utils import Catalogue, catalogue_type_to_id
from utils.image_embedding import (
    IMAGE_EMBEDDING_MODE_ENCODED_AT_RUNTIME,
    IMAGE_EMBEDDING_MODE_PRECOMPUTED,
    normalize_image_embedding_mode,
)


def comp_pairwise_distance(a: torch.Tensor, b: torch.Tensor, distance_metric: str = "euclidean") -> torch.Tensor:
    """Compute pairwise distance (lower => more similar)."""
    metric = distance_metric.lower()
    if metric == "euclidean":
        return torch.norm(a - b, p=2, dim=1)
    if metric == "cosine":
        return 1.0 - F.cosine_similarity(a, b, dim=1)
    raise ValueError(f"Unsupported distance metric: {distance_metric}. Use 'euclidean' or 'cosine'.")


def embed_triplet_batch(
    model: torch.nn.Module,
    q: torch.Tensor,
    pos: torch.Tensor,
    neg: torch.Tensor,
    catalogue_type_id: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    merged = torch.cat([q, pos, neg], dim=0)

    if catalogue_type_id is not None:
        merged_type_id = torch.cat([catalogue_type_id, catalogue_type_id, catalogue_type_id], dim=0)
        try:
            embeddings = F.normalize(model(merged, merged_type_id), p=2, dim=1)
        except TypeError:
            embeddings = F.normalize(model(merged), p=2, dim=1)
    else:
        embeddings = F.normalize(model(merged), p=2, dim=1)

    batch_size = q.size(0)
    return embeddings[:batch_size], embeddings[batch_size: 2 * batch_size], embeddings[2 * batch_size:]


def compute_catalogue_distances(
    catalogue: Catalogue,
    model: torch.nn.Module,
    transform: Callable[[str], torch.Tensor],
    device: str,
    distance_metric: str = "euclidean",
    metadata: Optional[Dict[str, object]] = None,
) -> np.ndarray:
    mode = IMAGE_EMBEDDING_MODE_ENCODED_AT_RUNTIME
    if metadata is not None:
        mode = normalize_image_embedding_mode(
            str(metadata.get("image_embedding_mode", IMAGE_EMBEDDING_MODE_ENCODED_AT_RUNTIME))
        )

    if mode == IMAGE_EMBEDDING_MODE_PRECOMPUTED:
        image_embedding_map = metadata.get("image_embedding_map", {}) if metadata is not None else {}
        image_embedding_dim = int(metadata.get("image_embedding_dim", 384)) if metadata is not None else 384
        zero_emb = np.zeros(image_embedding_dim, dtype=np.float32)
        q_tensor = torch.tensor(image_embedding_map.get(catalogue.query, zero_emb), dtype=torch.float32).unsqueeze(0)
        n_tensors = torch.tensor(
            np.stack([image_embedding_map.get(path, zero_emb) for path in catalogue.neighbors], axis=0),
            dtype=torch.float32,
        )
        all_tensors = torch.cat([q_tensor, n_tensors], dim=0).to(device)
    else:
        q_tensor = transform(catalogue.query).unsqueeze(0)
        n_tensors = torch.stack([transform(path) for path in catalogue.neighbors])
        all_tensors = torch.cat([q_tensor, n_tensors], dim=0).to(device)

    catalogue_type_id = torch.full(
        (all_tensors.size(0),),
        fill_value=catalogue_type_to_id(catalogue.catalogue_type),
        dtype=torch.long,
        device=all_tensors.device,
    )

    try:
        all_embs = F.normalize(model(all_tensors, catalogue_type_id), p=2, dim=1)
    except TypeError:
        all_embs = F.normalize(model(all_tensors), p=2, dim=1)

    q_emb = all_embs[:1]
    n_embs = all_embs[1:]
    q_emb_expanded = q_emb.expand_as(n_embs)
    distances = comp_pairwise_distance(q_emb_expanded, n_embs, distance_metric=distance_metric)
    return distances.detach().cpu().numpy()


def evaluate_triplets(model, dataloader, device, distance_metric: str = "euclidean") -> float:
    model.eval()
    correct_predictions = 0
    total_triplets = 0

    with torch.no_grad():
        for batch in dataloader:
            q = batch['query'].to(device)
            pos = batch['pos'].to(device)
            neg = batch['neg'].to(device)
            catalogue_type_id = batch.get('catalogue_type_id')
            if catalogue_type_id is not None:
                catalogue_type_id = catalogue_type_id.to(device)

            q_emb, pos_emb, neg_emb = embed_triplet_batch(model, q, pos, neg, catalogue_type_id=catalogue_type_id)
            pos_dist = comp_pairwise_distance(q_emb, pos_emb, distance_metric=distance_metric)
            neg_dist = comp_pairwise_distance(q_emb, neg_emb, distance_metric=distance_metric)

            correct_predictions += (pos_dist < neg_dist).sum().item()
            total_triplets += q.size(0)

    return correct_predictions / total_triplets


def evaluate_catalogue_metrics(
    catalogues: List[Catalogue],
    model: torch.nn.Module,
    transform: Callable[[str], torch.Tensor],
    device: str,
    ks: Tuple[int, ...] = (1, 5, 10),
    relevant_top_n: int = 5,
    map_k: int = 10,
    distance_metric: str = "euclidean",
    gain: str = "graded",
    metadata: Optional[Dict[str, object]] = None,
) -> Dict[str, float]:
    """Evaluate nDCG and cutoff ranking metrics over full catalogues."""
    if len(catalogues) == 0:
        raise ValueError("No catalogues provided for ranking metric evaluation.")

    model.eval()
    k_list = sorted(set(int(k) for k in ks if int(k) > 0)) or [10]
    sum_precision = {k: 0.0 for k in k_list}
    sum_recall = {k: 0.0 for k in k_list}
    sum_f1 = {k: 0.0 for k in k_list}
    sum_ap = {k: 0.0 for k in k_list}
    ndcg_values: List[float] = []

    with torch.no_grad():
        for catalogue in catalogues:
            neighbors = catalogue.neighbors
            num_neighbors = len(neighbors)
            if num_neighbors < 2:
                continue

            distances = compute_catalogue_distances(
                catalogue,
                model,
                transform,
                device,
                distance_metric=distance_metric,
                metadata=metadata,
            )
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

            discount = np.log2(np.arange(2, num_neighbors + 2))
            dcg = float(np.sum(pred_gain / discount))
            idcg = float(np.sum(ideal_gain / discount))
            ndcg_values.append(dcg / idcg if idcg > 0 else 0.0)

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
                sum_ap[k] += ap_accum / ap_denom if ap_denom > 0 else 0.0

    n_eval = len(ndcg_values)
    if n_eval == 0:
        raise ValueError("No valid catalogues with at least two neighbors were found for metric evaluation.")

    metrics: Dict[str, float] = {"ndcg": float(np.mean(ndcg_values))}
    for k in k_list:
        metrics[f"precision_at_{k}"] = sum_precision[k] / n_eval
        metrics[f"recall_at_{k}"] = sum_recall[k] / n_eval
        metrics[f"f1_at_{k}"] = sum_f1[k] / n_eval
        metrics[f"map_at_{k}"] = sum_ap[k] / n_eval

    return metrics