"""Evaluation helpers for the CF and MM heads of the joint model."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.joint import JointCFMMModel
from utils.catalogue_utils import Catalogue
from utils.cf.cf_data import load_img_tensor as _load_image_tensor
from utils.joint.joint_data import resolve_image_embedding_mode_plan
from utils.mm.mm_data import normalize_tabular_if_needed, tabular_vec_from_img_path
from utils.mm.mm_evaluation import comp_pairwise_distance

def evaluate_mm_triplet_accuracy(
    model: JointCFMMModel,
    dataloader: DataLoader,
    device: str,
    distance_metric: str = "cosine",
) -> float:
    model.eval()
    total = 0
    correct = 0

    with torch.no_grad():
        for batch in dataloader:
            q = model.encode(batch["query_img"].to(device), batch["query_tab"].to(device))
            p = model.encode(batch["pos_img"].to(device), batch["pos_tab"].to(device))
            n = model.encode(batch["neg_img"].to(device), batch["neg_tab"].to(device))

            d_pos = comp_pairwise_distance(q, p, distance_metric=distance_metric)
            d_neg = comp_pairwise_distance(q, n, distance_metric=distance_metric)
            correct += int((d_pos < d_neg).sum().item())
            total += int(q.size(0))

    return (correct / total) if total > 0 else 0.0

def evaluate_mm_catalogue_accuracy(
    catalogues: List[Catalogue],
    model: JointCFMMModel,
    metadata: Dict[str, Any],
    device: str,
    tabular_stats: Optional[Dict[str, np.ndarray]] = None,
    distance_metric: str = "cosine",
) -> float:
    """Evaluate MM triplet accuracy over a selected catalogue family."""
    mode_plan = resolve_image_embedding_mode_plan(metadata)
    include_catalogue_type_feature = bool(metadata.get("include_catalogue_type_feature", False))
    image_embedding_map = mode_plan["image_embedding_map"]
    image_transform = metadata.get("image_transform", transform_fn)
    cache_dir = metadata.get("cache_dir")
    use_cache = bool(metadata.get("use_cache", True))

    image_emb_dim = int(next(iter(image_embedding_map.values())).shape[-1]) if image_embedding_map else 384
    zero_emb = np.zeros(image_emb_dim, dtype=np.float32)
    correct = 0
    total = 0
    model.eval()

    with torch.no_grad():
        for catalogue in catalogues:
            q_tab_raw = tabular_vec_from_img_path(
                catalogue.query,
                include_catalogue_type_feature=include_catalogue_type_feature,
            )
            if q_tab_raw is None:
                continue
            q_tab = normalize_tabular_if_needed(q_tab_raw, tabular_stats)

            valid_neighbors: List[Tuple[str, np.ndarray]] = []
            for path in catalogue.neighbors:
                tab_raw = tabular_vec_from_img_path(
                    path,
                    include_catalogue_type_feature=include_catalogue_type_feature,
                )
                if tab_raw is not None:
                    valid_neighbors.append((path, normalize_tabular_if_needed(tab_raw, tabular_stats)))
            if len(valid_neighbors) < 2:
                continue

            if mode_plan["use_precomputed"] and image_embedding_map is not None:
                q_img = torch.tensor(image_embedding_map.get(catalogue.query, zero_emb), dtype=torch.float32).unsqueeze(0).to(device)
                neighbor_imgs = torch.tensor(
                    np.stack([image_embedding_map.get(path, zero_emb) for path, _ in valid_neighbors]),
                    dtype=torch.float32,
                    device=device,
                )
            else:
                q_img = _load_image_tensor(catalogue.query, image_transform, cache_dir, use_cache).unsqueeze(0).to(device)
                neighbor_imgs = torch.stack([
                    _load_image_tensor(path, image_transform, cache_dir, use_cache)
                    for path, _ in valid_neighbors
                ]).to(device)

            q_tab_tensor = torch.tensor(q_tab, dtype=torch.float32, device=device).unsqueeze(0)
            neighbor_tabs = torch.tensor(
                np.stack([tab for _, tab in valid_neighbors]),
                dtype=torch.float32,
                device=device,
            )
            query_embedding = model.encode(q_img, q_tab_tensor)
            neighbor_embeddings = model.encode(neighbor_imgs, neighbor_tabs)

            query_batch: List[torch.Tensor] = []
            positive_batch: List[torch.Tensor] = []
            negative_batch: List[torch.Tensor] = []
            for positive_idx, negative_idx in combinations(range(len(valid_neighbors)), 2):
                query_batch.append(query_embedding.squeeze(0))
                positive_batch.append(neighbor_embeddings[positive_idx])
                negative_batch.append(neighbor_embeddings[negative_idx])

            if query_batch:
                query_tensor = torch.stack(query_batch)
                positive_tensor = torch.stack(positive_batch)
                negative_tensor = torch.stack(negative_batch)
                positive_dist = comp_pairwise_distance(query_tensor, positive_tensor, distance_metric=distance_metric)
                negative_dist = comp_pairwise_distance(query_tensor, negative_tensor, distance_metric=distance_metric)
                correct += int((positive_dist < negative_dist).sum().item())
                total += len(query_batch)

    return correct / total if total > 0 else 0.0

def evaluate_catalogue_metrics_mm(
    catalogues: List[Catalogue],
    model: JointCFMMModel,
    metadata: Dict[str, Any],
    device: str,
    tabular_stats: Optional[Dict[str, np.ndarray]] = None,
    distance_metric: str = "cosine",
    gain: str = "linear",
) -> Dict[str, float]:
    """Evaluate MM ranking nDCG over full catalogues for the joint model."""
    if len(catalogues) == 0:
        raise ValueError("No catalogues provided for ranking metric evaluation.")

    mode_plan = resolve_image_embedding_mode_plan(metadata)
    image_embedding_mode = mode_plan["mode"]
    include_catalogue_type_feature = bool(metadata.get("include_catalogue_type_feature", False))
    image_transform = metadata.get("image_transform", transform_fn)
    cache_dir = metadata.get("cache_dir")
    use_cache = bool(metadata.get("use_cache", True))
    image_embedding_map = mode_plan["image_embedding_map"]

    image_emb_dim = int(next(iter(image_embedding_map.values())).shape[-1]) if image_embedding_map else 384
    zero_emb = np.zeros(image_emb_dim, dtype=np.float32)

    model.eval()
    ndcg_values: List[float] = []

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

            n_tabs: List[np.ndarray] = []
            valid_paths: List[str] = []
            for path in catalogue.neighbors:
                tab = tabular_vec_from_img_path(path, include_catalogue_type_feature=include_catalogue_type_feature)
                if tab is None:
                    continue
                n_tabs.append(normalize_tabular_if_needed(tab, tabular_stats))
                valid_paths.append(path)

            num_neighbors = len(valid_paths)
            if num_neighbors < 2:
                continue

            if mode_plan["use_precomputed"] and image_embedding_map is not None:
                q_img = torch.tensor(image_embedding_map.get(catalogue.query, zero_emb), dtype=torch.float32).unsqueeze(0).to(device)
                n_imgs = torch.tensor(
                    np.stack([image_embedding_map.get(path, zero_emb) for path in valid_paths], axis=0),
                    dtype=torch.float32,
                ).to(device)
            else:
                q_img = _load_image_tensor(catalogue.query, image_transform, cache_dir, use_cache).unsqueeze(0).to(device)
                n_imgs = torch.stack([
                    _load_image_tensor(path, image_transform, cache_dir, use_cache)
                    for path in valid_paths
                ]).to(device)

            q_tab_tensor = torch.tensor(q_tab, dtype=torch.float32).unsqueeze(0).to(device)
            n_tab_tensor = torch.tensor(np.stack(n_tabs, axis=0), dtype=torch.float32).to(device)

            q_emb = model.encode(q_img, q_tab_tensor)
            n_embs = model.encode(n_imgs, n_tab_tensor)
            q_emb_expanded = q_emb.expand_as(n_embs)
            distances = comp_pairwise_distance(
                q_emb_expanded,
                n_embs,
                distance_metric=distance_metric,
            ).detach().cpu().numpy()
            pred_indices = np.argsort(distances)

            gt_relevance = np.arange(num_neighbors, 0, -1, dtype=np.float64)
            if gain == "graded":
                pred_gain = (2.0 ** gt_relevance[pred_indices]) - 1.0
                ideal_gain = (2.0 ** gt_relevance) - 1.0
            elif gain == "linear":
                pred_gain = gt_relevance[pred_indices]
                ideal_gain = gt_relevance
            else:
                raise ValueError("Unsupported gain: choose 'graded' or 'linear'.")

            discounts = np.log2(np.arange(2, num_neighbors + 2))
            dcg = float(np.sum(pred_gain / discounts))
            idcg = float(np.sum(ideal_gain / discounts))
            ndcg_values.append(dcg / idcg if idcg > 0 else 0.0)

    if not ndcg_values:
        raise ValueError("No valid catalogues with at least two neighbors were found for metric evaluation.")
    return {"ndcg": float(np.mean(ndcg_values))}

def evaluate_cf_accuracy_precomputed(
    catalogues: List[Catalogue],
    model: nn.Module,
    metadata: Dict[str, Any],
    device: str,
    tabular_stats: Optional[Dict[str, np.ndarray]] = None,
    include_catalogue_type_feature: bool = False,
) -> float:
    """Pairwise ordering accuracy for precomputed-image mode."""
    model.eval()
    image_embedding_map = metadata["image_embedding_map"]
    user_image_embedding_map = metadata["user_image_embedding_map"]

    image_emb_dim = int(next(iter(image_embedding_map.values())).shape[-1]) if image_embedding_map else 384
    zero_emb = np.zeros(image_emb_dim, dtype=np.float32)

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

            for idx, path in enumerate(catalogue.neighbors):
                meta = parse_filename(os.path.basename(path))
                tab = tabular_vec_from_img_path(path, include_catalogue_type_feature=include_catalogue_type_feature)
                if meta is None or tab is None:
                    continue

                rank = int(meta["rank"]) if meta.get("rank") is not None else (idx + 1)
                n_imgs.append(image_embedding_map.get(path, zero_emb))
                n_tabs.append(normalize_tabular_if_needed(tab, tabular_stats))
                ranks.append(rank)

            valid_count = len(n_imgs)
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

def evaluate_catalogue_metrics_cf_precomputed(
    catalogues: List[Catalogue],
    model: nn.Module,
    metadata: Dict[str, Any],
    device: str,
    tabular_stats: Optional[Dict[str, np.ndarray]] = None,
    include_catalogue_type_feature: bool = False,
    gain: str = "linear",
) -> Dict[str, float]:
    """CF ranking metrics for precomputed-image mode."""
    model.eval()
    image_embedding_map = metadata["image_embedding_map"]
    user_image_embedding_map = metadata["user_image_embedding_map"]

    image_emb_dim = int(next(iter(image_embedding_map.values())).shape[-1]) if image_embedding_map else 384
    zero_emb = np.zeros(image_emb_dim, dtype=np.float32)

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
            q_img_np = user_image_embedding_map.get(catalogue.query, zero_emb)

            n_imgs: List[np.ndarray] = []
            n_tabs: List[np.ndarray] = []
            valid_paths: List[str] = []
            for path in catalogue.neighbors:
                meta = parse_filename(os.path.basename(path))
                tab = tabular_vec_from_img_path(path, include_catalogue_type_feature=include_catalogue_type_feature)
                if meta is None or tab is None:
                    continue
                n_imgs.append(image_embedding_map.get(path, zero_emb))
                n_tabs.append(normalize_tabular_if_needed(tab, tabular_stats))
                valid_paths.append(path)

            num_neighbors = len(valid_paths)
            if num_neighbors < 2:
                continue

            user_img_batch = torch.tensor(np.repeat(q_img_np[None, :], num_neighbors, axis=0), dtype=torch.float32, device=device)
            item_img_batch = torch.tensor(np.stack(n_imgs), dtype=torch.float32, device=device)
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

def evaluate_cf_accuracy_encoded_at_runtime(
    catalogues: List[Catalogue],
    model: nn.Module,
    device: str,
    tabular_stats: Optional[Dict[str, np.ndarray]],
    image_transform: Callable[[str], torch.Tensor],
    cache_dir: Optional[str],
    use_cache: bool,
    include_catalogue_type_feature: bool,
) -> float:
    model.eval()
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

            q_img_tensor = _load_image_tensor(catalogue.query, image_transform, cache_dir, use_cache)

            n_imgs: List[torch.Tensor] = []
            n_tabs: List[np.ndarray] = []
            ranks: List[int] = []

            for idx, path in enumerate(catalogue.neighbors):
                meta = parse_filename(os.path.basename(path))
                tab = tabular_vec_from_img_path(path, include_catalogue_type_feature=include_catalogue_type_feature)
                if meta is None or tab is None:
                    continue
                rank = int(meta["rank"]) if meta.get("rank") is not None else (idx + 1)
                n_imgs.append(_load_image_tensor(path, image_transform, cache_dir, use_cache))
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
                score_i, score_j = raw_scores[i], raw_scores[j]
                if rank_i < rank_j:
                    correct_pairs += int(score_i > score_j)
                else:
                    correct_pairs += int(score_j > score_i)
                total_pairs += 1

    return (correct_pairs / total_pairs) if total_pairs > 0 else 0.0

def evaluate_catalogue_metrics_cf_encoded_at_runtime(
    catalogues: List[Catalogue],
    model: nn.Module,
    device: str,
    tabular_stats: Optional[Dict[str, np.ndarray]],
    image_transform: Callable[[str], torch.Tensor],
    cache_dir: Optional[str],
    use_cache: bool,
    include_catalogue_type_feature: bool,
    gain: str = "linear",
) -> Dict[str, float]:
    model.eval()
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
            q_img = _load_image_tensor(catalogue.query, image_transform, cache_dir, use_cache)

            n_imgs: List[torch.Tensor] = []
            n_tabs: List[np.ndarray] = []
            valid_paths: List[str] = []

            for path in catalogue.neighbors:
                tab = tabular_vec_from_img_path(path, include_catalogue_type_feature=include_catalogue_type_feature)
                if tab is None:
                    continue
                n_imgs.append(_load_image_tensor(path, image_transform, cache_dir, use_cache))
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

            d_min, d_max = float(np.min(raw_scores)), float(np.max(raw_scores))
            if d_max > d_min:
                pred_scores = (raw_scores - d_min) / (d_max - d_min)
            else:
                pred_scores = np.ones(num_neighbors, dtype=np.float64)

            all_pred_scores.extend(pred_scores.tolist())
            all_gt_targets.extend(gt_targets.tolist())

            gt_relevance = np.arange(num_neighbors, 0, -1, dtype=np.float64)
            if gain == "graded":
                pred_gain = (2.0 ** gt_relevance[pred_indices]) - 1.0
                ideal_gain = (2.0 ** gt_relevance) - 1.0
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
