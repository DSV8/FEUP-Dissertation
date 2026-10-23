"""Joint CF/MM training loop for a shared image-tabular representation."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.image_embedding import (
    IMAGE_EMBEDDING_MODE_ENCODED_AT_RUNTIME,
    IMAGE_EMBEDDING_MODE_PRECOMPUTED,
    normalize_image_embedding_mode,
)
from cf import save_model
from utils.cf.cf_data import InteractionCFDataset, compute_img_embeddings
from utils.catalogue_utils import Catalogue
from models.joint import JointCFMMModel, UncertaintyWeightingLoss

from utils.joint.joint_data import (
    resolve_image_embedding_mode_plan,
    MMTripletInteraction,
    JointCFMMBundle,
    EmbeddingTripletDataset,
    EncodedAtRuntimeMMAdapterDataset,
    create_mm_triplet_interactions,
    prepare_joint_cf_mm_datasets,
    get_mm_dataloaders,
)

from utils.joint.joint_evaluation import (
    evaluate_mm_triplet_accuracy,
    evaluate_mm_catalogue_accuracy,
    evaluate_catalogue_metrics_mm,
    evaluate_cf_accuracy_precomputed,
    evaluate_catalogue_metrics_cf_precomputed,
    evaluate_cf_accuracy_encoded_at_runtime,
    evaluate_catalogue_metrics_cf_encoded_at_runtime,
)

from utils.joint.joint_data import (
    CFInteraction,
    CFEncodedAtRuntimeInteraction,
    EncodedAtRuntimeInteractionCFDataset,
    create_cf_precomputed_interactions,
    create_cf_encoded_at_runtime_interactions,
    infer_image_embedding_dim,
)

def save_joint_checkpoint(
    model: JointCFMMModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_score: float,
    checkpoint_path: str,
    uncertainty_module: Optional[UncertaintyWeightingLoss] = None,
) -> None:
    save_dir = os.path.dirname(checkpoint_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    payload: Dict[str, Any] = {
        "epoch": epoch,
        "best_score": best_score,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if uncertainty_module is not None:
        payload["uncertainty_state_dict"] = uncertainty_module.state_dict()

    torch.save(payload, checkpoint_path)

def load_joint_checkpoint(
    model: JointCFMMModel,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: str,
    device: str = "cpu",
    uncertainty_module: Optional[UncertaintyWeightingLoss] = None,
) -> Tuple[int, float]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if uncertainty_module is not None and "uncertainty_state_dict" in ckpt:
        uncertainty_module.load_state_dict(ckpt["uncertainty_state_dict"])
    model.to(device)
    start_epoch = ckpt.get("epoch", -1) + 1
    best_score = ckpt.get("best_score", float("-inf"))
    return start_epoch, best_score

def next_batch(loader: DataLoader, iterator: Any) -> Tuple[Dict[str, torch.Tensor], Any]:
    try:
        batch = next(iterator)
    except StopIteration:
        iterator = iter(loader)
        batch = next(iterator)
    return batch, iterator

def train_joint_cf_mm_model(
    model: JointCFMMModel,
    cf_train_loader: DataLoader,
    cf_test_loader: DataLoader,
    mm_train_loader: DataLoader,
    mm_test_loader: DataLoader,
    train_catalogues: List[Catalogue],
    test_catalogues: List[Catalogue],
    metadata: Dict[str, Any],
    device: str,
    num_epochs: int = 100,
    lr: float = 2e-4,
    weight_decay: float = 1e-4,
    cf_loss_weight: float = 1.0,
    mm_loss_weight: float = 1.0,
    use_uncertainty_weighting: bool = True,
    triplet_margin: float = 0.1,
    distance_metric: str = "cosine",
    loss_metric: str = "triplet_margin",
    use_rank_weighted_triplet: bool = False,
    rank_weight_alpha: float = 0.25,
    gain: str = "linear",
    use_wandb: bool = False,
    checkpoint_dir: Optional[str] = None,
    resume_checkpoint_path: Optional[str] = None,
    tabular_stats: Optional[Dict[str, np.ndarray]] = None,
) -> JointCFMMModel:
    uncertainty_module: Optional[UncertaintyWeightingLoss] = UncertaintyWeightingLoss().to(device) if use_uncertainty_weighting else None

    trainable_model_params = [p for p in model.parameters() if p.requires_grad]
    optimizer_params: List[torch.nn.Parameter] = list(trainable_model_params)
    if uncertainty_module is not None:
        optimizer_params.extend(list(uncertainty_module.parameters()))
    if len(optimizer_params) == 0:
        raise ValueError("No trainable parameters found for optimizer construction.")
    optimizer = torch.optim.Adam(optimizer_params, lr=lr, weight_decay=weight_decay)

    mse_loss = nn.MSELoss()
    normalized_distance_metric = distance_metric.strip().lower()
    if normalized_distance_metric == "euclidean":
        def embedding_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return torch.norm(a - b, p=2, dim=1)
    elif normalized_distance_metric == "cosine":
        def embedding_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return 1.0 - F.cosine_similarity(a, b, dim=1)
    else:
        raise ValueError("Unsupported distance metric: choose 'euclidean' or 'cosine'.")

    normalized_loss_metric = loss_metric.strip().lower() if isinstance(loss_metric, str) else str(loss_metric)
    if normalized_loss_metric in {"triplet", "triplet_margin"}:
        mm_triplet_criterion = nn.TripletMarginWithDistanceLoss(
            distance_function=embedding_distance,
            margin=triplet_margin,
            reduction="none",
        )
    elif normalized_loss_metric == "quadratic":
        mm_triplet_criterion = None
    else:
        raise ValueError(f"Unsupported loss_metric '{loss_metric}'. Supported: 'triplet_margin', 'quadratic'.")

    model.to(device)
    start_epoch = 0
    best_score = float("-inf")
    if resume_checkpoint_path is not None and os.path.exists(resume_checkpoint_path):
        start_epoch, best_score = load_joint_checkpoint(model, optimizer, resume_checkpoint_path, device=device, uncertainty_module=uncertainty_module)
        print(f"Resumed training from epoch {start_epoch}")

    for epoch in range(start_epoch, num_epochs):
        model.train()
        model.enforce_frozen_backbone_eval_mode()
        cf_iter = iter(cf_train_loader)
        mm_iter = iter(mm_train_loader)
        # Keep both objectives active each step; the shorter loader is cycled
        # so neither task is silently under-sampled within an epoch.
        steps = max(len(cf_train_loader), len(mm_train_loader))

        running_total_loss = 0.0
        running_cf_loss = 0.0
        running_mm_loss = 0.0

        for _ in range(steps):
            cf_batch, cf_iter = next_batch(cf_train_loader, cf_iter)
            mm_batch, mm_iter = next_batch(mm_train_loader, mm_iter)

            q_img = cf_batch["query_img"].to(device)
            i_img = cf_batch["item_img"].to(device)
            q_tab = cf_batch["query_tab"].to(device)
            i_tab = cf_batch["item_tab"].to(device)
            target = cf_batch["target"].to(device)

            cf_pred = model((q_img, i_img, q_tab, i_tab))
            cf_loss = mse_loss(cf_pred, target)

            tq_img = mm_batch["query_img"].to(device)
            tp_img = mm_batch["pos_img"].to(device)
            tn_img = mm_batch["neg_img"].to(device)
            tq_tab = mm_batch["query_tab"].to(device)
            tp_tab = mm_batch["pos_tab"].to(device)
            tn_tab = mm_batch["neg_tab"].to(device)
            rank_gap = mm_batch["rank_gap"].to(device)

            tq_emb = model.encode(tq_img, tq_tab)
            tp_emb = model.encode(tp_img, tp_tab)
            tn_emb = model.encode(tn_img, tn_tab)

            if normalized_loss_metric in {"triplet", "triplet_margin"}:
                per_sample_triplet = mm_triplet_criterion(tq_emb, tp_emb, tn_emb)
            else:
                d_pos = embedding_distance(tq_emb, tp_emb)
                d_neg = embedding_distance(tq_emb, tn_emb)
                relu = F.relu(d_pos - d_neg + triplet_margin)
                per_sample_triplet = relu * relu

            if use_rank_weighted_triplet:
                rank_gap = mm_batch["rank_gap"].to(device)
                weights = 1.0 + rank_weight_alpha * torch.clamp(rank_gap - 1.0, min=0.0)
                mm_loss = torch.mean(per_sample_triplet * weights)
            else:
                mm_loss = torch.mean(per_sample_triplet)

            if uncertainty_module is not None:
                total_loss = uncertainty_module(cf_loss, mm_loss)
            else:
                total_loss = (cf_loss_weight * cf_loss) + (mm_loss_weight * mm_loss)

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()

            running_total_loss += float(total_loss.item())
            running_cf_loss += float(cf_loss.item())
            running_mm_loss += float(mm_loss.item())

        mode_plan = resolve_image_embedding_mode_plan(metadata)
        image_embedding_mode = mode_plan["mode"]
        include_catalogue_type_feature = bool(metadata.get("include_catalogue_type_feature", False))
        if mode_plan["use_precomputed"]:
            train_cf_acc = evaluate_cf_accuracy_precomputed(
                train_catalogues,
                model,
                metadata,
                device,
                tabular_stats=tabular_stats,
                include_catalogue_type_feature=include_catalogue_type_feature,
            )
            test_cf_acc = evaluate_cf_accuracy_precomputed(
                test_catalogues,
                model,
                metadata,
                device,
                tabular_stats=tabular_stats,
                include_catalogue_type_feature=include_catalogue_type_feature,
            )
            train_cf_rank = evaluate_catalogue_metrics_cf_precomputed(
                train_catalogues,
                model,
                metadata,
                device,
                tabular_stats=tabular_stats,
                include_catalogue_type_feature=include_catalogue_type_feature,
                gain=gain,
            )
            test_cf_rank = evaluate_catalogue_metrics_cf_precomputed(
                test_catalogues,
                model,
                metadata,
                device,
                tabular_stats=tabular_stats,
                include_catalogue_type_feature=include_catalogue_type_feature,
                gain=gain,
            )
        else:
            train_cf_acc = evaluate_cf_accuracy_encoded_at_runtime(
                train_catalogues,
                model,
                device=device,
                tabular_stats=tabular_stats,
                image_transform=metadata["image_transform"],
                cache_dir=metadata.get("cache_dir"),
                use_cache=bool(metadata.get("use_cache", True)),
                include_catalogue_type_feature=include_catalogue_type_feature,
            )
            test_cf_acc = evaluate_cf_accuracy_encoded_at_runtime(
                test_catalogues,
                model,
                device=device,
                tabular_stats=tabular_stats,
                image_transform=metadata["image_transform"],
                cache_dir=metadata.get("cache_dir"),
                use_cache=bool(metadata.get("use_cache", True)),
                include_catalogue_type_feature=include_catalogue_type_feature,
            )
            train_cf_rank = evaluate_catalogue_metrics_cf_encoded_at_runtime(
                train_catalogues,
                model,
                device=device,
                tabular_stats=tabular_stats,
                image_transform=metadata["image_transform"],
                cache_dir=metadata.get("cache_dir"),
                use_cache=bool(metadata.get("use_cache", True)),
                include_catalogue_type_feature=include_catalogue_type_feature,
                gain=gain,
            )
            test_cf_rank = evaluate_catalogue_metrics_cf_encoded_at_runtime(
                test_catalogues,
                model,
                device=device,
                tabular_stats=tabular_stats,
                image_transform=metadata["image_transform"],
                cache_dir=metadata.get("cache_dir"),
                use_cache=bool(metadata.get("use_cache", True)),
                include_catalogue_type_feature=include_catalogue_type_feature,
                gain=gain,
            )
        train_mm_acc = evaluate_mm_triplet_accuracy(model, mm_train_loader, device, distance_metric=distance_metric)
        test_mm_acc = evaluate_mm_triplet_accuracy(model, mm_test_loader, device, distance_metric=distance_metric)
        train_mm_rank = evaluate_catalogue_metrics_mm(
            train_catalogues,
            model,
            metadata,
            device,
            tabular_stats=tabular_stats,
            distance_metric=distance_metric,
            gain=gain,
        )
        test_mm_rank = evaluate_catalogue_metrics_mm(
            test_catalogues,
            model,
            metadata,
            device,
            tabular_stats=tabular_stats,
            distance_metric=distance_metric,
            gain=gain,
        )

        test_family_rank_metrics: Dict[str, Dict[str, float]] = {}
        for family in ("Excl", "Fair"):
            family_test = [c for c in test_catalogues if c.catalogue_type == family]
            if not family_test:
                continue

            if mode_plan["use_precomputed"]:
                family_cf_rank = evaluate_catalogue_metrics_cf_precomputed(
                    family_test,
                    model,
                    metadata,
                    device,
                    tabular_stats=tabular_stats,
                    include_catalogue_type_feature=include_catalogue_type_feature,
                    gain=gain,
                )
            else:
                family_cf_rank = evaluate_catalogue_metrics_cf_encoded_at_runtime(
                    family_test,
                    model,
                    device=device,
                    tabular_stats=tabular_stats,
                    image_transform=metadata["image_transform"],
                    cache_dir=metadata.get("cache_dir"),
                    use_cache=bool(metadata.get("use_cache", True)),
                    include_catalogue_type_feature=include_catalogue_type_feature,
                    gain=gain,
                )
            family_mm_rank = evaluate_catalogue_metrics_mm(
                family_test,
                model,
                metadata,
                device,
                tabular_stats=tabular_stats,
                distance_metric=distance_metric,
                gain=gain,
            )
            family_cf_accuracy = (
                evaluate_cf_accuracy_precomputed(
                    family_test,
                    model,
                    metadata,
                    device,
                    tabular_stats=tabular_stats,
                    include_catalogue_type_feature=include_catalogue_type_feature,
                )
                if mode_plan["use_precomputed"]
                else evaluate_cf_accuracy_encoded_at_runtime(
                    family_test,
                    model,
                    device=device,
                    tabular_stats=tabular_stats,
                    image_transform=metadata["image_transform"],
                    cache_dir=metadata.get("cache_dir"),
                    use_cache=bool(metadata.get("use_cache", True)),
                    include_catalogue_type_feature=include_catalogue_type_feature,
                )
            )
            test_family_rank_metrics[family.lower()] = {
                "cf_accuracy": family_cf_accuracy,
                "cf_ndcg": family_cf_rank["ndcg"],
                "cf_rmse": family_cf_rank["rmse"],
                "mm_accuracy": evaluate_mm_catalogue_accuracy(
                    family_test,
                    model,
                    metadata,
                    device,
                    tabular_stats=tabular_stats,
                    distance_metric=distance_metric,
                ),
                "mm_ndcg": family_mm_rank["ndcg"],
            }

        log_dict: Dict[str, Any] = {
            "epoch": float(epoch + 1),
            "loss_total": running_total_loss / max(steps, 1),
            "loss_cf": running_cf_loss / max(steps, 1),
            "loss_mm": running_mm_loss / max(steps, 1),
            "distance_metric": distance_metric,
            "gain": gain,
            "loss_metric": normalized_loss_metric,
            "train_cf_accuracy": train_cf_acc,
            "test_cf_accuracy": test_cf_acc,
            "train_cf_ndcg": train_cf_rank["ndcg"],
            "test_cf_ndcg": test_cf_rank["ndcg"],
            "train_cf_rmse": train_cf_rank["rmse"],
            "test_cf_rmse": test_cf_rank["rmse"],
            "train_mm_triplet_accuracy": train_mm_acc,
            "test_mm_triplet_accuracy": test_mm_acc,
            "train_mm_ndcg": train_mm_rank["ndcg"],
            "test_mm_ndcg": test_mm_rank["ndcg"],
        }
        for family, metrics in test_family_rank_metrics.items():
            for key, value in metrics.items():
                log_dict[f"test_{family}_{key}"] = value

        if uncertainty_module is not None:
            log_dict.update(uncertainty_module.stats())

        if use_wandb:
            import wandb
            wandb.log(log_dict)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], L_total: {log_dict['loss_total']:.4f}, L_cf: {log_dict['loss_cf']:.4f}, L_mm: {log_dict['loss_mm']:.4f}, CF Acc train|test: {train_cf_acc * 100:.2f}% | {test_cf_acc * 100:.2f}%, CF NDCG train|test: {train_cf_rank['ndcg'] * 100:.2f}% | {test_cf_rank['ndcg'] * 100:.2f}%, CF RMSE train|test: {train_cf_rank['rmse']:.4f} | {test_cf_rank['rmse']:.4f}, MM Triplet Acc train|test: {train_mm_acc * 100:.2f}% | {test_mm_acc * 100:.2f}%, MM NDCG train|test: {train_mm_rank['ndcg'] * 100:.2f}% | {test_mm_rank['ndcg'] * 100:.2f}%"
        )
        for family, metrics in test_family_rank_metrics.items():
            print(
                f"  {family.title()} -> "
                f"Test CF Acc: {metrics['cf_accuracy'] * 100:.2f}%, "
                f"Test CF nDCG: {metrics['cf_ndcg'] * 100:.2f}%, "
                f"Test CF RMSE: {metrics['cf_rmse']:.4f}, "
                f"Test MM Acc: {metrics['mm_accuracy'] * 100:.2f}%, "
                f"Test MM nDCG: {metrics['mm_ndcg'] * 100:.2f}%"
            )

        checkpoint_score = 0.5 * (test_cf_acc + test_mm_acc)
        if checkpoint_dir is not None:
            os.makedirs(checkpoint_dir, exist_ok=True)
            save_joint_checkpoint(model, optimizer, epoch, best_score, os.path.join(checkpoint_dir, "last_checkpoint.pt"), uncertainty_module=uncertainty_module)
            save_model(model, os.path.join(checkpoint_dir, "last_model.pt"))

            if checkpoint_score > best_score:
                best_score = checkpoint_score
                save_joint_checkpoint(model, optimizer, epoch, best_score, os.path.join(checkpoint_dir, "best_checkpoint.pt"), uncertainty_module=uncertainty_module)
                save_model(model, os.path.join(checkpoint_dir, "best_model.pt"))

    return model
