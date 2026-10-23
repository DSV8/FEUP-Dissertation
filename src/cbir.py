"""CBIR pipeline: image triplets, metric learning, ranking evaluation, and plots."""

import os
import time
from typing import Dict, List, Optional, Tuple, Any, Callable
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb
from utils.catalogue_utils import (
    Catalogue,
    ImageSample,
    build_catalogues,
    catalogue_type_to_id,
    collect_catalogue_image_paths,
    create_triplets,
    infer_catalogue_type_id_from_image_path,
    normalize_catalogue_type,
    parse_catalogue_folder,
    parse_filename,
    split_catalogues,
    split_catalogues_canonical,
    split_catalogues_stratified_by_family,
)
from utils.cbir.cbir_data import (
    CatalogueConcatEncoder,
    EmbeddingTripletDataset,
    PrecomputedEmbeddingEncoder,
    TripletImageDataset,
    build_dataset_from_catalogues,
    build_precomputed_dataset_from_catalogues,
    build_ranking_transform,
    compute_img_embeddings_cbir,
    compute_rank_gap_from_paths,
    compute_rank_weight,
    prepare_datasets,
    prepare_datasets_from_catalogues,
    prepare_precomputed_datasets_from_catalogues,
)
from utils.image_embedding import (
    IMAGE_EMBEDDING_MODE_ENCODED_AT_RUNTIME,
    IMAGE_EMBEDDING_MODE_PRECOMPUTED,
    normalize_image_embedding_mode,
)
from utils.cbir.cbir_evaluation import (
    comp_pairwise_distance,
    compute_catalogue_distances,
    embed_triplet_batch,
    evaluate_catalogue_metrics,
    evaluate_triplets,
)
from utils.cbir.cbir_visualization import (
    visualize_catalogue,
    visualize_predicted_ranking,
    visualize_triplet,
)
from utils.image_preprocessing import pad_and_resize, transform_fn

# DataLoader construction shared by CBIR training and its baseline evaluator.
def get_dataloaders(train_dataset, test_dataset, batch_size=16, num_workers: int = 8, pin_memory: bool = True, persistent_workers: bool = True, prefetch_factor: int = 4) -> Tuple[DataLoader, DataLoader]:
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


def save_model(model, filepath: str):
    """Save only model weights for inference/evaluation use cases."""
    save_dir = os.path.dirname(filepath)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")


def load_model(model, filepath: str, device: str = "cpu"):
    """Load model weights and set the model to eval mode."""
    state_dict = torch.load(filepath, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"Model loaded from {filepath}")
    return model


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, best_test_acc: float, checkpoint_path: str):
    """Save a training checkpoint with model/optimizer states."""
    save_dir = os.path.dirname(checkpoint_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_test_acc": best_test_acc,
    }
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, checkpoint_path: str, device: str = "cpu") -> Tuple[int, float]:
    """Load a training checkpoint and return (start_epoch, best_test_acc)."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    model.to(device)

    start_epoch = checkpoint.get("epoch", -1) + 1
    best_test_acc = checkpoint.get("best_test_acc", float("-inf"))
    return start_epoch, best_test_acc

# Training and checkpoint selection for the image-only triplet objective.
def train_model(model: torch.nn.Module, train_loader: DataLoader, test_loader: DataLoader, device: str, num_epochs: int = 100, margin: float = 1e-4, lr: float = 1e-5, use_wandb: bool = False, test_catalogues: Optional[List[Catalogue]] = None, train_catalogues: Optional[List[Catalogue]] = None, checkpoint_dir: Optional[str] = None, resume_checkpoint_path: Optional[str] = None, distance_metric: str = "euclidean", profile_timing: bool = False, profile_first_n_epochs: int = 3, ranking_transform: Optional[Callable[[str], torch.Tensor]] = None, loss_metric: str = "triplet_margin", gain: str = "graded", use_rank_weighted_loss: bool = False, rank_weight_alpha: float = 0.25, test_loader_excl: Optional[DataLoader] = None, test_loader_fair: Optional[DataLoader] = None, test_catalogues_excl: Optional[List[Catalogue]] = None, test_catalogues_fair: Optional[List[Catalogue]] = None, metadata: Optional[Dict[str, Any]] = None):
    # Define optimizer and loss function (support multiple loss choices)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    normalized_distance_metric = distance_metric.strip().lower()
    if normalized_distance_metric == "euclidean":
        def embedding_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return torch.norm(a - b, p=2, dim=1)
    elif normalized_distance_metric == "cosine":
        def embedding_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return 1.0 - F.cosine_similarity(a, b, dim=1)
    else:
        raise ValueError("Unsupported distance metric: choose 'euclidean' or 'cosine'.")

    loss_metric_lower = loss_metric.strip().lower() if isinstance(loss_metric, str) else str(loss_metric)
    if loss_metric_lower in {"triplet", "triplet_margin"}:
        if use_rank_weighted_loss:
            criterion = torch.nn.TripletMarginWithDistanceLoss(
                distance_function=embedding_distance,
                margin=margin,
                reduction="none",
            )
        else:
            criterion = torch.nn.TripletMarginWithDistanceLoss(
                distance_function=embedding_distance,
                margin=margin,
            )
    elif loss_metric_lower in {"quadratic"}:
        # Quadratic hinge-style triplet loss: relu(d_pos - d_neg + margin)^2.
        if use_rank_weighted_loss:
            def _quadratic_loss(q: torch.Tensor, p: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
                d_pos = embedding_distance(q, p)
                d_neg = embedding_distance(q, n)
                relu = F.relu(d_pos - d_neg + margin)
                return relu * relu
            criterion = _quadratic_loss
        else:
            def _quadratic_loss_mean(q: torch.Tensor, p: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
                d_pos = embedding_distance(q, p)
                d_neg = embedding_distance(q, n)
                relu = F.relu(d_pos - d_neg + margin)
                return torch.mean(relu * relu)
            criterion = _quadratic_loss_mean
    else:
        raise ValueError(f"Unsupported loss_metric '{loss_metric}'. Supported: 'triplet_margin', 'quadratic'.")
    
    model.to(device)
    start_epoch = 0
    best_test_acc = float("-inf")
    rank_transform = ranking_transform if ranking_transform is not None else transform_fn

    if resume_checkpoint_path is not None and os.path.exists(resume_checkpoint_path):
        start_epoch, best_test_acc = load_checkpoint(model, optimizer, resume_checkpoint_path, device=device)
        print(f"Resumed training from epoch {start_epoch}")

    for epoch in range(start_epoch, num_epochs):
        model.train()  # Ensure train mode at each epoch start
        running_loss = 0.0

        epoch_timers: Dict[str, float] = {
            "train_batch_loop_s": 0.0,
            "eval_triplets_train_s": 0.0,
            "eval_triplets_test_s": 0.0,
            "eval_rank_train_s": 0.0,
            "eval_rank_test_s": 0.0,
        }

        timing_enabled = profile_timing and ((epoch - start_epoch) < profile_first_n_epochs)
        epoch_num = epoch + 1

        batch_loop_start = time.perf_counter() if timing_enabled else 0.0
        for batch in train_loader:
            q = batch['query'].to(device)
            pos = batch['pos'].to(device)
            neg = batch['neg'].to(device)
            catalogue_type_id = batch.get('catalogue_type_id')
            if catalogue_type_id is not None:
                catalogue_type_id = catalogue_type_id.to(device)

            rank_gap = batch.get('rank_gap')
            if rank_gap is not None:
                rank_gap = rank_gap.to(device)

            # Get embeddings
            q_emb, pos_emb, neg_emb = embed_triplet_batch(model, q, pos, neg, catalogue_type_id=catalogue_type_id)
            
            # Compute triplet loss
            per_sample_loss = criterion(q_emb, pos_emb, neg_emb)
            if per_sample_loss.ndim == 0:
                per_sample_loss = per_sample_loss.unsqueeze(0)

            if use_rank_weighted_loss and rank_gap is not None:
                sample_weights = compute_rank_weight(rank_gap, rank_weight_alpha)
                loss = torch.mean(per_sample_loss * sample_weights)
            else:
                loss = torch.mean(per_sample_loss)

            # Backpropagation and optimization step
            optimizer.zero_grad(set_to_none=True) # Clear gradients before backward pass (More efficient than zero_grad() as it avoids unnecessary tensor operations when gradients are not needed)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * q.size(0)

        if timing_enabled:
            epoch_timers["train_batch_loop_s"] = time.perf_counter() - batch_loop_start

        # Evaluation
        epoch_loss = running_loss

        eval_train_start = time.perf_counter() if timing_enabled else 0.0
        train_acc = evaluate_triplets(model, train_loader, device, distance_metric=distance_metric)
        if timing_enabled:
            epoch_timers["eval_triplets_train_s"] = time.perf_counter() - eval_train_start

        eval_test_start = time.perf_counter() if timing_enabled else 0.0
        test_acc = evaluate_triplets(model, test_loader, device, distance_metric=distance_metric)
        if timing_enabled:
            epoch_timers["eval_triplets_test_s"] = time.perf_counter() - eval_test_start

        test_excl_acc = None
        test_fair_acc = None
        if test_loader_excl is not None:
            test_excl_acc = evaluate_triplets(model, test_loader_excl, device, distance_metric=distance_metric)
        if test_loader_fair is not None:
            test_fair_acc = evaluate_triplets(model, test_loader_fair, device, distance_metric=distance_metric)

        # Log metrics
        log_dict = {
            'epoch': epoch_num,
            'loss': epoch_loss,
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'distance_metric': distance_metric,
        }
        if test_excl_acc is not None:
            log_dict['test_excl_accuracy'] = test_excl_acc
        if test_fair_acc is not None:
            log_dict['test_fair_accuracy'] = test_fair_acc

        train_rank_metrics: Optional[Dict[str, float]] = None
        test_rank_metrics: Optional[Dict[str, float]] = None
        if train_catalogues is not None and test_catalogues is not None:
            eval_rank_train_start = time.perf_counter() if timing_enabled else 0.0
            train_rank_metrics = evaluate_catalogue_metrics(train_catalogues, model, rank_transform, device, distance_metric=distance_metric, gain=gain, metadata=metadata)
            if timing_enabled:
                epoch_timers["eval_rank_train_s"] = time.perf_counter() - eval_rank_train_start

            eval_rank_test_start = time.perf_counter() if timing_enabled else 0.0
            test_rank_metrics = evaluate_catalogue_metrics(test_catalogues, model, rank_transform, device, distance_metric=distance_metric, gain=gain, metadata=metadata)
            if timing_enabled:
                epoch_timers["eval_rank_test_s"] = time.perf_counter() - eval_rank_test_start

            for key, value in train_rank_metrics.items():
                log_dict[f"train_{key}"] = value

            for key, value in test_rank_metrics.items():
                log_dict[f"test_{key}"] = value

            # Backward-compatible key for existing dashboards/scripts.
            log_dict["train_ndcg"] = train_rank_metrics["ndcg"]
            log_dict["test_ndcg"] = test_rank_metrics["ndcg"]

        if test_catalogues_excl is not None and len(test_catalogues_excl) > 0:
            test_rank_metrics_excl = evaluate_catalogue_metrics(
                test_catalogues_excl,
                model,
                rank_transform,
                device,
                distance_metric=distance_metric,
                gain=gain,
                metadata=metadata,
            )
            for key, value in test_rank_metrics_excl.items():
                log_dict[f"test_excl_{key}"] = value

        if test_catalogues_fair is not None and len(test_catalogues_fair) > 0:
            test_rank_metrics_fair = evaluate_catalogue_metrics(
                test_catalogues_fair,
                model,
                rank_transform,
                device,
                distance_metric=distance_metric,
                gain=gain,
                metadata=metadata,
            )
            for key, value in test_rank_metrics_fair.items():
                log_dict[f"test_fair_{key}"] = value

        if timing_enabled:
            log_dict.update(epoch_timers)
        
        if use_wandb:
            wandb.log(log_dict)

        if checkpoint_dir is not None:
            best_ckpt_path = os.path.join(checkpoint_dir, "best_checkpoint.pt")
            last_ckpt_path = os.path.join(checkpoint_dir, "last_checkpoint.pt")
            best_model_path = os.path.join(checkpoint_dir, "best_model.pt")
            last_model_path = os.path.join(checkpoint_dir, "last_model.pt")

            save_checkpoint(model, optimizer, epoch, best_test_acc, last_ckpt_path)
            save_model(model, last_model_path)

            checkpoint_score = test_acc
            if test_excl_acc is not None and test_fair_acc is not None:
                checkpoint_score = 0.5 * (test_excl_acc + test_fair_acc)

            if checkpoint_score > best_test_acc:
                best_test_acc = checkpoint_score
                save_checkpoint(model, optimizer, epoch, best_test_acc, best_ckpt_path)
                save_model(model, best_model_path)
        
        message = (
            f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, "
            f"Train Acc: {train_acc * 100:.2f}%, Test Acc: {test_acc * 100:.2f}%"
        )

        if train_rank_metrics is not None and test_rank_metrics is not None:
            message += (
                f", Train nDCG: {log_dict['train_ndcg'] * 100:.2f}, Test nDCG: {log_dict['test_ndcg'] * 100:.2f}, "
                f"Train R@1/5/10: {log_dict['train_recall_at_1']:.4f}/{log_dict['train_recall_at_5']:.4f}/{log_dict['train_recall_at_10']:.4f}, "
                f"Test R@1/5/10: {log_dict['test_recall_at_1']:.4f}/{log_dict['test_recall_at_5']:.4f}/{log_dict['test_recall_at_10']:.4f}, "
                f"Train mAP@10: {log_dict['train_map_at_10']:.4f}, Test mAP@10: {log_dict['test_map_at_10']:.4f}, "
                f"Train P@10/F1@10: {log_dict['train_precision_at_10']:.4f}/{log_dict['train_f1_at_10']:.4f}, "
                f"Test P@10/F1@10: {log_dict['test_precision_at_10']:.4f}/{log_dict['test_f1_at_10']:.4f}"
            )

        if test_excl_acc is not None:
            message += f", Test Excl Acc: {test_excl_acc * 100:.2f}%"
        if test_fair_acc is not None:
            message += f", Test Fair Acc: {test_fair_acc * 100:.2f}%"
        if "test_excl_ndcg" in log_dict:
            message += f", Test Excl nDCG: {log_dict['test_excl_ndcg'] * 100:.2f}%"
        if "test_fair_ndcg" in log_dict:
            message += f", Test Fair nDCG: {log_dict['test_fair_ndcg'] * 100:.2f}%"

        print(message)

        if timing_enabled:
            print(
                "Timing(s) -> "
                f"train_batch_loop={epoch_timers['train_batch_loop_s']:.2f}, "
                f"eval_triplets_train={epoch_timers['eval_triplets_train_s']:.2f}, "
                f"eval_triplets_test={epoch_timers['eval_triplets_test_s']:.2f}, "
                f"eval_rank_train={epoch_timers['eval_rank_train_s']:.2f}, "
                f"eval_rank_test={epoch_timers['eval_rank_test_s']:.2f}"
            )

__all__ = (
    "Catalogue",
    "ImageSample",
    "CatalogueConcatEncoder",
    "EmbeddingTripletDataset",
    "PrecomputedEmbeddingEncoder",
    "TripletImageDataset",
    "IMAGE_EMBEDDING_MODE_ENCODED_AT_RUNTIME",
    "IMAGE_EMBEDDING_MODE_PRECOMPUTED",
    "build_catalogues",
    "build_dataset_from_catalogues",
    "build_precomputed_dataset_from_catalogues",
    "build_ranking_transform",
    "catalogue_type_to_id",
    "collect_catalogue_image_paths",
    "comp_pairwise_distance",
    "compute_catalogue_distances",
    "compute_img_embeddings_cbir",
    "compute_rank_gap_from_paths",
    "compute_rank_weight",
    "create_triplets",
    "embed_triplet_batch",
    "evaluate_catalogue_metrics",
    "evaluate_triplets",
    "get_dataloaders",
    "infer_catalogue_type_id_from_image_path",
    "load_checkpoint",
    "load_model",
    "normalize_catalogue_type",
    "normalize_image_embedding_mode",
    "pad_and_resize",
    "parse_catalogue_folder",
    "parse_filename",
    "prepare_datasets",
    "prepare_datasets_from_catalogues",
    "prepare_precomputed_datasets_from_catalogues",
    "save_checkpoint",
    "save_model",
    "split_catalogues",
    "split_catalogues_canonical",
    "split_catalogues_stratified_by_family",
    "train_model",
    "transform_fn",
    "visualize_catalogue",
    "visualize_predicted_ranking",
    "visualize_triplet",
)

# Keep the historical cbir import surface available to experiment scripts.
