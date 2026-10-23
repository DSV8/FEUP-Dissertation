"""Multimodal retrieval training, evaluation, and visualization interfaces."""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb

from utils.image_embedding import (
	IMAGE_EMBEDDING_MODE_PRECOMPUTED,
	IMAGE_EMBEDDING_MODE_ENCODED_AT_RUNTIME,
	normalize_image_embedding_mode,
)
from utils.image_preprocessing import transform_fn
from utils.catalogue_utils import Catalogue
from models.multimodal import MultimodalEncoder

from utils.mm.mm_data import (
    resolve_image_embedding_mode_plan,
    MultimodalSample,
    MultimodalTriplet,
    EmbeddingMultimodalTriplet,
    compute_rank_weight,
    convert_brasize_to_underbust,
    convert_bracup_to_oubustdiff,
    catalogue_type_feature_from_image_path,
    tabular_vec_from_img_path,
    create_multimodal_triplets,
    build_dataset_from_catalogues,
    fit_tabular_minmax,
    apply_tabular_minmax,
    TripletMultimodalDataset,
    EmbeddingTripletMultimodalDataset,
    compute_img_embeddings_mm,
    build_embedding_triplet_dataset_from_path_dataset,
    prepare_precomputed_multimodal_datasets_from_catalogues,
    prepare_multimodal_datasets,
    prepare_multimodal_datasets_from_catalogues,
    get_dataloaders_multimodal,
    build_ranking_transform_multimodal,
    normalize_tabular_if_needed,
)

from utils.mm.mm_evaluation import (
    evaluate_multimodal,
    evaluate_catalogue_metrics_multimodal,
    comp_pairwise_distance,
)

from utils.mm.mm_visualization import (
    visualize_catalogue_multimodal,
    visualize_triplet_multimodal,
    visualize_predicted_ranking_multimodal,
)

def save_model(model: MultimodalEncoder, filepath: str) -> None:
	"""Save only model weights for inference/evaluation use cases."""
	save_dir = os.path.dirname(filepath)
	if save_dir:
		os.makedirs(save_dir, exist_ok=True)
	torch.save(model.state_dict(), filepath)
	print(f"Model saved to {filepath}")

def load_model(model: MultimodalEncoder, filepath: str, device: str = "cpu") -> MultimodalEncoder:
	"""Load model weights and set model to eval mode."""
	state_dict = torch.load(filepath, map_location=device)
	model.load_state_dict(state_dict)
	model.to(device)
	model.eval()
	print(f"Model loaded from {filepath}")
	return model

def save_checkpoint(model: MultimodalEncoder, optimizer: torch.optim.Optimizer, epoch: int, best_test_acc: float, checkpoint_path: str) -> None:
	"""Save training checkpoint with model and optimizer states."""
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

def load_checkpoint(model: MultimodalEncoder, optimizer: torch.optim.Optimizer, checkpoint_path: str, device: str = "cpu") -> Tuple[int, float]:
	"""Load a training checkpoint and return (start_epoch, best_test_acc)."""
	checkpoint = torch.load(checkpoint_path, map_location=device)
	model.load_state_dict(checkpoint["model_state_dict"])
	optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
	model.to(device)

	start_epoch = checkpoint.get("epoch", -1) + 1
	best_test_acc = checkpoint.get("best_test_acc", float("-inf"))
	return start_epoch, best_test_acc

def train_multimodal(
	model: MultimodalEncoder,
	train_loader: DataLoader,
	test_loader: DataLoader,
	device: torch.device,
	image_embedding_mode: str = IMAGE_EMBEDDING_MODE_ENCODED_AT_RUNTIME,
	image_embedding_map: Optional[Dict[str, np.ndarray]] = None,
	num_epochs: int = 100,
	margin: float = 1e-4,
	lr: float = 1e-5,
	use_wandb: bool = False,
	train_catalogues: Optional[List[Catalogue]] = None,
	test_catalogues: Optional[List[Catalogue]] = None,
	test_loader_excl: Optional[DataLoader] = None,
	test_loader_fair: Optional[DataLoader] = None,
	test_catalogues_excl: Optional[List[Catalogue]] = None,
	test_catalogues_fair: Optional[List[Catalogue]] = None,
	tabular_stats: Optional[Dict[str, np.ndarray]] = None,
	ks: Tuple[int, ...] = (1, 5, 10),
	relevant_top_n: int = 5,
	map_k: int = 10,
	distance_metric: str = "cosine",
	gain: str = "linear",
	checkpoint_dir: Optional[str] = None,
	resume_checkpoint_path: Optional[str] = None,
	profile_timing: bool = False,
	profile_first_n_epochs: int = 3,
	loss_metric: str = "triplet_margin",
	include_catalogue_type_feature: bool = False,
	use_rank_weighted_loss: bool = False,
	rank_weight_alpha: float = 0.25,
) -> None:
	"""
	Train a multimodal encoder using triplet loss.

	Args:
		model: Multimodal encoder to train
		train_loader: Training dataloader
		test_loader: Test dataloader (for evaluation during training)
		device: Device to train on
		num_epochs: Number of training epochs
		margin: Margin for triplet loss
		lr: Learning rate for optimizer
		use_wandb: Whether to log metrics to Weights and Biases
		checkpoint_dir: Directory for best/last checkpoints and models
		resume_checkpoint_path: Optional checkpoint path to resume training
	"""
	mode_plan = resolve_image_embedding_mode_plan(image_embedding_mode, image_embedding_map)
	image_embedding_mode = mode_plan["mode"]
	rank_eval_embedding_map = mode_plan["rank_eval_embedding_map"]
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

	# Support multiple loss choices. Default is margin-based triplet loss.
	loss_metric_lower = loss_metric.strip().lower() if isinstance(loss_metric, str) else str(loss_metric)
	if loss_metric_lower in {"triplet", "triplet_margin"}:
		criterion = nn.TripletMarginWithDistanceLoss(
			distance_function=embedding_distance,
			margin=margin,
			reduction="none",
		)
	elif loss_metric_lower in {"quadratic"}:
		# Quadratic hinge-style triplet loss:
		# loss = relu(d_pos - d_neg + margin)^2
		def quadratic_loss(q: torch.Tensor, p: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
			d_pos = embedding_distance(q, p)
			d_neg = embedding_distance(q, n)
			relu = F.relu(d_pos - d_neg + margin)
			return relu * relu
		criterion = quadratic_loss
	else:
		raise ValueError(f"Unsupported loss_metric '{loss_metric}'. Supported: 'triplet_margin', 'quadratic'.")

	model.to(device)
	start_epoch = 0
	best_test_acc = float("-inf")

	if resume_checkpoint_path is not None and os.path.exists(resume_checkpoint_path):
		start_epoch, best_test_acc = load_checkpoint(model, optimizer, resume_checkpoint_path, device=str(device))
		print(f"Resumed training from epoch {start_epoch}")

	for epoch in range(start_epoch, num_epochs):
		model.train()
		running_loss = 0.0

		epoch_timers: Dict[str, float] = {
			"train_batch_loop_s": 0.0,
			"eval_triplets_train_s": 0.0,
			"eval_triplets_test_s": 0.0,
			"eval_rank_train_s": 0.0,
			"eval_rank_test_s": 0.0,
		}

		timing_enabled = profile_timing and ((epoch - start_epoch) < profile_first_n_epochs)

		batch_loop_start = time.perf_counter() if timing_enabled else 0.0
		for batch in train_loader:
			q_img = batch["query_image"].to(device)
			pos_img = batch["pos_image"].to(device)
			neg_img = batch["neg_image"].to(device)
			rank_gap = batch.get("rank_gap")
			if rank_gap is not None:
				rank_gap = rank_gap.to(device)

			q_tab = batch["query_tabular"].to(device)
			pos_tab = batch["pos_tabular"].to(device)
			neg_tab = batch["neg_tabular"].to(device)

			# Get fused embeddings
			query_emb = model(q_img, q_tab)
			pos_emb = model(pos_img, pos_tab)
			neg_emb = model(neg_img, neg_tab)

			# Compute triplet loss
			per_sample_loss = criterion(query_emb, pos_emb, neg_emb)

			if use_rank_weighted_loss and rank_gap is not None:
				sample_weights = compute_rank_weight(rank_gap, rank_weight_alpha)
				loss = torch.mean(per_sample_loss * sample_weights)
			else:
				loss = torch.mean(per_sample_loss)

			# Backpropagation (use set_to_none for efficiency)
			optimizer.zero_grad(set_to_none=True)
			loss.backward()
			optimizer.step()

			running_loss += loss.item() * q_img.size(0)

		if timing_enabled:
			epoch_timers["train_batch_loop_s"] = time.perf_counter() - batch_loop_start

		# Evaluation
		eval_train_start = time.perf_counter() if timing_enabled else 0.0
		train_acc = evaluate_multimodal(model, train_loader, device, distance_metric=distance_metric)
		if timing_enabled:
			epoch_timers["eval_triplets_train_s"] = time.perf_counter() - eval_train_start

		eval_test_start = time.perf_counter() if timing_enabled else 0.0
		test_acc = evaluate_multimodal(model, test_loader, device, distance_metric=distance_metric)
		if timing_enabled:
			epoch_timers["eval_triplets_test_s"] = time.perf_counter() - eval_test_start

		test_excl_acc = None
		test_fair_acc = None
		if test_loader_excl is not None:
			test_excl_acc = evaluate_multimodal(model, test_loader_excl, device, distance_metric=distance_metric)
		if test_loader_fair is not None:
			test_fair_acc = evaluate_multimodal(model, test_loader_fair, device, distance_metric=distance_metric)

		epoch_loss = running_loss / len(train_loader.dataset)

		log_dict = {
			"epoch": epoch + 1,
			"loss": epoch_loss,
			"train_accuracy": train_acc,
			"test_accuracy": test_acc,
			"distance_metric": distance_metric,
		}
		if test_excl_acc is not None:
			log_dict["test_excl_accuracy"] = test_excl_acc
		if test_fair_acc is not None:
			log_dict["test_fair_accuracy"] = test_fair_acc

		if train_catalogues is not None and test_catalogues is not None:
			train_rank_metrics = evaluate_catalogue_metrics_multimodal(
				train_catalogues,
				model,
				device=str(device),
				tabular_stats=tabular_stats,
				image_transform=transform_fn,
				image_embedding_map=rank_eval_embedding_map,
				include_catalogue_type_feature=include_catalogue_type_feature,
				ks=ks,
				relevant_top_n=relevant_top_n,
				map_k=map_k,
				distance_metric=distance_metric,
				gain=gain,
			)
			test_rank_metrics = evaluate_catalogue_metrics_multimodal(
				test_catalogues,
				model,
				device=str(device),
				tabular_stats=tabular_stats,
				image_transform=transform_fn,
				image_embedding_map=rank_eval_embedding_map,
				include_catalogue_type_feature=include_catalogue_type_feature,
				ks=ks,
				relevant_top_n=relevant_top_n,
				map_k=map_k,
				distance_metric=distance_metric,
				gain=gain,
			)

			for key, value in train_rank_metrics.items():
				log_dict[f"train_{key}"] = value
			for key, value in test_rank_metrics.items():
				log_dict[f"test_{key}"] = value

			log_dict["train_ndcg"] = train_rank_metrics["ndcg"]
			log_dict["test_ndcg"] = test_rank_metrics["ndcg"]

		if test_catalogues_excl is not None and len(test_catalogues_excl) > 0:
			test_rank_metrics_excl = evaluate_catalogue_metrics_multimodal(
				test_catalogues_excl,
				model,
				device=str(device),
				tabular_stats=tabular_stats,
				image_transform=transform_fn,
				image_embedding_map=rank_eval_embedding_map,
				include_catalogue_type_feature=include_catalogue_type_feature,
				ks=ks,
				relevant_top_n=relevant_top_n,
				map_k=map_k,
				distance_metric=distance_metric,
				gain=gain,
			)
			for key, value in test_rank_metrics_excl.items():
				log_dict[f"test_excl_{key}"] = value

		if test_catalogues_fair is not None and len(test_catalogues_fair) > 0:
			test_rank_metrics_fair = evaluate_catalogue_metrics_multimodal(
				test_catalogues_fair,
				model,
				device=str(device),
				tabular_stats=tabular_stats,
				image_transform=transform_fn,
				image_embedding_map=rank_eval_embedding_map,
				include_catalogue_type_feature=include_catalogue_type_feature,
				ks=ks,
				relevant_top_n=relevant_top_n,
				map_k=map_k,
				distance_metric=distance_metric,
				gain=gain,
			)
			for key, value in test_rank_metrics_fair.items():
				log_dict[f"test_fair_{key}"] = value

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

		if train_catalogues is not None and test_catalogues is not None:
			message = (
				f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, "
				f"Train Acc: {train_acc * 100:.2f}, Test Acc: {test_acc * 100:.2f}, "
				f"Train nDCG: {log_dict['train_ndcg']:.4f}, Test nDCG: {log_dict['test_ndcg']:.4f}, "
				f"Train R@1/5/10: {log_dict.get('train_recall_at_1', 0.0):.4f}/{log_dict.get('train_recall_at_5', 0.0):.4f}/{log_dict.get('train_recall_at_10', 0.0):.4f}, "
				f"Test R@1/5/10: {log_dict.get('test_recall_at_1', 0.0):.4f}/{log_dict.get('test_recall_at_5', 0.0):.4f}/{log_dict.get('test_recall_at_10', 0.0):.4f}, "
				f"Train mAP@10: {log_dict.get('train_map_at_10', 0.0):.4f}, Test mAP@10: {log_dict.get('test_map_at_10', 0.0):.4f}, "
				f"Train P@10/F1@10: {log_dict.get('train_precision_at_10', 0.0):.4f}/{log_dict.get('train_f1_at_10', 0.0):.4f}, "
				f"Test P@10/F1@10: {log_dict.get('test_precision_at_10', 0.0):.4f}/{log_dict.get('test_f1_at_10', 0.0):.4f}"
			)
			if test_excl_acc is not None:
				message += f", Test Excl Acc: {test_excl_acc:.4f}"
			if test_fair_acc is not None:
				message += f", Test Fair Acc: {test_fair_acc:.4f}"
			if "test_excl_ndcg" in log_dict:
				message += f", Test Excl nDCG: {log_dict['test_excl_ndcg']:.4f}"
			if "test_fair_ndcg" in log_dict:
				message += f", Test Fair nDCG: {log_dict['test_fair_ndcg']:.4f}"
			print(message)
		else:
			message = (
				f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Train Acc: {train_acc * 100:.2f}%, Test Acc: {test_acc * 100:.2f}%"
			)
			if test_excl_acc is not None:
				message += f", Test Excl Acc: {test_excl_acc * 100:.2f}%"
			if test_fair_acc is not None:
				message += f", Test Fair Acc: {test_fair_acc * 100:.2f}%"
			print(message)

		# Print timing info when profiling is enabled
		if timing_enabled:
			print(
				"Timing(s) -> "
				f"train_batch_loop={epoch_timers['train_batch_loop_s']:.2f}, "
				f"eval_triplets_train={epoch_timers['eval_triplets_train_s']:.2f}, "
				f"eval_triplets_test={epoch_timers['eval_triplets_test_s']:.2f}, "
				f"eval_rank_train={epoch_timers['eval_rank_train_s']:.2f}, "
				f"eval_rank_test={epoch_timers['eval_rank_test_s']:.2f}"
			)
