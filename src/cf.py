"""Collaborative-filtering pipeline with pointwise rank supervision."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.image_embedding import (
	IMAGE_EMBEDDING_MODE_ENCODED_AT_RUNTIME,
	IMAGE_EMBEDDING_MODE_PRECOMPUTED,
	normalize_image_embedding_mode,
)
from utils.catalogue_utils import Catalogue
from utils.image_preprocessing import transform_fn

from utils.cf.cf_data import (
    CFDatasetBundle,
    CFInteraction,
    CFEncodedAtRuntimeInteraction,
    InteractionCFDataset,
    EncodedAtRuntimeInteractionCFDataset,
    CFEncodedAtRuntimeScoringWrapper,
    rank_to_target,
    create_cf_interactions,
    create_cf_encoded_at_runtime_interactions,
    _make_pointwise_scoring_inputs,
    compute_img_embeddings,
    load_img_tensor,
    infer_image_embedding_dim,
    prepare_cf_datasets,
    get_cf_dataloaders,
)

from utils.cf.cf_evaluation import (
    evaluate_cf_accuracy,
    evaluate_cf_accuracy_encoded_at_runtime,
    evaluate_catalogue_metrics_cf,
    evaluate_catalogue_metrics_cf_encoded_at_runtime,
)

from utils.cf.cf_visualization import (
    predict_catalogue_ranking_cf,
    predict_catalogue_ranking_cf_encoded_at_runtime,
    visualize_predicted_ranking_cf,
)

def save_model(model: nn.Module, filepath: str) -> None:
	save_dir = os.path.dirname(filepath)
	if save_dir:
		os.makedirs(save_dir, exist_ok=True)
	torch.save(model.state_dict(), filepath)
	print(f"Model saved to {filepath}")

def load_model(model: nn.Module, filepath: str, device: str = "cpu") -> nn.Module:
	model.load_state_dict(torch.load(filepath, map_location=device))
	model.to(device)
	model.eval()
	print(f"Model loaded from {filepath}")
	return model

def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, best_metric: float, checkpoint_path: str) -> None:
	save_dir = os.path.dirname(checkpoint_path)
	if save_dir:
		os.makedirs(save_dir, exist_ok=True)
	torch.save({
		"epoch":               epoch,
		"model_state_dict":    model.state_dict(),
		"optimizer_state_dict": optimizer.state_dict(),
		"best_metric":         best_metric,
	}, checkpoint_path)

def load_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, checkpoint_path: str, device: str = "cpu") -> Tuple[int, float]:
	ckpt = torch.load(checkpoint_path, map_location=device)
	model.load_state_dict(ckpt["model_state_dict"])
	optimizer.load_state_dict(ckpt["optimizer_state_dict"])
	model.to(device)
	start_epoch = ckpt.get("epoch", -1) + 1
	best_metric = ckpt.get("best_metric", float("-inf"))
	return start_epoch, best_metric

def train_cf_model(
	model: nn.Module,
	train_loader: DataLoader,
	test_loader: DataLoader,
	train_catalogues: List[Catalogue],
	test_catalogues: List[Catalogue],
	metadata: Dict[str, Any],
	device: str,
	num_epochs: int = 100,
	lr: float = 1e-4,
	weight_decay: float = 0.0,
	use_wandb: bool = False,
	checkpoint_dir: Optional[str] = None,
	resume_checkpoint_path: Optional[str] = None,
	gain: str = "linear",
	tabular_stats: Optional[Dict[str, np.ndarray]] = None,
	model_variant: str = "NCF",
) -> nn.Module:
	"""
	Train FM/NN/NCF with pointwise interaction supervision.
	"""
	optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
	normalized_variant = model_variant.strip().upper()
	if normalized_variant not in {"FM", "NN", "NCF"}:
		raise ValueError("Invalid model_variant for CF training. Use one of: 'FM', 'NN', 'NCF'.")
	criterion = nn.MSELoss()
	model.to(device)

	start_epoch = 0
	best_test_acc = float("-inf")

	if resume_checkpoint_path is not None and os.path.exists(resume_checkpoint_path):
		start_epoch, best_test_acc = load_checkpoint(model, optimizer, resume_checkpoint_path, device=device)
		print(f"Resumed training from epoch {start_epoch}")

	for epoch in range(start_epoch, num_epochs):
		model.train()
		running_loss = 0.0

		for batch in train_loader:
			pred_score = model(_make_pointwise_scoring_inputs(batch, device))
			target = batch["target"].to(device)
			loss = criterion(pred_score, target)
			batch_size_actual = pred_score.size(0)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			running_loss += loss.item() * batch_size_actual

		epoch_loss  = running_loss / len(train_loader.dataset)
		train_acc = evaluate_cf_accuracy(
			train_catalogues,
			model,
			metadata,
			device,
			tabular_stats=tabular_stats,
			model_variant=model_variant,
		)
		test_acc = evaluate_cf_accuracy(
			test_catalogues,
			model,
			metadata,
			device,
			tabular_stats=tabular_stats,
			model_variant=model_variant,
		)
		train_rank  = evaluate_catalogue_metrics_cf(train_catalogues, model, metadata, device, tabular_stats=tabular_stats, gain=gain, model_variant=model_variant)
		test_rank   = evaluate_catalogue_metrics_cf(test_catalogues,  model, metadata, device, tabular_stats=tabular_stats, gain=gain, model_variant=model_variant)

		family_rank_metrics: Dict[str, Dict[str, float]] = {}
		for family in ("Excl", "Fair"):
			family_train = [c for c in train_catalogues if c.catalogue_type == family]
			family_test = [c for c in test_catalogues if c.catalogue_type == family]
			if family_train and family_test:
				family_train_accuracy = evaluate_cf_accuracy(
					family_train, model, metadata, device, tabular_stats=tabular_stats, model_variant=model_variant,
				)
				family_test_accuracy = evaluate_cf_accuracy(
					family_test, model, metadata, device, tabular_stats=tabular_stats, model_variant=model_variant,
				)
				family_train_rank = evaluate_catalogue_metrics_cf(
					family_train, model, metadata, device, tabular_stats=tabular_stats, gain=gain, model_variant=model_variant,
				)
				family_test_rank = evaluate_catalogue_metrics_cf(
					family_test, model, metadata, device, tabular_stats=tabular_stats, gain=gain, model_variant=model_variant,
				)
				family_rank_metrics[f"train_{family.lower()}"] = family_train_rank
				family_rank_metrics[f"test_{family.lower()}"] = family_test_rank
				family_rank_metrics[f"train_{family.lower()}"]["accuracy"] = family_train_accuracy
				family_rank_metrics[f"test_{family.lower()}"]["accuracy"] = family_test_accuracy

		log_dict = {
			"epoch":             epoch + 1,
			"loss":              epoch_loss,
			"train_accuracy":    train_acc,
			"test_accuracy":     test_acc,
			"train_ndcg":        train_rank["ndcg"],
			"test_ndcg":         test_rank["ndcg"],
			"train_rmse":        train_rank["rmse"],
			"test_rmse":         test_rank["rmse"],
		}
		for split_family, metrics in family_rank_metrics.items():
			for key, value in metrics.items():
				log_dict[f"{split_family}_{key}"] = value

		if use_wandb:
			import wandb
			wandb.log(log_dict)

		print(
			f"Epoch [{epoch + 1}/{num_epochs}] "
			f"Loss: {epoch_loss:.4f}, "
			f"Train Acc: {train_acc * 100:.2f}%, Test Acc: {test_acc * 100:.2f}%, "
			f"Train nDCG: {train_rank['ndcg'] * 100:.2f}%, Test nDCG: {test_rank['ndcg'] * 100:.2f}%, "
			f"Train RMSE: {train_rank['rmse']:.4f}, Test RMSE: {test_rank['rmse']:.4f}"
		)
		for family in ("excl", "fair"):
			train_family = family_rank_metrics.get(f"train_{family}")
			test_family = family_rank_metrics.get(f"test_{family}")
			if train_family is not None and test_family is not None:
				print(
					f"  {family.title()} -> "
					f"Train Acc: {train_family['accuracy'] * 100:.2f}%, Test Acc: {test_family['accuracy'] * 100:.2f}%, "
					f"Train nDCG: {train_family['ndcg'] * 100:.2f}%, Test nDCG: {test_family['ndcg'] * 100:.2f}%, "
					f"Train RMSE: {train_family['rmse']:.4f}, Test RMSE: {test_family['rmse']:.4f}"
				)

		if checkpoint_dir is not None:
			os.makedirs(checkpoint_dir, exist_ok=True)
			save_checkpoint(model, optimizer, epoch, best_test_acc, os.path.join(checkpoint_dir, "last_checkpoint.pt"))
			save_model(model, os.path.join(checkpoint_dir, "last_model.pt"))
			if test_acc > best_test_acc:
				best_test_acc = test_acc
				save_checkpoint(model, optimizer, epoch, best_test_acc, os.path.join(checkpoint_dir, "best_checkpoint.pt"))
				save_model(model, os.path.join(checkpoint_dir, "best_model.pt"))

	return model
