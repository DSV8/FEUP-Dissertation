"""Configure and run image-plus-tabular metric-learning experiments."""

import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import wandb

from models.image_encoders import GC_ViT_224
from models.multimodal import MultimodalEncoder
from utils.cache_preprocessing import is_cache_available, preprocess_and_cache_dataset
from utils.image_preprocessing import transform_fn
from utils.catalogue_utils import build_catalogues, normalize_catalogue_type, split_catalogues_canonical
from mm import (
	build_dataset_from_catalogues,
	build_embedding_triplet_dataset_from_path_dataset,
	build_ranking_transform_multimodal,
	get_dataloaders_multimodal,
	prepare_multimodal_datasets_from_catalogues,
	prepare_precomputed_multimodal_datasets_from_catalogues,
	train_multimodal,
)
from utils.image_embedding import (
	IMAGE_EMBEDDING_MODE_ENCODED_AT_RUNTIME,
	IMAGE_EMBEDDING_MODE_PRECOMPUTED,
	normalize_image_embedding_mode,
)
from configuration import PROJECT_ROOT as PROJECT_ROOT_PATH, load_pipeline_config


CONFIG = load_pipeline_config("mm")
DATA_CONFIG = CONFIG["data"]
TRAINING_CONFIG = CONFIG["training"]
MODEL_CONFIG = CONFIG["model"]
LOADER_CONFIG = CONFIG["loader"]
EVALUATION_CONFIG = CONFIG["evaluation"]
NAS_DATASET_PATH = DATA_CONFIG["dataset_path"]
PROJECT_ROOT = str(PROJECT_ROOT_PATH)
CACHE_DIR = os.path.join(PROJECT_ROOT, CONFIG["paths"]["cache_dir"])


def ensure_cache(data_dir: str, cache_dir: str, catalogue_type: str) -> bool:
	if not is_cache_available(data_dir, cache_dir, catalogue_type=catalogue_type):
		preprocess_and_cache_dataset(
			nas_dataset_path=data_dir,
			cache_dir=cache_dir,
			verbose=False,
			catalogue_type=catalogue_type,
		)
	return True


def split_for_training(
	data_dir: str,
	catalogue_type: str,
	split_ratio: float,
	seed: int,
) -> Tuple[List[Any], List[Any]]:
	"""Build catalogue records and apply the shared canonical split policy."""
	catalogues = build_catalogues(data_dir)
	return split_catalogues_canonical(
		catalogues,
		split_ratio=split_ratio,
		seed=seed,
		catalogue_type=catalogue_type,
	)


def prepare_mm_datasets(
	train_catalogues: List[Any],
	test_catalogues: List[Any],
	image_encoder: torch.nn.Module,
	device: str,
	image_embedding_mode: str,
	include_catalogue_type_feature: bool,
	cache_dir: str,
	use_cache: bool,
) -> Tuple[Any, Any, Dict[str, Any], Optional[Dict[str, Any]]]:
	# Both embedding modes return the same logical triplet/task contract; only
	# image loading differs between the training and ranking datasets.
	if image_embedding_mode == IMAGE_EMBEDDING_MODE_PRECOMPUTED:
		train_dataset, test_dataset, tabular_stats, mm_metadata = prepare_precomputed_multimodal_datasets_from_catalogues(
			train_catalogues,
			test_catalogues,
			image_encoder=image_encoder,
			device=device,
			normalize_tabular=True,
			image_transform=transform_fn,
			cache_dir=cache_dir,
			use_cache=use_cache,
			include_catalogue_type_feature=include_catalogue_type_feature,
		)
		return train_dataset, test_dataset, tabular_stats, mm_metadata

	train_dataset, test_dataset, tabular_stats = prepare_multimodal_datasets_from_catalogues(
		train_catalogues,
		test_catalogues,
		normalize_tabular=True,
		image_transform=transform_fn,
		cache_dir=cache_dir,
		use_cache=use_cache,
		include_catalogue_type_feature=include_catalogue_type_feature,
	)
	return train_dataset, test_dataset, tabular_stats, None


def build_mixed_family_test_loaders(
	train_dataset: Any,
	test_catalogues_excl: List[Any],
	test_catalogues_fair: List[Any],
	ranking_transform,
	cache_dir: str,
	use_cache: bool,
	include_catalogue_type_feature: bool,
	tabular_stats: Dict[str, Any],
	image_embedding_mode: str,
	image_embedding_map: Optional[Dict[str, Any]],
	image_embedding_dim: Optional[int],
	batch_size: int,
	num_workers: int,
	pin_memory: bool,
	persistent_workers: bool,
	prefetch_factor: int,
) -> Tuple[Optional[Any], Optional[Any]]:
	active_cache_dir = cache_dir if use_cache else None
	test_dataset_excl = build_dataset_from_catalogues(
		test_catalogues_excl,
		image_transform=ranking_transform,
		cache_dir=active_cache_dir,
		include_catalogue_type_feature=include_catalogue_type_feature,
		tabular_stats=tabular_stats,
	)
	test_dataset_fair = build_dataset_from_catalogues(
		test_catalogues_fair,
		image_transform=ranking_transform,
		cache_dir=active_cache_dir,
		include_catalogue_type_feature=include_catalogue_type_feature,
		tabular_stats=tabular_stats,
	)

	if image_embedding_mode == IMAGE_EMBEDDING_MODE_PRECOMPUTED:
		test_dataset_excl = build_embedding_triplet_dataset_from_path_dataset(
			test_dataset_excl,
			image_embedding_map,
			image_embedding_dim,
		)
		test_dataset_fair = build_embedding_triplet_dataset_from_path_dataset(
			test_dataset_fair,
			image_embedding_map,
			image_embedding_dim,
		)

	test_loader_excl = None
	test_loader_fair = None
	if len(test_dataset_excl) > 0:
		_, test_loader_excl = get_dataloaders_multimodal(
			train_dataset,
			test_dataset_excl,
			batch_size=batch_size,
			num_workers=num_workers,
			pin_memory=pin_memory,
			persistent_workers=persistent_workers,
			prefetch_factor=prefetch_factor,
		)
	if len(test_dataset_fair) > 0:
		_, test_loader_fair = get_dataloaders_multimodal(
			train_dataset,
			test_dataset_fair,
			batch_size=batch_size,
			num_workers=num_workers,
			pin_memory=pin_memory,
			persistent_workers=persistent_workers,
			prefetch_factor=prefetch_factor,
		)

	return test_loader_excl, test_loader_fair


def main() -> None:
	split_ratio = DATA_CONFIG["split_ratio"]
	seed = DATA_CONFIG["seed"]
	catalogue_type = DATA_CONFIG["catalogue_type"]
	num_epochs = TRAINING_CONFIG["num_epochs"]
	batch_size = TRAINING_CONFIG["batch_size"]
	margin = TRAINING_CONFIG["margin"]
	lr = TRAINING_CONFIG["learning_rate"]
	device = "cuda" if torch.cuda.is_available() else "cpu"
	tabular_embedding_dim = MODEL_CONFIG["tabular_embedding_dim"]
	fusion_embedding_dim = MODEL_CONFIG["fusion_embedding_dim"]
	distance_metric = TRAINING_CONFIG["distance_metric"]
	loss_metric = TRAINING_CONFIG["loss_metric"]
	use_rank_weighted_loss = TRAINING_CONFIG["use_rank_weighted_loss"]
	rank_weight_alpha = TRAINING_CONFIG["rank_weight_alpha"]
	tabular_nhead = MODEL_CONFIG["tabular_nhead"]
	tabular_num_layers = MODEL_CONFIG["tabular_num_layers"]
	tabular_encoder_type = MODEL_CONFIG["tabular_encoder_type"]
	fusion_nhead = MODEL_CONFIG["fusion_nhead"]
	fusion_num_layers = MODEL_CONFIG["fusion_num_layers"]
	image_embedding_mode = TRAINING_CONFIG["image_embedding_mode"]
	fusion_mode = MODEL_CONFIG["fusion_mode"]

	metrics_ks = EVALUATION_CONFIG["metrics_ks"]
	metrics_relevant_top_n = EVALUATION_CONFIG["metrics_relevant_top_n"]
	metrics_map_k = EVALUATION_CONFIG["metrics_map_k"]
	metrics_gain = EVALUATION_CONFIG["metrics_gain"]

	num_workers = LOADER_CONFIG["num_workers"]
	pin_memory = LOADER_CONFIG["pin_memory"]
	persistent_workers = LOADER_CONFIG["persistent_workers"]
	prefetch_factor = LOADER_CONFIG["prefetch_factor"]

	torch.backends.cudnn.benchmark = True
	torch.backends.cudnn.deterministic = True
	os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,garbage_collection_threshold:0.9,expandable_segments:True"
	os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

	profile_timing = TRAINING_CONFIG["profile_timing"]
	profile_first_n_epochs = TRAINING_CONFIG["profile_first_n_epochs"]
	effective_num_epochs = profile_first_n_epochs if profile_timing else num_epochs

	normalized_catalogue_type = normalize_catalogue_type(catalogue_type)
	is_mixed_catalogue = normalized_catalogue_type == "Mixed"
	include_catalogue_type_feature = is_mixed_catalogue
	checkpoint_dir = os.path.join(PROJECT_ROOT, CONFIG["paths"]["checkpoint_dir"], "mm", normalized_catalogue_type)
	resume_training = TRAINING_CONFIG["resume_training"]
	resume_checkpoint_path = os.path.join(checkpoint_dir, "last_checkpoint.pt") if resume_training else None

	image_embedding_mode = normalize_image_embedding_mode(image_embedding_mode)

	print(f"NAS Dataset Path: {NAS_DATASET_PATH}")
	print(f"Cache Directory: {CACHE_DIR}")
	print(f"Device: {device}")
	print(f"Checkpoint Directory: {checkpoint_dir}")
	print(f"Normalized catalogue type: {normalized_catalogue_type}")
	print(f"Mixed mode: {is_mixed_catalogue}")
	print(f"Image embedding mode: {image_embedding_mode}")
	print(f"Tabular encoder type: {tabular_encoder_type}")
	print(f"Fusion mode: {fusion_mode}")

	use_cache = ensure_cache(NAS_DATASET_PATH, CACHE_DIR, catalogue_type)
	ranking_transform = build_ranking_transform_multimodal(CACHE_DIR, use_cache=use_cache)

	backbone_encoder = GC_ViT_224()
	train_catalogues, test_catalogues = split_for_training(
		NAS_DATASET_PATH,
		catalogue_type,
		split_ratio,
		seed,
	)
	print(f"Split: {len(train_catalogues)} train catalogues, {len(test_catalogues)} test catalogues")

	train_dataset, test_dataset, tabular_stats, mm_metadata = prepare_mm_datasets(
		train_catalogues,
		test_catalogues,
		image_encoder=backbone_encoder,
		device=device,
		image_embedding_mode=image_embedding_mode,
		include_catalogue_type_feature=include_catalogue_type_feature,
		cache_dir=CACHE_DIR,
		use_cache=use_cache,
	)

	image_embedding_map = None
	image_embedding_dim = None
	if mm_metadata is not None:
		image_embedding_map = mm_metadata["image_embedding_map"]
		image_embedding_dim = int(mm_metadata["image_embedding_dim"])

	print(f"Train dataset size: {len(train_dataset)}")
	print(f"Test dataset size: {len(test_dataset)}")

	test_catalogues_excl = [c for c in test_catalogues if c.catalogue_type == "Excl"]
	test_catalogues_fair = [c for c in test_catalogues if c.catalogue_type == "Fair"]
	train_catalogues_excl = [c for c in train_catalogues if c.catalogue_type == "Excl"]
	train_catalogues_fair = [c for c in train_catalogues if c.catalogue_type == "Fair"]
	print(
		"Family split counts -> "
		f"train Excl: {len(train_catalogues_excl)}, train Fair: {len(train_catalogues_fair)}, "
		f"test Excl: {len(test_catalogues_excl)}, test Fair: {len(test_catalogues_fair)}"
	)

	test_dataloader_excl = None
	test_dataloader_fair = None
	if is_mixed_catalogue:
		test_dataloader_excl, test_dataloader_fair = build_mixed_family_test_loaders(
			train_dataset=train_dataset,
			test_catalogues_excl=test_catalogues_excl,
			test_catalogues_fair=test_catalogues_fair,
			ranking_transform=ranking_transform,
			cache_dir=CACHE_DIR,
			use_cache=use_cache,
			include_catalogue_type_feature=include_catalogue_type_feature,
			tabular_stats=tabular_stats,
			image_embedding_mode=image_embedding_mode,
			image_embedding_map=image_embedding_map,
			image_embedding_dim=image_embedding_dim,
			batch_size=batch_size,
			num_workers=num_workers,
			pin_memory=pin_memory,
			persistent_workers=persistent_workers,
			prefetch_factor=prefetch_factor,
		)
		print(f"Mixed test split -> Excl catalogues: {len(test_catalogues_excl)}, Fair catalogues: {len(test_catalogues_fair)}")

	train_dataloader, test_dataloader = get_dataloaders_multimodal(
		train_dataset,
		test_dataset,
		batch_size=batch_size,
		num_workers=num_workers,
		pin_memory=pin_memory,
		persistent_workers=persistent_workers,
		prefetch_factor=prefetch_factor,
	)

	model = MultimodalEncoder(
		image_encoder=(torch.nn.Identity() if image_embedding_mode == IMAGE_EMBEDDING_MODE_PRECOMPUTED else backbone_encoder),
		tabular_in_features=6 if include_catalogue_type_feature else 5,
		tabular_embedding_dim=tabular_embedding_dim,
		fusion_embedding_dim=fusion_embedding_dim,
		tabular_nhead=tabular_nhead,
		tabular_num_layers=tabular_num_layers,
		tabular_encoder_type=tabular_encoder_type,
		fusion_nhead=fusion_nhead,
		fusion_num_layers=fusion_num_layers,
		image_encoder_output_dim=image_embedding_dim if image_embedding_mode == IMAGE_EMBEDDING_MODE_PRECOMPUTED else None,
		fusion_mode=fusion_mode,
	)
	print(f"Model architecture: {model.__class__.__name__}")

	print(f"Hyperparameters: num_epochs={effective_num_epochs}, margin={margin}, lr={lr}")
	print(f"Distance metric: {distance_metric}, Loss metric: {loss_metric}")
	print(f"Rank-weighted loss: {use_rank_weighted_loss}, alpha={rank_weight_alpha}")

	wandb.init(
		project="mm-triplet-learning",
		config={
			"dataset": "CINDERELLA",
			"model": "MultimodalEncoder",
			"num_epochs": num_epochs,
			"batch_size": batch_size,
			"learning_rate": lr,
			"margin": margin,
			"loss_metric": loss_metric,
			"use_rank_weighted_loss": use_rank_weighted_loss,
			"rank_weight_alpha": rank_weight_alpha,
			"tabular_embedding_dim": tabular_embedding_dim,
			"fusion_embedding_dim": fusion_embedding_dim,
			"tabular_nhead": tabular_nhead,
			"tabular_num_layers": tabular_num_layers,
			"fusion_nhead": fusion_nhead,
			"fusion_num_layers": fusion_num_layers,
			"distance_metric": distance_metric,
			"catalogue_type": catalogue_type,
			"normalized_catalogue_type": normalized_catalogue_type,
			"mixed_mode": is_mixed_catalogue,
			"include_catalogue_type_feature": include_catalogue_type_feature,
			"split_ratio": split_ratio,
			"seed": seed,
			"metrics_ks": metrics_ks,
			"metrics_relevant_top_n": metrics_relevant_top_n,
			"metrics_map_k": metrics_map_k,
			"metrics_gain": metrics_gain,
			"resume_training": resume_training,
			"num_workers": num_workers,
			"pin_memory": pin_memory,
			"persistent_workers": persistent_workers,
			"prefetch_factor": prefetch_factor,
			"image_embedding_mode": image_embedding_mode,
		},
	)

	train_multimodal(
		model,
		train_dataloader,
		test_dataloader,
		device,
		image_embedding_mode=image_embedding_mode,
		image_embedding_map=image_embedding_map,
		num_epochs=effective_num_epochs,
		margin=margin,
		lr=lr,
		use_wandb=True,
		train_catalogues=train_catalogues,
		test_catalogues=test_catalogues,
		test_loader_excl=test_dataloader_excl,
		test_loader_fair=test_dataloader_fair,
		test_catalogues_excl=test_catalogues_excl if is_mixed_catalogue else None,
		test_catalogues_fair=test_catalogues_fair if is_mixed_catalogue else None,
		tabular_stats=tabular_stats,
		ks=tuple(metrics_ks),
		relevant_top_n=metrics_relevant_top_n,
		map_k=metrics_map_k,
		distance_metric=distance_metric,
		loss_metric=loss_metric,
		checkpoint_dir=checkpoint_dir,
		resume_checkpoint_path=resume_checkpoint_path,
		profile_timing=profile_timing,
		profile_first_n_epochs=profile_first_n_epochs,
		include_catalogue_type_feature=include_catalogue_type_feature,
		gain=metrics_gain,
		use_rank_weighted_loss=use_rank_weighted_loss,
		rank_weight_alpha=rank_weight_alpha,
	)

	wandb.finish()


if __name__ == "__main__":
	main()
