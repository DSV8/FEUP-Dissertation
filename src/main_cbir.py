"""Configure and run image-only CBIR triplet training."""

import os
import sys
from pathlib import Path
from cbir import (
    prepare_datasets_from_catalogues,
    prepare_precomputed_datasets_from_catalogues,
    build_dataset_from_catalogues,
    train_model,
    get_dataloaders,
    build_ranking_transform,
    CatalogueConcatEncoder,
    PrecomputedEmbeddingEncoder,
    build_precomputed_dataset_from_catalogues,
)
from utils.cache_preprocessing import preprocess_and_cache_dataset, is_cache_available
from utils.catalogue_utils import build_catalogues, normalize_catalogue_type, split_catalogues_canonical
from utils.image_embedding import (
    IMAGE_EMBEDDING_MODE_ENCODED_AT_RUNTIME,
    IMAGE_EMBEDDING_MODE_PRECOMPUTED,
    normalize_image_embedding_mode,
)
from models.image_encoders import GC_ViT_224, ResNet50_Base_224
import torch
import wandb
from configuration import PROJECT_ROOT as PROJECT_ROOT_PATH, load_pipeline_config


config = load_pipeline_config("cbir")
data_config = config["data"]
training_config = config["training"]
model_config = config["model"]
loader_config = config["loader"]
evaluation_config = config["evaluation"]
PROJECT_ROOT = str(PROJECT_ROOT_PATH)
NAS_DATASET_PATH = data_config["dataset_path"]
CACHE_DIR = os.path.join(PROJECT_ROOT, config["paths"]["cache_dir"])

torch.backends.cudnn.benchmark = True
#torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cudnn.deterministic = True

# Apply CUDA configuration
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = (
    'max_split_size_mb:128,garbage_collection_threshold:0.9,expandable_segments:True'             # Adjust value based on GPU memory (num of MB) & Adjust threshold for garbage collection (0.0 to 1.0)
)
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Set to '1' for debugging (synchronous execution), '0' for normal asynchronous execution

# Training hyperparameters
split_ratio = data_config["split_ratio"]
seed = data_config["seed"]
catalogue_type = data_config["catalogue_type"]
num_epochs = training_config["num_epochs"]
batch_size = training_config["batch_size"]
margin = training_config["margin"]
lr = training_config["learning_rate"]
device = "cuda" if torch.cuda.is_available() else "cpu"
distance_metric = training_config["distance_metric"]
loss_metric = training_config["loss_metric"]
use_rank_weighted_loss = training_config["use_rank_weighted_loss"]
rank_weight_alpha = training_config["rank_weight_alpha"]
image_embedding_mode = training_config["image_embedding_mode"]

# Profiling configuration
profile_timing = training_config["profile_timing"]
profile_first_n_epochs = training_config["profile_first_n_epochs"]
effective_num_epochs = profile_first_n_epochs if profile_timing else num_epochs

# Throughput-oriented evaluation configuration

# DataLoader performance tuning
num_workers = loader_config["num_workers"]
pin_memory = loader_config["pin_memory"]
persistent_workers = loader_config["persistent_workers"]
prefetch_factor = loader_config["prefetch_factor"]

# Ranking metric configuration (for experiment tracking)
metrics_ks = evaluation_config["metrics_ks"]
metrics_relevant_top_n = evaluation_config["metrics_relevant_top_n"]
metrics_map_k = evaluation_config["metrics_map_k"]
metrics_gain = evaluation_config["metrics_gain"]


# Checkpoint configuration
normalized_catalogue_type = normalize_catalogue_type(catalogue_type)
is_mixed_catalogue = normalized_catalogue_type == "Mixed"
include_catalogue_type_id = is_mixed_catalogue
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, config["paths"]["checkpoint_dir"], "cbir", normalized_catalogue_type)
resume_training = training_config["resume_training"]
resume_checkpoint_path = os.path.join(CHECKPOINT_DIR, "last_checkpoint.pt") if resume_training else None

# CACHE SETUP
print(f"NAS Dataset Path: {NAS_DATASET_PATH}")
print(f"Cache Directory: {CACHE_DIR}")
print(f"Device: {device}")
print("Run Configuration:")
print(f"  Catalogue family: {catalogue_type}")
print(f"  Normalized catalogue family: {normalized_catalogue_type}")
print(f"  Mixed mode: {is_mixed_catalogue}")
print(f"  Distance metric: {distance_metric}")
print(f"  Checkpoint directory: {CHECKPOINT_DIR}")
image_embedding_mode = normalize_image_embedding_mode(image_embedding_mode)
print(f"  Image embedding mode: {image_embedding_mode}")

# Check if cache is available
cache_available = is_cache_available(NAS_DATASET_PATH, CACHE_DIR, catalogue_type=catalogue_type)

if cache_available:
    use_cache = True
else:
    preprocess_and_cache_dataset(
        nas_dataset_path=NAS_DATASET_PATH,
        cache_dir=CACHE_DIR,
        verbose=False,
        catalogue_type=catalogue_type,
    )
    use_cache = True

if model_config["backbone"] == "GC_ViT_224":
    base_model = GC_ViT_224()
elif model_config["backbone"] == "ResNet50_Base_224":
    base_model = ResNet50_Base_224()
else:
    raise ValueError(f"Unsupported CBIR backbone: {model_config['backbone']}")
model = CatalogueConcatEncoder(base_model) if include_catalogue_type_id else base_model
#model = ResNet50_Base_224()
ranking_transform = build_ranking_transform(CACHE_DIR, use_cache=use_cache)

print(f"Hyperparameters: num_epochs={effective_num_epochs}, margin={margin}, lr={lr}")
print(f"Distance metric: {distance_metric}, Loss metric: {loss_metric}")
print(f"Rank-weighted loss: {use_rank_weighted_loss}, alpha={rank_weight_alpha}")

# Split at catalogue level once so triplets and full-catalogue metrics use the
# same held-out cases. In Mixed mode, each family remains represented.
catalogues = build_catalogues(NAS_DATASET_PATH)
train_catalogues, test_catalogues = split_catalogues_canonical(
    catalogues,
    split_ratio=split_ratio,
    seed=seed,
    catalogue_type=catalogue_type,
)

print(f"Split: {len(train_catalogues)} train catalogues, {len(test_catalogues)} test catalogues")

cbir_metadata = None
if image_embedding_mode == IMAGE_EMBEDDING_MODE_PRECOMPUTED:
    train_dataset, test_dataset, cbir_metadata = prepare_precomputed_datasets_from_catalogues(
        train_catalogues=train_catalogues,
        test_catalogues=test_catalogues,
        image_encoder=base_model,
        device=device,
        cache_dir=CACHE_DIR,
        use_cache=use_cache,
        include_catalogue_type_id=include_catalogue_type_id,
    )
    model = PrecomputedEmbeddingEncoder(cbir_metadata["image_embedding_dim"])
else:
    train_dataset, test_dataset = prepare_datasets_from_catalogues(
        train_catalogues=train_catalogues,
        test_catalogues=test_catalogues,
        cache_dir=CACHE_DIR,
        use_cache=use_cache,
        include_catalogue_type_id=include_catalogue_type_id,
    )
    cbir_metadata = {"image_embedding_mode": IMAGE_EMBEDDING_MODE_ENCODED_AT_RUNTIME}

print(f"Model architecture: {model.__class__.__name__}")

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
    active_cache_dir = CACHE_DIR if use_cache else None
    test_dataset_excl = build_dataset_from_catalogues(
        test_catalogues_excl,
        transform=ranking_transform,
        cache_dir=active_cache_dir,
        include_catalogue_type_id=include_catalogue_type_id,
    )
    test_dataset_fair = build_dataset_from_catalogues(
        test_catalogues_fair,
        transform=ranking_transform,
        cache_dir=active_cache_dir,
        include_catalogue_type_id=include_catalogue_type_id,
    )

    if image_embedding_mode == IMAGE_EMBEDDING_MODE_PRECOMPUTED:
        test_dataset_excl = build_precomputed_dataset_from_catalogues(
            test_catalogues_excl,
            image_embedding_map=cbir_metadata["image_embedding_map"],
            image_embedding_dim=cbir_metadata["image_embedding_dim"],
            include_catalogue_type_id=include_catalogue_type_id,
        )
        test_dataset_fair = build_precomputed_dataset_from_catalogues(
            test_catalogues_fair,
            image_embedding_map=cbir_metadata["image_embedding_map"],
            image_embedding_dim=cbir_metadata["image_embedding_dim"],
            include_catalogue_type_id=include_catalogue_type_id,
        )

    if len(test_dataset_excl) > 0:
        _, test_dataloader_excl = get_dataloaders(
            train_dataset,
            test_dataset_excl,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        )
    if len(test_dataset_fair) > 0:
        _, test_dataloader_fair = get_dataloaders(
            train_dataset,
            test_dataset_fair,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        )

# Initialize Weights and Biases
wandb.init(
    project="cbir-triplet-learning",
    config={
        "dataset": "CINDERELLA",
        #"model": "ResNet50_Base_224",
        "model": model.__class__.__name__,
        "num_epochs": effective_num_epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "margin": margin,
        "loss_metric": loss_metric,
        "use_rank_weighted_loss": use_rank_weighted_loss,
        "rank_weight_alpha": rank_weight_alpha,
        "distance_metric": distance_metric,
        "catalogue_type": catalogue_type,
        "normalized_catalogue_type": normalized_catalogue_type,
        "mixed_mode": is_mixed_catalogue,
        "include_catalogue_type_id": include_catalogue_type_id,
        "split_ratio": split_ratio,
        "metrics_ks": metrics_ks,
        "metrics_relevant_top_n": metrics_relevant_top_n,
        "metrics_map_k": metrics_map_k,
        "metrics_gain": metrics_gain,
        "resume_training": resume_training,
        "profile_timing": profile_timing,
        "profile_first_n_epochs": profile_first_n_epochs,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
        "prefetch_factor": prefetch_factor,
        "image_embedding_mode": image_embedding_mode,
    }
)

train_dataloader, test_dataloader = get_dataloaders(train_dataset, test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers, prefetch_factor=prefetch_factor)
train_model(model, train_dataloader, test_dataloader, device, effective_num_epochs, margin, lr, use_wandb=True, test_catalogues=test_catalogues, train_catalogues=train_catalogues, checkpoint_dir=CHECKPOINT_DIR, resume_checkpoint_path=resume_checkpoint_path, distance_metric=distance_metric, profile_timing=profile_timing, profile_first_n_epochs=profile_first_n_epochs, ranking_transform=ranking_transform, loss_metric=loss_metric, gain=metrics_gain, use_rank_weighted_loss=use_rank_weighted_loss, rank_weight_alpha=rank_weight_alpha, test_loader_excl=test_dataloader_excl, test_loader_fair=test_dataloader_fair, test_catalogues_excl=test_catalogues_excl if is_mixed_catalogue else None, test_catalogues_fair=test_catalogues_fair if is_mixed_catalogue else None, metadata=cbir_metadata)

wandb.finish()
