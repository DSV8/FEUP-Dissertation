"""Evaluate pretrained GC-ViT without task-specific fine-tuning as a baseline."""
import os
import torch
from cbir import (
    build_ranking_transform,
    evaluate_catalogue_metrics,
    evaluate_triplets,
    get_dataloaders,
    prepare_datasets_from_catalogues,
)
from models.image_encoders import GC_ViT_224
from utils.catalogue_utils import build_catalogues, split_catalogues_canonical
from configuration import PROJECT_ROOT as PROJECT_ROOT_PATH, load_pipeline_config
import json
from datetime import datetime

config = load_pipeline_config("baseline_gcvit")
data_config = config["data"]
evaluation_config = config["evaluation"]
loader_config = config["loader"]
PROJECT_ROOT = str(PROJECT_ROOT_PATH)
NAS_DATASET_PATH = data_config["dataset_path"]
CACHE_DIR = os.path.join(PROJECT_ROOT, config["paths"]["cache_dir"])

device = "cuda" if torch.cuda.is_available() else "cpu"
catalogue_type = data_config["catalogue_type"]
split_ratio = data_config["split_ratio"]
seed = data_config["seed"]
batch_size = evaluation_config["batch_size"]
num_workers = loader_config["num_workers"]
pin_memory = loader_config["pin_memory"]
persistent_workers = loader_config["persistent_workers"]
prefetch_factor = loader_config["prefetch_factor"]

# Evaluation configuration
distance_metric = evaluation_config["distance_metric"]
metrics_ks = tuple(evaluation_config["metrics_ks"])
metrics_relevant_top_n = evaluation_config["metrics_relevant_top_n"]
metrics_map_k = evaluation_config["metrics_map_k"]
metrics_gain = evaluation_config["metrics_gain"]

print("=" * 80)
print("BASELINE GC_ViT EVALUATION (Untrained Model)")
print("=" * 80)
print(f"Device: {device}")
print(f"Distance metric: {distance_metric}")
print(f"Catalogue type: {catalogue_type}")
print(f"Timestamp: {datetime.now().isoformat()}")
print()

# Load untrained model
print("Loading pretrained GC-ViT baseline (no task-specific fine-tuning)...")
model = GC_ViT_224()
model.to(device)
model.eval()
print(f"Model: {model.__class__.__name__}")
print()

# Build the canonical split once for both triplet and ranking evaluation.
print("Building catalogues...")
catalogues = build_catalogues(NAS_DATASET_PATH)
train_catalogues, test_catalogues = split_catalogues_canonical(
    catalogues,
    split_ratio=split_ratio,
    seed=seed,
    catalogue_type=catalogue_type,
)
print(f"Train catalogues: {len(train_catalogues)}")
print(f"Test catalogues: {len(test_catalogues)}")
print()

# Prepare datasets and dataloaders from the same canonical split.
print("Preparing datasets...")
train_dataset, test_dataset = prepare_datasets_from_catalogues(
    train_catalogues,
    test_catalogues,
    cache_dir=CACHE_DIR,
    use_cache=True,
)
print(f"Train dataset size: {len(train_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")
print()

train_loader, test_loader = get_dataloaders(
    train_dataset,
    test_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    pin_memory=pin_memory,
    persistent_workers=persistent_workers,
    prefetch_factor=prefetch_factor
)

# Build ranking transform
ranking_transform = build_ranking_transform(CACHE_DIR, use_cache=True)

# Evaluate Triplet Loss Accuracy
print("Evaluating triplet loss accuracy...")
print("-" * 80)
train_triplet_acc = evaluate_triplets(model, train_loader, device, distance_metric=distance_metric)
test_triplet_acc = evaluate_triplets(model, test_loader, device, distance_metric=distance_metric)
print(f"Train Triplet Accuracy: {train_triplet_acc:.4f}")
print(f"Test Triplet Accuracy:  {test_triplet_acc:.4f}")
print()

# Evaluate Ranking Metrics
print("Evaluating ranking metrics...")
print("-" * 80)
train_rank_metrics = evaluate_catalogue_metrics(
    train_catalogues,
    model,
    ranking_transform,
    device,
    ks=metrics_ks,
    relevant_top_n=metrics_relevant_top_n,
    map_k=metrics_map_k,
    distance_metric=distance_metric,
    gain=metrics_gain,
)

test_rank_metrics = evaluate_catalogue_metrics(
    test_catalogues,
    model,
    ranking_transform,
    device,
    ks=metrics_ks,
    relevant_top_n=metrics_relevant_top_n,
    map_k=metrics_map_k,
    distance_metric=distance_metric,
    gain=metrics_gain,
)

print("Train Ranking Metrics:")
for metric_name, value in sorted(train_rank_metrics.items()):
    print(f"  {metric_name}: {value:.4f}")
print()

print("Test Ranking Metrics:")
for metric_name, value in sorted(test_rank_metrics.items()):
    print(f"  {metric_name}: {value:.4f}")
print()

# Summary
print("=" * 80)
print("BASELINE RESULTS SUMMARY")
print("=" * 80)
results = {
    "model": "GC_ViT_224",
    "state": "pretrained (no task-specific fine-tuning)",
    "timestamp": datetime.now().isoformat(),
    "configuration": {
        "device": device,
        "distance_metric": distance_metric,
        "catalogue_type": catalogue_type,
        "split_strategy": "canonical_stratified_by_family",
        "split_ratio": split_ratio,
        "seed": seed,
        "batch_size": batch_size,
    },
    "triplet_accuracy": {
        "train": float(train_triplet_acc),
        "test": float(test_triplet_acc),
    },
    "train_ranking_metrics": {k: float(v) for k, v in train_rank_metrics.items()},
    "test_ranking_metrics": {k: float(v) for k, v in test_rank_metrics.items()},
}

# Print summary
print(json.dumps(results, indent=2))

# Save results to file
output_file = os.path.join(PROJECT_ROOT, "baseline_gcvit_results.json")
with open(output_file, "w") as f:
    json.dump(results, f, indent=2)
print()
print(f"Results saved to: {output_file}")
