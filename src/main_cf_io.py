"""Configure and run the image-only collaborative-filtering ablation."""

import os

import torch
import wandb
from models.image_encoders import GC_ViT_224
from utils.cache_preprocessing import is_cache_available, preprocess_and_cache_dataset
from utils.image_preprocessing import transform_fn
from utils.catalogue_utils import normalize_catalogue_type
from cf import (
    CFEncodedAtRuntimeScoringWrapper,
    get_cf_dataloaders,
    prepare_cf_datasets,
    train_cf_model,
)
from utils.image_embedding import (
    IMAGE_EMBEDDING_MODE_ENCODED_AT_RUNTIME,
    IMAGE_EMBEDDING_MODE_PRECOMPUTED,
    normalize_image_embedding_mode,
)
from cf_io import build_image_only_cf_model, prepare_cf_io_datasets
from configuration import PROJECT_ROOT as PROJECT_ROOT_PATH, load_pipeline_config


config = load_pipeline_config("cf_io")
data_config = config["data"]
training_config = config["training"]
model_settings = config["model"]
loader_config = config["loader"]
split_ratio = data_config["split_ratio"]
seed = data_config["seed"]
catalogue_type = data_config["catalogue_type"]
num_epochs = training_config["num_epochs"]
batch_size = training_config["batch_size"]
lr = training_config["learning_rate"]
weight_decay = training_config["weight_decay"]
device = "cuda" if torch.cuda.is_available() else "cpu"
gain = training_config["gain"]
model_variant = model_settings["variant"]
image_embedding_mode = training_config["image_embedding_mode"]

projection_dim = model_settings["projection_dim"]
fm_k = model_settings["fm_k"]
lay_1, lay_2, lay_3, lay_4 = model_settings["hidden_dims"]
ncf_fusion_heads = model_settings["ncf_fusion_heads"]
ncf_fusion_layers = model_settings["ncf_fusion_layers"]
ncf_fusion_dim = model_settings["ncf_fusion_dim"]
ncf_fusion_mode = model_settings["ncf_fusion_mode"]
dropout = model_settings["dropout"]

# NAS dataset path
# Cache and checkpoint directories
PROJECT_ROOT = str(PROJECT_ROOT_PATH)
NAS_DATASET_PATH = data_config["dataset_path"]
CACHE_DIR = os.path.join(PROJECT_ROOT, config["paths"]["cache_dir"])

# Checkpoint configuration
normalized_catalogue_type = normalize_catalogue_type(catalogue_type)
normalized_model_variant = model_variant.strip().upper()
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, config["paths"]["checkpoint_dir"], "cf_io", normalized_catalogue_type, normalized_model_variant)
resume_training = training_config["resume_training"]
resume_checkpoint_path = os.path.join(CHECKPOINT_DIR, "last_checkpoint.pt") if resume_training else None

print(f"Dataset path:       {NAS_DATASET_PATH}")
print(f"Cache directory:    {CACHE_DIR}")
print(f"Checkpoint dir:     {CHECKPOINT_DIR}")
print(f"Device:             {device}")
print(f"Catalogue type:     {catalogue_type}")
print(f"Model variant:      {model_variant}")
print(f"NCF fusion mode:    {ncf_fusion_mode}")
image_embedding_mode = normalize_image_embedding_mode(image_embedding_mode)
print(f"Image emb mode:     {image_embedding_mode}")

# Ensure cache is available
if not is_cache_available(NAS_DATASET_PATH, CACHE_DIR, catalogue_type=catalogue_type):
    preprocess_and_cache_dataset(
        nas_dataset_path=NAS_DATASET_PATH,
        cache_dir=CACHE_DIR,
        verbose=False,
        catalogue_type=catalogue_type,
    )

# Encoder used to generate image embeddings
image_encoder = GC_ViT_224()

# Reuse CF's ranked interactions, but disable the tabular model branch so this
# run isolates the contribution of image features.
bundle = prepare_cf_io_datasets(
    data_dir=NAS_DATASET_PATH,
    image_encoder=image_encoder,
    split_ratio=split_ratio,
    seed=seed,
    catalogue_type=catalogue_type,
    device=device,
    image_transform=transform_fn,
    cache_dir=CACHE_DIR,
    use_cache=True,
    model_variant=model_variant,
    image_embedding_mode=image_embedding_mode,
    normalize_tabular=False,
)

print(f"Train interactions: {len(bundle.train_dataset)}")
print(f"Test  interactions: {len(bundle.test_dataset)}")
print(f"Image embedding dim: {bundle.image_embedding_dim}")

# Get dataloaders
train_loader, test_loader = get_cf_dataloaders(
    bundle.train_dataset,
    bundle.test_dataset,
    batch_size=batch_size,
    num_workers=loader_config["num_workers"],
    pin_memory=loader_config["pin_memory"],
    persistent_workers=loader_config["persistent_workers"],
    prefetch_factor=loader_config["prefetch_factor"],
)

image_embedding_dim = bundle.image_embedding_dim

# Instantiate model
model = build_image_only_cf_model(
    model_variant=model_variant,
    image_embedding_dim=image_embedding_dim,
    hidden_dims=[lay_1, lay_2, lay_3, lay_4],
    dropout=dropout,
    k=fm_k,
    fusion_dim=ncf_fusion_dim,
    fusion_heads=ncf_fusion_heads,
    fusion_layers=ncf_fusion_layers,
    fusion_dropout=dropout,
    device=device,
)

if image_embedding_mode == IMAGE_EMBEDDING_MODE_ENCODED_AT_RUNTIME:
    model = CFEncodedAtRuntimeScoringWrapper(model, image_encoder)

print(f"Model: {model.__class__.__name__}  variant={model_variant}")

# Ensure checkpoint directory exists
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Initialize weights and biases
wandb.init(
    project="cf-io-learning",
    config={
        "dataset": "CINDERELLA",
        "model": model.__class__.__name__,
        "model_variant": model_variant,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "split_ratio": split_ratio,
        "seed": seed,
        "catalogue_type": catalogue_type,
        "device": device,
        "gain": gain,
        "projection_dim": projection_dim,
        "fm_k": fm_k,
        "lay_1": lay_1,
        "lay_2": lay_2,
        "lay_3": lay_3,
        "lay_4": lay_4,
        "dropout": dropout,
        "ncf_fusion_dim": ncf_fusion_dim,
        "ncf_fusion_heads": ncf_fusion_heads,
        "ncf_fusion_layers": ncf_fusion_layers,
        "ncf_fusion_mode": ncf_fusion_mode,
        "resume_training": resume_training,
        "image_embedding_mode": image_embedding_mode,
    },
)

# Train model
model = train_cf_model(
    model=model,
    train_loader=train_loader,
    test_loader=test_loader,
    train_catalogues=bundle.train_catalogues,
    test_catalogues=bundle.test_catalogues,
    metadata=bundle.metadata,
    device=device,
    num_epochs=num_epochs,
    lr=lr,
    weight_decay=weight_decay,
    use_wandb=True,
    checkpoint_dir=CHECKPOINT_DIR,
    resume_checkpoint_path=resume_checkpoint_path,
    gain=gain,
    tabular_stats=bundle.tabular_stats,
    model_variant=model_variant,
)

wandb.finish()
