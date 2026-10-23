"""Configure and run pointwise collaborative-filtering training."""

import os
import torch
import wandb
from models.collaborative import FM_model, NCF_model, NN_model
from models.image_encoders import GC_ViT_224
from utils.cache_preprocessing import preprocess_and_cache_dataset, is_cache_available
from utils.image_preprocessing import transform_fn
from utils.catalogue_utils import normalize_catalogue_type
from cf import (
	get_cf_dataloaders,
	prepare_cf_datasets,
	train_cf_model,
	CFEncodedAtRuntimeScoringWrapper,
)
from utils.image_embedding import (
	IMAGE_EMBEDDING_MODE_ENCODED_AT_RUNTIME,
	IMAGE_EMBEDDING_MODE_PRECOMPUTED,
	normalize_image_embedding_mode,
)
from configuration import PROJECT_ROOT as PROJECT_ROOT_PATH, load_pipeline_config


config = load_pipeline_config("cf")
data_config = config["data"]
training_config = config["training"]
model_settings = config["model"]
PROJECT_ROOT = str(PROJECT_ROOT_PATH)
split_ratio    = data_config["split_ratio"]
seed           = data_config["seed"]
catalogue_type = data_config["catalogue_type"]
num_epochs     = training_config["num_epochs"]
batch_size     = training_config["batch_size"]
lr             = training_config["learning_rate"]
weight_decay   = training_config["weight_decay"]
device         = "cuda" if torch.cuda.is_available() else "cpu"
gain           = training_config["gain"]
model_variant = model_settings["variant"]
image_embedding_mode = training_config["image_embedding_mode"]

# Shared model architecture / representation dims
fusion_dim   = model_settings["fusion_dim"]
tabular_dim  = model_settings["tabular_dim"]
tabular_encoder_type = model_settings["tabular_encoder_type"]
nhead        = model_settings["nhead"]
num_layers   = model_settings["num_layers"]
dropout      = model_settings["dropout"]

# Score-model hyperparameters (used by FM/NN/NCF)
projection_dim = model_settings["projection_dim"]
fm_k = model_settings["fm_k"]
lay_1, lay_2, lay_3, lay_4 = model_settings["hidden_dims"]
ncf_fusion_heads = model_settings["ncf_fusion_heads"]
ncf_fusion_layers = model_settings["ncf_fusion_layers"]
ncf_fusion_dim = model_settings["ncf_fusion_dim"]
ncf_fusion_mode = model_settings["ncf_fusion_mode"]

NAS_DATASET_PATH = data_config["dataset_path"]
CACHE_DIR        = os.path.join(PROJECT_ROOT, config["paths"]["cache_dir"])

normalized_catalogue_type = normalize_catalogue_type(catalogue_type)
normalized_model_variant = model_variant.strip().upper()
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, config["paths"]["checkpoint_dir"], "cf", normalized_catalogue_type, normalized_model_variant)
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

if not is_cache_available(NAS_DATASET_PATH, CACHE_DIR, catalogue_type=catalogue_type):
	preprocess_and_cache_dataset(
		nas_dataset_path=NAS_DATASET_PATH,
		cache_dir=CACHE_DIR,
		verbose=False,
		catalogue_type=catalogue_type,
	)

image_encoder = GC_ViT_224()

# Dataset preparation owns the catalogue split, training-only tabular scaling,
# and either precomputed or runtime image encoding, as selected in YAML.
bundle = prepare_cf_datasets(
	data_dir=NAS_DATASET_PATH,
	image_encoder=image_encoder,
	split_ratio=split_ratio,
	seed=seed,
	catalogue_type=catalogue_type,
	device=device,
	image_transform=transform_fn,
	cache_dir=CACHE_DIR,
	use_cache=True,
	normalize_tabular=training_config["normalize_tabular"],
	model_variant=model_variant,
	image_embedding_mode=image_embedding_mode,
)

print(f"Train triplets: {len(bundle.train_dataset)}")
print(f"Test  triplets: {len(bundle.test_dataset)}")
print(f"Image embedding dim: {bundle.image_embedding_dim}")

train_loader, test_loader = get_cf_dataloaders(
	bundle.train_dataset,
	bundle.test_dataset,
	batch_size=batch_size,
)

normalized_model_variant = model_variant.strip().upper()
model_config = {
	"device": device,
	"dropout": dropout,
	"image_embedding_dim": bundle.image_embedding_dim,
	"projection_dim": projection_dim,
	"k": fm_k,
	"lay_1": lay_1,
	"lay_2": lay_2,
	"lay_3": lay_3,
	"lay_4": lay_4,
	"tabular_encoder_dim": tabular_dim,
	"tabular_nhead": nhead,
	"tabular_num_layers": num_layers,
	"tabular_encoder_type": tabular_encoder_type,
	"ncf_fusion_dim": ncf_fusion_dim,
	"ncf_fusion_heads": ncf_fusion_heads,
	"ncf_fusion_layers": ncf_fusion_layers,
	"ncf_dropout": dropout,
}
if normalized_model_variant == "FM":
	base_model = FM_model(model_config)
elif normalized_model_variant == "NN":
	base_model = NN_model(model_config)
elif normalized_model_variant == "NCF":
	fm_model = FM_model(model_config)
	nn_model = NN_model(model_config)
	base_model = NCF_model(
		model_config,
		nn_model=nn_model,
		fm_model=fm_model,
		fusion_mode=ncf_fusion_mode,
	)
else:
	raise ValueError("Invalid model_variant. Use one of: 'FM', 'NN', 'NCF'.")

if image_embedding_mode == IMAGE_EMBEDDING_MODE_ENCODED_AT_RUNTIME:
	model = CFEncodedAtRuntimeScoringWrapper(base_model, image_encoder)
else:
	model = base_model
print(f"Model: {model.__class__.__name__}  variant={model_variant}")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

wandb.init(
	project="cf-learning",
	config={
		"dataset":          "CINDERELLA",
		"model":            model.__class__.__name__,
		"model_variant":    model_variant,
		"num_epochs":       num_epochs,
		"batch_size":       batch_size,
		"learning_rate":    lr,
		"weight_decay":     weight_decay,
		"split_ratio":      split_ratio,
		"seed":             seed,
		"catalogue_type":   catalogue_type,
		"device":           device,
		"gain":             gain,
		"fusion_dim":       fusion_dim,
		"tabular_dim":      tabular_dim,
		"nhead":            nhead,
		"num_layers":       num_layers,
		"dropout":          dropout,
		"image_embedding_dim": bundle.image_embedding_dim,
		"projection_dim": projection_dim,
		"fm_k": fm_k,
		"lay_1": lay_1,
		"lay_2": lay_2,
		"lay_3": lay_3,
		"lay_4": lay_4,
		"ncf_fusion_dim": ncf_fusion_dim,
		"ncf_fusion_heads": ncf_fusion_heads,
		"ncf_fusion_layers": ncf_fusion_layers,
		"ncf_fusion_mode": ncf_fusion_mode,
		"resume_training":  resume_training,
		"image_embedding_mode": image_embedding_mode,
	},
)

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
