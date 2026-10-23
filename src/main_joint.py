"""Configure the shared-data, two-objective CF/MM experiment."""

import os
from typing import Callable, Optional

import torch
import wandb

from models.image_encoders import GC_ViT_224
from models.joint import JointCFMMModel
from utils.cache_preprocessing import is_cache_available, preprocess_and_cache_dataset
from utils.image_preprocessing import transform_fn
from utils.catalogue_utils import normalize_catalogue_type
from utils.image_embedding import (
    IMAGE_EMBEDDING_MODE_ENCODED_AT_RUNTIME,
    IMAGE_EMBEDDING_MODE_PRECOMPUTED,
    normalize_image_embedding_mode,
)
from joint import (
    JointCFMMBundle,
    prepare_joint_cf_mm_datasets,
    train_joint_cf_mm_model,
)
from configuration import PROJECT_ROOT as PROJECT_ROOT_PATH, load_pipeline_config


CONFIG = load_pipeline_config("joint")
DATA_CONFIG = CONFIG["data"]
TRAINING_CONFIG = CONFIG["training"]
MODEL_CONFIG = CONFIG["model"]
LOADER_CONFIG = CONFIG["loader"]
NAS_DATASET_PATH = DATA_CONFIG["dataset_path"]
PROJECT_ROOT = str(PROJECT_ROOT_PATH)
CACHE_DIR = os.path.join(PROJECT_ROOT, CONFIG["paths"]["cache_dir"])


def build_bundle(
    data_dir: str,
    image_encoder: torch.nn.Module,
    image_transform: Callable[[str], torch.Tensor],
    device: str,
    split_ratio: float,
    seed: int,
    catalogue_type: str,
    cache_dir: str,
    use_cache: bool,
    normalize_tabular: bool,
    image_embedding_mode: str,
    tabular_dim: int,
    tabular_embedding_dim: int,
    fusion_embedding_dim: int,
    tabular_nhead: int,
    tabular_num_layers: int,
    fusion_nhead: int,
    fusion_num_layers: int,
    fusion_mode: str,
) -> JointCFMMBundle:
    """Build both task datasets from one catalogue split and feature mapping."""
    return prepare_joint_cf_mm_datasets(
        data_dir=data_dir,
        image_encoder=image_encoder,
        split_ratio=split_ratio,
        seed=seed,
        catalogue_type=catalogue_type,
        device=device,
        image_transform=image_transform,
        cache_dir=cache_dir,
        use_cache=use_cache,
        normalize_tabular=normalize_tabular,
        image_embedding_mode=image_embedding_mode,
        tabular_dim=tabular_dim,
        tabular_embedding_dim=tabular_embedding_dim,
        fusion_embedding_dim=fusion_embedding_dim,
        tabular_nhead=tabular_nhead,
        tabular_num_layers=tabular_num_layers,
        fusion_nhead=fusion_nhead,
        fusion_num_layers=fusion_num_layers,
        fusion_mode=fusion_mode,
    )


def build_model(
    image_embedding_dim: int,
    tabular_dim: int,
    hidden_dim: int,
    dropout: float,
    tabular_embedding_dim: int,
    fusion_embedding_dim: int,
    tabular_nhead: int,
    tabular_num_layers: int,
    tabular_encoder_type: str,
    fusion_nhead: int,
    fusion_num_layers: int,
    device: str,
    cf_model_variant: str,
    cf_projection_dim: int,
    cf_fm_k: int,
    cf_lay_1: int,
    cf_lay_2: int,
    cf_lay_3: int,
    cf_lay_4: int,
    cf_ncf_fusion_heads: int,
    cf_ncf_fusion_layers: int,
    cf_ncf_fusion_dim: int,
    cf_ncf_dropout: float,
    fusion_mode: str,
    cf_use_fused_input: bool,
    freeze_image: bool,
    freeze_tabular: bool,
    image_encoder: Optional[torch.nn.Module] = None,
) -> JointCFMMModel:
    return JointCFMMModel(
        image_embedding_dim=image_embedding_dim,
        tabular_dim=tabular_dim,
        tabular_embedding_dim=tabular_embedding_dim,
        fusion_embedding_dim=fusion_embedding_dim,
        tabular_nhead=tabular_nhead,
        tabular_num_layers=tabular_num_layers,
        tabular_encoder_type=tabular_encoder_type,
        fusion_nhead=fusion_nhead,
        fusion_num_layers=fusion_num_layers,
        hidden_dim=hidden_dim,
        dropout=dropout,
        image_encoder=image_encoder,
        device=device,
        cf_model_variant=cf_model_variant,
        cf_projection_dim=cf_projection_dim,
        cf_fm_k=cf_fm_k,
        cf_lay_1=cf_lay_1,
        cf_lay_2=cf_lay_2,
        cf_lay_3=cf_lay_3,
        cf_lay_4=cf_lay_4,
        cf_ncf_fusion_heads=cf_ncf_fusion_heads,
        cf_ncf_fusion_layers=cf_ncf_fusion_layers,
        cf_ncf_fusion_dim=cf_ncf_fusion_dim,
        cf_ncf_dropout=cf_ncf_dropout,
        fusion_mode=fusion_mode,
        cf_use_fused_input=cf_use_fused_input,
        freeze_image=freeze_image,
        freeze_tabular=freeze_tabular,
    )


def train(
    bundle: JointCFMMBundle,
    model: JointCFMMModel,
    device: str,
    catalogue_type: str,
    num_epochs: int,
    batch_size_cf: int,
    batch_size_mm: int,
    lr: float,
    weight_decay: float,
    triplet_margin: float,
    distance_metric: str,
    gain: str,
    loss_metric: str,
    use_uncertainty_weighting: bool,
    cf_loss_weight: float,
    mm_loss_weight: float,
    use_rank_weighted_triplet: bool,
    rank_weight_alpha: float,
    checkpoint_dir: str,
    resume_checkpoint_path: Optional[str],
    use_wandb: bool,
) -> JointCFMMModel:
    from joint import get_mm_dataloaders
    from cf import get_cf_dataloaders

    # forward DataLoader perf settings from main() through train()
    # use defaults if not provided in caller via closure/outer scope
    num_workers = int(os.environ.get("NUM_WORKERS", "0"))
    pin_memory = bool(int(os.environ.get("PIN_MEMORY", "0")))
    persistent_workers = bool(int(os.environ.get("PERSISTENT_WORKERS", "0")))
    prefetch_factor = int(os.environ.get("PREFETCH_FACTOR", "2"))

    # The two tasks retain separate loaders because their supervision and
    # batch sizes differ, even though they update the same model.
    cf_train_loader, cf_test_loader = get_cf_dataloaders(
        bundle.cf_train_dataset,
        bundle.cf_test_dataset,
        batch_size=batch_size_cf,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    mm_train_loader, mm_test_loader = get_mm_dataloaders(
        bundle.mm_train_dataset,
        bundle.mm_test_dataset,
        batch_size=batch_size_mm,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )

    os.makedirs(checkpoint_dir, exist_ok=True)
    if use_wandb:
        wandb.init(
            project="joint-cf-mm-multitask",
            config={
                "dataset": "CINDERELLA",
                "model": model.__class__.__name__,
                "cf_model_variant": getattr(model, "cf_model_variant", "NCF"),
                "catalogue_type": normalize_catalogue_type(catalogue_type),
                "num_epochs": num_epochs,
                "batch_size_cf": batch_size_cf,
                "batch_size_mm": batch_size_mm,
                "learning_rate": lr,
                "weight_decay": weight_decay,
                "triplet_margin": triplet_margin,
                "distance_metric": distance_metric,
                "gain": gain,
                "loss_metric": loss_metric,
                "image_embedding_dim": bundle.image_embedding_dim,
                "use_uncertainty_weighting": use_uncertainty_weighting,
                "cf_loss_weight": cf_loss_weight,
                "mm_loss_weight": mm_loss_weight,
                "use_rank_weighted_triplet": use_rank_weighted_triplet,
                "rank_weight_alpha": rank_weight_alpha,
                "freeze_image": bool(getattr(model, "freeze_image", False)),
                "freeze_tabular": bool(getattr(model, "freeze_tabular", False)),
            },
        )

    trained_model = train_joint_cf_mm_model(
        model=model,
        cf_train_loader=cf_train_loader,
        cf_test_loader=cf_test_loader,
        mm_train_loader=mm_train_loader,
        mm_test_loader=mm_test_loader,
        train_catalogues=bundle.train_catalogues,
        test_catalogues=bundle.test_catalogues,
        metadata=bundle.metadata,
        device=device,
        num_epochs=num_epochs,
        lr=lr,
        weight_decay=weight_decay,
        cf_loss_weight=cf_loss_weight,
        mm_loss_weight=mm_loss_weight,
        use_uncertainty_weighting=use_uncertainty_weighting,
        triplet_margin=triplet_margin,
        distance_metric=distance_metric,
        gain=gain,
        loss_metric=loss_metric,
        use_rank_weighted_triplet=use_rank_weighted_triplet,
        rank_weight_alpha=rank_weight_alpha,
        use_wandb=use_wandb,
        checkpoint_dir=checkpoint_dir,
        resume_checkpoint_path=resume_checkpoint_path,
        tabular_stats=bundle.tabular_stats,
    )

    if use_wandb:
        wandb.finish()

    return trained_model


def main() -> None:
    split_ratio = DATA_CONFIG["split_ratio"]
    seed = DATA_CONFIG["seed"]
    catalogue_type = DATA_CONFIG["catalogue_type"]
    num_epochs = TRAINING_CONFIG["num_epochs"]
    batch_size_cf = TRAINING_CONFIG["batch_size_cf"]
    batch_size_mm = TRAINING_CONFIG["batch_size_mm"]
    lr = TRAINING_CONFIG["learning_rate"]
    weight_decay = TRAINING_CONFIG["weight_decay"]
    triplet_margin = TRAINING_CONFIG["triplet_margin"]
    distance_metric = TRAINING_CONFIG["distance_metric"]
    gain = TRAINING_CONFIG["gain"]
    loss_metric = TRAINING_CONFIG["loss_metric"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    use_uncertainty_weighting = TRAINING_CONFIG["use_uncertainty_weighting"]
    cf_loss_weight = TRAINING_CONFIG["cf_loss_weight"]
    mm_loss_weight = TRAINING_CONFIG["mm_loss_weight"]
    use_rank_weighted_triplet = TRAINING_CONFIG["use_rank_weighted_triplet"]
    rank_weight_alpha = TRAINING_CONFIG["rank_weight_alpha"]
    freeze_image = TRAINING_CONFIG["freeze_image"]
    freeze_tabular = TRAINING_CONFIG["freeze_tabular"]

    hidden_dim = MODEL_CONFIG["hidden_dim"]
    dropout = MODEL_CONFIG["dropout"]
    tabular_embedding_dim = MODEL_CONFIG["tabular_embedding_dim"]
    fusion_embedding_dim = MODEL_CONFIG["fusion_embedding_dim"]
    tabular_nhead = MODEL_CONFIG["tabular_nhead"]
    tabular_num_layers = MODEL_CONFIG["tabular_num_layers"]
    tabular_encoder_type = MODEL_CONFIG["tabular_encoder_type"]
    fusion_nhead = MODEL_CONFIG["fusion_nhead"]
    fusion_num_layers = MODEL_CONFIG["fusion_num_layers"]
    image_embedding_mode = TRAINING_CONFIG["image_embedding_mode"]
    image_embedding_mode = normalize_image_embedding_mode(image_embedding_mode)

    # DataLoader / CUDA environment and perf tuning
    num_workers = LOADER_CONFIG["num_workers"]
    pin_memory = LOADER_CONFIG["pin_memory"]
    persistent_workers = LOADER_CONFIG["persistent_workers"]
    prefetch_factor = LOADER_CONFIG["prefetch_factor"]
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,garbage_collection_threshold:0.9,expandable_segments:True"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
    os.environ["NUM_WORKERS"] = str(num_workers)
    os.environ["PIN_MEMORY"] = str(int(pin_memory))
    os.environ["PERSISTENT_WORKERS"] = str(int(persistent_workers))
    os.environ["PREFETCH_FACTOR"] = str(prefetch_factor)

    # CF branch variant used inside the joint multitask pipeline.
    # Options: "FM", "NN", "NCF"
    cf_model_variant = MODEL_CONFIG["cf_model_variant"]

    # FM/NN/NCF hyperparameters
    cf_projection_dim = MODEL_CONFIG["cf_projection_dim"]
    cf_fm_k = MODEL_CONFIG["cf_fm_k"]
    cf_lay_1, cf_lay_2, cf_lay_3, cf_lay_4 = MODEL_CONFIG["cf_hidden_dims"]
    cf_ncf_fusion_heads = MODEL_CONFIG["cf_ncf_fusion_heads"]
    cf_ncf_fusion_layers = MODEL_CONFIG["cf_ncf_fusion_layers"]
    cf_ncf_fusion_dim = MODEL_CONFIG["cf_ncf_fusion_dim"]
    cf_ncf_dropout = MODEL_CONFIG["cf_ncf_dropout"]
    fusion_mode = MODEL_CONFIG["fusion_mode"]

    resume_training = TRAINING_CONFIG["resume_training"]
    normalized_catalogue_type = normalize_catalogue_type(catalogue_type)
    include_catalogue_type_feature = normalized_catalogue_type == "Mixed"
    model_tabular_dim = 6 if include_catalogue_type_feature else 5
    checkpoint_dir = os.path.join(PROJECT_ROOT, CONFIG["paths"]["checkpoint_dir"], "joint", normalized_catalogue_type)
    resume_checkpoint_path = os.path.join(checkpoint_dir, "last_checkpoint.pt") if resume_training else None

    print(f"Dataset path:               {NAS_DATASET_PATH}")
    print(f"Cache directory:            {CACHE_DIR}")
    print(f"Checkpoint directory:       {checkpoint_dir}")
    print(f"Device:                     {device}")
    print(f"Catalogue type:             {normalized_catalogue_type}")
    print(f"Use uncertainty weighting:  {use_uncertainty_weighting}")
    print(f"Image embedding mode:       {image_embedding_mode}")
    print(f"Batch size CF:              {batch_size_cf}")
    print(f"Batch size MM:              {batch_size_mm}")
    print(f"CF model variant:           {cf_model_variant}")
    print(f"Fusion mode:                {fusion_mode}")
    print(f"MM distance metric:         {distance_metric}")
    print(f"Ranking gain:               {gain}")
    print(f"MM loss metric:             {loss_metric}")
    print(f"Freeze image:               {freeze_image}")
    print(f"Freeze tabular:             {freeze_tabular}")

    if not is_cache_available(NAS_DATASET_PATH, CACHE_DIR, catalogue_type=catalogue_type):
        preprocess_and_cache_dataset(
            nas_dataset_path=NAS_DATASET_PATH,
            cache_dir=CACHE_DIR,
            verbose=False,
            catalogue_type=catalogue_type,
        )

    image_encoder = GC_ViT_224()
    image_transform = transform_fn
    bundle = build_bundle(
        data_dir=NAS_DATASET_PATH,
        image_encoder=image_encoder,
        image_transform=image_transform,
        device=device,
        split_ratio=split_ratio,
        seed=seed,
        catalogue_type=catalogue_type,
        cache_dir=CACHE_DIR,
        use_cache=True,
        normalize_tabular=True,
        image_embedding_mode=image_embedding_mode,
        tabular_dim=model_tabular_dim,
        tabular_embedding_dim=tabular_embedding_dim,
        fusion_embedding_dim=fusion_embedding_dim,
        tabular_nhead=tabular_nhead,
        tabular_num_layers=tabular_num_layers,
        fusion_nhead=fusion_nhead,
        fusion_num_layers=fusion_num_layers,
        fusion_mode=fusion_mode,
    )

    print(f"CF samples -> train: {len(bundle.cf_train_dataset)}, test: {len(bundle.cf_test_dataset)}")
    print(f"MM triplets -> train: {len(bundle.mm_train_dataset)}, test: {len(bundle.mm_test_dataset)}")
    print(f"Image embedding dim: {bundle.image_embedding_dim}")

    include_catalogue_type_feature = bool(bundle.metadata.get("include_catalogue_type_feature", False))
    model_tabular_dim = 6 if include_catalogue_type_feature else 5

    model = build_model(
        image_embedding_dim=bundle.image_embedding_dim,
        tabular_dim=model_tabular_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        tabular_embedding_dim=tabular_embedding_dim,
        fusion_embedding_dim=fusion_embedding_dim,
        tabular_nhead=tabular_nhead,
        tabular_num_layers=tabular_num_layers,
        tabular_encoder_type=tabular_encoder_type,
        fusion_nhead=fusion_nhead,
        fusion_num_layers=fusion_num_layers,
        device=device,
        cf_model_variant=cf_model_variant,
        cf_projection_dim=cf_projection_dim,
        cf_fm_k=cf_fm_k,
        cf_lay_1=cf_lay_1,
        cf_lay_2=cf_lay_2,
        cf_lay_3=cf_lay_3,
        cf_lay_4=cf_lay_4,
        cf_ncf_fusion_heads=cf_ncf_fusion_heads,
        cf_ncf_fusion_layers=cf_ncf_fusion_layers,
        cf_ncf_fusion_dim=cf_ncf_fusion_dim,
        cf_ncf_dropout=cf_ncf_dropout,
        fusion_mode=fusion_mode,
        cf_use_fused_input=True,
        freeze_image=freeze_image,
        freeze_tabular=freeze_tabular,
        image_encoder=image_encoder if image_embedding_mode == IMAGE_EMBEDDING_MODE_ENCODED_AT_RUNTIME else None,
    )
    print(f"Model architecture: {model.__class__.__name__}")

    train(
        bundle=bundle,
        model=model,
        device=device,
        catalogue_type=catalogue_type,
        num_epochs=num_epochs,
        batch_size_cf=batch_size_cf,
        batch_size_mm=batch_size_mm,
        lr=lr,
        weight_decay=weight_decay,
        triplet_margin=triplet_margin,
        distance_metric=distance_metric,
        gain=gain,
        loss_metric=loss_metric,
        use_uncertainty_weighting=use_uncertainty_weighting,
        cf_loss_weight=cf_loss_weight,
        mm_loss_weight=mm_loss_weight,
        use_rank_weighted_triplet=use_rank_weighted_triplet,
        rank_weight_alpha=rank_weight_alpha,
        checkpoint_dir=checkpoint_dir,
        resume_checkpoint_path=resume_checkpoint_path,
        use_wandb=True,
    )


if __name__ == "__main__":
    main()
