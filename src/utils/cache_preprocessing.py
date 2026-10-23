"""Image tensor cache, family-aware paths, preprocessing, and cache checks."""

import hashlib
import os
import pickle
import re
from pathlib import Path
from typing import Optional

import torch

from utils.catalogue_utils import normalize_catalogue_type
from utils.image_preprocessing import transform_fn


def extract_catalogue_type_from_image_path(image_path: str) -> Optional[str]:
    """Infer catalogue type from any parent folder matching Excl_<id> or Fair_<id>."""
    current_path = os.path.dirname(image_path)
    while current_path and current_path != os.path.dirname(current_path):
        folder = os.path.basename(current_path)
        match = re.match(r"^(Excl|Fair)_\d+$", folder)
        if match:
            return match.group(1)
        current_path = os.path.dirname(current_path)
    return None


def get_cache_path(image_path: str, cache_dir: str) -> str:
    """Generate a unique cache file path using the image path and catalogue family."""
    catalogue_type = extract_catalogue_type_from_image_path(image_path)
    if catalogue_type is None:
        raise ValueError(f"Unable to infer catalogue type from image path: {image_path}")

    # Separate families in the directory tree and hash the full path so equal
    # filenames from different catalogues cannot collide.
    path_hash = hashlib.md5(image_path.encode()).hexdigest()
    return os.path.join(cache_dir, catalogue_type, f"{path_hash}.pkl")


def preprocess_and_cache_dataset(
    nas_dataset_path: str,
    cache_dir: str = "/nas-ctm01/homes/dsviana/retrieval/cache",
    verbose: bool = False,
    catalogue_type: Optional[str] = None,
) -> int:
    """Preprocess dataset images and cache their tensors."""
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    selected_type = normalize_catalogue_type(catalogue_type)
    images_processed = 0
    catalogues_found = 0

    if verbose:
        print(f"Scanning dataset at: {nas_dataset_path}")

    for catalogue_folder in sorted(os.listdir(nas_dataset_path)):
        catalogue_path = os.path.join(nas_dataset_path, catalogue_folder)
        if not os.path.isdir(catalogue_path):
            continue
        if not re.match(r"^(Excl|Fair)_\d+$", catalogue_folder):
            continue

        folder_type = catalogue_folder.split("_", 1)[0]
        if selected_type not in {None, "Mixed"} and folder_type != selected_type:
            continue

        catalogues_found += 1
        if verbose:
            print(f"Processing catalogue: {catalogue_folder}")

        for image_file in sorted(os.listdir(catalogue_path)):
            image_path = os.path.join(catalogue_path, image_file)
            if not os.path.isfile(image_path):
                continue
            if not image_file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif")):
                continue

            cache_path = get_cache_path(image_path, cache_dir)
            Path(os.path.dirname(cache_path)).mkdir(parents=True, exist_ok=True)
            if os.path.exists(cache_path):
                images_processed += 1
                continue

            try:
                tensor = transform_fn(image_path)
                with open(cache_path, "wb") as cache_file:
                    pickle.dump(tensor, cache_file)
                images_processed += 1
                if verbose and images_processed % 100 == 0:
                    print(f"  Cached {images_processed} images...")
            except Exception as error:
                print(f"Error processing {image_path}: {error}")
                continue

    if verbose:
        print("\nCache preprocessing complete!")
        print(f"Catalogues found: {catalogues_found}")
        print(f"Images cached: {images_processed}")
        print(f"Cache directory: {cache_dir}")

    return images_processed


def get_cached_image(image_path: str, cache_dir: str) -> Optional[torch.Tensor]:
    """Load a cached tensor, or return None when it is unavailable."""
    cache_path = get_cache_path(image_path, cache_dir)
    if not os.path.exists(cache_path):
        return None

    try:
        with open(cache_path, "rb") as cache_file:
            return pickle.load(cache_file)
    except Exception as error:
        print(f"Error loading cache from {cache_path}: {error}")
        return None


def save_cached_image(image_path: str, cache_dir: str, tensor: torch.Tensor) -> bool:
    """Persist a preprocessed tensor into the cache for one image path."""
    try:
        cache_path = get_cache_path(image_path, cache_dir)
        Path(os.path.dirname(cache_path)).mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as cache_file:
            pickle.dump(tensor, cache_file)
        return True
    except Exception as error:
        print(f"Error writing cache for {image_path}: {error}")
        return False


def get_or_create_cached_image(
    image_path: str,
    cache_dir: str,
    target_size: int = 224,
) -> Optional[torch.Tensor]:
    """Load a cached tensor, preprocessing and caching it on demand when absent."""
    cached_tensor = get_cached_image(image_path, cache_dir)
    if cached_tensor is not None:
        return cached_tensor
    if not os.path.exists(image_path):
        print(f"Image not found, cannot cache: {image_path}")
        return None

    try:
        tensor = transform_fn(image_path, target_size=target_size)
    except Exception as error:
        print(f"Error preprocessing {image_path}: {error}")
        return None

    save_cached_image(image_path, cache_dir, tensor)
    return tensor


def is_cache_available(
    nas_dataset_path: str,
    cache_dir: str,
    catalogue_type: Optional[str] = None,
) -> bool:
    """Check whether the cache contains approximately all dataset images."""
    if not os.path.exists(cache_dir):
        return False

    selected_type = normalize_catalogue_type(catalogue_type)
    if selected_type not in {None, "Mixed"}:
        cache_root = os.path.join(cache_dir, selected_type)
        if not os.path.exists(cache_root):
            return False
        cache_files = len([name for name in os.listdir(cache_root) if name.endswith(".pkl")])
    else:
        cache_files = 0
        for family in ("Excl", "Fair"):
            family_root = os.path.join(cache_dir, family)
            if os.path.exists(family_root):
                cache_files += len([name for name in os.listdir(family_root) if name.endswith(".pkl")])

    if cache_files == 0:
        return False

    image_count = 0
    for catalogue_folder in os.listdir(nas_dataset_path):
        catalogue_path = os.path.join(nas_dataset_path, catalogue_folder)
        if not os.path.isdir(catalogue_path) or not re.match(r"^(Excl|Fair)_\d+$", catalogue_folder):
            continue
        folder_type = catalogue_folder.split("_", 1)[0]
        if selected_type not in {None, "Mixed"} and folder_type != selected_type:
            continue
        image_count += sum(
            1
            for image_file in os.listdir(catalogue_path)
            if image_file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif"))
        )

    return cache_files >= image_count * 0.95