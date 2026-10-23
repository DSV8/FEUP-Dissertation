"""Catalogue discovery, filename parsing, triplet construction, and splitting."""

import os
import random
import re
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple


def normalize_catalogue_type(catalogue_type: Optional[str]) -> Optional[str]:
    """Normalize supported catalogue type aliases to canonical names."""
    if catalogue_type is None:
        return None

    normalized = catalogue_type.strip().lower()
    if normalized in {"e", "excl"}:
        return "Excl"
    if normalized in {"f", "fair"}:
        return "Fair"
    if normalized in {"m", "mixed"}:
        return "Mixed"

    raise ValueError("Invalid catalogue_type. Use one of: 'Excl', 'Fair', 'Mixed', 'E', 'F', or 'M'.")


def catalogue_type_to_id(catalogue_type: str) -> int:
    """Map catalogue type to integer id: Excl->0, Fair->1."""
    normalized = catalogue_type.strip().lower()
    if normalized in {"e", "excl"}:
        return 0
    if normalized in {"f", "fair"}:
        return 1
    raise ValueError(f"Unsupported catalogue type '{catalogue_type}'. Expected Excl/E or Fair/F.")


def parse_catalogue_folder(folder_name: str) -> Tuple[Optional[str], Optional[int]]:
    """Parse catalogue folder name to extract type and ID."""
    match = re.match(r"^(Excl|Fair)_(\d+)$", folder_name)
    if match:
        return match.group(1), int(match.group(2))
    return None, None


def infer_catalogue_type_id_from_image_path(image_path: str) -> int:
    """Infer catalogue type id from image parent folder name."""
    folder_name = os.path.basename(os.path.dirname(image_path))
    catalogue_type, _ = parse_catalogue_folder(folder_name)
    if catalogue_type is None:
        raise ValueError(f"Unable to infer catalogue type from path: {image_path}")
    return catalogue_type_to_id(catalogue_type)


@dataclass
class ImageSample:
    path: str
    image_type: str
    age: str
    height: str
    weight: str
    bra_cup: str
    bra_size: str
    rank: Optional[int]


@dataclass
class Catalogue:
    catalogue_type: str
    catalogue_id: int
    query: str
    neighbors: List[str]


def parse_filename(filename: str) -> Optional[Dict[str, object]]:
    """Parse image filename metadata for query and neighbor images."""
    img_name = os.path.splitext(filename)[0]

    query_pattern = r"^query_(.+?)_(.+?)_(.+?)_(.+?)_(.+?)$"
    query_match = re.match(query_pattern, img_name)
    if query_match:
        age, height, weight, bra_cup, bra_size = query_match.groups()
        return {
            "image_type": "query",
            "age": age,
            "height": height,
            "weight": weight,
            "bra_cup": bra_cup,
            "bra_size": bra_size,
            "rank": None,
        }

    neighbor_pattern = r"^neighbor_(.+?)_(.+?)_(.+?)_(.+?)_(.+?)_rank_(\d+)$"
    neighbor_match = re.match(neighbor_pattern, img_name)
    if neighbor_match:
        age, height, weight, bra_cup, bra_size, rank = neighbor_match.groups()
        return {
            "image_type": "neighbor",
            "age": age,
            "height": height,
            "weight": weight,
            "bra_cup": bra_cup,
            "bra_size": bra_size,
            "rank": int(rank),
        }
    return None


def build_catalogues(data_dir: str, catalogue_type: Optional[str] = None) -> List[Catalogue]:
    """Build catalogue objects from a directory of catalogue folders."""
    selected_type = normalize_catalogue_type(catalogue_type)
    catalogues: List[Catalogue] = []

    for catalogue_folder in sorted(os.listdir(data_dir)):
        catalogue_path = os.path.join(data_dir, catalogue_folder)
        if not os.path.isdir(catalogue_path):
            continue

        family, catalogue_id = parse_catalogue_folder(catalogue_folder)
        if family is None:
            continue
        if selected_type not in {None, "Mixed"} and family != selected_type:
            continue

        query_path: Optional[str] = None
        neighbor_samples: List[ImageSample] = []

        for image_file in os.listdir(catalogue_path):
            image_path = os.path.join(catalogue_path, image_file)
            if not os.path.isfile(image_path):
                continue

            metadata = parse_filename(image_file)
            if metadata is None:
                continue

            if metadata["image_type"] == "query":
                query_path = image_path
            else:
                neighbor_samples.append(
                    ImageSample(
                        path=image_path,
                        image_type=str(metadata["image_type"]),
                        age=str(metadata["age"]),
                        height=str(metadata["height"]),
                        weight=str(metadata["weight"]),
                        bra_cup=str(metadata["bra_cup"]),
                        bra_size=str(metadata["bra_size"]),
                        rank=metadata["rank"] if isinstance(metadata["rank"], int) else None,
                    )
                )

        if query_path is None or not neighbor_samples:
            continue

        neighbor_samples.sort(key=lambda sample: sample.rank)
        catalogues.append(
            Catalogue(
                catalogue_type=family,
                catalogue_id=catalogue_id,
                query=query_path,
                neighbors=[sample.path for sample in neighbor_samples],
            )
        )

    return catalogues


def collect_catalogue_image_paths(catalogues: List[Catalogue]) -> List[str]:
    """Collect unique query and neighbor image paths across catalogues."""
    paths = set()
    for catalogue in catalogues:
        paths.add(catalogue.query)
        paths.update(catalogue.neighbors)
    return sorted(paths)


def create_triplets(
    catalogues: List[Catalogue],
    include_catalogue_type_id: bool = False,
) -> List[Tuple[Any, ...]]:
    """Create query/positive/negative path tuples from catalogue rank order."""
    triplets: List[Tuple[Any, ...]] = []

    for catalogue in catalogues:
        query_path = catalogue.query
        catalogue_type_id = catalogue_type_to_id(catalogue.catalogue_type)

        for first_index, second_index in combinations(range(len(catalogue.neighbors)), 2):
            positive_path = catalogue.neighbors[first_index]
            negative_path = catalogue.neighbors[second_index]
            positive_metadata = parse_filename(os.path.basename(positive_path))
            negative_metadata = parse_filename(os.path.basename(negative_path))

            if positive_metadata is None or negative_metadata is None:
                continue
            if positive_metadata["rank"] < negative_metadata["rank"]:
                if include_catalogue_type_id:
                    triplets.append((query_path, positive_path, negative_path, catalogue_type_id))
                else:
                    triplets.append((query_path, positive_path, negative_path))

    return triplets


def split_catalogues(
    catalogues: List[Catalogue],
    split_ratio: float = 0.8,
    seed: Optional[int] = None,
) -> Tuple[List[Catalogue], List[Catalogue]]:
    """Split catalogues randomly into train and test sets."""
    rng = random.Random(seed)
    catalogues_copy = catalogues.copy()
    rng.shuffle(catalogues_copy)

    split_idx = int(len(catalogues_copy) * split_ratio)
    return catalogues_copy[:split_idx], catalogues_copy[split_idx:]


def split_catalogues_canonical(
    catalogues: List[Catalogue],
    split_ratio: float = 0.8,
    seed: Optional[int] = None,
    catalogue_type: Optional[str] = None,
) -> Tuple[List[Catalogue], List[Catalogue]]:
    """Split the full catalogue universe, then filter the selected family.

    Filtering after splitting keeps the catalogue IDs assigned to train/test
    consistent when experiments compare Excl, Fair, and Mixed subsets.
    """
    selected_type = normalize_catalogue_type(catalogue_type)
    train_catalogues, test_catalogues = split_catalogues_stratified_by_family(
        catalogues,
        split_ratio=split_ratio,
        seed=seed,
    )

    if selected_type in {None, "Mixed"}:
        return train_catalogues, test_catalogues

    return (
        [catalogue for catalogue in train_catalogues if catalogue.catalogue_type == selected_type],
        [catalogue for catalogue in test_catalogues if catalogue.catalogue_type == selected_type],
    )


def split_catalogues_stratified_by_family(
    catalogues: List[Catalogue],
    split_ratio: float = 0.8,
    seed: Optional[int] = None,
) -> Tuple[List[Catalogue], List[Catalogue]]:
    """Split catalogues by family, retaining representation in both splits."""
    if not 0.0 < split_ratio < 1.0:
        raise ValueError("split_ratio must be in (0, 1).")

    rng = random.Random(seed)
    by_family: Dict[str, List[Catalogue]] = {"Excl": [], "Fair": []}
    for catalogue in catalogues:
        if catalogue.catalogue_type in by_family:
            by_family[catalogue.catalogue_type].append(catalogue)

    train_catalogues: List[Catalogue] = []
    test_catalogues: List[Catalogue] = []

    for family_catalogues in by_family.values():
        family_copy = family_catalogues.copy()
        rng.shuffle(family_copy)

        count = len(family_copy)
        if count == 0:
            continue

        train_count = int(round(count * split_ratio))
        if count >= 2:
            train_count = max(1, min(train_count, count - 1))
        else:
            train_count = count

        train_catalogues.extend(family_copy[:train_count])
        test_catalogues.extend(family_copy[train_count:])

    rng.shuffle(train_catalogues)
    rng.shuffle(test_catalogues)
    return train_catalogues, test_catalogues