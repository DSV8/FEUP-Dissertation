"""Load shared and pipeline-specific YAML configuration for the project."""

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"


def _merge_config(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge overrides while leaving the input dictionaries untouched."""
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_config(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_pipeline_config(pipeline_name: str) -> Dict[str, Any]:
    """Load common settings and merge the selected pipeline's YAML file.

    Nested pipeline values replace matching common values; unspecified common
    settings remain available to every runner.
    """
    common_path = CONFIG_DIR / "common.yaml"
    pipeline_path = CONFIG_DIR / f"{pipeline_name}.yaml"

    with common_path.open("rb") as config_file:
        config = yaml.safe_load(config_file)
    with pipeline_path.open("rb") as config_file:
        pipeline_config = yaml.safe_load(config_file)

    return _merge_config(config, pipeline_config)