"""Shared names and validation for image-feature execution modes."""

IMAGE_EMBEDDING_MODE_PRECOMPUTED = "precomputed"
IMAGE_EMBEDDING_MODE_ENCODED_AT_RUNTIME = "encoded_at_runtime"

def normalize_image_embedding_mode(image_embedding_mode: str) -> str:
    """Validate a mode once before selecting image loading/encoding paths."""
    mode = str(image_embedding_mode).strip().lower()
    if mode in {IMAGE_EMBEDDING_MODE_PRECOMPUTED, IMAGE_EMBEDDING_MODE_ENCODED_AT_RUNTIME}:
        return mode
    raise ValueError("Invalid image_embedding_mode. Use either 'precomputed' or 'encoded_at_runtime'.")