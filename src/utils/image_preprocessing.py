"""Canonical image padding, resizing, tensor conversion, and normalization."""

from PIL import Image
import torch
from torchvision import transforms


def pad_img(image_path: str) -> Image.Image:
    """Pad an image equally to make it square, using black padding."""
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    if width != height:
        max_dim = max(width, height)
        pad_w = max_dim - width
        pad_h = max_dim - height
        padding = (
            pad_w // 2,
            pad_h // 2,
            pad_w - (pad_w // 2),
            pad_h - (pad_h // 2),
        )
        image = transforms.functional.pad(image, padding, fill=(0, 0, 0))

    return image


def transform_fn(image_path: str, target_size: int = 224) -> torch.Tensor:
    """Pad, resize, tensorize, and ImageNet-normalize an image."""
    padded_img = pad_img(image_path)
    image = transforms.Resize((target_size, target_size))(padded_img)
    tensor = transforms.ToTensor()(image)
    return transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )(tensor)


def pad_and_resize(image_path: str, target_size: int = 224) -> Image.Image:
    """Pad an image to square and resize it to the requested target size."""
    return transforms.Resize((target_size, target_size))(pad_img(image_path))