"""Image and mask IO helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from datasets.types import BoxAnnotation


def load_rgb_image(path: str | Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def load_binary_mask(
    path: str | Path,
    size: tuple[int, int] | None = None,
    threshold: int = 0,
) -> np.ndarray:
    with Image.open(path) as image:
        mask = image.convert("L")
        if size is not None and mask.size != size:
            mask = mask.resize(size, Image.Resampling.NEAREST)
        array = np.asarray(mask)
    return (array > threshold).astype(np.uint8)


def zeros_mask(size: tuple[int, int]) -> np.ndarray:
    width, height = size
    return np.zeros((height, width), dtype=np.uint8)


def boxes_to_mask(boxes: list[BoxAnnotation], size: tuple[int, int]) -> np.ndarray:
    width, height = size
    mask = zeros_mask(size)
    for box in boxes:
        clipped = box.clipped(width=width, height=height)
        x1, y1, x2, y2 = (int(round(value)) for value in clipped.to_xyxy())
        if x2 < x1 or y2 < y1:
            continue
        mask[y1 : y2 + 1, x1 : x2 + 1] = 1
    return mask


def image_to_tensor(image: Image.Image):
    import torch

    array = np.asarray(image, dtype=np.float32) / 255.0
    array = np.transpose(array, (2, 0, 1))
    return torch.from_numpy(array)


def mask_to_tensor(mask: np.ndarray):
    import torch

    return torch.from_numpy(mask.astype(np.float32, copy=False)).unsqueeze(0)
