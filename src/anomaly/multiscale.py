"""Single-scale and multi-scale anomaly heatmap inference."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from PIL import Image

from anomaly.heatmap import resize_heatmap_to_image
from anomaly.memory_bank import score_patch_features
from features.dinov2 import PatchFeatureExtractor


def parse_crop_sizes(value: str | Iterable[int]) -> list[int]:
    if isinstance(value, str):
        if not value.strip():
            return []
        return [int(part.strip()) for part in value.split(",") if part.strip()]
    return [int(size) for size in value]


def iter_crop_boxes(
    image_size: tuple[int, int],
    crop_size: int,
    overlap: float,
) -> Iterable[tuple[int, int, int, int]]:
    """Yield crop boxes covering the image, including right/bottom edges."""

    if crop_size <= 0:
        raise ValueError("crop_size must be positive")
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be in [0, 1)")

    width, height = image_size
    crop_w = min(crop_size, width)
    crop_h = min(crop_size, height)
    step = max(1, int(round(crop_size * (1.0 - overlap))))
    x_positions = _axis_positions(length=width, window=crop_w, step=step)
    y_positions = _axis_positions(length=height, window=crop_h, step=step)
    for y in y_positions:
        for x in x_positions:
            yield (x, y, x + crop_w, y + crop_h)


def compute_anomaly_heatmap(
    image: Image.Image,
    extractor: PatchFeatureExtractor,
    memory_bank: np.ndarray,
    crop_sizes: list[int] | None = None,
    crop_overlap: float = 0.25,
    fusion: str = "max",
    normalize_features: bool = True,
) -> np.ndarray:
    """Compute a full-resolution anomaly heatmap with optional local crops."""

    if fusion not in {"max", "mean"}:
        raise ValueError("fusion must be 'max' or 'mean'")

    image = image.convert("RGB")
    crop_sizes = crop_sizes or []
    global_heatmap = _score_image(
        image=image,
        extractor=extractor,
        memory_bank=memory_bank,
        normalize_features=normalize_features,
    )
    if not crop_sizes:
        return global_heatmap

    if fusion == "max":
        fused = global_heatmap.copy()
        counts = None
    else:
        fused = global_heatmap.copy()
        counts = np.ones_like(fused, dtype=np.float32)

    for crop_size in crop_sizes:
        for x1, y1, x2, y2 in iter_crop_boxes(image.size, crop_size, crop_overlap):
            crop = image.crop((x1, y1, x2, y2))
            crop_heatmap = _score_image(
                image=crop,
                extractor=extractor,
                memory_bank=memory_bank,
                normalize_features=normalize_features,
            )
            region = fused[y1:y2, x1:x2]
            if fusion == "max":
                fused[y1:y2, x1:x2] = np.maximum(region, crop_heatmap)
            else:
                fused[y1:y2, x1:x2] = region + crop_heatmap
                counts[y1:y2, x1:x2] += 1.0

    if counts is not None:
        fused = fused / np.maximum(counts, 1.0)
    return fused.astype(np.float32, copy=False)


def _score_image(
    image: Image.Image,
    extractor: PatchFeatureExtractor,
    memory_bank: np.ndarray,
    normalize_features: bool,
) -> np.ndarray:
    feature_map = extractor.extract(image)
    patch_heatmap = score_patch_features(
        feature_map,
        memory_bank,
        normalize=normalize_features,
    )
    return resize_heatmap_to_image(patch_heatmap, image.size, normalize=False)


def _axis_positions(length: int, window: int, step: int) -> list[int]:
    if length <= window:
        return [0]
    positions = list(range(0, length - window + 1, step))
    last = length - window
    if positions[-1] != last:
        positions.append(last)
    return positions
