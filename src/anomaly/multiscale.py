"""Single-scale and multi-scale anomaly heatmap inference."""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
from PIL import Image

from anomaly.heatmap import project_patch_heatmap_to_source
from anomaly.memory_bank import score_patch_features
from features.cache import FeatureCache
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
    feature_cache: FeatureCache | None = None,
    cache_key_for_view: Callable[[str], str] | None = None,
) -> np.ndarray:
    """Compute a full-resolution anomaly heatmap with optional local crops."""

    if fusion not in {"max", "mean"}:
        raise ValueError("fusion must be 'max' or 'mean'")

    image = image.convert("RGB")
    crop_sizes = crop_sizes or []
    if (feature_cache is None) != (cache_key_for_view is None):
        raise ValueError("feature_cache and cache_key_for_view must be provided together")

    global_view = f"global:0,0,{image.width},{image.height}"
    global_heatmap = _score_image(
        image=image,
        extractor=extractor,
        memory_bank=memory_bank,
        normalize_features=normalize_features,
        feature_cache=feature_cache,
        cache_key=cache_key_for_view(global_view) if cache_key_for_view else None,
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
                feature_cache=feature_cache,
                cache_key=(
                    cache_key_for_view(f"crop:{x1},{y1},{x2},{y2}")
                    if cache_key_for_view
                    else None
                ),
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
    feature_cache: FeatureCache | None,
    cache_key: str | None,
) -> np.ndarray:
    if feature_cache is None:
        feature_map = extractor.extract(image)
    else:
        assert cache_key is not None
        feature_map = feature_cache.get_or_compute(cache_key, lambda: extractor.extract(image))
    if feature_map.source_size != image.size:
        raise ValueError(
            "feature map source_size does not match the image being scored: "
            f"{feature_map.source_size} != {image.size}"
        )
    patch_heatmap = score_patch_features(
        feature_map,
        memory_bank,
        normalize=normalize_features,
    )
    return project_patch_heatmap_to_source(patch_heatmap, feature_map)


def _axis_positions(length: int, window: int, step: int) -> list[int]:
    if length <= window:
        return [0]
    positions = list(range(0, length - window + 1, step))
    last = length - window
    if positions[-1] != last:
        positions.append(last)
    return positions
