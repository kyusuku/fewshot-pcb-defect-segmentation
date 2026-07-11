"""DINOv2 patch feature extraction.

The real DINOv2 extractor loads Meta's official PyTorch Hub backbone. A small
color-patch extractor is also provided for fast local smoke tests without
downloading model weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class PatchFeatureMap:
    """Patch-grid features for one image."""

    features: np.ndarray
    image_size: tuple[int, int]
    patch_size: int
    source_size: tuple[int, int] | None = None
    content_box: tuple[int, int, int, int] | None = None

    def __post_init__(self) -> None:
        if self.features.ndim != 3:
            raise ValueError(f"features must be [grid_h, grid_w, dim], got {self.features.shape}")
        width, height = self.image_size
        if width <= 0 or height <= 0:
            raise ValueError("image_size must contain positive dimensions")

        source_size = self.source_size or self.image_size
        if source_size[0] <= 0 or source_size[1] <= 0:
            raise ValueError("source_size must contain positive dimensions")
        content_box = self.content_box or (0, 0, width, height)
        x1, y1, x2, y2 = content_box
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError("content_box must be non-empty and inside image_size")
        object.__setattr__(self, "source_size", source_size)
        object.__setattr__(self, "content_box", content_box)

    @property
    def grid_size(self) -> tuple[int, int]:
        return (int(self.features.shape[1]), int(self.features.shape[0]))

    def flatten(self) -> np.ndarray:
        return self.features.reshape(-1, self.features.shape[-1])

    def valid_patch_mask(self) -> np.ndarray:
        """Return patch centers that fall inside real, unpadded image content."""

        grid_h, grid_w = self.features.shape[:2]
        width, height = self.image_size
        x1, y1, x2, y2 = self.content_box
        x_centers = (np.arange(grid_w, dtype=np.float32) + 0.5) * width / grid_w
        y_centers = (np.arange(grid_h, dtype=np.float32) + 0.5) * height / grid_h
        valid_x = (x_centers >= x1) & (x_centers < x2)
        valid_y = (y_centers >= y1) & (y_centers < y2)
        return valid_y[:, None] & valid_x[None, :]


class PatchFeatureExtractor(Protocol):
    patch_size: int

    def extract(self, image: Image.Image) -> PatchFeatureMap:
        ...


class ColorPatchFeatureExtractor:
    """Deterministic patch-color features for smoke tests and debugging."""

    def __init__(self, image_size: int = 224, patch_size: int = 14) -> None:
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        self.image_size = image_size
        self.patch_size = patch_size

    def extract(self, image: Image.Image) -> PatchFeatureMap:
        source = image.convert("RGB")
        prepared, content_box = resize_and_pad_square_with_content_box(source, self.image_size)
        array = np.asarray(prepared, dtype=np.float32) / 255.0
        grid_h = self.image_size // self.patch_size
        grid_w = self.image_size // self.patch_size
        patches = array.reshape(grid_h, self.patch_size, grid_w, self.patch_size, 3)
        features = patches.mean(axis=(1, 3))
        return PatchFeatureMap(
            features=features.astype(np.float32, copy=False),
            image_size=prepared.size,
            patch_size=self.patch_size,
            source_size=source.size,
            content_box=content_box,
        )


class DINOv2PatchFeatureExtractor:
    """Extract DINOv2 normalized patch-token features."""

    def __init__(
        self,
        model_name: str = "dinov2_vits14",
        image_size: int = 518,
        patch_size: int = 14,
        device: str = "auto",
        model=None,
    ) -> None:
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")

        import torch

        self.model_name = model_name
        self.image_size = image_size
        self.patch_size = patch_size
        self.device = _resolve_device(device, torch)
        self.model = (
            model
            if model is not None
            else torch.hub.load("facebookresearch/dinov2", model_name)
        )
        if hasattr(self.model, "to"):
            self.model.to(self.device)
        if hasattr(self.model, "eval"):
            self.model.eval()

    def extract(self, image: Image.Image) -> PatchFeatureMap:
        import torch

        source = image.convert("RGB")
        prepared, content_box = resize_and_pad_square_with_content_box(source, self.image_size)
        tensor = _image_to_normalized_tensor(prepared, torch).to(self.device)
        with torch.no_grad():
            output = self.model.forward_features(tensor)
        if not isinstance(output, dict) or "x_norm_patchtokens" not in output:
            raise RuntimeError("DINOv2 backbone did not return x_norm_patchtokens")

        patch_tokens = output["x_norm_patchtokens"].detach().cpu().numpy()[0]
        grid = self.image_size // self.patch_size
        features = patch_tokens.reshape(grid, grid, patch_tokens.shape[-1])
        return PatchFeatureMap(
            features=features.astype(np.float32, copy=False),
            image_size=prepared.size,
            patch_size=self.patch_size,
            source_size=source.size,
            content_box=content_box,
        )


def build_feature_extractor(
    feature_backbone: str,
    image_size: int = 518,
    patch_size: int = 14,
    device: str = "auto",
) -> PatchFeatureExtractor:
    if feature_backbone == "patchcore_wrn50":
        from features.patchcore import PatchCoreFeatureExtractor

        return PatchCoreFeatureExtractor(image_size=image_size, device=device)
    if feature_backbone == "color_patch":
        return ColorPatchFeatureExtractor(image_size=image_size, patch_size=patch_size)
    return DINOv2PatchFeatureExtractor(
        model_name=feature_backbone,
        image_size=image_size,
        patch_size=patch_size,
        device=device,
    )


def resize_and_pad_square(image: Image.Image, size: int) -> Image.Image:
    prepared, _ = resize_and_pad_square_with_content_box(image, size)
    return prepared


def resize_and_pad_square_with_content_box(
    image: Image.Image,
    size: int,
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """Resize with aspect-ratio preservation and return content bounds in the square."""

    width, height = image.size
    scale = min(size / width, size / height)
    resized_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    resized = image.resize(resized_size, Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    offset = ((size - resized_size[0]) // 2, (size - resized_size[1]) // 2)
    canvas.paste(resized, offset)
    content_box = (
        offset[0],
        offset[1],
        offset[0] + resized_size[0],
        offset[1] + resized_size[1],
    )
    return canvas, content_box


def _resolve_device(device: str, torch_module) -> str:
    if device != "auto":
        return device
    if torch_module.cuda.is_available():
        return "cuda"
    if hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available():
        return "mps"
    return "cpu"


def _image_to_normalized_tensor(image: Image.Image, torch_module):
    array = np.asarray(image, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    array = (array - mean) / std
    array = np.transpose(array, (2, 0, 1))[None, ...]
    return torch_module.from_numpy(array)
