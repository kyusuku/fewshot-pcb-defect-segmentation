"""Anomaly heatmap utilities and debug rendering."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from features.dinov2 import PatchFeatureMap
from utils.image import load_binary_mask, load_rgb_image, zeros_mask


@dataclass(frozen=True)
class PatchHeatmapComponent:
    """One patch-score grid plus the exact geometry needed for projection."""

    patch_scores: np.ndarray
    image_size: tuple[int, int]
    patch_size: int
    source_size: tuple[int, int]
    content_box: tuple[int, int, int, int]

    def __post_init__(self) -> None:
        if not isinstance(self.patch_scores, np.ndarray) or self.patch_scores.ndim != 2:
            raise ValueError("patch_scores must be a two-dimensional numpy array")
        if self.patch_scores.dtype != np.float32:
            raise ValueError("patch_scores must have dtype float32")
        if not self.patch_scores.size:
            raise ValueError("patch_scores must be non-empty")
        if not np.isfinite(self.patch_scores).all():
            raise ValueError("patch_scores must contain only finite values")
        if not isinstance(self.patch_size, int) or isinstance(self.patch_size, bool):
            raise ValueError("patch_size must be a positive integer")
        if self.patch_size <= 0:
            raise ValueError("patch_size must be a positive integer")
        image_size = _positive_integer_tuple(self.image_size, "image_size", 2)
        source_size = _positive_integer_tuple(self.source_size, "source_size", 2)
        if image_size[0] % self.patch_size or image_size[1] % self.patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        expected_grid = (
            image_size[1] // self.patch_size,
            image_size[0] // self.patch_size,
        )
        if self.patch_scores.shape != expected_grid:
            raise ValueError(
                "patch_scores shape is inconsistent with image_size and patch_size: "
                f"{self.patch_scores.shape} != {expected_grid}"
            )
        content_box = _integer_tuple(self.content_box, "content_box", 4)
        x1, y1, x2, y2 = content_box
        if not (0 <= x1 < x2 <= image_size[0] and 0 <= y1 < y2 <= image_size[1]):
            raise ValueError("content_box must be non-empty and inside image_size")
        scores = self.patch_scores.copy()
        scores.setflags(write=False)
        object.__setattr__(self, "patch_scores", scores)
        object.__setattr__(self, "image_size", image_size)
        object.__setattr__(self, "source_size", source_size)
        object.__setattr__(self, "content_box", content_box)

    @classmethod
    def from_feature_map(
        cls,
        patch_scores: np.ndarray,
        feature_map: PatchFeatureMap,
    ) -> PatchHeatmapComponent:
        """Capture projection geometry from the feature map that produced scores."""

        return cls(
            patch_scores=patch_scores,
            image_size=feature_map.image_size,
            patch_size=feature_map.patch_size,
            source_size=feature_map.source_size,
            content_box=feature_map.content_box,
        )

    def valid_patch_mask(self) -> np.ndarray:
        """Return patch centers that fall inside the unpadded content box."""

        grid_h, grid_w = self.patch_scores.shape
        width, height = self.image_size
        x1, y1, x2, y2 = self.content_box
        x_centers = (np.arange(grid_w, dtype=np.float32) + 0.5) * width / grid_w
        y_centers = (np.arange(grid_h, dtype=np.float32) + 0.5) * height / grid_h
        valid_x = (x_centers >= x1) & (x_centers < x2)
        valid_y = (y_centers >= y1) & (y_centers < y2)
        return valid_y[:, None] & valid_x[None, :]

    def render(self) -> np.ndarray:
        """Project the stored patch scores back to their source-view geometry."""

        return project_patch_heatmap_to_source(self.patch_scores, self)


@dataclass(frozen=True)
class PositionedHeatmap:
    """A projected view positioned inside the query-image canvas."""

    box: tuple[int, int, int, int]
    component: PatchHeatmapComponent

    def __post_init__(self) -> None:
        box = _integer_tuple(self.box, "box", 4)
        x1, y1, x2, y2 = box
        if not (0 <= x1 < x2 and 0 <= y1 < y2):
            raise ValueError("box must have non-negative origin and positive area")
        if not isinstance(self.component, PatchHeatmapComponent):
            raise ValueError("component must be a PatchHeatmapComponent")
        object.__setattr__(self, "box", box)


@dataclass(frozen=True)
class ProjectedHeatmap:
    """Exact global/crop projection recipe for one anomaly heatmap."""

    source_size: tuple[int, int]
    fusion: str
    components: tuple[PositionedHeatmap, ...]

    def __post_init__(self) -> None:
        source_size = _positive_integer_tuple(self.source_size, "source_size", 2)
        if self.fusion not in {"max", "mean"}:
            raise ValueError("fusion must be 'max' or 'mean'")
        if not isinstance(self.components, (tuple, list)) or not self.components:
            raise ValueError("components must be a non-empty sequence")
        components = tuple(self.components)
        if not all(isinstance(item, PositionedHeatmap) for item in components):
            raise ValueError("components must contain PositionedHeatmap values")
        full_box = (0, 0, source_size[0], source_size[1])
        if components[0].box != full_box:
            raise ValueError("first component must cover the full source canvas")
        for item in components:
            x1, y1, x2, y2 = item.box
            if x2 > source_size[0] or y2 > source_size[1]:
                raise ValueError("component box must remain inside the source canvas")
            if item.component.source_size != (x2 - x1, y2 - y1):
                raise ValueError("component source_size must match its positioned box")
        object.__setattr__(self, "source_size", source_size)
        object.__setattr__(self, "components", components)

    def render(self) -> np.ndarray:
        """Reconstruct the full-resolution heatmap with the original operation order."""

        global_item, *crops = self.components
        fused = global_item.component.render()
        counts = np.ones_like(fused, dtype=np.float32) if self.fusion == "mean" else None
        for item in crops:
            x1, y1, x2, y2 = item.box
            crop_heatmap = item.component.render()
            region = fused[y1:y2, x1:x2]
            if self.fusion == "max":
                fused[y1:y2, x1:x2] = np.maximum(region, crop_heatmap)
            else:
                fused[y1:y2, x1:x2] = region + crop_heatmap
                assert counts is not None
                counts[y1:y2, x1:x2] += 1.0
        if counts is not None:
            fused = fused / np.maximum(counts, 1.0)
        return fused.astype(np.float32, copy=False)


def normalize_heatmap(heatmap: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    heatmap = heatmap.astype(np.float32, copy=False)
    minimum = float(np.min(heatmap))
    maximum = float(np.max(heatmap))
    if maximum - minimum <= eps:
        return np.zeros_like(heatmap, dtype=np.float32)
    return (heatmap - minimum) / (maximum - minimum)


def resize_heatmap_to_image(
    heatmap: np.ndarray,
    image_size: tuple[int, int],
    normalize: bool = True,
) -> np.ndarray:
    heatmap = np.asarray(heatmap, dtype=np.float32)
    if heatmap.ndim != 2:
        raise ValueError(f"Expected a 2D heatmap, got shape {heatmap.shape}")

    if normalize:
        heatmap = normalize_heatmap(heatmap)

    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError(f"Expected positive image size, got {image_size}")
    return _resize_float_bilinear(heatmap, output_shape=(height, width))


def project_patch_heatmap_to_source(
    patch_heatmap: np.ndarray,
    feature_map: PatchFeatureMap,
) -> np.ndarray:
    """Unpad a patch heatmap in prepared coordinates, then restore source geometry."""

    width, height = feature_map.image_size
    if feature_map.content_box == (0, 0, width, height):
        return resize_heatmap_to_image(
            patch_heatmap,
            feature_map.source_size,
            normalize=False,
        )

    valid = feature_map.valid_patch_mask()
    valid_rows = np.flatnonzero(valid.any(axis=1))
    valid_columns = np.flatnonzero(valid.any(axis=0))
    if valid_rows.size == 0 or valid_columns.size == 0:
        raise ValueError("feature map has no patch centers inside its content_box")
    content_patches = patch_heatmap[
        valid_rows[0] : valid_rows[-1] + 1,
        valid_columns[0] : valid_columns[-1] + 1,
    ]
    x1, y1, x2, y2 = feature_map.content_box
    content = resize_heatmap_to_image(
        content_patches,
        (x2 - x1, y2 - y1),
        normalize=False,
    )
    return resize_heatmap_to_image(content, feature_map.source_size, normalize=False)


def _resize_float_bilinear(heatmap: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    input_height, input_width = heatmap.shape
    output_height, output_width = output_shape
    if input_height == 0 or input_width == 0:
        raise ValueError("Cannot resize an empty heatmap")
    if (input_height, input_width) == (output_height, output_width):
        return heatmap.astype(np.float32, copy=True)

    y_coords = (
        np.arange(output_height, dtype=np.float32) + 0.5
    ) * input_height / output_height - 0.5
    x_coords = (np.arange(output_width, dtype=np.float32) + 0.5) * input_width / output_width - 0.5
    y_coords = np.clip(y_coords, 0.0, input_height - 1)
    x_coords = np.clip(x_coords, 0.0, input_width - 1)

    y0 = np.floor(y_coords).astype(np.int64)
    x0 = np.floor(x_coords).astype(np.int64)
    y1 = np.minimum(y0 + 1, input_height - 1)
    x1 = np.minimum(x0 + 1, input_width - 1)
    y_weight = (y_coords - y0).astype(np.float32)
    x_weight = (x_coords - x0).astype(np.float32)

    top_left = heatmap[y0[:, None], x0[None, :]]
    top_right = heatmap[y0[:, None], x1[None, :]]
    bottom_left = heatmap[y1[:, None], x0[None, :]]
    bottom_right = heatmap[y1[:, None], x1[None, :]]

    top = top_left * (1.0 - x_weight)[None, :] + top_right * x_weight[None, :]
    bottom = bottom_left * (1.0 - x_weight)[None, :] + bottom_right * x_weight[None, :]
    resized = top * (1.0 - y_weight)[:, None] + bottom * y_weight[:, None]
    return resized.astype(np.float32, copy=False)


def heatmap_to_rgb(heatmap: np.ndarray) -> Image.Image:
    normalized = normalize_heatmap(heatmap)
    red = (normalized * 255).astype(np.uint8)
    green = (np.clip(1.0 - np.abs(normalized - 0.75) * 2.0, 0.0, 1.0) * 220).astype(np.uint8)
    blue = ((1.0 - normalized) * 80).astype(np.uint8)
    rgb = np.stack([red, green, blue], axis=-1)
    return Image.fromarray(rgb, mode="RGB")


def overlay_heatmap(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.45) -> Image.Image:
    heatmap_rgb = heatmap_to_rgb(resize_heatmap_to_image(heatmap, image.size))
    return Image.blend(image.convert("RGB"), heatmap_rgb, alpha=alpha)


def save_heatmap_debug_panel(
    image_path: str | Path,
    mask_path: str | Path | None,
    heatmap: np.ndarray,
    output_path: str | Path,
    title: str,
) -> Path:
    image = load_rgb_image(image_path)
    resized_heatmap = resize_heatmap_to_image(heatmap, image.size)
    mask = load_binary_mask(mask_path, size=image.size) if mask_path else zeros_mask(image.size)
    mask_image = Image.fromarray((mask * 255).astype(np.uint8), mode="L").convert("RGB")
    panel = _make_panel_grid(
        [
            ("image", image),
            ("gt mask", mask_image),
            ("anomaly heatmap", heatmap_to_rgb(resized_heatmap)),
            ("overlay", overlay_heatmap(image, heatmap)),
        ],
        title=title,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(output_path)
    return output_path


def _make_panel_grid(panels: list[tuple[str, Image.Image]], title: str) -> Image.Image:
    padding = 12
    title_height = 28
    label_height = 22
    panel_width = max(image.width for _, image in panels)
    panel_height = max(image.height for _, image in panels)
    width = padding + len(panels) * (panel_width + padding)
    height = padding + title_height + panel_height + label_height + padding
    canvas = Image.new("RGB", (width, height), (245, 245, 242))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((padding, padding), title, fill=(30, 30, 30), font=font)

    y = padding + title_height
    for index, (label, image) in enumerate(panels):
        x = padding + index * (panel_width + padding)
        canvas.paste(image.convert("RGB"), (x, y))
        draw.rectangle((x, y, x + image.width - 1, y + image.height - 1), outline=(40, 40, 40))
        draw.text((x, y + panel_height + 5), label, fill=(30, 30, 30), font=font)

    return canvas


def _integer_tuple(value: object, name: str, length: int) -> tuple[int, ...]:
    if not isinstance(value, (tuple, list)) or len(value) != length:
        raise ValueError(f"{name} must contain exactly {length} integers")
    if any(not isinstance(item, int) or isinstance(item, bool) for item in value):
        raise ValueError(f"{name} must contain exactly {length} integers")
    return tuple(value)


def _positive_integer_tuple(value: object, name: str, length: int) -> tuple[int, ...]:
    result = _integer_tuple(value, name, length)
    if any(item <= 0 for item in result):
        raise ValueError(f"{name} must contain positive integer dimensions")
    return result
