"""Anomaly heatmap utilities and debug rendering."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from features.dinov2 import PatchFeatureMap
from utils.image import load_binary_mask, load_rgb_image, zeros_mask


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

    y_coords = (np.arange(output_height, dtype=np.float32) + 0.5) * input_height / output_height - 0.5
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
