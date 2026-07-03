"""Anomaly heatmap utilities and debug rendering."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from utils.image import load_binary_mask, load_rgb_image, zeros_mask


def normalize_heatmap(heatmap: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    heatmap = heatmap.astype(np.float32, copy=False)
    minimum = float(np.min(heatmap))
    maximum = float(np.max(heatmap))
    if maximum - minimum <= eps:
        return np.zeros_like(heatmap, dtype=np.float32)
    return (heatmap - minimum) / (maximum - minimum)


def resize_heatmap_to_image(heatmap: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    heatmap_image = Image.fromarray((normalize_heatmap(heatmap) * 255).astype(np.uint8), mode="L")
    resized = heatmap_image.resize(image_size, Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32) / 255.0


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
