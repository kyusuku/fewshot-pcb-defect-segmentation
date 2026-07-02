"""Visualization helpers for dataset sanity checks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from datasets.types import BoxAnnotation, DatasetRecord
from utils.image import load_binary_mask, load_rgb_image, zeros_mask


def visualize_record(
    record: DatasetRecord,
    output_path: str | Path,
    boxes: list[BoxAnnotation] | None = None,
    show: bool = False,
) -> Path:
    """Save a debug figure for one dataset record."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image = load_rgb_image(record.image_path)
    boxes = boxes or []

    if record.dataset == "deeppcb":
        _visualize_deeppcb(record, image, boxes, output_path, show=show)
    else:
        _visualize_visa(record, image, output_path, show=show)

    return output_path


def _visualize_visa(
    record: DatasetRecord,
    image: Image.Image,
    output_path: Path,
    show: bool = False,
) -> None:
    mask = (
        load_binary_mask(record.mask_path, size=image.size)
        if record.mask_path is not None
        else zeros_mask(image.size)
    )

    mask_image = Image.fromarray((mask * 255).astype(np.uint8), mode="L").convert("RGB")
    overlay = _mask_overlay(image, mask)
    canvas = _make_panel_grid(
        [
            ("image", image),
            ("gt mask", mask_image),
            (f"{record.category} | label={record.label}", overlay),
        ],
        title=record.sample_id,
    )
    canvas.save(output_path)
    if show:
        image.show()


def _visualize_deeppcb(
    record: DatasetRecord,
    image: Image.Image,
    boxes: list[BoxAnnotation],
    output_path: Path,
    show: bool = False,
) -> None:
    image_with_boxes = image.copy()
    draw = ImageDraw.Draw(image_with_boxes)
    for box in boxes:
        clipped = box.clipped(*image.size)
        xyxy = clipped.to_xyxy()
        draw.rectangle(xyxy, outline=(255, 40, 40), width=3)
        label = box.class_name or str(box.class_id)
        draw.text((xyxy[0] + 2, xyxy[1] + 2), label, fill=(255, 40, 40))

    template = load_rgb_image(record.template_path) if record.template_path else None
    panels = [("test image", image), (f"{len(boxes)} boxes", image_with_boxes)]
    if template is not None:
        panels.append(("template", template))
    canvas = _make_panel_grid(panels, title=record.sample_id)
    canvas.save(output_path)
    if show:
        image_with_boxes.show()


def safe_filename(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)


def _mask_overlay(image: Image.Image, mask: np.ndarray) -> Image.Image:
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    alpha = (mask > 0).astype(np.uint8) * 120
    red = np.zeros((image.size[1], image.size[0], 4), dtype=np.uint8)
    red[..., 0] = 255
    red[..., 1] = 64
    red[..., 3] = alpha
    overlay = Image.fromarray(red, mode="RGBA")
    return Image.alpha_composite(base, overlay).convert("RGB")


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
