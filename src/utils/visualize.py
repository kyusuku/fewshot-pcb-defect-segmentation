"""Visualization helpers for dataset sanity checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

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

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image)
    axes[0].set_title("image")
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("gt mask")
    axes[1].axis("off")

    axes[2].imshow(image)
    if mask.max() > 0:
        overlay = np.ma.masked_where(mask == 0, mask)
        axes[2].imshow(overlay, cmap="autumn", alpha=0.45, vmin=0, vmax=1)
    axes[2].set_title(f"{record.category} | label={record.label}")
    axes[2].axis("off")

    fig.suptitle(record.sample_id)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    if show:
        plt.show()
    plt.close(fig)


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
    num_cols = 3 if template is not None else 2
    fig, axes = plt.subplots(1, num_cols, figsize=(4 * num_cols, 4))
    axes_list: list[Any] = list(axes) if isinstance(axes, np.ndarray) else [axes]

    axes_list[0].imshow(image)
    axes_list[0].set_title("test image")
    axes_list[0].axis("off")

    axes_list[1].imshow(image_with_boxes)
    axes_list[1].set_title(f"{len(boxes)} boxes")
    axes_list[1].axis("off")

    if template is not None:
        axes_list[2].imshow(template)
        axes_list[2].set_title("template")
        axes_list[2].axis("off")

    fig.suptitle(record.sample_id)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    if show:
        plt.show()
    plt.close(fig)


def safe_filename(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)
