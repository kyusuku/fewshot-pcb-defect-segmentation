#!/usr/bin/env python
"""Render the reproducible method diagram for the paper."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


CANVAS_SIZE = (1800, 700)
BLOCKS = [
    {
        "id": "supports",
        "label": "Few normal\nsupports",
        "rect": (80, 110, 360, 230),
        "color": "#d7e9ff",
    },
    {
        "id": "memory",
        "label": "DINOv2\nmemory bank",
        "rect": (470, 110, 760, 230),
        "color": "#d7e9ff",
    },
    {
        "id": "anomaly",
        "label": "Multi-scale\nanomaly map",
        "rect": (870, 110, 1200, 230),
        "color": "#ffe3ae",
    },
    {
        "id": "sam2",
        "label": "Anomaly-guided\nSAM2 refinement",
        "rect": (870, 380, 1200, 500),
        "color": "#ccebd7",
    },
    {
        "id": "final",
        "label": "Proposal and SAM2\nintersection mask",
        "rect": (1280, 380, 1710, 500),
        "color": "#ccebd7",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-png",
        type=Path,
        default=Path("artifacts/paper_assets/method_figure.png"),
    )
    parser.add_argument(
        "--layout-json",
        type=Path,
        default=Path("artifacts/paper_assets/method_figure_layout.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    render_method_figure(
        args.output_png,
        args.layout_json,
        generation_command=shlex.join(sys.argv),
    )


def render_method_figure(
    output_png: str | Path,
    layout_json: str | Path,
    *,
    generation_command: str = "scripts/render_method_figure.py",
) -> None:
    output_png = Path(output_png)
    layout_json = Path(layout_json)
    image = Image.new("RGB", CANVAS_SIZE, "white")
    draw = ImageDraw.Draw(image)
    font = _load_font(size=28)
    small_font = _load_font(size=22)
    for block in BLOCKS:
        rect = tuple(block["rect"])
        draw.rounded_rectangle(rect, radius=8, fill=block["color"], outline="#243447", width=3)
        _center_text(draw, rect, str(block["label"]), font)
    _arrow(draw, (360, 170), (470, 170))
    _arrow(draw, (760, 170), (870, 170))
    _arrow(draw, (1035, 230), (1035, 380))
    _arrow(draw, (1200, 440), (1280, 440))
    _arrow(draw, (360, 440), (870, 440))
    draw.text((1065, 282), "calibrated proposal", fill="#5a3a00", font=small_font)
    query_label = "Query image"
    query_bbox = draw.textbbox((0, 0), query_label, font=font)
    query_y = 440 - (query_bbox[1] + query_bbox[3]) / 2
    draw.text((95, query_y), query_label, fill="#243447", font=font)
    evaluation_label = "binary mask used for evaluation"
    evaluation_bbox = draw.textbbox((0, 0), evaluation_label, font=small_font)
    final_rect = BLOCKS[-1]["rect"]
    final_center_x = (final_rect[0] + final_rect[2]) / 2
    evaluation_x = final_center_x - (evaluation_bbox[0] + evaluation_bbox[2]) / 2
    draw.text((evaluation_x, 525), evaluation_label, fill="#243447", font=small_font)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    layout_json.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_png)
    try:
        portable_output = output_png.relative_to(layout_json.parent).as_posix()
    except ValueError:
        portable_output = str(output_png)
    layout_json.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "canvas": list(CANVAS_SIZE),
                "generation_command": generation_command,
                "renderer_source_sha256": _sha256_file(Path(__file__)),
                "output_png": portable_output,
                "output_png_sha256": _sha256_file(output_png),
                "blocks": BLOCKS,
                "arrows": [
                    ["supports", "memory"],
                    ["memory", "anomaly"],
                    ["anomaly", "sam2"],
                    ["sam2", "final"],
                    ["query", "sam2"],
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int]) -> None:
    draw.line([start, end], fill="#243447", width=3)
    x1, y1 = start
    x2, y2 = end
    if abs(x2 - x1) >= abs(y2 - y1):
        direction = 1 if x2 >= x1 else -1
        points = [(x2, y2), (x2 - 18 * direction, y2 - 10), (x2 - 18 * direction, y2 + 10)]
    else:
        direction = 1 if y2 >= y1 else -1
        points = [(x2, y2), (x2 - 10, y2 - 18 * direction), (x2 + 10, y2 - 18 * direction)]
    draw.polygon(points, fill="#243447")


def _center_text(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    text: str,
    font: ImageFont.ImageFont,
) -> None:
    bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=6)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    x = rect[0] + (rect[2] - rect[0] - width) / 2
    y = rect[1] + (rect[3] - rect[1] - height) / 2
    draw.multiline_text((x, y), text, fill="#243447", font=font, spacing=6, align="center")


def _load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttf",
        "/Library/Fonts/Arial.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


if __name__ == "__main__":
    main()
