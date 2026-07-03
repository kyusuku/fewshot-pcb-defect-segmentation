"""Summarize per-category metric JSON files."""

from __future__ import annotations

import json
import re
from pathlib import Path


SUMMARY_FIELDS = [
    "category",
    "num_images",
    "image_auroc",
    "pixel_auroc",
    "best_pixel_f1",
    "best_pixel_iou",
    "num_pixels_evaluated",
]


def summarize_metric_files(paths: list[str | Path]) -> list[dict[str, float | int | str]]:
    rows = []
    for path in paths:
        path = Path(path)
        metrics = json.loads(path.read_text())
        rows.append(
            {
                "category": infer_category(path),
                "num_images": int(metrics.get("num_images", 0)),
                "image_auroc": float(metrics.get("image_auroc", 0.0)),
                "pixel_auroc": float(metrics.get("pixel_auroc", 0.0)),
                "best_pixel_f1": float(metrics.get("best_pixel_f1", 0.0)),
                "best_pixel_iou": float(metrics.get("best_pixel_iou", 0.0)),
                "num_pixels_evaluated": int(metrics.get("num_pixels_evaluated", 0)),
            }
        )
    rows.sort(key=lambda row: str(row["category"]))
    if rows:
        rows.append(_mean_row(rows))
    return rows


def infer_category(path: Path) -> str:
    candidates = [path.parent.name, path.name]
    for candidate in candidates:
        match = re.search(r"(pcb\d+)", candidate)
        if match:
            return match.group(1)
    return path.parent.name


def format_markdown_table(rows: list[dict[str, float | int | str]]) -> str:
    lines = [
        "| category | num_images | image_auroc | pixel_auroc | best_pixel_f1 | best_pixel_iou |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {category} | {num_images} | {image_auroc:.4f} | {pixel_auroc:.4f} | "
            "{best_pixel_f1:.4f} | {best_pixel_iou:.4f} |".format(**row)
        )
    return "\n".join(lines) + "\n"


def _mean_row(rows: list[dict[str, float | int | str]]) -> dict[str, float | int | str]:
    metric_keys = ["image_auroc", "pixel_auroc", "best_pixel_f1", "best_pixel_iou"]
    mean = {
        "category": "mean",
        "num_images": int(sum(int(row["num_images"]) for row in rows)),
        "num_pixels_evaluated": int(sum(int(row["num_pixels_evaluated"]) for row in rows)),
    }
    for key in metric_keys:
        mean[key] = sum(float(row[key]) for row in rows) / len(rows)
    return mean
