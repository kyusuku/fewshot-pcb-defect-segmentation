"""Evaluation helpers for predicted binary masks."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image

from utils.image import load_binary_mask


PER_MASK_FIELDS = [
    "sample_id",
    "category",
    "label",
    "mask_precision",
    "mask_recall",
    "mask_f1",
    "mask_iou",
    "pred_positive_pixels",
    "gt_positive_pixels",
    "pred_mask_path",
    "mask_path",
]


def mask_confusion_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Compute foreground precision, recall, F1, and IoU for two binary masks."""

    prediction = np.asarray(prediction).astype(bool)
    target = np.asarray(target).astype(bool)
    if prediction.shape != target.shape:
        raise ValueError("prediction and target masks must have the same shape")

    tp = float(np.sum(prediction & target))
    fp = float(np.sum(prediction & ~target))
    fn = float(np.sum(~prediction & target))
    union = tp + fp + fn
    if union == 0.0:
        return {
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "iou": 1.0,
            "pred_positive_pixels": 0.0,
            "gt_positive_pixels": 0.0,
        }

    precision = tp / (tp + fp) if tp + fp > 0.0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0.0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(tp / union),
        "pred_positive_pixels": float(np.sum(prediction)),
        "gt_positive_pixels": float(np.sum(target)),
    }


def evaluate_mask_rows(
    rows: list[dict[str, str]],
) -> tuple[dict[str, float], list[dict[str, str | float]]]:
    """Evaluate predicted mask rows from `run_mask_refinement.py` output."""

    per_row: list[dict[str, str | float]] = []
    all_metrics: list[dict[str, float]] = []
    anomaly_metrics: list[dict[str, float]] = []

    for row in rows:
        pred_mask_path = row.get("pred_mask_path") or ""
        if not pred_mask_path:
            continue
        prediction = load_binary_mask(pred_mask_path)
        label = int(row.get("label") or 0)
        mask_path = row.get("mask_path") or ""
        if mask_path:
            target = load_binary_mask(mask_path, size=(prediction.shape[1], prediction.shape[0]))
        elif label == 0:
            target = np.zeros(prediction.shape, dtype=np.uint8)
        else:
            raise ValueError(
                "mask_path is required for anomalous rows. Pass --source-scores-csv "
                "if evaluating older mask_scores.csv files."
            )
        if target.shape != prediction.shape:
            target = _resize_mask(target, prediction.shape)

        metrics = mask_confusion_metrics(prediction, target)
        all_metrics.append(metrics)
        if label == 1:
            anomaly_metrics.append(metrics)
        per_row.append(
            {
                "sample_id": row.get("sample_id", ""),
                "category": row.get("category", ""),
                "label": row.get("label", ""),
                "mask_precision": metrics["precision"],
                "mask_recall": metrics["recall"],
                "mask_f1": metrics["f1"],
                "mask_iou": metrics["iou"],
                "pred_positive_pixels": metrics["pred_positive_pixels"],
                "gt_positive_pixels": metrics["gt_positive_pixels"],
                "pred_mask_path": pred_mask_path,
                "mask_path": mask_path,
            }
        )

    summary = {
        "num_mask_images": float(len(all_metrics)),
        "num_anomaly_mask_images": float(len(anomaly_metrics)),
        "mean_mask_precision": _mean(all_metrics, "precision"),
        "mean_mask_recall": _mean(all_metrics, "recall"),
        "mean_mask_f1": _mean(all_metrics, "f1"),
        "mean_mask_iou": _mean(all_metrics, "iou"),
        "mean_anomaly_mask_precision": _mean(anomaly_metrics, "precision"),
        "mean_anomaly_mask_recall": _mean(anomaly_metrics, "recall"),
        "mean_anomaly_mask_f1": _mean(anomaly_metrics, "f1"),
        "mean_anomaly_mask_iou": _mean(anomaly_metrics, "iou"),
    }
    return summary, per_row


def merge_source_score_rows(
    mask_rows: list[dict[str, str]],
    source_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Fill missing ground-truth fields from the original heatmap score rows."""

    source_by_id = {row.get("sample_id", ""): row for row in source_rows}
    merged_rows = []
    for row in mask_rows:
        merged = dict(row)
        source = source_by_id.get(row.get("sample_id", ""))
        if source:
            for key in ("category", "label", "image_path", "mask_path"):
                if not merged.get(key) and source.get(key):
                    merged[key] = source[key]
        merged_rows.append(merged)
    return merged_rows


def resolve_mask_row_paths(
    rows: list[dict[str, str]],
    base_dir: str | Path,
) -> list[dict[str, str]]:
    """Resolve relative mask artifact paths against the CSV directory."""

    base_dir = Path(base_dir)
    resolved_rows = []
    for row in rows:
        resolved = dict(row)
        for key in ("pred_mask_path", "mask_path", "heatmap_path", "debug_path"):
            value = resolved.get(key) or ""
            if not value:
                continue
            path = Path(value)
            if not path.is_absolute() and not path.exists():
                resolved[key] = str(base_dir / path)
        resolved_rows.append(resolved)
    return resolved_rows


def _mean(rows: list[dict[str, float]], key: str) -> float:
    if not rows:
        return math.nan
    return float(sum(row[key] for row in rows) / len(rows))


def _resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    image = Image.fromarray((mask > 0).astype(np.uint8) * 255, mode="L")
    image = image.resize((width, height), Image.Resampling.NEAREST)
    return (np.asarray(image) > 0).astype(np.uint8)
