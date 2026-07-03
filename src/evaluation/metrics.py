"""Evaluation metrics for anomaly heatmaps and predicted masks."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image

from utils.image import load_binary_mask


def binary_roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute ROC AUC for binary labels where larger scores mean more anomalous."""

    labels = np.asarray(labels).astype(np.uint8).ravel()
    scores = np.asarray(scores, dtype=np.float64).ravel()
    if labels.shape[0] != scores.shape[0]:
        raise ValueError("labels and scores must have the same number of elements")

    num_pos = int(np.sum(labels == 1))
    num_neg = int(np.sum(labels == 0))
    if num_pos == 0 or num_neg == 0:
        return math.nan

    ranks = _average_ranks(scores)
    pos_rank_sum = float(np.sum(ranks[labels == 1]))
    auc = (pos_rank_sum - num_pos * (num_pos + 1) / 2.0) / (num_pos * num_neg)
    return float(auc)


def best_f1_iou(
    labels: np.ndarray,
    scores: np.ndarray,
    max_thresholds: int = 512,
) -> dict[str, float]:
    """Find the best binary threshold by F1 and report the IoU at that threshold."""

    labels = np.asarray(labels).astype(np.uint8).ravel()
    scores = np.asarray(scores, dtype=np.float32).ravel()
    if labels.shape[0] != scores.shape[0]:
        raise ValueError("labels and scores must have the same number of elements")
    if int(np.sum(labels == 1)) == 0:
        return {
            "best_f1": math.nan,
            "best_iou": math.nan,
            "best_threshold": math.nan,
            "num_thresholds": 0.0,
        }

    thresholds = _threshold_candidates(scores, max_thresholds=max_thresholds)
    best = {
        "best_f1": -1.0,
        "best_iou": 0.0,
        "best_threshold": math.nan,
        "num_thresholds": float(len(thresholds)),
    }
    for threshold in thresholds:
        prediction = scores >= threshold
        target = labels == 1
        tp = float(np.sum(prediction & target))
        fp = float(np.sum(prediction & ~target))
        fn = float(np.sum(~prediction & target))
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        iou = tp / (tp + fp + fn) if tp + fp + fn > 0 else 0.0
        if f1 > best["best_f1"]:
            best = {
                "best_f1": float(f1),
                "best_iou": float(iou),
                "best_threshold": float(threshold),
                "num_thresholds": float(len(thresholds)),
            }
    return best


def summarize_image_scores(rows: list[dict[str, str]]) -> dict[str, float]:
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.uint8)
    scores = np.asarray([float(row["image_score"]) for row in rows], dtype=np.float32)
    return {
        "num_images": float(len(rows)),
        "image_auroc": binary_roc_auc(labels, scores),
    }


def evaluate_heatmap_rows(
    rows: list[dict[str, str]],
    max_pixels: int | None = None,
    seed: int = 4880,
) -> dict[str, float]:
    """Evaluate image-level scores and pixel-level heatmaps from score CSV rows."""

    image_metrics = summarize_image_scores(rows)
    pixel_labels: list[np.ndarray] = []
    pixel_scores: list[np.ndarray] = []
    for row in rows:
        heatmap_path = row.get("heatmap_path") or ""
        if not heatmap_path:
            continue
        heatmap = np.load(heatmap_path).astype(np.float32, copy=False)
        mask_path = row.get("mask_path") or ""
        if mask_path:
            mask = load_binary_mask(mask_path, size=(heatmap.shape[1], heatmap.shape[0]))
        else:
            mask = np.zeros(heatmap.shape, dtype=np.uint8)
        if mask.shape != heatmap.shape:
            mask = _resize_mask(mask, heatmap.shape)
        pixel_labels.append(mask.ravel())
        pixel_scores.append(heatmap.ravel())

    metrics = dict(image_metrics)
    metrics["num_pixel_images"] = float(len(pixel_scores))
    if pixel_scores:
        labels = np.concatenate(pixel_labels, axis=0)
        scores = np.concatenate(pixel_scores, axis=0)
        labels, scores = sample_pixels(labels, scores, max_pixels=max_pixels, seed=seed)
        metrics["num_pixels_evaluated"] = float(scores.shape[0])
        metrics["pixel_auroc"] = binary_roc_auc(labels, scores)
        threshold_metrics = best_f1_iou(labels, scores)
        metrics["best_pixel_f1"] = threshold_metrics["best_f1"]
        metrics["best_pixel_iou"] = threshold_metrics["best_iou"]
        metrics["best_pixel_threshold"] = threshold_metrics["best_threshold"]
        metrics["num_pixel_thresholds"] = threshold_metrics["num_thresholds"]
    else:
        metrics["pixel_auroc"] = math.nan
        metrics["best_pixel_f1"] = math.nan
        metrics["best_pixel_iou"] = math.nan
        metrics["best_pixel_threshold"] = math.nan
        metrics["num_pixel_thresholds"] = 0.0
        metrics["num_pixels_evaluated"] = 0.0
    return metrics


def resolve_score_row_paths(
    rows: list[dict[str, str]],
    base_dir: str | Path,
) -> list[dict[str, str]]:
    """Resolve relative artifact paths against the score CSV directory when needed."""

    base_dir = Path(base_dir)
    resolved_rows = []
    for row in rows:
        resolved = dict(row)
        for key in ("image_path", "mask_path", "heatmap_path", "debug_path"):
            value = resolved.get(key) or ""
            if not value:
                continue
            path = Path(value)
            if not path.is_absolute() and not path.exists():
                resolved[key] = str(base_dir / path)
        resolved_rows.append(resolved)
    return resolved_rows


def metrics_to_jsonable(metrics: dict[str, float]) -> dict[str, float | int | None]:
    """Convert NaN values to null-friendly values for strict JSON consumers."""

    jsonable: dict[str, float | int | None] = {}
    for key, value in metrics.items():
        if key.startswith("num_"):
            jsonable[key] = int(value)
        elif isinstance(value, float) and math.isnan(value):
            jsonable[key] = None
        else:
            jsonable[key] = float(value)
    return jsonable


def sample_pixels(
    labels: np.ndarray,
    scores: np.ndarray,
    max_pixels: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if max_pixels is None or max_pixels <= 0 or scores.shape[0] <= max_pixels:
        return labels, scores
    rng = np.random.default_rng(seed)
    indices = rng.choice(scores.shape[0], size=max_pixels, replace=False)
    return labels[indices], scores[indices]


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.shape[0], dtype=np.float64)
    start = 0
    while start < values.shape[0]:
        end = start + 1
        while end < values.shape[0] and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def _threshold_candidates(scores: np.ndarray, max_thresholds: int) -> np.ndarray:
    unique = np.unique(scores)
    if unique.shape[0] <= max_thresholds:
        return unique[::-1]
    return np.linspace(float(scores.max()), float(scores.min()), num=max_thresholds)


def _resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    image = Image.fromarray((mask > 0).astype(np.uint8) * 255, mode="L")
    image = image.resize((width, height), Image.Resampling.NEAREST)
    return (np.asarray(image) > 0).astype(np.uint8)
