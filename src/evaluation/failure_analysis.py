"""Failure geometry and paired SAM2 effect descriptors."""

from __future__ import annotations

import math

import numpy as np

from evaluation.masks import mask_confusion_metrics
from sam_refine.fusion import agreement_features


def describe_mask_pair(
    target: np.ndarray,
    anomaly: np.ndarray,
    sam2: np.ndarray,
    heatmap: np.ndarray,
) -> dict[str, float]:
    """Describe one calibrated anomaly mask and its paired SAM2 mask."""

    target_mask, anomaly_mask, sam2_mask, heatmap_array = _prepare_inputs(
        target, anomaly, sam2, heatmap
    )
    components = _connected_components(target_mask)
    area = float(np.sum(target_mask))
    perimeter = _perimeter_proxy(target_mask)
    anomaly_metrics = mask_confusion_metrics(anomaly_mask, target_mask)
    sam2_metrics = mask_confusion_metrics(sam2_mask, target_mask)
    normalized = _normalize_heatmap(heatmap_array)
    if np.any(sam2_mask):
        mean_inside = float(normalized[sam2_mask].mean())
    else:
        mean_inside = 0.0
    return {
        "gt_area_fraction": float(target_mask.mean()),
        "gt_components": float(len(components)),
        "gt_thinness": float(perimeter * perimeter / max(4.0 * math.pi * area, 1.0)),
        "anomaly_f1": anomaly_metrics["f1"],
        "sam2_f1": sam2_metrics["f1"],
        "sam2_delta_f1": sam2_metrics["f1"] - anomaly_metrics["f1"],
        "anomaly_iou": anomaly_metrics["iou"],
        "sam2_iou": sam2_metrics["iou"],
        "sam2_delta_iou": sam2_metrics["iou"] - anomaly_metrics["iou"],
        "mean_anomaly_inside_sam2": mean_inside,
        **agreement_features(anomaly_mask, sam2_mask),
    }


def _prepare_inputs(
    target: np.ndarray,
    anomaly: np.ndarray,
    sam2: np.ndarray,
    heatmap: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    target_mask = np.asarray(target) > 0
    anomaly_mask = np.asarray(anomaly) > 0
    sam2_mask = np.asarray(sam2) > 0
    heatmap_array = np.asarray(heatmap, dtype=np.float32)
    if (
        target_mask.ndim != 2
        or anomaly_mask.ndim != 2
        or sam2_mask.ndim != 2
        or heatmap_array.ndim != 2
    ):
        raise ValueError("target, anomaly, sam2, and heatmap must be 2-D arrays")
    shapes = {target_mask.shape, anomaly_mask.shape, sam2_mask.shape, heatmap_array.shape}
    if len(shapes) != 1:
        raise ValueError("target, anomaly, sam2, and heatmap must have the same shape")
    if not np.isfinite(heatmap_array).all():
        raise ValueError("heatmap must contain only finite values")
    return target_mask, anomaly_mask, sam2_mask, heatmap_array


def _normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    minimum = float(heatmap.min())
    maximum = float(heatmap.max())
    if maximum <= minimum:
        return np.zeros(heatmap.shape, dtype=np.float32)
    return ((heatmap - minimum) / (maximum - minimum)).astype(np.float32, copy=False)


def _perimeter_proxy(mask: np.ndarray) -> float:
    padded = np.pad(mask.astype(bool), 1, mode="constant", constant_values=False)
    vertical = np.sum(padded[1:, :] != padded[:-1, :])
    horizontal = np.sum(padded[:, 1:] != padded[:, :-1])
    return float(vertical + horizontal)


def _connected_components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask).astype(bool)
    visited = np.zeros(mask.shape, dtype=bool)
    components: list[np.ndarray] = []
    height, width = mask.shape
    for y in range(height):
        for x in range(width):
            if visited[y, x] or not mask[y, x]:
                continue
            component = np.zeros(mask.shape, dtype=bool)
            stack = [(x, y)]
            visited[y, x] = True
            while stack:
                current_x, current_y = stack.pop()
                component[current_y, current_x] = True
                for next_x, next_y in (
                    (current_x - 1, current_y),
                    (current_x + 1, current_y),
                    (current_x, current_y - 1),
                    (current_x, current_y + 1),
                ):
                    if not (0 <= next_x < width and 0 <= next_y < height):
                        continue
                    if visited[next_y, next_x] or not mask[next_y, next_x]:
                        continue
                    visited[next_y, next_x] = True
                    stack.append((next_x, next_y))
            components.append(component)
    return components
