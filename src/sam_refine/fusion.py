"""Fuse calibrated anomaly proposals with raw SAM2 masks."""

from __future__ import annotations

import math

import numpy as np


_FUSION_MODES = {"anomaly", "sam2", "intersection", "union", "selective"}


def agreement_features(anomaly_mask, sam2_mask) -> dict[str, float]:
    """Summarize overlap and relative area for two binary masks."""

    anomaly, sam2 = _prepare_masks(anomaly_mask, sam2_mask)
    intersection_pixels = float(np.sum((anomaly > 0) & (sam2 > 0)))
    union_pixels = float(np.sum((anomaly > 0) | (sam2 > 0)))
    anomaly_pixels = float(np.sum(anomaly))
    sam2_pixels = float(np.sum(sam2))
    mask_iou = intersection_pixels / union_pixels if union_pixels else 1.0
    if anomaly_pixels:
        expansion = sam2_pixels / anomaly_pixels
    else:
        expansion = 1.0 if sam2_pixels == 0.0 else math.inf
    return {
        "anomaly_pixels": anomaly_pixels,
        "sam2_pixels": sam2_pixels,
        "intersection_pixels": intersection_pixels,
        "union_pixels": union_pixels,
        "mask_iou": float(mask_iou),
        "sam2_to_anomaly_area_ratio": float(expansion),
    }


def fuse_masks(
    anomaly_mask,
    sam2_mask,
    mode: str,
    min_iou: float = 0.25,
    max_expansion: float = 2.0,
) -> np.ndarray:
    """Return one binary mask according to the requested fusion policy."""

    if mode not in _FUSION_MODES:
        raise ValueError(f"mode must be one of {sorted(_FUSION_MODES)}")
    if not np.isfinite(min_iou) or not 0.0 <= min_iou <= 1.0:
        raise ValueError("min_iou must be finite and in [0, 1]")
    if not np.isfinite(max_expansion) or max_expansion <= 0.0:
        raise ValueError("max_expansion must be positive and finite")

    anomaly, sam2 = _prepare_masks(anomaly_mask, sam2_mask)
    if mode == "anomaly":
        return anomaly
    if mode == "sam2":
        return sam2

    intersection = (anomaly & sam2).astype(np.uint8, copy=False)
    if mode == "intersection":
        return intersection
    if mode == "union":
        return (anomaly | sam2).astype(np.uint8, copy=False)

    features = agreement_features(anomaly, sam2)
    if (
        features["mask_iou"] >= min_iou
        and features["sam2_to_anomaly_area_ratio"] <= max_expansion
    ):
        return intersection
    return anomaly


def _prepare_masks(anomaly_mask, sam2_mask) -> tuple[np.ndarray, np.ndarray]:
    anomaly = np.asarray(anomaly_mask)
    sam2 = np.asarray(sam2_mask)
    if anomaly.ndim != 2 or sam2.ndim != 2:
        raise ValueError("anomaly_mask and sam2_mask must be 2-D arrays")
    if anomaly.shape != sam2.shape:
        raise ValueError("anomaly_mask and sam2_mask must have identical shapes")
    return (anomaly > 0).astype(np.uint8), (sam2 > 0).astype(np.uint8)
