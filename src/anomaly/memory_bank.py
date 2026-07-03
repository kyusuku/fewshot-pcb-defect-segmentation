"""Nearest-neighbor anomaly scoring over patch features."""

from __future__ import annotations

import numpy as np

from features.dinov2 import PatchFeatureMap


def build_memory_bank(
    feature_maps: list[PatchFeatureMap],
    normalize: bool = True,
) -> np.ndarray:
    """Stack normal patch features into one memory bank."""

    if not feature_maps:
        raise ValueError("feature_maps must contain at least one support image")
    patches = [feature_map.flatten() for feature_map in feature_maps]
    memory_bank = np.concatenate(patches, axis=0).astype(np.float32, copy=False)
    return l2_normalize(memory_bank) if normalize else memory_bank


def score_patch_features(
    query_features: PatchFeatureMap,
    memory_bank: np.ndarray,
    normalize: bool = True,
    chunk_size: int = 4096,
) -> np.ndarray:
    """Return nearest-memory distance for each query patch as a 2D grid."""

    if memory_bank.ndim != 2 or memory_bank.shape[0] == 0:
        raise ValueError("memory_bank must be a non-empty [num_patches, dim] array")

    query = query_features.flatten().astype(np.float32, copy=False)
    bank = memory_bank.astype(np.float32, copy=False)
    if normalize:
        query = l2_normalize(query)
        bank = l2_normalize(bank)

    distances = []
    bank_norms = np.sum(bank * bank, axis=1)[None, :]
    for start in range(0, query.shape[0], chunk_size):
        chunk = query[start : start + chunk_size]
        chunk_norms = np.sum(chunk * chunk, axis=1)[:, None]
        squared = np.maximum(chunk_norms + bank_norms - 2.0 * chunk @ bank.T, 0.0)
        distances.append(np.sqrt(np.min(squared, axis=1)))

    flat_scores = np.concatenate(distances, axis=0)
    grid_h, grid_w = query_features.features.shape[:2]
    return flat_scores.reshape(grid_h, grid_w).astype(np.float32, copy=False)


def l2_normalize(array: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.maximum(norms, eps)
