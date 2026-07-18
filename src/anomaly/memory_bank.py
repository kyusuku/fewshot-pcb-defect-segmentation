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
    patches = [
        feature_map.flatten()[feature_map.valid_patch_mask().reshape(-1)]
        for feature_map in feature_maps
    ]
    memory_bank = np.concatenate(patches, axis=0).astype(np.float32, copy=False)
    if memory_bank.shape[0] == 0:
        raise ValueError("feature_maps contain no valid unpadded support patches")
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


def select_greedy_coreset(
    memory_bank: np.ndarray,
    ratio: float = 0.01,
    seed: int = 4880,
    projection_dim: int = 64,
) -> np.ndarray:
    """Select a deterministic approximate-greedy subset of patch features."""

    if (
        not isinstance(ratio, (int, float, np.integer, np.floating))
        or isinstance(ratio, (bool, np.bool_))
        or not np.isfinite(ratio)
        or not 0.0 < ratio <= 1.0
    ):
        raise ValueError("ratio must be finite and in (0, 1]")
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, (bool, np.bool_)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if (
        not isinstance(projection_dim, (int, np.integer))
        or isinstance(projection_dim, (bool, np.bool_))
        or projection_dim <= 0
    ):
        raise ValueError("projection_dim must be a positive integer")

    bank = np.asarray(memory_bank, dtype=np.float32)
    if bank.ndim != 2:
        raise ValueError("memory_bank must be a 2D array")
    if bank.shape[0] == 0:
        raise ValueError("memory_bank must be non-empty")
    if bank.shape[1] == 0:
        raise ValueError("memory_bank feature dimension must be non-empty")
    if not np.isfinite(bank).all():
        raise ValueError("memory_bank must contain only finite values")

    target = max(1, int(round(bank.shape[0] * ratio)))
    if target >= bank.shape[0]:
        return bank.copy()

    rng = np.random.default_rng(seed)
    projection = rng.standard_normal((bank.shape[1], projection_dim)).astype(np.float32)
    projected = bank @ projection / np.sqrt(float(projection_dim))
    selected = [int(rng.integers(0, bank.shape[0]))]
    minimum = np.full(bank.shape[0], np.inf, dtype=np.float32)
    while len(selected) < target:
        center = projected[selected[-1]]
        squared = np.sum((projected - center) ** 2, axis=1)
        minimum = np.minimum(minimum, squared)
        minimum[selected] = -1.0
        selected.append(int(np.argmax(minimum)))
    return bank[np.asarray(selected, dtype=np.int64)]
