"""Lossless heatmap storage shared by experiment producers and consumers."""

from __future__ import annotations

from pathlib import Path

import numpy as np


HEATMAP_STORAGE_FORMATS = ("npy", "npz_compressed")


def save_heatmap(
    path: str | Path,
    heatmap: np.ndarray,
    *,
    storage: str = "npy",
) -> Path:
    """Save one heatmap losslessly using the requested storage encoding."""

    path = Path(path)
    array = np.asarray(heatmap)
    if storage == "npy":
        if path.suffix != ".npy":
            raise ValueError("npy heatmaps must use a .npy path")
        np.save(path, array, allow_pickle=False)
    elif storage == "npz_compressed":
        if path.suffix != ".npz":
            raise ValueError("npz_compressed heatmaps must use a .npz path")
        np.savez_compressed(path, heatmap=array)
    else:
        raise ValueError(f"unsupported heatmap storage: {storage!r}")
    return path


def load_heatmap(
    path: str | Path,
    *,
    mmap_mode: str | None = None,
) -> np.ndarray:
    """Load legacy NPY or losslessly compressed NPZ heatmaps."""

    path = Path(path)
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as payload:
            if payload.files != ["heatmap"]:
                raise ValueError(
                    f"compressed heatmap at {str(path)!r} must contain exactly one 'heatmap' array"
                )
            return np.asarray(payload["heatmap"]).copy()
    return np.load(path, allow_pickle=False, mmap_mode=mmap_mode)
