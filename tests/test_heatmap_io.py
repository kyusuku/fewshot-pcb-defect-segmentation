from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from utils.heatmap_io import load_heatmap, save_heatmap


def test_compressed_heatmap_round_trip_is_lossless(tmp_path: Path) -> None:
    heatmap = np.linspace(0.0, 2.0, num=35, dtype=np.float32).reshape(5, 7)
    path = tmp_path / "heatmap.npz"

    saved = save_heatmap(path, heatmap, storage="npz_compressed")

    assert saved == path
    assert path.is_file()
    loaded = load_heatmap(path)
    assert loaded.dtype == np.float32
    np.testing.assert_array_equal(loaded, heatmap)


def test_heatmap_loader_keeps_npy_compatibility(tmp_path: Path) -> None:
    heatmap = np.arange(12, dtype=np.float32).reshape(3, 4)
    path = tmp_path / "heatmap.npy"
    np.save(path, heatmap)

    np.testing.assert_array_equal(load_heatmap(path, mmap_mode="r"), heatmap)


def test_compressed_heatmap_requires_one_named_array(tmp_path: Path) -> None:
    path = tmp_path / "ambiguous.npz"
    np.savez_compressed(path, first=np.zeros((2, 2)), second=np.ones((2, 2)))

    with pytest.raises(ValueError, match="exactly one 'heatmap'"):
        load_heatmap(path)
