from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from anomaly.heatmap import PatchHeatmapComponent, PositionedHeatmap, ProjectedHeatmap
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


def _component(
    scores: np.ndarray,
    source_size: tuple[int, int],
) -> PatchHeatmapComponent:
    return PatchHeatmapComponent(
        patch_scores=scores,
        image_size=(4, 4),
        patch_size=2,
        source_size=source_size,
        content_box=(0, 0, 4, 4),
    )


def _projected_heatmap() -> ProjectedHeatmap:
    return ProjectedHeatmap(
        source_size=(1400, 1000),
        fusion="max",
        components=(
            PositionedHeatmap(
                (0, 0, 1400, 1000),
                _component(
                    np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32),
                    (1400, 1000),
                ),
            ),
            PositionedHeatmap(
                (100, 100, 868, 868),
                _component(
                    np.array([[3.0, 2.0], [1.0, 0.0]], dtype=np.float32),
                    (768, 768),
                ),
            ),
        ),
    )


def test_component_archive_round_trip_is_exact_and_compact(tmp_path: Path) -> None:
    projected = _projected_heatmap()
    path = tmp_path / "heatmap.npz"

    saved = save_heatmap(path, projected, storage="npz_components")

    expected = projected.render()
    np.testing.assert_array_equal(load_heatmap(saved), expected)
    assert load_heatmap(saved).dtype == np.float32
    assert saved.stat().st_size < expected.nbytes // 4


def _read_payload(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {field: np.asarray(payload[field]).copy() for field in payload.files}


def test_component_archive_rejects_unknown_version_and_schema_drift(tmp_path: Path) -> None:
    valid = tmp_path / "valid.npz"
    save_heatmap(valid, _projected_heatmap(), storage="npz_components")
    payload = _read_payload(valid)

    unknown = tmp_path / "unknown.npz"
    np.savez_compressed(unknown, **{**payload, "format_version": np.array(2)})
    with pytest.raises(ValueError, match="format version"):
        load_heatmap(unknown)

    missing = tmp_path / "missing.npz"
    missing_payload = dict(payload)
    missing_payload.pop("fusion")
    np.savez_compressed(missing, **missing_payload)
    with pytest.raises(ValueError, match="schema"):
        load_heatmap(missing)

    extra = tmp_path / "extra.npz"
    np.savez_compressed(extra, **{**payload, "unexpected": np.array(1)})
    with pytest.raises(ValueError, match="schema"):
        load_heatmap(extra)


def test_component_archive_rejects_unsafe_or_invalid_values(tmp_path: Path) -> None:
    valid = tmp_path / "valid.npz"
    save_heatmap(valid, _projected_heatmap(), storage="npz_components")
    payload = _read_payload(valid)

    unsafe = tmp_path / "unsafe.npz"
    np.savez_compressed(
        unsafe,
        **{**payload, "patch_scores": np.array([[object()]], dtype=object)},
    )
    with pytest.raises(ValueError, match="object|pickle"):
        load_heatmap(unsafe)

    nonfinite = tmp_path / "nonfinite.npz"
    bad_scores = payload["patch_scores"].copy()
    bad_scores[0, 0, 0] = np.nan
    np.savez_compressed(nonfinite, **{**payload, "patch_scores": bad_scores})
    with pytest.raises(ValueError, match="finite"):
        load_heatmap(nonfinite)

    bad_fusion = tmp_path / "bad_fusion.npz"
    np.savez_compressed(bad_fusion, **{**payload, "fusion": np.array("median")})
    with pytest.raises(ValueError, match="fusion"):
        load_heatmap(bad_fusion)

    bad_geometry = tmp_path / "bad_geometry.npz"
    boxes = payload["boxes"].copy()
    boxes[1, 2] += 1
    np.savez_compressed(bad_geometry, **{**payload, "boxes": boxes})
    with pytest.raises(ValueError, match="source_size|box"):
        load_heatmap(bad_geometry)
