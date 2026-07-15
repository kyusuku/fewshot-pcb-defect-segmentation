"""Lossless heatmap storage shared by experiment producers and consumers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from anomaly.heatmap import PatchHeatmapComponent, PositionedHeatmap, ProjectedHeatmap


HEATMAP_STORAGE_FORMATS = ("npy", "npz_compressed", "npz_components")
_COMPONENT_FORMAT_VERSION = 1
_COMPONENT_STORAGE_KIND = "projected_patch_components"
_COMPONENT_FIELDS = {
    "format_version",
    "storage_kind",
    "source_size",
    "fusion",
    "boxes",
    "patch_scores",
    "image_sizes",
    "patch_sizes",
    "component_source_sizes",
    "content_boxes",
}


def save_heatmap(
    path: str | Path,
    heatmap: np.ndarray | ProjectedHeatmap,
    *,
    storage: str = "npy",
) -> Path:
    """Save one heatmap losslessly using the requested storage encoding."""

    path = Path(path)
    if storage == "npy":
        if path.suffix != ".npy":
            raise ValueError("npy heatmaps must use a .npy path")
        np.save(path, np.asarray(heatmap), allow_pickle=False)
    elif storage == "npz_compressed":
        if path.suffix != ".npz":
            raise ValueError("npz_compressed heatmaps must use a .npz path")
        np.savez_compressed(path, heatmap=np.asarray(heatmap))
    elif storage == "npz_components":
        _save_component_archive(path, heatmap)
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
            if payload.files == ["heatmap"]:
                return np.asarray(payload["heatmap"]).copy()
            fields = set(payload.files)
            if "storage_kind" not in fields:
                raise ValueError(
                    f"compressed heatmap at {str(path)!r} must contain exactly one 'heatmap' array"
                )
            return _load_component_archive(path, payload)
    return np.load(path, allow_pickle=False, mmap_mode=mmap_mode)


def _save_component_archive(
    path: Path,
    heatmap: np.ndarray | ProjectedHeatmap,
) -> None:
    if path.suffix != ".npz":
        raise ValueError("npz_components heatmaps must use a .npz path")
    if not isinstance(heatmap, ProjectedHeatmap):
        raise ValueError("npz_components storage requires a ProjectedHeatmap")
    components = [item.component for item in heatmap.components]
    shapes = {component.patch_scores.shape for component in components}
    if len(shapes) != 1:
        raise ValueError("component patch-score grids must share one shape")
    np.savez_compressed(
        path,
        format_version=np.array(_COMPONENT_FORMAT_VERSION, dtype=np.int64),
        storage_kind=np.array(_COMPONENT_STORAGE_KIND),
        source_size=np.asarray(heatmap.source_size, dtype=np.int64),
        fusion=np.array(heatmap.fusion),
        boxes=np.asarray([item.box for item in heatmap.components], dtype=np.int64),
        patch_scores=np.stack(
            [component.patch_scores for component in components],
            axis=0,
        ),
        image_sizes=np.asarray(
            [component.image_size for component in components],
            dtype=np.int64,
        ),
        patch_sizes=np.asarray(
            [component.patch_size for component in components],
            dtype=np.int64,
        ),
        component_source_sizes=np.asarray(
            [component.source_size for component in components],
            dtype=np.int64,
        ),
        content_boxes=np.asarray(
            [component.content_box for component in components],
            dtype=np.int64,
        ),
    )


def _load_component_archive(path: Path, payload) -> np.ndarray:
    fields = set(payload.files)
    if fields != _COMPONENT_FIELDS:
        missing = sorted(_COMPONENT_FIELDS - fields)
        extra = sorted(fields - _COMPONENT_FIELDS)
        raise ValueError(
            f"component heatmap schema mismatch at {str(path)!r}: missing={missing}, extra={extra}"
        )
    version = _integer_scalar(payload, "format_version")
    if version != _COMPONENT_FORMAT_VERSION:
        raise ValueError(f"unsupported component heatmap format version: {version}")
    storage_kind = _string_scalar(payload, "storage_kind")
    if storage_kind != _COMPONENT_STORAGE_KIND:
        raise ValueError(f"unsupported component heatmap storage kind: {storage_kind!r}")
    fusion = _string_scalar(payload, "fusion")
    source_size = _integer_array(payload, "source_size", (2,))
    patch_scores = _typed_array(payload, "patch_scores", np.dtype(np.float32), ndim=3)
    count = patch_scores.shape[0]
    if count == 0 or patch_scores.shape[1] == 0 or patch_scores.shape[2] == 0:
        raise ValueError("component heatmap patch_scores must be non-empty")
    boxes = _integer_array(payload, "boxes", (count, 4))
    image_sizes = _integer_array(payload, "image_sizes", (count, 2))
    patch_sizes = _integer_array(payload, "patch_sizes", (count,))
    component_source_sizes = _integer_array(
        payload,
        "component_source_sizes",
        (count, 2),
    )
    content_boxes = _integer_array(payload, "content_boxes", (count, 4))
    components = tuple(
        PositionedHeatmap(
            tuple(int(value) for value in boxes[index]),
            PatchHeatmapComponent(
                patch_scores=patch_scores[index],
                image_size=tuple(int(value) for value in image_sizes[index]),
                patch_size=int(patch_sizes[index]),
                source_size=tuple(int(value) for value in component_source_sizes[index]),
                content_box=tuple(int(value) for value in content_boxes[index]),
            ),
        )
        for index in range(count)
    )
    return ProjectedHeatmap(
        source_size=tuple(int(value) for value in source_size),
        fusion=fusion,
        components=components,
    ).render()


def _integer_scalar(payload, field: str) -> int:
    value = np.asarray(payload[field])
    if value.shape != () or value.dtype.kind not in {"i", "u"}:
        raise ValueError(f"component heatmap {field} must be an integer scalar")
    return int(value)


def _string_scalar(payload, field: str) -> str:
    value = np.asarray(payload[field])
    if value.shape != () or value.dtype.kind not in {"U", "S"}:
        raise ValueError(f"component heatmap {field} must be a string scalar")
    item = value.item()
    return item.decode("utf-8") if isinstance(item, bytes) else str(item)


def _integer_array(payload, field: str, shape: tuple[int, ...]) -> np.ndarray:
    value = np.asarray(payload[field])
    if value.dtype.kind not in {"i", "u"} or value.shape != shape:
        raise ValueError(f"component heatmap {field} must be an integer array with shape {shape}")
    return value


def _typed_array(payload, field: str, dtype: np.dtype, *, ndim: int) -> np.ndarray:
    value = np.asarray(payload[field])
    if value.dtype != dtype or value.ndim != ndim:
        raise ValueError(f"component heatmap {field} must have dtype {dtype} and {ndim} dimensions")
    return value
