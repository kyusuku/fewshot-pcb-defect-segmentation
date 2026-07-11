"""Normal-only threshold calibration for anomaly heatmaps."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NormalThreshold:
    quantile: float
    threshold: float
    num_images: int
    num_pixels: int
    source_split: str = "val"

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "quantile": self.quantile,
            "threshold": self.threshold,
            "num_images": self.num_images,
            "num_pixels": self.num_pixels,
            "source_split": self.source_split,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> NormalThreshold:
        quantile = float(payload["quantile"])
        threshold = float(payload["threshold"])
        num_images = int(payload["num_images"])
        num_pixels = int(payload["num_pixels"])
        source_split = str(payload["source_split"])
        if source_split != "val":
            raise ValueError("source_split must be 'val'")
        if not np.isfinite(quantile) or not 0.0 < quantile < 1.0:
            raise ValueError("quantile must be in (0, 1)")
        if not np.isfinite(threshold):
            raise ValueError("threshold must be finite")
        if num_images <= 0:
            raise ValueError("num_images must be positive")
        if num_pixels <= 0:
            raise ValueError("num_pixels must be positive")
        return cls(
            quantile=quantile,
            threshold=threshold,
            num_images=num_images,
            num_pixels=num_pixels,
            source_split=source_split,
        )


def fit_normal_threshold(
    rows: list[dict[str, str]],
    quantile: float = 0.995,
) -> NormalThreshold:
    """Fit a pixel threshold using explicitly normal validation heatmaps only."""

    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be in (0, 1)")
    if not rows:
        raise ValueError("rows must not be empty")

    for row_index, row in enumerate(rows):
        label = row.get("label")
        fold_split = row.get("fold_split")
        if label != "0" or fold_split != "val":
            raise ValueError(
                f"row {row_index} is not normal validation data: "
                f"label={label!r}, fold_split={fold_split!r}"
            )

    heatmaps = [
        _load_heatmap(row_index=row_index, path=row["heatmap_path"])
        for row_index, row in enumerate(rows)
    ]
    pixels = np.concatenate(heatmaps)
    return NormalThreshold(
        quantile=float(quantile),
        threshold=float(np.quantile(pixels, quantile)),
        num_images=len(rows),
        num_pixels=int(pixels.size),
    )


def _load_heatmap(row_index: int, path: str) -> np.ndarray:
    heatmap = np.load(path)
    context = f"heatmap row {row_index} at {path!r}"
    if heatmap.ndim != 2:
        raise ValueError(f"{context} must be exactly 2-D; got shape {heatmap.shape}")
    if heatmap.size == 0:
        raise ValueError(f"{context} must be non-empty")
    if not np.isfinite(heatmap).all():
        raise ValueError(f"{context} must contain only finite values")
    return heatmap.ravel()
