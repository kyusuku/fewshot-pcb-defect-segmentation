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
        return cls(
            quantile=float(payload["quantile"]),
            threshold=float(payload["threshold"]),
            num_images=int(payload["num_images"]),
            num_pixels=int(payload["num_pixels"]),
            source_split=str(payload["source_split"]),
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

    for row in rows:
        if row.get("label") != "0" or row.get("fold_split") != "val":
            raise ValueError("normal validation")

    heatmaps = [np.load(row["heatmap_path"]).ravel() for row in rows]
    pixels = np.concatenate(heatmaps)
    return NormalThreshold(
        quantile=float(quantile),
        threshold=float(np.quantile(pixels, quantile)),
        num_images=len(rows),
        num_pixels=int(pixels.size),
    )
