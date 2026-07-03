"""Convert anomaly heatmaps into SAM-style point and box prompts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PromptRegion:
    """Candidate anomalous region represented as an exclusive xyxy box."""

    box_xyxy: tuple[int, int, int, int]
    point_xy: tuple[float, float]
    area: int
    score: float


def heatmap_to_prompt_regions(
    heatmap: np.ndarray,
    threshold: float | None = None,
    percentile: float = 95.0,
    min_area: int = 8,
    max_regions: int = 8,
) -> list[PromptRegion]:
    """Extract connected high-score regions from a heatmap."""

    heatmap = np.asarray(heatmap, dtype=np.float32)
    if heatmap.ndim != 2:
        raise ValueError("heatmap must be a 2D array")
    if threshold is None:
        threshold = float(np.percentile(heatmap, percentile))
    binary = heatmap >= threshold
    if threshold <= 0:
        binary &= heatmap > 0

    regions = []
    visited = np.zeros(binary.shape, dtype=bool)
    height, width = binary.shape
    for y in range(height):
        for x in range(width):
            if visited[y, x] or not binary[y, x]:
                continue
            pixels = _flood_fill(binary, visited, x=x, y=y)
            if len(pixels) < min_area:
                continue
            ys = np.asarray([pixel[1] for pixel in pixels], dtype=np.int32)
            xs = np.asarray([pixel[0] for pixel in pixels], dtype=np.int32)
            x1 = int(xs.min())
            x2 = int(xs.max()) + 1
            y1 = int(ys.min())
            y2 = int(ys.max()) + 1
            local_scores = heatmap[ys, xs]
            score = float(local_scores.max())
            regions.append(
                PromptRegion(
                    box_xyxy=(x1, y1, x2, y2),
                    point_xy=((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                    area=int(len(pixels)),
                    score=score,
                )
            )

    regions.sort(key=lambda region: (region.score, region.area), reverse=True)
    return regions[:max_regions]


def _flood_fill(
    binary: np.ndarray,
    visited: np.ndarray,
    x: int,
    y: int,
) -> list[tuple[int, int]]:
    stack = [(x, y)]
    pixels: list[tuple[int, int]] = []
    height, width = binary.shape
    visited[y, x] = True
    while stack:
        current_x, current_y = stack.pop()
        pixels.append((current_x, current_y))
        for next_x, next_y in (
            (current_x - 1, current_y),
            (current_x + 1, current_y),
            (current_x, current_y - 1),
            (current_x, current_y + 1),
        ):
            if not (0 <= next_x < width and 0 <= next_y < height):
                continue
            if visited[next_y, next_x] or not binary[next_y, next_x]:
                continue
            visited[next_y, next_x] = True
            stack.append((next_x, next_y))
    return pixels
