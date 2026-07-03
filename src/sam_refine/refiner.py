"""SAM2-compatible mask refinement interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image

from sam_refine.prompts import PromptRegion


@dataclass(frozen=True)
class MaskPrediction:
    mask: np.ndarray
    score: float
    region: PromptRegion
    source: str


class MaskRefiner(Protocol):
    def refine(
        self,
        image: Image.Image,
        heatmap: np.ndarray,
        regions: list[PromptRegion],
    ) -> list[MaskPrediction]:
        ...


class FallbackMaskRefiner:
    """Deterministic threshold-in-box refiner used when SAM2 is unavailable."""

    def __init__(self, threshold_fraction: float = 0.5) -> None:
        if not 0.0 <= threshold_fraction <= 1.0:
            raise ValueError("threshold_fraction must be in [0, 1]")
        self.threshold_fraction = threshold_fraction

    def refine(
        self,
        image: Image.Image,
        heatmap: np.ndarray,
        regions: list[PromptRegion],
    ) -> list[MaskPrediction]:
        del image
        heatmap = np.asarray(heatmap, dtype=np.float32)
        predictions = []
        for region in regions:
            x1, y1, x2, y2 = region.box_xyxy
            local = heatmap[y1:y2, x1:x2]
            if local.size == 0:
                continue
            threshold = max(float(local.max()) * self.threshold_fraction, 0.0)
            mask = np.zeros(heatmap.shape, dtype=np.uint8)
            mask[y1:y2, x1:x2] = (local >= threshold).astype(np.uint8)
            if int(mask.sum()) == 0:
                continue
            score = float(np.mean(heatmap[mask > 0]))
            predictions.append(
                MaskPrediction(
                    mask=mask,
                    score=score,
                    region=region,
                    source="fallback",
                )
            )
        return predictions


class SAM2MaskRefiner:
    """Thin lazy adapter for Meta SAM2 image predictor."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        model_config: str,
        device: str = "auto",
        multimask_output: bool = True,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.model_config = model_config
        self.device = device
        self.multimask_output = multimask_output
        self._predictor = None

    def refine(
        self,
        image: Image.Image,
        heatmap: np.ndarray,
        regions: list[PromptRegion],
    ) -> list[MaskPrediction]:
        if not regions:
            return []
        heatmap = np.asarray(heatmap, dtype=np.float32)
        if heatmap.ndim != 2:
            raise ValueError("heatmap must be a 2D array")

        predictor = self._load_predictor()
        image = image.convert("RGB")
        predictor.set_image(np.asarray(image))
        predictions = []
        for region in regions:
            box = _scale_box_to_image(
                box_xyxy=region.box_xyxy,
                heatmap_shape=heatmap.shape,
                image_size=image.size,
            )
            point_coords = _scale_point_to_image(
                point_xy=region.point_xy,
                heatmap_shape=heatmap.shape,
                image_size=image.size,
            )[None, :]
            masks, scores, _ = predictor.predict(
                point_coords=point_coords,
                point_labels=np.asarray([1], dtype=np.int32),
                box=box,
                multimask_output=self.multimask_output,
            )
            mask, score = _best_mask(masks=masks, scores=scores)
            mask = _resize_binary_mask(mask, heatmap.shape)
            predictions.append(
                MaskPrediction(
                    mask=mask,
                    score=score,
                    region=region,
                    source="sam2",
                )
            )
        return predictions

    def _load_predictor(self):
        if self._predictor is not None:
            return self._predictor
        try:
            import torch
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except Exception as exc:  # pragma: no cover - depends on optional SAM2 install.
            raise RuntimeError(
                "SAM2 is not installed. Install facebookresearch/sam2 and provide "
                "a checkpoint/config, or run with --refiner fallback."
            ) from exc

        device = self.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model = build_sam2(self.model_config, str(self.checkpoint_path), device=device)
        self._predictor = SAM2ImagePredictor(model)
        return self._predictor


def _scale_box_to_image(
    box_xyxy: tuple[int, int, int, int],
    heatmap_shape: tuple[int, int],
    image_size: tuple[int, int],
) -> np.ndarray:
    heatmap_height, heatmap_width = heatmap_shape
    image_width, image_height = image_size
    x_scale = image_width / heatmap_width
    y_scale = image_height / heatmap_height
    x1, y1, x2, y2 = box_xyxy
    box = np.asarray(
        [x1 * x_scale, y1 * y_scale, x2 * x_scale, y2 * y_scale],
        dtype=np.float32,
    )
    box[[0, 2]] = np.clip(box[[0, 2]], 0.0, float(image_width))
    box[[1, 3]] = np.clip(box[[1, 3]], 0.0, float(image_height))
    return box


def _scale_point_to_image(
    point_xy: tuple[float, float],
    heatmap_shape: tuple[int, int],
    image_size: tuple[int, int],
) -> np.ndarray:
    heatmap_height, heatmap_width = heatmap_shape
    image_width, image_height = image_size
    x_scale = image_width / heatmap_width
    y_scale = image_height / heatmap_height
    x, y = point_xy
    point = np.asarray([x * x_scale, y * y_scale], dtype=np.float32)
    point[0] = np.clip(point[0], 0.0, max(float(image_width - 1), 0.0))
    point[1] = np.clip(point[1], 0.0, max(float(image_height - 1), 0.0))
    return point


def _best_mask(masks: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, float]:
    masks = np.asarray(masks)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if masks.ndim == 2:
        masks = masks[None, :, :]
    elif masks.ndim == 4:
        masks = masks.reshape((-1, masks.shape[-2], masks.shape[-1]))
    if masks.ndim != 3:
        raise RuntimeError(f"SAM2 returned masks with unsupported shape {masks.shape}")
    if masks.shape[0] != scores.size:
        raise RuntimeError(
            "SAM2 returned a different number of masks and scores: "
            f"{masks.shape[0]} masks vs {scores.size} scores"
        )
    best_index = int(np.argmax(scores))
    return masks[best_index], float(scores[best_index])


def _resize_binary_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    mask = (np.asarray(mask) > 0).astype(np.uint8)
    if mask.shape == shape:
        return mask
    height, width = shape
    image = Image.fromarray(mask * 255, mode="L")
    image = image.resize((width, height), Image.Resampling.NEAREST)
    return (np.asarray(image) > 0).astype(np.uint8)
