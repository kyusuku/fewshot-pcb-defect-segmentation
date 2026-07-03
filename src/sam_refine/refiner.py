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
        predictor = self._load_predictor()
        predictor.set_image(np.asarray(image.convert("RGB")))
        predictions = []
        for region in regions:
            box = np.asarray(region.box_xyxy, dtype=np.float32)
            masks, scores, _ = predictor.predict(
                box=box,
                multimask_output=self.multimask_output,
            )
            best_index = int(np.argmax(scores))
            mask = masks[best_index].astype(np.uint8)
            predictions.append(
                MaskPrediction(
                    mask=mask,
                    score=float(scores[best_index]),
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
