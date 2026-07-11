"""Shared calibration, heatmap, and mask artifact helpers."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from evaluation.calibration import NormalThreshold


MASK_SCORE_FIELDS = [
    "sample_id",
    "category",
    "label",
    "refiner",
    "raw_mask_source",
    "sam2_model_config",
    "sam2_checkpoint_sha256",
    "image_path",
    "mask_path",
    "num_regions",
    "num_masks",
    "mask_score",
    "mask_output",
    "selected_source",
    "prompt_mode",
    "point_mode",
    "proposal_threshold",
    "sam2_prompt_threshold",
    "sam2_calibration_sha256",
    "calibration_mismatch_override",
    "anomaly_pixels",
    "sam2_pixels",
    "intersection_pixels",
    "union_pixels",
    "mask_iou",
    "sam2_to_anomaly_area_ratio",
    "selective_min_iou",
    "selective_max_expansion",
    "sam2_mask_path",
    "pred_mask_path",
    "heatmap_path",
    "debug_path",
    "calibration_quantile",
    "calibration_threshold",
    "calibration_source_split",
    "calibration_num_images",
    "calibration_num_pixels",
]

PORTABLE_PATH_FIELDS = (
    "image_path",
    "mask_path",
    "sam2_mask_path",
    "pred_mask_path",
    "heatmap_path",
    "debug_path",
)


@dataclass(frozen=True)
class CalibrationArtifact:
    threshold: NormalThreshold
    sha256: str


def load_calibration_artifact(path: str | Path | None) -> CalibrationArtifact | None:
    if path is None:
        return None
    payload = Path(path).read_bytes()
    threshold = NormalThreshold.from_dict(json.loads(payload))
    return CalibrationArtifact(
        threshold=threshold,
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def raw_mask_provenance(
    refiner: str,
    sam2_model_config: str,
    sam2_checkpoint: str | Path,
) -> dict[str, str]:
    if refiner == "fallback":
        return {
            "refiner": "fallback",
            "raw_mask_source": "fallback",
            "sam2_model_config": "",
            "sam2_checkpoint_sha256": "",
        }
    if refiner != "sam2":
        raise ValueError("refiner must be 'fallback' or 'sam2'")
    if not sam2_model_config:
        raise ValueError("sam2_model_config is required for SAM2 provenance")
    checkpoint_path = Path(sam2_checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"SAM2 checkpoint does not exist: {checkpoint_path}")
    return {
        "refiner": "sam2",
        "raw_mask_source": "sam2",
        "sam2_model_config": sam2_model_config,
        "sam2_checkpoint_sha256": _sha256_file(checkpoint_path),
    }


def load_validated_heatmap(path: str | Path, context: str) -> np.ndarray:
    heatmap = np.load(path).astype(np.float32, copy=False)
    return validate_heatmap(heatmap, context=context)


def validate_heatmap(heatmap: np.ndarray, context: str) -> np.ndarray:
    heatmap = np.asarray(heatmap, dtype=np.float32)
    if heatmap.ndim != 2:
        raise ValueError(f"{context} must be exactly 2-D; got shape {heatmap.shape}")
    if heatmap.size == 0:
        raise ValueError(f"{context} must be non-empty")
    if not np.isfinite(heatmap).all():
        raise ValueError(f"{context} must contain only finite values")
    return heatmap


def anomaly_mask_from_heatmap(heatmap: np.ndarray, threshold: float) -> np.ndarray:
    mask = heatmap >= threshold
    if threshold <= 0.0:
        mask &= heatmap > 0.0
    return mask.astype(np.uint8)


def calibration_fields(calibration: NormalThreshold | None) -> dict[str, str]:
    if calibration is None:
        return {
            "calibration_quantile": "",
            "calibration_threshold": "",
            "calibration_source_split": "",
            "calibration_num_images": "",
            "calibration_num_pixels": "",
        }
    return {
        "calibration_quantile": f"{calibration.quantile:.8f}",
        "calibration_threshold": f"{calibration.threshold:.8f}",
        "calibration_source_split": calibration.source_split,
        "calibration_num_images": str(calibration.num_images),
        "calibration_num_pixels": str(calibration.num_pixels),
    }


def write_mask_scores(rows: list[dict[str, str]], output_path: str | Path) -> Path:
    """Write paths relative to mask_scores.csv so consumers are cwd-independent."""

    output_path = Path(output_path)
    base_dir = output_path.parent.resolve()
    portable_rows = []
    for row in rows:
        portable = {field: row.get(field, "") for field in MASK_SCORE_FIELDS}
        for key in PORTABLE_PATH_FIELDS:
            value = portable.get(key) or ""
            if value:
                portable[key] = os.path.relpath(Path(value).resolve(), start=base_dir)
        portable_rows.append(portable)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MASK_SCORE_FIELDS)
        writer.writeheader()
        writer.writerows(portable_rows)
    return output_path


def save_mask(mask: np.ndarray, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask > 0).astype(np.uint8) * 255, mode="L").save(output_path)
    return output_path


def save_refinement_debug(
    image: Image.Image,
    heatmap: np.ndarray,
    mask: np.ndarray,
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    heatmap_rgb = _heatmap_to_rgb(heatmap).resize(image.size, Image.Resampling.BILINEAR)
    overlay = Image.blend(image.convert("RGB"), heatmap_rgb, alpha=0.35).convert("RGBA")
    mask_resized = _resize_mask(mask, (image.size[1], image.size[0]))
    red = np.zeros((image.size[1], image.size[0], 4), dtype=np.uint8)
    red[..., 0] = 255
    red[..., 3] = (mask_resized > 0).astype(np.uint8) * 120
    mask_rgba = Image.fromarray(red, mode="RGBA")
    Image.alpha_composite(overlay, mask_rgba).convert("RGB").save(output_path)
    return output_path


def format_float(value: float) -> str:
    return "inf" if np.isinf(value) else f"{value:.8f}"


def _heatmap_to_rgb(heatmap: np.ndarray) -> Image.Image:
    minimum = float(heatmap.min())
    maximum = float(heatmap.max())
    if maximum > minimum:
        heatmap = (heatmap - minimum) / (maximum - minimum)
    else:
        heatmap = np.zeros_like(heatmap)
    red = (heatmap * 255).astype(np.uint8)
    green = (np.clip(1.0 - np.abs(heatmap - 0.75) * 2.0, 0.0, 1.0) * 220).astype(
        np.uint8
    )
    blue = ((1.0 - heatmap) * 80).astype(np.uint8)
    return Image.fromarray(np.stack([red, green, blue], axis=-1), mode="RGB")


def _resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    image = Image.fromarray((mask > 0).astype(np.uint8) * 255, mode="L")
    image = image.resize((width, height), Image.Resampling.NEAREST)
    return (np.asarray(image) > 0).astype(np.uint8)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
