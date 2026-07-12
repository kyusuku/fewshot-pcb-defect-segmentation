#!/usr/bin/env python
"""Fuse saved raw SAM2 masks with calibrated anomaly heatmaps offline."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from evaluation.masks import resolve_mask_row_paths
from sam_refine.artifacts import (
    anomaly_mask_from_heatmap,
    calibration_fields,
    format_float,
    load_calibration_artifact,
    load_validated_heatmap,
    save_mask,
    save_refinement_debug,
    write_mask_scores,
)
from sam_refine.fusion import fuse_masks, fuse_masks_with_metadata
from utils.image import load_binary_mask, load_rgb_image
from utils.visualize import safe_filename


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mask-scores-csv", type=Path, required=True)
    parser.add_argument("--calibration-json", type=Path, required=True)
    parser.add_argument(
        "--mask-output",
        choices=("anomaly", "intersection", "union", "selective"),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selective-min-iou", type=float, default=0.25)
    parser.add_argument("--selective-max-expansion", type=float, default=2.0)
    parser.add_argument("--allow-calibration-mismatch", action="store_true")
    parser.add_argument(
        "--smoke-debug-fallback",
        action="store_true",
        help="Allow only fallback-owned raw masks for non-paper smoke/debug fusion.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    calibration_artifact = load_calibration_artifact(args.calibration_json)
    if calibration_artifact is None:  # pragma: no cover - argparse always supplies it.
        raise RuntimeError("--calibration-json is required")
    calibration = calibration_artifact.threshold
    rows = read_rows(args.mask_scores_csv)
    if not rows:
        raise ValueError("mask scores CSV must contain at least one row")
    _validate_required_fields(rows)
    _validate_sam2_source_identity(rows, allow_smoke_fallback=args.smoke_debug_fallback)
    _validate_calibration_identity(
        rows,
        threshold=calibration.threshold,
        sha256=calibration_artifact.sha256,
        allow_mismatch=args.allow_calibration_mismatch,
    )
    rows = resolve_mask_row_paths(rows, base_dir=args.mask_scores_csv.parent)
    fuse_masks(
        np.zeros((1, 1), dtype=np.uint8),
        np.zeros((1, 1), dtype=np.uint8),
        mode=args.mask_output,
        min_iou=args.selective_min_iou,
        max_expansion=args.selective_max_expansion,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_rows = []

    for index, row in enumerate(rows):
        heatmap_path = _required_existing_path(row, index, "heatmap_path")
        sam2_mask_path = _required_existing_path(row, index, "sam2_mask_path")
        heatmap = load_validated_heatmap(
            heatmap_path,
            context=(
                f"heatmap row {index} sample_id={row.get('sample_id', '')!r} "
                f"at {heatmap_path!r}"
            ),
        )
        anomaly_mask = anomaly_mask_from_heatmap(heatmap, calibration.threshold)
        sam2_mask = load_binary_mask(sam2_mask_path)
        pred_mask, selected_source, agreement = fuse_masks_with_metadata(
            anomaly_mask,
            sam2_mask,
            mode=args.mask_output,
            min_iou=args.selective_min_iou,
            max_expansion=args.selective_max_expansion,
        )
        output_stem = f"{index:03d}_{safe_filename(row.get('sample_id', 'sample'))}"
        pred_mask_path = args.output_dir / f"{output_stem}_pred_mask.png"
        save_mask(pred_mask, pred_mask_path)

        debug_path = ""
        image_path = row.get("image_path") or ""
        if image_path:
            image_path = _required_existing_path(row, index, "image_path")
            debug_path = args.output_dir / f"{output_stem}_refined.png"
            save_refinement_debug(
                load_rgb_image(image_path),
                heatmap,
                pred_mask,
                debug_path,
            )

        updated = dict(row)
        updated.update(
            {
                "evidence_class": (
                    "smoke_debug_only" if args.smoke_debug_fallback else "paper_evidence"
                ),
                "mask_output": args.mask_output,
                "selected_source": selected_source,
                "proposal_threshold": f"{calibration.threshold:.8f}",
                "calibration_mismatch_override": (
                    "1" if args.allow_calibration_mismatch else "0"
                ),
                **{key: format_float(value) for key, value in agreement.items()},
                "selective_min_iou": f"{args.selective_min_iou:.8f}",
                "selective_max_expansion": f"{args.selective_max_expansion:.8f}",
                "pred_mask_path": str(pred_mask_path),
                "debug_path": str(debug_path) if debug_path else "",
                **calibration_fields(calibration),
            }
        )
        output_rows.append(updated)
        print(f"[{index + 1}/{len(rows)}] {row.get('sample_id', '')} mode={args.mask_output}")

    write_mask_scores(output_rows, args.output_dir / "mask_scores.csv")
    print(f"outputs: {args.output_dir}")


def read_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def _validate_required_fields(rows: list[dict[str, str]]) -> None:
    for index, row in enumerate(rows):
        for field in ("sam2_mask_path", "heatmap_path"):
            if not (row.get(field) or "").strip():
                raise ValueError(f"row {index} must have a nonempty {field}")


def _validate_calibration_identity(
    rows: list[dict[str, str]],
    threshold: float,
    sha256: str,
    allow_mismatch: bool,
) -> None:
    if allow_mismatch:
        return
    for index, row in enumerate(rows):
        stored_threshold = (row.get("sam2_prompt_threshold") or "").strip()
        stored_sha256 = (row.get("sam2_calibration_sha256") or "").strip()
        if not stored_threshold:
            raise ValueError(f"row {index} must have sam2_prompt_threshold")
        if not stored_sha256:
            raise ValueError(f"row {index} must have sam2_calibration_sha256")
        try:
            parsed_threshold = float(stored_threshold)
        except ValueError as exc:
            raise ValueError(
                f"row {index} has invalid sam2_prompt_threshold: {stored_threshold!r}"
            ) from exc
        if not np.isfinite(parsed_threshold) or parsed_threshold != threshold:
            raise ValueError(
                f"row {index} sam2_prompt_threshold {stored_threshold!r} does not "
                f"match supplied calibration threshold {threshold!r}"
            )
        if stored_sha256 != sha256:
            raise ValueError(
                f"row {index} sam2_calibration_sha256 {stored_sha256!r} does not "
                f"match supplied calibration SHA-256 {sha256!r}"
            )


def _validate_sam2_source_identity(
    rows: list[dict[str, str]], allow_smoke_fallback: bool = False
) -> None:
    expected_model_identity: tuple[str, str] | None = None
    for index, row in enumerate(rows):
        refiner = (row.get("refiner") or "").strip()
        raw_mask_source = (row.get("raw_mask_source") or "").strip()
        model_config = (row.get("sam2_model_config") or "").strip()
        checkpoint_sha256 = (row.get("sam2_checkpoint_sha256") or "").strip()
        if not refiner:
            raise ValueError(f"row {index} must have refiner provenance")
        if not raw_mask_source:
            raise ValueError(f"row {index} must have raw_mask_source provenance")
        expected_source = "fallback" if allow_smoke_fallback else "sam2"
        if refiner != expected_source or raw_mask_source != expected_source:
            raise ValueError(
                f"row {index} raw_mask_source must be {expected_source!r} for offline "
                f"{'smoke/debug' if allow_smoke_fallback else 'SAM2'} fusion; "
                f"got refiner={refiner!r}, raw_mask_source={raw_mask_source!r}"
            )
        if allow_smoke_fallback:
            if model_config or checkpoint_sha256:
                raise ValueError(f"row {index} fallback source must not claim a SAM2 model")
            continue
        if not model_config:
            raise ValueError(f"row {index} must have sam2_model_config")
        if len(checkpoint_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in checkpoint_sha256
        ):
            raise ValueError(
                f"row {index} must have a lowercase SHA-256 sam2_checkpoint_sha256"
            )
        model_identity = (model_config, checkpoint_sha256)
        if expected_model_identity is None:
            expected_model_identity = model_identity
        elif model_identity != expected_model_identity:
            raise ValueError(
                f"row {index} SAM2 model identity does not match earlier rows"
            )


def _required_existing_path(row: dict[str, str], index: int, field: str) -> str:
    value = row.get(field) or ""
    if not value:
        raise ValueError(f"row {index} must have a nonempty {field}")
    if not Path(value).is_file():
        raise FileNotFoundError(f"row {index} {field} does not exist: {value}")
    return value


if __name__ == "__main__":
    main()
