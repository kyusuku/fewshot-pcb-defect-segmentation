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
from sam_refine.fusion import agreement_features, fuse_masks
from utils.image import load_binary_mask, load_rgb_image
from utils.visualize import safe_filename

from run_mask_refinement import (
    anomaly_mask_from_heatmap,
    calibration_fields,
    load_calibration,
    save_mask,
    save_refinement_debug,
    write_mask_scores,
)


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    calibration = load_calibration(args.calibration_json)
    if calibration is None:  # pragma: no cover - argparse always supplies it.
        raise RuntimeError("--calibration-json is required")
    rows = read_rows(args.mask_scores_csv)
    _validate_required_fields(rows)
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
        heatmap = np.load(heatmap_path).astype(np.float32, copy=False)
        anomaly_mask = anomaly_mask_from_heatmap(heatmap, calibration.threshold)
        sam2_mask = load_binary_mask(sam2_mask_path)
        pred_mask = fuse_masks(
            anomaly_mask,
            sam2_mask,
            mode=args.mask_output,
            min_iou=args.selective_min_iou,
            max_expansion=args.selective_max_expansion,
        )
        agreement = agreement_features(anomaly_mask, sam2_mask)
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
                "mask_output": args.mask_output,
                "proposal_threshold": f"{calibration.threshold:.8f}",
                **{key: _format_float(value) for key, value in agreement.items()},
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


def _required_existing_path(row: dict[str, str], index: int, field: str) -> str:
    value = row.get(field) or ""
    if not value:
        raise ValueError(f"row {index} must have a nonempty {field}")
    if not Path(value).is_file():
        raise FileNotFoundError(f"row {index} {field} does not exist: {value}")
    return value


def _format_float(value: float) -> str:
    return "inf" if np.isinf(value) else f"{value:.8f}"


if __name__ == "__main__":
    main()
