#!/usr/bin/env python
"""Evaluate saved anomaly heatmaps from a baseline scores CSV."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from evaluation.calibration import NormalThreshold
from evaluation.metrics import (
    evaluate_heatmap_rows,
    evaluate_heatmap_rows_at_threshold,
    metrics_to_jsonable,
    resolve_score_row_paths,
)


PER_IMAGE_FIELDS = [
    "sample_id",
    "category",
    "label",
    "threshold",
    "mask_precision",
    "mask_recall",
    "mask_f1",
    "mask_iou",
    "pred_positive_pixels",
    "gt_positive_pixels",
    "true_positive_pixels",
    "false_positive_pixels",
    "false_negative_pixels",
    "heatmap_path",
    "mask_path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=1_000_000,
        help="Deterministically sample at most this many pixels for pixel metrics. Use 0 for all.",
    )
    parser.add_argument("--seed", type=int, default=4880)
    parser.add_argument("--calibration-json", type=Path)
    parser.add_argument("--per-image-csv", type=Path)
    args = parser.parse_args()
    if args.per_image_csv and not args.calibration_json:
        parser.error("--per-image-csv requires --calibration-json")
    return args


def main() -> None:
    args = parse_args()
    rows = read_score_rows(args.scores_csv)
    rows = resolve_score_row_paths(rows, base_dir=args.scores_csv.parent)
    max_pixels = None if args.max_pixels == 0 else args.max_pixels
    metrics = evaluate_heatmap_rows(rows, max_pixels=max_pixels, seed=args.seed)
    if args.calibration_json:
        calibration = NormalThreshold.from_dict(json.loads(args.calibration_json.read_text()))
        calibrated_metrics, per_image = evaluate_heatmap_rows_at_threshold(
            rows, calibration.threshold
        )
        metrics.update(calibrated_metrics)
        metrics["calibration_quantile"] = calibration.quantile
        metrics["calibration_threshold"] = calibration.threshold
        metrics["calibration_source_split"] = calibration.source_split
        metrics["calibration_num_images"] = calibration.num_images
        metrics["calibration_num_pixels"] = calibration.num_pixels
        if args.per_image_csv:
            write_per_image_csv(per_image, args.per_image_csv)
    jsonable = metrics_to_jsonable(metrics)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(jsonable, indent=2, sort_keys=True) + "\n")
    print(json.dumps(jsonable, indent=2, sort_keys=True))


def read_score_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_per_image_csv(
    rows: list[dict[str, str | float]], output_path: str | Path
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PER_IMAGE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


if __name__ == "__main__":
    main()
