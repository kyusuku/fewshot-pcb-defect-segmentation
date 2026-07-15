#!/usr/bin/env python
"""Evaluate predicted binary masks from mask refinement outputs."""

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

from evaluation.masks import (
    PER_MASK_FIELDS,
    evaluate_mask_rows,
    merge_source_score_rows,
    resolve_mask_row_paths,
)
from evaluation.metrics import metrics_to_jsonable, resolve_score_row_paths
from evaluation.reporting import relativize_report_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mask-scores-csv", type=Path, required=True)
    parser.add_argument(
        "--source-scores-csv",
        type=Path,
        default=None,
        help="Original DINOv2 scores.csv used to fill missing mask_path fields.",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_csv(args.mask_scores_csv)
    rows = resolve_mask_row_paths(rows, base_dir=args.mask_scores_csv.parent)
    if args.source_scores_csv is not None:
        source_rows = read_csv(args.source_scores_csv)
        source_rows = resolve_score_row_paths(source_rows, base_dir=args.source_scores_csv.parent)
        rows = merge_source_score_rows(rows, source_rows)

    metrics, per_row = evaluate_mask_rows(rows)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(metrics_to_jsonable(metrics), indent=2) + "\n")
    if args.output_csv is not None:
        write_per_row_csv(per_row, args.output_csv)
    print(json.dumps(metrics_to_jsonable(metrics), indent=2))


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_per_row_csv(rows: list[dict[str, str | float]], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    portable_rows = relativize_report_paths(
        rows,
        ("pred_mask_path", "mask_path"),
        output_path,
    )
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PER_MASK_FIELDS)
        writer.writeheader()
        writer.writerows(portable_rows)
    return output_path


if __name__ == "__main__":
    main()
