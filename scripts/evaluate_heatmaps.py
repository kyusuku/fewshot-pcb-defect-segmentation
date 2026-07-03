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

from evaluation.metrics import (
    evaluate_heatmap_rows,
    metrics_to_jsonable,
    resolve_score_row_paths,
)


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_score_rows(args.scores_csv)
    rows = resolve_score_row_paths(rows, base_dir=args.scores_csv.parent)
    max_pixels = None if args.max_pixels == 0 else args.max_pixels
    metrics = evaluate_heatmap_rows(rows, max_pixels=max_pixels, seed=args.seed)
    jsonable = metrics_to_jsonable(metrics)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(jsonable, indent=2, sort_keys=True) + "\n")
    print(json.dumps(jsonable, indent=2, sort_keys=True))


def read_score_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    main()
