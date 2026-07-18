#!/usr/bin/env python
"""Fit a normal-only pixel threshold from validation heatmaps."""

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

from evaluation.calibration import fit_normal_threshold
from evaluation.metrics import resolve_score_row_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores-csv", type=Path, required=True)
    parser.add_argument("--quantile", type=float, default=0.995)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.scores_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows = resolve_score_row_paths(rows, args.scores_csv.parent)
    result = fit_normal_threshold(rows, quantile=args.quantile)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result.to_dict(), indent=2) + "\n")
    print(f"threshold: {result.threshold:.8f}")


if __name__ == "__main__":
    main()
