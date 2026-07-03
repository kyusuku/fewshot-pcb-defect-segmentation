#!/usr/bin/env python
"""Load a few dataset samples and save image/mask/box visualizations."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from datasets.deeppcb import load_deeppcb_boxes
from datasets.registry import build_dataset_from_config
from utils.config import deep_update, load_yaml
from utils.visualize import safe_filename, visualize_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to dataset YAML config.")
    parser.add_argument("--root", help="Override dataset root path.")
    parser.add_argument("--split", help="Override split: train, val, test, or all.")
    parser.add_argument("--split-file", help="Optional split file for datasets that support one.")
    parser.add_argument(
        "--category",
        action="append",
        dest="categories",
        help="Restrict VisA categories. Can be passed multiple times.",
    )
    parser.add_argument("--limit", type=int, help="Number of samples to visualize.")
    parser.add_argument("--output-dir", help="Directory for debug PNGs.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle before taking samples.")
    parser.add_argument("--seed", type=int, default=4880, help="Shuffle seed.")
    parser.add_argument("--show", action="store_true", help="Display figures interactively.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)

    dataset_overrides = {
        "root": args.root,
        "split": args.split,
        "split_file": args.split_file,
        "categories": args.categories,
        "return_tensors": False,
    }
    deep_update(config, {"dataset": dataset_overrides})

    dataset = build_dataset_from_config(config)
    debug_config = config.get("debug", {})
    limit = args.limit or int(debug_config.get("limit", 4))
    output_dir = Path(args.output_dir or debug_config.get("output_dir", "outputs/debug_dataset"))

    indices = list(range(len(dataset)))
    if args.shuffle:
        random.Random(args.seed).shuffle(indices)
    indices = indices[:limit]

    print("Dataset summary:")
    print(dataset.describe())
    print(f"Saving {len(indices)} visualizations to {output_dir}")

    for rank, index in enumerate(indices):
        record = dataset.records[index]
        boxes = load_deeppcb_boxes(record.box_path) if record.box_path else []
        filename = f"{rank:03d}_{safe_filename(record.sample_id)}.png"
        output_path = visualize_record(record, output_dir / filename, boxes=boxes, show=args.show)
        print(f"[{rank + 1}/{len(indices)}] {record.sample_id} -> {output_path}")


if __name__ == "__main__":
    main()
