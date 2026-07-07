#!/usr/bin/env python
"""Create tiny synthetic VisA-like and DeepPCB-like data for smoke tests."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils.synthetic_data import create_synthetic_debug_datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/debug_fixture"),
        help="Directory where the synthetic fixture should be written.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete an existing output directory before writing the fixture.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)

    fixture = create_synthetic_debug_datasets(output_dir)
    print(f"root: {fixture.root}")
    print(f"visa_root: {fixture.visa_root}")
    print(f"deeppcb_root: {fixture.deeppcb_root}")
    print()
    print("VisA debug command:")
    print(
        "PYTHONPATH=src python scripts/debug_dataset.py "
        f"--config configs/datasets/visa_pcb.yaml --root {fixture.visa_root} "
        "--split test --category pcb1 --limit 2"
    )
    print()
    print("DeepPCB debug command:")
    print(
        "PYTHONPATH=src python scripts/debug_dataset.py "
        f"--config configs/datasets/deeppcb.yaml --root {fixture.deeppcb_root} "
        "--split test --limit 1"
    )


if __name__ == "__main__":
    main()
