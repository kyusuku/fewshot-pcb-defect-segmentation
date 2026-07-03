#!/usr/bin/env python
"""Create unified train/val/test manifest CSVs for VisA PCB and DeepPCB."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from datasets.deeppcb import DeepPCBDataset, load_deeppcb_boxes
from datasets.manifests import (
    apply_deeppcb_train_test_split,
    build_kfold_manifest_rows,
    split_train_validation,
    write_fold_manifest_csv,
    write_manifest_csv,
)
from datasets.visa import DEFAULT_PCB_CATEGORIES, VisAPCBDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--visa-root",
        type=Path,
        help="Root of prepared VisA or VisA_pytorch data.",
    )
    parser.add_argument(
        "--visa-manifest-csv",
        type=Path,
        help="Optional VisA annotation/split CSV. If omitted, the loader scans the tree.",
    )
    parser.add_argument(
        "--visa-category",
        action="append",
        dest="visa_categories",
        help="PCB category to include. Repeatable. Defaults to pcb1-pcb4.",
    )
    parser.add_argument("--deeppcb-root", type=Path, help="Root of DeepPCB PCBData directory.")
    parser.add_argument(
        "--deeppcb-train-split-file",
        type=Path,
        help=(
            "Optional official DeepPCB trainval.txt file. "
            "Defaults to <deeppcb-root>/trainval.txt if present."
        ),
    )
    parser.add_argument(
        "--deeppcb-test-split-file",
        type=Path,
        help=(
            "Optional official DeepPCB test.txt file. "
            "Defaults to <deeppcb-root>/test.txt if present."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/manifests"))
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument(
        "--skip-folds",
        action="store_true",
        help="Do not write five-fold CV manifests.",
    )
    parser.add_argument("--seed", type=int, default=4880)
    parser.add_argument(
        "--deeppcb-train-count",
        type=int,
        default=1000,
        help="Official DeepPCB uses first 1000 sorted pairs for train and remaining for test.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    wrote_any = False

    if args.visa_root:
        categories = tuple(args.visa_categories or DEFAULT_PCB_CATEGORIES)
        visa_records = []
        for split in ("train", "test"):
            try:
                dataset = VisAPCBDataset(
                    root=args.visa_root,
                    manifest_csv=args.visa_manifest_csv,
                    categories=categories,
                    split=split,
                    return_tensors=False,
                )
            except RuntimeError:
                continue
            visa_records.extend(dataset.records)
        if not visa_records:
            raise RuntimeError(f"No VisA records found under {args.visa_root}")
        if not args.skip_folds:
            fold_rows = build_kfold_manifest_rows(
                visa_records,
                num_folds=args.num_folds,
                seed=args.seed,
                stratify_by="category",
            )
            fold_output_path = write_fold_manifest_csv(
                fold_rows,
                args.output_dir / "visa_pcb_folds.csv",
            )
            _print_fold_summary("visa_pcb", fold_output_path, fold_rows)
        visa_records = split_train_validation(
            visa_records,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
        output_path = write_manifest_csv(visa_records, args.output_dir / "visa_pcb_manifest.csv")
        _print_split_summary("visa_pcb", output_path, visa_records)
        wrote_any = True

    if args.deeppcb_root:
        train_split_file = args.deeppcb_train_split_file or args.deeppcb_root / "trainval.txt"
        test_split_file = args.deeppcb_test_split_file or args.deeppcb_root / "test.txt"
        if train_split_file.exists() and test_split_file.exists():
            deeppcb_train = DeepPCBDataset(
                root=args.deeppcb_root,
                split="train",
                split_file=train_split_file,
                return_tensors=False,
            )
            deeppcb_test = DeepPCBDataset(
                root=args.deeppcb_root,
                split="test",
                split_file=test_split_file,
                return_tensors=False,
            )
            deeppcb_official_records = [*deeppcb_train.records, *deeppcb_test.records]
        else:
            deeppcb = DeepPCBDataset(root=args.deeppcb_root, split="all", return_tensors=False)
            deeppcb_official_records = apply_deeppcb_train_test_split(
                deeppcb.records,
                train_count=args.deeppcb_train_count,
                val_ratio=0.0,
                seed=args.seed,
            )
        if not args.skip_folds:
            fold_rows = build_kfold_manifest_rows(
                _with_deeppcb_class_ids(deeppcb_official_records),
                num_folds=args.num_folds,
                seed=args.seed,
                stratify_by="multilabel",
            )
            fold_output_path = write_fold_manifest_csv(
                fold_rows,
                args.output_dir / "deeppcb_folds.csv",
            )
            _print_fold_summary("deeppcb", fold_output_path, fold_rows)
        deeppcb_records = split_train_validation(
            deeppcb_official_records,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
        output_path = write_manifest_csv(deeppcb_records, args.output_dir / "deeppcb_manifest.csv")
        _print_split_summary("deeppcb", output_path, deeppcb_records)
        wrote_any = True

    if not wrote_any:
        raise SystemExit("Pass --visa-root, --deeppcb-root, or both.")


def _print_split_summary(dataset_name, output_path, records) -> None:
    counts: dict[str, int] = {}
    for record in records:
        counts[record.split] = counts.get(record.split, 0) + 1
    print(f"{dataset_name}: wrote {len(records)} rows to {output_path}")
    print(f"{dataset_name}: split counts {dict(sorted(counts.items()))}")


def _print_fold_summary(dataset_name, output_path, rows) -> None:
    counts: dict[tuple[int, str], int] = {}
    for row in rows:
        key = (int(row["fold_id"]), str(row["fold_split"]))
        counts[key] = counts.get(key, 0) + 1
    print(f"{dataset_name}: wrote {len(rows)} fold rows to {output_path}")
    print(f"{dataset_name}: fold counts {dict(sorted(counts.items()))}")


def _with_deeppcb_class_ids(records):
    enriched = []
    for record in records:
        class_ids = []
        if record.box_path:
            class_ids = sorted(box.class_id for box in load_deeppcb_boxes(record.box_path))
        metadata = {**record.metadata, "class_ids": class_ids}
        enriched.append(replace(record, metadata=metadata))
    return enriched


if __name__ == "__main__":
    main()
