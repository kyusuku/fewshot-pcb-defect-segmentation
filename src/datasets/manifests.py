"""Manifest creation and deterministic split helpers."""

from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Iterable

from datasets.types import DatasetRecord

MANIFEST_FIELDS = [
    "dataset",
    "category",
    "sample_id",
    "split",
    "image_path",
    "label",
    "mask_path",
    "box_path",
    "template_path",
    "metadata_json",
]

FOLD_MANIFEST_FIELDS = [*MANIFEST_FIELDS, "fold_id", "fold_split"]


def split_train_validation(
    records: Iterable[DatasetRecord],
    val_ratio: float = 0.2,
    seed: int = 4880,
) -> list[DatasetRecord]:
    """Split existing `train` records into train/val, grouped by category.

    Non-train records are returned with their original split, so locked official
    test partitions stay untouched.
    """

    if not 0.0 <= val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in [0, 1), got {val_ratio}")

    records = list(records)
    train_by_category: dict[str, list[DatasetRecord]] = defaultdict(list)
    for record in records:
        if record.split == "train":
            train_by_category[record.category].append(record)

    val_ids: set[tuple[str, str]] = set()
    rng = random.Random(seed)
    for category, category_records in sorted(train_by_category.items()):
        if val_ratio == 0.0 or len(category_records) <= 1:
            continue
        val_count = max(1, round(len(category_records) * val_ratio))
        val_count = min(val_count, len(category_records) - 1)
        shuffled = list(category_records)
        rng.shuffle(shuffled)
        val_ids.update((category, record.sample_id) for record in shuffled[:val_count])

    split_records: list[DatasetRecord] = []
    for record in records:
        if record.split == "train" and (record.category, record.sample_id) in val_ids:
            split_records.append(replace(record, split="val"))
        else:
            split_records.append(record)
    return split_records


def apply_deeppcb_train_test_split(
    records: Iterable[DatasetRecord],
    train_count: int = 1000,
    val_ratio: float = 0.2,
    seed: int = 4880,
) -> list[DatasetRecord]:
    """Apply DeepPCB's official-style 1000 train / remaining test split.

    The official README states that 1,000 images are separated for training and
    the remaining images are used for test. The repo file names are stable, so
    this function sorts by sample ID, marks the first `train_count` as train,
    then creates a deterministic validation split from that train partition.
    """

    if train_count <= 0:
        raise ValueError(f"train_count must be positive, got {train_count}")

    sorted_records = sorted(records, key=lambda record: record.sample_id)
    split_records: list[DatasetRecord] = []
    train_records = sorted_records[:train_count]
    test_records = sorted_records[train_count:]

    val_count = 0
    if val_ratio > 0.0 and len(train_records) > 1:
        val_count = max(1, round(len(train_records) * val_ratio))
        val_count = min(val_count, len(train_records) - 1)

    # Keep ordering stable and reserve validation from the end of the train pool.
    val_ids = {record.sample_id for record in train_records[-val_count:]} if val_count else set()
    for record in train_records:
        split = "val" if record.sample_id in val_ids else "train"
        split_records.append(replace(record, split=split))
    split_records.extend(replace(record, split="test") for record in test_records)
    return split_records


def build_kfold_manifest_rows(
    records: Iterable[DatasetRecord],
    num_folds: int = 5,
    seed: int = 4880,
    stratify_by: str = "category",
) -> list[dict[str, str | int]]:
    """Expand official train/test records into fold rows.

    Official test records stay locked as `fold_split=test` for every fold.
    Official train records are marked `val` in exactly one fold and `dev` in
    the remaining folds.
    """

    if num_folds < 2:
        raise ValueError(f"num_folds must be at least 2, got {num_folds}")
    if stratify_by not in {"category", "multilabel"}:
        raise ValueError(f"Unknown stratification mode: {stratify_by!r}")

    records = list(records)
    train_records = [record for record in records if record.split == "train"]
    if stratify_by == "multilabel":
        assignments = _assign_multilabel_folds(train_records, num_folds=num_folds, seed=seed)
    else:
        assignments = _assign_category_folds(train_records, num_folds=num_folds, seed=seed)

    rows: list[dict[str, str | int]] = []
    for fold_id in range(num_folds):
        for record in records:
            row = record_to_manifest_row(record)
            row["fold_id"] = fold_id
            if record.split == "train":
                row["fold_split"] = "val" if assignments[_record_key(record)] == fold_id else "dev"
            else:
                row["fold_split"] = "test"
            rows.append(row)
    return rows


def write_fold_manifest_csv(rows: Iterable[dict[str, str | int]], output_path: str | Path) -> Path:
    """Write fold-expanded manifest rows to CSV."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FOLD_MANIFEST_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return output_path


def write_manifest_csv(records: Iterable[DatasetRecord], output_path: str | Path) -> Path:
    """Write records to a unified manifest CSV."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow(record_to_manifest_row(record))
    return output_path


def record_to_manifest_row(record: DatasetRecord) -> dict[str, str | int]:
    return {
        "dataset": record.dataset,
        "category": record.category,
        "sample_id": record.sample_id,
        "split": record.split,
        "image_path": str(record.image_path),
        "label": int(record.label),
        "mask_path": str(record.mask_path) if record.mask_path else "",
        "box_path": str(record.box_path) if record.box_path else "",
        "template_path": str(record.template_path) if record.template_path else "",
        "metadata_json": json.dumps(record.metadata, sort_keys=True),
    }


def _assign_category_folds(
    records: Iterable[DatasetRecord],
    num_folds: int,
    seed: int,
) -> dict[tuple[str, str], int]:
    rng = random.Random(seed)
    by_category: dict[str, list[DatasetRecord]] = defaultdict(list)
    for record in records:
        by_category[record.category].append(record)

    assignments: dict[tuple[str, str], int] = {}
    for category, category_records in sorted(by_category.items()):
        shuffled = sorted(category_records, key=lambda record: record.sample_id)
        rng.shuffle(shuffled)
        for index, record in enumerate(shuffled):
            assignments[_record_key(record)] = index % num_folds
    return assignments


def _assign_multilabel_folds(
    records: Iterable[DatasetRecord],
    num_folds: int,
    seed: int,
) -> dict[tuple[str, str], int]:
    rng = random.Random(seed)
    records = sorted(records, key=lambda record: record.sample_id)
    assignments: dict[tuple[str, str], int] = {}
    fold_sizes = [0 for _ in range(num_folds)]
    fold_label_counts: list[dict[str, int]] = [defaultdict(int) for _ in range(num_folds)]

    single_label_groups: dict[str, list[DatasetRecord]] = defaultdict(list)
    multi_label_records: list[DatasetRecord] = []
    for record in records:
        labels = _record_multilabels(record)
        if len(labels) == 1:
            single_label_groups[labels[0]].append(record)
        else:
            multi_label_records.append(record)

    for label, label_records in sorted(single_label_groups.items()):
        shuffled = list(label_records)
        rng.shuffle(shuffled)
        for record in shuffled:
            fold_id = min(
                range(num_folds),
                key=lambda idx: (fold_label_counts[idx][label], fold_sizes[idx]),
            )
            _assign_record_to_fold(record, fold_id, assignments, fold_sizes, fold_label_counts)

    rng.shuffle(multi_label_records)
    multi_label_records.sort(
        key=lambda record: (-len(_record_multilabels(record)), record.sample_id)
    )
    for record in multi_label_records:
        labels = _record_multilabels(record)
        fold_id = min(
            range(num_folds),
            key=lambda idx: (
                sum(fold_label_counts[idx][label] for label in labels),
                fold_sizes[idx],
            ),
        )
        _assign_record_to_fold(record, fold_id, assignments, fold_sizes, fold_label_counts)

    return assignments


def _assign_record_to_fold(
    record: DatasetRecord,
    fold_id: int,
    assignments: dict[tuple[str, str], int],
    fold_sizes: list[int],
    fold_label_counts: list[dict[str, int]],
) -> None:
    assignments[_record_key(record)] = fold_id
    fold_sizes[fold_id] += 1
    for label in _record_multilabels(record):
        fold_label_counts[fold_id][label] += 1


def _record_key(record: DatasetRecord) -> tuple[str, str]:
    return (record.category, record.sample_id)


def _record_multilabels(record: DatasetRecord) -> tuple[str, ...]:
    class_ids = record.metadata.get("class_ids")
    if isinstance(class_ids, (list, tuple, set)) and class_ids:
        return tuple(sorted(str(class_id) for class_id in class_ids))
    return (record.category,)
