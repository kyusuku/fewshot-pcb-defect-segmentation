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
