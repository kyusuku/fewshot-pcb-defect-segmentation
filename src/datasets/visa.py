"""VisA PCB subset dataset loader."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Any

from datasets.sampling import sample_few_shot_normals
from datasets.types import DatasetRecord

DEFAULT_PCB_CATEGORIES = ("pcb1", "pcb2", "pcb3", "pcb4")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MASK_HINTS = {"mask", "masks", "ground_truth", "groundtruth", "annotation", "annotations", "gt"}
NORMAL_VALUES = {"0", "normal", "good", "ok", "negative", "false"}
ANOMALY_VALUES = {"1", "anomaly", "anomalous", "defect", "defective", "bad", "positive", "true"}


class VisAPCBDataset:
    """Dataset for PCB-related VisA subsets."""

    def __init__(
        self,
        root: str | Path,
        manifest_csv: str | Path | None = None,
        categories: list[str] | tuple[str, ...] = DEFAULT_PCB_CATEGORIES,
        split: str = "all",
        normal_only: bool = False,
        return_tensors: bool = True,
        mask_threshold: int = 0,
    ) -> None:
        self.root = Path(root).expanduser()
        self.manifest_csv = Path(manifest_csv).expanduser() if manifest_csv else None
        self.categories = tuple(categories)
        self.split = split
        self.normal_only = normal_only
        self.return_tensors = return_tensors
        self.mask_threshold = mask_threshold

        if not self.root.exists():
            raise FileNotFoundError(
                f"VisA root does not exist: {self.root}. "
                "Pass --root or edit configs/datasets/visa_pcb.yaml."
            )

        if self.manifest_csv:
            self.records = self._records_from_manifest(self.manifest_csv)
        else:
            self.records = self._records_from_tree()

        self.records = [record for record in self.records if self._keep_record(record)]
        if not self.records:
            raise RuntimeError(
                f"No VisA PCB samples found under {self.root} for categories={self.categories}, "
                f"split={self.split!r}, normal_only={self.normal_only}."
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        from utils.image import (
            image_to_tensor,
            load_binary_mask,
            load_rgb_image,
            mask_to_tensor,
            zeros_mask,
        )

        record = self.records[index]
        image = load_rgb_image(record.image_path)
        if record.mask_path and record.mask_path.exists():
            mask = load_binary_mask(record.mask_path, size=image.size, threshold=self.mask_threshold)
        else:
            mask = zeros_mask(image.size)

        sample: dict[str, Any] = record.to_dict()
        sample["record"] = record
        sample["image_size"] = {"width": image.size[0], "height": image.size[1]}

        if self.return_tensors:
            sample["image"] = image_to_tensor(image)
            sample["mask"] = mask_to_tensor(mask)
        else:
            sample["image"] = image
            sample["mask"] = mask

        return sample

    def describe(self) -> dict[str, Any]:
        by_category = Counter(record.category for record in self.records)
        by_split = Counter(record.split for record in self.records)
        by_label = Counter("anomaly" if record.label else "normal" for record in self.records)
        return {
            "dataset": "visa_pcb",
            "root": str(self.root),
            "manifest_csv": str(self.manifest_csv) if self.manifest_csv else None,
            "num_samples": len(self.records),
            "categories": dict(sorted(by_category.items())),
            "splits": dict(sorted(by_split.items())),
            "labels": dict(sorted(by_label.items())),
        }

    def few_shot_support(self, k: int, seed: int = 4880) -> dict[str, list[DatasetRecord]]:
        """Sample `k` normal support images per category for memory-bank construction."""

        return sample_few_shot_normals(self.records, k=k, seed=seed, categories=self.categories)

    def _records_from_manifest(self, manifest_csv: Path) -> list[DatasetRecord]:
        if not manifest_csv.exists():
            raise FileNotFoundError(f"VisA manifest CSV does not exist: {manifest_csv}")

        records: list[DatasetRecord] = []
        with manifest_csv.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row_number, row in enumerate(reader, start=2):
                try:
                    records.append(self._record_from_manifest_row(row))
                except Exception as exc:
                    raise ValueError(f"Could not parse VisA manifest row {row_number}: {row}") from exc
        return records

    def _record_from_manifest_row(self, row: dict[str, str]) -> DatasetRecord:
        image_value = _first_value(row, "image_path", "img_path", "path", "image", "filename", "file")
        if image_value is None:
            raise ValueError("Missing image path column")

        image_path = _resolve_path(self.root, image_value)
        mask_value = _first_value(row, "mask_path", "gt_path", "ground_truth", "mask", "annotation")
        mask_path = _resolve_path(self.root, mask_value) if mask_value else None

        category = _first_value(row, "category", "object", "object_type", "class", "cls")
        category = category or _infer_category(image_path, self.categories) or "unknown"

        split = _first_value(row, "split", "phase", "subset") or _infer_split(image_path) or "all"
        label_value = _first_value(row, "label", "is_anomaly", "anomaly", "target", "defect")
        label = _parse_label(label_value, image_path=image_path, mask_path=mask_path)
        sample_id = _first_value(row, "sample_id", "id", "name") or image_path.stem

        return DatasetRecord(
            dataset="visa_pcb",
            category=category,
            sample_id=f"{category}/{sample_id}",
            split=split,
            image_path=image_path,
            label=label,
            mask_path=mask_path if mask_path and mask_path.exists() else None,
            metadata={"source": "manifest"},
        )

    def _records_from_tree(self) -> list[DatasetRecord]:
        mask_index = _build_mask_index(self.root, self.categories)
        records: list[DatasetRecord] = []

        for image_path in sorted(self.root.rglob("*")):
            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            if _path_has_mask_hint(image_path):
                continue

            category = _infer_category(image_path, self.categories)
            if category is None:
                continue

            split = _infer_split(image_path) or "all"
            label = _infer_label_from_path(image_path)
            mask_path = mask_index.get((category, image_path.stem)) if label else None
            records.append(
                DatasetRecord(
                    dataset="visa_pcb",
                    category=category,
                    sample_id=f"{category}/{image_path.stem}",
                    split=split,
                    image_path=image_path,
                    label=label,
                    mask_path=mask_path,
                    metadata={"source": "tree_scan"},
                )
            )

        return records

    def _keep_record(self, record: DatasetRecord) -> bool:
        if record.category not in self.categories:
            return False
        if self.split != "all" and record.split != self.split:
            return False
        if self.normal_only and record.label != 0:
            return False
        return True


def _first_value(row: dict[str, str], *names: str) -> str | None:
    lower_to_key = {key.lower(): key for key in row}
    for name in names:
        key = lower_to_key.get(name.lower())
        if key is None:
            continue
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _resolve_path(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def _parse_label(label_value: str | None, image_path: Path, mask_path: Path | None) -> int:
    if label_value is not None:
        normalized = str(label_value).strip().lower()
        if normalized in NORMAL_VALUES:
            return 0
        if normalized in ANOMALY_VALUES:
            return 1
    if mask_path is not None:
        return 1
    return _infer_label_from_path(image_path)


def _infer_category(path: Path, categories: tuple[str, ...]) -> str | None:
    parts = {part.lower() for part in path.parts}
    for category in categories:
        if category.lower() in parts:
            return category
    return None


def _infer_split(path: Path) -> str | None:
    parts = {part.lower() for part in path.parts}
    for split in ("train", "val", "valid", "validation", "test"):
        if split in parts:
            return "val" if split in {"valid", "validation"} else split
    return None


def _infer_label_from_path(path: Path) -> int:
    parts = {part.lower() for part in path.parts}
    if parts & {"bad", "defect", "defects", "anomaly", "anomalous", "abnormal"}:
        return 1
    if parts & {"good", "normal", "ok"}:
        return 0
    return 0


def _path_has_mask_hint(path: Path) -> bool:
    return bool({part.lower() for part in path.parts} & MASK_HINTS)


def _build_mask_index(root: Path, categories: tuple[str, ...]) -> dict[tuple[str, str], Path]:
    index: dict[tuple[str, str], Path] = {}
    for mask_path in sorted(root.rglob("*")):
        if mask_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if not _path_has_mask_hint(mask_path):
            continue
        category = _infer_category(mask_path, categories)
        if category is None:
            continue
        index[(category, mask_path.stem)] = mask_path
    return index
