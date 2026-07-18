"""DeepPCB dataset loader.

DeepPCB provides aligned template/test pairs plus bounding boxes. Boxes are
localization labels and should only become pseudo-masks when explicitly needed
for prompts or qualitative visualization.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from datasets.types import BoxAnnotation, DatasetRecord

DEEPPCB_CLASS_NAMES = {
    1: "open",
    2: "short",
    3: "mousebite",
    4: "spur",
    5: "spurious_copper",
    6: "pin_hole",
}

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
TEST_MARKERS = ("_test", "-test", "test", "tested")
TEMPLATE_REPLACEMENTS = (
    ("_test", "_temp"),
    ("_test", "_template"),
    ("_test", "_tpl"),
    ("test", "temp"),
    ("test", "template"),
    ("tested", "template"),
)


def parse_deeppcb_annotation_line(line: str) -> BoxAnnotation:
    """Parse one `x1,y1,x2,y2,type` annotation line."""

    parts = [part.strip() for part in line.replace(" ", ",").split(",") if part.strip()]
    if len(parts) != 5:
        raise ValueError(f"Expected 5 comma-separated values, got {len(parts)}: {line!r}")

    x1, y1, x2, y2 = (float(value) for value in parts[:4])
    class_id = int(float(parts[4]))
    return BoxAnnotation(
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        class_id=class_id,
        class_name=DEEPPCB_CLASS_NAMES.get(class_id, f"class_{class_id}"),
    )


def load_deeppcb_boxes(path: Path) -> list[BoxAnnotation]:
    """Load all valid DeepPCB boxes from a text annotation file."""

    boxes: list[BoxAnnotation] = []
    if not path.exists():
        return boxes

    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            boxes.append(parse_deeppcb_annotation_line(stripped))
        except ValueError as exc:
            raise ValueError(f"Invalid DeepPCB annotation {path}:{line_number}: {exc}") from exc
    return boxes


class DeepPCBDataset:
    """Dataset for DeepPCB aligned template/test pairs."""

    def __init__(
        self,
        root: str | Path,
        split: str = "all",
        split_file: str | Path | None = None,
        return_tensors: bool = True,
        return_pseudo_mask: bool = True,
    ) -> None:
        self.root = Path(root).expanduser()
        self.split = split
        self.split_file = Path(split_file).expanduser() if split_file else None
        self.return_tensors = return_tensors
        self.return_pseudo_mask = return_pseudo_mask

        if not self.root.exists():
            raise FileNotFoundError(
                f"DeepPCB root does not exist: {self.root}. "
                "Pass --root or edit configs/datasets/deeppcb.yaml."
            )

        self._allowed_ids = self._load_split_ids(self.split_file)
        self.records = self._discover_records()
        if not self.records:
            raise RuntimeError(
                f"No DeepPCB samples found under {self.root}. Expected files like "
                "'*_test.jpg' with matching annotation .txt files."
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        from utils.image import (
            boxes_to_mask,
            image_to_tensor,
            load_rgb_image,
            mask_to_tensor,
        )

        record = self.records[index]
        image = load_rgb_image(record.image_path)
        template = load_rgb_image(record.template_path) if record.template_path else None
        boxes = (
            [box.clipped(*image.size) for box in load_deeppcb_boxes(record.box_path)]
            if record.box_path
            else []
        )

        sample: dict[str, Any] = record.to_dict()
        sample["record"] = record
        sample["boxes"] = [box.to_dict() for box in boxes]
        sample["box_annotations"] = boxes
        sample["image_size"] = {"width": image.size[0], "height": image.size[1]}
        sample["pseudo_mask_is_ground_truth"] = False

        if self.return_tensors:
            sample["image"] = image_to_tensor(image)
            sample["template"] = image_to_tensor(template) if template else None
        else:
            sample["image"] = image
            sample["template"] = template

        if self.return_pseudo_mask:
            pseudo_mask = boxes_to_mask(boxes, image.size)
            sample["pseudo_mask"] = (
                mask_to_tensor(pseudo_mask) if self.return_tensors else pseudo_mask
            )

        return sample

    def describe(self) -> dict[str, Any]:
        box_counter: Counter[str] = Counter()
        for record in self.records:
            if record.box_path:
                for box in load_deeppcb_boxes(record.box_path):
                    box_counter[box.class_name or str(box.class_id)] += 1

        return {
            "dataset": "deeppcb",
            "root": str(self.root),
            "split": self.split,
            "num_samples": len(self.records),
            "box_counts": dict(sorted(box_counter.items())),
        }

    def _discover_records(self) -> list[DatasetRecord]:
        records: list[DatasetRecord] = []
        for image_path in sorted(self.root.rglob("*")):
            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            if not _looks_like_test_image(image_path):
                continue
            if self._allowed_ids is not None and not _matches_split_id(
                image_path, self._allowed_ids
            ):
                continue

            annotation_path = _find_annotation_for_test_image(image_path)
            template_path = _find_template_for_test_image(image_path)
            sample_id = image_path.stem
            group = image_path.parent.name
            records.append(
                DatasetRecord(
                    dataset="deeppcb",
                    category=group,
                    sample_id=sample_id,
                    split=self.split,
                    image_path=image_path,
                    label=1,
                    box_path=annotation_path,
                    template_path=template_path,
                    metadata={"group": group},
                )
            )
        return records

    @staticmethod
    def _load_split_ids(split_file: Path | None) -> set[str] | None:
        if split_file is None:
            return None
        if not split_file.exists():
            raise FileNotFoundError(f"DeepPCB split file does not exist: {split_file}")
        ids: set[str] = set()
        for line in split_file.read_text().splitlines():
            value = line.strip()
            if not value or value.startswith("#"):
                continue
            for token in value.split():
                _add_split_id_variants(ids, token)
        return ids


def _looks_like_test_image(path: Path) -> bool:
    lower_stem = path.stem.lower()
    return any(marker in lower_stem for marker in TEST_MARKERS)


def _find_template_for_test_image(test_path: Path) -> Path | None:
    candidates = []
    stem = test_path.stem
    lower_stem = stem.lower()
    for source, replacement in TEMPLATE_REPLACEMENTS:
        if source in lower_stem:
            start = lower_stem.index(source)
            candidate_stem = stem[:start] + replacement + stem[start + len(source) :]
            candidates.append(test_path.with_name(candidate_stem + test_path.suffix))

    candidates.extend(
        [
            test_path.with_name(stem.replace("_test", "_temp") + test_path.suffix),
            test_path.with_name(stem.replace("_test", "_template") + test_path.suffix),
        ]
    )

    for candidate in candidates:
        if candidate.exists() and candidate != test_path:
            return candidate
    return None


def _find_annotation_for_test_image(test_path: Path) -> Path | None:
    stems = _candidate_annotation_stems(test_path)
    candidates = [test_path.with_name(f"{stem}.txt") for stem in stems]

    parent = test_path.parent
    sibling_annotation_dir = parent.parent / f"{parent.name}_not"
    candidates.extend(sibling_annotation_dir / f"{stem}.txt" for stem in stems)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _candidate_annotation_stems(test_path: Path) -> list[str]:
    stems = [test_path.stem]
    lower_stem = test_path.stem.lower()
    for marker in TEST_MARKERS:
        if marker in lower_stem:
            start = lower_stem.index(marker)
            candidate = test_path.stem[:start] + test_path.stem[start + len(marker) :]
            candidate = candidate.strip("_-")
            if candidate and candidate not in stems:
                stems.append(candidate)
    return stems


def _add_split_id_variants(ids: set[str], value: str) -> None:
    path = Path(value)
    stem = path.stem
    ids.update({value, path.name, stem})
    for marker in TEST_MARKERS:
        lower_stem = stem.lower()
        if marker in lower_stem:
            start = lower_stem.index(marker)
            base = (stem[:start] + stem[start + len(marker) :]).strip("_-")
            if base:
                ids.update({base, f"{base}_test"})
            return
    ids.add(f"{stem}_test")


def _matches_split_id(path: Path, allowed_ids: Iterable[str]) -> bool:
    allowed = set(allowed_ids)
    relative = str(path)
    return path.stem in allowed or path.name in allowed or relative in allowed
