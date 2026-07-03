from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from datasets.manifests import (
    apply_deeppcb_train_test_split,
    build_kfold_manifest_rows,
    split_train_validation,
    write_fold_manifest_csv,
    write_manifest_csv,
)
from datasets.types import DatasetRecord


class ManifestSplitTest(unittest.TestCase):
    def test_split_train_validation_keeps_test_locked_and_balances_categories(self) -> None:
        records = [
            DatasetRecord("visa_pcb", "pcb1", "pcb1/train_0", "train", Path("a.jpg"), 0),
            DatasetRecord("visa_pcb", "pcb1", "pcb1/train_1", "train", Path("b.jpg"), 0),
            DatasetRecord("visa_pcb", "pcb1", "pcb1/train_2", "train", Path("c.jpg"), 0),
            DatasetRecord("visa_pcb", "pcb2", "pcb2/train_0", "train", Path("d.jpg"), 0),
            DatasetRecord("visa_pcb", "pcb2", "pcb2/train_1", "train", Path("e.jpg"), 0),
            DatasetRecord("visa_pcb", "pcb2", "pcb2/train_2", "train", Path("f.jpg"), 0),
            DatasetRecord("visa_pcb", "pcb1", "pcb1/test_bad", "test", Path("g.jpg"), 1),
        ]

        split = split_train_validation(records, val_ratio=1 / 3, seed=4880)

        self.assertEqual(sum(record.split == "val" for record in split), 2)
        self.assertEqual(sum(record.split == "test" for record in split), 1)
        self.assertEqual(
            sorted(record.category for record in split if record.split == "val"),
            ["pcb1", "pcb2"],
        )
        self.assertEqual(
            [record.split for record in split if record.sample_id == "pcb1/test_bad"],
            ["test"],
        )

    def test_apply_deeppcb_train_test_split_uses_sorted_records(self) -> None:
        records = [
            DatasetRecord("deeppcb", "group", "00000002_test", "all", Path("2.jpg"), 1),
            DatasetRecord("deeppcb", "group", "00000000_test", "all", Path("0.jpg"), 1),
            DatasetRecord("deeppcb", "group", "00000001_test", "all", Path("1.jpg"), 1),
        ]

        split = apply_deeppcb_train_test_split(records, train_count=2, val_ratio=0.5, seed=4880)

        self.assertEqual(
            [record.sample_id for record in split],
            ["00000000_test", "00000001_test", "00000002_test"],
        )
        self.assertEqual([record.split for record in split], ["train", "val", "test"])

    def test_build_kfold_manifest_rows_keeps_test_locked_and_balances_categories(self) -> None:
        records = []
        for category in ("pcb1", "pcb2"):
            for index in range(5):
                records.append(
                    DatasetRecord(
                        "visa_pcb",
                        category,
                        f"{category}/train_{index}",
                        "train",
                        Path(f"{category}_{index}.jpg"),
                        0,
                    )
                )
        records.append(
            DatasetRecord("visa_pcb", "pcb1", "pcb1/test_normal", "test", Path("n.jpg"), 0)
        )
        records.append(DatasetRecord("visa_pcb", "pcb1", "pcb1/test_bad", "test", Path("b.jpg"), 1))

        rows = build_kfold_manifest_rows(records, num_folds=5, seed=4880, stratify_by="category")

        self.assertEqual(len(rows), len(records) * 5)
        val_by_category = {
            fold_id: {
                category: sum(
                    row["fold_id"] == fold_id
                    and row["category"] == category
                    and row["fold_split"] == "val"
                    for row in rows
                )
                for category in ("pcb1", "pcb2")
            }
            for fold_id in range(5)
        }
        self.assertTrue(
            all(
                count == 1
                for fold_counts in val_by_category.values()
                for count in fold_counts.values()
            )
        )
        train_val_counts = {
            row["sample_id"]: sum(
                other["sample_id"] == row["sample_id"] and other["fold_split"] == "val"
                for other in rows
            )
            for row in rows
            if row["split"] == "train"
        }
        self.assertTrue(all(count == 1 for count in train_val_counts.values()))
        self.assertTrue(
            all(row["fold_split"] == "test" for row in rows if row["split"] == "test")
        )

    def test_build_kfold_manifest_rows_can_balance_deeppcb_multilabels(self) -> None:
        records = [
            DatasetRecord(
                "deeppcb",
                "group",
                f"pair_{index}",
                "train",
                Path(f"{index}.jpg"),
                1,
                metadata={"class_ids": [1 if index % 2 == 0 else 2]},
            )
            for index in range(6)
        ]
        records.append(
            DatasetRecord(
                "deeppcb",
                "group",
                "pair_test",
                "test",
                Path("test.jpg"),
                1,
                metadata={"class_ids": [1, 2]},
            )
        )

        rows = build_kfold_manifest_rows(records, num_folds=3, seed=4880, stratify_by="multilabel")

        for fold_id in range(3):
            val_rows = [
                row for row in rows if row["fold_id"] == fold_id and row["fold_split"] == "val"
            ]
            self.assertEqual(len(val_rows), 2)
            class_ids = sorted(row["metadata_json"] for row in val_rows)
            self.assertTrue(any('"class_ids": [1]' in value for value in class_ids))
            self.assertTrue(any('"class_ids": [2]' in value for value in class_ids))

    def test_build_kfold_manifest_rows_counts_repeated_deeppcb_classes(self) -> None:
        records = [
            DatasetRecord(
                "deeppcb",
                "group",
                "many_opens",
                "train",
                Path("a.jpg"),
                1,
                metadata={"class_ids": [1, 1, 1]},
            ),
            DatasetRecord(
                "deeppcb",
                "group",
                "one_short",
                "train",
                Path("b.jpg"),
                1,
                metadata={"class_ids": [2]},
            ),
        ]

        rows = build_kfold_manifest_rows(records, num_folds=2, seed=4880, stratify_by="multilabel")

        many_opens = [
            row
            for row in rows
            if row["sample_id"] == "many_opens" and row["fold_split"] == "val"
        ]
        self.assertEqual(len(many_opens), 1)
        self.assertIn('"class_ids": [1, 1, 1]', many_opens[0]["metadata_json"])

    def test_write_fold_manifest_csv_preserves_fold_fields(self) -> None:
        records = [
            DatasetRecord(
                "visa_pcb",
                "pcb1",
                f"pcb1/train_{index}",
                "train",
                Path(f"{index}.jpg"),
                0,
            )
            for index in range(2)
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            rows = build_kfold_manifest_rows(records, num_folds=2, seed=4880)
            output_path = write_fold_manifest_csv(rows, Path(tmpdir) / "folds.csv")
            with output_path.open(newline="") as handle:
                csv_rows = list(csv.DictReader(handle))

        self.assertEqual({"fold_id", "fold_split"}.issubset(csv_rows[0]), True)
        self.assertEqual({row["fold_split"] for row in csv_rows}, {"dev", "val"})

    def test_write_manifest_csv_preserves_core_fields(self) -> None:
        records = [
            DatasetRecord(
                dataset="visa_pcb",
                category="pcb1",
                sample_id="pcb1/train_0",
                split="train",
                image_path=Path("VisA/pcb1/train/good/a.JPG"),
                label=0,
                metadata={"source": "unit"},
            )
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = write_manifest_csv(records, Path(tmpdir) / "manifest.csv")
            with output_path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(rows[0]["dataset"], "visa_pcb")
        self.assertEqual(rows[0]["category"], "pcb1")
        self.assertEqual(rows[0]["split"], "train")
        self.assertEqual(rows[0]["label"], "0")
        self.assertIn("source", rows[0]["metadata_json"])


if __name__ == "__main__":
    unittest.main()
