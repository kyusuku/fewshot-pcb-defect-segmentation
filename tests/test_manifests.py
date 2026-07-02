from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from datasets.manifests import (
    apply_deeppcb_train_test_split,
    split_train_validation,
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

        self.assertEqual([record.sample_id for record in split], ["00000000_test", "00000001_test", "00000002_test"])
        self.assertEqual([record.split for record in split], ["train", "val", "test"])

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
