from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from datasets.deeppcb import DeepPCBDataset, parse_deeppcb_annotation_line
from datasets.sampling import sample_few_shot_normals
from datasets.types import DatasetRecord


class DeepPCBAnnotationTest(unittest.TestCase):
    def test_parse_deeppcb_annotation_line(self) -> None:
        box = parse_deeppcb_annotation_line("10,20,30,40,3")

        self.assertEqual(box.to_xyxy(), (10.0, 20.0, 30.0, 40.0))
        self.assertEqual(box.class_id, 3)
        self.assertEqual(box.class_name, "mousebite")

    def test_discovers_official_sibling_annotation_folder(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "PCBData"
            image_dir = root / "group77000" / "77000"
            annotation_dir = root / "group77000" / "77000_not"
            image_dir.mkdir(parents=True)
            annotation_dir.mkdir(parents=True)

            Image.new("RGB", (16, 16), (0, 80, 50)).save(image_dir / "77000016_temp.jpg")
            Image.new("RGB", (16, 16), (120, 80, 50)).save(image_dir / "77000016_test.jpg")
            annotation_path = annotation_dir / "77000016.txt"
            annotation_path.write_text("1,2,8,9,4\n")

            dataset = DeepPCBDataset(root=root, split="all", return_tensors=False)

            self.assertEqual(len(dataset), 1)
            self.assertEqual(dataset.records[0].box_path, annotation_path)
            self.assertEqual(dataset.records[0].template_path, image_dir / "77000016_temp.jpg")

    def test_official_split_file_filters_by_first_column_stem(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "PCBData"
            image_dir = root / "group77000" / "77000"
            annotation_dir = root / "group77000" / "77000_not"
            image_dir.mkdir(parents=True)
            annotation_dir.mkdir(parents=True)
            for sample_id in ("77000016", "77000017"):
                Image.new("RGB", (16, 16), (0, 80, 50)).save(image_dir / f"{sample_id}_temp.jpg")
                Image.new("RGB", (16, 16), (120, 80, 50)).save(image_dir / f"{sample_id}_test.jpg")
                (annotation_dir / f"{sample_id}.txt").write_text("1,2,8,9,4\n")

            split_file = root / "trainval.txt"
            split_file.write_text(
                "group77000/77000/77000016.jpg group77000/77000_not/77000016.txt\n"
            )

            dataset = DeepPCBDataset(
                root=root,
                split="train",
                split_file=split_file,
                return_tensors=False,
            )

            self.assertEqual([record.sample_id for record in dataset.records], ["77000016_test"])


class FewShotSamplingTest(unittest.TestCase):
    def test_sample_few_shot_normals_is_category_balanced(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            records = [
                DatasetRecord("visa_pcb", "pcb1", "pcb1/a", "train", tmp_path / "a.jpg", 0),
                DatasetRecord("visa_pcb", "pcb1", "pcb1/b", "train", tmp_path / "b.jpg", 0),
                DatasetRecord("visa_pcb", "pcb1", "pcb1/bad", "test", tmp_path / "bad.jpg", 1),
                DatasetRecord("visa_pcb", "pcb2", "pcb2/a", "train", tmp_path / "c.jpg", 0),
                DatasetRecord("visa_pcb", "pcb2", "pcb2/b", "train", tmp_path / "d.jpg", 0),
            ]

            support = sample_few_shot_normals(records, k=1, seed=4880)

            self.assertEqual(set(support), {"pcb1", "pcb2"})
            self.assertTrue(all(len(samples) == 1 for samples in support.values()))
            self.assertTrue(
                all(sample.label == 0 for samples in support.values() for sample in samples)
            )


if __name__ == "__main__":
    unittest.main()
