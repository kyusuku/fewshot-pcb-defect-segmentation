from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from datasets.deeppcb import parse_deeppcb_annotation_line
from datasets.sampling import sample_few_shot_normals
from datasets.types import DatasetRecord


class DeepPCBAnnotationTest(unittest.TestCase):
    def test_parse_deeppcb_annotation_line(self) -> None:
        box = parse_deeppcb_annotation_line("10,20,30,40,3")

        self.assertEqual(box.to_xyxy(), (10.0, 20.0, 30.0, 40.0))
        self.assertEqual(box.class_id, 3)
        self.assertEqual(box.class_name, "mousebite")


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
