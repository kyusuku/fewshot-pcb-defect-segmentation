from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from datasets.deeppcb import DeepPCBDataset
from datasets.visa import VisAPCBDataset
from utils.synthetic_data import create_synthetic_debug_datasets
from utils.visualize import visualize_record


class SyntheticDebugDataTest(unittest.TestCase):
    def test_generated_fixtures_load_and_visualize(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture = create_synthetic_debug_datasets(Path(tmpdir))

            visa = VisAPCBDataset(
                root=fixture.visa_root,
                categories=("pcb1",),
                split="test",
                return_tensors=False,
            )
            deeppcb = DeepPCBDataset(
                root=fixture.deeppcb_root,
                split="test",
                return_tensors=False,
            )

            self.assertEqual(len(visa), 2)
            self.assertEqual(len(deeppcb), 1)
            self.assertEqual([record.label for record in visa.records], [1, 0])

            visa_sample = visa[0]
            self.assertEqual(visa_sample["mask"].shape, (64, 64))
            self.assertGreater(int(visa_sample["mask"].sum()), 0)

            deeppcb_sample = deeppcb[0]
            self.assertEqual(len(deeppcb_sample["boxes"]), 2)
            self.assertFalse(deeppcb_sample["pseudo_mask_is_ground_truth"])

            output_dir = Path(tmpdir) / "debug_outputs"
            visa_png = visualize_record(visa.records[0], output_dir / "visa.png")
            deeppcb_png = visualize_record(
                deeppcb.records[0],
                output_dir / "deeppcb.png",
                boxes=deeppcb_sample["box_annotations"],
            )

            self.assertTrue(visa_png.exists())
            self.assertTrue(deeppcb_png.exists())
            self.assertGreater(visa_png.stat().st_size, 0)
            self.assertGreater(deeppcb_png.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
