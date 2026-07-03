from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from sam_refine.prompts import heatmap_to_prompt_regions
from sam_refine.refiner import FallbackMaskRefiner


class PromptGenerationTest(unittest.TestCase):
    def test_heatmap_to_prompt_regions_extracts_connected_boxes(self) -> None:
        heatmap = np.zeros((10, 10), dtype=np.float32)
        heatmap[3:6, 2:5] = 0.9
        heatmap[7:9, 7:9] = 0.7

        regions = heatmap_to_prompt_regions(
            heatmap,
            threshold=0.5,
            min_area=2,
            max_regions=4,
        )

        self.assertEqual(len(regions), 2)
        self.assertEqual(regions[0].box_xyxy, (2, 3, 5, 6))
        self.assertEqual(regions[0].point_xy, (3.5, 4.5))
        self.assertGreater(regions[0].score, regions[1].score)


class FallbackMaskRefinerTest(unittest.TestCase):
    def test_fallback_refiner_thresholds_inside_prompt_boxes(self) -> None:
        image = Image.new("RGB", (10, 10), (0, 0, 0))
        heatmap = np.zeros((10, 10), dtype=np.float32)
        heatmap[3:6, 2:5] = 0.9
        regions = heatmap_to_prompt_regions(heatmap, threshold=0.5, min_area=2)

        predictions = FallbackMaskRefiner(threshold_fraction=0.5).refine(
            image=image,
            heatmap=heatmap,
            regions=regions,
        )

        self.assertEqual(len(predictions), 1)
        self.assertEqual(predictions[0].mask.shape, (10, 10))
        self.assertEqual(int(predictions[0].mask.sum()), 9)
        self.assertGreater(predictions[0].score, 0.0)


class MaskRefinementScriptTest(unittest.TestCase):
    def test_script_writes_mask_scores_and_predicted_masks(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            image_path = tmp_path / "image.png"
            heatmap_path = tmp_path / "heatmap.npy"
            scores_path = tmp_path / "scores.csv"
            output_dir = tmp_path / "refined"
            Image.new("RGB", (10, 10), (20, 40, 60)).save(image_path)
            heatmap = np.zeros((10, 10), dtype=np.float32)
            heatmap[3:6, 2:5] = 0.9
            np.save(heatmap_path, heatmap)
            with scores_path.open("w", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "sample_id",
                        "category",
                        "label",
                        "image_score",
                        "image_path",
                        "mask_path",
                        "heatmap_path",
                        "debug_path",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "sample_id": "pcb1/anomaly",
                        "category": "pcb1",
                        "label": "1",
                        "image_score": "0.9",
                        "image_path": str(image_path),
                        "mask_path": "",
                        "heatmap_path": str(heatmap_path),
                        "debug_path": "",
                    }
                )

            result = subprocess.run(
                [
                    sys.executable,
                    str(repo_root / "scripts" / "run_mask_refinement.py"),
                    "--scores-csv",
                    str(scores_path),
                    "--output-dir",
                    str(output_dir),
                    "--threshold",
                    "0.5",
                    "--refiner",
                    "fallback",
                ],
                check=False,
                cwd=repo_root,
                env={"PYTHONPATH": str(repo_root / "src")},
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            rows = _read_csv(output_dir / "mask_scores.csv")
            pred_mask_path = Path(rows[0]["pred_mask_path"])
            pred_mask = np.asarray(Image.open(pred_mask_path).convert("L")) > 0

        self.assertEqual(len(rows), 1)
        self.assertEqual(int(pred_mask.sum()), 9)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    unittest.main()
