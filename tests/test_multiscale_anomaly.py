from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from anomaly.memory_bank import build_memory_bank
from anomaly.multiscale import compute_anomaly_heatmap, iter_crop_boxes, parse_crop_sizes
from features.dinov2 import ColorPatchFeatureExtractor
from tests.test_anomaly_baseline import _read_csv, _write_tiny_visa_fold_manifest
from utils.synthetic_data import create_synthetic_debug_datasets


class MultiScaleAnomalyTest(unittest.TestCase):
    def test_iter_crop_boxes_covers_image_edges(self) -> None:
        boxes = list(iter_crop_boxes(image_size=(10, 8), crop_size=4, overlap=0.5))

        self.assertEqual(boxes[0], (0, 0, 4, 4))
        self.assertIn((6, 4, 10, 8), boxes)
        self.assertTrue(all(0 <= x1 < x2 <= 10 for x1, _, x2, _ in boxes))
        self.assertTrue(all(0 <= y1 < y2 <= 8 for _, y1, _, y2 in boxes))

    def test_parse_crop_sizes_ignores_empty_values(self) -> None:
        self.assertEqual(parse_crop_sizes("32, 64,,"), [32, 64])
        self.assertEqual(parse_crop_sizes(""), [])

    def test_compute_anomaly_heatmap_returns_full_image_shape(self) -> None:
        support = Image.new("RGB", (8, 8), (0, 0, 0))
        query = Image.new("RGB", (8, 8), (0, 0, 0))
        ImageDraw.Draw(query).rectangle((4, 4, 7, 7), fill=(255, 255, 255))
        extractor = ColorPatchFeatureExtractor(image_size=4, patch_size=2)
        memory_bank = build_memory_bank([extractor.extract(support)], normalize=False)

        heatmap = compute_anomaly_heatmap(
            image=query,
            extractor=extractor,
            memory_bank=memory_bank,
            crop_sizes=[4],
            crop_overlap=0.5,
            fusion="max",
            normalize_features=False,
        )

        self.assertEqual(heatmap.shape, (8, 8))
        self.assertGreater(float(heatmap.max()), 0.0)

    def test_compute_anomaly_heatmap_preserves_absolute_score_scale(self) -> None:
        support = Image.new("RGB", (8, 8), (0, 0, 0))
        query = Image.new("RGB", (8, 8), (0, 0, 0))
        ImageDraw.Draw(query).rectangle((4, 0, 7, 7), fill=(25, 25, 25))
        extractor = ColorPatchFeatureExtractor(image_size=4, patch_size=2)
        memory_bank = build_memory_bank([extractor.extract(support)], normalize=False)

        heatmap = compute_anomaly_heatmap(
            image=query,
            extractor=extractor,
            memory_bank=memory_bank,
            crop_sizes=[],
            normalize_features=False,
        )

        self.assertGreater(float(heatmap.max()), 0.0)
        self.assertLess(float(heatmap.max()), 0.5)


class MultiScaleBaselineScriptTest(unittest.TestCase):
    def test_script_accepts_multiscale_crop_arguments(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            fixture = create_synthetic_debug_datasets(tmp_path / "fixtures")
            manifest_path = tmp_path / "visa_pcb_folds.csv"
            _write_tiny_visa_fold_manifest(fixture.visa_root, manifest_path)
            output_dir = tmp_path / "outputs"

            result = subprocess.run(
                [
                    sys.executable,
                    str(repo_root / "scripts" / "run_dinov2_baseline.py"),
                    "--manifest",
                    str(manifest_path),
                    "--fold-id",
                    "0",
                    "--category",
                    "pcb1",
                    "--k",
                    "2",
                    "--limit",
                    "1",
                    "--feature-backbone",
                    "color_patch",
                    "--image-size",
                    "56",
                    "--patch-size",
                    "14",
                    "--crop-sizes",
                    "32",
                    "--crop-overlap",
                    "0.5",
                    "--fusion",
                    "max",
                    "--output-dir",
                    str(output_dir),
                ],
                check=False,
                cwd=repo_root,
                env={"PYTHONPATH": str(repo_root / "src")},
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            scores_path = output_dir / "scores.csv"
            rows = _read_csv(scores_path)
            heatmap = np.load(scores_path.parent / rows[0]["heatmap_path"])

        self.assertEqual(heatmap.shape, (64, 64))
        self.assertGreater(float(heatmap.max()), 0.0)


if __name__ == "__main__":
    unittest.main()
