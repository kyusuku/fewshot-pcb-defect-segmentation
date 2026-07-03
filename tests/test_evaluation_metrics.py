from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from evaluation.metrics import (
    best_f1_iou,
    binary_roc_auc,
    evaluate_heatmap_rows,
    summarize_image_scores,
)


class EvaluationMetricsTest(unittest.TestCase):
    def test_binary_roc_auc_scores_ranked_predictions(self) -> None:
        labels = np.array([0, 0, 1, 1], dtype=np.uint8)
        scores = np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float32)

        self.assertAlmostEqual(binary_roc_auc(labels, scores), 1.0)

    def test_best_f1_iou_finds_perfect_threshold(self) -> None:
        target = np.array([[0, 1], [0, 1]], dtype=np.uint8)
        scores = np.array([[0.1, 0.9], [0.2, 0.8]], dtype=np.float32)

        metrics = best_f1_iou(target.ravel(), scores.ravel())

        self.assertAlmostEqual(metrics["best_f1"], 1.0)
        self.assertAlmostEqual(metrics["best_iou"], 1.0)
        self.assertGreater(metrics["best_threshold"], 0.2)

    def test_evaluate_heatmap_rows_computes_image_and_pixel_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            normal_heatmap = tmp_path / "normal.npy"
            anomaly_heatmap = tmp_path / "anomaly.npy"
            normal_mask = tmp_path / "normal.png"
            anomaly_mask = tmp_path / "anomaly.png"
            np.save(normal_heatmap, np.zeros((4, 4), dtype=np.float32))
            np.save(
                anomaly_heatmap,
                np.array(
                    [
                        [0.0, 0.0, 0.1, 0.1],
                        [0.0, 0.7, 0.8, 0.1],
                        [0.0, 0.6, 0.9, 0.1],
                        [0.0, 0.0, 0.1, 0.1],
                    ],
                    dtype=np.float32,
                ),
            )
            Image.fromarray(np.zeros((4, 4), dtype=np.uint8), mode="L").save(normal_mask)
            mask = np.zeros((4, 4), dtype=np.uint8)
            mask[1:3, 1:3] = 255
            Image.fromarray(mask, mode="L").save(anomaly_mask)

            metrics = evaluate_heatmap_rows(
                [
                    {
                        "sample_id": "normal",
                        "label": "0",
                        "image_score": "0.0",
                        "heatmap_path": str(normal_heatmap),
                        "mask_path": str(normal_mask),
                    },
                    {
                        "sample_id": "anomaly",
                        "label": "1",
                        "image_score": "0.9",
                        "heatmap_path": str(anomaly_heatmap),
                        "mask_path": str(anomaly_mask),
                    },
                ]
            )

        self.assertAlmostEqual(metrics["image_auroc"], 1.0)
        self.assertAlmostEqual(metrics["pixel_auroc"], 1.0)
        self.assertAlmostEqual(metrics["best_pixel_f1"], 1.0)
        self.assertAlmostEqual(metrics["best_pixel_iou"], 1.0)

    def test_summarize_image_scores_handles_single_class_auc_as_nan(self) -> None:
        metrics = summarize_image_scores(
            [
                {"label": "1", "image_score": "0.2"},
                {"label": "1", "image_score": "0.4"},
            ]
        )

        self.assertTrue(np.isnan(metrics["image_auroc"]))


class EvaluateHeatmapsScriptTest(unittest.TestCase):
    def test_script_writes_metrics_json(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            heatmap_path = tmp_path / "heatmap.npy"
            mask_path = tmp_path / "mask.png"
            scores_path = tmp_path / "scores.csv"
            output_path = tmp_path / "metrics.json"
            np.save(heatmap_path, np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32))
            Image.fromarray(
                np.array([[0, 255], [0, 255]], dtype=np.uint8),
                mode="L",
            ).save(mask_path)
            with scores_path.open("w", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "sample_id",
                        "category",
                        "label",
                        "image_score",
                        "heatmap_path",
                        "mask_path",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "sample_id": "anomaly",
                        "category": "pcb1",
                        "label": "1",
                        "image_score": "1.0",
                        "heatmap_path": str(heatmap_path),
                        "mask_path": str(mask_path),
                    }
                )

            result = subprocess.run(
                [
                    sys.executable,
                    str(repo_root / "scripts" / "evaluate_heatmaps.py"),
                    "--scores-csv",
                    str(scores_path),
                    "--output-json",
                    str(output_path),
                ],
                check=False,
                cwd=repo_root,
                env={"PYTHONPATH": str(repo_root / "src")},
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            metrics = json.loads(output_path.read_text())

        self.assertEqual(metrics["num_images"], 1)
        self.assertAlmostEqual(metrics["best_pixel_iou"], 1.0)


if __name__ == "__main__":
    unittest.main()
