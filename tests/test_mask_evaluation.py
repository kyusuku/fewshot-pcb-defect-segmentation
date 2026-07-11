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

from evaluation.masks import (
    evaluate_mask_rows,
    mask_confusion_metrics,
    resolve_mask_row_paths,
    summarize_binary_metrics,
)


class MaskEvaluationTest(unittest.TestCase):
    def test_resolves_all_portable_mask_score_artifact_paths_from_csv_parent(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            row = {
                "image_path": "source/image.png",
                "mask_path": "source/ground_truth.png",
                "heatmap_path": "source/heatmap.npy",
                "sam2_mask_path": "artifacts/raw.png",
                "pred_mask_path": "artifacts/pred.png",
                "debug_path": "artifacts/debug.png",
            }

            resolved = resolve_mask_row_paths([row], base_dir=root)[0]

        for field, relative_path in row.items():
            self.assertEqual(resolved[field], str(root.resolve() / relative_path), msg=field)

    def test_mask_confusion_metrics_reports_overlap_quality(self) -> None:
        pred = np.array([[1, 0], [0, 1]], dtype=np.uint8)
        target = np.array([[1, 1], [0, 0]], dtype=np.uint8)

        metrics = mask_confusion_metrics(pred, target)

        self.assertAlmostEqual(metrics["precision"], 0.5)
        self.assertAlmostEqual(metrics["recall"], 0.5)
        self.assertAlmostEqual(metrics["f1"], 0.5)
        self.assertAlmostEqual(metrics["iou"], 1.0 / 3.0)
        self.assertEqual(metrics["true_positive_pixels"], 1.0)
        self.assertEqual(metrics["false_positive_pixels"], 1.0)
        self.assertEqual(metrics["false_negative_pixels"], 1.0)

    def test_mask_confusion_metrics_reports_zero_counts_for_two_empty_masks(self) -> None:
        metrics = mask_confusion_metrics(
            np.zeros((2, 2), dtype=np.uint8),
            np.zeros((2, 2), dtype=np.uint8),
        )

        self.assertEqual(metrics["true_positive_pixels"], 0.0)
        self.assertEqual(metrics["false_positive_pixels"], 0.0)
        self.assertEqual(metrics["false_negative_pixels"], 0.0)

    def test_binary_summary_aggregates_exact_counts_instead_of_mean_ratios(self) -> None:
        rows = [
            {
                "label": "1",
                "mask_precision": 1.0,
                "mask_recall": 1.0,
                "mask_f1": 1.0,
                "mask_iou": 1.0,
                "true_positive_pixels": 1.0,
                "false_positive_pixels": 0.0,
                "false_negative_pixels": 0.0,
            },
            {
                "label": "1",
                "mask_precision": 0.0,
                "mask_recall": 0.0,
                "mask_f1": 0.0,
                "mask_iou": 0.0,
                "true_positive_pixels": 0.0,
                "false_positive_pixels": 0.0,
                "false_negative_pixels": 3.0,
            },
        ]

        summary = summarize_binary_metrics(rows, prefix="calibrated")

        self.assertEqual(summary["calibrated_aggregate_pixel_precision"], 1.0)
        self.assertEqual(summary["calibrated_aggregate_pixel_recall"], 0.25)
        self.assertEqual(summary["calibrated_aggregate_pixel_f1"], 0.4)
        self.assertEqual(summary["calibrated_aggregate_pixel_iou"], 0.25)
        self.assertEqual(summary["calibrated_mean_anomaly_mask_f1"], 0.5)
        self.assertEqual(summary["calibrated_mean_mask_precision"], 0.5)
        self.assertEqual(summary["calibrated_mean_mask_recall"], 0.5)
        self.assertEqual(summary["calibrated_num_images"], 2.0)
        self.assertEqual(summary["calibrated_num_anomaly_images"], 2.0)
        self.assertEqual(summary["calibrated_aggregate_true_positive_pixels"], 1.0)
        self.assertEqual(summary["calibrated_aggregate_false_positive_pixels"], 0.0)
        self.assertEqual(summary["calibrated_aggregate_false_negative_pixels"], 3.0)

    def test_binary_summary_uses_nan_performance_for_no_rows(self) -> None:
        summary = summarize_binary_metrics([], prefix="calibrated")

        self.assertEqual(summary["calibrated_num_images"], 0.0)
        self.assertEqual(summary["calibrated_num_anomaly_images"], 0.0)
        self.assertEqual(summary["calibrated_aggregate_true_positive_pixels"], 0.0)
        self.assertEqual(summary["calibrated_aggregate_false_positive_pixels"], 0.0)
        self.assertEqual(summary["calibrated_aggregate_false_negative_pixels"], 0.0)
        for key, value in summary.items():
            if "precision" in key or "recall" in key or "f1" in key or "iou" in key:
                self.assertTrue(np.isnan(value), msg=key)

    def test_evaluate_mask_rows_summarizes_all_and_anomaly_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            anomaly_pred = root / "anomaly_pred.png"
            anomaly_gt = root / "anomaly_gt.png"
            normal_pred = root / "normal_pred.png"
            _save_mask(anomaly_pred, np.array([[255, 0], [0, 255]], dtype=np.uint8))
            _save_mask(anomaly_gt, np.array([[255, 255], [0, 0]], dtype=np.uint8))
            _save_mask(normal_pred, np.zeros((2, 2), dtype=np.uint8))

            metrics, per_row = evaluate_mask_rows(
                [
                    {
                        "sample_id": "pcb1/anomaly",
                        "category": "pcb1",
                        "label": "1",
                        "pred_mask_path": str(anomaly_pred),
                        "mask_path": str(anomaly_gt),
                    },
                    {
                        "sample_id": "pcb1/normal",
                        "category": "pcb1",
                        "label": "0",
                        "pred_mask_path": str(normal_pred),
                        "mask_path": "",
                    },
                ]
            )

        self.assertEqual(metrics["num_mask_images"], 2.0)
        self.assertEqual(metrics["num_anomaly_mask_images"], 1.0)
        self.assertAlmostEqual(metrics["mean_mask_iou"], (1.0 / 3.0 + 1.0) / 2.0)
        self.assertAlmostEqual(metrics["mean_anomaly_mask_iou"], 1.0 / 3.0)
        self.assertEqual(len(per_row), 2)
        self.assertAlmostEqual(float(per_row[0]["mask_f1"]), 0.5)


class EvaluateMasksScriptTest(unittest.TestCase):
    def test_script_writes_json_and_per_row_csv_with_source_score_join(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pred_path = root / "pred.png"
            gt_path = root / "gt.png"
            mask_scores_path = root / "mask_scores.csv"
            source_scores_path = root / "scores.csv"
            output_json = root / "mask_metrics.json"
            output_csv = root / "mask_metrics.csv"
            _save_mask(pred_path, np.array([[255, 0], [0, 255]], dtype=np.uint8))
            _save_mask(gt_path, np.array([[255, 255], [0, 0]], dtype=np.uint8))
            _write_csv(
                mask_scores_path,
                ["sample_id", "category", "label", "pred_mask_path"],
                [
                    {
                        "sample_id": "pcb1/anomaly",
                        "category": "pcb1",
                        "label": "1",
                        "pred_mask_path": str(pred_path),
                    }
                ],
            )
            _write_csv(
                source_scores_path,
                ["sample_id", "mask_path", "label"],
                [
                    {
                        "sample_id": "pcb1/anomaly",
                        "mask_path": str(gt_path),
                        "label": "1",
                    }
                ],
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(repo_root / "scripts" / "evaluate_masks.py"),
                    "--mask-scores-csv",
                    str(mask_scores_path),
                    "--source-scores-csv",
                    str(source_scores_path),
                    "--output-json",
                    str(output_json),
                    "--output-csv",
                    str(output_csv),
                ],
                check=False,
                cwd=repo_root,
                env={"PYTHONPATH": str(repo_root / "src")},
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            metrics = json.loads(output_json.read_text())
            rows = _read_csv(output_csv)

        self.assertAlmostEqual(metrics["mean_mask_f1"], 0.5)
        self.assertAlmostEqual(metrics["mean_anomaly_mask_iou"], 1.0 / 3.0)
        self.assertEqual(rows[0]["sample_id"], "pcb1/anomaly")
        self.assertAlmostEqual(float(rows[0]["mask_precision"]), 0.5)


def _save_mask(path: Path, array: np.ndarray) -> None:
    Image.fromarray(array, mode="L").save(path)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    unittest.main()
