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

from evaluation.metrics import evaluate_heatmap_rows_at_threshold


class CalibratedHeatmapEvaluationTest(unittest.TestCase):
    def test_perfect_heatmap_returns_exact_aggregate_and_per_image_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            heatmap_path = root / "heatmap.npy"
            mask_path = root / "mask.png"
            np.save(heatmap_path, np.asarray([[0.1, 0.9], [0.2, 0.8]], dtype=np.float32))
            _save_mask(mask_path, np.asarray([[0, 255], [0, 255]], dtype=np.uint8))

            summary, per_image = evaluate_heatmap_rows_at_threshold(
                [
                    {
                        "sample_id": "pcb1/a",
                        "category": "pcb1",
                        "label": "1",
                        "heatmap_path": str(heatmap_path),
                        "mask_path": str(mask_path),
                    }
                ],
                threshold=0.5,
            )

        self.assertEqual(summary["calibrated_aggregate_pixel_precision"], 1.0)
        self.assertEqual(summary["calibrated_aggregate_pixel_recall"], 1.0)
        self.assertEqual(summary["calibrated_aggregate_pixel_f1"], 1.0)
        self.assertEqual(summary["calibrated_aggregate_pixel_iou"], 1.0)
        self.assertEqual(summary["calibrated_mean_anomaly_mask_f1"], 1.0)
        self.assertEqual(per_image[0]["threshold"], 0.5)
        self.assertEqual(per_image[0]["true_positive_pixels"], 2.0)
        self.assertEqual(per_image[0]["false_positive_pixels"], 0.0)
        self.assertEqual(per_image[0]["false_negative_pixels"], 0.0)

    def test_normal_row_ignores_stray_nonempty_mask(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            heatmap_path = root / "heatmap.npy"
            mask_path = root / "stray_mask.png"
            np.save(heatmap_path, np.asarray([[0.9, 0.1], [0.8, 0.2]], dtype=np.float32))
            _save_mask(mask_path, np.asarray([[255, 0], [255, 0]], dtype=np.uint8))

            _, per_image = evaluate_heatmap_rows_at_threshold(
                [
                    {
                        "sample_id": "pcb1/normal",
                        "category": "pcb1",
                        "label": "0",
                        "heatmap_path": str(heatmap_path),
                        "mask_path": str(mask_path),
                    }
                ],
                threshold=0.5,
            )

        self.assertEqual(per_image[0]["gt_positive_pixels"], 0.0)
        self.assertEqual(per_image[0]["true_positive_pixels"], 0.0)
        self.assertEqual(per_image[0]["false_positive_pixels"], 2.0)

    def test_resizes_ground_truth_to_preserve_heatmap_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            heatmap_path = root / "heatmap.npy"
            mask_path = root / "mask.png"
            np.save(heatmap_path, np.asarray([[0.1, 0.9], [0.1, 0.9]], dtype=np.float32))
            _save_mask(
                mask_path,
                np.asarray(
                    [[0, 0, 255, 255], [0, 0, 255, 255], [0, 0, 255, 255], [0, 0, 255, 255]],
                    dtype=np.uint8,
                ),
            )

            summary, _ = evaluate_heatmap_rows_at_threshold(
                [
                    {
                        "sample_id": "pcb1/aligned",
                        "category": "pcb1",
                        "label": "1",
                        "heatmap_path": str(heatmap_path),
                        "mask_path": str(mask_path),
                    }
                ],
                threshold=0.5,
            )

        self.assertEqual(summary["calibrated_aggregate_pixel_iou"], 1.0)

    def test_requires_explicit_binary_label_with_row_context(self) -> None:
        for label in (None, "", "normal", "2"):
            row = {
                "sample_id": "pcb1/bad-label",
                "category": "pcb1",
                "heatmap_path": "unused.npy",
            }
            if label is not None:
                row["label"] = label
            with self.subTest(label=label):
                with self.assertRaises(ValueError) as caught:
                    evaluate_heatmap_rows_at_threshold([row], threshold=0.5)
                message = str(caught.exception)
                self.assertIn("row 0", message)
                self.assertIn("pcb1/bad-label", message)
                self.assertIn("label", message)

    def test_requires_nonempty_heatmap_path_with_row_context(self) -> None:
        for heatmap_path in (None, ""):
            row = {"sample_id": "pcb1/no-heatmap", "category": "pcb1", "label": "0"}
            if heatmap_path is not None:
                row["heatmap_path"] = heatmap_path
            with self.subTest(heatmap_path=heatmap_path):
                with self.assertRaises(ValueError) as caught:
                    evaluate_heatmap_rows_at_threshold([row], threshold=0.5)
                message = str(caught.exception)
                self.assertIn("row 0", message)
                self.assertIn("pcb1/no-heatmap", message)
                self.assertIn("heatmap_path", message)


class CalibrationScriptTest(unittest.TestCase):
    def test_calibration_cli_writes_explicit_normal_validation_threshold(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            first = root / "first.npy"
            second = root / "second.npy"
            scores_csv = root / "scores.csv"
            output_json = root / "nested" / "calibration.json"
            np.save(first, np.asarray([[0.0, 1.0]], dtype=np.float32))
            np.save(second, np.asarray([[2.0, 3.0]], dtype=np.float32))
            _write_csv(
                scores_csv,
                ["sample_id", "label", "fold_split", "heatmap_path"],
                [
                    {
                        "sample_id": "n1",
                        "label": "0",
                        "fold_split": "val",
                        "heatmap_path": first.name,
                    },
                    {
                        "sample_id": "n2",
                        "label": "0",
                        "fold_split": "val",
                        "heatmap_path": second.name,
                    },
                ],
            )

            result = _run_script(
                repo_root,
                "calibrate_heatmaps.py",
                "--scores-csv",
                str(scores_csv),
                "--quantile",
                "0.75",
                "--output-json",
                str(output_json),
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            payload = json.loads(output_json.read_text())
            raw = output_json.read_bytes()

        self.assertEqual(payload["threshold"], 2.25)
        self.assertEqual(payload["source_split"], "val")
        self.assertEqual(payload["num_images"], 2)
        self.assertTrue(raw.endswith(b"\n"))
        self.assertIn("threshold: 2.25000000", result.stdout)

    def test_calibration_cli_rejects_missing_fold_split_before_loading_heatmaps(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            scores_csv = root / "scores.csv"
            output_json = root / "calibration.json"
            _write_csv(
                scores_csv,
                ["sample_id", "label", "heatmap_path"],
                [{"sample_id": "n1", "label": "0", "heatmap_path": "missing.npy"}],
            )

            result = _run_script(
                repo_root,
                "calibrate_heatmaps.py",
                "--scores-csv",
                str(scores_csv),
                "--output-json",
                str(output_json),
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("fold_split=None", result.stderr)
        self.assertNotIn("FileNotFoundError", result.stderr)


class EvaluateHeatmapsCalibrationScriptTest(unittest.TestCase):
    def test_calibrated_cli_writes_summary_and_stable_per_image_csv(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            scores_csv, _ = _write_perfect_scores_fixture(root)
            calibration_json = root / "calibration.json"
            output_json = root / "metrics" / "metrics.json"
            per_image_csv = root / "metrics" / "per_image.csv"
            calibration_json.write_text(
                json.dumps(
                    {
                        "quantile": 0.995,
                        "threshold": 0.5,
                        "num_images": 2,
                        "num_pixels": 8,
                        "source_split": "val",
                    }
                )
            )

            result = _run_script(
                repo_root,
                "evaluate_heatmaps.py",
                "--scores-csv",
                str(scores_csv),
                "--output-json",
                str(output_json),
                "--calibration-json",
                str(calibration_json),
                "--per-image-csv",
                str(per_image_csv),
                "--max-pixels",
                "0",
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            metrics = json.loads(output_json.read_text())
            with per_image_csv.open(newline="") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
                fieldnames = reader.fieldnames

        self.assertEqual(metrics["calibration_quantile"], 0.995)
        self.assertEqual(metrics["calibration_threshold"], 0.5)
        self.assertEqual(metrics["calibration_source_split"], "val")
        self.assertEqual(metrics["calibration_num_images"], 2)
        self.assertEqual(metrics["calibration_num_pixels"], 8)
        self.assertEqual(metrics["calibrated_aggregate_pixel_f1"], 1.0)
        self.assertEqual(len(rows), 1)
        self.assertEqual(float(rows[0]["threshold"]), 0.5)
        self.assertEqual(
            fieldnames,
            [
                "dataset",
                "sample_id",
                "category",
                "label",
                "fold_split",
                "threshold",
                "mask_precision",
                "mask_recall",
                "mask_f1",
                "mask_iou",
                "pred_positive_pixels",
                "gt_positive_pixels",
                "true_positive_pixels",
                "false_positive_pixels",
                "false_negative_pixels",
                "heatmap_path",
                "mask_path",
            ],
        )

    def test_legacy_cli_still_succeeds_without_calibration(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            scores_csv, _ = _write_perfect_scores_fixture(root)
            output_json = root / "metrics.json"

            result = _run_script(
                repo_root,
                "evaluate_heatmaps.py",
                "--scores-csv",
                str(scores_csv),
                "--output-json",
                str(output_json),
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            metrics = json.loads(output_json.read_text())

        self.assertIn("best_pixel_f1", metrics)
        self.assertNotIn("calibration_threshold", metrics)

    def test_per_image_csv_without_calibration_fails_clearly(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            scores_csv, _ = _write_perfect_scores_fixture(root)
            result = _run_script(
                repo_root,
                "evaluate_heatmaps.py",
                "--scores-csv",
                str(scores_csv),
                "--output-json",
                str(root / "metrics.json"),
                "--per-image-csv",
                str(root / "per_image.csv"),
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--per-image-csv requires --calibration-json", result.stderr)

    def test_cli_rejects_invalid_calibration_json_integrity(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        invalid_cases = [
            ({"source_split": "test"}, "source_split must be 'val'"),
            ({"quantile": 0.0}, "quantile must be in (0, 1)"),
            ({"threshold": float("nan")}, "threshold must be finite"),
            ({"threshold": float("inf")}, "threshold must be finite"),
            ({"num_images": 0}, "num_images must be positive"),
            ({"num_pixels": 0}, "num_pixels must be positive"),
        ]
        for update, expected in invalid_cases:
            with self.subTest(update=update), tempfile.TemporaryDirectory() as tmpdir:
                root = Path(tmpdir)
                scores_csv, _ = _write_perfect_scores_fixture(root)
                calibration_json = root / "calibration.json"
                payload = {
                    "quantile": 0.995,
                    "threshold": 0.5,
                    "num_images": 2,
                    "num_pixels": 8,
                    "source_split": "val",
                    **update,
                }
                calibration_json.write_text(json.dumps(payload))

                result = _run_script(
                    repo_root,
                    "evaluate_heatmaps.py",
                    "--scores-csv",
                    str(scores_csv),
                    "--output-json",
                    str(root / "metrics.json"),
                    "--calibration-json",
                    str(calibration_json),
                )

                self.assertNotEqual(result.returncode, 0)
                self.assertIn(expected, result.stderr)


def _write_perfect_scores_fixture(root: Path) -> tuple[Path, Path]:
    heatmap_path = root / "heatmap.npy"
    mask_path = root / "mask.png"
    scores_csv = root / "scores.csv"
    np.save(heatmap_path, np.asarray([[0.1, 0.9], [0.2, 0.8]], dtype=np.float32))
    _save_mask(mask_path, np.asarray([[0, 255], [0, 255]], dtype=np.uint8))
    _write_csv(
        scores_csv,
        ["sample_id", "category", "label", "image_score", "heatmap_path", "mask_path"],
        [
            {
                "sample_id": "pcb1/a",
                "category": "pcb1",
                "label": "1",
                "image_score": "0.9",
                "heatmap_path": heatmap_path.name,
                "mask_path": mask_path.name,
            }
        ],
    )
    return scores_csv, heatmap_path


def _run_script(repo_root: Path, script: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(repo_root / "scripts" / script), *args],
        check=False,
        cwd=repo_root,
        env={"PYTHONPATH": str(repo_root / "src")},
        capture_output=True,
        text=True,
    )


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _save_mask(path: Path, array: np.ndarray) -> None:
    Image.fromarray(array, mode="L").save(path)


if __name__ == "__main__":
    unittest.main()
