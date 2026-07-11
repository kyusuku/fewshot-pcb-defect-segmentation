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

from sam_refine.fusion import agreement_features, fuse_masks


class MaskFusionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.anomaly = np.array([[1, 1, 0], [0, 1, 0]], dtype=np.int64)
        self.sam2 = np.array([[0, 2, 2], [0, 3, 0]], dtype=np.float32)

    def test_exact_fusion_modes_are_binary_uint8_and_shape_preserving(self) -> None:
        expected = {
            "anomaly": np.array([[1, 1, 0], [0, 1, 0]], dtype=np.uint8),
            "sam2": np.array([[0, 1, 1], [0, 1, 0]], dtype=np.uint8),
            "intersection": np.array([[0, 1, 0], [0, 1, 0]], dtype=np.uint8),
            "union": np.array([[1, 1, 1], [0, 1, 0]], dtype=np.uint8),
        }

        for mode, expected_mask in expected.items():
            with self.subTest(mode=mode):
                actual = fuse_masks(self.anomaly, self.sam2, mode=mode)
                np.testing.assert_array_equal(actual, expected_mask)
                self.assertEqual(actual.dtype, np.uint8)
                self.assertEqual(actual.shape, self.anomaly.shape)

    def test_agreement_features_report_exact_areas_iou_and_expansion(self) -> None:
        features = agreement_features(self.anomaly, self.sam2)

        self.assertEqual(features["anomaly_pixels"], 3.0)
        self.assertEqual(features["sam2_pixels"], 3.0)
        self.assertEqual(features["intersection_pixels"], 2.0)
        self.assertEqual(features["union_pixels"], 4.0)
        self.assertEqual(features["mask_iou"], 0.5)
        self.assertEqual(features["sam2_to_anomaly_area_ratio"], 1.0)

    def test_empty_mask_agreement_conventions_are_explicit(self) -> None:
        empty = np.zeros((2, 2), dtype=np.uint8)

        both_empty = agreement_features(empty, empty)
        anomaly_empty = agreement_features(empty, np.eye(2, dtype=np.uint8))

        self.assertEqual(both_empty["mask_iou"], 1.0)
        self.assertEqual(both_empty["sam2_to_anomaly_area_ratio"], 1.0)
        self.assertEqual(anomaly_empty["mask_iou"], 0.0)
        self.assertTrue(np.isinf(anomaly_empty["sam2_to_anomaly_area_ratio"]))

    def test_selective_uses_intersection_only_for_acceptable_agreement(self) -> None:
        anomaly = np.array([[1, 1], [1, 0]], dtype=np.uint8)
        high_agreement = np.array([[1, 1], [0, 0]], dtype=np.uint8)
        low_agreement = np.array([[0, 0], [0, 1]], dtype=np.uint8)
        overexpanded = np.ones((2, 2), dtype=np.uint8)

        np.testing.assert_array_equal(
            fuse_masks(
                anomaly,
                high_agreement,
                mode="selective",
                min_iou=0.5,
                max_expansion=2.0,
            ),
            high_agreement,
        )
        np.testing.assert_array_equal(
            fuse_masks(anomaly, low_agreement, mode="selective", min_iou=0.25),
            anomaly,
        )
        np.testing.assert_array_equal(
            fuse_masks(
                np.array([[1, 0], [0, 0]], dtype=np.uint8),
                overexpanded,
                mode="selective",
                min_iou=0.25,
                max_expansion=2.0,
            ),
            np.array([[1, 0], [0, 0]], dtype=np.uint8),
        )

    def test_invalid_inputs_and_controls_are_rejected(self) -> None:
        valid = np.zeros((2, 2), dtype=np.uint8)
        cases = [
            ((np.zeros((1, 2, 2)), valid), {"mode": "anomaly"}, "2-D"),
            ((valid, np.zeros((3, 2))), {"mode": "anomaly"}, "identical"),
            ((valid, valid), {"mode": "weighted"}, "mode"),
            ((valid, valid), {"mode": "selective", "min_iou": -0.1}, "min_iou"),
            ((valid, valid), {"mode": "selective", "min_iou": 1.1}, "min_iou"),
            ((valid, valid), {"mode": "selective", "max_expansion": 0.0}, "max_expansion"),
            ((valid, valid), {"mode": "selective", "max_expansion": np.inf}, "max_expansion"),
        ]

        for inputs, kwargs, message in cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, message):
                    fuse_masks(*inputs, **kwargs)


class FuseSavedMasksScriptTest(unittest.TestCase):
    def test_offline_intersection_uses_portable_paths_from_different_cwd(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_dir = root / "source"
            output_dir = root / "fused"
            other_cwd = root / "other"
            source_dir.mkdir()
            other_cwd.mkdir()
            heatmap = np.array(
                [[0.9, 0.9, 0.1], [0.9, 0.9, 0.1], [0.1, 0.1, 0.1]],
                dtype=np.float32,
            )
            np.save(source_dir / "heatmap.npy", heatmap)
            Image.fromarray(np.ones((3, 3), dtype=np.uint8) * 255, mode="L").save(
                source_dir / "raw.png"
            )
            scores_path = source_dir / "mask_scores.csv"
            _write_csv(
                scores_path,
                ["sample_id", "category", "label", "sam2_mask_path", "heatmap_path"],
                [
                    {
                        "sample_id": "pcb1/anomaly",
                        "category": "pcb1",
                        "label": "1",
                        "sam2_mask_path": "raw.png",
                        "heatmap_path": "heatmap.npy",
                    }
                ],
            )
            calibration_path = root / "calibration.json"
            calibration_path.write_text(
                json.dumps(
                    {
                        "quantile": 0.995,
                        "threshold": 0.5,
                        "num_images": 2,
                        "num_pixels": 18,
                        "source_split": "val",
                    }
                )
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(repo_root / "scripts" / "fuse_saved_masks.py"),
                    "--mask-scores-csv",
                    str(scores_path),
                    "--calibration-json",
                    str(calibration_path),
                    "--mask-output",
                    "intersection",
                    "--output-dir",
                    str(output_dir),
                ],
                check=False,
                cwd=other_cwd,
                env={"PYTHONPATH": str(repo_root / "src")},
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            rows = _read_csv(output_dir / "mask_scores.csv")
            pred_path = output_dir / rows[0]["pred_mask_path"]
            pred = np.asarray(Image.open(pred_path).convert("L")) > 0

        self.assertEqual(int(pred.sum()), 4)
        self.assertEqual(rows[0]["mask_output"], "intersection")
        self.assertEqual(rows[0]["proposal_threshold"], "0.50000000")
        self.assertEqual(rows[0]["intersection_pixels"], "4.00000000")
        self.assertFalse(Path(rows[0]["sam2_mask_path"]).is_absolute())
        self.assertFalse(Path(rows[0]["heatmap_path"]).is_absolute())
        self.assertFalse(Path(rows[0]["pred_mask_path"]).is_absolute())

    def test_missing_required_artifacts_fail_with_row_context(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            calibration_path = root / "calibration.json"
            calibration_path.write_text(
                json.dumps(
                    {
                        "quantile": 0.995,
                        "threshold": 0.5,
                        "num_images": 1,
                        "num_pixels": 4,
                        "source_split": "val",
                    }
                )
            )
            for missing_field in ("sam2_mask_path", "heatmap_path"):
                with self.subTest(missing_field=missing_field):
                    scores_path = root / f"missing_{missing_field}.csv"
                    row = {
                        "sample_id": "pcb1/anomaly",
                        "sam2_mask_path": "raw.png",
                        "heatmap_path": "heatmap.npy",
                    }
                    row[missing_field] = ""
                    _write_csv(scores_path, list(row), [row])
                    result = subprocess.run(
                        [
                            sys.executable,
                            str(repo_root / "scripts" / "fuse_saved_masks.py"),
                            "--mask-scores-csv",
                            str(scores_path),
                            "--calibration-json",
                            str(calibration_path),
                            "--mask-output",
                            "anomaly",
                            "--output-dir",
                            str(root / "out"),
                        ],
                        check=False,
                        cwd=root,
                        env={"PYTHONPATH": str(repo_root / "src")},
                        capture_output=True,
                        text=True,
                    )

                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("row 0", result.stderr)
                    self.assertIn(missing_field, result.stderr)

    def test_calibration_json_is_required(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [
                sys.executable,
                str(repo_root / "scripts" / "fuse_saved_masks.py"),
                "--mask-scores-csv",
                "unused.csv",
                "--mask-output",
                "anomaly",
                "--output-dir",
                "unused",
            ],
            check=False,
            cwd=repo_root,
            env={"PYTHONPATH": str(repo_root / "src")},
            capture_output=True,
            text=True,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--calibration-json", result.stderr)


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
