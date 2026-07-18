from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluation.calibration import NormalThreshold, fit_normal_threshold


class NormalThresholdCalibrationTest(unittest.TestCase):
    def test_fits_quantile_from_normal_validation_heatmaps(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            first_heatmap = tmp_path / "first.npy"
            second_heatmap = tmp_path / "second.npy"
            np.save(first_heatmap, np.array([[0.0, 1.0]], dtype=np.float32))
            np.save(second_heatmap, np.array([[2.0, 3.0]], dtype=np.float32))

            result = fit_normal_threshold(
                [
                    {
                        "label": "0",
                        "fold_split": "val",
                        "heatmap_path": str(first_heatmap),
                    },
                    {
                        "label": "0",
                        "fold_split": "val",
                        "heatmap_path": str(second_heatmap),
                    },
                ],
                quantile=0.75,
            )

        self.assertEqual(
            result,
            NormalThreshold(
                quantile=0.75,
                threshold=2.25,
                num_images=2,
                num_pixels=4,
                source_split="val",
            ),
        )

    def test_rejects_anomalous_or_non_validation_rows(self) -> None:
        invalid_rows = [
            {"label": "1", "fold_split": "val", "heatmap_path": "unused.npy"},
            {"label": "0", "fold_split": "test", "heatmap_path": "unused.npy"},
        ]

        for row in invalid_rows:
            with self.subTest(row=row):
                with self.assertRaises(ValueError) as caught:
                    fit_normal_threshold([row])
                message = str(caught.exception)
                self.assertIn("row 0", message)
                self.assertIn(f"label={row['label']!r}", message)
                self.assertIn(f"fold_split={row['fold_split']!r}", message)

    def test_rejects_missing_fold_split(self) -> None:
        with self.assertRaises(ValueError) as caught:
            fit_normal_threshold([{"label": "0", "heatmap_path": "unused.npy"}])
        self.assertIn("row 0", str(caught.exception))
        self.assertIn("fold_split=None", str(caught.exception))

    def test_validates_all_rows_before_loading_heatmaps(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            missing_heatmap = Path(tmpdir) / "missing.npy"
            rows = [
                {
                    "label": "0",
                    "fold_split": "val",
                    "heatmap_path": str(missing_heatmap),
                },
                {
                    "label": "1",
                    "fold_split": "test",
                    "heatmap_path": "unused.npy",
                },
            ]

            with self.assertRaises(ValueError) as caught:
                fit_normal_threshold(rows)
            message = str(caught.exception)
            self.assertIn("row 1", message)
            self.assertIn("label='1'", message)
            self.assertIn("fold_split='test'", message)

    def test_rejects_empty_2d_heatmap_with_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            heatmap_path = Path(tmpdir) / "empty.npy"
            np.save(heatmap_path, np.empty((0, 2), dtype=np.float32))
            row = {
                "label": "0",
                "fold_split": "val",
                "heatmap_path": str(heatmap_path),
            }

            with self.assertRaises(ValueError) as caught:
                fit_normal_threshold([row])

            message = str(caught.exception)
            self.assertIn("row 0", message)
            self.assertIn(str(heatmap_path), message)
            self.assertIn("non-empty", message)

    def test_rejects_non_finite_heatmap_with_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            heatmap_path = Path(tmpdir) / "non_finite.npy"
            np.save(heatmap_path, np.array([[0.0, np.nan]], dtype=np.float32))
            row = {
                "label": "0",
                "fold_split": "val",
                "heatmap_path": str(heatmap_path),
            }

            with self.assertRaises(ValueError) as caught:
                fit_normal_threshold([row])

            message = str(caught.exception)
            self.assertIn("row 0", message)
            self.assertIn(str(heatmap_path), message)
            self.assertIn("finite", message)

    def test_rejects_3d_heatmap_with_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            heatmap_path = Path(tmpdir) / "three_dimensional.npy"
            np.save(heatmap_path, np.zeros((1, 2, 2), dtype=np.float32))
            row = {
                "label": "0",
                "fold_split": "val",
                "heatmap_path": str(heatmap_path),
            }

            with self.assertRaises(ValueError) as caught:
                fit_normal_threshold([row])

            message = str(caught.exception)
            self.assertIn("row 0", message)
            self.assertIn(str(heatmap_path), message)
            self.assertIn("2-D", message)

    def test_json_round_trip_preserves_threshold(self) -> None:
        threshold = NormalThreshold(
            quantile=0.995,
            threshold=0.42,
            num_images=3,
            num_pixels=12,
        )

        payload = json.loads(json.dumps(threshold.to_dict()))

        self.assertEqual(NormalThreshold.from_dict(payload), threshold)

    def test_from_dict_rejects_invalid_calibration_integrity(self) -> None:
        valid = {
            "quantile": 0.995,
            "threshold": 0.42,
            "num_images": 3,
            "num_pixels": 12,
            "source_split": "val",
        }
        invalid_cases = [
            ({"source_split": "test"}, "source_split must be 'val'"),
            ({"quantile": 0.0}, r"quantile must be in \(0, 1\)"),
            ({"quantile": 1.0}, r"quantile must be in \(0, 1\)"),
            ({"threshold": float("nan")}, "threshold must be finite"),
            ({"threshold": float("inf")}, "threshold must be finite"),
            ({"num_images": 0}, "num_images must be positive"),
            ({"num_pixels": 0}, "num_pixels must be positive"),
        ]

        for update, message in invalid_cases:
            with self.subTest(update=update):
                with self.assertRaisesRegex(ValueError, message):
                    NormalThreshold.from_dict({**valid, **update})

    def test_rejects_invalid_quantiles(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            heatmap_path = Path(tmpdir) / "valid.npy"
            np.save(heatmap_path, np.array([[0.0, 1.0]], dtype=np.float32))
            rows = [
                {
                    "label": "0",
                    "fold_split": "val",
                    "heatmap_path": str(heatmap_path),
                }
            ]

            for quantile in (-0.1, 0.0, 1.0, 1.1):
                with self.subTest(quantile=quantile):
                    with self.assertRaisesRegex(ValueError, "quantile"):
                        fit_normal_threshold(rows, quantile=quantile)

    def test_rejects_empty_rows(self) -> None:
        with self.assertRaises(ValueError):
            fit_normal_threshold([])


if __name__ == "__main__":
    unittest.main()
