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
            with self.subTest(row=row), self.assertRaises(ValueError):
                fit_normal_threshold([row])

    def test_rejects_missing_fold_split(self) -> None:
        with self.assertRaises(ValueError):
            fit_normal_threshold([{"label": "0", "heatmap_path": "unused.npy"}])

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

            with self.assertRaisesRegex(ValueError, "^normal validation$"):
                fit_normal_threshold(rows)

    def test_json_round_trip_preserves_threshold(self) -> None:
        threshold = NormalThreshold(
            quantile=0.995,
            threshold=0.42,
            num_images=3,
            num_pixels=12,
        )

        payload = json.loads(json.dumps(threshold.to_dict()))

        self.assertEqual(NormalThreshold.from_dict(payload), threshold)

    def test_rejects_invalid_quantiles(self) -> None:
        for quantile in (-0.1, 0.0, 1.0, 1.1):
            with self.subTest(quantile=quantile), self.assertRaises(ValueError):
                fit_normal_threshold([], quantile=quantile)

    def test_rejects_empty_rows(self) -> None:
        with self.assertRaises(ValueError):
            fit_normal_threshold([])


if __name__ == "__main__":
    unittest.main()
