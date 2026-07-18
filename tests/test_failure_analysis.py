from __future__ import annotations

import numpy as np
import pytest

from evaluation.failure_analysis import describe_mask_pair


def test_describe_mask_pair_reports_geometry_and_sam2_delta() -> None:
    target = np.asarray([[1, 1, 0], [0, 0, 0]], dtype=np.uint8)
    anomaly = np.asarray([[1, 0, 0], [0, 0, 0]], dtype=np.uint8)
    sam2 = np.asarray([[1, 1, 0], [0, 1, 0]], dtype=np.uint8)
    heatmap = np.asarray([[1.0, 0.8, 0.0], [0.0, 0.7, 0.0]], dtype=np.float32)

    row = describe_mask_pair(target, anomaly, sam2, heatmap)

    assert row["gt_area_fraction"] == pytest.approx(2 / 6)
    assert row["gt_components"] == 1.0
    assert row["sam2_to_anomaly_area_ratio"] == 3.0
    assert row["sam2_delta_f1"] > 0.0
    assert row["mean_anomaly_inside_sam2"] > 0.8


def test_describe_mask_pair_validates_shapes_and_finite_heatmap() -> None:
    target = np.zeros((2, 2), dtype=np.uint8)
    anomaly = np.zeros((2, 2), dtype=np.uint8)
    sam2 = np.zeros((2, 2), dtype=np.uint8)

    with pytest.raises(ValueError, match="same shape"):
        describe_mask_pair(target, anomaly[:, :1], sam2, np.zeros((2, 2)))
    with pytest.raises(ValueError, match="finite"):
        describe_mask_pair(target, anomaly, sam2, np.full((2, 2), np.nan))
