from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.paper_tables import (
    aggregate_primary_rows,
    collect_primary_rows,
    format_primary_markdown,
)
from experiments.spec import RunSpec


def test_aggregate_primary_rows_macro_averages_categories_then_seeds() -> None:
    rows = [
        {"method": "a", "category": "pcb1", "seed": 1, "k": 1, "metric": 0.2},
        {"method": "a", "category": "pcb1", "seed": 2, "k": 1, "metric": 0.4},
        {"method": "a", "category": "pcb2", "seed": 1, "k": 1, "metric": 0.6},
        {"method": "a", "category": "pcb2", "seed": 2, "k": 1, "metric": 0.8},
    ]

    summary = aggregate_primary_rows(rows, metric="metric")

    assert summary[0]["mean"] == pytest.approx(0.5)
    assert summary[0]["num_categories"] == 2
    assert summary[0]["num_seeds"] == 2


def test_collect_primary_rows_requires_complete_method_category_seed_grid(
    tmp_path: Path,
) -> None:
    config = {
        "categories": ["pcb1"],
        "shots": [1],
        "seeds": [4880],
        "fold_id": 0,
        "methods": ["dinov2_multi", "dinov2_multi_sam2"],
    }
    heatmap = RunSpec("dinov2_multi", "pcb1", 0, 1, 4880)
    mask = RunSpec("dinov2_multi_sam2", "pcb1", 0, 1, 4880)
    _write_run(
        tmp_path,
        heatmap,
        "test/metrics.json",
        {
            "image_auroc": 0.9,
            "pixel_auroc": 0.8,
            "aupro": 0.7,
            "calibrated_aggregate_pixel_f1": 0.6,
            "calibrated_aggregate_pixel_iou": 0.5,
            "calibrated_mean_anomaly_mask_f1": 0.4,
            "calibrated_mean_anomaly_mask_iou": 0.3,
            "calibrated_mean_anomaly_mask_precision": 0.2,
            "calibrated_mean_anomaly_mask_recall": 0.1,
        },
    )
    _write_run(
        tmp_path,
        mask,
        "test/mask_metrics.json",
        {
            "num_mask_images": 2,
            "mean_anomaly_mask_f1": 0.55,
            "mean_anomaly_mask_iou": 0.45,
            "mean_anomaly_mask_precision": 0.35,
            "mean_anomaly_mask_recall": 0.25,
        },
    )

    rows = collect_primary_rows(config, tmp_path)

    assert [(row["method"], row["category"], row["seed"]) for row in rows] == [
        ("dinov2_multi", "pcb1", 4880),
        ("dinov2_multi_sam2", "pcb1", 4880),
    ]
    assert rows[0]["threshold_policy"] == "normal_q995"
    assert rows[1]["threshold_policy"] == "binary_model_output"
    assert rows[0]["git_commit"] == "abc123"
    assert "dinov2_multi_sam2" in format_primary_markdown(
        aggregate_primary_rows(rows, metric="mean_anomaly_mask_f1")
    )

    (tmp_path / mask.run_id / "test" / "mask_metrics.json").unlink()
    with pytest.raises(ValueError, match="missing metrics"):
        collect_primary_rows(config, tmp_path)


def _write_run(
    root: Path,
    run: RunSpec,
    metrics_relative: str,
    metrics: dict[str, float | int],
) -> None:
    run_dir = root / run.run_id
    metrics_path = run_dir / metrics_relative
    metrics_path.parent.mkdir(parents=True)
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    (run_dir / "provenance.json").write_text(
        json.dumps(
            {
                "run_id": run.run_id,
                "git_commit": "abc123",
                "git_dirty": False,
                "manifest": {"path": "data/manifests/visa_pcb_folds.csv", "sha256": "m"},
                "effective_execution_sha256": "e" * 64,
                "run_spec_sha256": run.identity_sha256,
            }
        ),
        encoding="utf-8",
    )
