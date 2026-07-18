from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.analyze_paper_results import (
    _assign_strata,
    _paired_statistics,
    analyze_paper_results,
)


def test_analyze_paper_results_writes_paired_failure_outputs(tmp_path: Path) -> None:
    output = tmp_path / "runs"
    analysis = tmp_path / "analysis"
    heatmap_run = "dinov2_multi__pcb1__fold0__k1__seed4880"
    sam2_run = "dinov2_multi_sam2__pcb1__fold0__k1__seed4880"
    fusion_run = "anomaly_consistent_sam2__pcb1__fold0__k1__seed4880"
    sample_id = "pcb1/anomaly_000"

    mask = np.asarray([[255, 255, 0], [0, 0, 0]], dtype=np.uint8)
    heatmap = np.asarray([[1.0, 0.8, 0.0], [0.0, 0.7, 0.0]], dtype=np.float32)
    sam2 = np.asarray([[255, 255, 0], [0, 255, 0]], dtype=np.uint8)
    fused = np.asarray([[255, 255, 0], [0, 0, 0]], dtype=np.uint8)

    heatmap_dir = output / heatmap_run / "test"
    sam2_dir = output / sam2_run / "test"
    fusion_dir = output / fusion_run / "test"
    heatmap_dir.mkdir(parents=True)
    sam2_dir.mkdir(parents=True)
    fusion_dir.mkdir(parents=True)
    np.save(heatmap_dir / "heatmap.npy", heatmap)
    _save_mask(heatmap_dir / "mask.png", mask)
    _save_mask(sam2_dir / "sam2.png", sam2)
    _save_mask(sam2_dir / "sam2_pred.png", sam2)
    _save_mask(fusion_dir / "fused.png", fused)

    _write_csv(
        heatmap_dir / "scores.csv",
        [
            {
                "dataset": "visa_pcb",
                "sample_id": sample_id,
                "category": "pcb1",
                "label": "1",
                "fold_split": "test",
                "mask_path": "mask.png",
                "heatmap_path": "heatmap.npy",
            }
        ],
    )
    _write_csv(
        heatmap_dir / "per_image.csv",
        [
            {
                "sample_id": sample_id,
                "category": "pcb1",
                "label": "1",
                "threshold": "1.0",
                "mask_f1": "0.6666666667",
                "mask_iou": "0.5",
            }
        ],
    )
    _write_csv(
        sam2_dir / "mask_scores.csv",
        [
            {
                "dataset": "visa_pcb",
                "sample_id": sample_id,
                "category": "pcb1",
                "label": "1",
                "fold_split": "test",
                "mask_path": str(heatmap_dir / "mask.png"),
                "heatmap_path": str(heatmap_dir / "heatmap.npy"),
                "sam2_mask_path": "sam2.png",
                "pred_mask_path": "sam2_pred.png",
            }
        ],
    )
    _write_csv(
        sam2_dir / "mask_per_image.csv",
        [
            {
                "sample_id": sample_id,
                "category": "pcb1",
                "label": "1",
                "mask_f1": "0.8",
                "mask_iou": "0.6666666667",
            }
        ],
    )
    _write_csv(
        fusion_dir / "mask_scores.csv",
        [
            {
                "dataset": "visa_pcb",
                "sample_id": sample_id,
                "category": "pcb1",
                "label": "1",
                "fold_split": "test",
                "mask_path": str(heatmap_dir / "mask.png"),
                "heatmap_path": str(heatmap_dir / "heatmap.npy"),
                "sam2_mask_path": str(sam2_dir / "sam2.png"),
                "pred_mask_path": "fused.png",
            }
        ],
    )
    _write_csv(
        fusion_dir / "mask_per_image.csv",
        [
            {
                "sample_id": sample_id,
                "category": "pcb1",
                "label": "1",
                "mask_f1": "1.0",
                "mask_iou": "1.0",
            }
        ],
    )

    analyze_paper_results(
        output_root=output,
        analysis_dir=analysis,
        categories=["pcb1"],
        shots=[1],
        seeds=[4880],
        validate_matrix=False,
        bootstrap_samples=32,
    )

    rows = _read_csv(analysis / "per_image_failure_analysis.csv")
    assert len(rows) == 1
    assert rows[0]["sample_id"] == sample_id
    assert float(rows[0]["sam2_delta_f1"]) > 0.0
    assert float(rows[0]["fusion_delta_f1"]) > 0.0
    assert rows[0]["area_stratum"] == "single"
    assert rows[0]["thinness_stratum"] == "single"

    statistics = json.loads((analysis / "paired_statistics.json").read_text())
    analysis_manifest = json.loads((analysis / "analysis_manifest.json").read_text())
    assert set(statistics["comparisons"]) == {
        "dinov2_multi_vs_dinov2_multi_sam2",
        "dinov2_multi_vs_anomaly_consistent_sam2",
        "dinov2_multi_sam2_vs_anomaly_consistent_sam2",
    }
    assert statistics["strata"]["area"]["edges"] == []
    assert statistics["strata"]["thinness"]["edges"] == []
    comparison = statistics["comparisons"]["dinov2_multi_vs_dinov2_multi_sam2"]
    assert comparison["scope"]["sample_inclusion"] == "anomaly_images_only"
    assert comparison["overall"]["f1"]["num_unique_images"] == 1
    assert set(comparison["by_k"]) == {"1"}
    assert set(comparison["by_category"]) == {"pcb1"}
    assert {item["path"] for item in analysis_manifest["generated_files"]} == {
        "paired_statistics.json",
        "per_image_failure_analysis.csv",
    }
    assert {item["run_id"] for item in analysis_manifest["source_runs"]} == {
        heatmap_run,
        sam2_run,
        fusion_run,
    }


def test_strata_keep_normal_rows_out_of_defect_tertiles() -> None:
    rows = [
        {"gt_area_fraction": 0.0, "gt_thinness": 0.0},
        {"gt_area_fraction": 0.2, "gt_thinness": 1.0},
    ]

    strata = _assign_strata(rows)

    assert rows[0]["area_stratum"] == "normal"
    assert rows[0]["thinness_stratum"] == "normal"
    assert rows[1]["area_stratum"] == "single"
    assert rows[1]["thinness_stratum"] == "single"
    assert strata["area"]["labels"] == ["normal", "single"]


def test_paired_statistics_excludes_normal_rows_from_primary_mask_claims() -> None:
    rows = [
        {
            "category": "pcb1",
            "k": 1,
            "seed": 4880,
            "sample_id": "pcb1/anomaly",
            "label": "1",
            "area_stratum": "single",
            "thinness_stratum": "single",
            "anomaly_f1": 0.4,
            "sam2_f1": 0.4,
            "fusion_f1": 0.4,
            "anomaly_iou": 0.3,
            "sam2_iou": 0.3,
            "fusion_iou": 0.3,
        },
        {
            "category": "pcb1",
            "k": 1,
            "seed": 4880,
            "sample_id": "pcb1/normal",
            "label": "0",
            "area_stratum": "normal",
            "thinness_stratum": "normal",
            "anomaly_f1": 0.0,
            "sam2_f1": 1.0,
            "fusion_f1": 1.0,
            "anomaly_iou": 0.0,
            "sam2_iou": 1.0,
            "fusion_iou": 1.0,
        },
    ]

    statistics = _paired_statistics(
        rows,
        {"area": {"labels": ["normal", "single"]}, "thinness": {"labels": []}},
        bootstrap_samples=16,
    )

    comparison = statistics["comparisons"]["dinov2_multi_vs_dinov2_multi_sam2"]
    assert comparison["overall"]["f1"]["mean_delta"] == 0.0
    assert comparison["scope"]["excluded_normal_rows"] == 1


def _save_mask(path: Path, data: np.ndarray) -> None:
    Image.fromarray(data, mode="L").save(path)


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
