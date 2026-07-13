from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

from scripts.build_paper_evidence import build_paper_evidence
from tests.test_paper_tables import _write_run
from experiments.spec import RunSpec


def test_build_paper_evidence_writes_compact_manifest(tmp_path: Path) -> None:
    config = {
        "categories": ["pcb1"],
        "shots": [1],
        "seeds": [4880],
        "fold_id": 0,
        "methods": ["dinov2_multi", "dinov2_multi_sam2"],
    }
    ablation_config = {
        "categories": ["pcb1"],
        "fold_id": 0,
        "k": 1,
        "seed": 4880,
        "ablations": [{"name": "fusion_union", "method": "anomaly_consistent_sam2"}],
    }
    heatmap = RunSpec("dinov2_multi", "pcb1", 0, 1, 4880)
    mask = RunSpec("dinov2_multi_sam2", "pcb1", 0, 1, 4880)
    ablation = RunSpec("anomaly_consistent_sam2", "pcb1", 0, 1, 4880, variant="fusion_union")
    for run, relative, metric in [
        (heatmap, "test/metrics.json", "calibrated_mean_anomaly_mask_f1"),
        (mask, "test/mask_metrics.json", "mean_anomaly_mask_f1"),
        (ablation, "test/mask_metrics.json", "mean_anomaly_mask_f1"),
    ]:
        _write_run(
            tmp_path / "primary" if run is not ablation else tmp_path / "ablation",
            run,
            relative,
            _metrics(metric, 0.5),
        )
    analysis = tmp_path / "analysis"
    analysis.mkdir()
    (analysis / "paired_statistics.json").write_text('{"comparisons":{},"strata":{}}\n')
    _write_csv(
        analysis / "per_image_failure_analysis.csv",
        [{"category": "pcb1", "sample_id": "a", "area_stratum": "single"}],
    )

    build_paper_evidence(
        config=config,
        output_root=tmp_path / "primary",
        ablation_config=ablation_config,
        ablation_output_root=tmp_path / "ablation",
        analysis_dir=analysis,
        evidence_dir=tmp_path / "evidence",
        validate_matrices=False,
        generation_command="test-command",
    )

    evidence = tmp_path / "evidence"
    assert (evidence / "primary_results.csv").is_file()
    assert "dinov2_multi_sam2" in (evidence / "primary_results.md").read_text()
    manifest = json.loads((evidence / "completion_manifest.json").read_text())
    generated = {item["path"] for item in manifest["generated_files"]}
    assert "primary_results.csv" in generated
    assert "paired_statistics.json" in generated
    assert all(not path.endswith((".png", ".npy", ".pt")) for path in generated)
    assert manifest["generation_command"] == "test-command"


def test_build_paper_evidence_incomplete_matrix_exits_one(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            str(repo / "scripts" / "build_paper_evidence.py"),
            "--config",
            str(repo / "configs" / "experiments" / "arxiv_smoke.yaml"),
            "--output-root",
            str(tmp_path / "missing-primary"),
            "--ablation-config",
            str(repo / "configs" / "experiments" / "arxiv_ablations.yaml"),
            "--ablation-output-root",
            str(tmp_path / "missing-ablation"),
            "--analysis-dir",
            str(tmp_path / "missing-analysis"),
            "--evidence-dir",
            str(tmp_path / "evidence"),
            "--device",
            "cpu",
        ],
        cwd=repo,
        env={"PYTHONPATH": str(repo / "src")},
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1


def _metrics(key: str, value: float) -> dict[str, float | int]:
    if key.startswith("calibrated"):
        return {
            "image_auroc": 0.9,
            "pixel_auroc": 0.8,
            "aupro": 0.7,
            "calibrated_aggregate_pixel_f1": value,
            "calibrated_aggregate_pixel_iou": value,
            "calibrated_mean_anomaly_mask_f1": value,
            "calibrated_mean_anomaly_mask_iou": value,
            "calibrated_mean_anomaly_mask_precision": value,
            "calibrated_mean_anomaly_mask_recall": value,
        }
    return {
        "num_mask_images": 1,
        "mean_anomaly_mask_f1": value,
        "mean_anomaly_mask_iou": value,
        "mean_anomaly_mask_precision": value,
        "mean_anomaly_mask_recall": value,
    }


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
