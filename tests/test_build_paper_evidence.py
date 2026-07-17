from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.build_paper_evidence import (
    _collect_ablation_rows,
    _validate_analysis_manifest,
    _validate_paper_assets,
    build_paper_evidence,
)
from scripts.render_method_figure import render_method_figure
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
        [
            {
                "category": "pcb1",
                "sample_id": "a",
                "area_stratum": "single",
                "image_path": "/private/dataset/pcb1.png",
                "heatmap_path": "/private/output/heatmap.npz",
            }
        ],
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
    assert manifest["ready_for_writing"] is False
    assert manifest["readiness_scope"] == "smoke_or_test_only"
    index = (evidence / "evidence_index.md").read_text()
    assert "Planned claim mapping" in index
    assert "Multi-scale DINOv2" in index
    assert "PatchCore-style baseline" in index
    assert "Anomaly-guided SAM2 refinement helps or hurts conditionally" in index
    assert "PatchCore reference baseline" not in index
    assert "Unconditional SAM2" not in index
    assert "test-command" in index
    failure_text = (evidence / "failure_strata.csv").read_text()
    assert "image_path" not in failure_text
    assert "/private/" not in failure_text


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
            "--qualitative-manifest",
            str(tmp_path / "missing-qualitative.csv"),
            "--method-figure-layout",
            str(tmp_path / "missing-method-layout.json"),
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


def test_ablation_rows_keep_heatmap_metrics_and_calibrated_policy(tmp_path: Path) -> None:
    config = {
        "categories": ["pcb1"],
        "fold_id": 0,
        "k": 4,
        "seed": 4880,
        "ablations": [{"name": "crop512", "method": "dinov2_multi"}],
    }
    run = RunSpec("dinov2_multi", "pcb1", 0, 4, 4880, variant="crop512")
    _write_run(
        tmp_path,
        run,
        "test/metrics.json",
        _metrics("calibrated_mean_anomaly_mask_f1", 0.5),
    )

    rows = _collect_ablation_rows(config, tmp_path)

    assert rows[0]["threshold_policy"] == "normal_q995"
    assert rows[0]["image_auroc"] == 0.9
    assert rows[0]["mean_anomaly_mask_f1"] == 0.5


def test_validate_paper_assets_requires_success_and_failure_for_each_category(
    tmp_path: Path,
) -> None:
    panels = {}
    for name in ("image", "mask", "anomaly_panel", "sam2_panel", "fusion_panel"):
        path = tmp_path / f"{name}.png"
        path.write_bytes(f"valid-{name}".encode())
        panels[name] = str(path)
    rows = []
    for role in ("success", "failure"):
        rows.append(
            {
                "category": "pcb1",
                "role": role,
                "sample_id": f"pcb1/{role}",
                "copied_paths_json": json.dumps(panels, sort_keys=True),
                **{
                    f"sha256_{name}": hashlib.sha256(Path(path).read_bytes()).hexdigest()
                    for name, path in panels.items()
                },
            }
        )
    qualitative = tmp_path / "qualitative_manifest.csv"
    _write_csv(qualitative, rows)
    method_png = tmp_path / "method_figure.png"
    method_layout = tmp_path / "method_figure_layout.json"
    render_method_figure(method_png, method_layout)

    _validate_paper_assets(qualitative, method_layout, categories=["pcb1"])

    _write_csv(qualitative, rows[:1])
    with pytest.raises(ValueError, match="success and failure"):
        _validate_paper_assets(qualitative, method_layout, categories=["pcb1"])


def test_validate_analysis_manifest_rejects_stale_generated_files(tmp_path: Path) -> None:
    analysis = tmp_path / "analysis"
    analysis.mkdir()
    per_image = analysis / "per_image_failure_analysis.csv"
    statistics = analysis / "paired_statistics.json"
    per_image.write_text("sample_id\na\n")
    statistics.write_text("{}\n")
    primary_rows = [
        {
            "run_id": "dinov2_multi__pcb1__fold0__k1__seed4880",
            "method": "dinov2_multi",
            "effective_execution_sha256": "a" * 64,
        },
        {
            "run_id": "dinov2_multi_sam2__pcb1__fold0__k1__seed4880",
            "method": "dinov2_multi_sam2",
            "effective_execution_sha256": "b" * 64,
        },
    ]
    manifest = {
        "schema_version": 1,
        "config_sha256": "",
        "source_runs": [
            {
                "run_id": row["run_id"],
                "effective_execution_sha256": row["effective_execution_sha256"],
            }
            for row in primary_rows
        ],
        "generated_files": [
            {"path": per_image.name, "sha256": hashlib.sha256(per_image.read_bytes()).hexdigest()},
            {
                "path": statistics.name,
                "sha256": hashlib.sha256(statistics.read_bytes()).hexdigest(),
            },
        ],
    }
    (analysis / "analysis_manifest.json").write_text(json.dumps(manifest))

    _validate_analysis_manifest(analysis, primary_rows, config_path=None)

    statistics.write_text('{"stale":true}\n')
    with pytest.raises(ValueError, match="checksum mismatch"):
        _validate_analysis_manifest(analysis, primary_rows, config_path=None)


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
