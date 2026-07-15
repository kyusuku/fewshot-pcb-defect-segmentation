from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import yaml

import experiments.spec as experiment_spec
from experiments.runner import matrix_report, run_matrix
from experiments.spec import expand_matrix, load_experiment_config, load_yaml_mapping
from scripts.analyze_paper_results import analyze_paper_results
from scripts.build_paper_evidence import build_paper_evidence
from tests.test_anomaly_baseline import _write_tiny_visa_fold_manifest
from utils.synthetic_data import create_synthetic_debug_datasets


def test_arxiv_smoke_builds_checked_analysis_and_evidence(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo = Path(__file__).resolve().parents[1]
    fixture_root = repo / "outputs" / f"pytest-arxiv-smoke-{tmp_path.name}"
    fixture_root.mkdir(parents=True, exist_ok=False)
    fixture = create_synthetic_debug_datasets(fixture_root / "fixture")
    manifest_path = fixture_root / "visa_pcb_folds.csv"
    _write_tiny_visa_fold_manifest(fixture.visa_root, manifest_path)
    base_config = load_yaml_mapping(repo / "configs" / "experiments" / "arxiv_smoke.yaml")
    base_config["manifest"] = manifest_path.relative_to(repo).as_posix()
    frozen_sha = hashlib.sha256(
        json.dumps(
            base_config,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    monkeypatch.setitem(experiment_spec._FROZEN_CONFIG_SHA256, "arxiv_smoke", frozen_sha)
    config_path = fixture_root / "arxiv_smoke.yaml"
    config_path.write_text(yaml.safe_dump(base_config, sort_keys=False), encoding="utf-8")
    output_root = fixture_root / "outputs"
    analysis_dir = fixture_root / "analysis"
    evidence_dir = fixture_root / "evidence"
    cache = fixture_root / "feature-cache"

    config = load_experiment_config(config_path)
    runs = expand_matrix(config)
    run_matrix(
        runs,
        config,
        config_path,
        output_root,
        output_root,
        "cpu",
        cache,
        allow_dirty=True,
    )
    report = matrix_report(
        runs,
        output_root,
        output_root,
        config,
        config_path,
        "cpu",
        cache,
    )
    assert report["ok"] is True
    analyze_paper_results(
        config_path=config_path,
        output_root=output_root,
        analysis_dir=analysis_dir,
        feature_cache_dir=cache,
        device="cpu",
    )

    build_paper_evidence(
        config=config,
        output_root=output_root,
        ablation_config={"ablations": [], "categories": []},
        ablation_output_root=tmp_path / "empty-ablations",
        analysis_dir=analysis_dir,
        evidence_dir=evidence_dir,
        validate_matrices=False,
        generation_command="pytest arxiv smoke",
    )

    assert (evidence_dir / "completion_manifest.json").exists()
    assert "anomaly_consistent_sam2" in (evidence_dir / "primary_results.md").read_text()
    shutil.rmtree(fixture_root)
