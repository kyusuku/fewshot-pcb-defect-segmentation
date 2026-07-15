from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from experiments.spec import load_experiment_config
from scripts.build_paper_evidence import build_paper_evidence


def test_arxiv_smoke_builds_checked_analysis_and_evidence(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    config_path = repo / "configs" / "experiments" / "arxiv_smoke.yaml"
    output_root = tmp_path / "outputs"
    analysis_dir = tmp_path / "analysis"
    evidence_dir = tmp_path / "evidence"
    env = {"PYTHONPATH": str(repo / "src")}

    _run(
        [
            sys.executable,
            str(repo / "scripts" / "run_experiment_matrix.py"),
            "--config",
            str(config_path),
            "--output-root",
            str(output_root),
            "--device",
            "cpu",
            "--allow-dirty",
        ],
        repo,
        env,
    )
    checker = _run(
        [
            sys.executable,
            str(repo / "scripts" / "check_experiment_matrix.py"),
            "--config",
            str(config_path),
            "--output-root",
            str(output_root),
            "--device",
            "cpu",
            "--output-json",
            str(output_root / "matrix_summary.json"),
        ],
        repo,
        env,
    )
    assert checker.returncode == 0
    assert (output_root / "matrix_summary.json").exists()
    _run(
        [
            sys.executable,
            str(repo / "scripts" / "analyze_paper_results.py"),
            "--config",
            str(config_path),
            "--output-root",
            str(output_root),
            "--analysis-dir",
            str(analysis_dir),
            "--device",
            "cpu",
        ],
        repo,
        env,
    )

    config = load_experiment_config(config_path)
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


def _run(command: list[str], cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess:
    result = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result
