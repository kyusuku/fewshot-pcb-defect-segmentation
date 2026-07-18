from __future__ import annotations

from pathlib import Path


def test_readme_offline_smoke_prepares_ignored_fixture_and_manifest() -> None:
    repo = Path(__file__).resolve().parents[1]
    readme = (repo / "README.md").read_text(encoding="utf-8")

    create_fixture = "scripts/create_synthetic_data.py --output-dir data/debug_fixture"
    create_manifest = "scripts/create_manifests.py"
    manifest_root = "--visa-root data/debug_fixture/VisA"
    pcb_scope = "--visa-category pcb1"
    matrix_run = "scripts/run_experiment_matrix.py"

    assert create_fixture in readme
    assert create_manifest in readme
    assert manifest_root in readme
    assert pcb_scope in readme
    assert readme.index(create_fixture) < readme.index(create_manifest)
    assert readme.index(create_manifest) < readme.index(matrix_run)
