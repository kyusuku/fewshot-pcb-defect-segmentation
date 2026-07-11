from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

import pytest

from experiments.provenance import (
    atomic_write_json,
    build_provenance,
    canonical_json,
    git_state,
    sha256_file,
)
from experiments.spec import RunSpec


def test_manifest_checksum_changes_with_content(tmp_path: Path) -> None:
    path = tmp_path / "manifest.csv"
    path.write_text("a\n")
    first = sha256_file(path)
    path.write_text("b\n")
    assert sha256_file(path) != first


def test_provenance_is_canonical_complete_and_does_not_leak_absolute_paths(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.csv"
    config_path = tmp_path / "arxiv_smoke.yaml"
    checkpoint = tmp_path / "model.pt"
    model_config = tmp_path / "model.yaml"
    _write_manifest(manifest, ["pcb1/0001", "pcb1/0002"])
    config = {"name": "arxiv_smoke", "shots": [1]}
    config_path.write_text("name: arxiv_smoke\nshots: [1]\n")
    checkpoint.write_bytes(b"weights")
    model_config.write_text("model: tiny\n")
    run = RunSpec(
        "dinov2_multi",
        "pcb1",
        0,
        2,
        4880,
        overrides={"crop_sizes": [32], "fusion": "max"},
    )

    record = build_provenance(
        run_spec=run,
        manifest_path=manifest,
        support_ids=["pcb1/0002", "pcb1/0001"],
        git_commit="abc123",
        git_dirty=False,
        config_path=config_path,
        config=config,
        checkpoint_paths={"sam2": checkpoint},
        model_config_paths={"sam2": model_config},
        cache_identity={"extractor_sha256": "a" * 64},
        library_names=("pytest",),
    )

    assert record["support_ids"] == ["pcb1/0001", "pcb1/0002"]
    assert record["git"] == {"commit": "abc123", "dirty": False}
    assert record["git_commit"] == "abc123"
    assert record["git_dirty"] is False
    assert record["run_spec"] == run.to_dict()
    assert record["overrides"] == run.overrides
    assert record["manifest"]["sha256"] == sha256_file(manifest)
    assert record["experiment_config"]["sha256"] == sha256_file(config_path)
    assert record["checkpoints"]["sam2"]["sha256"] == sha256_file(checkpoint)
    assert record["model_configs"]["sam2"]["sha256"] == sha256_file(model_config)
    assert record["cache_identity"]["sha256"]
    assert record["environment"]["python"]["version"]
    assert record["environment"]["platform"]["system"]
    assert record["environment"]["libraries"]["pytest"]
    serialized = canonical_json(record)
    assert str(tmp_path) not in serialized
    assert "/Users/" not in serialized


def test_provenance_hashes_change_with_config_checkpoint_and_cache_identity(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.csv"
    config_path = tmp_path / "config.yaml"
    checkpoint = tmp_path / "model.pt"
    _write_manifest(manifest, ["pcb1/0001"])
    config_path.write_text("version: 1\n")
    checkpoint.write_bytes(b"one")
    run = RunSpec("dinov2_single", "pcb1", 0, 1, 4880)

    def record(cache_value: str) -> dict[str, object]:
        return build_provenance(
            run,
            manifest,
            ["pcb1/0001"],
            "abc123",
            git_dirty=False,
            config_path=config_path,
            config={"version": int(config_path.read_text().split()[-1])},
            checkpoint_paths={"sam2": checkpoint},
            cache_identity={"value": cache_value},
            library_names=(),
        )

    first = record("one")
    config_path.write_text("version: 2\n")
    second = record("one")
    checkpoint.write_bytes(b"two")
    third = record("one")
    fourth = record("two")

    assert first["experiment_config"]["sha256"] != second["experiment_config"]["sha256"]
    assert second["checkpoints"]["sam2"]["sha256"] != third["checkpoints"]["sam2"]["sha256"]
    assert third["cache_identity"]["sha256"] != fourth["cache_identity"]["sha256"]


def test_git_state_reports_commit_and_dirty_state_truthfully(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "Test"], check=True)
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("clean\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "tracked.txt"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-qm", "initial"], check=True)

    clean = git_state(tmp_path)
    tracked.write_text("dirty\n")
    dirty = git_state(tmp_path)

    assert len(clean["commit"]) == 40
    assert clean["dirty"] is False
    assert dirty["commit"] == clean["commit"]
    assert dirty["dirty"] is True
    assert len(dirty["dirty_identity"]["unstaged_diff_sha256"]) == 64


def test_atomic_json_write_is_canonical_and_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "provenance.json"
    payload = {"z": 1, "a": {"values": [3, 2, 1]}}
    atomic_write_json(path, payload)

    assert json.loads(path.read_text()) == payload
    assert path.read_text() == canonical_json(payload) + "\n"
    assert list(path.parent.glob(".*.tmp")) == []


def test_canonical_json_rejects_nonfinite_and_nondeterministic_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        canonical_json({"bad": float("nan")})
    with pytest.raises(TypeError, match="JSON"):
        canonical_json({"bad": {1, 2}})


def test_provenance_rejects_duplicates_unsafe_ids_and_secret_identity_keys(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, ["pcb1/a"])
    run = RunSpec("dinov2_single", "pcb1", 0, 1, 4880)

    with pytest.raises(ValueError, match="duplicate support"):
        build_provenance(run, manifest, ["pcb1/a", "pcb1/a"], "abc", git_dirty=False)
    with pytest.raises(ValueError, match="support ID"):
        build_provenance(run, manifest, ["../private"], "abc", git_dirty=False)
    with pytest.raises(ValueError, match="sensitive"):
        build_provenance(
            run,
            manifest,
            ["pcb1/a"],
            "abc",
            git_dirty=False,
            cache_identity={"api_token": "do-not-store"},
        )


def test_build_provenance_requires_truthful_git_dirty_state(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, ["pcb1/a"])
    run = RunSpec("dinov2_single", "pcb1", 0, 1, 4880)
    with pytest.raises(ValueError, match="git_dirty"):
        build_provenance(run, manifest, ["pcb1/a"], "abc")


def test_support_count_must_match_run_pairing_dimension(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("sample_id\n")
    run = RunSpec("dinov2_single", "pcb1", 0, 2, 4880)

    with pytest.raises(ValueError, match="k=2.*1 unique support"):
        build_provenance(run, manifest, ["pcb1/a"], "abc", git_dirty=False)


def _write_manifest(path: Path, sample_ids: list[str]) -> None:
    fields = ["sample_id", "category", "fold_id", "fold_split", "label"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sample_id in sample_ids:
            writer.writerow(
                {
                    "sample_id": sample_id,
                    "category": "pcb1",
                    "fold_id": "0",
                    "fold_split": "dev",
                    "label": "0",
                }
            )
