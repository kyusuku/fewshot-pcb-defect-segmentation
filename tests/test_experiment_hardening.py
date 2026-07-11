from __future__ import annotations

import copy
import csv
import json
import subprocess
from pathlib import Path

import pytest
import yaml

from experiments.provenance import (
    build_provenance,
    compute_effective_execution_sha256,
    sha256_file,
    validate_resume_identity,
    validate_support_manifest,
)
from experiments.spec import (
    ReferencedArtifact,
    RunDependency,
    RunSpec,
    expand_matrix,
    load_experiment_config,
)


CONFIG_ROOT = Path("configs/experiments")


def test_programmatic_overrides_are_method_specific_and_recursively_safe() -> None:
    run = RunSpec(
        "dinov2_multi",
        "pcb1",
        0,
        1,
        4880,
        overrides={"crop_sizes": [512], "crop_overlap": 0.25, "fusion": "max"},
    )
    assert RunSpec.from_dict(run.to_dict()) == run

    invalid = [
        ("dinov2_single", {"crop_sizes": [512]}, "does not allow overrides"),
        ("dinov2_multi", {"crop_sizes": "512"}, "crop_sizes"),
        ("dinov2_multi_sam2", {"max_mask_area_fraction": 2.0}, "fraction"),
        ("anomaly_consistent_sam2", {"mask_output": "unknown"}, "mask_output"),
        ("dinov2_multi", {"api_token": "secret"}, "sensitive"),
        ("dinov2_multi", {"fusion": {"password": "secret"}}, "sensitive"),
        ("dinov2_multi", {"fusion": "/Users/private/value"}, "absolute"),
        ("dinov2_multi", {"fusion": "C:/private/value"}, "absolute"),
        ("dinov2_multi", {"fusion": "../private"}, "traversal"),
        ("dinov2_multi", {"fusion": "méan"}, "ASCII"),
        ("dinov2_multi", {"fusion": {"mode": "max"}}, "fusion"),
    ]
    for method, overrides, message in invalid:
        with pytest.raises(ValueError, match=message):
            RunSpec(method, "pcb1", 0, 1, 4880, overrides=overrides)


@pytest.mark.parametrize(
    "unsafe_path",
    [
        ".",
        "..",
        "../private.csv",
        "/private/data.csv",
        "C:/private/data.csv",
        "C:\\private\\data.csv",
        "//server/share/data.csv",
        "data/é.csv",
        "data/control\x00.csv",
    ],
)
def test_public_config_paths_reject_ambiguous_or_private_forms(
    tmp_path: Path, unsafe_path: str
) -> None:
    config = load_experiment_config(CONFIG_ROOT / "arxiv_primary.yaml")
    config["manifest"] = unsafe_path
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    with pytest.raises(ValueError, match="path|ASCII|traversal|relative"):
        load_experiment_config(path)


def test_duplicate_yaml_keys_are_rejected_before_schema_validation(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.yaml"
    path.write_text("name: arxiv_primary\nname: arxiv_smoke\n")
    with pytest.raises(ValueError, match="duplicate YAML key.*name"):
        load_experiment_config(path)


def test_nested_choice_fields_reject_wrong_types_as_validation_errors(tmp_path: Path) -> None:
    config = load_experiment_config(CONFIG_ROOT / "arxiv_primary.yaml")
    config["sam2_only"]["device"] = []
    path = tmp_path / "wrong-type.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    with pytest.raises(ValueError, match="sam2_only.device"):
        load_experiment_config(path)


def test_frozen_configs_fully_specify_patchcore_and_sam2_only() -> None:
    primary = load_experiment_config(CONFIG_ROOT / "arxiv_primary.yaml")
    smoke = load_experiment_config(CONFIG_ROOT / "arxiv_smoke.yaml")

    assert primary["patchcore"] == {
        "backbone": "wide_resnet50_2",
        "weights": "IMAGENET1K_V2",
        "source": "torchvision",
        "layers": ["layer2", "layer3"],
        "image_size": 512,
        "patch_size": 8,
        "coreset_ratio": 0.01,
        "projection_dim": 64,
        "normalization": "imagenet",
        "seed_policy": "run_seed",
    }
    assert smoke["patchcore"] == primary["patchcore"]
    assert primary["sam2_only"] == {
        "refiner": "sam2",
        "query_policy": "full_test",
        "query_fold_split": "test",
        "limit": None,
        "prompt_longest_side": 1024,
        "grid_size": 3,
        "max_regions": 9,
        "box_scale": 1.0,
        "max_mask_area_fraction": 0.25,
        "checkpoint": "weights/sam2.1_hiera_tiny.pt",
        "model_config": "configs/sam2.1/sam2.1_hiera_t.yaml",
        "device": "auto",
    }
    assert smoke["sam2_only"]["refiner"] == "fallback"
    assert smoke["sam2_only"]["query_policy"] == "bounded_fixture"
    assert smoke["sam2_only"]["limit"] == 2


def test_dependencies_declare_referenced_artifacts_and_expected_identity() -> None:
    primary = expand_matrix(load_experiment_config(CONFIG_ROOT / "arxiv_primary.yaml"))
    guided = next(
        run
        for run in primary
        if run.method == "dinov2_multi_sam2"
        and run.category == "pcb1"
        and run.k == 1
        and run.seed == 4880
    )
    heatmap_dependency = guided.dependencies[0]
    heatmap_reference = heatmap_dependency.referenced_artifacts[0]

    upstream = next(run for run in primary if run.run_id == heatmap_dependency.run_id)
    assert heatmap_dependency.expected_run_spec_sha256 == upstream.identity_sha256
    assert heatmap_reference == ReferencedArtifact(
        csv_path="test/scores.csv",
        path_column="heatmap_path",
        required_per_row=True,
        path_scope="csv_parent",
        checksum="sha256_file",
    )

    ablations = expand_matrix(load_experiment_config(CONFIG_ROOT / "arxiv_ablations.yaml"))
    fusion = next(
        run
        for run in ablations
        if run.variant == "fusion_union" and run.category == "pcb1"
    )
    raw_reference = fusion.dependencies[1].referenced_artifacts[0]
    assert raw_reference.csv_path == "test/mask_scores.csv"
    assert raw_reference.path_column == "sam2_mask_path"
    assert raw_reference.checksum == "sha256_file"


def test_dependency_sequences_and_artifacts_are_strict() -> None:
    with pytest.raises(ValueError, match="sequence.*not a string"):
        RunDependency("dinov2_multi__pcb1__fold0__k1__seed4880", artifacts="scores.csv")
    with pytest.raises(ValueError, match="normalized"):
        RunDependency(
            "dinov2_multi__pcb1__fold0__k1__seed4880",
            artifacts=("test/./scores.csv",),
        )
    with pytest.raises(ValueError, match="sequence.*not a string"):
        RunDependency(
            "dinov2_multi__pcb1__fold0__k1__seed4880",
            referenced_artifacts="test/scores.csv",
        )
    with pytest.raises(ValueError, match="SHA-256"):
        RunDependency(
            "dinov2_multi__pcb1__fold0__k1__seed4880",
            expected_run_spec_sha256="abc",
        )


def test_dirty_provenance_is_rejected_by_default_and_hashed_when_allowed(
    tmp_path: Path,
) -> None:
    root = _init_repo(tmp_path)
    manifest = _write_manifest(root / "manifest.csv", [_support_row("pcb1/a")])
    config_path = root / "config.yaml"
    config = {"name": "debug", "value": 1}
    config_path.write_text(yaml.safe_dump(config, sort_keys=True))
    staged = root / "staged.txt"
    unstaged = root / "unstaged.txt"
    staged.write_text("base\n")
    unstaged.write_text("base\n")
    _commit_all(root)
    staged.write_text("staged change\n")
    subprocess.run(["git", "-C", str(root), "add", "staged.txt"], check=True)
    unstaged.write_text("unstaged change\n")
    untracked = root / "notes.txt"
    untracked.write_text("untracked evidence\n")
    run = RunSpec("dinov2_single", "pcb1", 0, 1, 4880)

    with pytest.raises(ValueError, match="dirty repository"):
        build_provenance(
            run,
            manifest,
            ["pcb1/a"],
            repo_root=root,
            config_path=config_path,
            config=config,
            library_names=(),
        )

    record = build_provenance(
        run,
        manifest,
        ["pcb1/a"],
        repo_root=root,
        config_path=config_path,
        config=config,
        library_names=(),
        allow_dirty=True,
    )
    dirty = record["git"]["dirty_identity"]
    assert len(dirty["staged_diff_sha256"]) == 64
    assert len(dirty["unstaged_diff_sha256"]) == 64
    assert dirty["untracked"] == [
        {"path": "notes.txt", "sha256": sha256_file(untracked)}
    ]
    assert "untracked evidence" not in json.dumps(record)


def test_dirty_provenance_rejects_sensitive_untracked_paths(tmp_path: Path) -> None:
    root = _init_repo(tmp_path)
    manifest = _write_manifest(root / "manifest.csv", [_support_row("pcb1/a")])
    config_path = root / "config.yaml"
    config_path.write_text("name: debug\n")
    _commit_all(root)
    (root / "api_token.txt").write_text("never-record-this\n")
    with pytest.raises(ValueError, match="sensitive untracked"):
        build_provenance(
            RunSpec("dinov2_single", "pcb1", 0, 1, 4880),
            manifest,
            ["pcb1/a"],
            repo_root=root,
            config_path=config_path,
            config={"name": "debug"},
            library_names=(),
            allow_dirty=True,
        )


def test_manifest_support_rows_are_verified_exactly() -> None:
    with pytest.raises(ValueError, match="category"):
        _validate_rows([_support_row("pcb1/a", category="pcb2")], ["pcb1/a"])
    with pytest.raises(ValueError, match="fold"):
        _validate_rows([_support_row("pcb1/a", fold_id="1")], ["pcb1/a"])
    with pytest.raises(ValueError, match="normal.*label=0"):
        _validate_rows([_support_row("pcb1/a", label="1")], ["pcb1/a"])
    with pytest.raises(ValueError, match="fold_split=dev"):
        _validate_rows([_support_row("pcb1/a", fold_split="val")], ["pcb1/a"])
    with pytest.raises(ValueError, match="exactly once"):
        _validate_rows([], ["pcb1/a"])
    duplicate = [_support_row("pcb1/a"), _support_row("pcb1/a")]
    with pytest.raises(ValueError, match="exactly once"):
        _validate_rows(duplicate, ["pcb1/a"])


def test_config_file_must_equal_supplied_canonical_config(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path / "manifest.csv", [_support_row("pcb1/a")])
    config_path = tmp_path / "config.yaml"
    config_path.write_text("name: from_file\n")
    with pytest.raises(ValueError, match="does not match supplied config"):
        build_provenance(
            RunSpec("dinov2_single", "pcb1", 0, 1, 4880),
            manifest,
            ["pcb1/a"],
            "a" * 40,
            git_dirty=False,
            config_path=config_path,
            config={"name": "different"},
            library_names=(),
        )


def test_same_run_id_with_different_effective_identity_is_not_resumable(
    tmp_path: Path,
) -> None:
    manifest = _write_manifest(tmp_path / "manifest.csv", [_support_row("pcb1/a")])
    config_path = tmp_path / "config.yaml"
    config = {"name": "debug"}
    config_path.write_text(yaml.safe_dump(config))
    original = RunSpec(
        "dinov2_multi", "pcb1", 0, 1, 4880, overrides={"crop_sizes": [512]}
    )
    changed = RunSpec(
        "dinov2_multi", "pcb1", 0, 1, 4880, overrides={"crop_sizes": [768]}
    )
    record = build_provenance(
        original,
        manifest,
        ["pcb1/a"],
        "a" * 40,
        git_dirty=False,
        config_path=config_path,
        config=config,
        library_names=(),
    )
    assert original.run_id == changed.run_id
    expected = copy.deepcopy(record)
    expected["run_spec"] = changed.to_dict()
    expected["run_spec_sha256"] = changed.identity_sha256
    expected_hash = compute_effective_execution_sha256(expected)
    with pytest.raises(ValueError, match="effective execution identity"):
        validate_resume_identity(
            record,
            expected_effective_execution_sha256=expected_hash,
        )
    validate_resume_identity(
        record,
        expected_effective_execution_sha256=record["effective_execution_sha256"],
    )


def _validate_rows(rows: list[dict[str, str]], support_ids: list[str]) -> None:
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as directory:
        manifest = _write_manifest(Path(directory) / "manifest.csv", rows)
        validate_support_manifest(
            manifest,
            RunSpec("dinov2_single", "pcb1", 0, len(support_ids), 4880),
            support_ids,
        )


def _support_row(sample_id: str, **updates: str) -> dict[str, str]:
    row = {
        "dataset": "visa_pcb",
        "sample_id": sample_id,
        "category": "pcb1",
        "fold_id": "0",
        "fold_split": "dev",
        "label": "0",
    }
    row.update(updates)
    return row


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> Path:
    fields = ["dataset", "sample_id", "category", "fold_id", "fold_split", "label"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _init_repo(root: Path) -> Path:
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(
        ["git", "-C", str(root), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
    return root


def _commit_all(root: Path) -> None:
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-qm", "fixture"], check=True)
