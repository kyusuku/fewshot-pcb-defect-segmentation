from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

from experiments.runner import (
    _validate_calibration_payload,
    _normalize_output_identity,
    build_commands,
    execution_identity,
    execute_run,
    expected_artifacts,
    method_dependencies,
    output_references,
    select_support_ids,
    topological_order,
)
from experiments.spec import RunDependency, RunSpec, expand_matrix, load_experiment_config


CONFIG_ROOT = Path("configs/experiments")


def _smoke_runs() -> list[RunSpec]:
    return expand_matrix(load_experiment_config(CONFIG_ROOT / "arxiv_smoke.yaml"))


def test_task8_dependencies_drive_deterministic_topological_order() -> None:
    runs = list(reversed(_smoke_runs()))
    ordered = topological_order(runs)
    positions = {run.run_id: index for index, run in enumerate(ordered)}
    for run in ordered:
        for dependency in run.dependencies:
            if dependency.run_id in positions:
                assert positions[dependency.run_id] < positions[run.run_id]
    assert [run.run_id for run in ordered] == [run.run_id for run in topological_order(runs)]


def test_cycle_error_names_cycle() -> None:
    first_base = RunSpec("dinov2_single", "pcb1", 0, 1, 4880)
    second_base = RunSpec("dinov2_multi", "pcb1", 0, 1, 4880)
    first = RunSpec(
        first_base.method,
        first_base.category,
        first_base.fold_id,
        first_base.k,
        first_base.seed,
        dependencies=(RunDependency(second_base.run_id, (), (), second_base.identity_sha256),),
    )
    second = RunSpec(
        second_base.method,
        second_base.category,
        second_base.fold_id,
        second_base.k,
        second_base.seed,
        dependencies=(RunDependency(first.run_id, (), (), first.identity_sha256),),
    )
    with pytest.raises(ValueError, match="cycle.*dinov2"):
        topological_order([first, second])


def test_expected_heatmap_artifacts_cover_calibrated_evaluation(tmp_path: Path) -> None:
    run = RunSpec("dinov2_single", "pcb1", 0, 1, 4880)
    paths = expected_artifacts(run, tmp_path)
    assert paths == [
        tmp_path / run.run_id / "val" / "scores.csv",
        tmp_path / run.run_id / "calibration.json",
        tmp_path / run.run_id / "test" / "scores.csv",
        tmp_path / run.run_id / "test" / "metrics.json",
        tmp_path / run.run_id / "test" / "per_image.csv",
        tmp_path / run.run_id / "provenance.json",
        tmp_path / run.run_id / "status.json",
    ]


def test_heatmap_commands_are_full_split_explicit_and_share_cache(tmp_path: Path) -> None:
    config = load_experiment_config(CONFIG_ROOT / "arxiv_smoke.yaml")
    run = next(run for run in _smoke_runs() if run.method == "dinov2_multi")
    cache = tmp_path / "cache"
    commands = build_commands(run, config, tmp_path / "out", tmp_path / "deps", "cpu", cache)
    assert len(commands) == 4
    assert [Path(command[1]).name for command in commands] == [
        "run_dinov2_baseline.py",
        "calibrate_heatmaps.py",
        "run_dinov2_baseline.py",
        "evaluate_heatmaps.py",
    ]
    assert commands[0][commands[0].index("--query-fold-split") + 1] == "val"
    assert commands[2][commands[2].index("--query-fold-split") + 1] == "test"
    for command in (commands[0], commands[2]):
        assert "--all" in command
        assert command[command.index("--feature-cache-dir") + 1] == str(cache)
        assert command[command.index("--feature-backbone") + 1] == "color_patch"
        assert command[command.index("--crop-sizes") + 1] == "32"
        assert command[command.index("--crop-overlap") + 1] == "0.25"
        assert command[command.index("--fusion") + 1] == "max"
    assert commands[3][commands[3].index("--calibration-json") + 1].endswith(
        "calibration.json"
    )


def test_patchcore_command_emits_every_frozen_method_parameter(tmp_path: Path) -> None:
    config = load_experiment_config(CONFIG_ROOT / "arxiv_primary.yaml")
    run = next(
        run
        for run in expand_matrix(config)
        if run.method == "patchcore" and run.category == "pcb1"
        and run.k == 1 and run.seed == 4880
    )
    command = build_commands(
        run, config, tmp_path / "out", tmp_path / "deps", "cpu", tmp_path / "cache"
    )[0]
    expected = {
        "--patchcore-weights": "IMAGENET1K_V2",
        "--patchcore-source": "torchvision",
        "--patchcore-layers": "layer2,layer3",
        "--feature-normalization": "imagenet",
        "--coreset-seed-policy": "run_seed",
    }
    for flag, value in expected.items():
        assert command[command.index(flag) + 1] == value


def test_guided_and_fusion_commands_use_real_dependency_columns_without_rerun(
    tmp_path: Path,
) -> None:
    config = load_experiment_config(CONFIG_ROOT / "arxiv_primary.yaml")
    runs = expand_matrix(config)
    guided = next(
        run
        for run in runs
        if run.method == "dinov2_multi_sam2" and run.category == "pcb1"
        and run.k == 1 and run.seed == 4880
    )
    fusion = next(
        run
        for run in runs
        if run.method == "anomaly_consistent_sam2" and run.category == "pcb1"
        and run.k == 1 and run.seed == 4880
    )
    guided_commands = build_commands(
        guided, config, tmp_path / "out", tmp_path / "deps", "cuda", tmp_path / "cache"
    )
    fusion_commands = build_commands(
        fusion, config, tmp_path / "out", tmp_path / "deps", "cuda", tmp_path / "cache"
    )
    assert Path(guided_commands[0][1]).name == "run_mask_refinement.py"
    assert guided_commands[0][guided_commands[0].index("--mask-output") + 1] == "sam2"
    assert "--sam2-checkpoint" in guided_commands[0]
    assert [Path(command[1]).name for command in fusion_commands] == [
        "fuse_saved_masks.py",
        "evaluate_masks.py",
    ]
    assert "--mask-scores-csv" in fusion_commands[0]
    assert "--scores-csv" not in fusion_commands[0]


def test_smoke_fusion_is_explicit_online_fallback_only(tmp_path: Path) -> None:
    config = load_experiment_config(CONFIG_ROOT / "arxiv_smoke.yaml")
    run = next(run for run in _smoke_runs() if run.method == "anomaly_consistent_sam2")
    commands = build_commands(
        run, config, tmp_path / "out", tmp_path / "deps", "cpu", tmp_path / "cache"
    )
    assert Path(commands[0][1]).name == "run_mask_refinement.py"
    assert commands[0][commands[0].index("--refiner") + 1] == "fallback"
    assert commands[0][commands[0].index("--mask-output") + 1] == "intersection"


def test_support_ids_match_baseline_algorithm_and_pair_across_methods(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    fields = ["dataset", "sample_id", "category", "fold_id", "fold_split", "label"]
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in range(5):
            writer.writerow(
                {
                    "dataset": "visa_pcb",
                    "sample_id": f"pcb1/normal_{index}",
                    "category": "pcb1",
                    "fold_id": "0",
                    "fold_split": "dev",
                    "label": "0",
                }
            )
    single = RunSpec("dinov2_single", "pcb1", 0, 2, 4880)
    multi = RunSpec("dinov2_multi", "pcb1", 0, 2, 4880)
    assert select_support_ids(single, manifest) == select_support_ids(multi, manifest)
    assert len(select_support_ids(single, manifest)) == 2
    assert select_support_ids(RunSpec("sam2_only", "pcb1", 0, 0, 0), manifest) == []


def test_method_dependencies_reflect_task8_contract() -> None:
    run = next(run for run in _smoke_runs() if run.method == "anomaly_consistent_sam2")
    assert method_dependencies(run) == [dependency.run_id for dependency in run.dependencies]


def test_mask_output_contract_only_claims_writer_owned_files() -> None:
    guided = next(run for run in _smoke_runs() if run.method == "dinov2_multi_sam2")
    fusion = next(run for run in _smoke_runs() if run.method == "anomaly_consistent_sam2")
    assert [reference.path_column for reference in output_references(guided)] == [
        "pred_mask_path",
        "sam2_mask_path",
    ]
    assert [reference.path_column for reference in output_references(fusion)] == [
        "pred_mask_path"
    ]


def test_calibration_contract_uses_serialized_val_split() -> None:
    _validate_calibration_payload(
        {"source_split": "val", "num_images": 1, "num_pixels": 4096}
    )
    with pytest.raises(ValueError, match="validation"):
        _validate_calibration_payload(
            {"source_split": "test", "num_images": 1, "num_pixels": 4096}
        )


def test_output_identity_is_recorded_relative_to_run_root(tmp_path: Path) -> None:
    artifact = tmp_path / "test" / "heatmap.npy"
    artifact.parent.mkdir()
    artifact.write_bytes(b"heatmap")
    normalized = _normalize_output_identity(
        {"path": str(artifact), "sha256": "a" * 64}, tmp_path
    )
    assert normalized == {"path": "test/heatmap.npy", "sha256": "a" * 64}
    with pytest.raises(ValueError, match="escapes"):
        _normalize_output_identity(
            {"path": str(tmp_path.parent / "outside.npy"), "sha256": "a" * 64},
            tmp_path,
        )


def test_execution_identity_contains_exact_git_state(tmp_path: Path) -> None:
    run = RunSpec("dinov2_single", "pcb1", 0, 1, 4880)
    config = CONFIG_ROOT / "arxiv_smoke.yaml"
    identity = execution_identity(
        run,
        [["python", "script.py"]],
        config,
        tmp_path / "deps",
        "cpu",
        ["pcb1/example"],
        tmp_path / "cache",
        {},
    )
    assert len(identity["git"]["commit"]) == 40
    assert isinstance(identity["git"]["dirty"], bool)
    assert identity["manifest"]["sha256"] == "unavailable" or len(
        identity["manifest"]["sha256"]
    ) == 64
    assert len(identity["config_canonical_sha256"]) == 64
    assert identity["device_identity"]["kind"] == "cpu"
    assert identity["cache_identity"]["schema"] == "feature-cache-v2"
    assert identity["evidence_class"] == "smoke_debug_only"


def test_failed_subprocess_writes_atomic_failed_status_without_shell_injection(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.csv"
    fields = ["dataset", "sample_id", "category", "fold_id", "fold_split", "label"]
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "dataset": "visa_pcb",
                "sample_id": "pcb1/support",
                "category": "pcb1",
                "fold_id": "0",
                "fold_split": "dev",
                "label": "0",
            }
        )
    config = {"name": "debug"}
    config_path = tmp_path / "config.yaml"
    config_path.write_text("name: debug\n")
    output = tmp_path / "outputs"
    marker = tmp_path / "injected"
    run = RunSpec("dinov2_single", "pcb1", 0, 1, 4880)
    command = [
        sys.executable,
        "-c",
        "import sys; assert sys.argv[1].startswith('$(touch'); sys.exit(7)",
        f"$(touch {marker})",
    ]
    with pytest.raises(Exception) as caught:
        execute_run(
            run,
            config,
            config_path,
            manifest,
            output,
            [command],
            ["pcb1/support"],
            {"commands": [command], "selected_device": "cpu", "feature_cache": "cache"},
            allow_dirty=True,
        )
    status_path = output / run.run_id / "status.json"
    assert status_path.is_file(), str(caught.value)
    status = json.loads(status_path.read_text())
    assert status["state"] == "failed"
    assert status["command_index"] == 0
    assert status["returncode"] == 7
    assert not marker.exists()
