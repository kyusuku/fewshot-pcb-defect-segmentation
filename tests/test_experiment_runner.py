from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest
import numpy as np

from experiments.runner import (
    _validate_calibration_payload,
    _validate_recorded_dependency_closure,
    _validate_sample_rows,
    _validate_finite_metrics,
    _validate_manifest_output_rows,
    _validate_calibration_contract,
    _validate_method_metrics,
    _validate_heatmap_per_image_rows,
    _validate_mask_per_rows,
    _normalize_output_identity,
    _create_owned_staging,
    _promote_complete_attempt,
    _prune_owned_failures,
    _prune_owned_archives,
    _read_ownership_marker,
    _write_final_ownership_marker,
    _write_ownership_marker,
    acquire_run_lock,
    build_commands,
    execution_identity,
    execute_run,
    expected_artifacts,
    method_dependencies,
    output_references,
    resolve_sam2_model_config,
    release_run_lock,
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
        tmp_path / run.run_id / "val" / "memory_bank_provenance.json",
        tmp_path / run.run_id / "calibration.json",
        tmp_path / run.run_id / "test" / "scores.csv",
        tmp_path / run.run_id / "test" / "memory_bank_provenance.json",
        tmp_path / run.run_id / "test" / "metrics.json",
        tmp_path / run.run_id / "test" / "per_image.csv",
        tmp_path / run.run_id / ".runner_ownership.json",
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
        assert command[command.index("--heatmap-format") + 1] == "npz_compressed"
        assert command[command.index("--debug-limit") + 1] == "0"
        assert command[command.index("--feature-backbone") + 1] == "color_patch"
        assert command[command.index("--crop-sizes") + 1] == "32"
        assert command[command.index("--crop-overlap") + 1] == "0.25"
        assert command[command.index("--fusion") + 1] == "max"
    assert commands[3][commands[3].index("--calibration-json") + 1].endswith("calibration.json")


def test_patchcore_command_emits_every_frozen_method_parameter(tmp_path: Path) -> None:
    config = load_experiment_config(CONFIG_ROOT / "arxiv_primary.yaml")
    run = next(
        run
        for run in expand_matrix(config)
        if run.method == "patchcore" and run.category == "pcb1" and run.k == 1 and run.seed == 4880
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


def test_all_48_ablations_build_exact_commands_and_external_dependencies(
    tmp_path: Path,
) -> None:
    config = load_experiment_config(CONFIG_ROOT / "arxiv_ablations.yaml")
    runs = expand_matrix(config)
    assert len(runs) == 48
    for run in runs:
        commands = build_commands(
            run,
            config,
            tmp_path / "ablations",
            tmp_path / "primary",
            "cpu",
            tmp_path / "cache",
        )
        assert commands
        flat = [value for command in commands for value in command]
        for dependency in run.dependencies:
            assert str(tmp_path / "primary" / dependency.run_id) in " ".join(flat)
        for key, value in run.overrides.items():
            flag = {
                "crop_sizes": "--crop-sizes",
                "crop_overlap": "--crop-overlap",
                "fusion": "--fusion",
                "prompt_mode": "--prompt-mode",
                "point_mode": "--point-mode",
                "max_mask_area_fraction": "--max-mask-area-fraction",
                "mask_output": "--mask-output",
                "min_iou": "--selective-min-iou",
                "max_expansion": "--selective-max-expansion",
            }[key]
            expected = (
                ",".join(str(item) for item in value)
                if isinstance(value, list)
                else ("none" if value is None else str(value))
            )
            assert any(
                flag in command and command[command.index(flag) + 1] == expected
                for command in commands
            )


def test_sam2_hydra_identifier_is_preserved_and_resolved_from_installed_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = tmp_path / "sam2"
    config = package / "configs" / "sam2.1" / "sam2.1_hiera_t.yaml"
    config.parent.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    config.write_text("model: tiny\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    import importlib
    import sys

    sys.modules.pop("sam2", None)
    importlib.invalidate_caches()
    identity = resolve_sam2_model_config("configs/sam2.1/sam2.1_hiera_t.yaml", required=True)
    assert identity["path"] == "configs/sam2.1/sam2.1_hiera_t.yaml"
    assert len(identity["sha256"]) == 64

    primary = load_experiment_config(CONFIG_ROOT / "arxiv_primary.yaml")
    run = next(run for run in expand_matrix(primary) if run.method == "sam2_only")
    command = build_commands(
        run, primary, tmp_path / "out", tmp_path / "deps", "cpu", tmp_path / "cache"
    )[0]
    assert command[command.index("--sam2-model-config") + 1] == (
        "configs/sam2.1/sam2.1_hiera_t.yaml"
    )
    assert command[command.index("--heatmap-format") + 1] == "npz_compressed"
    assert command[command.index("--debug-limit") + 1] == "0"


def test_guided_and_fusion_commands_use_real_dependency_columns_without_rerun(
    tmp_path: Path,
) -> None:
    config = load_experiment_config(CONFIG_ROOT / "arxiv_primary.yaml")
    runs = expand_matrix(config)
    guided = next(
        run
        for run in runs
        if run.method == "dinov2_multi_sam2"
        and run.category == "pcb1"
        and run.k == 1
        and run.seed == 4880
    )
    fusion = next(
        run
        for run in runs
        if run.method == "anomaly_consistent_sam2"
        and run.category == "pcb1"
        and run.k == 1
        and run.seed == 4880
    )
    guided_commands = build_commands(
        guided, config, tmp_path / "out", tmp_path / "deps", "cuda", tmp_path / "cache"
    )
    assert guided_commands[0][guided_commands[0].index("--debug-limit") + 1] == "0"
    fusion_commands = build_commands(
        fusion, config, tmp_path / "out", tmp_path / "deps", "cuda", tmp_path / "cache"
    )
    assert fusion_commands[0][fusion_commands[0].index("--debug-limit") + 1] == "0"
    assert Path(guided_commands[0][1]).name == "run_mask_refinement.py"
    assert guided_commands[0][guided_commands[0].index("--mask-output") + 1] == "sam2"
    assert "--sam2-checkpoint" in guided_commands[0]
    assert [Path(command[1]).name for command in fusion_commands] == [
        "fuse_saved_masks.py",
        "evaluate_masks.py",
    ]
    assert "--mask-scores-csv" in fusion_commands[0]
    assert "--scores-csv" not in fusion_commands[0]


def test_smoke_fusion_reuses_declared_fallback_mask_offline(tmp_path: Path) -> None:
    config = load_experiment_config(CONFIG_ROOT / "arxiv_smoke.yaml")
    run = next(run for run in _smoke_runs() if run.method == "anomaly_consistent_sam2")
    commands = build_commands(
        run, config, tmp_path / "out", tmp_path / "deps", "cpu", tmp_path / "cache"
    )
    assert Path(commands[0][1]).name == "fuse_saved_masks.py"
    assert "--smoke-debug-fallback" in commands[0]
    assert "--refiner" not in commands[0]
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
    assert [reference.path_column for reference in output_references(fusion)] == ["pred_mask_path"]


def test_calibration_contract_uses_serialized_val_split() -> None:
    _validate_calibration_payload({"source_split": "val", "num_images": 1, "num_pixels": 4096})
    with pytest.raises(ValueError, match="validation"):
        _validate_calibration_payload({"source_split": "test", "num_images": 1, "num_pixels": 4096})


def test_output_identity_is_recorded_relative_to_run_root(tmp_path: Path) -> None:
    artifact = tmp_path / "test" / "heatmap.npy"
    artifact.parent.mkdir()
    artifact.write_bytes(b"heatmap")
    normalized = _normalize_output_identity({"path": str(artifact), "sha256": "a" * 64}, tmp_path)
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
        tmp_path / "out",
        tmp_path / "deps",
        "cpu",
        ["pcb1/example"],
        tmp_path / "cache",
        {},
    )
    assert len(identity["git"]["commit"]) == 40
    assert isinstance(identity["git"]["dirty"], bool)
    assert (
        identity["manifest"]["sha256"] == "unavailable" or len(identity["manifest"]["sha256"]) == 64
    )
    assert len(identity["config_canonical_sha256"]) == 64
    assert identity["device_identity"]["kind"] == "cpu"
    assert identity["cache_identity"]["schema"] == "feature-cache-v2"
    assert identity["evidence_class"] == "smoke_debug_only"
    serialized = json.dumps(identity)
    assert str(tmp_path) not in serialized
    assert identity["commands"] == [["$PYTHON_EXECUTABLE", "script.py"]]


def test_execution_identity_canonicalizes_absolute_operational_roots(tmp_path: Path) -> None:
    config = load_experiment_config(CONFIG_ROOT / "arxiv_smoke.yaml")
    run = next(run for run in _smoke_runs() if run.method == "dinov2_multi")
    output = tmp_path / "private" / "out"
    dependency = tmp_path / "private" / "deps"
    cache = tmp_path / "private" / "cache"
    commands = build_commands(run, config, output, dependency, "cpu", cache)
    identity = execution_identity(
        run,
        commands,
        CONFIG_ROOT / "arxiv_smoke.yaml",
        output,
        dependency,
        "cpu",
        ["pcb1/support"],
        cache,
        {},
    )
    serialized = json.dumps(identity)
    assert str(tmp_path) not in serialized
    assert "$PROJECT_ROOT" in serialized
    assert "$RUN_ROOT" in serialized
    assert identity["dependency_root"] == "$DEPENDENCY_ROOT"
    assert identity["feature_cache"] == "$CACHE_ROOT"


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


def test_preflight_failure_writes_sanitized_status_and_provenance(tmp_path: Path) -> None:
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
    config_path = tmp_path / "secret-private-config.yaml"
    config_path.write_text("name: file-value\n")
    output = tmp_path / "outputs"
    run = RunSpec("dinov2_single", "pcb1", 0, 1, 4880)
    with pytest.raises(ValueError):
        execute_run(
            run,
            {"name": "different-value"},
            config_path,
            manifest,
            output,
            [[sys.executable, "-c", "raise AssertionError('must not run')"]],
            ["pcb1/support"],
            {"commands": [], "selected_device": "cpu", "feature_cache": "$CACHE_ROOT"},
            allow_dirty=True,
        )
    run_dir = output / run.run_id
    status = json.loads((run_dir / "status.json").read_text())
    provenance = json.loads((run_dir / "provenance.json").read_text())
    assert status["state"] == "failed"
    assert status["phase"] == "preflight"
    serialized = json.dumps(provenance)
    assert str(tmp_path) not in serialized
    assert "different-value" not in serialized


def test_live_run_lock_rejects_second_owner(tmp_path: Path) -> None:
    run = RunSpec("dinov2_single", "pcb1", 0, 1, 4880)
    lock = acquire_run_lock(tmp_path, run, {"identity": "first"})
    try:
        with pytest.raises(RuntimeError, match="live conflicting owner"):
            acquire_run_lock(tmp_path, run, {"identity": "second"})
    finally:
        release_run_lock(lock)


def test_foreign_host_lock_is_never_reclaimed_automatically(tmp_path: Path) -> None:
    run = RunSpec("dinov2_single", "pcb1", 0, 1, 4880)
    lock_dir = tmp_path / ".locks"
    lock_dir.mkdir()
    lock_path = lock_dir / f"{run.run_id}.lock"
    lock_path.write_text(
        json.dumps(
            {
                "run_id": run.run_id,
                "pid": 999999,
                "host": "foreign-host",
                "requested_execution_sha256": "a" * 64,
            }
        )
    )
    with pytest.raises(RuntimeError, match="foreign-host"):
        acquire_run_lock(tmp_path, run, {"identity": "current"})
    assert lock_path.is_file()


def test_staging_parent_symlink_and_planted_unknown_dirs_are_preserved(tmp_path: Path) -> None:
    run = RunSpec("dinov2_single", "pcb1", 0, 1, 4880)
    outside = tmp_path / "outside"
    outside.mkdir()
    linked_root = tmp_path / "linked-output"
    linked_root.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="staging parent.*symlink"):
        _create_owned_staging(linked_root, run, {"identity": "x"})
    assert list(outside.iterdir()) == []

    planted = tmp_path / ".planted-user-directory"
    planted.mkdir()
    (planted / "keep.txt").write_text("user data")
    staging = _create_owned_staging(tmp_path, run, {"identity": "x"})
    assert (planted / "keep.txt").read_text() == "user data"
    assert staging != planted


def test_failed_final_promotion_restores_old_final_and_preserves_staging(
    tmp_path: Path,
) -> None:
    run = RunSpec("dinov2_single", "pcb1", 0, 1, 4880)
    final = tmp_path / run.run_id
    final.mkdir()
    _write_ownership_marker(final, run, {"identity": "old"}, nonce="oldnonce")
    (final / "old.txt").write_text("old")
    staging = _create_owned_staging(tmp_path, run, {"identity": "new"})
    (staging / "new.txt").write_text("new")

    def fail_new_promotion(source: Path, destination: Path) -> None:
        raise OSError("injected promotion failure")

    with pytest.raises(OSError, match="injected"):
        _promote_complete_attempt(
            staging,
            final,
            tmp_path,
            replace_staging=fail_new_promotion,
        )
    assert (final / "old.txt").read_text() == "old"
    assert (staging / "new.txt").read_text() == "new"


def test_hidden_sibling_archives_preserve_relative_links_and_bounded_retention(
    tmp_path: Path,
) -> None:
    run = RunSpec("dinov2_multi_sam2", "pcb1", 0, 1, 4880)
    dependency_file = tmp_path / "dependency" / "test" / "heatmap.npy"
    dependency_file.parent.mkdir(parents=True)
    dependency_file.write_bytes(b"heatmap")
    final = tmp_path / run.run_id
    (final / "test").mkdir(parents=True)
    _write_ownership_marker(final, run, {"identity": "old"}, nonce="oldnonce")
    relative = Path("../../dependency/test/heatmap.npy")
    (final / "test" / "link.txt").write_text(str(relative))
    staging = _create_owned_staging(tmp_path, run, {"identity": "new"})
    _promote_complete_attempt(staging, final, tmp_path)
    archives = sorted(tmp_path.glob(f".archive-{run.run_id}-*"))
    assert len(archives) == 1
    assert (archives[0] / "test" / relative).resolve() == dependency_file.resolve()
    for index in range(5):
        archive = tmp_path / f".archive-{run.run_id}-extra{index}"
        archive.mkdir()
        _write_ownership_marker(archive, run, {"identity": str(index)}, nonce=str(index))
    unknown = tmp_path / f".archive-{run.run_id}-unknown"
    unknown.mkdir()
    (unknown / "keep.txt").write_text("keep")
    _prune_owned_archives(tmp_path, run, retain=3)
    owned = [
        path
        for path in tmp_path.glob(f".archive-{run.run_id}-*")
        if (path / ".runner_ownership.json").is_file()
    ]
    assert len(owned) == 3
    assert (unknown / "keep.txt").read_text() == "keep"


def test_final_ownership_marker_is_portable_and_sanitized(tmp_path: Path) -> None:
    run = RunSpec("dinov2_single", "pcb1", 0, 1, 4880)
    staging = _create_owned_staging(tmp_path, run, {"identity": "requested"})
    effective_identity = "e" * 64

    _write_final_ownership_marker(staging, run, effective_identity)

    marker = _read_ownership_marker(staging)
    assert marker == {
        "schema_version": 1,
        "kind": "final_run",
        "run_id": run.run_id,
        "portable_effective_identity": effective_identity,
        "nonce": staging.name.removeprefix(f".staging-{run.run_id}-"),
    }
    serialized = json.dumps(marker)
    assert "host" not in serialized
    assert "pid" not in serialized
    assert "pre_execution_sha256" not in serialized


def test_owned_failure_retention_is_bounded_and_unknown_dirs_are_preserved(
    tmp_path: Path,
) -> None:
    run = RunSpec("dinov2_single", "pcb1", 0, 1, 4880)
    for index in range(5):
        failure = tmp_path / f".failure-{run.run_id}-extra{index}"
        failure.mkdir()
        _write_ownership_marker(failure, run, {"identity": str(index)}, nonce=str(index))
    unknown = tmp_path / f".failure-{run.run_id}-unknown"
    unknown.mkdir()
    (unknown / "keep.txt").write_text("keep")

    _prune_owned_failures(tmp_path, run, retain=3)

    owned = [
        path
        for path in tmp_path.glob(f".failure-{run.run_id}-*")
        if (path / ".runner_ownership.json").is_file()
    ]
    assert len(owned) == 3
    assert (unknown / "keep.txt").read_text() == "keep"


def test_symlinked_run_directory_is_rejected_before_write(tmp_path: Path) -> None:
    output = tmp_path / "outputs"
    outside = tmp_path / "outside"
    output.mkdir()
    outside.mkdir()
    run = RunSpec("dinov2_single", "pcb1", 0, 1, 4880)
    (output / run.run_id).symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        execute_run(
            run,
            {"name": "bad"},
            tmp_path / "missing.yaml",
            tmp_path / "missing.csv",
            output,
            [],
            [],
            {"commands": [], "selected_device": "cpu", "feature_cache": "$CACHE_ROOT"},
            allow_dirty=True,
        )
    assert list(outside.iterdir()) == []


def test_dependency_closure_rejects_old_guided_parent_identity() -> None:
    recorded = {"heatmap-run": "a" * 64}
    _validate_recorded_dependency_closure(recorded, recorded)
    with pytest.raises(ValueError, match="ancestor.*stale"):
        _validate_recorded_dependency_closure(recorded, {"heatmap-run": "b" * 64})


def test_semantic_rows_require_unique_exact_manifest_sample_set() -> None:
    rows = [
        {"sample_id": "pcb1/a", "category": "pcb1", "fold_split": "test"},
        {"sample_id": "pcb1/b", "category": "pcb1", "fold_split": "test"},
    ]
    _validate_sample_rows(rows, {"pcb1/a", "pcb1/b"}, "pcb1", "test", "scores")
    with pytest.raises(ValueError, match="duplicate sample IDs"):
        _validate_sample_rows([rows[0], rows[0]], {"pcb1/a"}, "pcb1", "test", "scores")
    with pytest.raises(ValueError, match="manifest row set"):
        _validate_sample_rows(rows[:1], {"pcb1/a", "pcb1/b"}, "pcb1", "test", "scores")


def test_metrics_allow_declared_metadata_but_require_finite_numeric_values() -> None:
    _validate_finite_metrics({"image_auroc": 0.9, "calibration_source_split": "val"})
    with pytest.raises(ValueError, match="finite"):
        _validate_finite_metrics({"image_auroc": float("nan")})
    with pytest.raises(ValueError, match="numeric"):
        _validate_finite_metrics({"image_auroc": "high"})


def test_manifest_row_semantics_reject_wrong_label_and_mask_identity(tmp_path: Path) -> None:
    image = tmp_path / "image.png"
    mask = tmp_path / "mask.png"
    wrong_mask = tmp_path / "wrong-mask.png"
    image.write_bytes(b"image")
    mask.write_bytes(b"mask")
    wrong_mask.write_bytes(b"wrong")
    expected = {
        "pcb1/a": {
            "dataset": "visa_pcb",
            "sample_id": "pcb1/a",
            "category": "pcb1",
            "fold_split": "test",
            "label": "1",
            "image_path": str(image),
            "mask_path": str(mask),
        }
    }
    valid = [dict(expected["pcb1/a"])]
    _validate_manifest_output_rows(valid, expected, tmp_path, tmp_path, "scores")
    wrong_label = [dict(valid[0], label="0")]
    with pytest.raises(ValueError, match="label"):
        _validate_manifest_output_rows(wrong_label, expected, tmp_path, tmp_path, "scores")
    wrong_mask_row = [dict(valid[0], mask_path=str(wrong_mask))]
    with pytest.raises(ValueError, match="mask identity"):
        _validate_manifest_output_rows(wrong_mask_row, expected, tmp_path, tmp_path, "scores")


def test_calibration_contract_rejects_incomplete_and_pixel_count_mismatch(
    tmp_path: Path,
) -> None:
    heatmap = tmp_path / "heatmap.npy"
    np.save(heatmap, np.zeros((2, 3), dtype=np.float32))
    rows = [{"sample_id": "pcb1/n", "label": "0", "heatmap_path": heatmap.name}]
    valid = {
        "quantile": 0.995,
        "threshold": 0.0,
        "num_images": 1,
        "num_pixels": 6,
        "source_split": "val",
    }
    _validate_calibration_contract(valid, 0.995, rows, tmp_path)
    with pytest.raises((KeyError, ValueError, TypeError)):
        _validate_calibration_contract({"threshold": 0.5}, 0.995, rows, tmp_path)
    with pytest.raises(ValueError, match="pixel count"):
        _validate_calibration_contract(dict(valid, num_pixels=5), 0.995, rows, tmp_path)
    with pytest.raises(ValueError, match="exact validation quantile"):
        _validate_calibration_contract(dict(valid, threshold=123.0), 0.995, rows, tmp_path)


def test_method_metric_schemas_reject_garbage_and_count_mismatch() -> None:
    with pytest.raises(ValueError, match="missing required"):
        _validate_method_metrics(
            RunSpec("dinov2_single", "pcb1", 0, 1, 4880),
            {"image_auroc": 0.9},
            test_rows=[{"label": "1"}],
            calibration=None,
        )


def test_per_image_rows_reject_nan_confusion_and_mean_mismatch() -> None:
    expected = {
        "pcb1/a": {
            "dataset": "visa_pcb",
            "sample_id": "pcb1/a",
            "category": "pcb1",
            "fold_split": "test",
            "label": "1",
        }
    }
    row = {
        **expected["pcb1/a"],
        "threshold": "0.5",
        "mask_precision": "1.0",
        "mask_recall": "1.0",
        "mask_f1": "1.0",
        "mask_iou": "1.0",
        "pred_positive_pixels": "2",
        "gt_positive_pixels": "2",
        "true_positive_pixels": "2",
        "false_positive_pixels": "0",
        "false_negative_pixels": "0",
    }
    metrics = {
        "calibrated_mean_mask_precision": 1.0,
        "calibrated_mean_mask_recall": 1.0,
        "calibrated_mean_mask_f1": 1.0,
        "calibrated_mean_mask_iou": 1.0,
        "calibrated_mean_anomaly_mask_precision": 1.0,
        "calibrated_mean_anomaly_mask_recall": 1.0,
        "calibrated_mean_anomaly_mask_f1": 1.0,
        "calibrated_mean_anomaly_mask_iou": 1.0,
        "calibrated_aggregate_pixel_precision": 1.0,
        "calibrated_aggregate_pixel_recall": 1.0,
        "calibrated_aggregate_pixel_f1": 1.0,
        "calibrated_aggregate_pixel_iou": 1.0,
    }
    _validate_heatmap_per_image_rows([row], expected, 0.5, metrics)
    with pytest.raises(ValueError, match="finite"):
        _validate_heatmap_per_image_rows([dict(row, mask_f1="nan")], expected, 0.5, metrics)
    with pytest.raises(ValueError, match="confusion"):
        _validate_heatmap_per_image_rows(
            [dict(row, pred_positive_pixels="3")], expected, 0.5, metrics
        )
    with pytest.raises(ValueError, match="mean"):
        _validate_heatmap_per_image_rows(
            [row], expected, 0.5, dict(metrics, calibrated_mean_mask_f1=0.0)
        )


def test_mask_per_rows_reject_wrong_label_nan_count_and_mean_mismatch() -> None:
    expected = {
        "pcb1/a": {
            "dataset": "visa_pcb",
            "sample_id": "pcb1/a",
            "category": "pcb1",
            "fold_split": "test",
            "label": "1",
        }
    }
    row = {
        **expected["pcb1/a"],
        "mask_precision": "0.5",
        "mask_recall": "0.5",
        "mask_f1": "0.5",
        "mask_iou": "0.5",
        "pred_positive_pixels": "2",
        "gt_positive_pixels": "2",
    }
    metrics = {
        "mean_mask_precision": 0.5,
        "mean_mask_recall": 0.5,
        "mean_mask_f1": 0.5,
        "mean_mask_iou": 0.5,
        "mean_anomaly_mask_precision": 0.5,
        "mean_anomaly_mask_recall": 0.5,
        "mean_anomaly_mask_f1": 0.5,
        "mean_anomaly_mask_iou": 0.5,
    }
    _validate_mask_per_rows([row], expected, metrics)
    with pytest.raises(ValueError, match="label"):
        _validate_mask_per_rows([dict(row, label="0")], expected, metrics)
    with pytest.raises(ValueError, match="finite"):
        _validate_mask_per_rows([dict(row, mask_iou="inf")], expected, metrics)
    with pytest.raises(ValueError, match="integral"):
        _validate_mask_per_rows([dict(row, pred_positive_pixels="1.5")], expected, metrics)
    with pytest.raises(ValueError, match="mean"):
        _validate_mask_per_rows([row], expected, dict(metrics, mean_mask_f1=0.1))
    with pytest.raises(ValueError, match="numeric"):
        _validate_method_metrics(
            RunSpec("dinov2_multi_sam2", "pcb1", 0, 1, 4880),
            {"num_mask_images": "one"},
            test_rows=[{"label": "1"}],
            calibration=None,
        )
