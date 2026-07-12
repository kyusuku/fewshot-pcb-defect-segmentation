"""Dependency-aware execution and validation for frozen experiment matrices."""

from __future__ import annotations

import csv
import importlib.metadata
import importlib.util
import json
import math
import os
import random
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Sequence

from experiments.provenance import (
    atomic_write_json,
    build_provenance,
    canonical_json,
    compute_effective_execution_sha256,
    git_state,
    resolve_referenced_artifacts,
    sha256_file,
    sha256_json,
    validate_resume_identity,
    validate_support_manifest,
)
from experiments.spec import ReferencedArtifact, RunSpec, load_experiment_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]
HEATMAP_METHODS = {"patchcore", "dinov2_single", "dinov2_multi"}
GUIDED_METHODS = {"dinov2_single_sam2", "dinov2_multi_sam2"}
MASK_METHODS = {*GUIDED_METHODS, "sam2_only", "anomaly_consistent_sam2"}


def method_dependencies(run: RunSpec) -> list[str]:
    """Return the Task 8 dependency IDs without reconstructing lossy specs."""

    return [dependency.run_id for dependency in run.dependencies]


def expected_artifacts(run: RunSpec, root: str | Path) -> list[Path]:
    directory = Path(root) / run.run_id
    return _expected_artifacts_in(run, directory)


def _expected_artifacts_in(run: RunSpec, directory: Path) -> list[Path]:
    common = [directory / "provenance.json", directory / "status.json"]
    if run.method in HEATMAP_METHODS:
        return [
            directory / "val" / "scores.csv",
            directory / "val" / "memory_bank_provenance.json",
            directory / "calibration.json",
            directory / "test" / "scores.csv",
            directory / "test" / "memory_bank_provenance.json",
            directory / "test" / "metrics.json",
            directory / "test" / "per_image.csv",
            *common,
        ]
    return [
        directory / "test" / "mask_scores.csv",
        directory / "test" / "mask_metrics.json",
        directory / "test" / "mask_per_image.csv",
        *common,
    ]


def output_references(run: RunSpec) -> tuple[ReferencedArtifact, ...]:
    if run.method in HEATMAP_METHODS:
        return (
            ReferencedArtifact("val/scores.csv", "heatmap_path"),
            ReferencedArtifact("test/scores.csv", "heatmap_path"),
        )
    if run.method == "sam2_only":
        return (
            ReferencedArtifact("test/mask_scores.csv", "pred_mask_path"),
            ReferencedArtifact("test/mask_scores.csv", "heatmap_path"),
        )
    references = [ReferencedArtifact("test/mask_scores.csv", "pred_mask_path")]
    if run.method in GUIDED_METHODS:
        references.append(ReferencedArtifact("test/mask_scores.csv", "sam2_mask_path"))
    return tuple(references)


def topological_order(runs: Sequence[RunSpec]) -> list[RunSpec]:
    """Return a stable dependency-first order, rejecting duplicate IDs and cycles."""

    by_id: dict[str, RunSpec] = {}
    for run in runs:
        if run.run_id in by_id:
            raise ValueError(f"duplicate run ID: {run.run_id}")
        by_id[run.run_id] = run
    state: dict[str, int] = {}
    stack: list[str] = []
    ordered: list[RunSpec] = []

    def visit(run_id: str) -> None:
        marker = state.get(run_id, 0)
        if marker == 2:
            return
        if marker == 1:
            start = stack.index(run_id)
            cycle = stack[start:] + [run_id]
            raise ValueError(f"dependency cycle: {' -> '.join(cycle)}")
        state[run_id] = 1
        stack.append(run_id)
        run = by_id[run_id]
        for dependency in sorted(run.dependencies, key=lambda value: value.run_id):
            if dependency.run_id in by_id:
                visit(dependency.run_id)
        stack.pop()
        state[run_id] = 2
        ordered.append(run)

    for run_id in sorted(by_id):
        visit(run_id)
    return ordered


def select_support_ids(run: RunSpec, manifest_path: str | Path) -> list[str]:
    """Select exactly the baseline runner's sorted-shuffle-truncate support set."""

    if run.method == "sam2_only":
        return validate_support_manifest(manifest_path, run, [])
    with Path(manifest_path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        _require_unique_header(reader, "manifest")
        rows = list(reader)
    support = sorted(
        (
            row
            for row in rows
            if row.get("dataset") == "visa_pcb"
            and row.get("category") == run.category
            and row.get("fold_id") == str(run.fold_id)
            and row.get("fold_split") == "dev"
            and row.get("label") == "0"
        ),
        key=lambda row: row["sample_id"],
    )
    random.Random(run.seed).shuffle(support)
    selected = [row["sample_id"] for row in support[: run.k]]
    return validate_support_manifest(manifest_path, run, selected)


def build_commands(
    run: RunSpec,
    config: Mapping[str, object],
    output_root: str | Path,
    dependency_root: str | Path,
    device: str,
    feature_cache_dir: str | Path,
    *,
    internal_run_ids: set[str] | None = None,
    python_executable: str = sys.executable,
) -> list[list[str]]:
    """Expand one run into exact shell-free argv lists."""

    config = effective_method_config(config)
    output_root = Path(output_root)
    dependency_root = Path(dependency_root)
    run_dir = output_root / run.run_id
    manifest = _project_path(config["manifest"])
    if run.method in HEATMAP_METHODS:
        common = _baseline_common(run, manifest, python_executable)
        return _heatmap_commands(
            run,
            config,
            common,
            run_dir,
            device,
            Path(feature_cache_dir),
            python_executable,
        )
    if run.method == "sam2_only":
        common = _baseline_common(run, manifest, python_executable)
        return _sam2_only_commands(
            run, config, common, run_dir, device, python_executable
        )
    dependency_dirs = {
        dependency.run_id: (
            output_root / dependency.run_id
            if internal_run_ids and dependency.run_id in internal_run_ids
            else dependency_root / dependency.run_id
        )
        for dependency in run.dependencies
    }
    if run.method in GUIDED_METHODS:
        return _guided_commands(
            run, config, run_dir, dependency_dirs, device, python_executable
        )
    if run.method == "anomaly_consistent_sam2":
        return _fusion_commands(
            run, config, run_dir, dependency_dirs, device, python_executable
        )
    raise ValueError(f"unsupported method: {run.method}")


def _heatmap_commands(
    run: RunSpec,
    config: Mapping[str, object],
    common: list[str],
    run_dir: Path,
    device: str,
    cache_dir: Path,
    python: str,
) -> list[list[str]]:
    dinov2 = config["dinov2"]
    multi = dict(config["multi_scale"])
    multi.update(run.overrides)
    if run.method == "patchcore":
        patchcore = config["patchcore"]
        backbone = "patchcore_wrn50"
        image_size = patchcore["image_size"]
        patch_size = patchcore["patch_size"]
        coreset_ratio = patchcore["coreset_ratio"]
        projection_dim = patchcore["projection_dim"]
        crop_sizes = ""
        patchcore_explicit = [
            "--patchcore-weights",
            str(patchcore["weights"]),
            "--patchcore-source",
            str(patchcore["source"]),
            "--patchcore-layers",
            ",".join(str(value) for value in patchcore["layers"]),
            "--feature-normalization",
            str(patchcore["normalization"]),
            "--coreset-seed-policy",
            str(patchcore["seed_policy"]),
        ]
    else:
        backbone = dinov2["backbone"]
        image_size = dinov2["image_size"]
        patch_size = dinov2["patch_size"]
        coreset_ratio = config["patchcore"]["coreset_ratio"]
        projection_dim = config["patchcore"]["projection_dim"]
        crop_sizes = "" if run.method == "dinov2_single" else ",".join(
            str(value) for value in multi["crop_sizes"]
        )
        patchcore_explicit = []
    explicit = [
        "--k",
        str(run.k),
        "--seed",
        str(run.seed),
        "--feature-backbone",
        str(backbone),
        "--image-size",
        str(image_size),
        "--patch-size",
        str(patch_size),
        "--device",
        device,
        "--coreset-ratio",
        str(coreset_ratio),
        "--coreset-projection-dim",
        str(projection_dim),
        "--crop-sizes",
        crop_sizes,
        "--crop-overlap",
        str(multi["crop_overlap"]),
        "--fusion",
        str(multi["fusion"]),
        "--feature-cache-dir",
        str(cache_dir),
        "--debug-limit",
        "0",
        "--all",
        *patchcore_explicit,
    ]
    val_scores = run_dir / "val" / "scores.csv"
    calibration = run_dir / "calibration.json"
    test_scores = run_dir / "test" / "scores.csv"
    return [
        [*common, *explicit, "--query-fold-split", "val", "--output-dir", str(run_dir / "val")],
        [
            python,
            str(PROJECT_ROOT / "scripts" / "calibrate_heatmaps.py"),
            "--scores-csv",
            str(val_scores),
            "--quantile",
            str(config["calibration"]["quantile"]),
            "--output-json",
            str(calibration),
        ],
        [*common, *explicit, "--query-fold-split", "test", "--output-dir", str(run_dir / "test")],
        [
            python,
            str(PROJECT_ROOT / "scripts" / "evaluate_heatmaps.py"),
            "--scores-csv",
            str(test_scores),
            "--output-json",
            str(run_dir / "test" / "metrics.json"),
            "--max-pixels",
            "1000000",
            "--seed",
            str(run.seed),
            "--calibration-json",
            str(calibration),
            "--per-image-csv",
            str(run_dir / "test" / "per_image.csv"),
        ],
    ]


def _sam2_only_commands(
    run: RunSpec,
    config: Mapping[str, object],
    common: list[str],
    run_dir: Path,
    device: str,
    python: str,
) -> list[list[str]]:
    values = config["sam2_only"]
    command = [
        *common,
        "--query-fold-split",
        str(values["query_fold_split"]),
        "--prompt-longest-side",
        str(values["prompt_longest_side"]),
        "--grid-size",
        str(values["grid_size"]),
        "--max-regions",
        str(values["max_regions"]),
        "--box-scale",
        str(values["box_scale"]),
        "--refiner",
        str(values["refiner"]),
        "--max-mask-area-fraction",
        _optional_number(values["max_mask_area_fraction"]),
        "--device",
        device,
        "--output-dir",
        str(run_dir / "test"),
    ]
    if values["limit"] is None:
        command.append("--all")
    else:
        command.extend(["--limit", str(values["limit"])])
    if values["refiner"] == "sam2":
        command.extend(
            [
                "--sam2-checkpoint",
                str(_project_path(values["checkpoint"])),
                "--sam2-model-config",
                str(values["model_config"]),
            ]
        )
    else:
        command.extend(
            ["--fallback-threshold-fraction", str(values["fallback_threshold_fraction"])]
        )
    return [command, _mask_evaluation_command(run_dir, python, None)]


def _guided_commands(
    run: RunSpec,
    config: Mapping[str, object],
    run_dir: Path,
    dependency_dirs: Mapping[str, Path],
    device: str,
    python: str,
) -> list[list[str]]:
    dependency = dependency_dirs[run.dependencies[0].run_id]
    sam2 = dict(config["sam2"])
    sam2.update(run.overrides)
    command = _refinement_command(
        scores_csv=dependency / "test" / "scores.csv",
        calibration_json=dependency / "calibration.json",
        output_dir=run_dir / "test",
        sam2=sam2,
        device=device,
        mask_output="sam2",
        python=python,
    )
    return [command, _mask_evaluation_command(run_dir, python, dependency / "test" / "scores.csv")]


def _fusion_commands(
    run: RunSpec,
    config: Mapping[str, object],
    run_dir: Path,
    dependency_dirs: Mapping[str, Path],
    device: str,
    python: str,
) -> list[list[str]]:
    heatmap = dependency_dirs[run.dependencies[0].run_id]
    mask_output = str(run.overrides.get("mask_output", "intersection"))
    minimum_iou = run.overrides.get("min_iou", 0.25)
    max_expansion = run.overrides.get("max_expansion", 2.0)
    masks = dependency_dirs[run.dependencies[1].run_id]
    command = [
        python,
        str(PROJECT_ROOT / "scripts" / "fuse_saved_masks.py"),
        "--mask-scores-csv",
        str(masks / "test" / "mask_scores.csv"),
        "--calibration-json",
        str(heatmap / "calibration.json"),
        "--mask-output",
        mask_output,
        "--output-dir",
        str(run_dir / "test"),
        "--selective-min-iou",
        str(minimum_iou),
        "--selective-max-expansion",
        str(max_expansion),
    ]
    if config["name"] == "arxiv_smoke":
        command.append("--smoke-debug-fallback")
    return [command, _mask_evaluation_command(run_dir, python, heatmap / "test" / "scores.csv")]


def _refinement_command(
    *,
    scores_csv: Path,
    calibration_json: Path,
    output_dir: Path,
    sam2: Mapping[str, object],
    device: str,
    mask_output: str,
    python: str,
    minimum_iou: object = 0.25,
    max_expansion: object = 2.0,
) -> list[str]:
    command = [
        python,
        str(PROJECT_ROOT / "scripts" / "run_mask_refinement.py"),
        "--scores-csv",
        str(scores_csv),
        "--output-dir",
        str(output_dir),
        "--calibration-json",
        str(calibration_json),
        "--percentile",
        "95.0",
        "--min-area",
        "8",
        "--max-regions",
        "8",
        "--refiner",
        str(sam2["refiner"]),
        "--mask-output",
        mask_output,
        "--point-mode",
        str(sam2["point_mode"]),
        "--prompt-mode",
        str(sam2["prompt_mode"]),
        "--selective-min-iou",
        str(minimum_iou),
        "--selective-max-expansion",
        str(max_expansion),
        "--fallback-threshold-fraction",
        "0.5",
        "--max-mask-area-fraction",
        _optional_number(sam2["max_mask_area_fraction"]),
        "--device",
        device,
    ]
    if sam2["refiner"] == "sam2":
        command.extend(
            [
                "--sam2-checkpoint",
                str(_project_path(sam2["checkpoint"])),
                "--sam2-model-config",
                str(sam2["model_config"]),
            ]
        )
    return command


def _mask_evaluation_command(
    run_dir: Path, python: str, source_scores: Path | None
) -> list[str]:
    command = [
        python,
        str(PROJECT_ROOT / "scripts" / "evaluate_masks.py"),
        "--mask-scores-csv",
        str(run_dir / "test" / "mask_scores.csv"),
        "--output-json",
        str(run_dir / "test" / "mask_metrics.json"),
        "--output-csv",
        str(run_dir / "test" / "mask_per_image.csv"),
    ]
    if source_scores is not None:
        command.extend(["--source-scores-csv", str(source_scores)])
    return command


def run_matrix(
    runs: Sequence[RunSpec],
    config: Mapping[str, object],
    config_path: str | Path,
    output_root: str | Path,
    dependency_root: str | Path,
    device: str,
    feature_cache_dir: str | Path,
    *,
    dry_run: bool = False,
    resume: bool = False,
    allow_dirty: bool = False,
) -> None:
    ordered = topological_order(runs)
    internal_ids = {run.run_id for run in ordered}
    manifest = _project_path(config["manifest"])
    selected_device = resolve_device(device)
    for run in ordered:
        commands = build_commands(
            run,
            config,
            output_root,
            dependency_root,
            selected_device,
            feature_cache_dir,
            internal_run_ids=internal_ids,
        )
        dependency_locations = _dependency_locations(
            run, Path(output_root), Path(dependency_root), internal_ids
        )
        if dry_run:
            print(f"RUN {run.run_id}")
            portable = portable_command_identity(
                commands,
                run,
                output_root=output_root,
                dependency_root=dependency_root,
                cache_root=feature_cache_dir,
            )
            for command in portable:
                print("ARGV " + canonical_json(command))
            continue
        portable = portable_command_identity(
            commands,
            run,
            output_root=output_root,
            dependency_root=dependency_root,
            cache_root=feature_cache_dir,
        )
        preflight_identity = {
            "run_spec_sha256": run.identity_sha256,
            "commands": portable,
            "config_sha256": sha256_file(config_path),
            "selected_device": selected_device,
        }
        lock = acquire_run_lock(output_root, run, preflight_identity)
        entered_execute = False
        try:
            support_ids = select_support_ids(run, manifest)
            dependency_identities = {
                dependency.run_id: validate_dependency(
                    dependency,
                    dependency_locations[dependency.run_id],
                )
                for dependency in run.dependencies
            }
            execution = execution_identity(
                run,
                commands,
                config_path,
                output_root,
                dependency_root,
                selected_device,
                support_ids,
                feature_cache_dir,
                dependency_identities,
            )
            if resume and is_complete(run, output_root, execution):
                print(f"SKIP complete {run.run_id}")
                continue
            entered_execute = True
            execute_run(
                run,
                config,
                config_path,
                manifest,
                output_root,
                commands,
                support_ids,
                execution,
                allow_dirty=allow_dirty,
            )
        except BaseException as exc:
            if not entered_execute:
                _record_matrix_preflight_failure(
                    run, output_root, preflight_identity, exc
                )
            raise
        finally:
            release_run_lock(lock)


def execute_run(
    run: RunSpec,
    config: Mapping[str, object],
    config_path: str | Path,
    manifest_path: str | Path,
    output_root: str | Path,
    commands: Sequence[Sequence[str]],
    support_ids: Sequence[str],
    execution: Mapping[str, object],
    *,
    allow_dirty: bool,
) -> None:
    output_root = Path(output_root)
    run_dir = output_root / run.run_id
    if run_dir.is_symlink():
        raise ValueError(f"run directory must not be a symlink: {run.run_id}")
    staging = output_root / f".{run.run_id}.staging-{sha256_json(execution)[:12]}"
    if staging.is_symlink():
        raise ValueError(f"staging directory must not be a symlink: {run.run_id}")
    if staging.exists():
        shutil.rmtree(staging)
    staged_commands = [
        [value.replace(str(run_dir), str(staging)) for value in command]
        for command in commands
    ]
    provenance_path = staging / "provenance.json"
    status_path = staging / "status.json"
    try:
        checkpoints, model_configs = _model_files(run, effective_method_config(config))
        pre = build_provenance(
            run,
            manifest_path,
            support_ids,
            repo_root=PROJECT_ROOT,
            config_path=config_path,
            config=config,
            checkpoint_paths=checkpoints,
            model_config_paths=model_configs,
            cache_identity={"feature_cache": execution["feature_cache"]},
            artifact_identities={"phase": "pre_execution"},
            allow_dirty=allow_dirty,
        )
    except BaseException as exc:
        _write_preflight_failure(run, staging, execution, exc)
        _promote_failed_attempt(staging, run_dir, output_root, execution)
        raise
    pre["execution"] = dict(execution)
    pre["effective_execution_sha256"] = compute_effective_execution_sha256(pre)
    atomic_write_json(provenance_path, pre)
    atomic_write_json(
        status_path,
        {"state": "running", "requested_execution_sha256": sha256_json(execution)},
    )
    command_index = -1
    try:
        environment = os.environ.copy()
        existing = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = str(PROJECT_ROOT / "src") + (
            os.pathsep + existing if existing else ""
        )
        for command_index, command in enumerate(staged_commands):
            subprocess.run(
                list(command),
                check=True,
                cwd=PROJECT_ROOT,
                env=environment,
            )
        _validate_data_artifacts_in(run, staging, manifest_path=manifest_path)
        _validate_evidence_class(run, staging, execution)
        artifacts = _collect_artifact_identities_in(run, staging)
        final = build_provenance(
            run,
            manifest_path,
            support_ids,
            repo_root=PROJECT_ROOT,
            config_path=config_path,
            config=config,
            checkpoint_paths=checkpoints,
            model_config_paths=model_configs,
            cache_identity={"feature_cache": execution["feature_cache"]},
            artifact_identities={"outputs": artifacts, "phase": "complete"},
            allow_dirty=allow_dirty,
        )
        final["execution"] = dict(execution)
        final["observed_identities"] = _collect_observed_identities_in(run, staging)
        final["effective_execution_sha256"] = compute_effective_execution_sha256(final)
        atomic_write_json(provenance_path, final)
        atomic_write_json(
            status_path,
            {
                "state": "complete",
                "effective_execution_sha256": final["effective_execution_sha256"],
                "requested_execution_sha256": sha256_json(execution),
            },
        )
        _promote_complete_attempt(staging, run_dir, output_root)
        validate_completed_run(run, output_root, execution)
        print(f"COMPLETE {run.run_id}")
    except BaseException as exc:
        returncode = exc.returncode if isinstance(exc, subprocess.CalledProcessError) else None
        atomic_write_json(
            status_path,
            {
                "state": "failed",
                "command_index": command_index,
                "returncode": returncode,
                "error": f"{type(exc).__name__}: execution failed",
            },
        )
        _promote_failed_attempt(staging, run_dir, output_root, execution)
        raise


def execution_identity(
    run: RunSpec,
    commands: Sequence[Sequence[str]],
    config_path: str | Path,
    output_root: str | Path,
    dependency_root: str | Path,
    selected_device: str,
    support_ids: Sequence[str],
    feature_cache_dir: str | Path,
    dependency_identities: Mapping[str, str],
) -> dict[str, object]:
    loaded_config = load_experiment_config(config_path)
    method_config = effective_method_config(loaded_config)
    manifest_path = _project_path(loaded_config["manifest"])
    checkpoints, model_configs = _model_files(run, method_config)
    portable_commands = portable_command_identity(
        commands,
        run,
        output_root=output_root,
        dependency_root=dependency_root,
        cache_root=feature_cache_dir,
    )
    return {
        "commands": portable_commands,
        "config_path": _portable_path(Path(config_path), {}),
        "config_sha256": sha256_file(config_path),
        "config_canonical_sha256": sha256_json(loaded_config),
        "base_config": _base_config_identity(loaded_config),
        "manifest": {
            "path": str(loaded_config["manifest"]),
            "sha256": sha256_file(manifest_path) if manifest_path.is_file() else "unavailable",
        },
        "dependency_root": "$DEPENDENCY_ROOT",
        "dependency_effective_identities": dict(sorted(dependency_identities.items())),
        "selected_device": selected_device,
        "support_ids": list(support_ids),
        "feature_cache": "$CACHE_ROOT",
        "cache_identity": {
            "schema": "feature-cache-v2",
            "root": "$CACHE_ROOT",
        },
        "runtime": runtime_identity(),
        "source_revisions": source_revisions(),
        "git": git_state(PROJECT_ROOT),
        "device_identity": device_identity(selected_device),
        "checkpoints": _execution_file_identities(checkpoints),
        "model_configs": _execution_file_identities(model_configs),
        "run_spec_sha256": run.identity_sha256,
        "evidence_class": (
            "smoke_debug_only"
            if loaded_config["name"] == "arxiv_smoke"
            else "paper_evidence"
        ),
    }


def runtime_identity() -> dict[str, object]:
    versions: dict[str, str] = {}
    for name in ("numpy", "Pillow", "PyYAML", "torch", "torchvision"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "unavailable"
    return {
        "python_executable": Path(sys.executable).name,
        "python_version": sys.version.split()[0],
        "libraries": versions,
    }


def source_revisions() -> dict[str, str]:
    return {
        "repository": _git_revision(PROJECT_ROOT),
        "dinov2_torch_hub": _torch_hub_revision("facebookresearch_dinov2"),
        "sam2": _candidate_git_revision((PROJECT_ROOT / "external" / "sam2",)),
        "torchvision": _package_revision("torchvision"),
    }


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def device_identity(selected: str) -> dict[str, str]:
    identity = {"kind": selected, "name": selected}
    if selected == "cuda":
        try:
            import torch

            identity["name"] = torch.cuda.get_device_name(0)
        except (ImportError, RuntimeError, AssertionError):
            identity["name"] = "unavailable"
    elif selected == "mps":
        identity["name"] = "Apple Metal Performance Shaders"
    return identity


def validate_dependency(dependency, run_dir: Path, _visited: set[str] | None = None) -> str:
    if run_dir.is_symlink():
        raise ValueError(f"dependency {dependency.run_id} run directory is a symlink")
    if (run_dir / "status.json").is_symlink() or (run_dir / "provenance.json").is_symlink():
        raise ValueError(f"dependency {dependency.run_id} metadata must not be symlinked")
    visited = set() if _visited is None else _visited
    if dependency.run_id in visited:
        raise ValueError(f"dependency closure cycle at {dependency.run_id}")
    visited.add(dependency.run_id)
    status = _read_json(run_dir / "status.json", "dependency status")
    provenance = _read_json(run_dir / "provenance.json", "dependency provenance")
    if status.get("state") != "complete":
        raise ValueError(f"dependency {dependency.run_id} is not complete")
    if provenance.get("run_id") != dependency.run_id:
        raise ValueError(f"dependency {dependency.run_id} provenance run ID mismatch")
    if provenance.get("run_spec_sha256") != dependency.expected_run_spec_sha256:
        raise ValueError(f"dependency {dependency.run_id} run-spec identity is stale")
    effective = provenance.get("effective_execution_sha256")
    validate_resume_identity(provenance, expected_effective_execution_sha256=effective)
    if status.get("effective_execution_sha256") != effective:
        raise ValueError(f"dependency {dependency.run_id} status identity is stale")
    root = run_dir.resolve()
    for relative in dependency.artifacts:
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"dependency {dependency.run_id} artifact invalid: {relative}")
    for reference in dependency.referenced_artifacts:
        resolve_referenced_artifacts(run_dir, reference)
    _validate_recorded_outputs(provenance, run_dir)
    recorded_spec = RunSpec.from_dict(provenance["run_spec"])
    if provenance.get("observed_identities") != _collect_observed_identities_in(
        recorded_spec, run_dir
    ):
        raise ValueError(f"dependency {dependency.run_id} observed identity is stale")
    current_ancestors = {
        ancestor.run_id: validate_dependency(
            ancestor, run_dir.parent / ancestor.run_id, visited
        )
        for ancestor in recorded_spec.dependencies
    }
    execution = provenance.get("execution")
    if not isinstance(execution, Mapping):
        raise ValueError(f"dependency {dependency.run_id} execution identity is missing")
    recorded_ancestors = execution.get("dependency_effective_identities")
    if not isinstance(recorded_ancestors, Mapping):
        raise ValueError(f"dependency {dependency.run_id} closure identity is missing")
    _validate_recorded_dependency_closure(recorded_ancestors, current_ancestors)
    visited.remove(dependency.run_id)
    return effective


def _validate_recorded_dependency_closure(
    recorded: Mapping[str, object], current: Mapping[str, object]
) -> None:
    if dict(recorded) != dict(current):
        raise ValueError("dependency ancestor effective identity is stale")


def validate_data_artifacts(
    run: RunSpec, root: str | Path, *, manifest_path: str | Path | None = None
) -> None:
    run_dir = Path(root) / run.run_id
    if run_dir.is_symlink():
        raise ValueError(f"run {run.run_id} directory must not be a symlink")
    if (run_dir / "status.json").is_symlink() or (run_dir / "provenance.json").is_symlink():
        raise ValueError(f"run {run.run_id} metadata must not be symlinked")
    _validate_data_artifacts_in(run, run_dir, manifest_path=manifest_path)


def _validate_data_artifacts_in(
    run: RunSpec, run_dir: Path, *, manifest_path: str | Path | None = None
) -> None:
    run_dir = run_dir.resolve()
    data_paths = _expected_artifacts_in(run, run_dir)[:-2]
    for path in data_paths:
        resolved = path.resolve()
        if not resolved.is_relative_to(run_dir):
            raise ValueError(f"artifact escapes run root: {path.name}")
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"missing or empty artifact: {path.relative_to(run_dir)}")
    expected = _expected_manifest_ids(run, manifest_path) if manifest_path is not None else {}
    if run.method in HEATMAP_METHODS:
        val_rows = _read_validated_csv(
            run_dir / "val" / "scores.csv", {"sample_id", "heatmap_path"}
        )
        test_rows = _read_validated_csv(
            run_dir / "test" / "scores.csv", {"sample_id", "heatmap_path"}
        )
        per_image = _read_validated_csv(
            run_dir / "test" / "per_image.csv", {"sample_id", "mask_f1"}
        )
        if expected:
            _validate_sample_rows(val_rows, expected["val"], run.category, "val", "scores")
            _validate_sample_rows(test_rows, expected["test"], run.category, "test", "scores")
            _validate_sample_rows(
                per_image, expected["test"], run.category, None, "per_image"
            )
        calibration = _read_json(run_dir / "calibration.json", "calibration")
        _validate_calibration_payload(calibration)
    else:
        mask_rows = _read_validated_csv(
            run_dir / "test" / "mask_scores.csv", {"sample_id", "pred_mask_path"}
        )
        per_mask = _read_validated_csv(
            run_dir / "test" / "mask_per_image.csv", {"sample_id"}
        )
        if expected:
            _validate_sample_rows(mask_rows, expected["test"], run.category, None, "masks")
            _validate_sample_rows(per_mask, expected["test"], run.category, None, "per_mask")
    metrics_name = "metrics.json" if run.method in HEATMAP_METHODS else "mask_metrics.json"
    _validate_finite_metrics(_read_json(run_dir / "test" / metrics_name, "metrics"))
    for reference in output_references(run):
        resolve_referenced_artifacts(run_dir, reference)


def collect_artifact_identities(run: RunSpec, root: str | Path) -> list[dict[str, object]]:
    run_dir = Path(root) / run.run_id
    return _collect_artifact_identities_in(run, run_dir)


def _collect_artifact_identities_in(
    run: RunSpec, run_dir: Path
) -> list[dict[str, object]]:
    paths = _expected_artifacts_in(run, run_dir)[:-2]
    identities = [
        {"path": path.relative_to(run_dir).as_posix(), "sha256": sha256_file(path)}
        for path in paths
    ]
    for reference in output_references(run):
        identities.extend(
            _normalize_output_identity(identity, run_dir)
            for identity in resolve_referenced_artifacts(run_dir, reference)
        )
    unique = {(item["path"], item["sha256"]): item for item in identities}
    return [unique[key] for key in sorted(unique)]


def _normalize_output_identity(
    identity: Mapping[str, object], run_dir: str | Path
) -> dict[str, str]:
    root = Path(run_dir).resolve()
    path = Path(str(identity.get("path", "")))
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("recorded output identity escapes run root")
    checksum = identity.get("sha256")
    if not isinstance(checksum, str):
        raise ValueError("recorded output identity is missing SHA-256")
    return {"path": resolved.relative_to(root).as_posix(), "sha256": checksum}


def collect_observed_identities(run: RunSpec, root: str | Path) -> dict[str, object]:
    run_dir = Path(root) / run.run_id
    return _collect_observed_identities_in(run, run_dir)


def _collect_observed_identities_in(
    run: RunSpec, run_dir: Path
) -> dict[str, object]:
    observed: dict[str, object] = {"source_revisions": source_revisions()}
    if run.method in HEATMAP_METHODS:
        memory_records = []
        for split in ("val", "test"):
            path = run_dir / split / "memory_bank_provenance.json"
            if path.is_file():
                memory_records.append({"split": split, "identity": _read_json(path, "memory")})
        _validate_observed_consistency(memory_records)
        observed["extractors"] = memory_records
    return observed


def _validate_observed_consistency(records: Sequence[Mapping[str, object]]) -> None:
    if len(records) != 2:
        raise ValueError("heatmap runs require val and test observed extractor identities")
    val_identity = records[0].get("identity")
    test_identity = records[1].get("identity")
    if not isinstance(val_identity, Mapping) or not isinstance(test_identity, Mapping):
        raise ValueError("observed extractor identities must be mappings")
    fields = ("feature_backbone", "feature_cache_enabled", "extractor_source_revision")
    for field in fields:
        if val_identity.get(field) != test_identity.get(field):
            raise ValueError(f"val/test observed identity mismatch: {field}")
    for field in ("extractor_identity",):
        if canonical_json(val_identity.get(field)) != canonical_json(test_identity.get(field)):
            raise ValueError(f"val/test observed identity mismatch: {field}")


def is_complete(run: RunSpec, root: str | Path, execution: Mapping[str, object]) -> bool:
    try:
        validate_completed_run(run, root, execution)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False
    return True


def validate_completed_run(
    run: RunSpec, root: str | Path, execution: Mapping[str, object]
) -> None:
    run_dir = Path(root) / run.run_id
    status = _read_json(run_dir / "status.json", "status")
    provenance = _read_json(run_dir / "provenance.json", "provenance")
    if status.get("state") != "complete":
        raise ValueError(f"run {run.run_id} status is not complete")
    if provenance.get("run_spec_sha256") != run.identity_sha256:
        raise ValueError(f"run {run.run_id} run-spec identity is stale")
    if provenance.get("execution") != dict(execution):
        raise ValueError(f"run {run.run_id} requested execution identity is stale")
    effective = provenance.get("effective_execution_sha256")
    validate_resume_identity(provenance, expected_effective_execution_sha256=effective)
    if status.get("effective_execution_sha256") != effective:
        raise ValueError(f"run {run.run_id} status effective identity is stale")
    if status.get("requested_execution_sha256") != sha256_json(execution):
        raise ValueError(f"run {run.run_id} requested identity is stale")
    manifest_identity = execution.get("manifest")
    manifest_path = None
    if isinstance(manifest_identity, Mapping):
        manifest_path = _project_path(manifest_identity.get("path"))
    validate_data_artifacts(run, root, manifest_path=manifest_path)
    _validate_evidence_class(run, run_dir, execution)
    _validate_recorded_outputs(provenance, run_dir)
    observed = _collect_observed_identities_in(run, run_dir)
    if provenance.get("observed_identities") != observed:
        raise ValueError("observed extractor identity is stale")


def matrix_report(
    runs: Sequence[RunSpec],
    output_root: str | Path,
    dependency_root: str | Path,
    config: Mapping[str, object],
    config_path: str | Path,
    device: str,
    feature_cache_dir: str | Path,
) -> dict[str, object]:
    ordered = topological_order(runs)
    internal_ids = {run.run_id for run in ordered}
    manifest = _project_path(config["manifest"])
    selected_device = resolve_device(device)
    reports = []
    for run in ordered:
        errors: list[dict[str, str]] = []
        try:
            support = select_support_ids(run, manifest)
            locations = _dependency_locations(
                run, Path(output_root), Path(dependency_root), internal_ids
            )
            dependency_ids = {
                dependency.run_id: validate_dependency(dependency, locations[dependency.run_id])
                for dependency in run.dependencies
            }
            commands = build_commands(
                run,
                config,
                output_root,
                dependency_root,
                selected_device,
                feature_cache_dir,
                internal_run_ids=internal_ids,
            )
            execution = execution_identity(
                run,
                commands,
                config_path,
                output_root,
                dependency_root,
                selected_device,
                support,
                feature_cache_dir,
                dependency_ids,
            )
            validate_completed_run(run, output_root, execution)
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            errors.append(_checker_error(exc))
        reports.append({"run_id": run.run_id, "ok": not errors, "errors": errors})
    return {"ok": all(report["ok"] for report in reports), "runs": reports}


def _checker_error(exc: BaseException) -> dict[str, str]:
    message = str(exc).lower()
    if (
        isinstance(exc, (FileNotFoundError, OSError))
        or "missing" in message
        or "no such" in message
    ):
        category = "missing"
    elif "failed" in message or "not complete" in message:
        category = "failed"
    elif "duplicate" in message:
        category = "duplicate"
    elif "dependency" in message or "ancestor" in message:
        category = "dependency"
    elif "stale" in message or "identity" in message or "checksum" in message:
        category = "stale"
    elif "manifest row set" in message or "metric" in message or "evidence class" in message:
        category = "semantic"
    else:
        category = "artifact"
    return {"category": category, "message": f"{category} validation failure"}


def _validate_recorded_outputs(provenance: Mapping[str, object], run_dir: Path) -> None:
    identity = provenance.get("artifact_identities")
    if not isinstance(identity, Mapping):
        raise ValueError("provenance artifact identities are missing")
    canonical = identity.get("canonical")
    if not isinstance(canonical, Mapping) or not isinstance(canonical.get("outputs"), list):
        raise ValueError("provenance output identities are missing")
    root = run_dir.resolve()
    for item in canonical["outputs"]:
        if not isinstance(item, Mapping) or set(item) != {"path", "sha256"}:
            raise ValueError("malformed recorded output identity")
        path = (root / str(item["path"])).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("recorded output is missing or escapes run root")
        if sha256_file(path) != item["sha256"]:
            raise ValueError(f"recorded output checksum mismatch: {item['path']}")


def _validate_csv(path: Path, required: set[str]) -> None:
    _read_validated_csv(path, required)


def _read_validated_csv(path: Path, required: set[str]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        headers = _require_unique_header(reader, path.name)
        missing = required - set(headers)
        if missing:
            raise ValueError(f"{path.name} missing columns: {sorted(missing)}")
        rows = list(reader)
        if not rows:
            raise ValueError(f"{path.name} must contain at least one row")
        return rows


def _expected_manifest_ids(
    run: RunSpec, manifest_path: str | Path
) -> dict[str, set[str]]:
    with Path(manifest_path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        _require_unique_header(reader, "manifest")
        rows = list(reader)
    return {
        split: {
            row["sample_id"]
            for row in rows
            if row.get("dataset") == "visa_pcb"
            and row.get("category") == run.category
            and row.get("fold_id") == str(run.fold_id)
            and row.get("fold_split") == split
        }
        for split in ("val", "test")
    }


def _validate_sample_rows(
    rows: Sequence[Mapping[str, str]],
    expected_ids: set[str],
    category: str,
    split: str | None,
    context: str,
) -> None:
    sample_ids = [row.get("sample_id", "") for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"{context} contains duplicate sample IDs")
    if set(sample_ids) != expected_ids:
        raise ValueError(f"{context} does not match exact manifest row set")
    for row in rows:
        row_category = row.get("category")
        if row_category not in (None, "", category):
            raise ValueError(f"{context} category does not match {category}")
        if split is not None and row.get("fold_split") != split:
            raise ValueError(f"{context} split does not match {split}")


def _validate_finite_metrics(payload: Mapping[str, object]) -> None:
    if not payload:
        raise ValueError("metrics must be a nonempty mapping")
    for key, value in payload.items():
        if key == "calibration_source_split" and value == "val":
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"metric {key!r} must be numeric")
        if not math.isfinite(float(value)):
            raise ValueError(f"metric {key!r} must be finite")


def _validate_evidence_class(
    run: RunSpec, run_dir: Path, execution: Mapping[str, object]
) -> None:
    if run.method not in {*GUIDED_METHODS, "anomaly_consistent_sam2"}:
        return
    rows = _read_validated_csv(
        run_dir / "test" / "mask_scores.csv",
        {"sample_id", "evidence_class", "refiner", "raw_mask_source"},
    )
    expected_class = execution.get("evidence_class")
    for row in rows:
        if row.get("evidence_class") != expected_class:
            raise ValueError("mask evidence class does not match experiment class")
        expected_source = "fallback" if expected_class == "smoke_debug_only" else "sam2"
        if row.get("refiner") != expected_source or row.get("raw_mask_source") != expected_source:
            raise ValueError("mask source cannot be accepted for this evidence class")


def _validate_calibration_payload(payload: Mapping[str, object]) -> None:
    if payload.get("source_split") != "val":
        raise ValueError("calibration source must be the validation split")
    for field in ("num_images", "num_pixels"):
        value = payload.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"calibration {field} must be a positive integer")


def _require_unique_header(reader: csv.DictReader, context: str) -> list[str]:
    headers = reader.fieldnames
    if not headers:
        raise ValueError(f"{context} must have a CSV header")
    if len(headers) != len(set(headers)):
        raise ValueError(f"{context} has duplicate CSV columns")
    return headers


def _read_json(path: Path, context: str) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"{context} must be a nonempty JSON object")
    return payload


def _dependency_locations(
    run: RunSpec, output_root: Path, dependency_root: Path, internal_ids: set[str]
) -> dict[str, Path]:
    return {
        dependency.run_id: (
            output_root / dependency.run_id
            if dependency.run_id in internal_ids
            else dependency_root / dependency.run_id
        )
        for dependency in run.dependencies
    }


def _baseline_script(method: str) -> str:
    if method == "patchcore":
        return "run_patchcore_baseline.py"
    if method in HEATMAP_METHODS:
        return "run_dinov2_baseline.py"
    if method == "sam2_only":
        return "run_sam2_baseline.py"
    raise ValueError(f"method has no baseline script: {method}")


def _baseline_common(run: RunSpec, manifest: Path, python: str) -> list[str]:
    return [
        python,
        str(PROJECT_ROOT / "scripts" / _baseline_script(run.method)),
        "--manifest",
        str(manifest),
        "--fold-id",
        str(run.fold_id),
        "--category",
        run.category,
    ]


def _project_path(value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else PROJECT_ROOT / path


def portable_command_identity(
    commands: Sequence[Sequence[str]],
    run: RunSpec,
    *,
    output_root: str | Path,
    dependency_root: str | Path,
    cache_root: str | Path,
) -> list[list[str]]:
    output = Path(output_root)
    dependency = Path(dependency_root)
    cache = Path(cache_root)
    replacements = {
        output / run.run_id: "$RUN_ROOT",
        cache: "$CACHE_ROOT",
        dependency: "$DEPENDENCY_ROOT",
        output: "$OUTPUT_ROOT",
        PROJECT_ROOT: "$PROJECT_ROOT",
    }
    portable: list[list[str]] = []
    for command in commands:
        values = []
        for index, value in enumerate(command):
            values.append(
                "$PYTHON_EXECUTABLE"
                if index == 0
                else _portable_path(Path(value), replacements, original=value)
            )
        portable.append(values)
    return portable


def _portable_path(
    path: Path,
    replacements: Mapping[Path, str],
    *,
    original: str | None = None,
) -> str:
    value = str(path) if original is None else original
    for root, placeholder in sorted(
        replacements.items(), key=lambda item: len(str(item[0])), reverse=True
    ):
        root_value = str(root)
        if value == root_value:
            return placeholder
        prefix = root_value.rstrip("/") + "/"
        if value.startswith(prefix):
            return placeholder + "/" + value[len(prefix) :]
        if path.is_absolute() and root.is_absolute():
            try:
                relative = path.resolve().relative_to(root.resolve())
            except ValueError:
                continue
            return placeholder + "/" + relative.as_posix()
    if path.is_absolute():
        return "$EXTERNAL_PATH/" + path.name
    return value


def effective_method_config(config: Mapping[str, object]) -> dict[str, object]:
    if config.get("name") != "arxiv_ablations":
        return dict(config)
    base_path = _project_path(config["base_config"])
    base = load_experiment_config(base_path)
    merged = dict(base)
    merged.update(
        {
            "name": config["name"],
            "manifest": config["manifest"],
            "dependency_output_root": config["dependency_output_root"],
        }
    )
    return merged


def _base_config_identity(config: Mapping[str, object]) -> dict[str, object]:
    if config.get("name") != "arxiv_ablations":
        return {}
    path = _project_path(config["base_config"])
    base = load_experiment_config(path)
    return {
        "path": str(config["base_config"]),
        "file_sha256": sha256_file(path),
        "canonical_sha256": sha256_json(base),
        "declared_canonical_sha256": config["base_config_canonical_sha256"],
    }


def _optional_number(value: object) -> str:
    return "none" if value is None else str(value)


def _model_files(
    run: RunSpec, config: Mapping[str, object]
) -> tuple[dict[str, Path], dict[str, Path]]:
    if run.method == "sam2_only":
        section = config["sam2_only"]
    elif run.method in GUIDED_METHODS or (
        run.method == "anomaly_consistent_sam2" and config["name"] == "arxiv_smoke"
    ):
        section = config["sam2"]
    else:
        return {}, {}
    if section["refiner"] != "sam2":
        return {}, {}
    return (
        {"sam2": _project_path(section["checkpoint"])},
        {
            "sam2": Path(
                resolve_sam2_model_config(section["model_config"], required=True)[
                    "resolved_path"
                ]
            )
        },
    )


def resolve_sam2_model_config(identifier: object, *, required: bool) -> dict[str, str]:
    value = str(identifier)
    result = {"path": value, "resolved_path": "unavailable", "sha256": "unavailable"}
    try:
        spec = importlib.util.find_spec("sam2")
    except (ImportError, ValueError):
        spec = None
    roots = list(spec.submodule_search_locations or ()) if spec is not None else []
    for root in roots:
        candidate = Path(root) / value
        if candidate.is_file():
            result.update(resolved_path=str(candidate.resolve()), sha256=sha256_file(candidate))
            return result
    if required:
        raise ValueError(
            f"installed sam2 package does not provide Hydra config identifier {value!r}"
        )
    return result


def _execution_file_identities(paths: Mapping[str, Path]) -> dict[str, object]:
    return {
        name: {
            "path": path.name,
            "sha256": sha256_file(path) if path.is_file() else "unavailable",
        }
        for name, path in sorted(paths.items())
    }


def _git_revision(path: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        revision = result.stdout.strip()
        return revision if len(revision) == 40 else "unavailable"
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _candidate_git_revision(paths: Sequence[Path]) -> str:
    for path in paths:
        if path.exists():
            revision = _git_revision(path)
            if revision != "unavailable":
                return revision
    return "unavailable"


def _torch_hub_revision(prefix: str) -> str:
    try:
        import torch

        hub = Path(torch.hub.get_dir())
    except (ImportError, OSError):
        return "unavailable"
    candidates = sorted(path for path in hub.glob(f"{prefix}*") if path.is_dir())
    return _candidate_git_revision(candidates)


def _package_revision(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _sanitize_error(message: str, run_dir: Path) -> str:
    sanitized = message.replace(str(PROJECT_ROOT), "<project>")
    sanitized = sanitized.replace(str(run_dir), "<run>")
    return sanitized[:2000]


def _write_preflight_failure(
    run: RunSpec,
    run_dir: Path,
    execution: Mapping[str, object],
    exc: BaseException,
) -> None:
    requested = sha256_json(execution)
    atomic_write_json(
        run_dir / "provenance.json",
        {
            "schema_version": 1,
            "run_id": run.run_id,
            "phase": "preflight_failed",
            "requested_execution_sha256": requested,
            "error_type": type(exc).__name__,
        },
    )
    atomic_write_json(
        run_dir / "status.json",
        {
            "state": "failed",
            "phase": "preflight",
            "command_index": -1,
            "returncode": None,
            "requested_execution_sha256": requested,
            "error": f"{type(exc).__name__}: preflight validation failed",
        },
    )


def _record_matrix_preflight_failure(
    run: RunSpec,
    output_root: str | Path,
    preflight_identity: Mapping[str, object],
    exc: BaseException,
) -> None:
    output = Path(output_root)
    run_dir = output / run.run_id
    if run_dir.is_symlink():
        raise ValueError(f"run directory must not be a symlink: {run.run_id}") from exc
    staging = output / f".{run.run_id}.preflight-{sha256_json(preflight_identity)[:12]}"
    if staging.exists():
        shutil.rmtree(staging)
    _write_preflight_failure(run, staging, preflight_identity, exc)
    _promote_failed_attempt(staging, run_dir, output, preflight_identity)


def acquire_run_lock(
    output_root: str | Path, run: RunSpec, execution: Mapping[str, object]
) -> Path:
    lock_dir = Path(output_root) / ".locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    if lock_dir.is_symlink():
        raise ValueError("runner lock directory must not be a symlink")
    lock_path = lock_dir / f"{run.run_id}.lock"
    payload = {
        "run_id": run.run_id,
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "requested_execution_sha256": sha256_json(execution),
    }
    for _ in range(2):
        try:
            descriptor = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            owner = _read_json(lock_path, "run lock")
            if owner.get("host") == socket.gethostname() and _pid_is_live(owner.get("pid")):
                raise RuntimeError(f"run {run.run_id} has a live conflicting owner")
            lock_path.unlink()
            continue
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(canonical_json(payload) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return lock_path
    raise RuntimeError(f"unable to acquire lock for {run.run_id}")


def _promote_complete_attempt(staging: Path, run_dir: Path, output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    if run_dir.exists():
        if run_dir.is_symlink() or not _runner_owned_directory(run_dir):
            raise ValueError(f"refusing to replace non-runner-owned directory: {run_dir.name}")
        archive_root = output_root / ".archive"
        archive_root.mkdir(parents=True, exist_ok=True)
        status = _read_json(run_dir / "status.json", "archived status")
        identity = str(
            status.get("effective_execution_sha256")
            or status.get("requested_execution_sha256")
            or "unknown"
        )[:12]
        archive = archive_root / f"{run_dir.name}-{identity}"
        suffix = 1
        while archive.exists():
            archive = archive_root / f"{run_dir.name}-{identity}-{suffix}"
            suffix += 1
        os.replace(run_dir, archive)
    os.replace(staging, run_dir)


def _promote_failed_attempt(
    staging: Path,
    run_dir: Path,
    output_root: Path,
    execution: Mapping[str, object],
) -> None:
    if not staging.exists():
        return
    output_root.mkdir(parents=True, exist_ok=True)
    if not run_dir.exists():
        os.replace(staging, run_dir)
        return
    failures = output_root / ".failures"
    failures.mkdir(parents=True, exist_ok=True)
    target = failures / f"{run_dir.name}-{sha256_json(execution)[:12]}"
    if target.exists():
        shutil.rmtree(target)
    os.replace(staging, target)


def _runner_owned_directory(path: Path) -> bool:
    try:
        provenance = _read_json(path / "provenance.json", "runner provenance")
        status = _read_json(path / "status.json", "runner status")
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return provenance.get("run_id") == path.name and status.get("state") in {
        "running",
        "complete",
        "failed",
    }


def release_run_lock(lock_path: str | Path) -> None:
    Path(lock_path).unlink(missing_ok=True)


def _pid_is_live(value: object) -> bool:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        return False
    try:
        os.kill(value, 0)
    except (OSError, OverflowError):
        return False
    return True
