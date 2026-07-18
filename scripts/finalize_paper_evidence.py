#!/usr/bin/env python
"""Create the public post-freeze evidence revision from validated compact inputs."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shlex
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Mapping, Sequence

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from evaluation.paper_tables import (
    HEATMAP_METHODS,
    MASK_METHODS,
    METRIC_COLUMNS,
    aggregate_ablation_rows,
    aggregate_all_primary_metrics,
)
from experiments.provenance import sha256_file, sha256_json
from experiments.spec import RunSpec, expand_matrix, load_experiment_config
from scripts.render_method_figure import BLOCKS


PRIMARY_SUMMARY_FIELDS = [
    "method",
    "k",
    "category",
    "metric",
    "mean",
    "seed_std",
    "ci_low",
    "ci_high",
    "num_categories",
    "num_seeds",
    "ci_resampling_unit",
    "aggregation",
    "uncertainty_scope",
]
ABLATION_SUMMARY_FIELDS = [
    "variant",
    "method",
    "k",
    "seed",
    "category",
    "metric",
    "mean",
    "num_categories",
    "aggregation",
    "uncertainty_scope",
]
QUALITATIVE_PANEL_NAMES = (
    "image",
    "mask",
    "anomaly_panel",
    "sam2_panel",
    "fusion_panel",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-evidence-dir", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--primary-config", type=Path, required=True)
    parser.add_argument("--ablation-config", type=Path, required=True)
    parser.add_argument("--primary-check", type=Path, required=True)
    parser.add_argument("--ablation-check", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--postprocessor-commit", required=True)
    parser.add_argument("--method-figure-layout", type=Path, required=True)
    parser.add_argument("--qualitative-figure-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        _require_current_commit(args.postprocessor_commit)
        finalize_paper_evidence(
            source_evidence_dir=args.source_evidence_dir,
            audit_dir=args.audit_dir,
            primary_check=args.primary_check,
            ablation_check=args.ablation_check,
            output_dir=args.output_dir,
            source_commit=args.source_commit,
            postprocessor_commit=args.postprocessor_commit,
            generation_command=shlex.join(sys.argv),
            primary_config=args.primary_config,
            ablation_config=args.ablation_config,
            method_figure_layout=args.method_figure_layout,
            qualitative_figure_dir=args.qualitative_figure_dir,
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc


def finalize_paper_evidence(
    *,
    source_evidence_dir: str | Path,
    audit_dir: str | Path,
    primary_check: str | Path,
    ablation_check: str | Path,
    output_dir: str | Path,
    source_commit: str,
    postprocessor_commit: str,
    generation_command: str,
    primary_config: Mapping[str, object] | str | Path,
    ablation_config: Mapping[str, object] | str | Path,
    method_figure_layout: str | Path,
    qualitative_figure_dir: str | Path,
) -> dict[str, object]:
    """Validate frozen inputs and create one immutable public evidence directory."""

    source_evidence_dir = Path(source_evidence_dir)
    audit_dir = Path(audit_dir)
    primary_check = Path(primary_check)
    ablation_check = Path(ablation_check)
    publish_dir = Path(output_dir)
    if publish_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing evidence directory: {publish_dir}")
    _validate_git_sha(source_commit, "source commit")
    _validate_git_sha(postprocessor_commit, "postprocessor commit")
    primary_config_data, primary_config_identity = _load_config_identity(primary_config)
    ablation_config_data, ablation_config_identity = _load_config_identity(ablation_config)
    primary_runs = expand_matrix(primary_config_data)
    ablation_runs = expand_matrix(ablation_config_data)
    expected_primary_ids = {run.run_id for run in primary_runs}
    expected_ablation_ids = {run.run_id for run in ablation_runs}
    expected_primary_runs = len(expected_primary_ids)
    expected_ablation_runs = len(expected_ablation_ids)
    if expected_primary_runs != len(primary_runs) or expected_ablation_runs != len(ablation_runs):
        raise ValueError("frozen configs expand to duplicate run IDs")

    source_manifest_path = source_evidence_dir / "completion_manifest.json"
    source_manifest = _read_json(source_manifest_path)
    source_runs = _validate_source_manifest(
        source_evidence_dir,
        source_manifest,
        source_commit=source_commit,
        expected_runs=expected_primary_runs + expected_ablation_runs,
    )
    audit_manifest_path = audit_dir / "audit_manifest.json"
    audit_manifest = _read_json(audit_manifest_path)
    _validate_generated_hashes(audit_dir, audit_manifest)
    if audit_manifest.get("schema_version") != 2:
        raise ValueError("frozen audit manifest schema is stale")
    if audit_manifest.get("source_commit") != source_commit:
        raise ValueError("frozen audit source commit mismatch")
    if audit_manifest.get("postprocessor_commit") != postprocessor_commit:
        raise ValueError("frozen audit postprocessor commit mismatch")
    invariants = _read_json(audit_dir / "method_invariants.json")
    if invariants.get("ok") is not True:
        raise ValueError("frozen method invariant audit is not clean")
    primary_report = _validate_checker(primary_check, expected_primary_runs, "primary")
    ablation_report = _validate_checker(ablation_check, expected_ablation_runs, "ablation")
    source_run_ids = {str(item["run_id"]) for item in source_runs if isinstance(item, Mapping)}
    source_by_id = {str(item["run_id"]): item for item in source_runs if isinstance(item, Mapping)}
    primary_checker_ids = _checker_run_ids(primary_report)
    ablation_checker_ids = _checker_run_ids(ablation_report)
    if primary_checker_ids != expected_primary_ids or ablation_checker_ids != expected_ablation_ids:
        raise ValueError("checker run IDs do not match frozen configs")
    if primary_checker_ids | ablation_checker_ids != source_run_ids:
        raise ValueError("checker run IDs do not match source evidence")

    _validate_config_bindings(
        audit_dir,
        primary_identity=primary_config_identity,
        ablation_identity=ablation_config_identity,
        primary_ids=expected_primary_ids,
        ablation_ids=expected_ablation_ids,
        primary_runs=primary_runs,
        ablation_runs=ablation_runs,
        source_by_id=source_by_id,
    )
    _validate_invariant_coverage(invariants, primary_runs)

    staging_dir = publish_dir.with_name(f".{publish_dir.name}.building-{uuid.uuid4().hex}")
    if staging_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing staging directory: {staging_dir}")
    output_dir = staging_dir

    output_dir.mkdir(parents=True)
    generated_paths: list[Path] = []
    for item in source_manifest["generated_files"]:
        name = str(item["path"])
        if name in {"evidence_index.md", "primary_results.md"}:
            continue
        source_path = source_evidence_dir / name
        destination = output_dir / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, destination)
        generated_paths.append(destination)
    generated_paths.extend(_copy_method_figure(Path(method_figure_layout), output_dir))
    generated_paths.extend(
        _copy_qualitative_figures(
            Path(qualitative_figure_dir),
            output_dir,
            categories=[str(value) for value in primary_config_data["categories"]],
            source_selection=output_dir / "qualitative_manifest.csv",
        )
    )

    source_copy = output_dir / f"source_completion_manifest_{source_commit[:7]}.json"
    shutil.copyfile(source_manifest_path, source_copy)
    generated_paths.append(source_copy)
    for item in audit_manifest["generated_files"]:
        source_path = audit_dir / str(item["path"])
        destination = output_dir / source_path.name
        shutil.copyfile(source_path, destination)
        generated_paths.append(destination)
    audit_copy = output_dir / "frozen_audit_manifest.json"
    shutil.copyfile(audit_manifest_path, audit_copy)
    generated_paths.append(audit_copy)
    primary_check_copy = output_dir / "matrix_primary_check.json"
    ablation_check_copy = output_dir / "matrix_ablations_check.json"
    shutil.copyfile(primary_check, primary_check_copy)
    shutil.copyfile(ablation_check, ablation_check_copy)
    generated_paths.extend([primary_check_copy, ablation_check_copy])

    primary_rows = _read_csv(output_dir / "primary_results.csv")
    ablation_rows = _read_csv(output_dir / "ablation_results.csv")
    _validate_csv_run_specs(
        primary_rows,
        primary_runs,
        source_by_id=source_by_id,
        source_commit=source_commit,
        label="primary",
    )
    _validate_csv_run_specs(
        ablation_rows,
        ablation_runs,
        source_by_id=source_by_id,
        source_commit=source_commit,
        label="ablation",
    )
    _validate_metric_coverage(primary_rows, label="primary")
    _validate_metric_coverage(ablation_rows, label="ablation")
    primary_summary = aggregate_all_primary_metrics(primary_rows, metrics=METRIC_COLUMNS)
    ablation_summary = aggregate_ablation_rows(ablation_rows, metrics=METRIC_COLUMNS)
    primary_summary_csv = _write_csv(
        output_dir / "primary_summary.csv", primary_summary, PRIMARY_SUMMARY_FIELDS
    )
    ablation_summary_csv = _write_csv(
        output_dir / "ablation_summary.csv", ablation_summary, ABLATION_SUMMARY_FIELDS
    )
    primary_summary_md = output_dir / "primary_summary.md"
    primary_summary_md.write_text(_primary_markdown(primary_summary), encoding="utf-8")
    ablation_summary_md = output_dir / "ablation_summary.md"
    ablation_summary_md.write_text(_ablation_markdown(ablation_summary), encoding="utf-8")
    generated_paths.extend(
        [primary_summary_csv, ablation_summary_csv, primary_summary_md, ablation_summary_md]
    )

    validation_summary = {
        "schema_version": 1,
        "source_commit": source_commit,
        "primary": _checker_summary(primary_report, expected_primary_runs),
        "ablations": _checker_summary(ablation_report, expected_ablation_runs),
        "method_invariants": {
            "ok": True,
            "runs_checked": invariants.get("runs_checked"),
            "images_checked": invariants.get("images_checked"),
        },
    }
    validation_path = _write_json(output_dir / "validation_summary.json", validation_summary)
    conclusion = _evidence_conclusion(_read_json(output_dir / "paired_statistics.json"))
    conclusion_path = _write_json(output_dir / "evidence_conclusion.json", conclusion)
    metric_path = _write_json(output_dir / "metric_definitions.json", _metric_definitions())
    revision = {
        "schema_version": 1,
        "source_commit": source_commit,
        "postprocessor_commit": postprocessor_commit,
        "source_completion_manifest_sha256": sha256_file(source_manifest_path),
        "frozen_audit_manifest_sha256": sha256_file(audit_manifest_path),
        "primary_checker_sha256": sha256_file(primary_check),
        "ablation_checker_sha256": sha256_file(ablation_check),
        "primary_config": primary_config_identity,
        "ablation_config": ablation_config_identity,
        "numeric_results_changed": False,
        "scope": "presentation, aggregation, terminology, and frozen-run audit only",
    }
    revision_path = _write_json(output_dir / "evidence_revision.json", revision)
    generated_paths.extend([validation_path, conclusion_path, metric_path, revision_path])
    index_path = output_dir / "evidence_index.md"
    index_path.write_text(
        _evidence_index(conclusion, source_commit=source_commit), encoding="utf-8"
    )
    generated_paths.append(index_path)

    generated_paths = sorted(set(generated_paths), key=lambda path: path.as_posix())
    manifest = {
        "schema_version": 3,
        "generation_command": generation_command,
        "ready_for_writing": True,
        "readiness_scope": "complete_paper_evidence",
        "evidence_revision": revision,
        "source_runs": source_runs,
        "generated_files": [
            {"path": path.relative_to(output_dir).as_posix(), "sha256": sha256_file(path)}
            for path in generated_paths
        ],
    }
    _reject_private_paths(output_dir)
    _write_json(output_dir / "completion_manifest.json", manifest)
    _reject_private_paths(output_dir)
    output_dir.replace(publish_dir)
    return manifest


def _copy_method_figure(layout_path: Path, output_dir: Path) -> list[Path]:
    layout = _read_json(layout_path)
    renderer_path = PROJECT_ROOT / "scripts/render_method_figure.py"
    if (
        layout.get("schema_version") != 2
        or layout.get("renderer_source_sha256") != sha256_file(renderer_path)
        or layout.get("blocks") != json.loads(json.dumps(BLOCKS))
    ):
        raise ValueError("method figure does not match the current renderer contract")
    relative = Path(str(layout.get("output_png", "")))
    if (
        not relative.name
        or relative.is_absolute()
        or ".." in relative.parts
        or relative.suffix.lower() != ".png"
    ):
        raise ValueError("method figure layout contains an unsafe PNG path")
    source_png = layout_path.parent / relative
    expected_hash = str(layout.get("output_png_sha256", ""))
    if not source_png.is_file() or sha256_file(source_png) != expected_hash:
        raise ValueError("method figure PNG checksum mismatch")
    destination_png = output_dir / "method_figure.png"
    destination_layout = output_dir / "method_figure_layout.json"
    shutil.copyfile(source_png, destination_png)
    published_layout = dict(layout)
    published_layout["output_png"] = destination_png.name
    _write_json(destination_layout, published_layout)
    return [destination_png, destination_layout]


def _copy_qualitative_figures(
    source_dir: Path,
    output_dir: Path,
    *,
    categories: Sequence[str],
    source_selection: Path,
) -> list[Path]:
    manifest_path = source_dir / "qualitative_figure_manifest.json"
    manifest = _read_json(manifest_path)
    renderer_path = PROJECT_ROOT / "scripts/render_qualitative_figures.py"
    if (
        manifest.get("schema_version") != 2
        or manifest.get("renderer_source_sha256") != sha256_file(renderer_path)
        or not re.fullmatch(r"[0-9a-f]{64}", str(manifest.get("source_manifest_sha256", "")))
    ):
        raise ValueError("qualitative figure does not match the current renderer contract")
    selection_rows = _read_csv(source_selection)
    expected_selection = {}
    for row in selection_rows:
        key = (str(row.get("category", "")), str(row.get("role", "")))
        if key in expected_selection:
            raise ValueError("qualitative source selections are duplicated")
        panel_hashes = {
            panel: str(row.get(f"sha256_{panel}", "")) for panel in QUALITATIVE_PANEL_NAMES
        }
        if any(not re.fullmatch(r"[0-9a-f]{64}", value) for value in panel_hashes.values()):
            raise ValueError("qualitative source panel identity is malformed")
        expected_selection[key] = (
            str(row.get("sample_id", "")),
            float(row.get("sam2_delta_f1", "nan")),
            panel_hashes,
        )
    expected_roles = {
        (category, role) for category in categories for role in ("success", "failure")
    }
    if set(expected_selection) != expected_roles:
        raise ValueError("qualitative source selections are incomplete")
    figures = manifest.get("figures")
    if not isinstance(figures, list) or len(figures) != len(expected_roles):
        raise ValueError("qualitative figure manifest is incomplete")
    observed_roles: set[tuple[str, str]] = set()
    destination_dir = output_dir / "qualitative_figures"
    destination_dir.mkdir(parents=True)
    generated = []
    for item in figures:
        if not isinstance(item, Mapping):
            raise ValueError("qualitative figure entry must be an object")
        key = (str(item.get("category", "")), str(item.get("role", "")))
        if key not in expected_roles or key in observed_roles:
            raise ValueError("qualitative figure roles are missing or duplicated")
        observed_roles.add(key)
        selection_sample, selection_delta, selection_panel_hashes = expected_selection[key]
        if (
            str(item.get("sample_id", "")) != selection_sample
            or float(item.get("sam2_delta_f1", "nan")) != selection_delta
        ):
            raise ValueError("qualitative figure does not match selected evidence case")
        if item.get("source_panel_sha256") != selection_panel_hashes:
            raise ValueError("qualitative figure source panel identity mismatch")
        relative = Path(str(item.get("path", "")))
        if not relative.name or relative.is_absolute() or ".." in relative.parts:
            raise ValueError("qualitative figure path is missing or unsafe")
        source = source_dir / relative
        if not source.is_file() or item.get("sha256") != sha256_file(source):
            raise ValueError("qualitative figure checksum mismatch")
        destination = destination_dir / relative.name
        if destination.exists():
            raise ValueError("qualitative figure filenames must be unique")
        shutil.copyfile(source, destination)
        generated.append(destination)
    if observed_roles != expected_roles:
        raise ValueError("qualitative figure roles are incomplete")
    destination_manifest = destination_dir / manifest_path.name
    published_manifest = dict(manifest)
    published_manifest["asset_selection_manifest_sha256"] = published_manifest.pop(
        "source_manifest_sha256"
    )
    published_manifest["public_selection_manifest_sha256"] = sha256_file(source_selection)
    published_manifest["figures"] = [
        {**dict(item), "path": Path(str(item["path"])).name}
        for item in figures
        if isinstance(item, Mapping)
    ]
    _write_json(destination_manifest, published_manifest)
    generated.append(destination_manifest)
    return generated


def _load_config_identity(
    value: Mapping[str, object] | str | Path,
) -> tuple[dict[str, object], dict[str, str]]:
    if isinstance(value, Mapping):
        config = dict(value)
        return config, {
            "path": "inline",
            "sha256": sha256_json(config),
            "canonical_sha256": sha256_json(config),
        }
    path = Path(value)
    config = load_experiment_config(path)
    try:
        portable_path = path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        portable_path = path.name
    return config, {
        "path": portable_path,
        "sha256": sha256_file(path),
        "canonical_sha256": sha256_json(config),
    }


def _validate_config_bindings(
    audit_dir: Path,
    *,
    primary_identity: Mapping[str, str],
    ablation_identity: Mapping[str, str],
    primary_ids: set[str],
    ablation_ids: set[str],
    primary_runs: Sequence[RunSpec],
    ablation_runs: Sequence[RunSpec],
    source_by_id: Mapping[str, Mapping[str, object]],
) -> None:
    bindings = _read_json(audit_dir / "config_bindings.json")
    for label, identity, run_ids, runs in (
        ("primary", primary_identity, primary_ids, primary_runs),
        ("ablations", ablation_identity, ablation_ids, ablation_runs),
    ):
        item = bindings.get(label)
        if not isinstance(item, Mapping):
            raise ValueError(f"missing frozen {label} config binding")
        expected = {
            "config_sha256": identity["sha256"],
            "config_canonical_sha256": identity["canonical_sha256"],
            "run_count": len(run_ids),
            "run_ids_sha256": sha256_json(sorted(run_ids)),
            "run_spec_identities_sha256": sha256_json(
                {run.run_id: run.identity_sha256 for run in runs}
            ),
            "effective_execution_identities_sha256": sha256_json(
                {
                    run_id: str(source_by_id[run_id]["effective_execution_sha256"])
                    for run_id in sorted(run_ids)
                }
            ),
        }
        if any(item.get(key) != value for key, value in expected.items()):
            raise ValueError(f"frozen {label} config binding mismatch")


def _validate_invariant_coverage(
    invariants: Mapping[str, object], primary_runs: Sequence[RunSpec]
) -> None:
    expected_guided = {run.run_id for run in primary_runs if run.method == "dinov2_multi_sam2"}
    expected_fusion = {
        run.run_id for run in primary_runs if run.method == "anomaly_consistent_sam2"
    }
    if not expected_guided and not expected_fusion:
        return
    guided = invariants.get("guided_run_ids_checked")
    fusion = invariants.get("fusion_run_ids_checked")
    if not isinstance(guided, list) or set(map(str, guided)) != expected_guided:
        raise ValueError("method invariant audit does not cover exact guided run set")
    if not isinstance(fusion, list) or set(map(str, fusion)) != expected_fusion:
        raise ValueError("method invariant audit does not cover exact fusion run set")
    if (
        invariants.get("runs_checked") != len(expected_guided)
        or int(invariants.get("images_checked", 0)) <= 0
    ):
        raise ValueError("method invariant audit coverage counts are incomplete")


def _validate_csv_run_specs(
    rows: Sequence[Mapping[str, object]],
    runs: Sequence[RunSpec],
    *,
    source_by_id: Mapping[str, Mapping[str, object]],
    source_commit: str,
    label: str,
) -> None:
    expected = {run.run_id: run for run in runs}
    observed = [str(row.get("run_id", "")) for row in rows]
    if len(set(observed)) != len(observed) or set(observed) != set(expected):
        raise ValueError(f"{label} CSV run IDs do not match frozen config")
    for row in rows:
        run_id = str(row["run_id"])
        source = source_by_id.get(run_id)
        if source is None:
            raise ValueError(f"{label} CSV run is absent from source manifest: {run_id}")
        if row.get("run_spec_sha256") != expected[run_id].identity_sha256:
            raise ValueError(f"{label} CSV run-spec identity mismatch: {run_id}")
        run = expected[run_id]
        expected_fields = {
            "method": run.method,
            "category": run.category,
            "k": str(run.k),
            "seed": str(run.seed),
        }
        if run.variant != "primary":
            expected_fields["variant"] = run.variant
        for field, value in expected_fields.items():
            if str(row.get(field, "")) != value:
                raise ValueError(f"{label} CSV run field mismatch: {run_id}, {field}")
        if row.get("git_commit") != source_commit or _truthy(row.get("git_dirty")):
            raise ValueError(f"{label} CSV run is stale or dirty: {run_id}")
        for field in ("manifest_sha256", "effective_execution_sha256"):
            value = str(row.get(field, ""))
            if not re.fullmatch(r"[0-9a-f]{64}", value) or value != str(source.get(field, "")):
                raise ValueError(f"{label} CSV/source {field} mismatch: {run_id}")


def _validate_metric_coverage(rows: Sequence[Mapping[str, object]], *, label: str) -> None:
    for row in rows:
        method = str(row.get("method", ""))
        if method in HEATMAP_METHODS:
            required = METRIC_COLUMNS
        elif method in MASK_METHODS:
            required = METRIC_COLUMNS[-4:]
        else:
            raise ValueError(f"{label} CSV contains unsupported method: {method}")
        for metric in required:
            value = row.get(metric)
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{label} CSV missing required metric: {row.get('run_id')}, {metric}"
                ) from exc
            if number != number or number in (float("inf"), float("-inf")):
                raise ValueError(f"{label} CSV metric is not finite: {row.get('run_id')}, {metric}")


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _validate_git_sha(value: str, label: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{40}", value):
        raise ValueError(f"{label} must be a lowercase 40-character Git SHA")


def _require_current_commit(expected: str) -> None:
    _validate_git_sha(expected, "postprocessor commit")
    observed = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
    ).strip()
    if observed != expected:
        raise ValueError("postprocessor commit does not match current repository HEAD")
    tracked_status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=PROJECT_ROOT,
        text=True,
    ).strip()
    if tracked_status:
        raise ValueError("postprocessor tracked worktree is dirty")


def _validate_source_manifest(
    root: Path,
    manifest: Mapping[str, object],
    *,
    source_commit: str,
    expected_runs: int,
) -> list[object]:
    if manifest.get("ready_for_writing") is not True:
        raise ValueError("source evidence is not writing-ready")
    if manifest.get("readiness_scope") != "complete_paper_evidence":
        raise ValueError("source evidence readiness scope is incomplete")
    _validate_generated_hashes(root, manifest)
    source_runs = manifest.get("source_runs")
    if not isinstance(source_runs, list) or len(source_runs) != expected_runs:
        raise ValueError(f"source evidence must contain exactly {expected_runs} runs")
    for item in source_runs:
        if not isinstance(item, Mapping):
            raise ValueError("source run identity must be an object")
        if item.get("git_commit") != source_commit or bool(item.get("git_dirty")):
            raise ValueError("source run is stale or dirty")
        for field in ("manifest_sha256", "effective_execution_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", str(item.get(field, ""))):
                raise ValueError(f"source run {field} is malformed")
    run_ids = [str(item["run_id"]) for item in source_runs if isinstance(item, Mapping)]
    if len(set(run_ids)) != len(run_ids):
        raise ValueError("source evidence contains duplicate run IDs")
    return list(source_runs)


def _validate_generated_hashes(root: Path, manifest: Mapping[str, object]) -> None:
    generated = manifest.get("generated_files")
    if not isinstance(generated, list):
        raise ValueError("generated file manifest must be a list")
    observed: set[str] = set()
    for item in generated:
        if not isinstance(item, Mapping):
            raise ValueError("generated file identity must be an object")
        name = str(item.get("path", ""))
        if not name or name in observed or Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("generated file path is missing, duplicate, or unsafe")
        observed.add(name)
        path = root / name
        if not path.is_file() or item.get("sha256") != sha256_file(path):
            raise ValueError(f"generated file checksum mismatch: {name}")


def _validate_checker(path: Path, expected_runs: int, label: str) -> dict[str, object]:
    report = _read_json(path)
    runs = report.get("runs")
    if report.get("ok") is not True or not isinstance(runs, list) or len(runs) != expected_runs:
        raise ValueError(f"{label} checker does not contain {expected_runs} valid runs")
    if any(
        not isinstance(item, Mapping)
        or item.get("ok") is not True
        or item.get("errors") not in ([], None)
        for item in runs
    ):
        raise ValueError(f"{label} checker contains bad runs")
    return dict(report)


def _checker_summary(report: Mapping[str, object], expected: int) -> dict[str, object]:
    runs = report["runs"]
    assert isinstance(runs, list)
    return {"ok": True, "expected": expected, "complete": len(runs), "bad": 0}


def _checker_run_ids(report: Mapping[str, object]) -> set[str]:
    runs = report["runs"]
    assert isinstance(runs, list)
    run_ids = {str(item["run_id"]) for item in runs if isinstance(item, Mapping)}
    if len(run_ids) != len(runs):
        raise ValueError("checker contains missing or duplicate run IDs")
    return run_ids


def _evidence_conclusion(statistics: Mapping[str, object]) -> dict[str, object]:
    comparisons = statistics.get("comparisons", {})
    if not isinstance(comparisons, Mapping):
        raise ValueError("paired statistics comparisons are malformed")
    comparison = comparisons.get("dinov2_multi_vs_anomaly_consistent_sam2", {})
    if not isinstance(comparison, Mapping) or not isinstance(comparison.get("overall"), Mapping):
        raise ValueError("primary paired comparison is missing")
    overall = comparison["overall"]
    assert isinstance(overall, Mapping)
    f1 = overall.get("f1", {})
    iou = overall.get("iou", {})
    if not isinstance(f1, Mapping) or not isinstance(iou, Mapping):
        raise ValueError("primary paired F1/IoU intervals are missing")
    if float(f1["ci_low"]) > 0.0 and float(iou["ci_low"]) > 0.0:
        outcome = "anomaly_consistent_sam2_improves_robustly"
    elif float(f1["ci_high"]) < 0.0 and float(iou["ci_high"]) < 0.0:
        outcome = "sam2_remains_inferior"
    else:
        outcome = "mixed_or_inconclusive"
    return {
        "schema_version": 1,
        "outcome": outcome,
        "comparison": "anomaly_consistent_sam2 minus calibrated dinov2_multi",
        "sample_inclusion": "anomaly_images_only",
        "delta_definition": "candidate_minus_baseline",
        "f1": dict(f1),
        "iou": dict(iou),
        "decision_rule": {
            "robust_improvement": "both 95% confidence interval lower bounds are above zero",
            "robustly_inferior": "both 95% confidence interval upper bounds are below zero",
            "otherwise": "mixed_or_inconclusive",
        },
    }


def _metric_definitions() -> dict[str, object]:
    return {
        "schema_version": 1,
        "image_auroc": "all official-test images in each category",
        "pixel_auroc": "deterministic global sample of at most 1000000 pixels per run; the sample is seeded by the run support seed and therefore may vary across support seeds",
        "aupro": "all full-resolution heatmaps; 256 thresholds; normalized through FPR 0.3",
        "calibrated_heatmap_masks": "all full-resolution pixels at the normal-validation q=0.995 threshold",
        "binary_sam2_masks": "saved binary model outputs evaluated on identical official-test images",
        "oracle_metrics": "test-optimal diagnostics only; excluded from primary comparisons; oracle pixel sampling follows the same deterministic per-run support-seed policy",
        "primary_aggregation": "per-category and equal-weight category macro; support-seed mean, sample standard deviation, and explicitly labeled support-seed interval",
        "paired_statistics": "candidate-minus-baseline repeated-measures bootstrap over shared support-seed indices and category-level test-image clusters, with category and k fixed",
        "ablation_aggregation": "single support seed 4880 at k=4; descriptive category and equal-weight category macro only",
    }


def _evidence_index(conclusion: Mapping[str, object], *, source_commit: str) -> str:
    return "\n".join(
        [
            "# Evidence Index",
            "",
            "Readiness scope: **complete paper evidence**.",
            "",
            f"Frozen conclusion: **{conclusion['outcome']}**.",
            "",
            "No model parameters are trained or fine-tuned. Each run builds a support-feature memory bank, optionally applies a PatchCore-style coreset, calibrates on normal validation images, and performs frozen-model inference.",
            "",
            "| Planned claim or artifact | Evidence source | Guardrail |",
            "| --- | --- | --- |",
            "| Anomaly-consistent SAM2 versus calibrated multi-scale DINOv2 | `evidence_conclusion.json`, `paired_statistics.json` | Anomalous images only; repeated-measures CI |",
            "| Multi-scale versus single-scale DINOv2 | `baseline_paired_statistics.json`, `primary_summary.csv` | Identical support/test pairing |",
            "| Multi-scale DINOv2 versus PatchCore-style baseline | `baseline_paired_statistics.json`, `primary_summary.csv` | Repository reproduction, not reference PatchCore code |",
            "| Anomaly-guided multi-scale versus single-scale SAM2 | `baseline_paired_statistics.json` | Not the deterministic SAM2-only baseline |",
            "| Deterministic SAM2-only baseline | `primary_summary.csv` | k=0, seed=0 descriptive result; no support-seed CI |",
            "| Pre-registered sensitivity variants | `ablation_summary.csv` | k=4, seed=4880 descriptive only |",
            "| Failure geometry and SAM2 behavior | `failure_strata.csv`, `paired_statistics.json` | Normal rows excluded from paired defect intervals |",
            "| Pixel contracts for selective fusion | `method_invariants.json` | Every primary image checked |",
            "| Matrix completion | `validation_summary.json`, `matrix_primary_check.json`, `matrix_ablations_check.json` | 364 + 48, zero bad |",
            "| Runtime, model, checkpoint, config identities | `runtime_provenance.json`, `evidence_revision.json` | "
            f"Source runs remain frozen at {source_commit[:7]} |",
            "| Metric definitions | `metric_definitions.json` | Calibrated and oracle metrics remain separate |",
            "| Qualitative examples | `qualitative_manifest.csv` | One positive and one negative SAM2-delta case per category; no aggregate implication |",
            "| Method figure | `method_figure_layout.json` | Deterministic layout and PNG checksum |",
            "",
            "The five support seeds repeat normal-support sampling within fold 0; they are not independent folds. Prior VisA test-set exposure must be disclosed. DeepPCB boxes are not segmentation masks.",
            "",
        ]
    )


def _primary_markdown(rows: Sequence[Mapping[str, object]]) -> str:
    lines = [
        "Support-seed summaries. `macro` gives each PCB category equal weight.",
        "",
        "| method | k | category | metric | mean | seed std | 95% interval | uncertainty |",
        "| --- | ---: | --- | --- | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        descriptive = row["uncertainty_scope"] == "descriptive_single_run"
        interval = (
            "not estimated"
            if descriptive
            else f"[{float(row['ci_low']):.4f}, {float(row['ci_high']):.4f}]"
        )
        seed_std = "not estimated" if descriptive else f"{float(row['seed_std']):.4f}"
        lines.append(
            f"| {row['method']} | {row['k']} | {row['category']} | {row['metric']} | "
            f"{float(row['mean']):.4f} | {seed_std} | {interval} | {row['uncertainty_scope']} |"
        )
    return "\n".join(lines) + "\n"


def _ablation_markdown(rows: Sequence[Mapping[str, object]]) -> str:
    lines = [
        "Descriptive pre-registered sensitivity results at k=4 and support seed 4880.",
        "",
        "| variant | method | category | metric | mean |",
        "| --- | --- | --- | --- | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['method']} | {row['category']} | "
            f"{row['metric']} | {float(row['mean']):.4f} |"
        )
    return "\n".join(lines) + "\n"


def _read_csv(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    fields: Sequence[str],
) -> Path:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _read_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise ValueError(f"missing evidence input: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"evidence JSON must be an object: {path.name}")
    return payload


def _write_json(path: Path, payload: Mapping[str, object]) -> Path:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _reject_private_paths(root: Path) -> None:
    needles = (
        b"/root/",
        b"/Users/",
        b"/home/",
        b"/private/",
        b"/tmp/",
        b"autodl-tmp",
        b".staging-",
    )
    windows_absolute = re.compile(rb"[A-Za-z]:[\\/]")
    unc_absolute = re.compile(rb"\\\\[^\\\r\n]+\\")
    hits = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        content = _private_path_scan_content(path)
        if (
            any(needle in content for needle in needles)
            or windows_absolute.search(content)
            or unc_absolute.search(content)
        ):
            hits.append(path.relative_to(root).as_posix())
    if hits:
        raise ValueError(f"private paths leaked into public evidence: {hits}")


def _private_path_scan_content(path: Path) -> bytes:
    if path.suffix.lower() != ".png":
        return path.read_bytes()

    text: list[str] = []
    with Image.open(path) as image:
        metadata_sources = [image.info, getattr(image, "text", {}), image.getexif()]
        for metadata in metadata_sources:
            for key, value in metadata.items():
                if isinstance(key, str):
                    text.append(key)
                if isinstance(value, str):
                    text.append(value)
    return "\n".join(text).encode("utf-8")


if __name__ == "__main__":
    main()
