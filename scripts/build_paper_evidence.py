#!/usr/bin/env python
"""Build compact, provenance-backed paper evidence artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from evaluation.paper_tables import (
    MASK_METHODS,
    METRIC_COLUMNS,
    aggregate_primary_rows,
    collect_primary_rows,
    format_primary_markdown,
    metric_fields,
)
from experiments.runner import matrix_report
from experiments.spec import RunSpec, expand_matrix, load_experiment_config


PRIMARY_FIELDS = [
    "run_id",
    "method",
    "category",
    "k",
    "seed",
    "threshold_policy",
    *METRIC_COLUMNS,
    "git_commit",
    "git_dirty",
    "manifest_path",
    "manifest_sha256",
    "effective_execution_sha256",
    "run_spec_sha256",
]
ABLATION_FIELDS = ["variant", *PRIMARY_FIELDS]
ORACLE_FIELDS = [
    "run_id",
    "method",
    "category",
    "k",
    "seed",
    "oracle_best_pixel_f1",
    "oracle_best_pixel_iou",
    "oracle_best_pixel_threshold",
    "best_pixel_f1",
    "best_pixel_iou",
    "best_pixel_threshold",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--ablation-config", type=Path, required=True)
    parser.add_argument("--ablation-output-root", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--qualitative-manifest", type=Path, required=True)
    parser.add_argument("--method-figure-layout", type=Path, required=True)
    parser.add_argument("--dependency-root", type=Path)
    parser.add_argument("--ablation-dependency-root", type=Path)
    parser.add_argument("--feature-cache-dir", type=Path)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        build_paper_evidence(
            config=load_experiment_config(args.config),
            output_root=args.output_root,
            ablation_config=load_experiment_config(args.ablation_config),
            ablation_output_root=args.ablation_output_root,
            analysis_dir=args.analysis_dir,
            evidence_dir=args.evidence_dir,
            qualitative_manifest=args.qualitative_manifest,
            method_figure_layout=args.method_figure_layout,
            validate_matrices=True,
            config_path=args.config,
            ablation_config_path=args.ablation_config,
            dependency_root=args.dependency_root,
            ablation_dependency_root=args.ablation_dependency_root,
            feature_cache_dir=args.feature_cache_dir,
            device=args.device,
            generation_command=" ".join(sys.argv),
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc


def build_paper_evidence(
    *,
    config: Mapping[str, object],
    output_root: str | Path,
    ablation_config: Mapping[str, object],
    ablation_output_root: str | Path,
    analysis_dir: str | Path,
    evidence_dir: str | Path,
    qualitative_manifest: str | Path | None = None,
    method_figure_layout: str | Path | None = None,
    validate_matrices: bool = True,
    config_path: str | Path | None = None,
    ablation_config_path: str | Path | None = None,
    dependency_root: str | Path | None = None,
    ablation_dependency_root: str | Path | None = None,
    feature_cache_dir: str | Path | None = None,
    device: str = "auto",
    generation_command: str = "scripts/build_paper_evidence.py",
) -> dict[str, object]:
    output_root = Path(output_root)
    ablation_output_root = Path(ablation_output_root)
    analysis_dir = Path(analysis_dir)
    evidence_dir = Path(evidence_dir)
    if validate_matrices:
        _require_matrix_ok(
            config,
            Path(config_path) if config_path is not None else Path("primary.yaml"),
            output_root,
            Path(dependency_root) if dependency_root is not None else output_root,
            Path(feature_cache_dir)
            if feature_cache_dir is not None
            else output_root / ".feature_cache",
            device,
        )
        _require_matrix_ok(
            ablation_config,
            Path(ablation_config_path)
            if ablation_config_path is not None
            else Path("ablations.yaml"),
            ablation_output_root,
            (
                Path(ablation_dependency_root)
                if ablation_dependency_root is not None
                else ablation_output_root
            ),
            (
                Path(feature_cache_dir)
                if feature_cache_dir is not None
                else ablation_output_root / ".feature_cache"
            ),
            device,
        )
        if qualitative_manifest is None or method_figure_layout is None:
            raise ValueError("complete evidence requires qualitative and method figure manifests")

    if qualitative_manifest is not None or method_figure_layout is not None:
        if qualitative_manifest is None or method_figure_layout is None:
            raise ValueError("qualitative and method figure manifests must be supplied together")
        _validate_paper_assets(
            Path(qualitative_manifest),
            Path(method_figure_layout),
            categories=[str(value) for value in config["categories"]],  # type: ignore[index]
        )

    evidence_dir.mkdir(parents=True, exist_ok=True)
    primary_rows = collect_primary_rows(config, output_root)
    if validate_matrices:
        _validate_analysis_manifest(
            analysis_dir,
            primary_rows,
            config_path=Path(config_path) if config_path is not None else None,
        )
    ablation_rows = _collect_ablation_rows(ablation_config, ablation_output_root)
    generated_paths = [
        _write_csv(evidence_dir / "primary_results.csv", primary_rows, PRIMARY_FIELDS),
        _write_primary_markdown(evidence_dir / "primary_results.md", primary_rows),
        _write_csv(
            evidence_dir / "oracle_diagnostics.csv",
            _collect_oracle_rows(primary_rows, output_root),
            ORACLE_FIELDS,
        ),
        _write_csv(evidence_dir / "ablation_results.csv", ablation_rows, ABLATION_FIELDS),
        _copy_required(
            analysis_dir / "paired_statistics.json",
            evidence_dir / "paired_statistics.json",
        ),
        _write_public_csv(
            analysis_dir / "per_image_failure_analysis.csv",
            evidence_dir / "failure_strata.csv",
        ),
    ]
    if qualitative_manifest is not None and method_figure_layout is not None:
        generated_paths.extend(
            [
                _write_public_csv(
                    Path(qualitative_manifest),
                    evidence_dir / "qualitative_manifest.csv",
                ),
                _copy_required(
                    Path(method_figure_layout),
                    evidence_dir / "method_figure_layout.json",
                ),
            ]
        )
    analysis_manifest = analysis_dir / "analysis_manifest.json"
    if analysis_manifest.is_file():
        generated_paths.append(
            _copy_required(analysis_manifest, evidence_dir / "analysis_manifest.json")
        )
    ready_for_writing = (
        validate_matrices and qualitative_manifest is not None and method_figure_layout is not None
    )
    generated_paths.append(
        _write_index(
            evidence_dir / "evidence_index.md",
            generated_paths,
            generation_command=generation_command,
            ready_for_writing=ready_for_writing,
        )
    )
    manifest = _completion_manifest(
        primary_rows=primary_rows,
        ablation_rows=ablation_rows,
        generated_paths=generated_paths,
        evidence_dir=evidence_dir,
        generation_command=generation_command,
        ready_for_writing=ready_for_writing,
    )
    manifest_path = evidence_dir / "completion_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _require_matrix_ok(
    config: Mapping[str, object],
    config_path: Path,
    output_root: Path,
    dependency_root: Path,
    feature_cache_dir: Path,
    device: str,
) -> None:
    report = matrix_report(
        expand_matrix(config),
        output_root,
        dependency_root,
        dict(config),
        config_path,
        device,
        feature_cache_dir,
    )
    if not report["ok"]:
        raise ValueError("experiment matrix is incomplete or stale")


def _collect_ablation_rows(
    config: Mapping[str, object],
    output_root: Path,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for ablation in config.get("ablations", []):  # type: ignore[union-attr]
        if not isinstance(ablation, Mapping):
            continue
        method = str(ablation["method"])
        variant = str(ablation["name"])
        for category in config["categories"]:  # type: ignore[index]
            run = RunSpec(
                method,
                str(category),
                int(config["fold_id"]),
                int(config["k"]),
                int(config["seed"]),
                variant=variant,
            )
            run_dir = output_root / run.run_id
            metrics_path = (
                run_dir / "test" / "mask_metrics.json"
                if method in MASK_METHODS
                else run_dir / "test" / "metrics.json"
            )
            if not metrics_path.is_file():
                raise ValueError(f"missing metrics for {run.run_id}")
            provenance_path = run_dir / "provenance.json"
            if not provenance_path.is_file():
                raise ValueError(f"missing provenance for {run.run_id}")
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "variant": variant,
                    "run_id": run.run_id,
                    "method": method,
                    "category": run.category,
                    "k": run.k,
                    "seed": run.seed,
                    **metric_fields(method, metrics),
                    "git_commit": provenance.get("git_commit", ""),
                    "git_dirty": bool(provenance.get("git_dirty", False)),
                    "manifest_path": _manifest_value(provenance, "path"),
                    "manifest_sha256": _manifest_value(provenance, "sha256"),
                    "effective_execution_sha256": provenance.get("effective_execution_sha256", ""),
                    "run_spec_sha256": provenance.get("run_spec_sha256", ""),
                }
            )
    return rows


def _collect_oracle_rows(
    primary_rows: list[dict[str, object]],
    output_root: Path,
) -> list[dict[str, object]]:
    rows = []
    for row in primary_rows:
        metrics_path = output_root / str(row["run_id"]) / "test" / "metrics.json"
        if not metrics_path.is_file():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "run_id": row["run_id"],
                "method": row["method"],
                "category": row["category"],
                "k": row["k"],
                "seed": row["seed"],
                "oracle_best_pixel_f1": metrics.get("oracle_best_pixel_f1"),
                "oracle_best_pixel_iou": metrics.get("oracle_best_pixel_iou"),
                "oracle_best_pixel_threshold": metrics.get("oracle_best_pixel_threshold"),
                "best_pixel_f1": metrics.get("best_pixel_f1"),
                "best_pixel_iou": metrics.get("best_pixel_iou"),
                "best_pixel_threshold": metrics.get("best_pixel_threshold"),
            }
        )
    return rows


def _write_primary_markdown(path: Path, primary_rows: list[dict[str, object]]) -> Path:
    summary = aggregate_primary_rows(primary_rows, metric="mean_anomaly_mask_f1")
    path.write_text(format_primary_markdown(summary), encoding="utf-8")
    return path


def _write_csv(
    path: Path,
    rows: list[dict[str, object]],
    fieldnames: Sequence[str],
) -> Path:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _copy_required(source: Path, destination: Path) -> Path:
    if not source.is_file():
        raise ValueError(f"missing required analysis artifact: {source.name}")
    shutil.copyfile(source, destination)
    return destination


def _write_public_csv(source: Path, destination: Path) -> Path:
    """Copy compact evidence while removing local/raw artifact path columns."""

    if not source.is_file():
        raise ValueError(f"missing required analysis artifact: {source.name}")
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"evidence CSV has no header: {source.name}")
        public_fields = [
            field
            for field in reader.fieldnames
            if not field.endswith("_path")
            and not field.endswith("_panel_path")
            and not field.endswith("_paths_json")
        ]
        rows = list(reader)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=public_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return destination


def _validate_paper_assets(
    qualitative_manifest: Path,
    method_figure_layout: Path,
    *,
    categories: Sequence[str],
) -> None:
    if not qualitative_manifest.is_file():
        raise ValueError("missing qualitative asset manifest")
    with qualitative_manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required_roles = {
        (category, role) for category in categories for role in ("success", "failure")
    }
    observed_roles = {(row.get("category", ""), row.get("role", "")) for row in rows}
    if not required_roles.issubset(observed_roles):
        raise ValueError("qualitative manifest must contain success and failure for every category")
    panel_names = ("image", "mask", "anomaly_panel", "sam2_panel", "fusion_panel")
    for row_index, row in enumerate(rows):
        try:
            copied_paths = json.loads(row.get("copied_paths_json", ""))
        except json.JSONDecodeError as exc:
            raise ValueError(f"qualitative row {row_index} has invalid copied paths") from exc
        if not isinstance(copied_paths, Mapping):
            raise ValueError(f"qualitative row {row_index} copied paths must be an object")
        for panel_name in panel_names:
            value = copied_paths.get(panel_name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"qualitative row {row_index} is missing {panel_name}")
            path = Path(value)
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            if not path.is_file():
                raise ValueError(f"qualitative row {row_index} {panel_name} does not exist")
            if row.get(f"sha256_{panel_name}", "") != _sha256_file(path):
                raise ValueError(f"qualitative row {row_index} {panel_name} checksum mismatch")

    if not method_figure_layout.is_file():
        raise ValueError("missing method figure layout manifest")
    layout = json.loads(method_figure_layout.read_text(encoding="utf-8"))
    if not isinstance(layout, Mapping):
        raise ValueError("method figure layout must be a JSON object")
    output_value = layout.get("output_png")
    if not isinstance(output_value, str) or not output_value:
        raise ValueError("method figure layout must identify output_png")
    output_png = Path(output_value)
    if not output_png.is_absolute():
        output_png = method_figure_layout.parent / output_png
    if not output_png.is_file():
        raise ValueError("method figure PNG does not exist")
    if layout.get("output_png_sha256") != _sha256_file(output_png):
        raise ValueError("method figure PNG checksum mismatch")


def _validate_analysis_manifest(
    analysis_dir: Path,
    primary_rows: Sequence[Mapping[str, object]],
    *,
    config_path: Path | None,
) -> None:
    manifest_path = analysis_dir / "analysis_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("missing analysis manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping) or manifest.get("schema_version") != 1:
        raise ValueError("analysis manifest is malformed")
    if config_path is not None:
        if not config_path.is_file() or manifest.get("config_sha256") != _sha256_file(config_path):
            raise ValueError("analysis manifest config checksum mismatch")

    expected_rows = {
        str(row["run_id"]): str(row.get("effective_execution_sha256", ""))
        for row in primary_rows
        if str(row.get("method", ""))
        in {"dinov2_multi", "dinov2_multi_sam2", "anomaly_consistent_sam2"}
    }
    source_runs = manifest.get("source_runs")
    if not isinstance(source_runs, list):
        raise ValueError("analysis manifest source_runs must be a list")
    observed_rows: dict[str, str] = {}
    for item in source_runs:
        if not isinstance(item, Mapping):
            raise ValueError("analysis manifest source run must be an object")
        run_id = str(item.get("run_id", ""))
        if not run_id or run_id in observed_rows:
            raise ValueError("analysis manifest has missing or duplicate run IDs")
        observed_rows[run_id] = str(item.get("effective_execution_sha256", ""))
    if observed_rows != expected_rows:
        raise ValueError("analysis manifest source run identities are incomplete or stale")

    generated_files = manifest.get("generated_files")
    if not isinstance(generated_files, list):
        raise ValueError("analysis manifest generated_files must be a list")
    expected_names = {"per_image_failure_analysis.csv", "paired_statistics.json"}
    observed_names: set[str] = set()
    for item in generated_files:
        if not isinstance(item, Mapping):
            raise ValueError("analysis manifest generated file must be an object")
        name = str(item.get("path", ""))
        if name in observed_names:
            raise ValueError("analysis manifest has duplicate generated files")
        observed_names.add(name)
        path = analysis_dir / name
        if not path.is_file() or item.get("sha256") != _sha256_file(path):
            raise ValueError(f"analysis generated file checksum mismatch: {name}")
    if observed_names != expected_names:
        raise ValueError("analysis manifest generated file set is incomplete")


def _write_index(
    path: Path,
    generated_paths: Sequence[Path],
    *,
    generation_command: str,
    ready_for_writing: bool,
) -> Path:
    status = "complete paper evidence" if ready_for_writing else "smoke or test evidence only"
    lines = [
        "# Evidence Index",
        "",
        f"Readiness scope: **{status}**.",
        "",
        "Generation command:",
        "",
        f"`{generation_command}`",
        "",
        "## Planned claim mapping",
        "",
        "| Planned claim | Evidence source | Guardrail |",
        "| --- | --- | --- |",
        "| Multi-scale DINOv2 versus single-scale proposals | `primary_results.csv` | "
        "Per-category and category-macro, fixed primary geometry |",
        "| PatchCore-style baseline | `primary_results.csv` | Same support protocol and split; "
        "repository implementation, not the reference PatchCore code |",
        "| Anomaly-guided SAM2 refinement helps or hurts conditionally | "
        "`paired_statistics.json`, "
        "`failure_strata.csv` | Anomaly images only; paired repeated-measures CI |",
        "| Anomaly-consistent SAM2 versus proposal and SAM2 masks | `paired_statistics.json`, "
        "`primary_results.md` | No improvement claim when delta CI includes zero |",
        "| Pre-registered sensitivity results | `ablation_results.csv` | Alternatives remain "
        "ablations, not replacement primaries |",
        "| Oracle threshold diagnostics | `oracle_diagnostics.csv` | Diagnostic only; never "
        "presented as deployable |",
        "",
        "## Planned table and figure mapping",
        "",
        "| Artifact | Evidence source |",
        "| --- | --- |",
        "| Primary quantitative table | `primary_results.csv`, `primary_results.md` |",
        "| Ablation table | `ablation_results.csv` |",
        "| Paired statistical table | `paired_statistics.json` |",
        "| Failure-stratified table | `failure_strata.csv` |",
        "| Category success/failure figure set | `qualitative_manifest.csv` |",
        "| Method/pipeline figure | `method_figure_layout.json` |",
        "",
        "## Generated compact files",
        "",
    ]
    for generated in generated_paths:
        lines.append(f"- `{generated.name}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _completion_manifest(
    *,
    primary_rows: list[dict[str, object]],
    ablation_rows: list[dict[str, object]],
    generated_paths: Sequence[Path],
    evidence_dir: Path,
    generation_command: str,
    ready_for_writing: bool,
) -> dict[str, object]:
    return {
        "schema_version": 2,
        "generation_command": generation_command,
        "ready_for_writing": ready_for_writing,
        "readiness_scope": (
            "complete_paper_evidence" if ready_for_writing else "smoke_or_test_only"
        ),
        "source_runs": [_source_run(row) for row in [*primary_rows, *ablation_rows]],
        "generated_files": [
            {
                "path": path.relative_to(evidence_dir).as_posix(),
                "sha256": _sha256_file(path),
            }
            for path in generated_paths
        ],
    }


def _source_run(row: Mapping[str, object]) -> dict[str, object]:
    return {
        "run_id": row["run_id"],
        "git_commit": row.get("git_commit", ""),
        "git_dirty": row.get("git_dirty", False),
        "manifest_path": row.get("manifest_path", ""),
        "manifest_sha256": row.get("manifest_sha256", ""),
        "effective_execution_sha256": row.get("effective_execution_sha256", ""),
    }


def _manifest_value(provenance: Mapping[str, object], key: str) -> str:
    manifest = provenance.get("manifest", {})
    if not isinstance(manifest, Mapping):
        return ""
    return str(manifest.get(key, ""))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
