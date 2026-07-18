"""Post-freeze statistical and pixel-invariant audits for paper runs."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from evaluation.masks import resolve_mask_row_paths
from evaluation.statistics import repeated_measures_paired_delta
from experiments.provenance import sha256_json
from experiments.spec import RunSpec
from sam_refine.artifacts import anomaly_mask_from_heatmap
from utils.heatmap_io import load_heatmap
from utils.image import load_binary_mask


_BASELINE_COMPARISONS = {
    "dinov2_multi_minus_dinov2_single": ("dinov2_single", "dinov2_multi"),
    "dinov2_multi_minus_patchcore_style": ("patchcore", "dinov2_multi"),
    "dinov2_multi_sam2_minus_dinov2_single_sam2": (
        "dinov2_single_sam2",
        "dinov2_multi_sam2",
    ),
}


def runtime_provenance_summary(
    output_root: str | Path,
    *,
    category: str,
    k: int,
    seed: int,
    fold_id: int,
    source_commit: str,
) -> dict[str, object]:
    """Extract a compact, path-safe identity summary from representative runs."""

    output_root = Path(output_root)
    representatives: dict[str, object] = {}
    manifest_sha256 = ""
    environment: object = None
    execution_environment: object = None
    source_revisions: dict[str, str] = {}
    for method in ("patchcore", "dinov2_multi", "dinov2_multi_sam2"):
        run = RunSpec(method, category, fold_id, k, seed)
        path = output_root / run.run_id / "provenance.json"
        if not path.is_file():
            raise ValueError(f"missing representative provenance: {run.run_id}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("git_commit") != source_commit or bool(payload.get("git_dirty")):
            raise ValueError(f"representative run is not from clean source commit: {run.run_id}")
        manifest = payload.get("manifest", {})
        if not isinstance(manifest, Mapping):
            raise ValueError(f"representative manifest is malformed: {run.run_id}")
        observed_manifest = str(manifest.get("sha256", ""))
        if not manifest_sha256:
            manifest_sha256 = observed_manifest
        elif observed_manifest != manifest_sha256:
            raise ValueError("representative runs disagree on dataset manifest")
        observed_environment = payload.get("environment")
        if environment is None:
            environment = observed_environment
        elif observed_environment != environment:
            raise ValueError("representative runs disagree on runtime environment")
        execution = payload.get("execution", {})
        if not isinstance(execution, Mapping):
            raise ValueError(f"execution identity is malformed: {run.run_id}")
        observed_execution_environment = _compact_execution_environment(execution)
        if execution_environment is None:
            execution_environment = observed_execution_environment
        elif observed_execution_environment != execution_environment:
            raise ValueError("representative runs disagree on execution environment")
        observed = payload.get("observed_identities", {})
        if not isinstance(observed, Mapping):
            raise ValueError(f"observed identities are malformed: {run.run_id}")
        revisions = observed.get("source_revisions", {})
        if isinstance(revisions, Mapping):
            for key, value in revisions.items():
                value = str(value)
                previous = source_revisions.get(str(key))
                if previous is not None and previous != value:
                    raise ValueError(f"source revision mismatch for {key}")
                source_revisions[str(key)] = value
        extractors = observed.get("extractors", [])
        if not isinstance(extractors, list):
            raise ValueError(f"extractor identities are malformed: {run.run_id}")
        representatives[method] = {
            "run_id": run.run_id,
            "effective_execution_sha256": payload.get("effective_execution_sha256", ""),
            "run_spec_sha256": payload.get("run_spec_sha256", ""),
            "extractors": [_compact_extractor(item) for item in extractors],
            "checkpoints": payload.get("checkpoints", {}),
            "model_configs": payload.get("model_configs", {}),
        }
    return {
        "schema_version": 1,
        "source_commit": source_commit,
        "manifest_sha256": manifest_sha256,
        "environment": environment,
        "execution_environment": execution_environment,
        "source_revisions": source_revisions,
        "representative_runs": representatives,
    }


def validate_run_config_binding(
    output_root: str | Path,
    *,
    runs: Sequence[RunSpec],
    source_commit: str,
    config_sha256: str,
    config_canonical_sha256: str,
) -> dict[str, object]:
    """Bind every frozen run to the exact run spec and experiment config identities."""

    output_root = Path(output_root)
    if len({run.run_id for run in runs}) != len(runs):
        raise ValueError("config binding run IDs must be unique")
    effective_identities: dict[str, str] = {}
    run_spec_identities: dict[str, str] = {}
    for run in runs:
        path = output_root / run.run_id / "provenance.json"
        if not path.is_file():
            raise ValueError(f"missing config-binding provenance: {run.run_id}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("run_id") != run.run_id:
            raise ValueError(f"run identity mismatch: {run.run_id}")
        if payload.get("git_commit") != source_commit or bool(payload.get("git_dirty")):
            raise ValueError(f"source identity mismatch: {run.run_id}")
        if payload.get("run_spec_sha256") != run.identity_sha256:
            raise ValueError(f"run-spec identity mismatch: {run.run_id}")
        config = payload.get("experiment_config", {})
        execution = payload.get("execution", {})
        if not isinstance(config, Mapping) or not isinstance(execution, Mapping):
            raise ValueError(f"config identity is malformed: {run.run_id}")
        if (
            config.get("sha256") != config_sha256
            or execution.get("config_sha256") != config_sha256
            or execution.get("config_canonical_sha256") != config_canonical_sha256
            or execution.get("run_spec_sha256") != run.identity_sha256
        ):
            raise ValueError(f"config identity mismatch: {run.run_id}")
        effective = str(payload.get("effective_execution_sha256", ""))
        if not re.fullmatch(r"[0-9a-f]{64}", effective):
            raise ValueError(f"effective execution identity is malformed: {run.run_id}")
        effective_identities[run.run_id] = effective
        run_spec_identities[run.run_id] = run.identity_sha256
    run_ids = sorted(run_spec_identities)
    return {
        "config_sha256": config_sha256,
        "config_canonical_sha256": config_canonical_sha256,
        "run_count": len(run_ids),
        "run_ids_sha256": sha256_json(run_ids),
        "run_spec_identities_sha256": sha256_json(run_spec_identities),
        "effective_execution_identities_sha256": sha256_json(effective_identities),
    }


def paired_baseline_statistics(
    output_root: str | Path,
    *,
    categories: Sequence[str],
    shots: Sequence[int],
    seeds: Sequence[int],
    fold_id: int,
    bootstrap_samples: int = 2000,
) -> dict[str, object]:
    """Compare preregistered baselines on identical anomalous test images."""

    output_root = Path(output_root)
    comparisons: dict[str, object] = {}
    for name, (baseline, candidate) in _BASELINE_COMPARISONS.items():
        rows = _paired_rows(
            output_root,
            baseline=baseline,
            candidate=candidate,
            categories=categories,
            shots=shots,
            seeds=seeds,
            fold_id=fold_id,
        )
        comparisons[name] = {
            "baseline": baseline,
            "candidate": candidate,
            "scope": {
                "sample_inclusion": "anomaly_images_only",
                "paired_unit": "category_k_seed_sample_id",
            },
            "overall": _paired_interval(rows, bootstrap_samples),
            "by_k": {
                str(k): _paired_interval(
                    [row for row in rows if int(row["k"]) == int(k)], bootstrap_samples
                )
                for k in sorted({int(row["k"]) for row in rows})
            },
            "by_category": {
                category: _paired_interval(
                    [row for row in rows if str(row["category"]) == category],
                    bootstrap_samples,
                )
                for category in sorted({str(row["category"]) for row in rows})
            },
        }
    return {
        "schema_version": 1,
        "delta_definition": "candidate_minus_baseline",
        "statistical_design": {
            "sample_inclusion": "anomaly_images_only",
            "paired_unit": "category_k_seed_sample_id",
            "resampling_unit": "support_seed_and_test_image",
            "fixed_strata": ["category", "k"],
            "category_aggregation": "macro_average",
        },
        "comparisons": comparisons,
    }


def audit_method_invariants(
    output_root: str | Path,
    *,
    categories: Sequence[str],
    shots: Sequence[int],
    seeds: Sequence[int],
    fold_id: int,
) -> dict[str, object]:
    """Check every primary guided/fused mask against its frozen pixel contract."""

    output_root = Path(output_root)
    counts = {
        "runs_checked": 0,
        "images_checked": 0,
        "provenance_violating_rows": 0,
        "mask_output_violating_rows": 0,
        "guided_pred_not_raw_sam2_pixels": 0,
        "fusion_raw_sam2_not_guided_raw_pixels": 0,
        "fusion_outside_raw_sam2_pixels": 0,
        "fusion_outside_anomaly_proposal_pixels": 0,
        "fusion_not_exact_intersection_pixels": 0,
    }
    guided_run_ids_checked: list[str] = []
    fusion_run_ids_checked: list[str] = []
    for category in categories:
        for k in shots:
            for seed in seeds:
                guided = RunSpec("dinov2_multi_sam2", category, fold_id, int(k), int(seed))
                fusion = RunSpec("anomaly_consistent_sam2", category, fold_id, int(k), int(seed))
                guided_dir = output_root / guided.run_id / "test"
                fusion_dir = output_root / fusion.run_id / "test"
                guided_rows = _resolved_mask_rows(guided_dir / "mask_scores.csv")
                fusion_rows = _resolved_mask_rows(fusion_dir / "mask_scores.csv")
                guided_by_id = _unique_rows(guided_rows, guided.run_id)
                fusion_by_id = _unique_rows(fusion_rows, fusion.run_id)
                if set(guided_by_id) != set(fusion_by_id):
                    raise ValueError(
                        f"guided/fusion sample IDs differ for {category}, k={k}, seed={seed}"
                    )
                counts["runs_checked"] += 1
                guided_run_ids_checked.append(guided.run_id)
                fusion_run_ids_checked.append(fusion.run_id)
                for sample_id in sorted(guided_by_id):
                    guided_row = guided_by_id[sample_id]
                    fusion_row = fusion_by_id[sample_id]
                    counts["images_checked"] += 1
                    counts["provenance_violating_rows"] += int(
                        not _paper_sam2_row(guided_row) or not _paper_sam2_row(fusion_row)
                    )
                    counts["mask_output_violating_rows"] += int(
                        guided_row.get("mask_output") != "sam2"
                        or guided_row.get("selected_source") != "sam2"
                        or fusion_row.get("mask_output") != "intersection"
                        or fusion_row.get("selected_source") != "intersection"
                    )
                    guided_raw = load_binary_mask(_required_path(guided_row, "sam2_mask_path"))
                    guided_pred = load_binary_mask(_required_path(guided_row, "pred_mask_path"))
                    fusion_raw = load_binary_mask(_required_path(fusion_row, "sam2_mask_path"))
                    fusion_pred = load_binary_mask(_required_path(fusion_row, "pred_mask_path"))
                    heatmap = load_heatmap(_required_path(fusion_row, "heatmap_path"))
                    anomaly = anomaly_mask_from_heatmap(
                        heatmap, _finite_threshold(fusion_row.get("proposal_threshold"))
                    )
                    shapes = {
                        guided_raw.shape,
                        guided_pred.shape,
                        fusion_raw.shape,
                        fusion_pred.shape,
                        anomaly.shape,
                    }
                    if len(shapes) != 1:
                        raise ValueError(f"native mask shapes differ for {sample_id}: {shapes}")
                    exact_intersection = (guided_raw > 0) & (anomaly > 0)
                    counts["guided_pred_not_raw_sam2_pixels"] += int(
                        np.count_nonzero(guided_pred != guided_raw)
                    )
                    counts["fusion_raw_sam2_not_guided_raw_pixels"] += int(
                        np.count_nonzero(fusion_raw != guided_raw)
                    )
                    counts["fusion_outside_raw_sam2_pixels"] += int(
                        np.count_nonzero((fusion_pred > 0) & (guided_raw == 0))
                    )
                    counts["fusion_outside_anomaly_proposal_pixels"] += int(
                        np.count_nonzero((fusion_pred > 0) & (anomaly == 0))
                    )
                    counts["fusion_not_exact_intersection_pixels"] += int(
                        np.count_nonzero((fusion_pred > 0) != exact_intersection)
                    )
    violation_fields = [key for key in counts if key.endswith("rows") or key.endswith("pixels")]
    return {
        "schema_version": 1,
        "ok": all(counts[key] == 0 for key in violation_fields),
        "contracts": [
            "guided prediction equals saved raw SAM2 mask when mask_output=sam2",
            "anomaly-consistent prediction is a subset of the saved raw SAM2 mask",
            "anomaly-consistent prediction is a subset of the calibrated anomaly proposal",
            "anomaly-consistent prediction equals the exact raw-SAM2/anomaly intersection",
            "every row is paper_evidence with refiner=sam2 and raw_mask_source=sam2",
        ],
        "guided_run_ids_checked": guided_run_ids_checked,
        "fusion_run_ids_checked": fusion_run_ids_checked,
        **counts,
    }


def _paired_rows(
    output_root: Path,
    *,
    baseline: str,
    candidate: str,
    categories: Sequence[str],
    shots: Sequence[int],
    seeds: Sequence[int],
    fold_id: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for category in categories:
        for k in shots:
            for seed in seeds:
                baseline_rows = _metric_rows(
                    output_root / RunSpec(baseline, category, fold_id, int(k), int(seed)).run_id,
                    baseline,
                )
                candidate_rows = _metric_rows(
                    output_root / RunSpec(candidate, category, fold_id, int(k), int(seed)).run_id,
                    candidate,
                )
                baseline_by_id = _unique_rows(baseline_rows, baseline)
                candidate_by_id = _unique_rows(candidate_rows, candidate)
                anomaly_ids = {
                    sample_id
                    for sample_id, row in baseline_by_id.items()
                    if str(row.get("label", "")) == "1"
                }
                if anomaly_ids != {
                    sample_id
                    for sample_id, row in candidate_by_id.items()
                    if str(row.get("label", "")) == "1"
                }:
                    raise ValueError(
                        f"paired anomaly image IDs differ for {baseline} and {candidate}"
                    )
                for sample_id in sorted(anomaly_ids):
                    baseline_row = baseline_by_id[sample_id]
                    candidate_row = candidate_by_id[sample_id]
                    rows.append(
                        {
                            "category": category,
                            "k": int(k),
                            "seed": int(seed),
                            "sample_id": sample_id,
                            "baseline_f1": float(baseline_row["mask_f1"]),
                            "candidate_f1": float(candidate_row["mask_f1"]),
                            "baseline_iou": float(baseline_row["mask_iou"]),
                            "candidate_iou": float(candidate_row["mask_iou"]),
                        }
                    )
    return rows


def _paired_interval(rows: list[dict[str, object]], bootstrap_samples: int) -> dict[str, object]:
    return {
        "num_pairs": len(rows),
        "f1": repeated_measures_paired_delta(
            rows,
            baseline_field="baseline_f1",
            candidate_field="candidate_f1",
            samples=bootstrap_samples,
            seed=4880,
        ),
        "iou": repeated_measures_paired_delta(
            rows,
            baseline_field="baseline_iou",
            candidate_field="candidate_iou",
            samples=bootstrap_samples,
            seed=4880,
        ),
    }


def _metric_rows(run_dir: Path, method: str) -> list[dict[str, str]]:
    filename = "mask_per_image.csv" if "sam2" in method else "per_image.csv"
    return _read_csv(run_dir / "test" / filename)


def _resolved_mask_rows(path: Path) -> list[dict[str, str]]:
    return resolve_mask_row_paths(_read_csv(path), path.parent)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise ValueError(f"missing frozen audit input: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _unique_rows(rows: Sequence[Mapping[str, str]], context: str) -> dict[str, Mapping[str, str]]:
    output: dict[str, Mapping[str, str]] = {}
    for row in rows:
        sample_id = str(row.get("sample_id", ""))
        if not sample_id or sample_id in output:
            raise ValueError(f"missing or duplicate sample ID in {context}")
        output[sample_id] = row
    return output


def _paper_sam2_row(row: Mapping[str, str]) -> bool:
    return (
        row.get("evidence_class") == "paper_evidence"
        and row.get("refiner") == "sam2"
        and row.get("raw_mask_source") == "sam2"
    )


def _required_path(row: Mapping[str, str], field: str) -> Path:
    value = str(row.get(field, ""))
    if not value or not Path(value).is_file():
        raise ValueError(f"missing {field} for {row.get('sample_id', '')}")
    return Path(value)


def _finite_threshold(value: object) -> float:
    try:
        threshold = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("proposal threshold must be finite") from exc
    if not np.isfinite(threshold):
        raise ValueError("proposal threshold must be finite")
    return threshold


def _compact_extractor(item: object) -> dict[str, object]:
    if not isinstance(item, Mapping):
        raise ValueError("extractor identity entry must be an object")
    identity = item.get("identity", {})
    if not isinstance(identity, Mapping):
        raise ValueError("extractor identity must be an object")
    extractor = identity.get("extractor_identity", identity)
    if not isinstance(extractor, Mapping):
        raise ValueError("nested extractor identity must be an object")
    compact = {
        key: extractor[key]
        for key in (
            "extractor_class",
            "device",
            "configuration",
            "model_class",
            "model_state",
        )
        if key in extractor
    }
    for key in (
        "coreset_applied",
        "coreset_ratio",
        "coreset_projection_dim",
        "coreset_seed",
        "memory_bank_size_full",
        "memory_bank_size_used",
    ):
        if key in identity:
            compact[key] = identity[key]
    return compact


def _compact_execution_environment(execution: Mapping[str, object]) -> dict[str, object]:
    selected = str(execution.get("selected_device", ""))
    device = execution.get("device_identity", {})
    runtime = execution.get("runtime", {})
    if not selected or not isinstance(device, Mapping) or not isinstance(runtime, Mapping):
        raise ValueError("execution environment is missing device or runtime identity")
    libraries = runtime.get("libraries", {})
    if not isinstance(libraries, Mapping):
        raise ValueError("execution runtime libraries are malformed")
    torch_version = str(libraries.get("torch", ""))
    match = re.search(r"\+cu(\d{2})(\d)$", torch_version)
    cuda_version = f"{int(match.group(1))}.{int(match.group(2))}" if match else "unavailable"
    return {
        "selected_device": selected,
        "device_identity": dict(device),
        "runtime": dict(runtime),
        "cuda_available": selected == "cuda",
        "cuda_version": cuda_version,
    }
