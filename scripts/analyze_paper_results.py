#!/usr/bin/env python
"""Analyze paired anomaly/SAM2 failures for paper evidence."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from evaluation.failure_analysis import describe_mask_pair
from evaluation.masks import mask_confusion_metrics, resolve_mask_row_paths
from evaluation.metrics import resolve_score_row_paths
from evaluation.statistics import repeated_measures_paired_delta
from experiments.provenance import sha256_file
from experiments.runner import matrix_report
from experiments.spec import RunSpec, expand_matrix, load_experiment_config
from sam_refine.artifacts import anomaly_mask_from_heatmap
from utils.heatmap_io import load_heatmap
from utils.image import load_binary_mask


FAILURE_FIELDS = [
    "category",
    "k",
    "seed",
    "sample_id",
    "label",
    "anomaly_run_id",
    "sam2_run_id",
    "fusion_run_id",
    "image_path",
    "mask_path",
    "heatmap_path",
    "sam2_mask_path",
    "fusion_mask_path",
    "anomaly_panel_path",
    "sam2_panel_path",
    "fusion_panel_path",
    "gt_area_fraction",
    "gt_components",
    "gt_thinness",
    "area_stratum",
    "thinness_stratum",
    "anomaly_f1",
    "sam2_f1",
    "sam2_delta_f1",
    "fusion_f1",
    "fusion_delta_f1",
    "fusion_vs_sam2_delta_f1",
    "anomaly_iou",
    "sam2_iou",
    "sam2_delta_iou",
    "fusion_iou",
    "fusion_delta_iou",
    "fusion_vs_sam2_delta_iou",
    "mean_anomaly_inside_sam2",
    "anomaly_pixels",
    "sam2_pixels",
    "intersection_pixels",
    "union_pixels",
    "mask_iou",
    "sam2_to_anomaly_area_ratio",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--dependency-root", type=Path)
    parser.add_argument("--feature-cache-dir", type=Path)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analyze_paper_results(
        output_root=args.output_root,
        analysis_dir=args.analysis_dir,
        config_path=args.config,
        dependency_root=args.dependency_root,
        feature_cache_dir=args.feature_cache_dir,
        device=args.device,
    )


def analyze_paper_results(
    *,
    output_root: str | Path,
    analysis_dir: str | Path,
    config_path: str | Path | None = None,
    categories: Sequence[str] | None = None,
    shots: Sequence[int] | None = None,
    seeds: Sequence[int] | None = None,
    validate_matrix: bool = True,
    dependency_root: str | Path | None = None,
    feature_cache_dir: str | Path | None = None,
    device: str = "auto",
    bootstrap_samples: int = 2000,
) -> dict[str, object]:
    """Write per-image failure descriptors and paired statistics."""

    output_root = Path(output_root)
    analysis_dir = Path(analysis_dir)
    config: Mapping[str, object] | None = None
    if config_path is not None:
        config = load_experiment_config(config_path)
        categories = tuple(str(value) for value in config["categories"])
        shots = tuple(int(value) for value in config["shots"])
        seeds = tuple(int(value) for value in config["seeds"])
        fold_id = int(config["fold_id"])
        if validate_matrix:
            _validate_completed_matrix(
                config,
                Path(config_path),
                output_root,
                Path(dependency_root) if dependency_root is not None else output_root,
                (
                    Path(feature_cache_dir)
                    if feature_cache_dir is not None
                    else output_root / ".feature_cache"
                ),
                device,
            )
    else:
        fold_id = 0
    if categories is None or shots is None or seeds is None:
        raise ValueError("categories, shots, and seeds are required without config_path")

    rows = _collect_failure_rows(
        output_root,
        categories=categories,
        shots=shots,
        seeds=seeds,
        fold_id=fold_id,
    )
    strata = _assign_strata(rows)
    statistics = _paired_statistics(rows, strata, bootstrap_samples=bootstrap_samples)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    per_image_path = analysis_dir / "per_image_failure_analysis.csv"
    statistics_path = analysis_dir / "paired_statistics.json"
    _write_csv(per_image_path, rows, FAILURE_FIELDS)
    _write_json(statistics_path, statistics)
    analysis_manifest_path = analysis_dir / "analysis_manifest.json"
    _write_json(
        analysis_manifest_path,
        _analysis_manifest(
            rows,
            output_root=output_root,
            config_path=Path(config_path) if config_path is not None else None,
            per_image_path=per_image_path,
            statistics_path=statistics_path,
        ),
    )
    return {
        "rows": rows,
        "statistics": statistics,
        "per_image_csv": str(analysis_dir / "per_image_failure_analysis.csv"),
        "paired_statistics_json": str(analysis_dir / "paired_statistics.json"),
        "analysis_manifest_json": str(analysis_manifest_path),
    }


def _validate_completed_matrix(
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


def _collect_failure_rows(
    output_root: Path,
    *,
    categories: Sequence[str],
    shots: Sequence[int],
    seeds: Sequence[int],
    fold_id: int,
) -> list[dict[str, str | float | int]]:
    rows: list[dict[str, str | float | int]] = []
    for category in categories:
        for k in shots:
            for seed in seeds:
                heatmap_run = RunSpec("dinov2_multi", category, fold_id, int(k), int(seed))
                sam2_run = RunSpec("dinov2_multi_sam2", category, fold_id, int(k), int(seed))
                fusion_run = RunSpec(
                    "anomaly_consistent_sam2", category, fold_id, int(k), int(seed)
                )
                rows.extend(
                    _collect_run_triplet(
                        output_root,
                        heatmap_run=heatmap_run,
                        sam2_run=sam2_run,
                        fusion_run=fusion_run,
                    )
                )
    return rows


def _collect_run_triplet(
    output_root: Path,
    *,
    heatmap_run: RunSpec,
    sam2_run: RunSpec,
    fusion_run: RunSpec,
) -> list[dict[str, str | float | int]]:
    heatmap_dir = output_root / heatmap_run.run_id / "test"
    sam2_dir = output_root / sam2_run.run_id / "test"
    fusion_dir = output_root / fusion_run.run_id / "test"
    heatmap_rows = resolve_score_row_paths(_read_csv(heatmap_dir / "scores.csv"), heatmap_dir)
    heatmap_metrics = {
        row["sample_id"]: row
        for row in _read_optional_csv(heatmap_dir / "per_image.csv")
        if row.get("sample_id")
    }
    sam2_rows = {
        row["sample_id"]: row
        for row in resolve_mask_row_paths(_read_csv(sam2_dir / "mask_scores.csv"), sam2_dir)
    }
    fusion_rows = {
        row["sample_id"]: row
        for row in resolve_mask_row_paths(_read_csv(fusion_dir / "mask_scores.csv"), fusion_dir)
    }
    output: list[dict[str, str | float | int]] = []
    for heatmap_row in heatmap_rows:
        sample_id = heatmap_row["sample_id"]
        if sample_id not in sam2_rows or sample_id not in fusion_rows:
            raise ValueError(f"missing paired SAM2/fusion rows for {sample_id}")
        heatmap = load_heatmap(heatmap_row["heatmap_path"]).astype(np.float32, copy=False)
        target = _target_mask(heatmap_row, heatmap.shape)
        threshold = _proposal_threshold(
            heatmap_metrics.get(sample_id, {}), output_root / heatmap_run.run_id
        )
        anomaly = anomaly_mask_from_heatmap(heatmap, threshold)
        sam2_mask = load_binary_mask(
            sam2_rows[sample_id]["sam2_mask_path"],
            size=(heatmap.shape[1], heatmap.shape[0]),
        )
        fusion_mask = load_binary_mask(
            fusion_rows[sample_id]["pred_mask_path"],
            size=(heatmap.shape[1], heatmap.shape[0]),
        )
        descriptor = describe_mask_pair(target, anomaly, sam2_mask, heatmap)
        fusion_metrics = mask_confusion_metrics(fusion_mask, target)
        output.append(
            {
                "category": heatmap_run.category,
                "k": heatmap_run.k,
                "seed": heatmap_run.seed,
                "sample_id": sample_id,
                "label": heatmap_row.get("label", ""),
                "anomaly_run_id": heatmap_run.run_id,
                "sam2_run_id": sam2_run.run_id,
                "fusion_run_id": fusion_run.run_id,
                "image_path": heatmap_row.get("image_path", ""),
                "mask_path": heatmap_row.get("mask_path", ""),
                "heatmap_path": heatmap_row.get("heatmap_path", ""),
                "sam2_mask_path": sam2_rows[sample_id].get("sam2_mask_path", ""),
                "fusion_mask_path": fusion_rows[sample_id].get("pred_mask_path", ""),
                "anomaly_panel_path": heatmap_row.get("debug_path", ""),
                "sam2_panel_path": sam2_rows[sample_id].get("debug_path", ""),
                "fusion_panel_path": fusion_rows[sample_id].get("debug_path", ""),
                **descriptor,
                "fusion_f1": fusion_metrics["f1"],
                "fusion_delta_f1": fusion_metrics["f1"] - descriptor["anomaly_f1"],
                "fusion_vs_sam2_delta_f1": fusion_metrics["f1"] - descriptor["sam2_f1"],
                "fusion_iou": fusion_metrics["iou"],
                "fusion_delta_iou": fusion_metrics["iou"] - descriptor["anomaly_iou"],
                "fusion_vs_sam2_delta_iou": fusion_metrics["iou"] - descriptor["sam2_iou"],
            }
        )
    return output


def _target_mask(row: Mapping[str, str], shape: tuple[int, int]) -> np.ndarray:
    if row.get("label") == "0":
        return np.zeros(shape, dtype=np.uint8)
    mask_path = row.get("mask_path", "")
    if not mask_path:
        raise ValueError(f"mask_path is required for anomalous row {row.get('sample_id')}")
    return load_binary_mask(mask_path, size=(shape[1], shape[0]))


def _proposal_threshold(metric_row: Mapping[str, str], run_dir: Path) -> float:
    if metric_row.get("threshold"):
        return _finite_float(metric_row["threshold"], "per-image threshold")
    calibration_path = run_dir / "calibration.json"
    if calibration_path.is_file():
        payload = json.loads(calibration_path.read_text(encoding="utf-8"))
        return _finite_float(payload.get("threshold"), "calibration threshold")
    raise ValueError("missing calibrated threshold for heatmap run")


def _assign_strata(rows: list[dict[str, str | float | int]]) -> dict[str, object]:
    defect_rows = [row for row in rows if float(row["gt_area_fraction"]) > 0.0]
    area = _stratum_spec([float(row["gt_area_fraction"]) for row in defect_rows], "area")
    thinness = _stratum_spec([float(row["gt_thinness"]) for row in defect_rows], "thinness")
    area_assignments = iter(area["assignments"])
    thinness_assignments = iter(thinness["assignments"])
    has_normal = False
    for row in rows:
        if float(row["gt_area_fraction"]) <= 0.0:
            row["area_stratum"] = "normal"
            row["thinness_stratum"] = "normal"
            has_normal = True
            continue
        row["area_stratum"] = next(area_assignments)
        row["thinness_stratum"] = next(thinness_assignments)
    area_labels = list(area["labels"])
    thinness_labels = list(thinness["labels"])
    if has_normal:
        area_labels.insert(0, "normal")
        thinness_labels.insert(0, "normal")
    return {
        "area": {"edges": area["edges"], "labels": area_labels},
        "thinness": {"edges": thinness["edges"], "labels": thinness_labels},
    }


def _stratum_spec(values: list[float], kind: str) -> dict[str, object]:
    if len(values) < 3:
        return {"edges": [], "labels": ["single"], "assignments": ["single"] * len(values)}
    edges_array = np.quantile(np.asarray(values, dtype=np.float64), [1 / 3, 2 / 3])
    edges = [float(value) for value in edges_array]
    if edges[0] == edges[1]:
        return {"edges": [], "labels": ["single"], "assignments": ["single"] * len(values)}
    labels = ["small", "medium", "large"] if kind == "area" else ["compact", "medium", "thin"]
    assignments = []
    for value in values:
        if value <= edges[0]:
            assignments.append(labels[0])
        elif value <= edges[1]:
            assignments.append(labels[1])
        else:
            assignments.append(labels[2])
    return {"edges": edges, "labels": labels, "assignments": assignments}


def _paired_statistics(
    rows: list[dict[str, str | float | int]],
    strata: Mapping[str, object],
    *,
    bootstrap_samples: int,
) -> dict[str, object]:
    anomaly_rows = [row for row in rows if str(row.get("label", "")) == "1"]
    if not anomaly_rows:
        raise ValueError("paired paper statistics require anomalous test rows")
    return {
        "comparisons": {
            "dinov2_multi_vs_dinov2_multi_sam2": _comparison_breakdown(
                anomaly_rows,
                "anomaly",
                "sam2",
                bootstrap_samples,
                excluded_normal_rows=len(rows) - len(anomaly_rows),
            ),
            "dinov2_multi_vs_anomaly_consistent_sam2": _comparison_breakdown(
                anomaly_rows,
                "anomaly",
                "fusion",
                bootstrap_samples,
                excluded_normal_rows=len(rows) - len(anomaly_rows),
            ),
            "dinov2_multi_sam2_vs_anomaly_consistent_sam2": _comparison_breakdown(
                anomaly_rows,
                "sam2",
                "fusion",
                bootstrap_samples,
                excluded_normal_rows=len(rows) - len(anomaly_rows),
            ),
        },
        "strata": dict(strata),
        "statistical_design": {
            "sample_inclusion": "anomaly_images_only",
            "paired_unit": "category_k_seed_sample_id",
            "resampling_unit": "support_seed_and_test_image",
            "fixed_strata": ["category", "k"],
            "category_aggregation": "macro_average",
        },
    }


def _comparison_breakdown(
    rows: list[dict[str, str | float | int]],
    baseline: str,
    candidate: str,
    bootstrap_samples: int,
    *,
    excluded_normal_rows: int,
) -> dict[str, object]:
    by_k = {
        str(k): _comparison(
            [row for row in rows if int(row["k"]) == k],
            baseline,
            candidate,
            bootstrap_samples,
        )
        for k in sorted({int(row["k"]) for row in rows})
    }
    by_category = {
        category: _comparison(
            [row for row in rows if str(row["category"]) == category],
            baseline,
            candidate,
            bootstrap_samples,
        )
        for category in sorted({str(row["category"]) for row in rows})
    }
    by_area_stratum = {
        stratum: _comparison(
            [row for row in rows if str(row["area_stratum"]) == stratum],
            baseline,
            candidate,
            bootstrap_samples,
        )
        for stratum in sorted({str(row["area_stratum"]) for row in rows})
    }
    by_thinness_stratum = {
        stratum: _comparison(
            [row for row in rows if str(row["thinness_stratum"]) == stratum],
            baseline,
            candidate,
            bootstrap_samples,
        )
        for stratum in sorted({str(row["thinness_stratum"]) for row in rows})
    }
    return {
        "scope": {
            "sample_inclusion": "anomaly_images_only",
            "excluded_normal_rows": excluded_normal_rows,
        },
        "overall": _comparison(rows, baseline, candidate, bootstrap_samples),
        "by_k": by_k,
        "by_category": by_category,
        "by_area_stratum": by_area_stratum,
        "by_thinness_stratum": by_thinness_stratum,
    }


def _comparison(
    rows: list[dict[str, str | float | int]],
    baseline: str,
    candidate: str,
    bootstrap_samples: int,
) -> dict[str, object]:
    return {
        "num_pairs": len(rows),
        "f1": repeated_measures_paired_delta(
            rows,
            baseline_field=f"{baseline}_f1",
            candidate_field=f"{candidate}_f1",
            samples=bootstrap_samples,
            seed=4880,
        ),
        "iou": repeated_measures_paired_delta(
            rows,
            baseline_field=f"{baseline}_iou",
            candidate_field=f"{candidate}_iou",
            samples=bootstrap_samples,
            seed=4880,
        ),
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_optional_csv(path: Path) -> list[dict[str, str]]:
    return _read_csv(path) if path.is_file() else []


def _write_csv(
    path: Path,
    rows: list[dict[str, str | float | int]],
    fieldnames: Sequence[str],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")


def _analysis_manifest(
    rows: Sequence[Mapping[str, object]],
    *,
    output_root: Path,
    config_path: Path | None,
    per_image_path: Path,
    statistics_path: Path,
) -> dict[str, object]:
    if config_path is None:
        portable_config_path = ""
    else:
        try:
            portable_config_path = config_path.resolve().relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            portable_config_path = config_path.name
    run_ids = sorted(
        {
            str(row[field])
            for row in rows
            for field in ("anomaly_run_id", "sam2_run_id", "fusion_run_id")
        }
    )
    source_runs = []
    for run_id in run_ids:
        provenance_path = output_root / run_id / "provenance.json"
        provenance = (
            json.loads(provenance_path.read_text(encoding="utf-8"))
            if provenance_path.is_file()
            else {}
        )
        source_runs.append(
            {
                "run_id": run_id,
                "effective_execution_sha256": provenance.get("effective_execution_sha256", ""),
                "run_spec_sha256": provenance.get("run_spec_sha256", ""),
                "git_commit": provenance.get("git_commit", ""),
            }
        )
    return {
        "schema_version": 1,
        "config_path": portable_config_path,
        "config_sha256": (
            sha256_file(config_path) if config_path is not None and config_path.is_file() else ""
        ),
        "source_runs": source_runs,
        "generated_files": [
            {"path": per_image_path.name, "sha256": sha256_file(per_image_path)},
            {"path": statistics_path.name, "sha256": sha256_file(statistics_path)},
        ],
    }


def _finite_float(value: object, context: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} must be finite") from exc
    if not np.isfinite(parsed):
        raise ValueError(f"{context} must be finite")
    return parsed


if __name__ == "__main__":
    main()
