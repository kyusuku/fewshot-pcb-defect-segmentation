"""Provenance-backed compact tables for paper evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np

from evaluation.statistics import bootstrap_mean_ci
from experiments.spec import RunSpec


HEATMAP_METHODS = {"patchcore", "dinov2_single", "dinov2_multi"}
MASK_METHODS = {
    "sam2_only",
    "dinov2_single_sam2",
    "dinov2_multi_sam2",
    "anomaly_consistent_sam2",
}
METRIC_COLUMNS = [
    "image_auroc",
    "pixel_auroc",
    "aupro",
    "aggregate_pixel_f1",
    "aggregate_pixel_iou",
    "mean_anomaly_mask_f1",
    "mean_anomaly_mask_iou",
    "mean_anomaly_mask_precision",
    "mean_anomaly_mask_recall",
]


def collect_primary_rows(
    config: Mapping[str, object],
    output_root: str | Path,
) -> list[dict[str, object]]:
    """Collect one compact metrics row per configured primary run."""

    output_root = Path(output_root)
    rows: list[dict[str, object]] = []
    for method in config["methods"]:  # type: ignore[index]
        for run in _configured_runs(config, str(method)):
            run_dir = output_root / run.run_id
            metrics_path = _metrics_path(run_dir, run.method)
            if not metrics_path.is_file():
                raise ValueError(f"missing metrics for {run.run_id}")
            provenance_path = run_dir / "provenance.json"
            if not provenance_path.is_file():
                raise ValueError(f"missing provenance for {run.run_id}")
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "run_id": run.run_id,
                    "method": run.method,
                    "category": run.category,
                    "k": run.k,
                    "seed": run.seed,
                    **metric_fields(run.method, metrics),
                    "git_commit": provenance.get("git_commit", ""),
                    "git_dirty": bool(provenance.get("git_dirty", False)),
                    "manifest_path": _manifest_value(provenance, "path"),
                    "manifest_sha256": _manifest_value(provenance, "sha256"),
                    "effective_execution_sha256": provenance.get("effective_execution_sha256", ""),
                    "run_spec_sha256": provenance.get("run_spec_sha256", ""),
                }
            )
    return rows


def aggregate_primary_rows(
    rows: list[dict[str, object]],
    metric: str,
) -> list[dict[str, float | int | str]]:
    grouped: dict[tuple[str, int], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((str(row["method"]), int(row["k"])), []).append(row)
    output = []
    for (method, k), method_rows in sorted(grouped.items()):
        categories = sorted({str(row["category"]) for row in method_rows})
        seeds = sorted({int(row["seed"]) for row in method_rows})
        by_cell: dict[tuple[str, int], float] = {}
        for row in method_rows:
            key = (str(row["category"]), int(row["seed"]))
            if key in by_cell:
                raise ValueError(f"duplicate primary summary cell: {key!r}")
            by_cell[key] = float(row[metric])
        for category in categories:
            observed = {seed for cell_category, seed in by_cell if cell_category == category}
            if observed != set(seeds):
                raise ValueError("every category must contain the same support seeds")

        seed_macro_values = [
            float(np.mean([by_cell[(category, seed)] for category in categories])) for seed in seeds
        ]
        output.append(
            _aggregate_summary_row(
                method=method,
                k=k,
                category="macro",
                metric=metric,
                seed_values=seed_macro_values,
                num_categories=len(categories),
            )
        )
        for category in categories:
            output.append(
                _aggregate_summary_row(
                    method=method,
                    k=k,
                    category=category,
                    metric=metric,
                    seed_values=[by_cell[(category, seed)] for seed in seeds],
                    num_categories=1,
                )
            )
    return output


def aggregate_all_primary_metrics(
    rows: list[dict[str, object]],
    *,
    metrics: list[str] | None = None,
) -> list[dict[str, float | int | str]]:
    """Aggregate every applicable paper metric without mixing unavailable fields."""

    output: list[dict[str, float | int | str]] = []
    for metric in metrics or METRIC_COLUMNS:
        metric_rows = [row for row in rows if row.get(metric) not in (None, "")]
        if not metric_rows:
            continue
        for summary in aggregate_primary_rows(metric_rows, metric=metric):
            if int(summary["num_seeds"]) == 1:
                summary["uncertainty_scope"] = "descriptive_single_run"
                summary["seed_std"] = ""
                summary["ci_low"] = ""
                summary["ci_high"] = ""
            else:
                summary["uncertainty_scope"] = "support_seed_interval"
            output.append(summary)
    return output


def aggregate_ablation_rows(
    rows: list[dict[str, object]],
    *,
    metrics: list[str] | None = None,
) -> list[dict[str, float | int | str]]:
    """Summarize single-seed ablations descriptively by category and macro mean."""

    output: list[dict[str, float | int | str]] = []
    for metric in metrics or METRIC_COLUMNS:
        metric_rows = [row for row in rows if row.get(metric) not in (None, "")]
        grouped: dict[tuple[str, str, int, int], dict[str, float]] = {}
        for row in metric_rows:
            key = (
                str(row["variant"]),
                str(row["method"]),
                int(row["k"]),
                int(row["seed"]),
            )
            category = str(row["category"])
            category_values = grouped.setdefault(key, {})
            if category in category_values:
                raise ValueError(f"duplicate ablation summary cell: {key!r}, {category!r}")
            value = float(row[metric])
            if not np.isfinite(value):
                raise ValueError(f"ablation metric {metric} must be finite")
            category_values[category] = value
        for (variant, method, k, seed), category_values in sorted(grouped.items()):
            common = {
                "variant": variant,
                "method": method,
                "k": k,
                "seed": seed,
                "metric": metric,
                "uncertainty_scope": "descriptive_single_support_seed",
            }
            output.append(
                {
                    **common,
                    "category": "macro",
                    "mean": float(np.mean(list(category_values.values()))),
                    "num_categories": len(category_values),
                    "aggregation": "category_macro",
                }
            )
            for category, value in sorted(category_values.items()):
                output.append(
                    {
                        **common,
                        "category": category,
                        "mean": value,
                        "num_categories": 1,
                        "aggregation": "within_category",
                    }
                )
    return output


def format_primary_markdown(rows: list[dict[str, object]]) -> str:
    lines = [
        "Support-seed summaries. Categories are fixed; `macro` gives each category equal weight.",
        "",
        "| method | k | category | metric | mean | seed std | support-seed 95% CI | seeds |",
        "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(
        rows,
        key=lambda item: (
            str(item["method"]),
            int(item["k"]),
            0 if str(item["category"]) == "macro" else 1,
            str(item["category"]),
        ),
    ):
        lines.append(
            "| {method} | {k} | {category} | {metric} | {mean:.4f} | {seed_std:.4f} | "
            "[{ci_low:.4f}, {ci_high:.4f}] | {num_seeds} |".format(
                method=row["method"],
                k=int(row["k"]),
                category=row["category"],
                metric=row["metric"],
                mean=float(row["mean"]),
                seed_std=float(row["seed_std"]),
                ci_low=float(row["ci_low"]),
                ci_high=float(row["ci_high"]),
                num_seeds=int(row["num_seeds"]),
            )
        )
    return "\n".join(lines) + "\n"


def _aggregate_summary_row(
    *,
    method: str,
    k: int,
    category: str,
    metric: str,
    seed_values: list[float],
    num_categories: int,
) -> dict[str, float | int | str]:
    interval = bootstrap_mean_ci(seed_values, samples=2000, seed=4880)
    seed_std = float(np.std(seed_values, ddof=1)) if len(seed_values) > 1 else 0.0
    return {
        "method": method,
        "k": k,
        "category": category,
        "metric": metric,
        "mean": interval["mean"],
        "seed_std": seed_std,
        "ci_low": interval["ci_low"],
        "ci_high": interval["ci_high"],
        "num_categories": num_categories,
        "num_seeds": len(seed_values),
        "ci_resampling_unit": "support_seed",
        "aggregation": "category_macro" if category == "macro" else "within_category",
    }


def _configured_runs(config: Mapping[str, object], method: str) -> list[RunSpec]:
    categories = [str(value) for value in config["categories"]]  # type: ignore[index]
    fold_id = int(config["fold_id"])
    if method == "sam2_only":
        return [RunSpec(method, category, fold_id, 0, 0) for category in categories]
    shots = [int(value) for value in config["shots"]]  # type: ignore[index]
    seeds = [int(value) for value in config["seeds"]]  # type: ignore[index]
    return [
        RunSpec(method, category, fold_id, k, seed)
        for category in categories
        for k in shots
        for seed in seeds
    ]


def _metrics_path(run_dir: Path, method: str) -> Path:
    if method in HEATMAP_METHODS:
        return run_dir / "test" / "metrics.json"
    if method in MASK_METHODS:
        return run_dir / "test" / "mask_metrics.json"
    raise ValueError(f"unsupported method for paper table: {method}")


def metric_fields(method: str, metrics: Mapping[str, object]) -> dict[str, object]:
    if method in HEATMAP_METHODS:
        return {
            "threshold_policy": "normal_q995",
            "image_auroc": _optional_metric(metrics, "image_auroc"),
            "pixel_auroc": _optional_metric(metrics, "pixel_auroc"),
            "aupro": _optional_metric(metrics, "aupro"),
            "aggregate_pixel_f1": _required_metric(metrics, "calibrated_aggregate_pixel_f1"),
            "aggregate_pixel_iou": _required_metric(metrics, "calibrated_aggregate_pixel_iou"),
            "mean_anomaly_mask_f1": _required_metric(metrics, "calibrated_mean_anomaly_mask_f1"),
            "mean_anomaly_mask_iou": _required_metric(metrics, "calibrated_mean_anomaly_mask_iou"),
            "mean_anomaly_mask_precision": _required_metric(
                metrics, "calibrated_mean_anomaly_mask_precision"
            ),
            "mean_anomaly_mask_recall": _required_metric(
                metrics, "calibrated_mean_anomaly_mask_recall"
            ),
        }
    return {
        "threshold_policy": "binary_model_output",
        "image_auroc": None,
        "pixel_auroc": None,
        "aupro": None,
        "aggregate_pixel_f1": None,
        "aggregate_pixel_iou": None,
        "mean_anomaly_mask_f1": _required_metric(metrics, "mean_anomaly_mask_f1"),
        "mean_anomaly_mask_iou": _required_metric(metrics, "mean_anomaly_mask_iou"),
        "mean_anomaly_mask_precision": _required_metric(metrics, "mean_anomaly_mask_precision"),
        "mean_anomaly_mask_recall": _required_metric(metrics, "mean_anomaly_mask_recall"),
    }


def _optional_metric(metrics: Mapping[str, object], key: str) -> float | None:
    return None if metrics.get(key) is None else float(metrics[key])


def _required_metric(metrics: Mapping[str, object], key: str) -> float:
    if metrics.get(key) is None:
        raise ValueError(f"missing required metric {key}")
    return float(metrics[key])


def _manifest_value(provenance: Mapping[str, object], key: str) -> str:
    manifest = provenance.get("manifest", {})
    if not isinstance(manifest, Mapping):
        return ""
    return str(manifest.get(key, ""))
