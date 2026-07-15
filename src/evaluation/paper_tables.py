"""Provenance-backed compact tables for paper evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

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
                    **_metric_fields(run.method, metrics),
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
        values = [float(row[metric]) for row in method_rows]
        interval = bootstrap_mean_ci(values, samples=2000, seed=4880)
        output.append(
            {
                "method": method,
                "k": k,
                "metric": metric,
                "mean": interval["mean"],
                "ci_low": interval["ci_low"],
                "ci_high": interval["ci_high"],
                "num_categories": len({str(row["category"]) for row in method_rows}),
                "num_seeds": len({int(row["seed"]) for row in method_rows}),
            }
        )
    return output


def format_primary_markdown(rows: list[dict[str, object]]) -> str:
    lines = [
        "| method | k | metric | mean | 95% CI | categories | seeds |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(rows, key=lambda item: (str(item["method"]), int(item["k"]))):
        lines.append(
            "| {method} | {k} | {metric} | {mean:.4f} | [{ci_low:.4f}, {ci_high:.4f}] | "
            "{num_categories} | {num_seeds} |".format(
                method=row["method"],
                k=int(row["k"]),
                metric=row["metric"],
                mean=float(row["mean"]),
                ci_low=float(row["ci_low"]),
                ci_high=float(row["ci_high"]),
                num_categories=int(row["num_categories"]),
                num_seeds=int(row["num_seeds"]),
            )
        )
    return "\n".join(lines) + "\n"


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


def _metric_fields(method: str, metrics: Mapping[str, object]) -> dict[str, object]:
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
