"""Stage 4 comparison summaries across heatmap and mask baselines."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path


STAGE4_FIELDS = [
    "category",
    "method",
    "threshold_policy",
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

METHOD_ORDER = {
    "dinov2_single_heatmap": 0,
    "sam2_only": 1,
    "single_dinov2_sam2": 2,
    "dinov2_ms768_heatmap": 3,
    "ms768_dinov2_sam2": 4,
}


def summarize_stage4_files(paths: list[str | Path]) -> list[dict[str, float | str | None]]:
    """Summarize saved `metrics.json` and `mask_metrics.json` files."""

    rows = []
    for path in paths:
        path = Path(path)
        metrics = json.loads(path.read_text())
        category = infer_category(path)
        method = infer_method(path)
        rows.append(_row_from_metrics(category=category, method=method, metrics=metrics))

    rows.sort(key=_sort_key)
    rows.extend(_mean_rows(rows))
    return rows


def infer_category(path: Path) -> str:
    candidates = [path.parent.name, path.name]
    for candidate in candidates:
        match = re.search(r"(pcb\d+)", candidate)
        if match:
            return match.group(1)
    return path.parent.name


def infer_method(path: Path) -> str:
    name = path.parent.name
    if name.startswith("sam2_only_"):
        return "sam2_only"
    if name.endswith("_sam2") and "ms768" in name:
        return "ms768_dinov2_sam2"
    if name.endswith("_sam2"):
        return "single_dinov2_sam2"
    if "ms768" in name:
        return "dinov2_ms768_heatmap"
    if "dinov2" in name:
        return "dinov2_single_heatmap"
    return name


def format_stage4_markdown(rows: list[dict[str, float | str | None]]) -> str:
    lines = [
        "| category | method | threshold policy | image AUROC | pixel AUROC | AUPRO | "
        "aggregate pixel F1 | aggregate pixel IoU | mean anomaly mask F1 | "
        "mean anomaly mask IoU | precision | recall |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {category} | {method} | {threshold_policy} | {image_auroc} | "
            "{pixel_auroc} | {aupro} | {aggregate_pixel_f1} | "
            "{aggregate_pixel_iou} | {mean_anomaly_mask_f1} | "
            "{mean_anomaly_mask_iou} | {mean_anomaly_mask_precision} | "
            "{mean_anomaly_mask_recall} |".format(
                category=row["category"],
                method=row["method"],
                threshold_policy=row.get("threshold_policy", "-"),
                image_auroc=_format_metric(row.get("image_auroc")),
                pixel_auroc=_format_metric(row.get("pixel_auroc")),
                aupro=_format_metric(row.get("aupro")),
                aggregate_pixel_f1=_format_metric(row.get("aggregate_pixel_f1")),
                aggregate_pixel_iou=_format_metric(row.get("aggregate_pixel_iou")),
                mean_anomaly_mask_f1=_format_metric(row.get("mean_anomaly_mask_f1")),
                mean_anomaly_mask_iou=_format_metric(row.get("mean_anomaly_mask_iou")),
                mean_anomaly_mask_precision=_format_metric(
                    row.get("mean_anomaly_mask_precision")
                ),
                mean_anomaly_mask_recall=_format_metric(
                    row.get("mean_anomaly_mask_recall")
                ),
            )
        )
    return "\n".join(lines) + "\n"


def _row_from_metrics(
    category: str,
    method: str,
    metrics: dict[str, float],
) -> dict[str, float | str | None]:
    if "image_auroc" in metrics or "pixel_auroc" in metrics:
        required = {
            "calibrated_aggregate_pixel_f1",
            "calibrated_aggregate_pixel_iou",
            "calibrated_mean_anomaly_mask_f1",
            "calibrated_mean_anomaly_mask_iou",
            "calibrated_mean_anomaly_mask_precision",
            "calibrated_mean_anomaly_mask_recall",
        }
        if not required.issubset(metrics):
            raise ValueError("calibrated heatmap metrics required")
        return {
            "category": category,
            "method": method,
            "threshold_policy": "normal_q995",
            "image_auroc": _optional_float(metrics.get("image_auroc")),
            "pixel_auroc": _optional_float(metrics.get("pixel_auroc")),
            "aupro": _optional_float(metrics.get("aupro")),
            "aggregate_pixel_f1": _optional_float(
                metrics.get("calibrated_aggregate_pixel_f1")
            ),
            "aggregate_pixel_iou": _optional_float(
                metrics.get("calibrated_aggregate_pixel_iou")
            ),
            "mean_anomaly_mask_f1": _optional_float(
                metrics.get("calibrated_mean_anomaly_mask_f1")
            ),
            "mean_anomaly_mask_iou": _optional_float(
                metrics.get("calibrated_mean_anomaly_mask_iou")
            ),
            "mean_anomaly_mask_precision": _optional_float(
                metrics.get("calibrated_mean_anomaly_mask_precision")
            ),
            "mean_anomaly_mask_recall": _optional_float(
                metrics.get("calibrated_mean_anomaly_mask_recall")
            ),
        }
    return {
        "category": category,
        "method": method,
        "threshold_policy": "binary_model_output",
        "image_auroc": None,
        "pixel_auroc": None,
        "aupro": None,
        "aggregate_pixel_f1": None,
        "aggregate_pixel_iou": None,
        "mean_anomaly_mask_f1": _optional_float(metrics.get("mean_anomaly_mask_f1")),
        "mean_anomaly_mask_iou": _optional_float(metrics.get("mean_anomaly_mask_iou")),
        "mean_anomaly_mask_precision": _optional_float(
            metrics.get("mean_anomaly_mask_precision")
        ),
        "mean_anomaly_mask_recall": _optional_float(
            metrics.get("mean_anomaly_mask_recall")
        ),
    }


def _mean_rows(
    rows: list[dict[str, float | str | None]],
) -> list[dict[str, float | str | None]]:
    rows_by_method: dict[str, list[dict[str, float | str | None]]] = {}
    for row in rows:
        rows_by_method.setdefault(str(row["method"]), []).append(row)

    mean_rows = []
    for method, method_rows in rows_by_method.items():
        mean = {
            "category": "mean",
            "method": method,
            "threshold_policy": method_rows[0].get("threshold_policy"),
        }
        for key in STAGE4_FIELDS[3:]:
            values = [row.get(key) for row in method_rows]
            numeric_values = [
                float(value)
                for value in values
                if value is not None and not math.isnan(float(value))
            ]
            mean[key] = sum(numeric_values) / len(numeric_values) if numeric_values else None
        mean_rows.append(mean)
    mean_rows.sort(key=_sort_key)
    return mean_rows


def _sort_key(row: dict[str, float | str | None]) -> tuple[int, str, int]:
    category = str(row["category"])
    category_key = 1 if category == "mean" else 0
    return (category_key, category, METHOD_ORDER.get(str(row["method"]), 999))


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _format_metric(value: object) -> str:
    if value is None:
        return "-"
    value = float(value)
    if math.isnan(value):
        return "-"
    return f"{value:.4f}"
