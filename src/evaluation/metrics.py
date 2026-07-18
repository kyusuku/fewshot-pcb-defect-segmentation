"""Evaluation metrics for anomaly heatmaps and predicted masks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from evaluation.masks import mask_confusion_metrics, summarize_binary_metrics
from utils.heatmap_io import load_heatmap
from utils.image import load_binary_mask


@dataclass(frozen=True)
class _HeatmapRowInfo:
    row: dict[str, str]
    path: str
    offset: int
    size: int


def binary_roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute ROC AUC for binary labels where larger scores mean more anomalous."""

    labels = np.asarray(labels).astype(np.uint8).ravel()
    scores = np.asarray(scores, dtype=np.float64).ravel()
    if labels.shape[0] != scores.shape[0]:
        raise ValueError("labels and scores must have the same number of elements")

    num_pos = int(np.sum(labels == 1))
    num_neg = int(np.sum(labels == 0))
    if num_pos == 0 or num_neg == 0:
        return math.nan

    ranks = _average_ranks(scores)
    pos_rank_sum = float(np.sum(ranks[labels == 1]))
    auc = (pos_rank_sum - num_pos * (num_pos + 1) / 2.0) / (num_pos * num_neg)
    return float(auc)


def best_f1_iou(
    labels: np.ndarray,
    scores: np.ndarray,
    max_thresholds: int = 512,
) -> dict[str, float]:
    """Find the best binary threshold by F1 and report the IoU at that threshold."""

    labels = np.asarray(labels).astype(np.uint8).ravel()
    scores = np.asarray(scores, dtype=np.float32).ravel()
    if labels.shape[0] != scores.shape[0]:
        raise ValueError("labels and scores must have the same number of elements")
    if int(np.sum(labels == 1)) == 0:
        return {
            "best_f1": math.nan,
            "best_iou": math.nan,
            "best_threshold": math.nan,
            "num_thresholds": 0.0,
        }

    thresholds = _threshold_candidates(scores, max_thresholds=max_thresholds)
    best = {
        "best_f1": -1.0,
        "best_iou": 0.0,
        "best_threshold": math.nan,
        "num_thresholds": float(len(thresholds)),
    }
    for threshold in thresholds:
        prediction = scores >= threshold
        target = labels == 1
        tp = float(np.sum(prediction & target))
        fp = float(np.sum(prediction & ~target))
        fn = float(np.sum(~prediction & target))
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        iou = tp / (tp + fp + fn) if tp + fp + fn > 0 else 0.0
        if f1 > best["best_f1"]:
            best = {
                "best_f1": float(f1),
                "best_iou": float(iou),
                "best_threshold": float(threshold),
                "num_thresholds": float(len(thresholds)),
            }
    return best


def average_pro_score(
    targets: list[np.ndarray],
    scores: list[np.ndarray],
    max_fpr: float = 0.3,
    max_thresholds: int = 256,
) -> float:
    """Compute normalized area under the per-region-overlap curve.

    The PRO value at a threshold is the mean overlap over connected ground-truth
    anomaly regions. The x-axis is pixel false-positive rate on normal pixels,
    and the returned area is normalized by `max_fpr`.
    """

    if not 0.0 < max_fpr <= 1.0:
        raise ValueError("max_fpr must be in (0, 1]")
    if len(targets) != len(scores):
        raise ValueError("targets and scores must contain the same number of images")

    prepared_targets: list[np.ndarray] = []
    prepared_scores: list[np.ndarray] = []
    regions_by_image: list[list[np.ndarray]] = []
    num_negative_pixels = 0
    for target, score in zip(targets, scores):
        target = np.asarray(target).astype(bool)
        score = np.asarray(score, dtype=np.float32)
        if target.shape != score.shape:
            raise ValueError("each target and score map must have the same shape")
        prepared_targets.append(target)
        prepared_scores.append(score)
        regions = _connected_components(target)
        regions_by_image.append(regions)
        num_negative_pixels += int(np.sum(~target))

    num_regions = sum(len(regions) for regions in regions_by_image)
    if num_regions == 0 or num_negative_pixels == 0:
        return math.nan

    all_scores = np.concatenate([score.ravel() for score in prepared_scores], axis=0)
    thresholds = _threshold_candidates(all_scores, max_thresholds=max_thresholds)
    fpr_to_pro: dict[float, float] = {}
    for threshold in thresholds:
        false_positives = 0
        region_overlaps = []
        for target, score, regions in zip(prepared_targets, prepared_scores, regions_by_image):
            prediction = score >= threshold
            false_positives += int(np.sum(prediction & ~target))
            for region in regions:
                region_overlaps.append(float(np.sum(prediction & region)) / float(np.sum(region)))
        fpr = false_positives / float(num_negative_pixels)
        pro = float(sum(region_overlaps) / len(region_overlaps))
        fpr_to_pro[fpr] = max(pro, fpr_to_pro.get(fpr, 0.0))

    points = sorted(fpr_to_pro.items())
    if not points or points[0][0] > 0.0:
        points.insert(0, (0.0, 0.0))
    clipped = [(fpr, pro) for fpr, pro in points if fpr <= max_fpr]
    if not clipped or clipped[0][0] > 0.0:
        clipped.insert(0, (0.0, 0.0))
    if clipped[-1][0] < max_fpr:
        clipped.append((max_fpr, _interpolate_pro(points, max_fpr)))

    area = 0.0
    for (x1, y1), (x2, y2) in zip(clipped, clipped[1:]):
        area += (x2 - x1) * (y1 + y2) / 2.0
    return float(area / max_fpr)


def summarize_image_scores(rows: list[dict[str, str]]) -> dict[str, float]:
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.uint8)
    scores = np.asarray([float(row["image_score"]) for row in rows], dtype=np.float32)
    return {
        "num_images": float(len(rows)),
        "image_auroc": binary_roc_auc(labels, scores),
    }


def evaluate_heatmap_rows(
    rows: list[dict[str, str]],
    max_pixels: int | None = None,
    seed: int = 4880,
) -> dict[str, float]:
    """Evaluate image-level scores and pixel-level heatmaps from score CSV rows."""

    image_metrics = summarize_image_scores(rows)
    row_infos, total_pixels, thresholds = _inspect_heatmap_rows(rows, max_thresholds=256)
    selected_indices = _sample_global_indices(total_pixels, max_pixels=max_pixels, seed=seed)
    pixel_labels: list[np.ndarray] = []
    pixel_scores: list[np.ndarray] = []
    false_positive_counts = np.zeros(thresholds.shape, dtype=np.float64)
    region_overlap_sums = np.zeros(thresholds.shape, dtype=np.float64)
    num_negative_pixels = 0
    num_regions = 0
    for info in row_infos:
        heatmap = load_heatmap(info.path).astype(np.float32, copy=False)
        mask = _load_evaluation_target(info.row, heatmap.shape)
        selected_start = int(np.searchsorted(selected_indices, info.offset, side="left"))
        selected_end = int(np.searchsorted(selected_indices, info.offset + info.size, side="left"))
        local_indices = selected_indices[selected_start:selected_end] - info.offset
        if local_indices.size:
            pixel_labels.append(mask.ravel()[local_indices])
            pixel_scores.append(heatmap.ravel()[local_indices])

        image_negative_pixels, image_regions = _accumulate_streaming_pro_image(
            heatmap=heatmap,
            mask=mask,
            thresholds=thresholds,
            false_positive_counts=false_positive_counts,
            region_overlap_sums=region_overlap_sums,
        )
        num_negative_pixels += image_negative_pixels
        num_regions += image_regions
        del heatmap, mask

    metrics = dict(image_metrics)
    metrics["num_pixel_images"] = float(len(row_infos))
    if pixel_scores:
        labels = np.concatenate(pixel_labels, axis=0)
        scores = np.concatenate(pixel_scores, axis=0)
        metrics["num_pixels_evaluated"] = float(scores.shape[0])
        metrics["pixel_auroc"] = binary_roc_auc(labels, scores)
        metrics["aupro"] = _streaming_average_pro_score(
            thresholds=thresholds,
            false_positive_counts=false_positive_counts,
            region_overlap_sums=region_overlap_sums,
            num_negative_pixels=num_negative_pixels,
            num_regions=num_regions,
        )
        threshold_metrics = best_f1_iou(labels, scores)
        metrics["best_pixel_f1"] = threshold_metrics["best_f1"]
        metrics["best_pixel_iou"] = threshold_metrics["best_iou"]
        metrics["best_pixel_threshold"] = threshold_metrics["best_threshold"]
        metrics["oracle_best_pixel_f1"] = threshold_metrics["best_f1"]
        metrics["oracle_best_pixel_iou"] = threshold_metrics["best_iou"]
        metrics["oracle_best_pixel_threshold"] = threshold_metrics["best_threshold"]
        metrics["num_pixel_thresholds"] = threshold_metrics["num_thresholds"]
    else:
        metrics["pixel_auroc"] = math.nan
        metrics["aupro"] = math.nan
        metrics["best_pixel_f1"] = math.nan
        metrics["best_pixel_iou"] = math.nan
        metrics["best_pixel_threshold"] = math.nan
        metrics["oracle_best_pixel_f1"] = math.nan
        metrics["oracle_best_pixel_iou"] = math.nan
        metrics["oracle_best_pixel_threshold"] = math.nan
        metrics["num_pixel_thresholds"] = 0.0
        metrics["num_pixels_evaluated"] = 0.0
    return metrics


def evaluate_heatmap_rows_at_threshold(
    rows: list[dict[str, str]],
    threshold: float,
) -> tuple[dict[str, float], list[dict[str, str | float]]]:
    """Evaluate heatmaps as binary masks at one pre-calibrated threshold."""

    for row_index, row in enumerate(rows):
        context = f"row {row_index} sample_id={row.get('sample_id', '')!r}"
        label = row.get("label")
        if label not in {"0", "1"}:
            raise ValueError(f"{context} must have label exactly '0' or '1'; got {label!r}")
        if not (row.get("heatmap_path") or "").strip():
            raise ValueError(f"{context} must have a nonempty heatmap_path")
        if label == "1" and not (row.get("mask_path") or "").strip():
            raise ValueError(f"{context} must have mask_path for an anomalous row")

    per_image: list[dict[str, str | float]] = []
    for row in rows:
        heatmap_path = row["heatmap_path"]
        heatmap = load_heatmap(heatmap_path).astype(np.float32, copy=False)
        label = row["label"]
        mask_path = row.get("mask_path") or ""
        if label == "0":
            target = np.zeros(heatmap.shape, dtype=np.uint8)
        else:
            target = load_binary_mask(mask_path, size=(heatmap.shape[1], heatmap.shape[0]))
        if target.shape != heatmap.shape:
            target = _resize_mask(target, heatmap.shape)

        metrics = mask_confusion_metrics(heatmap >= threshold, target)
        per_image.append(
            {
                "dataset": row.get("dataset", ""),
                "sample_id": row.get("sample_id", ""),
                "category": row.get("category", ""),
                "label": row.get("label", ""),
                "fold_split": row.get("fold_split", ""),
                "threshold": float(threshold),
                "mask_precision": metrics["precision"],
                "mask_recall": metrics["recall"],
                "mask_f1": metrics["f1"],
                "mask_iou": metrics["iou"],
                "pred_positive_pixels": metrics["pred_positive_pixels"],
                "gt_positive_pixels": metrics["gt_positive_pixels"],
                "true_positive_pixels": metrics["true_positive_pixels"],
                "false_positive_pixels": metrics["false_positive_pixels"],
                "false_negative_pixels": metrics["false_negative_pixels"],
                "heatmap_path": heatmap_path,
                "mask_path": mask_path,
            }
        )
    return summarize_binary_metrics(per_image, prefix="calibrated"), per_image


def resolve_score_row_paths(
    rows: list[dict[str, str]],
    base_dir: str | Path,
) -> list[dict[str, str]]:
    """Resolve relative artifact paths against the score CSV directory when needed."""

    base_dir = Path(base_dir).resolve()
    project_root = Path(__file__).resolve().parents[2]
    resolved_rows = []
    for row in rows:
        resolved = dict(row)
        for key in ("image_path", "mask_path", "heatmap_path", "debug_path"):
            value = resolved.get(key) or ""
            if not value:
                continue
            path = Path(value)
            if path.is_absolute():
                continue
            csv_relative = base_dir / path
            legacy_repo_relative = project_root / path
            resolved_path = csv_relative
            if not csv_relative.exists() and legacy_repo_relative.exists():
                resolved_path = legacy_repo_relative
            resolved[key] = str(resolved_path)
        resolved_rows.append(resolved)
    return resolved_rows


def metrics_to_jsonable(
    metrics: dict[str, float | int | str],
) -> dict[str, float | int | str | None]:
    """Convert NaN values to null-friendly values for strict JSON consumers."""

    jsonable: dict[str, float | int | str | None] = {}
    for key, value in metrics.items():
        if isinstance(value, str):
            jsonable[key] = value
        elif isinstance(value, int):
            jsonable[key] = value
        elif key.startswith("num_"):
            jsonable[key] = int(value)
        elif isinstance(value, float) and math.isnan(value):
            jsonable[key] = None
        else:
            jsonable[key] = float(value)
    return jsonable


def _inspect_heatmap_rows(
    rows: list[dict[str, str]],
    max_thresholds: int,
) -> tuple[list[_HeatmapRowInfo], int, np.ndarray]:
    """Inspect map metadata and discover bounded AUPRO threshold candidates."""

    infos: list[_HeatmapRowInfo] = []
    total_pixels = 0
    unique_scores: set[float] | None = set()
    global_min = math.inf
    global_max = -math.inf
    for row in rows:
        heatmap_path = row.get("heatmap_path") or ""
        if not heatmap_path:
            continue
        heatmap = load_heatmap(heatmap_path, mmap_mode="r")
        if heatmap.ndim != 2 or heatmap.size == 0:
            raise ValueError(
                f"heatmap at {heatmap_path!r} must be a nonempty 2-D array; "
                f"got shape {heatmap.shape}"
            )
        current_min = float(np.min(heatmap))
        current_max = float(np.max(heatmap))
        if not math.isfinite(current_min) or not math.isfinite(current_max):
            raise ValueError(f"heatmap at {heatmap_path!r} must contain only finite values")
        global_min = min(global_min, current_min)
        global_max = max(global_max, current_max)
        if unique_scores is not None:
            flattened = heatmap.reshape(-1)
            chunk_size = max_thresholds + 1
            for start in range(0, flattened.size, chunk_size):
                unique_scores.update(
                    float(value) for value in np.unique(flattened[start : start + chunk_size])
                )
                if len(unique_scores) > max_thresholds:
                    unique_scores = None
                    break
        size = int(heatmap.size)
        infos.append(
            _HeatmapRowInfo(
                row=row,
                path=heatmap_path,
                offset=total_pixels,
                size=size,
            )
        )
        total_pixels += size

    if not infos:
        thresholds = np.asarray([], dtype=np.float64)
    elif unique_scores is not None:
        thresholds = np.asarray(sorted(unique_scores, reverse=True), dtype=np.float64)
    else:
        thresholds = np.linspace(global_max, global_min, num=max_thresholds)
    return infos, total_pixels, thresholds


def _sample_global_indices(
    total_pixels: int,
    max_pixels: int | None,
    seed: int,
) -> np.ndarray:
    if max_pixels is None or max_pixels <= 0 or total_pixels <= max_pixels:
        return np.arange(total_pixels, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(total_pixels, size=max_pixels, replace=False))


def _load_evaluation_target(row: dict[str, str], shape: tuple[int, int]) -> np.ndarray:
    mask_path = row.get("mask_path") or ""
    if mask_path:
        mask = load_binary_mask(mask_path, size=(shape[1], shape[0]))
    else:
        mask = np.zeros(shape, dtype=np.uint8)
    if mask.shape != shape:
        mask = _resize_mask(mask, shape)
    return mask


def _streaming_average_pro_score(
    thresholds: np.ndarray,
    false_positive_counts: np.ndarray,
    region_overlap_sums: np.ndarray,
    num_negative_pixels: int,
    num_regions: int,
    max_fpr: float = 0.3,
) -> float:
    if num_regions == 0 or num_negative_pixels == 0:
        return math.nan

    fpr_to_pro: dict[float, float] = {}
    for false_positives, overlap_sum in zip(false_positive_counts, region_overlap_sums):
        fpr = float(false_positives) / float(num_negative_pixels)
        pro = float(overlap_sum) / float(num_regions)
        fpr_to_pro[fpr] = max(pro, fpr_to_pro.get(fpr, 0.0))

    points = sorted(fpr_to_pro.items())
    if not points or points[0][0] > 0.0:
        points.insert(0, (0.0, 0.0))
    clipped = [(fpr, pro) for fpr, pro in points if fpr <= max_fpr]
    if not clipped or clipped[0][0] > 0.0:
        clipped.insert(0, (0.0, 0.0))
    if clipped[-1][0] < max_fpr:
        clipped.append((max_fpr, _interpolate_pro(points, max_fpr)))

    area = 0.0
    for (x1, y1), (x2, y2) in zip(clipped, clipped[1:]):
        area += (x2 - x1) * (y1 + y2) / 2.0
    return float(area / max_fpr)


def _accumulate_streaming_pro_image(
    heatmap: np.ndarray,
    mask: np.ndarray,
    thresholds: np.ndarray,
    false_positive_counts: np.ndarray,
    region_overlap_sums: np.ndarray,
) -> tuple[int, int]:
    """Accumulate PRO statistics while retaining only one image's arrays."""

    target = mask.astype(bool)
    negative_scores = np.sort(heatmap[~target])
    false_positive_counts += negative_scores.size - np.searchsorted(
        negative_scores, thresholds, side="left"
    )
    regions = _connected_components(target)
    for region in regions:
        region_scores = np.sort(heatmap[region])
        region_overlap_sums += (
            region_scores.size - np.searchsorted(region_scores, thresholds, side="left")
        ) / float(region_scores.size)
    return int(negative_scores.size), len(regions)


def sample_pixels(
    labels: np.ndarray,
    scores: np.ndarray,
    max_pixels: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if max_pixels is None or max_pixels <= 0 or scores.shape[0] <= max_pixels:
        return labels, scores
    rng = np.random.default_rng(seed)
    indices = rng.choice(scores.shape[0], size=max_pixels, replace=False)
    return labels[indices], scores[indices]


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.shape[0], dtype=np.float64)
    start = 0
    while start < values.shape[0]:
        end = start + 1
        while end < values.shape[0] and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def _threshold_candidates(scores: np.ndarray, max_thresholds: int) -> np.ndarray:
    unique = np.unique(scores)
    if unique.shape[0] <= max_thresholds:
        return unique[::-1]
    return np.linspace(float(scores.max()), float(scores.min()), num=max_thresholds)


def _connected_components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask).astype(bool)
    visited = np.zeros(mask.shape, dtype=bool)
    components = []
    height, width = mask.shape
    for y in range(height):
        for x in range(width):
            if visited[y, x] or not mask[y, x]:
                continue
            component = np.zeros(mask.shape, dtype=bool)
            stack = [(x, y)]
            visited[y, x] = True
            while stack:
                current_x, current_y = stack.pop()
                component[current_y, current_x] = True
                for next_x, next_y in (
                    (current_x - 1, current_y),
                    (current_x + 1, current_y),
                    (current_x, current_y - 1),
                    (current_x, current_y + 1),
                ):
                    if not (0 <= next_x < width and 0 <= next_y < height):
                        continue
                    if visited[next_y, next_x] or not mask[next_y, next_x]:
                        continue
                    visited[next_y, next_x] = True
                    stack.append((next_x, next_y))
            components.append(component)
    return components


def _interpolate_pro(points: list[tuple[float, float]], target_fpr: float) -> float:
    if not points:
        return 0.0
    previous_x, previous_y = points[0]
    if target_fpr <= previous_x:
        return previous_y
    for current_x, current_y in points[1:]:
        if current_x >= target_fpr:
            if current_x == previous_x:
                return max(previous_y, current_y)
            weight = (target_fpr - previous_x) / (current_x - previous_x)
            return float(previous_y + weight * (current_y - previous_y))
        previous_x, previous_y = current_x, current_y
    return previous_y


def _resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    image = Image.fromarray((mask > 0).astype(np.uint8) * 255, mode="L")
    image = image.resize((width, height), Image.Resampling.NEAREST)
    return (np.asarray(image) > 0).astype(np.uint8)
