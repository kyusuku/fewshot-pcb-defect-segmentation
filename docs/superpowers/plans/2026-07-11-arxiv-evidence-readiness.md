# ArXiv Evidence Readiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and execute the reproducible evaluation, selective-SAM2, baseline, experiment, statistics, and evidence pipeline required to begin writing an arXiv-style report.

**Architecture:** Keep feature extraction, anomaly scoring, SAM2 refinement, evaluation, experiment orchestration, and evidence generation as separate modules. Every run is described by an immutable `RunSpec`, writes provenance before execution, and is considered usable only when the matrix validator confirms its declared artifacts. Heavy primary runs execute on AutoDL; local execution covers unit tests, synthetic smokes, result aggregation, documentation, and notebook work.

**Tech Stack:** Python 3.11, NumPy, Pillow, PyTorch, torchvision, PyYAML, SAM2, pytest, Ruff, CSV/JSON/Markdown artifacts.

---

## File Map

### Evaluation

- Create `src/evaluation/calibration.py`: normal-only threshold fitting and serialization.
- Create `src/evaluation/statistics.py`: deterministic bootstrap intervals and paired deltas.
- Create `src/evaluation/failure_analysis.py`: per-image geometry and SAM2-effect descriptors.
- Modify `src/evaluation/metrics.py`: explicit oracle aliases and calibrated heatmap masks.
- Modify `src/evaluation/masks.py`: shared aggregate/per-image binary-mask summaries.
- Modify `src/evaluation/stage4.py`: compare only like-for-like metric definitions.
- Create `scripts/calibrate_heatmaps.py`: fit the pre-registered 99.5% normal threshold.
- Modify `scripts/evaluate_heatmaps.py`: emit oracle and calibrated metrics plus per-image CSV.
- Create `scripts/analyze_paper_results.py`: confidence intervals, paired analysis, and strata.

### SAM2 refinement

- Modify `src/sam_refine/prompts.py`: local anomaly-maximum prompts and explicit point mode.
- Modify `src/sam_refine/refiner.py`: point-only, box-only, and combined predictor calls.
- Create `src/sam_refine/fusion.py`: anomaly/SAM2 intersection, union, and agreement features.
- Modify `scripts/run_mask_refinement.py`: calibration input, prompt mode, and fusion output.

### Baselines and compute efficiency

- Create `src/features/patchcore.py`: frozen Wide-ResNet50 PatchCore-style features.
- Modify `src/features/dinov2.py`: dispatch the PatchCore extractor through the common protocol.
- Create `src/features/cache.py`: deterministic `PatchFeatureMap` cache keys and storage.
- Modify `src/anomaly/multiscale.py`: use optional cached global/crop feature maps.
- Modify `scripts/run_dinov2_baseline.py`: bounded debug rendering, provenance, and cache options.
- Create `scripts/run_patchcore_baseline.py`: same support/query protocol with PatchCore defaults.

### Experiment orchestration

- Create `src/experiments/__init__.py`.
- Create `src/experiments/spec.py`: immutable run specifications and matrix expansion.
- Create `src/experiments/provenance.py`: checksums, git/environment metadata, support IDs.
- Create `src/experiments/runner.py`: dependency-aware commands and resumable status records.
- Create `configs/experiments/arxiv_primary.yaml`: frozen `k`, seeds, categories, and methods.
- Create `configs/experiments/arxiv_ablations.yaml`: frozen crop, prompt, area-cap, and fusion variants.
- Create `configs/experiments/arxiv_smoke.yaml`: tiny color-patch/fallback matrix.
- Create `scripts/run_experiment_matrix.py`: list, dry-run, filter, resume, and execute runs.
- Create `scripts/check_experiment_matrix.py`: detect missing, duplicate, stale, and failed runs.

### Paper evidence

- Create `src/evaluation/paper_tables.py`: seed/category aggregation with provenance.
- Create `scripts/build_paper_evidence.py`: tables, evidence index, and completion manifest.
- Create `scripts/curate_paper_assets.py`: deterministic success/failure selection.
- Create `scripts/render_method_figure.py`: reproducible method diagram using Pillow.
- Modify `notebooks/pcb_defect_pipeline.ipynb`: present the frozen method and final evidence.
- Create `docs/evidence/README.md`: tracked evidence policy and generated-summary index.
- Create `docs/arxiv_readiness_checklist.md`: requirement-by-requirement completion audit.
- Modify `README.md`: exact local, smoke, AutoDL, matrix, and evidence commands.
- Modify `pyproject.toml`: package `experiments*` and declare any new dev-only dependency.

## Task 1: Normal-Only Threshold Calibration

**Files:**
- Create: `src/evaluation/calibration.py`
- Create: `tests/test_calibration.py`

- [ ] **Step 1: Write the failing calibration tests**

```python
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluation.calibration import NormalThreshold, fit_normal_threshold


class NormalThresholdTest(unittest.TestCase):
    def test_fit_uses_only_normal_validation_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            first = root / "first.npy"
            second = root / "second.npy"
            np.save(first, np.asarray([[0.0, 1.0]], dtype=np.float32))
            np.save(second, np.asarray([[2.0, 3.0]], dtype=np.float32))
            rows = [
                {"label": "0", "fold_split": "val", "heatmap_path": str(first)},
                {"label": "0", "fold_split": "val", "heatmap_path": str(second)},
            ]

            result = fit_normal_threshold(rows, quantile=0.75)

        self.assertEqual(result.source_split, "val")
        self.assertEqual(result.num_images, 2)
        self.assertEqual(result.num_pixels, 4)
        self.assertAlmostEqual(result.threshold, 2.25)

    def test_fit_rejects_anomalous_or_non_validation_rows(self) -> None:
        with self.assertRaisesRegex(ValueError, "normal validation"):
            fit_normal_threshold(
                [{"label": "1", "fold_split": "test", "heatmap_path": "x.npy"}],
                quantile=0.995,
            )

        with self.assertRaisesRegex(ValueError, "normal validation"):
            fit_normal_threshold(
                [{"label": "0", "heatmap_path": "x.npy"}],
                quantile=0.995,
            )

    def test_round_trip_json_preserves_provenance(self) -> None:
        threshold = NormalThreshold(
            quantile=0.995,
            threshold=0.42,
            num_images=10,
            num_pixels=1000,
            source_split="val",
        )
        restored = NormalThreshold.from_dict(json.loads(json.dumps(threshold.to_dict())))
        self.assertEqual(restored, threshold)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests and confirm the missing module failure**

Run: `PYTHONPATH=src venv/bin/python -m pytest tests/test_calibration.py -q`

Expected: FAIL during collection with `ModuleNotFoundError: No module named 'evaluation.calibration'`.

- [ ] **Step 3: Implement the calibration module**

```python
"""Normal-only calibration for operational anomaly masks."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class NormalThreshold:
    quantile: float
    threshold: float
    num_images: int
    num_pixels: int
    source_split: str = "val"

    def to_dict(self) -> dict[str, float | int | str]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "NormalThreshold":
        return cls(
            quantile=float(payload["quantile"]),
            threshold=float(payload["threshold"]),
            num_images=int(payload["num_images"]),
            num_pixels=int(payload["num_pixels"]),
            source_split=str(payload["source_split"]),
        )


def fit_normal_threshold(
    rows: list[dict[str, str]],
    quantile: float = 0.995,
) -> NormalThreshold:
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be in (0, 1)")
    if not rows or any(
        int(row.get("label", "1")) != 0 or row.get("fold_split") != "val"
        for row in rows
    ):
        raise ValueError("calibration requires only normal validation rows")
    heatmaps = [
        np.load(row["heatmap_path"]).astype(np.float32, copy=False).ravel()
        for row in rows
    ]
    pixels = np.concatenate(heatmaps)
    return NormalThreshold(
        quantile=quantile,
        threshold=float(np.quantile(pixels, quantile)),
        num_images=len(rows),
        num_pixels=int(pixels.size),
    )
```

- [ ] **Step 4: Run the calibration tests**

Run: `PYTHONPATH=src venv/bin/python -m pytest tests/test_calibration.py -q`

Expected: `4 passed`.

- [ ] **Step 5: Commit calibration**

```bash
git add src/evaluation/calibration.py tests/test_calibration.py
git commit -m "feat: add normal-only heatmap calibration"
```

## Task 2: Comparable Oracle and Calibrated Heatmap Metrics

**Files:**
- Modify: `src/evaluation/metrics.py:155-206`
- Modify: `src/evaluation/masks.py:29-124`
- Modify: `scripts/evaluate_heatmaps.py:24-55`
- Create: `scripts/calibrate_heatmaps.py`
- Modify: `tests/test_evaluation_metrics.py`
- Create: `tests/test_calibrated_evaluation.py`

- [ ] **Step 1: Add failing tests for like-for-like heatmap masks**

```python
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from evaluation.metrics import evaluate_heatmap_rows_at_threshold


class CalibratedHeatmapEvaluationTest(unittest.TestCase):
    def test_returns_aggregate_and_per_image_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            heatmap_path = root / "heatmap.npy"
            mask_path = root / "mask.png"
            np.save(heatmap_path, np.asarray([[0.1, 0.9], [0.2, 0.8]], dtype=np.float32))
            Image.fromarray(np.asarray([[0, 255], [0, 255]], dtype=np.uint8)).save(mask_path)
            rows = [{
                "sample_id": "pcb1/a",
                "category": "pcb1",
                "label": "1",
                "heatmap_path": str(heatmap_path),
                "mask_path": str(mask_path),
            }]

            summary, per_image = evaluate_heatmap_rows_at_threshold(rows, threshold=0.5)

        self.assertEqual(summary["calibrated_aggregate_pixel_f1"], 1.0)
        self.assertEqual(summary["calibrated_mean_anomaly_mask_f1"], 1.0)
        self.assertEqual(per_image[0]["mask_f1"], 1.0)
        self.assertEqual(per_image[0]["threshold"], 0.5)
```

Also extend `tests/test_evaluation_metrics.py`:

```python
def test_heatmap_summary_uses_explicit_oracle_names(self) -> None:
    metrics = evaluate_heatmap_rows(self.rows, max_pixels=0, seed=4880)
    self.assertEqual(metrics["oracle_best_pixel_f1"], metrics["best_pixel_f1"])
    self.assertEqual(metrics["oracle_best_pixel_iou"], metrics["best_pixel_iou"])
    self.assertEqual(metrics["oracle_best_pixel_threshold"], metrics["best_pixel_threshold"])
```

- [ ] **Step 2: Verify the new API is missing**

Run: `PYTHONPATH=src venv/bin/python -m pytest tests/test_evaluation_metrics.py tests/test_calibrated_evaluation.py -q`

Expected: FAIL importing `evaluate_heatmap_rows_at_threshold`.

- [ ] **Step 3: Add explicit oracle aliases to `evaluate_heatmap_rows`**

Immediately after the existing best-threshold fields in `src/evaluation/metrics.py`, add:

```python
metrics["oracle_best_pixel_f1"] = threshold_metrics["best_f1"]
metrics["oracle_best_pixel_iou"] = threshold_metrics["best_iou"]
metrics["oracle_best_pixel_threshold"] = threshold_metrics["best_threshold"]
```

In the no-pixel branch, set the same three keys to `math.nan`.

- [ ] **Step 4: Implement thresholded heatmap evaluation using the mask metric definition**

Add to `src/evaluation/metrics.py`:

```python
from evaluation.masks import mask_confusion_metrics, summarize_binary_metrics


def evaluate_heatmap_rows_at_threshold(
    rows: list[dict[str, str]],
    threshold: float,
) -> tuple[dict[str, float], list[dict[str, str | float]]]:
    per_image: list[dict[str, str | float]] = []
    for row in rows:
        heatmap = np.load(row["heatmap_path"]).astype(np.float32, copy=False)
        mask_path = row.get("mask_path") or ""
        target = (
            load_binary_mask(mask_path, size=(heatmap.shape[1], heatmap.shape[0]))
            if mask_path
            else np.zeros(heatmap.shape, dtype=np.uint8)
        )
        metrics = mask_confusion_metrics(heatmap >= threshold, target)
        per_image.append({
            "sample_id": row.get("sample_id", ""),
            "category": row.get("category", ""),
            "label": row.get("label", ""),
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
            "heatmap_path": row["heatmap_path"],
            "mask_path": mask_path,
        })
    summary = summarize_binary_metrics(per_image, prefix="calibrated")
    return summary, per_image
```

First extend both return branches of `mask_confusion_metrics` in
`src/evaluation/masks.py` to return the exact counts (all three are `0.0` for
the empty/empty case):

```python
"true_positive_pixels": tp,
"false_positive_pixels": fp,
"false_negative_pixels": fn,
```

Then add:

```python
def summarize_binary_metrics(
    rows: list[dict[str, str | float]],
    prefix: str,
) -> dict[str, float]:
    anomaly = [row for row in rows if int(row["label"]) == 1]
    aggregate_tp = sum(float(row["true_positive_pixels"]) for row in rows)
    aggregate_fp = sum(float(row["false_positive_pixels"]) for row in rows)
    aggregate_fn = sum(float(row["false_negative_pixels"]) for row in rows)
    aggregate_precision = (
        aggregate_tp / (aggregate_tp + aggregate_fp)
        if aggregate_tp + aggregate_fp > 0.0 else 0.0
    )
    aggregate_recall = (
        aggregate_tp / (aggregate_tp + aggregate_fn)
        if aggregate_tp + aggregate_fn > 0.0 else 0.0
    )
    aggregate_f1 = (
        2.0 * aggregate_precision * aggregate_recall
        / (aggregate_precision + aggregate_recall)
        if aggregate_precision + aggregate_recall > 0.0 else 0.0
    )
    aggregate_iou = (
        aggregate_tp / (aggregate_tp + aggregate_fp + aggregate_fn)
        if aggregate_tp + aggregate_fp + aggregate_fn > 0.0 else 0.0
    )
    return {
        f"{prefix}_aggregate_pixel_precision": float(aggregate_precision),
        f"{prefix}_aggregate_pixel_recall": float(aggregate_recall),
        f"{prefix}_aggregate_pixel_f1": float(aggregate_f1),
        f"{prefix}_aggregate_pixel_iou": float(aggregate_iou),
        f"{prefix}_mean_mask_f1": _mean_rows(rows, "mask_f1"),
        f"{prefix}_mean_mask_iou": _mean_rows(rows, "mask_iou"),
        f"{prefix}_mean_anomaly_mask_f1": _mean_rows(anomaly, "mask_f1"),
        f"{prefix}_mean_anomaly_mask_iou": _mean_rows(anomaly, "mask_iou"),
    }


def _mean_rows(rows: list[dict[str, str | float]], key: str) -> float:
    return float(sum(float(row[key]) for row in rows) / len(rows)) if rows else math.nan
```

- [ ] **Step 5: Add the calibration CLI**

Create `scripts/calibrate_heatmaps.py`:

```python
#!/usr/bin/env python
"""Fit a normal-only pixel threshold from validation heatmaps."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from evaluation.calibration import fit_normal_threshold
from evaluation.metrics import resolve_score_row_paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores-csv", type=Path, required=True)
    parser.add_argument("--quantile", type=float, default=0.995)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    with args.scores_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows = resolve_score_row_paths(rows, args.scores_csv.parent)
    result = fit_normal_threshold(rows, quantile=args.quantile)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result.to_dict(), indent=2) + "\n")
    print(f"threshold: {result.threshold:.8f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Extend the heatmap evaluation CLI**

Add arguments to `scripts/evaluate_heatmaps.py`:

```python
parser.add_argument("--calibration-json", type=Path)
parser.add_argument("--per-image-csv", type=Path)
```

After the existing oracle metrics are computed, add:

```python
if args.calibration_json:
    calibration = NormalThreshold.from_dict(json.loads(args.calibration_json.read_text()))
    calibrated, per_image = evaluate_heatmap_rows_at_threshold(rows, calibration.threshold)
    metrics.update(calibrated)
    metrics["calibration_quantile"] = calibration.quantile
    metrics["calibration_threshold"] = calibration.threshold
    if args.per_image_csv:
        write_csv(per_image, args.per_image_csv)
```

Import `NormalThreshold`, `evaluate_heatmap_rows_at_threshold`, and reuse a
small `write_csv` helper that creates the parent directory and writes keys from
the first row.

Modify `run_dinov2_baseline.py` so every score row contains
`"fold_split": args.query_fold_split`, and add `fold_split` to
`write_scores_csv` fieldnames. The calibration CLI must fail if this field is
missing or is not `val`; it must never infer validation provenance from the
filename or command context.

- [ ] **Step 7: Run focused and full tests**

Run: `PYTHONPATH=src venv/bin/python -m pytest tests/test_calibration.py tests/test_evaluation_metrics.py tests/test_calibrated_evaluation.py tests/test_mask_evaluation.py -q`

Expected: all focused tests pass.

Run: `PYTHONPATH=src venv/bin/python -m pytest -q`

Expected: the original 56 tests plus the new tests pass.

- [ ] **Step 8: Commit comparable evaluation**

```bash
git add src/evaluation/metrics.py src/evaluation/masks.py scripts/evaluate_heatmaps.py scripts/calibrate_heatmaps.py tests/test_evaluation_metrics.py tests/test_calibrated_evaluation.py
git commit -m "feat: compare calibrated heatmap masks fairly"
```

## Task 3: Deterministic Confidence Intervals and Paired Deltas

**Files:**
- Create: `src/evaluation/statistics.py`
- Create: `tests/test_statistics.py`

- [ ] **Step 1: Write deterministic bootstrap tests**

```python
import unittest

from evaluation.statistics import bootstrap_mean_ci, paired_bootstrap_delta


class BootstrapStatisticsTest(unittest.TestCase):
    def test_constant_values_have_zero_width_interval(self) -> None:
        result = bootstrap_mean_ci([0.5, 0.5, 0.5], samples=200, seed=4880)
        self.assertEqual(result, {"mean": 0.5, "ci_low": 0.5, "ci_high": 0.5})

    def test_paired_delta_preserves_pairing(self) -> None:
        result = paired_bootstrap_delta(
            baseline=[0.1, 0.3, 0.5],
            candidate=[0.2, 0.4, 0.6],
            samples=200,
            seed=4880,
        )
        self.assertAlmostEqual(result["mean_delta"], 0.1)
        self.assertGreater(result["ci_low"], 0.09)
        self.assertLess(result["ci_high"], 0.11)

    def test_paired_delta_rejects_mismatched_lengths(self) -> None:
        with self.assertRaisesRegex(ValueError, "same length"):
            paired_bootstrap_delta([0.1], [0.1, 0.2])
```

- [ ] **Step 2: Confirm the module is missing**

Run: `PYTHONPATH=src venv/bin/python -m pytest tests/test_statistics.py -q`

Expected: FAIL importing `evaluation.statistics`.

- [ ] **Step 3: Implement statistics without SciPy**

```python
"""Small deterministic bootstrap utilities for paper tables."""

from __future__ import annotations

import numpy as np


def bootstrap_mean_ci(
    values: list[float],
    confidence: float = 0.95,
    samples: int = 2000,
    seed: int = 4880,
) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("values must not be empty")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(samples, array.size))
    means = array[indices].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    return {
        "mean": float(array.mean()),
        "ci_low": float(np.quantile(means, tail)),
        "ci_high": float(np.quantile(means, 1.0 - tail)),
    }


def paired_bootstrap_delta(
    baseline: list[float],
    candidate: list[float],
    confidence: float = 0.95,
    samples: int = 2000,
    seed: int = 4880,
) -> dict[str, float]:
    if len(baseline) != len(candidate):
        raise ValueError("baseline and candidate must have the same length")
    delta = np.asarray(candidate, dtype=np.float64) - np.asarray(baseline, dtype=np.float64)
    result = bootstrap_mean_ci(delta.tolist(), confidence=confidence, samples=samples, seed=seed)
    return {
        "mean_delta": result["mean"],
        "ci_low": result["ci_low"],
        "ci_high": result["ci_high"],
    }
```

- [ ] **Step 4: Verify and commit statistics**

Run: `PYTHONPATH=src venv/bin/python -m pytest tests/test_statistics.py -q`

Expected: `3 passed`.

```bash
git add src/evaluation/statistics.py tests/test_statistics.py
git commit -m "feat: add deterministic result statistics"
```

## Task 4: Anomaly-Maximum Prompt Variants

**Files:**
- Modify: `src/sam_refine/prompts.py:12-67`
- Modify: `src/sam_refine/refiner.py:56-126`
- Modify: `tests/test_sam_refine.py`

- [ ] **Step 1: Add failing prompt-location and predictor-call tests**

Add to `tests/test_sam_refine.py`:

```python
def test_prompt_point_uses_local_anomaly_maximum(self) -> None:
    heatmap = np.zeros((6, 6), dtype=np.float32)
    heatmap[1:4, 1:4] = 0.6
    heatmap[3, 2] = 1.0
    region = heatmap_to_prompt_regions(
        heatmap,
        threshold=0.5,
        min_area=1,
        max_regions=1,
        point_mode="anomaly_max",
    )[0]
    self.assertEqual(region.point_xy, (2.0, 3.0))


def test_sam2_point_only_omits_box(self) -> None:
    refiner = SAM2MaskRefiner(
        checkpoint_path="weights/fake.pt",
        model_config="configs/fake.yaml",
        prompt_mode="point",
    )
    predictor = _FakeSAM2Predictor()
    refiner._predictor = predictor
    refiner.refine(self.image, self.heatmap, [self.region])
    self.assertIsNone(predictor.calls[0]["box"])
    self.assertIsNotNone(predictor.calls[0]["point_coords"])


def test_sam2_box_only_omits_point(self) -> None:
    refiner = SAM2MaskRefiner(
        checkpoint_path="weights/fake.pt",
        model_config="configs/fake.yaml",
        prompt_mode="box",
    )
    predictor = _FakeSAM2Predictor()
    refiner._predictor = predictor
    refiner.refine(self.image, self.heatmap, [self.region])
    self.assertIsNotNone(predictor.calls[0]["box"])
    self.assertIsNone(predictor.calls[0]["point_coords"])
```

- [ ] **Step 2: Run tests and confirm signature failures**

Run: `PYTHONPATH=src venv/bin/python -m pytest tests/test_sam_refine.py -q`

Expected: FAIL because `point_mode` and `prompt_mode` are not accepted.

- [ ] **Step 3: Implement anomaly-maximum points**

Change the prompt function signature:

```python
def heatmap_to_prompt_regions(
    heatmap: np.ndarray,
    threshold: float | None = None,
    percentile: float = 95.0,
    min_area: int = 8,
    max_regions: int = 8,
    point_mode: str = "anomaly_max",
) -> list[PromptRegion]:
```

Validate `point_mode in {"anomaly_max", "box_center"}`. Replace the existing
`point_xy` assignment with:

```python
if point_mode == "anomaly_max":
    maximum_index = int(np.argmax(local_scores))
    point_xy = (float(xs[maximum_index]), float(ys[maximum_index]))
else:
    point_xy = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
```

- [ ] **Step 4: Implement predictor prompt modes**

Add `prompt_mode: str = "point_box"` to `SAM2MaskRefiner.__init__`, validate it
against `{"point", "box", "point_box"}`, and store it. Replace the predictor
call arguments with:

```python
use_point = self.prompt_mode in {"point", "point_box"}
use_box = self.prompt_mode in {"box", "point_box"}
masks, scores, _ = predictor.predict(
    point_coords=point_coords if use_point else None,
    point_labels=np.asarray([1], dtype=np.int32) if use_point else None,
    box=box if use_box else None,
    multimask_output=self.multimask_output,
)
```

- [ ] **Step 5: Run and commit prompt variants**

Run: `PYTHONPATH=src venv/bin/python -m pytest tests/test_sam_refine.py -q`

Expected: all SAM refinement tests pass.

```bash
git add src/sam_refine/prompts.py src/sam_refine/refiner.py tests/test_sam_refine.py
git commit -m "feat: add anomaly-aware sam2 prompt variants"
```

## Task 5: Anomaly-Consistency Mask Fusion

**Files:**
- Create: `src/sam_refine/fusion.py`
- Modify: `scripts/run_mask_refinement.py:26-198`
- Create: `scripts/fuse_saved_masks.py`
- Create: `tests/test_mask_fusion.py`
- Modify: `tests/test_sam_refine.py`

- [ ] **Step 1: Write failing fusion tests**

```python
import unittest

import numpy as np

from sam_refine.fusion import agreement_features, fuse_masks


class MaskFusionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.anomaly = np.asarray([[1, 1], [0, 0]], dtype=np.uint8)
        self.sam = np.asarray([[0, 1], [0, 1]], dtype=np.uint8)

    def test_intersection_is_primary_anomaly_consistent_output(self) -> None:
        np.testing.assert_array_equal(
            fuse_masks(self.anomaly, self.sam, "intersection"),
            np.asarray([[0, 1], [0, 0]], dtype=np.uint8),
        )

    def test_union_and_source_modes_are_explicit(self) -> None:
        self.assertEqual(int(fuse_masks(self.anomaly, self.sam, "union").sum()), 3)
        np.testing.assert_array_equal(fuse_masks(self.anomaly, self.sam, "anomaly"), self.anomaly)
        np.testing.assert_array_equal(fuse_masks(self.anomaly, self.sam, "sam2"), self.sam)

    def test_selective_fallback_keeps_anomaly_when_agreement_is_low(self) -> None:
        disjoint = np.asarray([[0, 0], [1, 1]], dtype=np.uint8)
        np.testing.assert_array_equal(
            fuse_masks(self.anomaly, disjoint, "selective"),
            self.anomaly,
        )

    def test_agreement_features_report_expansion_and_iou(self) -> None:
        features = agreement_features(self.anomaly, self.sam)
        self.assertEqual(features["intersection_pixels"], 1.0)
        self.assertAlmostEqual(features["mask_iou"], 1.0 / 3.0)
        self.assertEqual(features["sam2_to_anomaly_area_ratio"], 1.0)
```

- [ ] **Step 2: Confirm the fusion module is missing**

Run: `PYTHONPATH=src venv/bin/python -m pytest tests/test_mask_fusion.py -q`

Expected: FAIL importing `sam_refine.fusion`.

- [ ] **Step 3: Implement fusion and agreement features**

```python
"""Binary fusion between anomaly proposals and SAM2 masks."""

from __future__ import annotations

import numpy as np


def fuse_masks(
    anomaly_mask: np.ndarray,
    sam2_mask: np.ndarray,
    mode: str,
    min_iou: float = 0.25,
    max_expansion: float = 2.0,
) -> np.ndarray:
    anomaly = np.asarray(anomaly_mask).astype(bool)
    sam2 = np.asarray(sam2_mask).astype(bool)
    if anomaly.shape != sam2.shape:
        raise ValueError("anomaly and SAM2 masks must have the same shape")
    features = agreement_features(anomaly, sam2)
    selective = (
        anomaly & sam2
        if features["mask_iou"] >= min_iou
        and features["sam2_to_anomaly_area_ratio"] <= max_expansion
        else anomaly
    )
    operations = {
        "anomaly": anomaly,
        "sam2": sam2,
        "intersection": anomaly & sam2,
        "union": anomaly | sam2,
        "selective": selective,
    }
    if mode not in operations:
        raise ValueError(f"unknown fusion mode: {mode}")
    return operations[mode].astype(np.uint8)


def agreement_features(anomaly_mask: np.ndarray, sam2_mask: np.ndarray) -> dict[str, float]:
    anomaly = np.asarray(anomaly_mask).astype(bool)
    sam2 = np.asarray(sam2_mask).astype(bool)
    intersection = float(np.sum(anomaly & sam2))
    union = float(np.sum(anomaly | sam2))
    anomaly_area = float(np.sum(anomaly))
    sam2_area = float(np.sum(sam2))
    return {
        "anomaly_pixels": anomaly_area,
        "sam2_pixels": sam2_area,
        "intersection_pixels": intersection,
        "mask_iou": intersection / union if union else 1.0,
        "sam2_to_anomaly_area_ratio": sam2_area / anomaly_area if anomaly_area else float("inf"),
    }
```

- [ ] **Step 4: Extend `run_mask_refinement.py` with calibrated fusion**

Add arguments:

```python
parser.add_argument("--calibration-json", type=Path)
parser.add_argument(
    "--mask-output",
    choices=("sam2", "anomaly", "intersection", "union", "selective"),
    default="sam2",
)
parser.add_argument(
    "--point-mode",
    choices=("anomaly_max", "box_center"),
    default="anomaly_max",
)
parser.add_argument(
    "--prompt-mode",
    choices=("point", "box", "point_box"),
    default="point_box",
)
```

Load `NormalThreshold` once. For each heatmap, choose the prompt/fusion threshold
from calibration when provided, otherwise preserve the existing explicit
threshold/percentile behavior. Replace the final mask construction with:

```python
sam2_mask = union_masks([prediction.mask for prediction in predictions], shape=heatmap.shape)
proposal_threshold = calibration.threshold if calibration else args.threshold
if proposal_threshold is None:
    proposal_threshold = float(np.percentile(heatmap, args.percentile))
anomaly_mask = (heatmap >= proposal_threshold).astype(np.uint8)
pred_mask = fuse_masks(anomaly_mask, sam2_mask, mode=args.mask_output)
agreement = agreement_features(anomaly_mask, sam2_mask)
```

Always save the raw union mask as `*_sam2_mask.png`, add `sam2_mask_path`,
`mask_output`, `proposal_threshold`, `mask_iou`, and
`sam2_to_anomaly_area_ratio` to `mask_scores.csv`, and save the chosen fused
mask separately as `pred_mask_path`. Pass `point_mode` into
`heatmap_to_prompt_regions` and `prompt_mode` into `SAM2MaskRefiner`.

Create `scripts/fuse_saved_masks.py` with arguments `--mask-scores-csv`,
`--calibration-json`, `--mask-output`, and `--output-dir`. It loads each
`sam2_mask_path` and `heatmap_path`, recreates the calibrated anomaly mask,
calls `fuse_masks`, saves the new `pred_mask_path`, and writes a compatible
`mask_scores.csv`. It performs no SAM2 inference. This is the required path for
the anomaly-consistent and union ablations.

The selective ablation is frozen to `min_iou=0.25` and `max_expansion=2.0`:
when agreement passes both rules it returns the intersection; otherwise it
falls back to the calibrated anomaly mask. These thresholds are not retuned
from target test masks.

- [ ] **Step 5: Add a script integration test for intersection output**

Extend the existing temporary heatmap test in `tests/test_sam_refine.py` to run:

```python
"--threshold", "0.5",
"--mask-output", "intersection",
"--refiner", "fallback",
```

Assert that the saved mask is the intersection and that the new CSV fields are
present with `mask_output == "intersection"`.

- [ ] **Step 6: Run and commit fusion**

Run: `PYTHONPATH=src venv/bin/python -m pytest tests/test_mask_fusion.py tests/test_sam_refine.py tests/test_mask_evaluation.py -q`

Expected: all focused tests pass.

```bash
git add src/sam_refine/fusion.py scripts/run_mask_refinement.py scripts/fuse_saved_masks.py tests/test_mask_fusion.py tests/test_sam_refine.py
git commit -m "feat: add anomaly-consistent sam2 fusion"
```

## Task 6: PatchCore Baseline Through the Common Feature Interface

**Files:**
- Create: `src/features/patchcore.py`
- Modify: `src/features/dinov2.py:96-116`
- Modify: `src/anomaly/memory_bank.py`
- Create: `scripts/run_patchcore_baseline.py`
- Create: `tests/test_patchcore_features.py`

- [ ] **Step 1: Write a network-free PatchCore extractor test**

```python
from __future__ import annotations

import unittest

import numpy as np
from PIL import Image

from anomaly.memory_bank import select_greedy_coreset
from features.patchcore import PatchCoreFeatureExtractor


class _FakeBackbone:
    def __call__(self, tensor):
        import torch

        batch = tensor.shape[0]
        return {
            "layer2": torch.ones((batch, 4, 8, 8), device=tensor.device),
            "layer3": torch.full((batch, 6, 4, 4), 2.0, device=tensor.device),
        }


class PatchCoreFeatureTest(unittest.TestCase):
    def test_combines_layer2_and_upsampled_layer3(self) -> None:
        extractor = PatchCoreFeatureExtractor(
            image_size=64,
            device="cpu",
            backbone=_FakeBackbone(),
        )
        result = extractor.extract(Image.new("RGB", (48, 32), (20, 40, 60)))
        self.assertEqual(result.features.shape, (8, 8, 10))
        self.assertEqual(result.image_size, (64, 64))
        self.assertTrue(np.isfinite(result.features).all())

    def test_greedy_coreset_is_deterministic_subset(self) -> None:
        bank = np.arange(24, dtype=np.float32).reshape(8, 3)
        first = select_greedy_coreset(bank, ratio=0.25, seed=4880)
        second = select_greedy_coreset(bank, ratio=0.25, seed=4880)
        self.assertEqual(first.shape, (2, 3))
        np.testing.assert_array_equal(first, second)
```

- [ ] **Step 2: Confirm the PatchCore module is missing**

Run: `PYTHONPATH=src venv/bin/python -m pytest tests/test_patchcore_features.py -q`

Expected: FAIL importing `features.patchcore`.

- [ ] **Step 3: Implement the PatchCore-style feature extractor**

Create `src/features/patchcore.py`:

```python
"""Frozen Wide-ResNet features for a PatchCore-style memory-bank baseline."""

from __future__ import annotations

import numpy as np
from PIL import Image

from features.dinov2 import PatchFeatureMap, _image_to_normalized_tensor, _resolve_device


class PatchCoreFeatureExtractor:
    def __init__(
        self,
        image_size: int = 512,
        device: str = "auto",
        backbone=None,
    ) -> None:
        import torch

        self.image_size = image_size
        self.patch_size = 8
        self.device = _resolve_device(device, torch)
        if backbone is None:
            from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2
            from torchvision.models.feature_extraction import create_feature_extractor

            model = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.DEFAULT)
            backbone = create_feature_extractor(
                model,
                return_nodes={"layer2": "layer2", "layer3": "layer3"},
            )
        self.backbone = backbone.to(self.device) if hasattr(backbone, "to") else backbone
        if hasattr(self.backbone, "eval"):
            self.backbone.eval()

    def extract(self, image: Image.Image) -> PatchFeatureMap:
        import torch
        import torch.nn.functional as functional

        prepared = image.convert("RGB").resize(
            (self.image_size, self.image_size), Image.Resampling.BICUBIC
        )
        tensor = _image_to_normalized_tensor(prepared, torch).to(self.device)
        with torch.no_grad():
            output = self.backbone(tensor)
        layer2 = output["layer2"]
        layer3 = functional.interpolate(
            output["layer3"], size=layer2.shape[-2:], mode="bilinear", align_corners=False
        )
        features = torch.cat([layer2, layer3], dim=1)
        features = functional.avg_pool2d(features, kernel_size=3, stride=1, padding=1)
        array = features[0].permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)
        return PatchFeatureMap(
            features=array,
            image_size=prepared.size,
            patch_size=self.patch_size,
        )
```

- [ ] **Step 4: Dispatch PatchCore without changing DINO defaults**

Add this deterministic approximate-greedy coreset to
`src/anomaly/memory_bank.py`:

```python
def select_greedy_coreset(
    memory_bank: np.ndarray,
    ratio: float = 0.01,
    seed: int = 4880,
    projection_dim: int = 64,
) -> np.ndarray:
    if not 0.0 < ratio <= 1.0:
        raise ValueError("ratio must be in (0, 1]")
    bank = np.asarray(memory_bank, dtype=np.float32)
    target = max(1, int(round(bank.shape[0] * ratio)))
    if target >= bank.shape[0]:
        return bank.copy()
    rng = np.random.default_rng(seed)
    projection = rng.standard_normal((bank.shape[1], projection_dim)).astype(np.float32)
    projected = bank @ projection / np.sqrt(float(projection_dim))
    selected = [int(rng.integers(0, bank.shape[0]))]
    minimum = np.full(bank.shape[0], np.inf, dtype=np.float32)
    while len(selected) < target:
        center = projected[selected[-1]]
        squared = np.sum((projected - center) ** 2, axis=1)
        minimum = np.minimum(minimum, squared)
        minimum[selected] = -1.0
        selected.append(int(np.argmax(minimum)))
    return bank[np.asarray(selected, dtype=np.int64)]
```

Expose `--coreset-ratio` with default `0.01` in the shared runner. Apply
`select_greedy_coreset` only when `feature_backbone == "patchcore_wrn50"`, after
the normalized full bank is built and before query scoring. Record the ratio,
projection dimension, and seed in provenance. The report must call this the
repository's PatchCore reproduction and disclose the 1% coreset.

Then dispatch PatchCore without changing DINO defaults.

At the top of `build_feature_extractor` in `src/features/dinov2.py`, add:

```python
if feature_backbone == "patchcore_wrn50":
    from features.patchcore import PatchCoreFeatureExtractor

    return PatchCoreFeatureExtractor(image_size=image_size, device=device)
```

The local import avoids a module cycle during import and keeps torchvision model
construction lazy.

- [ ] **Step 5: Add a compatibility CLI with explicit PatchCore defaults**

Create `scripts/run_patchcore_baseline.py`:

```python
#!/usr/bin/env python
"""Run the PatchCore-style baseline with the shared anomaly runner."""

from __future__ import annotations

import sys

from run_dinov2_baseline import main


def _append_default(flag: str, value: str) -> None:
    if flag not in sys.argv:
        sys.argv.extend([flag, value])


if __name__ == "__main__":
    _append_default("--feature-backbone", "patchcore_wrn50")
    _append_default("--image-size", "512")
    _append_default("--patch-size", "8")
    main()
```

- [ ] **Step 6: Verify network-free tests and CLI help**

Run: `PYTHONPATH=src venv/bin/python -m pytest tests/test_patchcore_features.py tests/test_anomaly_baseline.py -q`

Expected: focused tests pass without downloading weights.

Run: `PYTHONPATH=src venv/bin/python scripts/run_patchcore_baseline.py --help`

Expected: exit 0 and the shared few-shot arguments are listed.

- [ ] **Step 7: Commit PatchCore**

```bash
git add src/features/patchcore.py src/features/dinov2.py src/anomaly/memory_bank.py scripts/run_patchcore_baseline.py tests/test_patchcore_features.py
git commit -m "feat: add patchcore memory-bank baseline"
```

## Task 7: Feature Caching and Bounded Debug Rendering

**Files:**
- Create: `src/features/cache.py`
- Modify: `src/anomaly/multiscale.py:35-93`
- Modify: `scripts/run_dinov2_baseline.py:25-174`
- Create: `tests/test_feature_cache.py`
- Modify: `tests/test_anomaly_baseline.py`

- [ ] **Step 1: Write cache round-trip and invalidation tests**

```python
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from features.cache import FeatureCache, feature_cache_key
from features.dinov2 import PatchFeatureMap


class FeatureCacheTest(unittest.TestCase):
    def test_round_trip_preserves_feature_map(self) -> None:
        original = PatchFeatureMap(
            features=np.arange(12, dtype=np.float32).reshape(2, 2, 3),
            image_size=(28, 28),
            patch_size=14,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = FeatureCache(Path(tmpdir))
            cache.save("abc", original)
            restored = cache.load("abc")
        self.assertIsNotNone(restored)
        np.testing.assert_array_equal(restored.features, original.features)
        self.assertEqual(restored.image_size, original.image_size)

    def test_key_changes_for_result_affecting_inputs(self) -> None:
        first = feature_cache_key(
            "pcb1/a", "image-a", "dino-rev-a", "dinov2_vits14", 518, 14, "global"
        )
        second = feature_cache_key(
            "pcb1/a", "image-a", "dino-rev-a", "dinov2_vits14", 518, 14,
            "crop:0,0,768,768",
        )
        third = feature_cache_key(
            "pcb1/a", "image-a", "wrn-rev-a", "patchcore_wrn50", 512, 8, "global"
        )
        self.assertEqual(len({first, second, third}), 3)
```

- [ ] **Step 2: Confirm the cache module is missing**

Run: `PYTHONPATH=src venv/bin/python -m pytest tests/test_feature_cache.py -q`

Expected: FAIL importing `features.cache`.

- [ ] **Step 3: Implement cache storage and keys**

```python
"""Ignored on-disk cache for patch feature maps."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from features.dinov2 import PatchFeatureMap


def feature_cache_key(
    sample_id: str,
    image_sha256: str,
    extractor_revision: str,
    backbone: str,
    image_size: int,
    patch_size: int,
    view: str,
) -> str:
    payload = "|".join((
        sample_id,
        image_sha256,
        extractor_revision,
        backbone,
        str(image_size),
        str(patch_size),
        view,
    ))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class FeatureCache:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def load(self, key: str) -> PatchFeatureMap | None:
        path = self.root / f"{key}.npz"
        if not path.exists():
            return None
        payload = np.load(path)
        return PatchFeatureMap(
            features=payload["features"].astype(np.float32, copy=False),
            image_size=(int(payload["image_size"][0]), int(payload["image_size"][1])),
            patch_size=int(payload["patch_size"]),
        )

    def save(self, key: str, feature_map: PatchFeatureMap) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"{key}.npz"
        np.savez_compressed(
            path,
            features=feature_map.features.astype(np.float32),
            image_size=np.asarray(feature_map.image_size, dtype=np.int32),
            patch_size=np.asarray(feature_map.patch_size, dtype=np.int32),
        )
        return path

    def get_or_compute(self, key: str, compute) -> PatchFeatureMap:
        cached = self.load(key)
        if cached is not None:
            return cached
        result = compute()
        self.save(key, result)
        return result
```

- [ ] **Step 4: Thread optional cache identifiers through multi-scale scoring**

Extend `compute_anomaly_heatmap` with:

```python
feature_cache: FeatureCache | None = None,
cache_prefix: str | None = None,
```

Create a local helper:

```python
def _extract(
    image: Image.Image,
    extractor: PatchFeatureExtractor,
    feature_cache: FeatureCache | None,
    cache_key: str | None,
):
    if feature_cache is None or cache_key is None:
        return extractor.extract(image)
    return feature_cache.get_or_compute(cache_key, lambda: extractor.extract(image))
```

Use `<cache_prefix>:global` for the full image and
`<cache_prefix>:crop:<x1>,<y1>,<x2>,<y2>` for each crop. Pass the resulting
feature map into scoring instead of re-extracting it.

- [ ] **Step 5: Add runner cache and debug-limit arguments**

Add to `run_dinov2_baseline.py`:

```python
parser.add_argument("--feature-cache-dir", type=Path)
parser.add_argument(
    "--debug-limit",
    type=int,
    default=8,
    help="Render at most this many debug panels; use 0 to render none.",
)
```

Build `FeatureCache` when requested. Use full cache keys containing sample ID,
source-image SHA-256, extractor source revision, backbone, image size, patch
size, and view. Render a panel only when
`rank < args.debug_limit`; otherwise store an empty `debug_path` while still
saving the heatmap required by SAM2 and evaluation.

- [ ] **Step 6: Add a CLI test proving debug output is bounded**

Extend `tests/test_anomaly_baseline.py` to run a two-query color-patch fixture
with `--debug-limit 1`. Assert both heatmaps exist, exactly one debug PNG exists,
and the second `scores.csv` row has an empty `debug_path`.

- [ ] **Step 7: Verify and commit caching**

Run: `PYTHONPATH=src venv/bin/python -m pytest tests/test_feature_cache.py tests/test_multiscale_anomaly.py tests/test_anomaly_baseline.py -q`

Expected: focused tests pass.

```bash
git add src/features/cache.py src/anomaly/multiscale.py scripts/run_dinov2_baseline.py tests/test_feature_cache.py tests/test_anomaly_baseline.py
git commit -m "feat: cache features and bound debug artifacts"
```

## Task 8: Immutable Experiment Specifications and Provenance

**Files:**
- Create: `src/experiments/__init__.py`
- Create: `src/experiments/spec.py`
- Create: `src/experiments/provenance.py`
- Create: `configs/experiments/arxiv_primary.yaml`
- Create: `configs/experiments/arxiv_ablations.yaml`
- Create: `configs/experiments/arxiv_smoke.yaml`
- Modify: `pyproject.toml:26-28`
- Create: `tests/test_experiment_spec.py`
- Create: `tests/test_provenance.py`

- [ ] **Step 1: Write matrix cardinality and stable-ID tests**

```python
from pathlib import Path

from experiments.spec import RunSpec, expand_matrix, load_experiment_config


def test_primary_matrix_has_364_deduplicated_runs() -> None:
    config = load_experiment_config(Path("configs/experiments/arxiv_primary.yaml"))
    runs = expand_matrix(config)
    assert len(runs) == 364
    assert len({run.run_id for run in runs}) == 364


def test_ablation_matrix_has_48_deduplicated_runs() -> None:
    config = load_experiment_config(Path("configs/experiments/arxiv_ablations.yaml"))
    runs = expand_matrix(config)
    assert len(runs) == 48
    assert len({run.run_id for run in runs}) == 48


def test_sam2_only_does_not_repeat_across_shots_or_seeds() -> None:
    config = load_experiment_config(Path("configs/experiments/arxiv_primary.yaml"))
    runs = [run for run in expand_matrix(config) if run.method == "sam2_only"]
    assert len(runs) == 4
    assert {(run.k, run.seed) for run in runs} == {(0, 0)}


def test_run_id_contains_pairing_dimensions() -> None:
    run = RunSpec("dinov2_multi", "pcb2", fold_id=0, k=2, seed=4881)
    assert run.run_id == "dinov2_multi__pcb2__fold0__k2__seed4881"
```

- [ ] **Step 2: Write provenance checksum tests**

```python
def test_manifest_checksum_changes_with_content(tmp_path) -> None:
    path = tmp_path / "manifest.csv"
    path.write_text("a\n")
    first = sha256_file(path)
    path.write_text("b\n")
    assert sha256_file(path) != first


def test_provenance_records_support_ids(tmp_path) -> None:
    (tmp_path / "manifest.csv").write_text("sample_id\npcb1/0001\n")
    record = build_provenance(
        run_spec=RunSpec("dinov2_single", "pcb1", 0, 1, 4880),
        manifest_path=tmp_path / "manifest.csv",
        support_ids=["pcb1/0001"],
        git_commit="abc123",
    )
    assert record["support_ids"] == ["pcb1/0001"]
    assert record["run_id"].startswith("dinov2_single__pcb1")
```

- [ ] **Step 3: Confirm experiment modules are missing**

Run: `PYTHONPATH=src venv/bin/python -m pytest tests/test_experiment_spec.py tests/test_provenance.py -q`

Expected: FAIL importing `experiments`.

- [ ] **Step 4: Implement `RunSpec` and matrix expansion**

```python
"""Experiment configuration expansion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


PAIRED_METHODS = (
    "patchcore",
    "dinov2_single",
    "dinov2_multi",
    "dinov2_single_sam2",
    "dinov2_multi_sam2",
    "anomaly_consistent_sam2",
)


@dataclass(frozen=True)
class RunSpec:
    method: str
    category: str
    fold_id: int
    k: int
    seed: int
    variant: str = "primary"

    @property
    def run_id(self) -> str:
        base = (
            f"{self.method}__{self.category}__fold{self.fold_id}"
            f"__k{self.k}__seed{self.seed}"
        )
        return base if self.variant == "primary" else f"{base}__{self.variant}"


def load_experiment_config(path: str | Path) -> dict[str, object]:
    return yaml.safe_load(Path(path).read_text())


def expand_matrix(config: dict[str, object]) -> list[RunSpec]:
    fold_id = int(config["fold_id"])
    categories = [str(value) for value in config["categories"]]
    if "ablations" in config:
        return sorted(
            (
                RunSpec(
                    method=str(ablation["method"]),
                    category=category,
                    fold_id=fold_id,
                    k=int(config["k"]),
                    seed=int(config["seed"]),
                    variant=str(ablation["name"]),
                )
                for ablation in config["ablations"]
                for category in categories
            ),
            key=lambda run: run.run_id,
        )
    shots = [int(value) for value in config["shots"]]
    seeds = [int(value) for value in config["seeds"]]
    methods = [str(value) for value in config["methods"]]
    runs = [
        RunSpec(method, category, fold_id, k, seed)
        for method in methods
        if method != "sam2_only"
        for category in categories
        for k in shots
        for seed in seeds
    ]
    if "sam2_only" in methods:
        runs.extend(RunSpec("sam2_only", category, fold_id, 0, 0) for category in categories)
    return sorted(runs, key=lambda run: run.run_id)
```

- [ ] **Step 5: Implement provenance helpers**

```python
"""Immutable provenance records for experiment outputs."""

from __future__ import annotations

import hashlib
import platform
import sys
from pathlib import Path

from experiments.spec import RunSpec


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_provenance(
    run_spec: RunSpec,
    manifest_path: str | Path,
    support_ids: list[str],
    git_commit: str,
) -> dict[str, object]:
    return {
        "run_id": run_spec.run_id,
        "method": run_spec.method,
        "category": run_spec.category,
        "fold_id": run_spec.fold_id,
        "k": run_spec.k,
        "seed": run_spec.seed,
        "variant": run_spec.variant,
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "support_ids": sorted(support_ids),
        "git_commit": git_commit,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
```

- [ ] **Step 6: Add the frozen primary and smoke configs**

`configs/experiments/arxiv_primary.yaml`:

```yaml
name: arxiv_primary
manifest: data/manifests/visa_pcb_folds.csv
fold_id: 0
categories: [pcb1, pcb2, pcb3, pcb4]
shots: [1, 2, 4]
seeds: [4880, 4881, 4882, 4883, 4884]
methods:
  - patchcore
  - dinov2_single
  - dinov2_multi
  - sam2_only
  - dinov2_single_sam2
  - dinov2_multi_sam2
  - anomaly_consistent_sam2
dinov2:
  backbone: dinov2_vits14
  image_size: 518
  patch_size: 14
multi_scale:
  crop_sizes: [768]
  crop_overlap: 0.25
  fusion: max
calibration:
  quantile: 0.995
sam2:
  checkpoint: weights/sam2.1_hiera_tiny.pt
  model_config: configs/sam2.1/sam2.1_hiera_t.yaml
  prompt_mode: point_box
  point_mode: anomaly_max
  max_mask_area_fraction: 0.25
```

`configs/experiments/arxiv_smoke.yaml` uses one category, `shots: [1]`,
`seeds: [4880]`, `methods: [dinov2_single, dinov2_multi,
dinov2_multi_sam2, anomaly_consistent_sam2]`, `backbone: color_patch`, `image_size: 56`,
`patch_size: 14`, `crop_sizes: [32]`, and fallback refinement.

Create `configs/experiments/arxiv_ablations.yaml` with `k: 4`, `seed: 4880`,
all four categories, and these 12 named variants:

```yaml
name: arxiv_ablations
manifest: data/manifests/visa_pcb_folds.csv
dependency_output_root: outputs/arxiv_primary
fold_id: 0
categories: [pcb1, pcb2, pcb3, pcb4]
k: 4
seed: 4880
ablations:
  - {name: crop512_o025_max, method: dinov2_multi, overrides: {crop_sizes: [512], crop_overlap: 0.25, fusion: max}}
  - {name: crop768_o000_max, method: dinov2_multi, overrides: {crop_sizes: [768], crop_overlap: 0.0, fusion: max}}
  - {name: crop768_o050_max, method: dinov2_multi, overrides: {crop_sizes: [768], crop_overlap: 0.5, fusion: max}}
  - {name: crop768_o025_mean, method: dinov2_multi, overrides: {crop_sizes: [768], crop_overlap: 0.25, fusion: mean}}
  - {name: prompt_point_max, method: dinov2_multi_sam2, overrides: {prompt_mode: point, point_mode: anomaly_max, max_mask_area_fraction: 0.25}}
  - {name: prompt_box, method: dinov2_multi_sam2, overrides: {prompt_mode: box, point_mode: anomaly_max, max_mask_area_fraction: 0.25}}
  - {name: prompt_point_box_center, method: dinov2_multi_sam2, overrides: {prompt_mode: point_box, point_mode: box_center, max_mask_area_fraction: 0.25}}
  - {name: area_cap_none, method: dinov2_multi_sam2, overrides: {prompt_mode: point_box, point_mode: anomaly_max, max_mask_area_fraction: null}}
  - {name: area_cap_010, method: dinov2_multi_sam2, overrides: {prompt_mode: point_box, point_mode: anomaly_max, max_mask_area_fraction: 0.10}}
  - {name: area_cap_050, method: dinov2_multi_sam2, overrides: {prompt_mode: point_box, point_mode: anomaly_max, max_mask_area_fraction: 0.50}}
  - {name: fusion_union, method: anomaly_consistent_sam2, overrides: {mask_output: union}}
  - {name: fusion_selective, method: anomaly_consistent_sam2, overrides: {mask_output: selective, min_iou: 0.25, max_expansion: 2.0}}
```

The runner reads the named variant's `overrides` verbatim and writes them into
provenance. Prompt/area/fusion variants depend on the primary
`dinov2_multi__<category>__fold0__k4__seed4880` outputs; fusion variants also
depend on the corresponding saved primary raw SAM2 masks.

- [ ] **Step 7: Package experiments and verify**

Add `"experiments*"` to the setuptools include list in `pyproject.toml`.

Run: `PYTHONPATH=src venv/bin/python -m pytest tests/test_experiment_spec.py tests/test_provenance.py -q`

Expected: all tests pass; primary cardinality is 364 and ablation cardinality is 48.

- [ ] **Step 8: Commit experiment specifications**

```bash
git add src/experiments/__init__.py src/experiments/spec.py src/experiments/provenance.py configs/experiments/arxiv_primary.yaml configs/experiments/arxiv_ablations.yaml configs/experiments/arxiv_smoke.yaml pyproject.toml tests/test_experiment_spec.py tests/test_provenance.py
git commit -m "feat: define reproducible paper experiment matrix"
```

## Task 9: Dependency-Aware, Resumable Experiment Runner

**Files:**
- Create: `src/experiments/runner.py`
- Create: `scripts/run_experiment_matrix.py`
- Create: `scripts/check_experiment_matrix.py`
- Create: `tests/test_experiment_runner.py`
- Create: `tests/test_experiment_matrix_scripts.py`

- [ ] **Step 1: Write dependency and completion tests**

```python
from experiments.runner import expected_artifacts, method_dependencies
from experiments.spec import RunSpec


def test_anomaly_consistent_run_depends_on_multi_sam2() -> None:
    run = RunSpec("anomaly_consistent_sam2", "pcb1", 0, 2, 4880)
    assert method_dependencies(run) == [
        RunSpec("dinov2_multi", "pcb1", 0, 2, 4880),
        RunSpec("dinov2_multi_sam2", "pcb1", 0, 2, 4880),
    ]


def test_heatmap_run_requires_validation_calibration_and_test_metrics(tmp_path) -> None:
    run = RunSpec("dinov2_single", "pcb1", 0, 1, 4880)
    artifacts = expected_artifacts(run, tmp_path)
    assert artifacts == [
        tmp_path / run.run_id / "val" / "scores.csv",
        tmp_path / run.run_id / "calibration.json",
        tmp_path / run.run_id / "test" / "scores.csv",
        tmp_path / run.run_id / "test" / "metrics.json",
        tmp_path / run.run_id / "test" / "per_image.csv",
        tmp_path / run.run_id / "provenance.json",
        tmp_path / run.run_id / "status.json",
    ]
```

- [ ] **Step 2: Write a smoke dry-run CLI test**

Run the matrix script with `--config configs/experiments/arxiv_smoke.yaml
--dry-run`. Assert exit 0, no output directories are created, and stdout lists
the calibration command between validation and test evaluation.

- [ ] **Step 3: Confirm runner modules are missing**

Run: `PYTHONPATH=src venv/bin/python -m pytest tests/test_experiment_runner.py tests/test_experiment_matrix_scripts.py -q`

Expected: FAIL importing `experiments.runner`.

- [ ] **Step 4: Implement method dependencies and artifact contracts**

```python
"""Dependency and artifact contracts for experiment runs."""

from __future__ import annotations

from pathlib import Path

from experiments.spec import RunSpec


def method_dependencies(run: RunSpec) -> list[RunSpec]:
    if run.method == "dinov2_single_sam2":
        return [RunSpec("dinov2_single", run.category, run.fold_id, run.k, run.seed)]
    if run.method == "dinov2_multi_sam2":
        return [RunSpec("dinov2_multi", run.category, run.fold_id, run.k, run.seed)]
    if run.method == "anomaly_consistent_sam2":
        return [
            RunSpec("dinov2_multi", run.category, run.fold_id, run.k, run.seed),
            RunSpec("dinov2_multi_sam2", run.category, run.fold_id, run.k, run.seed),
        ]
    return []


def expected_artifacts(run: RunSpec, root: str | Path) -> list[Path]:
    directory = Path(root) / run.run_id
    common = [directory / "provenance.json", directory / "status.json"]
    if run.method in {"patchcore", "dinov2_single", "dinov2_multi"}:
        return [
            directory / "val" / "scores.csv",
            directory / "calibration.json",
            directory / "test" / "scores.csv",
            directory / "test" / "metrics.json",
            directory / "test" / "per_image.csv",
            *common,
        ]
    return [directory / "test" / "mask_scores.csv", directory / "test" / "mask_metrics.json", *common]


def is_complete(run: RunSpec, root: str | Path) -> bool:
    return all(path.exists() for path in expected_artifacts(run, root))
```

- [ ] **Step 5: Implement explicit command builders**

In `src/experiments/runner.py`, add `build_commands(run, config, root,
device) -> list[list[str]]`. Heatmap methods return commands in this exact order:

1. run the baseline on `--query-fold-split val` with the run's support seed;
2. fit `calibration.json` with `scripts/calibrate_heatmaps.py`;
3. run the same baseline on the locked test split;
4. evaluate with `--calibration-json` and write `per_image.csv`.

SAM2-guided methods consume the dependency's test `scores.csv` and calibration.
The anomaly-consistent method consumes the already-saved raw SAM2 mask rather
than rerunning SAM2. Every command is a list of arguments passed to
`subprocess.run(command, check=True)` without `shell=True`.

- [ ] **Step 6: Implement status and provenance-before-execution behavior**

Before the first subprocess, write `provenance.json` and a `status.json` with
`{"state": "running"}`. On success write `{"state": "complete"}`; on an
exception write `{"state": "failed", "error": str(exc)}` and re-raise. With
`--resume`, skip only when `is_complete` is true and `status.json` says
`complete`.

Extend the provenance record before writing it with the exact expanded command
lists, experiment-config path and SHA-256, dependency run IDs, dependency root,
torch/torchvision versions, CUDA device name, the local DINOv2 Torch Hub git
revision when present, and the local SAM2 git revision. If a model source is not
a git checkout, record `"unavailable"` rather than omitting the field.

- [ ] **Step 7: Implement matrix and checker CLIs**

Before execution, topologically order runs so every in-matrix dependency is
complete before its consumer. A dependency cycle raises an error naming the
cycle. Dependencies outside the current matrix are resolved under
`--dependency-root`; this defaults to `--output-root` for primary runs.

`scripts/run_experiment_matrix.py` arguments:

```text
--config PATH --output-root PATH --dependency-root PATH --device DEVICE
--dry-run --resume --run-id ID --method NAME --category NAME
```

`scripts/check_experiment_matrix.py` loads the same config, expands all runs,
checks duplicate IDs, expected artifacts, provenance manifest checksum, and
status. It accepts the same `--dependency-root`, verifies every external
dependency there, prints JSON, and exits 1 if any run or dependency is missing,
failed, duplicate, or stale.

- [ ] **Step 8: Verify dry-run, smoke execution, and resume**

Run: `PYTHONPATH=src venv/bin/python -m pytest tests/test_experiment_runner.py tests/test_experiment_matrix_scripts.py -q`

Expected: focused tests pass.

Run the smoke matrix on synthetic data and then run it again with `--resume`.
Expected: the first run completes all smoke methods; the second prints only
`SKIP complete` records.

- [ ] **Step 9: Commit the runner**

```bash
git add src/experiments/runner.py scripts/run_experiment_matrix.py scripts/check_experiment_matrix.py tests/test_experiment_runner.py tests/test_experiment_matrix_scripts.py
git commit -m "feat: orchestrate resumable paper experiments"
```

## Task 10: Failure Geometry and Paired SAM2 Analysis

**Files:**
- Create: `src/evaluation/failure_analysis.py`
- Create: `scripts/analyze_paper_results.py`
- Create: `tests/test_failure_analysis.py`
- Create: `tests/test_analyze_paper_results.py`

- [ ] **Step 1: Write hand-computed geometry tests**

```python
import numpy as np

from evaluation.failure_analysis import describe_mask_pair


def test_describe_mask_pair_reports_geometry_and_sam2_delta() -> None:
    target = np.asarray([[1, 1, 0], [0, 0, 0]], dtype=np.uint8)
    anomaly = np.asarray([[1, 0, 0], [0, 0, 0]], dtype=np.uint8)
    sam2 = np.asarray([[1, 1, 0], [0, 1, 0]], dtype=np.uint8)
    heatmap = np.asarray([[1.0, 0.8, 0.0], [0.0, 0.7, 0.0]], dtype=np.float32)

    row = describe_mask_pair(target, anomaly, sam2, heatmap)

    assert row["gt_area_fraction"] == 2 / 6
    assert row["gt_components"] == 1.0
    assert row["sam2_to_anomaly_area_ratio"] == 3.0
    assert row["sam2_delta_f1"] > 0.0
    assert row["mean_anomaly_inside_sam2"] > 0.8
```

- [ ] **Step 2: Confirm failure-analysis APIs are missing**

Run: `PYTHONPATH=src venv/bin/python -m pytest tests/test_failure_analysis.py -q`

Expected: FAIL importing `evaluation.failure_analysis`.

- [ ] **Step 3: Implement descriptors with existing metric primitives**

Create `describe_mask_pair(target, anomaly, sam2, heatmap)`. Use
`mask_confusion_metrics` for both predictions, the existing four-connected
component logic for component count, edge transitions for a perimeter proxy,
and `agreement_features` for proposal/SAM2 agreement. Return:

```python
{
    "gt_area_fraction": float(target.mean()),
    "gt_components": float(len(components)),
    "gt_thinness": float(perimeter * perimeter / max(4.0 * np.pi * area, 1.0)),
    "anomaly_f1": anomaly_metrics["f1"],
    "sam2_f1": sam2_metrics["f1"],
    "sam2_delta_f1": sam2_metrics["f1"] - anomaly_metrics["f1"],
    "anomaly_iou": anomaly_metrics["iou"],
    "sam2_iou": sam2_metrics["iou"],
    "sam2_delta_iou": sam2_metrics["iou"] - anomaly_metrics["iou"],
    "mean_anomaly_inside_sam2": float(normalized_heatmap[sam2 > 0].mean()) if sam2.any() else 0.0,
    **agreement_features(anomaly, sam2),
}
```

- [ ] **Step 4: Implement the joined analysis CLI**

`scripts/analyze_paper_results.py` takes `--config`, `--output-root`, and
`--analysis-dir`. It validates the matrix first, joins paired rows by
`(category, k, seed, sample_id)`, computes descriptors, writes
`per_image_failure_analysis.csv`, and writes `paired_statistics.json` containing
bootstrap deltas for:

- multi-scale anomaly versus unconditional multi-scale SAM2;
- multi-scale anomaly versus anomaly-consistent SAM2;
- unconditional multi-scale SAM2 versus anomaly-consistent SAM2.

It also writes small/medium/large defect-area and compact/thin strata with exact
bin edges derived from pooled tertiles and recorded in JSON.

- [ ] **Step 5: Verify and commit analysis**

Run: `PYTHONPATH=src venv/bin/python -m pytest tests/test_failure_analysis.py tests/test_analyze_paper_results.py -q`

Expected: focused tests pass on temporary CSV/PNG/NPY fixtures.

```bash
git add src/evaluation/failure_analysis.py scripts/analyze_paper_results.py tests/test_failure_analysis.py tests/test_analyze_paper_results.py
git commit -m "feat: analyze when sam2 helps pcb masks"
```

## Task 11: Like-for-Like Paper Tables and Evidence Index

**Files:**
- Create: `src/evaluation/paper_tables.py`
- Modify: `src/evaluation/stage4.py`
- Create: `scripts/build_paper_evidence.py`
- Create: `docs/evidence/README.md`
- Modify: `tests/test_stage4_summary.py`
- Create: `tests/test_paper_tables.py`
- Create: `tests/test_build_paper_evidence.py`

- [ ] **Step 1: Write tests that reject mixed metric definitions**

Replace the Stage 4 fixture expectation that maps heatmap `best_pixel_f1` into
the same `mask_f1` column as SAM2. The new test fixture must provide
`calibrated_mean_anomaly_mask_f1` for heatmaps and
`mean_anomaly_mask_f1` for masks, and assert:

```python
self.assertAlmostEqual(
    method_rows[("pcb1", "dinov2_single_heatmap")]["mean_anomaly_mask_f1"],
    0.2,
)
self.assertEqual(
    method_rows[("pcb1", "dinov2_single_heatmap")]["threshold_policy"],
    "normal_q995",
)
self.assertEqual(
    method_rows[("pcb1", "sam2_only")]["threshold_policy"],
    "binary_model_output",
)
```

Add a negative test: a heatmap metrics JSON containing only `best_pixel_f1`
raises `ValueError("calibrated heatmap metrics required")` when used in the
paper comparison.

- [ ] **Step 2: Write seed/category aggregation tests**

```python
from evaluation.paper_tables import aggregate_primary_rows


def test_aggregate_primary_rows_macro_averages_categories_then_seeds() -> None:
    rows = [
        {"method": "a", "category": "pcb1", "seed": 1, "metric": 0.2},
        {"method": "a", "category": "pcb1", "seed": 2, "metric": 0.4},
        {"method": "a", "category": "pcb2", "seed": 1, "metric": 0.6},
        {"method": "a", "category": "pcb2", "seed": 2, "metric": 0.8},
    ]
    summary = aggregate_primary_rows(rows, metric="metric")
    assert summary[0]["mean"] == 0.5
    assert summary[0]["num_categories"] == 2
    assert summary[0]["num_seeds"] == 2
```

- [ ] **Step 3: Run tests and confirm current Stage 4 behavior fails**

Run: `PYTHONPATH=src venv/bin/python -m pytest tests/test_stage4_summary.py tests/test_paper_tables.py -q`

Expected: FAIL because Stage 4 still mixes oracle aggregate F1 with mean per-image F1.

- [ ] **Step 4: Replace ambiguous Stage 4 fields**

Use this exact comparison schema in `src/evaluation/stage4.py`:

```python
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
```

For heatmap JSON, read only `calibrated_*` mask fields and set
`threshold_policy="normal_q995"`. For SAM2/fused binary masks, read the existing
`mean_anomaly_mask_*` fields and set
`threshold_policy="binary_model_output"`. Keep oracle fields in a separate
oracle-only table, never in the primary binary-mask table.

- [ ] **Step 5: Implement seed-aware paper aggregation**

`src/evaluation/paper_tables.py` must:

1. read completed run metrics plus provenance;
2. require exactly the configured categories and seeds for every paired method;
3. group by `(method, k, category, seed)`;
4. calculate category/seed means and standard deviations;
5. use `bootstrap_mean_ci` for 95% intervals;
6. return stable, explicitly named columns.

The public functions are `collect_primary_rows(config, output_root)`,
`aggregate_primary_rows(rows, metric)`, and `format_primary_markdown(rows)`.
`collect_primary_rows` reads each declared metrics/provenance pair and raises on
any missing category/seed. `aggregate_primary_rows` uses this concrete grouping:

```python
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
        output.append({
            "method": method,
            "k": k,
            "metric": metric,
            "mean": interval["mean"],
            "ci_low": interval["ci_low"],
            "ci_high": interval["ci_high"],
            "num_categories": len({str(row["category"]) for row in method_rows}),
            "num_seeds": len({int(row["seed"]) for row in method_rows}),
        })
    return output
```

`format_primary_markdown` renders those explicit keys in stable method/shot
order and formats metrics to four decimals.

- [ ] **Step 6: Implement the evidence builder**

`scripts/build_paper_evidence.py` takes `--config`, `--output-root`,
`--ablation-config`, `--ablation-output-root`, `--analysis-dir`, and
`--evidence-dir`. It first runs the same completion checks as
`check_experiment_matrix.py` on both matrices; incomplete matrices exit 1. On
success it writes:

```text
docs/evidence/generated/primary_results.csv
docs/evidence/generated/primary_results.md
docs/evidence/generated/oracle_diagnostics.csv
docs/evidence/generated/ablation_results.csv
docs/evidence/generated/paired_statistics.json
docs/evidence/generated/failure_strata.csv
docs/evidence/generated/evidence_index.md
docs/evidence/generated/completion_manifest.json
```

`completion_manifest.json` records every source run ID, git commit, manifest
checksum, generation command, and generated-file SHA-256. The generated
directory contains compact summaries only; no raw images, masks, heatmaps, or
weights.

- [ ] **Step 7: Add the tracked evidence policy**

Create `docs/evidence/README.md` stating:

- generated compact tables and JSON summaries are public evidence;
- raw artifacts stay under ignored `outputs/` and `artifacts/`;
- each generated file must appear in `completion_manifest.json`;
- generation fails rather than silently omitting incomplete runs;
- test-optimal metrics are diagnostics labeled `oracle_*`.

- [ ] **Step 8: Verify and commit evidence generation**

Run: `PYTHONPATH=src venv/bin/python -m pytest tests/test_stage4_summary.py tests/test_paper_tables.py tests/test_build_paper_evidence.py -q`

Expected: focused tests pass on a complete temporary matrix and the incomplete
matrix fixture exits 1.

```bash
git add src/evaluation/stage4.py src/evaluation/paper_tables.py scripts/build_paper_evidence.py docs/evidence/README.md tests/test_stage4_summary.py tests/test_paper_tables.py tests/test_build_paper_evidence.py
git commit -m "feat: generate provenance-backed paper tables"
```

## Task 12: Deterministic Paper Figures and Qualitative Curation

**Files:**
- Create: `scripts/curate_paper_assets.py`
- Create: `scripts/render_method_figure.py`
- Create: `tests/test_curate_paper_assets.py`
- Create: `tests/test_render_method_figure.py`

- [ ] **Step 1: Write deterministic selection tests**

```python
from curate_paper_assets import select_examples


def test_select_examples_returns_success_and_failure_per_category() -> None:
    rows = [
        {"category": "pcb1", "sample_id": "a", "sam2_delta_f1": "0.4"},
        {"category": "pcb1", "sample_id": "b", "sam2_delta_f1": "-0.3"},
        {"category": "pcb2", "sample_id": "c", "sam2_delta_f1": "0.2"},
        {"category": "pcb2", "sample_id": "d", "sam2_delta_f1": "-0.1"},
    ]
    selected = select_examples(rows, successes_per_category=1, failures_per_category=1)
    assert [(row["category"], row["sample_id"], row["role"]) for row in selected] == [
        ("pcb1", "a", "success"),
        ("pcb1", "b", "failure"),
        ("pcb2", "c", "success"),
        ("pcb2", "d", "failure"),
    ]
```

- [ ] **Step 2: Write method-figure rendering tests**

Run the renderer into a temporary path and assert the PNG exists, is RGB, has
non-zero dimensions, and contains more than four unique colors. This prevents a
blank or truncated figure from passing.

- [ ] **Step 3: Implement qualitative selection and copying**

`scripts/curate_paper_assets.py` reads
`per_image_failure_analysis.csv`, sorts deterministically by category and F1
delta, selects one success and one failure per category by default, and copies
the matching source image, ground-truth mask, anomaly panel, SAM2 panel, and
anomaly-consistent panel into:

```text
artifacts/paper_assets/qualitative/<category>/<role>/<sample_id-safe>/
```

It writes `artifacts/paper_assets/qualitative_manifest.csv` containing role,
sample ID, source run IDs, source paths, copied paths, metrics, and SHA-256.
Missing required panels are fatal.

- [ ] **Step 4: Implement the method figure with Pillow**

`scripts/render_method_figure.py` draws five labeled blocks:

```text
Few normal supports -> DINOv2 memory bank -> Multi-scale anomaly map
                                              | calibrated proposal
                                              v
Query image -------------------------------> SAM2 prompts/refinement
                                              |
                         proposal intersection + SAM2 mask -> final mask
```

Use a white 1800x700 canvas, consistent blue/amber/green blocks, 3-pixel arrows,
and fonts discovered from the system with a Pillow default fallback. Save both
`artifacts/paper_assets/method_figure.png` and a JSON layout manifest containing
labels, rectangles, colors, and generation command.

- [ ] **Step 5: Verify and commit figure tooling**

Run: `PYTHONPATH=src venv/bin/python -m pytest tests/test_curate_paper_assets.py tests/test_render_method_figure.py -q`

Expected: focused tests pass; test artifacts remain inside temporary folders.

```bash
git add scripts/curate_paper_assets.py scripts/render_method_figure.py tests/test_curate_paper_assets.py tests/test_render_method_figure.py
git commit -m "feat: curate reproducible paper figures"
```

## Task 13: Smoke Protocol, Notebook, and Public Documentation

**Files:**
- Modify: `notebooks/pcb_defect_pipeline.ipynb`
- Modify: `notebooks/README.md`
- Modify: `README.md`
- Modify: `docs/PRD.md`
- Modify: `docs/milestone2_readiness_checklist.md`
- Create: `docs/arxiv_readiness_checklist.md`
- Modify: `docs/autodl_data_setup.md`
- Create: `tests/test_arxiv_smoke.py`
- Create: `tests/test_public_hygiene.py`

- [ ] **Step 1: Add an end-to-end smoke test**

The test creates the existing synthetic VisA fixture, creates a fold manifest,
runs `configs/experiments/arxiv_smoke.yaml`, checks the matrix, runs analysis,
and builds evidence. It asserts:

```python
assert (output_root / "matrix_summary.json").exists()
assert checker.returncode == 0
assert (evidence_dir / "completion_manifest.json").exists()
assert "anomaly_consistent_sam2" in (evidence_dir / "primary_results.md").read_text()
```

The smoke must use `color_patch` and the fallback refiner so it requires no
network, GPU, or model checkpoint.

- [ ] **Step 2: Add a public-hygiene test**

The test runs `git ls-files` and fails if a tracked path is under `data/`,
`outputs/`, `weights/`, `external/`, `artifacts/`, `reports/`, or `submissions/`,
or has a forbidden model/archive extension. It also scans tracked text files for
`sk-`, `OPENAI_API_KEY=`, `SUPABASE_SERVICE_ROLE_KEY=`, and private course-report
filenames.

- [ ] **Step 3: Run the smoke tests before documentation changes**

Run: `PYTHONPATH=src venv/bin/python -m pytest tests/test_arxiv_smoke.py tests/test_public_hygiene.py -q`

Expected: the smoke passes once Tasks 1-12 are complete; hygiene may identify
only intentional documentation paths, never ignored data/model artifacts.

- [ ] **Step 4: Update the notebook as a presentation layer**

Keep core logic in `src/` and `scripts/`. Add or refresh these top-to-bottom
sections in `notebooks/pcb_defect_pipeline.ipynb`:

1. frozen research question and prior-test-exposure disclosure;
2. dataset and repeated few-shot protocol;
3. single-scale, multi-scale, PatchCore, and SAM2 method summary;
4. normal-only calibration versus oracle diagnostics;
5. primary table loaded from `docs/evidence/generated/primary_results.csv`;
6. paired confidence intervals and failure strata;
7. selected success/failure figures;
8. limitations and decision-rule outcome;
9. exact reproduction commands.

Clear stale exploratory outputs. Execute only lightweight table/figure-loading
cells locally; heavy experiments remain scripts.

- [ ] **Step 5: Update README and research docs**

`README.md` must include:

- one exact local setup;
- the offline smoke command;
- one single-run command;
- primary AutoDL matrix and resume commands;
- matrix validation, analysis, asset, and evidence commands;
- a clear distinction between oracle and calibrated metrics;
- the statement that VisA is primary and DeepPCB boxes are secondary;
- the no-CLIP and public-artifact boundaries.

Update `docs/PRD.md` stage statuses from live evidence, not intention. Replace
the old milestone checklist verdict with a link to the new arXiv checklist while
preserving its historical results and date.

- [ ] **Step 6: Create the arXiv readiness checklist**

`docs/arxiv_readiness_checklist.md` contains one row per design deliverable with
columns:

```text
Requirement | Status | Authoritative evidence | Verification command | Notes
```

Initial statuses are derived programmatically from the matrix/evidence
completion manifest. A missing file or failed command is `incomplete`, never
silently `complete`.

- [ ] **Step 7: Update AutoDL instructions**

Add commands using `/root/autodl-tmp/fewshot-pcb-defect-segmentation`, the Python
3.11 venv, CUDA check, `tmux`, `--resume`, the primary matrix config, checker,
and commands to archive only compact summaries and selected paper assets for
transfer back to the local checkout.

- [ ] **Step 8: Verify notebook, links, smoke, and hygiene**

Run:

```bash
PYTHONPATH=src venv/bin/python -m pytest tests/test_arxiv_smoke.py tests/test_public_hygiene.py -q
PYTHONPATH=src venv/bin/python -m pytest -q
venv/bin/python -m ruff check src scripts tests
git diff --check
```

Expected: all tests pass, Ruff passes, and diff check is clean.

- [ ] **Step 9: Commit documentation and smoke protocol**

Stage the exact task files, including the previously untracked user-approved
milestone evidence only after reviewing it against the live results:

```bash
git add README.md notebooks/pcb_defect_pipeline.ipynb notebooks/README.md docs/PRD.md docs/milestone2_readiness_checklist.md docs/arxiv_readiness_checklist.md docs/autodl_data_setup.md tests/test_arxiv_smoke.py tests/test_public_hygiene.py
git commit -m "docs: add arxiv-ready reproduction workflow"
```

## Task 14: Execute and Validate the Full AutoDL Matrix

**Files:**
- Generated, ignored: `outputs/arxiv_primary/**`
- Generated, ignored: `artifacts/paper_assets/**`
- Generated, tracked after review: `docs/evidence/generated/**`
- Modify after evidence freeze: `docs/arxiv_readiness_checklist.md`

- [ ] **Step 1: Verify the remote runtime before spending GPU time**

On AutoDL:

```bash
cd /root/autodl-tmp/fewshot-pcb-defect-segmentation
source venv/bin/activate
python --version
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
PYTHONPATH=src python -m pytest -q
PYTHONPATH=src python scripts/check_experiment_matrix.py \
  --config configs/experiments/arxiv_primary.yaml \
  --output-root outputs/arxiv_primary
```

Expected before execution: Python 3.10-3.12, CUDA `True`, tests pass, and the
checker reports 364 missing runs rather than malformed configuration.

- [ ] **Step 2: Run one real DINOv2/SAM2 primary-spec smoke**

```bash
PYTHONPATH=src python scripts/run_experiment_matrix.py \
  --config configs/experiments/arxiv_primary.yaml \
  --output-root outputs/arxiv_primary \
  --run-id dinov2_multi__pcb1__fold0__k1__seed4880 \
  --device cuda
PYTHONPATH=src python scripts/run_experiment_matrix.py \
  --config configs/experiments/arxiv_primary.yaml \
  --output-root outputs/arxiv_primary \
  --run-id dinov2_multi_sam2__pcb1__fold0__k1__seed4880 \
  --device cuda
PYTHONPATH=src python scripts/run_experiment_matrix.py \
  --config configs/experiments/arxiv_primary.yaml \
  --output-root outputs/arxiv_primary \
  --run-id anomaly_consistent_sam2__pcb1__fold0__k1__seed4880 \
  --device cuda
```

Expected: all three runs complete, their provenance/status files exist, and the
consistent run reuses the saved SAM2 mask.

- [ ] **Step 3: Inspect the real smoke artifacts**

Evaluate the three metric JSON files and inspect at least one normal, one SAM2
success, and one SAM2 failure panel. Confirm mask/image alignment and that
intersection never contains pixels outside the calibrated anomaly mask.

- [ ] **Step 4: Launch the full resumable matrix in `tmux`**

```bash
tmux new -s pcb-arxiv
cd /root/autodl-tmp/fewshot-pcb-defect-segmentation
source venv/bin/activate
PYTHONPATH=src python scripts/run_experiment_matrix.py \
  --config configs/experiments/arxiv_primary.yaml \
  --output-root outputs/arxiv_primary \
  --device cuda \
  --resume 2>&1 | tee outputs/arxiv_primary.log
PYTHONPATH=src python scripts/run_experiment_matrix.py \
  --config configs/experiments/arxiv_ablations.yaml \
  --output-root outputs/arxiv_ablations \
  --dependency-root outputs/arxiv_primary \
  --device cuda \
  --resume 2>&1 | tee outputs/arxiv_ablations.log
```

Detach with `Ctrl-b d`. Reattach with `tmux attach -t pcb-arxiv`. After any
interruption, rerun the interrupted matrix command with `--resume`.

- [ ] **Step 5: Require complete 364-run primary and 48-run ablation matrices**

```bash
PYTHONPATH=src python scripts/check_experiment_matrix.py \
  --config configs/experiments/arxiv_primary.yaml \
  --output-root outputs/arxiv_primary \
  --output-json outputs/arxiv_primary/matrix_summary.json
```

Then validate ablations:

```bash
PYTHONPATH=src python scripts/check_experiment_matrix.py \
  --config configs/experiments/arxiv_ablations.yaml \
  --output-root outputs/arxiv_ablations \
  --dependency-root outputs/arxiv_primary \
  --output-json outputs/arxiv_ablations/matrix_summary.json
```

Expected: primary exits 0 with `complete=364`; ablations exit 0 with
`complete=48`; both report zero missing, failed, duplicate, or stale runs. Any
other result stops evidence generation.

- [ ] **Step 6: Generate analysis, figures, and compact evidence**

```bash
PYTHONPATH=src python scripts/analyze_paper_results.py \
  --config configs/experiments/arxiv_primary.yaml \
  --output-root outputs/arxiv_primary \
  --analysis-dir outputs/arxiv_primary/analysis
PYTHONPATH=src python scripts/curate_paper_assets.py \
  --analysis-csv outputs/arxiv_primary/analysis/per_image_failure_analysis.csv \
  --output-root outputs/arxiv_primary \
  --asset-dir artifacts/paper_assets
PYTHONPATH=src python scripts/render_method_figure.py \
  --output artifacts/paper_assets/method_figure.png
PYTHONPATH=src python scripts/build_paper_evidence.py \
  --config configs/experiments/arxiv_primary.yaml \
  --output-root outputs/arxiv_primary \
  --ablation-config configs/experiments/arxiv_ablations.yaml \
  --ablation-output-root outputs/arxiv_ablations \
  --analysis-dir outputs/arxiv_primary/analysis \
  --evidence-dir docs/evidence/generated
```

Expected: every declared table, statistical file, qualitative manifest, method
figure, evidence index, and completion manifest exists.

- [ ] **Step 7: Transfer compact artifacts back to the local checkout**

Transfer `docs/evidence/generated/` and `artifacts/paper_assets/`; do not transfer
raw datasets, feature caches, checkpoints, or the full heatmap tree unless a
specific missing panel requires it. Verify transferred SHA-256 values against
the completion and qualitative manifests.

- [ ] **Step 8: Review evidence before committing summaries**

Check:

- all four categories, three shot counts, and five seeds are present;
- primary tables use calibrated like-for-like metrics;
- oracle diagnostics are separately labeled;
- confidence intervals and paired deltas match source per-image rows;
- the final decision rule is positive, mixed, or negative based on the evidence;
- no claim depends on a smoke output or historical `k=5` result.

- [ ] **Step 9: Commit the frozen compact evidence**

```bash
git add docs/evidence/generated docs/arxiv_readiness_checklist.md
git commit -m "results: freeze arxiv evidence summaries"
```

Do not add `outputs/`, `artifacts/`, model weights, datasets, or logs.

## Task 15: Final Paper-Readiness Completion Audit

**Files:**
- Modify: `docs/arxiv_readiness_checklist.md`
- Modify: `README.md`
- Modify: `notebooks/pcb_defect_pipeline.ipynb`
- Modify only if results require it: `docs/evidence/generated/evidence_index.md`

- [ ] **Step 1: Re-run all local verification from a clean process**

```bash
PYTHONPATH=src venv/bin/python -m pytest -q
venv/bin/python -m ruff check src scripts tests
PYTHONPATH=src venv/bin/python scripts/run_experiment_matrix.py \
  --config configs/experiments/arxiv_smoke.yaml \
  --output-root /tmp/pcb-arxiv-smoke \
  --resume
PYTHONPATH=src venv/bin/python scripts/check_experiment_matrix.py \
  --config configs/experiments/arxiv_smoke.yaml \
  --output-root /tmp/pcb-arxiv-smoke
git diff --check
```

Expected: tests, Ruff, smoke matrix, checker, and diff hygiene all pass.

- [ ] **Step 2: Prove every design deliverable from authoritative evidence**

For each of the 14 paper-readiness deliverables in the approved design, record
one direct file and one verification command in
`docs/arxiv_readiness_checklist.md`. Do not mark a row complete from intention,
an old milestone table, a smoke run, or absence of an obvious problem.

- [ ] **Step 3: Perform the public repository audit**

```bash
git status --short --branch
PYTHONPATH=src venv/bin/python -m pytest tests/test_public_hygiene.py -q
```

Expected: the hygiene test passes with no forbidden tracked artifacts or likely
secrets.

- [ ] **Step 4: Inspect final qualitative and notebook rendering**

Open the method figure and all eight category success/failure selections. Read
the notebook top to bottom and confirm every displayed number comes from the
frozen evidence tables. Remove stale output cells or claims that differ from the
final decision rule.

- [ ] **Step 5: Freeze the evidence-led conclusion**

Choose exactly one conclusion supported by paired intervals:

- anomaly-consistent SAM2 improves robustly;
- improvement is conditional/mixed by category or geometry;
- SAM2 remains inferior and the project is a controlled negative study.

Write that conclusion in the evidence index, readiness checklist, README status,
and notebook. Do not change the primary configuration or rerun a search for a
better test result after this decision.

- [ ] **Step 6: Run final status and commit the audit**

```bash
git diff --check
PYTHONPATH=src venv/bin/python -m pytest -q
venv/bin/python -m ruff check src scripts tests
git status --short --branch
git add README.md notebooks/pcb_defect_pipeline.ipynb docs/arxiv_readiness_checklist.md docs/evidence/generated/evidence_index.md
git commit -m "docs: certify arxiv evidence readiness"
```

Expected: only intentionally ignored local artifacts remain; the tracked
readiness checklist has no incomplete requirement. At this point—and only at
this point—the active project goal may be marked complete and report writing can
begin.

## Plan Self-Review: Specification Coverage

| Approved design requirement | Implemented by |
| --- | --- |
| Normal-only 99.5% calibration and separate oracle diagnostics | Tasks 1-2 |
| Comparable aggregate and per-image mask metrics | Tasks 2 and 11 |
| Five-seed confidence intervals and paired statistics | Tasks 3, 10, and 11 |
| Anomaly-maximum point, point/box modes, and SAM2 filtering | Tasks 4-5 and ablation matrix in Task 8 |
| Primary anomaly-consistent intersection and selective fallback | Task 5 |
| PatchCore external baseline under identical support protocol | Task 6 |
| Query/crop feature caching and bounded debug artifacts | Task 7 |
| `k = 1, 2, 4`, five seeds, four categories, seven primary methods | Task 8 primary matrix |
| Crop, overlap, fusion, prompt, point, and area-cap ablations | Task 8 frozen 48-run ablation matrix |
| Immutable provenance, exact support IDs, checksums, and resume behavior | Tasks 8-9 |
| Defect-area, components, thinness, anomaly density, expansion, and SAM2 delta analysis | Task 10 |
| Like-for-like tables, evidence manifest, and failure-on-incomplete behavior | Task 11 |
| Per-category success/failure assets and method figure | Task 12 |
| Notebook presentation layer and exact local/AutoDL commands | Task 13 |
| Full real GPU experiment execution and transfer verification | Task 14 |
| Public hygiene, Apache-2.0 boundary, VisA/DeepPCB distinction, and no CLIP | Tasks 13 and 15 |
| Positive, mixed, or negative evidence-led conclusion | Tasks 14-15 |

No approved design requirement is intentionally deferred outside this plan.
