"""Deterministic bootstrap statistics for experiment comparisons."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


_BOOTSTRAP_BATCH_SIZE = 256


def bootstrap_mean_ci(
    values: list[float],
    confidence: float = 0.95,
    samples: int = 2000,
    seed: int = 4880,
) -> dict[str, float]:
    """Return the mean and percentile-bootstrap confidence interval."""

    _validate_bootstrap_options(confidence=confidence, samples=samples)
    values_array = _as_finite_array(values, name="values")

    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, _BOOTSTRAP_BATCH_SIZE):
        batch_size = min(_BOOTSTRAP_BATCH_SIZE, samples - start)
        indices = rng.integers(
            0,
            values_array.size,
            size=(batch_size, values_array.size),
        )
        bootstrap_means[start : start + batch_size] = values_array[indices].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    ci_low, ci_high = np.percentile(
        bootstrap_means,
        [100.0 * tail, 100.0 * (1.0 - tail)],
    )
    return {
        "mean": float(values_array.mean()),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
    }


def paired_bootstrap_delta(
    baseline: list[float],
    candidate: list[float],
    confidence: float = 0.95,
    samples: int = 2000,
    seed: int = 4880,
) -> dict[str, float]:
    """Return a confidence interval for paired candidate-minus-baseline deltas."""

    baseline_array = _as_finite_array(baseline, name="baseline")
    candidate_array = _as_finite_array(candidate, name="candidate")
    if baseline_array.shape != candidate_array.shape:
        raise ValueError("baseline and candidate must have the same length")

    interval = bootstrap_mean_ci(
        (candidate_array - baseline_array).tolist(),
        confidence=confidence,
        samples=samples,
        seed=seed,
    )
    return {
        "mean_delta": interval["mean"],
        "ci_low": interval["ci_low"],
        "ci_high": interval["ci_high"],
    }


def repeated_measures_paired_delta(
    rows: Sequence[Mapping[str, object]],
    *,
    baseline_field: str,
    candidate_field: str,
    confidence: float = 0.95,
    samples: int = 2000,
    seed: int = 4880,
) -> dict[str, float | int | str]:
    """Bootstrap paired deltas without treating repeated images as independent.

    Categories and shot counts are fixed reporting strata. Support-seed indices
    are resampled once per replicate and reused across all strata. Test-image
    clusters are resampled once per category and reused across shots and seeds.
    Candidate-minus-baseline pairing is therefore preserved at every level.
    """

    _validate_bootstrap_options(confidence=confidence, samples=samples)
    if not rows:
        raise ValueError("rows must not be empty")

    cells: dict[tuple[str, int, int, str], float] = {}
    for index, row in enumerate(rows):
        try:
            category = str(row["category"])
            k = int(row["k"])
            support_seed = int(row["seed"])
            sample_id = str(row["sample_id"])
            baseline = float(row[baseline_field])
            candidate = float(row[candidate_field])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"row {index} has an invalid repeated-measures field") from exc
        if not category or not sample_id:
            raise ValueError(f"row {index} has an empty category or sample_id")
        if not np.isfinite(baseline) or not np.isfinite(candidate):
            raise ValueError(f"row {index} paired metrics must be finite")
        key = (category, k, support_seed, sample_id)
        if key in cells:
            raise ValueError(f"duplicate repeated-measures cell: {key!r}")
        cells[key] = candidate - baseline

    categories = sorted({key[0] for key in cells})
    shots = sorted({key[1] for key in cells})
    support_seeds = sorted({key[2] for key in cells})
    samples_by_category: dict[str, list[str]] = {}
    tensors: dict[str, np.ndarray] = {}
    for category in categories:
        reference_samples: set[str] | None = None
        for k in shots:
            observed_seeds = {
                cell_seed
                for cell_category, cell_k, cell_seed, _ in cells
                if cell_category == category and cell_k == k
            }
            if observed_seeds != set(support_seeds):
                raise ValueError("every category/k stratum must contain the same support seeds")
            for support_seed in support_seeds:
                observed_samples = {
                    sample_id
                    for cell_category, cell_k, cell_seed, sample_id in cells
                    if cell_category == category and cell_k == k and cell_seed == support_seed
                }
                if reference_samples is None:
                    reference_samples = observed_samples
                elif observed_samples != reference_samples:
                    raise ValueError(
                        "every repeated run must contain the same paired test images "
                        "within a category"
                    )
        if not reference_samples:
            raise ValueError(f"category {category!r} has no paired test images")
        category_samples = sorted(reference_samples)
        samples_by_category[category] = category_samples
        tensors[category] = np.asarray(
            [
                [
                    [
                        cells[(category, k, support_seed, sample_id)]
                        for sample_id in category_samples
                    ]
                    for support_seed in support_seeds
                ]
                for k in shots
            ],
            dtype=np.float64,
        )

    stratum_means = [
        float(tensors[category][k_index].mean())
        for category in categories
        for k_index in range(len(shots))
    ]
    point_mean = float(np.mean(stratum_means))
    seed_macro_values = np.asarray(
        [
            np.mean(
                [
                    tensors[category][k_index, seed_index, :].mean()
                    for category in categories
                    for k_index in range(len(shots))
                ]
            )
            for seed_index in range(len(support_seeds))
        ],
        dtype=np.float64,
    )
    seed_std = float(seed_macro_values.std(ddof=1)) if seed_macro_values.size > 1 else 0.0

    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, _BOOTSTRAP_BATCH_SIZE):
        batch_size = min(_BOOTSTRAP_BATCH_SIZE, samples - start)
        seed_indices = rng.integers(
            0,
            len(support_seeds),
            size=(batch_size, len(support_seeds)),
        )
        replicate_sum = np.zeros(batch_size, dtype=np.float64)
        for category in categories:
            category_tensor = tensors[category]
            num_images = category_tensor.shape[2]
            image_indices = rng.integers(
                0,
                num_images,
                size=(batch_size, num_images),
            )
            for k_index in range(len(shots)):
                selected = category_tensor[k_index][
                    seed_indices[:, :, None],
                    image_indices[:, None, :],
                ]
                replicate_sum += selected.mean(axis=(1, 2))
        bootstrap_means[start : start + batch_size] = replicate_sum / (len(categories) * len(shots))

    tail = (1.0 - confidence) / 2.0
    ci_low, ci_high = np.percentile(
        bootstrap_means,
        [100.0 * tail, 100.0 * (1.0 - tail)],
    )
    return {
        "mean_delta": point_mean,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "seed_std_delta": seed_std,
        "num_rows": len(rows),
        "num_unique_images": sum(len(value) for value in samples_by_category.values()),
        "num_categories": len(categories),
        "num_k": len(shots),
        "num_seeds": len(support_seeds),
        "resampling_unit": "support_seed_and_test_image",
        "aggregation": "fixed_k_category_macro",
        "bootstrap_samples": samples,
        "confidence": float(confidence),
    }


def _validate_bootstrap_options(confidence: float, samples: int) -> None:
    if not np.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    if (
        isinstance(samples, (bool, np.bool_))
        or not isinstance(samples, (int, np.integer))
        or samples <= 0
    ):
        raise ValueError("samples must be a positive integer")


def _as_finite_array(values: list[float], name: str) -> np.ndarray:
    values_array = np.asarray(values, dtype=np.float64)
    if values_array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if values_array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(values_array)):
        raise ValueError(f"{name} must contain only finite values")
    return values_array
