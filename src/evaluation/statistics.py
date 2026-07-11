"""Deterministic bootstrap statistics for experiment comparisons."""

from __future__ import annotations

import numpy as np


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
    indices = rng.integers(
        0,
        values_array.size,
        size=(samples, values_array.size),
    )
    bootstrap_means = values_array[indices].mean(axis=1)
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
    if baseline_array.size != candidate_array.size:
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


def _validate_bootstrap_options(confidence: float, samples: int) -> None:
    if not np.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    if not isinstance(samples, (int, np.integer)) or samples <= 0:
        raise ValueError("samples must be a positive integer")


def _as_finite_array(values: list[float], name: str) -> np.ndarray:
    values_array = np.asarray(values, dtype=np.float64)
    if values_array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(values_array)):
        raise ValueError(f"{name} must contain only finite values")
    return values_array
