from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from evaluation.statistics import bootstrap_mean_ci, paired_bootstrap_delta


class BootstrapStatisticsTest(unittest.TestCase):
    def test_constant_values_have_zero_width_interval(self) -> None:
        result = bootstrap_mean_ci([0.5, 0.5, 0.5])

        self.assertEqual(result, {"mean": 0.5, "ci_low": 0.5, "ci_high": 0.5})

    def test_paired_delta_is_candidate_minus_baseline(self) -> None:
        result = paired_bootstrap_delta(
            baseline=[0.1, 0.3, 0.5],
            candidate=[0.2, 0.4, 0.6],
        )

        self.assertAlmostEqual(result["mean_delta"], 0.1)
        self.assertAlmostEqual(result["ci_low"], 0.1)
        self.assertAlmostEqual(result["ci_high"], 0.1)

    def test_same_seed_and_input_return_identical_results(self) -> None:
        values = [0.1, 0.2, 0.4, 0.8]

        first = bootstrap_mean_ci(values, samples=250, seed=17)
        second = bootstrap_mean_ci(values, samples=250, seed=17)

        self.assertEqual(first, second)

    def test_seeded_nonconstant_interval_matches_numerical_regression(self) -> None:
        result = bootstrap_mean_ci(
            [0.1, 0.4, 0.9, 1.6],
            confidence=0.8,
            samples=12,
            seed=42,
        )

        self.assertEqual(result, {"mean": 0.75, "ci_low": 0.675, "ci_high": 1.125})

    def test_resampling_uses_bounded_batches_without_changing_results(self) -> None:
        values = [0.1, 0.2, 0.4, 0.8]
        samples = 513
        expected = bootstrap_mean_ci(values, samples=samples, seed=29)
        recording_rng = mock.Mock(wraps=np.random.default_rng(29))

        with mock.patch(
            "evaluation.statistics.np.random.default_rng",
            return_value=recording_rng,
        ):
            result = bootstrap_mean_ci(values, samples=samples, seed=29)

        requested_sizes = [call.kwargs["size"] for call in recording_rng.integers.call_args_list]
        self.assertEqual(result, expected)
        self.assertEqual(sum(size[0] for size in requested_sizes), samples)
        self.assertGreater(len(requested_sizes), 1)
        self.assertLessEqual(max(size[0] for size in requested_sizes), 256)

    def test_mismatched_paired_lengths_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "same length"):
            paired_bootstrap_delta([0.1, 0.2], [0.3])

    def test_empty_inputs_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            bootstrap_mean_ci([])
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            paired_bootstrap_delta([], [])

    def test_non_finite_inputs_are_rejected(self) -> None:
        for value in [float("nan"), float("inf"), float("-inf")]:
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "finite"):
                    bootstrap_mean_ci([0.1, value])

        with self.assertRaisesRegex(ValueError, "finite"):
            paired_bootstrap_delta([0.1, 0.2], [0.2, float("nan")])

    def test_nested_inputs_are_rejected_as_not_one_dimensional(self) -> None:
        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            bootstrap_mean_ci([[0.1, 0.2], [0.3, 0.4]])
        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            paired_bootstrap_delta([[0.1, 0.2]], [[0.2, 0.3]])

    def test_confidence_outside_open_unit_interval_is_rejected(self) -> None:
        for confidence in [0.0, 1.0, -0.1, 1.1, float("nan")]:
            with self.subTest(confidence=confidence):
                with self.assertRaisesRegex(ValueError, "confidence"):
                    bootstrap_mean_ci([0.1], confidence=confidence)
                with self.assertRaisesRegex(ValueError, "confidence"):
                    paired_bootstrap_delta([0.1], [0.2], confidence=confidence)

    def test_non_positive_sample_counts_are_rejected(self) -> None:
        for samples in [0, -1]:
            with self.subTest(samples=samples):
                with self.assertRaisesRegex(ValueError, "samples"):
                    bootstrap_mean_ci([0.1], samples=samples)
                with self.assertRaisesRegex(ValueError, "samples"):
                    paired_bootstrap_delta([0.1], [0.2], samples=samples)

    def test_boolean_sample_counts_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "samples"):
            bootstrap_mean_ci([0.1], samples=True)
        with self.assertRaisesRegex(ValueError, "samples"):
            paired_bootstrap_delta([0.1], [0.2], samples=True)

    def test_paired_resampling_retains_pairing(self) -> None:
        result = paired_bootstrap_delta(
            baseline=[-10_000.0, 0.0, 10_000.0],
            candidate=[-9_999.0, 1.0, 10_001.0],
            samples=250,
            seed=23,
        )

        self.assertEqual(
            result,
            {"mean_delta": 1.0, "ci_low": 1.0, "ci_high": 1.0},
        )


if __name__ == "__main__":
    unittest.main()
