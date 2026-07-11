from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from sam_refine.fusion import agreement_features, fuse_masks, fusion_source


class MaskFusionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.anomaly = np.array([[1, 1, 0], [0, 1, 0]], dtype=np.int64)
        self.sam2 = np.array([[0, 2, 2], [0, 3, 0]], dtype=np.float32)

    def test_exact_fusion_modes_are_binary_uint8_and_shape_preserving(self) -> None:
        expected = {
            "anomaly": np.array([[1, 1, 0], [0, 1, 0]], dtype=np.uint8),
            "sam2": np.array([[0, 1, 1], [0, 1, 0]], dtype=np.uint8),
            "intersection": np.array([[0, 1, 0], [0, 1, 0]], dtype=np.uint8),
            "union": np.array([[1, 1, 1], [0, 1, 0]], dtype=np.uint8),
        }

        for mode, expected_mask in expected.items():
            with self.subTest(mode=mode):
                actual = fuse_masks(self.anomaly, self.sam2, mode=mode)
                np.testing.assert_array_equal(actual, expected_mask)
                self.assertEqual(actual.dtype, np.uint8)
                self.assertEqual(actual.shape, self.anomaly.shape)

    def test_agreement_features_report_exact_areas_iou_and_expansion(self) -> None:
        features = agreement_features(self.anomaly, self.sam2)

        self.assertEqual(features["anomaly_pixels"], 3.0)
        self.assertEqual(features["sam2_pixels"], 3.0)
        self.assertEqual(features["intersection_pixels"], 2.0)
        self.assertEqual(features["union_pixels"], 4.0)
        self.assertEqual(features["mask_iou"], 0.5)
        self.assertEqual(features["sam2_to_anomaly_area_ratio"], 1.0)

    def test_empty_mask_agreement_conventions_are_explicit(self) -> None:
        empty = np.zeros((2, 2), dtype=np.uint8)

        both_empty = agreement_features(empty, empty)
        anomaly_empty = agreement_features(empty, np.eye(2, dtype=np.uint8))

        self.assertEqual(both_empty["mask_iou"], 1.0)
        self.assertEqual(both_empty["sam2_to_anomaly_area_ratio"], 1.0)
        self.assertEqual(anomaly_empty["mask_iou"], 0.0)
        self.assertTrue(np.isinf(anomaly_empty["sam2_to_anomaly_area_ratio"]))

    def test_selective_uses_intersection_only_for_acceptable_agreement(self) -> None:
        anomaly = np.array([[1, 1], [1, 0]], dtype=np.uint8)
        high_agreement = np.array([[1, 1], [0, 0]], dtype=np.uint8)
        low_agreement = np.array([[0, 0], [0, 1]], dtype=np.uint8)
        overexpanded = np.ones((2, 2), dtype=np.uint8)

        np.testing.assert_array_equal(
            fuse_masks(
                anomaly,
                high_agreement,
                mode="selective",
                min_iou=0.5,
                max_expansion=2.0,
            ),
            high_agreement,
        )
        np.testing.assert_array_equal(
            fuse_masks(anomaly, low_agreement, mode="selective", min_iou=0.25),
            anomaly,
        )
        np.testing.assert_array_equal(
            fuse_masks(
                np.array([[1, 0], [0, 0]], dtype=np.uint8),
                overexpanded,
                mode="selective",
                min_iou=0.25,
                max_expansion=2.0,
            ),
            np.array([[1, 0], [0, 0]], dtype=np.uint8),
        )
        self.assertEqual(
            fusion_source(anomaly, high_agreement, mode="selective", min_iou=0.5),
            "intersection",
        )
        self.assertEqual(
            fusion_source(anomaly, low_agreement, mode="selective", min_iou=0.25),
            "anomaly_fallback",
        )

    def test_invalid_inputs_and_controls_are_rejected(self) -> None:
        valid = np.zeros((2, 2), dtype=np.uint8)
        cases = [
            ((np.zeros((1, 2, 2)), valid), {"mode": "anomaly"}, "2-D"),
            ((valid, np.zeros((3, 2))), {"mode": "anomaly"}, "identical"),
            ((valid, valid), {"mode": "weighted"}, "mode"),
            ((valid, valid), {"mode": "selective", "min_iou": -0.1}, "min_iou"),
            ((valid, valid), {"mode": "selective", "min_iou": 1.1}, "min_iou"),
            ((valid, valid), {"mode": "selective", "max_expansion": 0.0}, "max_expansion"),
            ((valid, valid), {"mode": "selective", "max_expansion": np.inf}, "max_expansion"),
        ]

        for inputs, kwargs, message in cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, message):
                    fuse_masks(*inputs, **kwargs)

    def test_non_finite_numeric_masks_are_rejected_before_binary_conversion(self) -> None:
        finite = np.zeros((2, 2), dtype=np.float32)

        for value in (np.nan, np.inf, -np.inf):
            for position in ("anomaly", "sam2"):
                with self.subTest(value=value, position=position):
                    invalid = finite.copy()
                    invalid[0, 0] = value
                    inputs = (invalid, finite) if position == "anomaly" else (finite, invalid)
                    with self.assertRaisesRegex(ValueError, rf"{position}_mask.*finite"):
                        fuse_masks(*inputs, mode="union")


class FuseSavedMasksScriptTest(unittest.TestCase):
    def test_offline_intersection_uses_portable_paths_from_different_cwd(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_dir = root / "source"
            output_dir = root / "fused"
            other_cwd = root / "other"
            source_dir.mkdir()
            other_cwd.mkdir()
            heatmap = np.array(
                [[0.9, 0.9, 0.1], [0.9, 0.9, 0.1], [0.1, 0.1, 0.1]],
                dtype=np.float32,
            )
            np.save(source_dir / "heatmap.npy", heatmap)
            Image.fromarray(np.ones((3, 3), dtype=np.uint8) * 255, mode="L").save(
                source_dir / "raw.png"
            )
            calibration_payload = {
                "quantile": 0.995,
                "threshold": 0.5,
                "num_images": 2,
                "num_pixels": 18,
                "source_split": "val",
            }
            calibration_bytes = json.dumps(calibration_payload).encode()
            calibration_sha256 = hashlib.sha256(calibration_bytes).hexdigest()
            scores_path = source_dir / "mask_scores.csv"
            _write_csv(
                scores_path,
                [
                    "sample_id",
                    "category",
                    "label",
                    "sam2_mask_path",
                    "heatmap_path",
                    "sam2_prompt_threshold",
                    "sam2_calibration_sha256",
                ],
                [
                    {
                        "sample_id": "pcb1/anomaly",
                        "category": "pcb1",
                        "label": "1",
                        "sam2_mask_path": "raw.png",
                        "heatmap_path": "heatmap.npy",
                        "sam2_prompt_threshold": "0.50000000",
                        "sam2_calibration_sha256": calibration_sha256,
                    }
                ],
            )
            calibration_path = root / "calibration.json"
            calibration_path.write_bytes(calibration_bytes)

            result = subprocess.run(
                [
                    sys.executable,
                    str(repo_root / "scripts" / "fuse_saved_masks.py"),
                    "--mask-scores-csv",
                    str(scores_path),
                    "--calibration-json",
                    str(calibration_path),
                    "--mask-output",
                    "intersection",
                    "--output-dir",
                    str(output_dir),
                ],
                check=False,
                cwd=other_cwd,
                env={"PYTHONPATH": str(repo_root / "src")},
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            rows = _read_csv(output_dir / "mask_scores.csv")
            pred_path = output_dir / rows[0]["pred_mask_path"]
            pred = np.asarray(Image.open(pred_path).convert("L")) > 0

        self.assertEqual(int(pred.sum()), 4)
        self.assertEqual(rows[0]["mask_output"], "intersection")
        self.assertEqual(rows[0]["selected_source"], "intersection")
        self.assertEqual(rows[0]["proposal_threshold"], "0.50000000")
        self.assertEqual(rows[0]["sam2_prompt_threshold"], "0.50000000")
        self.assertEqual(rows[0]["sam2_calibration_sha256"], calibration_sha256)
        self.assertEqual(rows[0]["calibration_mismatch_override"], "0")
        self.assertEqual(rows[0]["intersection_pixels"], "4.00000000")
        self.assertFalse(Path(rows[0]["sam2_mask_path"]).is_absolute())
        self.assertFalse(Path(rows[0]["heatmap_path"]).is_absolute())
        self.assertFalse(Path(rows[0]["pred_mask_path"]).is_absolute())

    def test_missing_required_artifacts_fail_with_row_context(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            calibration_path = root / "calibration.json"
            calibration_path.write_text(
                json.dumps(
                    {
                        "quantile": 0.995,
                        "threshold": 0.5,
                        "num_images": 1,
                        "num_pixels": 4,
                        "source_split": "val",
                    }
                )
            )
            for missing_field in ("sam2_mask_path", "heatmap_path"):
                with self.subTest(missing_field=missing_field):
                    scores_path = root / f"missing_{missing_field}.csv"
                    row = {
                        "sample_id": "pcb1/anomaly",
                        "sam2_mask_path": "raw.png",
                        "heatmap_path": "heatmap.npy",
                    }
                    row[missing_field] = ""
                    _write_csv(scores_path, list(row), [row])
                    result = subprocess.run(
                        [
                            sys.executable,
                            str(repo_root / "scripts" / "fuse_saved_masks.py"),
                            "--mask-scores-csv",
                            str(scores_path),
                            "--calibration-json",
                            str(calibration_path),
                            "--mask-output",
                            "anomaly",
                            "--output-dir",
                            str(root / "out"),
                        ],
                        check=False,
                        cwd=root,
                        env={"PYTHONPATH": str(repo_root / "src")},
                        capture_output=True,
                        text=True,
                    )

                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("row 0", result.stderr)
                    self.assertIn(missing_field, result.stderr)

    def test_offline_rejects_invalid_heatmaps_with_row_and_path_context(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        invalid_cases = [
            (np.array([[np.nan]], dtype=np.float32), "finite"),
            (np.array([[np.inf]], dtype=np.float32), "finite"),
            (np.array([[-np.inf]], dtype=np.float32), "finite"),
            (np.zeros((1, 1, 1), dtype=np.float32), "2-D"),
            (np.empty((0, 1), dtype=np.float32), "non-empty"),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for index, (heatmap, expected) in enumerate(invalid_cases):
                with self.subTest(expected=expected, index=index):
                    case_dir = root / str(index)
                    scores_path, calibration_path = _write_offline_fixture(
                        case_dir,
                        heatmap=heatmap,
                    )
                    result = _run_offline_fusion(
                        repo_root,
                        scores_path,
                        calibration_path,
                        case_dir / "output",
                    )

                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("row 0", result.stderr)
                    self.assertIn("heatmap.npy", result.stderr)
                    self.assertIn(expected, result.stderr)

    def test_offline_rejects_threshold_and_hash_identity_mismatches(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            threshold_dir = root / "threshold"
            source_scores_path = _write_refinement_source_fixture(threshold_dir)
            raw_calibration_path = threshold_dir / "raw_calibration.json"
            raw_calibration_path.write_bytes(_calibration_bytes(threshold=0.8))
            raw_output_dir = threshold_dir / "raw"
            raw_result = subprocess.run(
                [
                    sys.executable,
                    str(repo_root / "scripts" / "run_mask_refinement.py"),
                    "--scores-csv",
                    str(source_scores_path),
                    "--output-dir",
                    str(raw_output_dir),
                    "--calibration-json",
                    str(raw_calibration_path),
                    "--refiner",
                    "fallback",
                    "--min-area",
                    "1",
                ],
                check=False,
                cwd=repo_root,
                env={"PYTHONPATH": str(repo_root / "src")},
                capture_output=True,
                text=True,
            )
            self.assertEqual(raw_result.returncode, 0, msg=raw_result.stderr)
            supplied_calibration_path = threshold_dir / "supplied_calibration.json"
            supplied_calibration_path.write_bytes(_calibration_bytes(threshold=0.5))
            threshold_result = _run_offline_fusion(
                repo_root,
                raw_output_dir / "mask_scores.csv",
                supplied_calibration_path,
                threshold_dir / "output",
            )

            different_same_threshold_bytes = json.dumps(
                json.loads(_calibration_bytes(threshold=0.5)),
                indent=2,
            ).encode()
            hash_dir = root / "hash"
            scores_path, calibration_path = _write_offline_fixture(
                hash_dir,
                heatmap=np.ones((2, 2), dtype=np.float32),
                calibration_threshold=0.5,
                identity_threshold=0.5,
                identity_sha256=hashlib.sha256(
                    different_same_threshold_bytes
                ).hexdigest(),
            )
            hash_result = _run_offline_fusion(
                repo_root,
                scores_path,
                calibration_path,
                hash_dir / "output",
            )

        self.assertNotEqual(threshold_result.returncode, 0)
        self.assertIn("row 0", threshold_result.stderr)
        self.assertIn("sam2_prompt_threshold", threshold_result.stderr)
        self.assertNotEqual(hash_result.returncode, 0)
        self.assertIn("row 0", hash_result.stderr)
        self.assertIn("sam2_calibration_sha256", hash_result.stderr)

    def test_legacy_identity_requires_explicit_override_and_preserves_missing_fields(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            scores_path, calibration_path = _write_offline_fixture(
                root,
                heatmap=np.ones((2, 2), dtype=np.float32),
                include_identity=False,
            )
            rejected = _run_offline_fusion(
                repo_root,
                scores_path,
                calibration_path,
                root / "rejected",
            )
            accepted = _run_offline_fusion(
                repo_root,
                scores_path,
                calibration_path,
                root / "accepted",
                allow_mismatch=True,
            )

            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("sam2_prompt_threshold", rejected.stderr)
            self.assertEqual(accepted.returncode, 0, msg=accepted.stderr)
            rows = _read_csv(root / "accepted" / "mask_scores.csv")

        self.assertEqual(rows[0]["sam2_prompt_threshold"], "")
        self.assertEqual(rows[0]["sam2_calibration_sha256"], "")
        self.assertEqual(rows[0]["calibration_mismatch_override"], "1")
        self.assertEqual(rows[0]["selected_source"], "anomaly")

    def test_header_only_csv_fails_before_creating_output_artifacts(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            scores_path = root / "mask_scores.csv"
            _write_csv(
                scores_path,
                [
                    "sample_id",
                    "sam2_mask_path",
                    "heatmap_path",
                    "sam2_prompt_threshold",
                    "sam2_calibration_sha256",
                ],
                [],
            )
            calibration_path = root / "calibration.json"
            calibration_path.write_bytes(_calibration_bytes())
            output_dir = root / "output"

            result = _run_offline_fusion(
                repo_root,
                scores_path,
                calibration_path,
                output_dir,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("must contain at least one row", result.stderr)
            self.assertFalse(output_dir.exists())

    def test_calibration_json_is_required(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [
                sys.executable,
                str(repo_root / "scripts" / "fuse_saved_masks.py"),
                "--mask-scores-csv",
                "unused.csv",
                "--mask-output",
                "anomaly",
                "--output-dir",
                "unused",
            ],
            check=False,
            cwd=repo_root,
            env={"PYTHONPATH": str(repo_root / "src")},
            capture_output=True,
            text=True,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--calibration-json", result.stderr)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _calibration_bytes(threshold: float = 0.5) -> bytes:
    return json.dumps(
        {
            "quantile": 0.995,
            "threshold": threshold,
            "num_images": 2,
            "num_pixels": 8,
            "source_split": "val",
        }
    ).encode()


def _write_offline_fixture(
    root: Path,
    heatmap: np.ndarray,
    calibration_threshold: float = 0.5,
    identity_threshold: float | None = None,
    identity_sha256: str | None = None,
    include_identity: bool = True,
) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    np.save(root / "heatmap.npy", heatmap)
    shape = heatmap.shape if heatmap.ndim == 2 and heatmap.size else (1, 1)
    Image.fromarray(np.ones(shape, dtype=np.uint8) * 255, mode="L").save(root / "raw.png")
    calibration_bytes = _calibration_bytes(calibration_threshold)
    calibration_path = root / "calibration.json"
    calibration_path.write_bytes(calibration_bytes)
    row = {
        "sample_id": "pcb1/anomaly",
        "sam2_mask_path": "raw.png",
        "heatmap_path": "heatmap.npy",
    }
    if include_identity:
        stored_threshold = (
            calibration_threshold if identity_threshold is None else identity_threshold
        )
        row["sam2_prompt_threshold"] = f"{stored_threshold:.8f}"
        row["sam2_calibration_sha256"] = identity_sha256 or hashlib.sha256(
            calibration_bytes
        ).hexdigest()
    scores_path = root / "mask_scores.csv"
    _write_csv(scores_path, list(row), [row])
    return scores_path, calibration_path


def _write_refinement_source_fixture(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    image_path = root / "image.png"
    heatmap_path = root / "heatmap.npy"
    Image.new("RGB", (2, 2), (20, 40, 60)).save(image_path)
    np.save(heatmap_path, np.ones((2, 2), dtype=np.float32))
    scores_path = root / "scores.csv"
    _write_csv(
        scores_path,
        ["sample_id", "category", "label", "image_path", "mask_path", "heatmap_path"],
        [
            {
                "sample_id": "pcb1/anomaly",
                "category": "pcb1",
                "label": "1",
                "image_path": str(image_path),
                "mask_path": "",
                "heatmap_path": str(heatmap_path),
            }
        ],
    )
    return scores_path


def _run_offline_fusion(
    repo_root: Path,
    scores_path: Path,
    calibration_path: Path,
    output_dir: Path,
    allow_mismatch: bool = False,
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(repo_root / "scripts" / "fuse_saved_masks.py"),
        "--mask-scores-csv",
        str(scores_path),
        "--calibration-json",
        str(calibration_path),
        "--mask-output",
        "anomaly",
        "--output-dir",
        str(output_dir),
    ]
    if allow_mismatch:
        command.append("--allow-calibration-mismatch")
    return subprocess.run(
        command,
        check=False,
        cwd=repo_root,
        env={"PYTHONPATH": str(repo_root / "src")},
        capture_output=True,
        text=True,
    )


if __name__ == "__main__":
    unittest.main()
