from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from anomaly.heatmap import normalize_heatmap, resize_heatmap_to_image
from anomaly.memory_bank import build_memory_bank, score_patch_features
from features.dinov2 import PatchFeatureMap
from scripts.run_dinov2_baseline import select_rows
from utils.synthetic_data import create_synthetic_debug_datasets


class MemoryBankTest(unittest.TestCase):
    def test_nearest_neighbor_scores_patch_features(self) -> None:
        support = [
            PatchFeatureMap(
                features=np.array([[[0.0, 0.0], [1.0, 0.0]]], dtype=np.float32),
                image_size=(2, 1),
                patch_size=1,
            )
        ]
        query = PatchFeatureMap(
            features=np.array([[[0.0, 0.0], [2.0, 0.0]]], dtype=np.float32),
            image_size=(2, 1),
            patch_size=1,
        )

        bank = build_memory_bank(support, normalize=False)
        scores = score_patch_features(query, bank, normalize=False)

        self.assertTrue(np.allclose(scores, np.array([[0.0, 1.0]], dtype=np.float32)))

    def test_normalized_heatmap_resizes_to_image_size(self) -> None:
        heatmap = np.array([[0.0, 2.0], [1.0, 3.0]], dtype=np.float32)

        normalized = normalize_heatmap(heatmap)
        resized = resize_heatmap_to_image(normalized, image_size=(8, 4))

        self.assertEqual(resized.shape, (4, 8))
        self.assertGreaterEqual(float(resized.min()), 0.0)
        self.assertLessEqual(float(resized.max()), 1.0)

    def test_absolute_heatmap_resize_preserves_scores_above_one(self) -> None:
        heatmap = np.array([[0.0, 2.0], [4.0, 6.0]], dtype=np.float32)

        resized = resize_heatmap_to_image(heatmap, image_size=(4, 4), normalize=False)

        self.assertEqual(resized.shape, (4, 4))
        self.assertGreater(float(resized.max()), 5.0)
        self.assertLessEqual(float(resized.max()), 6.0)


class DINOv2BaselineScriptTest(unittest.TestCase):
    def test_select_rows_interleaves_test_normals_and_anomalies_for_small_limits(self) -> None:
        rows = _tiny_manifest_rows_for_selection()

        _, query_rows = select_rows(
            rows=rows,
            fold_id=0,
            category="pcb1",
            k=2,
            query_fold_split="test",
            limit=2,
            seed=4880,
        )

        self.assertEqual([row["label"] for row in query_rows], ["1", "0"])

    def test_script_writes_color_patch_smoke_outputs(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            fixture = create_synthetic_debug_datasets(tmp_path / "fixtures")
            manifest_path = tmp_path / "visa_pcb_folds.csv"
            _write_tiny_visa_fold_manifest(fixture.visa_root, manifest_path)
            output_dir = tmp_path / "outputs"

            result = subprocess.run(
                [
                    sys.executable,
                    str(repo_root / "scripts" / "run_dinov2_baseline.py"),
                    "--manifest",
                    str(manifest_path),
                    "--fold-id",
                    "0",
                    "--category",
                    "pcb1",
                    "--k",
                    "2",
                    "--limit",
                    "1",
                    "--query-fold-split",
                    "test",
                    "--feature-backbone",
                    "color_patch",
                    "--image-size",
                    "56",
                    "--patch-size",
                    "14",
                    "--output-dir",
                    str(output_dir),
                ],
                check=False,
                cwd=repo_root,
                env={"PYTHONPATH": str(repo_root / "src")},
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            output_files = sorted(path.name for path in output_dir.glob("*.png"))
            scores_path = output_dir / "scores.csv"
            rows = _read_csv(scores_path)
            provenance = json.loads(
                (output_dir / "memory_bank_provenance.json").read_text()
            )
            self.assertFalse(Path(rows[0]["heatmap_path"]).is_absolute())
            self.assertFalse(Path(rows[0]["image_path"]).is_absolute())
            self.assertFalse(Path(rows[0]["mask_path"]).is_absolute())
            self.assertTrue((scores_path.parent / rows[0]["heatmap_path"]).is_file())
            self.assertTrue((scores_path.parent / rows[0]["image_path"]).is_file())
            self.assertTrue((scores_path.parent / rows[0]["mask_path"]).is_file())

        self.assertEqual(output_files, ["000_pcb1_anomaly_000.png"])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["sample_id"], "pcb1/anomaly_000")
        self.assertEqual(rows[0]["fold_split"], "test")
        self.assertGreater(float(rows[0]["image_score"]), 0.0)
        self.assertEqual(provenance["feature_backbone"], "color_patch")
        self.assertFalse(provenance["coreset_applied"])
        self.assertEqual(provenance["coreset_ratio"], 0.01)
        self.assertEqual(provenance["coreset_projection_dim"], 64)
        self.assertEqual(provenance["coreset_seed"], 4880)
        self.assertEqual(
            provenance["memory_bank_size_used"],
            provenance["memory_bank_size_full"],
        )

    def test_debug_limit_one_keeps_all_heatmaps_and_bounds_panels(self) -> None:
        rows, png_names = _run_debug_limit_fixture(debug_limit=1)

        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["heatmap_path"] for row in rows))
        self.assertTrue(all(row["debug_path"] == "" for row in rows[1:]))
        self.assertNotEqual(rows[0]["debug_path"], "")
        self.assertEqual(png_names, ["000_pcb1_anomaly_000.png"])

    def test_debug_limit_zero_keeps_all_heatmaps_and_skips_panels(self) -> None:
        rows, png_names = _run_debug_limit_fixture(debug_limit=0)

        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["heatmap_path"] for row in rows))
        self.assertTrue(all(row["debug_path"] == "" for row in rows))
        self.assertEqual(png_names, [])

    def test_portable_scores_work_for_calibration_and_evaluation_from_other_cwd(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            generation_cwd = root / "generation"
            consumer_cwd = root / "consumer"
            generation_cwd.mkdir()
            consumer_cwd.mkdir()
            fixture = create_synthetic_debug_datasets(root / "fixtures")
            manifest_path = root / "visa_pcb_folds.csv"
            _write_tiny_visa_fold_manifest(fixture.visa_root, manifest_path)

            baseline = subprocess.run(
                [
                    sys.executable,
                    str(repo_root / "scripts" / "run_dinov2_baseline.py"),
                    "--manifest",
                    str(manifest_path),
                    "--fold-id",
                    "0",
                    "--category",
                    "pcb1",
                    "--k",
                    "2",
                    "--limit",
                    "1",
                    "--query-fold-split",
                    "val",
                    "--feature-backbone",
                    "color_patch",
                    "--image-size",
                    "56",
                    "--patch-size",
                    "14",
                    "--output-dir",
                    "portable_outputs",
                ],
                check=False,
                cwd=generation_cwd,
                env={"PYTHONPATH": str(repo_root / "src")},
                capture_output=True,
                text=True,
            )
            self.assertEqual(baseline.returncode, 0, msg=baseline.stderr)
            scores_path = generation_cwd / "portable_outputs" / "scores.csv"
            calibration_path = root / "calibration.json"
            calibration = subprocess.run(
                [
                    sys.executable,
                    str(repo_root / "scripts" / "calibrate_heatmaps.py"),
                    "--scores-csv",
                    str(scores_path),
                    "--quantile",
                    "0.75",
                    "--output-json",
                    str(calibration_path),
                ],
                check=False,
                cwd=consumer_cwd,
                env={"PYTHONPATH": str(repo_root / "src")},
                capture_output=True,
                text=True,
            )
            self.assertEqual(calibration.returncode, 0, msg=calibration.stderr)
            metrics_path = root / "metrics.json"
            evaluation = subprocess.run(
                [
                    sys.executable,
                    str(repo_root / "scripts" / "evaluate_heatmaps.py"),
                    "--scores-csv",
                    str(scores_path),
                    "--output-json",
                    str(metrics_path),
                    "--calibration-json",
                    str(calibration_path),
                ],
                check=False,
                cwd=consumer_cwd,
                env={"PYTHONPATH": str(repo_root / "src")},
                capture_output=True,
                text=True,
            )
            self.assertEqual(evaluation.returncode, 0, msg=evaluation.stderr)
            metrics = json.loads(metrics_path.read_text())

        self.assertEqual(metrics["calibration_source_split"], "val")


def _write_tiny_visa_fold_manifest(visa_root: Path, output_path: Path) -> None:
    fieldnames = [
        "dataset",
        "category",
        "sample_id",
        "split",
        "image_path",
        "label",
        "mask_path",
        "box_path",
        "template_path",
        "metadata_json",
        "fold_id",
        "fold_split",
    ]
    rows = []
    for index in range(3):
        rows.append(
            {
                "dataset": "visa_pcb",
                "category": "pcb1",
                "sample_id": f"pcb1/train_normal_{index}",
                "split": "train",
                "image_path": str(
                    visa_root / "pcb1" / "train" / "normal" / f"pcb1_train_normal_{index:03d}.png"
                ),
                "label": "0",
                "mask_path": "",
                "box_path": "",
                "template_path": "",
                "metadata_json": "{}",
                "fold_id": "0",
                "fold_split": "dev",
            }
        )
    rows.append(
        {
            "dataset": "visa_pcb",
            "category": "pcb1",
            "sample_id": "pcb1/anomaly_000",
            "split": "test",
            "image_path": str(visa_root / "pcb1" / "test" / "anomaly" / "pcb1_anomaly_000.png"),
            "label": "1",
            "mask_path": str(
                visa_root
                / "pcb1"
                / "ground_truth"
                / "anomaly"
                / "pcb1_anomaly_000.png"
            ),
            "box_path": "",
            "template_path": "",
            "metadata_json": "{}",
            "fold_id": "0",
            "fold_split": "test",
        }
    )
    rows.append(
        {
            "dataset": "visa_pcb",
            "category": "pcb1",
            "sample_id": "pcb1/test_normal_000",
            "split": "test",
            "image_path": str(
                visa_root / "pcb1" / "train" / "normal" / "pcb1_train_normal_001.png"
            ),
            "label": "0",
            "mask_path": "",
            "box_path": "",
            "template_path": "",
            "metadata_json": "{}",
            "fold_id": "0",
            "fold_split": "test",
        }
    )
    rows.append(
        {
            "dataset": "visa_pcb",
            "category": "pcb1",
            "sample_id": "pcb1/val_normal_000",
            "split": "train",
            "image_path": str(
                visa_root / "pcb1" / "train" / "normal" / "pcb1_train_normal_000.png"
            ),
            "label": "0",
            "mask_path": "",
            "box_path": "",
            "template_path": "",
            "metadata_json": "{}",
            "fold_id": "0",
            "fold_split": "val",
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _tiny_manifest_rows_for_selection() -> list[dict[str, str]]:
    rows = []
    for index in range(3):
        rows.append(
            {
                "dataset": "visa_pcb",
                "category": "pcb1",
                "sample_id": f"pcb1/train_normal_{index}",
                "split": "train",
                "image_path": f"train_{index}.png",
                "label": "0",
                "mask_path": "",
                "fold_id": "0",
                "fold_split": "dev",
            }
        )
    for label, prefix in (("1", "anomaly"), ("0", "normal")):
        for index in range(2):
            rows.append(
                {
                    "dataset": "visa_pcb",
                    "category": "pcb1",
                    "sample_id": f"pcb1/{prefix}_{index}",
                    "split": "test",
                    "image_path": f"{prefix}_{index}.png",
                    "label": label,
                    "mask_path": "",
                    "fold_id": "0",
                    "fold_split": "test",
                }
            )
    return rows


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _run_debug_limit_fixture(debug_limit: int) -> tuple[list[dict[str, str]], list[str]]:
    repo_root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        fixture = create_synthetic_debug_datasets(tmp_path / "fixtures")
        manifest_path = tmp_path / "visa_pcb_folds.csv"
        _write_tiny_visa_fold_manifest(fixture.visa_root, manifest_path)
        output_dir = tmp_path / "outputs"
        result = subprocess.run(
            [
                sys.executable,
                str(repo_root / "scripts" / "run_dinov2_baseline.py"),
                "--manifest",
                str(manifest_path),
                "--fold-id",
                "0",
                "--category",
                "pcb1",
                "--k",
                "2",
                "--limit",
                "2",
                "--query-fold-split",
                "test",
                "--feature-backbone",
                "color_patch",
                "--image-size",
                "56",
                "--patch-size",
                "14",
                "--debug-limit",
                str(debug_limit),
                "--output-dir",
                str(output_dir),
            ],
            check=False,
            cwd=repo_root,
            env={"PYTHONPATH": str(repo_root / "src")},
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise AssertionError(result.stderr)
        rows = _read_csv(output_dir / "scores.csv")
        heatmap_paths = [output_dir / row["heatmap_path"] for row in rows]
        if not all(path.is_file() for path in heatmap_paths):
            raise AssertionError("not all required heatmaps were written")
        png_names = sorted(path.name for path in output_dir.glob("*.png"))
    return rows, png_names


if __name__ == "__main__":
    unittest.main()
