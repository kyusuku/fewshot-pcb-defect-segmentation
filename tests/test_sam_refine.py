from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
from PIL import Image

import sam_refine.artifacts as artifacts
from sam_refine.prompts import PromptRegion, heatmap_to_prompt_regions
from sam_refine.refiner import FallbackMaskRefiner, SAM2MaskRefiner
from scripts.run_mask_refinement import build_refiner


class PromptGenerationTest(unittest.TestCase):
    def test_heatmap_to_prompt_regions_extracts_connected_boxes(self) -> None:
        heatmap = np.zeros((10, 10), dtype=np.float32)
        heatmap[3:6, 2:5] = 0.9
        heatmap[7:9, 7:9] = 0.7

        regions = heatmap_to_prompt_regions(
            heatmap,
            threshold=0.5,
            min_area=2,
            max_regions=4,
        )

        self.assertEqual(len(regions), 2)
        self.assertEqual(regions[0].box_xyxy, (2, 3, 5, 6))
        self.assertEqual(regions[0].point_xy, (2.0, 3.0))
        self.assertGreater(regions[0].score, regions[1].score)

    def test_anomaly_max_point_uses_off_center_component_maximum(self) -> None:
        heatmap = np.zeros((6, 7), dtype=np.float32)
        heatmap[1:5, 2:6] = 0.6
        heatmap[4, 5] = 1.0

        regions = heatmap_to_prompt_regions(heatmap, threshold=0.5, min_area=1)

        self.assertEqual(regions[0].point_xy, (5.0, 4.0))

    def test_anomaly_max_point_stays_inside_irregular_component(self) -> None:
        heatmap = np.zeros((5, 5), dtype=np.float32)
        heatmap[0:4, 0] = 0.6
        heatmap[3, 0:4] = 0.6
        heatmap[3, 3] = 1.0

        regions = heatmap_to_prompt_regions(heatmap, threshold=0.5, min_area=1)

        point_x, point_y = regions[0].point_xy
        self.assertEqual(regions[0].box_xyxy, (0, 0, 4, 4))
        self.assertEqual(regions[0].point_xy, (3.0, 3.0))
        self.assertGreaterEqual(heatmap[int(point_y), int(point_x)], 0.5)
        self.assertNotEqual(regions[0].point_xy, (2.0, 2.0))

    def test_anomaly_max_ties_use_topmost_then_leftmost_pixel(self) -> None:
        heatmap = np.zeros((4, 5), dtype=np.float32)
        heatmap[1:3, 1:4] = 0.6
        heatmap[1, 3] = 1.0
        heatmap[2, 1] = 1.0

        regions = heatmap_to_prompt_regions(heatmap, threshold=0.5, min_area=1)

        self.assertEqual(regions[0].point_xy, (3.0, 1.0))

    def test_box_center_mode_preserves_legacy_point(self) -> None:
        heatmap = np.zeros((10, 10), dtype=np.float32)
        heatmap[3:6, 2:5] = 0.9

        regions = heatmap_to_prompt_regions(
            heatmap,
            threshold=0.5,
            min_area=1,
            point_mode="box_center",
        )

        self.assertEqual(regions[0].point_xy, (3.5, 4.5))

    def test_invalid_point_mode_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "point_mode"):
            heatmap_to_prompt_regions(
                np.ones((2, 2), dtype=np.float32),
                point_mode="centroid",
            )


class FallbackMaskRefinerTest(unittest.TestCase):
    def test_fallback_refiner_thresholds_inside_prompt_boxes(self) -> None:
        image = Image.new("RGB", (10, 10), (0, 0, 0))
        heatmap = np.zeros((10, 10), dtype=np.float32)
        heatmap[3:6, 2:5] = 0.9
        regions = heatmap_to_prompt_regions(heatmap, threshold=0.5, min_area=2)

        predictions = FallbackMaskRefiner(threshold_fraction=0.5).refine(
            image=image,
            heatmap=heatmap,
            regions=regions,
        )

        self.assertEqual(len(predictions), 1)
        self.assertEqual(predictions[0].mask.shape, (10, 10))
        self.assertEqual(int(predictions[0].mask.sum()), 9)
        self.assertGreater(predictions[0].score, 0.0)


class SAM2MaskRefinerTest(unittest.TestCase):
    def test_empty_regions_do_not_require_loading_sam2(self) -> None:
        image = Image.new("RGB", (20, 30), (0, 0, 0))
        heatmap = np.zeros((10, 10), dtype=np.float32)

        predictions = SAM2MaskRefiner(
            checkpoint_path="weights/missing.pt",
            model_config="configs/missing.yaml",
        ).refine(image=image, heatmap=heatmap, regions=[])

        self.assertEqual(predictions, [])

    def test_sam2_refiner_scales_prompts_and_resizes_masks(self) -> None:
        image = Image.new("RGB", (20, 30), (0, 0, 0))
        heatmap = np.zeros((10, 10), dtype=np.float32)
        heatmap[3:6, 2:5] = 1.0
        region = PromptRegion(
            box_xyxy=(2, 3, 5, 6),
            point_xy=(3.5, 4.5),
            area=9,
            score=0.9,
        )
        predictor = _FakeSAM2Predictor()
        refiner = SAM2MaskRefiner(
            checkpoint_path="weights/fake.pt",
            model_config="configs/fake.yaml",
            prompt_mode="point_box",
        )
        refiner._predictor = predictor

        predictions = refiner.refine(image=image, heatmap=heatmap, regions=[region])

        self.assertEqual(predictor.image_shape, (30, 20, 3))
        self.assertTrue(predictor.image_writeable)
        self.assertEqual(len(predictor.calls), 1)
        call = predictor.calls[0]
        np.testing.assert_allclose(
            call["box"],
            np.asarray([4.0, 9.0, 10.0, 18.0], dtype=np.float32),
        )
        np.testing.assert_allclose(
            call["point_coords"],
            np.asarray([[7.0, 13.5]], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            call["point_labels"],
            np.asarray([1], dtype=np.int32),
        )
        self.assertTrue(call["multimask_output"])
        self.assertEqual(len(predictions), 1)
        self.assertEqual(predictions[0].mask.shape, heatmap.shape)
        self.assertEqual(int(predictions[0].mask.sum()), 9)
        self.assertGreater(predictions[0].score, 0.0)
        self.assertEqual(predictions[0].source, "sam2")

    def test_point_prompt_mode_omits_box(self) -> None:
        image, heatmap, region = _sam2_prompt_fixture()
        predictor = _FakeSAM2Predictor()
        refiner = SAM2MaskRefiner(
            checkpoint_path="weights/fake.pt",
            model_config="configs/fake.yaml",
            prompt_mode="point",
        )
        refiner._predictor = predictor

        refiner.refine(image=image, heatmap=heatmap, regions=[region])

        call = predictor.calls[0]
        self.assertIsNone(call["box"])
        np.testing.assert_allclose(
            call["point_coords"],
            np.asarray([[7.0, 13.5]], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            call["point_labels"],
            np.asarray([1], dtype=np.int32),
        )

    def test_box_prompt_mode_omits_point_coordinates_and_labels(self) -> None:
        image, heatmap, region = _sam2_prompt_fixture()
        predictor = _FakeSAM2Predictor()
        refiner = SAM2MaskRefiner(
            checkpoint_path="weights/fake.pt",
            model_config="configs/fake.yaml",
            prompt_mode="box",
        )
        refiner._predictor = predictor

        refiner.refine(image=image, heatmap=heatmap, regions=[region])

        call = predictor.calls[0]
        np.testing.assert_allclose(
            call["box"],
            np.asarray([4.0, 9.0, 10.0, 18.0], dtype=np.float32),
        )
        self.assertIsNone(call["point_coords"])
        self.assertIsNone(call["point_labels"])

    def test_invalid_prompt_mode_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "prompt_mode"):
            SAM2MaskRefiner(
                checkpoint_path="weights/fake.pt",
                model_config="configs/fake.yaml",
                prompt_mode="center",
            )

    def test_sam2_refiner_prefers_anomaly_aligned_mask_over_larger_sam2_mask(self) -> None:
        image = Image.new("RGB", (20, 30), (0, 0, 0))
        heatmap = np.zeros((10, 10), dtype=np.float32)
        heatmap[3:6, 2:5] = 1.0
        region = PromptRegion(
            box_xyxy=(2, 3, 5, 6),
            point_xy=(3.5, 4.5),
            area=9,
            score=0.9,
        )
        refiner = SAM2MaskRefiner(
            checkpoint_path="weights/fake.pt",
            model_config="configs/fake.yaml",
        )
        refiner._predictor = _OversizedSAM2Predictor()

        predictions = refiner.refine(image=image, heatmap=heatmap, regions=[region])

        self.assertEqual(len(predictions), 1)
        self.assertEqual(int(predictions[0].mask.sum()), 9)
        np.testing.assert_array_equal(predictions[0].mask, (heatmap > 0).astype(np.uint8))

    def test_sam2_refiner_rejects_masks_above_max_area_fraction(self) -> None:
        image = Image.new("RGB", (10, 10), (0, 0, 0))
        heatmap = np.ones((10, 10), dtype=np.float32)
        region = PromptRegion(
            box_xyxy=(0, 0, 10, 10),
            point_xy=(5.0, 5.0),
            area=100,
            score=1.0,
        )
        refiner = SAM2MaskRefiner(
            checkpoint_path="weights/fake.pt",
            model_config="configs/fake.yaml",
            max_mask_area_fraction=0.5,
        )
        refiner._predictor = _HighConfidenceFullMaskPredictor()

        predictions = refiner.refine(image=image, heatmap=heatmap, regions=[region])

        self.assertEqual(len(predictions), 1)
        self.assertEqual(int(predictions[0].mask.sum()), 9)


class MaskRefinementScriptTest(unittest.TestCase):
    def test_build_refiner_passes_max_mask_area_fraction_to_sam2(self) -> None:
        refiner = build_refiner(
            Namespace(
                refiner="sam2",
                sam2_checkpoint=Path("weights/fake.pt"),
                sam2_model_config="configs/fake.yaml",
                device="cpu",
                max_mask_area_fraction=0.25,
                fallback_threshold_fraction=0.5,
                prompt_mode="box",
            )
        )

        self.assertIsInstance(refiner, SAM2MaskRefiner)
        self.assertEqual(refiner.max_mask_area_fraction, 0.25)
        self.assertEqual(refiner.prompt_mode, "box")

    def test_script_writes_mask_scores_and_predicted_masks(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            image_path = tmp_path / "image.png"
            heatmap_path = tmp_path / "heatmap.npy"
            mask_path = tmp_path / "mask.png"
            scores_path = tmp_path / "scores.csv"
            output_dir = tmp_path / "refined"
            Image.new("RGB", (10, 10), (20, 40, 60)).save(image_path)
            heatmap = np.zeros((10, 10), dtype=np.float32)
            heatmap[3:6, 2:5] = 0.9
            np.save(heatmap_path, heatmap)
            mask = np.zeros((10, 10), dtype=np.uint8)
            mask[3:6, 2:5] = 255
            Image.fromarray(mask, mode="L").save(mask_path)
            with scores_path.open("w", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "sample_id",
                        "category",
                        "label",
                        "image_score",
                        "image_path",
                        "mask_path",
                        "heatmap_path",
                        "debug_path",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "sample_id": "pcb1/anomaly",
                        "category": "pcb1",
                        "label": "1",
                        "image_score": "0.9",
                        "image_path": str(image_path),
                        "mask_path": str(mask_path),
                        "heatmap_path": str(heatmap_path),
                        "debug_path": "",
                    }
                )

            result = subprocess.run(
                [
                    sys.executable,
                    str(repo_root / "scripts" / "run_mask_refinement.py"),
                    "--scores-csv",
                    str(scores_path),
                    "--output-dir",
                    str(output_dir),
                    "--threshold",
                    "0.5",
                    "--refiner",
                    "fallback",
                ],
                check=False,
                cwd=repo_root,
                env={"PYTHONPATH": str(repo_root / "src")},
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            rows = _read_csv(output_dir / "mask_scores.csv")
            pred_mask_path = output_dir / rows[0]["pred_mask_path"]
            sam2_mask_path = output_dir / rows[0]["sam2_mask_path"]
            pred_mask = np.asarray(Image.open(pred_mask_path).convert("L")) > 0
            sam2_mask = np.asarray(Image.open(sam2_mask_path).convert("L")) > 0

        self.assertEqual(len(rows), 1)
        self.assertEqual(int(pred_mask.sum()), 9)
        np.testing.assert_array_equal(pred_mask, sam2_mask)
        self.assertEqual(rows[0]["mask_output"], "sam2")
        self.assertEqual(rows[0]["prompt_mode"], "point_box")
        self.assertEqual(rows[0]["point_mode"], "anomaly_max")
        self.assertEqual(rows[0]["proposal_threshold"], "0.50000000")
        self.assertEqual(float(rows[0]["sam2_prompt_threshold"]), 0.5)
        self.assertEqual(rows[0]["sam2_calibration_sha256"], "")
        self.assertEqual(rows[0]["selected_source"], "sam2")
        self.assertEqual(rows[0]["calibration_mismatch_override"], "0")
        self.assertEqual(rows[0]["refiner"], "fallback")
        self.assertEqual(rows[0]["raw_mask_source"], "fallback")
        self.assertEqual(rows[0]["sam2_model_config"], "")
        self.assertEqual(rows[0]["sam2_checkpoint_sha256"], "")
        self.assertFalse(Path(rows[0]["mask_path"]).is_absolute())
        self.assertFalse(Path(rows[0]["heatmap_path"]).is_absolute())
        self.assertFalse(Path(rows[0]["sam2_mask_path"]).is_absolute())
        self.assertFalse(Path(rows[0]["pred_mask_path"]).is_absolute())

    def test_sam2_raw_mask_provenance_hashes_checkpoint_and_records_model_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "sam2.pt"
            checkpoint_path.write_bytes(b"fake-sam2-checkpoint")

            provenance = artifacts.raw_mask_provenance(
                refiner="sam2",
                sam2_model_config="sam2_hiera_s.yaml",
                sam2_checkpoint=checkpoint_path,
            )

        self.assertEqual(provenance["refiner"], "sam2")
        self.assertEqual(provenance["raw_mask_source"], "sam2")
        self.assertEqual(provenance["sam2_model_config"], "sam2_hiera_s.yaml")
        self.assertEqual(
            provenance["sam2_checkpoint_sha256"],
            hashlib.sha256(b"fake-sam2-checkpoint").hexdigest(),
        )

    def test_intersection_saves_raw_sam2_and_selected_mask_with_stable_fields(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_path, heatmap_path, mask_path, scores_path = _write_refinement_fixture(root)
            output_dir = root / "refined"

            result = subprocess.run(
                [
                    sys.executable,
                    str(repo_root / "scripts" / "run_mask_refinement.py"),
                    "--scores-csv",
                    str(scores_path),
                    "--output-dir",
                    str(output_dir),
                    "--threshold",
                    "0.8",
                    "--mask-output",
                    "intersection",
                    "--point-mode",
                    "box_center",
                    "--prompt-mode",
                    "box",
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

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            rows = _read_csv(output_dir / "mask_scores.csv")
            raw = np.asarray(Image.open(output_dir / rows[0]["sam2_mask_path"]).convert("L")) > 0
            selected = (
                np.asarray(Image.open(output_dir / rows[0]["pred_mask_path"]).convert("L")) > 0
            )
            portable_source_paths_resolve = (
                (output_dir / rows[0]["image_path"]).samefile(image_path)
                and (output_dir / rows[0]["mask_path"]).samefile(mask_path)
                and (output_dir / rows[0]["heatmap_path"]).samefile(heatmap_path)
            )

        self.assertEqual(int(raw.sum()), 9)
        self.assertEqual(int(selected.sum()), 8)
        self.assertEqual(rows[0]["mask_output"], "intersection")
        self.assertEqual(rows[0]["prompt_mode"], "box")
        self.assertEqual(rows[0]["point_mode"], "box_center")
        self.assertEqual(rows[0]["selected_source"], "intersection")
        self.assertEqual(float(rows[0]["sam2_prompt_threshold"]), 0.8)
        self.assertEqual(rows[0]["sam2_calibration_sha256"], "")
        self.assertEqual(rows[0]["anomaly_pixels"], "8.00000000")
        self.assertEqual(rows[0]["sam2_pixels"], "9.00000000")
        self.assertEqual(rows[0]["intersection_pixels"], "8.00000000")
        self.assertEqual(rows[0]["union_pixels"], "9.00000000")
        self.assertAlmostEqual(float(rows[0]["mask_iou"]), 8.0 / 9.0)
        self.assertAlmostEqual(float(rows[0]["sam2_to_anomaly_area_ratio"]), 9.0 / 8.0)
        self.assertEqual(rows[0]["selective_min_iou"], "0.25000000")
        self.assertEqual(rows[0]["selective_max_expansion"], "2.00000000")
        self.assertTrue(portable_source_paths_resolve)

    def test_calibration_threshold_overrides_explicit_threshold_and_writes_provenance(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _, _, _, scores_path = _write_refinement_fixture(root)
            output_dir = root / "refined"
            calibration_path = root / "calibration.json"
            calibration_path.write_text(
                json.dumps(
                    {
                        "quantile": 0.995,
                        "threshold": 0.8,
                        "num_images": 3,
                        "num_pixels": 300,
                        "source_split": "val",
                    }
                )
            )
            expected_calibration_sha256 = hashlib.sha256(calibration_path.read_bytes()).hexdigest()

            result = subprocess.run(
                [
                    sys.executable,
                    str(repo_root / "scripts" / "run_mask_refinement.py"),
                    "--scores-csv",
                    str(scores_path),
                    "--output-dir",
                    str(output_dir),
                    "--calibration-json",
                    str(calibration_path),
                    "--threshold",
                    "0.95",
                    "--percentile",
                    "10",
                    "--mask-output",
                    "anomaly",
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

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            rows = _read_csv(output_dir / "mask_scores.csv")
            selected = (
                np.asarray(Image.open(output_dir / rows[0]["pred_mask_path"]).convert("L")) > 0
            )

        self.assertEqual(int(selected.sum()), 8)
        self.assertEqual(rows[0]["proposal_threshold"], "0.80000000")
        self.assertEqual(rows[0]["calibration_quantile"], "0.99500000")
        self.assertEqual(rows[0]["calibration_threshold"], "0.80000000")
        self.assertEqual(rows[0]["calibration_source_split"], "val")
        self.assertEqual(rows[0]["calibration_num_images"], "3")
        self.assertEqual(rows[0]["calibration_num_pixels"], "300")
        self.assertEqual(
            rows[0]["sam2_calibration_sha256"],
            expected_calibration_sha256,
        )
        self.assertEqual(float(rows[0]["sam2_prompt_threshold"]), 0.8)

    def test_script_rejects_invalid_heatmaps_with_row_and_path_context(self) -> None:
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
                    case_dir.mkdir()
                    _, heatmap_path, _, scores_path = _write_refinement_fixture(case_dir)
                    np.save(heatmap_path, heatmap)
                    result = subprocess.run(
                        [
                            sys.executable,
                            str(repo_root / "scripts" / "run_mask_refinement.py"),
                            "--scores-csv",
                            str(scores_path),
                            "--output-dir",
                            str(case_dir / "output"),
                            "--threshold",
                            "0.5",
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

                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("row 0", result.stderr)
                    self.assertIn("heatmap.npy", result.stderr)
                    self.assertIn(expected, result.stderr)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write_refinement_fixture(root: Path) -> tuple[Path, Path, Path, Path]:
    image_path = root / "image.png"
    heatmap_path = root / "heatmap.npy"
    mask_path = root / "mask.png"
    scores_path = root / "scores.csv"
    Image.new("RGB", (3, 3), (20, 40, 60)).save(image_path)
    heatmap = np.full((3, 3), 0.9, dtype=np.float32)
    heatmap[1, 1] = 0.5
    np.save(heatmap_path, heatmap)
    Image.fromarray((heatmap >= 0.8).astype(np.uint8) * 255, mode="L").save(mask_path)
    with scores_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "category",
                "label",
                "image_score",
                "image_path",
                "mask_path",
                "heatmap_path",
                "debug_path",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "sample_id": "pcb1/anomaly",
                "category": "pcb1",
                "label": "1",
                "image_score": "0.9",
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "heatmap_path": str(heatmap_path),
                "debug_path": "",
            }
        )
    return image_path, heatmap_path, mask_path, scores_path


def _sam2_prompt_fixture() -> tuple[Image.Image, np.ndarray, PromptRegion]:
    image = Image.new("RGB", (20, 30), (0, 0, 0))
    heatmap = np.zeros((10, 10), dtype=np.float32)
    heatmap[3:6, 2:5] = 1.0
    region = PromptRegion(
        box_xyxy=(2, 3, 5, 6),
        point_xy=(3.5, 4.5),
        area=9,
        score=0.9,
    )
    return image, heatmap, region


class _FakeSAM2Predictor:
    def __init__(self) -> None:
        self.image_shape: tuple[int, ...] | None = None
        self.image_writeable: bool | None = None
        self.calls: list[dict[str, object]] = []

    def set_image(self, image: np.ndarray) -> None:
        self.image_shape = image.shape
        self.image_writeable = bool(image.flags.writeable)

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        masks = np.zeros((2, 30, 20), dtype=np.uint8)
        masks[0, 0:2, 0:2] = 1
        masks[1, 9:18, 4:10] = 1
        scores = np.asarray([0.2, 0.8], dtype=np.float32)
        return masks, scores, None


class _OversizedSAM2Predictor:
    def set_image(self, image: np.ndarray) -> None:
        del image

    def predict(self, **kwargs):
        del kwargs
        masks = np.zeros((2, 30, 20), dtype=np.uint8)
        masks[0, 9:18, 4:10] = 1
        masks[1, :, :] = 1
        scores = np.asarray([0.2, 0.95], dtype=np.float32)
        return masks, scores, None


class _HighConfidenceFullMaskPredictor:
    def set_image(self, image: np.ndarray) -> None:
        del image

    def predict(self, **kwargs):
        del kwargs
        masks = np.zeros((2, 10, 10), dtype=np.uint8)
        masks[0, 2:5, 2:5] = 1
        masks[1, :, :] = 1
        scores = np.asarray([0.2, 0.99], dtype=np.float32)
        return masks, scores, None


if __name__ == "__main__":
    unittest.main()
