from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
from PIL import Image

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
        self.assertEqual(regions[0].point_xy, (3.5, 4.5))
        self.assertGreater(regions[0].score, regions[1].score)


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
            )
        )

        self.assertIsInstance(refiner, SAM2MaskRefiner)
        self.assertEqual(refiner.max_mask_area_fraction, 0.25)

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
            pred_mask_path = Path(rows[0]["pred_mask_path"])
            pred_mask = np.asarray(Image.open(pred_mask_path).convert("L")) > 0

        self.assertEqual(len(rows), 1)
        self.assertEqual(int(pred_mask.sum()), 9)
        self.assertEqual(rows[0]["mask_path"], str(mask_path))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


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
