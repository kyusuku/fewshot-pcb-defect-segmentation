from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from utils.heatmap_io import load_heatmap

from scripts.run_sam2_baseline import (
    _optional_fraction,
    build_grid_prompt_regions,
    prompt_shape_for_image,
    select_query_rows,
)


class SAM2OnlyBaselinePromptTest(unittest.TestCase):
    def test_optional_fraction_accepts_explicit_none(self) -> None:
        self.assertIsNone(_optional_fraction("none"))
        self.assertEqual(_optional_fraction("0.25"), 0.25)

    def test_prompt_shape_preserves_image_aspect_ratio(self) -> None:
        self.assertEqual(prompt_shape_for_image((640, 320), longest_side=64), (32, 64))
        self.assertEqual(prompt_shape_for_image((320, 640), longest_side=64), (64, 32))

    def test_build_grid_prompt_regions_covers_prompt_shape(self) -> None:
        regions = build_grid_prompt_regions(
            prompt_shape=(6, 8),
            grid_size=2,
            max_regions=4,
            box_scale=1.0,
        )

        self.assertEqual(
            [region.box_xyxy for region in regions],
            [
                (0, 0, 4, 3),
                (4, 0, 8, 3),
                (0, 3, 4, 6),
                (4, 3, 8, 6),
            ],
        )
        self.assertEqual(regions[0].point_xy, (2.0, 1.5))
        self.assertEqual(regions[0].area, 12)
        self.assertEqual(regions[0].score, 1.0)

    def test_none_limit_selects_all_queries(self) -> None:
        rows = [
            {
                "dataset": "visa_pcb",
                "category": "pcb1",
                "sample_id": f"pcb1/{index}",
                "label": str(index % 2),
                "fold_id": "0",
                "fold_split": "test",
            }
            for index in range(3)
        ]
        selected = select_query_rows(rows, 0, "pcb1", "test", limit=None)
        self.assertEqual(len(selected), 3)


class SAM2OnlyBaselineScriptTest(unittest.TestCase):
    def test_cli_supports_all_and_rejects_nonpositive_limit(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        help_result = subprocess.run(
            [sys.executable, str(repo_root / "scripts" / "run_sam2_baseline.py"), "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        invalid_result = subprocess.run(
            [
                sys.executable,
                str(repo_root / "scripts" / "run_sam2_baseline.py"),
                "--limit",
                "0",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(help_result.returncode, 0)
        self.assertIn("--all", help_result.stdout)
        self.assertNotEqual(invalid_result.returncode, 0)
        self.assertIn("positive", invalid_result.stderr)

    def test_script_writes_mask_scores_without_support_images_or_heatmaps(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            image_path = tmp_path / "image.png"
            mask_path = tmp_path / "mask.png"
            manifest_path = tmp_path / "visa_pcb_folds.csv"
            output_dir = tmp_path / "sam2_only"
            Image.new("RGB", (12, 8), (20, 40, 60)).save(image_path)
            mask = np.zeros((8, 12), dtype=np.uint8)
            mask[2:5, 3:7] = 255
            Image.fromarray(mask, mode="L").save(mask_path)
            _write_manifest_without_support(manifest_path, image_path, mask_path)

            result = subprocess.run(
                [
                    sys.executable,
                    str(repo_root / "scripts" / "run_sam2_baseline.py"),
                    "--manifest",
                    str(manifest_path),
                    "--fold-id",
                    "0",
                    "--category",
                    "pcb1",
                    "--query-fold-split",
                    "test",
                    "--all",
                    "--prompt-longest-side",
                    "12",
                    "--grid-size",
                    "1",
                    "--refiner",
                    "fallback",
                    "--debug-limit",
                    "0",
                    "--heatmap-format",
                    "npz_compressed",
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
            rows = _read_csv(output_dir / "mask_scores.csv")
            pred_mask_path = output_dir / rows[0]["pred_mask_path"]
            heatmap_path = output_dir / rows[0]["heatmap_path"]
            pred_mask = np.asarray(Image.open(pred_mask_path).convert("L")) > 0
            heatmap = load_heatmap(heatmap_path)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["sample_id"], "pcb1/anomaly_000")
        self.assertEqual(rows[0]["category"], "pcb1")
        self.assertEqual(rows[0]["label"], "1")
        self.assertEqual(rows[0]["num_regions"], "1")
        self.assertEqual(rows[0]["num_masks"], "1")
        self.assertEqual(rows[0]["mask_path"], "../mask.png")
        self.assertEqual(pred_mask.shape, (8, 12))
        self.assertEqual(int(pred_mask.sum()), 96)
        self.assertEqual(heatmap.shape, (8, 12))
        self.assertEqual(heatmap_path.suffix, ".npz")
        self.assertEqual(rows[0]["debug_path"], "")


def _write_manifest_without_support(
    path: Path,
    image_path: Path,
    mask_path: Path,
) -> None:
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
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "dataset": "visa_pcb",
                "category": "pcb1",
                "sample_id": "pcb1/anomaly_000",
                "split": "test",
                "image_path": str(image_path),
                "label": "1",
                "mask_path": str(mask_path),
                "box_path": "",
                "template_path": "",
                "metadata_json": "{}",
                "fold_id": "0",
                "fold_split": "test",
            }
        )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    unittest.main()
