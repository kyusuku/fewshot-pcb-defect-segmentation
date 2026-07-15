from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from evaluation.stage4 import format_stage4_markdown, summarize_stage4_files


class Stage4SummaryTest(unittest.TestCase):
    def test_summarize_stage4_files_infers_methods_and_adds_means(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = [
                _write_heatmap_metrics(
                    root / "dinov2_vits14_pcb1_fold0_full" / "metrics.json",
                    image_auroc=0.7,
                    pixel_auroc=0.8,
                    aupro=0.75,
                    f1=0.2,
                    iou=0.1,
                ),
                _write_heatmap_metrics(
                    root / "dinov2_vits14_pcb2_fold0_full" / "metrics.json",
                    image_auroc=0.9,
                    pixel_auroc=0.6,
                    aupro=0.55,
                    f1=0.4,
                    iou=0.3,
                ),
                _write_mask_metrics(
                    root / "sam2_only_pcb1_fold0_full" / "mask_metrics.json",
                    f1=0.05,
                    iou=0.03,
                    precision=0.04,
                    recall=0.08,
                ),
                _write_mask_metrics(
                    root
                    / "dinov2_vits14_pcb1_fold0_ms768_o025_max_full_sam2"
                    / "mask_metrics.json",
                    f1=0.3,
                    iou=0.2,
                    precision=0.25,
                    recall=0.45,
                ),
            ]

            rows = summarize_stage4_files(paths)

        method_rows = {(row["category"], row["method"]): row for row in rows}
        self.assertAlmostEqual(
            method_rows[("pcb1", "dinov2_single_heatmap")]["image_auroc"],
            0.7,
        )
        self.assertAlmostEqual(
            method_rows[("pcb1", "dinov2_single_heatmap")]["aupro"],
            0.75,
        )
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
        self.assertAlmostEqual(
            method_rows[("pcb1", "sam2_only")]["mean_anomaly_mask_f1"],
            0.05,
        )
        self.assertAlmostEqual(
            method_rows[("pcb1", "ms768_dinov2_sam2")]["mean_anomaly_mask_recall"],
            0.45,
        )
        self.assertAlmostEqual(
            method_rows[("mean", "dinov2_single_heatmap")]["image_auroc"],
            0.8,
        )

    def test_format_stage4_markdown_uses_dashes_for_missing_metrics(self) -> None:
        markdown = format_stage4_markdown(
            [
                {
                    "category": "pcb1",
                    "method": "sam2_only",
                    "threshold_policy": "binary_model_output",
                    "image_auroc": None,
                    "pixel_auroc": None,
                    "aupro": None,
                    "aggregate_pixel_f1": None,
                    "aggregate_pixel_iou": None,
                    "mean_anomaly_mask_f1": 0.1,
                    "mean_anomaly_mask_iou": 0.05,
                    "mean_anomaly_mask_precision": 0.2,
                    "mean_anomaly_mask_recall": 0.3,
                }
            ]
        )

        self.assertIn(
            "| pcb1 | sam2_only | binary_model_output | - | - | - | - | - | 0.1000 | 0.0500 | 0.2000 | 0.3000 |",
            markdown,
        )

    def test_heatmap_requires_calibrated_binary_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            path = _write_json(
                root / "dinov2_vits14_pcb1_fold0_full" / "metrics.json",
                {
                    "num_images": 10,
                    "image_auroc": 0.7,
                    "pixel_auroc": 0.8,
                    "aupro": 0.75,
                    "best_pixel_f1": 0.9,
                    "best_pixel_iou": 0.8,
                },
            )

            with self.assertRaisesRegex(ValueError, "calibrated heatmap metrics required"):
                summarize_stage4_files([path])


class Stage4SummaryScriptTest(unittest.TestCase):
    def test_script_writes_csv_and_markdown(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            heatmap_metrics = _write_heatmap_metrics(
                root / "dinov2_vits14_pcb1_fold0_full" / "metrics.json",
                image_auroc=0.7,
                pixel_auroc=0.8,
                aupro=0.75,
                f1=0.2,
                iou=0.1,
            )
            mask_metrics = _write_mask_metrics(
                root / "sam2_only_pcb1_fold0_full" / "mask_metrics.json",
                f1=0.05,
                iou=0.03,
                precision=0.04,
                recall=0.08,
            )
            output_csv = root / "stage4.csv"
            output_md = root / "stage4.md"

            result = subprocess.run(
                [
                    sys.executable,
                    str(repo_root / "scripts" / "summarize_stage4.py"),
                    "--metrics-json",
                    str(heatmap_metrics),
                    str(mask_metrics),
                    "--output-csv",
                    str(output_csv),
                    "--output-md",
                    str(output_md),
                ],
                check=False,
                cwd=repo_root,
                env={"PYTHONPATH": str(repo_root / "src")},
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            rows = _read_csv(output_csv)
            markdown = output_md.read_text()

        self.assertEqual(rows[0]["method"], "dinov2_single_heatmap")
        self.assertIn(
            "| category | method | threshold policy | image AUROC | pixel AUROC | AUPRO |",
            markdown,
        )


def _write_heatmap_metrics(
    path: Path,
    image_auroc: float,
    pixel_auroc: float,
    aupro: float,
    f1: float,
    iou: float,
) -> Path:
    return _write_json(
        path,
        {
            "num_images": 10,
            "image_auroc": image_auroc,
            "pixel_auroc": pixel_auroc,
            "aupro": aupro,
            "calibrated_aggregate_pixel_f1": f1,
            "calibrated_aggregate_pixel_iou": iou,
            "calibrated_mean_anomaly_mask_f1": f1,
            "calibrated_mean_anomaly_mask_iou": iou,
            "calibrated_mean_anomaly_mask_precision": f1 + 0.01,
            "calibrated_mean_anomaly_mask_recall": f1 + 0.02,
        },
    )


def _write_mask_metrics(
    path: Path,
    f1: float,
    iou: float,
    precision: float,
    recall: float,
) -> Path:
    return _write_json(
        path,
        {
            "num_mask_images": 10,
            "mean_anomaly_mask_f1": f1,
            "mean_anomaly_mask_iou": iou,
            "mean_anomaly_mask_precision": precision,
            "mean_anomaly_mask_recall": recall,
        },
    )


def _write_json(path: Path, payload: dict[str, float]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    unittest.main()
