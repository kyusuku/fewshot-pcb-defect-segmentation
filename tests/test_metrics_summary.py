from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from evaluation.summary import summarize_metric_files


class MetricsSummaryTest(unittest.TestCase):
    def test_summarize_metric_files_adds_mean_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pcb1 = _write_metrics(root / "dinov2_vits14_pcb1_fold0_full" / "metrics.json", 0.8, 0.7)
            pcb2 = _write_metrics(root / "dinov2_vits14_pcb2_fold0_full" / "metrics.json", 0.6, 0.5)

            rows = summarize_metric_files([pcb1, pcb2])

        self.assertEqual([row["category"] for row in rows], ["pcb1", "pcb2", "mean"])
        self.assertEqual(rows[0]["num_images"], 10)
        self.assertAlmostEqual(rows[-1]["image_auroc"], 0.7)
        self.assertAlmostEqual(rows[-1]["pixel_auroc"], 0.6)


class MetricsSummaryScriptTest(unittest.TestCase):
    def test_script_writes_csv_and_markdown(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            metrics_a = _write_metrics(
                root / "dinov2_vits14_pcb1_fold0_full" / "metrics.json",
                0.8,
                0.7,
            )
            metrics_b = _write_metrics(
                root / "dinov2_vits14_pcb2_fold0_full" / "metrics.json",
                0.6,
                0.5,
            )
            output_csv = root / "summary.csv"
            output_md = root / "summary.md"

            result = subprocess.run(
                [
                    sys.executable,
                    str(repo_root / "scripts" / "summarize_metrics.py"),
                    "--metrics-json",
                    str(metrics_a),
                    str(metrics_b),
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
            csv_rows = _read_csv(output_csv)
            markdown = output_md.read_text()

        self.assertEqual(csv_rows[-1]["category"], "mean")
        self.assertIn("| category | num_images | image_auroc | pixel_auroc |", markdown)


def _write_metrics(path: Path, image_auroc: float, pixel_auroc: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "num_images": 10,
                "image_auroc": image_auroc,
                "pixel_auroc": pixel_auroc,
                "best_pixel_f1": 0.2,
                "best_pixel_iou": 0.1,
                "num_pixels_evaluated": 1000,
            }
        )
    )
    return path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    unittest.main()
