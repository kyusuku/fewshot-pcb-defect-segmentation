from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from utils.synthetic_data import create_synthetic_debug_datasets


class CreateManifestsScriptTest(unittest.TestCase):
    def test_script_writes_visa_and_deeppcb_manifests(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            fixture = create_synthetic_debug_datasets(tmp_path / "fixtures")
            output_dir = tmp_path / "manifests"

            result = subprocess.run(
                [
                    sys.executable,
                    str(repo_root / "scripts" / "create_manifests.py"),
                    "--visa-root",
                    str(fixture.visa_root),
                    "--deeppcb-root",
                    str(fixture.deeppcb_root),
                    "--output-dir",
                    str(output_dir),
                    "--visa-category",
                    "pcb1",
                    "--val-ratio",
                    "0.34",
                    "--deeppcb-train-count",
                    "1",
                ],
                check=False,
                cwd=repo_root,
                env={"PYTHONPATH": str(repo_root / "src")},
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            visa_rows = _read_csv(output_dir / "visa_pcb_manifest.csv")
            deeppcb_rows = _read_csv(output_dir / "deeppcb_manifest.csv")

        self.assertEqual({row["category"] for row in visa_rows}, {"pcb1"})
        self.assertIn("val", {row["split"] for row in visa_rows})
        self.assertIn("test", {row["split"] for row in visa_rows})
        self.assertEqual({row["dataset"] for row in deeppcb_rows}, {"deeppcb"})

    def test_script_uses_deeppcb_official_split_files_when_available(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            deeppcb_root = tmp_path / "PCBData"
            _write_deeppcb_official_pair(deeppcb_root, "77000016")
            _write_deeppcb_official_pair(deeppcb_root, "77000017")
            (deeppcb_root / "trainval.txt").write_text(
                "group77000/77000/77000016.jpg group77000/77000_not/77000016.txt\n"
            )
            (deeppcb_root / "test.txt").write_text(
                "group77000/77000/77000017.jpg group77000/77000_not/77000017.txt\n"
            )
            output_dir = tmp_path / "manifests"

            result = subprocess.run(
                [
                    sys.executable,
                    str(repo_root / "scripts" / "create_manifests.py"),
                    "--deeppcb-root",
                    str(deeppcb_root),
                    "--output-dir",
                    str(output_dir),
                    "--val-ratio",
                    "0.0",
                ],
                check=False,
                cwd=repo_root,
                env={"PYTHONPATH": str(repo_root / "src")},
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            rows = _read_csv(output_dir / "deeppcb_manifest.csv")

        self.assertEqual(
            {row["sample_id"]: row["split"] for row in rows},
            {"77000016_test": "train", "77000017_test": "test"},
        )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write_deeppcb_official_pair(root: Path, sample_id: str) -> None:
    image_dir = root / "group77000" / "77000"
    annotation_dir = root / "group77000" / "77000_not"
    image_dir.mkdir(parents=True, exist_ok=True)
    annotation_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), (0, 80, 50)).save(image_dir / f"{sample_id}_temp.jpg")
    Image.new("RGB", (16, 16), (120, 80, 50)).save(image_dir / f"{sample_id}_test.jpg")
    (annotation_dir / f"{sample_id}.txt").write_text("1,2,8,9,4\n")


if __name__ == "__main__":
    unittest.main()
