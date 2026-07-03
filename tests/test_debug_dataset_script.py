from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


class DebugDatasetScriptTest(unittest.TestCase):
    def test_deeppcb_split_file_argument_filters_samples(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            deeppcb_root = tmp_path / "PCBData"
            _write_deeppcb_official_pair(deeppcb_root, "77000016")
            _write_deeppcb_official_pair(deeppcb_root, "77000017")
            split_file = deeppcb_root / "test.txt"
            split_file.write_text(
                "group77000/77000/77000017.jpg group77000/77000_not/77000017.txt\n"
            )
            output_dir = tmp_path / "debug"

            result = subprocess.run(
                [
                    sys.executable,
                    str(repo_root / "scripts" / "debug_dataset.py"),
                    "--config",
                    str(repo_root / "configs" / "datasets" / "deeppcb.yaml"),
                    "--root",
                    str(deeppcb_root),
                    "--split",
                    "test",
                    "--split-file",
                    str(split_file),
                    "--limit",
                    "4",
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
            output_files = sorted(output_dir.glob("*.png"))

        self.assertIn("'num_samples': 1", result.stdout)
        self.assertEqual([path.name for path in output_files], ["000_77000017_test.png"])


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
