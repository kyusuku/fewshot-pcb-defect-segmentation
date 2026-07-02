from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

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


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    unittest.main()
