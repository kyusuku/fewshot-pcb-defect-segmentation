from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class CreateSyntheticDataScriptTest(unittest.TestCase):
    def test_script_creates_reusable_debug_fixture(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "debug_fixture"
            result = subprocess.run(
                [
                    sys.executable,
                    str(repo_root / "scripts" / "create_synthetic_data.py"),
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
            self.assertIn("visa_root:", result.stdout)
            self.assertIn("deeppcb_root:", result.stdout)
            self.assertTrue(
                (output_dir / "VisA" / "pcb1" / "test" / "anomaly" / "pcb1_anomaly_000.png").is_file()
            )
            self.assertTrue(
                (output_dir / "DeepPCB" / "PCBData" / "group00000" / "00000000_test.jpg").is_file()
            )


if __name__ == "__main__":
    unittest.main()
