from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_dry_run_prints_ordered_argv_and_writes_nothing(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    output = tmp_path / "out"
    result = subprocess.run(
        [
            sys.executable,
            str(repo / "scripts" / "run_experiment_matrix.py"),
            "--config",
            str(repo / "configs" / "experiments" / "arxiv_smoke.yaml"),
            "--output-root",
            str(output),
            "--device",
            "cpu",
            "--dry-run",
        ],
        cwd=repo,
        env={"PYTHONPATH": str(repo / "src")},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert not output.exists()
    lines = result.stdout.splitlines()
    argv = [
        json.loads(line.removeprefix("ARGV "))
        for line in lines
        if line.startswith("ARGV ")
    ]
    assert all(isinstance(command, list) for command in argv)
    val_index = next(
        i
        for i, line in enumerate(lines)
        if _argv_option(line, "--query-fold-split") == "val"
    )
    calibration_index = next(i for i, line in enumerate(lines) if "calibrate_heatmaps.py" in line)
    test_index = next(
        i
        for i, line in enumerate(lines)
        if _argv_option(line, "--query-fold-split") == "test"
    )
    assert val_index < calibration_index < test_index
    assert all(line.startswith(("RUN ", "ARGV ")) for line in lines)


def test_checker_missing_matrix_is_deterministic_json_and_exit_one(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    command = [
        sys.executable,
        str(repo / "scripts" / "check_experiment_matrix.py"),
        "--config",
        str(repo / "configs" / "experiments" / "arxiv_smoke.yaml"),
        "--output-root",
        str(tmp_path / "missing"),
        "--device",
        "cpu",
    ]
    environment = {"PYTHONPATH": str(repo / "src")}
    first = subprocess.run(
        command, cwd=repo, env=environment, capture_output=True, text=True
    )
    second = subprocess.run(
        command, cwd=repo, env=environment, capture_output=True, text=True
    )
    assert first.returncode == second.returncode == 1
    assert first.stdout == second.stdout
    assert '"ok":false' in first.stdout


def _argv_option(line: str, flag: str) -> str | None:
    if not line.startswith("ARGV "):
        return None
    command = json.loads(line.removeprefix("ARGV "))
    return command[command.index(flag) + 1] if flag in command else None
