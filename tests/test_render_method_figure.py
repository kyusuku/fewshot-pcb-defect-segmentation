from __future__ import annotations

import json
import hashlib
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from scripts.render_method_figure import BLOCKS, _load_font, render_method_figure


def test_render_method_figure_writes_nonblank_png_and_manifest(tmp_path: Path) -> None:
    png_path = tmp_path / "method_figure.png"
    manifest_path = tmp_path / "method_figure.json"

    render_method_figure(png_path, manifest_path, generation_command="test-render")

    image = Image.open(png_path)
    assert image.mode == "RGB"
    assert image.size[0] > 0
    assert image.size[1] > 0
    pixels = np.asarray(image)
    assert np.unique(pixels.reshape(-1, pixels.shape[-1]), axis=0).shape[0] > 4
    manifest = json.loads(manifest_path.read_text())
    assert manifest["generation_command"] == "test-render"
    assert len(manifest["blocks"]) == 5
    assert manifest["output_png"] == "method_figure.png"
    assert manifest["output_png_sha256"] == hashlib.sha256(png_path.read_bytes()).hexdigest()


def test_method_figure_labels_are_wrapped_and_fit_inside_blocks() -> None:
    assert [block["label"] for block in BLOCKS] == [
        "Few normal\nsupports",
        "DINOv2\nmemory bank",
        "Multi-scale\nanomaly map",
        "Anomaly-guided\nSAM2 refinement",
        "Proposal and SAM2\nintersection mask",
    ]
    draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    font = _load_font(size=28)
    for block in BLOCKS:
        left, top, right, bottom = block["rect"]
        box = draw.multiline_textbbox((0, 0), block["label"], font=font, spacing=6)
        assert box[2] - box[0] <= right - left - 24
        assert box[3] - box[1] <= bottom - top - 24


def test_render_method_figure_cli_records_exact_shell_quoted_command(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    script = repo / "scripts" / "render_method_figure.py"
    png_path = tmp_path / "method figure.png"
    layout_path = tmp_path / "method layout.json"
    command = [
        sys.executable,
        str(script),
        "--output-png",
        str(png_path),
        "--layout-json",
        str(layout_path),
    ]

    result = subprocess.run(command, cwd=repo, capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stderr
    layout = json.loads(layout_path.read_text())
    assert layout["generation_command"] == shlex.join(command[1:])
