from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.render_method_figure import render_method_figure


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
