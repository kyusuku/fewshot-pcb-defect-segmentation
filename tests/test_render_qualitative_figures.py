from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from scripts.render_qualitative_figures import render_qualitative_figures


def test_render_qualitative_figures_validates_panels_and_writes_public_montages(
    tmp_path: Path,
) -> None:
    asset_root = tmp_path / "transfer"
    rows = []
    for role, delta, base in (("success", 0.2, 20), ("failure", -0.1, 80)):
        paths = {}
        hashes = {}
        for index, panel in enumerate(
            ("image", "mask", "anomaly_panel", "sam2_panel", "fusion_panel")
        ):
            relative = Path("artifacts") / "paper" / role / f"{panel}.png"
            path = asset_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (32, 24), (base + index * 10, 30, 40)).save(path)
            paths[panel] = relative.as_posix()
            hashes[f"sha256_{panel}"] = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(
            {
                "category": "pcb1",
                "role": role,
                "sample_id": f"pcb1/{role}",
                "sam2_delta_f1": delta,
                "copied_paths_json": json.dumps(paths),
                **hashes,
            }
        )
    source_manifest = tmp_path / "qualitative_manifest.csv"
    _write_csv(source_manifest, rows)

    output = tmp_path / "figures"
    manifest = render_qualitative_figures(
        source_manifest,
        asset_root=asset_root,
        output_dir=output,
        categories=["pcb1"],
        generation_command="test qualitative renderer",
    )

    assert manifest["generation_command"] == "test qualitative renderer"
    assert len(manifest["figures"]) == 2
    assert {(item["category"], item["role"]) for item in manifest["figures"]} == {
        ("pcb1", "success"),
        ("pcb1", "failure"),
    }
    for item in manifest["figures"]:
        path = output / item["path"]
        assert path.is_file()
        assert item["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert Image.open(path).size[0] > Image.open(path).size[1]
    serialized = (output / "qualitative_figure_manifest.json").read_text()
    assert "copied_paths_json" not in serialized
    assert str(asset_root) not in serialized

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        render_qualitative_figures(
            source_manifest,
            asset_root=asset_root,
            output_dir=output,
            categories=["pcb1"],
            generation_command="test qualitative renderer",
        )


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
