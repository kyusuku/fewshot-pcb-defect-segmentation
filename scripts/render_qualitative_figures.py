#!/usr/bin/env python
"""Render public qualitative montages from checksum-validated selected assets."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import sys
from pathlib import Path
from typing import Mapping, Sequence

from PIL import Image, ImageDraw, ImageFont, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from experiments.provenance import sha256_file


PANELS = (
    ("image", "Input image"),
    ("mask", "Ground truth"),
    ("anomaly_panel", "Anomaly score"),
    ("sam2_panel", "Guided SAM2"),
    ("fusion_panel", "Anomaly-consistent"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--categories", nargs="+", default=["pcb1", "pcb2", "pcb3", "pcb4"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        render_qualitative_figures(
            args.source_manifest,
            asset_root=args.asset_root,
            output_dir=args.output_dir,
            categories=args.categories,
            generation_command=shlex.join(sys.argv),
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc


def render_qualitative_figures(
    source_manifest: str | Path,
    *,
    asset_root: str | Path,
    output_dir: str | Path,
    categories: Sequence[str],
    generation_command: str,
) -> dict[str, object]:
    """Validate selected panels and render one labeled montage per selected case."""

    source_manifest = Path(source_manifest)
    asset_root = Path(asset_root)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing figure directory: {output_dir}")
    rows = _read_csv(source_manifest)
    by_role: dict[tuple[str, str], Mapping[str, str]] = {}
    for row in rows:
        key = (str(row.get("category", "")), str(row.get("role", "")))
        if key in by_role:
            raise ValueError(f"duplicate qualitative role: {key!r}")
        by_role[key] = row
    expected = {(category, role) for category in categories for role in ("success", "failure")}
    if set(by_role) != expected:
        raise ValueError(
            "qualitative selections must contain success and failure for every category"
        )

    validated: list[tuple[Mapping[str, str], dict[str, Path], dict[str, str]]] = []
    for key in sorted(expected):
        row = by_role[key]
        delta = float(row["sam2_delta_f1"])
        if (key[1] == "success" and delta <= 0.0) or (key[1] == "failure" and delta >= 0.0):
            raise ValueError(f"qualitative role has dishonest delta sign: {key!r}")
        try:
            copied = json.loads(row.get("copied_paths_json", ""))
        except json.JSONDecodeError as exc:
            raise ValueError(f"qualitative copied paths are malformed: {key!r}") from exc
        if not isinstance(copied, Mapping):
            raise ValueError(f"qualitative copied paths must be an object: {key!r}")
        paths: dict[str, Path] = {}
        hashes: dict[str, str] = {}
        for panel, _ in PANELS:
            value = str(copied.get(panel, ""))
            relative = Path(value)
            if not value or relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"qualitative panel path is missing or unsafe: {key!r}, {panel}")
            path = asset_root / relative
            expected_hash = str(row.get(f"sha256_{panel}", ""))
            if not path.is_file() or sha256_file(path) != expected_hash:
                raise ValueError(f"qualitative panel checksum mismatch: {key!r}, {panel}")
            paths[panel] = path
            hashes[panel] = expected_hash
        validated.append((row, paths, hashes))

    output_dir.mkdir(parents=True)
    figure_entries = []
    for row, paths, hashes in validated:
        filename = f"{row['category']}_{row['role']}.png"
        output_path = output_dir / filename
        _render_montage(row, paths, output_path)
        figure_entries.append(
            {
                "category": row["category"],
                "role": row["role"],
                "sample_id": row["sample_id"],
                "sam2_delta_f1": float(row["sam2_delta_f1"]),
                "path": filename,
                "sha256": sha256_file(output_path),
                "source_panel_sha256": hashes,
            }
        )
    manifest = {
        "schema_version": 2,
        "generation_command": generation_command,
        "renderer_source_sha256": sha256_file(Path(__file__)),
        "source_manifest_sha256": sha256_file(source_manifest),
        "role_definition": {
            "success": "selected anomalous case with strictly positive guided-SAM2 delta F1",
            "failure": "selected anomalous case with strictly negative guided-SAM2 delta F1",
        },
        "figures": figure_entries,
    }
    (output_dir / "qualitative_figure_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _render_montage(
    row: Mapping[str, str],
    paths: Mapping[str, Path],
    output_path: Path,
) -> None:
    panel_size = (280, 280)
    gap = 18
    margin = 24
    header_height = 58
    label_height = 38
    width = margin * 2 + len(PANELS) * panel_size[0] + (len(PANELS) - 1) * gap
    height = margin + header_height + panel_size[1] + label_height + margin
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = _load_font(22)
    label_font = _load_font(18)
    delta = float(row["sam2_delta_f1"])
    title = (
        f"{str(row['category']).upper()} {row['role']} | {row['sample_id']} | "
        f"guided SAM2 delta F1 {delta:+.4f}"
    )
    draw.text((margin, margin), title, fill="#182230", font=title_font)
    y = margin + header_height
    for index, (panel, label) in enumerate(PANELS):
        x = margin + index * (panel_size[0] + gap)
        with Image.open(paths[panel]) as image:
            fitted = ImageOps.contain(image.convert("RGB"), panel_size, Image.Resampling.LANCZOS)
        tile = Image.new("RGB", panel_size, "#f2f4f7")
        tile.paste(
            fitted,
            ((panel_size[0] - fitted.width) // 2, (panel_size[1] - fitted.height) // 2),
        )
        canvas.paste(tile, (x, y))
        draw.rectangle((x, y, x + panel_size[0], y + panel_size[1]), outline="#344054", width=2)
        label_box = draw.textbbox((0, 0), label, font=label_font)
        label_width = label_box[2] - label_box[0]
        draw.text(
            (x + (panel_size[0] - label_width) / 2, y + panel_size[1] + 9),
            label,
            fill="#182230",
            font=label_font,
        )
    canvas.save(output_path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise ValueError("missing qualitative selection manifest")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_font(size: int) -> ImageFont.ImageFont:
    for candidate in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttf",
        "/Library/Fonts/Arial.ttf",
    ):
        path = Path(candidate)
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


if __name__ == "__main__":
    main()
