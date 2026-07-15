#!/usr/bin/env python
"""Curate deterministic qualitative examples for paper figures."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils.heatmap_io import load_heatmap
from utils.visualize import safe_filename


PANEL_FIELDS = {
    "image": "image_path",
    "mask": "mask_path",
    "anomaly_panel": "anomaly_panel_path",
    "sam2_panel": "sam2_panel_path",
    "fusion_panel": "fusion_panel_path",
}
MANIFEST_FIELDS = [
    "category",
    "role",
    "sample_id",
    "anomaly_run_id",
    "sam2_run_id",
    "fusion_run_id",
    "sam2_delta_f1",
    "source_paths_json",
    "copied_paths_json",
    "sha256_image",
    "sha256_mask",
    "sha256_anomaly_panel",
    "sha256_sam2_panel",
    "sha256_fusion_panel",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failure-analysis-csv", type=Path, required=True)
    parser.add_argument("--asset-dir", type=Path, default=Path("artifacts/paper_assets"))
    parser.add_argument("--successes-per-category", type=int, default=1)
    parser.add_argument("--failures-per-category", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    curate_assets(
        args.failure_analysis_csv,
        args.asset_dir,
        successes_per_category=args.successes_per_category,
        failures_per_category=args.failures_per_category,
    )


def select_examples(
    rows: Sequence[Mapping[str, str]],
    *,
    successes_per_category: int = 1,
    failures_per_category: int = 1,
) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    eligible_rows = [row for row in rows if row.get("label", "1") != "0"]
    categories = sorted({row["category"] for row in eligible_rows})
    for category in categories:
        category_rows = [dict(row) for row in eligible_rows if row["category"] == category]
        successes = sorted(
            category_rows,
            key=lambda row: (-float(row["sam2_delta_f1"]), row["sample_id"]),
        )[:successes_per_category]
        failures = sorted(
            category_rows,
            key=lambda row: (float(row["sam2_delta_f1"]), row["sample_id"]),
        )[:failures_per_category]
        for row in successes:
            selected_row = dict(row)
            selected_row["role"] = "success"
            selected.append(selected_row)
        for row in failures:
            selected_row = dict(row)
            selected_row["role"] = "failure"
            selected.append(selected_row)
    return selected


def curate_assets(
    failure_analysis_csv: str | Path,
    asset_dir: str | Path,
    *,
    successes_per_category: int = 1,
    failures_per_category: int = 1,
) -> Path:
    failure_analysis_csv = Path(failure_analysis_csv)
    asset_dir = Path(asset_dir)
    rows = _read_csv(failure_analysis_csv)
    selected = select_examples(
        rows,
        successes_per_category=successes_per_category,
        failures_per_category=failures_per_category,
    )
    manifest_rows = []
    for row in selected:
        output_dir = (
            asset_dir
            / "qualitative"
            / row["category"]
            / row["role"]
            / safe_filename(row["sample_id"])
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        copied: dict[str, str] = {}
        source_paths: dict[str, str] = {}
        hashes: dict[str, str] = {}
        for panel_name, field_name in PANEL_FIELDS.items():
            source = _panel_source(row, panel_name, field_name)
            if source is None or not source.is_file():
                raise ValueError(f"missing required {panel_name} panel for {row['sample_id']}")
            destination = output_dir / f"{panel_name}.png"
            if panel_name == "anomaly_panel" and source.suffix in {".npy", ".npz"}:
                _render_heatmap_panel(source, destination)
            else:
                shutil.copyfile(source, destination)
            source_paths[panel_name] = str(source)
            copied[panel_name] = str(destination)
            hashes[f"sha256_{panel_name}"] = _sha256_file(destination)
        manifest_rows.append(
            {
                "category": row["category"],
                "role": row["role"],
                "sample_id": row["sample_id"],
                "anomaly_run_id": row.get("anomaly_run_id", ""),
                "sam2_run_id": row.get("sam2_run_id", ""),
                "fusion_run_id": row.get("fusion_run_id", ""),
                "sam2_delta_f1": row.get("sam2_delta_f1", ""),
                "source_paths_json": json.dumps(source_paths, sort_keys=True),
                "copied_paths_json": json.dumps(copied, sort_keys=True),
                **hashes,
            }
        )
    manifest_path = asset_dir / "qualitative_manifest.csv"
    _write_csv(manifest_path, manifest_rows, MANIFEST_FIELDS)
    return manifest_path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _panel_source(
    row: Mapping[str, str],
    panel_name: str,
    field_name: str,
) -> Path | None:
    value = row.get(field_name, "")
    if value:
        return Path(value)
    if panel_name == "anomaly_panel" and row.get("heatmap_path"):
        return Path(row["heatmap_path"])
    if panel_name == "sam2_panel" and row.get("sam2_mask_path"):
        return Path(row["sam2_mask_path"])
    if panel_name == "fusion_panel" and row.get("fusion_mask_path"):
        return Path(row["fusion_mask_path"])
    return None


def _render_heatmap_panel(source: Path, destination: Path) -> None:
    heatmap = load_heatmap(source).astype(np.float32, copy=False)
    minimum = float(np.min(heatmap))
    maximum = float(np.max(heatmap))
    if maximum > minimum:
        normalized = (heatmap - minimum) / (maximum - minimum)
    else:
        normalized = np.zeros_like(heatmap)
    image = Image.fromarray((normalized * 255).astype(np.uint8), mode="L").convert("RGB")
    image.save(destination)


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, str]],
    fieldnames: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
