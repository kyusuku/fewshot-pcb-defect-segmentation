from __future__ import annotations

import csv
from pathlib import Path

from scripts.curate_paper_assets import curate_assets, select_examples


def test_select_examples_returns_success_and_failure_per_category() -> None:
    rows = [
        {"category": "pcb1", "sample_id": "a", "sam2_delta_f1": "0.4"},
        {"category": "pcb1", "sample_id": "b", "sam2_delta_f1": "-0.3"},
        {"category": "pcb2", "sample_id": "c", "sam2_delta_f1": "0.2"},
        {"category": "pcb2", "sample_id": "d", "sam2_delta_f1": "-0.1"},
    ]

    selected = select_examples(rows, successes_per_category=1, failures_per_category=1)

    assert [(row["category"], row["sample_id"], row["role"]) for row in selected] == [
        ("pcb1", "a", "success"),
        ("pcb1", "b", "failure"),
        ("pcb2", "c", "success"),
        ("pcb2", "d", "failure"),
    ]


def test_curate_assets_copies_required_panels_and_writes_manifest(tmp_path: Path) -> None:
    sources = tmp_path / "sources"
    sources.mkdir()
    paths = {}
    for name in ("image", "mask", "anomaly", "sam2", "fusion"):
        path = sources / f"{name}.png"
        path.write_bytes(f"{name}-data".encode("ascii"))
        paths[name] = path
    analysis_csv = tmp_path / "per_image_failure_analysis.csv"
    _write_csv(
        analysis_csv,
        [
            {
                "category": "pcb1",
                "sample_id": "pcb1/a",
                "sam2_delta_f1": "0.4",
                "anomaly_run_id": "dinov2_multi__pcb1__fold0__k1__seed4880",
                "sam2_run_id": "dinov2_multi_sam2__pcb1__fold0__k1__seed4880",
                "fusion_run_id": "anomaly_consistent_sam2__pcb1__fold0__k1__seed4880",
                "image_path": str(paths["image"]),
                "mask_path": str(paths["mask"]),
                "anomaly_panel_path": str(paths["anomaly"]),
                "sam2_panel_path": str(paths["sam2"]),
                "fusion_panel_path": str(paths["fusion"]),
            }
        ],
    )

    manifest_path = curate_assets(analysis_csv, tmp_path / "paper_assets")

    rows = _read_csv(manifest_path)
    assert rows[0]["role"] == "success"
    assert rows[0]["sample_id"] == "pcb1/a"
    copied = tmp_path / "paper_assets" / "qualitative" / "pcb1" / "success" / "pcb1_a"
    assert (copied / "image.png").read_bytes() == b"image-data"
    assert (copied / "anomaly_panel.png").read_bytes() == b"anomaly-data"
    assert rows[0]["sha256_image"]


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
