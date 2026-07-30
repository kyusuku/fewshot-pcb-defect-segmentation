from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image
from pypdf import PdfReader
from reportlab.pdfbase.pdfmetrics import stringWidth

from scripts.render_visual_big_picture import (
    PAGE_SPECS,
    PROJECT_ROOT,
    VISUAL_RENDERERS,
    build_visual_big_picture,
    collect_step_numbers,
    extract_real_assets,
    load_f1_table,
    sha256_file,
    validate_source_assets,
)


def test_page_registry_covers_fourteen_pages_and_all_steps_in_order() -> None:
    assert [page.number for page in PAGE_SPECS] == list(range(1, 15))
    assert collect_step_numbers(PAGE_SPECS) == list(range(1, 17))


def test_every_page_has_teaching_copy_and_source_kind() -> None:
    for page in PAGE_SPECS:
        assert page.title
        assert page.what_enters
        assert page.what_happens
        assert page.what_comes_out
        assert page.what_to_notice
        assert page.source_kind in {"real", "illustration", "mixed"}


def test_source_assets_exist_and_frozen_montages_match_manifest() -> None:
    validated = validate_source_assets()

    assert validated["pcb1_success"]
    assert validated["pcb1_failure"]
    assert all(validated[f"pcb{index}_normal"] for index in range(1, 5))


def test_real_asset_extraction_preserves_expected_panel_size(tmp_path: Path) -> None:
    assets = extract_real_assets(tmp_path)

    for name in (
        "success_anomaly",
        "success_guided",
        "success_ac",
        "failure_guided",
        "failure_ac",
    ):
        with Image.open(assets[name]) as image:
            assert image.size == (280, 280)
    assert Image.open(assets["success_query"]).size == (1404, 1070)
    assert Image.open(assets["success_ground_truth"]).size == (1404, 1070)


def test_f1_table_is_loaded_from_frozen_primary_summary() -> None:
    assert load_f1_table() == {
        "Multi-DINO": {1: 0.210, 2: 0.228, 4: 0.244},
        "Guided SAM2": {1: 0.257, 2: 0.284, 4: 0.303},
        "AC-SAM2": {1: 0.268, 2: 0.297, 4: 0.317},
    }


@pytest.mark.filterwarnings("error")
def test_build_writes_pdf_markdown_manifest_and_fourteen_previews(tmp_path: Path) -> None:
    result = build_visual_big_picture(tmp_path, render_pngs=True)

    assert set(result) == {"pdf", "markdown", "manifest", "contact_sheet", "pages"}
    assert len(PdfReader(str(result["pdf"])).pages) == 14
    assert len(result["pages"]) == 14
    assert all(path.is_file() and path.stat().st_size > 10_000 for path in result["pages"])
    assert result["markdown"].read_text(encoding="utf-8").count("## Page ") == 14
    manifest = json.loads(result["manifest"].read_text(encoding="utf-8"))
    assert manifest["page_count"] == 14
    assert manifest["step_numbers"] == list(range(1, 17))
    assert manifest["pdf_sha256"] == sha256_file(result["pdf"])


def test_every_page_visual_has_a_renderer() -> None:
    assert set(VISUAL_RENDERERS) == {page.visual for page in PAGE_SPECS}


def test_calibration_tau_label_stays_inside_visual_column(tmp_path: Path) -> None:
    result = build_visual_big_picture(tmp_path, render_pngs=False)
    positions: list[tuple[str, float, float]] = []

    def visit_text(text, _cm, text_matrix, _font, font_size) -> None:
        if "tau = 99.5th percentile" in text:
            positions.append((text.strip(), float(text_matrix[4]), float(font_size)))

    PdfReader(str(result["pdf"])).pages[7].extract_text(visitor_text=visit_text)

    assert positions
    for text, x, size in positions:
        assert x + stringWidth(text, "Helvetica-Bold", size) <= 527


def test_multiscale_captions_do_not_overlap(tmp_path: Path) -> None:
    result = build_visual_big_picture(tmp_path, render_pngs=False)
    positions: dict[str, float] = {}

    def visit_text(text, _cm, text_matrix, _font, _font_size) -> None:
        stripped = text.strip()
        if stripped in {"full image + local crops", "768 x 768 crops | 25% overlap"}:
            positions[stripped] = float(text_matrix[5])

    PdfReader(str(result["pdf"])).pages[9].extract_text(visitor_text=visit_text)

    assert set(positions) == {"full image + local crops", "768 x 768 crops | 25% overlap"}
    assert (
        abs(positions["full image + local crops"] - positions["768 x 768 crops | 25% overlap"])
        >= 12
    )


def test_cli_builds_the_named_output_package(tmp_path: Path) -> None:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "render_visual_big_picture.py"),
        "--output-dir",
        str(tmp_path),
    ]

    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "visual_big_picture.pdf" in result.stdout
    assert (tmp_path / "visual_big_picture.pdf").is_file()
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_labels"] == [
        "real experiment example",
        "explanatory illustration",
        "real example + explanatory overlay",
    ]
    assert manifest["reported_f1"] == {
        method: {str(k): value for k, value in series.items()}
        for method, series in load_f1_table().items()
    }


def test_pdf_page_size_is_a4_landscape(tmp_path: Path) -> None:
    result = build_visual_big_picture(tmp_path, render_pngs=False)
    reader = PdfReader(str(result["pdf"]))
    width = float(reader.pages[0].mediabox.width)
    height = float(reader.pages[0].mediabox.height)

    assert abs(width - 841.89) < 0.2
    assert abs(height - 595.28) < 0.2
