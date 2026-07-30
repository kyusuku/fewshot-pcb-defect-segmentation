# Visual Big Picture PDF Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a verified 14-page A4 landscape visual atlas that explains every step of the few-shot PCB defect-segmentation pipeline with real VisA examples and deterministic teaching diagrams.

**Architecture:** A single focused Python renderer owns the page registry, validated evidence loading, ReportLab composition, readable Markdown export, and provenance manifest. Focused tests lock the 14-page/16-step contract, evidence identity, frozen metric values, and output structure. Poppler renders every PDF page to PNG for final visual inspection; no DINOv2, SAM2, or experiment rerun is required.

**Tech Stack:** Python 3.11 bundled runtime, Pillow, ReportLab, PyYAML, pypdf, pytest, Ruff, Poppler (`pdfinfo`, `pdftoppm`).

---

## Execution Context

Implement in the current checkout, not a new worktree. The teaching PDF needs ignored local VisA images under `data/processed/`, which a fresh worktree would not contain. Preserve all unrelated working-tree changes and stage only the paths named in each task.

Use these bundled tools:

```bash
CODEX_PDF_PY=/Users/kyusuku/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3
CODEX_PDF_BIN=/Users/kyusuku/.cache/codex-runtimes/codex-primary-runtime/dependencies/bin/override
```

## File Map

- Create `scripts/render_visual_big_picture.py`: page/content registry, asset validation and cropping, ReportLab drawing, Markdown/manifest export, preview rendering, CLI.
- Create `tests/test_render_visual_big_picture.py`: content, provenance, metric, PDF, Markdown, and preview contracts.
- Generate `output/visual_big_picture/visual_big_picture.pdf`: final PDF.
- Generate `output/visual_big_picture/visual_big_picture.md`: readable source exported from the same page registry.
- Generate `output/visual_big_picture/manifest.json`: provenance, source hashes, page metadata, and output hashes.
- Generate `output/visual_big_picture/pages/page-01.png` through `page-14.png`: visual QA renders.
- Generate `output/visual_big_picture/preview-contact-sheet.png`: one-image overview for final inspection.

Do not modify `docs/evidence/generated/`, the final report, notebooks, configs, or dataset files.

### Task 1: Lock the 14-page and 16-step content contract

**Files:**
- Create: `tests/test_render_visual_big_picture.py`
- Create: `scripts/render_visual_big_picture.py`

- [ ] **Step 1: Write the failing registry tests**

Create `tests/test_render_visual_big_picture.py` with:

```python
from __future__ import annotations

from scripts.render_visual_big_picture import PAGE_SPECS, collect_step_numbers


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
```

- [ ] **Step 2: Run the tests and verify the import fails**

Run:

```bash
"$CODEX_PDF_PY" -m pytest tests/test_render_visual_big_picture.py -q
```

Expected: collection fails because `scripts.render_visual_big_picture` does not exist.

- [ ] **Step 3: Add the page data model and complete registry**

Create `scripts/render_visual_big_picture.py` with the following data model and registry. Keep the exact wording compact enough for A4 landscape pages.

```python
#!/usr/bin/env python
"""Render the illustrated big-picture guide for the PCB segmentation pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "visual_big_picture"


@dataclass(frozen=True)
class PageSpec:
    number: int
    title: str
    steps: tuple[int, ...]
    source_kind: str
    visual: str
    what_enters: str
    what_happens: str
    what_comes_out: str
    what_to_notice: str


PAGE_SPECS = (
    PageSpec(1, "The inspection question", (), "mixed", "overview",
             "A few defect-free boards and one new PCB image.",
             "Frozen DINOv2 finds unusual pixels; frozen SAM2 follows prompted boundaries; exact intersection keeps their agreement.",
             "A predicted binary defect mask.",
             "Ground truth is absent from the prediction path."),
    PageSpec(2, "Two models, two jobs", (), "mixed", "roles",
             "An image plus normal reference features or spatial prompts.",
             "DINOv2 supplies anomaly evidence. SAM2 supplies coherent visual regions.",
             "A score map from DINOv2 and a region mask from SAM2.",
             "Neither pretrained model directly understands the full task."),
    PageSpec(3, "Choose the dataset", (1,), "mixed", "categories",
             "VisA fold 0 with pcb1, pcb2, pcb3, and pcb4.",
             "Each PCB category is treated as its own anomaly-detection problem.",
             "Four separate category pipelines and four separate normal memories.",
             "Normal patches are never mixed across PCB categories. VisA is the pixel benchmark; DeepPCB boxes are not true masks."),
    PageSpec(4, "Give every image one role", (2, 3), "mixed", "roles_sampling",
             "Normal development images, held-out normal calibration images, and test images.",
             "Select k in {1, 2, 4} normal supports and repeat the selection with five fixed seeds.",
             "A support set, a separate calibration set, and an untouched test set.",
             "The protocol uses few support images, but also uses normal-only calibration images."),
    PageSpec(5, "Preprocess without losing alignment", (4,), "mixed", "preprocess",
             "One selected normal RGB image and its valid content extent.",
             "Resize with aspect ratio preserved, center on a 518 x 518 canvas, normalize, and mark padding invalid.",
             "A DINOv2-ready tensor plus a valid-content mask.",
             "Padding must not become fake normal PCB evidence."),
    PageSpec(6, "Describe the PCB as patches", (5,), "mixed", "features",
             "A 518 x 518 preprocessed support image.",
             "Frozen ViT-S/14 emits one local feature vector per 14 x 14 patch.",
             "A 37 x 37 grid containing up to 1,369 local feature vectors.",
             "One support image is many local normal examples, not one number."),
    PageSpec(7, "Build the normal memory", (6,), "illustration", "memory",
             "Valid DINOv2 patch vectors from k normal boards of one category.",
             "Concatenate every valid vector into that category's memory bank.",
             "A searchable collection of normal local appearances.",
             "The primary DINOv2 method keeps the complete valid bank."),
    PageSpec(8, "Calibrate suspicion using only normal images", (7,), "illustration", "calibration",
             "An independent normal validation set and the category memory bank.",
             "Score validation pixels and choose tau at the 0.995 normal-score quantile.",
             "A category-specific threshold tau.",
             "No anomaly mask or test ground truth chooses the threshold."),
    PageSpec(9, "Compare a new PCB with normal memory", (8,), "mixed", "query_compare",
             "A new query PCB and its category's normal memory bank.",
             "For each query patch, measure the distance to its nearest normal-memory patch.",
             "A coarse grid of anomaly distances.",
             "Large distance means no stored normal patch looks similar."),
    PageSpec(10, "Project a multi-scale anomaly heatmap", (9,), "mixed", "multiscale",
             "Scores from the full image and overlapping 768 x 768 crops.",
             "Map every view back to original coordinates and keep the maximum where views overlap.",
             "A continuous full-resolution anomaly heatmap.",
             "Multi-scale was motivated by small defects, but was not reliably better than single-scale."),
    PageSpec(11, "Turn scores into automatic prompts", (10, 11), "mixed", "proposal_prompts",
             "The heatmap and normal-only threshold tau.",
             "Threshold to proposal A, split four-connected components, remove tiny regions, and make one box plus one peak point per retained component.",
             "A binary anomaly proposal and automatic SAM2 prompts.",
             "The proposal may be fragmented or noisy; it is evidence, not the final outline."),
    PageSpec(12, "Let SAM2 follow coherent regions", (12, 13), "mixed", "sam2_union",
             "The original query image plus anomaly-generated point and box prompts.",
             "Frozen SAM2 predicts candidate masks; selection uses confidence, alignment, prompt containment, and an area cap before masks are combined.",
             "The Guided SAM2 mask S.",
             "SAM2 can outline the wrong normal structure because it knows grouping, not defectiveness."),
    PageSpec(13, "Keep only exact agreement", (14, 15), "mixed", "intersection_limit",
             "The calibrated anomaly proposal A and Guided SAM2 mask S.",
             "Compute AC-SAM2 = A intersect S pixel by pixel.",
             "The anomaly-consistent final mask.",
             "Intersection can remove false positives but cannot recover a pixel missing from A or S."),
    PageSpec(14, "Evaluate only after prediction", (16,), "real", "evaluation",
             "A completed prediction and the previously hidden test mask.",
             "Measure precision, recall, F1, IoU, and continuous-map metrics; aggregate across categories, shots, and seeds.",
             "Post-hoc metrics and the verified support-size F1 trend.",
             "DINOv2 inspects, SAM2 outlines, and AC-SAM2 keeps only their agreement."),
)


def collect_step_numbers(pages: Sequence[PageSpec]) -> list[int]:
    return [step for page in pages for step in page.steps]
```

- [ ] **Step 4: Run the registry tests**

Run:

```bash
"$CODEX_PDF_PY" -m pytest tests/test_render_visual_big_picture.py -q
```

Expected: `2 passed`.

- [ ] **Step 5: Commit the content contract**

```bash
git add scripts/render_visual_big_picture.py tests/test_render_visual_big_picture.py
git commit -m "test: lock visual guide page contract"
```

### Task 2: Validate and prepare real evidence assets

**Files:**
- Modify: `scripts/render_visual_big_picture.py`
- Modify: `tests/test_render_visual_big_picture.py`

- [ ] **Step 1: Add failing asset and frozen-value tests**

Add the imports to the top of the test module, extend the existing renderer import, and append these tests:

```python
import json
from pathlib import Path

from PIL import Image

from scripts.render_visual_big_picture import (
    PROJECT_ROOT,
    extract_real_assets,
    load_f1_table,
    sha256_file,
    validate_source_assets,
)


def test_source_assets_exist_and_frozen_montages_match_manifest() -> None:
    validated = validate_source_assets()
    assert validated["pcb1_success"]
    assert validated["pcb1_failure"]
    assert all(validated[f"pcb{index}_normal"] for index in range(1, 5))


def test_real_asset_extraction_preserves_expected_panel_size(tmp_path: Path) -> None:
    assets = extract_real_assets(tmp_path)
    for name in ("success_anomaly", "success_guided", "success_ac", "failure_guided", "failure_ac"):
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
```

- [ ] **Step 2: Run the new tests and verify missing-symbol failures**

Run:

```bash
"$CODEX_PDF_PY" -m pytest tests/test_render_visual_big_picture.py -q
```

Expected: collection fails because the asset and metric helpers do not exist.

- [ ] **Step 3: Implement source validation, montage cropping, and metric loading**

Add imports and constants:

```python
import csv
import hashlib
import json
import shutil

from PIL import Image


EVIDENCE_DIR = PROJECT_ROOT / "docs" / "evidence" / "generated"
QUALITATIVE_DIR = EVIDENCE_DIR / "qualitative_figures"
DATASET_ROOT = PROJECT_ROOT / "data" / "processed" / "VisA_pytorch" / "1cls"
NORMAL_EXAMPLES = {
    f"pcb{index}_normal": DATASET_ROOT / f"pcb{index}" / "train" / "good" / "0000.JPG"
    for index in range(1, 5)
}
REAL_CASES = {
    "success_query": DATASET_ROOT / "pcb1" / "test" / "bad" / "085.JPG",
    "success_ground_truth": DATASET_ROOT / "pcb1" / "ground_truth" / "bad" / "085.png",
    "failure_query": DATASET_ROOT / "pcb1" / "test" / "bad" / "054.JPG",
    "failure_ground_truth": DATASET_ROOT / "pcb1" / "ground_truth" / "bad" / "054.png",
}
PANEL_INDEX = {"input": 0, "ground_truth": 1, "anomaly": 2, "guided": 3, "ac": 4}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_source_assets() -> dict[str, bool]:
    manifest_path = QUALITATIVE_DIR / "qualitative_figure_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    figures = {entry["path"]: entry for entry in manifest["figures"]}
    checked: dict[str, bool] = {}
    for role in ("success", "failure"):
        filename = f"pcb1_{role}.png"
        path = QUALITATIVE_DIR / filename
        checked[f"pcb1_{role}"] = path.is_file() and sha256_file(path) == figures[filename]["sha256"]
    for name, path in NORMAL_EXAMPLES.items():
        checked[name] = path.is_file()
    for name, path in REAL_CASES.items():
        checked[name] = path.is_file()
    if not all(checked.values()):
        missing = sorted(name for name, valid in checked.items() if not valid)
        raise ValueError(f"missing or mismatched visual-guide sources: {missing}")
    return checked


def _crop_montage_panel(montage: Path, panel_name: str, destination: Path) -> None:
    index = PANEL_INDEX[panel_name]
    x = 24 + index * (280 + 18)
    y = 24 + 58
    with Image.open(montage) as image:
        panel = image.convert("RGB").crop((x, y, x + 280, y + 280))
    panel.save(destination)


def extract_real_assets(work_dir: Path) -> dict[str, Path]:
    validate_source_assets()
    work_dir.mkdir(parents=True, exist_ok=True)
    assets: dict[str, Path] = {}
    for name, source in {**NORMAL_EXAMPLES, **REAL_CASES}.items():
        destination = work_dir / f"{name}{source.suffix.lower()}"
        shutil.copyfile(source, destination)
        assets[name] = destination
    panel_names = {"anomaly": "anomaly", "guided": "guided", "ac": "ac"}
    for role in ("success", "failure"):
        montage = QUALITATIVE_DIR / f"pcb1_{role}.png"
        for suffix, panel_name in panel_names.items():
            destination = work_dir / f"{role}_{suffix}.png"
            _crop_montage_panel(montage, panel_name, destination)
            assets[f"{role}_{suffix}"] = destination
    return assets


def load_f1_table() -> dict[str, dict[int, float]]:
    method_labels = {
        "dinov2_multi": "Multi-DINO",
        "dinov2_multi_sam2": "Guided SAM2",
        "anomaly_consistent_sam2": "AC-SAM2",
    }
    values = {label: {} for label in method_labels.values()}
    with (EVIDENCE_DIR / "primary_summary.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if (
                row["method"] in method_labels
                and row["category"] == "macro"
                and row["metric"] == "mean_anomaly_mask_f1"
                and int(row["k"]) in {1, 2, 4}
            ):
                values[method_labels[row["method"]]][int(row["k"])] = round(float(row["mean"]), 3)
    if any(set(series) != {1, 2, 4} for series in values.values()):
        raise ValueError("frozen F1 table is incomplete")
    return values
```

- [ ] **Step 4: Run the asset tests**

Run:

```bash
"$CODEX_PDF_PY" -m pytest tests/test_render_visual_big_picture.py -q
```

Expected: `5 passed`.

- [ ] **Step 5: Commit validated evidence loading**

```bash
git add scripts/render_visual_big_picture.py tests/test_render_visual_big_picture.py
git commit -m "feat: validate visual guide evidence sources"
```

### Task 3: Add deterministic ReportLab page composition

**Files:**
- Modify: `scripts/render_visual_big_picture.py`
- Modify: `tests/test_render_visual_big_picture.py`

- [ ] **Step 1: Add failing PDF and source-output tests**

Extend the test-module imports and append:

```python
from pypdf import PdfReader

from scripts.render_visual_big_picture import build_visual_big_picture


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
```

- [ ] **Step 2: Run the new test and verify it fails**

Run:

```bash
"$CODEX_PDF_PY" -m pytest tests/test_render_visual_big_picture.py::test_build_writes_pdf_markdown_manifest_and_fourteen_previews -q
```

Expected: collection fails because `build_visual_big_picture` does not exist.

- [ ] **Step 3: Add drawing primitives and the build pipeline**

Add the following imports, page constants, and helpers. Use only ASCII hyphens in text rendered into the PDF.

```python
import argparse
import os
import subprocess
import tempfile

from PIL import ImageDraw, ImageFont, ImageOps
from reportlab.lib.colors import HexColor, white
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas


PAGE_SIZE = landscape(A4)
PAGE_WIDTH, PAGE_HEIGHT = PAGE_SIZE
COLORS = {
    "ink": HexColor("#17324D"),
    "muted": HexColor("#52677D"),
    "line": HexColor("#D6DEE7"),
    "paper": HexColor("#F7F9FC"),
    "blue": HexColor("#D7E9FF"),
    "blue_strong": HexColor("#3C78B5"),
    "amber": HexColor("#FFE3AE"),
    "amber_strong": HexColor("#C77900"),
    "teal": HexColor("#CCEBD7"),
    "teal_strong": HexColor("#32847A"),
    "navy": HexColor("#15395F"),
}


def _wrap(text: str, font: str, size: float, max_width: float) -> list[str]:
    lines: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if current and stringWidth(candidate, font, size) > max_width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def _text(c: canvas.Canvas, x: float, y: float, text: str, *, size: float = 12,
          color=COLORS["ink"], max_width: float = 250, leading: float | None = None) -> float:
    leading = leading or size * 1.28
    c.setFillColor(color)
    c.setFont("Helvetica", size)
    for line in _wrap(text, "Helvetica", size, max_width):
        c.drawString(x, y, line)
        y -= leading
    return y


def _rounded_box(c: canvas.Canvas, x: float, y: float, width: float, height: float,
                 fill, *, stroke=COLORS["line"], radius: float = 10) -> None:
    c.setFillColor(fill)
    c.setStrokeColor(stroke)
    c.roundRect(x, y, width, height, radius, fill=1, stroke=1)


def _arrow(c: canvas.Canvas, x1: float, y1: float, x2: float, y2: float,
           *, color=COLORS["ink"]) -> None:
    c.setStrokeColor(color)
    c.setFillColor(color)
    c.setLineWidth(2)
    c.line(x1, y1, x2, y2)
    c.line(x2, y2, x2 - 9, y2 + 5)
    c.line(x2, y2, x2 - 9, y2 - 5)


def _draw_image(c: canvas.Canvas, path: Path, x: float, y: float, width: float,
                height: float) -> None:
    with Image.open(path) as image:
        fitted = ImageOps.contain(image.convert("RGB"), (int(width * 2), int(height * 2)))
    c.drawImage(ImageReader(fitted), x + (width - fitted.width / 2) / 2,
                y + (height - fitted.height / 2) / 2,
                fitted.width / 2, fitted.height / 2, preserveAspectRatio=True, mask="auto")


def _header(c: canvas.Canvas, page: PageSpec) -> None:
    c.setFillColor(COLORS["paper"])
    c.rect(0, 0, PAGE_WIDTH, PAGE_HEIGHT, fill=1, stroke=0)
    c.setFillColor(COLORS["muted"])
    c.setFont("Helvetica-Bold", 10)
    label = "OVERVIEW" if not page.steps else "STEPS " + ", ".join(map(str, page.steps))
    c.drawString(42, PAGE_HEIGHT - 40, label)
    c.setFillColor(COLORS["ink"])
    c.setFont("Helvetica-Bold", 25)
    c.drawString(42, PAGE_HEIGHT - 72, page.title)
    source = {"real": "REAL EXPERIMENT EXAMPLE", "illustration": "EXPLANATORY ILLUSTRATION", "mixed": "REAL EXAMPLE + EXPLANATORY OVERLAY"}[page.source_kind]
    c.setFont("Helvetica-Bold", 8)
    c.setFillColor(COLORS["muted"])
    c.drawRightString(PAGE_WIDTH - 42, PAGE_HEIGHT - 40, source)


def _teaching_copy(c: canvas.Canvas, page: PageSpec) -> None:
    x, y, width = 555, 435, 245
    for label, body in (
        ("WHAT ENTERS", page.what_enters),
        ("WHAT HAPPENS", page.what_happens),
        ("WHAT COMES OUT", page.what_comes_out),
        ("WHAT TO NOTICE", page.what_to_notice),
    ):
        c.setFillColor(COLORS["muted"])
        c.setFont("Helvetica-Bold", 8)
        c.drawString(x, y, label)
        y = _text(c, x, y - 17, body, size=10.5, max_width=width, leading=13.5) - 13


def _draw_page(c: canvas.Canvas, page: PageSpec, assets: dict[str, Path],
               f1: dict[str, dict[int, float]]) -> None:
    _header(c, page)
    _draw_visual(c, page.visual, assets, f1)
    _teaching_copy(c, page)
    c.setFillColor(COLORS["muted"])
    c.setFont("Helvetica", 8)
    c.drawString(42, 24, "Few-Shot PCB Defect Segmentation - visual big picture")
    c.drawRightString(PAGE_WIDTH - 42, 24, f"{page.number} / 14")


def _write_markdown(path: Path) -> None:
    parts = ["# Visual Big Picture\n"]
    for page in PAGE_SPECS:
        steps = "Overview" if not page.steps else "Steps " + ", ".join(map(str, page.steps))
        parts.extend([
            f"## Page {page.number}: {page.title}\n",
            f"**{steps} - {page.source_kind}.**\n",
            f"- **What enters:** {page.what_enters}\n",
            f"- **What happens:** {page.what_happens}\n",
            f"- **What comes out:** {page.what_comes_out}\n",
            f"- **What to notice:** {page.what_to_notice}\n",
        ])
    path.write_text("\n".join(parts), encoding="utf-8")


def _render_previews(pdf_path: Path, pages_dir: Path) -> list[Path]:
    pages_dir.mkdir(parents=True, exist_ok=True)
    for old_preview in pages_dir.glob("page-*.png"):
        old_preview.unlink()
    prefix = pages_dir / "page"
    executable = os.environ.get("PDFTOPPM", "pdftoppm")
    subprocess.run([executable, "-png", "-r", "120", str(pdf_path), str(prefix)], check=True)
    generated = sorted(
        pages_dir.glob("page-*.png"),
        key=lambda path: int(path.stem.rsplit("-", 1)[1]),
    )
    renamed: list[Path] = []
    for index, path in enumerate(generated, start=1):
        destination = pages_dir / f"page-{index:02d}.png"
        if path != destination:
            path.replace(destination)
        renamed.append(destination)
    return renamed


def _contact_sheet(page_paths: Sequence[Path], output_path: Path) -> None:
    thumbs = []
    for path in page_paths:
        with Image.open(path) as image:
            thumbs.append(ImageOps.contain(image.convert("RGB"), (420, 297)))
    sheet = Image.new("RGB", (1260, 5 * 325), "#E8EDF3")
    for index, thumb in enumerate(thumbs):
        x = (index % 3) * 420
        y = (index // 3) * 325
        sheet.paste(thumb, (x, y))
    sheet.save(output_path)


def build_visual_big_picture(output_dir: Path, *, render_pngs: bool = True) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "visual_big_picture.pdf"
    markdown_path = output_dir / "visual_big_picture.md"
    manifest_path = output_dir / "manifest.json"
    contact_sheet = output_dir / "preview-contact-sheet.png"
    f1 = load_f1_table()
    with tempfile.TemporaryDirectory(prefix="visual-big-picture-") as temp_dir:
        assets = extract_real_assets(Path(temp_dir))
        pdf = canvas.Canvas(str(pdf_path), pagesize=PAGE_SIZE, pageCompression=1)
        pdf.setTitle("Visual Big Picture - Few-Shot PCB Defect Segmentation")
        for page in PAGE_SPECS:
            _draw_page(pdf, page, assets, f1)
            pdf.showPage()
        pdf.save()
    _write_markdown(markdown_path)
    page_paths = _render_previews(pdf_path, output_dir / "pages") if render_pngs else []
    if page_paths:
        _contact_sheet(page_paths, contact_sheet)
    manifest = {
        "schema_version": 1,
        "page_count": len(PAGE_SPECS),
        "step_numbers": collect_step_numbers(PAGE_SPECS),
        "pages": [{"number": page.number, "title": page.title, "steps": list(page.steps), "source_kind": page.source_kind} for page in PAGE_SPECS],
        "source_sha256": {
            "pcb1_success.png": sha256_file(QUALITATIVE_DIR / "pcb1_success.png"),
            "pcb1_failure.png": sha256_file(QUALITATIVE_DIR / "pcb1_failure.png"),
            "primary_summary.csv": sha256_file(EVIDENCE_DIR / "primary_summary.csv"),
        },
        "pdf_sha256": sha256_file(pdf_path),
        "markdown_sha256": sha256_file(markdown_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"pdf": pdf_path, "markdown": markdown_path, "manifest": manifest_path,
            "contact_sheet": contact_sheet, "pages": page_paths}
```

- [ ] **Step 4: Add a temporary `_draw_visual` stub and confirm the structural test passes**

Add this temporary minimal function; Task 4 replaces its body with all 14 visuals:

```python
def _draw_visual(c: canvas.Canvas, visual: str, assets: dict[str, Path],
                 f1: dict[str, dict[int, float]]) -> None:
    _rounded_box(c, 42, 82, 485, 390, white)
    c.setFillColor(COLORS["ink"])
    c.setFont("Helvetica-Bold", 18)
    c.drawCentredString(284, 275, visual.replace("_", " ").title())
```

Run:

```bash
PATH="$CODEX_PDF_BIN:$PATH" "$CODEX_PDF_PY" -m pytest tests/test_render_visual_big_picture.py::test_build_writes_pdf_markdown_manifest_and_fourteen_previews -q
```

Expected: `1 passed`.

- [ ] **Step 5: Commit the output pipeline**

```bash
git add scripts/render_visual_big_picture.py tests/test_render_visual_big_picture.py
git commit -m "feat: add visual guide PDF pipeline"
```

### Task 4: Implement the fourteen page visuals

**Files:**
- Modify: `scripts/render_visual_big_picture.py`
- Modify: `tests/test_render_visual_big_picture.py`

- [ ] **Step 1: Add a failing visual-coverage test**

Extend the renderer import and append:

```python
from scripts.render_visual_big_picture import VISUAL_RENDERERS


def test_every_page_visual_has_a_renderer() -> None:
    assert set(VISUAL_RENDERERS) == {page.visual for page in PAGE_SPECS}
```

- [ ] **Step 2: Run the coverage test and verify it fails**

Run:

```bash
"$CODEX_PDF_PY" -m pytest tests/test_render_visual_big_picture.py::test_every_page_visual_has_a_renderer -q
```

Expected: collection fails because `VISUAL_RENDERERS` does not exist.

- [ ] **Step 3: Replace the stub with the complete renderer dispatch**

Implement one function per page visual and use this exact mapping:

```python
VISUAL_RENDERERS = {
    "overview": _visual_overview,
    "roles": _visual_roles,
    "categories": _visual_categories,
    "roles_sampling": _visual_roles_sampling,
    "preprocess": _visual_preprocess,
    "features": _visual_features,
    "memory": _visual_memory,
    "calibration": _visual_calibration,
    "query_compare": _visual_query_compare,
    "multiscale": _visual_multiscale,
    "proposal_prompts": _visual_proposal_prompts,
    "sam2_union": _visual_sam2_union,
    "intersection_limit": _visual_intersection_limit,
    "evaluation": _visual_evaluation,
}


def _draw_visual(c: canvas.Canvas, visual: str, assets: dict[str, Path],
                 f1: dict[str, dict[int, float]]) -> None:
    VISUAL_RENDERERS[visual](c, assets, f1)
```

The functions must follow these deterministic compositions:

1. `_visual_overview`: two horizontal lanes. Top lane shows three normal thumbnails -> blue memory block -> amber heatmap. Bottom lane shows query thumbnail -> teal SAM2 block -> navy final mask. A lock icon/text places ground truth below a dashed “evaluation only” line, never on a prediction arrow.
2. `_visual_roles`: two equal cards. Left overlays a 7 x 7 patch grid on a normal thumbnail and labels “feature vectors -> nearest-normal distance.” Right shows the success query with one amber point and box leading to the Guided SAM2 crop. A final narrow strip shows `A intersect S`.
3. `_visual_categories`: four real normal thumbnails from `pcb1_normal` through `pcb4_normal`, each labeled. Draw four separate blue memory cylinders below; no connecting arrows between categories.
4. `_visual_roles_sampling`: three columns labeled Support, Calibration, Test. Support shows k = 1, 2, 4 thumbnail counts and seeds 4880-4884. Calibration shows a normal-only score strip ending at `tau = Q_0.995`. Test shows success query with a closed-eye/lock label “mask hidden until evaluation.”
5. `_visual_preprocess`: original normal thumbnail -> aspect-preserving contained image on a square canvas -> 518 x 518 image with a pale padding area and blue valid-content rectangle. Do not stretch the PCB.
6. `_visual_features`: real normal thumbnail with a 37 x 37 grid, a zoomed 5 x 5 patch area, and three vector bars. Label `518 / 14 = 37` and `37 x 37 = 1,369 patches`.
7. `_visual_memory`: three small support-image grids feed rows of blue vector bars into one cylinder labeled `M_pcb1`. A query vector outside the bank is shown only as the next-step preview, not inserted into memory.
8. `_visual_calibration`: draw a deterministic histogram-like sequence of 200 blue bars whose heights use `0.18 + 0.55 * exp(-((i - 75) / 48)^2) + 0.12 * sin(i / 9)^2`. Mark the 199th bar with an amber line labeled `tau = 99.5th percentile`; label all samples “held-out normal pixels.”
9. `_visual_query_compare`: success query at left, three highlighted query patches in the center, and three nearest-memory comparisons at right. Two comparisons use short blue distance arrows; one uses a long amber arrow and is labeled “high anomaly score.” Show `a_p(x) = min ||f(x)_p - m||_2` below.
10. `_visual_multiscale`: success query with one full-image frame plus four overlapping crop rectangles; arrows merge into the real `success_anomaly` panel. Label fusion “map back + pixelwise maximum,” and add the caution “motivation, not a proven gain.”
11. `_visual_proposal_prompts`: real anomaly panel -> a deterministic threshold illustration derived by converting that panel to grayscale and selecting pixels >= its 92nd percentile for display only -> connected-component outlines -> success query with amber boxes and peak points. Label the thresholded panel “explanatory proposal view; tau is calibrated in the real pipeline.”
12. `_visual_sam2_union`: success query with prompt overlays -> three overlapping teal candidate-shape diagrams -> real `success_guided` panel. Add a small `area <= 25%` selection gate and labels for confidence, anomaly alignment, containment, and area.
13. `_visual_intersection_limit`: top row shows proposal A illustration, real `success_guided`, and real `success_ac` with the AND symbol. Bottom row uses the full verified `pcb1_failure.png` montage, cropped only to fit, with caption “Failure case: intersection cannot restore missing proposal pixels.”
14. `_visual_evaluation`: success query prediction beside `success_ground_truth` with the ground-truth lock now open; a three-series line chart reads values from `f1`; a small experiment tree shows `4 categories x 3 shots x 5 seeds x 6 support-dependent methods = 360`, plus `4 SAM2-only runs = 364`.

For reusable masks and prompt overlays, add these exact helpers:

```python
def _illustrative_proposal(anomaly_path: Path) -> Image.Image:
    with Image.open(anomaly_path) as image:
        gray = image.convert("L")
    values = sorted(gray.getdata())
    threshold = values[int(0.92 * (len(values) - 1))]
    return gray.point(lambda value: 255 if value >= threshold else 0).convert("RGB")


def _draw_prompt_overlay(c: canvas.Canvas, image_path: Path, x: float, y: float,
                         width: float, height: float) -> None:
    _draw_image(c, image_path, x, y, width, height)
    c.setStrokeColor(COLORS["amber_strong"])
    c.setFillColor(COLORS["amber_strong"])
    c.setLineWidth(2)
    c.rect(x + width * 0.43, y + height * 0.42, width * 0.27, height * 0.18, fill=0, stroke=1)
    c.circle(x + width * 0.57, y + height * 0.51, 4, fill=1, stroke=0)
```

All page functions must draw within the visual area `x=42..527`, `y=82..472`. Reserve `x=555..800` for teaching copy. Keep every body label at 9 pt or larger.

- [ ] **Step 4: Run the complete focused test file**

Run:

```bash
PATH="$CODEX_PDF_BIN:$PATH" "$CODEX_PDF_PY" -m pytest tests/test_render_visual_big_picture.py -q
```

Expected: `7 passed`.

- [ ] **Step 5: Commit the completed page visuals**

```bash
git add scripts/render_visual_big_picture.py tests/test_render_visual_big_picture.py
git commit -m "feat: illustrate PCB pipeline visual atlas"
```

### Task 5: Add CLI, provenance detail, and automated output checks

**Files:**
- Modify: `scripts/render_visual_big_picture.py`
- Modify: `tests/test_render_visual_big_picture.py`

- [ ] **Step 1: Add failing CLI and manifest tests**

Extend the test-module imports and append:

```python
import subprocess
import sys


def test_cli_builds_the_named_output_package(tmp_path: Path) -> None:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "render_visual_big_picture.py"),
        "--output-dir",
        str(tmp_path),
    ]
    result = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True, check=False)
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
```

- [ ] **Step 2: Run the new tests and verify the CLI is absent**

Run:

```bash
PATH="$CODEX_PDF_BIN:$PATH" "$CODEX_PDF_PY" -m pytest tests/test_render_visual_big_picture.py::test_cli_builds_the_named_output_package tests/test_render_visual_big_picture.py::test_pdf_page_size_is_a4_landscape -q
```

Expected: the page-size test passes; the CLI test fails because the script does not parse arguments or print its output path.

- [ ] **Step 3: Complete the manifest and CLI**

Add to the manifest created in `build_visual_big_picture`:

```python
"source_labels": [
    "real experiment example",
    "explanatory illustration",
    "real example + explanatory overlay",
],
"reported_f1": f1,
"notes": [
    "No generative-image model was used.",
    "The 92nd-percentile proposal view is an explanatory visualization, not a reconstruction of the frozen calibrated proposal.",
    "Ground truth is used only on evaluation or explicitly post-hoc comparison pages.",
],
```

Add the CLI:

```python
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--skip-previews", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_visual_big_picture(args.output_dir, render_pngs=not args.skip_previews)
    print(result["pdf"])


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests and lint**

Run:

```bash
PATH="$CODEX_PDF_BIN:$PATH" "$CODEX_PDF_PY" -m pytest tests/test_render_visual_big_picture.py -q
"$CODEX_PDF_PY" -m ruff check scripts/render_visual_big_picture.py tests/test_render_visual_big_picture.py
"$CODEX_PDF_PY" -m ruff format --check scripts/render_visual_big_picture.py tests/test_render_visual_big_picture.py
```

Expected: all tests pass; Ruff reports no lint or formatting errors.

- [ ] **Step 5: Commit the CLI and provenance contract**

```bash
git add scripts/render_visual_big_picture.py tests/test_render_visual_big_picture.py
git commit -m "feat: verify visual guide output package"
```

### Task 6: Build and visually verify the final PDF

**Files:**
- Generate: `output/visual_big_picture/visual_big_picture.pdf`
- Generate: `output/visual_big_picture/visual_big_picture.md`
- Generate: `output/visual_big_picture/manifest.json`
- Generate: `output/visual_big_picture/pages/page-01.png` through `page-14.png`
- Generate: `output/visual_big_picture/preview-contact-sheet.png`

- [ ] **Step 1: Build the final package with the exact test command**

Run:

```bash
PATH="$CODEX_PDF_BIN:$PATH" "$CODEX_PDF_PY" scripts/render_visual_big_picture.py --output-dir output/visual_big_picture
```

Expected: stdout ends with `output/visual_big_picture/visual_big_picture.pdf` and the command exits 0. This is a presentation-only build; it reads local VisA examples and frozen evidence but does not load DINOv2/SAM2 weights or run inference.

- [ ] **Step 2: Verify PDF metadata and page count**

Run:

```bash
PATH="$CODEX_PDF_BIN:$PATH" pdfinfo output/visual_big_picture/visual_big_picture.pdf
```

Expected: `Pages: 14`, `Page size: 841.89 x 595.276 pts (A4)`, and no syntax error.

- [ ] **Step 3: Verify output and source contracts**

Run:

```bash
PATH="$CODEX_PDF_BIN:$PATH" "$CODEX_PDF_PY" -m pytest tests/test_render_visual_big_picture.py -q
"$CODEX_PDF_PY" -m ruff check scripts/render_visual_big_picture.py tests/test_render_visual_big_picture.py
git diff --check -- scripts/render_visual_big_picture.py tests/test_render_visual_big_picture.py
```

Expected: all focused tests pass, Ruff passes, and `git diff --check` emits no output.

- [ ] **Step 4: Inspect every rendered page**

Open `output/visual_big_picture/preview-contact-sheet.png`, then inspect individual PNGs at original resolution. Check:

- all 14 page titles and page numbers;
- every step number 1 through 16 appears once and in order;
- no clipped, overlapping, tiny, or missing text;
- PCB images preserve aspect ratio;
- diagrams stay inside the left visual area;
- real/mixed/illustration labels are correct;
- ground truth appears only on evaluation or explicitly post-hoc comparison pages;
- the success and failure masks remain legible;
- the results chart reads 0.210/0.228/0.244, 0.257/0.284/0.303, and 0.268/0.297/0.317;
- the last page reads 360 support-dependent runs plus 4 SAM2-only runs equals 364.

If any page fails, change only the renderer, rebuild the complete package, and repeat the page inspection from the new PNGs.

- [ ] **Step 5: Confirm the worktree scope**

Run:

```bash
git status --short
git diff --stat -- scripts/render_visual_big_picture.py tests/test_render_visual_big_picture.py
```

Expected: the renderer and its test are the only implementation paths changed by this plan. Existing unrelated modifications remain untouched. The generated `output/visual_big_picture/` package may remain untracked because it is a user-facing deliverable and contains embedded dataset imagery.

- [ ] **Step 6: Report completion without committing prohibited assets**

Provide:

- a clickable link to `output/visual_big_picture/visual_big_picture.pdf`;
- a clickable link to `output/visual_big_picture/preview-contact-sheet.png`;
- the exact build and test commands above;
- the assumption that local VisA fold-0 files are present under `data/processed/VisA_pytorch/1cls/`;
- confirmation that no dependencies or source assets were missing;
- a summary of the renderer/test files created;
- the next useful step: read the PDF and request any page-specific wording or visual adjustment.

Do not commit the generated PDF, page PNGs, extracted `.assets`, or embedded dataset imagery.
