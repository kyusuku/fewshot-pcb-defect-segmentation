#!/usr/bin/env python
"""Render the illustrated big-picture guide for the PCB segmentation pipeline."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageOps
from reportlab.lib.colors import HexColor, white
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "visual_big_picture"
EVIDENCE_DIR = PROJECT_ROOT / "docs" / "evidence" / "generated"
QUALITATIVE_DIR = EVIDENCE_DIR / "qualitative_figures"
DATASET_ROOT = PROJECT_ROOT / "data" / "processed" / "VisA_pytorch" / "1cls"
NORMAL_EXAMPLES = {
    f"pcb{index}_normal": DATASET_ROOT
    / f"pcb{index}"
    / "train"
    / "good"
    / "0000.JPG"
    for index in range(1, 5)
}
REAL_CASES = {
    "success_query": DATASET_ROOT / "pcb1" / "test" / "bad" / "085.JPG",
    "success_ground_truth": DATASET_ROOT
    / "pcb1"
    / "ground_truth"
    / "bad"
    / "085.png",
    "failure_query": DATASET_ROOT / "pcb1" / "test" / "bad" / "054.JPG",
    "failure_ground_truth": DATASET_ROOT
    / "pcb1"
    / "ground_truth"
    / "bad"
    / "054.png",
}
PANEL_INDEX = {"input": 0, "ground_truth": 1, "anomaly": 2, "guided": 3, "ac": 4}
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
    PageSpec(
        1,
        "The inspection question",
        (),
        "mixed",
        "overview",
        "A few defect-free boards and one new PCB image.",
        "Frozen DINOv2 finds unusual pixels; frozen SAM2 follows prompted boundaries; "
        "exact intersection keeps their agreement.",
        "A predicted binary defect mask.",
        "Ground truth is absent from the prediction path.",
    ),
    PageSpec(
        2,
        "Two models, two jobs",
        (),
        "mixed",
        "roles",
        "An image plus normal reference features or spatial prompts.",
        "DINOv2 supplies anomaly evidence. SAM2 supplies coherent visual regions.",
        "A score map from DINOv2 and a region mask from SAM2.",
        "Neither pretrained model directly understands the full task.",
    ),
    PageSpec(
        3,
        "Choose the dataset",
        (1,),
        "mixed",
        "categories",
        "VisA fold 0 with pcb1, pcb2, pcb3, and pcb4.",
        "Each PCB category is treated as its own anomaly-detection problem.",
        "Four separate category pipelines and four separate normal memories.",
        "Normal patches are never mixed across PCB categories. VisA is the pixel "
        "benchmark; DeepPCB boxes are not true masks.",
    ),
    PageSpec(
        4,
        "Give every image one role",
        (2, 3),
        "mixed",
        "roles_sampling",
        "Normal development images, held-out normal calibration images, and test images.",
        "Select k in {1, 2, 4} normal supports and repeat the selection with five fixed "
        "seeds.",
        "A support set, a separate calibration set, and an untouched test set.",
        "The protocol uses few support images, but also uses normal-only calibration "
        "images.",
    ),
    PageSpec(
        5,
        "Preprocess without losing alignment",
        (4,),
        "mixed",
        "preprocess",
        "One selected normal RGB image and its valid content extent.",
        "Resize with aspect ratio preserved, center on a 518 x 518 canvas, normalize, "
        "and mark padding invalid.",
        "A DINOv2-ready tensor plus a valid-content mask.",
        "Padding must not become fake normal PCB evidence.",
    ),
    PageSpec(
        6,
        "Describe the PCB as patches",
        (5,),
        "mixed",
        "features",
        "A 518 x 518 preprocessed support image.",
        "Frozen ViT-S/14 emits one local feature vector per 14 x 14 patch.",
        "A 37 x 37 grid containing up to 1,369 local feature vectors.",
        "One support image is many local normal examples, not one number.",
    ),
    PageSpec(
        7,
        "Build the normal memory",
        (6,),
        "illustration",
        "memory",
        "Valid DINOv2 patch vectors from k normal boards of one category.",
        "Concatenate every valid vector into that category's memory bank.",
        "A searchable collection of normal local appearances.",
        "The primary DINOv2 method keeps the complete valid bank.",
    ),
    PageSpec(
        8,
        "Calibrate suspicion using only normal images",
        (7,),
        "illustration",
        "calibration",
        "An independent normal validation set and the category memory bank.",
        "Score validation pixels and choose tau at the 0.995 normal-score quantile.",
        "A category-specific threshold tau.",
        "No anomaly mask or test ground truth chooses the threshold.",
    ),
    PageSpec(
        9,
        "Compare a new PCB with normal memory",
        (8,),
        "mixed",
        "query_compare",
        "A new query PCB and its category's normal memory bank.",
        "For each query patch, measure the distance to its nearest normal-memory patch.",
        "A coarse grid of anomaly distances.",
        "Large distance means no stored normal patch looks similar.",
    ),
    PageSpec(
        10,
        "Project a multi-scale anomaly heatmap",
        (9,),
        "mixed",
        "multiscale",
        "Scores from the full image and overlapping 768 x 768 crops.",
        "Map every view back to original coordinates and keep the maximum where views "
        "overlap.",
        "A continuous full-resolution anomaly heatmap.",
        "Multi-scale was motivated by small defects, but was not reliably better than "
        "single-scale.",
    ),
    PageSpec(
        11,
        "Turn scores into automatic prompts",
        (10, 11),
        "mixed",
        "proposal_prompts",
        "The heatmap and normal-only threshold tau.",
        "Threshold to proposal A, split four-connected components, remove tiny regions, "
        "and make one box plus one peak point per retained component.",
        "A binary anomaly proposal and automatic SAM2 prompts.",
        "The proposal may be fragmented or noisy; it is evidence, not the final outline.",
    ),
    PageSpec(
        12,
        "Let SAM2 follow coherent regions",
        (12, 13),
        "mixed",
        "sam2_union",
        "The original query image plus anomaly-generated point and box prompts.",
        "Frozen SAM2 predicts candidate masks; selection uses confidence, alignment, "
        "prompt containment, and an area cap before masks are combined.",
        "The Guided SAM2 mask S.",
        "SAM2 can outline the wrong normal structure because it knows grouping, not "
        "defectiveness.",
    ),
    PageSpec(
        13,
        "Keep only exact agreement",
        (14, 15),
        "mixed",
        "intersection_limit",
        "The calibrated anomaly proposal A and Guided SAM2 mask S.",
        "Compute AC-SAM2 = A intersect S pixel by pixel.",
        "The anomaly-consistent final mask.",
        "Intersection can remove false positives but cannot recover a pixel missing from "
        "A or S.",
    ),
    PageSpec(
        14,
        "Evaluate only after prediction",
        (16,),
        "real",
        "evaluation",
        "A completed prediction and the previously hidden test mask.",
        "Measure precision, recall, F1, IoU, and continuous-map metrics; aggregate across "
        "categories, shots, and seeds.",
        "Post-hoc metrics and the verified support-size F1 trend.",
        "DINOv2 inspects, SAM2 outlines, and AC-SAM2 keeps only their agreement.",
    ),
)


def collect_step_numbers(pages: Sequence[PageSpec]) -> list[int]:
    return [step for page in pages for step in page.steps]


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
        checked[f"pcb1_{role}"] = (
            path.is_file() and sha256_file(path) == figures[filename]["sha256"]
        )
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
    values: dict[str, dict[int, float]] = {label: {} for label in method_labels.values()}
    with (EVIDENCE_DIR / "primary_summary.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if (
                row["method"] in method_labels
                and row["category"] == "macro"
                and row["metric"] == "mean_anomaly_mask_f1"
                and int(row["k"]) in {1, 2, 4}
            ):
                label = method_labels[row["method"]]
                values[label][int(row["k"])] = round(float(row["mean"]), 3)
    if any(set(series) != {1, 2, 4} for series in values.values()):
        raise ValueError("frozen F1 table is incomplete")
    return values


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


def _text(
    pdf: canvas.Canvas,
    x: float,
    y: float,
    text: str,
    *,
    size: float = 12,
    color=COLORS["ink"],
    max_width: float = 250,
    leading: float | None = None,
    font: str = "Helvetica",
) -> float:
    leading = leading or size * 1.28
    pdf.setFillColor(color)
    pdf.setFont(font, size)
    for line in _wrap(text, font, size, max_width):
        pdf.drawString(x, y, line)
        y -= leading
    return y


def _rounded_box(
    pdf: canvas.Canvas,
    x: float,
    y: float,
    width: float,
    height: float,
    fill,
    *,
    stroke=COLORS["line"],
    radius: float = 10,
) -> None:
    pdf.setFillColor(fill)
    pdf.setStrokeColor(stroke)
    pdf.roundRect(x, y, width, height, radius, fill=1, stroke=1)


def _arrow(
    pdf: canvas.Canvas,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    color=COLORS["ink"],
) -> None:
    pdf.setStrokeColor(color)
    pdf.setFillColor(color)
    pdf.setLineWidth(2)
    pdf.line(x1, y1, x2, y2)
    pdf.line(x2, y2, x2 - 9, y2 + 5)
    pdf.line(x2, y2, x2 - 9, y2 - 5)


def _draw_image(
    pdf: canvas.Canvas,
    path: Path,
    x: float,
    y: float,
    width: float,
    height: float,
) -> None:
    with Image.open(path) as image:
        fitted = ImageOps.contain(
            image.convert("RGB"),
            (max(1, int(width * 2)), max(1, int(height * 2))),
            Image.Resampling.LANCZOS,
        )
    draw_width = fitted.width / 2
    draw_height = fitted.height / 2
    pdf.drawImage(
        ImageReader(fitted),
        x + (width - draw_width) / 2,
        y + (height - draw_height) / 2,
        draw_width,
        draw_height,
        preserveAspectRatio=True,
        mask="auto",
    )


def _header(pdf: canvas.Canvas, page: PageSpec) -> None:
    pdf.setFillColor(COLORS["paper"])
    pdf.rect(0, 0, PAGE_WIDTH, PAGE_HEIGHT, fill=1, stroke=0)
    pdf.setFillColor(COLORS["muted"])
    pdf.setFont("Helvetica-Bold", 10)
    label = "OVERVIEW" if not page.steps else "STEPS " + ", ".join(map(str, page.steps))
    pdf.drawString(42, PAGE_HEIGHT - 40, label)
    pdf.setFillColor(COLORS["ink"])
    pdf.setFont("Helvetica-Bold", 25)
    pdf.drawString(42, PAGE_HEIGHT - 72, page.title)
    source = {
        "real": "REAL EXPERIMENT EXAMPLE",
        "illustration": "EXPLANATORY ILLUSTRATION",
        "mixed": "REAL EXAMPLE + EXPLANATORY OVERLAY",
    }[page.source_kind]
    pdf.setFont("Helvetica-Bold", 8)
    pdf.setFillColor(COLORS["muted"])
    pdf.drawRightString(PAGE_WIDTH - 42, PAGE_HEIGHT - 40, source)


def _teaching_copy(pdf: canvas.Canvas, page: PageSpec) -> None:
    x, y, width = 555, 435, 245
    for label, body in (
        ("WHAT ENTERS", page.what_enters),
        ("WHAT HAPPENS", page.what_happens),
        ("WHAT COMES OUT", page.what_comes_out),
        ("WHAT TO NOTICE", page.what_to_notice),
    ):
        pdf.setFillColor(COLORS["muted"])
        pdf.setFont("Helvetica-Bold", 8)
        pdf.drawString(x, y, label)
        y = _text(
            pdf,
            x,
            y - 17,
            body,
            size=10.5,
            max_width=width,
            leading=13.5,
        ) - 13


def _draw_visual(
    pdf: canvas.Canvas,
    visual: str,
    assets: dict[str, Path],
    f1: dict[str, dict[int, float]],
) -> None:
    del assets, f1
    _rounded_box(pdf, 42, 82, 485, 390, white)
    pdf.setFillColor(COLORS["ink"])
    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawCentredString(284, 275, visual.replace("_", " ").title())


def _draw_page(
    pdf: canvas.Canvas,
    page: PageSpec,
    assets: dict[str, Path],
    f1: dict[str, dict[int, float]],
) -> None:
    _header(pdf, page)
    _draw_visual(pdf, page.visual, assets, f1)
    _teaching_copy(pdf, page)
    pdf.setFillColor(COLORS["muted"])
    pdf.setFont("Helvetica", 8)
    pdf.drawString(42, 24, "Few-Shot PCB Defect Segmentation - visual big picture")
    pdf.drawRightString(PAGE_WIDTH - 42, 24, f"{page.number} / 14")


def _write_markdown(path: Path) -> None:
    parts = ["# Visual Big Picture\n"]
    for page in PAGE_SPECS:
        steps = "Overview" if not page.steps else "Steps " + ", ".join(map(str, page.steps))
        parts.extend(
            [
                f"## Page {page.number}: {page.title}\n",
                f"**{steps} - {page.source_kind}.**\n",
                f"- **What enters:** {page.what_enters}\n",
                f"- **What happens:** {page.what_happens}\n",
                f"- **What comes out:** {page.what_comes_out}\n",
                f"- **What to notice:** {page.what_to_notice}\n",
            ]
        )
    path.write_text("\n".join(parts), encoding="utf-8")


def _render_previews(pdf_path: Path, pages_dir: Path) -> list[Path]:
    pages_dir.mkdir(parents=True, exist_ok=True)
    for old_preview in pages_dir.glob("page-*.png"):
        old_preview.unlink()
    prefix = pages_dir / "page"
    executable = os.environ.get("PDFTOPPM", "pdftoppm")
    subprocess.run(
        [executable, "-png", "-r", "120", str(pdf_path), str(prefix)],
        check=True,
        capture_output=True,
        text=True,
    )
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
            thumbs.append(
                ImageOps.contain(image.convert("RGB"), (420, 297), Image.Resampling.LANCZOS)
            )
    sheet = Image.new("RGB", (1260, 5 * 325), "#E8EDF3")
    for index, thumb in enumerate(thumbs):
        x = (index % 3) * 420
        y = (index // 3) * 325
        sheet.paste(thumb, (x, y))
    sheet.save(output_path)


def build_visual_big_picture(
    output_dir: Path,
    *,
    render_pngs: bool = True,
) -> dict[str, object]:
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
        "pages": [
            {
                "number": page.number,
                "title": page.title,
                "steps": list(page.steps),
                "source_kind": page.source_kind,
            }
            for page in PAGE_SPECS
        ],
        "source_sha256": {
            "pcb1_success.png": sha256_file(QUALITATIVE_DIR / "pcb1_success.png"),
            "pcb1_failure.png": sha256_file(QUALITATIVE_DIR / "pcb1_failure.png"),
            "primary_summary.csv": sha256_file(EVIDENCE_DIR / "primary_summary.csv"),
        },
        "pdf_sha256": sha256_file(pdf_path),
        "markdown_sha256": sha256_file(markdown_path),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "pdf": pdf_path,
        "markdown": markdown_path,
        "manifest": manifest_path,
        "contact_sheet": contact_sheet,
        "pages": page_paths,
    }
