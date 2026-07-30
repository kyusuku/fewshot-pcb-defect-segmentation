#!/usr/bin/env python
"""Render the illustrated big-picture guide for the PCB segmentation pipeline."""

from __future__ import annotations

import csv
import hashlib
import json
import math
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


def _visual_canvas(pdf: canvas.Canvas) -> None:
    _rounded_box(pdf, 42, 82, 485, 390, white)


def _small_label(
    pdf: canvas.Canvas,
    x: float,
    y: float,
    text: str,
    *,
    color=COLORS["muted"],
    size: float = 8,
    centered: bool = False,
) -> None:
    pdf.setFillColor(color)
    pdf.setFont("Helvetica-Bold", size)
    if centered:
        pdf.drawCentredString(x, y, text)
    else:
        pdf.drawString(x, y, text)


def _image_tile(
    pdf: canvas.Canvas,
    path: Path,
    x: float,
    y: float,
    width: float,
    height: float,
    label: str,
    *,
    border=COLORS["line"],
) -> None:
    _rounded_box(pdf, x, y, width, height, white, stroke=border, radius=6)
    _draw_image(pdf, path, x + 4, y + 4, width - 8, height - 8)
    _small_label(pdf, x + width / 2, y - 13, label, centered=True)


def _draw_pil_image(
    pdf: canvas.Canvas,
    image: Image.Image,
    x: float,
    y: float,
    width: float,
    height: float,
) -> None:
    fitted = ImageOps.contain(
        image.convert("RGB"),
        (max(1, int(width * 2)), max(1, int(height * 2))),
        Image.Resampling.NEAREST,
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


def _draw_grid(
    pdf: canvas.Canvas,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    columns: int,
    rows: int,
    color=COLORS["blue_strong"],
    line_width: float = 0.45,
) -> None:
    pdf.setStrokeColor(color)
    pdf.setLineWidth(line_width)
    for index in range(columns + 1):
        x_pos = x + width * index / columns
        pdf.line(x_pos, y, x_pos, y + height)
    for index in range(rows + 1):
        y_pos = y + height * index / rows
        pdf.line(x, y_pos, x + width, y_pos)


def _draw_vector(
    pdf: canvas.Canvas,
    x: float,
    y: float,
    values: Sequence[float],
    *,
    color=COLORS["blue_strong"],
    width: float = 75,
    height: float = 28,
) -> None:
    slot = width / len(values)
    pdf.setFillColor(color)
    for index, value in enumerate(values):
        pdf.rect(x + index * slot, y, max(1.5, slot - 2), height * value, fill=1, stroke=0)


def _draw_cylinder(
    pdf: canvas.Canvas,
    x: float,
    y: float,
    width: float,
    height: float,
    label: str,
) -> None:
    pdf.setFillColor(COLORS["blue"])
    pdf.setStrokeColor(COLORS["blue_strong"])
    pdf.rect(x, y + 7, width, height - 14, fill=1, stroke=0)
    pdf.ellipse(x, y + height - 14, x + width, y + height, fill=1, stroke=1)
    pdf.ellipse(x, y, x + width, y + 14, fill=1, stroke=1)
    pdf.line(x, y + 7, x, y + height - 7)
    pdf.line(x + width, y + 7, x + width, y + height - 7)
    _small_label(pdf, x + width / 2, y + height / 2 - 3, label, centered=True, size=8.5)


def _draw_prompt_overlay(
    pdf: canvas.Canvas,
    image_path: Path,
    x: float,
    y: float,
    width: float,
    height: float,
) -> None:
    _draw_image(pdf, image_path, x, y, width, height)
    pdf.setStrokeColor(COLORS["amber_strong"])
    pdf.setFillColor(COLORS["amber_strong"])
    pdf.setLineWidth(2)
    pdf.rect(
        x + width * 0.43,
        y + height * 0.42,
        width * 0.27,
        height * 0.18,
        fill=0,
        stroke=1,
    )
    pdf.circle(x + width * 0.57, y + height * 0.51, 4, fill=1, stroke=0)


def _illustrative_proposal(anomaly_path: Path) -> Image.Image:
    with Image.open(anomaly_path) as image:
        gray = image.convert("L")
    values = sorted(gray.get_flattened_data())
    threshold = values[int(0.92 * (len(values) - 1))]
    return gray.point(lambda value: 255 if value >= threshold else 0).convert("RGB")


def _visual_overview(
    pdf: canvas.Canvas,
    assets: dict[str, Path],
    f1: dict[str, dict[int, float]],
) -> None:
    del f1
    _visual_canvas(pdf)
    _small_label(pdf, 58, 440, "LEARN NORMAL APPEARANCE", color=COLORS["blue_strong"])
    for index in range(3):
        _image_tile(
            pdf,
            assets["pcb1_normal"],
            58 + index * 69,
            351,
            58,
            66,
            f"normal {index + 1}",
        )
    _arrow(pdf, 257, 384, 287, 384)
    _rounded_box(pdf, 291, 351, 92, 66, COLORS["blue"], stroke=COLORS["blue_strong"])
    _small_label(pdf, 337, 389, "DINOv2", centered=True, color=COLORS["ink"], size=9)
    _small_label(pdf, 337, 374, "normal memory", centered=True, size=8)
    _arrow(pdf, 387, 384, 411, 384)
    _image_tile(pdf, assets["success_anomaly"], 416, 351, 91, 66, "anomaly map")

    _small_label(pdf, 58, 303, "INSPECT A NEW PCB", color=COLORS["amber_strong"])
    _image_tile(pdf, assets["success_query"], 58, 154, 145, 125, "new query")
    _arrow(pdf, 209, 216, 246, 216)
    _rounded_box(pdf, 250, 177, 108, 80, COLORS["teal"], stroke=COLORS["teal_strong"])
    _small_label(pdf, 304, 225, "Guided SAM2", centered=True, color=COLORS["ink"], size=9)
    _small_label(pdf, 304, 207, "point + box", centered=True, size=8)
    _arrow(pdf, 362, 216, 397, 216)
    _image_tile(pdf, assets["success_ac"], 402, 154, 105, 125, "final mask", border=COLORS["navy"])

    pdf.setDash(4, 3)
    pdf.setStrokeColor(COLORS["line"])
    pdf.line(58, 121, 507, 121)
    pdf.setDash()
    _small_label(pdf, 282, 98, "GROUND TRUTH LOCKED UNTIL EVALUATION", centered=True)


def _visual_roles(
    pdf: canvas.Canvas,
    assets: dict[str, Path],
    f1: dict[str, dict[int, float]],
) -> None:
    del f1
    _visual_canvas(pdf)
    _rounded_box(pdf, 56, 213, 210, 230, COLORS["blue"])
    _small_label(pdf, 72, 421, "DINOv2 - DOES THIS LOOK NORMAL?", color=COLORS["blue_strong"])
    _draw_image(pdf, assets["pcb1_normal"], 73, 270, 176, 125)
    _draw_grid(pdf, 87, 282, 148, 101, columns=7, rows=7)
    _small_label(pdf, 161, 245, "patches -> features -> distances", centered=True)

    _rounded_box(pdf, 282, 213, 229, 230, COLORS["teal"])
    _small_label(pdf, 298, 421, "SAM2 - WHICH PIXELS BELONG TOGETHER?", color=COLORS["teal_strong"])
    _draw_prompt_overlay(pdf, assets["success_query"], 297, 278, 92, 105)
    _arrow(pdf, 393, 331, 412, 331, color=COLORS["teal_strong"])
    _draw_image(pdf, assets["success_guided"], 418, 278, 78, 105)
    _small_label(pdf, 396, 245, "prompt -> coherent region", centered=True)

    _rounded_box(pdf, 70, 105, 440, 73, COLORS["paper"])
    for x, label, fill in (
        (102, "A\nanomaly", COLORS["amber"]),
        (230, "S\nregion", COLORS["teal"]),
        (385, "A AND S\nfinal", COLORS["navy"]),
    ):
        _rounded_box(pdf, x, 119, 82, 43, fill, stroke=fill, radius=7)
        text_color = white if fill == COLORS["navy"] else COLORS["ink"]
        lines = label.split("\n")
        for index, line in enumerate(lines):
            _small_label(
                pdf,
                x + 41,
                143 - index * 14,
                line,
                centered=True,
                color=text_color,
                size=8.5,
            )
    _arrow(pdf, 188, 141, 220, 141)
    _arrow(pdf, 316, 141, 375, 141)


def _visual_categories(
    pdf: canvas.Canvas,
    assets: dict[str, Path],
    f1: dict[str, dict[int, float]],
) -> None:
    del f1
    _visual_canvas(pdf)
    for index in range(4):
        x = 57 + index * 116
        _image_tile(
            pdf,
            assets[f"pcb{index + 1}_normal"],
            x,
            292,
            102,
            115,
            f"pcb{index + 1}",
        )
        _draw_cylinder(pdf, x + 13, 177, 76, 58, f"M_pcb{index + 1}")
        _arrow(pdf, x + 51, 278, x + 51, 241, color=COLORS["blue_strong"])
    _rounded_box(pdf, 72, 104, 425, 42, COLORS["paper"])
    _small_label(
        pdf,
        284,
        129,
        "4,416 images | 4,016 normal | 400 anomalous | fold 0",
        centered=True,
        color=COLORS["ink"],
        size=9,
    )
    _small_label(
        pdf,
        284,
        112,
        "four board layouts -> four separate definitions of normal",
        centered=True,
    )


def _visual_roles_sampling(
    pdf: canvas.Canvas,
    assets: dict[str, Path],
    f1: dict[str, dict[int, float]],
) -> None:
    del f1
    _visual_canvas(pdf)
    columns = (
        (55, "SUPPORT", COLORS["blue"], COLORS["blue_strong"]),
        (216, "CALIBRATION", COLORS["paper"], COLORS["amber_strong"]),
        (377, "TEST", COLORS["paper"], COLORS["teal_strong"]),
    )
    for x, label, fill, accent in columns:
        _rounded_box(pdf, x, 118, 142, 306, fill, stroke=accent)
        _small_label(pdf, x + 71, 399, label, centered=True, color=accent, size=9)

    _draw_image(pdf, assets["pcb1_normal"], 67, 285, 118, 89)
    for row, k in enumerate((1, 2, 4)):
        y = 250 - row * 43
        _small_label(pdf, 70, y + 9, f"k = {k}", color=COLORS["ink"], size=8.5)
        for index in range(k):
            pdf.setFillColor(COLORS["blue_strong"])
            pdf.circle(124 + index * 14, y + 11, 5, fill=1, stroke=0)
    _small_label(pdf, 126, 137, "seeds 4880-4884", centered=True)

    _small_label(pdf, 287, 357, "normal only", centered=True, color=COLORS["ink"])
    for index in range(28):
        height = 12 + 46 * math.exp(-((index - 12) / 9) ** 2)
        pdf.setFillColor(COLORS["blue_strong"])
        pdf.rect(231 + index * 4.1, 246, 3, height, fill=1, stroke=0)
    pdf.setStrokeColor(COLORS["amber_strong"])
    pdf.setLineWidth(2)
    pdf.line(337, 238, 337, 320)
    _small_label(pdf, 337, 221, "tau = Q_0.995", centered=True, color=COLORS["amber_strong"])
    _small_label(pdf, 287, 164, "181 held-out normal images", centered=True)

    _draw_image(pdf, assets["success_query"], 392, 248, 112, 112)
    _rounded_box(pdf, 398, 166, 100, 49, COLORS["paper"], stroke=COLORS["muted"])
    _small_label(pdf, 448, 195, "MASK HIDDEN", centered=True, color=COLORS["ink"], size=9)
    _small_label(pdf, 448, 179, "until evaluation", centered=True)


def _visual_preprocess(
    pdf: canvas.Canvas,
    assets: dict[str, Path],
    f1: dict[str, dict[int, float]],
) -> None:
    del f1
    _visual_canvas(pdf)
    _image_tile(pdf, assets["pcb1_normal"], 57, 236, 126, 151, "1. RGB image")
    _arrow(pdf, 188, 312, 211, 312)
    _rounded_box(pdf, 216, 236, 126, 151, COLORS["paper"])
    _draw_image(pdf, assets["pcb1_normal"], 223, 248, 112, 127)
    _small_label(pdf, 279, 218, "2. preserve aspect ratio", centered=True)
    _arrow(pdf, 347, 312, 370, 312)
    _rounded_box(pdf, 375, 236, 126, 151, COLORS["paper"], stroke=COLORS["blue_strong"])
    pdf.setFillColor(COLORS["blue"])
    pdf.rect(384, 257, 108, 110, fill=1, stroke=0)
    _draw_image(pdf, assets["pcb1_normal"], 390, 270, 96, 84)
    pdf.setStrokeColor(COLORS["blue_strong"])
    pdf.setLineWidth(2)
    pdf.rect(390, 270, 96, 84, fill=0, stroke=1)
    _small_label(pdf, 438, 218, "3. center on 518 x 518", centered=True)

    chips = (
        (76, "RGB"),
        (180, "no stretching"),
        (318, "valid-content mask"),
    )
    for x, label in chips:
        width = 82 if label == "RGB" else 112
        _rounded_box(pdf, x, 125, width, 43, COLORS["blue"])
        _small_label(pdf, x + width / 2, 142, label, centered=True, color=COLORS["ink"])


def _visual_features(
    pdf: canvas.Canvas,
    assets: dict[str, Path],
    f1: dict[str, dict[int, float]],
) -> None:
    del f1
    _visual_canvas(pdf)
    _rounded_box(pdf, 57, 184, 226, 232, COLORS["paper"])
    _draw_image(pdf, assets["pcb1_normal"], 68, 208, 204, 180)
    _draw_grid(pdf, 76, 224, 188, 148, columns=37, rows=37, line_width=0.18)
    _small_label(pdf, 170, 195, "37 x 37 feature grid", centered=True)

    _rounded_box(pdf, 310, 251, 178, 165, COLORS["blue"])
    _small_label(pdf, 399, 397, "zoom: 5 x 5 patches", centered=True, color=COLORS["ink"])
    _draw_grid(pdf, 331, 273, 136, 102, columns=5, rows=5, line_width=1)
    for row, values in enumerate(
        ((0.3, 0.7, 0.45, 0.85, 0.55), (0.75, 0.4, 0.6, 0.25, 0.9), (0.5, 0.8, 0.2, 0.65, 0.4))
    ):
        _draw_vector(pdf, 325, 183 - row * 28, values, width=150, height=20)
    _small_label(pdf, 399, 95, "518 / 14 = 37 | up to 1,369 local vectors", centered=True, size=9)


def _visual_memory(
    pdf: canvas.Canvas,
    assets: dict[str, Path],
    f1: dict[str, dict[int, float]],
) -> None:
    del assets, f1
    _visual_canvas(pdf)
    for index in range(3):
        x = 61 + index * 102
        _rounded_box(pdf, x, 291, 82, 111, COLORS["blue"])
        _draw_grid(pdf, x + 10, 317, 62, 62, columns=6, rows=4, line_width=0.6)
        _small_label(pdf, x + 41, 301, f"support {index + 1}", centered=True)
        _arrow(pdf, x + 82, 344, 347, 286 - index * 22, color=COLORS["blue_strong"])
    _draw_cylinder(pdf, 355, 200, 135, 165, "M_pcb1")
    for row in range(6):
        _draw_vector(
            pdf,
            380,
            234 + row * 17,
            (0.2 + 0.08 * row, 0.7, 0.45, 0.85 - 0.05 * row, 0.55),
            width=84,
            height=11,
        )
    _rounded_box(pdf, 68, 113, 215, 73, COLORS["paper"], stroke=COLORS["amber_strong"])
    _small_label(pdf, 83, 168, "NEXT STEP PREVIEW", color=COLORS["amber_strong"])
    _draw_vector(pdf, 92, 130, (0.8, 0.25, 0.7, 0.35, 0.9), color=COLORS["amber_strong"])
    _small_label(pdf, 177, 142, "query vector stays outside memory", color=COLORS["ink"])


def _visual_calibration(
    pdf: canvas.Canvas,
    assets: dict[str, Path],
    f1: dict[str, dict[int, float]],
) -> None:
    del assets, f1
    _visual_canvas(pdf)
    chart_x, chart_y, chart_width, chart_height = 62, 160, 430, 230
    pdf.setStrokeColor(COLORS["line"])
    pdf.line(chart_x, chart_y, chart_x + chart_width, chart_y)
    pdf.line(chart_x, chart_y, chart_x, chart_y + chart_height)
    bar_width = chart_width / 200
    heights = [
        0.18 + 0.55 * math.exp(-((index - 75) / 48) ** 2) + 0.12 * math.sin(index / 9) ** 2
        for index in range(200)
    ]
    for index, value in enumerate(heights):
        pdf.setFillColor(COLORS["blue_strong"])
        pdf.rect(
            chart_x + index * bar_width,
            chart_y,
            max(0.8, bar_width - 0.3),
            chart_height * value,
            fill=1,
            stroke=0,
        )
    tau_x = chart_x + 198 * bar_width
    pdf.setStrokeColor(COLORS["amber_strong"])
    pdf.setLineWidth(2.5)
    pdf.line(tau_x, chart_y - 8, tau_x, chart_y + chart_height + 8)
    pdf.setFillColor(COLORS["amber_strong"])
    pdf.setFont("Helvetica-Bold", 8)
    pdf.drawRightString(tau_x - 5, 411, "tau = 99.5th percentile")
    _small_label(pdf, 277, 129, "held-out normal pixel scores, sorted from low to high", centered=True)
    _small_label(pdf, 62, 105, "below tau: looks normal", color=COLORS["blue_strong"])
    _small_label(pdf, 492, 105, "suspicious", centered=True, color=COLORS["amber_strong"])


def _visual_query_compare(
    pdf: canvas.Canvas,
    assets: dict[str, Path],
    f1: dict[str, dict[int, float]],
) -> None:
    del f1
    _visual_canvas(pdf)
    _image_tile(pdf, assets["success_query"], 57, 208, 186, 185, "new pcb1 query")
    pdf.setStrokeColor(COLORS["amber_strong"])
    pdf.setLineWidth(2)
    for x, y in ((105, 305), (145, 270), (177, 330)):
        pdf.rect(x, y, 18, 18, fill=0, stroke=1)

    _small_label(pdf, 316, 412, "query patch", centered=True, color=COLORS["ink"])
    _small_label(pdf, 446, 412, "nearest normal patch", centered=True, color=COLORS["ink"])
    rows = (
        (340, COLORS["blue_strong"], 35, "small distance"),
        (280, COLORS["blue_strong"], 42, "small distance"),
        (220, COLORS["amber_strong"], 82, "large distance"),
    )
    for index, (y, color, distance, label) in enumerate(rows):
        values = (
            (0.3 + 0.1 * index, 0.75, 0.4, 0.65, 0.5)
            if index < 2
            else (0.85, 0.2, 0.72, 0.3, 0.92)
        )
        _draw_vector(pdf, 278, y, values, color=color, width=64, height=24)
        _arrow(pdf, 348, y + 11, 348 + distance, y + 11, color=color)
        _draw_vector(pdf, 442, y, (0.4, 0.7, 0.45, 0.62, 0.52), width=58, height=24)
        _small_label(pdf, 382, y - 14, label, centered=True, color=color)
    _rounded_box(pdf, 80, 108, 410, 48, COLORS["paper"])
    _small_label(
        pdf,
        285,
        132,
        "a_p(x) = min || f(x)_p - m ||_2  :  far from every normal patch = suspicious",
        centered=True,
        color=COLORS["ink"],
        size=9,
    )


def _visual_multiscale(
    pdf: canvas.Canvas,
    assets: dict[str, Path],
    f1: dict[str, dict[int, float]],
) -> None:
    del f1
    _visual_canvas(pdf)
    _image_tile(pdf, assets["success_query"], 57, 184, 220, 208, "full image + local crops")
    pdf.setStrokeColor(COLORS["amber_strong"])
    pdf.setLineWidth(1.6)
    for x, y in ((78, 280), (124, 280), (78, 225), (124, 225)):
        pdf.rect(x, y, 94, 72, fill=0, stroke=1)
    _small_label(pdf, 167, 405, "global view", centered=True, color=COLORS["blue_strong"])
    _small_label(pdf, 167, 151, "768 x 768 crops | 25% overlap", centered=True)
    _arrow(pdf, 283, 287, 340, 287, color=COLORS["amber_strong"])
    _small_label(pdf, 312, 307, "map back", centered=True, color=COLORS["amber_strong"])
    _small_label(pdf, 312, 273, "+ max", centered=True, color=COLORS["amber_strong"])
    _image_tile(pdf, assets["success_anomaly"], 349, 202, 145, 171, "continuous heatmap")
    _rounded_box(pdf, 90, 104, 390, 38, COLORS["paper"])
    _small_label(
        pdf,
        285,
        120,
        "Small-defect motivation - paired results did not prove a reliable gain.",
        centered=True,
        color=COLORS["ink"],
        size=8.5,
    )


def _visual_proposal_prompts(
    pdf: canvas.Canvas,
    assets: dict[str, Path],
    f1: dict[str, dict[int, float]],
) -> None:
    del f1
    _visual_canvas(pdf)
    proposal = _illustrative_proposal(assets["success_anomaly"])
    _image_tile(pdf, assets["success_anomaly"], 56, 252, 126, 132, "continuous scores")
    _arrow(pdf, 186, 318, 213, 318, color=COLORS["amber_strong"])
    _rounded_box(pdf, 218, 252, 126, 132, COLORS["paper"])
    _draw_pil_image(pdf, proposal, 224, 258, 114, 120)
    _small_label(pdf, 281, 237, "binary proposal A", centered=True)
    _arrow(pdf, 348, 318, 375, 318, color=COLORS["amber_strong"])
    _rounded_box(pdf, 380, 252, 126, 132, COLORS["paper"])
    _draw_prompt_overlay(pdf, assets["success_query"], 386, 258, 114, 120)
    _small_label(pdf, 443, 237, "box + peak point", centered=True)

    _rounded_box(pdf, 64, 126, 438, 70, COLORS["amber"])
    _small_label(pdf, 82, 178, "COMPONENT FILTER", color=COLORS["amber_strong"])
    _small_label(pdf, 82, 157, "4-connected | discard < 8 px | keep at most 8 regions", color=COLORS["ink"], size=9)
    _small_label(
        pdf,
        82,
        138,
        "Proposal image is explanatory; the real threshold comes from normal calibration.",
        color=COLORS["ink"],
        size=8,
    )


def _visual_sam2_union(
    pdf: canvas.Canvas,
    assets: dict[str, Path],
    f1: dict[str, dict[int, float]],
) -> None:
    del f1
    _visual_canvas(pdf)
    _rounded_box(pdf, 56, 240, 126, 151, COLORS["paper"])
    _draw_prompt_overlay(pdf, assets["success_query"], 62, 251, 114, 128)
    _small_label(pdf, 119, 225, "image + prompts", centered=True)
    _arrow(pdf, 187, 315, 213, 315, color=COLORS["teal_strong"])

    for index, (x, y) in enumerate(((226, 325), (268, 295), (226, 265))):
        _rounded_box(pdf, x, y, 72, 51, COLORS["teal"], stroke=COLORS["teal_strong"])
        pdf.setFillColor(COLORS["teal_strong"])
        pdf.ellipse(x + 14, y + 10, x + 58, y + 40, fill=1, stroke=0)
        _small_label(pdf, x + 36, y - 12, f"candidate {index + 1}", centered=True, size=7.5)
    _arrow(pdf, 346, 315, 375, 315, color=COLORS["teal_strong"])
    _image_tile(pdf, assets["success_guided"], 380, 240, 126, 151, "Guided SAM2 S")

    _rounded_box(pdf, 65, 115, 435, 74, COLORS["paper"])
    criteria = ("confidence", "anomaly alignment", "prompt containment", "area <= 25%")
    for index, label in enumerate(criteria):
        x = 80 + (index % 2) * 205
        y = 164 - (index // 2) * 29
        pdf.setFillColor(COLORS["teal_strong"])
        pdf.circle(x, y + 2, 4, fill=1, stroke=0)
        _small_label(pdf, x + 12, y - 2, label, color=COLORS["ink"], size=8.5)


def _visual_intersection_limit(
    pdf: canvas.Canvas,
    assets: dict[str, Path],
    f1: dict[str, dict[int, float]],
) -> None:
    del f1
    _visual_canvas(pdf)
    proposal = _illustrative_proposal(assets["success_anomaly"])
    _rounded_box(pdf, 56, 302, 108, 112, COLORS["paper"])
    _draw_pil_image(pdf, proposal, 62, 308, 96, 100)
    _small_label(pdf, 110, 286, "proposal A", centered=True)
    _small_label(pdf, 183, 352, "AND", centered=True, color=COLORS["amber_strong"], size=10)
    _image_tile(pdf, assets["success_guided"], 207, 302, 108, 112, "Guided S")
    _small_label(pdf, 334, 352, "=", centered=True, color=COLORS["navy"], size=15)
    _image_tile(pdf, assets["success_ac"], 358, 302, 108, 112, "AC-SAM2", border=COLORS["navy"])
    _rounded_box(pdf, 476, 327, 35, 62, COLORS["navy"], stroke=COLORS["navy"])
    _small_label(pdf, 493.5, 365, "A", centered=True, color=white, size=9)
    _small_label(pdf, 493.5, 350, "cap", centered=True, color=white, size=7)
    _small_label(pdf, 493.5, 336, "S", centered=True, color=white, size=9)

    _rounded_box(pdf, 56, 106, 455, 138, COLORS["paper"])
    _draw_image(pdf, QUALITATIVE_DIR / "pcb1_failure.png", 63, 126, 441, 108)
    _small_label(
        pdf,
        284,
        113,
        "Failure case: intersection cannot restore pixels missing from the proposal.",
        centered=True,
        color=COLORS["amber_strong"],
        size=8.5,
    )


def _draw_f1_chart(
    pdf: canvas.Canvas,
    f1: dict[str, dict[int, float]],
    x: float,
    y: float,
    width: float,
    height: float,
) -> None:
    pdf.setStrokeColor(COLORS["line"])
    pdf.line(x, y, x + width, y)
    pdf.line(x, y, x, y + height)
    for value in (0.2, 0.25, 0.3):
        y_pos = y + (value - 0.18) / 0.16 * height
        pdf.setStrokeColor(COLORS["line"])
        pdf.line(x, y_pos, x + width, y_pos)
        _small_label(pdf, x - 6, y_pos - 3, f"{value:.2f}", centered=True, size=7)
    series_colors = {
        "Multi-DINO": COLORS["muted"],
        "Guided SAM2": COLORS["teal_strong"],
        "AC-SAM2": COLORS["navy"],
    }
    shots = (1, 2, 4)
    for method, series in f1.items():
        points = []
        for index, shot in enumerate(shots):
            x_pos = x + 25 + index * (width - 50) / 2
            y_pos = y + (series[shot] - 0.18) / 0.16 * height
            points.append((x_pos, y_pos))
        pdf.setStrokeColor(series_colors[method])
        pdf.setLineWidth(2)
        pdf.line(*points[0], *points[1])
        pdf.line(*points[1], *points[2])
        for index, (x_pos, y_pos) in enumerate(points):
            pdf.setFillColor(series_colors[method])
            pdf.circle(x_pos, y_pos, 3.5, fill=1, stroke=0)
            if method == "AC-SAM2":
                _small_label(
                    pdf,
                    x_pos,
                    y_pos + 10,
                    f"{series[shots[index]]:.3f}",
                    centered=True,
                    color=COLORS["navy"],
                    size=7,
                )
    for index, shot in enumerate(shots):
        x_pos = x + 25 + index * (width - 50) / 2
        _small_label(pdf, x_pos, y - 15, f"k={shot}", centered=True, size=7.5)
    for index, method in enumerate(("Multi-DINO", "Guided SAM2", "AC-SAM2")):
        legend_x = x + index * 82
        pdf.setStrokeColor(series_colors[method])
        pdf.setLineWidth(3)
        pdf.line(legend_x, y + height + 19, legend_x + 16, y + height + 19)
        _small_label(pdf, legend_x + 20, y + height + 16, method, size=6.8)


def _visual_evaluation(
    pdf: canvas.Canvas,
    assets: dict[str, Path],
    f1: dict[str, dict[int, float]],
) -> None:
    _visual_canvas(pdf)
    _image_tile(pdf, assets["success_ac"], 55, 303, 105, 112, "prediction")
    _small_label(pdf, 177, 354, "vs", centered=True, color=COLORS["muted"], size=10)
    _image_tile(pdf, assets["success_ground_truth"], 194, 303, 105, 112, "ground truth")
    _small_label(pdf, 177, 326, "lock open", centered=True, color=COLORS["teal_strong"])
    _draw_f1_chart(pdf, f1, 331, 284, 165, 112)

    _rounded_box(pdf, 55, 116, 444, 127, COLORS["paper"])
    _small_label(pdf, 74, 221, "REPEATED-MEASURES EXPERIMENT", color=COLORS["ink"], size=9)
    nodes = (
        (72, "4\ncategories", COLORS["blue"]),
        (177, "3\nshots", COLORS["blue"]),
        (282, "5\nseeds", COLORS["blue"]),
        (387, "6\nmethods", COLORS["teal"]),
    )
    for index, (x, label, fill) in enumerate(nodes):
        _rounded_box(pdf, x, 157, 76, 49, fill)
        for row, line in enumerate(label.split("\n")):
            _small_label(pdf, x + 38, 185 - row * 15, line, centered=True, color=COLORS["ink"])
        if index < len(nodes) - 1:
            _small_label(pdf, x + 91, 177, "x", centered=True, color=COLORS["muted"], size=10)
    _small_label(pdf, 277, 139, "360 support-dependent runs + 4 SAM2-only = 364", centered=True, color=COLORS["navy"], size=9)
    _small_label(pdf, 277, 101, "DINOv2 inspects | SAM2 outlines | AC-SAM2 keeps agreement", centered=True, color=COLORS["ink"], size=9)


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


def _draw_visual(
    pdf: canvas.Canvas,
    visual: str,
    assets: dict[str, Path],
    f1: dict[str, dict[int, float]],
) -> None:
    VISUAL_RENDERERS[visual](pdf, assets, f1)


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
