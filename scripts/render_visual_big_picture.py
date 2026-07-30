#!/usr/bin/env python
"""Render the illustrated big-picture guide for the PCB segmentation pipeline."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from PIL import Image


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
