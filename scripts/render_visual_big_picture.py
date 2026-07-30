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
