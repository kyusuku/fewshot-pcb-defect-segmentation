"""Build the comprehensive, evidence-bound teaching notebook.

This script formats existing frozen evidence for presentation. It does not run
experiments, recompute metrics, or modify any evidence artifact.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import textwrap


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "notebooks/pcb_defect_pipeline.ipynb"
EVIDENCE_DIR = REPO_ROOT / "docs/evidence/generated"


def _source(text: str) -> list[str]:
    normalized = textwrap.dedent(text).strip("\n") + "\n"
    return normalized.splitlines(keepends=True)


def _cell_id(index: int, cell_type: str, source: list[str]) -> str:
    payload = f"{index}:{cell_type}:{''.join(source)}".encode()
    return hashlib.sha256(payload).hexdigest()[:8]


def markdown_cell(index: int, text: str) -> dict[str, object]:
    source = _source(text)
    return {
        "cell_type": "markdown",
        "id": _cell_id(index, "markdown", source),
        "metadata": {},
        "source": source,
    }


def code_cell(index: int, text: str) -> dict[str, object]:
    source = _source(text)
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": _cell_id(index, "code", source),
        "metadata": {},
        "outputs": [],
        "source": source,
    }


def read_json(name: str) -> dict[str, object]:
    path = EVIDENCE_DIR / name
    if not path.is_file():
        raise FileNotFoundError(f"missing frozen evidence: {path.relative_to(REPO_ROOT)}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(name: str) -> list[dict[str, str]]:
    path = EVIDENCE_DIR / name
    if not path.is_file():
        raise FileNotFoundError(f"missing frozen evidence: {path.relative_to(REPO_ROOT)}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _fmt(value: object, digits: int = 4, signed: bool = False) -> str:
    number = float(value)
    return f"{number:+.{digits}f}" if signed else f"{number:.{digits}f}"


def _interval(low: object, high: object) -> str:
    if low in (None, "") or high in (None, ""):
        return "descriptive"
    return f"[{_fmt(low)}, {_fmt(high)}]"


def _format_markdown(template: str, **values: str) -> str:
    """Dedent a Markdown template before inserting column-zero table blocks."""

    return textwrap.dedent(template).strip("\n").format(**values)


def _primary_f1_table() -> str:
    rows = [
        row
        for row in read_csv("primary_summary.csv")
        if row["category"] == "macro" and row["metric"] == "mean_anomaly_mask_f1"
    ]
    method_order = {
        "patchcore": 0,
        "dinov2_single": 1,
        "dinov2_multi": 2,
        "sam2_only": 3,
        "dinov2_single_sam2": 4,
        "dinov2_multi_sam2": 5,
        "anomaly_consistent_sam2": 6,
    }
    rows.sort(key=lambda row: (method_order[row["method"]], int(row["k"])))
    lines = [
        "| Method | k | Mean anomaly-image F1 | Support-seed interval |",
        "| --- | ---: | ---: | --- |",
    ]
    labels = {
        "patchcore": "PatchCore-style",
        "dinov2_single": "Single-scale DINOv2",
        "dinov2_multi": "Multi-scale DINOv2",
        "sam2_only": "SAM2-only grid prompts",
        "dinov2_single_sam2": "Single-scale-guided SAM2",
        "dinov2_multi_sam2": "Multi-scale-guided SAM2",
        "anomaly_consistent_sam2": "Anomaly-consistent SAM2",
    }
    for row in rows:
        k = "0 (fixed)" if row["method"] == "sam2_only" else row["k"]
        lines.append(
            f"| {labels[row['method']]} | {k} | {_fmt(row['mean'])} | "
            f"{_interval(row['ci_low'], row['ci_high'])} |"
        )
    return "\n".join(lines)


def _heatmap_k4_table() -> str:
    rows = [
        row
        for row in read_csv("primary_summary.csv")
        if row["category"] == "macro"
        and row["k"] == "4"
        and row["method"] in {"patchcore", "dinov2_single", "dinov2_multi"}
    ]
    lookup = {(row["method"], row["metric"]): row for row in rows}
    metrics = (
        ("image_auroc", "Image AUROC"),
        ("pixel_auroc", "Pixel AUROC"),
        ("aupro", "AUPRO"),
        ("aggregate_pixel_f1", "Calibrated aggregate F1"),
        ("aggregate_pixel_iou", "Calibrated aggregate IoU"),
    )
    methods = (
        ("patchcore", "PatchCore-style"),
        ("dinov2_single", "Single-scale DINOv2"),
        ("dinov2_multi", "Multi-scale DINOv2"),
    )
    lines = [
        "| Method | " + " | ".join(label for _, label in metrics) + " |",
        "| --- | " + " | ".join("---:" for _ in metrics) + " |",
    ]
    for method, label in methods:
        values = [_fmt(lookup[(method, metric)]["mean"]) for metric, _ in metrics]
        lines.append(f"| {label} | " + " | ".join(values) + " |")
    return "\n".join(lines)


def _category_shot_delta_table() -> str:
    comparison = read_json("paired_statistics.json")["comparisons"][
        "dinov2_multi_vs_anomaly_consistent_sam2"
    ]
    lines = [
        "| Group | Item | F1 delta | 95% CI | Reading |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for category in ("pcb1", "pcb2", "pcb3", "pcb4"):
        result = comparison["by_category"][category]["f1"]
        reading = "robust positive" if float(result["ci_low"]) > 0 else "inconclusive"
        lines.append(
            f"| category | {category} | {_fmt(result['mean_delta'], signed=True)} | "
            f"{_interval(result['ci_low'], result['ci_high'])} | {reading} |"
        )
    for k in ("1", "2", "4"):
        result = comparison["by_k"][k]["f1"]
        reading = "robust positive" if float(result["ci_low"]) > 0 else "inconclusive"
        lines.append(
            f"| shot count k | {k} | {_fmt(result['mean_delta'], signed=True)} | "
            f"{_interval(result['ci_low'], result['ci_high'])} | {reading} |"
        )
    return "\n".join(lines)


def _comparison_table() -> str:
    conclusion = read_json("evidence_conclusion.json")
    supporting = read_json("baseline_paired_statistics.json")["comparisons"]
    paired = read_json("paired_statistics.json")["comparisons"]
    comparisons = [
        (
            "Anomaly-consistent SAM2 - calibrated multi-scale DINOv2",
            conclusion,
            "robust improvement",
        ),
        (
            "Multi-scale DINOv2 - single-scale DINOv2",
            supporting["dinov2_multi_minus_dinov2_single"]["overall"],
            "inconclusive",
        ),
        (
            "Multi-scale DINOv2 - PatchCore-style",
            supporting["dinov2_multi_minus_patchcore_style"]["overall"],
            "robust improvement",
        ),
        (
            "Multi-scale-guided SAM2 - single-scale-guided SAM2",
            supporting["dinov2_multi_sam2_minus_dinov2_single_sam2"]["overall"],
            "inconclusive",
        ),
        (
            "Anomaly-consistent SAM2 - raw multi-scale-guided SAM2",
            paired["dinov2_multi_sam2_vs_anomaly_consistent_sam2"]["overall"],
            "robust improvement",
        ),
    ]
    lines = [
        "| Paired candidate-minus-baseline comparison | F1 delta (95% CI) | IoU delta (95% CI) | Reading |",
        "| --- | --- | --- | --- |",
    ]
    for label, result, reading in comparisons:
        f1 = result["f1"]
        iou = result["iou"]
        lines.append(
            f"| {label} | {_fmt(f1['mean_delta'], signed=True)} "
            f"{_interval(f1['ci_low'], f1['ci_high'])} | "
            f"{_fmt(iou['mean_delta'], signed=True)} "
            f"{_interval(iou['ci_low'], iou['ci_high'])} | {reading} |"
        )
    return "\n".join(lines)


def _ablation_f1_table() -> str:
    rows = [
        row
        for row in read_csv("ablation_summary.csv")
        if row["category"] == "macro" and row["metric"] == "mean_anomaly_mask_f1"
    ]
    rows.sort(key=lambda row: row["variant"])
    lines = [
        "| Ablation variant | Base method | Macro anomaly-image F1 |",
        "| --- | --- | ---: |",
    ]
    for row in rows:
        lines.append(f"| `{row['variant']}` | `{row['method']}` | {_fmt(row['mean'])} |")
    return "\n".join(lines)


def _strata_table() -> str:
    comparison = read_json("paired_statistics.json")["comparisons"][
        "dinov2_multi_vs_anomaly_consistent_sam2"
    ]
    lines = [
        "| Group | Stratum | F1 delta | 95% CI | Interpretation |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for group_key, group_label in (
        ("by_area_stratum", "defect area"),
        ("by_thinness_stratum", "shape/thinness"),
    ):
        for stratum, result in comparison[group_key].items():
            f1 = result["f1"]
            low = float(f1["ci_low"])
            high = float(f1["ci_high"])
            reading = "robust positive" if low > 0 else "inconclusive" if high >= 0 else "negative"
            lines.append(
                f"| {group_label} | {stratum} | {_fmt(f1['mean_delta'], signed=True)} | "
                f"{_interval(f1['ci_low'], f1['ci_high'])} | {reading} |"
            )
    return "\n".join(lines)


EVIDENCE_SETUP_CODE = r'''
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


def find_project_root(start: Path) -> Path:
    start = start.resolve()
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "docs/evidence").is_dir():
            return candidate
    raise FileNotFoundError("run the notebook from inside the repository")


PROJECT_ROOT = find_project_root(Path.cwd())
EVIDENCE_RELATIVE = Path("docs/evidence/generated")
EVIDENCE_DIR = PROJECT_ROOT / EVIDENCE_RELATIVE


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"missing frozen evidence: {path}")
    return path


def read_csv(name: str) -> list[dict[str, str]]:
    with require_file(EVIDENCE_DIR / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(name: str) -> dict:
    return json.loads(require_file(EVIDENCE_DIR / name).read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
'''


EVIDENCE_CHECK_CODE = r'''
manifest = read_json("completion_manifest.json")
assert manifest["ready_for_writing"] is True
assert manifest["readiness_scope"] == "complete_paper_evidence"
assert len(manifest["source_runs"]) == 412
for item in manifest["generated_files"]:
    path = require_file(EVIDENCE_DIR / item["path"])
    assert sha256(path) == item["sha256"], path
validation = read_json("validation_summary.json")
assert validation["primary"]["ok"] and validation["primary"]["bad"] == 0
assert validation["ablations"]["ok"] and validation["ablations"]["bad"] == 0
assert validation["method_invariants"]["ok"]
print("ready_for_writing:", manifest["ready_for_writing"])
print("source runs:", len(manifest["source_runs"]))
print("primary matrix:", validation["primary"]["complete"], "/", validation["primary"]["expected"])
print("ablation matrix:", validation["ablations"]["complete"], "/", validation["ablations"]["expected"])
print("method invariants:", validation["method_invariants"])
'''


RESULT_CHECK_CODE = r'''
conclusion = read_json("evidence_conclusion.json")
assert conclusion["outcome"] == "anomaly_consistent_sam2_improves_robustly"
print("outcome:", conclusion["outcome"])
for metric in ("f1", "iou"):
    result = conclusion[metric]
    print(
        f"{metric.upper()} delta={result['mean_delta']:.4f}, "
        f"95% CI [{result['ci_low']:.4f}, {result['ci_high']:.4f}]"
    )
'''


ORIENTATION_SECTIONS = (
    r'''
    # Few-Shot PCB Defect Segmentation

    ## A complete teaching walkthrough of the project

    **Project:** Few-Shot PCB Defect Segmentation via Multi-Scale DINOv2 Anomaly
    Proposals and SAM2 Mask Refinement

    This is the one notebook to read when you want to understand the entire
    project well enough to present it in class. It explains the motivation,
    datasets, frozen-feature pipeline, mathematical operations, repository code,
    experiment design, AutoDL execution, frozen results, limitations, and likely
    questions. Core implementation remains in `src/` and experiment entry points
    remain in `scripts/`; the notebook teaches and traces that code rather than
    creating a second implementation.

    > **One-sentence result:** under a controlled few-shot normal-only VisA PCB
    > protocol, intersecting the calibrated multi-scale DINOv2 proposal with its
    > anomaly-guided SAM2 mask robustly improves mean anomalous-image mask F1 and
    > IoU over the calibrated DINOv2 mask alone.

    No model parameters are trained or fine-tuned. There is no project-specific
    loss function, optimizer, epoch schedule, or learned classifier. The work is
    training-free inference with frozen pretrained backbones plus a support-image
    memory bank built independently for each category and run.
    ''',
    r'''
    ## 0. How to Use This Notebook

    Read the notebook in order once. Parts I-III build the conceptual and
    implementation story; Part IV explains how evidence was produced; Part V
    turns that understanding into a class presentation and Q&A preparation.

    **Markdown contains everything essential.** You do not have to execute cells
    to see the method, equations, result tables, or conclusions. The few Python
    cells use only the standard library. They verify that the tracked evidence is
    complete and checksum-correct; they do not load datasets, models, or GPUs.

    A useful study method is:

    1. First pass: read the executive story and the pipeline diagram.
    2. Second pass: explain every input/output shape in the method chapters.
    3. Third pass: reconstruct why the experiment matrix contains 364 primary
       and 48 ablation runs.
    4. Fourth pass: practice the slide outline without reading full paragraphs.
    5. Final pass: answer the Q&A bank aloud and use the cheat sheet to check facts.

    Throughout the notebook, **Source of truth** boxes name the exact file that
    implements a step. The displayed code is either a compact excerpt of that
    implementation or faithful pseudocode. If a classroom question asks for a
    detail beyond the presentation, these paths show where the answer comes from.

    The results are frozen outputs transferred from AutoDL and published under
    `docs/evidence/generated/`. This notebook formats them for reading; it does
    not alter them.
    ''',
    r'''
    ## Part I — Orientation

    ## 1. Executive Story

    ### The industrial problem

    A PCB inspection system must locate defects such as scratches, missing or
    extra material, and thin irregular structures. Pixel-accurate defect masks
    are valuable because they indicate defect extent, but dense defect labels are
    expensive and rare. Normal boards are much easier to collect. The project
    therefore asks whether a small set of normal examples can define expected
    appearance and whether two frozen foundation models can turn deviations from
    that appearance into useful masks.

    ### The research question

    > Can anomaly-guided SAM2 prompting improve few-shot PCB defect segmentation
    > over calibrated DINOv2 anomaly masks, especially for small or thin defects?

    ### The method in plain language

    - A few normal images provide examples of what each PCB category should look like.
    - DINOv2 converts image patches into semantic feature vectors.
    - Query patches far from every normal support patch receive high anomaly scores.
    - Global and overlapping local views form a multi-scale anomaly heatmap.
    - A threshold learned only from normal validation images converts the map into
      a binary anomaly proposal.
    - Connected anomalous regions provide automatic point and box prompts to SAM2.
    - SAM2 proposes object-shaped masks, but it does not know which pixels are
      anomalous.
    - The final mask keeps only pixels supported by both systems: anomaly proposal
      intersection guided SAM2.

    ### The frozen answer

    The final anomaly-consistent mask improves F1 by **+0.0666**, with 95% CI
    **[0.0538, 0.0788]**, and IoU by **+0.0565**, with 95% CI
    **[0.0462, 0.0665]**, relative to calibrated multi-scale DINOv2 on anomalous
    VisA PCB test images. Both lower bounds are above zero, satisfying the
    pre-declared robust-improvement decision rule.
    ''',
    r'''
    ### What the project contributes—and what it does not

    | Category | Defensible statement |
    | --- | --- |
    | Problem contribution | A controlled normal-only few-shot segmentation protocol for four VisA PCB categories. |
    | System contribution | A reproducible bridge from anomaly scores to automatic SAM2 points and boxes. |
    | Method contribution | An exact anomaly-consistency constraint, `M = A ∩ S`, that prevents SAM2 from adding unsupported pixels. |
    | Evidence contribution | Paired repeated-support evaluation, failure strata, ablations, and checksum-bound artifacts. |

    The project **does not** invent DINOv2, nearest-neighbor anomaly scoring,
    PatchCore, SAM2, or promptable segmentation. It makes no first-method or
    state-of-the-art claim. A simple sequence of strong pretrained models would
    be engineering, not enough novelty by itself; the scoped research value is
    the PCB protocol, proposal-to-prompt bridge, constraint, and evidence about
    when the combination helps.

    The supporting findings keep the story honest. Multi-scale versus
    single-scale DINOv2 is inconclusive, so the report cannot claim that
    multi-scale inference universally helps. Multi-scale DINOv2 does robustly
    outperform this repository's PatchCore-style baseline in paired mask
    comparisons. Multi-scale-guided versus single-scale-guided raw SAM2 is also
    inconclusive. The strongest supported claim is specifically about
    anomaly-consistent SAM2 relative to its calibrated multi-scale proposal.

    **Presentation cue:** state the problem, bridge, and final constraint before
    naming every implementation detail. Your audience then knows why each later
    component exists.
    ''',
    r'''
    ## 2. Conceptual Background

    ### Supervised segmentation versus normal-only anomaly segmentation

    In conventional supervised segmentation, a model sees many images paired
    with pixel masks and learns parameters that map pixels to defect classes. In
    this project, target-category defect masks are unavailable to the method.
    Only a few normal support images are used to represent expected appearance.
    The ground-truth test masks are used after prediction for evaluation, not to
    adapt the method.

    **Few-shot** means the support set is deliberately small: `k = 1, 2, or 4`
    normal images for one category. **Normal-only** means every support label is
    zero. **Frozen-feature** means DINOv2 and SAM2 weights never change.
    **Training-free** does not mean that nothing is computed: the run still
    extracts features, samples support images, builds a memory bank, calibrates a
    threshold, performs nearest-neighbor search, generates prompts, and executes
    SAM2 inference. The distinction is that none of those operations optimizes
    neural-network parameters on VisA or DeepPCB.

    | Term | What changes per run? | Learned with defect labels? |
    | --- | --- | --- |
    | DINOv2 weights | Nothing | No |
    | SAM2 weights | Nothing | No |
    | Normal memory bank | Support patch vectors change with category, k, and seed | No |
    | Calibration threshold | Fitted from normal validation heatmaps | No |
    | Prompts | Generated from each query anomaly map | No |
    | Final mask | Computed for each query | No |

    This distinction answers a common Q&A trap: “You have a memory bank, so did
    you train?” No. The bank stores frozen descriptors; it is non-parametric
    reference data, not a fitted neural network.
    ''',
    r'''
    ### DINOv2, PatchCore, and SAM2 in this project

    **DINOv2** is a self-supervised Vision Transformer. Instead of returning only
    one image-level vector, it exposes a grid of normalized patch-token vectors.
    Semantically similar patches tend to lie near one another in feature space.
    That makes DINOv2 useful as a frozen representation: a query patch unlike all
    normal support patches can be treated as anomalous.

    **PatchCore** is the conventional memory-bank comparison. This repository's
    PatchCore-style baseline takes intermediate Wide-ResNet50-2 feature maps,
    aligns and pools them, then selects a small deterministic coreset. It tests
    whether the DINOv2 representation offers an advantage under the same support
    and query protocol. It is a repository reproduction, not the authors'
    reference PatchCore implementation.

    **SAM2** is a promptable segmentation foundation model. A point or a box says
    what region to segment; SAM2 returns one or more plausible object masks. That
    is not anomaly detection. A normal copper trace may be a coherent object and
    therefore attractive to SAM2 even though it is not defective. The project
    uses DINOv2 to answer “where is appearance abnormal?” and SAM2 to answer “what
    boundary is consistent with this prompt?” The final intersection requires
    both answers to agree.

    A helpful mental model is:

    ```text
    DINOv2 = anomaly evidence     SAM2 = shape prior
                    \             /
                     agreement mask
    ```

    This division of responsibility explains both the strength and the main
    failure mode. SAM2 can clean boundaries, but raw SAM2 can expand onto normal,
    object-like structures. The anomaly constraint is therefore central rather
    than cosmetic.
    ''',
)


DATA_SECTIONS = (
    r'''
    ## Part II — Data and Software Foundations

    ## 3. Datasets and Benchmark Boundary

    ### VisA is the primary segmentation benchmark

    The Visual Anomaly (VisA) dataset supplies image-level normal/anomalous labels
    and pixel-level masks for anomalous images. This project uses its four PCB
    categories: `pcb1`, `pcb2`, `pcb3`, and `pcb4`. A VisA record contains the
    category, stable sample ID, split, image path, binary label, and optional mask
    path. The loader converts these into the shared immutable `DatasetRecord` in
    `src/datasets/types.py`; category-specific path parsing lives in
    `src/datasets/visa.py`.

    The method may inspect normal training/development images to form support and
    calibration sets. The official VisA test images are queries. Anomalous test
    masks are hidden from the inference path and used only by evaluation. Normal
    test images are important for detection and false-positive metrics, while
    paired anomalous-image mask comparisons exclude normal rows because an empty
    ground-truth mask makes per-defect boundary interpretation different.

    ### DeepPCB is secondary

    DeepPCB supplies aligned template/test pairs and bounding boxes. A box marks a
    rectangular defect extent; it does not identify which pixels inside the box
    are truly defective. Therefore DeepPCB can support box-level localization or
    coarse pseudo-mask visualization, but it cannot be mixed into VisA's true
    pixel-segmentation table. Its loader is `src/datasets/deeppcb.py`.

    | Dataset | Annotation | Valid project use | Invalid claim |
    | --- | --- | --- | --- |
    | VisA PCB | Image labels + pixel masks | Primary detection and segmentation evidence | None if protocol is disclosed |
    | DeepPCB | Bounding boxes + aligned template | Secondary box localization and qualitative checks | Calling boxes true segmentation masks |

    This boundary is a scientific design choice, not a loader limitation. It
    prevents a large rectangular pseudo-mask from being scored as if it were a
    precise defect contour.
    ''',
    r'''
    ## 4. Few-Shot Protocol

    The frozen primary study uses fold `0`, categories `pcb1`-`pcb4`, shots
    `k in {1, 2, 4}`, and support seeds `4880`-`4884`. For each category/k/seed
    cell, `sample_few_shot_normals` filters out every anomalous record, shuffles
    normal candidates with a deterministic Python RNG, selects exactly `k`, and
    sorts the selected sample IDs. If too few normals exist, it raises instead of
    silently running a weaker experiment.

    ```python
    # Compact excerpt of src/datasets/sampling.py
    grouped = normals_grouped_by_category(records)
    rng = random.Random(seed)
    candidates = list(grouped[category])
    rng.shuffle(candidates)
    support[category] = sorted(
        candidates[:k], key=lambda record: record.sample_id
    )
    ```

    **Source of truth:** `src/datasets/sampling.py` and
    `data/manifests/visa_pcb_folds.csv` (local/ignored manifest used by runs).

    ### What a seed means

    Each seed changes which normal support images represent the category. The
    query set does not change. The five seeds therefore measure sensitivity to
    support selection on the same test images. They are **not five folds**, and
    this is not five-fold cross-validation. Category and shot count are fixed
    reporting strata; support selection is repeated within each stratum.

    ### Fold and test-set disclosure

    Fold 0 partitions normal development images for support and validation. The
    official test set is locked and shared across fold IDs. Earlier development
    inspected VisA test results, so this test set is not an untouched holdout.
    The final configuration and evidence revision are frozen and checksum-bound,
    but a presentation must disclose prior test exposure rather than implying
    prospective blind evaluation.

    | Information | Used to build the method? | Used only to score it? |
    | --- | --- | --- |
    | Normal support images | Yes: memory bank | No |
    | Normal validation images | Yes: threshold calibration | No |
    | Test query pixels | Yes: inference input | No |
    | Test image labels | No | Yes |
    | Test defect masks | No | Yes |
    ''',
    r'''
    ## 5. Repository Map and Artifact Flow

    The repository deliberately separates scientific logic, runnable entry
    points, experiment declarations, heavy outputs, and presentation:

    ```text
    configs/                 fixed dataset and experiment choices
    src/datasets/            VisA/DeepPCB records, manifests, sampling
    src/features/            DINOv2, PatchCore, feature cache
    src/anomaly/             memory bank, patch scores, multi-scale projection
    src/sam_refine/          prompt regions, SAM2 adapter, mask fusion
    src/evaluation/          calibration, metrics, statistics, evidence tables
    src/experiments/         run specification, provenance, orchestration
    scripts/                 runnable smoke, matrix, audit, and evidence commands
    docs/evidence/generated/ compact frozen AutoDL tables/JSON/figures
    notebooks/               this explanation and presentation layer
    ```

    The end-to-end artifact flow is:

    ```text
    official dataset
      -> deterministic manifest
      -> YAML matrix config
      -> immutable RunSpec
      -> support IDs + source/config/checkpoint hashes
      -> feature cache and exact heatmap/mask outputs on AutoDL
      -> per-run metrics and status
      -> matrix checker
      -> statistical analysis
      -> frozen compact evidence
      -> this notebook
    ```

    `src/experiments/spec.py` expands YAML declarations into run IDs such as
    `dinov2_multi__pcb1__fold0__k1__seed4880`.
    `src/experiments/provenance.py` binds result-affecting inputs to hashes.
    `src/experiments/runner.py` executes dependencies and records status. The
    evidence package records source commit
    `ff4c07208278376b27a4954d560c715f51453c5e` and evidence postprocessor commit
    `45a5f93c556e1c1a7a9e39f535434f9244ea1a7a`.

    Heavy datasets, weights, caches, heatmaps, masks, and raw runs are ignored.
    The tracked evidence is intentionally compact: tables, statistical JSON,
    manifests, and eight selected panels. A result is not trusted because a file
    exists; the completion manifest and its SHA-256 inventory must validate.
    ''',
    r'''
    ### Evidence integrity gate

    The next two optional cells establish a fail-closed boundary. They find the
    repository without assuming a private absolute path, load only compact
    tracked files, require the final writing-ready scope, check all declared
    hashes, and verify the primary/ablation matrix counts plus method invariants.

    This gate is why the static tables later in the notebook can be traced back
    to immutable AutoDL-derived artifacts. It is not evidence that every research
    choice is perfect; it is evidence that the displayed package is complete,
    internally consistent, and unchanged since finalization.
    ''',
)


METHOD_SECTIONS_A = (
    r'''
    ## Part III — Method, Step by Step

    ## 6. Image Preparation and Alignment

    The method compares patch locations across support images, query images,
    heatmaps, prompts, SAM2 masks, and ground truth. A visually plausible mask is
    scientifically useless if these coordinate systems drift.

    `DINOv2PatchFeatureExtractor.extract` first converts the source to RGB, then
    resizes it while preserving aspect ratio and pads it to a square. With the
    primary DINOv2 setting, the prepared canvas is `518 x 518`. The helper returns
    `content_box=(x1,y1,x2,y2)`, the rectangle containing real resized content;
    padding outside that box is not allowed into the normal memory bank.

    ```python
    # src/features/dinov2.py
    scale = min(size / width, size / height)
    resized_size = (round(width * scale), round(height * scale))
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    offset = ((size - resized_width) // 2, (size - resized_height) // 2)
    canvas.paste(resized, offset)
    content_box = (offset_x, offset_y, offset_x + resized_width,
                   offset_y + resized_height)
    ```

    `PatchFeatureMap` stores five pieces together: feature grid, prepared image
    size, patch size, original `source_size`, and `content_box`. It checks that
    the grid dimensions match `image_size / patch_size`. Its
    `valid_patch_mask()` keeps patch centers inside real content, preventing black
    padding from becoming a common “normal” feature.

    When a patch-score grid is rendered, bicubic interpolation projects it
    through the recorded content geometry into source-image coordinates. Binary
    masks use nearest-neighbor resizing so labels remain binary. Multi-scale crop
    components remember their source boxes and are inserted into exactly those
    boxes. This explicit geometry is how image/mask alignment survives resizing.

    **Source of truth:** `src/features/dinov2.py`, `src/anomaly/heatmap.py`, and
    `src/utils/image.py`.
    ''',
    r'''
    ## 7. DINOv2 Patch-Feature Extraction

    The primary backbone is `dinov2_vits14`: a small Vision Transformer with
    patch size 14. The `518 x 518` prepared image contains `518/14 = 37` patches
    per axis, so one view produces a `37 x 37` grid, or 1,369 patch tokens. For
    ViT-S/14, each token has dimension `D=384`.

    $$
    x \in \mathbb{R}^{518\times518\times3}
    \xrightarrow{\text{DINOv2 ViT-S/14}}
    Z \in \mathbb{R}^{37\times37\times384}
    $$

    RGB pixels are scaled to `[0,1]`, normalized with ImageNet mean and standard
    deviation, transposed to channels-first, and batched. The extractor runs
    `forward_features` under `torch.no_grad()` and reads
    `x_norm_patchtokens`. The `norm` in that key matters: the model already
    returns normalized transformer patch representations, and the memory-bank
    code performs an explicit L2 normalization again before distance scoring.

    ```python
    # Compact excerpt of src/features/dinov2.py
    prepared, content_box = resize_and_pad_square_with_content_box(source, 518)
    tensor = _image_to_normalized_tensor(prepared, torch).to(device)
    with torch.no_grad():
        output = model.forward_features(tensor)
    patch_tokens = output["x_norm_patchtokens"].cpu().numpy()[0]
    features = patch_tokens.reshape(37, 37, patch_tokens.shape[-1])
    ```

    DINOv2 is not told what a PCB defect is. Its value is transferable visual
    structure learned during self-supervised pretraining. Normal-only nearest
    neighbors convert that generic representation into a category-specific
    anomaly detector without changing the backbone.

    **Input:** one RGB image. **Output:** `PatchFeatureMap[37,37,384]` plus geometry.
    **Source of truth:** `src/features/dinov2.py`.
    ''',
    r'''
    ## 8. Normal Memory Bank and Anomaly Score

    For a given category/k/seed run, features are extracted from the `k` selected
    normal supports. Valid, unpadded patch vectors are flattened and concatenated:

    $$
    \mathcal{B} = \{\hat z_{i,p}: i\in\text{support images},\ p\in\text{valid patches}\},
    \qquad \hat z = \frac{z}{\max(\lVert z\rVert_2,\epsilon)}.
    $$

    For every query patch vector $q$, the anomaly score is its Euclidean distance
    to the closest normal memory vector:

    $$
    a(q)=\min_{b\in\mathcal{B}}\lVert \hat q-\hat b\rVert_2.
    $$

    A low score means at least one normal patch looks similar in feature space. A
    high score means no stored normal patch is close. Because normalized vectors
    satisfy $\lVert q-b\rVert_2^2=2-2q^Tb$, Euclidean distance here is monotonic
    with cosine dissimilarity.

    ```python
    # src/anomaly/memory_bank.py
    memory_bank = concatenate(valid_support_patches)
    bank = l2_normalize(memory_bank)
    query = l2_normalize(query_features.flatten())
    squared = maximum(q_norms + bank_norms - 2.0 * query @ bank.T, 0.0)
    patch_scores = sqrt(min(squared, axis=1)).reshape(grid_h, grid_w)
    ```

    Distances are evaluated in query chunks (default 4,096) to avoid constructing
    an unnecessarily large query-by-bank matrix. The representative AutoDL
    `pcb1/k1/seed4880` DINOv2 run recorded 1,073 valid memory vectors; the exact
    count depends on support-image aspect ratios because padded patch centers are
    excluded.

    The `37 x 37` patch-score grid is not yet a mask. Projection metadata turns it
    into a smooth source-resolution heatmap, and later calibration decides which
    scores count as anomalous.

    **Source of truth:** `src/anomaly/memory_bank.py` and
    `src/anomaly/heatmap.py`.
    ''',
    r'''
    ## 9. PatchCore-Style Baseline

    A strong experiment needs a meaningful conventional comparison. The
    PatchCore-style baseline uses an ImageNet-pretrained Wide-ResNet50-2 rather
    than DINOv2. It extracts `layer2` and `layer3`, upsamples layer3 to layer2's
    spatial grid, concatenates channels, and applies `3 x 3` average pooling.
    With image size 512 and effective patch size/stride 8, the feature grid is
    `64 x 64` before padding validity is applied.

    ```python
    # src/features/patchcore.py
    output = backbone(tensor)
    layer3 = interpolate(output["layer3"], size=output["layer2"].shape[-2:])
    features = cat([output["layer2"], layer3], dim=1)
    features = avg_pool2d(features, kernel_size=3, stride=1, padding=1)
    ```

    A full bank can be large. The baseline projects features deterministically to
    64 dimensions and uses approximate greedy farthest-first selection to retain
    1% of patches. The projection and coreset seed are bound to the run seed. The
    representative AutoDL provenance records 3,072 full vectors reduced to 31.
    Query anomaly scores then use the same nearest-memory distance idea as the
    DINOv2 path.

    Why include this baseline? If DINOv2 were compared only with its own variants,
    we could not tell whether the foundation-model representation matters. The
    frozen paired evidence shows multi-scale DINOv2 improves anomaly-image F1
    over this repository's PatchCore-style baseline by `+0.0365`, with 95% CI
    `[0.0206, 0.0505]`.

    Why say **PatchCore-style**? The implementation follows the core memory-bank
    and coreset concept under this repository's shared protocol, but it is not the
    original authors' code and should not be presented as an exact reproduction.

    **Source of truth:** `src/features/patchcore.py` and
    `src/anomaly/memory_bank.py`.
    ''',
)


METHOD_SECTIONS_B = (
    r'''
    ## 10. Multi-Scale Anomaly Proposals

    A global `518 x 518` view provides context but compresses tiny defects into a
    small number of patch tokens. Multi-scale inference adds overlapping local
    crops from the original source image. The frozen primary configuration uses
    crop size 768 pixels, overlap 0.25, and max fusion.

    For a crop width $c$ and overlap fraction $o$, the nominal step is
    $\operatorname{round}(c(1-o))$. The implementation always adds the final
    right/bottom position, so the image boundary is covered even when the step
    does not divide the length. Each crop is independently resized/padded for the
    same feature extractor, scored against the same normal bank, and projected
    back into its source-image box.

    ```python
    # src/anomaly/multiscale.py
    components = [score(global_image) at full_image_box]
    for crop_size in crop_sizes:
        for box in iter_crop_boxes(image.size, crop_size, crop_overlap):
            crop_component = score(image.crop(box))
            components.append(PositionedHeatmap(box, crop_component))
    return ProjectedHeatmap(
        source_size=image.size,
        fusion="max",
        components=tuple(components),
    )
    ```

    `max` fusion asks whether either global context or a high-resolution local
    view finds a pixel anomalous. `mean` fusion averages the covering components.
    Max can recover a small high local score, but it can also preserve local false
    positives on repetitive traces. That trade-off is why crop size, overlap, and
    fusion are ablated instead of assumed beneficial.

    ### Exact compact heatmaps

    Raw full-resolution float32 heatmaps for hundreds of runs consume substantial
    disk space. The final run stores the global/local patch-score grids, component
    boxes, image sizes, patch sizes, content boxes, crop order, and fusion rule in
    compressed NPZ archives. `src/utils/heatmap_io.py` reconstructs the exact
    source-resolution float32 heatmap through `ProjectedHeatmap.render()`. This is
    lossless recipe storage, not quantization and not lower-resolution evaluation.

    **Important result:** multi-scale versus single-scale DINOv2 is inconclusive:
    F1 `+0.0045`, 95% CI `[-0.0058, 0.0142]`. Multi-scale remains part of the
    frozen proposal path, but the project does not claim a universal scale gain.
    ''',
    r'''
    ## 11. Normal-Only Calibration

    A continuous anomaly map must become a binary proposal before prompt
    extraction and final mask comparison. Choosing the threshold that maximizes
    test F1 would use the answers from anomalous test masks and exaggerate
    deployable performance. The operational threshold is therefore fitted only
    on normal validation heatmaps.

    Let $V$ contain every pixel score from the normal validation images for one
    run. The frozen threshold is the 99.5th percentile:

    $$
    \tau = Q_{0.995}(V), \qquad A(u,v)=\mathbf{1}[H(u,v)\ge\tau].
    $$

    Roughly speaking, this sets a nominal validation-pixel false-positive tail of
    0.5%. It does not guarantee a 0.5% false-positive rate on different test
    images because score distributions can shift.

    ```python
    # src/evaluation/calibration.py
    for row in rows:
        if row["label"] != "0" or row["fold_split"] != "val":
            raise ValueError("calibration requires normal validation rows")
    pixels = concatenate(load_heatmap(row["heatmap_path"]).ravel()
                         for row in rows)
    threshold = float(np.quantile(pixels, 0.995))
    ```

    The project also retains **oracle diagnostics**: thresholds selected to
    maximize a test metric. Oracle numbers answer “how separable are these scores
    if the best test threshold were known?” They do not answer “how would the
    system operate without test labels?” Every primary heatmap mask and every
    anomaly-consistent fusion uses normal-only calibration. Never compare an
    oracle DINOv2 number with a calibrated SAM2 number as if they were the same
    protocol.

    **Input:** normal validation heatmaps. **Output:** scalar $\tau$ with
    provenance. **Source of truth:** `src/evaluation/calibration.py` and
    `scripts/calibrate_heatmaps.py`.
    ''',
    r'''
    ## 12. Heatmap-to-Prompt Conversion

    SAM2 needs prompts; the anomaly map automates them. The calibrated binary map
    is scanned with four-connected flood fill. Components smaller than
    `min_area` are discarded. Each retained component provides an exclusive
    `(x1,y1,x2,y2)` box, its area, maximum anomaly score, and a positive point.

    The primary `point_mode=anomaly_max` chooses the highest-scoring pixel inside
    the component. Ties are deterministic: smallest row, then smallest column.
    The alternative `box_center` is an ablation. For a curved or thin defect, a
    geometric box center can fall on a normal pixel, so the anomaly maximum is a
    more semantically aligned positive prompt.

    ```python
    # src/sam_refine/prompts.py
    binary = heatmap >= threshold
    pixels = flood_fill(binary, start=(x, y))
    x1, x2 = min(xs), max(xs) + 1
    y1, y2 = min(ys), max(ys) + 1
    score = heatmap[ys, xs].max()
    point_xy = coordinates_of_deterministic_local_maximum(score)
    region = PromptRegion((x1, y1, x2, y2), point_xy,
                          area=len(pixels), score=score)
    ```

    Regions are sorted by `(maximum score, area)` descending and capped by
    `max_regions`. The primary prompt mode sends both the point and box. Ablations
    test point-only, box-only, and point-plus-box-center.

    Coordinate scaling matters. Prompt regions initially live in heatmap/source
    coordinates. Before SAM2 prediction, boxes and points are multiplied by
    image-to-heatmap scale factors and clipped to the image bounds. This keeps a
    maximum at heatmap position `(x,y)` aimed at the corresponding RGB location.

    **Source of truth:** `src/sam_refine/prompts.py` and prompt scaling helpers in
    `src/sam_refine/refiner.py`.
    ''',
    r'''
    ## 13. SAM2 Refinement and Candidate Selection

    The real AutoDL path uses frozen SAM2.1 Hiera Tiny. The adapter loads the
    configured checkpoint, sets the query image, and calls the image predictor
    for each anomaly-derived region. Depending on `prompt_mode`, the call supplies
    a point, a box, or both. `multimask_output` can return multiple plausible
    masks and confidence scores.

    Selecting only the highest SAM2 confidence can favor a large coherent object
    such as a PCB trace. The repository ranks each candidate with four factors:

    $$
    r(S)=
    \frac{s_{SAM2}(0.25+\bar H_S)(0.25+c_{prompt})}
         {0.25+|S|/(HW)}.
    $$

    Here $\bar H_S$ is mean normalized anomaly inside the mask,
    $c_{prompt}$ is the fraction of mask pixels inside the prompt box, and the
    denominator penalizes large masks. Candidates exceeding the primary 0.25
    image-area cap are skipped when possible. If every candidate exceeds the cap,
    the best normally ranked candidate is retained instead of returning nothing.

    ```python
    # src/sam_refine/refiner.py
    selection_score = (
        sam_score
        * (0.25 + anomaly_score_inside_mask)
        * (0.25 + prompt_containment)
        / (0.25 + mask_area_fraction)
    )
    ```

    SAM2 masks are resized back to the heatmap/source grid with nearest-neighbor
    interpolation. Accepted region masks are combined into the raw guided-SAM2
    mask $S$.

    A deterministic fallback refiner exists for offline smoke tests. It exercises
    prompt and artifact plumbing without pretending to be real SAM2 evidence.
    The primary frozen results use the real checkpoint, whose SHA-256 is recorded
    in runtime provenance.

    **Input:** RGB image, heatmap, prompt regions. **Output:** raw binary SAM2 mask.
    **Source of truth:** `src/sam_refine/refiner.py` and
    `scripts/run_mask_refinement.py`.
    ''',
    r'''
    ## 14. Anomaly-Consistent Fusion

    Let $A$ be the normal-calibrated multi-scale DINOv2 proposal and $S$ the raw
    multi-scale-guided SAM2 mask. The primary final output is exactly:

    $$
    \boxed{M=A\cap S}
    $$

    ```python
    # src/sam_refine/fusion.py
    anomaly = (anomaly_mask > 0).astype(np.uint8)
    sam2 = (sam2_mask > 0).astype(np.uint8)
    intersection = (anomaly & sam2).astype(np.uint8)
    return intersection
    ```

    This is an **anomaly-consistency constraint**. SAM2 is allowed to remove
    proposal pixels that do not fit its shape prediction, but it cannot introduce
    any pixel that DINOv2 did not mark anomalous under normal-only calibration.
    The guarantee is set-theoretic: $M\subseteq A$. It directly targets the
    observed failure where SAM2 expands along visually coherent normal traces.

    The trade-off is equally clear. Intersection can increase precision by
    removing false-positive expansion, but it cannot recover a true defect pixel
    missing from $A$ and may reduce recall. That is why the final claim is
    empirical, not guaranteed by the equation alone.

    Other implemented modes are:

    - `anomaly`: return $A$;
    - `sam2`: return $S$;
    - `union`: return $A\cup S$;
    - `selective`: use the intersection only when mask IoU and expansion rules
      pass, otherwise fall back to $A$.

    The frozen decision selected exact intersection. The selective and union
    modes remain descriptive ablations and are not silently substituted after
    seeing test results.

    **Source of truth:** `src/sam_refine/fusion.py` and
    `scripts/fuse_saved_masks.py`.
    ''',
    r'''
    ## 15. One-Image Trace

    Follow one query all the way through the system:

    1. **Manifest record:** `DatasetRecord` identifies category, sample ID, RGB
       path, test split, label, and evaluation mask path.
    2. **Support selection:** the category/k/seed RunSpec selects exactly `k`
       normal development records. Query information does not affect selection.
    3. **Support features:** each support becomes a padded `518 x 518` tensor and
       a `37 x 37 x 384` DINOv2 patch map. Invalid padded patch centers are removed.
    4. **Memory bank:** remaining support vectors are concatenated and L2-normalized.
    5. **Query global view:** the same extractor produces query patch vectors;
       nearest-normal distance produces a `37 x 37` score grid.
    6. **Local views:** overlapping source crops repeat extraction and scoring.
       Component metadata maps every crop grid back to its source rectangle.
    7. **Heatmap:** global/local grids are rendered at source resolution and fused
       by max. Exact compact storage preserves the component recipe.
    8. **Calibration:** the run-specific normal-validation 99.5th-percentile
       threshold turns the heatmap into binary proposal $A$.
    9. **Prompts:** connected proposal regions generate anomaly-maximum points and
       boxes in source coordinates.
    10. **SAM2:** prompts plus RGB pixels yield candidates. Anomaly-aware ranking
        and area policy select/merge them into $S$.
    11. **Fusion:** bitwise intersection produces $M=A\cap S$.
    12. **Evaluation:** $M$ is compared with the VisA mask. F1/IoU rows are saved;
        paired analysis later aligns this sample across methods and support runs.

    The qualitative panels later show five views of this trace: input, ground
    truth, anomaly score, raw guided SAM2, and anomaly-consistent output. Notice
    that ground truth participates only in step 12.

    ### Pipeline diagram

    ![Method diagram](../docs/evidence/generated/method_figure.png)
    ''',
    r'''
    ### Method summary table for presentation

    | Stage | Input | Operation | Output | Main risk |
    | --- | --- | --- | --- | --- |
    | Support | k normal images | Frozen patch extraction | Normal vectors | Support set may not cover normal variation |
    | Scoring | Query vectors + bank | Nearest-normal distance | Patch scores | Semantic similarity may miss subtle appearance changes |
    | Multi-scale | Global + local grids | Geometric projection and max | Source-resolution heatmap | Local crops may increase false positives |
    | Calibration | Normal validation maps | 99.5th percentile | Threshold and proposal A | Validation/test shift |
    | Prompting | Connected A regions | Boxes + anomaly maxima | SAM2 prompts | Proposal may miss a true defect |
    | SAM2 | RGB + prompts | Frozen mask prediction/ranking | Raw mask S | Object prior may expand onto normal structure |
    | Fusion | A and S | Exact intersection | Final mask M | Recall cannot exceed proposal support |

    If you can explain every row—especially why the output of one stage is the
    input to the next—you understand the algorithmic core. The project does not
    contain a hidden training stage between any two rows.
    ''',
)


def experiment_sections() -> tuple[str, ...]:
    primary_table = _primary_f1_table()
    heatmap_k4_table = _heatmap_k4_table()
    category_shot_table = _category_shot_delta_table()
    comparison_table = _comparison_table()
    ablation_table = _ablation_f1_table()
    strata_table = _strata_table()
    return (
        r'''
        ## Part IV — Experiments and Evidence

        ## 16. Primary Experiment Matrix

        The primary config is `configs/experiments/arxiv_primary.yaml`. It freezes
        categories, shots, seeds, methods, backbone geometry, crop settings,
        calibration quantile, SAM2 checkpoint/config, prompt policy, area cap, and
        PatchCore settings. A simplified exact view is:

        ```yaml
        categories: [pcb1, pcb2, pcb3, pcb4]
        shots: [1, 2, 4]
        seeds: [4880, 4881, 4882, 4883, 4884]
        methods:
          - patchcore
          - dinov2_single
          - dinov2_multi
          - sam2_only
          - dinov2_single_sam2
          - dinov2_multi_sam2
          - anomaly_consistent_sam2
        dinov2: {backbone: dinov2_vits14, image_size: 518, patch_size: 14}
        multi_scale: {crop_sizes: [768], crop_overlap: 0.25, fusion: max}
        calibration: {quantile: 0.995}
        sam2:
          prompt_mode: point_box
          point_mode: anomaly_max
          max_mask_area_fraction: 0.25
        ```

        Six methods depend on support selection: PatchCore, single DINOv2, multi
        DINOv2, single-guided SAM2, multi-guided SAM2, and anomaly-consistent
        SAM2. Their count is:

        $$4\ \text{categories}\times3\ k\text{ values}\times5\ \text{seeds}
        \times6\ \text{methods}=360.$$

        SAM2-only uses deterministic uniform grid prompts and no support bank, so
        it runs once per category: `4 x 1 = 4`. Thus the package contains
        **364 primary runs**. Counting SAM2-only 60 times would create fake
        repeated evidence; its table rows are correctly labeled `k=0`, seed 0,
        descriptive.
        ''',
        r'''
        ### What each primary method isolates

        | Method | Question answered | Depends on normal support? |
        | --- | --- | --- |
        | PatchCore-style | Does a conventional CNN memory bank work under the protocol? | Yes |
        | Single-scale DINOv2 | How strong is the simplest frozen DINOv2 anomaly map? | Yes |
        | Multi-scale DINOv2 | Do local crops change proposal quality? | Yes |
        | SAM2-only | Can uniform promptable segmentation find defects without anomaly evidence? | No |
        | Single-scale-guided SAM2 | Does single-scale anomaly prompting help SAM2? | Yes |
        | Multi-scale-guided SAM2 | What does raw SAM2 refinement do to multi-scale proposals? | Yes |
        | Anomaly-consistent SAM2 | Does constraining raw SAM2 to proposal support improve masks? | Yes |

        Pairing is critical. When two support-dependent methods are compared, the
        category, `k`, seed, support IDs, test sample IDs, and ordering match. A
        paired delta therefore reflects method change rather than a different
        support draw.

        `src/experiments/spec.py` creates immutable RunSpecs. Expensive DINOv2
        query/support views are cached with keys derived from every
        result-affecting input. SAM2 runs depend on saved proposal artifacts rather
        than re-running feature extraction. The matrix runner can `--resume`, but
        the checker rejects missing, failed, duplicated, or stale outputs.

        The real matrix ran on AutoDL with Linux x86_64, Python 3.10.8, PyTorch
        2.5.1+cu121, torchvision 0.20.1+cu121, CUDA 12.1, and an NVIDIA GeForce RTX
        4090. This Mac notebook does not reproduce that heavy computation; it
        reads the finalized evidence transferred from it.
        ''',
        r'''
        ## 17. Ablation Matrix

        Ablations change one declared method choice while fixing `k=4`, seed
        `4880`, and all four categories. There are 12 variants, giving
        **48 ablation runs** (`12 x 4`). Together the frozen package has
        **412 source runs** (`364 + 48`).

        | Family | Variants | Scientific purpose |
        | --- | --- | --- |
        | Crop geometry | 512/0.25/max; 768/0; 768/0.5; 768/0.25/mean | Test resolution, overlap, and fusion sensitivity |
        | Prompt type | anomaly-max point; box; point+box center | Test whether prompt geometry drives SAM2 quality |
        | Area cap | none; 0.10; 0.50 | Test candidate rejection sensitivity around primary 0.25 |
        | Mask fusion | union; selective fallback | Compare alternatives with exact intersection |

        These are **descriptive single-support-seed** results. They help explain
        mechanism and sensitivity but do not have the repeated-support uncertainty
        of the primary comparisons. A numerically attractive ablation must not
        replace the frozen primary method after viewing test performance.

        **Source of truth:** `configs/experiments/arxiv_ablations.yaml`,
        `scripts/run_experiment_matrix.py`, and
        `scripts/check_experiment_matrix.py`.
        ''',
        r'''
        ## 18. Metrics

        A segmentation result can be good in one sense and poor in another, so the
        project reports complementary metrics.

        For true positives `TP`, false positives `FP`, false negatives `FN`, and
        true negatives `TN`:

        $$P=\frac{TP}{TP+FP},\qquad R=\frac{TP}{TP+FN}$$

        $$F1=\frac{2PR}{P+R}=\frac{2TP}{2TP+FP+FN}$$

        $$IoU=\frac{TP}{TP+FP+FN}.$$

        - **Precision** asks how much predicted defect area is truly defective.
        - **Recall** asks how much true defect area is recovered.
        - **F1** balances precision and recall.
        - **IoU** measures direct region overlap and is usually numerically lower
          than F1 for the same mask.
        - **Image AUROC** ranks normal versus anomalous images using an image score.
        - **Pixel AUROC** ranks normal versus anomalous pixels across thresholds.
        - **AUPRO** integrates per-region overlap while limiting false-positive rate,
          making it useful for localized industrial defects.

        ### Aggregation matters

        `aggregate_pixel_f1` pools pixels before forming counts; large images or
        large defects can dominate. `mean_anomaly_mask_f1` computes one F1 per
        anomalous image and averages, giving each defect image equal weight within
        the declared aggregation. Category-macro reporting then gives each PCB
        category equal weight. Never place values with different aggregation or
        threshold rules in one comparison column.

        `metric_definitions.json` is the compact authoritative dictionary. Oracle
        metrics are diagnostic. Primary mask comparisons use normal-calibrated
        heatmap masks or saved binary method masks on identical images.
        ''',
        r'''
        ## 19. Statistical Design

        The main statistic is a paired candidate-minus-baseline delta on anomalous
        images. For one aligned cell:

        $$d_{c,k,s,i}=m^{candidate}_{c,k,s,i}-m^{baseline}_{c,k,s,i}.$$

        Here `c` is category, `k` shot count, `s` support seed, and `i` test image.
        Categories and shot counts are fixed strata. A bootstrap replicate samples
        support-seed indices once and reuses them across strata; it samples test
        image clusters once per category and reuses them across shots and seeds.
        Pairing is preserved at every level. The final replicate mean is a macro
        average across category/k strata.

        ```python
        # Conceptual structure of src/evaluation/statistics.py
        seed_indices = rng.integers(0, num_seeds, size=num_seeds)
        for category in categories:
            image_indices = resample_test_images_for_this_category()
            for k in shots:
                selected = delta_tensor[category, k][seed_indices, image_indices]
                replicate_sum += selected.mean()
        bootstrap_mean = replicate_sum / (num_categories * num_shots)
        ```

        Why not treat all 6,000 paired rows as independent? The same test image is
        evaluated under multiple support seeds and shot counts. Naive row
        resampling would overstate effective sample size. The reported 95%
        percentile interval uses 2,000 deterministic bootstrap replicates and
        names its resampling unit `support_seed_and_test_image`.

        A confidence interval wholly above zero supports a robust positive mean
        delta under this resampling design. If it crosses zero, the comparison is
        **inconclusive**, not proof that methods are identical. The interval also
        does not establish generalization beyond these categories, split, models,
        and prior test exposure.

        **Source of truth:** `src/evaluation/statistics.py`,
        `paired_statistics.json`, and `baseline_paired_statistics.json`.
        ''',
        _format_markdown(
            """
        ## 20. Frozen AutoDL Results

        The table below is inserted at notebook-build time by selecting the
        `category=macro`, `metric=mean_anomaly_mask_f1` rows from the frozen
        `primary_summary.csv`. These are displayed values, not recomputed metrics.

        {primary_table}

        Read across shots rather than selecting one favorite cell. Support-dependent
        methods have five-seed intervals. SAM2-only has one deterministic result
        and therefore no support-seed confidence interval.

        The values also show why the final claim is about a paired delta rather
        than simply “the largest number in one row.” The primary analysis aligns
        every anomalous image and support draw, then asks whether the candidate
        changes F1/IoU consistently.
        """,
            primary_table=primary_table,
        ),
        _format_markdown(
            """
        ### K=4 heatmap ranking metrics

        This table selects the frozen category-macro `k=4` rows for the three
        methods that directly produce continuous anomaly heatmaps. It answers a
        different question from the per-image mask table: how well do image/pixel
        scores rank anomalies, how well do regions overlap across thresholds, and
        how does the normal-calibrated operating mask behave?

        {heatmap_k4_table}

        No single method dominates every metric. For example, single-scale
        DINOv2 has the highest pixel AUROC here, while multi-scale DINOv2 has the
        highest AUPRO and calibrated aggregate F1 among the three. This reinforces
        why the paper uses named metrics and paired final-mask comparisons rather
        than declaring one method universally best from one column.

        ### F1 delta by category and shot

        The primary anomaly-consistent-minus-multi-scale comparison is also
        reported within each category and each shot count:

        {category_shot_table}

        Every displayed category and shot-count interval is above zero. This does
        not create independent studies—the same test images still repeat across
        support settings—but it shows that the overall positive mean is not driven
        by only one category or one value of k.
        """,
            heatmap_k4_table=heatmap_k4_table,
            category_shot_table=category_shot_table,
        ),
        _format_markdown(
            """
        ### Primary and supporting paired comparisons

        {comparison_table}

        The pre-declared outcome is
        **`anomaly_consistent_sam2_improves_robustly`**. Relative to calibrated
        multi-scale DINOv2, F1 changes by **+0.0666**, 95% CI
        **[0.0538, 0.0788]**, and IoU changes by **+0.0565**, 95% CI
        **[0.0462, 0.0665]**. Both lower bounds exceed zero.

        Three nuances are presentation-critical:

        1. Multi-scale versus single-scale DINOv2 is inconclusive because both
           F1 and IoU intervals include zero.
        2. Multi-scale DINOv2 is robustly better than this repository's
           PatchCore-style baseline in the paired anomaly-image comparison.
        3. Multi-scale-guided versus single-scale-guided raw SAM2 is inconclusive;
           the robust final improvement comes from anomaly consistency, not a
           proved universal benefit of multi-scale prompting.

        The comparison scope is anomalous VisA PCB test images. Normal images are
        excluded from these per-defect paired intervals but remain relevant to
        detection and calibration metrics.
        """,
            comparison_table=comparison_table,
        ),
        _format_markdown(
            """
        ## 21. Ablations and Failure Analysis

        ### Descriptive ablation F1

        {ablation_table}

        Because all rows use only `k=4`, seed `4880`, differences can be useful
        clues but do not measure support-set variability. For example, point-only
        prompting performs poorly relative to box or combined prompting; an
        isolated point does not give SAM2 enough spatial extent for these sparse
        defects. Several area-cap settings produce the same summarized value,
        indicating that the representative candidates did not frequently cross
        those alternative caps in a way that changed the selected masks.

        The mean-fusion crop ablation can be numerically attractive in this one
        support draw. It remains an ablation because the primary `768/0.25/max`
        configuration was frozen before final target evaluation. Selecting the
        best-looking ablation after seeing test masks would invalidate the
        controlled story.
        """,
            ablation_table=ablation_table,
        ),
        _format_markdown(
            """
        ### Where anomaly consistency helps

        {strata_table}

        Defect-area strata and thinness strata are computed from ground-truth
        geometry for analysis only; they do not select the method. The largest
        benefit appears for thin defects, consistent with the mechanism: SAM2 can
        help restrict a rough anomaly response to a narrow coherent structure,
        while intersection blocks expansion beyond anomalous support.

        The large-area F1 stratum is inconclusive in the final conclusion package.
        That does not contradict a positive overall mean. It means uncertainty in
        that subgroup is wide enough that the data do not support a robust
        large-area-specific claim. The notebook therefore says “especially thin”
        only where the strata support it and avoids claiming every geometry wins.

        Failure analysis also records component count, area fraction, thinness,
        anomaly/SAM2 overlap, SAM2-to-proposal expansion, and mean anomaly inside
        the SAM2 mask. These explain mechanisms; they do not train a selector.
        """,
            strata_table=strata_table,
        ),
        r'''
        ## 22. Qualitative Evidence

        Each category contributes one deterministic success and one failure case
        based on raw guided-SAM2 F1 delta. The panels contain five columns:

        1. **Input image:** what the frozen models receive.
        2. **Ground truth:** evaluation-only defect pixels.
        3. **Anomaly score:** continuous multi-scale DINOv2 evidence.
        4. **Guided SAM2:** raw mask from anomaly-generated prompts.
        5. **Anomaly-consistent:** final intersection with calibrated proposal.

        A success case teaches how SAM2 and anomaly evidence complement each
        other. A failure case is equally important: it prevents cherry-picked
        qualitative storytelling and shows where shape priors or proposals break.

        | Category | Success | Failure |
        | --- | --- | --- |
        | PCB1 | ![PCB1 success](../docs/evidence/generated/qualitative_figures/pcb1_success.png) | ![PCB1 failure](../docs/evidence/generated/qualitative_figures/pcb1_failure.png) |
        | PCB2 | ![PCB2 success](../docs/evidence/generated/qualitative_figures/pcb2_success.png) | ![PCB2 failure](../docs/evidence/generated/qualitative_figures/pcb2_failure.png) |
        | PCB3 | ![PCB3 success](../docs/evidence/generated/qualitative_figures/pcb3_success.png) | ![PCB3 failure](../docs/evidence/generated/qualitative_figures/pcb3_failure.png) |
        | PCB4 | ![PCB4 success](../docs/evidence/generated/qualitative_figures/pcb4_success.png) | ![PCB4 failure](../docs/evidence/generated/qualitative_figures/pcb4_failure.png) |

        These eight images illustrate behavior; they do not establish the average
        improvement. The paired 6,000-row comparisons establish the aggregate
        claim. In a presentation, show one success and one failure, then return to
        the confidence interval so the audience sees both mechanism and evidence.
        ''',
        r'''
        ### How to discuss a qualitative panel

        Start at the ground truth and identify whether the defect is thin,
        compact, or spatially fragmented. Compare the anomaly score with the
        ground truth: does DINOv2 localize the right region but spill into normal
        structure, or does it miss the defect? Then compare raw guided SAM2 with
        the proposal. If SAM2 expands along a normal object, the final
        intersection should remove pixels not supported by DINOv2. If DINOv2
        misses a true region, intersection cannot recover it.

        The title reports the sample ID and raw guided-SAM2 delta, which is the
        deterministic selection criterion. “Success” means raw guided SAM2 has a
        positive delta relative to its proposal for that selected case; “failure”
        means negative. It is not a category label and does not imply the final
        intersection always follows the raw delta.

        Three safe qualitative statements are:

        - “This case illustrates SAM2 following a thin candidate boundary.”
        - “This case illustrates expansion onto coherent normal structure.”
        - “The intersection enforces proposal support, but cannot add missing
          anomaly pixels.”

        Avoid “this image proves the method works.” One image cannot prove an
        average or an uncertainty interval.
        ''',
    )


INTERPRETATION_SECTIONS = (
    r'''
    ## Part V — Interpretation and Presentation

    ## 23. Research Answer and Novelty Boundary

    The frozen answer is narrow and defensible:

    > Under fold-0, normal-only few-shot VisA PCB evaluation, exact intersection
    > of a normal-calibrated multi-scale DINOv2 proposal with its anomaly-guided
    > SAM2 mask robustly improves anomalous-image mean F1 and IoU relative to the
    > calibrated proposal alone.

    This result supports the proposal-to-prompt bridge and anomaly-consistency
    constraint. It does not prove multi-scale inference is universally superior,
    because the single-/multi-scale interval crosses zero. It does not prove
    SAM2 is an anomaly detector. It does not establish state of the art because
    the study was not designed as an exhaustive external benchmark comparison.

    ### Novelty layers

    1. **Not novel:** using a pretrained DINOv2 backbone.
    2. **Not novel:** nearest-neighbor patch anomaly scoring; AnomalyDINO and
       related work already establish this family.
    3. **Not novel:** promptable segmentation with SAM2.
    4. **Scoped contribution:** automatically translating calibrated anomaly
       components into deterministic SAM2 points/boxes for normal-only PCB data.
    5. **Scoped contribution:** enforcing exact proposal support after SAM2 with
       `M=A∩S` and evaluating it under paired repeated-support evidence.
    6. **Evidence contribution:** reporting when the bridge helps or fails rather
       than presenting only a successful visual.

    In class, avoid saying “we combined DINOv2 and SAM2, which is novel.” Say:
    “The foundation models are existing components. Our contribution is a
    controlled anomaly-to-prompt pipeline and an exact consistency constraint,
    together with PCB-specific evidence showing that the constraint improves mask
    quality under the stated protocol.”

    The closest-work and citation boundary is tracked in
    `docs/citation_inventory.md`. The report should cite DINOv2, PatchCore,
    AnomalyDINO, and SAM2 rather than implying their methods originated here.
    ''',
    r'''
    ## 24. Limitations and Threats to Validity

    A strong presentation volunteers limitations before the audience discovers
    them.

    ### Internal validity

    - **Prior test exposure:** earlier development inspected VisA test results.
      The final configuration is frozen, but the test set is not an untouched
      prospective holdout.
    - **One normal-data fold:** the primary study uses fold 0. The five support
      seeds repeat support sampling; they are not five independent data splits.
    - **One-seed ablations:** all 48 ablation runs use `k=4`, seed 4880, so their
      sensitivity conclusions are descriptive.
    - **Threshold choice:** the 0.995 rule is fixed and normal-only, but its
      deployment behavior depends on how representative validation normals are.

    ### External validity

    - Only four VisA PCB categories provide primary segmentation evidence.
    - DeepPCB cannot extend the pixel-mask claim because it has boxes.
    - Results may change with other board manufacturing processes, cameras,
      resolutions, defect classes, DINOv2 sizes, or SAM2 checkpoints.
    - A training-free method reduces label requirements but still depends on large
      pretrained models and GPU-efficient execution for the full matrix.

    ### Method limitations

    - Intersection cannot recover a defect missed by DINOv2.
    - Nearest-neighbor banks can mistake unseen normal variation for anomaly.
    - SAM2 may segment coherent normal structure near a prompt.
    - Multi-scale max fusion can amplify crop-specific false positives.
    - No learned calibration adapts thresholds across domain shift.

    ### Future work that follows from evidence

    Repeat complete folds or new datasets; perform a prospectively frozen external
    evaluation; study normal-validation calibration under domain shift; test
    uncertainty-aware or leave-one-category-out selection without target test
    labels; and benchmark additional frozen backbones. These are future studies,
    not retroactive changes to the frozen evidence reported here.
    ''',
    r'''
    ## 25. Class Presentation Guide

    The following 12-slide sequence fits a detailed technical class presentation.
    Each slide has one job. If time is short, combine Slides 3-4 and 9-10 rather
    than rushing every slide.

    ### Slide 1: Problem and motivation

    Show a PCB image and a sparse defect mask. Say that pixel labels are expensive
    while normal boards are abundant. End with the question: can frozen models and
    a few normal examples produce useful defect masks?

    ### Slide 2: Research question and headline answer

    State the exact question and the final F1/IoU deltas with confidence intervals.
    Do not explain every method block yet. Tell the audience the final idea is
    “DINOv2 finds abnormal pixels, SAM2 proposes shape, intersection requires
    agreement.”

    ### Slide 3: Supervision and dataset protocol

    Explain VisA `pcb1`-`pcb4`, `k={1,2,4}`, five support seeds, normal validation,
    and the locked test set. Explicitly say the method sees no defect labels and
    that the seeds are not folds. Mention DeepPCB only as box-level secondary data.

    ### Slide 4: Frozen models and why no training

    Introduce DINOv2 patch tokens, the non-parametric normal memory bank, and SAM2
    prompts. Show the table of what changes per run. Answer the likely “where is
    the loss?” question before it is asked.

    ### Slide 5: DINOv2 anomaly scoring

    Show the `518 -> 37x37x384` tensor path and nearest-normal distance equation.
    Explain that every query patch asks whether any normal patch is close in
    feature space. Mention geometry/padding validity.

    ### Slide 6: Multi-scale proposal and calibration

    Show global plus overlapping local views, exact projection, and max fusion.
    Then show the normal-validation 99.5th-percentile threshold. Emphasize that
    oracle test thresholds are diagnostic only.

    ### Slide 7: From heatmap to SAM2

    Walk through connected components, anomaly-maximum points, boxes, candidate
    masks, and anomaly-aware ranking. Explain why SAM2 alone cannot know which
    object is defective.

    ### Slide 8: Anomaly-consistent fusion

    Put `M=A∩S` in the center. Explain the set guarantee `M subset of A`, expected
    precision/recall trade-off, and why union is risky. This is the conceptual
    climax of the method.

    ### Slide 9: Experiment matrix and baselines

    List seven methods. Derive `360 + 4 = 364` primary and `12 x 4 = 48` ablation
    runs. Say that AutoDL used a 4090 and that source/config/model identities were
    bound into provenance.

    ### Slide 10: Quantitative results and statistics

    Show the paired comparison table, not a wall of raw metrics. Explain that the
    bootstrap preserves support-seed and repeated-image structure. Label the two
    scale comparisons inconclusive.

    ### Slide 11: Success, failure, and limitations

    Show one success and one failure panel. Discuss thin-defect benefit, large-area
    uncertainty, prior test exposure, fold-0 scope, and one-seed ablations. This
    slide demonstrates scientific honesty.

    ### Slide 12: Contribution and takeaway

    Close with three points: normal-only frozen pipeline; anomaly-to-prompt bridge;
    exact anomaly-consistency constraint with robust paired improvement. End by
    stating what is not claimed: no new backbone, no SOTA claim, no five-fold
    cross-validation.
    ''',
    r'''
    ### Suggested opening and transitions

    **Opening (about 30 seconds):**

    “Industrial PCB inspection needs pixel-level localization, but defective
    examples and masks are scarce. We ask whether a few normal images and two
    frozen foundation models can produce accurate defect masks without training.
    DINOv2 supplies anomaly evidence, SAM2 supplies a shape prior, and our final
    mask keeps only pixels on which they agree.”

    **Transition into method:** “First we need a detector that can learn normality
    without parameter training. That is the role of the DINOv2 memory bank.”

    **Transition into SAM2:** “The heatmap knows abnormality but has rough
    boundaries. SAM2 knows object-like boundaries but not abnormality, so the
    anomaly map becomes its automatic prompt.”

    **Transition into fusion:** “Prompting alone does not stop SAM2 from expanding
    onto normal traces. The final constraint returns authority over anomaly status
    to the calibrated proposal.”

    **Transition into evidence:** “The equation motivates the method, but only a
    paired experiment can tell us whether its precision/recall trade-off helps.”

    **Closing (about 20 seconds):** “The supported result is not that every
    foundation-model combination works. It is that, under this controlled PCB
    protocol, exact anomaly consistency makes guided SAM2 reliably safer and
    improves F1 and IoU, with the strongest benefit for thin defects.”

    When presenting equations, define every symbol before interpreting the result.
    When presenting a number, state its baseline, inclusion rule, and uncertainty.
    ''',
)


QA_SECTIONS = (
    r'''
    ## 26. Anticipated Q&A — Method and Supervision

    ### Q1. Why did you not train a model?

    The research setting assumes defect masks are scarce but normal boards are
    available. Frozen DINOv2 already supplies transferable patch descriptors, and
    frozen SAM2 supplies prompt-conditioned shapes. The project tests whether a
    non-parametric normal memory bank plus deterministic prompting/fusion can solve
    the task without target-category defect supervision. No parameters are
    optimized, so there is no loss, optimizer, or epoch count. The support bank and
    calibration threshold are recomputed data artifacts, not trained weights.

    ### Q2. Is building the memory bank a form of training?

    No in the parameter-learning sense. It stores L2-normalized descriptors from
    selected normal images. Query scoring is nearest-neighbor retrieval. Nothing
    backpropagates, and the DINOv2 weights remain byte-identical. It is reasonable
    to call bank construction a per-run fitting or reference-building step, but not
    neural-network training.

    ### Q3. Why DINOv2?

    DINOv2 exposes strong self-supervised patch tokens that transfer to images
    outside its pretraining labels. The tokens preserve more local structure than
    a single global embedding, and one/few-shot anomaly work already shows that
    feature-space similarity is competitive. This project does not claim that
    choice as novel; it tests DINOv2 inside a controlled PCB prompt-generation
    pipeline.

    ### Q4. Why use patch tokens instead of the class token?

    Segmentation needs localization. A class token summarizes the whole image and
    cannot directly say which patch is abnormal. The `37 x 37` patch-token grid
    assigns a descriptor to spatial regions, so nearest-normal distance becomes a
    heatmap that can be projected back to pixels and converted into prompts.

    ### Q5. Why use Euclidean distance after L2 normalization?

    For unit vectors, squared Euclidean distance equals `2 - 2 cosine_similarity`.
    The ordering is therefore equivalent to cosine dissimilarity while making the
    nearest-neighbor implementation simple. Normalization also prevents feature
    magnitude from dominating semantic direction.

    ### Q6. Why not average the normal support features into one prototype?

    PCB appearance is spatially and semantically multi-modal: connectors, traces,
    background, solder, and other normal structures should not collapse into one
    vector. A patch memory bank preserves multiple modes. Each query patch only
    needs one close normal counterpart, which is more flexible than distance to a
    single average.

    ### Q7. What happens if the support images do not contain enough normal variation?

    Legitimate unseen structure may be far from the bank and become a false
    positive. Repeating support seeds measures some sensitivity, and increasing
    `k` adds coverage, but the method cannot guarantee coverage of a shifted
    production distribution. Better normal diversity and domain-shift monitoring
    are practical future improvements.

    ### Q8. Why use multi-scale crops?

    Resizing an entire PCB to 518 pixels can make a tiny defect occupy very few
    patch tokens. A local crop magnifies that region before patch extraction.
    Global context and local resolution are then projected to one source grid.
    This is a motivated component, not a proved universal win: the final paired
    multi-scale-versus-single-scale result is inconclusive.

    ### Q9. Why max fusion rather than mean fusion?

    Max preserves a strong anomaly found in either global or local view, which is
    attractive for small sparse defects. Mean can suppress a localized peak but
    may reduce crop-specific false positives. The frozen primary config uses max;
    mean is reported as a one-seed ablation and cannot be selected post hoc just
    because one descriptive number looks better.

    ### Q10. What guarantees that resizing does not misalign the masks?

    `PatchFeatureMap` stores source size, prepared size, patch size, and real
    content box. Only valid patch centers enter the bank. `ProjectedHeatmap`
    retains crop boxes and projection geometry, and binary masks use
    nearest-neighbor resizing. Tests cover shape and alignment invariants, and the
    frozen method-invariant audit passes for the primary images.
    ''',
    r'''
    ## Anticipated Q&A — SAM2, Calibration, and Metrics

    ### Q11. Why use SAM2 if it is not anomaly-aware?

    Because it supplies a useful shape prior once another method identifies a
    suspicious region. DINOv2 answers where appearance differs from normal; SAM2
    can turn a point/box in that region into a coherent boundary. The project does
    not delegate anomaly status to SAM2: the final intersection requires DINOv2
    support.

    ### Q12. Why not use SAM2 alone?

    SAM2 is promptable, not a prompt-free defect detector. The SAM2-only baseline
    uses deterministic image-grid prompts to measure what happens without anomaly
    guidance. It cannot know which segmented object is defective, and it has no
    support-set uncertainty because it uses no normal bank. Its role is a control,
    not the proposed method.

    ### Q13. How are the SAM2 prompts generated?

    The normal-calibrated binary anomaly proposal is decomposed into
    four-connected components. Each retained component yields an exclusive box
    and one positive point at its deterministic maximum anomaly pixel. The primary
    mode sends both. Regions are ranked by anomaly maximum and area and capped to
    a configurable count.

    ### Q14. Why use the anomaly maximum instead of the box center?

    A box around a thin or curved component can have a center on normal
    background. The anomaly maximum is guaranteed to lie on the selected component
    and aims the positive point at the strongest evidence. Box center remains a
    declared ablation rather than an untested assumption.

    ### Q15. How do you choose among multiple SAM2 masks?

    The ranking multiplies SAM2 confidence by terms for mean normalized anomaly
    and prompt containment, then divides by an area penalty. The primary policy
    also skips candidates above 25% of the image when alternatives exist. This
    reduces the chance that a confident but oversized object mask wins solely on
    SAM2 confidence.

    ### Q16. Why intersection instead of union?

    Union lets SAM2 introduce pixels with no calibrated anomaly evidence, exactly
    the expansion failure observed on coherent PCB structures. Intersection
    guarantees the final mask is a subset of the proposal and makes SAM2 a
    boundary filter rather than an anomaly classifier. It may lose recall, so its
    benefit is validated empirically. Union and selective fallback are ablations.

    ### Q17. Why is the calibration quantile 0.995?

    It was pre-registered as a normal-validation operating rule corresponding to
    the upper 0.5% validation-pixel score tail. It avoids target defect labels and
    provides one reproducible rule across runs. It is not theoretically optimal
    for every deployment; sensitivity to calibration and domain shift is future
    work.

    ### Q18. What is the difference between calibrated and oracle results?

    Calibrated masks use a threshold derived from normal validation scores and
    represent the operational pipeline. Oracle diagnostics choose the best
    threshold using test labels, showing potential score separability but not a
    deployable operating point. The notebook never uses oracle masks in the
    primary anomaly-consistent comparison.

    ### Q19. Why report both F1 and IoU?

    Both summarize overlap, but IoU penalizes the union directly and is stricter.
    F1 is the harmonic mean of pixel precision and recall. For one binary mask,
    they are monotonically related, yet reporting both aligns with segmentation
    conventions and makes the primary decision robust across two named metrics.

    ### Q20. Why include AUROC and AUPRO if F1 is the main claim?

    AUROC evaluates score ranking across thresholds; AUPRO emphasizes overlap of
    connected defect regions at controlled false-positive rates. They characterize
    heatmaps before one threshold is chosen. The final claim concerns calibrated
    binary mask quality, where per-image F1/IoU are directly interpretable.
    ''',
    r'''
    ## Anticipated Q&A — Protocol, Evidence, Novelty, and Reproducibility

    ### Q21. Are the five seeds five folds?

    No. All five seeds sample different normal supports within fold 0 and evaluate
    the same official test images. They measure support-set sensitivity, not
    independent dataset splitting. Describing them as five folds or five-fold
    cross-validation would be incorrect.

    ### Q22. Why use fold 0 only?

    The full DINOv2/SAM2 matrix is computationally expensive, and the frozen study
    scoped the primary evidence to one normal-data fold with repeated supports.
    This is enough for the stated within-protocol result but limits generality.
    Complete fold repetition is a clear next study, not something implied by the
    current evidence.

    ### Q23. How do you get 364 primary runs?

    Six support-dependent methods use `4 categories x 3 shots x 5 seeds`, which is
    60 runs per method and 360 total. SAM2-only is support-independent and runs
    once for each of four categories. `360 + 4 = 364`. The 12 one-seed ablation
    variants across four categories add 48, yielding 412 source runs.

    ### Q24. Why use a paired bootstrap?

    Candidate and baseline predictions exist for the same category, shot, support
    seed, and test image. Pairing removes variation shared by those cells. Because
    test images repeat across seeds and shots, resampling independent rows would
    overstate sample size. The implemented bootstrap jointly resamples support
    seeds and category-specific test-image clusters while fixing category/k strata.

    ### Q25. What does “inconclusive” mean here?

    It means the 95% interval for the paired mean delta includes zero under the
    declared resampling design. The data do not support a robust positive or
    negative direction. It does not prove exact equality. This is why the project
    avoids a universal multi-scale claim even though the point estimate is
    slightly positive for mask F1.

    ### Q26. What is actually novel?

    Not the backbones or nearest-neighbor scoring. The scoped contribution is the
    controlled normal-only PCB protocol, deterministic translation of calibrated
    anomaly regions into SAM2 prompts, exact anomaly-consistency intersection, and
    paired/failure-stratified evidence showing when that constraint helps.

    ### Q27. Is this state of the art?

    The evidence does not establish that. The study compares repository baselines
    under one controlled protocol; it does not reproduce every modern industrial
    anomaly method on identical splits. The safe claim is relative improvement
    over the named calibrated multi-scale DINOv2 baseline, not SOTA.

    ### Q28. What is the biggest limitation?

    The strongest validity limitation is that the study uses fold 0 and the same
    official test images across support seeds, with prior test-set exposure during
    development. The method is reproducibly frozen, but the evidence is not an
    untouched external validation. A prospectively locked new dataset/fold study
    would strengthen the claim most.

    ### Q29. Why can you not use DeepPCB to strengthen the segmentation result?

    DeepPCB annotations are boxes. Treating every pixel in a box as defective
    would create noisy rectangular pseudo-masks and could reward over-segmentation.
    DeepPCB is valid for box localization or qualitative secondary analysis, but
    not for the true pixel-mask claim reported on VisA.

    ### Q30. How can someone reproduce or audit the result?

    The repo records dataset acquisition/layout, manifest creation, fixed YAML
    configs, RunSpec IDs, source/model/config/checkpoint hashes, exact AutoDL
    commands, matrix checkers, statistical scripts, and a final completion
    manifest. The tracked evidence package contains tables, paired statistics,
    runtime provenance, selected figures, and SHA-256 bindings. A local offline
    synthetic smoke exercises the command path without claiming to reproduce the
    real GPU numbers.
    ''',
)


APPENDIX_SECTIONS = (
    r'''
    ## 27. Glossary and Symbol Table

    | Term | Meaning in this project |
    | --- | --- |
    | Anomaly detection | Finding inputs or pixels that differ from normal reference data. |
    | Anomaly segmentation | Producing a pixel mask for anomalous regions. |
    | Few-shot | Using a deliberately small support set; here k is 1, 2, or 4. |
    | Normal-only | Support and calibration use only label-0 images. |
    | Frozen backbone | A pretrained neural network whose parameters never change. |
    | DINOv2 | Self-supervised Vision Transformer used for normalized patch tokens. |
    | Patch token | Feature vector representing one spatial image patch. |
    | Memory bank | Stored normal patch vectors used for nearest-neighbor scoring. |
    | PatchCore-style | Repository CNN-feature/coreset baseline inspired by PatchCore. |
    | Coreset | Small representative subset of a larger memory bank. |
    | Heatmap | Continuous per-pixel anomaly score after patch-grid projection. |
    | Calibration | Choosing an operational threshold from normal validation data. |
    | Oracle | Test-label-optimized diagnostic, not an operational result. |
    | Prompt region | Connected anomaly component summarized by a box and point. |
    | SAM2 | Frozen promptable segmentation model used as a shape prior. |
    | Anomaly consistency | Requirement that final pixels remain supported by anomaly proposal A. |
    | Macro average | Equal-weight average across declared categories or strata. |
    | Paired delta | Candidate metric minus baseline metric on the same experimental cell. |
    | Bootstrap CI | Interval from deterministic resampling of declared units. |
    | AUPRO | Area under per-region-overlap curve within a false-positive range. |
    | Provenance | Source, config, data, model, environment, and hash identity for a run. |
    | Ablation | Controlled variant used to study one design choice. |

    | Symbol | Definition | Typical shape/value |
    | --- | --- | --- |
    | $x$ | Source RGB image | `H x W x 3` |
    | $Z$ | DINOv2 patch feature grid | `37 x 37 x 384` |
    | $\mathcal{B}$ | Normal memory bank | `N_valid x 384` |
    | $q,b$ | Normalized query and bank patch vectors | `384` |
    | $a(q)$ | Nearest-normal patch anomaly score | scalar |
    | $H(u,v)$ | Source-resolution continuous anomaly heatmap | `H x W` |
    | $\tau$ | Normal-validation 0.995 quantile threshold | scalar per run |
    | $A$ | Calibrated binary anomaly proposal | `H x W`, binary |
    | $S$ | Raw anomaly-guided SAM2 mask | `H x W`, binary |
    | $M$ | Final anomaly-consistent mask, $A\cap S$ | `H x W`, binary |
    | $c,k,s,i$ | Category, shot count, support seed, test image | pairing indices |
    ''',
    r'''
    ## 28. Final Cheat Sheet

    ### Facts to remember before class

    - **Task:** few-shot normal-only PCB defect segmentation.
    - **Main dataset:** VisA `pcb1`-`pcb4` with pixel masks.
    - **Secondary dataset:** DeepPCB with boxes, not segmentation masks.
    - **Training:** none; DINOv2 and SAM2 are frozen.
    - **Supports:** `k={1,2,4}` normal images.
    - **Seeds:** `4880-4884`, repeated support selection—not folds.
    - **Backbone:** DINOv2 ViT-S/14, input 518, patch 14, grid `37x37`, dimension 384.
    - **Anomaly score:** nearest L2 distance to an L2-normalized normal bank.
    - **Multi-scale primary:** crop 768, overlap 0.25, max fusion.
    - **Calibration:** normal validation 99.5th percentile.
    - **Prompts:** connected-component box plus anomaly-maximum positive point.
    - **SAM2:** SAM2.1 Hiera Tiny, candidate area cap 0.25.
    - **Final equation:** `M = A intersection S`.
    - **Primary runs:** 364. **Ablation runs:** 48. **Total:** 412.
    - **Compute:** AutoDL NVIDIA GeForce RTX 4090, CUDA 12.1.
    - **Main F1 delta:** `+0.0666`, 95% CI `[0.0538, 0.0788]`.
    - **Main IoU delta:** `+0.0565`, 95% CI `[0.0462, 0.0665]`.
    - **Scale claim:** multi versus single DINOv2 is inconclusive.
    - **PatchCore claim:** multi DINOv2 is robustly better in paired mask F1/IoU.
    - **Geometry finding:** benefit is largest for thin defects; large-area result
      is not a robust subgroup claim.
    - **Novelty:** protocol + anomaly-to-prompt bridge + exact consistency
      constraint + controlled evidence; not a new backbone or SOTA claim.
    - **Biggest limitation:** fold-0 scope and prior exposure to shared official
      test images.

    ### Three-sentence answer if called on unexpectedly

    “We represent each PCB category with frozen DINOv2 patch features from a few
    normal images and score query patches by nearest-normal distance. A
    normal-validation threshold turns the multi-scale heatmap into point/box
    prompts for frozen SAM2, then exact intersection prevents SAM2 from adding
    pixels unsupported by anomaly evidence. On the frozen VisA PCB study, this
    improves anomalous-image F1 by 0.0666 and IoU by 0.0565 with paired 95%
    intervals above zero, while multi-scale alone remains inconclusive.”

    ### Final self-check

    Before presenting, confirm you can derive 364, explain `37x37x384`, distinguish
    calibrated and oracle metrics, explain why the seeds are not folds, defend
    intersection's recall limitation, and state novelty without claiming DINOv2
    or SAM2 as your invention.
    ''',
    r'''
    ## 29. Reproduction Appendix

    Reproduction has three levels. Do not confuse a local smoke test with the
    frozen real-data AutoDL study.

    ### A. Environment

    Use Python 3.10-3.12 with PyTorch-compatible packages:

    ```bash
    python3.11 -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    python -m pip install -e ".[dev]"
    ```

    Real DINOv2 may download official weights through PyTorch Hub. Real SAM2
    requires the separately installed SAM2 repository, config, and ignored
    checkpoint. Dataset and weight paths never belong in git.

    ### B. Offline CPU-safe synthetic smoke

    This checks orchestration, geometry, calibration, fallback prompting, fusion,
    metrics, and evidence plumbing without network, GPU, real data, or weights:

    ```bash
    PYTHONPATH=src python scripts/create_synthetic_data.py --output-dir data/debug_fixture
    PYTHONPATH=src python scripts/create_manifests.py \
      --visa-root data/debug_fixture/VisA \
      --visa-category pcb1 \
      --output-dir data/manifests
    PYTHONPATH=src python scripts/run_experiment_matrix.py \
      --config configs/experiments/arxiv_smoke.yaml \
      --output-root outputs/arxiv_smoke \
      --device cpu \
      --allow-dirty
    PYTHONPATH=src python scripts/check_experiment_matrix.py \
      --config configs/experiments/arxiv_smoke.yaml \
      --output-root outputs/arxiv_smoke \
      --device cpu \
      --output-json outputs/arxiv_smoke/matrix_summary.json
    ```

    Passing this smoke proves the tiny command path works. It does not reproduce
    the AutoDL numbers or establish real-data performance.
    ''',
    r'''
    ### C. Full primary and ablation matrices on AutoDL

    After uploading official data, manifest, SAM2 installation, and checkpoint as
    described in `docs/autodl_data_setup.md`:

    ```bash
    PYTHONPATH=src python scripts/run_experiment_matrix.py \
      --config configs/experiments/arxiv_primary.yaml \
      --output-root outputs/arxiv_primary \
      --feature-cache-dir outputs/.feature_cache \
      --device cuda \
      --resume

    PYTHONPATH=src python scripts/check_experiment_matrix.py \
      --config configs/experiments/arxiv_primary.yaml \
      --output-root outputs/arxiv_primary \
      --feature-cache-dir outputs/.feature_cache \
      --device cuda \
      --output-json outputs/arxiv_primary/matrix_summary.json

    PYTHONPATH=src python scripts/run_experiment_matrix.py \
      --config configs/experiments/arxiv_ablations.yaml \
      --output-root outputs/arxiv_ablations \
      --dependency-root outputs/arxiv_primary \
      --feature-cache-dir outputs/.feature_cache \
      --device cuda \
      --resume
    ```

    Analysis and paper evidence are produced by
    `scripts/analyze_paper_results.py`, `scripts/curate_paper_assets.py`,
    `scripts/render_method_figure.py`, and `scripts/build_paper_evidence.py`.
    Final publication used the frozen audit/finalization commands recorded in
    `frozen_audit_manifest.json` and `completion_manifest.json`; use those records
    rather than reconstructing private absolute paths.

    ### D. Verify this teaching notebook

    ```bash
    venv/bin/python scripts/build_teaching_notebook.py --check
    PYTHONPATH=src venv/bin/python -m pytest tests/test_presentation_notebook.py -q
    ```

    The first command proves deterministic notebook generation. The second checks
    chapter coverage, depth, source paths, images, output-free cells, evidence
    hashes, matrix counts, and top-to-bottom execution.
    ''',
    r'''
    ## End of Walkthrough

    You now have the complete chain required for presentation:

    ```text
    scarce defect labels
      -> few normal supports
      -> frozen DINOv2 patch features
      -> nearest-normal anomaly scores
      -> global/local projection
      -> normal-only calibration
      -> automatic points and boxes
      -> frozen SAM2 shape candidates
      -> exact anomaly-consistency intersection
      -> paired VisA evaluation
      -> frozen AutoDL evidence
    ```

    The most important discipline is to keep three layers separate:

    1. **Algorithm:** what the code computes.
    2. **Evidence:** what the frozen experiment supports.
    3. **Claim:** what you are allowed to say in class.

    Algorithmically, the final mask is simple. Evidentially, its value comes from
    paired repeated-support evaluation, not visual intuition. Scientifically, the
    claim is intentionally scoped to the controlled VisA PCB study. That
    separation is what makes the project understandable, reproducible, and
    defensible under Q&A.
    ''',
)


def _validate_frozen_evidence() -> None:
    manifest = read_json("completion_manifest.json")
    validation = read_json("validation_summary.json")
    conclusion = read_json("evidence_conclusion.json")
    if manifest.get("ready_for_writing") is not True:
        raise RuntimeError("frozen evidence is not writing-ready")
    if manifest.get("readiness_scope") != "complete_paper_evidence":
        raise RuntimeError("unexpected evidence readiness scope")
    if len(manifest.get("source_runs", [])) != 412:
        raise RuntimeError("expected exactly 412 frozen source runs")
    if validation["primary"].get("complete") != 364 or validation["primary"].get("bad") != 0:
        raise RuntimeError("primary matrix must be complete at 364/364 with zero bad runs")
    if validation["ablations"].get("complete") != 48 or validation["ablations"].get("bad") != 0:
        raise RuntimeError("ablation matrix must be complete at 48/48 with zero bad runs")
    if conclusion.get("outcome") != "anomaly_consistent_sam2_improves_robustly":
        raise RuntimeError("unexpected frozen evidence conclusion")
    expected = {
        "f1": (0.06664446687355506, 0.053798811697808646, 0.07879064833302009),
        "iou": (0.05650546376886057, 0.04624032040163985, 0.06654617114428095),
    }
    for metric, values in expected.items():
        result = conclusion[metric]
        observed = (result["mean_delta"], result["ci_low"], result["ci_high"])
        if observed != values:
            raise RuntimeError(f"frozen {metric} result does not match teaching narrative")


def build_cells() -> list[dict[str, object]]:
    _validate_frozen_evidence()
    specs: list[tuple[str, str]] = []

    def add_markdown(sections: tuple[str, ...]) -> None:
        specs.extend(("markdown", section) for section in sections)

    add_markdown(ORIENTATION_SECTIONS)
    add_markdown(DATA_SECTIONS)
    specs.append(("code", EVIDENCE_SETUP_CODE))
    specs.append(("code", EVIDENCE_CHECK_CODE))
    add_markdown(METHOD_SECTIONS_A)
    add_markdown(METHOD_SECTIONS_B)
    add_markdown(experiment_sections())
    specs.append(("code", RESULT_CHECK_CODE))
    add_markdown(INTERPRETATION_SECTIONS)
    add_markdown(QA_SECTIONS)
    add_markdown(APPENDIX_SECTIONS)

    cells = []
    for index, (cell_type, source) in enumerate(specs):
        factory = markdown_cell if cell_type == "markdown" else code_cell
        cells.append(factory(index, source))
    return cells


def build_notebook() -> dict[str, object]:
    return {
        "cells": build_cells(),
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def serialized_notebook() -> str:
    return json.dumps(build_notebook(), indent=1, ensure_ascii=False) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    rendered = serialized_notebook()
    if args.check:
        if not output.is_file() or output.read_text(encoding="utf-8") != rendered:
            print(f"notebook is out of date: {output}", file=sys.stderr)
            return 1
        print(f"notebook is current: {output}")
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
