# ArXiv Evidence Readiness Research Design

**Date:** 2026-07-11  
**Project:** Few-Shot PCB Defect Segmentation  
**Selected direction:** Evidence-led selective SAM2 refinement

## Objective

Finish the project to the point where report writing can begin without further
method design, implementation, or evidence collection. At that point the
repository must contain reproducible experiment code and commands, while the
local ignored artifact set must contain complete quantitative tables,
statistical analyses, and curated figures suitable for an arXiv-style report.

This design does not require SAM2 to win. It requires a fair, controlled answer
to whether SAM2 helps, where it fails, and whether a simple anomaly-consistency
constraint can make its use safer for sparse PCB defects.

## Scientific Positioning

### Provisional title

**When Segment Anything Is Not Anomaly-Aware: Anomaly-Consistent SAM2
Refinement for Few-Shot PCB Defect Segmentation**

The title may be shortened after the final results are frozen, but the final
wording must match the evidence. It must not imply that SAM2 improves every
category or metric.

### Research question

> Under normal-only few-shot supervision, when does anomaly-guided SAM2 improve
> PCB defect masks, and can agreement with the originating anomaly map prevent
> SAM2 from over-segmenting normal PCB structures?

### Hypotheses

1. Multi-scale DINOv2 proposals improve localization of small PCB defects over
   single-scale patch anomaly maps.
2. Unconditional SAM2 refinement is category- and geometry-dependent: it can
   recover defect extent, but it can also expand onto visually coherent copper
   traces and background structures.
3. Intersecting SAM2 masks with a calibrated anomaly proposal provides a simple,
   training-free anomaly-consistency constraint that improves precision and
   mean mask F1 without using target-category defect labels.
4. SAM2 benefit is predictable from measurable properties such as defect area,
   connected-component structure, anomaly density inside the SAM2 mask, mask
   expansion, and heatmap/SAM2 agreement.

### Claims the report may make

- A controlled empirical claim about multi-scale DINOv2 proposals on the four
  VisA PCB categories.
- A controlled empirical claim about when SAM2 helps or hurts PCB anomaly masks.
- A method claim for a lightweight anomaly-consistency fusion rule only if the
  final untuned or validation-calibrated results support it.
- A negative-result claim about unconditional SAM2 refinement if it remains
  inferior after the pre-registered ablations.

### Claims the report must not make

- DINOv2 nearest-neighbor anomaly scoring is novel.
- Multi-scale foundation-model features are novel by themselves.
- SAM2 or promptable segmentation is novel.
- DeepPCB boxes are segmentation ground truth.
- Test-optimal thresholds represent deployable performance.
- A result from one support sample, one seed, one category, or a smoke subset is
  representative of the complete method.

### Related-work constraints on novelty

The implementation and report must be positioned against at least these primary
sources before finalizing novelty language:

- DINOv2: <https://arxiv.org/abs/2304.07193>
- PatchCore: <https://arxiv.org/abs/2106.08265>
- AnomalyDINO: <https://arxiv.org/abs/2405.14529>
- SAM2: <https://arxiv.org/abs/2408.00714>
- Few-shot industrial defect segmentation with foundation models:
  <https://arxiv.org/abs/2502.01216>
- RadioCore multi-scale foundation-feature anomaly segmentation:
  <https://openaccess.thecvf.com/content/CVPR2026W/VISION26/html/Ali_RadioCore_Few-Shot_Industrial_Anomaly_Segmentation_with_Multi-Scale_Radio_ViT_Features_CVPRW_2026_paper.html>
- Quantified limits of SAM-family models on low-contrast and tree-like targets:
  <https://openaccess.thecvf.com/content/WACV2026/html/Zhang_Quantifying_the_Limits_of_Segmentation_Foundation_Models_Modeling_Challenges_in_WACV_2026_paper.html>

These works make a generic "DINOv2 plus SAM2" or "multi-scale features" claim
insufficient. The defensible contribution must come from the controlled PCB
study, anomaly-consistency mechanism, and failure analysis.

## Current Evidence and Its Limits

The current fold-0, `k=5` outputs establish that the pipeline runs on all four
VisA PCB categories. Existing summary tables show a strong multi-scale heatmap
result and mixed SAM2 behavior. The test suite currently passes 56 tests in the
Python 3.11 environment.

The VisA test metrics have already been inspected during development and helped
identify the current `crop_size=768`, `crop_overlap=0.25`, `fusion=max`
configuration. The report must disclose this prior exposure and must not call
the VisA test set an untouched holdout. This design freezes that multi-scale
configuration for the primary repeated-seed experiment; subsequent target-test
results cannot be used to replace it with a better-looking configuration.

The existing Stage 4 table is not a valid direct mask-quality comparison:

- heatmap `best_pixel_f1` is computed globally over sampled test pixels at a
  test-optimal threshold;
- SAM2 `mean_anomaly_mask_f1` is a mean of per-image binary-mask F1 values.

Those values answer different questions and must not occupy the same comparison
column without an explicit metric definition.

An exploratory re-evaluation of the existing outputs using the same per-image
F1 definition found the following anomaly-image means:

| Category | Multi-scale heatmap | SAM2 refinement |
| --- | ---: | ---: |
| `pcb1` | 0.3844 | 0.2769 |
| `pcb2` | 0.3087 | 0.2588 |
| `pcb3` | 0.2329 | 0.2568 |
| `pcb4` | 0.1910 | 0.2751 |
| Mean | 0.2792 | 0.2669 |

A simple heatmap/SAM2 intersection reached exploratory mean per-image F1
`0.2918`. These values were derived from the test set and are hypothesis-forming
only. They are not final results and must not be used to select final
hyperparameters.

## Method Design

### 1. Few-shot anomaly proposals

For each category and run:

1. sample `k` normal support images using a recorded random seed;
2. extract frozen DINOv2 ViT-S/14 patch features;
3. construct an L2-normalized normal patch memory bank;
4. score query patches by nearest-normal-feature distance;
5. resize patch scores to the query image while preserving mask alignment;
6. optionally score overlapping local crops and fuse them with the global map.

Single-scale and multi-scale runs must use the same support images and query
records. Query features should be cached independently of support selection so
the seed and shot matrix does not repeat expensive backbone inference.

### 2. Proposal calibration

Every heatmap method produces two clearly separated thresholded outputs:

- **Oracle diagnostic:** the threshold maximizing test-set pixel F1. This is
  reported only as `oracle_best_*` and never used by the operational pipeline.
- **Normal-calibrated output:** a category- and run-specific threshold equal to
  the 99.5th percentile of pixel anomaly scores on the normal validation split.
  This corresponds to a nominal 0.5% validation-pixel false-positive target and
  uses no anomalous image or mask.

The 99.5th-percentile rule is pre-registered here. It may be included in a
sensitivity plot with 99.0% and 99.9%, but the primary operational result may
not be changed after examining test masks.

### 3. SAM2 variants

The experiment suite includes:

- **SAM2-only:** deterministic uniform prompts without anomaly evidence;
- **single-scale DINOv2 + SAM2:** prompts from single-scale heatmaps;
- **multi-scale DINOv2 + SAM2:** prompts from multi-scale heatmaps;
- **anomaly-consistent SAM2:** multi-scale SAM2 masks intersected with the
  corresponding normal-calibrated anomaly proposal.

SAM2-only is intentionally semantic-free. Its purpose is to demonstrate whether
promptable segmentation alone can identify defects; it is not expected to be a
strong anomaly detector.

The default SAM2 prompt is one connected-region box plus one positive point at
the local anomaly maximum. Center points remain an ablation because a box center
can fall on a normal pixel for thin or irregular defects.

### 4. Anomaly-consistency fusion

Let `A` be the normal-calibrated binary anomaly proposal and `S` be the union of
accepted SAM2 masks for the same image. The primary proposed output is

`M = A ∩ S`.

This rule adds no trained parameters and guarantees that SAM2 cannot introduce
pixels unsupported by the calibrated anomaly detector. It tests whether SAM2
can act as a boundary prior rather than an anomaly classifier.

The following are ablations, not silently substituted primary methods:

- `A` alone;
- `S` alone;
- `A ∪ S`;
- mask-candidate area filtering;
- point-only, box-only, center-point-plus-box, and anomaly-maximum-point-plus-box
  prompting;
- a selective fallback that retains `A` when SAM2/anomaly agreement is low.

Any selective fallback thresholds must be fixed before target test evaluation.
They may be chosen from normal validation behavior or by leave-one-category-out
calibration, and their provenance must be written into the run metadata.

## Benchmark and Experiment Protocol

### Primary benchmark

- Dataset: VisA.
- Categories: `pcb1`, `pcb2`, `pcb3`, and `pcb4`.
- Supervision: normal support images only for the target category.
- Query set: the locked official VisA test split.
- DeepPCB: qualitative or box-level secondary evidence only.

### Repeated few-shot protocol

- Shots: `k ∈ {1, 2, 4}`.
- Support seeds: `4880`, `4881`, `4882`, `4883`, and `4884`.
- Categories: all four PCB categories.
- Every paired method comparison reuses the exact same support set and query
  ordering.
- Primary DINOv2 geometry: `image_size=518`, `patch_size=14`.
- Primary multi-scale configuration: `crop_size=768`,
  `crop_overlap=0.25`, `fusion=max`.
- The support sample IDs, source manifest checksum, git commit, environment, and
  complete arguments are stored with every run.

The existing `k=5`, seed-`4880`, fold-0 results remain historical/milestone
evidence. They are not mixed into the primary paper matrix.

### Required method matrix

1. PatchCore as the conventional non-DINO memory-bank baseline, run under the
   same support protocol.
2. Single-scale DINOv2 heatmap.
3. Multi-scale DINOv2 heatmap.
4. SAM2-only uniform-prompt baseline.
5. Single-scale DINOv2-guided SAM2.
6. Multi-scale DINOv2-guided SAM2.
7. Multi-scale anomaly-consistent SAM2.

If an external baseline cannot be reproduced, the evidence index must record the
exact technical blocker. Reported values from a paper may be cited separately
but may not be placed in the same table as if they used this repository's split.

### Pre-registered ablations

- single-scale versus multi-scale;
- crop size and overlap;
- max versus mean heatmap fusion;
- `k = 1, 2, 4`;
- point-only, box-only, and combined prompts;
- center point versus local anomaly-maximum point;
- SAM2 candidate area cap;
- heatmap, SAM2, intersection, union, and selective fallback masks.

The existing `768 / 0.25 / max` configuration is frozen as the primary method.
The explicitly listed alternatives are reported as ablations rather than used
to replace the primary result after test evaluation. No best-test-configuration
selection is performed per category.

## Evaluation Design

### Detection and localization metrics

- image AUROC;
- pixel AUROC;
- AUPRO;
- oracle aggregate pixel F1 and IoU, clearly labeled;
- normal-calibrated aggregate pixel precision, recall, F1, and IoU;
- normal-calibrated mean per-image precision, recall, F1, and IoU;
- anomaly-image-only mean per-image mask metrics.

All methods shown in a metric column must use exactly the same definition,
sample inclusion rule, resolution, and aggregation.

### Statistical reporting

- mean and standard deviation across the five support seeds;
- 95% bootstrap confidence intervals over test images for primary mask metrics;
- paired per-image comparison between multi-scale DINOv2, unconditional SAM2,
  and anomaly-consistent SAM2;
- per-category results plus a macro average over categories;
- no claim of improvement when the relevant confidence interval includes zero
  unless it is explicitly described as inconclusive.

### Failure-stratified analysis

For each anomalous test image, derive:

- ground-truth defect area fraction;
- number of connected components;
- component compactness or thinness proxy;
- heatmap/SAM2 mask IoU;
- SAM2-to-proposal area expansion ratio;
- mean normalized anomaly score inside the SAM2 mask;
- change in F1 and IoU caused by SAM2.

The final analysis must show where SAM2 helps and hurts rather than reporting
only global means. At minimum, include small-versus-large defect strata and four
category-level comparisons.

## Software Architecture

### Reusable modules

- `src/evaluation/`: comparable aggregate/per-image metrics, calibration,
  confidence intervals, and paired analysis.
- `src/sam_refine/`: prompt variants, SAM2 candidate selection, and
  anomaly-consistency fusion.
- `src/features/`: cacheable query/support feature extraction with explicit
  cache keys.
- `src/anomaly/`: memory-bank scoring and multi-scale heatmap generation.
- `src/experiments/`: run specifications, provenance records, matrix expansion,
  and completion validation.

### Runnable scripts

- a tiny end-to-end smoke command;
- a validation-calibration command;
- a single experiment-run command;
- an experiment-matrix launcher suitable for AutoDL;
- a matrix completion checker;
- a result-table generator;
- a statistical/error-analysis generator;
- a paper-asset curation command.

The implementation plan will assign exact filenames, interfaces, tests, and
commands to these responsibilities.

### Artifact policy

Raw data, model weights, feature caches, heatmaps, masks, logs, and rendered
figures remain ignored. Compact, non-private evidence summaries may be tracked
under `docs/evidence/` when they contain no raw dataset material. Each tracked
summary must include enough provenance to identify the ignored source runs.

Repeated experiment runs should not render a debug PNG for every image. The
runner must support a bounded debug sample while retaining the compact data
needed for metrics and curated figures.

## Testing and Verification

### Unit and integration coverage

- calibration uses only rows from the requested validation split;
- oracle and calibrated metric names cannot be confused;
- aggregate and per-image mask metrics match hand-computed fixtures;
- fusion operations preserve shape and binary semantics;
- anomaly-maximum points lie inside their originating prompt component;
- paired experiment records use identical support IDs and test rows;
- run metadata and cache keys change when any result-affecting input changes;
- the matrix checker detects missing, duplicate, failed, and stale runs;
- synthetic tiny-subset smoke tests exercise the full command path.

### Reproducibility verification

- fresh Python 3.10-3.12 environment setup succeeds;
- the full unit suite and lint checks pass;
- a tiny color-patch/fallback-refiner run succeeds without model downloads;
- a tiny real DINOv2/SAM2 run succeeds with documented local paths;
- two identical smoke runs produce identical support IDs and metrics;
- public-repository hygiene finds no data, weights, outputs, secrets, or private
  course files staged for commit.

## Paper-Readiness Deliverables

The project is ready to start writing only when all of the following exist and
have been checked:

1. a frozen research question and defensible novelty statement;
2. a completed primary experiment matrix with no missing runs;
3. a fair external baseline or a documented reproduction blocker;
4. unified quantitative tables with provenance;
5. seed variation, confidence intervals, and paired comparisons;
6. all pre-registered ablations;
7. success and failure figures for every category;
8. a pipeline/method figure;
9. failure-stratified analysis explaining SAM2 behavior;
10. a related-work evidence and citation list covering DINOv2, AnomalyDINO,
    PatchCore, SAM/SAM2, VisA, DeepPCB, recent few-shot industrial segmentation,
    and recent work on segmentation-foundation-model limits;
11. exact local and AutoDL reproduction commands;
12. a top-to-bottom notebook that presents results but delegates implementation
    to `src/` and `scripts/`;
13. an arXiv evidence index mapping every planned claim/table/figure to its
    source run and generation command;
14. passing tests, clean public-repository hygiene, Apache-2.0 licensing, and an
    explicit limitations section outline.

## Decision Rules

### If anomaly-consistent SAM2 improves robustly

Lead with the selective-refinement method. Report unconditional SAM2 as the
failure-prone ablation and show that anomaly consistency recovers precision.

### If improvements are mixed or statistically inconclusive

Lead with the conditional empirical finding. State which categories and defect
geometries benefit and avoid a universal improvement claim.

### If all SAM2 variants remain inferior

Do not hide or endlessly tune the result. Reframe the report as a controlled
negative study of SAM2 for sparse PCB anomalies, with multi-scale DINOv2 as the
strong practical baseline and the failure analysis as the main contribution.

In every outcome, the final title, abstract claims, and conclusion must follow
the frozen evidence rather than the initial project title.

## Explicit Non-Goals

- no CLIP component;
- no large supervised segmentation model;
- no test-mask-driven method selection;
- no claim that a course-scale study establishes state of the art across all
  industrial anomaly datasets;
- no migration of core pipeline logic into the notebook;
- no raw data, weights, checkpoints, outputs, runs, or private reports committed
  to the public repository.
