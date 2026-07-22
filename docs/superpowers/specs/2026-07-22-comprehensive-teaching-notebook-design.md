# Comprehensive Teaching Notebook Design

**Date:** 2026-07-22

**Project:** Few-Shot PCB Defect Segmentation

**Selected direction:** Layered, self-contained classroom notebook with optional evidence-verification cells

## Objective

Replace the current compact evidence-handoff notebook with a complete teaching
artifact that lets the project author understand and present the entire project
without opening another explanatory document. A reader should be able to move
from the industrial problem through the mathematical method, implementation
architecture, experiment protocol, frozen results, limitations, and likely
classroom questions in one top-to-bottom reading.

The notebook remains a presentation and explanation layer. Core behavior stays
in `src/`, runnable experiments stay in `scripts/`, and frozen results stay in
`docs/evidence/generated/`. The notebook explains those artifacts accurately
and links to them for traceability; it does not duplicate the pipeline into an
untested second implementation.

## Intended Reader and Success Standard

The primary reader is an undergraduate computer-vision student who understands
basic image classification and segmentation but may not already understand
DINOv2, few-shot normal-only anomaly detection, PatchCore, SAM2 prompting, or
repeated-measures confidence intervals.

After reading the notebook, the reader must be able to:

1. state the problem, hypothesis, contribution, and final answer in plain language;
2. explain why the method is training-free and what is computed per category;
3. trace one image from dataset record to final binary mask;
4. derive the memory-bank anomaly score and final mask-fusion rule;
5. explain tensor shapes and coordinate transformations at each pipeline stage;
6. distinguish VisA segmentation evidence from the DeepPCB box-only boundary;
7. reconstruct the primary and ablation experiment matrices;
8. define every reported metric and distinguish calibrated from oracle results;
9. interpret the final confidence intervals without overstating the evidence;
10. defend novelty, limitations, failure cases, and reproducibility under Q&A;
11. deliver a detailed classroom presentation using the notebook alone; and
12. locate the implementation and reproduction command for every major step.

## Selected Approach

The notebook will use a layered hybrid structure:

- **Readable without execution:** all essential explanations, equations, frozen
  result tables, conclusions, commands, and Q&A answers appear in Markdown.
- **Optionally executable:** small standard-library cells verify checksums and
  print additional evidence details. They do not require a GPU, datasets,
  PyTorch, DINOv2 weights, or a SAM2 checkpoint.
- **Evidence-bound:** numerical claims come from the frozen public evidence
  package. The notebook's first executable gate checks the completion manifest
  and SHA-256 hashes before any optional evidence cells are trusted.
- **Implementation-linked:** concise pseudocode and essential source excerpts
  explain behavior, while exact paths identify the authoritative modules.
- **Presentation-oriented:** chapter summaries, speaker cues, a slide outline,
  a glossary, and a substantial anticipated-Q&A bank turn understanding into a
  usable class presentation.

This is preferred over a heavy runnable tutorial because a full DINOv2/SAM2 run
requires external data, weights, GPU resources, and substantial time. It is also
preferred over a slide-style notebook because compressed slides would omit the
mathematical and methodological depth needed for Q&A.

## Notebook Architecture

### Part I: Orientation

1. **Title, purpose, and how to use the notebook**
   - Identify the file as the single teaching walkthrough.
   - Separate reading cells from optional verification cells.
   - Provide a one-minute project summary and a chapter map.

2. **Executive story**
   - Define the industrial inspection problem.
   - State the research question and frozen answer.
   - Explain the contribution and non-contributions.
   - Provide a compact end-to-end pipeline table.

3. **Conceptual background**
   - Contrast supervised segmentation, few-shot learning, and normal-only
     anomaly detection.
   - Explain frozen features and why no optimizer, loss, or epoch count exists.
   - Introduce Vision Transformer patch tokens, DINOv2, PatchCore, and SAM2.
   - Explain why a promptable segmenter is not automatically anomaly-aware.

### Part II: Data and Software Foundations

4. **Datasets and benchmark boundary**
   - Explain VisA `pcb1`-`pcb4`, image labels, and pixel masks.
   - Explain DeepPCB boxes and why they are not segmentation ground truth.
   - Describe manifests, fold 0, normal support, normal validation, and the
     official test set.
   - Disclose prior test-set exposure and repeated-test-image limitations.

5. **Few-shot protocol**
   - Define `k in {1, 2, 4}` and seeds `4880`-`4884`.
   - Explain deterministic support sampling and identical paired test rows.
   - Show what information is and is not available at inference time.

6. **Repository map and artifact flow**
   - Map each pipeline stage to `src/`, `scripts/`, `configs/`, and evidence.
   - Distinguish source code, heavy ignored outputs, compact tracked evidence,
     and the notebook presentation layer.
   - Trace provenance from manifest/config/source revision to a result row.

### Part III: Method, Step by Step

7. **Image preparation and alignment**
   - Explain RGB conversion, aspect-ratio-preserving resize, square padding,
     `content_box`, valid patch centers, and nearest-neighbor mask resizing.
   - Show why the same geometry must be used for images, heatmaps, and masks.

8. **DINOv2 patch-feature extraction**
   - Explain ViT-S/14, ImageNet normalization, the `518 x 518` input, patch size
     14, the `37 x 37` patch grid, and normalized patch-token embeddings.
   - Show the symbolic tensor path from image to `[H_p, W_p, D]` features.

9. **Normal memory bank and anomaly score**
   - Explain stacking valid support patches and L2 normalization.
   - Define nearest-normal Euclidean distance for every query patch.
   - Explain chunked computation and conversion from patch scores to an image
     heatmap.

10. **PatchCore-style baseline**
    - Explain Wide-ResNet layer 2/layer 3 features, alignment, pooling, random
      projection, and deterministic approximate-greedy coreset selection.
    - State that this is the repository's reproduction, not reference code.

11. **Multi-scale anomaly proposals**
    - Explain global inference, overlapping `768`-pixel crops, 0.25 overlap,
      local-score placement, coverage accounting, and max fusion.
    - Explain exact compact heatmap archives and why compact storage does not
      change evaluation resolution or numerical values.
    - State that multi-scale improvement over single-scale is inconclusive.

12. **Normal-only calibration**
    - Define the validation-pixel quantile threshold `q=0.995`.
    - Explain the deployment interpretation and separation from test-optimal
      oracle thresholds.
    - Show how the calibrated heatmap becomes binary proposal `A`.

13. **Heatmap-to-prompt conversion**
    - Explain connected components, boxes, and positive points at local anomaly
      maxima.
    - Describe `point`, `box`, and `point_box` modes and the center-point
      ablation.

14. **SAM2 refinement and candidate selection**
    - Explain SAM2.1 Hiera Tiny as a frozen promptable segmentation model.
    - Trace prompt coordinates into image space and masks back into heatmap
      space.
    - Explain the candidate score factors: SAM2 confidence, anomaly strength,
      prompt containment, area penalty, and the 0.25 area cap.

15. **Anomaly-consistent fusion**
    - Define anomaly mask `A`, guided SAM2 mask `S`, and primary output
      `M = A intersection S`.
    - Explain why intersection prevents SAM2 from introducing unsupported
      normal pixels while allowing boundary cleanup.
    - Cover anomaly-only, SAM2-only, union, and selective-fallback alternatives.

16. **One-image trace**
    - Present a numbered, shape-aware trace from manifest row to final mask.
    - Point to one tracked qualitative panel to ground each abstract stage.

### Part IV: Experiments and Evidence

17. **Primary experiment matrix**
    - List all seven methods and the fixed configs.
    - Derive the 364 primary runs, explicitly explaining that `sam2_only` is a
      deterministic category-level baseline rather than a support-dependent run.
    - Explain dependency reuse, feature caching, frozen revisions, and AutoDL.

18. **Ablation matrix**
    - List all 12 variants across four categories at `k=4`, seed `4880`.
    - Derive the 48 runs and explain why these results are descriptive.

19. **Metrics**
    - Define precision, recall, F1, IoU, image AUROC, pixel AUROC, and AUPRO.
    - Explain aggregate-pixel versus mean-per-image aggregation.
    - Explain anomaly-image inclusion and the normal-image boundary.

20. **Statistical design**
    - Explain paired candidate-minus-baseline deltas.
    - Explain fixed category/shot strata and joint resampling of support seeds
      and test-image clusters.
    - Explain what a 95% confidence interval does and does not prove.

21. **Frozen quantitative results**
    - Show compact primary method/shot tables in Markdown.
    - Show the primary F1 and IoU improvements with confidence intervals.
    - Show supporting PatchCore, single-/multi-scale, and guided-SAM2 comparisons.
    - Label inconclusive comparisons explicitly.

22. **Ablations and failure analysis**
    - Summarize prompt, crop, fusion, and area-cap variants.
    - Explain thinness and area strata.
    - Interpret why thin defects benefit more and why large-area evidence remains
      less decisive.

23. **Qualitative evidence**
    - Retain one deterministic success and failure case per PCB category.
    - Teach how to read input, ground truth, anomaly score, guided SAM2, and
      anomaly-consistent columns.
    - State that selected examples illustrate behavior but do not prove averages.

### Part V: Interpretation and Presentation

24. **Research answer and novelty boundary**
    - Center the defensible contribution on the controlled PCB protocol,
      anomaly-to-prompt bridge, and exact anomaly-consistency constraint.
    - Reject new-backbone, first-method, and state-of-the-art claims.

25. **Limitations, threats to validity, and future work**
    - Cover fold-0 scope, shared test images, prior test exposure, four PCB
      categories, frozen foundation-model dependence, and one-seed ablations.
    - Separate evidence-supported future work from changes that would invalidate
      the frozen study.

26. **Class presentation guide**
    - Provide a 10-12 slide sequence, key message per slide, and suggested
      transitions.
    - Include a short opening, a concise method explanation, and a closing claim.

27. **Anticipated Q&A bank**
    - Include at least 25 concrete questions and evidence-grounded answers.
    - Cover training, supervision, model choice, tensor flow, prompts, calibration,
      baselines, metrics, statistics, results, novelty, limitations, compute,
      reproducibility, and next steps.

28. **Glossary, symbol table, and final cheat sheet**
    - Define all essential acronyms and variables.
    - End with the facts and numbers most important to remember before class.

29. **Reproduction appendix**
    - Include environment setup, offline smoke, primary/ablation matrix, checker,
      analysis, and evidence-finalization commands.
    - Clearly label local CPU-safe commands versus AutoDL GPU commands.

## Evidence and Accuracy Rules

1. Frozen numerical claims must come from `docs/evidence/generated/`.
2. The completion manifest must report `ready_for_writing: true`, scope
   `complete_paper_evidence`, 412 source runs, 364 primary runs, and 48 ablations.
3. The notebook must not silently recompute or alter frozen results.
4. Oracle metrics must always be labeled diagnostic and test-optimal.
5. Five support seeds must never be described as five folds.
6. DeepPCB boxes must never be described as segmentation masks.
7. The project must be described as training-free; no loss, optimizer, or epochs
   are invented.
8. Multi-scale and guided-scale comparisons must remain inconclusive where their
   confidence intervals cross zero.
9. The final robust claim is anomaly-consistent SAM2 versus calibrated
   multi-scale DINOv2 on anomalous VisA PCB test images.
10. Qualitative examples must not be presented as aggregate proof.

## Notebook Construction and Maintainability

A deterministic standard-library builder under `scripts/` will define the cell
sequence and write valid `nbformat 4.5` JSON. Keeping the long narrative as
structured Python strings makes cell boundaries, identifiers, and regeneration
reviewable while avoiding fragile hand-edited JSON.

The generated notebook will:

- contain stable cell IDs;
- contain no execution counts or stored code outputs;
- use only repository-relative paths;
- embed all essential tables and explanations as Markdown;
- link tracked method and qualitative figures;
- include lightweight code cells that execute top-to-bottom with the Python
  standard library; and
- fail closed when the frozen evidence package is missing or inconsistent.

The builder is a maintenance tool, not pipeline logic. The notebook remains the
artifact the student reads.

## Error Handling

- Missing repository root: raise a clear instruction to run from inside the repo.
- Missing evidence file: identify the exact missing relative path.
- Failed checksum: stop before displaying optional computed summaries.
- Non-writing-ready manifest: state the failed evidence gate.
- Missing image asset: the notebook test fails with the unresolved path.
- Builder drift: a `--check` mode exits nonzero when the committed notebook no
  longer matches deterministic generation.

## Testing Strategy

### Structural teaching-contract tests

Tests will verify:

- all required chapters and concepts exist;
- all seven methods, twelve ablations, core equations, source paths, and frozen
  conclusions appear;
- the Markdown narrative has sufficient depth rather than headings alone;
- the notebook includes at least 25 Q&A items and a presentation outline;
- every linked repository asset exists;
- the notebook contains no stored outputs or execution counts; and
- generated notebook JSON matches the builder's deterministic output.

These tests are written first and observed failing against the current compact
notebook.

### Execution and evidence tests

The existing top-to-bottom execution test remains. It must verify the completion
manifest, file checksums, matrix counts, method invariants, and frozen outcome
using only compact tracked evidence.

### Human-readable QA

Automated checks cannot prove teaching quality. Final review therefore includes:

- extraction and top-to-bottom reading of every Markdown cell;
- inspection of the notebook's heading order and cell boundaries;
- visual inspection of the method diagram and representative success/failure
  panels;
- a requirement-by-requirement content audit against this design; and
- a mock Q&A pass using only facts present in the notebook.

## Completion Criteria

The task is complete only when:

1. the committed notebook is deterministically generated and valid JSON;
2. every architecture chapter in this design is present and substantive;
3. a reader does not need README, PRD, source, or evidence files for conceptual
   understanding, even though paths remain available for traceability;
4. all essential numerical tables and interpretations are visible without
   running cells;
5. optional cells execute top-to-bottom without heavy dependencies;
6. the presentation outline and at least 25 Q&A answers are included;
7. notebook-specific tests pass;
8. the full local test suite passes;
9. the builder `--check`, JSON validation, and diff checks pass;
10. visual assets referenced by the notebook are legible; and
11. the final audit finds no unsupported claims, missing required sections, or
    unresolved reader dependencies.
