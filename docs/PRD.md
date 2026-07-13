# Staged Research PRD

## Project

**Title:** Few-Shot PCB Defect Segmentation via Multi-Scale DINOv2 Anomaly
Proposals and SAM2 Mask Refinement

**Repository:** `fewshot-pcb-defect-segmentation`

This document is a staged product/research requirements document for the final
project. It is intentionally practical: each stage should produce runnable code,
debug artifacts, and measurable evidence before the next stage becomes the main
focus.

## Research Objective

Build a reproducible few-shot PCB defect segmentation pipeline that learns normal
PCB appearance from a small support set, proposes anomalous regions with DINOv2
patch features, improves small-defect localization with multi-scale crops, and
uses SAM2 to refine candidate regions into masks.

The project should answer one main research question:

> Can anomaly-guided SAM2 prompting improve few-shot PCB defect segmentation over
> DINOv2 heatmaps alone, especially for small or thin defects?

## Why This Could Be Publishable

The project is not publishable just because it connects DINOv2 and SAM2. A plain
pipeline of strong pretrained models is useful engineering, but not a strong
research contribution by itself.

The project becomes worth a public arXiv-style report if it provides evidence for
a focused claim:

1. **Domain-specific problem:** PCB defects are small, sparse, and visually close
   to normal copper traces, which makes prompt-free segmentation difficult.
2. **Few-shot constraint:** The method uses only a few normal images per category,
   so it targets realistic industrial inspection settings with scarce labels.
3. **Anomaly-guided prompting:** DINOv2 is not only a detector; its anomaly map is
   converted into structured SAM2 prompts. The key idea is that anomaly scores can
   automate the prompt selection that SAM2 otherwise needs from a human or a
   detector.
4. **Multi-scale localization:** Global DINOv2 features can miss small defects.
   Local crops may recover high-resolution anomaly evidence, and the PRD requires
   ablations proving whether this helps.
5. **Careful benchmark boundary:** VisA is evaluated as true segmentation, while
   DeepPCB is treated as box-level localization or pseudo-mask visualization only.
   This avoids overstating DeepPCB boxes as segmentation ground truth.

The strongest paper story would be:

> Foundation segmentation models are powerful but not anomaly-aware, while
> anomaly detectors localize defects but often produce rough masks. A small,
> training-free bridge between DINOv2 anomaly evidence and SAM2 prompts can improve
> mask quality for PCB inspection under few-shot normal-only supervision.

The project is not strong enough for arXiv if the final report only says "we ran
DINOv2, then SAM2." It needs quantitative comparisons, ablations, failure cases,
and a clear explanation of when SAM2 helps or hurts.

## Related Work Positioning

- **DINOv2** provides strong general visual features without task-specific
  finetuning, which makes it a good backbone for few-shot industrial anomaly
  detection. This project should cite the original DINOv2 paper:
  <https://arxiv.org/abs/2304.07193>.
- **AnomalyDINO** already shows that patch similarity over DINOv2 features is a
  strong training-free one/few-shot anomaly baseline. This project should not
  claim DINOv2 nearest-neighbor anomaly scoring as novel by itself:
  <https://arxiv.org/abs/2405.14529>.
- **SAM2** is a promptable segmentation model for images and videos. This project
  should not claim promptable segmentation as novel by itself:
  <https://arxiv.org/abs/2408.00714>.

The intended contribution is the PCB-specific bridge between these ideas:
multi-scale anomaly maps generate prompts for SAM2, and the resulting masks are
evaluated under few-shot normal-only PCB inspection constraints.

## Non-Goals

- Do not implement CLIP unless explicitly added later.
- Do not train a large supervised segmentation model.
- Do not treat DeepPCB bounding boxes as true segmentation masks.
- Do not commit raw datasets, checkpoints, pretrained weights, generated outputs,
  private reports, course-only files, or API keys.
- Do not make the final notebook the source of truth for core logic. The notebook
  should call modular code from `src/`.

## Stage Summary

| Stage | Name | Goal | Current Status | Exit Criteria |
| --- | --- | --- | --- | --- |
| 0 | Repo and Data Foundation | Reproducible public repo, loaders, manifests, debug views | Mostly complete | Real data manifests and debug PNGs verified |
| 1 | DINOv2-Only Baseline | Few-shot memory bank and heatmap scoring | Initial implementation complete | Per-category VisA metrics saved and summarized |
| 2 | Multi-Scale DINOv2 Proposals | Improve small-defect localization with crops | Initial implementation complete | Single-scale vs multi-scale ablation complete |
| 3 | SAM2 Mask Refinement | Refine anomaly proposals into masks | Adapter/fallback implemented | Real SAM2 run works on a small subset |
| 4 | Baseline Comparison | Compare method variants fairly | Frozen primary and ablation matrices implemented; full AutoDL results pending | Matrix checker passes on all primary/ablation runs |
| 5 | Evaluation and Report Assets | Produce final tables, figures, and notebook | Local smoke, analysis, evidence builder, and figure tooling implemented | `docs/evidence/generated/completion_manifest.json` from full matrix |
| 6 | Public Release Polish | Make repo and paper artifacts publication-safe | README, evidence policy, and hygiene tests active | Full arXiv checklist complete |

The current publication-readiness source of truth is
[`docs/arxiv_readiness_checklist.md`](arxiv_readiness_checklist.md). The older
milestone status below is retained as historical project context; final claims
must be derived from the frozen matrix, analysis, evidence manifest, and checklist.

## Stage 0: Repo and Data Foundation

### Purpose

Create the foundation that makes every later experiment reproducible.

### Requirements

- Provide modular Python code under `src/`.
- Keep configs under `configs/`.
- Keep runnable scripts under `scripts/`.
- Provide dataset loaders for VisA PCB subsets and DeepPCB.
- Generate deterministic train/validation/test manifests and five-fold manifests.
- Save visual sanity checks for image, mask, box, and pseudo-mask alignment.
- Keep raw datasets and generated artifacts ignored by Git.

### Exit Criteria

- `scripts/create_manifests.py` runs on official VisA and DeepPCB roots.
- `scripts/debug_dataset.py` saves real VisA and DeepPCB debug images.
- Manifest counts and sample paths are inspected.
- Synthetic fixtures remain only a smoke test, not proof of real data readiness.

## Stage 1: DINOv2-Only Baseline

### Purpose

Establish the simplest strong baseline: DINOv2 patch features plus nearest-normal
feature distance.

### Requirements

- Sample `k` normal images per category from the development fold.
- Extract DINOv2 patch features from support and query images.
- Build a normal feature memory bank.
- Compute anomaly scores by distance to normal features.
- Resize patch scores to image-space heatmaps without breaking mask alignment.
- Save heatmaps, overlays, and `scores.csv`.
- Evaluate image AUROC, pixel AUROC, best pixel F1, and best pixel IoU.

### Exit Criteria

- One full VisA fold runs for `pcb1`, `pcb2`, `pcb3`, and `pcb4`.
- Metrics are saved as per-category `metrics.json` files.
- `scripts/summarize_metrics.py` produces a mean summary table.
- At least one qualitative panel per category is inspected.

## Stage 2: Multi-Scale DINOv2 Proposals

### Purpose

Test whether local crop inference improves localization of small PCB defects.

### Requirements

- Run global image inference.
- Run local crop inference with configurable crop sizes and overlaps.
- Fuse global and local heatmaps using `max` or `mean`.
- Preserve coordinate alignment when inserting crop heatmaps back into the full
  image grid.
- Compare against single-scale DINOv2 under the same support/query split.

### Exit Criteria

- Single-scale and multi-scale metrics are reported on the same fold.
- Ablations include at least one crop size and one overlap setting.
- Qualitative examples show where multi-scale helps and where it creates false
  positives.

## Stage 3: SAM2 Mask Refinement

### Purpose

Convert anomaly heatmaps into promptable segmentation requests and evaluate
whether SAM2 improves mask quality.

### Requirements

- Convert heatmaps into candidate connected regions.
- Generate box prompts and positive point prompts from each region.
- Run deterministic fallback refinement when SAM2 is not installed.
- Run real SAM2 refinement when checkpoint and config are provided.
- Resize SAM2 masks back to the heatmap grid before evaluation.
- Score masks using anomaly strength, SAM2 confidence, area, overlap, and simple
  stability checks where available.

### Exit Criteria

- `scripts/run_mask_refinement.py --refiner fallback` works on smoke outputs.
- `scripts/run_mask_refinement.py --refiner sam2` works on at least one small
  real VisA subset.
- Refined masks are evaluated against VisA ground-truth masks.
- Failure cases are saved, especially over-segmentation and missed tiny defects.

## Stage 4: Baseline Comparison

### Purpose

Make the final claim credible by comparing against fair baselines.

### Required Baselines

1. DINOv2-only anomaly heatmap.
2. SAM2-only or simple prompt baseline.
3. Single-scale DINOv2 + SAM2.
4. Multi-scale DINOv2 + SAM2.

### Metrics

- Image AUROC.
- Pixel AUROC.
- AUPRO.
- Best pixel F1.
- Best pixel IoU.
- DeepPCB box-level localization or coarse pseudo-mask metrics only when clearly
  labeled as such.

### Exit Criteria

- `scripts/run_sam2_baseline.py` generates the SAM2-only/simple prompt baseline.
- All baselines use the same categories, folds, support size, and query split.
- Metrics are summarized per category and as a mean.
- The final method improves at least one meaningful segmentation metric without
  causing unacceptable image-level degradation.
- If the final method does not improve, the report explains why and reframes the
  contribution as an empirical negative result.

## Stage 5: Evaluation and Report Assets

### Purpose

Prepare the evidence needed for Milestone 2, the final report, and an optional
arXiv-style technical report.

### Requirements

- Save tables for each baseline and category.
- Save qualitative figures for success and failure cases.
- Keep a final readable notebook that calls the modular code.
- Include a short method diagram or pipeline figure.
- Keep experiment commands reproducible from README or a dedicated experiments
  document.

### Exit Criteria

- Milestone 2 can report a clear baseline model, evaluation protocol, initial
  results, and improvement plan.
- Final notebook can be read top-to-bottom without hiding core logic.
- Results can be regenerated from committed scripts and ignored local data.

## Stage 6: Public Release Polish

### Purpose

Make the repository safe and useful for public release.

### Requirements

- Keep Apache-2.0 license.
- Keep README setup and dataset instructions accurate.
- Keep `.gitignore` strict for data, outputs, checkpoints, weights, reports,
  notebooks checkpoints, `.env`, and large artifacts.
- Avoid committing private course files or raw model outputs.
- Add clear citations and acknowledgements in the final report or README.

### Exit Criteria

- Fresh clone instructions are coherent.
- Unit tests pass.
- Public repo contains no raw datasets, weights, generated outputs, reports, or
  secrets.
- Final paper/report states limitations honestly.

## Publishability Gates

Before calling this arXiv-worthy, the project should pass these gates:

1. **Novelty gate:** The report clearly identifies anomaly-guided SAM2 prompting
   and multi-scale proposal fusion as the contribution, not the existence of
   DINOv2 or SAM2.
2. **Evidence gate:** Multi-scale DINOv2 + SAM2 is compared against DINOv2-only
   and single-scale DINOv2 + SAM2 on the same split.
3. **Ablation gate:** The report shows which component matters: k-shot support
   size, crop scale, prompt type, mask scoring, or SAM2 refinement.
4. **Failure gate:** The report includes cases where SAM2 worsens masks or where
   anomaly maps produce misleading prompts.
5. **Reproducibility gate:** Dataset acquisition, manifests, commands, configs,
   and metrics are documented well enough for another student to rerun.

## Main Risks

- DINOv2 heatmaps may already be strong enough that SAM2 adds little.
- SAM2 may over-segment PCB traces because it segments objects, not defects.
- Multi-scale crops may increase false positives on repetitive trace patterns.
- DeepPCB boxes may tempt overclaiming segmentation quality.
- CPU-only experiments may limit the number of folds and ablations that can run
  before deadlines.

## Current Next Step

Run real SAM2 on a small VisA PCB subset using saved DINOv2 heatmaps, then
evaluate whether refined masks improve best pixel F1 or IoU over the DINOv2-only
heatmap threshold baseline.
