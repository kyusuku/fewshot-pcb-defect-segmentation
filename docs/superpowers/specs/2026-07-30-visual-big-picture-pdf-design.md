# Visual Big Picture PDF Design

**Date:** 2026-07-30  
**Status:** Approved  
**Audience:** A reader who wants to understand the PCB defect-segmentation pipeline before studying implementation details or the final report.

## Goal

Transform the existing 16-step “Big picture” explanation into an A4 landscape visual atlas. The PDF must answer three questions at every stage:

1. What enters this step?
2. What changes here?
3. What is passed to the next step?

The result is a teaching artifact, not a replacement report and not new experimental evidence.

## Deliverables

- A 14-page A4 landscape PDF at `output/visual_big_picture/visual_big_picture.pdf`.
- A readable source document beside the PDF.
- A deterministic renderer for explanatory diagrams and page composition.
- Rendered page previews for visual verification.
- A short provenance note identifying real experiment examples and explanatory illustrations.

The deliverables remain under `output/visual_big_picture/`. Raw datasets, checkpoints, pretrained weights, and run outputs are not copied into the deliverable package.

## Page Sequence

1. The inspection question and complete end-to-end pipeline.
2. DINOv2 and SAM2 as complementary roles.
3. VisA PCB categories and why each category has its own memory.
4. Support, calibration, and test roles; few-shot sampling with k in {1, 2, 4}.
5. RGB conversion, aspect-preserving resize, 518 by 518 canvas, and valid-content mask.
6. Frozen DINOv2 patch extraction and the 37 by 37 feature grid.
7. Category-specific normal memory-bank construction.
8. Normal-only score calibration and the 0.995 quantile threshold.
9. Query-patch nearest-neighbour comparison against normal memory.
10. Global plus overlapping local views and the full-resolution multi-scale heatmap.
11. Thresholded anomaly proposal, connected components, point prompts, and box prompts.
12. Frozen SAM2 refinement, candidate-mask selection, and guided-mask union.
13. Exact anomaly-consistent intersection, including the recall ceiling and a failure case.
14. Post-hoc ground-truth evaluation, reported F1 trend, experiment repetition, and the inspector/outline-artist/safety-rule memory aid.

Every original numbered step appears visually. Two closely related steps may share a page, but each retains a distinct number, input, transformation, and output.

## Representative Example

The guide uses the frozen `pcb1/085` qualitative success case as its main query trace because the verified evidence package already links its input image, ground truth, anomaly score, Guided SAM2 mask, and anomaly-consistent mask. The verified `pcb1/054` failure case illustrates the recall and over-segmentation limitation.

Support and dataset-role pages use local VisA normal samples selected deterministically from the fold-0 manifest. These images are embedded in the PDF only; raw dataset files are not copied into the output package.

## Visual Language

- Blue: normal/reference information and DINOv2 support features.
- Amber: anomaly scores, thresholding, and DINOv2 proposals.
- Teal: SAM2 prompts and Guided SAM2 regions.
- Navy: exact anomaly-consistent agreement and final output.
- White on black: binary masks, matching the frozen qualitative figures.

Each visual is marked as one of:

- **Real experiment example:** copied or cropped from verified project evidence, or rendered directly from an aligned local VisA image/mask pair.
- **Explanatory illustration:** a deterministic diagram of an abstract operation such as feature extraction, memory-bank lookup, calibration, or set intersection.

No generative-image model is used to create PCB defects, masks, heatmaps, or reported outcomes.

## Page Anatomy

Each teaching page contains:

1. A step number and short action title.
2. One dominant visual, with smaller supporting panels only when comparison is essential.
3. Three short text blocks: “What enters,” “What happens,” and “What comes out.”
4. A “What to notice” caption focused on the visual evidence.
5. A source label distinguishing real evidence from explanatory illustration.

Equations remain only where they identify an exact operation. Each equation is immediately translated into one plain-language sentence.

## Scientific Guardrails

- DINOv2 is described as a frozen feature extractor; nearest-neighbour comparison produces anomaly scores.
- SAM2 is described as a frozen promptable segmenter that does not know whether a region is defective.
- The memory bank and threshold are category-specific.
- Calibration uses only held-out normal images.
- Ground truth is hidden from prediction and appears only in clearly labeled evaluation or post-hoc comparison panels.
- Multi-scale inference is not claimed to be reliably better than single-scale inference.
- AC-SAM2 is shown as the exact intersection of the calibrated proposal and Guided SAM2 mask.
- The intersection is explicitly shown as unable to recover pixels missing from either input.
- VisA is the main pixel-segmentation benchmark; DeepPCB remains secondary because its boxes are not true segmentation masks.
- The reported support-size F1 values and run counts must be read from the frozen evidence package or checked against the provided source text before rendering.

## Implementation Boundaries

- Use Python and existing project dependencies for deterministic composition.
- Keep the renderer separate from notebooks.
- Reuse frozen images under `docs/evidence/generated/` and verified numeric files under the same directory.
- Do not alter frozen evidence files.
- Do not run DINOv2, SAM2, or the full experiment matrix; this task is presentation-only.
- Do not modify or stage unrelated user-owned changes.

## Verification

Completion requires:

- The PDF opens successfully and contains exactly 14 pages.
- Every page renders to a nonblank preview image.
- No text, equation, or image is clipped at A4 landscape size.
- All 16 original step numbers appear in order.
- Real examples and explanatory illustrations are labeled consistently.
- Ground truth does not appear as an input to any prediction step.
- Frozen qualitative assets and reported values match their source files.
- A final visual review checks readability at normal zoom and verifies the success and failure examples.

## Out of Scope

- Re-running experiments or changing numerical results.
- Modifying the final project report.
- Creating a slide deck or interactive website.
- Adding CLIP, training DINOv2 or SAM2, or changing the segmentation method.
- Committing raw VisA images or any other prohibited dataset artifact.
