# Project Context

## Title

Few-Shot PCB Defect Segmentation via Multi-Scale DINOv2 Anomaly Proposals and SAM2 Mask Refinement

## Repository

`fewshot-pcb-defect-segmentation`

## Goal

Build a clean, reproducible research codebase for few-shot PCB defect detection and segmentation. The system learns normal PCB appearance from a small number of normal images, detects anomalous regions using DINOv2 patch features, improves small-defect localization with multi-scale crops, and refines candidate regions into masks using SAM2.

## Datasets

### VisA PCB Subsets

VisA PCB subsets are the main benchmark. Use PCB-like categories only, not the full VisA dataset. VisA provides normal and anomalous images, image-level labels, and pixel-level anomaly masks. Use it for image AUROC, pixel AUROC, AUPRO, F1, IoU, and qualitative mask visualizations.

### DeepPCB

DeepPCB is the secondary PCB-specific benchmark. It contains aligned template/test image pairs, 640 x 640 crops, six defect types, bounding boxes, and class IDs. Use it for localization, reference-based comparison, and qualitative SAM2 refinement. Do not treat DeepPCB boxes as true segmentation masks except as coarse pseudo-masks when explicitly needed.

The six DeepPCB defect types are open, short, mousebite, spur, pin hole, and spurious copper.

## Method

1. Load VisA and DeepPCB datasets.
2. Sample k normal images per category for few-shot training.
3. Extract DINOv2 patch features from normal images.
4. Build a normal feature memory bank.
5. Extract DINOv2 features from test images.
6. Compute anomaly scores using distance to the normal memory bank.
7. Convert patch anomaly scores into heatmaps.
8. Add multi-scale inference using global image and local crops.
9. Fuse multi-scale anomaly maps.
10. Convert anomaly maps into SAM2 prompts such as points, boxes, or candidate regions.
11. Use SAM2 to refine masks.
12. Score, refine, and filter masks using anomaly strength, overlap, size/shape, and stability.
13. Evaluate against ground-truth masks on VisA.
14. Evaluate DeepPCB using boxes or coarse pseudo-masks only where appropriate.

## Baselines

- DINOv2-only anomaly heatmap.
- SAM2-only or simple prompt baseline.
- Single-scale DINOv2 + SAM2.
- Proposed multi-scale DINOv2 + SAM2.

## Constraints

- Start with a minimal working pipeline before optimizing.
- Do not implement CLIP unless explicitly requested later.
- Use PyTorch.
- Use modular Python files under `src/`.
- Use YAML configs under `configs/`.
- Use runnable scripts under `scripts/`.
- Every script should be testable on a tiny subset.
- Add visual debug outputs early: image, mask/box, heatmap overlay, and final predicted mask.
- Preserve image/mask alignment during resizing.
- Clearly separate code from datasets and model weights.
- Do not commit raw datasets, pretrained weights, outputs, reports, private course files, or API keys.
- Use Apache-2.0 for public release.
