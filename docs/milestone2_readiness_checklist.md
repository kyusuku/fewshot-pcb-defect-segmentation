# Milestone 2 Readiness Checklist

Last refreshed: 2026-07-08.

This is a codebase readiness checklist and evidence index, not a milestone
report draft. Use it to confirm that the implemented baseline, evaluation
commands, results, qualitative artifacts, and next steps are ready before
writing the report separately.

## Verdict

The repository is ready for the baseline-focused milestone 2 requirements:
runnable baseline implementation, reproducible evaluation commands, initial
quantitative results, qualitative artifact locations, and documented next steps.

The final project is not complete yet. It still needs a full SAM2-only baseline
across all PCB categories, curated qualitative success/failure figures, and more
analysis of why SAM2 refinement sometimes underperforms heatmap thresholding.

## Baseline Model

The milestone baseline is **DINOv2 ViT-S/14 patch-feature nearest-neighbor
anomaly scoring** on the VisA PCB subsets.

- **Inputs:** VisA PCB fold 0 categories `pcb1`, `pcb2`, `pcb3`, and `pcb4`.
- **Few-shot support:** `k=5` normal development images per category.
- **Feature extractor:** pretrained DINOv2 ViT-S/14 through the existing
  `dinov2_vits14` backend.
- **Image geometry:** images resized for DINOv2 patch extraction with
  `image_size=518` and `patch_size=14`.
- **Memory bank:** all support-image patch features are L2-normalized and stored
  as the normal feature memory bank.
- **Anomaly score:** each query patch is scored by distance to its nearest normal
  feature. The image score is the maximum anomaly value over the heatmap.
- **Segmentation output:** patch scores are converted into an image-space
  heatmap. The best threshold over validation/evaluation pixels gives the pixel
  F1 and IoU reported below.

This baseline is relevant because it matches the project setting: only normal
support images are used, no supervised defect-mask training is required, and the
same heatmaps later generate SAM2 prompts for the proposed method.

## Training and Evaluation Procedure

There is no gradient-based training in the baseline. The "training" step is the
deterministic selection of few-shot normal support images and construction of the
normal feature memory bank.

Required training-pipeline fields for the milestone prompt:

- **Loss function:** not applicable. The baseline is training-free and does not
  optimize a supervised or self-supervised objective on the PCB data.
- **Optimizer:** not applicable. No SGD/Adam optimizer, learning rate, momentum,
  or weight decay is used.
- **Training details:** `k=5` normal support images per category, seed `4880`,
  fold `0`, no epochs, no batches, no learning-rate schedule.
- **Support/query split:** normal support images come from the development fold;
  evaluation uses the fold 0 test split.

Evaluation uses the VisA PCB fold 0 test split. Metrics are image AUROC, pixel
AUROC, AUPRO, best pixel F1, and best pixel IoU. Pixel metrics use the existing
deterministic evaluator with at most 1,000,000 sampled pixels per category by
default.

Metric suitability:

- **Image AUROC:** measures image-level defect detection quality without choosing
  a fixed threshold.
- **Pixel AUROC:** measures dense anomaly localization quality.
- **AUPRO:** emphasizes connected-region overlap, which is useful for sparse PCB
  defects.
- **Best pixel F1 and IoU:** summarize thresholded segmentation quality against
  VisA pixel masks.

Full-fold runs were generated from ignored local data and outputs. Public repo
files remain safe: `data/`, `outputs/`, `weights/`, `external/`, dataset
archives, PDFs, and private reports are ignored.

## Reproduction Commands

Use the Python 3.11 environment for local verification. In this checkout,
`venv/` is Python 3.11 with PyTorch available; `.venv/` is Python 3.14 and should
not be used for PyTorch experiments.

Run a full single-scale DINOv2 baseline for one category:

```bash
env PYTHONPATH=src venv/bin/python scripts/run_dinov2_baseline.py \
  --manifest data/manifests/visa_pcb_folds.csv \
  --fold-id 0 \
  --category pcb1 \
  --k 5 \
  --limit 100000 \
  --feature-backbone dinov2_vits14 \
  --image-size 518 \
  --patch-size 14 \
  --output-dir outputs/dinov2_vits14_pcb1_fold0_full
```

Run the multi-scale ablation for the same category:

```bash
env PYTHONPATH=src venv/bin/python scripts/run_dinov2_baseline.py \
  --manifest data/manifests/visa_pcb_folds.csv \
  --fold-id 0 \
  --category pcb1 \
  --k 5 \
  --limit 100000 \
  --feature-backbone dinov2_vits14 \
  --image-size 518 \
  --patch-size 14 \
  --crop-sizes 768 \
  --crop-overlap 0.25 \
  --fusion max \
  --output-dir outputs/dinov2_vits14_pcb1_fold0_ms768_o025_max_full
```

Evaluate a saved heatmap run:

```bash
env PYTHONPATH=src venv/bin/python scripts/evaluate_heatmaps.py \
  --scores-csv outputs/dinov2_vits14_pcb1_fold0_full/scores.csv \
  --output-json outputs/dinov2_vits14_pcb1_fold0_full/metrics.json
```

Summarize the four single-scale categories:

```bash
env PYTHONPATH=src venv/bin/python scripts/summarize_metrics.py \
  --metrics-json \
    outputs/dinov2_vits14_pcb1_fold0_full/metrics.json \
    outputs/dinov2_vits14_pcb2_fold0_full/metrics.json \
    outputs/dinov2_vits14_pcb3_fold0_full/metrics.json \
    outputs/dinov2_vits14_pcb4_fold0_full/metrics.json \
  --output-csv outputs/summary/dinov2_vits14_fold0_summary.csv \
  --output-md outputs/summary/dinov2_vits14_fold0_summary.md
```

Summarize the four-category Stage 4 comparison:

```bash
env PYTHONPATH=src venv/bin/python scripts/summarize_stage4.py \
  --metrics-json \
    outputs/dinov2_vits14_pcb1_fold0_full/metrics.json \
    outputs/dinov2_vits14_pcb1_fold0_full_sam2/mask_metrics.json \
    outputs/dinov2_vits14_pcb1_fold0_ms768_o025_max_full/metrics.json \
    outputs/dinov2_vits14_pcb1_fold0_ms768_o025_max_full_sam2/mask_metrics.json \
    outputs/dinov2_vits14_pcb2_fold0_full/metrics.json \
    outputs/dinov2_vits14_pcb2_fold0_full_sam2/mask_metrics.json \
    outputs/dinov2_vits14_pcb2_fold0_ms768_o025_max_full/metrics.json \
    outputs/dinov2_vits14_pcb2_fold0_ms768_o025_max_full_sam2/mask_metrics.json \
    outputs/dinov2_vits14_pcb3_fold0_full/metrics.json \
    outputs/dinov2_vits14_pcb3_fold0_full_sam2/mask_metrics.json \
    outputs/dinov2_vits14_pcb3_fold0_ms768_o025_max_full/metrics.json \
    outputs/dinov2_vits14_pcb3_fold0_ms768_o025_max_full_sam2/mask_metrics.json \
    outputs/dinov2_vits14_pcb4_fold0_full/metrics.json \
    outputs/dinov2_vits14_pcb4_fold0_full_sam2/mask_metrics.json \
    outputs/dinov2_vits14_pcb4_fold0_ms768_o025_max_full/metrics.json \
    outputs/dinov2_vits14_pcb4_fold0_ms768_o025_max_full_sam2/mask_metrics.json \
  --output-csv outputs/summary/stage4_fold0_multiscale_comparison.csv \
  --output-md outputs/summary/stage4_fold0_multiscale_comparison.md
```

## Initial Results

### Single-Scale DINOv2 Baseline

Source: refreshed local `outputs/summary/dinov2_vits14_fold0_summary.md`.

| category | num_images | image_auroc | pixel_auroc | aupro | best_pixel_f1 | best_pixel_iou |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| pcb1 | 200 | 0.7154 | 0.8766 | 0.6225 | 0.3921 | 0.2439 |
| pcb2 | 200 | 0.6547 | 0.8680 | 0.5198 | 0.1056 | 0.0557 |
| pcb3 | 201 | 0.6737 | 0.9071 | 0.4297 | 0.1883 | 0.1039 |
| pcb4 | 201 | 0.6694 | 0.9022 | 0.5360 | 0.1265 | 0.0675 |
| mean | 802 | 0.6783 | 0.8885 | 0.5270 | 0.2031 | 0.1178 |

### Multi-Scale DINOv2 Ablation

Source: refreshed local `outputs/summary/dinov2_vits14_fold0_ms768_summary.md`.

| category | num_images | image_auroc | pixel_auroc | aupro | best_pixel_f1 | best_pixel_iou |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| pcb1 | 200 | 0.6890 | 0.9915 | 0.9137 | 0.6539 | 0.4857 |
| pcb2 | 200 | 0.7284 | 0.9459 | 0.8295 | 0.2727 | 0.1579 |
| pcb3 | 201 | 0.7336 | 0.9355 | 0.7865 | 0.1982 | 0.1100 |
| pcb4 | 201 | 0.7571 | 0.8799 | 0.6963 | 0.1692 | 0.0924 |
| mean | 802 | 0.7270 | 0.9382 | 0.8065 | 0.3235 | 0.2115 |

The multi-scale heatmap improves mean image AUROC, pixel AUROC, AUPRO, F1, and
IoU over the single-scale heatmap baseline. This is the clearest current
positive result available for the milestone evidence package.

### Early Stage 4 Comparison

Source: refreshed local
`outputs/summary/stage4_fold0_multiscale_comparison.md`.

| category | method | image AUROC | pixel AUROC | AUPRO | mask F1 | mask IoU | precision | recall |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pcb1 | dinov2_single_heatmap | 0.7154 | 0.8766 | 0.6225 | 0.3921 | 0.2439 | - | - |
| pcb1 | single_dinov2_sam2 | - | - | - | 0.1681 | 0.1098 | 0.1546 | 0.4635 |
| pcb1 | dinov2_ms768_heatmap | 0.6890 | 0.9915 | 0.9137 | 0.6539 | 0.4857 | - | - |
| pcb1 | ms768_dinov2_sam2 | - | - | - | 0.2769 | 0.2051 | 0.2242 | 0.6322 |
| pcb2 | dinov2_single_heatmap | 0.6547 | 0.8680 | 0.5198 | 0.1056 | 0.0557 | - | - |
| pcb2 | single_dinov2_sam2 | - | - | - | 0.1273 | 0.0751 | 0.0828 | 0.4100 |
| pcb2 | dinov2_ms768_heatmap | 0.7284 | 0.9459 | 0.8295 | 0.2727 | 0.1579 | - | - |
| pcb2 | ms768_dinov2_sam2 | - | - | - | 0.2588 | 0.1719 | 0.1865 | 0.5948 |
| pcb3 | dinov2_single_heatmap | 0.6737 | 0.9071 | 0.4297 | 0.1883 | 0.1039 | - | - |
| pcb3 | single_dinov2_sam2 | - | - | - | 0.1107 | 0.0744 | 0.0856 | 0.2675 |
| pcb3 | dinov2_ms768_heatmap | 0.7336 | 0.9355 | 0.7865 | 0.1982 | 0.1100 | - | - |
| pcb3 | ms768_dinov2_sam2 | - | - | - | 0.2568 | 0.1930 | 0.2133 | 0.4114 |
| pcb4 | dinov2_single_heatmap | 0.6694 | 0.9022 | 0.5360 | 0.1265 | 0.0675 | - | - |
| pcb4 | single_dinov2_sam2 | - | - | - | 0.1934 | 0.1297 | 0.1607 | 0.3184 |
| pcb4 | dinov2_ms768_heatmap | 0.7571 | 0.8799 | 0.6963 | 0.1692 | 0.0924 | - | - |
| pcb4 | ms768_dinov2_sam2 | - | - | - | 0.2751 | 0.1868 | 0.2747 | 0.3829 |
| mean | dinov2_single_heatmap | 0.6783 | 0.8885 | 0.5270 | 0.2031 | 0.1178 | - | - |
| mean | single_dinov2_sam2 | - | - | - | 0.1499 | 0.0972 | 0.1209 | 0.3649 |
| mean | dinov2_ms768_heatmap | 0.7270 | 0.9382 | 0.8065 | 0.3235 | 0.2115 | - | - |
| mean | ms768_dinov2_sam2 | - | - | - | 0.2669 | 0.1892 | 0.2247 | 0.5053 |

The current SAM2 refinement increases recall but often lowers precision and mask
F1 relative to direct heatmap thresholding. This is useful error-analysis
evidence: multi-scale anomaly maps are strong, while SAM2 prompt/mask selection
still needs refinement.

## Qualitative Evidence

Useful local figure locations for report drafting:

- Single-scale heatmap panels:
  `outputs/dinov2_vits14_pcb1_fold0_full/*.png`
- Multi-scale heatmap panels:
  `outputs/dinov2_vits14_pcb1_fold0_ms768_o025_max_full/*.png`
- Multi-scale DINOv2 + SAM2 panels and masks:
  `outputs/dinov2_vits14_pcb1_fold0_ms768_o025_max_full_sam2/*_refined.png`
  and `*_pred_mask.png`
- SAM2-only smoke examples:
  `outputs/sam2_only_pcb1_fold0_limit2_smoke/*_sam2_only.png`

Before writing the report, choose one success and one failure per category if
possible. The most important failure cases are over-segmented copper traces and
missed thin defects.

## Assumptions and Local Paths

- Main benchmark: VisA PCB subsets only.
- Secondary DeepPCB use remains qualitative or box/pseudo-mask based, not true
  segmentation ground truth.
- Local manifests are under ignored `data/manifests/`.
- Local model/SAM2 files are under ignored `weights/` and `external/`.
- The refreshed tables above come from ignored `outputs/` and are intentionally
  copied here as report evidence, not committed as raw experiment artifacts.
- The Python environment used for verification is `venv/`, not `.venv/`.

## Missing or Incomplete Before Final Report

- Run `scripts/run_sam2_baseline.py` for all four PCB categories, not only the
  current `pcb1` smoke output.
- Curate qualitative success/failure figures and export them into an ignored
  report-assets folder.
- Decide whether final tables should emphasize heatmap threshold masks, SAM2
  refined masks, or both.
- Run final heavy jobs on AutoDL rather than tying up the local laptop.
- Add final citations for DINOv2, AnomalyDINO, SAM2, VisA, and DeepPCB in the
  written report.

## Next Steps

1. Run full SAM2-only baseline jobs on AutoDL for `pcb1` through `pcb4`.
2. Pick representative qualitative figures after comparing the heatmap panels
   against SAM2 refined masks.
3. Iterate on SAM2 mask selection or prompt filtering to recover the heatmap
   F1/IoU gains without losing recall.
4. Add final citations for DINOv2, AnomalyDINO, SAM2, VisA, and DeepPCB in the
   separately written report.
