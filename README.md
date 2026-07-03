# Few-Shot PCB Defect Segmentation

Research codebase for **Few-Shot PCB Defect Segmentation via Multi-Scale DINOv2 Anomaly Proposals and SAM2 Mask Refinement**.

Repository name: `fewshot-pcb-defect-segmentation`

This project implements a few-shot PCB defect detection and segmentation pipeline. The intended method learns normal PCB appearance from a small number of normal images, detects anomalous regions with DINOv2 patch features, improves small-defect localization with multi-scale crops, and refines candidate regions into masks with SAM2.

Current status: dataset loaders, official split manifests, DINOv2-style anomaly
heatmap baseline, optional multi-scale crop fusion, heatmap evaluation, and a
SAM2-compatible mask refinement stage with a deterministic fallback refiner.

See [docs/PRD.md](docs/PRD.md) for the staged research plan, exit criteria, and
publishability gates.

## Benchmarks

- **VisA PCB subsets**: `pcb1`, `pcb2`, `pcb3`, `pcb4`, with image labels and pixel masks.
- **DeepPCB**: aligned template/test PCB pairs with box annotations.

VisA is the main segmentation benchmark. DeepPCB is secondary and should not be treated as pixel-level segmentation ground truth because it provides boxes, not masks.

## Pipeline

1. Sample k normal VisA images per PCB category.
2. Extract DINOv2 patch features.
3. Build a normal feature memory bank.
4. Score test patches by nearest normal-feature distance.
5. Convert patch scores to anomaly heatmaps.
6. Optionally add multi-scale crop inference and fuse heatmaps.
7. Convert anomaly maps to SAM2-style point/box prompts.
8. Refine and filter candidate masks with SAM2 or the fallback refiner.
9. Evaluate VisA with AUROC, F1, IoU, and qualitative visualizations.
10. Evaluate DeepPCB with boxes or coarse pseudo-masks only where appropriate.

## Setup

Use a PyTorch-compatible Python version. The system Python in this folder may be too new for PyTorch, so prefer Python 3.10-3.12.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e ".[dev]"
```

## Expected Data

Keep raw datasets outside git. The default configs use placeholder local paths:

```text
data/
  VisA/
    ...
  DeepPCB/
    PCBData/
      group00041/
        00041/
          00041000_test.jpg
          00041000_temp.jpg
        00041_not/
          00041000.txt
```

See [docs/dataset_layout.md](docs/dataset_layout.md) for supported path conventions and config details.
See [docs/data_acquisition.md](docs/data_acquisition.md) for official download plus train/validation/test and five-fold manifest generation commands.

Raw datasets, pretrained weights, checkpoints, outputs, private reports, PDFs, and API keys are ignored by `.gitignore`.

## Debug Loaders

Synthetic smoke test without real datasets:

```bash
PYTHONPATH=src python -c "from pathlib import Path; from utils.synthetic_data import create_synthetic_debug_datasets; create_synthetic_debug_datasets(Path('/tmp/pcb_debug_fixture'))"
python scripts/debug_dataset.py \
  --config configs/datasets/visa_pcb.yaml \
  --root /tmp/pcb_debug_fixture/VisA \
  --split test \
  --category pcb1 \
  --limit 2
python scripts/debug_dataset.py \
  --config configs/datasets/deeppcb.yaml \
  --root /tmp/pcb_debug_fixture/DeepPCB/PCBData \
  --split test \
  --limit 1
```

VisA:

```bash
python scripts/debug_dataset.py \
  --config configs/datasets/visa_pcb.yaml \
  --root /absolute/path/to/VisA \
  --split test \
  --limit 4
```

DeepPCB:

```bash
python scripts/debug_dataset.py \
  --config configs/datasets/deeppcb.yaml \
  --root /absolute/path/to/DeepPCB/PCBData \
  --split test \
  --split-file /absolute/path/to/DeepPCB/PCBData/test.txt \
  --limit 4
```

Outputs are written to `outputs/debug_dataset/` by default.

## DINOv2 Baseline

Use the color-patch backend for a fast smoke test without downloading weights.
It exercises the same memory-bank, heatmap, multi-scale, and artifact-writing
path as the real DINOv2 backend:

```bash
PYTHONPATH=src python scripts/run_dinov2_baseline.py \
  --manifest data/manifests/visa_pcb_folds.csv \
  --fold-id 0 \
  --category pcb1 \
  --k 2 \
  --limit 2 \
  --feature-backbone color_patch \
  --image-size 56 \
  --patch-size 14 \
  --crop-sizes 128 \
  --crop-overlap 0.25 \
  --output-dir outputs/dinov2_color_patch_smoke
```

For the real DINOv2 baseline, use a PyTorch-compatible Python environment and
run with `--feature-backbone dinov2_vits14`. The first run downloads model
weights through PyTorch Hub.

## Evaluation

Evaluate saved heatmaps from a baseline `scores.csv`:

```bash
PYTHONPATH=src python scripts/evaluate_heatmaps.py \
  --scores-csv outputs/dinov2_color_patch_smoke/scores.csv \
  --output-json outputs/dinov2_color_patch_smoke/metrics.json
```

The evaluator reports image AUROC, pixel AUROC, best pixel F1, and best pixel
IoU when both normal and anomalous samples with masks are available. Pixel
metrics use a deterministic sample of up to 1,000,000 pixels by default for
fast full-fold iteration; pass `--max-pixels 0` for exact full-resolution pixel
metrics.

Summarize multiple category-level metric files:

```bash
PYTHONPATH=src python scripts/summarize_metrics.py \
  --metrics-json outputs/dinov2_vits14_pcb*/metrics.json \
  --output-csv outputs/summary/dinov2_vits14_fold0_summary.csv \
  --output-md outputs/summary/dinov2_vits14_fold0_summary.md
```

## Mask Refinement

Run the current deterministic fallback refiner:

```bash
PYTHONPATH=src python scripts/run_mask_refinement.py \
  --scores-csv outputs/dinov2_color_patch_smoke/scores.csv \
  --output-dir outputs/mask_refinement_smoke \
  --threshold 0.5 \
  --refiner fallback
```

The fallback refiner is a runnable stand-in for debugging prompts and mask
artifacts. Real SAM2 inference requires installing `facebookresearch/sam2`,
downloading a checkpoint into ignored `weights/`, and running with
`--refiner sam2 --sam2-checkpoint ... --sam2-model-config ...`:

```bash
PYTHONPATH=src python scripts/run_mask_refinement.py \
  --scores-csv outputs/dinov2_vits14_pcb1_fold0_full/scores.csv \
  --output-dir outputs/dinov2_vits14_pcb1_fold0_full_sam2 \
  --percentile 95 \
  --refiner sam2 \
  --sam2-checkpoint weights/sam2.1_hiera_tiny.pt \
  --sam2-model-config configs/sam2.1/sam2.1_hiera_t.yaml \
  --device auto
```

For SAM2 mode, connected high-anomaly regions become box prompts plus one
positive point prompt at the region center. Predicted masks are resized back to
the anomaly-heatmap grid before saving and evaluation.

## Repository Layout

```text
configs/          YAML configs
scripts/          runnable debug/train/eval scripts
src/datasets/     VisA and DeepPCB dataset loaders
src/features/     DINOv2 feature extraction
src/anomaly/      memory bank and anomaly heatmaps
src/sam_refine/   SAM2 prompt and mask refinement
src/evaluation/   metrics and qualitative outputs
src/utils/        shared helpers
notebooks/        final readable pipeline notebook plus exploration notes
```

The final notebook should be a readable end-to-end companion that calls the
modular code in `src/`; dataset loaders, model code, metrics, and reusable
utilities should remain outside the notebook.

## License

This project is released under the Apache License 2.0. See [LICENSE](LICENSE).
