# Few-Shot PCB Defect Segmentation

Research codebase for **Few-Shot PCB Defect Segmentation via Multi-Scale DINOv2 Anomaly Proposals and SAM2 Mask Refinement**.

Repository name: `fewshot-pcb-defect-segmentation`

This project implements a few-shot PCB defect detection and segmentation pipeline. The intended method learns normal PCB appearance from a small number of normal images, detects anomalous regions with DINOv2 patch features, improves small-defect localization with multi-scale crops, and refines candidate regions into masks with SAM2.

Current status: dataset loaders, official split manifests, and a first DINOv2-only
anomaly heatmap baseline script.

## Benchmarks

- **VisA PCB subsets**: `pcb1`, `pcb2`, `pcb3`, `pcb4`, with image labels and pixel masks.
- **DeepPCB**: aligned template/test PCB pairs with box annotations.

VisA is the main segmentation benchmark. DeepPCB is secondary and should not be treated as pixel-level segmentation ground truth because it provides boxes, not masks.

## Planned Pipeline

1. Sample k normal VisA images per PCB category.
2. Extract DINOv2 patch features.
3. Build a normal feature memory bank.
4. Score test patches by nearest normal-feature distance.
5. Convert patch scores to anomaly heatmaps.
6. Add multi-scale crop inference and fuse heatmaps.
7. Convert anomaly maps to SAM2 prompts.
8. Refine and filter SAM2 masks.
9. Evaluate VisA with AUROC, AUPRO, F1, IoU, and qualitative visualizations.
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

Use the color-patch backend for a fast smoke test without downloading weights:

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
  --output-dir outputs/dinov2_color_patch_smoke
```

For the real DINOv2 baseline, use a PyTorch-compatible Python environment and
run with `--feature-backbone dinov2_vits14`. The first run downloads model
weights through PyTorch Hub.

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
