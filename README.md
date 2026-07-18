# Few-Shot PCB Defect Segmentation

Research codebase for **Few-Shot PCB Defect Segmentation via Multi-Scale DINOv2 Anomaly Proposals and SAM2 Mask Refinement**.

Repository name: `fewshot-pcb-defect-segmentation`

This project implements a training-free few-shot PCB defect detection and
segmentation pipeline. It represents normal PCB appearance using frozen features
from a small set of normal images, detects anomalous regions with DINOv2 patch
features, combines global and local anomaly evidence, and refines candidate
regions with frozen SAM2 masks. No model parameters are trained or fine-tuned.

Current status: the frozen AutoDL study and compact evidence package are complete.
The primary checker passed all 364 runs, the ablation checker passed all 48 runs,
and the completion manifest reports `ready_for_writing: true`. The source runs are
frozen at commit `ff4c07208278376b27a4954d560c715f51453c5e`; the evidence
postprocessor is frozen at `45a5f93c556e1c1a7a9e39f535434f9244ea1a7a`.

See [docs/PRD.md](docs/PRD.md) for the staged research plan, exit criteria, and
publishability gates.
See [docs/arxiv_readiness_checklist.md](docs/arxiv_readiness_checklist.md) for
the current arXiv-readiness audit.
See [docs/evidence/generated/evidence_index.md](docs/evidence/generated/evidence_index.md)
for the claim-to-evidence map and
[docs/citation_inventory.md](docs/citation_inventory.md) for the scoped novelty
and citation boundary.

## Benchmarks

- **VisA PCB subsets**: `pcb1`, `pcb2`, `pcb3`, `pcb4`, with image labels and pixel masks.
- **DeepPCB**: aligned template/test PCB pairs with box annotations.

VisA is the main segmentation benchmark. DeepPCB is secondary and should not be treated as pixel-level segmentation ground truth because it provides boxes, not masks.

## Frozen Findings

On anomalous VisA PCB test images, anomaly-consistent SAM2 improves over the
calibrated multi-scale DINOv2 mask by **0.0666 F1** (95% CI
`[0.0538, 0.0788]`) and **0.0565 IoU** (95% CI `[0.0462, 0.0665]`). These
paired intervals resample both normal-support seeds and test-image clusters while
treating category and shot count as fixed strata.

The supporting comparisons delimit the claim:

- Multi-scale versus single-scale DINOv2 is inconclusive: F1 `+0.0045`, 95% CI
  `[-0.0058, 0.0142]`; IoU `+0.0041`, 95% CI `[-0.0030, 0.0109]`.
- Multi-scale DINOv2 improves over the repository's PatchCore-style baseline:
  F1 `+0.0365`, 95% CI `[0.0206, 0.0505]`; IoU `+0.0254`, 95% CI
  `[0.0145, 0.0352]`.
- Multi-scale-guided versus single-scale-guided raw SAM2 is inconclusive: F1
  `+0.0120`, 95% CI `[-0.0052, 0.0289]`; IoU `+0.0127`, 95% CI
  `[-0.0004, 0.0260]`.

The defensible contribution is the controlled PCB protocol, proposal-to-prompt
bridge, and exact anomaly-consistency constraint—not a new backbone, a “first”
claim, or a state-of-the-art claim. The frozen study uses fold 0 with five
repeated normal-support samples; it is not five-fold cross-validation, and prior
VisA test-set exposure must be disclosed.

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
See [docs/autodl_data_setup.md](docs/autodl_data_setup.md) for the no-download AutoDL upload workflow.

Raw datasets, pretrained weights, checkpoints, outputs, private reports, PDFs, and API keys are ignored by `.gitignore`.

## ArXiv Evidence Workflow

The paper-facing protocol uses VisA PCB as the primary segmentation benchmark.
DeepPCB is secondary because it provides boxes, not true masks. CLIP is not part
of this method.

Offline smoke, with no network, GPU, checkpoint, or pre-existing dataset. On a
fresh clone, first create the ignored synthetic fixture and the manifest expected
by the frozen smoke config. If they already exist, skip these two setup commands;
no overwrite or deletion is required.

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
PYTHONPATH=src python scripts/analyze_paper_results.py \
  --config configs/experiments/arxiv_smoke.yaml \
  --output-root outputs/arxiv_smoke \
  --analysis-dir outputs/arxiv_smoke_analysis \
  --device cpu
```

Single primary run example:

```bash
PYTHONPATH=src python scripts/run_experiment_matrix.py \
  --config configs/experiments/arxiv_primary.yaml \
  --output-root outputs/arxiv_primary \
  --feature-cache-dir outputs/.feature_cache \
  --device auto \
  --run-id dinov2_multi__pcb1__fold0__k1__seed4880
```

Full primary matrix on AutoDL:

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
```

Paper-matrix DINOv2 and PatchCore heatmaps are retained as versioned exact
component archives. Each archive stores the global/local patch-score grids,
projection geometry, crop order, and fusion rule; the shared loader reconstructs
the same full-resolution float32 heatmap used by calibration, SAM2 prompting,
evaluation, and curation. This is not quantization or lower-resolution
evaluation. Legacy materialized `.npy` and `.npz` heatmaps remain readable, and
all archives remain covered by per-run SHA-256 provenance.

No cleanup policy is required for the frozen study: datasets and prior outputs
were not deleted. The compact archives make the paper matrix fit while preserving
exact regeneration of dependent heatmaps and masks.

Ablations, analysis, assets, and compact evidence:

```bash
PYTHONPATH=src python scripts/run_experiment_matrix.py \
  --config configs/experiments/arxiv_ablations.yaml \
  --output-root outputs/arxiv_ablations \
  --dependency-root outputs/arxiv_primary \
  --feature-cache-dir outputs/.feature_cache \
  --device cuda \
  --resume
PYTHONPATH=src python scripts/analyze_paper_results.py \
  --config configs/experiments/arxiv_primary.yaml \
  --output-root outputs/arxiv_primary \
  --analysis-dir outputs/arxiv_analysis \
  --feature-cache-dir outputs/.feature_cache \
  --device cuda
PYTHONPATH=src python scripts/curate_paper_assets.py \
  --failure-analysis-csv outputs/arxiv_analysis/per_image_failure_analysis.csv \
  --asset-dir artifacts/paper_assets
PYTHONPATH=src python scripts/render_method_figure.py
PYTHONPATH=src python scripts/build_paper_evidence.py \
  --config configs/experiments/arxiv_primary.yaml \
  --output-root outputs/arxiv_primary \
  --ablation-config configs/experiments/arxiv_ablations.yaml \
  --ablation-output-root outputs/arxiv_ablations \
  --analysis-dir outputs/arxiv_analysis \
  --evidence-dir docs/evidence/generated \
  --dependency-root outputs/arxiv_primary \
  --ablation-dependency-root outputs/arxiv_primary \
  --feature-cache-dir outputs/.feature_cache \
  --qualitative-manifest artifacts/paper_assets/qualitative_manifest.csv \
  --method-figure-layout artifacts/paper_assets/method_figure_layout.json \
  --device cuda
```

Primary paper tables use normal-validation calibrated binary masks
(`normal_q995`) for heatmaps and binary model outputs for SAM2/fused masks.
Oracle/test-optimal metrics are retained only as `oracle_*` diagnostics.

Study-scope limitation: the paper matrix uses `fold_id=0`; it is not a
five-fold cross-validation result. VisA's official test set is locked and is
the same across fold IDs, while fold 0 partitions only normal training images
for support and validation. The five seeds measure sensitivity to the sampled
normal support set; they are not five independent dataset splits. Earlier
development inspected VisA test results, so the report must disclose that the
test set is not an untouched holdout.

## Debug Loaders

Synthetic smoke test without real datasets:

```bash
PYTHONPATH=src python scripts/create_synthetic_data.py --output-dir data/debug_fixture
python scripts/debug_dataset.py \
  --config configs/datasets/visa_pcb.yaml \
  --root data/debug_fixture/VisA \
  --split test \
  --category pcb1 \
  --limit 2
python scripts/debug_dataset.py \
  --config configs/datasets/deeppcb.yaml \
  --root data/debug_fixture/DeepPCB/PCBData \
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

The evaluator reports image AUROC, pixel AUROC, AUPRO, best pixel F1, and best
pixel IoU when both normal and anomalous samples with masks are available. Pixel
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

Summarize Stage 4 heatmap and mask baselines together:

```bash
PYTHONPATH=src python scripts/summarize_stage4.py \
  --metrics-json \
    outputs/dinov2_vits14_pcb1_fold0_full/metrics.json \
    outputs/dinov2_vits14_pcb1_fold0_ms768_o025_max_full/metrics.json \
    outputs/dinov2_vits14_pcb1_fold0_full_sam2/mask_metrics.json \
    outputs/dinov2_vits14_pcb1_fold0_ms768_o025_max_full_sam2/mask_metrics.json \
  --output-csv outputs/summary/stage4_fold0_multiscale_comparison.csv \
  --output-md outputs/summary/stage4_fold0_multiscale_comparison.md
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
git clone --depth 1 https://github.com/facebookresearch/sam2.git external/sam2
SAM2_BUILD_CUDA=0 python -m pip install -e external/sam2
curl -L https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt \
  -o weights/sam2.1_hiera_tiny.pt
```

```bash
PYTHONPATH=src python scripts/run_mask_refinement.py \
  --scores-csv outputs/dinov2_vits14_pcb1_fold0_full/scores.csv \
  --output-dir outputs/dinov2_vits14_pcb1_fold0_full_sam2 \
  --percentile 95 \
  --refiner sam2 \
  --sam2-checkpoint weights/sam2.1_hiera_tiny.pt \
  --sam2-model-config configs/sam2.1/sam2.1_hiera_t.yaml \
  --max-mask-area-fraction 0.25 \
  --device auto
```

For SAM2 mode, connected high-anomaly regions become box prompts plus one
positive point prompt at the region center. Predicted masks are resized back to
the anomaly-heatmap grid before saving and evaluation. When SAM2 returns
multiple candidate masks, the adapter selects the mask using SAM2 confidence,
anomaly strength, prompt containment, and mask area instead of SAM2 confidence
alone. Use `--max-mask-area-fraction` to reject oversized SAM2 candidates during
mask selection; if all candidates exceed the cap, the adapter falls back to the
best normally ranked candidate.

Run the SAM2-only/simple prompt baseline without DINOv2 anomaly guidance:

```bash
PYTHONPATH=src python scripts/run_sam2_baseline.py \
  --manifest data/manifests/visa_pcb_folds.csv \
  --fold-id 0 \
  --category pcb1 \
  --query-fold-split test \
  --limit 8 \
  --prompt-longest-side 128 \
  --grid-size 3 \
  --max-regions 9 \
  --refiner sam2 \
  --sam2-checkpoint weights/sam2.1_hiera_tiny.pt \
  --sam2-model-config configs/sam2.1/sam2.1_hiera_t.yaml \
  --output-dir outputs/sam2_only_pcb1_fold0
```

This baseline deliberately uses fixed image-grid prompts instead of DINOv2
heatmaps. It is a comparison point for measuring how much anomaly-guided
prompting helps over promptable segmentation alone.

Evaluate saved refined masks against VisA ground-truth masks:

```bash
PYTHONPATH=src python scripts/evaluate_masks.py \
  --mask-scores-csv outputs/mask_refinement_smoke/mask_scores.csv \
  --source-scores-csv outputs/dinov2_color_patch_smoke/scores.csv \
  --output-json outputs/mask_refinement_smoke/mask_metrics.json \
  --output-csv outputs/mask_refinement_smoke/mask_metrics.csv
```

New `mask_scores.csv` files include `mask_path` directly. The
`--source-scores-csv` argument is useful for older refinement outputs that need
ground-truth mask paths joined from the original DINOv2 `scores.csv`.

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
