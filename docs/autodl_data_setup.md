# AutoDL Data Setup

AutoDL SSH sessions may not be able to download from AWS, GitHub, or model-hosting
URLs. Keep the public repository lightweight and upload official datasets into
ignored local folders instead.

Do not commit VisA, DeepPCB, checkpoints, generated outputs, or archives to Git.
The repo should contain code, configs, documentation, tests, and tiny synthetic
fixtures only.

## Smoke Test Without Official Data

Fresh clones can still run end-to-end loader and visualization checks using
synthetic data:

```bash
cd /root/autodl-tmp/fewshot-pcb-defect-segmentation
PYTHONPATH=src python scripts/create_synthetic_data.py --output-dir data/debug_fixture
```

Then run the printed debug commands, or use:

```bash
PYTHONPATH=src python scripts/debug_dataset.py \
  --config configs/datasets/visa_pcb.yaml \
  --root data/debug_fixture/VisA \
  --split test \
  --category pcb1 \
  --limit 2 \
  --output-dir outputs/debug/synthetic_visa

PYTHONPATH=src python scripts/debug_dataset.py \
  --config configs/datasets/deeppcb.yaml \
  --root data/debug_fixture/DeepPCB/PCBData \
  --split test \
  --limit 1 \
  --output-dir outputs/debug/synthetic_deeppcb
```

## Recommended Official VisA Route

The most reliable route is to prepare VisA on a machine with normal internet,
then upload the prepared one-class folder to AutoDL.

On the machine with the prepared VisA folder:

```bash
tar -czf VisA_pytorch_1cls.tar.gz -C /path/to/VisA_pytorch 1cls
scp -P <autodl_port> VisA_pytorch_1cls.tar.gz \
  root@connect.westc.seetacloud.com:/root/autodl-tmp/fewshot-pcb-defect-segmentation/data/processed/
```

On AutoDL:

```bash
cd /root/autodl-tmp/fewshot-pcb-defect-segmentation
mkdir -p data/processed/VisA_pytorch
tar -xzf data/processed/VisA_pytorch_1cls.tar.gz -C data/processed/VisA_pytorch
```

The expected final path is:

```text
data/processed/VisA_pytorch/1cls
```

Verify it:

```bash
PYTHONPATH=src python scripts/debug_dataset.py \
  --config configs/datasets/visa_pcb.yaml \
  --root data/processed/VisA_pytorch/1cls \
  --split test \
  --limit 4 \
  --output-dir outputs/debug/visa_pcb
```

## DeepPCB Upload Route

If you have `DeepPCB-master.zip` locally, upload it to AutoDL:

```bash
scp -P <autodl_port> DeepPCB-master.zip \
  root@connect.westc.seetacloud.com:/root/autodl-tmp/fewshot-pcb-defect-segmentation/data/raw/
```

On AutoDL:

```bash
cd /root/autodl-tmp/fewshot-pcb-defect-segmentation
python -m zipfile -e data/raw/DeepPCB-master.zip data/raw
```

The expected final path is:

```text
data/raw/DeepPCB-master/PCBData
```

Verify it:

```bash
PYTHONPATH=src python scripts/debug_dataset.py \
  --config configs/datasets/deeppcb.yaml \
  --root data/raw/DeepPCB-master/PCBData \
  --split test \
  --split-file data/raw/DeepPCB-master/PCBData/test.txt \
  --limit 4 \
  --output-dir outputs/debug/deeppcb
```

## Manifest Generation

After VisA and/or DeepPCB are present:

```bash
PYTHONPATH=src python scripts/create_manifests.py \
  --visa-root data/processed/VisA_pytorch/1cls \
  --deeppcb-root data/raw/DeepPCB-master/PCBData \
  --output-dir data/manifests \
  --val-ratio 0.2 \
  --seed 4880
```

If only VisA is available, omit `--deeppcb-root`. If only DeepPCB is available,
omit `--visa-root`.

## AutoDL Primary Matrix Workflow

Use the ignored AutoDL checkout under `/root/autodl-tmp`:

```bash
cd /root/autodl-tmp/fewshot-pcb-defect-segmentation
python3.11 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e ".[dev]"
```

Check CUDA before launching the matrix:

```bash
python - <<'PY'
import torch
print("cuda_available", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
PY
```

Run long jobs inside `tmux` and always use `--resume`:

```bash
tmux new -s pcb-arxiv
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

Then run ablations against the completed primary outputs:

```bash
PYTHONPATH=src python scripts/run_experiment_matrix.py \
  --config configs/experiments/arxiv_ablations.yaml \
  --output-root outputs/arxiv_ablations \
  --dependency-root outputs/arxiv_primary \
  --feature-cache-dir outputs/.feature_cache \
  --device cuda \
  --resume
PYTHONPATH=src python scripts/check_experiment_matrix.py \
  --config configs/experiments/arxiv_ablations.yaml \
  --output-root outputs/arxiv_ablations \
  --dependency-root outputs/arxiv_primary \
  --feature-cache-dir outputs/.feature_cache \
  --device cuda \
  --output-json outputs/arxiv_ablations/matrix_summary.json
```

Build compact summaries and selected paper assets only:

```bash
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
tar -czf arxiv_compact_evidence.tar.gz docs/evidence/generated artifacts/paper_assets
```

Transfer `arxiv_compact_evidence.tar.gz` back to the local checkout. Do not
archive raw `outputs/`, checkpoints, datasets, feature caches, or heatmaps for
public release.

The final matrix is a fold-0 repeated-support study, not five-fold
cross-validation. Fold 0 partitions normal training images for support and
normal-only calibration; the official VisA test set is unchanged across folds.
The five seeds measure support-sampling variability rather than independent
test sets. The report must also disclose prior development-time inspection of
VisA test results.
