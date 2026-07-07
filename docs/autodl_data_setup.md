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
