# Dataset Loading Contract

The first code milestone standardizes VisA and DeepPCB samples into one dictionary shape so later DINOv2, SAM2, and evaluation code do not need dataset-specific path logic.

## Common Fields

Every loader keeps a list of `DatasetRecord` objects with:

- `dataset`: `visa_pcb` or `deeppcb`
- `category`: VisA PCB category or DeepPCB group/class context
- `sample_id`: stable readable ID
- `split`: `train`, `val`, `test`, or `all`
- `image_path`: query image path
- `label`: `0` for normal and `1` for anomalous/defective
- `mask_path`: VisA ground-truth mask path when available
- `box_path`: DeepPCB annotation text path when available
- `template_path`: DeepPCB aligned template image path when available
- `metadata`: dataset-specific extras

`__getitem__` returns the record metadata plus loaded image tensors, binary masks for VisA, and box lists for DeepPCB.

## VisA PCB

The loader supports two modes.

1. **Manifest mode**: set `manifest_csv` to the official or generated CSV. The parser accepts common column names such as `image_path`, `mask_path`, `category`, `split`, and `label`.
2. **Directory scan mode**: leave `manifest_csv: null`. The loader scans under `root`, filters categories to `pcb1`-`pcb4`, skips mask-like directories for image discovery, and indexes mask-like directories by image stem.

Normal VisA samples receive an all-zero mask at load time. Anomalous samples convert every mask pixel above `mask_threshold` to binary foreground.

## DeepPCB

The loader scans for tested images whose names contain `_test`, `test`, or `tested`. It searches next to each tested image for:

- matching template image names using `_temp`, `_template`, or `template`
- matching annotation text file with the same stem

Annotation lines are parsed as:

```text
x1,y1,x2,y2,type
```

Class IDs use the DeepPCB convention:

| ID | Name |
| --- | --- |
| 1 | open |
| 2 | short |
| 3 | mousebite |
| 4 | spur |
| 5 | spurious_copper |
| 6 | pin_hole |

DeepPCB boxes are localization annotations, not pixel-accurate segmentation masks. The loader can generate rectangular pseudo-masks for prompts or visualization, but these should not be reported as segmentation ground truth.

## Debug Visualization

Run `scripts/debug_dataset.py` with a config and dataset root. The script saves one PNG per sample with:

- RGB query image
- VisA binary mask overlay when available
- DeepPCB boxes and class labels
- DeepPCB template image when available
