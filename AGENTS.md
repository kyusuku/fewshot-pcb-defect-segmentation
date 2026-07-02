# AGENTS.md

## Project

ECE4880J Computer Vision final project:
Few-Shot PCB Defect Segmentation via Multi-Scale DINOv2 Anomaly Proposals and SAM2 Mask Refinement.

Public repository name: `fewshot-pcb-defect-segmentation`.

## Coding style

- Use Python and PyTorch.
- Keep the code modular.
- Prefer clear research code over production abstractions.
- Avoid putting the whole pipeline in notebooks.
- Use `src/` for reusable code and `scripts/` for runnable experiments.
- Use YAML configs under `configs/`.
- Add comments where the math or tensor shapes may be confusing.
- Keep implementation choices simple until the minimal pipeline runs end to end.

## Important directories

- `src/datasets/`: dataset loaders for VisA and DeepPCB.
- `src/features/`: DINOv2 feature extraction.
- `src/anomaly/`: memory bank and anomaly scoring.
- `src/sam_refine/`: SAM2 prompt generation and mask refinement.
- `src/evaluation/`: AUROC, AUPRO, F1, IoU, visualization.
- `src/utils/`: config, image, plotting, and shared helper code.
- `configs/`: YAML configs.
- `scripts/`: runnable training/evaluation/debug scripts.
- `notebooks/`: exploration only, never the core pipeline.

## Method constraints

- Main benchmark is VisA PCB subsets.
- DeepPCB is secondary because it has bounding boxes, not true segmentation masks.
- Do not add CLIP unless explicitly requested.
- First build a minimal working version.
- Always support few-shot normal sampling with k normal images per category.
- Preserve image/mask alignment during resizing.
- Every script should run on a tiny subset.
- Add visual debug outputs early: image, mask/box, heatmap overlay, and final predicted mask.

## Public repo constraints

- License project code under Apache-2.0.
- Do not commit raw datasets, pretrained weights, checkpoints, outputs, runs, reports, private course files, API keys, or `.env` files.
- Keep placeholder paths in configs and document how to override them locally.
- Do not add CLIP unless explicitly requested later.

## Done criteria

Before saying a task is done:

- Code should run on a tiny debug subset.
- Include the exact command used to test it.
- Report any assumptions about dataset paths.
- Mention any missing dependencies or files.
- Summarize files created or changed.
- Say what should be implemented next.
