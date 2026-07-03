# Notebooks

The final project can include one readable end-to-end notebook that walks
through the full pipeline for presentation and review:

1. load prepared VisA/DeepPCB manifests and CV fold files,
2. sample few-shot normal support images,
3. extract DINOv2 features,
4. build anomaly heatmaps,
5. create SAM2 prompts,
6. refine masks,
7. evaluate and visualize results.

Keep reusable implementation in `src/` and runnable entry points in `scripts/`.
The notebook should call those modules rather than becoming the source of truth.
Exploratory notebooks are fine during development, but they should not replace
tested code paths.
