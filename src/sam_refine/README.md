# SAM2 Refinement

Candidate prompt generation, SAM2-compatible inference wrappers, and fallback
mask scoring modules.

- `prompts.py`: connected-component point/box prompt extraction from anomaly
  heatmaps.
- `refiner.py`: deterministic fallback refiner plus a lazy SAM2 image-predictor
  adapter for environments with SAM2 installed.

The SAM2 adapter keeps the rest of the repo runnable without SAM2 installed. It
loads SAM2 only when `--refiner sam2` is selected, scales heatmap-space prompts
to the RGB image coordinate frame, sends both a box prompt and positive point
prompt, then resizes returned masks back to the heatmap grid.
