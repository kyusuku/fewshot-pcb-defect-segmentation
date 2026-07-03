# SAM2 Refinement

Candidate prompt generation, SAM2-compatible inference wrappers, and fallback
mask scoring modules.

- `prompts.py`: connected-component point/box prompt extraction from anomaly
  heatmaps.
- `refiner.py`: deterministic fallback refiner plus a lazy SAM2 image-predictor
  adapter for environments with SAM2 installed.
