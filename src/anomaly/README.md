# Anomaly

Memory-bank construction, nearest-neighbor scoring, and anomaly heatmap helpers.

- `memory_bank.py`: stack normal patch features and score query patches by
  nearest-memory distance.
- `heatmap.py`: normalize, resize, overlay, and save debug heatmap panels.
- `multiscale.py`: score the full image plus optional overlapping crops and
  fuse full-resolution heatmaps with max or mean fusion.
