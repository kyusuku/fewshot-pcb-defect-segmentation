# Evaluation

Metrics for saved anomaly heatmaps and predicted masks.

- `metrics.py`: pure NumPy image AUROC, pixel AUROC, best threshold F1/IoU, and
  score-row helpers used by `scripts/evaluate_heatmaps.py`.
- `summary.py`: aggregate per-category `metrics.json` files into CSV and
  Markdown result tables.
