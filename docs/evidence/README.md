# Paper Evidence Policy

This directory tracks compact, public evidence summaries for the paper.

- Generated CSV, Markdown, and JSON summaries are public evidence.
- Raw images, masks, heatmaps, model weights, checkpoints, and run directories stay under ignored `outputs/` and `artifacts/` paths.
- Every generated summary file must be listed in `completion_manifest.json` with a SHA-256 checksum.
- Evidence generation fails instead of silently omitting incomplete or stale runs.
- Test-optimal metrics are diagnostics and must be labeled with `oracle_*`; primary comparison tables use calibrated heatmap masks or binary model outputs.
