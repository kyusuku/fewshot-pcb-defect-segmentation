# Paper Evidence Policy

This directory tracks compact, public evidence summaries for the paper.

- Generated CSV, Markdown, and JSON summaries are public evidence.
- Raw images, masks, heatmaps, model weights, checkpoints, and run directories stay under ignored `outputs/` and `artifacts/` paths.
- Every generated summary file must be listed in `completion_manifest.json` with a SHA-256 checksum.
- Evidence generation fails instead of silently omitting incomplete or stale runs.
- Test-optimal metrics are diagnostics and must be labeled with `oracle_*`; primary comparison tables use calibrated heatmap masks or binary model outputs.
- `completion_manifest.json` is writing-ready only when it says `ready_for_writing: true`; smoke/test manifests explicitly say false.
- Paired primary-mask intervals include anomalous test images only. They preserve method pairing and use repeated-measures resampling of support seeds and test-image clusters, with category and shot count treated as fixed strata.
- Run-level tables report the mean and sample standard deviation across support seeds, per-category results, and an equal-weight category macro average. Their confidence intervals are explicitly labeled as support-seed intervals.
- The study uses fold 0 only. The five seeds are repeated normal-support samples, not five independent folds, and the report must disclose prior VisA test-set exposure.
- Qualitative and method-figure manifests are checksum-validated before the final completion manifest can become writing-ready.
