# Paper Evidence Policy

This directory tracks the compact, public evidence package for manuscript
writing. The finalized package is under `generated/`.

## Frozen package status

- `generated/completion_manifest.json` is schema 3 and reports
  `ready_for_writing: true` with scope `complete_paper_evidence`.
- All 364 primary runs and 48 ablation runs passed their matrix checkers with
  zero bad runs.
- The evidence postprocessor verified 12,030 per-image invariants across 60
  paired method runs.
- The 35 files declared by the completion manifest have SHA-256 checksums. The
  manifest itself is the 36th file in the generated directory tree.
- Source results are frozen at `ff4c07208278376b27a4954d560c715f51453c5e`;
  presentation/evidence postprocessing is frozen at
  `45a5f93c556e1c1a7a9e39f535434f9244ea1a7a`.
- No model parameters were trained or fine-tuned.

The frozen decision is
`anomaly_consistent_sam2_improves_robustly`: relative to calibrated multi-scale
DINOv2, F1 changes by `+0.066644` (95% CI `[0.053799, 0.078791]`) and IoU by
`+0.056505` (95% CI `[0.046240, 0.066546]`). See
`generated/evidence_index.md` for the claim-to-file map and
`generated/evidence_conclusion.json` for the exact machine-readable decision.

- Generated CSV, Markdown, JSON, and curated paper PNGs are public evidence.
- Full run directories, uncurated images, masks, model weights, checkpoints, and
  feature caches stay under ignored `outputs/` and `artifacts/` paths.
- Matrix heatmaps remain checker-valid as exact projected-component archives: loaders reconstruct the full-resolution float32 maps without quantization or downsampled evaluation, and legacy materialized formats remain readable.
- Every generated summary file must be listed in `completion_manifest.json` with a SHA-256 checksum.
- Evidence generation fails instead of silently omitting incomplete or stale runs.
- Test-optimal metrics are diagnostics and must be labeled with `oracle_*`; primary comparison tables use calibrated heatmap masks or binary model outputs.
- `completion_manifest.json` is writing-ready only when it says `ready_for_writing: true`; smoke/test manifests explicitly say false.
- Paired primary-mask intervals include anomalous test images only. They preserve method pairing and use repeated-measures resampling of support seeds and test-image clusters, with category and shot count treated as fixed strata.
- Run-level tables report the mean and sample standard deviation across support seeds, per-category results, and an equal-weight category macro average. Their confidence intervals are explicitly labeled as support-seed intervals.
- The study uses fold 0 only. The five seeds are repeated normal-support samples, not five independent folds, and the report must disclose prior VisA test-set exposure.
- Qualitative and method-figure manifests are checksum-validated before the final completion manifest can become writing-ready.

The exact projected-component archives remove the need for a destructive disk
cleanup policy. They preserve the patch-score grids and projection metadata needed
to reconstruct the full-resolution float32 heatmaps used by all dependent stages.
The frozen run did not delete datasets or prior experiment outputs.

## Verification

From the repository root:

```bash
PYTHONPATH=src venv/bin/python -m pytest tests/test_presentation_notebook.py -q
PYTHONPATH=src venv/bin/python -m pytest tests/test_public_hygiene.py -q
PYTHONPATH=src:. venv/bin/python scripts/finalize_paper_evidence.py --help
```

To independently verify the package checksums:

```bash
PYTHONPATH=src venv/bin/python - <<'PY'
import hashlib
import json
from pathlib import Path

root = Path("docs/evidence/generated")
manifest = json.loads((root / "completion_manifest.json").read_text())
for record in manifest["generated_files"]:
    relative_path = record["path"]
    actual = hashlib.sha256((root / relative_path).read_bytes()).hexdigest()
    assert actual == record["sha256"], relative_path
print(f"verified {len(manifest['generated_files'])} generated files")
PY
```
