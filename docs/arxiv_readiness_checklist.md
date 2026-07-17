# ArXiv Readiness Checklist

Current status: all frozen experimental evidence required to begin manuscript
writing is present and checksum-bound. The generated completion manifest reports
`ready_for_writing: true` for 364 primary runs and 48 ablation runs, with zero
checker failures. A final fresh-clone release verification remains before the
repository handoff is declared complete; manuscript drafting is intentionally
outside this repository task.

| Requirement | Status | Authoritative evidence | Verification command or contract | Notes |
| --- | --- | --- | --- | --- |
| Normal-only calibration | complete | `src/evaluation/calibration.py`, `docs/evidence/generated/metric_definitions.json` | `PYTHONPATH=src venv/bin/python -m pytest tests/test_calibration.py tests/test_calibrated_evaluation.py -q` | Uses validation normals only; oracle metrics are diagnostics. |
| Comparable calibrated metrics | complete | `docs/evidence/generated/primary_results.csv`, `docs/evidence/generated/primary_summary.csv` | `scripts/build_paper_evidence.py` checksum-binds both tables | Heatmaps use calibrated masks; SAM2/fused methods use binary outputs. |
| Statistics and paired deltas | complete | `docs/evidence/generated/paired_statistics.json`, `docs/evidence/generated/baseline_paired_statistics.json` | `PYTHONPATH=src venv/bin/python -m pytest tests/test_statistics.py tests/test_analyze_paper_results.py -q` | Deterministic repeated-measures bootstrap over support seeds and test images. |
| Prompt variants and anomaly-consistent fusion | complete | `src/sam_refine/`, `docs/evidence/generated/method_invariants.json` | 12,030 image contracts across 60 method runs passed | Raw SAM2 provenance and exact intersection semantics are enforced. |
| PatchCore-style baseline | complete | `src/features/patchcore.py`, `docs/evidence/generated/baseline_paired_statistics.json` | `PYTHONPATH=src venv/bin/python -m pytest tests/test_patchcore_features.py -q` | Repository reproduction, not the reference PatchCore implementation. |
| Immutable experiment specs | complete | `configs/experiments/*.yaml`, `docs/evidence/generated/config_bindings.json` | `PYTHONPATH=src venv/bin/python -m pytest tests/test_experiment_spec.py tests/test_provenance.py -q` | Frozen source commit is `ff4c07208278376b27a4954d560c715f51453c5e`. |
| Full primary matrix | complete | `docs/evidence/generated/matrix_primary_check.json` | Checker result: 364 expected, 364 complete, 0 bad | Four VisA PCB categories, k=1/2/4, five support seeds, fold 0. |
| Full ablation matrix | complete | `docs/evidence/generated/matrix_ablations_check.json` | Checker result: 48 expected, 48 complete, 0 bad | Pre-registered one-seed variants are descriptive, not repeated-seed estimates. |
| Frozen GPU audit and provenance | complete | `docs/evidence/generated/frozen_audit_manifest.json`, `docs/evidence/generated/runtime_provenance.json`, `docs/evidence/generated/evidence_revision.json` | Source and postprocessor revisions plus model/config/checkpoint hashes are bound | AutoDL audit used an RTX 4090; numeric source results were not changed by postprocessing. |
| Failure geometry analysis | complete | `docs/evidence/generated/failure_strata.csv`, `docs/evidence/generated/paired_statistics.json` | `PYTHONPATH=src venv/bin/python -m pytest tests/test_failure_analysis.py tests/test_analyze_paper_results.py -q` | Large-area F1/IoU intervals cross zero; small, medium, and thin strata improve robustly. |
| Paper tables and evidence index | complete | `docs/evidence/generated/evidence_index.md`, `docs/evidence/generated/completion_manifest.json` | Every one of 35 declared files has a manifest SHA-256 | Completion manifest schema 3 has `ready_for_writing: true`. |
| Deterministic figures and curation | complete | `docs/evidence/generated/method_figure.png`, `docs/evidence/generated/qualitative_manifest.csv` | One positive and one negative guided-SAM2 delta example per category | Nine public PNGs were checksum-verified and visually inspected. |
| Presentation notebook | complete | `notebooks/pcb_defect_pipeline.ipynb` | `PYTHONPATH=src venv/bin/python -m pytest tests/test_presentation_notebook.py -q` | Loads only compact evidence, verifies hashes, and executes top-to-bottom with no stored outputs. |
| Citation and novelty boundary | complete | `docs/citation_inventory.md` | Primary sources and closest-work boundaries reviewed | No “first” or state-of-the-art claim; DINOv2 scoring and SAM2 are not claimed as novel. |
| End-to-end offline smoke | complete | `tests/test_arxiv_smoke.py` | `PYTHONPATH=src venv/bin/python -m pytest tests/test_arxiv_smoke.py -q` | Uses synthetic data, color-patch features, and fallback refinement. |
| Public repository hygiene | complete | `tests/test_public_hygiene.py` | `PYTHONPATH=src venv/bin/python -m pytest tests/test_public_hygiene.py -q` | Blocks raw data, weights, run archives, private paths, and key-shaped secrets. |
| Final fresh-clone release check | pending | pushed feature-branch commit | Full tests, Ruff, deterministic smoke/checker, hash/path/secret/blob audits | This is the only remaining repository handoff gate. |

## Frozen decision

The pre-declared decision rule is satisfied: both lower bounds are above zero for
anomaly-consistent SAM2 relative to calibrated multi-scale DINOv2 on anomalous
images. F1 improves by `0.066644` (95% CI `[0.053799, 0.078791]`) and IoU by
`0.056505` (95% CI `[0.046240, 0.066546]`). Multi-scale versus single-scale
DINOv2 remains inconclusive and must not be written as a confirmed improvement.

## Required manuscript disclosures

- No model parameters were trained or fine-tuned. Each run builds a normal-support
  feature memory bank, optionally selects a coreset, calibrates on normal
  validation images, and performs frozen-model inference.
- The study uses fold 0. Five seeds are repeated normal-support samples, not five
  independent dataset folds.
- VisA test results were inspected during earlier development, so the test set is
  not an untouched holdout.
- DeepPCB boxes are not pixel-level masks and are outside the primary frozen
  segmentation matrix.
- The one-seed ablations are descriptive sensitivity checks.
- The evidence supports the exact anomaly-consistent intersection, not a broad
  claim that SAM2 or multi-scale inference always helps.
