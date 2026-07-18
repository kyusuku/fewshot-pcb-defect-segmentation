# Evidence Index

Readiness scope: **complete paper evidence**.

Frozen conclusion: **anomaly_consistent_sam2_improves_robustly**.

No model parameters are trained or fine-tuned. Each run builds a support-feature memory bank, optionally applies a PatchCore-style coreset, calibrates on normal validation images, and performs frozen-model inference.

| Planned claim or artifact | Evidence source | Guardrail |
| --- | --- | --- |
| Anomaly-consistent SAM2 versus calibrated multi-scale DINOv2 | `evidence_conclusion.json`, `paired_statistics.json` | Anomalous images only; repeated-measures CI |
| Multi-scale versus single-scale DINOv2 | `baseline_paired_statistics.json`, `primary_summary.csv` | Identical support/test pairing |
| Multi-scale DINOv2 versus PatchCore-style baseline | `baseline_paired_statistics.json`, `primary_summary.csv` | Repository reproduction, not reference PatchCore code |
| Anomaly-guided multi-scale versus single-scale SAM2 | `baseline_paired_statistics.json` | Not the deterministic SAM2-only baseline |
| Deterministic SAM2-only baseline | `primary_summary.csv` | k=0, seed=0 descriptive result; no support-seed CI |
| Pre-registered sensitivity variants | `ablation_summary.csv` | k=4, seed=4880 descriptive only |
| Failure geometry and SAM2 behavior | `failure_strata.csv`, `paired_statistics.json` | Normal rows excluded from paired defect intervals |
| Pixel contracts for selective fusion | `method_invariants.json` | Every primary image checked |
| Matrix completion | `validation_summary.json`, `matrix_primary_check.json`, `matrix_ablations_check.json` | 364 + 48, zero bad |
| Runtime, model, checkpoint, config identities | `runtime_provenance.json`, `evidence_revision.json` | Source runs remain frozen at ff4c072 |
| Metric definitions | `metric_definitions.json` | Calibrated and oracle metrics remain separate |
| Qualitative examples | `qualitative_manifest.csv` | One positive and one negative SAM2-delta case per category; no aggregate implication |
| Method figure | `method_figure_layout.json` | Deterministic layout and PNG checksum |

The five support seeds repeat normal-support sampling within fold 0; they are not independent folds. Prior VisA test-set exposure must be disclosed. DeepPCB boxes are not segmentation masks.
