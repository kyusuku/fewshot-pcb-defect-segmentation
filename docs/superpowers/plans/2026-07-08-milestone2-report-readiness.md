# Milestone 2 Report Readiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the repository ready for the user to write the milestone 2 progress report from tracked readiness evidence and reproducible commands.

**Architecture:** Keep experiment outputs ignored, but commit a concise readiness checklist under `docs/` that records the baseline architecture, protocol, refreshed local results, assumptions, and next steps without drafting the report. Link that checklist from `README.md` so the evidence path is discoverable from a fresh clone.

**Tech Stack:** Markdown documentation, existing Python/PyTorch scripts, existing pytest suite.

---

### Task 1: Add the milestone readiness checklist

**Files:**
- Create: `docs/milestone2_readiness_checklist.md`

- [x] **Step 1: Create the checklist document**

Create `docs/milestone2_readiness_checklist.md` with these sections:

```markdown
# Milestone 2 Readiness Checklist

## Verdict

## Baseline Model

## Training and Evaluation Procedure

## Initial Results

## Qualitative Evidence

## Assumptions and Local Paths

## Missing or Incomplete Before Final Report

## Next Steps
```

The document must explicitly state that it is not a report draft. It must state that the baseline is training-free DINOv2 ViT-S/14 patch-feature nearest-neighbor anomaly scoring with `k=5` normal support images per PCB category, evaluated on VisA PCB fold 0. It must include the refreshed single-scale DINOv2 table and the refreshed stage-comparison table from ignored local outputs.

- [x] **Step 2: Include exact commands**

Add exact commands for the report-critical operations:

```bash
PYTHONPATH=src python scripts/run_dinov2_baseline.py ...
PYTHONPATH=src python scripts/evaluate_heatmaps.py ...
PYTHONPATH=src python scripts/summarize_metrics.py ...
PYTHONPATH=src python scripts/summarize_stage4.py ...
```

Use concrete paths from the current local run, and note that `outputs/`, `data/`, `weights/`, and `external/` remain ignored.

### Task 2: Link the report note from README

**Files:**
- Modify: `README.md`

- [x] **Step 1: Add a short status pointer**

After the PRD pointer near the top of `README.md`, add a sentence linking to `docs/milestone2_readiness_checklist.md`.

Expected wording:

```markdown
For the milestone 2 readiness checklist and evidence index, see
[docs/milestone2_readiness_checklist.md](docs/milestone2_readiness_checklist.md).
```

### Task 3: Verify the repository

**Files:**
- Test: `tests/`

- [x] **Step 1: Run the unit suite**

Run:

```bash
venv/bin/python -m pytest
```

Expected: all tests pass.

- [x] **Step 2: Run a tiny baseline smoke**

Run:

```bash
env PYTHONPATH=src venv/bin/python scripts/run_dinov2_baseline.py \
  --manifest data/manifests/visa_pcb_folds.csv \
  --fold-id 0 \
  --category pcb1 \
  --k 2 \
  --limit 2 \
  --feature-backbone color_patch \
  --image-size 56 \
  --patch-size 14 \
  --crop-sizes 128 \
  --crop-overlap 0.25 \
  --output-dir outputs/milestone2_color_patch_smoke
```

Expected: command exits successfully and writes `scores.csv`, heatmaps, and debug panels under ignored `outputs/milestone2_color_patch_smoke/`.

- [x] **Step 3: Evaluate the tiny smoke**

Run:

```bash
env PYTHONPATH=src venv/bin/python scripts/evaluate_heatmaps.py \
  --scores-csv outputs/milestone2_color_patch_smoke/scores.csv \
  --output-json outputs/milestone2_color_patch_smoke/metrics.json
```

Expected: command exits successfully and writes image and pixel metrics where the two-sample smoke contains enough label/mask coverage.

- [x] **Step 4: Inspect git status**

Run:

```bash
git status --short
```

Expected: only the new docs plan, the milestone readiness checklist, and the README link are tracked changes. Ignored outputs, data, weights, environments, and private PDFs must remain untracked.
