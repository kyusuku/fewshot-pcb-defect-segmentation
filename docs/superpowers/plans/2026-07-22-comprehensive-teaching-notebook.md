# Comprehensive Teaching Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the compact evidence-handoff notebook with a self-contained, detailed project walkthrough that is sufficient for a classroom presentation and anticipated Q&A while preserving every algorithm, config, frozen AutoDL result, metric, and conclusion exactly.

**Architecture:** Add a deterministic standard-library notebook builder that reads the existing frozen evidence package without modifying it, formats authoritative results into Markdown, and generates a long-form teaching notebook with optional lightweight verification cells. Extend the existing notebook contract tests before generation so narrative depth, chapter coverage, source traceability, Q&A coverage, evidence integrity, and output-free execution are enforceable.

**Tech Stack:** Python 3.11 standard library, Jupyter `nbformat` 4.5 JSON, Markdown/LaTeX, existing CSV/JSON/PNG evidence, pytest.

---

## Non-Negotiable Scope Boundary

- Do not modify files under `src/`.
- Do not modify experiment or dataset configs under `configs/`.
- Do not modify any file under `docs/evidence/generated/`.
- Do not modify existing experiment, evaluation, feature, anomaly, SAM2, or
  evidence-generation scripts.
- Do not recompute, round-trip, or overwrite AutoDL results.
- Do not change any frozen numerical claim or conclusion.
- New executable behavior is limited to a standard-library notebook builder and
  lightweight notebook cells that read and verify tracked evidence.

## File Map

- Create `scripts/build_teaching_notebook.py`: deterministic notebook generator,
  evidence-table formatter, and `--check` drift command.
- Modify `tests/test_presentation_notebook.py`: complete teaching-contract,
  self-contained-reading, source-link, asset, Q&A, and execution tests.
- Regenerate `notebooks/pcb_defect_pipeline.ipynb`: the comprehensive teaching
  notebook read by the user.
- Modify `notebooks/README.md`: document the notebook's new teaching role and
  regeneration/check commands.
- Do not touch implementation, configs, or frozen evidence.

## Task 1: Record the Immutable Baseline

**Files:**
- Read only: `src/**`
- Read only: `configs/**`
- Read only: `docs/evidence/generated/**`
- Temporary: `/tmp/pcb-teaching-notebook-baseline.sha256`

- [ ] **Step 1: Record hashes for algorithms, configs, and frozen evidence**

Run:

```bash
{
  rg --files src configs docs/evidence/generated
} | sort | xargs shasum -a 256 > /tmp/pcb-teaching-notebook-baseline.sha256
```

Expected: a non-empty hash manifest; no repository file changes.

- [ ] **Step 2: Record the pre-existing dirty-worktree boundary**

Run:

```bash
git status --short --branch
```

Expected: preserve the user's existing changes to `README.md`, notebook cell IDs,
`scripts/render_method_figure.py`, `docs/milestone2_readiness_checklist.md`,
`output/`, and `tmp/`. Only files explicitly named in this plan may be staged.

## Task 2: Add the Failing Teaching Contract

**Files:**
- Modify: `tests/test_presentation_notebook.py`
- Test: `tests/test_presentation_notebook.py`

- [ ] **Step 1: Add reusable notebook helpers**

Add imports and helpers:

```python
import re


def _load_notebook() -> tuple[Path, dict[str, object]]:
    repo_root = Path(__file__).resolve().parents[1]
    notebook_path = repo_root / "notebooks/pcb_defect_pipeline.ipynb"
    return notebook_path, json.loads(notebook_path.read_text(encoding="utf-8"))


def _joined_cell_text(notebook: dict[str, object], cell_type: str | None = None) -> str:
    cells = notebook["cells"]
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in cells
        if cell_type is None or cell["cell_type"] == cell_type
    )
```

- [ ] **Step 2: Add a failing complete-chapter test**

Add:

```python
def test_teaching_notebook_covers_the_complete_project() -> None:
    _, notebook = _load_notebook()
    text = _joined_cell_text(notebook)
    required_headings = (
        "How to Use This Notebook",
        "Executive Story",
        "Conceptual Background",
        "Datasets and Benchmark Boundary",
        "Few-Shot Protocol",
        "Repository Map and Artifact Flow",
        "Image Preparation and Alignment",
        "DINOv2 Patch-Feature Extraction",
        "Normal Memory Bank and Anomaly Score",
        "PatchCore-Style Baseline",
        "Multi-Scale Anomaly Proposals",
        "Normal-Only Calibration",
        "Heatmap-to-Prompt Conversion",
        "SAM2 Refinement and Candidate Selection",
        "Anomaly-Consistent Fusion",
        "One-Image Trace",
        "Primary Experiment Matrix",
        "Ablation Matrix",
        "Metrics",
        "Statistical Design",
        "Frozen AutoDL Results",
        "Ablations and Failure Analysis",
        "Qualitative Evidence",
        "Research Answer and Novelty Boundary",
        "Limitations and Threats to Validity",
        "Class Presentation Guide",
        "Anticipated Q&A",
        "Glossary and Symbol Table",
        "Final Cheat Sheet",
        "Reproduction Appendix",
    )
    for heading in required_headings:
        assert heading in text, heading
```

- [ ] **Step 3: Add failing depth, source, presentation, and Q&A tests**

Add:

```python
def test_teaching_notebook_is_detailed_and_source_traceable() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    _, notebook = _load_notebook()
    markdown = _joined_cell_text(notebook, "markdown")
    words = re.findall(r"\b[\w'-]+\b", markdown)
    assert len(words) >= 7000

    required_paths = (
        "src/datasets/visa.py",
        "src/datasets/sampling.py",
        "src/features/dinov2.py",
        "src/features/patchcore.py",
        "src/anomaly/memory_bank.py",
        "src/anomaly/multiscale.py",
        "src/evaluation/calibration.py",
        "src/sam_refine/prompts.py",
        "src/sam_refine/refiner.py",
        "src/sam_refine/fusion.py",
        "src/experiments/spec.py",
        "src/evaluation/statistics.py",
        "configs/experiments/arxiv_primary.yaml",
        "configs/experiments/arxiv_ablations.yaml",
    )
    for relative in required_paths:
        assert relative in markdown
        assert (repo_root / relative).is_file()


def test_teaching_notebook_contains_presentation_and_qa_material() -> None:
    _, notebook = _load_notebook()
    markdown = _joined_cell_text(notebook, "markdown")
    assert len(re.findall(r"^### Slide \d+:", markdown, flags=re.MULTILINE)) >= 10
    assert len(re.findall(r"^### Q\d+\.", markdown, flags=re.MULTILINE)) >= 25
    for topic in (
        "Why did you not train a model?",
        "Why DINOv2?",
        "Why use SAM2 if it is not anomaly-aware?",
        "Why intersection instead of union?",
        "Are the five seeds five folds?",
        "What is actually novel?",
        "What is the biggest limitation?",
    ):
        assert topic in markdown
```

- [ ] **Step 4: Add failing evidence and asset tests**

Add:

```python
def test_teaching_notebook_preserves_frozen_claims_and_assets() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    _, notebook = _load_notebook()
    text = _joined_cell_text(notebook)
    for required in (
        "364 primary runs",
        "48 ablation runs",
        "412 source runs",
        "+0.0666",
        "[0.0538, 0.0788]",
        "+0.0565",
        "[0.0462, 0.0665]",
        "multi-scale versus single-scale DINOv2 is inconclusive",
        "not five-fold cross-validation",
        "No model parameters are trained or fine-tuned",
    ):
        assert required.lower() in text.lower()

    linked_pngs = set(re.findall(r"\((\.\./docs/evidence/generated/[^)]+\.png)\)", text))
    assert len(linked_pngs) >= 9
    for relative in linked_pngs:
        assert (repo_root / "notebooks" / relative).resolve().is_file()
```

- [ ] **Step 5: Run the notebook test and verify RED**

Run:

```bash
PYTHONPATH=src venv/bin/python -m pytest tests/test_presentation_notebook.py -q
```

Expected: FAIL because the current 16-cell notebook lacks most chapters, has
far fewer than 7,000 Markdown words, and has no presentation/Q&A bank.

- [ ] **Step 6: Commit the red test contract**

```bash
git add tests/test_presentation_notebook.py
git commit -m "test: define comprehensive teaching notebook contract"
```

## Task 3: Build the Deterministic Notebook Generator

**Files:**
- Create: `scripts/build_teaching_notebook.py`
- Test: `tests/test_presentation_notebook.py`

- [ ] **Step 1: Implement deterministic cell construction**

The builder must use this interface:

```python
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import textwrap


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "notebooks/pcb_defect_pipeline.ipynb"
EVIDENCE_DIR = REPO_ROOT / "docs/evidence/generated"


def _source(text: str) -> list[str]:
    normalized = textwrap.dedent(text).strip("\n") + "\n"
    return normalized.splitlines(keepends=True)


def _cell_id(index: int, cell_type: str, source: list[str]) -> str:
    payload = f"{index}:{cell_type}:{''.join(source)}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:8]


def markdown_cell(index: int, text: str) -> dict[str, object]:
    source = _source(text)
    return {
        "cell_type": "markdown",
        "id": _cell_id(index, "markdown", source),
        "metadata": {},
        "source": source,
    }


def code_cell(index: int, text: str) -> dict[str, object]:
    source = _source(text)
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": _cell_id(index, "code", source),
        "metadata": {},
        "outputs": [],
        "source": source,
    }
```

- [ ] **Step 2: Implement read-only evidence helpers**

Use only:

```python
def read_json(name: str) -> dict[str, object]:
    path = EVIDENCE_DIR / name
    if not path.is_file():
        raise FileNotFoundError(f"missing frozen evidence: {path.relative_to(REPO_ROOT)}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(name: str) -> list[dict[str, str]]:
    path = EVIDENCE_DIR / name
    if not path.is_file():
        raise FileNotFoundError(f"missing frozen evidence: {path.relative_to(REPO_ROOT)}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
```

Evidence formatting functions must select exact existing rows from
`primary_summary.csv`, `ablation_summary.csv`, `evidence_conclusion.json`,
`baseline_paired_statistics.json`, and `paired_statistics.json`. They may format
display precision but may not write to evidence files or compute new metrics.

- [ ] **Step 3: Implement notebook serialization and drift check**

Use:

```python
def build_notebook() -> dict[str, object]:
    cells = build_cells()
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def serialized_notebook() -> str:
    return json.dumps(build_notebook(), indent=1, ensure_ascii=False) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    rendered = serialized_notebook()
    if args.check:
        if not args.output.is_file() or args.output.read_text(encoding="utf-8") != rendered:
            print(f"notebook is out of date: {args.output}", file=sys.stderr)
            return 1
        print(f"notebook is current: {args.output.relative_to(REPO_ROOT)}")
        return 0
    args.output.write_text(rendered, encoding="utf-8")
    print(f"wrote {args.output.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Add a deterministic-generation test**

Add to `tests/test_presentation_notebook.py`:

```python
def test_teaching_notebook_matches_deterministic_builder() -> None:
    import importlib.util

    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts/build_teaching_notebook.py"
    spec = importlib.util.spec_from_file_location("build_teaching_notebook", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    notebook_path = repo_root / "notebooks/pcb_defect_pipeline.ipynb"
    assert notebook_path.read_text(encoding="utf-8") == module.serialized_notebook()
```

## Task 4: Write the Complete Teaching Narrative

**Files:**
- Modify: `scripts/build_teaching_notebook.py`
- Generate: `notebooks/pcb_defect_pipeline.ipynb`
- Test: `tests/test_presentation_notebook.py`

- [ ] **Step 1: Create all orientation and background cells**

Implement Markdown cells for title/how-to-use, executive story, conceptual
background, datasets, protocol, and repository map. Include a one-minute summary,
normal-only supervision table, stage-to-path map, and explicit training-free
explanation.

- [ ] **Step 2: Create every method cell with equations and code traceability**

Implement the ordered chapters from image geometry through fusion. Required
equations include:

```text
z = DINOv2(x) in R^(Hp x Wp x D)
B = {z_i,p : support image i, valid patch p}
a(q) = min_(b in B) ||normalize(q) - normalize(b)||_2
tau = Quantile_0.995({a_v,p : normal validation pixels})
A(u,v) = 1[a(u,v) >= tau]
M = A intersection S
F1 = 2PR/(P+R)
IoU = TP/(TP+FP+FN)
```

For each chapter include: purpose, input/output, tensor or coordinate shape,
algorithm steps, why the choice exists, failure mode, and authoritative source
path. Code shown in the notebook must be concise excerpts or faithful pseudocode
of existing modules; it must not define an alternative runnable pipeline.

- [ ] **Step 3: Create experiment, metric, statistics, and result cells**

Explain and derive:

- primary methods: PatchCore, single DINOv2, multi DINOv2, SAM2-only, guided
  single SAM2, guided multi SAM2, anomaly-consistent SAM2;
- `4 categories x 3 shots x 5 seeds x 6 support-dependent methods + 4 SAM2-only = 364`;
- `12 ablations x 4 categories = 48`;
- `364 + 48 = 412` source runs;
- calibrated versus oracle metrics;
- paired support-seed/test-image bootstrap; and
- exact frozen results and inconclusive supporting comparisons.

All result tables are inserted into Markdown at build time from the tracked
evidence files.

- [ ] **Step 4: Create interpretation, presentation, Q&A, and appendix cells**

Include:

- research answer and scoped novelty;
- limitations and threats to validity;
- 10-12 slide presentation sequence;
- at least 25 numbered Q&A entries with complete answers;
- glossary and symbol table;
- final pre-class cheat sheet; and
- exact local smoke and AutoDL reproduction commands already documented in the
  repository.

- [ ] **Step 5: Retain and explain all tracked qualitative figures**

Embed the existing method diagram and eight success/failure figures by relative
path. Add a reading guide for the five columns and state why individual examples
are illustrative rather than statistical proof.

- [ ] **Step 6: Generate the notebook**

Run:

```bash
venv/bin/python scripts/build_teaching_notebook.py
```

Expected: `wrote notebooks/pcb_defect_pipeline.ipynb`.

- [ ] **Step 7: Run the notebook contract and verify GREEN**

Run:

```bash
PYTHONPATH=src venv/bin/python -m pytest tests/test_presentation_notebook.py -q
```

Expected: every presentation-notebook test passes.

- [ ] **Step 8: Commit builder, notebook, and green tests**

```bash
git add scripts/build_teaching_notebook.py tests/test_presentation_notebook.py notebooks/pcb_defect_pipeline.ipynb
git commit -m "docs: build comprehensive PCB teaching notebook"
```

## Task 5: Update the Notebook Guide

**Files:**
- Modify: `notebooks/README.md`

- [ ] **Step 1: Document the new reader contract**

Update the guide to state that the notebook is the single comprehensive teaching
walkthrough, is readable without execution, covers the full method/protocol/
evidence/presentation/Q&A story, and remains subordinate to modular code for
implementation truth.

- [ ] **Step 2: Document regeneration and verification**

Add:

```bash
venv/bin/python scripts/build_teaching_notebook.py
venv/bin/python scripts/build_teaching_notebook.py --check
PYTHONPATH=src venv/bin/python -m pytest tests/test_presentation_notebook.py -q
```

- [ ] **Step 3: Commit the guide**

```bash
git add notebooks/README.md
git commit -m "docs: document comprehensive notebook workflow"
```

## Task 6: Full Verification and Human QA

**Files:**
- Verify: all modified files
- Compare: `/tmp/pcb-teaching-notebook-baseline.sha256`

- [ ] **Step 1: Verify deterministic generation and valid JSON**

Run:

```bash
venv/bin/python scripts/build_teaching_notebook.py --check
venv/bin/python -m json.tool notebooks/pcb_defect_pipeline.ipynb > /dev/null
```

Expected: notebook current; JSON command exits 0.

- [ ] **Step 2: Execute notebook-specific tests**

Run:

```bash
PYTHONPATH=src venv/bin/python -m pytest tests/test_presentation_notebook.py -q
```

Expected: all notebook tests pass with no warning or error.

- [ ] **Step 3: Execute the full regression suite**

Run:

```bash
PYTHONPATH=src venv/bin/python -m pytest -q
```

Expected: all tests pass. This is regression evidence only; it does not replace
the content audit.

- [ ] **Step 4: Prove algorithms, configs, and results are unchanged**

Run:

```bash
{
  rg --files src configs docs/evidence/generated
} | sort | xargs shasum -a 256 > /tmp/pcb-teaching-notebook-final.sha256
diff -u /tmp/pcb-teaching-notebook-baseline.sha256 /tmp/pcb-teaching-notebook-final.sha256
git diff --exit-code -- src configs docs/evidence/generated
```

Expected: both comparisons produce no output and exit 0.

- [ ] **Step 5: Run formatting and scope checks**

Run:

```bash
git diff --check
git status --short --branch
```

Expected: no whitespace errors; changed files match the plan plus preserved
pre-existing user changes.

- [ ] **Step 6: Perform the human-readable completion audit**

Extract notebook headings and Markdown, then verify every design requirement:

```bash
jq -r '.cells[] | select(.cell_type == "markdown") | .source | join("")' notebooks/pcb_defect_pipeline.ipynb
jq -r '.cells[] | .source | join("") | split("\n")[0]' notebooks/pcb_defect_pipeline.ipynb
```

Audit all chapters, equations, code-path explanations, frozen tables, figures,
slide sequence, Q&A answers, glossary, and commands. Any unsupported claim or
reader dependency is a failure that must be corrected and re-verified.

- [ ] **Step 7: Visually inspect referenced figures**

Open the method figure plus representative success and failure panels and verify
that labels remain legible at notebook display size. Do not regenerate or edit
the frozen images.
